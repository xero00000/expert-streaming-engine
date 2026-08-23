#!/usr/bin/env python3
"""Validate idle-only, failure-atomic runtime KV and expert-cache rebalancing.

The report contains hashes and resource accounting, never prompts or generated
text. A nonzero exit means the live transaction is not acceptable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = "ESE runtime resource transactions preserve deterministic continuation."
BUSY_PROMPT = "ESE rejects resource mutation while an inference slot is active."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(response: dict) -> str:
    content = response.get("content")
    if not isinstance(content, str) or not content:
        raise RuntimeError("completion returned no generated content")
    return hashlib.sha256(content.encode()).hexdigest()


def request_json(
    url: str,
    payload: dict | None = None,
    timeout: float = 10.0,
    expected_status: int = 200,
) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            body = json.load(response)
    except urllib.error.HTTPError as error:
        status = error.code
        body = json.load(error)
    if status != expected_status:
        raise RuntimeError(f"{url} returned HTTP {status}, expected {expected_status}: {body}")
    return body


class Server:
    def __init__(
        self,
        args: argparse.Namespace,
        port: int,
        failure_kind: str | None = None,
        expert_mode: bool = False,
    ):
        self.args = args
        self.port = port
        self.output = tempfile.TemporaryFile(mode="w+")
        command = [
            str(args.server.resolve()),
            "-m", str(args.model.resolve()),
            "-c", str(args.context),
            "-np", "1",
            "--host", "127.0.0.1",
            "--port", str(port),
            "-ngl", str(args.gpu_layers),
            "--memory-policy", "auto",
            "--max-ram", args.max_ram,
            "--min-kv-quality", "f16",
            "--max-context", str(args.context),
        ]
        if expert_mode:
            command.extend([
                "-ub", "1",
                "--cpu-moe",
                "--expert-vram-cache-mib", str(args.expert_initial_mib),
                "--expert-vram-reserve-mib", str(args.expert_reserve_mib),
                "--expert-cache-min-observations", "1",
            ])
        else:
            command.append("-no-ooae")
        command.extend(args.server_arg)
        environment = os.environ.copy()
        if args.gpu_layers == 0:
            environment["CUDA_VISIBLE_DEVICES"] = ""
        if failure_kind == "kv":
            environment["ESE_TURBO_RETIER_FAIL_AFTER_ROWS"] = "1"
        elif failure_kind == "expert":
            environment["ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_COPIES"] = "0"
        elif failure_kind == "expert-device":
            environment["ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_DEVICES"] = str(
                args.expert_failure_after_devices
            )
        self.process = subprocess.Popen(
            command,
            cwd=args.server.resolve().parents[2],
            env=environment,
            stdout=self.output,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def wait_ready(self) -> None:
        deadline = time.monotonic() + self.args.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited during startup with code {self.process.returncode}")
            try:
                request_json(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return
            except (OSError, urllib.error.URLError, json.JSONDecodeError, RuntimeError):
                time.sleep(0.1)
        raise RuntimeError("server did not become ready before the startup timeout")

    def stop(self) -> str:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.output.seek(0)
        log = self.output.read()
        self.output.close()
        if self.process.returncode != 0:
            raise RuntimeError(
                f"server exited with code {self.process.returncode}:\n"
                + "\n".join(log.splitlines()[-30:])
            )
        return log


def completion(port: int, predict: int, timeout: float, busy: bool = False) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/completion",
        {
            "prompt": BUSY_PROMPT if busy else PROMPT,
            "n_predict": predict,
            "temperature": 0 if not busy else 0.8,
            "seed": 4242,
            "cache_prompt": True,
            "ignore_eos": busy,
        },
        timeout=timeout,
    )


def rebalance(port: int, context: int, dry_run: bool, expected_status: int = 200) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/ese/resources/rebalance",
        {"dry_run": dry_run, "context": context},
        timeout=30,
        expected_status=expected_status,
    )


def rebalance_expert(
    port: int, bytes_per_device: int, dry_run: bool, expected_status: int = 200
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/ese/resources/rebalance",
        {
            "dry_run": dry_run,
            "expert_cache_bytes_per_device": bytes_per_device,
        },
        timeout=30,
        expected_status=expected_status,
    )


def resources(port: int) -> dict:
    return request_json(f"http://127.0.0.1:{port}/v1/ese/resources")


def require_geometry(snapshot: dict, active: int, maximum: int) -> None:
    kv = snapshot.get("kv", {})
    plan = snapshot.get("plan", {})
    if kv.get("capacity_tokens") != active or kv.get("max_capacity_tokens") != maximum:
        raise RuntimeError(f"unexpected live KV geometry: {kv}")
    if plan.get("context") != active:
        raise RuntimeError("logical resource plan does not match the active KV geometry")


def require_http_views(port: int, active: int, maximum: int) -> None:
    props = request_json(f"http://127.0.0.1:{port}/props")
    if (
        props.get("n_ctx") != active
        or props.get("n_ctx_max") != maximum
        or props.get("resource_plan", {}).get("context") != active
    ):
        raise RuntimeError("/props does not match the committed runtime geometry")
    models = request_json(f"http://127.0.0.1:{port}/models")
    codex_models = models.get("models", [])
    v1_models = models.get("data", [])
    if (
        not codex_models
        or codex_models[0].get("context_window") != active
        or codex_models[0].get("load_time_max_context_window") != maximum
        or not v1_models
        or v1_models[0].get("max_model_len") != active
    ):
        raise RuntimeError("model discovery endpoints expose stale context geometry")


def validate_success(args: argparse.Namespace) -> dict:
    server = Server(args, args.port)
    try:
        server.wait_ready()
        require_geometry(resources(args.port), args.context, args.context)
        require_http_views(args.port, args.context, args.context)
        dry = rebalance(args.port, args.target_context, True)
        if dry.get("mutated") is not False or dry.get("target_plan", {}).get("context") != args.target_context:
            raise RuntimeError("dry run did not return the requested immutable target")
        if not dry.get("preparation_peak", {}).get("prepares_kv"):
            raise RuntimeError("KV dry run omitted double-buffer preparation-peak accounting")

        before = completion(args.port, args.predict, args.request_timeout)
        shrink = rebalance(args.port, args.target_context, False)
        if shrink.get("status") != "committed" or shrink.get("current_context") != args.target_context:
            raise RuntimeError("shrink transaction did not commit")
        require_geometry(resources(args.port), args.target_context, args.context)
        require_http_views(args.port, args.target_context, args.context)
        after_shrink = completion(args.port, args.predict, args.request_timeout)

        grow = rebalance(args.port, args.context, False)
        if grow.get("status") != "committed" or grow.get("current_context") != args.context:
            raise RuntimeError("growth transaction did not commit")
        require_geometry(resources(args.port), args.context, args.context)
        require_http_views(args.port, args.context, args.context)
        after_grow = completion(args.port, args.predict, args.request_timeout)

        hashes = [content_hash(item) for item in (before, after_shrink, after_grow)]
        if len(set(hashes)) != 1:
            raise RuntimeError("deterministic output changed across shrink/grow transactions")

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                completion, args.port, args.busy_predict, args.request_timeout, True
            )
            deadline = time.monotonic() + 5
            observed_busy = False
            while time.monotonic() < deadline:
                if not resources(args.port).get("safe_point", True):
                    observed_busy = True
                    break
                time.sleep(0.01)
            if not observed_busy:
                raise RuntimeError("long completion ended before the busy safe point was observed")
            busy_rejection = rebalance(args.port, args.target_context, False, 503)
            future.result(timeout=args.request_timeout)
        require_geometry(resources(args.port), args.context, args.context)
        if "idle" not in busy_rejection.get("error", {}).get("message", ""):
            raise RuntimeError("busy mutation did not explain the idle-only contract")
        return {
            "output_sha256": hashes[0],
            "shrink": {"from": args.context, "to": args.target_context},
            "grow": {"from": args.target_context, "to": args.context},
            "busy_rejection_http": 503,
        }
    finally:
        server.stop()


def validate_failure(args: argparse.Namespace) -> dict:
    port = args.port + 1
    server = Server(args, port, failure_kind="kv")
    try:
        server.wait_ready()
        before = completion(port, args.predict, args.request_timeout)
        failure = rebalance(port, args.target_context, False, 500)
        require_geometry(resources(port), args.context, args.context)
        after = completion(port, args.predict, args.request_timeout)
        before_hash = content_hash(before)
        after_hash = content_hash(after)
        if before_hash != after_hash:
            raise RuntimeError("failed transaction changed deterministic continuation")
        message = failure.get("error", {}).get("message", "")
        if "original cache remains active" not in message:
            raise RuntimeError("failed transaction did not report rollback usability")
        return {"output_sha256": before_hash, "failure_http": 500, "old_engine_usable": True}
    finally:
        server.stop()


def require_expert_capacity(
    snapshot: dict,
    expected: int,
    require_resident: bool,
    required_resident_devices: set[int] | None = None,
) -> dict:
    devices = snapshot.get("devices", [])
    if not devices:
        raise RuntimeError("expert-cache validation requires at least one accelerator")
    summaries = []
    for device in devices:
        cache = device.get("expert_cache", {})
        capacity = cache.get("capacity_bytes")
        allocated = cache.get("allocated_bytes", expected + 1)
        resident = cache.get("resident_bytes", 0)
        if (
            capacity != expected
            or allocated > expected
            or resident > expected
            or (expected > 0 and allocated == 0)
        ):
            raise RuntimeError(f"unexpected expert-cache accounting: {cache}")
        summaries.append({
            "device": device.get("id"),
            "capacity_bytes": capacity,
            "allocated_bytes": allocated,
            "resident_bytes": resident,
        })
    resident_devices = {
        item["device"] for item in summaries if item["resident_bytes"] > 0
    }
    if require_resident and not resident_devices:
        raise RuntimeError("the validation request did not populate an expert cache")
    if required_resident_devices and not required_resident_devices.issubset(resident_devices):
        raise RuntimeError(
            "expert-cache replacement lost resident devices: "
            f"expected {sorted(required_resident_devices)}, got {sorted(resident_devices)}"
        )
    return {"devices": summaries}


def validate_expert_success(args: argparse.Namespace) -> dict:
    initial = args.expert_initial_mib * 1024 * 1024
    target = args.expert_target_mib * 1024 * 1024
    port = args.port + 2
    server = Server(args, port, expert_mode=True)
    try:
        server.wait_ready()
        completion(port, args.predict, args.request_timeout)
        before = completion(port, args.predict, args.request_timeout)
        initial_stats = require_expert_capacity(resources(port), initial, True)
        resident_devices = {
            item["device"]
            for item in initial_stats["devices"]
            if item["resident_bytes"] > 0
        }
        combined = request_json(
            f"http://127.0.0.1:{port}/v1/ese/resources/rebalance",
            {
                "dry_run": False,
                "context": args.target_context,
                "expert_cache_bytes_per_device": target,
            },
            timeout=30,
            expected_status=501,
        )
        if "one resource class" not in combined.get("error", {}).get("message", ""):
            raise RuntimeError("combined mutation was not rejected at the explicit boundary")
        require_geometry(resources(port), args.context, args.context)
        require_expert_capacity(resources(port), initial, True)
        dry = rebalance_expert(port, target, True)
        target_devices = dry.get("target_plan", {}).get("devices", [])
        if dry.get("mutated") is not False or not any(
            item.get("id", -1) >= 0 and item.get("expert_cache_bytes") == target
            for item in target_devices
        ):
            raise RuntimeError("expert-cache dry run did not preserve the requested target")
        if not dry.get("preparation_peak", {}).get("prepares_expert_cache"):
            raise RuntimeError(
                "expert-cache dry run omitted double-buffer preparation-peak accounting"
            )

        shrink = rebalance_expert(port, target, False)
        if shrink.get("status") != "committed" or shrink.get("scope") != "expert-cache-only":
            raise RuntimeError("prepared expert-cache shrink did not commit")
        target_stats = require_expert_capacity(
            resources(port), target, True, resident_devices
        )
        after_shrink = completion(port, args.predict, args.request_timeout)

        grow = rebalance_expert(port, initial, False)
        if grow.get("status") != "committed" or grow.get("scope") != "expert-cache-only":
            raise RuntimeError("prepared expert-cache growth did not commit")
        grown_stats = require_expert_capacity(
            resources(port), initial, True, resident_devices
        )
        after_grow = completion(port, args.predict, args.request_timeout)

        disable = rebalance_expert(port, 0, False)
        if disable.get("status") != "committed" or disable.get("scope") != "expert-cache-only":
            raise RuntimeError("expert-cache disable transaction did not commit")
        disabled_stats = require_expert_capacity(resources(port), 0, False)
        after_disable = completion(port, args.predict, args.request_timeout)

        enable = rebalance_expert(port, initial, False)
        if enable.get("status") != "committed" or enable.get("scope") != "expert-cache-only":
            raise RuntimeError("expert-cache enable transaction did not commit")
        require_expert_capacity(resources(port), initial, False)
        after_enable = completion(port, args.predict, args.request_timeout)
        enabled_stats = require_expert_capacity(
            resources(port), initial, True, resident_devices
        )

        hashes = [content_hash(item) for item in (
            before, after_shrink, after_grow, after_disable, after_enable
        )]
        if len(set(hashes)) != 1:
            raise RuntimeError("deterministic output changed across expert-cache replacements")
        return {
            "output_sha256": hashes[0],
            "combined_boundary_http": 501,
            "shrink": {"from": initial, "to": target},
            "grow": {"from": target, "to": initial},
            "initial": initial_stats,
            "target": target_stats,
            "grown": grown_stats,
            "disabled": disabled_stats,
            "re_enabled": enabled_stats,
        }
    finally:
        server.stop()


def validate_expert_failure(
    args: argparse.Namespace,
    failure_kind: str = "expert",
    port_offset: int = 3,
) -> dict:
    initial = args.expert_initial_mib * 1024 * 1024
    target = args.expert_target_mib * 1024 * 1024
    port = args.port + port_offset
    server = Server(args, port, failure_kind=failure_kind, expert_mode=True)
    try:
        server.wait_ready()
        completion(port, args.predict, args.request_timeout)
        before = completion(port, args.predict, args.request_timeout)
        before_stats = require_expert_capacity(resources(port), initial, True)
        failure = rebalance_expert(port, target, False, 500)
        after_stats = require_expert_capacity(resources(port), initial, True)
        if after_stats != before_stats:
            raise RuntimeError(
                "failed expert-cache replacement changed live device accounting"
            )
        after = completion(port, args.predict, args.request_timeout)
        before_hash = content_hash(before)
        after_hash = content_hash(after)
        if before_hash != after_hash:
            raise RuntimeError("failed expert-cache replacement changed deterministic output")
        message = failure.get("error", {}).get("message", "")
        if "original cache remains active" not in message:
            raise RuntimeError("failed expert-cache replacement did not report rollback usability")
        return {
            "injection": failure_kind,
            "output_sha256": before_hash,
            "failure_http": 500,
            "old_engine_usable": True,
            "before": before_stats,
            "after": after_stats,
        }
    finally:
        server.stop()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, default=Path("build/bin/llama-server"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--target-context", type=int, default=512)
    parser.add_argument("--predict", type=int, default=16)
    parser.add_argument("--busy-predict", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--max-ram", default="4GiB")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=180)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--expert-initial-mib", type=int, default=0)
    parser.add_argument("--expert-target-mib", type=int, default=0)
    parser.add_argument("--expert-reserve-mib", type=int, default=256)
    parser.add_argument("--expert-failure-after-devices", type=int, default=-1)
    args = parser.parse_args()
    for path, label in ((args.server, "server"), (args.model, "model")):
        if not path.is_file():
            parser.error(f"--{label} must name an existing file")
    if (
        args.context <= args.target_context
        or args.target_context < 256
        or args.context % 256
        or args.target_context % 256
        or args.predict < 1
        or args.busy_predict < 64
        or (args.expert_initial_mib and (
            args.gpu_layers < 1
            or args.expert_target_mib < 1
            or args.expert_target_mib >= args.expert_initial_mib
            or args.expert_reserve_mib < 0
        ))
        or (args.expert_target_mib and not args.expert_initial_mib)
        or args.expert_failure_after_devices < -1
        or (args.expert_failure_after_devices >= 0 and not args.expert_initial_mib)
    ):
        parser.error("invalid context, prediction, or expert-cache validation geometry")

    success = validate_success(args)
    failure = validate_failure(args)
    expert = None
    if args.expert_initial_mib:
        failures = {
            "first_resident_copy": validate_expert_failure(args),
        }
        if args.expert_failure_after_devices >= 0:
            failures["after_prepared_devices"] = validate_expert_failure(
                args, failure_kind="expert-device", port_offset=4
            )
        expert = {
            "committed_path": validate_expert_success(args),
            "injected_failures": failures,
        }
    report = {
        "schema": 1,
        "status": "pass",
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "server_sha256": sha256_file(args.server),
        "configuration": {
            "context": args.context,
            "target_context": args.target_context,
            "gpu_layers": args.gpu_layers,
            "expert_initial_mib": args.expert_initial_mib or None,
            "expert_target_mib": args.expert_target_mib or None,
        },
        "committed_path": success,
        "injected_failure": failure,
        "expert_cache": expert,
    }
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
