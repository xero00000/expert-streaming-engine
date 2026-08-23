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
COMBINED_FAILURE_STAGES = (
    "after-kv-prepare",
    "after-expert-prepare",
    "after-kv-publish",
    "before-logical-publish",
)


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


def require_rollback_message(failure: dict, label: str) -> None:
    message = failure.get("error", {}).get("message", "").lower()
    accepted = (
        "original cache remains active",
        "original resources remain active",
        "original resources were restored",
        "original cache retained",
    )
    if not any(fragment in message for fragment in accepted):
        raise RuntimeError(f"{label} did not report rollback usability")


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
        elif failure_kind in COMBINED_FAILURE_STAGES:
            environment["ESE_RESOURCE_REBALANCE_FAIL_STAGE"] = failure_kind
        elif failure_kind == "expert-published-device":
            environment[
                "ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES"
            ] = "1"
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


def rebalance_combined(
    port: int,
    context: int,
    bytes_per_device: int,
    dry_run: bool,
    expected_status: int = 200,
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/ese/resources/rebalance",
        {
            "dry_run": dry_run,
            "context": context,
            "expert_cache_bytes_per_device": bytes_per_device,
        },
        timeout=30,
        expected_status=expected_status,
    )


def rebalance_combined_observing_props(
    port: int,
    context: int,
    bytes_per_device: int,
) -> tuple[dict, int]:
    samples = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            rebalance_combined, port, context, bytes_per_device, False
        )
        while not future.done():
            props = request_json(f"http://127.0.0.1:{port}/props")
            plan_context = props.get("resource_plan", {}).get("context")
            if props.get("n_ctx") != plan_context:
                raise RuntimeError(
                    "/props exposed context and resource plan from different commits"
                )
            samples += 1
            time.sleep(0.001)
        result = future.result()
    props = request_json(f"http://127.0.0.1:{port}/props")
    if props.get("n_ctx") != props.get("resource_plan", {}).get("context"):
        raise RuntimeError("/props exposed a split logical state after commit")
    return result, samples + 1


def resources(port: int) -> dict:
    return request_json(f"http://127.0.0.1:{port}/v1/ese/resources")


def kv_geometry(snapshot: dict, include_occupancy: bool = False) -> dict:
    kv = snapshot.get("kv", {})
    keys = ["capacity_tokens", "max_capacity_tokens", "allocated_bytes"]
    if include_occupancy:
        keys.append("used_cells")
    if any(not isinstance(kv.get(key), int) for key in keys):
        raise RuntimeError(f"incomplete live KV accounting: {kv}")
    return {key: kv[key] for key in keys}


def expert_accounting(snapshot: dict) -> list[dict]:
    devices = snapshot.get("devices", [])
    if not devices:
        raise RuntimeError("expert-cache validation requires at least one accelerator")
    accounting = []
    for device in devices:
        cache = device.get("expert_cache", {})
        item = {
            "device": device.get("id"),
            "capacity_bytes": cache.get("capacity_bytes"),
            "allocated_bytes": cache.get("allocated_bytes"),
            "resident_bytes": cache.get("resident_bytes"),
        }
        if not isinstance(item["device"], int) or any(
            not isinstance(item[key], int)
            for key in ("capacity_bytes", "allocated_bytes", "resident_bytes")
        ):
            raise RuntimeError(f"incomplete per-device expert accounting: {device}")
        accounting.append(item)
    return sorted(accounting, key=lambda item: item["device"])


def exact_transaction_state(snapshot: dict) -> dict:
    plan = snapshot.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("resource snapshot omitted the serialized resource plan")
    return {
        "kv": kv_geometry(snapshot, include_occupancy=True),
        "expert_cache": expert_accounting(snapshot),
        "plan": plan,
    }


def require_serialized_plan(
    snapshot: dict,
    context: int,
    expert_capacity: int,
    expected_plan: dict | None = None,
) -> dict:
    plan = snapshot.get("plan")
    if not isinstance(plan, dict) or plan.get("context") != context:
        raise RuntimeError("serialized resource plan exposes stale context geometry")
    live_devices = {item["device"] for item in expert_accounting(snapshot)}
    plan_devices = {
        item.get("id"): item
        for item in plan.get("devices", [])
        if item.get("id") in live_devices
    }
    if set(plan_devices) != live_devices or any(
        item.get("expert_cache_bytes") != expert_capacity
        for item in plan_devices.values()
    ):
        raise RuntimeError("serialized resource plan exposes stale expert-cache geometry")
    if expected_plan is not None and plan != expected_plan:
        raise RuntimeError("serialized resource plan differs from the committed response")
    return plan


def require_combined_preparation_peak(dry_run: dict) -> None:
    peak = dry_run.get("preparation_peak", {})
    if not peak.get("prepares_kv") or not peak.get("prepares_expert_cache"):
        raise RuntimeError("combined dry run did not prepare both resource pools")
    current_devices = {
        item.get("id"): item for item in dry_run.get("current_plan", {}).get("devices", [])
    }
    target_devices = {
        item.get("id"): item for item in dry_run.get("target_plan", {}).get("devices", [])
    }
    peak_devices = {item.get("id"): item for item in peak.get("devices", [])}
    if not current_devices or set(current_devices) != set(target_devices) or set(
        current_devices
    ) != set(peak_devices):
        raise RuntimeError("combined dry run returned inconsistent device topology")
    for device_id, item in peak_devices.items():
        current = current_devices[device_id]
        target = target_devices[device_id]
        expected = {
            "target_live_bytes": target.get("planned_bytes"),
            "prepared_kv_bytes": target.get("kv_bytes"),
            "prepared_expert_cache_bytes": target.get("expert_cache_bytes"),
        }
        if any(item.get(key) != value for key, value in expected.items()):
            raise RuntimeError(
                f"combined preparation peak is inconsistent on device {device_id}: {item}"
            )
        current_live = item.get("current_live_bytes")
        current_without_expert = (
            current.get("planned_bytes", 0)
            - current.get("expert_cache_bytes", 0)
        )
        if not isinstance(current_live, int) or current_live < current_without_expert:
            raise RuntimeError(
                f"combined preparation peak omitted realized live bytes on device {device_id}"
            )
        expected_peak = (
            current_live
            + expected["prepared_kv_bytes"]
            + expected["prepared_expert_cache_bytes"]
        )
        usable = item.get("capacity_bytes", 0) - item.get("reserve_bytes", 0)
        if (
            item.get("peak_bytes") != expected_peak
            or item.get("peak_headroom_bytes") != usable - expected_peak
        ):
            raise RuntimeError(
                f"combined preparation peak arithmetic failed on device {device_id}: {item}"
            )


def require_geometry(snapshot: dict, active: int, maximum: int) -> None:
    kv = snapshot.get("kv", {})
    plan = snapshot.get("plan", {})
    if kv.get("capacity_tokens") != active or kv.get("max_capacity_tokens") != maximum:
        raise RuntimeError(f"unexpected live KV geometry: {kv}")
    if plan.get("context") != active:
        raise RuntimeError("logical resource plan does not match the active KV geometry")


def require_http_views(
    port: int,
    active: int,
    maximum: int,
    expected_plan: dict | None = None,
) -> None:
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
    if expected_plan is not None and props.get("resource_plan") != expected_plan:
        raise RuntimeError("/props exposes a stale serialized resource plan")


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
        require_rollback_message(failure, "failed KV transaction")
        return {"output_sha256": before_hash, "failure_http": 500, "old_engine_usable": True}
    finally:
        server.stop()


def require_expert_capacity(
    snapshot: dict,
    expected: int,
    require_resident: bool,
    required_resident_devices: set[int] | None = None,
) -> dict:
    summaries = expert_accounting(snapshot)
    for item in summaries:
        capacity = item["capacity_bytes"]
        allocated = item["allocated_bytes"]
        resident = item["resident_bytes"]
        if (
            capacity != expected
            or allocated > expected
            or resident > expected
            or (expected > 0 and allocated == 0)
        ):
            raise RuntimeError(f"unexpected expert-cache accounting: {item}")
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
        initial_snapshot = resources(port)
        require_geometry(initial_snapshot, args.context, args.context)
        initial_geometry = kv_geometry(initial_snapshot)
        initial_stats = require_expert_capacity(initial_snapshot, initial, True)
        initial_plan = require_serialized_plan(
            initial_snapshot, args.context, initial
        )
        require_http_views(port, args.context, args.context, initial_plan)
        resident_devices = {
            item["device"]
            for item in initial_stats["devices"]
            if item["resident_bytes"] > 0
        }
        reconcile_dry = rebalance_expert(port, initial, True)
        reconcile_peak = reconcile_dry.get("preparation_peak", {})
        if not reconcile_peak.get("prepares_expert_cache") or any(
            item.get("prepared_expert_cache_bytes") != initial
            for item in reconcile_peak.get("devices", [])
            if item.get("id", -1) >= 0
        ):
            raise RuntimeError(
                "same-target expert reconciliation omitted preparation-peak accounting"
            )
        reconcile = rebalance_expert(port, initial, False)
        if reconcile.get("scope") != "expert-cache-only" or reconcile.get("mutated") is not False:
            raise RuntimeError("explicit same-target expert-cache reconciliation was suppressed")
        if require_expert_capacity(resources(port), initial, True) != initial_stats:
            raise RuntimeError("same-target reconciliation changed a realized expert cache")

        combined_dry = rebalance_combined(
            port, args.target_context, target, True
        )
        combined_target = combined_dry.get("target_plan", {})
        combined_target_devices = [
            item
            for item in combined_target.get("devices", [])
            if item.get("id", -1) >= 0
        ]
        if (
            combined_dry.get("mutated") is not False
            or combined_dry.get("current_plan") != initial_plan
            or combined_target.get("context") != args.target_context
            or len(combined_target_devices) != len(initial_stats["devices"])
            or any(
                item.get("expert_cache_bytes") != target
                for item in combined_target_devices
            )
        ):
            raise RuntimeError("combined dry run did not preserve the exact requested target")
        require_combined_preparation_peak(combined_dry)

        combined_shrink, shrink_props_samples = rebalance_combined_observing_props(
            port, args.target_context, target
        )
        if (
            combined_shrink.get("status") != "committed"
            or combined_shrink.get("scope") != "kv-and-expert"
            or combined_shrink.get("previous_context") != args.context
            or combined_shrink.get("current_context") != args.target_context
            or combined_shrink.get("previous_expert_cache_bytes_per_device") != initial
            or combined_shrink.get("current_expert_cache_bytes_per_device") != target
            or not isinstance(combined_shrink.get("current_plan"), dict)
        ):
            raise RuntimeError("combined KV/expert shrink did not commit atomically")
        combined_target_snapshot = resources(port)
        require_geometry(combined_target_snapshot, args.target_context, args.context)
        target_geometry = kv_geometry(combined_target_snapshot)
        combined_target_stats = require_expert_capacity(
            combined_target_snapshot, target, True, resident_devices
        )
        combined_target_plan = require_serialized_plan(
            combined_target_snapshot,
            args.target_context,
            target,
            combined_shrink.get("current_plan"),
        )
        require_http_views(
            port, args.target_context, args.context, combined_target_plan
        )
        after_combined_shrink = completion(port, args.predict, args.request_timeout)

        combined_grow, grow_props_samples = rebalance_combined_observing_props(
            port, args.context, initial
        )
        if (
            combined_grow.get("status") != "committed"
            or combined_grow.get("scope") != "kv-and-expert"
            or combined_grow.get("previous_context") != args.target_context
            or combined_grow.get("current_context") != args.context
            or combined_grow.get("previous_expert_cache_bytes_per_device") != target
            or combined_grow.get("current_expert_cache_bytes_per_device") != initial
            or not isinstance(combined_grow.get("current_plan"), dict)
        ):
            raise RuntimeError("combined KV/expert restore did not commit atomically")
        combined_grown_snapshot = resources(port)
        require_geometry(combined_grown_snapshot, args.context, args.context)
        grown_geometry = kv_geometry(combined_grown_snapshot)
        if grown_geometry != initial_geometry:
            raise RuntimeError(
                "combined round trip did not restore the exact load-time KV geometry"
            )
        combined_grown_stats = require_expert_capacity(
            combined_grown_snapshot, initial, True, resident_devices
        )
        combined_grown_plan = require_serialized_plan(
            combined_grown_snapshot,
            args.context,
            initial,
            combined_grow.get("current_plan"),
        )
        require_http_views(port, args.context, args.context, combined_grown_plan)
        after_combined_grow = completion(port, args.predict, args.request_timeout)

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
        expert_target_snapshot = resources(port)
        require_geometry(expert_target_snapshot, args.context, args.context)
        expert_target_stats = require_expert_capacity(
            expert_target_snapshot, target, True, resident_devices
        )
        after_shrink = completion(port, args.predict, args.request_timeout)

        grow = rebalance_expert(port, initial, False)
        if grow.get("status") != "committed" or grow.get("scope") != "expert-cache-only":
            raise RuntimeError("prepared expert-cache growth did not commit")
        expert_grown_snapshot = resources(port)
        require_geometry(expert_grown_snapshot, args.context, args.context)
        expert_grown_stats = require_expert_capacity(
            expert_grown_snapshot, initial, True, resident_devices
        )
        after_grow = completion(port, args.predict, args.request_timeout)

        disable = rebalance_expert(port, 0, False)
        if disable.get("status") != "committed" or disable.get("scope") != "expert-cache-only":
            raise RuntimeError("expert-cache disable transaction did not commit")
        disabled_stats = require_expert_capacity(resources(port), 0, False)
        disable_again = rebalance_expert(port, 0, False)
        if disable_again.get("scope") != "expert-cache-only" or disable_again.get("mutated") is not False:
            raise RuntimeError("repeated explicit zero was not handled as an owner-thread no-op")
        require_expert_capacity(resources(port), 0, False)
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
            before,
            after_combined_shrink,
            after_combined_grow,
            after_shrink,
            after_grow,
            after_disable,
            after_enable,
        )]
        if len(set(hashes)) != 1:
            raise RuntimeError(
                "deterministic output changed across combined or expert-cache replacements"
            )
        return {
            "output_sha256": hashes[0],
            "accelerator_count": len(initial_stats["devices"]),
            "combined": {
                "scope": "kv-and-expert",
                "preparation_peak": combined_dry["preparation_peak"],
                "shrink": {
                    "kv": {"from": initial_geometry, "to": target_geometry},
                    "expert_cache": {
                        "from": initial_stats,
                        "to": combined_target_stats,
                    },
                },
                "restore": {
                    "kv": {"from": target_geometry, "to": grown_geometry},
                    "expert_cache": {
                        "from": combined_target_stats,
                        "to": combined_grown_stats,
                    },
                },
                "serialized_plan_verified": True,
                "http_views_verified": True,
                "concurrent_props_samples": {
                    "shrink": shrink_props_samples,
                    "restore": grow_props_samples,
                },
            },
            "shrink": {"from": initial, "to": target},
            "grow": {"from": target, "to": initial},
            "initial": initial_stats,
            "target": expert_target_stats,
            "grown": expert_grown_stats,
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
        require_rollback_message(failure, "failed expert-cache replacement")
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


def validate_combined_failure(
    args: argparse.Namespace,
    failure_kind: str,
    port_offset: int,
) -> dict:
    initial = args.expert_initial_mib * 1024 * 1024
    target = args.expert_target_mib * 1024 * 1024
    port = args.port + port_offset
    server = Server(args, port, failure_kind=failure_kind, expert_mode=True)
    try:
        server.wait_ready()
        completion(port, args.predict, args.request_timeout)
        before = completion(port, args.predict, args.request_timeout)
        before_snapshot = resources(port)
        require_geometry(before_snapshot, args.context, args.context)
        before_stats = require_expert_capacity(before_snapshot, initial, True)
        resident_devices = {
            item["device"]
            for item in before_stats["devices"]
            if item["resident_bytes"] > 0
        }
        before_geometry = kv_geometry(before_snapshot)
        before_plan = require_serialized_plan(
            before_snapshot, args.context, initial
        )
        before_state = exact_transaction_state(before_snapshot)

        failure = rebalance_combined(
            port,
            args.target_context,
            target,
            False,
            expected_status=500,
        )

        # This snapshot must precede any completion: KV occupancy and expert
        # residency are part of the rollback invariant, not eventual state.
        immediate_snapshot = resources(port)
        immediate_state = exact_transaction_state(immediate_snapshot)
        if immediate_state != before_state:
            raise RuntimeError(
                f"{failure_kind} changed live KV, expert, or plan state before recovery: "
                f"before={before_state}, after={immediate_state}"
            )
        require_geometry(immediate_snapshot, args.context, args.context)
        require_expert_capacity(immediate_snapshot, initial, True)
        require_serialized_plan(
            immediate_snapshot, args.context, initial, before_plan
        )
        require_http_views(port, args.context, args.context, before_plan)
        request_json(f"http://127.0.0.1:{port}/health")
        recovery_dry_run = rebalance_combined(
            port, args.target_context, target, True
        )
        require_combined_preparation_peak(recovery_dry_run)

        after = completion(port, args.predict, args.request_timeout)
        before_hash = content_hash(before)
        after_hash = content_hash(after)
        if before_hash != after_hash:
            raise RuntimeError(
                f"{failure_kind} changed deterministic output after rollback"
            )
        recovered_snapshot = resources(port)
        require_geometry(recovered_snapshot, args.context, args.context)
        require_expert_capacity(recovered_snapshot, initial, True)
        require_serialized_plan(
            recovered_snapshot, args.context, initial, before_plan
        )
        require_http_views(port, args.context, args.context, before_plan)
        require_rollback_message(failure, failure_kind)

        # Every test fault is one-shot. A successful target commit and exact
        # restore in this same process prove that rollback released both opaque
        # transaction owners instead of merely leaving inference usable.
        retry = rebalance_combined(
            port, args.target_context, target, False
        )
        if retry.get("status") != "committed" or retry.get("scope") != "kv-and-expert":
            raise RuntimeError(
                f"{failure_kind} left a transaction owner that blocked retry"
            )
        retry_snapshot = resources(port)
        require_geometry(retry_snapshot, args.target_context, args.context)
        require_expert_capacity(retry_snapshot, target, True, resident_devices)
        retry_plan = require_serialized_plan(
            retry_snapshot,
            args.target_context,
            target,
            retry.get("current_plan"),
        )
        require_http_views(port, args.target_context, args.context, retry_plan)

        restore = rebalance_combined(port, args.context, initial, False)
        if restore.get("status") != "committed" or restore.get("scope") != "kv-and-expert":
            raise RuntimeError(f"{failure_kind} retry could not restore initial resources")
        restored_snapshot = resources(port)
        require_geometry(restored_snapshot, args.context, args.context)
        if kv_geometry(restored_snapshot) != before_geometry:
            raise RuntimeError(f"{failure_kind} retry changed KV allocation geometry")
        require_expert_capacity(restored_snapshot, initial, True, resident_devices)
        restored_plan = require_serialized_plan(
            restored_snapshot,
            args.context,
            initial,
            restore.get("current_plan"),
        )
        require_http_views(port, args.context, args.context, restored_plan)
        after_retry = completion(port, args.predict, args.request_timeout)
        if content_hash(after_retry) != before_hash:
            raise RuntimeError(
                f"{failure_kind} retry/restore changed deterministic output"
            )
        return {
            "injection": failure_kind,
            "output_sha256": before_hash,
            "failure_http": 500,
            "exact_pre_completion_state_restored": True,
            "serialized_plan_restored": True,
            "http_endpoints_usable": True,
            "same_process_retry_committed": True,
            "same_process_restore_committed": True,
            "state": before_state,
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
        committed_path = validate_expert_success(args)
        failures = {
            "first_resident_copy": validate_expert_failure(args),
        }
        if args.expert_failure_after_devices >= 0:
            failures["after_prepared_devices"] = validate_expert_failure(
                args, failure_kind="expert-device", port_offset=4
            )
        combined_failures = {
            stage: validate_combined_failure(args, stage, 10 + index)
            for index, stage in enumerate(COMBINED_FAILURE_STAGES)
        }
        if committed_path["accelerator_count"] >= 2:
            combined_failures["after_first_published_expert_device"] = (
                validate_combined_failure(
                    args,
                    failure_kind="expert-published-device",
                    port_offset=10 + len(COMBINED_FAILURE_STAGES),
                )
            )
        expert = {
            "committed_path": committed_path,
            "injected_failures": failures,
            "combined_injected_failures": combined_failures,
        }
    report = {
        "schema": 2,
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
