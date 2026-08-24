#!/usr/bin/env python3
"""Validate idle-only, failure-atomic runtime resource rebalancing.

The report contains hashes and resource accounting, never prompts or generated
text. A nonzero exit means the live transaction is not acceptable.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import signal
import socket
import struct
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
    "after-expert-publish",
    "before-logical-publish",
)
TRANSIENT_FAILURE_STAGES = (
    "after-transient-prepare",
    "after-transient-publish",
)
TRANSIENT_POLICIES = (
    "shared",
    "mtp-only",
    "multimodal-only",
    "off",
)
CANCELLATION_CLEANUP_TIMEOUT_SECONDS = 30.0
HANDOFF_A_PROMPT = "ESE handoff sequence A establishes deterministic text residency."
HANDOFF_SYSTEM_PROMPT = (
    "ESE transient handoff system mutation must survive post-mutation launch failure."
)
HANDOFF_B_PREFIX = " ".join(
    ["ESE", "handoff", "sequence", "B"]
    + [f"unique-marker-{index:02d}" for index in range(24)]
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


def chat_content_hash(response: dict) -> str:
    choices = response.get("choices", [])
    content = choices[0].get("message", {}).get("content") if choices else None
    if not isinstance(content, str) or not content:
        raise RuntimeError("chat completion returned no generated content")
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
        transient_mode: bool = False,
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
        if transient_mode:
            command.extend([
                "--mmproj", str(args.mmproj.resolve()),
                "--spec-type", "mtp:n_max=1,p_min=0.0",
                "--transient-mtp-mib", str(args.transient_mtp_mib),
                "--transient-mmproj-mib", str(args.transient_mmproj_mib),
            ])
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
        elif failure_kind in COMBINED_FAILURE_STAGES + TRANSIENT_FAILURE_STAGES:
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


def completion_with_system_prompt(
    port: int,
    system_prompt: str,
    predict: int,
    timeout: float,
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/completion",
        {
            "prompt": PROMPT,
            "system_prompt": system_prompt,
            "n_predict": predict,
            "temperature": 0,
            "seed": 4242,
            "cache_prompt": True,
        },
        timeout=timeout,
    )


def chat_completion(
    port: int,
    text: str,
    predict: int,
    timeout: float,
    cache_prompt: bool = True,
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": text}],
            "max_tokens": predict,
            "temperature": 0,
            "seed": 4242,
            "cache_prompt": cache_prompt,
        },
        timeout=timeout,
    )


def encoded_image(image_path: Path) -> tuple[str, str]:
    suffix = image_path.suffix.lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(suffix)
    if media_type is None:
        raise RuntimeError("--image must be JPEG, PNG, or WebP")
    return media_type, base64.b64encode(image_path.read_bytes()).decode("ascii")


def multimodal_payload(
    image_path: Path,
    predict: int,
    text: str = "Name the main subject in one word.",
) -> dict:
    media_type, encoded = encoded_image(image_path)
    return {
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{encoded}",
                    },
                },
            ],
        }],
        "max_tokens": predict,
        "temperature": 0,
        "seed": 4242,
        "cache_prompt": True,
    }


def legacy_multimodal_batch_payload(
    image_path: Path,
    predict: int,
    id_slot: int | None = None,
) -> dict:
    _, encoded = encoded_image(image_path)
    payload = {
        "prompt": [
            {
                "prompt_string": (
                    f"{HANDOFF_B_PREFIX} batch-member-{index} "
                    "<__media__> answer briefly."
                ),
                "multimodal_data": [encoded],
            }
            for index in range(2)
        ],
        "n_predict": predict,
        "temperature": 0,
        "seed": 4242,
        "cache_prompt": True,
        "ignore_eos": True,
    }
    if id_slot is not None:
        payload["id_slot"] = id_slot
    return payload


def multimodal_completion(
    port: int,
    image_path: Path,
    predict: int,
    timeout: float,
    expected_status: int = 200,
    text: str = "Name the main subject in one word.",
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        multimodal_payload(image_path, predict, text),
        timeout=timeout,
        expected_status=expected_status,
    )


def malformed_multimodal_completion(
    port: int,
    image_path: Path,
    timeout: float,
) -> dict:
    _, encoded = encoded_image(image_path)
    return request_json(
        f"http://127.0.0.1:{port}/completion",
        {
            "prompt": {
                # The media marker acquires the pre-tokenization lease, while
                # this deliberately invalid type forces tokenization to throw
                # before any task can be posted.
                "prompt_string": {"invalid": True},
                "multimodal_data": [encoded],
            },
            "n_predict": 1,
        },
        timeout=timeout,
        expected_status=400,
    )


def system_prompt_multimodal_launch_failure(
    port: int,
    image_path: Path,
    timeout: float,
) -> dict:
    payload = legacy_multimodal_batch_payload(image_path, 1)
    payload["prompt"] = payload["prompt"][0]
    payload["system_prompt"] = HANDOFF_SYSTEM_PROMPT
    payload["repeat_last_n"] = -2
    payload["ignore_eos"] = False
    return request_json(
        f"http://127.0.0.1:{port}/completion",
        payload,
        timeout=timeout,
        expected_status=400,
    )


def open_cancellable_json_request(
    port: int,
    path: str,
    payload: dict,
    timeout: float,
) -> socket.socket:
    body = json.dumps(payload).encode()
    request = (
        f"POST {path} HTTP/1.1\r\n"
        f"Host: 127.0.0.1:{port}\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode() + body
    connection = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    connection.settimeout(timeout)
    connection.sendall(request)
    return connection


def close_cancellable_request(connection: socket.socket) -> None:
    # A graceful FIN after a complete HTTP request only says the client will
    # send no more bytes; the peer can still legally finish the response.  Use
    # an abortive close so this gate models an actual browser/client request
    # cancellation and the server observes the disconnect before responding.
    try:
        connection.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_LINGER,
            struct.pack("ii", 1, 0),
        )
    except OSError:
        pass
    connection.close()


def multimodal_completion_observing_resources(
    port: int,
    image_path: Path,
    predict: int,
    timeout: float,
    text: str = "Name the main subject in one word.",
) -> tuple[dict, dict]:
    evidence = {
        "samples": 0,
        "active_lease": False,
        "multimodal_resident": False,
        "active_multimodal_owner": False,
        "unsafe_while_leased": False,
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            multimodal_completion,
            port,
            image_path,
            predict,
            timeout,
            text=text,
        )
        while not future.done():
            snapshot = resources(port)
            transient = snapshot.get("transient", {})
            modules = transient.get("modules", [])
            active = transient.get("active_leases", 0) > 0 or transient.get("pins", 0) > 0
            multimodal_resident = any(
                item.get("id") == "multimodal" and item.get("resident") is True
                for item in modules
            )
            evidence["samples"] += 1
            evidence["active_lease"] |= active
            evidence["multimodal_resident"] |= multimodal_resident
            evidence["active_multimodal_owner"] |= active and multimodal_resident
            evidence["unsafe_while_leased"] |= active and snapshot.get("safe_point") is False
            time.sleep(0.001)
        response = future.result()
    if not (
        evidence["samples"] > 0
        and evidence["active_lease"]
        and evidence["multimodal_resident"]
        and evidence["active_multimodal_owner"]
        and evidence["unsafe_while_leased"]
    ):
        raise RuntimeError(
            f"real media request did not expose its owner-thread transient lease: {evidence}"
        )
    return response, evidence


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


def rebalance_transient(
    port: int,
    policy: str,
    dry_run: bool,
    expected_status: int = 200,
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/ese/resources/rebalance",
        {"dry_run": dry_run, "transient_policy": policy},
        timeout=30,
        expected_status=expected_status,
    )


def rebalance_all(
    port: int,
    context: int,
    bytes_per_device: int,
    transient_policy: str,
    dry_run: bool,
    expected_status: int = 200,
) -> dict:
    return request_json(
        f"http://127.0.0.1:{port}/v1/ese/resources/rebalance",
        {
            "dry_run": dry_run,
            "context": context,
            "expert_cache_bytes_per_device": bytes_per_device,
            "transient_policy": transient_policy,
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


def rebalance_all_observing_views(
    port: int,
    context: int,
    bytes_per_device: int,
    transient_policy: str,
    projector_modalities: dict,
) -> tuple[dict, dict]:
    samples = {"props": 0, "simple_props": 0, "resources": 0}
    prior_props = request_json(f"http://127.0.0.1:{port}/props")
    prior_plan = prior_props.get("resource_plan", {})
    prior_context = prior_props.get("n_ctx")
    prior_policy = prior_plan.get("transient_policy")
    if prior_context != prior_plan.get("context"):
        raise RuntimeError("/props exposed split logical state before three-pool commit")
    require_policy_modalities(prior_props, prior_policy, projector_modalities)
    allowed_simple_states = (
        (prior_context, policy_modalities(prior_policy, projector_modalities)),
        (context, policy_modalities(transient_policy, projector_modalities)),
    )
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            rebalance_all,
            port,
            context,
            bytes_per_device,
            transient_policy,
            False,
        )
        while not future.done():
            props = request_json(f"http://127.0.0.1:{port}/props")
            plan = props.get("resource_plan", {})
            if props.get("n_ctx") != plan.get("context"):
                raise RuntimeError(
                    "/props exposed context and resource plan from different commits"
                )
            require_policy_modalities(
                props, plan.get("transient_policy"), projector_modalities
            )
            samples["props"] += 1

            simple_props = request_json(f"http://127.0.0.1:{port}/v1/props")
            simple_state = (
                simple_props.get("n_ctx"),
                simple_props.get("modalities"),
            )
            if simple_state not in allowed_simple_states:
                raise RuntimeError(
                    "/v1/props exposed context and modalities from different commits"
                )
            samples["simple_props"] += 1

            snapshot = resources(port)
            snapshot_plan = snapshot.get("plan", {})
            transient = snapshot.get("transient", {})
            if (
                snapshot.get("kv", {}).get("capacity_tokens")
                != snapshot_plan.get("context")
                or transient.get("policy") != snapshot_plan.get("transient_policy")
            ):
                raise RuntimeError(
                    "/v1/ese/resources exposed split KV/transient logical state"
                )
            samples["resources"] += 1
            time.sleep(0.001)
        result = future.result()
    props = request_json(f"http://127.0.0.1:{port}/props")
    if props.get("n_ctx") != props.get("resource_plan", {}).get("context"):
        raise RuntimeError("/props exposed split logical state after three-pool commit")
    require_policy_modalities(
        props,
        props.get("resource_plan", {}).get("transient_policy"),
        projector_modalities,
    )
    samples["props"] += 1
    simple_props = request_json(f"http://127.0.0.1:{port}/v1/props")
    if simple_props.get("n_ctx") != context:
        raise RuntimeError("/v1/props exposed stale geometry after three-pool commit")
    require_policy_modalities(simple_props, transient_policy, projector_modalities)
    samples["simple_props"] += 1
    snapshot = resources(port)
    if (
        snapshot.get("kv", {}).get("capacity_tokens")
        != snapshot.get("plan", {}).get("context")
        or snapshot.get("transient", {}).get("policy")
        != snapshot.get("plan", {}).get("transient_policy")
    ):
        raise RuntimeError(
            "/v1/ese/resources exposed split logical state after three-pool commit"
        )
    samples["resources"] += 1
    return result, samples


def resources(port: int) -> dict:
    return request_json(f"http://127.0.0.1:{port}/v1/ese/resources")


def wait_for_resource_snapshot(
    port: int,
    predicate,
    timeout: float,
    label: str,
) -> dict:
    deadline = time.monotonic() + timeout
    last_snapshot: dict = {}
    while time.monotonic() < deadline:
        last_snapshot = resources(port)
        if predicate(last_snapshot):
            return last_snapshot
        time.sleep(0.01)
    raise RuntimeError(f"timed out waiting for {label}: {last_snapshot}")


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


def transient_policy_capacity(policy: str, mtp_bytes: int, multimodal_bytes: int) -> int:
    if policy == "shared":
        return max(mtp_bytes, multimodal_bytes)
    if policy == "mtp-only":
        return mtp_bytes
    if policy == "multimodal-only":
        return multimodal_bytes
    if policy == "off":
        return 0
    raise RuntimeError(f"unknown transient policy in validator: {policy}")


def transient_snapshot_state(snapshot: dict) -> dict:
    transient = snapshot.get("transient")
    plan = snapshot.get("plan")
    if not isinstance(transient, dict) or not isinstance(plan, dict):
        raise RuntimeError("resource snapshot omitted transient or serialized-plan state")

    devices = []
    for device in transient.get("devices", []):
        item = {
            key: device.get(key)
            for key in (
                "device",
                "capacity_bytes",
                "reserve_bytes",
                "usable_bytes",
                "resident_configured_bound_bytes",
            )
        }
        if not isinstance(item["device"], int) or any(
            not isinstance(item[key], int)
            for key in (
                "capacity_bytes",
                "reserve_bytes",
                "usable_bytes",
                "resident_configured_bound_bytes",
            )
        ):
            raise RuntimeError(f"incomplete transient device accounting: {device}")
        devices.append(item)

    modules = []
    for module in transient.get("modules", []):
        item = {
            key: module.get(key)
            for key in (
                "id",
                "kind",
                "device",
                "bytes",
                "enabled",
                "resident",
                "pins",
            )
        }
        if (
            not isinstance(item["id"], str)
            or not isinstance(item["kind"], int)
            or not isinstance(item["device"], int)
            or not isinstance(item["bytes"], int)
            or not isinstance(item["enabled"], bool)
            or not isinstance(item["resident"], bool)
            or not isinstance(item["pins"], int)
        ):
            raise RuntimeError(f"incomplete transient module accounting: {module}")
        modules.append(item)

    plan_devices = [
        {"id": item.get("id"), "transient_bytes": item.get("transient_bytes")}
        for item in plan.get("devices", [])
    ]
    if not devices or not modules or any(
        not isinstance(item["id"], int)
        or not isinstance(item["transient_bytes"], int)
        for item in plan_devices
    ):
        raise RuntimeError("transient snapshot omitted device or module topology")

    locks = {
        key: transient.get(key)
        for key in (
            "active_leases",
            "pins",
            "pending_restores",
            "reconfiguration_open",
        )
    }
    if any(not isinstance(locks[key], int) for key in (
        "active_leases", "pins", "pending_restores"
    )) or not isinstance(locks["reconfiguration_open"], bool):
        raise RuntimeError("transient snapshot omitted safe-point ownership state")

    return {
        "policy": transient.get("policy"),
        "devices": sorted(devices, key=lambda item: item["device"]),
        "modules": sorted(modules, key=lambda item: item["id"]),
        "locks": locks,
        "plan": {
            key: plan.get(key)
            for key in (
                "transient_device",
                "transient_policy",
                "transient_mtp_bytes",
                "transient_multimodal_bytes",
                "transient_capacity_bytes",
                "transient_swap",
                "draft_resident",
            )
        }
        | {"devices": sorted(plan_devices, key=lambda item: item["id"])},
    }


def require_zero_transient_ownership(snapshot: dict, label: str) -> dict:
    state = transient_snapshot_state(snapshot)
    expected = {
        "active_leases": 0,
        "pins": 0,
        "pending_restores": 0,
        "reconfiguration_open": False,
    }
    if state["locks"] != expected or any(
        module["pins"] != 0 for module in state["modules"]
    ):
        raise RuntimeError(f"{label} leaked a transient lease or pin: {state['locks']}")
    return state["locks"]


def require_runtime_capabilities(snapshot: dict, require_transient: bool = False) -> None:
    capabilities = snapshot.get("runtime_rebalance", {})
    mutable_pools = capabilities.get("mutable_pools", [])
    required = {"kv", "expert-cache"}
    if require_transient:
        required.add("transient")
    if (
        capabilities.get("mutation_enabled") is not True
        or capabilities.get("mode") != "idle-atomic-multi-pool"
        or capabilities.get("combined_mutation") is not True
        or capabilities.get("dry_run_endpoint") != "/v1/ese/resources/rebalance"
        or not isinstance(mutable_pools, list)
        or not required.issubset(set(mutable_pools))
    ):
        raise RuntimeError(f"resource snapshot omitted runtime capabilities: {capabilities}")


def get_projector_modalities(port: int) -> dict:
    modalities = request_json(f"http://127.0.0.1:{port}/props").get("modalities")
    if (
        not isinstance(modalities, dict)
        or modalities.get("vision") is not True
        or not isinstance(modalities.get("audio"), bool)
    ):
        raise RuntimeError("configured projector did not publish vision capabilities")
    return {
        "vision": modalities["vision"],
        "audio": modalities["audio"],
    }


def policy_modalities(policy: str, projector_modalities: dict) -> dict:
    return (
        projector_modalities
        if policy in {"shared", "multimodal-only"}
        else {"vision": False, "audio": False}
    )


def require_policy_modalities(
    props: dict,
    policy: str,
    projector_modalities: dict,
) -> None:
    expected = policy_modalities(policy, projector_modalities)
    if props.get("modalities") != expected:
        raise RuntimeError(
            f"/props modalities do not match transient policy {policy}: "
            f"expected={expected}, actual={props.get('modalities')}"
        )


def require_transient_policy(
    snapshot: dict,
    policy: str,
    mtp_bytes: int,
    multimodal_bytes: int,
    expected_plan: dict | None = None,
) -> dict:
    if policy not in TRANSIENT_POLICIES:
        raise RuntimeError(f"unsupported transient policy expectation: {policy}")
    state = transient_snapshot_state(snapshot)
    transient = snapshot["transient"]
    plan = snapshot["plan"]
    capacity = transient_policy_capacity(policy, mtp_bytes, multimodal_bytes)
    if (
        state["policy"] != policy
        or plan.get("transient_policy") != policy
        or plan.get("transient_mtp_bytes") != mtp_bytes
        or plan.get("transient_multimodal_bytes") != multimodal_bytes
        or plan.get("transient_capacity_bytes") != capacity
        or plan.get("transient_swap") is not (policy == "shared")
        or plan.get("draft_resident") is not (policy in {"shared", "mtp-only"})
    ):
        raise RuntimeError(f"transient plan does not match {policy}: {plan}")
    if expected_plan is not None and plan != expected_plan:
        raise RuntimeError("transient serialized plan differs from the committed response")

    plan_device = plan.get("transient_device")
    plan_devices = {item.get("id"): item for item in plan.get("devices", [])}
    if plan_device not in plan_devices or any(
        item.get("transient_bytes") != (capacity if device_id == plan_device else 0)
        for device_id, item in plan_devices.items()
    ):
        raise RuntimeError("serialized plan exposes stale transient device accounting")

    modules = {item["id"]: item for item in state["modules"]}
    if set(modules) != {"mtp", "multimodal"}:
        raise RuntimeError(f"unexpected transient module topology: {modules}")
    if (
        modules["mtp"]["bytes"] != mtp_bytes
        or modules["multimodal"]["bytes"] != multimodal_bytes
        or modules["mtp"]["device"] != plan_device
        or modules["multimodal"]["device"] != plan_device
    ):
        raise RuntimeError("transient module bounds or placement differ from the plan")

    enabled = {
        "shared": {"mtp", "multimodal"},
        "mtp-only": {"mtp"},
        "multimodal-only": {"multimodal"},
        "off": set(),
    }[policy]
    resident = {item["id"] for item in state["modules"] if item["resident"]}
    if {item["id"] for item in state["modules"] if item["enabled"]} != enabled:
        raise RuntimeError(f"transient enabled modules do not match {policy}")
    if policy == "shared":
        if len(resident) != 1 or not resident.issubset(enabled):
            raise RuntimeError("shared policy must retain exactly one transient owner")
    elif resident != enabled:
        raise RuntimeError(f"transient resident owner does not match {policy}")

    if len(state["devices"]) != 1 or state["devices"][0]["device"] != plan_device:
        raise RuntimeError("transient telemetry did not preserve single-device placement")
    device = state["devices"][0]
    resident_bound = sum(modules[module_id]["bytes"] for module_id in resident)
    if (
        device["capacity_bytes"] - device["reserve_bytes"] != capacity
        or device["usable_bytes"] != capacity
        or device["resident_configured_bound_bytes"] != resident_bound
    ):
        raise RuntimeError(f"transient budget/residency accounting is inconsistent: {device}")
    if (
        state["locks"]
        != {
            "active_leases": 0,
            "pins": 0,
            "pending_restores": 0,
            "reconfiguration_open": False,
        }
        or snapshot.get("safe_point") is not True
        or transient.get("byte_accounting")
        != "configured-peak-bounds-not-backend-measurement"
        or transient.get("placement_scope") != "configured-main-gpu-only"
    ):
        raise RuntimeError("transient owner was not published at a clean safe point")
    require_runtime_capabilities(snapshot, require_transient=True)
    return state


def exact_transaction_state(snapshot: dict) -> dict:
    plan = snapshot.get("plan")
    if not isinstance(plan, dict):
        raise RuntimeError("resource snapshot omitted the serialized resource plan")
    state = {
        "kv": kv_geometry(snapshot, include_occupancy=True),
        "expert_cache": expert_accounting(snapshot),
        "plan": plan,
    }
    if "transient" in snapshot:
        state["transient"] = transient_snapshot_state(snapshot)
    return state


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


def require_combined_preparation_peak(
    dry_run: dict,
    include_transient: bool = False,
) -> None:
    peak = dry_run.get("preparation_peak", {})
    if not peak.get("prepares_kv") or not peak.get("prepares_expert_cache"):
        raise RuntimeError("combined dry run did not prepare both resource pools")
    if include_transient and not peak.get("prepares_transient"):
        raise RuntimeError("combined dry run did not prepare the transient pool")
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
            "prepared_transient_bytes": (
                target.get("transient_bytes") if include_transient else 0
            ),
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
            + expected["prepared_transient_bytes"]
        )
        usable = item.get("capacity_bytes", 0) - item.get("reserve_bytes", 0)
        if (
            item.get("peak_bytes") != expected_peak
            or item.get("peak_headroom_bytes") != usable - expected_peak
        ):
            raise RuntimeError(
                f"combined preparation peak arithmetic failed on device {device_id}: {item}"
            )


def require_transient_preparation_peak(
    dry_run: dict,
    policy: str,
    mtp_bytes: int,
    multimodal_bytes: int,
    prepares: bool,
) -> None:
    peak = dry_run.get("preparation_peak", {})
    target = dry_run.get("target_plan", {})
    if (
        dry_run.get("status") != "validated"
        or dry_run.get("dry_run") is not True
        or dry_run.get("mutated") is not False
        or target.get("transient_policy") != policy
        or target.get("transient_capacity_bytes")
        != transient_policy_capacity(policy, mtp_bytes, multimodal_bytes)
        or peak.get("prepares_transient") is not prepares
        or peak.get("prepares_kv") is not False
        or peak.get("prepares_expert_cache") is not False
    ):
        raise RuntimeError(f"transient dry run did not preserve {policy}: {dry_run}")
    current_devices = {
        item.get("id"): item for item in dry_run.get("current_plan", {}).get("devices", [])
    }
    target_devices = {
        item.get("id"): item for item in target.get("devices", [])
    }
    peak_devices = {item.get("id"): item for item in peak.get("devices", [])}
    if not current_devices or not (
        set(current_devices) == set(target_devices) == set(peak_devices)
    ):
        raise RuntimeError("transient dry run returned inconsistent device topology")
    for device_id, item in peak_devices.items():
        target_device = target_devices[device_id]
        prepared = target_device.get("transient_bytes") if prepares else 0
        current_live = item.get("current_live_bytes")
        usable = item.get("capacity_bytes", 0) - item.get("reserve_bytes", 0)
        if (
            not isinstance(current_live, int)
            or item.get("target_live_bytes") != target_device.get("planned_bytes")
            or item.get("prepared_kv_bytes") != 0
            or item.get("prepared_expert_cache_bytes") != 0
            or item.get("prepared_transient_bytes") != prepared
            or item.get("peak_bytes") != current_live + prepared
            or item.get("peak_headroom_bytes") != usable - current_live - prepared
        ):
            raise RuntimeError(
                f"transient preparation-peak arithmetic failed on device {device_id}: {item}"
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
    transient_policy: str | None = None,
    transient_mtp_bytes: int = 0,
    transient_multimodal_bytes: int = 0,
    projector_modalities: dict | None = None,
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
    simple_props = request_json(f"http://127.0.0.1:{port}/v1/props")
    if (
        simple_props.get("n_ctx") != active
        or simple_props.get("n_ctx_max") != maximum
    ):
        raise RuntimeError("/v1/props exposes stale context geometry")
    if transient_policy is not None:
        if projector_modalities is None:
            raise RuntimeError("transient HTTP validation requires projector modalities")
        require_policy_modalities(props, transient_policy, projector_modalities)
        require_policy_modalities(
            simple_props, transient_policy, projector_modalities
        )
    snapshot = resources(port)
    require_runtime_capabilities(snapshot, require_transient=transient_policy is not None)
    if transient_policy is not None:
        require_transient_policy(
            snapshot,
            transient_policy,
            transient_mtp_bytes,
            transient_multimodal_bytes,
            expected_plan,
        )


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


def require_transient_commit(
    response: dict,
    previous_policy: str,
    current_policy: str,
    scope: str = "transient-only",
) -> dict:
    if (
        response.get("status") != "committed"
        or response.get("dry_run") is not False
        or response.get("scope") != scope
        or response.get("previous_transient_policy") != previous_policy
        or response.get("current_transient_policy") != current_policy
        or not isinstance(response.get("current_plan"), dict)
    ):
        raise RuntimeError(
            f"transient policy {previous_policy}->{current_policy} did not commit: {response}"
        )
    return response["current_plan"]


def transient_resident_owner(state: dict) -> str:
    resident = [item["id"] for item in state["modules"] if item["resident"]]
    if len(resident) != 1:
        raise RuntimeError(f"shared transient state has no unique resident owner: {resident}")
    return resident[0]


def restore_shared_owner(
    port: int,
    desired_owner: str,
    mtp_bytes: int,
    multimodal_bytes: int,
) -> tuple[dict, dict]:
    current_snapshot = resources(port)
    current = require_transient_policy(
        current_snapshot, "shared", mtp_bytes, multimodal_bytes
    )
    if transient_resident_owner(current) == desired_owner:
        return current, current_snapshot["plan"]
    owner_policy = {
        "mtp": "mtp-only",
        "multimodal": "multimodal-only",
    }.get(desired_owner)
    if owner_policy is None:
        raise RuntimeError(f"cannot restore unknown transient owner: {desired_owner}")
    select = rebalance_transient(port, owner_policy, False)
    select_plan = require_transient_commit(
        select, "shared", owner_policy
    )
    require_transient_policy(
        resources(port), owner_policy, mtp_bytes, multimodal_bytes, select_plan
    )
    shared = rebalance_transient(port, "shared", False)
    shared_plan = require_transient_commit(shared, owner_policy, "shared")
    shared_snapshot = resources(port)
    restored = require_transient_policy(
        shared_snapshot, "shared", mtp_bytes, multimodal_bytes, shared_plan
    )
    if transient_resident_owner(restored) != desired_owner:
        raise RuntimeError("shared policy did not retain the selected transient owner")
    return restored, shared_plan


def require_disabled_multimodal(response: dict, policy: str) -> None:
    message = response.get("error", {}).get("message", "").lower()
    if "disabled by the active transient policy" not in message:
        raise RuntimeError(
            f"{policy} did not explain its multimodal rejection: {response}"
        )


def validate_system_prompt_launch_failure(
    args: argparse.Namespace,
    port: int,
    expected_plan: dict,
    mtp_bytes: int,
    multimodal_bytes: int,
    baseline_hash: str,
) -> dict:
    before_snapshot = resources(port)
    before_state = require_transient_policy(
        before_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if transient_resident_owner(before_state) != "mtp":
        raise RuntimeError("system-prompt handoff gate did not start from warm MTP")
    if request_json(f"http://127.0.0.1:{port}/props").get("system_prompt") != "":
        raise RuntimeError("system-prompt handoff gate requires the default empty prompt")

    failure = system_prompt_multimodal_launch_failure(
        port, args.image, args.request_timeout
    )
    message = failure.get("error", {}).get("message", "").lower()
    if "repeat_last_n" not in message or ">= -1" not in message:
        raise RuntimeError(
            "media/system request did not reach the intended post-mutation launch error"
        )
    request_json(f"http://127.0.0.1:{port}/health")
    changed_props = request_json(f"http://127.0.0.1:{port}/props")
    if changed_props.get("system_prompt") != HANDOFF_SYSTEM_PROMPT:
        raise RuntimeError("post-launch error discarded the committed system prompt")

    failed_snapshot = resources(port)
    failed_state = require_transient_policy(
        failed_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if transient_resident_owner(failed_state) != "multimodal":
        raise RuntimeError(
            "post-system-prompt launch error restored stale prior MTP residency"
        )
    failed_cleanup = require_zero_transient_ownership(
        failed_snapshot, "post-system-prompt launch error"
    )

    changed_system_text_hash = content_hash(completion(
        port, args.predict, args.request_timeout
    ))
    rebuilt_snapshot = resources(port)
    rebuilt_state = require_transient_policy(
        rebuilt_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if transient_resident_owner(rebuilt_state) != "mtp":
        raise RuntimeError("next text request did not rebuild MTP after system mutation")
    if (
        request_json(f"http://127.0.0.1:{port}/props").get("system_prompt")
        != HANDOFF_SYSTEM_PROMPT
    ):
        raise RuntimeError("next text request did not retain the committed system prompt")

    reset_hash = content_hash(completion_with_system_prompt(
        port, "", args.predict, args.request_timeout
    ))
    if reset_hash != baseline_hash:
        raise RuntimeError("clearing the system prompt did not restore deterministic output")
    reset_snapshot = resources(port)
    reset_state = require_transient_policy(
        reset_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if reset_state != before_state:
        raise RuntimeError(
            "system-prompt handoff gate did not restore its initial transient state"
        )
    if request_json(f"http://127.0.0.1:{port}/props").get("system_prompt") != "":
        raise RuntimeError("system-prompt handoff gate did not clear its test mutation")
    request_json(f"http://127.0.0.1:{port}/health")
    return {
        "failure_http": 400,
        "post_system_prompt_launch_error": True,
        "server_alive": True,
        "system_prompt_sha256": hashlib.sha256(
            HANDOFF_SYSTEM_PROMPT.encode()
        ).hexdigest(),
        "system_change_retained": True,
        "stale_prior_mtp_not_restored": True,
        "multimodal_remained_resident": True,
        "zero_active_leases_and_pins": (
            failed_cleanup["active_leases"] == 0
            and failed_cleanup["pins"] == 0
        ),
        "next_text_rebuilt_mtp": True,
        "changed_system_text_sha256": changed_system_text_hash,
        "test_mutation_cleared": True,
    }


def validate_serialized_disjoint_rebalances(
    args: argparse.Namespace,
    port: int,
    initial_expert: int,
    mtp_bytes: int,
    multimodal_bytes: int,
    projector_modalities: dict,
    baseline_hash: str,
) -> dict:
    before_snapshot = resources(port)
    before_geometry = kv_geometry(before_snapshot)
    before_plan = require_serialized_plan(
        before_snapshot, args.context, initial_expert
    )
    before_transient = require_transient_policy(
        before_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        before_plan,
    )

    # The public handler intentionally serializes these disjoint requests.
    # Submit them together and require each to commit from the fresh state it
    # observes under that mutex, with the second response publishing the exact
    # combined final plan rather than overwriting its peer from a stale plan.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        kv_future = executor.submit(
            rebalance, port, args.target_context, False
        )
        transient_future = executor.submit(
            rebalance_transient, port, "mtp-only", False
        )
        kv_result = kv_future.result(timeout=args.request_timeout)
        transient_result = transient_future.result(timeout=args.request_timeout)

    if (
        kv_result.get("status") != "committed"
        or kv_result.get("scope") != "kv-only"
        or kv_result.get("previous_context") != args.context
        or kv_result.get("current_context") != args.target_context
    ):
        raise RuntimeError("serialized disjoint KV mutation did not commit coherently")
    require_transient_commit(
        transient_result, "shared", "mtp-only", scope="transient-only"
    )

    combined_snapshot = resources(port)
    require_geometry(combined_snapshot, args.target_context, args.context)
    require_expert_capacity(combined_snapshot, initial_expert, True)
    combined_plan = require_serialized_plan(
        combined_snapshot, args.target_context, initial_expert
    )
    require_transient_policy(
        combined_snapshot,
        "mtp-only",
        mtp_bytes,
        multimodal_bytes,
        combined_plan,
    )
    if json.dumps(combined_plan, sort_keys=True) not in {
        json.dumps(kv_result.get("current_plan"), sort_keys=True),
        json.dumps(transient_result.get("current_plan"), sort_keys=True),
    }:
        raise RuntimeError("neither serialized response published the final combined plan")
    require_http_views(
        port,
        args.target_context,
        args.context,
        combined_plan,
        "mtp-only",
        mtp_bytes,
        multimodal_bytes,
        projector_modalities,
    )

    kv_restore = rebalance(port, args.context, False)
    if (
        kv_restore.get("status") != "committed"
        or kv_restore.get("scope") != "kv-only"
        or kv_restore.get("current_context") != args.context
    ):
        raise RuntimeError("serialized disjoint gate could not restore KV geometry")
    transient_restore = rebalance_transient(port, "shared", False)
    restored_plan = require_transient_commit(
        transient_restore, "mtp-only", "shared", scope="transient-only"
    )
    restored_snapshot = resources(port)
    require_geometry(restored_snapshot, args.context, args.context)
    if kv_geometry(restored_snapshot) != before_geometry:
        raise RuntimeError("serialized disjoint round trip changed KV allocation geometry")
    require_expert_capacity(restored_snapshot, initial_expert, True)
    restored_transient = require_transient_policy(
        restored_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        restored_plan,
    )
    if restored_transient != before_transient:
        raise RuntimeError("serialized disjoint round trip changed transient ownership")
    require_http_views(
        port,
        args.context,
        args.context,
        restored_plan,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        projector_modalities,
    )
    if content_hash(completion(port, args.predict, args.request_timeout)) != baseline_hash:
        raise RuntimeError("serialized disjoint round trip changed deterministic output")
    return {
        "concurrent_submission": True,
        "serialized_by_http_mutex": True,
        "both_committed": True,
        "coherent_combined_plan": True,
        "physical_manager_aligned": True,
        "round_trip": True,
    }


def validate_multitask_multimodal_handoff(
    args: argparse.Namespace,
    port: int,
    expected_state: dict,
    expected_plan: dict,
    mtp_bytes: int,
    multimodal_bytes: int,
) -> dict:
    def multimodal_is_resident(snapshot: dict) -> bool:
        return any(
            module.get("id") == "multimodal" and module.get("resident") is True
            for module in snapshot.get("transient", {}).get("modules", [])
        )

    # Both tasks are valid and fully tokenized, but an impossible requested
    # slot keeps them deferred. Cancelling the client proves all-fail group
    # resolution rolls the shared owner back from multimodal to exact prior MTP.
    all_fail_connection: socket.socket | None = None
    try:
        all_fail_connection = open_cancellable_json_request(
            port,
            "/completion",
            legacy_multimodal_batch_payload(
                args.image,
                max(args.busy_predict, 1024),
                id_slot=2_147_483_647,
            ),
            args.request_timeout,
        )
        all_fail_live = wait_for_resource_snapshot(
            port,
            lambda snapshot: (
                snapshot.get("slots", {}).get("processing") == 0
                and snapshot.get("slots", {}).get("deferred", 0) >= 2
                and snapshot.get("transient", {}).get("active_leases", 0) >= 2
                and snapshot.get("transient", {}).get("pins", 0) >= 2
                and multimodal_is_resident(snapshot)
            ),
            args.request_timeout,
            "both image batch members to defer before launch",
        )
    finally:
        if all_fail_connection is not None:
            close_cancellable_request(all_fail_connection)

    all_fail_snapshot = wait_for_resource_snapshot(
        port,
        lambda snapshot: (
            snapshot.get("safe_point") is True
            and snapshot.get("slots", {}).get("deferred") == 0
            and snapshot.get("transient", {}).get("active_leases") == 0
            and snapshot.get("transient", {}).get("pins") == 0
            and snapshot.get("transient", {}).get("pending_restores") == 0
        ),
        min(args.request_timeout, CANCELLATION_CLEANUP_TIMEOUT_SECONDS),
        "all-fail image batch rollback",
    )
    all_fail_state = require_transient_policy(
        all_fail_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if all_fail_state != expected_state:
        raise RuntimeError(
            "two-item all-fail image batch did not restore exact prior MTP owner"
        )

    # With no requested slot, the first member launches and the second defers
    # behind it. Cancelling both must still commit multimodal because at least
    # one member reached an inference slot.
    partial_connection: socket.socket | None = None
    try:
        partial_connection = open_cancellable_json_request(
            port,
            "/completion",
            legacy_multimodal_batch_payload(
                args.image, max(args.busy_predict, 1024)
            ),
            args.request_timeout,
        )
        partial_live = wait_for_resource_snapshot(
            port,
            lambda snapshot: (
                snapshot.get("slots", {}).get("processing") == 1
                and snapshot.get("slots", {}).get("deferred", 0) >= 1
                and snapshot.get("transient", {}).get("active_leases", 0) >= 2
                and snapshot.get("transient", {}).get("pins", 0) >= 2
                and multimodal_is_resident(snapshot)
            ),
            args.request_timeout,
            "one launched and one deferred image batch member",
        )
    finally:
        if partial_connection is not None:
            close_cancellable_request(partial_connection)

    partial_snapshot = wait_for_resource_snapshot(
        port,
        lambda snapshot: (
            snapshot.get("safe_point") is True
            and snapshot.get("slots", {}).get("deferred") == 0
            and snapshot.get("transient", {}).get("active_leases") == 0
            and snapshot.get("transient", {}).get("pins") == 0
            and snapshot.get("transient", {}).get("pending_restores") == 0
        ),
        min(args.request_timeout, CANCELLATION_CLEANUP_TIMEOUT_SECONDS),
        "partial-launch image batch commit",
    )
    partial_state = require_transient_policy(
        partial_snapshot,
        "shared",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if transient_resident_owner(partial_state) != "multimodal":
        raise RuntimeError(
            "partial-launch image batch did not commit multimodal residency"
        )
    require_zero_transient_ownership(
        partial_snapshot, "partial-launch image batch"
    )
    return {
        "members": 2,
        "all_fail": {
            "deferred": all_fail_live["slots"]["deferred"],
            "exact_prior_mtp_restored": True,
            "zero_active_leases_and_pins": True,
        },
        "partial_launch": {
            "processing": partial_live["slots"]["processing"],
            "deferred": partial_live["slots"]["deferred"],
            "multimodal_committed": True,
            "zero_active_leases_and_pins": True,
        },
    }


def validate_deferred_multimodal_cancel(
    args: argparse.Namespace,
    port: int,
    expected_state: dict,
    expected_plan: dict,
    mtp_bytes: int,
    multimodal_bytes: int,
) -> dict:
    connection: socket.socket | None = None
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        busy_future = executor.submit(
            completion,
            port,
            args.busy_predict,
            args.request_timeout,
            True,
        )
        wait_for_resource_snapshot(
            port,
            lambda snapshot: snapshot.get("slots", {}).get("processing") == 1,
            args.request_timeout,
            "the sole inference slot to become busy",
        )
        try:
            connection = open_cancellable_json_request(
                port,
                "/v1/chat/completions",
                multimodal_payload(args.image, args.predict),
                args.request_timeout,
            )
            deferred_snapshot = wait_for_resource_snapshot(
                port,
                lambda snapshot: (
                    snapshot.get("slots", {}).get("processing") == 1
                    and snapshot.get("slots", {}).get("deferred", 0) > 0
                    and (
                        snapshot.get("transient", {}).get("active_leases", 0) > 0
                        or snapshot.get("transient", {}).get("pins", 0) > 0
                    )
                ),
                args.request_timeout,
                "a lease-owning multimodal request to defer behind the busy slot",
            )
            if busy_future.done():
                raise RuntimeError(
                    "the original busy request ended before deferred cancellation"
                )
        finally:
            if connection is not None:
                close_cancellable_request(connection)

        cancelled_snapshot = wait_for_resource_snapshot(
            port,
            lambda snapshot: (
                snapshot.get("slots", {}).get("processing") == 1
                and snapshot.get("slots", {}).get("deferred") == 0
                and snapshot.get("transient", {}).get("active_leases") == 0
                and snapshot.get("transient", {}).get("pins") == 0
                and snapshot.get("transient", {}).get("pending_restores") == 0
            ),
            min(args.request_timeout, CANCELLATION_CLEANUP_TIMEOUT_SECONDS),
            "deferred multimodal cancellation cleanup",
        )
        if busy_future.done():
            raise RuntimeError(
                "the image request was not cancelled while the original slot remained busy"
            )
        require_zero_transient_ownership(
            cancelled_snapshot, "deferred multimodal cancellation"
        )
        busy_future.result()

    final_snapshot = resources(port)
    final_state = require_transient_policy(
        final_snapshot,
        "multimodal-only",
        mtp_bytes,
        multimodal_bytes,
        expected_plan,
    )
    if final_state != expected_state:
        raise RuntimeError(
            "never-launched deferred image did not restore exact prior transient state"
        )
    return {
        "all_slots_busy": True,
        "deferred_observed": deferred_snapshot["slots"]["deferred"],
        "active_lease_observed": True,
        "cancelled_before_launch": True,
        "zero_active_leases_and_pins": True,
        "exact_prior_owner_restored": True,
    }


def validate_transient_success(args: argparse.Namespace) -> dict:
    mib = 1024 * 1024
    mtp_bytes = args.transient_mtp_mib * mib
    multimodal_bytes = args.transient_mmproj_mib * mib
    initial_expert = args.expert_initial_mib * mib
    target_expert = args.expert_target_mib * mib
    port = args.port + 30
    server = Server(args, port, expert_mode=True, transient_mode=True)
    try:
        server.wait_ready()
        projector_modalities = get_projector_modalities(port)
        completion(port, args.predict, args.request_timeout)
        baseline_completion = completion(port, args.predict, args.request_timeout)
        baseline_hash = content_hash(baseline_completion)
        handoff_a_hash = chat_content_hash(chat_completion(
            port,
            HANDOFF_A_PROMPT,
            args.predict,
            args.request_timeout,
        ))
        initial_snapshot = resources(port)
        require_geometry(initial_snapshot, args.context, args.context)
        require_expert_capacity(initial_snapshot, initial_expert, True)
        initial_transient = require_transient_policy(
            initial_snapshot, "shared", mtp_bytes, multimodal_bytes
        )
        initial_owner = transient_resident_owner(initial_transient)
        if initial_owner != "mtp":
            raise RuntimeError("text handoff A did not establish MTP residency")
        initial_plan = require_serialized_plan(
            initial_snapshot, args.context, initial_expert
        )
        require_http_views(
            port,
            args.context,
            args.context,
            initial_plan,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )

        system_prompt_handoff_evidence = validate_system_prompt_launch_failure(
            args,
            port,
            initial_plan,
            mtp_bytes,
            multimodal_bytes,
            baseline_hash,
        )

        # A failed parse acquires the pre-tokenization image owner but posts no
        # task. Its guard must roll back the exact prior MTP owner immediately.
        malformed = malformed_multimodal_completion(
            port, args.image, args.request_timeout
        )
        if not malformed.get("error", {}).get("message"):
            raise RuntimeError("malformed media request did not return a parse error")
        malformed_snapshot = resources(port)
        malformed_state = require_transient_policy(
            malformed_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            initial_plan,
        )
        if malformed_state != initial_transient:
            raise RuntimeError(
                "failed multimodal parse did not roll back its exact prior owner"
            )

        batch_handoff_evidence = validate_multitask_multimodal_handoff(
            args,
            port,
            initial_transient,
            initial_plan,
            mtp_bytes,
            multimodal_bytes,
        )
        if chat_content_hash(chat_completion(
            port,
            HANDOFF_A_PROMPT,
            args.predict,
            args.request_timeout,
        )) != handoff_a_hash:
            raise RuntimeError(
                "text recovery changed after partial-launch image batch commit"
            )
        post_batch_snapshot = resources(port)
        post_batch_state = require_transient_policy(
            post_batch_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            initial_plan,
        )
        if transient_resident_owner(post_batch_state) != "mtp":
            raise RuntimeError(
                "text recovery did not rebuild MTP after partial-launch image batch"
            )

        # Once image task B reaches the slot, multimodal becomes the committed
        # owner. Text C shares B's long unique prefix and must rebuild MTP from
        # fresh state; a second B -> C pass without prompt-cache reuse provides
        # deterministic parity against that handoff.
        image_prompt = HANDOFF_B_PREFIX + " Describe the image in one short sentence."
        shared_text_prompt = HANDOFF_B_PREFIX + " Continue in one short text-only sentence."
        image_response, shared_media_evidence = multimodal_completion_observing_resources(
            port,
            args.image,
            args.predict,
            args.request_timeout,
            text=image_prompt,
        )
        image_output_hash = chat_content_hash(image_response)
        image_snapshot = resources(port)
        image_state = require_transient_policy(
            image_snapshot, "shared", mtp_bytes, multimodal_bytes
        )
        if transient_resident_owner(image_state) != "multimodal":
            raise RuntimeError("launched image B did not commit multimodal residency")
        successful_image_cleanup = require_zero_transient_ownership(
            image_snapshot, "successful image B"
        )

        shared_text_hash = chat_content_hash(chat_completion(
            port,
            shared_text_prompt,
            args.predict,
            args.request_timeout,
            cache_prompt=True,
        ))
        post_media_snapshot = resources(port)
        post_media_state = require_transient_policy(
            post_media_snapshot, "shared", mtp_bytes, multimodal_bytes
        )
        if transient_resident_owner(post_media_state) != "mtp":
            raise RuntimeError("text C did not rebuild and commit fresh MTP residency")

        replay_image = multimodal_completion(
            port,
            args.image,
            args.predict,
            args.request_timeout,
            text=image_prompt,
        )
        if chat_content_hash(replay_image) != image_output_hash:
            raise RuntimeError("replayed image B changed deterministic image output")
        replay_image_snapshot = resources(port)
        replay_image_state = require_transient_policy(
            replay_image_snapshot, "shared", mtp_bytes, multimodal_bytes
        )
        if transient_resident_owner(replay_image_state) != "multimodal":
            raise RuntimeError("replayed image B did not commit multimodal residency")
        require_zero_transient_ownership(
            replay_image_snapshot, "successful replayed image B"
        )

        uncached_text_hash = chat_content_hash(chat_completion(
            port,
            shared_text_prompt,
            args.predict,
            args.request_timeout,
            cache_prompt=False,
        ))
        if uncached_text_hash != shared_text_hash:
            raise RuntimeError(
                "A -> image B -> text C changed deterministic prefix parity"
            )
        freshness_snapshot = resources(port)
        freshness_state = require_transient_policy(
            freshness_snapshot, "shared", mtp_bytes, multimodal_bytes
        )
        if transient_resident_owner(freshness_state) != "mtp":
            raise RuntimeError("uncached text C did not leave fresh MTP resident")
        if content_hash(completion(port, args.predict, args.request_timeout)) != baseline_hash:
            raise RuntimeError("text continuation changed after image handoff recovery")
        shared_reference_snapshot = resources(port)
        shared_reference_state = require_transient_policy(
            shared_reference_snapshot, "shared", mtp_bytes, multimodal_bytes
        )

        policy_report: dict[str, dict] = {}

        # An explicit same-policy request must validate the complete target but
        # must not manufacture a physical replacement when manager and plan agree.
        shared_dry = rebalance_transient(port, "shared", True)
        require_transient_preparation_peak(
            shared_dry, "shared", mtp_bytes, multimodal_bytes, prepares=False
        )
        shared_noop = rebalance_transient(port, "shared", False)
        if (
            shared_noop.get("status") != "committed"
            or shared_noop.get("scope") != "none"
            or shared_noop.get("mutated") is not False
            or shared_noop.get("current_plan") != initial_plan
        ):
            raise RuntimeError("explicit shared-policy reconciliation was not a clean no-op")
        shared_snapshot = resources(port)
        if require_transient_policy(
            shared_snapshot, "shared", mtp_bytes, multimodal_bytes, initial_plan
        ) != shared_reference_state:
            raise RuntimeError("same-policy shared request changed transient ownership")
        policy_report["shared"] = {
            "dry_run": True,
            "commit_scope": "none",
            "round_trip": True,
            "failed_parse_exact_rollback": True,
            "image_committed_multimodal_owner": True,
            "text_rebuilt_fresh_mtp_owner": True,
            "successful_image_zero_leases_and_pins": (
                successful_image_cleanup["active_leases"] == 0
                and successful_image_cleanup["pins"] == 0
            ),
            "handoff_a_sha256": handoff_a_hash,
            "shared_prefix_text_sha256": shared_text_hash,
            "deterministic_prefix_parity": True,
            "media_lease_evidence": shared_media_evidence,
            "multitask_handoff": batch_handoff_evidence,
            "system_prompt_handoff": system_prompt_handoff_evidence,
        }

        if getattr(args, "transient_only", False):
            policy_report["shared"]["concurrent_disjoint_rebalance"] = {
                "capability_split": True,
                "reason": "conventional KV/expert transactions are gated by a separate model",
            }
        else:
            policy_report["shared"]["concurrent_disjoint_rebalance"] = (
                validate_serialized_disjoint_rebalances(
                    args,
                    port,
                    initial_expert,
                    mtp_bytes,
                    multimodal_bytes,
                    projector_modalities,
                    baseline_hash,
                )
            )

        mtp_dry = rebalance_transient(port, "mtp-only", True)
        require_transient_preparation_peak(
            mtp_dry, "mtp-only", mtp_bytes, multimodal_bytes, prepares=True
        )
        mtp_commit = rebalance_transient(port, "mtp-only", False)
        mtp_plan = require_transient_commit(
            mtp_commit, "shared", "mtp-only"
        )
        mtp_snapshot = resources(port)
        mtp_state = require_transient_policy(
            mtp_snapshot, "mtp-only", mtp_bytes, multimodal_bytes, mtp_plan
        )
        require_http_views(
            port,
            args.context,
            args.context,
            mtp_plan,
            "mtp-only",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        blocked_image = multimodal_completion(
            port,
            args.image,
            args.predict,
            args.request_timeout,
            expected_status=400,
        )
        require_disabled_multimodal(blocked_image, "mtp-only")
        if require_transient_policy(
            resources(port), "mtp-only", mtp_bytes, multimodal_bytes, mtp_plan
        ) != mtp_state:
            raise RuntimeError("disabled image request changed mtp-only ownership")
        if content_hash(completion(port, args.predict, args.request_timeout)) != baseline_hash:
            raise RuntimeError("mtp-only policy changed deterministic continuation")
        mtp_restore = rebalance_transient(port, "shared", False)
        mtp_restore_plan = require_transient_commit(
            mtp_restore, "mtp-only", "shared"
        )
        require_transient_policy(
            resources(port), "shared", mtp_bytes, multimodal_bytes, mtp_restore_plan
        )
        policy_report["mtp-only"] = {
            "dry_run": True,
            "commit_scope": "transient-only",
            "round_trip": True,
            "multimodal_rejection_http": 400,
        }

        multimodal_dry = rebalance_transient(port, "multimodal-only", True)
        require_transient_preparation_peak(
            multimodal_dry,
            "multimodal-only",
            mtp_bytes,
            multimodal_bytes,
            prepares=True,
        )
        multimodal_commit = rebalance_transient(port, "multimodal-only", False)
        multimodal_plan = require_transient_commit(
            multimodal_commit, "shared", "multimodal-only"
        )
        multimodal_state = require_transient_policy(
            resources(port),
            "multimodal-only",
            mtp_bytes,
            multimodal_bytes,
            multimodal_plan,
        )
        require_http_views(
            port,
            args.context,
            args.context,
            multimodal_plan,
            "multimodal-only",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        multimodal_response, multimodal_media_evidence = (
            multimodal_completion_observing_resources(
                port,
                args.image,
                args.predict,
                args.request_timeout,
                text=image_prompt,
            )
        )
        if chat_content_hash(multimodal_response) != image_output_hash:
            raise RuntimeError("multimodal-only policy changed deterministic image output")
        if require_transient_policy(
            resources(port),
            "multimodal-only",
            mtp_bytes,
            multimodal_bytes,
            multimodal_plan,
        ) != multimodal_state:
            raise RuntimeError("multimodal-only image request changed exact owner state")
        require_zero_transient_ownership(
            resources(port), "successful multimodal-only image"
        )
        deferred_cancel_evidence = validate_deferred_multimodal_cancel(
            args,
            port,
            multimodal_state,
            multimodal_plan,
            mtp_bytes,
            multimodal_bytes,
        )
        multimodal_restore = rebalance_transient(port, "shared", False)
        multimodal_restore_plan = require_transient_commit(
            multimodal_restore, "multimodal-only", "shared"
        )
        require_transient_policy(
            resources(port),
            "shared",
            mtp_bytes,
            multimodal_bytes,
            multimodal_restore_plan,
        )
        if content_hash(completion(port, args.predict, args.request_timeout)) != baseline_hash:
            raise RuntimeError("multimodal-only round trip changed deterministic continuation")
        policy_report["multimodal-only"] = {
            "dry_run": True,
            "commit_scope": "transient-only",
            "round_trip": True,
            "multimodal_request": True,
            "media_lease_evidence": multimodal_media_evidence,
            "deferred_cancel": deferred_cancel_evidence,
        }

        off_dry = rebalance_transient(port, "off", True)
        require_transient_preparation_peak(
            off_dry, "off", mtp_bytes, multimodal_bytes, prepares=True
        )
        off_commit = rebalance_transient(port, "off", False)
        off_plan = require_transient_commit(off_commit, "shared", "off")
        off_state = require_transient_policy(
            resources(port), "off", mtp_bytes, multimodal_bytes, off_plan
        )
        require_http_views(
            port,
            args.context,
            args.context,
            off_plan,
            "off",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        blocked_image = multimodal_completion(
            port,
            args.image,
            args.predict,
            args.request_timeout,
            expected_status=400,
        )
        require_disabled_multimodal(blocked_image, "off")
        if require_transient_policy(
            resources(port), "off", mtp_bytes, multimodal_bytes, off_plan
        ) != off_state:
            raise RuntimeError("disabled image request changed off-policy ownership")
        off_restore = rebalance_transient(port, "shared", False)
        off_restore_plan = require_transient_commit(off_restore, "off", "shared")
        require_transient_policy(
            resources(port),
            "shared",
            mtp_bytes,
            multimodal_bytes,
            off_restore_plan,
        )
        if content_hash(completion(port, args.predict, args.request_timeout)) != baseline_hash:
            raise RuntimeError("off/shared round trip changed deterministic continuation")
        policy_report["off"] = {
            "dry_run": True,
            "commit_scope": "transient-only",
            "round_trip": True,
            "multimodal_rejection_http": 400,
        }

        # Text and image requests can each select the shared resident owner.
        # Restore whichever owner was captured before the sweep, rather than
        # assuming a load-time module that earlier warmup requests may replace.
        final_transient, normalized_plan = restore_shared_owner(
            port, initial_owner, mtp_bytes, multimodal_bytes
        )
        normalized_snapshot = resources(port)
        require_transient_policy(
            normalized_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            normalized_plan,
        )
        if final_transient != initial_transient:
            raise RuntimeError(
                "transient policy sweep did not restore the exact captured owner state"
            )

        if getattr(args, "transient_only", False):
            return {
                "output_sha256": baseline_hash,
                "image_output_sha256": image_output_hash,
                "policies": policy_report,
                "exact_transient_round_trip": True,
                "capability_split": "transient-only",
                "combined": None,
            }

        before_combined_geometry = kv_geometry(normalized_snapshot)
        before_combined_transient = final_transient
        resident_devices = {
            item["device"]
            for item in require_expert_capacity(
                normalized_snapshot, initial_expert, True
            )["devices"]
            if item["resident_bytes"] > 0
        }
        combined_policy = {
            "mtp": "mtp-only",
            "multimodal": "multimodal-only",
        }[initial_owner]
        combined_dry = rebalance_all(
            port,
            args.target_context,
            target_expert,
            combined_policy,
            True,
        )
        if (
            combined_dry.get("target_plan", {}).get("context")
            != args.target_context
            or combined_dry.get("target_plan", {}).get("transient_policy")
            != combined_policy
        ):
            raise RuntimeError("three-pool dry run did not preserve the requested target")
        require_combined_preparation_peak(combined_dry, include_transient=True)

        combined_commit, commit_samples = rebalance_all_observing_views(
            port,
            args.target_context,
            target_expert,
            combined_policy,
            projector_modalities,
        )
        combined_plan = require_transient_commit(
            combined_commit,
            "shared",
            combined_policy,
            scope="kv-expert-and-transient",
        )
        combined_snapshot = resources(port)
        require_geometry(combined_snapshot, args.target_context, args.context)
        require_expert_capacity(
            combined_snapshot, target_expert, True, resident_devices
        )
        require_serialized_plan(
            combined_snapshot,
            args.target_context,
            target_expert,
            combined_plan,
        )
        require_transient_policy(
            combined_snapshot,
            combined_policy,
            mtp_bytes,
            multimodal_bytes,
            combined_plan,
        )
        require_http_views(
            port,
            args.target_context,
            args.context,
            combined_plan,
            combined_policy,
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )

        combined_restore, restore_samples = rebalance_all_observing_views(
            port,
            args.context,
            initial_expert,
            "shared",
            projector_modalities,
        )
        restored_plan = require_transient_commit(
            combined_restore,
            combined_policy,
            "shared",
            scope="kv-expert-and-transient",
        )
        restored_snapshot = resources(port)
        require_geometry(restored_snapshot, args.context, args.context)
        if kv_geometry(restored_snapshot) != before_combined_geometry:
            raise RuntimeError("three-pool round trip changed KV allocation geometry")
        require_expert_capacity(
            restored_snapshot, initial_expert, True, resident_devices
        )
        require_serialized_plan(
            restored_snapshot, args.context, initial_expert, restored_plan
        )
        restored_transient = require_transient_policy(
            restored_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            restored_plan,
        )
        if restored_transient != before_combined_transient:
            raise RuntimeError("three-pool round trip changed exact transient owner state")
        require_http_views(
            port,
            args.context,
            args.context,
            restored_plan,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        if content_hash(completion(port, args.predict, args.request_timeout)) != baseline_hash:
            raise RuntimeError("three-pool round trip changed deterministic continuation")

        return {
            "output_sha256": baseline_hash,
            "image_output_sha256": image_output_hash,
            "policies": policy_report,
            "exact_transient_round_trip": True,
            "combined": {
                "scope": "kv-expert-and-transient",
                "preparation_peak": combined_dry["preparation_peak"],
                "exact_transient_round_trip": True,
                "serialized_plan_verified": True,
                "http_capabilities_verified": True,
                "concurrent_view_samples": {
                    "commit": commit_samples,
                    "restore": restore_samples,
                },
            },
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


def validate_transient_failure(
    args: argparse.Namespace,
    failure_kind: str,
    port_offset: int,
) -> dict:
    mib = 1024 * 1024
    mtp_bytes = args.transient_mtp_mib * mib
    multimodal_bytes = args.transient_mmproj_mib * mib
    initial_expert = args.expert_initial_mib * mib
    port = args.port + port_offset
    server = Server(
        args,
        port,
        failure_kind=failure_kind,
        expert_mode=True,
        transient_mode=True,
    )
    try:
        server.wait_ready()
        projector_modalities = get_projector_modalities(port)
        completion(port, args.predict, args.request_timeout)
        before = completion(port, args.predict, args.request_timeout)
        before_hash = content_hash(before)
        before_snapshot = resources(port)
        require_geometry(before_snapshot, args.context, args.context)
        require_expert_capacity(before_snapshot, initial_expert, True)
        before_plan = require_serialized_plan(
            before_snapshot, args.context, initial_expert
        )
        before_transient = require_transient_policy(
            before_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            before_plan,
        )
        before_owner = transient_resident_owner(before_transient)
        target_policy = {
            "mtp": "multimodal-only",
            "multimodal": "mtp-only",
        }[before_owner]
        before_state = exact_transaction_state(before_snapshot)

        failure = rebalance_transient(
            port, target_policy, False, expected_status=500
        )
        immediate_snapshot = resources(port)
        if exact_transaction_state(immediate_snapshot) != before_state:
            raise RuntimeError(
                f"{failure_kind} changed exact transient state before recovery"
            )
        require_geometry(immediate_snapshot, args.context, args.context)
        require_expert_capacity(immediate_snapshot, initial_expert, True)
        require_transient_policy(
            immediate_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            before_plan,
        )
        require_http_views(
            port,
            args.context,
            args.context,
            before_plan,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        request_json(f"http://127.0.0.1:{port}/health")
        recovery_dry_run = rebalance_transient(port, target_policy, True)
        require_transient_preparation_peak(
            recovery_dry_run,
            target_policy,
            mtp_bytes,
            multimodal_bytes,
            prepares=True,
        )
        require_rollback_message(failure, failure_kind)
        if content_hash(completion(port, args.predict, args.request_timeout)) != before_hash:
            raise RuntimeError(
                f"{failure_kind} changed deterministic output after rollback"
            )

        retry = rebalance_transient(port, target_policy, False)
        retry_plan = require_transient_commit(
            retry, "shared", target_policy, scope="transient-only"
        )
        require_geometry(resources(port), args.context, args.context)
        require_transient_policy(
            resources(port),
            target_policy,
            mtp_bytes,
            multimodal_bytes,
            retry_plan,
        )

        restore = rebalance_transient(port, "shared", False)
        restore_plan = require_transient_commit(
            restore, target_policy, "shared", scope="transient-only"
        )
        require_transient_policy(
            resources(port),
            "shared",
            mtp_bytes,
            multimodal_bytes,
            restore_plan,
        )
        final_transient, normalized_plan = restore_shared_owner(
            port, before_owner, mtp_bytes, multimodal_bytes
        )
        final_snapshot = resources(port)
        if final_transient != before_transient:
            raise RuntimeError(
                f"{failure_kind} retry/restore changed exact transient owner state"
            )
        require_geometry(final_snapshot, args.context, args.context)
        require_expert_capacity(final_snapshot, initial_expert, True)
        require_serialized_plan(
            final_snapshot, args.context, initial_expert, normalized_plan
        )
        require_http_views(
            port,
            args.context,
            args.context,
            normalized_plan,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        if content_hash(completion(port, args.predict, args.request_timeout)) != before_hash:
            raise RuntimeError(
                f"{failure_kind} retry/restore changed deterministic output"
            )
        return {
            "injection": failure_kind,
            "output_sha256": before_hash,
            "failure_http": 500,
            "exact_pre_completion_state_restored": True,
            "exact_transient_owner_restored": True,
            "serialized_plan_restored": True,
            "http_capabilities_verified": True,
            "same_process_retry_committed": True,
            "same_process_restore_committed": True,
            "scope": "transient-only",
        }
    finally:
        server.stop()


def validate_transient_combined_failure(
    args: argparse.Namespace,
    failure_kind: str,
    port_offset: int,
) -> dict:
    mib = 1024 * 1024
    mtp_bytes = args.transient_mtp_mib * mib
    multimodal_bytes = args.transient_mmproj_mib * mib
    initial_expert = args.expert_initial_mib * mib
    target_expert = args.expert_target_mib * mib
    port = args.port + port_offset
    server = Server(
        args,
        port,
        failure_kind=failure_kind,
        expert_mode=True,
        transient_mode=True,
    )
    try:
        server.wait_ready()
        projector_modalities = get_projector_modalities(port)
        completion(port, args.predict, args.request_timeout)
        before = completion(port, args.predict, args.request_timeout)
        before_hash = content_hash(before)
        before_snapshot = resources(port)
        require_geometry(before_snapshot, args.context, args.context)
        before_expert = require_expert_capacity(
            before_snapshot, initial_expert, True
        )
        resident_devices = {
            item["device"]
            for item in before_expert["devices"]
            if item["resident_bytes"] > 0
        }
        before_plan = require_serialized_plan(
            before_snapshot, args.context, initial_expert
        )
        before_transient = require_transient_policy(
            before_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            before_plan,
        )
        before_owner = transient_resident_owner(before_transient)
        target_policy = {
            "mtp": "multimodal-only",
            "multimodal": "mtp-only",
        }[before_owner]
        before_geometry = kv_geometry(before_snapshot)
        before_state = exact_transaction_state(before_snapshot)

        failure = rebalance_all(
            port,
            args.target_context,
            target_expert,
            target_policy,
            False,
            expected_status=500,
        )

        # This snapshot is deliberately taken before any recovery completion.
        # It covers KV occupancy, expert residency, transient owner pointers and
        # enablement/budgets, and the serialized logical plan.
        immediate_snapshot = resources(port)
        immediate_state = exact_transaction_state(immediate_snapshot)
        if immediate_state != before_state:
            raise RuntimeError(
                f"{failure_kind} changed exact three-pool state before recovery: "
                f"before={before_state}, after={immediate_state}"
            )
        require_geometry(immediate_snapshot, args.context, args.context)
        require_expert_capacity(immediate_snapshot, initial_expert, True)
        require_transient_policy(
            immediate_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            before_plan,
        )
        require_http_views(
            port,
            args.context,
            args.context,
            before_plan,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        request_json(f"http://127.0.0.1:{port}/health")
        recovery_dry_run = rebalance_all(
            port,
            args.target_context,
            target_expert,
            target_policy,
            True,
        )
        require_combined_preparation_peak(
            recovery_dry_run, include_transient=True
        )
        require_rollback_message(failure, failure_kind)

        if content_hash(completion(port, args.predict, args.request_timeout)) != before_hash:
            raise RuntimeError(
                f"{failure_kind} changed deterministic output after rollback"
            )
        recovered_snapshot = resources(port)
        require_geometry(recovered_snapshot, args.context, args.context)
        require_expert_capacity(recovered_snapshot, initial_expert, True)
        require_transient_policy(
            recovered_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            before_plan,
        )

        # The injected fault is one-shot. Commit the same three-pool target in
        # this process, restore all logical resources, then normalize the shared
        # owner so the final transient state exactly matches the pre-fault owner.
        retry = rebalance_all(
            port,
            args.target_context,
            target_expert,
            target_policy,
            False,
        )
        retry_plan = require_transient_commit(
            retry,
            "shared",
            target_policy,
            scope="kv-expert-and-transient",
        )
        retry_snapshot = resources(port)
        require_geometry(retry_snapshot, args.target_context, args.context)
        require_expert_capacity(
            retry_snapshot, target_expert, True, resident_devices
        )
        require_serialized_plan(
            retry_snapshot,
            args.target_context,
            target_expert,
            retry_plan,
        )
        require_transient_policy(
            retry_snapshot,
            target_policy,
            mtp_bytes,
            multimodal_bytes,
            retry_plan,
        )
        require_http_views(
            port,
            args.target_context,
            args.context,
            retry_plan,
            target_policy,
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )

        restore = rebalance_all(
            port,
            args.context,
            initial_expert,
            "shared",
            False,
        )
        restore_plan = require_transient_commit(
            restore,
            target_policy,
            "shared",
            scope="kv-expert-and-transient",
        )
        restored_snapshot = resources(port)
        require_geometry(restored_snapshot, args.context, args.context)
        if kv_geometry(restored_snapshot) != before_geometry:
            raise RuntimeError(f"{failure_kind} retry changed KV allocation geometry")
        require_expert_capacity(
            restored_snapshot, initial_expert, True, resident_devices
        )
        require_serialized_plan(
            restored_snapshot, args.context, initial_expert, restore_plan
        )
        require_transient_policy(
            restored_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            restore_plan,
        )

        final_transient, normalized_plan = restore_shared_owner(
            port, before_owner, mtp_bytes, multimodal_bytes
        )
        final_snapshot = resources(port)
        require_transient_policy(
            final_snapshot,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            normalized_plan,
        )
        if final_transient != before_transient:
            raise RuntimeError(
                f"{failure_kind} retry/restore changed exact transient owner state"
            )
        require_geometry(final_snapshot, args.context, args.context)
        require_expert_capacity(
            final_snapshot, initial_expert, True, resident_devices
        )
        require_serialized_plan(
            final_snapshot, args.context, initial_expert, normalized_plan
        )
        require_http_views(
            port,
            args.context,
            args.context,
            normalized_plan,
            "shared",
            mtp_bytes,
            multimodal_bytes,
            projector_modalities,
        )
        if content_hash(completion(port, args.predict, args.request_timeout)) != before_hash:
            raise RuntimeError(
                f"{failure_kind} retry/restore changed deterministic output"
            )
        if require_transient_policy(
            resources(port),
            "shared",
            mtp_bytes,
            multimodal_bytes,
            normalized_plan,
        ) != before_transient:
            raise RuntimeError(
                f"{failure_kind} post-restore text request changed transient ownership"
            )
        return {
            "injection": failure_kind,
            "output_sha256": before_hash,
            "failure_http": 500,
            "exact_pre_completion_state_restored": True,
            "exact_transient_owner_restored": True,
            "serialized_plan_restored": True,
            "http_capabilities_verified": True,
            "same_process_retry_committed": True,
            "same_process_restore_committed": True,
            "scope": "kv-expert-and-transient",
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
    parser.add_argument(
        "--transient-only",
        action="store_true",
        help=(
            "validate MTP/mmproj ownership without conventional KV resizing; "
            "use a separate non-recurrent model for KV/expert transaction evidence"
        ),
    )
    parser.add_argument(
        "--mmproj",
        type=Path,
        help="matching multimodal projector for transient-policy validation",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="JPEG, PNG, or WebP fixture for real transient media requests",
    )
    parser.add_argument(
        "--transient-mtp-mib",
        type=int,
        default=0,
        help="measured peak MTP allocation bound",
    )
    parser.add_argument(
        "--transient-mmproj-mib",
        type=int,
        default=0,
        help="measured peak multimodal-projector allocation bound",
    )
    args = parser.parse_args()
    required_paths = [(args.server, "server"), (args.model, "model")]
    if args.mmproj is not None:
        required_paths.append((args.mmproj, "mmproj"))
    if args.image is not None:
        required_paths.append((args.image, "image"))
    for path, label in required_paths:
        if not path.is_file():
            parser.error(f"--{label} must name an existing file")
    transient_requested = any((
        args.mmproj is not None,
        args.image is not None,
        args.transient_mtp_mib != 0,
        args.transient_mmproj_mib != 0,
    ))
    transient_configured = (
        args.mmproj is not None
        and args.image is not None
        and args.transient_mtp_mib > 0
        and args.transient_mmproj_mib > 0
    )
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
        or (transient_requested and not transient_configured)
        or (args.transient_only and not transient_configured)
        or (transient_requested and (
            args.gpu_layers < 1
            or not args.expert_initial_mib
            or args.image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}
        ))
    ):
        parser.error(
            "invalid context, prediction, expert-cache, or transient validation geometry"
        )

    success = None if args.transient_only else validate_success(args)
    failure = None if args.transient_only else validate_failure(args)
    expert = None
    if args.expert_initial_mib and not args.transient_only:
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
    transient = None
    if transient_requested:
        committed_path = validate_transient_success(args)
        if args.transient_only:
            injected_failures = {
                stage: validate_transient_failure(args, stage, 40 + index)
                for index, stage in enumerate(TRANSIENT_FAILURE_STAGES)
            }
        else:
            injected_failures = {
                stage: validate_transient_combined_failure(args, stage, 40 + index)
                for index, stage in enumerate(
                    COMBINED_FAILURE_STAGES + TRANSIENT_FAILURE_STAGES
                )
            }
        transient = {
            "committed_path": committed_path,
            "evidence_scope": (
                "transient-only" if args.transient_only else "combined-three-pool"
            ),
        }
        if args.transient_only:
            transient["injected_failures"] = injected_failures
        else:
            # Preserve the schema-v3 field consumed by existing combined-gate
            # evidence and downstream report tooling.
            transient["combined_injected_failures"] = injected_failures
    report = {
        "schema": 3,
        "status": "pass",
        "model_sha256": sha256_file(args.model),
        "server_sha256": sha256_file(args.server),
        "configuration": {
            "context": args.context,
            "target_context": args.target_context,
            "gpu_layers": args.gpu_layers,
            "expert_initial_mib": args.expert_initial_mib or None,
            "expert_target_mib": args.expert_target_mib or None,
            "transient_mtp_mib": args.transient_mtp_mib or None,
            "transient_mmproj_mib": args.transient_mmproj_mib or None,
            "transient_only": args.transient_only,
        },
        "committed_path": success,
        "injected_failure": failure,
        "expert_cache": expert,
        "transient": transient,
    }
    if args.mmproj is not None:
        report["mmproj_sha256"] = sha256_file(args.mmproj)
        report["image_sha256"] = sha256_file(args.image)
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
