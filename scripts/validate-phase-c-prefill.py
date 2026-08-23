#!/usr/bin/env python3
"""Validate bounded expert-prefill staging against the established path.

The report contains hashes and timing/allocator telemetry, never prompt or
generated text.  A nonzero exit means the staged path is not acceptable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


PROMPT = (
    "ESE validates deterministic mixture of experts prefill streaming on three GPUs. "
    "ESE validates deterministic mixture of experts prefill streaming on three GPUs. "
    "Write one short sentence describing a reliable local inference engine."
)
STATS_PREFIX = "expert_prefill_stats: "


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_json(url: str, payload: dict | None = None, timeout: float = 5.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            request_json(f"http://127.0.0.1:{port}/health")
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            time.sleep(0.2)
    raise RuntimeError(f"server did not become ready within {timeout:.0f}s")


def run_server(args: argparse.Namespace, port: int, staging_mib: int) -> dict:
    command = [
        str(args.launcher.resolve()), "serve", str(args.model.resolve()),
        "--binary", str(args.server.resolve()), "--policy", "stream",
        "--context", str(args.context), "--port", str(port),
        "--expert-ram-cache", args.ram_cache,
        "--expert-ram-staging", args.ram_staging,
        "--expert-vram-cache", args.vram_cache,
        "--expert-prefill-staging", f"{staging_mib}MiB",
        "--no-auto-hybrid",
    ]
    if args.tensor_split:
        command.extend(("--tensor-split", args.tensor_split))

    with tempfile.TemporaryFile(mode="w+") as output:
        process = subprocess.Popen(
            command,
            cwd=args.launcher.resolve().parent,
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
        )
        responses: list[dict] = []
        try:
            wait_ready(port, process, args.startup_timeout)
            for _ in range(args.runs):
                response = request_json(
                    f"http://127.0.0.1:{port}/completion",
                    {
                        "prompt": PROMPT,
                        "n_predict": args.predict,
                        "temperature": 0,
                        "seed": 4242,
                        "cache_prompt": False,
                    },
                    timeout=args.request_timeout,
                )
                content = response.get("content")
                timings = response.get("timings", {})
                if not isinstance(content, str) or not content:
                    raise RuntimeError("completion returned no generated content")
                responses.append({
                    "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
                    "prompt_tokens_per_second": timings.get("prompt_per_second"),
                    "decode_tokens_per_second": timings.get("predicted_per_second"),
                })
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            output.seek(0)
            log = output.read()
    if process.returncode != 0:
        raise RuntimeError(f"server exited with code {process.returncode}:\n" + "\n".join(log.splitlines()[-30:]))
    stats = [
        json.loads(line.split(STATS_PREFIX, 1)[1])
        for line in log.splitlines() if STATS_PREFIX in line
    ]
    return {"responses": responses, "stats": stats}


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def summarize(run: dict) -> dict:
    prompt = [item["prompt_tokens_per_second"] for item in run["responses"]]
    decode = [item["decode_tokens_per_second"] for item in run["responses"]]
    if not all(isinstance(value, (int, float)) and value > 0 for value in prompt + decode):
        raise RuntimeError("server returned incomplete timing data")
    return {
        "output_sha256": [item["output_sha256"] for item in run["responses"]],
        "prompt_tokens_per_second": {"samples": prompt, "median": median(prompt)},
        "decode_tokens_per_second": {"samples": decode, "median": median(decode)},
    }


def validate_telemetry(stats: list[dict], staging_mib: int, gpu_count: int) -> dict:
    devices = [item for item in stats if item.get("level") == "device"]
    total = next((item for item in stats if item.get("level") == "total"), None)
    if len(devices) < gpu_count:
        raise RuntimeError(f"expected staging on {gpu_count} GPUs, observed {len(devices)}")
    capacity = staging_mib * 1024 * 1024
    if any(item.get("lanes") != 2 or item.get("allocated_bytes", capacity + 1) > capacity for item in devices):
        raise RuntimeError("device telemetry did not prove the two-lane allocation bound")
    if not total or total.get("selected_components", 0) < 1 or total.get("h2d_bytes", 0) < 1:
        raise RuntimeError("staging telemetry did not prove prompt transfers")
    if total.get("h2d_batches", 0) > total.get("h2d_components", 0):
        raise RuntimeError("H2D batch telemetry is inconsistent")
    if total.get("fallbacks") != 0 or total.get("route_global_sync_fallbacks") != 0:
        raise RuntimeError("staging used a correctness or global-synchronization fallback")
    if total.get("d2d_bytes", 0) < 1:
        raise RuntimeError("warm requests did not prove persistent-cache D2D reuse")
    return {"devices": devices, "total": total}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, default=Path("./ese"))
    parser.add_argument("--server", type=Path, default=Path("build/bin/llama-server"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--staging-mib", type=int, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--tensor-split", default="")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--predict", type=int, default=24)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--ram-cache", default="64MiB")
    parser.add_argument("--ram-staging", default="8MiB")
    parser.add_argument("--vram-cache", default="64MiB")
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--startup-timeout", type=float, default=180)
    parser.add_argument("--request-timeout", type=float, default=180)
    args = parser.parse_args()
    if args.runs < 2 or args.staging_mib < 1 or args.gpu_count < 1:
        parser.error("--runs must be >= 2 and staging/GPU counts must be positive")
    for path, label in ((args.launcher, "launcher"), (args.server, "server"), (args.model, "model")):
        if not path.is_file():
            parser.error(f"--{label} must name an existing file")

    baseline = run_server(args, args.port, 0)
    staged = run_server(args, args.port + 1, args.staging_mib)
    baseline_summary = summarize(baseline)
    staged_summary = summarize(staged)
    if baseline_summary["output_sha256"] != staged_summary["output_sha256"]:
        raise RuntimeError("staged output hashes differ from the established path")
    telemetry = validate_telemetry(staged["stats"], args.staging_mib, args.gpu_count)
    report = {
        "schema": 1,
        "model": str(args.model.resolve()),
        "model_sha256": sha256_file(args.model),
        "server_sha256": sha256_file(args.server),
        "configuration": {
            "runs": args.runs, "context": args.context, "predict": args.predict,
            "staging_mib_per_gpu": args.staging_mib, "gpu_count": args.gpu_count,
            "tensor_split": args.tensor_split or None,
        },
        "baseline": baseline_summary,
        "staged": staged_summary,
        "prompt_speed_ratio": (
            staged_summary["prompt_tokens_per_second"]["median"] /
            baseline_summary["prompt_tokens_per_second"]["median"]
        ),
        "telemetry": telemetry,
        "status": "pass",
    }
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
