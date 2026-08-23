#!/usr/bin/env python3
"""Phase 2 expert-cache acceptance runner.

Runs exact-output storage parity, bounded churn, cold/warm latency summaries,
and optional 1/2/3-GPU sidecar-only checks. Results are emitted as JSON so a
PR gate can retain the evidence without scraping human-formatted tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path


PROMPT = "The answer is"
PROMPT_TIME_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+) ms")
DECODE_TIME_RE = re.compile(r"eval time\s*=\s*([0-9.]+) ms /\s*(\d+) tokens")
STATS_RE = re.compile(r"expert_cache_stats:\s*(\{.*\})")
COMPUTE_CAPABILITY_RE = re.compile(r"^\d+\.\d+$")
REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision() -> dict[str, object]:
    def git(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        raise RuntimeError(f"cannot identify validation source revision: {head.stderr.strip()}")
    tracked = git("status", "--porcelain", "--untracked-files=no")
    untracked = git("ls-files", "--others", "--exclude-standard")
    if tracked.returncode != 0 or untracked.returncode != 0:
        detail = tracked.stderr.strip() or untracked.stderr.strip()
        raise RuntimeError(f"cannot determine validation source state: {detail}")
    # Local validation builds and the downloaded fixture intentionally live in
    # .phase2-* paths. They do not affect the committed source tree.
    relevant_untracked = sorted(
        path for path in untracked.stdout.splitlines()
        if path and not path.startswith(".phase2-")
    )
    tracked_changes = tracked.stdout.splitlines()
    return {
        "commit": head.stdout.strip(),
        "dirty": bool(tracked_changes or relevant_untracked),
        "tracked_changes": tracked_changes,
        "untracked_source": relevant_untracked,
    }


def gpu_inventory() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,compute_cap,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown nvidia-smi error"
        raise RuntimeError(f"cannot inventory GPUs for the requested matrix: {detail}")

    inventory: list[dict[str, object]] = []
    for row in csv.reader(io.StringIO(completed.stdout), skipinitialspace=True):
        if not row:
            continue
        if len(row) != 6:
            raise RuntimeError(f"unexpected nvidia-smi inventory row: {row!r}")
        index, uuid, name, compute_capability, memory_mib, driver = [value.strip() for value in row]
        if not uuid.startswith("GPU-") or not COMPUTE_CAPABILITY_RE.fullmatch(compute_capability):
            raise RuntimeError(f"incomplete GPU architecture evidence from nvidia-smi: {row!r}")
        try:
            parsed_index = int(index)
            parsed_memory = int(memory_mib)
        except ValueError as error:
            raise RuntimeError(f"invalid numeric GPU inventory field: {row!r}") from error
        inventory.append({
            "index": parsed_index,
            "uuid": uuid,
            "name": name,
            "compute_capability": compute_capability,
            "memory_mib": parsed_memory,
            "driver_version": driver,
        })
    if not inventory:
        raise RuntimeError("nvidia-smi returned no GPUs for the requested matrix")
    inventory.sort(key=lambda item: int(item["index"]))
    return inventory


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def run_cli(cli: Path, model: Path, extra: list[str], env: dict[str, str] | None = None) -> dict:
    command = [
        str(cli), "-m", str(model), "-p", PROMPT, "-n", "8", "-s", "4242",
        "--temp", "0", "--no-display-prompt", "--simple-io", "--no-warmup", "-t", "4",
        *extra,
    ]
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    wall_ms = (time.perf_counter() - started) * 1000.0
    output = completed.stderr
    generated_text = completed.stdout.strip()
    prompt_time = PROMPT_TIME_RE.search(output)
    decode_time = DECODE_TIME_RE.search(output)
    stats = []
    for match in STATS_RE.finditer(output):
        stats.append(json.loads(match.group(1)))
    return {
        "returncode": completed.returncode,
        "generated": generated_text or None,
        "prompt_ms": float(prompt_time.group(1)) if prompt_time else None,
        "decode_ms": float(decode_time.group(1)) if decode_time else None,
        "decode_tokens": int(decode_time.group(2)) if decode_time else None,
        "wall_ms": wall_ms,
        "stats": stats,
        "tail": "\n".join((completed.stdout + completed.stderr).splitlines()[-20:]),
    }


def require_success(result: dict, label: str) -> None:
    if result["returncode"] != 0 or result["generated"] is None:
        raise RuntimeError(f"{label} failed:\n{result['tail']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--gpu-counts", default="", help="comma-separated counts, for example 1,2,3")
    args = parser.parse_args()
    if args.runs < 2:
        parser.error("--runs must be at least 2")
    if not args.cli.is_file() or not args.model.is_file():
        parser.error("--cli and --model must name existing files")
    try:
        gpu_counts = [int(value) for value in args.gpu_counts.split(",") if value]
    except ValueError as error:
        parser.error(f"--gpu-counts must contain positive integers: {error}")
    if any(count < 1 for count in gpu_counts) or len(set(gpu_counts)) != len(gpu_counts):
        parser.error("--gpu-counts must contain unique positive integers")
    revision = source_revision()
    if gpu_counts and revision["dirty"]:
        raise RuntimeError(
            "GPU acceptance evidence requires a clean committed source tree; "
            "commit or discard the reported source changes first"
        )
    inventory = gpu_inventory() if gpu_counts else []
    if gpu_counts and max(gpu_counts) > len(inventory):
        raise RuntimeError(
            f"requested up to {max(gpu_counts)} GPUs, but nvidia-smi found {len(inventory)}"
        )

    report: dict[str, object] = {
        "schema": 3,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "cli": str(args.cli),
        "cli_sha256": sha256_file(args.cli),
        "source": revision,
        "runs": args.runs,
        "architecture_coverage": {
            "inventory": inventory,
            "ada_or_newer": "maintainer-waived-unavailable-hardware",
            "waiver_scope": "runtime architecture coverage only; parity, bounds, events, and sidecar-only assertions remain required",
        },
    }
    # The acceptance binary may be CUDA-enabled and the host may expose a
    # heterogeneous multi-GPU set. Keep storage/RAM reference evidence
    # explicitly CPU-only; numbered GPU lanes below own all GPU visibility and
    # offload settings.
    cpu_only = ["-ngl", "0"]
    cpu_env = dict(os.environ)
    cpu_env["CUDA_VISIBLE_DEVICES"] = ""
    baseline = run_cli(args.cli, args.model, cpu_only, cpu_env)
    require_success(baseline, "baseline")
    report["baseline_output"] = baseline["generated"]

    storage: dict[str, object] = {}
    common_cache = [
        "--expert-ram-cache-mib", "8", "--expert-ram-staging-mib", "2",
        "--expert-sidecar-only", "--defer-experts",
    ]
    for backend in ("mmap", "pread", "io_uring"):
        result = run_cli(
            args.cli,
            args.model,
            [*cpu_only, *common_cache, "--expert-storage-backend", backend],
            cpu_env,
        )
        if result["returncode"] != 0 and backend == "io_uring" and "io_uring is unavailable" in result["tail"]:
            storage[backend] = {"status": "unsupported", "reason": "kernel/sandbox denied io_uring"}
            continue
        require_success(result, backend)
        if result["generated"] != baseline["generated"]:
            raise RuntimeError(f"{backend} output differs from baseline")
        ram_stats = next((item for item in result["stats"] if item.get("level") == "ram"), None)
        if not ram_stats or ram_stats["peak_resident_bytes"] > ram_stats["capacity_bytes"]:
            raise RuntimeError(f"{backend} did not prove its RAM bound")
        storage[backend] = {"status": "pass", "stats": ram_stats}
    report["storage_parity"] = storage

    cold_ms: list[float] = []
    warm_ms: list[float] = []
    for _ in range(args.runs):
        result = run_cli(
            args.cli,
            args.model,
            [*cpu_only, *common_cache, "--expert-storage-backend", "pread"],
            cpu_env,
        )
        require_success(result, "latency run")
        if result["generated"] != baseline["generated"]:
            raise RuntimeError("latency run output differs from baseline")
        cold_ms.append(result["prompt_ms"])
        warm_ms.append(result["decode_ms"] / result["decode_tokens"])
    report["latency"] = {
        "cold_prompt": distribution(cold_ms),
        "warm_decode_per_token": distribution(warm_ms),
    }

    gpu_results: dict[str, object] = {}
    for count in gpu_counts:
        selected_gpus = inventory[:count]
        # UUID selection avoids ambiguity between CUDA's performance ordering
        # and the physical indices reported by NVML/nvidia-smi.
        visible = ",".join(str(item["uuid"]) for item in selected_gpus)
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = visible
        # Layer placement gives this compact fixture deterministic ownership on
        # every visible device. Larger expert-parallel models can additionally
        # use graph/row distribution, but TinyMoE's one-token graph otherwise
        # resolves every routed activation to device 0 and cannot prove the
        # per-device cache bounds required by this acceptance matrix.
        multi_gpu = ["-sm", "layer", "-ts", ",".join("1" for _ in range(count))] if count > 1 else []
        gpu_common = [
            "-ngl", "99", "-ub", "1",
            "-ot", ".ffn_.*_exps.=CPU",
            *multi_gpu,
        ]
        gpu_resident = run_cli(args.cli, args.model, [
            "-ngl", "99", "-ub", "1", *multi_gpu,
        ], env)
        require_success(gpu_resident, f"{count}-GPU resident reference")
        gpu_baseline = run_cli(args.cli, args.model, gpu_common, env)
        require_success(gpu_baseline, f"{count}-GPU baseline")
        legacy_env = dict(env)
        legacy_env["LLAMA_EXPERT_GPU_CACHE_SLOTS"] = "4"
        legacy = run_cli(args.cli, args.model, gpu_common, legacy_env)
        require_success(legacy, f"{count}-GPU legacy slot cache")
        legacy_totals = [
            item for item in legacy["stats"] if item.get("level") == "vram-total"
        ]
        if legacy["generated"] != gpu_resident["generated"]:
            raise RuntimeError(
                f"{count}-GPU legacy slot cache output differs from the resident CUDA reference"
            )
        if (
            not legacy_totals
            or legacy_totals[-1]["forced_fallbacks"] != 0
            or legacy_totals[-1]["admissions"] == 0
            or legacy_totals[-1]["uploads"] == 0
        ):
            raise RuntimeError(f"{count}-GPU legacy slot cache did not complete compact staging")
        vram_policy = [
            "--expert-storage-backend", "pread",
            # TinyMoE uses top-2 routing; 4 MiB yields two full expert slots,
            # exercising same-route slot protection as well as forced churn.
            "--expert-vram-cache-mib", "4",
            "--expert-vram-reserve-mib", "256",
            "--expert-cache-min-observations", "2",
        ]
        non_asserted_cache = [item for item in common_cache if item != "--expert-sidecar-only"]
        compact_safety = run_cli(args.cli, args.model, [
            *non_asserted_cache,
            *vram_policy,
            *gpu_common,
        ], env)
        require_success(compact_safety, f"{count}-GPU compact admission safety")
        compact_safety_repeat = run_cli(args.cli, args.model, [
            *non_asserted_cache,
            *vram_policy,
            *gpu_common,
        ], env)
        require_success(compact_safety_repeat, f"{count}-GPU compact admission safety repeat")
        if compact_safety_repeat["generated"] != compact_safety["generated"]:
            raise RuntimeError(f"{count}-GPU compact admission output is not deterministic")
        if compact_safety["generated"] != gpu_resident["generated"]:
            raise RuntimeError(
                f"{count}-GPU compact admission output differs from the resident CUDA reference"
            )
        compact_safety_totals = [
            item for item in compact_safety["stats"] if item.get("level") == "vram-total"
        ]
        # Decode graphs already contain compact expert tensors. Deferring an
        # admission in that graph has no full-tensor fallback and used to abort
        # at staging time. A minimum-observation policy must therefore admit
        # compact routes immediately, with neither a rejection nor fallback.
        if (
            not compact_safety_totals
            or compact_safety_totals[-1]["admissions"] == 0
            or compact_safety_totals[-1]["uploads"] == 0
            or compact_safety_totals[-1]["rejected_admissions"] != 0
            or compact_safety_totals[-1]["forced_fallbacks"] != 0
        ):
            raise RuntimeError(f"{count}-GPU run did not prove compact admission safety")
        result = run_cli(args.cli, args.model, [
            *common_cache,
            *vram_policy,
            *gpu_common,
        ], env)
        require_success(result, f"{count}-GPU")
        result_repeat = run_cli(args.cli, args.model, [
            *common_cache,
            *vram_policy,
            *gpu_common,
        ], env)
        require_success(result_repeat, f"{count}-GPU repeat")
        if result_repeat["generated"] != result["generated"]:
            raise RuntimeError(f"{count}-GPU cache output is not deterministic")
        if result["generated"] != gpu_resident["generated"]:
            raise RuntimeError(
                f"{count}-GPU cache output differs from the resident CUDA reference"
            )
        vram = [item for item in result["stats"] if item.get("level") == "vram"]
        totals = [item for item in result["stats"] if item.get("level") == "vram-total"]
        devices = {item.get("device") for item in vram}
        if (
            not vram
            or not totals
            or len(devices) != count
            or None in devices
            or any(item["resident_bytes"] > item["capacity_bytes"] for item in vram)
            or any(item["route_id_bytes"] <= 0 or item["resident_bytes"] < item["route_id_bytes"] for item in vram)
        ):
            raise RuntimeError(f"{count}-GPU run did not prove VRAM bounds/telemetry")
        required_total_fields = {
            "hits", "misses", "admissions", "evictions", "uploads", "lease_uploads",
            "forced_fallbacks", "rejected_admissions", "transfer_submit_ns", "transfer_wait_ns",
            "route_observations", "route_prediction_matches", "prediction_admission_contributions",
            "reuse_distance_sum", "load_bytes", "eviction_cost_bytes",
        }
        if not required_total_fields.issubset(totals[-1]):
            missing = sorted(required_total_fields.difference(totals[-1]))
            raise RuntimeError(f"{count}-GPU telemetry is missing fields: {missing}")
        if (
            totals[-1]["forced_fallbacks"] != 0
            or totals[-1]["lease_uploads"] == 0
            or totals[-1]["admissions"] == 0
            or totals[-1]["evictions"] == 0
            or totals[-1]["misses"] == 0
            or totals[-1]["route_observations"] == 0
            or totals[-1]["load_bytes"] == 0
            or totals[-1]["eviction_cost_bytes"] == 0
            or totals[-1]["transfer_submit_ns"] == 0
        ):
            raise RuntimeError(f"{count}-GPU run did not prove lease-only cache churn")
        gpu_results[str(count)] = {
            "status": "pass",
            "split_mode": "layer" if count > 1 else "single-device",
            "physical_gpus": selected_gpus,
            "gpu_resident_reference_output": gpu_resident["generated"],
            "gpu_resident_matches_cpu_reference":
                gpu_resident["generated"] == baseline["generated"],
            "host_active_copy_reference_output": gpu_baseline["generated"],
            "legacy_slot_cache": {
                "slots": 4,
                "output": legacy["generated"],
                "total": legacy_totals[-1],
            },
            "compact_admission_output": compact_safety["generated"],
            "cached_output": result["generated"],
            "parity": "exact repeated generation against a fully resident CUDA reference; CPU storage parity is enforced separately",
            "compact_admission_safety": compact_safety_totals[-1],
            "devices": vram,
            "total": totals[-1],
        }
    report["gpu"] = gpu_results

    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
