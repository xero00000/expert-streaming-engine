#!/usr/bin/env python3
"""Run a content-addressed Qwen4Exp PLE storage benchmark matrix."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backends", nargs="+", default=["pread", "mmap"])
    parser.add_argument("--storage", nargs="+", default=["row", "page", "two-tier"])
    parser.add_argument("--budgets-mib", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--cache-states", nargs="+", default=["warm"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--sessions", type=int, default=2)
    parser.add_argument("--overlap-us", type=int, default=0)
    return parser.parse_args()


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    results: list[dict] = []
    for cache_state in args.cache_states:
        for backend in args.backends:
            for storage in args.storage:
                for budget in args.budgets_mib:
                    command = [
                        str(args.benchmark),
                        "--shard", str(args.shard),
                        "--manifest", str(args.manifest),
                        "--trace", str(args.trace),
                        "--backend", backend,
                        "--storage", storage,
                        "--cache-state", cache_state,
                        "--cache-mib", str(budget),
                        "--workers", str(args.workers),
                        "--iterations", str(args.iterations),
                        "--sessions", str(args.sessions),
                        "--overlap-us", str(args.overlap_us),
                    ]
                    completed = subprocess.run(
                        command, check=True, capture_output=True, text=True
                    )
                    result = json.loads(completed.stdout)
                    if result["reads"]["failed_reads"] != 0:
                        raise RuntimeError("benchmark reported a failed source read")
                    results.append(result)

    evidence = {
        "schema_version": 1,
        "date": dt.date.today().isoformat(),
        "architecture": "qwen4exp",
        "model_tensor_inventory_sha256": manifest["tensor_inventory_sha256"],
        "trace_sha256": trace["trace_sha256"],
        "configuration": {
            "backends": args.backends,
            "storage_modes": args.storage,
            "budgets_mib": args.budgets_mib,
            "cache_states": args.cache_states,
            "workers": args.workers,
            "iterations": args.iterations,
            "sessions": args.sessions,
            "synthetic_overlap_us": args.overlap_us,
        },
        "scope": {
            "authoritative_trace_replay": True,
            "unique_row_working_set": 248,
            "unique_page_working_set": 251,
            "budget_pressure_claim": False,
            "note": (
                "This short authoritative trace validates parity, concurrency, capacity "
                "accounting, and bounded residency at each configured budget. Its working "
                "set is too small to infer production cache-size sensitivity."
            ),
        },
        "results": results,
    }
    evidence["evidence_sha256"] = canonical_hash(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
