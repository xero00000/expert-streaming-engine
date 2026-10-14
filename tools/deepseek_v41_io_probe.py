#!/usr/bin/env python3
"""Bounded synthetic V4.1 storage probe, not an inference benchmark.

Writes and removes one private 512 MiB file. Eviction advice is per-file;
physical reads are recorded because advice does not guarantee a cold cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
import time


def read_bytes():
    return int(dict(line.split(": ") for line in Path("/proc/self/io").read_text().splitlines())["read_bytes"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    args = parser.parse_args()
    expert_bytes = json.loads(args.preflight.read_text())["expert_bundle_bytes_including_scales"]
    size = 512 * 1024 * 1024
    if not 0 < expert_bytes <= size // 2:
        raise ValueError("expert size exceeds bounded probe capacity")
    results = []
    with tempfile.TemporaryFile(dir=args.directory) as file:
        fd = file.fileno()
        # Incompressible material prevents filesystem compression producing a bogus bandwidth result.
        for _ in range(size // (1024 * 1024)):
            file.write(os.urandom(1024 * 1024))
        file.flush()
        os.fsync(fd)
        for repeat in range(3):
            for kind, block_size in (("expert_bundle", expert_bytes), ("engram_page", 4096)):
                offsets = list(range(0, size - block_size + 1, block_size))
                random.Random(4100 + repeat).shuffle(offsets)
                offsets = offsets[:min(4096, len(offsets))]
                # Rotate order to reduce a fixed worker-order bias.
                workers_order = (1, 2, 4)
                workers_order = workers_order[repeat:] + workers_order[:repeat]
                for workers in workers_order:
                    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_RANDOM)
                    def read_one(offset):
                        start = time.perf_counter()
                        data = os.pread(fd, block_size, offset)
                        if len(data) != block_size:
                            raise IOError("short probe read")
                        return time.perf_counter() - start
                    with ThreadPoolExecutor(max_workers=workers) as pool:
                        before = read_bytes()
                        start = time.perf_counter()
                        latencies = list(pool.map(read_one, offsets))
                        elapsed = time.perf_counter() - start
                        physical = read_bytes() - before
                    logical = len(offsets) * block_size
                    results.append({"kind": kind, "workers": workers, "repeat": repeat,
                        "requests": len(offsets), "request_bytes": block_size,
                        "logical_bytes": logical, "process_physical_read_bytes": physical,
                        "elapsed_seconds": elapsed, "logical_GB_per_second": logical / elapsed / 1e9,
                        "read_service_p50_ms": statistics.median(latencies) * 1000,
                        "read_service_p95_ms": sorted(latencies)[int(.95 * (len(latencies) - 1))] * 1000})
    args.output.write_text(json.dumps({"scope": "Synthetic buffered pread, no ESE execution, GPU transfer or routing trace",
        "cache_control": "Per-file POSIX_FADV_DONTNEED; physical reads recorded, device cache uncontrolled",
        "fixture_bytes": size, "fixture_removed": True, "directory": str(args.directory.resolve()),
        "results": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
