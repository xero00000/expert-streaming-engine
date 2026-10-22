# Concurrent RAM-cache fills: development checkpoint

September 4, 2026. Source: `ee86cae61` plus the existing local desktop changes and this uncommitted cache refactor. This is a development correctness checkpoint, not release acceptance or a sustained inference speed claim.

## Implemented behavior

`ram_cache::acquire()` now reserves resident space and a disjoint range of the existing fixed staging arena under the metadata mutex, reads outside the mutex, then copies and publishes. The resident copy also runs outside the mutex. Ready hits can proceed during an unrelated blocked read. Concurrent requests for the same immutable component share one load and its completion/error.

Reserved eviction candidates remain readable while storage is in flight. If a candidate gains a lease, publication finds another admissible region or fails while preserving old bytes. Failed reads release reservations and preserve existing payloads. `erase()` rejects reserved entries; `clear()` rejects active fills. Completion ownership pins the result until coalesced callers acquire their leases. Callers must still finish active operations and leases before destroying the cache, as before.

The physical resident and staging allocations remain fixed. New statistics expose active/peak fills, staging use, reservations, coalescing, failures, and contended lock wait. Reservation bytes overlap resident regions and must not be added as a second physical allocation. Storage duration is aggregate worker time, not exposed token latency or physical SSD traffic. Lock timing runs only on a contended initial acquisition to avoid two clock reads on every warm hit.

This implements cache-side concurrency. It does not add scheduler workers, make the existing synchronous io_uring reader completion-driven, replace the allocator, or provide CUDA-pinned upload buffers.

## Validation

- The controlled blocked-miss/ready-hit regression fails against the pre-edit implementation and passes against this implementation.
- New tests cover independent overlapping fills, bounded staging/backpressure, duplicate success/error coalescing, no read for pinned-capacity rejection, victim leasing during I/O, failed-fill rollback, and multithreaded eviction with leased-byte verification. Twenty stress repetitions passed during development.
- Final CPU focused CTest: 10/10 passed (`qwen4exp|expert-cache|resource-planner`). Final CUDA focused CTest: 2/2 passed (`test-expert-cache`, `test-resource-planner`). Native io_uring executes in the unrestricted environment.
- The model-backed transaction ownership/rollback/finalize/teardown binary passed. This particular lifecycle test constructs CPU contexts even in its CUDA build.
- AddressSanitizer and UndefinedBehaviorSanitizer passed on the final cache source. ThreadSanitizer passed five repetitions with `TSAN_OPTIONS=ignore_noninstrumented_modules=1`. Unsuppressed TSAN reported accesses involving noninstrumented libstdc++ exception ownership and libomp. A standalone standard-library-only exception_ptr reproducer produces the same exception warnings and is archived. This is not a claim of an entirely unsuppressed TSAN pass.

## GPU tests despite NVML mismatch

Direct CUDA driver initialization succeeds with loaded kernel driver 595.84 although nvidia-smi/NVML 595.91 reports a mismatch. Devices were selected by CUDA UUID: RTX 3060 Ti, RTX 2080 SUPER, RTX 3080. No driver changes were made. NVML failure is not used to skip CUDA execution.

The TinyMoE fixed F16 fixture (200,437,152 bytes; full SHA-256 in the artifact JSON) ran 128 requested greedy output tokens, seed 4242, context 2048, microbatch 1, under resident, pread-cache, and io_uring-cache modes for each of one, two, and three GPUs. Cache runs used 8 MiB RAM, 2 MiB staging, 8 MiB VRAM, 64 MiB reserve, and `GGML_CUDA_NO_PINNED=1`. This is a short correctness fixture, not a filled-context or production-quality benchmark.

A paired pre-edit/current comparison completed all nine configurations (18 executions). Each configuration produced identical before/after output bytes. The pre-edit build restores the exact cache header/source and llama.cpp saved at the start of the work, while importing the same unchanged CUDA ggml shared library as the current build. Compiler, commands, binary/source hashes, CUDA UUIDs, stderr and stdout are archived. This isolates the RAM change from existing local GPU edits.

All cached A/B runs performed actual lease uploads and GPU eviction, reported zero forced fallbacks, and stayed within their RAM/VRAM limits. Current runs also ended with zero failed/active fills and no outstanding staging/reservations. Earlier diagnostic 4 MiB runs did not activate GPU expert caching; they are retained separately and are not counted as cache-path validation.

**Existing cross-mode discrepancy remains:** on two GPUs, resident output differs from cached output; on three GPUs, cached output differs from resident output. Both pread and io_uring agree with each other. The exact pre-edit local source reproduces these differences, so they were not introduced by this refactor. Their numerical/correctness cause is not established. The historical v2 two-GPU comparison did not show that difference. Preserve this as an open local-engine investigation; do not turn the fixture into release parity evidence.

## Synthetic performance evidence

The standalone cache microbenchmark uses 32 KiB synthetic payloads, a fixed resident/staging budget, and eleven randomized paired repetitions. These are median total milliseconds, not token throughput. The final run occurred after the build/model checks finished.

| Case | Before ms | After ms | Before / after |
| --- | ---: | ---: | ---: |
| warm-single | 4.376 | 2.616 | 1.67× |
| warm-four | 29.416 | 22.161 | 1.33× |
| fill-single | 3.908 | 4.276 | 0.91× |
| delayed-four | 64.192 | 21.261 | 3.02× |

The four-thread delayed-source case shows approximately 3× useful concurrency. The immediate single-thread fill case is about 9% slower (roughly 0.36 microseconds extra per fill), so reservation bookkeeping is not free. Warm hits improved in this synthetic test after removing unconditional timer calls. These short measurements are directional; they do not establish production inference gains.

The model tests report `peak_active_loads=1`: the existing caller still requests fills serially. Therefore this checkpoint demonstrates the concurrency primitive and preserved before/after outputs, not storage overlap on the production token path.

## Next gates

1. Investigate the pre-existing TinyMoE multi-GPU cross-mode difference using intermediate/logit comparisons, separately from Qwen4's existing placement gate.
2. Trace the actual scheduler's misses, then connect a bounded batch/worker request interface to this cache so known independent payloads can load concurrently. Preserve the failure, lease and byte-budget guarantees above. The synchronous io_uring ring mutex still limits independent reads on a shared source.
3. Measure exposed waiting time and sustained model latency at fixed quality/context and cold/warm cache states. Include admission overhead and buffer copies. Only then evaluate a bounded pinned upload ring and promote measured wins.

## Artifacts and reproduction

Workspace artifacts: `../results/benchmarks/ese-cache-concurrency-20260904/` from the repository root. `checkpoint.json` identifies the final source/binaries. `cache-concurrency-only.patch` contains only this change relative to the saved dirty source, while `before.patch` preserves the pre-existing tracked diff. `gpu-ab.json`, `model-checks.json`, and the per-run logs retain successes and cross-mode failures; `microbench.json` and its standalone source/runner retain the performance experiment.

`baseline-shared-backend/CMakeLists.txt` documents the imported-backend build. The redundant full CUDA rebuild was stopped and its generated directory removed after the shared-backend baseline succeeded. Its diagnostic logs are retained. No installed engine or production launcher was switched.
