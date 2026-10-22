# Cache storage aliases and mixed-layout staging

September 4, 2026, local development checkpoint. Changes preserve the existing source work and model/quantization. Evidence is under `../../results/benchmarks/ese-cache-alias-20260904/`, `ese-mixed-layout-20260904/`, and `ese-sustained-nvme-20260904/`.

## Duplicate storage allocation

A Qwen3.6 mixed-Q2K screening run with an 8 GiB RAM cache and 4 GiB expert cache on each of three GPUs ran out of GPU memory. After realizing the caches, the graph allocator requested another 3,619,580,032-byte buffer on GPU 0. Compact graph inputs were unallocated tensor metadata until upload, so graph reservation allocated copies of data already owned by the cache.

Compact inputs now bind to preflighted cache buffers before graph allocation. The cache remains the sole payload owner; upload/compute events continue to protect reuse. Compact copies require an active decode pass and a realized device cache. The regression requires GPU graph scratch below 1 MiB for a small graph with an approximately 8 MiB realized cache, and fails against the prior implementation. The original 4 GiB-per-device configuration then completes.

## Mixed-layout corruption

Inspecting output exposed a separate, pre-existing failure: cached Qwen produced repeated exclamation marks while both resident GPU and CPU-expert modes produced coherent answers. Reconstructing the pre-alias Release backend with all other compiled objects held identical reproduced the same bad output at a 1 GiB cache budget. The alias change did not cause it.

`ggml_active_expert_cache_stage()` selected the required component layout, then route preparation called cache initialization again. Initialization walked the complete layout catalog, selecting its last layout and clearing readiness along the way. Upload offsets subsequently used that layout's expert stride rather than the source tensor's stride. This was a correctness error as well as unnecessary invalidation.

Completed preflight is now keyed to the layout catalog generation. Repeated initialization preserves the active component layout; catalog changes still trigger preflight, released caches clear the stamp, and fully prepared transaction replacements carry the generation they validated.

The CUDA regression combines F16 and F32 experts for the same component at different layers, distinct route-ID tensors, exact diagonal weights, and expert-specific biases. It checks every output against an analytical oracle and an uncached result. It fails before this fix and passes afterward. Reusing the same graph after freeing replaced cache buffers, and after publication rollback, also passes.

The fixed Qwen 32-token screens produce coherent output and identical token IDs with synchronous versus two-worker loading. The earlier approximately 14.7 tok/s repetitive-output screens are **invalid performance evidence** and are excluded from comparisons. The corrected screens are around 2.3–2.6 tok/s in this particular configuration; they are short, have uncontrolled OS-cache reuse, and cannot establish a worker-count speedup.

## Validation boundaries

Release tests pass on the RTX 3060 Ti, RTX 2080 SUPER, and RTX 3080 using CUDA UUIDs. CPU and ASAN/UBSAN logs are retained in the mixed-layout evidence directory. The NVML driver/library mismatch does not prevent these CUDA tests.

The benchmark GGUF is the verified NVMe copy of `Qwen3.6-35B-A3B-mixed-q2k-imat.gguf`, SHA-256 `9b9bdc97b9a4c84dfbd92a9dca78a2905f9181bbb49e0e2d4d1f26b30ff9438c`. No HDD inference is involved. Every benchmark records actual prompt/output counts, configured context, load/TTFT/decode times, per-token gaps, process physical reads, commands, device UUIDs, and binary hashes. OS cache is reused without a reset and is not described as cold NVMe.

Legitimate changes between active layouts still clear component readiness. The fixed short screen has 7,011 metadata hits but 22,140 component uploads and 11.5 GB of H2D payload over 31 decode steps, with no slot evictions. This makes payload preservation across layout transitions a concrete next profiling target. It does not justify dropping synchronization or changing quantization.

Sustained results belong in the accompanying benchmark manifest after output inspection. No faster default, cross-placement numerical tolerance, or production deployment is promoted by this checkpoint.
