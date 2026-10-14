# Mixed-layout payload retention: validation checkpoint

September 4, 2026. Candidate implementation with a completed short-context decode comparison; broader promotion remains pending. Benchmark files are in `/home/xero/Desktop/test/results/benchmarks/ese-layout-retention-20260904`.

## Problem and implementation

The corrected pre-retention cache produces coherent model output, but a layer changing a component's tensor layout invalidates unrelated retained weights. The completed Qwen baseline's first synchronous run uploaded 80,331,079,680 payload bytes over 255 decode steps despite zero evictions. Metadata slot hits therefore substantially overstate usable payload hits.

The candidate gives each component a fixed physical slot stride and each entry a stored layout ID. Selecting a layer's tensor metadata changes the reader, while readiness follows the slot's actual payload. Uploads use the source's logical expert size at the fixed destination offset. Up and gate share a physical stride because the fused quantized kernel uses one tensor's stride for both pointers. That stride remains charged to the existing capacity budget.

Cache replacement copies each entry using its stored layout and remaps layout IDs. Publication, rollback, same-graph aliases, and resource statistics follow the resulting allocation. Prefill copies from cache entries use the stored logical payload and physical offset. The `vram-payload` telemetry separates ready hits/misses, layout switches/reloads, and padding upload bytes.

CUDA's normal tensor initialization clears quantization padding. On shared layout aliases, that padding can belong to a retained larger-format expert in the last slot. The cache now reserves the largest backend allocation span across the layout catalog, initializes the allocation once, and owns the final slot's padding clear on upload. `GGML_BACKEND_BUFFER_USAGE_EXPERT_CACHE` distinguishes these persistent buffers from temporary compute buffers, whose CUDA matmul paths can clear padding during execution. Existing enum values are unchanged.

## Evidence completed before the final ownership rebuild

- Fixed-stride analytical CUDA tests cover F16/F32 layout changes, repeated routes without new acquisitions, evictions, cache replacement, rollback, and reused graphs. The frozen baseline fails the retention assertion.
- Packed versus padded expert kernels produce bit-identical results for F16, Q4_0, and Q2_K, including fused up/gate execution. The final test revision also exercises a two-token indexed path and checks that retained tail bytes survive alias initialization and indexed execution. The final suite passes separately on the 3060 Ti, 2080 Super, and 3080.
- The first retention candidate fails a quantized alias-initialization regression that overwrites a retained F32 payload. Explicit cache-owned padding fixes that regression. Logs and the failing library are retained.
- The second candidate matched all teacher-forced logits exactly for NVMe Qwen3.6 35B-A3B (64 steps, 248,320 logits per step) and GPT-OSS 120B (32 steps, 201,088 logits per step), with no non-finite values. See `model-parity.json`. The final explicit buffer-usage build repeated both comparisons with the same exact result (`final-model-parity.log`).
- CPU and ASAN cache tests passed on the preceding candidate. CPU and ASAN tests passed again after the explicit buffer-usage change and the final test edits (`final-test-cpu.log`, `final-test-asan.log`). Final CUDA suites also pass on all three individual devices. The separate multi-GPU publication/rollback gate also passed, including injected failure after one device swap (`final-cuda-multigpu.log`).

These checks establish bounded, specific correctness evidence; they do not establish arbitrary model/task parity, long-context correctness, or sustained speed.

## Sustained comparison contract

`run_sustained_ab.py` compares frozen baseline and candidate libraries with the same Qwen model, source quantization, three GPU UUIDs, 1,024 allocated context, actual prompt fill reported, 256 generated tokens, greedy sampling, 8 GiB expert RAM cache, 64 MiB staging, 4 GiB expert cache per GPU, and zero load workers. Five prompts each receive five paired repeats with deterministic randomized order.

Both versions use absolute `LD_PRELOAD` paths and verified `ldd` output. The matrix fingerprints the kernel, loaded NVIDIA kernel module, CUDA driver library, CPU, physical RAM, and GPU identities, and stops if that configuration changes between runs. Conda binaries contain DT_RPATH; an LD_LIBRARY_PATH-only override is insufficient. Baseline libraries must remain preserved until comparisons finish.

Each run starts a fresh process and cold ESE cache. OS page cache is reused, and physical read_bytes is measured; these are not advertised as cold-storage trials. All source model paths must resolve to an NVMe mount. HDD originals are archival and are not inference inputs.

The runner records per-token gaps, TTFT, prompt processing, generation throughput, physical reads, memory/resource data, transfer telemetry, binary hashes, commands, and linkage. `analyze_sustained_ab.py` checks completed pairs and resource bounds and reports prompt-clustered paired bootstrap intervals plus per-prompt tail latency. Its counter extraction was checked against all ten completed baseline cache runs. Total GPU process-memory high-water mark remains unmeasured; reported cache allocation is not a substitute for that metric. It requires exact paired output token IDs and valid timing identities and stops on errors or repetitive output. Timing and physical-read variation must be interpreted together. Broader filled-context and soak validation remains required before a general release claim.

## Current status

Final Release CUDA rebuild is complete. The final suite passes independently on all three GPUs and with all devices visible. Rebuilt Qwen64 and GPT32 logit comparisons are exact. All 50 sustained A/B runs completed, and all 25 paired output sequences match exactly (`sustained-ab.json`, `sustained-ab-summary.json`).

Median decode throughput is 2.992 tokens/s for the baseline and 9.110 tokens/s for the candidate. The median paired improvement is 3.083x, with a prompt-clustered bootstrap 95% interval of 2.963–3.401x. Median paired P95/P99 token-gap ratios are 0.612/0.799. The median paired payload-upload ratio is 0.1284, an 87.16% reduction. Both versions allocate 12,875,317,248 bytes across their three expert caches; all measured cache/RAM bounds pass. These results apply to this constrained-cache Qwen fixture, not resident inference or long-context performance.

First-token nonregression is unresolved: the median paired ready-TTFT ratio is 1.0309, with a 95% interval of 0.9711–1.0890. It does not establish nonregression within 2%. Physical source reads and run order vary, so a balanced TTFT-only control warms the NVMe file before each process and verifies actual physical prefill reads. The completed control has 20 runs (five prompts, two balanced paired repeats), all with zero physical prefill reads and exact paired outputs. Median paired ready-TTFT ratio is 0.9953, with a bootstrap 95% interval of 0.9858–1.0137, satisfying the predeclared 2% nonregression bound for this warm-source condition. See `ttft-control-summary.json`. Its two-token outputs are not sustained decode evidence, and the result does not settle cold-source TTFT.

The initial final-suite launch incorrectly inherited `GGML_CUDA_NO_PINNED=1`, which disables the small pinned allocation explicitly required by a unit test. The runner configuration was corrected, all per-card suites passed, and no engine change was needed. Model runs retain their pinning safeguard. The failed configuration and logs are preserved under `runner-config-failure/`.
