# Streaming checkpoint — 2026-10-02

Strata was stopped at the user's request and remains stopped. Desktop only; no publication or architecture-gate changes.

## Actual worker execution verified

Added scripts/validate-expert-workers.py. It preserves CPU expert mappings with GGML_CUDA_NO_PINNED=1, uses strict lease-only reads, requires nonzero uploads/worker submissions and two concurrent fills, checks every requested device's cache accounting, checks RAM/staging bounds and empty teardown state, and compares exact output across backends/worker settings within each placement. Reports record command, relevant environment, CLI/library hashes and raw counters; incomplete runs remain explicitly incomplete.

The Release TinyMoE screen ran 24 processes: 1/2/3 GPUs, pread/io_uring, workers 0/2, two repeats and 128 decoded tokens. All gates passed. Sources/OS caches were not evicted, and the fixture is small; this is actual streaming feature validation, not larger-than-RAM performance certification or cross-placement numerical parity.

Initial median decode t/s:

| Visible GPUs | pread sync | pread 2 workers | io_uring sync | io_uring 2 workers |
| --- | ---: | ---: | ---: | ---: |
| 0 | 50.36 | 48.09 | 50.47 | 49.67 |
| 0,2 | 65.40 | 63.45 | 65.17 | 64.85 |
| 0,1,2 | 31.14 | 29.87 | 31.08 | 30.21 |

Workers remain opt-in. Three cards are slower here despite more aggregate cache: layer execution and transfer boundaries matter. These placements are not compared as equivalent-memory optimization ratios. GPU identifiers are CUDA-visible indices; the run logs retain device discovery information.

## Rejected allocator experiment

A shortcut for first-victim allocations without other reservations avoided building temporary gap trees. Native cache and strict sanitizer tests passed; a further 24-process screen retained exact same-placement outputs and resource bounds. Sequential before/after timing varied by configuration without a consistent win. This was not a randomized binary A/B or sufficient promotion evidence. The shortcut was removed and Release binaries rebuilt. No allocator speedup is claimed. Raw screens remain in ../strata/benchmarks/ese-active-worker-matrix-20261002.json and ese-single-victim-screen-20261002.json. Candidate report hashes differ from final rebuilt libraries.

## Expanded feature validation

- Two-GPU prefill staging acceptance passes with exact output hashes, two bounded lanes per device, H2D and D2D reuse, zero fallback counters.
- Three-GPU live KV/expert transaction acceptance passes, including failure after preparation of a device and partial expert publication, combined rollback paths, deterministic continuation and busy rejection.
- Whole-visible-device native cache CTest passes.
- Strict ASAN/UBSAN cache test passed for the experimental shortcut before removal; earlier checkpoint covers final retained cache behavior. No separate sanitizer rebuild after shortcut removal in this checkpoint.
- Final Release focused cache/resource/parser tests, 85 Python tests and diff whitespace check pass.

Artifacts: ../strata/benchmarks/ese-prefill-2gpu-20261002.json, ese-rebalance-3gpu-20261002.json. The small fixture does not validate arbitrary-model MTP, media, KV quality or long-context soaks.

## Strata ideas: transferable, not universal

Local inspected version: test/strata/Strata-main. docs/MODELS.md explicitly describes one model family, Qwen3.8-Flash-Next variants, not a general architecture engine. docs/HOW_IT_WORKS.md and docs/DETAILS.md describe concurrent CPU experts and GPU cached experts, pinned expert RAM, measured PCIe miss fraction, packed formats, overlapped quantized prefill, MTP verification and prompt lookup. Source core/expert_source.cpp contains lookahead submission and core/verify.cpp uses a CUDA host callback for synchronization. Main engine license is MIT (2026 Niko1221 and contributors); third-party assets/model-derived artifacts retain separate licenses. No Strata source was copied.

Potential broadly useful MoE transfers:

1. Residency-aware CPU/GPU miss partition using actual compute/PCIe costs and event-safe joins. ESE already has calibrated hybrid support, but retaining routes while choosing placement by live residency needs additional validation.
2. Whole-layer quantized prefill transfer/compute overlap and cache-hit D2D reuse. ESE already implements bounded two-lane staging; tune/profile it rather than duplicate it.
3. Keeping RAM and VRAM copies complementary when appropriate, so scarce host memory does not duplicate every GPU expert. Requires transactional ownership/eviction and capacity evidence; not a drop-in cache policy.
4. Bounded causal next-layer storage advice. ESE already has advice queues; prediction usefulness must exceed extra I/O/contended memory costs.
5. Accepted-token speculation on compatible model/tokenizer/state paths. Acceptance and draft memory determine benefit; not all models have MTP.

Not transferable as a universal speed multiplier: Qwen4Exp PLE/DeltaNet/QSA graph, its active-expert geometry, GSQ-RCO quantization layout, architecture-specific state rollback and draft layer. The local Strata docs themselves state that mapped experts missing RAM/VRAM are much slower with a small GPU. Much larger sparse models might benefit from generic scheduling improvements, but physical NVMe/PCIe bandwidth, route reuse, supported operators and active weight bytes set limits. Dense models cannot exploit sparse expert selection in the same way.

Next work should establish a supported larger-model NVMe baseline and measured critical path, finish io_uring syscall fault ownership validation, and resolve existing cached/cold dense output drift. Archival large models currently resolve to external storage; do not call those runs intended NVMe performance evidence without verified staging. No new general model-speed claim is warranted by this checkpoint.
