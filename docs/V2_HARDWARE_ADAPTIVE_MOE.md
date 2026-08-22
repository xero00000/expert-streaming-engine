# ESE v2: hardware-adaptive MoE

This document is the implementation contract for the v2 hardware-adaptive MoE work. Development occurs in the private v2 repository; completed, independently implemented phases can later be reviewed for public `main`.

## Invariants

- GGUF remains the canonical model format.
- The storage hierarchy remains disk to bounded RAM leases to bounded per-device VRAM caches.
- Calibration must exercise ESE/GGML production primitives. Nominal PCIe rates and synthetic-only estimates cannot drive policy.
- FreeToken is an algorithmic reference, not a source-code dependency. No FreeToken engine, Python serving stack, Triton core, or copied Apache-2.0 implementation enters ESE's MIT codebase.
- A missing probe is reported as `unavailable`; it is never silently replaced by a synthetic value.

## Phase A: hardware calibration

The user-facing commands are:

```sh
./ese build
./ese calibrate
./ese hardware-profile
```

Model-backed calibration defaults to 21 samples per series. Fewer samples can be requested for diagnostics, but fewer than seven can never produce planner-ready evidence.

Profiles default to `~/.cache/ese/hardware-profile.json`. They are versioned, written atomically with mode `0600`, and bound to a fingerprint covering CPU/NUMA topology, GPU UUID/model/device order, driver, PCI bus identity, CUDA compute capability, kernel, and the NVIDIA topology matrix. A changed identity makes the profile stale.

### Measurement maturity

| Measurement | Current source | Planner-ready |
| --- | --- | --- |
| Sustained host copy | native timed `memcpy` | No; baseline only |
| H2D/D2H | `ggml_backend_tensor_set/get` on every detected CUDA backend | No; needs confidence scoring |
| H2D under host-memory contention | Per-device CUDA upload concurrent with host copy | No; baseline only |
| CPU MoE | Real `ggml_mul_mat_id` over two actual GGUF expert payloads for every discovered type/geometry, checked against both single-thread execution and an independently dequantized scalar matvec | No; requires sufficiently confident samples for every format |
| Adaptive expert-cache upload | Warm steady-state real split-GGUF payload read through a bounded production RAM-cache lease, uploaded with the production async primitive on every CUDA backend, synchronized while the lease is held, then released | No; still needs a separately labeled cold distribution and live scheduler route/event timing |
| CPU MoE + cache upload contention | Per-device concurrent model-format routed matvec and warm steady-state exact `pread` → bounded RAM lease → production async upload path | No; still needs a separately labeled cold distribution and the live scheduler's route/event timing |

The planner must not consume Phase A data until the last two rows use their production paths and results are keyed by expert GGML type, geometry, GPU, and NUMA node.
Baseline profiles therefore carry `benchmark_source.planner_ready: false`.
Without `--model`, provenance records `model_used: false`. With a GGUF file, first shard, or model directory, ESE scans every shard without loading tensor payloads and uses the representative expert type, geometry, count, and component byte size for the cache-upload probe. The fixed F16 CPU result remains clearly separate and is not presented as model-specific compute calibration.

The native resource planner now has a pure integer split solver for calibrated decode misses. It minimizes the maximum of concurrent CPU and upload completion time, handles all-CPU/all-upload/mixed small counts, retains the previous split within a configurable hysteresis band, and refuses incomplete, low-confidence, non-finite, or otherwise invalid calibration. It is not connected to live decode until profile confidence and the remaining correctness gates are satisfied.

A native calibration table parser now independently rejects forged readiness, missing or duplicated device/format coverage, incomplete exact-lease evidence, low confidence, and invalid numerical-reference evidence. Accepted entries are keyed by CUDA backend, GGML type, input width, expert width, and component byte size and expose explicit CPU/upload nanoseconds per expert component. The launcher still owns topology-fingerprint freshness checks; live decode consumption remains gated.

Each timed series records its sample count, raw coefficient of variation, robust relative standard error of the median, and a confidence score combining seven-sample coverage with that uncertainty. The median absolute deviation estimator resists isolated scheduler/interrupt outliers but still penalizes broad or bimodal samples. The profile gate requires at least `0.80` confidence for both CPU and upload on every device/format. Changing the provenance flag alone cannot bypass missing model-backed or per-device evidence.

### Remaining Phase A gate

- Resolve and test the standalone fused `ggml_moe_up_gate` probe; its first calibration harness exposed allocator corruption, so Phase A currently uses the stable routed `ggml_mul_mat_id` primitive.
- Instrument the live scheduler route/events around the exact leased upload path.
- Feed only complete, current, sufficiently confident profiles into the native resource controller.

## Later phases

1. Phase B: native CPU/GPU hybrid decode co-execution.
2. Phase C: double-buffered prefill streaming.
3. Phase D: live KV/expert/transient resource rebalancing.
4. Phase E: optional aligned FastStore sidecar bound to GGUF identity.
5. Phase F: semantic cache anchors and newer hardware kernels.

Each phase needs correctness tests, local benchmark evidence, documented hardware coverage and limitations, and a clean private review checkpoint before it is proposed for public `main`.
