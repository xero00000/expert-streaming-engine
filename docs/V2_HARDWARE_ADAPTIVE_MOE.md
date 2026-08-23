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

The native resource planner now has a pure integer split solver for calibrated decode misses. It minimizes the maximum of concurrent CPU and upload completion time, handles all-CPU/all-upload/mixed small counts, retains the previous split within a configurable hysteresis band, and refuses incomplete, low-confidence, non-finite, or otherwise invalid calibration. The launcher applies the same golden-tested completion rule after its stronger topology and profile gate, then passes only a mixed result into live decode.

A native calibration table parser now independently rejects forged readiness, missing or duplicated device/format coverage, incomplete exact-lease evidence, low confidence, and invalid numerical-reference evidence. Accepted entries are keyed by CUDA backend, GGML type, input width, expert width, and component byte size and expose explicit CPU/upload nanoseconds per expert component. A calibrated split entry point fails closed if any component is missing, sums all components required by one routed expert, then applies the integer solver and hysteresis. The launcher owns topology-fingerprint freshness checks and now mirrors the integer solver to select a conservative split across every calibrated GPU and distinct model layer layout. It passes that result to the native decode graph only for a genuinely mixed plan.

Each timed series records its sample count, raw coefficient of variation, robust relative standard error of the median, and a confidence score combining seven-sample coverage with that uncertainty. The median absolute deviation estimator resists isolated scheduler/interrupt outliers but still penalizes broad or bimodal samples. The profile gate requires at least `0.80` confidence for both CPU and upload on every device/format. Changing the provenance flag alone cannot bypass missing model-backed or per-device evidence.

### Remaining Phase A gate

- Resolve and test the standalone fused `ggml_moe_up_gate` probe; its first calibration harness exposed allocator corruption, so Phase A currently uses the stable routed `ggml_mul_mat_id` primitive.
- Instrument the live scheduler route/events around the exact leased upload path.
- Keep the launcher and native calibrated split solvers covered by shared golden cases as the controller evolves.

## Phase B: heterogeneous decode execution

The graph scheduler now supports a one-CUDA-backend-plus-CPU parallel split and
an ESE reduction with an explicit destination branch. CPU results are copied to
the destination backend only after the CPU split completes; CUDA work remains
asynchronous until the reduction dependency is reached. Scheduler workers use
ordinary threads whenever a CPU split is present so the CPU backend retains its
own OpenMP worker team.

The native CUDA test constructs two real routed `ggml_mul_mat_id` branches: a
full host expert tensor on CPU and a compact one-expert tensor on CUDA. Negative
route sentinels mask positions not owned by each branch, and the explicit CUDA
reduction reconstructs the unsplit result. GPU-owned positions are bit-exact and
the complete hybrid output must remain within `0.005` NRMSE of the unsplit CUDA
reference. A CPU-only build also compiles and runs the same test suite, skipping
only the unavailable CUDA path.

The llama decode graph can now build complementary CPU and GPU branches for a
one-token input graph. `--expert-hybrid-gpu-experts N` assigns the first `N`
top-k route positions to the compact CUDA-cache branch and masks those
positions from the full host-weight branch; the explicit reduction reconstructs
the layer result on the GPU. The option is disabled by default and requires a
nonzero bounded expert VRAM cache. Prompt graphs remain on their established
path even when their final layer is collapsed to one requested output.

`ese plan` and `ese serve` automatically inspect the topology-bound profile at
`~/.cache/ese/hardware-profile.json`. The launcher requires exact GGUF component
geometry, complete per-device evidence, and model top-k metadata before adding
the native split option. Automatic activation also requires the calibrated
`pread` bounded-lease storage path; stream mode uses it by default, while cache
mode must select it explicitly. `--hardware-profile PATH` selects another profile and
`--no-auto-hybrid` opts out. Missing, stale, incomplete, all-CPU, and all-GPU
solutions leave the established execution path unchanged and report the reason
in JSON under `hybrid_routing`.

The integration also closes three scheduler lifetime/correctness gaps exposed
only by a real model graph: heterogeneous reduction sources cannot assume their
array index is a CUDA backend index, negative route sentinels must be skipped by
the active-expert fallback bitmap, and temporary cache redirects must be
restored before graph metadata is recycled.

On the local `TinyMoE-100m-2x8-fixed-f16.gguf` fixture, a deterministic 16-token
run produced byte-identical text with and without a one-GPU/one-CPU split. The
single-GPU decode changed from `65.15` to `103.04` tokens/s; async graph scheduling
changed from `44.64` to `58.37` tokens/s. These are development-fixture results,
not published product benchmarks, and primarily prove end-to-end execution and
parity. Larger-model numerical parity and production benchmark selection remain
Phase B gates.

## Later phases

1. Complete Phase B live native CPU/GPU hybrid decode co-execution.
2. Phase C: double-buffered prefill streaming.
3. Phase D: live KV/expert/transient resource rebalancing.
4. Phase E: optional aligned FastStore sidecar bound to GGUF identity.
5. Phase F: semantic cache anchors and newer hardware kernels.

Each phase needs correctness tests, local benchmark evidence, documented hardware coverage and limitations, and a clean private review checkpoint before it is proposed for public `main`.
