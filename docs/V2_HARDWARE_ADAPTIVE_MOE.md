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
| CPU MoE | Real `ggml_mul_mat_id` over two actual GGUF expert payloads for every discovered type/geometry, checked against both single-thread execution and an independently dequantized scalar matvec; separate and merged fused gate/up layouts have a bit-exact native regression | Yes, only with sufficiently confident samples for every format |
| Adaptive expert-cache upload | Separate page-cache-cold and warm steady-state real split-GGUF payload distributions read through a bounded production RAM-cache lease, uploaded with the production async primitive on every CUDA backend, synchronized while the lease is held, then released | Yes, with complete per-device evidence and the workload A/B gate |
| CPU MoE + cache upload contention | Per-device concurrent model-format routed matvec and warm steady-state exact `pread` → bounded RAM lease → production async upload path, plus separately labeled page-cache-cold upload distributions and live per-layer route/event telemetry | Yes, with complete per-device evidence and the workload A/B gate |

The planner must not consume Phase A data until the last two rows use their production paths and results are keyed by expert GGML type, geometry, GPU, and NUMA node.
Baseline profiles therefore carry `benchmark_source.planner_ready: false`.
Without `--model`, provenance records `model_used: false`. With a GGUF file, first shard, or model directory, ESE scans every shard without loading tensor payloads and uses the representative expert type, geometry, count, and component byte size for the cache-upload probe. The fixed F16 CPU result remains clearly separate and is not presented as model-specific compute calibration.

The native resource planner now has a pure integer split solver for calibrated decode misses. It minimizes the maximum of concurrent CPU and upload completion time, handles all-CPU/all-upload/mixed small counts, retains the previous split within a configurable hysteresis band, and refuses incomplete, low-confidence, non-finite, or otherwise invalid calibration. The launcher applies the same golden-tested completion rule after its stronger topology and profile gate, then passes only a mixed result into live decode.

A native calibration table parser now independently rejects forged readiness, missing or duplicated device/format coverage, incomplete exact-lease evidence, low confidence, and invalid numerical-reference evidence. Accepted entries are keyed by CUDA backend, GGML type, input width, expert width, and component byte size and expose explicit CPU/upload nanoseconds per expert component. A calibrated split entry point fails closed if any component is missing, sums all components required by one routed expert, then applies the integer solver and hysteresis. The launcher owns topology-fingerprint freshness checks and now mirrors the integer solver to select a conservative split across every calibrated GPU and distinct model layer layout. It passes that result to the native decode graph only for a genuinely mixed plan.

Each timed series records its sample count, raw coefficient of variation, robust relative standard error of the median, and a confidence score combining seven-sample coverage with that uncertainty. The median absolute deviation estimator resists isolated scheduler/interrupt outliers but still penalizes broad or bimodal samples. The profile gate requires at least `0.80` confidence for both CPU and upload on every device/format. Changing the provenance flag alone cannot bypass missing model-backed or per-device evidence.

On Linux, cold upload calibration advises the kernel to discard the exact expert
extent with `POSIX_FADV_DONTNEED` before every sample and labels the resulting
series `page-cache-cold`. Platforms that cannot provide that operation report
the cold series as unavailable; the warm planner series remains present and is
the only distribution consumed by automatic split selection.

### Runtime verification gate

Calibration alone cannot activate mixed routing. Before `ese plan` or
`ese serve` adds `--expert-hybrid-gpu-experts`, the exact model, hardware, and
performance-relevant launch configuration must pass a deterministic workload
A/B test:

```sh
./ese validate-hybrid MODEL.gguf --hardware-profile ~/.cache/ese/hardware-profile.json
```

The validator starts separate established-path and hybrid servers, performs
warmups followed by at least three identical seeded decode samples, requires
exact generated-output hashes, and requires the hybrid median to exceed the
established path by at least 2% by default. Passing and failing results are
written atomically with mode `0600` to
`~/.cache/ese/hybrid-verifications.json`. Evidence is bound to sampled model
contents, the full topology fingerprint, and context, KV, batching, threading,
cache, routing, and native-override settings. Prompts and generated text are not
stored.

Approval also requires reconciled live `vram-layer`/`vram-total` telemetry from
the hybrid server, at least three cache misses, mixed CPU/GPU route positions,
and zero forced host-tensor fallbacks. The validator sums lease acquisition,
upload submission, and event-wait time and compares that measured upload path
with the most conservative matching calibrated device/layout prediction. More
than `4x` drift is treated as a contradiction and cannot authorize automatic
routing. Missing, stale, malformed, parity-failed, telemetry-failed, slower, or
contradictory evidence leaves the established path unchanged.

### Remaining Phase A gate

- Add native per-layer CPU-branch compute timing so the validator can compare both sides of the calibrated balance; upload-side route/event contradictions already fail closed.
- Extend the preflight workload window into a periodically refreshed serving-time window before allowing automatic split changes during a long-lived server.
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
in JSON under `hybrid_routing`. A mixed calibrated result is still only a
candidate: it must also pass the model/hardware/configuration-bound workload A/B
gate described above.

The integration also closes three scheduler lifetime/correctness gaps exposed
only by a real model graph: heterogeneous reduction sources cannot assume their
array index is a CUDA backend index, negative route sentinels must be skipped by
the active-expert fallback bitmap, and temporary cache redirects must be
restored before graph metadata is recycled.

The bounded VRAM cache also supports models whose expert tensor type or geometry
changes between layers. Slot capacity is derived from the largest observed
component layout instead of extrapolating from the first layer. Each component
reuses one bounded allocation across its layouts, drains outstanding leases and
compute before switching metadata, and invalidates the affected ready bits.
Compact staging is fail-closed: an initialization failure cannot fall back to
the original full-tensor offsets inside a compact allocation.

The CPU fused-MoE workspace planner now keys activation conversion storage from
the activation tensor itself. The previous condition incorrectly depended on a
separate gate tensor, so merged gate/up layouts could omit required workspace
and corrupt the allocator arena. A native regression executes separate and
merged `ggml_moe_up_gate` graphs and requires bit-exact output parity.

At context shutdown, the scheduler emits one `vram-layer` JSON telemetry record
per exercised MoE layer. It reports route readback latency, active and explicitly
GPU-owned route positions, cache hits and misses, bounded-lease acquisition
latency, lease-backed upload count, upload submission time, event-wait time, and
bytes loaded. These timers observe the existing asynchronous path and do not add
a synchronization point. The aggregate `vram-total` record remains available for
whole-request accounting.

On the local `TinyMoE-100m-2x8-fixed-f16.gguf` fixture, a deterministic 16-token
run produced byte-identical text with and without a one-GPU/one-CPU split. The
single-GPU decode changed from `65.15` to `103.04` tokens/s; async graph scheduling
changed from `44.64` to `58.37` tokens/s. These are development-fixture results,
not published product benchmarks, and primarily prove end-to-end execution and
parity.

A deterministic two-token DeepSeek-V4-Flash `UD-IQ1_S` validation then exercised
43 deferred MoE layers, four distinct expert layouts, three mixed-generation
GPUs, the 4 GiB bounded RAM cache, and a 256 MiB bounded VRAM cache on each GPU.
The established path and a one-GPU-position hybrid split produced byte-identical
output (`SHA-256 62b805db2808dd609730a26b3ed8c9f631274e9071bea28267866d1854372658`).
The hybrid run reported zero forced host-tensor fallbacks and stayed inside each
configured cache bound. Its short-run wall time changed from 65.81 seconds to
4.14 seconds, but this cold, two-token correctness trial is not a publishable
throughput benchmark. Longer steady-state benchmarking and live route/event
comparison against the calibrated prediction remain Phase B gates.

The fail-closed workload validator has also run end-to-end on the local TinyMoE
fixture with three measured four-token samples after warmup. The established
path median was `69.72` tokens/s and the one-GPU-position hybrid median was
`93.25` tokens/s (`1.337x`), with exact output hashes for every paired sample.
All 10 layers reconciled with the aggregate telemetry, 130 of 260 route
positions exercised the GPU partition, all 53 misses accounted for 159
lease-backed component uploads, and forced fallbacks remained zero. Observed
upload-path time was `2.880x` the conservative development-profile prediction,
inside the `4x` fail-closed bound. This uses a tiny fixture and a test profile;
it is gate-development evidence, not a product benchmark.

## Later phases

1. Finish Phase B native CPU-branch telemetry, two-sided contradiction handling, and longer workload validation.
2. Phase C: double-buffered prefill streaming.
3. Phase D: live KV/expert/transient resource rebalancing.
4. Phase E: optional aligned FastStore sidecar bound to GGUF identity.
5. Phase F: semantic cache anchors and newer hardware kernels.

Each phase needs correctness tests, local benchmark evidence, documented hardware coverage and limitations, and a clean private review checkpoint before it is proposed for public `main`.
