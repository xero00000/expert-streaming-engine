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
and zero forced host-tensor fallbacks. The native scheduler also times every
explicitly tagged CPU hybrid branch without adding a GPU synchronization point.
The validator compares both CPU compute and the sum of lease acquisition,
upload submission, and event-wait time with the most conservative matching
calibrated device/layout predictions. More than `4x` drift on either path is
treated as a contradiction and cannot authorize automatic routing. Missing,
stale, malformed, parity-failed, telemetry-failed, slower, or contradictory
evidence leaves the established path unchanged.

When heterogeneous devices disagree and the conservative global solver chooses
an established endpoint, advanced users may evaluate a specific mixed split
with `--hybrid-candidate N`. This never overrides calibration readiness and
never directly activates routing: that exact candidate still needs matching
schema-v3 workload evidence, and normal automatic planning remains
conservative.

### Long-lived serving guard

Passing preflight evidence now supplies its conservative CPU and upload costs to
the native scheduler. The scheduler evaluates independent rolling windows after
outstanding upload events have completed, including the asynchronous
multi-backend path. A forced host-tensor fallback or CPU/upload drift above the
same `4x` bound permanently revokes mixed routing for that process. Subsequent
graphs use the already-reserved established path; noisy recovery cannot
oscillate the process back into hybrid mode. Prefill-only cache traffic is
excluded until an actual mixed CPU branch has executed.

`GET /props` exposes the live state as `disabled`, `monitoring`, or `revoked`
with the revocation reason and calibrated bounds. A validator run that observes
revocation is rejected even when its output parity and median speedup would
otherwise pass. The launcher/native solvers and guard parameter boundary remain
covered by shared surface and golden tests as the controller evolves.

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
latency, lease-backed upload count, upload submission time, event-wait time,
bytes loaded, tagged CPU-branch compute time, and CPU-branch call count. These
timers observe the existing asynchronous path and do not add a GPU
synchronization point. The aggregate `vram-total` record remains available for
whole-request accounting.

On the local `TinyMoE-100m-2x8-fixed-f16.gguf` fixture, a deterministic 16-token
run produced byte-identical text with and without a one-GPU/one-CPU split. The
single-GPU decode changed from `65.15` to `103.04` tokens/s; async graph scheduling
changed from `44.64` to `58.37` tokens/s. These are development-fixture results,
not published product benchmarks, and primarily prove end-to-end execution and
parity.

A schema-v3 DeepSeek-V4-Flash `UD-IQ1_S` validation exercised 43 deferred MoE
layers, four distinct expert layouts, three mixed-generation GPUs, the 4 GiB
bounded RAM cache, and a 256 MiB bounded VRAM cache on each GPU. Three seeded,
two-token samples compared the established path with a one-GPU-position hybrid
candidate. Every corresponding output hash matched. Median generation changed
from `0.394` to `2.968` tokens/s (`7.54x`). The accepted hybrid run recorded 344
route positions split evenly between CPU and GPU, 115 misses, 502 lease uploads,
172 explicitly timed CPU calls, and zero forced fallbacks. Observed CPU and
upload costs were respectively `0.702x` and `0.388x` their conservative bounds.
This is real model-backed gate evidence for the tested machine and launch
signature, not a general product benchmark.

The fail-closed workload validator has also run end-to-end on the local TinyMoE
fixture with five measured 16-token samples after warmup. The established path
median was `52.04` tokens/s and the one-GPU-position hybrid median was
`88.79` tokens/s (`1.706x`), with exact output hashes for every paired sample.
All 10 layers reconciled with the aggregate telemetry, 910 of 1,820 route
positions exercised the GPU partition, all 369 misses accounted for 1,107
lease-backed component uploads, all 910 CPU branch calls were timed, and forced
fallbacks remained zero. Observed CPU and upload-path time were respectively
`0.228x` and `2.092x` their conservative development-profile predictions,
inside the `4x` fail-closed bounds. This uses a tiny fixture and a test profile;
it is gate-development evidence, not a product benchmark.

A later one-GPU RTX 3080 calibration intentionally provided a negative live
guard trial. `/props` began in `monitoring`; after 70 timed CPU branch calls the
observed CPU window was `11.61x` its calibrated prediction and the native guard
changed to `revoked` with reason `cpu-drift`. A second request stayed revoked,
and shutdown telemetry showed 70 GPU route positions—all before the first
request's revocation—while the later request used the established path. Output parity still
matched and the three-sample hybrid median was `1.034x`, demonstrating that a
short apparent speed win cannot bypass contradictory serving telemetry.

## Phase C: bounded double-buffered prefill streaming

Prompt MoE layers can now stage only their selected expert components into two
fixed CUDA lanes per participating device. The lanes are separate from the
persistent decode cache and are sized from the largest complete expert layer.
Layer `N+1` uses a dedicated transfer stream while layer `N` computes; explicit
transfer and compute events fence lane reuse without a backend-wide CUDA
synchronization. Adjacent host components are coalesced, persistent-cache hits
use device-to-device copies, and bounded RAM leases remain held until their
transfer event completes.

The resource planner accounts for the full two-lane allocation independently on
every device. Automatic staging is optional and is dropped before context or KV
quality is reduced. An explicit positive `--expert-prefill-staging` request is
required: an unsupported policy, insufficient planned capacity, allocation
failure, or loss of the configured VRAM reserve fails closed. `0` disables the
feature. The native surface is `--expert-prefill-staging-mib`.

Shutdown telemetry records the allocated lanes per device and per-layer/total
selected components, H2D/D2D components and batches, bytes, lease and transfer
timings, and fallbacks. Acceptance requires two bounded lanes on every requested
device, exact output parity, H2D traffic, warm D2D reuse, coalesced batch counts,
and zero staging or global-synchronization fallbacks. It is automated by:

```sh
scripts/validate-phase-c-prefill.py \
  --model MODEL.gguf --staging-mib SIZE --gpu-count COUNT \
  --tensor-split SPLIT
```

The local three-GPU TinyMoE development fixture exercised RTX 3060 Ti (SM86),
RTX 2080 SUPER (SM75), and RTX 3080 (SM86) with a deliberate `20,30,50` split.
Each device allocated exactly two 13.5 MiB lanes (27 MiB total). Two paired
seeded runs produced identical output hashes. Median prompt throughput changed
from `182.10` to `251.32` tokens/s (`1.38x`); decode stayed effectively neutral
at `30.15` versus `30.75` tokens/s. The staged run selected 363 components,
coalesced 321 H2D components into 216 batches, reused 42 components through D2D
copies, and reported zero fallbacks. This is correctness and development-fixture
evidence, not a general product benchmark.

A later fused-MoE regression trial found that the fuse-down staging pool was
allocated in elements even though its byte-addressed buffer receives full
floating-point rows. Allocating `ggml_nbytes(next)` removes the out-of-bounds
CUDA write. Seeded fused and unfused single-Ampere runs then produced identical
tokens; fused decode measured 136.4 versus 54.6 tokens/s on the fixture. The
default fused path also completed on the mixed SM75/SM86 three-GPU split at
189.7 tokens/s and remained healthy through clean shutdown. These small-model
figures are regression evidence rather than general performance claims.

After compact-cache parity disabled fused `mul_multi_add`, the validator's old
wording exposed a fixture-specific near tie: the first two token probabilities
differed by about `1e-5`, so normal CPU/GPU floating-point differences selected
different tokens even though forced staging fallback reproduced the staged
distribution. The corruption gate now uses a longer stable prompt that still
exercises every layer and requires exact 24-token hashes. On the current v2 tip,
three paired runs passed on one, two, and three participating GPUs with zero
fallbacks or global synchronizations. Median prompt ratios were `1.41x`,
`2.44x`, and `2.91x` respectively; all lanes stayed at exactly 27 MiB per
device and the three-GPU run proved 201 H2D plus 240 D2D component transfers.
These remain development-fixture measurements rather than product benchmarks.

## Phase D: runtime resource rebalancing (in progress)

`GET /v1/ese/resources` is routed through the inference task queue and reports
a coherent owner-thread snapshot without synchronizing a device. It includes
actual KV capacity, occupied cells and allocation bytes; bounded expert-RAM
capacity, residency and active leases; and per-device expert-cache and prefill
staging capacity, allocation and residency alongside the resolved global plan.

`POST /v1/ese/resources/rebalance` can derive a target `context` and
`expert_cache_bytes_per_device`, prove integer accounting and per-device reserve
bounds, reject targets smaller than live KV occupancy, and return current and
target plans. Dry runs also prove the server's immutable load-time KV maximum.
An explicit `"dry_run": false` can commit KV, expert cache, or both resource
classes atomically while every slot is idle and the deferred queue is empty.
The off-to-the-side KV replacement migrates occupied rows before publication;
allocation, conversion, or injected migration failure leaves the old cache and
logical plan active. Per-slot logical limits are changed only after both
requested physical pools finalize.

The KV replacement exposes an opaque reversible transaction through
`llama_kv_cache_prepare_resize()` / `llama_kv_cache_prepare_retier()` and the
publish, rollback, finalize, and free calls in `llama.h`. Prepare owns the
candidate allocation and completes every row migration; publication is a
no-allocation storage/scalar swap; rollback applies the same swap in reverse;
and finalization alone releases the old checkpoint and retires the old buffers.
Freeing a prepared handle drops its candidate, while freeing a published but
unfinalized handle rolls it back. This lets the resource controller prepare KV
and expert pools before publishing either. `ESE_KV_TRANSACTION_FAIL_AFTER_PUBLISH=1`
exercises the gap between publication and finalization, proving that a later
resource publication can fail without committing only the KV half.

The per-device expert cache now has an equivalent prepared replacement. It
allocates every backend layout at the target bound while the old cache remains
active, migrates the most recently used resident entries and admission history,
then publishes all devices together. Allocation, metadata-copy, resident-copy,
or injected preparation failure releases the new allocations and restores the
old configuration. Zero disables the cache, and a later nonzero target prepares
it from a persistent value-only layout and route-capacity catalog (never
scratch-graph tensor pointers). Unknown fields, signed/overflowing values,
non-whole per-slot context, reserve violations, and unavailable target
allocations fail closed. Preparation-peak accounting uses realized live cache
allocations, so a same-target repair or legacy slot allocation cannot bypass a
device reserve merely because the serialized plan differs from physical state.

The replacement is exposed as an opaque reversible transaction at both the
GGML scheduler and public llama-context boundaries: `prepare` owns complete
off-side allocations without changing live policy, `publish` performs only
no-throw swaps and scalar assignments, `rollback` restores the old cache and
policy, and `finalize` retires the off-side owner. The llama handle remains
valid after finalization until `free`, while freeing a published handle
automatically rolls it back. Only one transaction may be open for a scheduler
or context. Context teardown releases it before scheduler/backend destruction.
A monotonic layout/route catalog generation rejects a prepared cache if graph
geometry changed before publication, and the legacy one-call replacement
remains a prepare -> publish -> finalize wrapper. The server composes this with
the KV transaction under the shared publication boundary described below.

A local CPU TinyMoE endpoint trial reported the actual 30 MiB KV allocation and
64 MiB bounded RAM tier. After a 1,502-token prompt, a two-token target was
rejected because it would discard occupied cells. Both that failure and a valid
2,048-token dry run left the live capacity and current plan at 4,096 tokens.

The model-backed transaction gate is automated by:

```sh
scripts/validate-phase-d-rebalance.py --model MODEL.gguf
```

Models with recurrent or hybrid KV state intentionally cannot provide the
conventional resize evidence above. For an MTP-capable model in that class,
`--transient-only` runs the full shared/mtp-only/multimodal-only/off request
matrix plus the after-prepare and after-publish transient fault gates, while
skipping conventional KV, expert-only, and three-pool transactions. The report
labels this as capability-split evidence. KV/expert and combined publication
must still pass separately on a non-recurrent MoE model; this mode does not
weaken or bypass the recurrent-cache fail-closed rule.

On the checked TinyMoE fixture, both CPU and mixed RTX 2080 SUPER/RTX 3060
Ti/RTX 3080 (`20,30,50`) trials shrank an occupied 1,024-token F16 cache to 512
tokens and grew it back to 1,024 while all three seeded completions retained the
same output hash. A concurrent 256-token generation kept the endpoint at the
old geometry and received the required HTTP 503 idle-safe-point rejection.
With failure injected after the first migrated row, the endpoint returned HTTP
500, retained both the 1,024-token allocation and logical plan, reproduced the
pre-failure output hash, and remained healthy. `/props`, `/models`, and the live
resource plan tracked each committed geometry. The native lifecycle test also
decodes at the shrunken geometry before regrowing, preventing a cache/graph
context mismatch from escaping the gate.

The same validator can add the expert-cache transaction matrix on CUDA with
`--expert-initial-mib` and `--expert-target-mib`. That gate covers shrink, grow,
disable, re-enable, preserved deterministic output, retained residency, an
injected preparation failure, and continued use of the old engine. The local
single-Ampere and mixed Turing/Ampere gates both passed 8 MiB -> 4 MiB ->
8 MiB replacement, disable -> inference -> re-enable, and exact pre/post
failure accounting. The mixed gate prepared all three devices and additionally
failed after one complete device preparation to prove cleanup of an earlier
candidate. Use `--expert-failure-after-devices 1` for that multi-device gate.
The compact-cache path currently disables the experimental fused
`mul_multi_add` reduction. A seeded TinyMoE model-level gate found that compact
remapping plus that fusion changed the first generated token, while the
unfused reduction reproduced the uncached output exactly. Both bounded byte
capacity and legacy `LLAMA_EXPERT_GPU_CACHE_SLOTS=4` launches are covered by
the parity gate; the fusion must remain off until an equivalent full-model
proof passes.
The private Phase D branch now gives the transient pool an equivalent prepared
policy transaction. `off`, `shared`, `mtp-only`, and `multimodal-only` derive
enabled modules and stable per-device capacity from measured MTP and projector
bounds. Prepare constructs exact off-side MTP/mmproj owners without mutating live
state; publish performs owner swaps; rollback restores the prior policy, budget,
and owners; finalization retires the old owners. Overlapping request leases form
one residency epoch so a successful image request cannot be undone by a failed
concurrent request and paired with stale target/MTP cache state.

The server now accepts every non-empty combination of KV, expert cache, and
transient policy in one owner-thread transaction. It proves the combined
preparation peak, publishes each reversible pool, acquires the logical-plan lock,
finalizes the physical owners, and publishes context, capabilities, and plan as
one reader-visible state. A queued mutation carries the exact source plan and is
rejected before preparation if another commit made that snapshot stale. Media
cache shrink boundaries are also resolved before preparation, preventing an
irreversible KV shrink from cutting through or retaining stale image chunks.

Phase D remains marked in progress until the full model-backed transient matrix
and mixed-GPU evidence from `validate-phase-d-rebalance.py` are recorded on this
private branch; the CPU, failure-injection, sanitizer, and exact MTP-owner gates
are necessary but not substitutes for that final hardware run.

### Combined KV/expert publication contract

The planner now rejects a target unless the current live allocation plus every
changed replacement pool fits below each immutable device reserve. Dry-run and
commit responses expose the per-device preparation peak and remaining
headroom. Final-fit accounting alone is intentionally insufficient because the
old pools remain live until the transaction is irreversible.

The combined implementation is split into reversible pool transactions. Each
pool has four states: prepare off-side, publish by ownership swap, roll back by
the inverse swap, and finalize by retiring the old allocation. The owner thread
prepares both pools, synchronizes their migrations, publishes KV, publishes all
expert devices, then updates context/configuration and the serialized plan as a
single mutex-protected snapshot. No inference task can run between those
operations, and concurrent `/props` readers cannot combine state from two
commits.

Failure injection must cover both preparation and reversible publication:

1. after KV preparation;
2. after expert-cache preparation;
3. after KV publication;
4. after expert device N publication;
5. after both physical pools publish but before logical configuration publish.

Every failure gate retains the old context, per-device cache accounting,
resource-plan JSON, deterministic output hash, and a usable inference engine.
Faults are one-shot so the same server must then commit the target and restore
the initial plan, proving that no opaque transaction owner leaked.

The model-backed gate passed on one RTX 3060 Ti and on the mixed RTX 2080
SUPER/RTX 3060 Ti/RTX 3080 topology. Both ran 1,024 -> 512 -> 1,024 KV and
8 MiB -> 4 MiB -> 8 MiB per-device expert-cache transactions with identical
deterministic output. The mixed run also restored all three devices after one
expert device had published. Concurrent `/props` sampling observed 97 coherent
snapshots in the single-Ampere run and 60 in the mixed run.

## Later phases

1. Complete Phase D live transient-resource rebalancing.
2. Phase E: optional aligned FastStore sidecar bound to GGUF identity.
3. Phase F: semantic cache anchors and newer hardware kernels.

Each phase needs correctness tests, local benchmark evidence, documented hardware coverage and limitations, and a clean private review checkpoint before it is proposed for public `main`.
