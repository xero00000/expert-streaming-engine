# Port Roadmap

This roadmap brings the best applicable ideas from `buun-llama-cpp` into ESE without replacing ESE's defining capability: disk-backed expert execution.

Reference analyzed for planning:

```text
spiritbuun/buun-llama-cpp
planning tree observed: a2bd802d81936bab8a066cbf789a427776fb4839
MIT license
```

Phase 1 implementation source pin:

```text
commit: 799e3995cd4f19aa9f6a3fa9fb5b4674422bf0ee
tree:   a2bd802d81936bab8a066cbf789a427776fb4839
file:   ggml/src/ggml-turbo-quant.c
```

A tree SHA and a commit SHA are recorded separately. Every code port must pin the exact source commit it uses, retain attribution, and separate mechanical adaptation from ESE-specific redesign.

## Status

| Capability | Status | Unified treatment |
| --- | --- | --- |
| Deferred expert storage | Integrated | `stream` policy; pread default, mmap optional |
| Route-aware page-cache prefetch | Integrated | opt-in mmap hint for `cache` / `stream` |
| CPU MoE plus optional GPU-resident tail | Integrated | `hybrid` / `cache` / `stream` |
| One startup interface | Integrated | `ese doctor/build/plan/serve` |
| GGUF-aware startup policy | Foundation | standard-library planner |
| DeepSeek V4, DSpark, MTP integration-line work | Integrated | native options remain available |
| Maple/TQ2_0 CPU and CUDA work | Integrated | integration-line native path |
| Fixed Turbo2/3/4/8 CPU + CUDA row codecs | Foundation | checked internal ABI, exact-byte parity, no host fallback |
| Fixed Turbo2/3/4/8 CUDA Flash Attention | Foundation | direct compressed decode reads; device-only F16 staging remains for prefill |
| Fixed Turbo2/3/4/8 per-head padding | Foundation | odd K/V heads padded independently; logical output width restored |
| Turbo1/2/3-TCQ CPU + CUDA | Foundation | substantive Viterbi references, signed FWHT, O(1) state lookup, direct decode reads |
| Adaptive VRAM MoE cache | Validation pending | Phase 2 implementation complete; 1/2/3-GPU runtime gate remains |
| Multi-GPU expert-parallel cache | Validation pending | graph/tensor-distributed matrix in Phase 2 / issue #5 |
| Core Turbo KV types, CUDA, TCQ, and VBR | Phase 1 released | issue #4 closed; internal promotion gates remain explicit |
| Transient mmproj/MTP sharing | Validated | Phase 3 / issue #6; real Qwen image→text swap |
| Adaptive MTP depth / mapped vocabulary | Validated | Phase 3 / issue #6; CPU, Turing, Ampere, five workload panels |
| Global dynamic resource controller | Implemented and locally validated | Phase 4 / issue #7 |

## Phase 1 — KV ladder, TCQ, and VBR

Phase 1 was merged and released as `v0.1.0`. The release establishes the
checked internal foundation described below; it does not waive the later
promotion gates for formats that intentionally remain absent from the public
cache-type parser.

### Fixed Turbo codecs

The fixed-codec foundation provides checked deterministic CPU references for
Turbo2, Turbo3, Turbo4, and Turbo8 without registering public cache types. See
`docs/TURBO_KV_PHASE1.md`.

Next, integrate fixed formats in stages:

1. complete core type traits and checked dispatch;
2. serialization and numeric type-ID compatibility;
3. CUDA encode/decode — row codec complete;
4. Flash Attention integration — direct compressed decode reads complete for
   batches up to eight, including unusual-head-dimension padding; tiled direct
   prefill remains;
5. lifecycle and quality validation;
6. only then expose stable cache flags.

ESE replaces the pinned simplified/zero-fill Turbo2/Turbo3 encoders with
complete deterministic references while retaining the pinned fixed storage
layouts and centroid tables.

Acceptance:

- prove the claimed native path executes;
- deterministic reference vectors;
- full bits/value accounting;
- F16-reference KLD and perplexity;
- prompt-processing and decode measured separately;
- context-depth and one/multi-slot coverage;
- available NVIDIA architecture coverage, with explicit maintainer-approved
  exceptions recorded for unavailable hardware.

### TCQ codecs

Port `turbo3_tcq`, `turbo2_tcq`, and `turbo1_tcq`, including codebook provenance, FWHT/sign rotation, Viterbi encode, O(1) decode, context-adaptive scale, and fused attention paths.

The deterministic CPU references, embedded pinned codebooks, signed FWHT,
free-initial-state Viterbi encoding, O(1) sliding-window state lookup, native
CUDA row codecs, and direct decode-attention paths are complete. Context-
adaptive scale policy and lifecycle metadata remain in progress.

Acceptance additionally requires bit-exact CPU-reference decode, bounded temporary memory, KLD distribution rather than only a mean, and measured policy for compute-poor GPUs.

### Static mixed-tier KV

Before dynamic VBR, support an explicit per-layer/per-side map. K and V may use different tiers; every layer reports its type; checkpoints record tier metadata; resize and rollback tests pass.

The internal policy core now builds strict independent K/V maps, rejects
out-of-range and overlapping assignments, accounts for padded Turbo rows, and
feeds the existing first/last layer overrides into one deterministic plan. An
experimental C API accepts complete K and V layer arrays independently without
adding Turbo/TCQ names to the public CLI parser. Checkpoint records carry the
actual type and row size for every layer. Live retiering now prepares a full
replacement cache, converts occupied rows with bounded per-head host staging,
and publishes only after every conversion succeeds. Injected conversion
failure, multi-slot continuation, resize, shift, defrag, and representation-aware
checkpoint rollback are covered on a conventional attention model. A model-
backed recurrent test decodes successfully and proves retiering fails closed;
hybrid, transposed, MLA, and split layouts remain explicitly unsupported.

### Dynamic VBR

Inputs:

```text
target context
KV VRAM budget
minimum quality floor
current fill
model sensitivity order
```

Transitions must be failure-atomic, preserve multi-slot behavior and checkpoints, never cross the declared floor, and have reproducible quality evidence.

The deterministic solver core now consumes context, byte budget, quality
floor, and a model-provided layer-side sensitivity order. It selects K and V
independently and reports every layer plus actual and budgeted bytes. Measured
model sensitivity tables and live cache-storage retiering remain in progress.

## Phase 2 — One NVMe → RAM → VRAM expert hierarchy

The native hierarchy and the `cache`/`stream` launcher presets are implemented
on the Phase 2 branch. CPU/storage parity, forced churn, compile, server, and
sanitizer gates pass. The one-, two-, and three-GPU Turing/Ampere runtime
matrix remains the final pre-merge gate; Ada-or-newer coverage has a documented
solo-maintainer hardware exception. See
`docs/PHASE2_EXPERT_CACHE_VALIDATION.md`.

### Common expert descriptor

Implemented as immutable checked 64-bit descriptors containing layer, expert,
component, one or more shard extents, dtype/quant geometry,
dimensions/strides/axis, and source/model identity. File identity is rechecked
when the cache opens each source.

### Bounded RAM cache

Implemented with an exact fixed resident arena, explicit staging bound,
lease-scoped lifetime, deterministic aligned first-fit/LRU eviction, and
mmap/pread/io_uring sources. Deferred `stream` operation does not require the
original expert pages to remain resident.

### Adaptive VRAM cache

The per-device cache consumes the same leases from every host/storage backend.
Admission includes hysteresis, minimum observations, route frequency,
prediction, reuse distance, and load cost; eviction includes deterministic age
and ready-byte cost. Dedicated CUDA transfer streams and upload/compute events
provide readiness without global compute synchronization.

### Multi-GPU expert parallelism

Per-device capacities/reserves include route metadata, placement follows the
device owning each activation, and the acceptance runner uses graph split to
exercise row/tensor distribution across every requested GPU. Prompt-sized CPU
MoE remains on the scheduler path where independent device work can overlap.

Acceptance:

- exact token, route, intermediate, and output parity;
- forced eviction/churn tests;
- configured RAM and VRAM bounds never exceeded;
- no original-tensor fallback in asserted sidecar-only mode;
- one-, two-, and three-GPU execution;
- cold/warm distributions and structured telemetry.

## Phase 3 — Transient modules and speculation

Status: validated on issue #6. The failure-atomic controller, real server MTP/mmproj
adapters, adaptive MTP target-only arm, mapped draft-head graph, focused tests,
and structured telemetry are implemented. A Qwen 3.6 image request followed by
text/MTP generation passed on Turing/Ampere hardware with no stale state; the five
workload panels measured a 12.9% aggregate decode gain and 80.8% draft acceptance.

### General transient residency

Share a bounded VRAM budget among MTP/draft heads, vision/audio modules, rerankers, embedding heads, and temporary LoRAs. Transactions must reserve, quiesce only affected streams, swap, restore, and roll back atomically.

### Adaptive MTP depth

Select draft depth from recent acceptance and measured net throughput; automatically reduce or disable speculation when draft plus verification loses.

### Mapped draft vocabulary

Port reduced draft-vocabulary projection while retaining full target verification. Temperature-zero output must remain byte-identical. Map provenance and bias must be documented across code, prose, tools, multilingual, and long-context prompts.

## Phase 4 — Global native resource controller

Budget:

```text
dense weights
expert RAM cache
expert VRAM cache
KV
graph workspace
I/O staging
MTP/draft
transient multimodal modules
safety reserve
```

Target declarative interface:

```text
--memory-policy auto
--max-ram 40GiB
--reserve-vram 1GiB
--min-kv-quality turbo4
--max-context 128K
```

The runtime must print a reproducible allocation plan before serving and rebalance failure-atomically without silent precision or backend fallback.

Implemented in `common/resource-planner.*` and the native model initialization
path. A metadata-only probe supplies real dense/expert geometry, largest expert
component, KV costs, format compatibility, and per-device memory. The solved
plan drives native fit, context/KV types, expert tiers, prompt cache,
batch/ubatch, and transient capacity. `/props` and `/metrics` expose the plan;
release and sanitizer tests cover deterministic presets, multi-device bounds,
one/two-slot lifecycle inputs, and rollback. See
`docs/PHASE4_GLOBAL_RESOURCE_CONTROLLER.md`.

## Phase 5 — Portability and release gates

- architecture-specific CUDA policy for Ampere, Ada, Hopper, and Blackwell;
- ROCm fixed codecs and cache path;
- Windows/MSVC support for promoted features;
- Linux mmap/pread/io_uring;
- CPU reference correctness;
- API/server regression suite;
- reproducible release artifacts.

## Non-goals

- requiring the full model to fit RAM;
- merging all buun changes wholesale into a divergent base;
- publishing VBR/TCQ flags before quality and lifecycle validation;
- keeping duplicate cache implementations;
- promoting one machine's best run as a universal default;
- hiding native commands or silently changing precision.
