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
| Deferred mmap expert storage | Integrated | `stream` policy |
| Route-aware page-cache prefetch | Integrated | enabled by `stream` |
| CPU MoE plus optional GPU-resident tail | Integrated | `hybrid` / `stream` |
| One startup interface | Integrated | `ese doctor/build/plan/serve` |
| GGUF-aware startup policy | Foundation | standard-library planner |
| DeepSeek V4, DSpark, MTP integration-line work | Integrated | native options remain available |
| Maple/TQ2_0 CPU and CUDA work | Integrated | integration-line native path |
| Turbo4/Turbo8 CPU + CUDA row codecs | Foundation | checked internal ABI, exact-byte parity, no host fallback |
| Turbo4/Turbo8 CUDA Flash Attention bridge | Foundation | device-only F16 staging; fused reads still pending |
| Turbo4/Turbo8 per-head padding | Foundation | odd K/V heads padded independently; logical output width restored |
| Adaptive VRAM MoE cache | Planned | Phase 2 / issue #5 |
| Multi-GPU expert-parallel cache | Planned | Phase 2 / issue #5 |
| Core Turbo KV types, CUDA, TCQ, and VBR | In progress | Phase 1 / issue #4 |
| Transient mmproj/MTP sharing | Planned | Phase 3 / issue #6 |
| Adaptive MTP depth / mapped vocabulary | Planned | Phase 3 / issue #6 |
| Global dynamic resource controller | Planned | Phase 4 / issue #7 |

## Phase 1 — KV ladder, TCQ, and VBR

### Fixed Turbo codecs

The first completed slice provides checked deterministic CPU references for Turbo4 and Turbo8 without registering public cache types. See `docs/TURBO_KV_PHASE1.md`.

Next, integrate fixed formats in stages:

1. complete core type traits and checked dispatch;
2. serialization and numeric type-ID compatibility;
3. CUDA encode/decode — row codec complete;
4. Flash Attention compatibility bridge — complete, including unusual-head-
   dimension padding; fused reads remain;
5. lifecycle and quality validation;
6. only then expose stable cache flags.

Turbo3 and Turbo2 require complete correctness references rather than the simplified/zero-fill CPU paths present at the pinned source revision.

Acceptance:

- prove the claimed native path executes;
- deterministic reference vectors;
- full bits/value accounting;
- F16-reference KLD and perplexity;
- prompt-processing and decode measured separately;
- context-depth and one/multi-slot coverage;
- Ampere plus one newer NVIDIA architecture.

### TCQ codecs

Port `turbo3_tcq`, `turbo2_tcq`, and `turbo1_tcq`, including codebook provenance, FWHT/sign rotation, Viterbi encode, O(1) decode, context-adaptive scale, and fused attention paths.

Acceptance additionally requires bit-exact CPU-reference decode, bounded temporary memory, KLD distribution rather than only a mean, and measured policy for compute-poor GPUs.

### Static mixed-tier KV

Before dynamic VBR, support an explicit per-layer/per-side map. K and V may use different tiers; every layer reports its type; checkpoints record tier metadata; resize and rollback tests pass.

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

## Phase 2 — One NVMe → RAM → VRAM expert hierarchy

The current `stream` and static `hybrid` paths should become levels of one bounded native cache.

### Common expert descriptor

Use immutable checked 64-bit descriptors containing layer, expert, component, one or more shard extents, dtype/quant geometry, dimensions/strides/axis, and source identity.

### Bounded RAM cache

Add explicit byte capacity, admission, lease lifetime, deterministic eviction, reusable staging, and mmap/pread/io_uring backends. Correctness and limits cannot depend on an unobservable unlimited OS page cache.

### Adaptive VRAM cache

Consume expert leases from any host/storage backend. Add hysteresis, minimum observations, route frequency, predicted route, reuse distance, load cost, eviction cost, asynchronous promotion/demotion, and event-scoped readiness.

### Multi-GPU expert parallelism

Add per-device capacities/reserves, topology-aware placement, row/tensor distribution, and CPU miss overlap where profitable.

Acceptance:

- exact token, route, intermediate, and output parity;
- forced eviction/churn tests;
- configured RAM and VRAM bounds never exceeded;
- no original-tensor fallback in asserted sidecar-only mode;
- one-, two-, and three-GPU execution;
- cold/warm distributions and structured telemetry.

## Phase 3 — Transient modules and speculation

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
