# Port Roadmap

This roadmap converts the useful ideas from `buun-llama-cpp` into ESE without replacing ESE's defining capability: disk-backed, bounded expert execution.

Reference source for the current analysis:

```text
spiritbuun/buun-llama-cpp
master observed at a2bd802d81936bab8a066cbf789a427776fb4839
MIT license
```

A code port must pin the actual source commit used and preserve attribution. Similar behavior already present independently in ESE should be consolidated rather than duplicated.

## Status key

- **Integrated** — present on the unified core line.
- **Foundation** — public interface or test boundary exists; native implementation remains.
- **Planned** — acceptance gates defined, code not presented as complete.
- **Rejected** — not appropriate for ESE's architecture or failed measurement.

## Consolidation status

| Capability | Status | Unified treatment |
| --- | --- | --- |
| Deferred mmap expert storage | Integrated | `stream` policy |
| Route-aware page-cache prefetch | Integrated | enabled by `stream` |
| CPU MoE plus GPU-resident tail | Integrated | `--gpu-resident-moe` |
| Adaptive VRAM MoE cache | Integrated | `cache` policy |
| Multi-GPU expert-parallel cache | Integrated | automatic on 2+ NVIDIA GPUs |
| DeepSeek V4 Flash support | Integrated | native integration branch |
| DSpark / native MTP paths | Integrated | native options remain available |
| One startup interface | Integrated | `ese doctor/build/plan/serve` |
| GGUF-aware auto-fit startup policy | Foundation | standard-library launcher |
| Global dynamic C++ resource controller | Planned | Phase 2 |
| VBR / Turbo / TCQ KV | Planned | Phase 1 |
| `mmproj` / MTP transient swapping | Planned | Phase 3 |
| Adaptive MTP depth and vocabulary trim | Planned | Phase 3 |

## Phase 1 — KV ladder, TCQ, and VBR

### 1A. Fixed Turbo codecs

Port and validate fixed:

```text
turbo8
turbo4
turbo3
turbo2
```

Required work:

- GGML type declarations and storage accounting;
- CPU reference dequantization;
- CUDA kernels, then ROCm where practical;
- Flash Attention integration;
- odd head-dimension padding;
- conversion/serialization tests;
- fixed-type CLI exposure.

Acceptance:

- no fallback while a native codec is claimed;
- deterministic encode/decode reference vectors;
- KLD, perplexity, prompt-processing, decode, and memory matrix;
- one- and multi-slot coverage;
- at least Ampere plus one newer NVIDIA architecture.

### 1B. TCQ codecs

Port:

```text
turbo3_tcq
turbo2_tcq
turbo1_tcq
```

Required work:

- codebook file format and embedded defaults;
- FWHT/sign-rotation path;
- Viterbi encoder;
- O(1) sliding-window decode;
- context-adaptive norm scale;
- fused attention decode;
- codebook-training utilities isolated from runtime.

Acceptance:

- bit-exact decode against CPU reference;
- no unbounded temporary allocation;
- median and tail KLD reported against F16;
- speed reported separately for prefill and decode;
- compute-poor GPU fallback policy measured, not guessed.

### 1C. Static mixed-tier KV

Before runtime VBR, support an explicit per-layer/per-side tier map. This proves that graph construction, Flash Attention, allocation, and cache indexing can handle heterogeneous storage.

Acceptance:

- K and V may use different tiers;
- every layer/side reports its active type;
- context checkpoints record tier metadata;
- resize and failure rollback tests pass;
- context fill stops cleanly at the calculated limit.

### 1D. Dynamic VBR controller

Implement the quality ladder and model-specific sensitivity order.

Controller inputs:

```text
target context
KV VRAM budget
minimum quality floor
current fill
per-layer sensitivity order
```

Controller outputs:

```text
next layer/side to retier
new storage type
memory released
quality-floor proof
```

Acceptance:

- starts at the highest quality that fits;
- transitions are failure-atomic;
- no token/logit discontinuity beyond the measured codec error;
- multi-slot unified KV is covered;
- checkpoints survive retiering;
- model-specific orders have reproducible KLD evidence;
- generic fallback order is clearly identified.

## Phase 2 — One NVMe → RAM → VRAM expert hierarchy

The existing stream path and RAM-backed MoE cache must become levels of one bounded cache rather than separate modes internally.

### 2A. Common expert descriptor

Use the existing immutable 64-bit sidecar extent model for every backing source:

```text
layer
expert
component: gate/up/down or fused form
one or more shard extents
dtype/quant geometry
dimensions/strides/axis
checksum or source identity
```

### 2B. Bounded RAM cache

Add explicit capacity, admission, lease lifetime, eviction, and telemetry around disk-backed expert materialization. The OS page cache may remain a fast backend, but correctness and limits cannot depend on unobservable global page-cache behavior.

### 2C. VRAM cache over either backing source

Make adaptive GPU residency consume expert leases from:

```text
mmap/page cache
pread
io_uring
RAM cache
```

The cache policy must not care which storage backend supplied a valid expert.

### 2D. Admission policy

Start with a measurable score:

```text
recent route frequency
+ predicted next-route probability
+ reuse-distance benefit
+ load-cost benefit
- eviction cost
- churn penalty
```

Use hysteresis and minimum-observation thresholds. Compare against LRU, LFU, and the existing policy.

### 2E. Async promotion and expert parallelism

- dedicated transfer streams;
- event-scoped readiness;
- no compute-stream global synchronization;
- per-device capacity and topology;
- optional row/tensor distribution across GPUs;
- CPU computation overlaps misses where profitable.

Acceptance for Phase 2:

- total RAM and VRAM remain within configured budgets under forced churn;
- exact token, route, and intermediate parity;
- deterministic forced eviction;
- `pread`, `io_uring`, and mmap backends;
- one-, two-, and three-device execution;
- cold and warm latency distributions;
- no original-tensor fallback while sidecar-only mode is asserted.

## Phase 3 — Transient modules and speculation

### 3A. Generalized transient GPU residency

Adapt the `mmproj`/MTP swap concept into a generic interface for stateless modules:

```text
MTP/draft head
vision projection/encoder
audio encoder
reranker
embedding head
temporary LoRA
```

A swap transaction must reserve capacity, quiesce only the affected streams, move or release the old module, activate the requested module, then restore prior residency if required.

### 3B. Adaptive MTP depth

Choose draft depth from recent acceptance and measured net speed. Disable speculation automatically when target verification plus draft cost loses to ordinary decode.

### 3C. Mapped draft vocabulary

Port the reduced draft-vocabulary idea with full target verification.

Acceptance:

- target output remains byte-identical at temperature zero;
- map provenance and corpus bias are documented;
- unsupported tokens remain impossible for the draft but available to target verification;
- memory and LM-head speed deltas are reported;
- prose, code, tool use, and multilingual prompts are separated.

### 3D. Speculation telemetry

Record:

```text
proposed tokens
accepted tokens
acceptance by depth
draft time
verification time
net tokens/s
disable/retune decisions
```

## Phase 4 — Global resource controller

Move startup-only planning into a native runtime controller.

Budget participants:

```text
dense weights
expert RAM cache
expert VRAM cache
KV
graph workspace
I/O staging
MTP/draft
vision/audio modules
safety reserve
```

The controller chooses among:

- KV quality;
- context capacity;
- expert residency;
- draft residency/depth;
- batch and ubatch limits;
- transient module swaps.

User-facing constraints should be declarative:

```text
--memory-policy auto
--max-ram 40GiB
--reserve-vram 1GiB
--min-kv-quality turbo4
--max-context 128K
```

The runtime must expose the resulting allocation plan before serving requests.

## Phase 5 — Portability and release gates

- CUDA: Ampere, Ada, Hopper, Blackwell policies measured separately;
- ROCm fixed codecs and cache path;
- Windows/MSVC build restoration for supported features;
- Linux `pread` and `io_uring`;
- CPU-only correctness reference;
- API/server regression suite;
- release artifacts and reproducible build metadata.

## Explicit non-goals

- Replacing disk-backed experts with a requirement that the full model fit in RAM.
- Copying all buun changes wholesale into a divergent ik/llama base.
- Publishing VBR/TCQ flags before lifecycle and quality validation.
- Keeping duplicate implementations of the same MoE cache.
- Promoting model-specific benchmarks as universal defaults.
- Hiding native commands or silently changing precision.
