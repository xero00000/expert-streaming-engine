# ESE Architecture

## Goal

Expert Streaming Engine exposes one inference runtime whose useful capacity is not limited by requiring every routed expert to stay permanently resident in VRAM or RAM.

The public model matches native capabilities:

```text
resident  ordinary GPU offload / native fit
hybrid    experts on CPU, dense tensors and an optional MoE tail on GPU
cache     bounded RAM expert leases feeding adaptive per-device VRAM caches
stream    the same hierarchy with deferred disk-backed experts
```

All policies end in the same `llama-server` binary and API.

## Control plane and data plane

The standard-library `ese` launcher is the control plane. It:

1. reads the GGUF header and scalar metadata;
2. discovers and totals split shards;
3. detects available host RAM and NVIDIA VRAM;
4. selects a policy or honors an explicit one;
5. reserves VRAM headroom for KV/workspace planning;
6. derives a proportional multi-GPU tensor split;
7. prints or executes the exact native command.

It does not read tensor payloads, rewrite a model, or intercept inference.

The C++/CUDA runtime is the data plane. It owns model loading, routing, computation, synchronization, KV, and API service.

## Current memory paths

### Resident

```text
GGUF → native loader → GPU tensors
                       + KV
                       + graph workspace
```

`ese` uses normal `-ngl 99`. When a dense model does not fit the conservative VRAM threshold, it adds the native `--fit` planner rather than inventing a separate offload scheme.

### Hybrid

```text
GGUF → dense/non-expert tensors → GPU(s)
     └→ routed experts          → CPU RAM
                                  └→ optional final MoE layers on GPU
```

The conservative default is `--cpu-moe`. `--gpu-resident-moe N` converts a readable GGUF block count into `--n-cpu-moe BLOCKS-N`. This is static placement, not an adaptive VRAM expert cache.

### Cache and stream

```text
GGUF shard extents (mmap / pread / io_uring)
      │ checked immutable descriptors
      ▼
fixed 64-byte-aligned RAM arena
      │ lease-scoped expert components
      ▼
per-device adaptive VRAM cache
      │ dedicated transfer stream + readiness events
      ▼
CUDA MoE decode
```

`cache` defaults to mmap-backed sources when the model fits the safe RAM
budget. `stream` selects `pread` and `--defer-experts` so correctness and the
RAM bound do not depend on retaining all expert mappings. Both presets use the
same native descriptor, lease, RAM, and VRAM controller. The older mmap
page-cache prefetch remains an explicit optional optimization, not a required
cache level.

The launcher sets `GGML_CUDA_NO_PINNED=1` so CPU MoE does not recreate a
whole-model pinned allocation. RAM capacity and reusable staging are separate
bounds. VRAM capacity and safety reserve are enforced independently on every
participating device, including the route-remap arena.

## Startup policy

Auto mode currently uses transparent conservative rules:

1. A sparse model larger than 90% of available RAM selects `stream`.
2. GPT-OSS exceeding the safe resident-memory budget selects `stream`.
3. A MoE larger than 85% of free VRAM selects `cache` if RAM remains sufficient.
4. A model within 85% of free VRAM selects `resident`.
5. Missing GPU telemetry defaults MoE models to the bounded RAM cache and dense models to the native resident path.

These are startup decisions, not hidden runtime migrations. `ese plan --json` exposes every input and output.

## Native expert hierarchy and global-controller boundary

Phase 2 implements the expert-specific hierarchy:

```text
NVMe/model shards → bounded RAM expert cache → adaptive per-device VRAM cache
```

Admission combines minimum observations, route frequency, prior-route
prediction, reuse distance, and load cost. Deterministic eviction combines LRU
age and ready-component cost. Upload and slot-reuse dependencies use CUDA
events without a global compute-stream synchronization. Prompt-sized MoE work
stays on the established CPU graph, where independent GPU work can overlap;
single-token decode uses compact cache slots.

Phase 4 will place this controller inside a global budget shared with:

Budget participants will include:

```text
dense weights
expert RAM cache
expert VRAM cache
KV cache
MTP/draft model
vision/audio transient modules
graph workspace
I/O staging
per-device safety reserve
```

That global rebalance policy is not yet a launcher promise; only the bounded
expert hierarchy above is implemented in Phase 2.

## Design invariants

### Bounded memory

Every new cache requires an explicit or calculated capacity and a safety reserve. “Use the remaining memory” is not a bound.

### No unsupported flags or silent fallback

The launcher must generate only options present in the consolidated native parser. CI statically checks the policy-specific surface and checks the compiled server help. Accelerated paths must report whether they actually execute.

### Exact storage descriptions

Disk-backed expert tensors use immutable checked 64-bit extents. Validation covers shard bounds, overflow, overlap, alignment, dimensions, strides, axes, missing shards, and duplicate keys.

### Lease-scoped lifetime

Loaded or staged experts remain valid until the consuming graph or synchronized callback completes. Eviction may not invalidate in-flight tensors.

### Failure-atomic reconfiguration

Future cache resize, KV retiering, transient-module swapping, and recurrent-state changes must either complete fully or preserve the prior usable state.

### Deterministic validation

Lossless changes require token, route, logit/intermediate, and output parity where applicable. Lossy KV codecs require KLD, perplexity, task-quality, and speed measurements against F16.

### Observable decisions

Policy, capacities, hits, misses, evictions, bytes read, staging waits, per-device residency, and speculation acceptance must be available as structured telemetry.

## Why VBR/TCQ is staged

VBR and trellis-coded KV touch Flash Attention, cache lifecycle, checkpoints, multi-slot behavior, GPU kernels, and quality. The port is intentionally split into:

1. fixed Turbo scalar codecs;
2. TCQ encode/decode kernels and codebooks;
3. static per-layer/per-side mixed tiers;
4. model sensitivity orders;
5. dynamic VBR retiering;
6. checkpoint, resize, multi-slot, and context-limit lifecycle;
7. global resource-controller integration.

Until those gates pass, `--kv auto` selects only KV types already supported by this engine.

## Unified branch layout

```text
ese                         executable front door
tools/ese.py                GGUF reader, hardware detection, policy planner
tests/test_ese_launcher.py  deterministic control-plane tests
tests/test_native_surface.py parser-surface guard
docs/ESE_*.md               focused user and architecture docs
docs/PORT_ROADMAP.md        native port gates and status
.github/workflows/ese-ci.yml
```

Native directories are not renamed in this consolidation, minimizing conflicts with future ik/llama synchronization.
