# ESE Architecture

## Goal

Expert Streaming Engine should expose one inference runtime whose capacity is determined by the useful memory hierarchy—not by a requirement that every routed expert stay permanently resident in RAM or VRAM.

The public model is deliberately small:

```text
resident  model fits in aggregate VRAM
cache     model fits in RAM; hot experts use spare VRAM
stream    model or expert set exceeds the safe RAM budget
```

All three policies end in the same native `llama-server` binary and API.

## Control plane and data plane

The `ese` launcher is the control plane. It:

1. reads the GGUF header and scalar metadata;
2. discovers and totals split shards;
3. detects available host RAM and NVIDIA VRAM;
4. selects a policy or honors an explicit one;
5. allocates a VRAM safety reserve;
6. derives a proportional multi-GPU tensor split;
7. prints or executes the exact native command.

It never reads tensor payloads, rewrites a model, or intercepts inference.

The C++/CUDA runtime is the data plane. It owns model loading, routing, computation, caches, synchronization, and API service.

## Memory hierarchy

### Resident

```text
GGUF → host loader → GPU tensors
                      + KV
                      + graph workspace
```

Use this when the model fits with enough headroom for KV, scratch, and transient allocations.

### Cache

```text
GGUF → host-resident experts ──────────────┐
                         cache miss → CPU  │
                                          ▼
                                  adaptive VRAM cache
                                          │
                               one or more GPUs
```

Dense and non-expert tensors remain offloaded normally. Routed experts remain addressable in host memory. Spare VRAM is treated as a hot cache rather than left idle. On multiple GPUs, expert-parallel mode distributes resident expert work across eligible devices.

### Stream

```text
GGUF on NVMe
      │
      ├─ deferred mmap
      ▼
bounded OS page cache / staging
      │
router top-k + optional logit tail prefetch
      ▼
CPU MoE + selected GPU-resident MoE layers
```

The model file is the backing store. `GGML_CUDA_NO_PINNED=1` prevents a pinned host allocation from defeating deferred paging. Route-aware prefetch requests only relevant expert ranges. The intended invariant is that the active working set remains bounded even when the complete expert set is larger than RAM.

## Unified planner

Auto policy currently uses transparent conservative rules:

1. A sparse model larger than 90% of available RAM selects `stream`.
2. A MoE model larger than 85% of free VRAM selects `cache`, provided the RAM limit did not already require streaming.
3. A model smaller than 85% of free VRAM selects `resident`.
4. Missing GPU telemetry falls back to `cache` for MoE and `resident` for dense models.

These are startup decisions, not hidden runtime migrations. `ese plan --json` exposes all inputs and the resulting command.

The planned C++ resource controller will eventually rebalance:

```text
dense weights
expert RAM cache
expert VRAM cache
KV cache
MTP/draft model
vision projection
graph workspace
I/O staging
VRAM safety reserve
```

The launcher already provides the stable policy vocabulary and extension boundary for that controller.

## Design invariants

### Bounded memory

Every new cache must have a configured or automatically calculated capacity. “Use the remaining memory” must still retain a safety reserve and fail predictably.

### No silent fallback

A requested accelerated path must report whether it is active. Validation must distinguish native execution from compilation-only coverage or an unnoticed CPU fallback.

### Exact storage descriptions

Disk-backed expert tensors are identified through immutable 64-bit extents. Validation covers shard bounds, overflow, overlap, alignment, dimensions, strides, axes, missing shards, and duplicate keys.

### Lease-scoped lifetime

A loaded or staged expert remains valid until the graph or synchronized callback that consumes it has completed. Eviction cannot invalidate an in-flight tensor.

### Failure-atomic reconfiguration

Cache resize, KV retiering, transient-module swapping, and recurrent-state changes must either complete fully or leave the prior state usable.

### Deterministic validation

Lossless changes require token, route, logit/intermediate, and output parity where applicable. Lossy KV codecs require KLD, perplexity, task-quality, and speed measurements against an F16 reference.

### Observable decisions

Policy, cache capacity, hits, misses, evictions, bytes read, staging waits, per-device residency, and speculation acceptance must be available as structured telemetry.

## Why VBR/TCQ is not merged blindly

Variable-bit-rate KV and trellis-coded codecs are attractive because KV cannot be paged like immutable expert weights. They also touch Flash Attention, cache lifecycle, context checkpoints, multi-slot behavior, CUDA/ROCm kernels, and quality.

The port is therefore split into independently reviewable layers:

1. fixed Turbo scalar codecs and reference tests;
2. TCQ encode/decode kernels and codebooks;
3. per-layer sensitivity orders;
4. static mixed-tier KV;
5. dynamic VBR controller;
6. checkpoint, resize, multi-slot, and context-limit lifecycle;
7. global resource-controller integration.

Until those gates pass, `ese --kv auto` chooses only native KV types already present in this engine.

## Source layout added by the unified branch

```text
ese                         executable front door
tools/ese.py                planner, GGUF reader, hardware detection
tests/test_ese_launcher.py  deterministic control-plane tests
docs/ESE_*.md               focused user and architecture docs
docs/PORT_ROADMAP.md        code-port gates and status
.github/workflows/ese-ci.yml
```

No native source directory is renamed in this first consolidation, minimizing conflict with future upstream synchronization.
