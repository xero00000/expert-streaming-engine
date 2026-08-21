# Phase 4 global resource controller

Phase 4 moves the final memory decision into the native runtime. The `ese`
launcher still discovers files and hardware and supplies safe limits, but the
C++ controller probes the real GGUF metadata and owns the resulting policy.

## Interface

```bash
llama-server -m model.gguf \
  --memory-policy auto \
  --max-ram 40GiB \
  --reserve-vram 1GiB \
  --min-kv-quality turbo4 \
  --max-context 128K \
  --resource-preference balanced
```

`--memory-policy` also accepts `resident`, `cache`, `hybrid` (a compatibility
alias for cache), and `stream`. Memory sizes accept bytes, KiB, MiB, and GiB;
context accepts K and M binary suffixes. Invalid values are startup errors.

The controller is opt-in for direct native invocations. Omitting
`--memory-policy` preserves the existing independent tuning flags. The `ese`
launcher enables it automatically.

## Planning sequence

1. Load a metadata-only CPU probe and build the checked expert descriptor
   index without reading tensor payloads.
2. Separate dense and routed-expert bytes and measure the largest indivisible
   expert component.
3. Query every selected device's free/total memory and apply the same explicit
   reserve to each device.
4. Estimate complete KV bytes for every internal quality at an aligned sample
   context. Formats whose block geometry is incompatible with the model's K or
   V head width are unavailable, not fallback candidates.
5. Solve policy, context, quality, RAM expert cache, prompt cache, per-device
   expert cache, batch/ubatch, and transient capacity deterministically.
6. Map that plan to the existing native fit, expert hierarchy, KV, and
   transient-module implementations, then perform the real model load.

Balanced mode preserves the requested context at the highest fitting quality.
If no quality can preserve it, context is maximized first without crossing the
quality floor. Latency mode prioritizes quality; throughput mode prioritizes
context. Context is solved in the exact 256-token Flash Attention or 32-token
non-Flash alignment used by context creation, so padding cannot cross a bound.

## Budget invariants

- `planned + reserve <= free` on every GPU.
- CPU-only KV/workspace and RAM caches share one physical `--max-ram` bound.
- Nonzero expert RAM capacity is either large enough for the largest component
  or reduced to zero.
- MTP and multimodal allocations are charged once to the selected transient
  device and share `max(MTP, mmproj)` capacity.
- Requested storage backends never silently fall back. An unavailable
  `io_uring` source fails startup.
- KV formats below `--min-kv-quality` are never considered.
- Prepare or commit exceptions restore the prior logical plan and invoke the
  backend rollback hook. Transition counters distinguish prepare and commit
  failures.

## Observability

The complete single-line `resource_plan` JSON is printed before the real model
load and therefore before requests are accepted. The same object is returned
as `resource_plan` by `/props`. With `--metrics`, Prometheus gauges report the
selected context, planned RAM, transient capacity, and remaining host/device
headroom. Existing expert and transient telemetry continues to report actual
runtime hits, misses, evictions, transfers, swaps, and rollback failures.

## Validation

`test-resource-planner` covers deterministic repeatability, two-device reserve
accounting, single-device transient charging, balanced/latency/throughput
tradeoffs, hard backend and quality floors, resident/cache/stream presets,
CPU-only unified RAM accounting, human-readable parsing, and boolean/exception
rollback. It runs in release and ASAN/UBSAN CI jobs.

Local model-backed validation used the checked TinyMoE fixture:

- CUDA `sm_86` + `sm_75` (RTX 3060 Ti + RTX 2080 SUPER), 2 GiB RAM,
  1 GiB reserve per GPU, 16K, and two slots: compiled both architectures,
  loaded all layers across both devices, preserved both reserves, reached the
  HTTP listener, and reported the exact per-device plan and headroom.
- `auto`, 2 GiB, 16K, two slots: selected cache/F16, bounded the expert RAM
  tier, reduced the prompt cache, reached the HTTP listener, and exposed an
  identical `/props` plan plus Prometheus gauges.
- `cache`, 400 MiB, 16K: rejected incompatible compressed formats, retained
  F16, reduced context to the exact aligned 12,032-token capacity, and reached
  the listener without exceeding the declared host plan.
- `stream`, 400 MiB: failed closed because dense weights, mandatory I/O
  staging, workspace, and the minimum KV floor could not fit.
