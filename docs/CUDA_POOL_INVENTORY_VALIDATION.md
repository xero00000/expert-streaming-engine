# CUDA workspace pool inventory — 2026-10-12

`llama_resource_get_memory_pools` now reports category `CUDA_POOL` (14):
physical backing owned by each context's CUDA workspace pools, including
allocations cached for reuse. Pools on remote devices are assigned to that
device's scheduler backend, or -2 when the scheduler has no matching backend.
Physical backing is distinct from graph, handoff and model tensor buffers.

The private `ggml_backend_cuda_get_pool_memory_stats` query separates physical,
active, cached, virtual-reservation and peak-physical bytes. It reads existing
pools without creating them or synchronizing execution. Call it on the owner
thread after graph completion. Invalid arguments return false and preserve the
output. The public inventory adds only physical bytes, avoiding double charges
for the active/cached partition and virtual reservation.

VMM physical backing is its committed mapping; active storage is its current
aligned allocation stack. Unused committed backing remains cached. Its 32 GiB
virtual reservation is not physical VRAM. Legacy physical backing includes
active allocations and cached buffers; the query sums cached slots and computes
active bytes from the remainder. A retained high-water mark tracks legacy
physical growth even when a full cache later frees backing.

## Evidence

Evidence bundle: `/home/xero/Desktop/test/results/ese-cuda-pool-inventory-20261012`.

- 26 scoped Release regressions and 18 CPU ASAN/UBSAN/leak regressions pass.
  The CPU suite exercises existing memory/session paths; it does not execute
  CUDA pools. The default new test only checks rejection without GPU init.
- Real cuBLAS projections pass independently known exact outputs on the
  3060 Ti, 2080 SUPER and 3080. Queries start at zero without creating pools,
  observe growth, then observe reuse without additional physical charges.
  Each snapshot's public inventory matches private physical bytes. Physical
  device IDs map to scheduler backend 0 even for device 1/2 fixtures.
- Native VMM pools retain 8,388,608 physical bytes after the large projection;
  active bytes are zero and cached bytes equal physical bytes. Virtual bytes
  are 34,359,738,368. Subsequent smaller projections retain the same backing.
- A frozen diagnostic variant changes only the shared factory/query predicate
  to select legacy pools. The source transform is byte-audited. Its identical
  projections and inventory checks pass all three GPUs, retaining 8,946,176
  physical/cached bytes with zero virtual reservation. Production selection
  continues to use device VMM capability.
- Both pool implementations pass isolated unsuppressed all-API/leak memcheck
  on each GPU: six terminal exit-0 runs, zero errors and zero leaked bytes.
- An initial test compile failed because the private context header expects
  `<random>` to be included first. Corrected before running tests; that failed
  build is retained and excluded. Final source/runtime generations are frozen.

## Limits

This is the inventory checkpoint. A separate
[admission and recovery validation](CUDA_POOL_ADMISSION_VALIDATION.md) covers
explicit pool caps, live lease snapshots and legacy cache overflow.

This checkpoint qualifies snapshots after completed graph execution. Nonzero active-byte
snapshots, legacy cache overflow/high-water decay, split-backend remote pool
allocations and sustained concurrent mutation are not exercised. The query is
not a thread-safe asynchronous monitor.

Vendor cuBLAS workspaces, CUDA graphs, driver/context allocations and OS page
cache remain outside this pool field. No aggregate allocation limit or safe
CUDA pool OOM recovery is implemented by this inventory change. The tests do not measure
full-model prefill peaks, throughput, long-context correctness or whole-model
Turing parity. The original Qwen 33-token gate remains failed; normal loading
and release-wide acceptance remain gated.
