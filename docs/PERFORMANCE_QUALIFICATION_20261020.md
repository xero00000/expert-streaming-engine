# ESE performance qualification — 20 October 2026

Task 4 starts after the local inference/recovery checks in task 3. The user's
publication hold remains in force. A benchmark result does not authorize a
push, hosted workflow, tag or publication.

## Acceptance contract to finalize

The model, prompt, hardware/memory budget and speed threshold require the
user's selection. An asynchronous question was submitted while task 3's
validation continued. No passing threshold or production workload is assumed.

Record the following before collecting qualification measurements:

- Exact model and quantization, local path and SHA-256; any conversion or
  upstream reference needed for the claimed model support.
- Selected GPU UUIDs, CPU thread count, RAM and per-GPU VRAM ceilings/reserves,
  model/expert placement, KV types, cache/staging capacities and storage path.
- Exact prompt/token IDs, filled prompt length, generated token count, seed,
  sampling settings and supported concurrency.
- Required minimum sustained decode rate and any startup, prefill or first
  response limits; state whether these apply per request or in aggregate.
- Comparison baseline: exact ESE and, if requested, Strata source/binary hashes
  and equivalent operating conditions. Compilation or unlike configurations
  cannot establish a performance comparison.

## Local inventory already available

The host has RTX 3060 Ti (8 GB), RTX 2080 SUPER (8 GB), and RTX 3080 (10 GB)
GPUs. UUIDs and compiler/driver records are retained in task 3's evidence.
The TinyMoE fixture is a regression workload; its speed does not qualify a
larger user model. Local model candidates include
`/home/xero/Desktop/test/models/Qwen3.8-35B-A3B-Q4_K_M.gguf` (21,713,462,944
bytes). Availability does not imply model selection or numerical qualification.
The models remain on their existing storage; no downloads or moves are needed
to define the contract.

## Ordered measurements once the contract is filled

1. Freeze the native candidate, exact commands, model/prompt hashes and resource
   settings. Confirm no unrelated load is using the selected devices.
2. Run a bounded correctness and memory-budget check on the selected model and
   settings before timing. Retain failures, untruncated token counts and output.
3. Measure at least three fresh-process starts and three repeated requests per
   configuration. Record OS page-cache state and distinguish process-cold from
   storage-cold starts. Do not label an already cached file storage-cold or
   globally drop caches shared by other work.
4. Retain startup-to-ready, prompt prefill, first response and sustained decode
   timings separately, along with raw responses, actual token counts and
   RAM/VRAM/cache/queue/lease snapshots. Measure performance without tracing,
   sanitizer builds or concurrent compilation; diagnose bottlenecks separately.
5. Compare matched resident/cache/stream settings and requested ESE/Strata
   baseline only when their placement and resource conditions are equivalent.
   Report individual samples and median/range, including failed trials.
6. Profile the measured bottleneck, make one change, and rerun the affected
   correctness/recovery checks before repeating the same timing protocol.
7. Apply the agreed threshold without changing it after seeing results. Record
   task 4's outcome and proceed to task 5's filled-64K/concurrency qualification.

Existing tools include `llama-bench`, `scripts/compare-llama-bench.py`, the
phase-2 cache runner and the retained HTTP/resource evidence wrappers. Select
the runner that records the agreed workload; the present fixture timings have
no production acceptance threshold.
