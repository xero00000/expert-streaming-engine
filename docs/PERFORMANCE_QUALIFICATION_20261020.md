# ESE performance qualification — 20 October 2026

Task 4 starts after the local inference/recovery checks in task 3. The user's
publication hold remains in force. A benchmark result does not authorize a
push, hosted workflow, tag or publication.

## Acceptance contract

The user selected **match Strata in speed**. For the first direct comparison,
freeze one common supported model and prompt on this machine; compare repeated
sustained decode medians (ESE / Strata at least 1.0). Record every sample and
cold/repeated behavior; retain startup, prefill and first-response separately.
This does not lift the publication hold or qualify other models/context sizes.

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
7. Apply the Strata reference without changing it after seeing results. Record
   task 4's outcome and proceed to task 5's filled-64K/concurrency qualification.

Existing tools include `llama-bench`, `scripts/compare-llama-bench.py`, the
phase-2 cache runner and the retained HTTP/resource evidence wrappers. Select
the runner that records the agreed workload; the present fixture timings have
no production acceptance threshold.

## Direct comparison preparation

The saved 5 October Strata benchmark failed before readiness and has no speed
metrics. Its gated Qwen4Exp workload is different from the supported qwen35moe
35B model used in the preliminary resident/cache smoke tests. Those smoke
timings cannot establish Strata parity.

The installed Strata 0.1.45 also supports qwen35moe, using a single GPU for this
family. The common comparison therefore selects the RTX 3080, 8 CPU threads,
8,192 context cells, a frozen 512-token prompt, 128 greedy generated tokens and
no MTP. The Q4_K_M candidate is rejected by Strata's F32 projection loader.
The existing Qwen3.6-35B-A3B-UD-IQ4_XS model has suitable tensor formats but
omits the explicit zero-MTP metadata that Strata requires. A separate benchmark
copy adds only `qwen35moe.nextn_predict_layers = 0`; all original metadata and
every tensor's type, shape and SHA-256 must verify unchanged. Both engines use
that identical copy. Original files and running-service configs are preserved.

The prompt includes reasoning output; rates are generated-token rates, not
answer-only rates. Initial measurements retain the shared desktop's ambient
GPU load, so quiet-system qualification remains distinct. Engine-protocol
Strata first-token latency excludes HTTP; compare it separately from ESE HTTP
latency. Both runners reject missing/invalid timing receipts, prompt reuse,
wrong actual token counts, early stops and truncation. They retain raw logs,
commands, model/binary/runner identities and sampled process/driver memory.
