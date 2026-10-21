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

## First completed measurements

Both completed baselines use the identical comparison file (SHA-256
`2b442ab092f23747224e62b74b8a14a45e777629d28aa97427d0c1b62a49010e`).
Its original IQ4_XS file hashes to
`649d7508507b84638732c4f52c24c8b15843c6dca2f3ff793ae07c14a67ebbb3`;
the model-preparation manifest verifies all 733 tensors unchanged. Native ESE
remains source `9e55c4925`; only benchmark tooling has changed.

| Measurement | Strata 0.1.45 | ESE mapped cache, unfused control |
| --- | ---: | ---: |
| Fresh processes / requests each | 3 / 3 | 3 / 3 |
| Decode median, all nine requests | 45.29 tok/s | 31.18 tok/s |
| Decode median, six repeated requests | 45.75 tok/s | 31.38 tok/s |
| First request decode, each fresh process | 44.04 / 41.90 / 42.28 | 2.28 / 2.12 / 23.13 |
| Startup to ready, seconds | 191.98 / 82.31 / 13.01 | 1.53 / 5.91 / 4.76 |
| Median prompt prefill | 623.40 tok/s | 123.77 tok/s |
| Median first token | 0.82 s (engine protocol) | 4.14 s (HTTP) |
| Process high-water RSS | 32.53 GiB | 25.48 GiB |
| Sampled process swap peak | 1.19 GiB | 0 |
| Sampled total RTX 3080 usage | 9,431 MiB | 8,236 MiB |

ESE has not met the target: its overall ratio is 0.688 and its repeated-request
ratio is 0.686. These were sequential runs on a shared desktop, not randomized
quiet-system pairs. Strata and ESE have different actual placements/footprints:
Strata uses adaptive CPU/GPU expert execution and ESE uses bounded expert
uploads. ESE's 32 GiB tracked-capacity ceiling and 1 GiB VRAM reserve do not
assert an equivalent Strata host limit or a process-RSS cap. Device samples
include desktop allocations and are not exact driver peaks.

All completed requests process 512 prompt tokens, generate 128 tokens, and
report no reuse/early stop/truncation. Each engine independently repeats its
output exactly across nine requests and exits normally after each process.
This is bounded workload evidence, not upstream full-model numerical acceptance.

The initial ESE pinned-model control was stopped during its first request:
startup took 218.46 s and process high-water RSS reached 35.89 GiB before a
complete token receipt. The failed trial and operator-abort note are retained.
The successful ESE run changes only `GGML_CUDA_NO_MODEL_PINNED=1`: expert
weights remain mapped/deferred, avoiding a duplicate 14.63 GiB pinned model
copy. The bounded cache is 16 GiB RAM / 4 GiB VRAM, with native automatic
two-lane prefill staging (860 MiB), 64 MiB host staging and a 1 GiB GPU reserve.
RAM-cache leases and worker queues drain to zero; no forced host-tensor fallback
occurs in completed trials. Fresh-process cache loading affects early decode
and is retained, not averaged into a warm-only claim.

A three-request pilot using production fused reduction preserves the unfused
output exactly but gives about 31.29 tok/s on repeated requests: no demonstrated
speedup. Current tuning targets the observed upload work; the unfused baseline's
first process records 102,135 lease-backed component uploads across three
requests. Cache-capacity/prefill pilots must retain the same model, prompt,
counts and independent replay checks before any passing configuration is
repeated over three fresh processes.

Evidence lives under
`/home/xero/Desktop/test/results/ese-release-work-20261020/task4/`:
`comparison-model-manifest.json`, `comparison-summary.json`,
`strata-iq4-3080/`, `ese-iq4-3080-cache/`,
`ese-iq4-3080-cache-mapped/` and `ese-iq4-3080-cache-fused-pilot/`.
Raw protocol/SSE responses, exact commands, timing receipts, memory samples,
resources and failed controls remain available. No publication action occurred.

### Failures exposed during tuning

The 4,864 MiB VRAM-cache pilot completes its first 512/128 request but aborts
on the second prefill with `sidecar-only prefill staging refused original-tensor
fallback`. The cache clamps itself to about 4,715 MiB on this run; subsequent
live free memory falls below the 1 GiB prefill reserve. The process exits by
SIGABRT. This configuration is rejected; neither its timing nor relaxed
reserve/fallback settings qualify it. Retain `ese-iq4-3080-cache-475g-pilot/`
as a startup-versus-repeated-request admission/recovery issue requiring follow-up.

Hardware calibration also rejects the IQ4_XS CPU matrix path against the
independent scalar oracle (NRMSE 0.127736). An isolated diagnostic keeps the
same weights/activations but uses the direct row-dot path: NRMSE 0.003938.
Matching activation precision does not explain the matrix discrepancy.
A deterministic synthetic IQ4_XS case reproduces saturation: 8.03149 is returned
where scalar dequantization requires 904. The AVX2 matrix path's unsigned
lookup reaches 241, overflowing its saturating 16-bit pair sum before offset
correction. A narrow signed-lookup/abs-sign repair and regression are under
validation. The independent 0.08 calibration limit remains unchanged; mixed
routing remains disabled until calibration and its separate workload gate pass.

Native source `f1c381c5f` completes the signed AVX2 repair: all 27 synthetic
cases pass, and full model-backed calibration is planner-ready with CPU oracle
errors below 0.004 for each of the three expert formats. Its qualification
passes 95 CPU tests, 100 CUDA tests plus one expected NCCL skip per GPU, and
five strict sanitizer targets. The table above still describes the earlier
`9e55c4925` native benchmark; no speed improvement is inferred from a correctness
repair or a cache-hot calibration result.

The initial calibrated hybrid A/B trial is invalid and retained in
`3080-iq4-hybrid-gate-f1c381c5f.log` and `hybrid-gate-f1c381c5f-raw/`.
The existing validator appended text after the chat generation prefix and
accepted early EOS as a timing sample. It has been repaired to preserve the
supplied prompt, force the requested generation length, verify actual/timing
counts and duration arithmetic, and reject truncation. Four new regression
tests fail before the repair and pass after it; the complete 160-test Python
suite passes. Verification schema 4 rejects earlier measurement receipts.
`validate-hybrid --evidence-dir NEW_DIRECTORY` now preserves both native logs
on success or failure. A new full-token trial is required before any mixed
configuration can qualify for the matched Strata comparison.
