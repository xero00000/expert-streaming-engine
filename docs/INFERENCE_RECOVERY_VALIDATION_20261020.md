# Consolidated inference correctness and recovery — 20 October 2026

Task 3 remains in progress. This evidence validates the bounded fixture paths
listed below. It does not establish release readiness or upstream full-model
acceptance. The strict automatic RAM-planning headroom failure remains open.

## Candidate and retained references

Native source: `9e0d3b0379d187ac643c4fc94019bc4e2ef694aa`.
The native-receipt validator and its six regressions are committed at
`60ab8fdf2`; they change no native executable source.
The production Qwen gate is OFF in CPU, CUDA and sanitizer builds.
The candidate CLI/server report this commit. See `source.json` and
`runtime-versions.json` in
`/home/xero/Desktop/test/results/ese-release-work-20261020/task3/` for source,
binary/library hashes, compilers, device UUIDs and gate settings.

The retained TinyMoE F16 fixture has SHA-256
`c559eaabd8a56589b8abe09d678fbb56c673a3dcc28907e193d27b25ecd60684`.
Resident execution on each backend is the reference for path comparisons.
Comparing storage/cache/stream variants does not independently verify the
conversion or the original upstream model. The generated three-layer Llama
and new four-expert MoE fixtures require no external model download.

## Repairs and regression controls

- `d249dd3e2` corrects an obsolete acceptance budget. With current padding
  headroom, 4 MiB holds one usable expert triplet, so top-2 execution retains
  CPU experts. The runner now checks that case separately against its CPU
  placement reference and uses 6 MiB for two active GPU slots. The original
  failing trial and capacity diagnostic runs are retained.
- `c69fd21d8` removes an incorrect CPU softmax/router adjacency shortcut. It
  read unwritten softmax output, skipped live probabilities and ignored masks
  and scale. The replacement executes the normal operations. Its poisoned
  replay oracle checks probabilities, selected IDs and gathered weights with
  one/four threads, changed logits and scaled/masked inputs. It fails against
  the previous source and passes after the repair. Performance remains a
  separate task; the shortcut must not be reinstated without a valid oracle.
- `9e0d3b037` replaces the BOS/timing warmup heuristic with explicit context
  warmup state. Ordinary first BOS microbatches now use the configured top-k.
  `llama_set_warmup` synchronizes and invalidates graphs on transitions;
  common initialization explicitly brackets its optional warmup. Allocation
  reservation remains conservative. The generated MoE test fails on the old
  first-request route width, then checks exact logits through cold replays,
  timing resets, both graph-reuse modes and explicit warmup transitions.

Before the warmup repair, retained TinyMoE requests changed even without a
client cancellation, with prompt caching disabled, with one thread, with F32
KV, with flash attention disabled and with graph reuse disabled. These failed
controls are retained and are not passed qualification runs. The final native
cold replay matches all 32,000 logits exactly. Ordinary CPU HTTP serving with
and without startup warmup also matches content and all returned probabilities.

## Final local results

- Full Release CPU/CUDA builds completed. CPU CTest passed **94/94**. Each
  RTX 3060 Ti, RTX 2080 SUPER and RTX 3080 CUDA suite has **99 passes, zero
  failures and one expected single-device NCCL skip**. The default BF16 batch
  test reports an internal skip; task 3 does not claim a new explicitly enabled
  BF16 batch matrix. Task 2 records its earlier explicit kernel checks.
- **152 Python unittests passed.** The six new receipt-verifier tests accept
  native allocations that differ from forecasts and reject incorrect pools,
  native topology, byte counts and peak/headroom arithmetic. The prior verifier
  rejects the valid native-receipt regression. Documentation and whitespace
  checks passed.
- Both targeted Clang 22 Debug ASAN/UBSAN/leak suites passed: expert cache and
  server decode recovery. Sanitizers use strict halt-on-error and leak detection.
  These are CPU sanitizer checks, with no claimed GPU memory-checker run.
- Exact forced-token parity covers **512,000 logits and 320 route IDs per
  configuration** over 16 steps. The CUDA-enabled runner has eleven records:
  CPU resident/cache/stream, resident/cache on each GPU, and resident/cache on
  all three. All six optimized-path comparisons are exact. A separate CPU-only
  compiler build also passes its three-path matrix. The probe disables fused
  multiply/multi-add on both sides to match cache dispatch; uninstrumented
  controls retain their normal execution settings.
- The corrected uninstrumented cache matrix passes for one/two/three GPUs.
  Every participating GPU has two active slots within 6 MiB, with 138/136/134
  aggregate evictions respectively and zero original-tensor fallbacks. CPU
  mmap, pread and io_uring storage variants and native stream match their CPU
  reference exactly; the undersized-cache CPU controls also pass.
- **24 sustained 128-token worker trials pass**, two rounds across one/two/three
  GPUs, pread/io_uring and zero/two workers. Queue, RAM/staging, lease and GPU
  bounds pass and outstanding ownership drains completely.
- Bounded prefill passes against the established path on one/two/three GPUs,
  with two requests per path and exact output hashes. Each GPU has two lanes
  within 64 MiB, real H2D/D2D reuse and no correctness/global-sync fallback.
  The three-GPU launcher trial uses `--tensor-split 1,1,8` so MoE layers occupy
  all three cards; the equal-share trial used only two expert devices and is
  retained as a failed placement-coverage trial. No speed threshold is claimed.
- CPU KV shrink/grow, injected KV rollback and continuation pass. The RTX 2080
  SUPER and three-GPU runs also pass expert-cache reconciliation, combined
  1024/512-token KV and 64/32 MiB cache changes, busy rejection, exact output
  and publication/rollback checks. Multi-device controls include failure after
  one device prepares and after the first device publishes. The verifier now
  compares live allocation inventories and a separate native KV-only receipt,
  preserving exact byte assertions instead of comparing receipts to estimates.
- **21 real HTTP disconnect/retry trials pass**: three each for CPU resident,
  CPU stream, warmup-enabled CPU, one/two/three-GPU cache, and staged CUDA
  serving. Every trial observes cancellation in the server log, drains resource
  ownership and preserves cold retry content and all returned probabilities.
  Pre-cancellation cold controls must match too. A tiny dense cancellation
  attempt completed too quickly to prove cancellation and is excluded.
- **Seven final HTTP decode-failure controls pass**: four real CPU graph aborts
  with/without a system prefix, two injected error statuses and one recoverable
  capacity status. Cached/cold retries and the second slot's continuation match.
  The shim is freshly compiled against this checkout and loads no old backend.
- Studio's staged CUDA runtime was refreshed from the candidate. Its version,
  package marker and dependency closure pass without `LD_LIBRARY_PATH` or
  `LD_PRELOAD`; its three real disconnect/inference trials also pass in that
  environment. This is a staged runtime, not an installed release package.

Start with `task3-summary.json` in the evidence directory. Raw model outputs,
route/logit values, command/environment records, HTTP responses, allocator
snapshots, JUnit files and per-file checksums are retained. Earlier failed
runs remain visible and are not included in the passing counts.

## Scope and outstanding work

Retain all commands, raw output/route/logit records, responses, resource
snapshots, failed trials and hashes. Callback probes synchronize selected
intermediates and can change graph submission; retain uninstrumented CLI and
HTTP controls alongside them. The probe executable's historical Qwen name and
capture environment names do not enable or qualify the Qwen architecture.

Strict automatic `--memory-policy auto --max-ram 4GiB` selected cache capacity
that left insufficient tracked context metadata/output headroom. Startup
rejected `4,295,429,216` required bytes against `4,294,967,296` usable bytes.
Preserve the rejection and repair the planning forecast; do not weaken host
admission or treat an explicit smaller-cache configuration as proof that the
automatic plan is fixed. Transaction tests can use bounded explicit caches
independently.

Physical driver allocation exhaustion, larger original-model numerical
acceptance, research Qwen/MTP/multimodal serving, filled 64K context,
concurrent workloads, performance targets and installed package/update
qualification remain outside these fixture results. No push, hosted workflow,
tag or publication is authorized by this validation.
