# CUDA pool admission and graph recovery — 2026-10-12

A private owner-thread setter applies a physical-backing cap to an individual
CUDA backend/device pool. It includes active and cached backing. Setting a cap
can create the pool object but allocates no GPU backing. A cap below already
committed backing is rejected without changing the old cap. Default admission
remains unlimited (`UINT64_MAX`). This cap is not exposed as a launcher default
or a combined context limit.

VMM admission checks aligned, granularity-rounded growth before `cuMemCreate`.
Legacy admission checks rounded look-ahead backing before device allocation;
existing cached allocations can be reused without additional physical charge.
Both implementations check size/rounding overflow. Pool statistics include
the cap and rejection count. The private statistics structure has grown;
rebuild its consumers together with the library.

A pool admission failure now unwinds temporary allocation leases and returns
`GGML_STATUS_ALLOC_FAILED` from graph execution. If CUDA graph capture has
started, the failure ends and discards that capture, releases its global
capture counter, drains created streams and forgets the failed cached graph.
Retries cannot replay a partial capture. Primary CUDA allocator OOM results
also enter this recovery path; the legacy runtime error flag is cleared.

## Qualified checks

Evidence: `/home/xero/Desktop/test/results/ese-cuda-pool-admission-20261012`.
Source, owned patch, runtime generations, diagnostic transforms, helper builds,
commands, logs and terminal receipts are frozen in this bundle.

- Final scoped Release suite passes 26 tests; strict CPU ASAN/UBSAN/leak suite
  passes 18. The final source refinement only changes CUDA OOM error clearing
  and capture telemetry; CPU runtime source remains the same generation.
- Real FP16 cuBLAS projections pass on all three GPUs under both VMM and
  legacy pools, with capture enabled and disabled. Zero cap rejects a cold
  allocation before physical backing changes. A cap equal to warm backing
  rejects growth; a smaller cap is refused without changing the old cap.
- A 6 MiB cap admits the first large temporary buffer and rejects the second.
  Leases unwind to zero active bytes, admitted backing remains within the cap,
  and the projection output retains its sentinel. Small retries pass exact
  independently known outputs; removing the cap permits large exact outputs.
  Reuse does not double-charge retained physical backing. Telemetry confirms
  rejection during capture (1) and ordinary execution (0).
- A separate native CUDA helper, compiled with the actual backend's flags and
  context layout, holds two live leases and checks their exact active-byte sum.
  Rejection preserves both leases and their independent byte patterns. It
  checks `SIZE_MAX` rejection without backing mutation, then 257 simultaneous
  allocations and reverse-order release on both implementations/all three GPUs.
- Legacy cache overflow releases physical backing while retaining the high
  water mark: 134144 live physical/cached bytes, peak 138496, active zero after
  this helper. VMM retains 2097152 physical/cached bytes, active zero, plus its
  separate 34359738368-byte virtual reservation.
- Twelve isolated GPU projection memchecks cover two implementations, three
  GPUs and capture on/off. Two additional memcheck processes execute active
  lease/overflow checks across all three GPUs for each implementation. All 14
  final processes terminate 0, report zero CUDA API/memory errors and zero
  leaked bytes. These are component tests, not whole-model Turing coverage.
- A same-ABI negative runtime ignores only budget admission while preserving
  overflow checks. The identical cold-cap fixture fails with exit 1 at its
  expected allocation-status assertion. Both this transform and legacy-only
  pool selection are byte-audited against the final source.

## Limits

This is the explicit pool-cap checkpoint. Separate
[shared compute validation](SHARED_COMPUTE_BUDGET_VALIDATION.md) covers
automatic combined graph-arena/workspace-pool admission.

Caps are explicit, per pool and unlimited by default. Automatic combined
admission across fixed weights, KV, graph arenas, expert caches, pools and
vendor/driver allocations remains incomplete. A free-memory snapshot is not a
reservation against other processes.

Recoverable status is not whole-graph or whole-session atomicity. Earlier
ordinary graph nodes may already have written outputs or recurrent state
before a later allocation fails. The tested projection rejects before writing
its output; that sentinel check does not establish arbitrary session rollback.
Full-model journals, cancellation, sustained concurrency and actual prefill
still require their own acceptance checks.

Actual physical OOM is not forced by these tests. Virtual-address reservation,
mapping/access failures, host metadata allocation failures and vendor-library
OOM remain outside the qualified recovery scope. Split-backend remote pools,
HIP/MUSA execution, unified memory and graph capture with a full legacy cache
are unqualified. The active helper is a separately built evidence executable,
not part of the default CTest run.

No weights, routes, goldens or comparison tolerances changed. The original
33-token Qwen numerical gate remains failed, normal loading stays gated and
no throughput or complete-release claim is made.
