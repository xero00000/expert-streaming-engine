# Shared graph/workspace admission — 2026-10-12

Graph arenas and CUDA workspace pools can now reserve against the same physical
admission ledger. A mutex serializes reservations, including pending driver
allocations, so concurrent owners cannot each spend the same remaining bytes.
The ledger reports current/peak reserved bytes, its cap and rejected requests.
Consumers retain it until their allocations and charges are released.

Graph replacements reserve their candidate while the old arena remains charged.
Failed admission or allocation releases candidate reservations and preserves the
old graph. Successful replacement retires the old charge only after publication.
Reused arenas and buffer-type aliases retain a single charge. Reattaching an
existing arena to a different ledger reserves its current backing before
releasing the former owner's charge; insufficient capacity leaves it attached
to the original ledger.

CUDA pools charge granularity-rounded VMM growth or rounded legacy backing
against that same ledger. Cached backing stays charged. Legacy overflow refunds
freed backing; destruction releases the remaining physical charge. Pool-local
caps remain available in addition to shared admission.

Context initialization attaches main-scheduler CUDA graph arenas and its pools
to one ledger per selected device. Unselected visible devices have null entries
and receive no free-memory query through this planner. The owner refreshes the
cap from current CUDA free memory, current ledger reservations and the configured
VRAM reserve before initialization reservation, ordinary decode graph allocation
and execution. Existing graph-specific explicit caps remain in force.

## Qualified evidence

Bundle: `/home/xero/Desktop/test/results/ese-shared-compute-budget-20261012`.

- Final suites pass 27 Release and 19 strict CPU ASAN/UBSAN/leak regressions.
- Two real CPU graph allocators share a cap. Replacement rejection preserves
  both old graphs' pointers/results. Admission accounts for old plus candidate;
  successful retirement and reuse release or retain exactly the correct charge.
  Buffer-type aliases, failed ledger reattachment and retained-owner lifetime
  checks pass. Eight simultaneous owners competing for 64 bytes admit exactly
  one 64-byte request, record seven rejections and release back to zero.
- Real CUDA graph/pool checks pass on the 3060 Ti, 2080 SUPER and 3080, using
  production VMM and a byte-audited legacy-only diagnostic variant. Cached pool
  backing blocks graph replacement; graph backing blocks pool growth. Old graph
  output remains exact and cached projection work can retry. A larger shared cap
  admits replacement with both old and candidate charged, then refunds the old
  arena. Graph deletion preserves the cached-pool charge; detaching the pool
  removes its ledger charge without freeing its retained backing.
- Actual tiny nonzero Llama contexts continue to pass 1024-request backoff,
  128-token chunked prompt execution and the unchanged output bound on all
  three GPUs with the automatic shared planner active. An existing session stays
  byte-exact after independent context failures, as in the prior fixture.
- Device-selection fixture passes all five modes, including noncontiguous GPU
  selection, last-device-only and both CPU modes. It asserts that non-null
  compute ledgers match exactly the expected device set.
- Fifteen final isolated unsuppressed all-API/leak memchecks pass: twelve
  graph/pool cases across both pool types, three GPUs and capture on/off, plus
  three actual-context prefill fixtures. All terminate 0, report zero errors and
  zero leaked bytes. This is component/small-model coverage, not whole-model
  Turing qualification.
- A same-ABI diagnostic ignores only the shared cap while retaining integer
  overflow protection. The identical CPU fixture fails its expected replacement
  rejection with exit 1. The transform preserves every other source byte.
- Initial CPU test executions failed because the fixture permitted its input to
  be overwritten in place, invalidating repeated-output checks. Marked its input
  as retained input/output. The failed logs, including leaks from early fixture
  exit, remain excluded. Corrected direct and full sanitizer runs pass.

## Limits

This is combined admission for graph arenas and pools, not a complete context
memory limit. Fixed weights and KV allocations already reduce the initial
free-memory snapshot, but their growth, handoff/expert/PLE buffers, host heaps,
vendor workspaces and driver allocations do not all participate in this ledger.
Other processes can change free memory after a snapshot. The planner's refresh
is an owner-thread operation; concurrent calls on one context are unsupported.

Generic backends that report more buffer bytes than requested receive a
post-allocation adjustment and rejection/rollback if it cannot fit. That cannot
guarantee admission before every physical byte for such a backend. Qualified
CUDA graph buffers report their requested size. Direct allocator failure with
an attached shared ledger, actual physical OOM and mapping/vendor failures are
not forced by this stage.

Secondary MTP/DFlash schedulers, remote split-pool execution, encoder-family
execution, arbitrary recurrent rollback, cancellation and sustained serving
concurrency are unqualified. Free-memory-query overhead is not performance
benchmarked. No original-model filled-context prefill or throughput acceptance
is established here. The original 33-token gate remains failed; normal Qwen
loading and complete-release promotion stay gated.
