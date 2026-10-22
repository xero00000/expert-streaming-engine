# PLE vector allocation accounting — 2026-10-11

Provider ticket vectors, row/page locator maps and temporary span/row-ID vectors
now use the same lifetime-safe metadata allocator as their ticket/owner ledger.
Returned reader buffer vectors and row/page/two-tier lease/view vectors retain
that allocator when moved or copied. The native inventory therefore continues
to count retained batch storage after the ticket is consumed or destroyed.

The allocator uses standard typed allocation callbacks to preserve alignment,
checks its quota before each request, counts successful requested sizes, and
propagates its ledger through container moves/copies/swaps. Its shared tracker
can outlive the provider. Empty user-created batches start without a ledger;
provider-created storage always carries the originating one.

Pointer/count input overloads permit tracked temporary vectors to flow between
providers and readers without copying into an untracked standard vector. Existing
vector input APIs remain. Nonempty null inputs are rejected. Page row views now
move into the consumed batch alongside their pinned pages instead of allocating
another copy.

Evidence: `/home/xero/Desktop/test/results/ese-ple-vector-accounting-20261011`.

- Provider quota matrix390 cases:156 rejected/234 admitted (rows48/82,
  pages62/68, two-tier46/84). Existing ticket/cancellation, child rollback and
  exact retry checks pass. Each provider's1,000 owner-ID sweep stays bounded.
- Retained batches remain counted after ticket and owner release. Under the
  remaining fixed quota, new cold submissions reject before child dispatch.
  Swapping batches from two independent providers keeps each allocation in its
  originating ledger. Releasing the corresponding moved batch drains that
  ledger. A returned batch outlives its provider and still yields exact rows.
- Reader result-vector quota failure leaves the ticket retryable and does not
  increment worker failures. Raising the quota returns exact bytes. Copying a
  returned vector adds exactly its capacity times element size to the ledger;
  releasing it removes the allocation.
- The older worker failure fixture now restores quota only after workers are
  idle, so returned-vector allocation cannot mask the stored worker error.
- Five focused Release and five strict CPU ASAN/UBSAN/leak tests pass, including
  native inventory and DeepSeek Engram. Broader rebuilt20 Release and13 CPU
  sanitizer regressions pass. Original reader129-case publication matrix passes
  with62 rejections/67 admissions.
- Two offside reconstructed libraries with the current ABI fail the same tests:
  an untracked reader result escapes quota; an untracked row result leaves both
  retained provider ledger counts at zero. Current reader/provider positives pass.
  Controls model the excluded allocations, not an old released engine binary.

C1/C3 remain partial. Worker source request descriptors still use the source's
standard-vector interface; byte-vector objects, tracker/implementation/cache
controls, allocator/OS overhead, histories, host transactions and device-private
allocations also require accounting and/or budget enforcement. This is allocation
request telemetry, not total RSS. No new whole-model/GPU numerical, long-context
or performance qualification is claimed; normal Qwen loading remains gated.
The original33-token failure and all remaining current-release requirements stay
open.
