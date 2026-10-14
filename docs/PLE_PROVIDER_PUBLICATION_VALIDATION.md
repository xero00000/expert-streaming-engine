# PLE provider ticket publication and registry metadata — 2026-10-11

Row, page and two-tier providers now account for actual ticket/control-block,
owner-map-node and owner-vector allocator requests. Telemetry includes live/peak
bytes, quota/failure counts and physical owner cardinality. The native resource
inventory includes row and both two-tier provider ledgers in host metadata.
An optional per-provider quota rejects allocation before exceeding its limit;
default construction keeps the quota unlimited.

Previously, row/page providers dispatched reads before allocating owner records;
the two-tier provider similarly submitted page work before registering its own
ticket. A later allocation failure left work with no returned outer ticket.
All three now prepare their registration first and hold the owner lock through
child dispatch. If child admission fails, only the unpublished ticket is removed.
Existing tickets retain cancellation ownership. Expired owner IDs are pruned
before publication, preventing weak controls/owner entries accumulating forever.

Evidence: `/home/xero/Desktop/test/results/ese-ple-provider-publication-20261011`.

- Each provider is tested at 65 metadata quota levels for both an existing and
  new owner (390 cases total): rows34/pages36/two-tier34 rejections, with the
  remaining286 admissions. Rejection sends no child read and preserves original
  submitted/logical/staging counters and owner cancellation. Admission and later
  retry return the exact expected quantized-row dequantization.
- Child staging exhaustion rolls back the newly registered ticket while leaving
  the original same-owner ticket cancellable. Metadata quota cannot be lowered
  below live allocations; removing the owners releases tracked weak controls.
- Each provider completes1,000 distinct owner IDs under a fixed metadata quota
  while retaining at most one expired-owner entry (3,000 IDs total).
- Four focused Release and four strict CPU ASAN/UBSAN/leak tests pass, including
  native row/two-tier inventory and the shared DeepSeek Engram implementation.
  Broader rebuilt20 Release and13 CPU sanitizer regressions pass. The final
  fixture-only diagnostic addition also passes in both builds.
- Two offside reconstructed libraries use the same current allocator/quota/ABI:
  dispatch-before-owner-allocation fails at rows owner7/slack208 with submissions
  2 versus1; disabling expired-owner cleanup fails with two owner entries.
  Current positive fixtures pass. These are behavior controls, not released
  historical binaries. Owned negative subprocesses disable core dumps.

This is partial memory/failure qualification, not total process memory or complete
C1/C3 closure. Provider state vector backing, staged batches, transient request
vectors, implementation/tracker storage, cache controls, allocator overhead,
thread resources, histories, host transactions and device-private memory still
need accounting and/or budget enforcement. The per-provider quota covers only
the explicitly tracked requests; two-tier page metadata has its own ledger.

No new whole-model/GPU numerical, long-context or performance result is claimed.
Normal Qwen production loading remains gated. The original33-token numerical
failure and the remaining current release gates stay open.
