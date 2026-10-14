# PLE cache allocation accounting — 2026-10-11

PLE cache telemetry now distinguishes occupied row/page payload from backing
allocation capacity. `block_cache_stats` reports arena and slot-vector bytes,
live index-node allocation requests, their peak, and an optional index-node
allocation quota. The map allocator counts its rebound node type rather than
estimating node size from the key/value pair. The existing payload-cache
capacity and default admission behavior are retained.

The native resource inventory includes full PLE arenas in `PLE_CACHE` and
slot/index storage in `HOST_METADATA`. Row-only and page-plus-row providers
are both covered; allocated empty arenas are included even with zero occupied
payload. Multiple live provider allocations are summed separately.

## Admission recovery

Index insertion previously followed slot publication and old-index removal.
If that insertion failed, an empty slot could remain occupied/pinned without
an index entry, preventing later admissions. Index-node allocation now happens
before slot bytes, old membership, pins and admission telemetry change.

The optional quota bounds index-node allocation requests only. It does not
bound arena/slot allocation or total process RAM. A full-cache replacement
needs temporary headroom for one incoming index node before the old node is
released. Default construction leaves the index quota unlimited.

## Evidence

Results: `/home/xero/Desktop/test/results/ese-ple-allocation-accounting-20261011`.

- Cache tests check eager empty-arena/slot allocation, live/peak index requests
  across admission and eviction, and real allocator-quota failures with empty
  slots and unpinned eviction candidates. Old bytes/index membership remain
  available and a second failed admission does not find an orphaned slot.
- The identical Release fixture passes with current code and fails with an
  offside library restoring the old publication order while retaining the new
  allocator/quota. This is a reconstructed order control, not an old released
  binary. Its expected failure is `PLE failed admission left an unusable slot`.
- Row-store integration exercises the native inventory before occupancy and
  after actual reads, checking row-only and two-tier arena sums and precise
  index-metadata growth. Existing dequantization, leases, cancellation and
  page-straddle coverage are preserved.
- Focused four-test Release and CPU ASAN/UBSAN/leak runs pass. Final broader
  regression checks pass 18 Release and 11 CPU ASAN/UBSAN/leak tests.
- Initial integration-fixture compile errors (a duplicate local name and the
  internal sampling header's required random include) were corrected. An
  earlier four-test run used the old row-store fixture after its build failed;
  that result is excluded from the new integration qualification. Final rebuilt
  fixture logs are retained separately.

## Limits

Counters cover allocation requests/capacities, not total RSS. Allocator headers,
shared control objects, worker/ticket/future bookkeeping, asynchronous read
staging, sequence histories, OS page cache and driver/provider-private memory
still need accounting. Source buffers shared outside the context are not
claimed as context-owned allocations.

These changes are a later source generation than the completed cache-format
model checks. They do not certify full-model output parity, complete memory
budgets, long context, performance or production promotion. C1 remains open.
