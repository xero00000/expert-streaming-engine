# PLE reader publication and metadata — 2026-10-11

Reader queue, inflight/owner indexes, tickets, operations, promise/future state,
worker vectors and the fixed latency ring now report actual allocator requests.
An optional metadata quota rejects allocations before exceeding its limit;
it cannot be lowered below current live use. The default remains unlimited.
The native resource inventory includes this tracked storage in host metadata.

Submission prepares operations and reservations before publication, rolls back
partial queue/index/owner insertion on failure, and commits telemetry only after
success. Workers propagate preparation and buffer-allocation failures through
futures, release reservations, and allow later exact retries. Worker temporaries
are released before idle is published. Expired owners are pruned on submission;
statistics queries remain read-only. Startup failure joins existing workers.

Evidence: `/home/xero/Desktop/test/results/ese-ple-reader-publication-20261011`.

- 129 quota levels with an existing blocked read: 62 rejected, 67 admitted.
  Rejection preserves submitted/logical/coalesced counters, queues, reservations
  and the original owner's cancellation/read identity. Successful/retried bytes
  and duplicate-buffer identity are exact.
- Quota failures in worker preparation and buffer-vector allocation recover.
  The corrected second fixture uses 16 queued operations and 128 bytes headroom;
  the initial 64/1024 fixture unexpectedly admitted because deque block release
  supplied additional headroom. That failed experiment is retained.
- 1,000 successive owner IDs stay within the fixed metadata quota and retain
  at most one physical owner entry after old tickets expire.
- Four offside reconstructed variants use the same current allocators/ABI.
  Early telemetry/no rollback fails the fixture; rollback removal alone leaves
  8 reserved bytes rather than 4 after rejection. Moving either worker reserve
  outside its handler reproduces SIGABRT. The current executable passes.
  These are reconstructed behavior controls, not old released binaries.
- Final 19 Release and 12 CPU ASAN/UBSAN/leak regressions pass, including
  shared DeepSeek Engram storage and native provider/inventory checks.

This is partial C1/C3 qualification. Outer provider bookkeeping, public batch
vectors, tracker/impl objects, byte-vector objects, allocator headers, OS thread
resources, histories, host transactions and private device memory remain outside
this count. No complete RSS/budget, long stress, whole-model/GPU numerical or
performance claim is made. The original 33-token gate remains unresolved.
