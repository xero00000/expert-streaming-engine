# PLE history and host transaction allocation inventory — 2026-10-11

Persistent PLE histories now count actual allocator requests for token-tree
nodes, sequence-map nodes/hash buckets and copy-preparation sets. Their quota
is checked before allocation and reports live/peak bytes, failures and successful
request counts. Independent history copies use independent ledgers; assignment
prepares a complete candidate under the destination budget before swapping.
The default budget remains unlimited.

The hasher's two predecessors now reside in a fixed array instead of a heap
vector. Integer hash/EOS semantics are unchanged and match the existing pinned
traces. New-sequence append failures remove the unpublished empty entry.
Keeping one sequence erases its peers in place, avoiding the previous
clear-and-reinsert allocation after removing the target's owner node.

The native inventory counts both published and pending histories once, plus
known history-object storage. It also counts KV and nested split/checkpoint
vector backing, prepared host transaction objects/alternate-cache vectors,
Qwen attention input metadata, and host checkpoint payloads for both cache
banks. Existing payload/context-buffer deduplication remains; vector storage is
separate from the tensor payloads it references.

Evidence: `/home/xero/Desktop/test/results/ese-history-accounting-20261011`.

- Two zero-headroom append failures preserve history; invalid budget lowering
  rejects. EOS and subsequent hashing match the fixed reference trace.
- Copying leaves the original ledger unchanged. Failed assignment preserves the
  destination tokens and byte count; lifting the budget permits exact assignment.
- Keep preserves selected history and makes no allocation request.
- Copy/restore preparation is tested at129 quota levels each:28 copy failures,
  14 restore failures and216 admissions. Failure leaves complete histories intact.
  Initial one-token allocation is560bytes on this host; tests use measured bytes.
- Native inventory sums independent published/pending histories; dropping the
  pending root removes its allocation without changing the published root.
  Prepared retier transactions increase host metadata, and rollback/free restores
  its original count while existing KV/checkpoint ownership tests still pass.
- Six focused Release and six strict CPU ASAN/UBSAN/leak tests pass. Broader
  rebuilt20 Release and13 CPU sanitizer regressions pass, including DeepSeek
  Engram, provider quotas, cache gathers, lifecycle/checkpoint and loader gates.
- Offside same-ABI controls demonstrate two failure cases: a naive shared-ledger
  history copy changes original bytes672 to1344 and double-reports both roots;
  the old keep order performs request9 rather than preserving count8. Both
  fixtures return1; current Release/CPU sanitizer positives return0. The shared
  copy control reconstructs a naive allocator adaptation, not released history
  behavior. Owned negatives disable core dumps.

C1/C3 and the full release remain incomplete. This counts requested allocation
sizes/capacities, not allocator headers or total RSS. Returned serialization
vectors, legacy-restore temporaries, cell sequence-ID set nodes, implementation/
tracker/cache controls, worker source descriptors, OS resources and backend
private/handoff allocations remain outside the ledger. Per-history quotas are
independent; no aggregate context budget or prefill backoff is qualified yet.
No fresh whole-model/GPU numerical, long-context or performance claim is made.
Normal Qwen loading stays gated and the original33-token failure remains open.
