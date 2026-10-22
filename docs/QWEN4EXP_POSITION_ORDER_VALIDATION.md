# Qwen position transforms and persistent token chronology

Evidence: `/home/xero/Desktop/test/results/ese-qwen-position-order-20261011`.
This qualifies short private-row position transforms on the original coder
artifact, using the F32 arithmetic policy, F32 K/V/indexer and non-Flash
attention on the selected RTX 3060 Ti / RTX 3080 pair. B4 remains in progress.

Cached tokens now carry a chronological ordinal separately from rotary
position. PLE stores tokens by ordinal, so integer position division retains
all tokens even when positions coincide. Position-tagged removal rebuilds
hash history in original token order. Copy and restore preserve ordinals;
malformed chronological history is staged and rejected before publication.
New-token hash updates keep constant-time maximum-position bookkeeping.

Ordered QSA groups rows by ordinal and masks future queries by ordinal. Block
RoPE positions still come from each block's first actual rotary position.
The independent ordinal survives cache copies, defragmentation (whole-cell
moves), retiering and full/per-sequence snapshots. Actual defragmentation and
retiering after compression still need model-backed qualification.

Qwen add/divide wrappers validate divisors, ranges, metadata overflow and
shared ownership before changing checkpoints or side histories. Valid private
row transforms update PLE positions and the saved-output position ledger
alongside cache metadata. Negative shifted positions remove corresponding
history entries. Individual-owner transforms of shared rows currently fail
before mutation: cloning K/V/raw-indexer rows for independent owners remains
required work. This is not a completed shared-sequence implementation.

Experimental Qwen KV snapshots now carry marker `0x514b5632`, eight bytes of
ordinal per cached cell, and eight additional bytes per PLE history entry.
Older snapshots are rejected. Conventional architecture serialization is
unchanged and its synthetic roundtrip passes. Saving a real Qwen cache applies
pending key rotations before recording weights and positions; otherwise the
generic snapshot format would discard the pending deltas. Whole-session
malformed restore and arbitrary allocator/backend failure atomicity remain
separate gates.

Verification:

- 13 focused Release CTests and three strict CPU ASAN/UBSAN/leak CTests pass.
  New cases cover coincident positions, partial position reordering, exact
  rolling hashes after copy/restore/removal, pending deltas and output positions,
  causal masks, version/order rejection and malformed per-sequence KV/history
  recovery. Initial test build omitted three required API flag arguments; the
  stale executable run afterward is explicitly excluded from qualification.
- Real model: 24 exact logit/route lifecycle comparisons, including uniform
  shift, division by two, partial shift, full-state roundtrip and sequence
  migration. The original nine-token baseline logits, input IDs and routes
  are byte-identical to the preceding validated runtime. Every original
  serialized K/V/indexer/recurrent payload is byte-identical after accounting
  for the new metadata representation (118,480,896 prefix payload bytes and
  118,536,192 continuation payload bytes).
- A separate six-case transform run under unsuppressed all-API GPU memcheck
  finishes with zero errors and zero leaked bytes. RAM/staging caps and complete
  lease/load/reservation drain pass in both model runs.
- Independent F64 partial-NeoX rotation of the saved original F32 cache keys
  matches all 36 layer/transform cases (147,456 values), maximum error
  1.9073486328125e-6, at the unchanged `2e-5*(1+abs(reference))` bound.
  Nonrotary key bytes, all values, raw indexer keys and recurrent/PLE tensors
  remain exact through each transform. Actual positions/ordinals/history are
  checked separately, including equal and non-monotonic rotary positions.

This does not compare transformed inference with a full upstream re-prefill:
context shifting preserves historical hidden/recurrent computations and
changes the positions of cached keys. It does not qualify long-context sparse
selection, alternative cache types/devices, shared-row isolation, overlapping
private copies, or speculation after transforms. The 33-token strict numerical
discrepancy remains open with its original goldens and bounds preserved.
No speed claim or production promotion is made. Full memory accounting,
stress/concurrency, kernels/cache/speculation, fair Strata benchmarks,
serving, README rebrand and GitHub publication all remain active.
