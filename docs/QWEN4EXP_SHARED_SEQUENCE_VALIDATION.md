# Qwen shared-prefix isolation and overlapping copies

Evidence: `/home/xero/Desktop/test/results/ese-qwen-shared-sequence-20261011`.
B4 remains in progress. The normal production architecture gate is unchanged.

Copying between independently restored versions of the same prefix previously
left duplicate chronological tokens in different physical rows. The new
regression fails before repair with duplicate token order in attention. Copy
now removes redundant destination ownership for copied ordinals before sharing
the source rows. It preserves unrelated owners and the hybrid recurrent-slot
`src` fields. Empty/out-of-range copies preserve destination state; invalid
sequence owners are rejected before publication.

Position add/divide now isolates changed shared rows for the selected owner.
The operation prepares free-row reservations, metadata, PLE history and output
positions, validates cache layouts/capacity/overflow, then copies exact K/V/raw
indexer row bytes through the backend. Only afterward are ownership, positions
and checkpoints published. Recurrent-slot ownership is kept independent of
physical attention-row movement. The visible cache extent grows to include
new rows. Removing negative shifted tokens detaches the selected owner without
copying or changing the other owners' rows.

This implementation supports contiguous, token-major, unsplit cache tensors.
It rejects unsupported split/transposed layouts and insufficient free rows
before live-state mutation. Prepared metadata and PLE-history clones require
host memory proportional to the affected cache/history; complete allocation
accounting and arbitrary backend/allocation-failure qualification remain open.
No host fallback is introduced for the tested same-device CUDA row copies.

Verification:

- 13 focused Release CTests and three strict CPU ASAN/UBSAN/leak CTests pass.
  Component cases cover exact F32, F16 and packed Q8_0 row bytes across K/V/raw
  indexer copies, preservation of other owners' metadata/history/checkpoints,
  capacity failure, negative removal, empty/invalid copies and overlap repair.
- CUDA row-copy components pass unsuppressed all-API memcheck on RTX 3060 Ti,
  RTX 2080 SUPER and RTX 3080, with zero errors and zero leaked bytes on each.
  This is component coverage, not whole-model Turing qualification. An initial
  optional-GPU test compile omitted two required backend-init arguments; it
  was corrected before the GPU tests and the failed log is retained.
- Real original-weight model on the selected Ampere pair: 32 exact logit/route
  comparisons, including shared uniform shift, division and partial shift,
  selected-private controls, untouched-owner continuations and overlapping
  restored-prefix copies. Unselected sequence snapshots remain byte-identical.
- Separate 14-case shared/position model run under unsuppressed all-API GPU
  memcheck finishes with zero errors and zero leaked bytes. RAM/staging caps
  and complete lease/load/reservation drain pass in both model runs.
- Independent snapshot audit canonicalizes physical rows by persistent ordinal.
  All selected-owner K/V/indexer/recurrent payloads match the private controls;
  all other-owner payloads match the original prefix, across both runs and all
  three transformations. The original nine-token logits/input/routes and both
  prefix/continuation sequence snapshots remain byte-identical to the preceding
  validated generation.

A separate, unregistered next-task regression deliberately fails: malformed
sequence restore clears the destination's previous saved-output/checkpoint
ownership even when cache/history recovery succeeds. This is not counted as a
passing test or a fixed bug. Its source, binary, raw failure and exit record are
retained as `next-restore-ledger-*` evidence. Fix this next, then qualify
speculation after transformed prefixes, interleaved recovery, compressed cache
defragmentation/retiering and failure stress.

Long contexts, split/other cache layouts, full-model device coverage, concurrent
serving and the 33-token upstream numerical discrepancy remain open. No
threshold changes, speed claim, capability promotion, launcher change or
publication occurs. The complete resource/kernel/cache/speculation/GLM-Maya/
Strata/serving/README/GitHub roadmap remains active.
