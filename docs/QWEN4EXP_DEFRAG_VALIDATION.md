# Hybrid cache defragmentation

Validated 2026-10-11. Evidence:
`/home/xero/Desktop/test/results/ese-qwen-defrag-20261011/qualification-final.json`.

Two errors were repaired. Attention-cell movement now preserves the recurrent
source controls at both physical indices. Value movement follows `v_trans`,
rather than assuming the Flash Attention setting determines the stored layout.
Hybrid caches store token-major values even with Flash Attention disabled.

The original-model negative moves actual rows and changes recurrent controls
from[0,1] to[15,16], then fails before an invalid recurrent read. The independent
pre-fix non-Flash copy graph changes value-row bytes. Both failures are retained.

Verified scope:

- 18 overlapping K/V/indexer raw-copy cases pass for F32,F16,Q8 storage, preserving
  recurrent bytes; Release and strict CPU sanitizers pass.
- Six native model runs cover unchanged positions, division, whole shifts,
  partial shifts, F16 K/V and Q8 keys with F16 values. All30 numerical comparisons
  are exact, including both continuations/checkpoints and sequence copying.
- Five additional comparisons pass under all-API GPU memcheck after a partial
  shift: exit0, zero errors/leaks,356.96s. Total model comparisons35.
- Independent canonicalization by chronological ordinal proves28 before/after
  and rollback snapshots preserve metadata and every K/V/recurrent/indexer byte.
- 17 Release/five CPU sanitizer checks pass; RAM resident/staging caps and
  lease/load/reservation drain pass in every accepted run.

The first negative attempt recreated no holes because snapshot restore compacts
the cache. It is excluded; the corrected fixture recreates holes after restore.
The first Q8 run fails the old cache-gather type assertion and was stopped during
its backtrace; its logs/exit remain. Q8 gathering was added and independently
tested in the separate `ese-qwen-q8-cache-order-20261011` stage. Accepted Q8 and
final memory runs use that newer frozen generation; earlier runs retain their
original runtime identities.

This is short unsplit text inference on RTX3060Ti/RTX3080, non-Flash cache.
Compressed formats are compared with the same-format control, not with an F32
quality reference. Split layouts, long context, broader hardware/families,
retiering and upstream numerical/performance gates remain separate. Normal
Qwen4Exp loading remains gated; no throughput improvement is claimed.
