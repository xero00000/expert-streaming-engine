# Failed sequence restore ownership and CUDA copy context

Validated 2026-10-11. Evidence:
`/home/xero/Desktop/test/results/ese-sequence-restore-ownership-20261011/qualification-final.json`.

Malformed per-sequence state previously cleared saved output/checkpoint ownership
before cache recovery. Buffer rollback now preserves that ownership; file loading
stages prompt tokens and commits them only after successful transactional restore.
Valid replacement state still invalidates the previous checkpoint and output.
Failed recovery invalidates ownership rather than advertising stale state.

The original model test uses different incoming prompts, so the failing candidate
changes K/V, recurrent and indexer payloads before a late truncation. All four
buffer/file recoveries in the native and corrected GPU runs reproduce the target
sequence bytes exactly. Full-session bytes remain unchanged on failure, caller
prompt tokens/count remain unchanged, and both interleaved checkpoints recover.

The initial GPU sanitizer run exposed 160 CUDA API errors: asynchronous tensor
copy did not select the source device when the caller's current context belonged
to another GPU. Selecting the device before both same-backend and same-device
copy branches repairs the error. No numerical kernel was changed or error hidden.

Verified results:

- 15 Release tests and five CPU ASAN/UBSAN/leak tests pass.
- Pre-fix async-copy control reports 24 stream/context errors; corrected 24 cases
  on all three physical GPUs pass with zero errors/leaks and exact output bytes.
- Original native model run: 36 exact lifecycle comparisons. Stronger
  different-candidate native run: 14 exact comparisons. These preceded the CUDA
  context fix and carry no corrected GPU sanitizer claim.
- Corrected full-model GPU run: exit0, 14 exact logit/route comparisons,
  zero all-API memcheck errors and zero leaked bytes, 913.93 seconds.
- RAM resident/staging peaks stay within configured caps; loads, leases,
  reservations and staging drain. Nine-token baseline logits/input/routes and
  prefix/continuation snapshots match the preceding frozen shared-prefix runtime.

The failed 160-error run, stale-client failures and initial build failures remain
in the evidence. `audit-final.py` checks terminal results, caps/drain, retained
negative controls, baseline identity and the independent payload parser.

Scope is short original-weight Qwen text inference, F32/non-Flash cache, unsplit
layout on the selected RTX3060Ti/RTX3080 pair. Copy components also cover the
RTX2080SUPER. This does not establish full-session failure atomicity, arbitrary
allocator/backend failure recovery, bounded file-transaction host memory,
long-context correctness, full upstream parity or a performance improvement.
The production architecture gate remains closed. The separate 33-token numerical
gate and the complete roadmap remain open.
