# Speculative journals after position transforms

Validated 2026-10-11. Scoped evidence:
`/home/xero/Desktop/test/results/ese-qwen-transformed-journal-20261011/qualification-final.json`.

Partially shifting a prefix can put an earlier token at a larger rotary
position than the latest token. Journal save now validates the last output by
chronological ordinal and stores its actual rotary position. Base-only recovery
restores that position instead of deriving it from the maximum cache position.
The next verification window still begins beyond the maximum retained position.

All30 original-model cases pass: shifts, division and partial shifts, both
independent sequence owners and every accepted cut0–4. Complete sequence snapshots,
owned logits, MoE/QSA routes, continued outputs and independent discard are exact.
An immediate empty next window also saves/restores correctly. A focused partial
shift case passes all-API GPU memcheck with zero errors and zero leaked bytes;
its terminal runtime is423.81s. RAM resident/staging caps and lease/load drain pass.

17 Release tests and five strict CPU ASAN/UBSAN/leak tests pass. Additional fixtures
cover physically reordered cache rows, signed-zero output/state bytes and stale
output rejection without replacing a live checkpoint; final fixture checks pass
in both builds. The matched pre-fix model rejects a valid partially shifted save.

The first corrected run omitted the four previously qualified stable batch
settings and fails byte identity in the accepted sequence, while the untouched
sequence remains exact. The repeated failure, raw snapshots and independent
payload comparison are retained. Successful runs explicitly enable
`ESE_CUDA_DENSE_BATCH_VECTOR`, `ESE_CUDA_ATTENTION_STABLE_BATCH`,
`ESE_CUDA_FUSED_BATCH_VECTOR` and `ESE_CUDA_BF16_STABLE_BATCH`. This is not evidence
that default dispatch has exact scalar/batch identity. No tolerance was changed.

Scope is short experimental Qwen text inference on the selected RTX3060Ti/RTX3080
pair, complete-row journals, F32/non-Flash cache and up to four proposed inputs.
This does not certify actual drafting, long context, arbitrary backend failure
atomicity, other layouts/devices, full upstream parity or performance. The
production loader gate remains closed and the separate33-token gate remains open.
