# Qwen cache format handling — 2026-10-11

Ordered cache gathers now accept BF16, Q4_0, Q4_1, Q5_0 and Q5_1 alongside
F16, F32 and Q8_0. F16/F32 retain the existing gather representation; the
other formats decode selected rows into F32 and leave persistent bytes
unchanged. This removes late assertions for formats the native backend can
already read and write.

Qwen context setup rejects unsupported global cache formats before backend
initialization. Cache allocation checks effective per-layer key/value types,
indexer type and block-aligned head widths before creating cache tensors.
Retiering checks requested active-layer formats before preparing replacements.
Quantized values retain the Flash Attention requirement. Generic model cache
policies remain unchanged; the production Qwen loader gate remains closed.

Four Q4/Q5 codec bridges replace incompatible generic function-pointer casts
with typed calls. Encoding and decoding arithmetic is unchanged. This fork's
Q4_0 runtime encoder deliberately uses IQK's optimized implementation, which
differs from the scalar reference; migration checks use that selected codec.

## Verified scope

Evidence: `/home/xero/Desktop/test/results/ese-qwen-cache-formats-20261011`.

- 217 gather cases pass on CPU and all three GPUs: the original 54 F16/F32
  cases, 162 BF16/packed cases and an odd-width BF16 case. Strided heads/rows,
  offsets, reverse order and repeated indices are covered. Decoded values and
  persistent source bytes are exact. All three GPU component memchecks report
  zero API errors and zero leaked bytes. Those component runs precede the
  later context block-width assertion removal and diagnostic fixture additions;
  their gather/codec source is unchanged.
- Final 18 Release and seven CPU ASAN/UBSAN/leak regression tests pass.
  CPU migration checks match direct selected codecs and independent IEEE BF16
  conversion, reconstruct exact decoded rows, and preserve recurrent storage.
  Unsupported context/migration requests preserve old storage/checkpoints.
- Original-weight model runs with BF16/BF16, Q4_0/F16 and Q5_1/F16 each pass
  four exact rollback/checkpoint continuation comparisons. A BF16/BF16 GPU
  model memcheck repeats those four comparisons with zero errors and leaks.
  All runs use the selected Ampere pair and the frozen diagnostic runtime.
- The independent snapshot parser verifies 16 published/committed snapshots:
  chronology/positions/history unchanged, recurrent/indexer bytes unchanged,
  key/value migration bytes matching IEEE conversion or the selected runtime
  codec, and published/committed payload identity. Resource caps and drain pass.
- The initial migration fixture assumed scalar Q4_0 encoding and failed.
  Its corrected oracle uses the configured runtime encoder without changing
  production encoding or loosening byte equality. Failed logs remain retained.

## Remaining gates

These are short, unsplit, token-major cache/lifecycle checks with non-Flash
model attention. Exact continuation comparisons use controls in the same
converted representation; they do not certify quantization loss versus F32,
full upstream parity, long context, Flash Attention combinations, split layouts,
other architectures or all-visible/Turing whole-model GPU sanitization.

The original 33-token numerical gate remains failed; normal Qwen loading stays gated.
This stage makes no throughput, release-readiness or complete memory-budget
claim. Later PLE allocation-accounting changes are a separate source generation.
