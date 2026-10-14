# DeepSeek HC normalization validation — 2026-10-13

CPU and CUDA `HC_PRE` now add epsilon after each completed row or column
reduction. This matches the pinned `hc_split_sinkhorn` implementation. Adding
epsilon to the initial accumulator changes float rounding and caused the
initialized token reference to fail. The standalone `SINKHORN` operation is
outside this correction.

Evidence: `/home/xero/Desktop/test/results/ese-deepseek-hc-attribution-20261013`.
Pinned reference kernel SHA-256:
`1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455`.
The saved model weights, inputs, references and their tolerances are unchanged.

## Qualification

- A permanent scalar check covers 1–8 streams, 2 and 20 iterations, and three
  tokens: 48 normalization matrices and 1224 combination coefficients per run.
  CPU coefficients must equal the scalar oracle exactly. CUDA coefficients
  use a separate 3e-7 component bound. The old frozen library fails this check
  at two streams; the corrected CPU and all three GPUs pass.
- Full Linux CPU Release, CUDA Release and CPU Debug ASan/UBSan builds pass.
  Complete registered suites pass 91/92, 95/96 and 91/92 respectively. CPU
  sanitizers enable leak checking and stop on sanitizer errors. The only
  remaining suite failure is `test-deepseek-v41-block`.
- The original ten-token whole-model reference now passes in all three builds,
  including its abort and injected-failure replay. No reference tolerance was
  changed to obtain this result.
- Direct primitive checks pass on RTX 3060 Ti, RTX 2080 SUPER and RTX 3080.
  Separate Ampere-pair memory checks report zero errors and zero leaked bytes.
- The previous CPU Release evidence directory remains unchanged; the new CPU
  qualification uses a separate build directory. A parent-driver stop left an
  initial sanitizer run without a wrapper exit receipt; its full JUnit result
  is retained, and a fresh captured run supplies the final qualification.

## Block difference at the initial qualification

The block reference first differs at token 7, layer 7, element 82. At the
attention residual expansion, native arithmetic exactly matches the reference
when both receive the same native coefficients. The coefficient differences
remain at most 2.98e-8 in the same-input comparison. A scalar `exp/sum`
softmax replay reproduces the native combination coefficients; the pinned CPU
replay's fused Torch softmax differs slightly.

One intermediate reaches a BF16 tie: the source float is
`-1.6445313692092896`, while native produces `-1.64453125`. Their BF16 values
are `-1.6484375` and `-1.640625`. The saved same-input attribution isolates this
difference; it does not justify an additional arithmetic change or close the
block gate.

This is initialized small-model evidence. Trained-model quality, original Qwen
output parity, full 64K inference, concurrency, matched benchmarks and Windows
packaging remain unqualified. Normal Qwen loading stays gated. The release
remains unready.

## Subsequent expansion qualification — 2026-10-13

The retained block failure above is resolved by native FP64 coefficient and
residual expansion. The original HC_PRE normalization oracle remains unchanged.
Current full CPU/CUDA/strict sanitizer suites pass 92/92, 100/100 and 92/92;
separate block/token controls and zero-error, zero-leak primitive memory checks
pass on all three cards. See
[HC expansion validation](DEEPSEEK_HC_EXPANSION_VALIDATION_20261013.md) for the
correction, full-fixture proof and evidence limits.
