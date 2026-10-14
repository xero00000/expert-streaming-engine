# Qwen4Exp recurrent alpha and current-gate requalification

The historical captured-input CPU gate failure is resolved by the current
implementation's existing explicit projection precision. This checkpoint
regenerates and qualifies the current oracle; it does not introduce another
production math change.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-alpha-precision-20261011`.
The initial generation preserves the unchanged test; `v2` contains the final
test with actual BF16 alpha weights, equivalent F32 weights and a diagnostic
negative control. Historical fixtures and shared model shards remain unchanged.

## Reference and diagnosis

The generator verifies modeling source revision
`fc5c5bde8e656dad91cbf34e61940d984b1c7b91` against its pinned SHA-256. The pinned
model configuration supplies `output_gate_type = sigmoid`. The older captured
fixture lacks that contract and cannot qualify the current complete layer.

A separately regenerated one-token fixture uses the original captured F32
activation and per-tensor-verified real GGUF weights. It regenerates
byte-identically: file SHA-256
`2c9d171eeab39b7b160c868c280902af621f8b478c63b6e3bca9997cd117c8a2`,
content digest `e80b742a203833f796210f9991517ece8574bba919d0fe31350e36f8059c1e45`.
A five-token synthetic-input fixture uses the same real weights and current
gate contract, covering convolution rollover and persistent recurrent state.
Its content digest is
`9afb9e4d02a1482b949e88b572a386951e97b364551d27c567b7180c3fc6be1e`.

The independent F64 dot-product study is in the separate
`ese-qwen-alpha-oracle-20261011` bundle. At historical stored head 38, it yields
alpha `0.283768856354011`; nearest F32 is `0.2837688624858856`. Historical default
CPU accumulation gave `0.28376856446266174`. Current explicit-precision CPU
execution gives the nearest F32 value, and its head-38 gate is exactly the
pinned `-115.39457702636719`.

## Final checks

The native test's new real-alpha graph verifies that re-encoding the originally
BF16 weights is bit-exact, then evaluates both BF16 and equivalent F32 alpha
projections with F32 activations, the real bias and decay coefficient. Both
projections and gates must match the pinned reference at the unchanged absolute
`<2e-5` tolerance. The captured gate maximum error is `3.8147e-6`; the five-token
gate maximum is `7.62939e-6` in the final native CPU checks. It also executes the
complete recurrent-layer F32-weight oracle for both batched and cached decode.

`QWEN4EXP_TEST_ALPHA_DEFAULT=1` affects only the test's F32 alpha projection.
The negative control reproduces the historical gate failure, approximately
`2.28882e-5`, while the explicit BF16 gate passes. This test-only control is not
a production runtime option.

Both fixtures pass on CPU and all three GPUs (3060 Ti, 2080 SUPER, 3080),
including unsuppressed CUDA memcheck with all API errors enabled: 14 final runs,
all exit 0, zero memory/API errors in the six instrumented GPU runs.
Both final CPU ASan/UBSan/leak checks are terminal exit 0, with all projection,
gate, complete-layer and persistent-state comparisons passing. Eight focused
ordinary CTests pass. `v2/qualification-audit.json` records all 16 final runs,
unchanged tolerance checks, the negative control and fixture reproducibility.

## Scope

The captured activation is historical input, not a new full-model capture.
The complete-layer oracle dequantizes projection weights to F32; the additional
alpha check exercises the actual BF16 representation. Passing this checkpoint
does not qualify every production quantized projection or expert kernel,
full-model upstream logits/routes, long contexts, arbitrary serving workloads
or a speed improvement. Production promotion remains gated by the full roadmap.
