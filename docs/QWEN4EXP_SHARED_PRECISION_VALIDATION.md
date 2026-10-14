# Original-weight shared projections and padded CUDA inputs

Evidence: `/home/xero/Desktop/test/results/ese-qwen-shared-f32-20261011`.
This is a scoped correctness qualification. Full upstream model parity and
performance qualification remain open; the production architecture gate is
unchanged.

## Shared expert arithmetic

The experimental `ESE_QWEN4EXP_EXPERT_F32=1` policy now requests explicit
precision for shared gate, up and down projections. Its shared FFN uses
separate projections rather than the activation-quantizing fused kernel.
Unsupported split tensors, biases/scales, adapters and activation contracts
are rejected under this policy. Default policy and other architectures keep
their existing builder choices.

CPU explicit packed projections now include Q5_K, IQ4_NL, IQ4_XS and Q8_0,
in addition to the previously qualified Q4_K/Q6_K and floating formats.
They decode one native block at a time into a fixed 1 KiB worker-local array,
accumulate original decoded weights against F32 activations in F64, and round
the result to F32. Typed decoder calls avoid incompatible function-pointer
dispatch. The CUDA explicit path already preserves F32 activations through
dequantization and floating matrix multiplication; it is requested here.
This does not imply CPU/CUDA dot products have universal bit equality.

Original payloads from layers 0, 4 and 47 cover all six actual shared packed
formats, including different gate/up formats. The new `test-shared-f32`
executes separate gate/up, SiLU/product and down stages for 1/2/8 tokens,
three graph replays and padded input strides. The independent fixture oracle
decodes original weights in Python, accumulates projection dots in F64 and
applies F32 activation/product rounding. Every value must stay within
`2e-5 * (1 + abs(expected))`; the threshold is fixed before native execution.
The old CPU explicit IQ4_XS path fails this same oracle at the first up value,
with error `0.000765242`.

## Existing CUDA stride defect

The initial default CUDA component check produces nonfinite output with
padded token rows. Its direct quantization path reads flat rows despite the
input's row stride. The initial failure and executable are retained.

Direct quantized matrix dispatch now requires contiguous weight, input and
output tensors. Other layouts use the existing packing path. Reuse of a
following quantized matrix node also checks its weight/output layout before
consuming it. The tested padded inputs now match equivalent contiguous
inputs byte-for-byte. The tests do not certify every possible weight/output
view or unsupported layout.

## Qualified component checks

All 78 final component profiles pass: CPU at 1/8 threads and each of the
RTX 3060 Ti, RTX 2080 SUPER and RTX 3080. Fifteen explicit native and nine
all-API GPU memory profiles each compare 147,840 values, including replay.
Nine additional all-API padded-default GPU checks also pass and reproduce
their contiguous/default artifacts exactly. All 18 GPU component memory runs
report zero errors and zero leaked bytes. Six strict
CPU ASan/UBSan/leak profiles also pass.

Fifteen original/default and fifteen current/default profiles produce
byte-identical output artifacts. Fifteen additional padded-default profiles
match their contiguous controls exactly. Each GPU is independently exposed
for component validation. Twelve focused normal CTests pass, including
production loader rejection, recurrent/state tests and existing kernel tests.

## Model comparisons and remaining boundary

On the exact same captured layer-zero FFN input, shared output maximum
error falls from `0.00293624` to `2.38419e-7` against the pinned upstream
method. Routed output remains at `4.47e-8` and the shared gate at `5.96e-8`.
Complete FFN output error is `1.04308e-7` on that input. End-to-end layer-zero
FFN error is `1.63913e-7`, and layer-zero residual error `9.68575e-8`.

The complete first-token logit maximum error decreases from `0.158851`
to `0.122928`, still outside full-model parity. The next isolated discrepancy
is layer-one PLE: input embeddings are byte-exact, but the same-input key
projection error is `0.000888526`, value projection `0.000182755`, and
post-normalization key error `0.0582542`. PLE projection precision requires
its own correction and qualification.

Final model controls are terminal and audited. CPU-expert/hybrid execution
produces byte-identical 2,234,880 teacher values and per-token routes. The
full-model hybrid all-API memory check reports zero errors/leaked bytes,
with bounded RAM/staging and clean final drain. Batched verification at
depths one/eight matches nine target outputs/routes plus complete saved
state and continuation for both depths. Independent probe observations
verify explicit precision and actual GPU execution for all 48 shared gate,
up and down projections on each of 18 tokens across two placements. Matched
default model teachers/routes remain byte-identical to the preceding frozen
runtime. `qualification-final.json` records these scoped checks. Full upstream/state/long-context/resource stress,
kernel/cache/speculation, GLM/Maya, matched Strata, serving and publication
work remain on the roadmap. No model speedup or release promotion is claimed.
