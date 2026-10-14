# Original-weight F32 expert arithmetic prototype

The CPU/CUDA prototype removes the activation quantization difference identified
in the hybrid expert investigation. Both paths consume the original quantized
weights and original F32 activations, accumulate 32 ordered F64 partial sums,
reduce with the same tree, and round projections to F32. The fused SiLU/product
uses F64 intermediates and rounds its result to F32.

This is an experiment, not production dispatch or a model-support promotion.
Evidence: `/home/xero/Desktop/test/results/ese-expert-f32-prototype-20261011`.

## Implementation and reference

- `ggml/src/ggml-expert-f32.h`: CPU block decoding, 256-float temporary and
  32 double accumulators (1.25 KiB), without materializing a full expert.
- `ggml/src/ggml-cuda/expert-f32.cuh`: original IQ3_XXS gate/up and Q2_0 down
  decoding, warp reductions and fused up/gate execution. Explicit conversion
  handles subnormal F32 inputs and outputs even with CUDA fast math enabled.
- `tools/expert-f32-probe.cu`: original-payload replay, independent full-row
  dequantization/long-double oracle, comparisons, memory checks and component
  timings. Its oracle also checks the complete hidden/down path using an
  independently computed long-double SiLU/product.

The retained layer-20/expert-10 payload has 2560 input channels, 640 intermediate
channels and 2560 output channels. Original up/gate/down bytes total 1,715,200;
their SHA-256 and the captured input hash are verified against `weights.json`.
No weight is requantized. The input comes from decode step 3 of the retained
failed-residency trial.

## Qualification

Each GPU checks all 4,915,200 decoded weight values against the CPU decoder,
byte-for-byte. Four inputs cover the real capture, a deterministic alternating
pattern, zeros and F32 subnormals. Up, gate, fused hidden and down outputs match
CPU/GPU byte-for-byte, including zero signs. Each run checks 17,920 output
values for those four inputs.

The independent full-row/long-double oracle passes the unchanged absolute
`<2e-5` bound for projections, hidden and complete down output. All reported
comparison errors are zero in these cases. All three GPUs pass native and
unsuppressed all-API memcheck/leak runs: zero errors and zero leaked allocations.
The first three-GPU generation was retained separately before adding the
independent hidden oracle; the final generation revalidates that additional
check. This is one actual expert, two quantization families and contiguous
single-token buffers, not general model or serving qualification.

## Measured cost and remaining work

The 200-replay resident GPU graph timings cover fused up/gate plus down. Weights
are already uploaded; the graph has no debug projection outputs. These are
component timings, excluding storage, transfers, resource planning and scheduler
integration. The 3060 Ti/2080 SUPER/3080 measurements are approximately
0.50/0.41/0.29 ms per expert, respectively. The scalar single-thread CPU reference
is untuned; its captured-input cost is approximately 19 ms in this build.
Neither measurement establishes an engine speed improvement.

Next: improve measured CPU/GPU cost while preserving a qualified common
arithmetic contract, widen inputs/layouts/types, and integrate indexed/fused
dispatch with bounded leases, scratch and explicit fallback reporting. Then
validate whole-model logits/routes and placement invariance against independent
references. The old hybrid baseline failure remains recorded in
`CUDA_IQ3_SCALE_VALIDATION.md`; this standalone prototype does not waive it.
The wider loader/lifecycle/resource/64K/cache/speculation/GLM-Maya/Strata/launcher
and publication roadmap remains open.
