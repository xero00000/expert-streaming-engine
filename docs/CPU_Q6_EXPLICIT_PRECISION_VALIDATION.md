# CPU Q6_K explicit projection precision

CPU `GGML_PREC_F32` matrix multiplication now preserves F32 activations for
Q6_K weights. Previously, only floating weight types honored that path; Q6_K
still entered the activation-quantizing kernels despite the precision request.
Default-precision dispatch and indexed expert arithmetic are unchanged.

The correction dequantizes one 256-weight block into a fixed 1 KiB worker-local
array, accumulates against the original F32 input in F64 and publishes F32.
It does not allocate a complete dequantized CPU weight matrix. This is a
precision correction, not a measured speed improvement.

Evidence: `/home/xero/Desktop/test/results/ese-q6-cpu-precision-20261011`.
`v2` preserves final runtime libraries, component controls and model regressions;
`v3` adds the final test's original model-payload projection checks, using those
same immutable libraries. Earlier failures and generations remain separate.

## Independent projection and dispatch controls

The existing Q6 oracle now executes on CPU as well as CUDA. Six contiguous
cases cover widths 256/2560/10240, output rows 9/321/129 and one/two input
tokens. Eight layout cases add padded weight rows, poisoned activation padding,
broadcasting and 1/2/8/128-token batches. The independent oracle dequantizes
original packed weights and computes F64 products before F32 publication.
The unchanged absolute tolerance is `<2e-5`.

Old CPU code fails the first case at approximately `2.95162e-4`. The repaired
CPU matches all 14 synthetic projection cases exactly. All three GPUs pass
supported cases and CUDA memcheck with all API errors enabled, zero errors.
The maximum recorded GPU error is `7.45058e-7`.

CUDA rejects broadcasting along the fourth axis. Four cases per GPU run are
explicitly evaluated with CPU buffers/backend after checking that capability;
four layout cases and all six contiguous cases execute on the requested GPU.
This tests the fallback operator's math, not automatic scheduler/device-copy
behavior. It does not claim native CUDA support for the rejected layouts.

The initial harness forced an unsupported CUDA operation. It exposed a separate
dispatcher error: returning `-1` from a boolean function reported success and
left the output unwritten. The refusal now returns `false`, following the
backend's existing unsupported-operation assertion path. The same executable
under old libraries reports success with an unchanged output canary; the repaired
library rejects the forced call. Ordinary scheduling already consults the
support query and remains the intended way to select a fallback.

Six default-precision output digests match old/new libraries byte-for-byte on
CPU and each GPU: 24 controls total. Both original failures are retained.
The final operator oracle passes strict CPU ASan/UBSan/leak checks.

## Original model tensors

The final real-layer test retains original Q6 bytes while reading and verifying
the GGUF tensor payload hashes. It tests the production layer-0 QKV matrix
(`10240 x 2560`) on the pinned fixture input and the output matrix
(`2560 x 6144`) on the pinned gated output. It never requantizes these tensors.

Both the captured-input and five-token synthetic-input fixtures pass on CPU
and all three GPUs natively and under unsuppressed CUDA memcheck. Both also pass
strict CPU ASan/UBSan/leak checks: 16 final real-layer runs. The captured old-CPU
QKV error is `0.0262485`; the repaired maximum is `5.72205e-6`, within the
unchanged `<2e-5` tolerance. The existing BF16/F32 alpha and complete recurrent
F32-weight batch/cached-decode checks also pass in those runs.

Ten final normal focused CTests pass. These component tests do not yet assemble
an entirely packed recurrent graph: the real Z projection is Q4_K, whose CPU
precision request remains separate work. Indexed expert activation arithmetic
and whole-model upstream parity also remain open.

## Model regression scope

All four frozen-library model controls are terminal/pass: native/all-API
sequence-bank focus and ordinary CPU/GPU teacher runs. The focus verifies
target 0, four accepted tokens, microbatch 2, complete state/logits, continuation,
logical QSA IDs and independent discard; full-model memcheck reports zero errors.
Each ordinary teacher passes 21 lifecycle/checkpoint checks with 8,194,560 F32
teacher values and baseline logits/routes byte-identical to the preceding
qualified generation. RAM/staging caps and final lease/load/reservation drain
pass. These preserve controlled behavior; they do not prove full-model agreement
with the upstream implementation, broad resource safety or a Strata speed win.
