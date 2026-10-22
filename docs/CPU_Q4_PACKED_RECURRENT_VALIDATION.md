# CPU Q4_K precision and original-projection recurrent graph

CPU explicit F32 matrix multiplication now preserves F32 activations with
Q4_K weights, using the same bounded block-dequantization path as Q6_K.
The worker-local temporary remains 256 floats (1 KiB); the default quantized
dispatch and indexed expert arithmetic are unchanged.

Evidence: `/home/xero/Desktop/test/results/ese-q4-packed-recurrence-20261011`.
Model payloads and historical reference fixtures are read-only. The current
sigmoid-gate fixtures retain their pinned modeling/configuration identities.

## Precision and original tensor checks

The original layer-0 Z projection is Q4_K, with 6144 output rows and width 2560.
Old CPU code ignored the precision request and produced maximum error
`0.0375867` on the captured input. The repaired original-payload projection
has maximum error `2.86102e-6`, below the unchanged absolute `<2e-5` tolerance.

The synthetic projection oracle now tests both Q4_K and Q6_K: six contiguous
and eight layout cases per family, including poisoned input padding, padded
weight rows, broadcasting and batches of 1/2/8/128 tokens. CUDA's unsupported
fourth-axis broadcasting uses explicitly labelled CPU buffers/backend, as in
the preceding Q6 study; those cases do not qualify automatic scheduler copies.
The other cases execute on the requested backend.

Twelve default-precision output digests match old/new libraries on each of
CPU, 3060 Ti, 2080 SUPER and 3080: 48 byte-exact controls. The old-library
original Q4 projection failure is retained separately.

## Complete recurrent graph

The test now retains every original tensor payload after checking its recorded
GGUF SHA-256. It runs the previous F32-weight oracle, then a second complete
graph with all five original projection tensors:

- QKV and output: original Q6_K bytes.
- Z: original Q4_K bytes.
- Alpha and beta: original BF16 bytes.

The convolution, bias, decay and norm parameters retain their F32 representation.
An assertion requires all five original projection payloads to be used. Original
quantized tensors are never requantized for this graph.

The captured-input one-token fixture and five-token synthetic-input fixture
both pass batched and persistent-cache single-token execution. Comparisons cover
every projection, gate, convolved input, recurrent output, gated output, final
output, convolution state and recurrent state at the unchanged `<2e-5`
tolerance. The five-token case crosses the four-tap convolution rollover.

All 14 native/GPU-memory runs and both strict CPU ASan/UBSan/leak runs pass.
All three GPUs pass unsuppressed CUDA memcheck with all API errors enabled,
zero errors. Maximum error across all reported comparisons is `7.62939e-6`.
Ten focused normal CTests pass. `layer-qualification-audit.json` verifies both
float/original graph diagnostic counts and the default controls.

## Remaining scope

All four full-model controls are terminal/pass and audited: native/all-API
sequence-bank focus and ordinary CPU/GPU teacher runs. Full-model memcheck
reports zero errors. Each ordinary teacher passes 21 lifecycle/checkpoint checks,
with 8,194,560 F32 teacher values and baseline logits/routes byte-identical to the
preceding qualified generation. RAM/staging caps and final lease/load/reservation
drain pass. The bank focus covers target 0, four accepted tokens and microbatch 2;
it checks complete state, continuation, logical QSA IDs and independent discard.
This checkpoint does not establish whole-model upstream logits/routes,
cross-device indexed-expert arithmetic, broader quantization families,
arbitrary serving/long-context safety or a speed improvement. Those release
gates and the Strata comparison remain open.
