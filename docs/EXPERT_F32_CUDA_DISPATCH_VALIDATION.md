# Explicit expert arithmetic: CUDA indexed and fused dispatch

`GGML_PREC_F32` selects the original-weight/F32-input expert implementation
for CUDA `MUL_MAT_ID` and `MOE_FUSED_UP_GATE`. It covers IQ2_S, IQ3_XXS,
IQ3_S, IQ4_NL, IQ4_XS and Q2_0, using the previously qualified decoders and
ordered F64 reduction. Fused execution supports unbiased, unclipped SiLU.
Default-only operations retain their existing dispatch.

Evidence: `/home/xero/Desktop/test/results/ese-expert-f32-indexed-20261011`.
The authoritative final results are `qualified-layouts`, `qualified-graphs`,
`runtime-manifest.json`, `focused-ctest.log` and `qualification-audit.json`.
Earlier logs preserve intermediate versions and failures; they are not final
qualification. The final runners preload the frozen GGML/llama libraries and
execute frozen test binaries.

## Layout and execution contract

The kernels read routing IDs on the device, retain F32 activations and write
only logical output elements. They respect independent up/gate row and expert
plane pitches, strided activation slots/tokens, strided ID rows and padded
outputs. Input slots broadcast by route-slot modulo input-slot count. Duplicate
IDs execute independently; negative and out-of-range IDs produce positive zero.
Merged weights contain gate rows followed by up rows. The up offset uses the
logical gate row count times the row stride, preserving extra plane padding.

Support queries validate types, shapes, block widths, alignment and launch
dimensions, and reject split-buffer parents. Queries permit host/other-device
source buffers so the scheduler can arrange copies. Execution additionally
requires every operand and output in a regular CUDA buffer on the selected
GPU. These kernels introduce no activation quantization or transfer scratch.
CPU prepared-activation/lease workspace remains separately accounted for.

## Qualification

The 82 layout runs cover four original-payload format pairs from layers
0/1/17/47, separate and merged gate/up, all three GPUs, native execution and
unsuppressed all-API memory/leak checks. CPU native and strict ASan/UBSan/leak
runs cover one/eight threads and direct/required leases: 64 CPU profiles.
Padding canaries, invalid/duplicate routes, partial gate admissions, lease
memory caps and final drain are checked. Additional GPU geometries exercise
one/eight/128 tokens and one/two/six input slots. Logical outputs match the
scalar arithmetic reference byte-for-byte for the tested cases.

The 36 graph runs cover four format pairs on three GPUs in native, all-API
memcheck and explicitly disabled-graph modes. Each checks 54 precision,
input-pointer/routing-data and repeated-replay cases, three paired-projection
replays, and 19 unsupported metadata/split-parent cases. All four combinations
of default/explicit fused and down-projection precision are exercised. Explicit
outputs match scalar arithmetic; returning to default reproduces the initial
default result byte-for-byte. An external observer verifies actual graph
capture/launch and successful pointer updates; disabled runs make no graph
calls. Final GPU memory checks require zero API errors and zero leaked bytes.
Eleven focused existing CTests also pass.

## Bugs reproduced and repaired

Legacy fusion originally consumed an explicitly precise following projection,
bypassing its dispatcher. The mixed-policy regression fails before the guards
with `down explicit precision bypassed`. Both legacy gate/up/down fusion and
paired indexed projections now leave explicit next nodes to their own dispatch.

Legacy capture restrictions disabled the new batched expert kernels, despite
their device-only routing and allocation-free launches. The explicit kernels
now qualify capture directly, while legacy restrictions remain for legacy ops.

Switching arithmetic policy changes captured kernel topology even at unchanged
tensor addresses/layouts. The first candidate passes arithmetic but fails
unsuppressed memcheck with six error-910 reports from two rejected updates.
Policy changes now create a fresh executable; pointer-only updates retain the
existing successful update path. No sanitizer errors are suppressed.

## Remaining scope

This qualifies indexed/fused component execution for these cases, not a proof
of CPU/GPU bit identity for every possible floating-point input. It does not
enable the policy in the model graph or establish whole-model logits/routes,
cache-remapping interoperability, upstream parity, production readiness or an
end-to-end speed improvement. Model integration and placement/cache validation
are next. The full loader/lifecycle/resource/long-context/cache/speculation,
matched Strata, launcher and publication roadmap remains open.
