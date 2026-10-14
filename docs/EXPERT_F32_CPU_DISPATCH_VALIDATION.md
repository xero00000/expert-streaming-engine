# Shared expert arithmetic: six formats and CPU dispatch

Explicit `GGML_PREC_F32` now selects original-weight/F32-input expert arithmetic
on CPU `MUL_MAT_ID` and `MOE_FUSED_UP_GATE`. Fused precision occupies parameter
slot 2, leaving the unary and clipping slots intact. Default operations retain
their previous dispatch. CUDA currently refuses this explicit request until its
indexed path is connected; it cannot silently substitute quantized activations.

Evidence: `/home/xero/Desktop/test/results/ese-expert-f32-prepared-20261011`.

This document records the prepared-CPU stage. The subsequent CUDA connection,
merged/padded execution and graph-policy repairs are documented separately in
[`EXPERT_F32_CUDA_DISPATCH_VALIDATION.md`](EXPERT_F32_CUDA_DISPATCH_VALIDATION.md).

## Arithmetic coverage and cost

The loaded model's 144 expert tensors use IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
and Q2_0; all 48 up/gate pairs have matching types. Added CUDA scalar decoders
cover the four families absent from the first prototype. Original expert-10
payloads from layers 0, 1, 17 and 47 cover all six. The retained layer-20
activation is a common arithmetic input, not a captured activation for those
other layers. Their payload hashes and shapes are retained in `case-inventory.json`.

All 24 native and unsuppressed all-API GPU memory/leak runs pass. Each run checks
all 4,915,200 decoded values and four captured/alternating/zero/subnormal inputs,
with CPU/GPU outputs byte-exact and independent long-double checks passing the
unchanged `<2e-5` bound. All reported comparison errors are zero. These are
contiguous single-expert CUDA prototypes, not indexed GPU backend qualification.

Preparing F64 activations once per projection and reusing them across rows
approximately halves the scalar prototype's captured-input CPU cost from
19 ms to 9.5 ms on this machine. The prepared build also enables AVX2/FMA,
so this comparison does not isolate input reuse from compiler flags. All four
input cases retain byte-exact scalar outputs. This is a component experiment;
no engine throughput improvement is established.

## CPU execution, memory and leases

The graph plan accounts for eight bytes per prepared activation, selected-expert
flags and duplicate-safe route jobs. Each worker retains the 1.25 KiB block
decoder/reduction temporary. Shared groups hold at most four experts; optional
partial up/gate admissions release their temporary up lease safely. All weights
remain leased until every output-row chunk completes.

Four format pairs pass native and strict CPU ASan/UBSan/leak runs. Each run covers
one/eight threads and direct/required leases: 32 profiles, 30,720 final output
values per profile, byte-exact against the scalar arithmetic reference. Fixtures
rotate original rows to distinguish four synthetic experts, poison leased
backing tensors and padding, use different up/gate pitches and padded input/ID
rows, and exercise duplicate, negative and out-of-range IDs. Cache pressure
rejects a gate admission after its up lease succeeds. Final leases drain and
peak lease bytes stay within the fixture budget. Nine existing normal CTests pass.

The initial strict run diagnosed five typed decoders being cast to the generic
`to_float` callback signature. Correctly typed adapters remove that function-type
UB without changing decoder math; Q2_0 already has the generic signature. The
failed run is retained. An initial sanitizer launch also picked an older conda
libstdc++ via LD_LIBRARY_PATH; the corrected sanitizer environment uses its build
runtime, without suppressing checks. Both failures remain separate from accepted
results in `qualification-audit.json`.

## Remaining qualification

CPU support currently covers these six types, F32 activations and unclipped,
unbiased SiLU fusion; unsupported requests fail support queries. CUDA indexed
dispatch, its layout/device checks and the model graph opt-in remain unfinished.
Default decoder identity controls, further metadata/failure tests, merged-layout
tests and whole-model upstream/placement validation still need completion.
The earlier hybrid numerical-baseline failure is not waived. Wider formats,
performance tuning and all other release/publication roadmap gates remain open.
