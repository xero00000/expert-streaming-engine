# Qwen4Exp selection-limited sparse-attention validation

The experimental QSA pipeline now dispatches its final F32 logarithm on CUDA.
The existing `ggml_cuda_op_log` kernel was implemented but missing from both
the execution switch and the backend support query. The support query accepts
contiguous F32 input/output only, matching the CPU operator's type contract.
Other layouts retain their existing fallback. No architecture gate changes.

Evidence is in
`/home/xero/Desktop/test/results/ese-qsa-selection-20261011`.
The original candidate and failures are preserved; `v2` contains the repair,
frozen sources/libraries, commands, environments and initial component results.
`final-components` preserves the expanded test with the actual four-head geometry,
using those same immutable repaired libraries. The final `v3` generation adds
the logarithm's full F32 edge-case correction and repeats component/model checks.

## Independent selection checks

`tests/test-qwen4exp-qsa-selection.cpp` exercises the actual sequence-owned
attention layout/input filler and the QSA graph primitives: F16 raw-key gather,
four-row pooling, RMS normalization, partial NEOX rotation, scoring, top-two
block selection and union with the causal partial tail.

The 144 cases combine widths 8/128, head counts 2/4, sequence lengths
31/32/33/255/256/257/511/512/513, one/four queries and zero/nonzero position
offsets. Nonzero-offset inputs include a five-position gap. Each case runs with
contiguous and shuffled physical placement, plus unrelated-sequence rows filled
with adversarial finite values. Padding crosses 256/512-row boundaries.

An independent F64 oracle computes scores from the original logical F16 rows;
the allowed score error is `2e-4 * (1 + abs(expected))`. Independent ranking
checks exact selected block IDs, with a required non-tied selection boundary.
Independent membership checks require exact 0/-infinity masks. Both physical
placements must produce byte-identical scores, selections and masks. This is
an independent mathematical oracle, not a new pinned-upstream full-model trace.

All 144 cases pass on CPU, RTX 3060 Ti, RTX 2080 SUPER and RTX 3080. Each GPU also
passes unsuppressed compute-sanitizer memcheck with `--report-api-errors all`:
exit 0, zero errors. Exact final score-error maxima are in
`v3/component-audit.json` (maximum `2.09950878e-5`).
All numerical graph nodes are explicitly placed on the tested backend; one I32
`REPEAT` remains on CPU per graph and is reported, not hidden. This is not a
claim that the entire graph is GPU-only.

The expanded executable with frozen original libraries fails because contiguous
F32 `LOG` is unsupported. The final executable with the first dispatch repair
fails at input `0x1p-149`: the existing fast-math logarithm returned -infinity
instead of a finite result. A float-to-double promotion also flushes that input.
The final kernel reconstructs F32 subnormals from their integer significands
before the F64 logarithm, and recognizes zero/one directly by their bits.
The repaired version checks finite logarithms, positive/negative subnormals,
minimum normal/maximum finite values, neighbors of one, zero and signed zero,
infinity, negative inputs, NaNs and changed-input replay. The host oracle also
reconstructs subnormals, avoiding dependence on host flush-to-zero state.
Unsupported strided/F16 logarithms are rejected by the CUDA support query.

Ten focused normal CTests pass. The 144-case CPU pipeline also passes strict
ASan/UBSan/leak checking. The initial CPU harness incorrectly read intermediate
storage after allocator reuse; retaining scores/top-blocks/masks as outputs
corrected the harness. That initial failure is preserved and is not attributed
to model math.

## Model controls and remaining scope

Both native and all-API real-artifact sequence-bank focus runs pass both
continuation/logit/route checks, including complete state recovery, independent
discard and logical QSA IDs. Their seven captured teacher/baseline/reference/
actual streams are byte-identical to the preceding qualified graph-update
generation. The focus covers target sequence 0, accepted length 4, microbatch 2;
the preceding broader bank matrices are separate historical evidence.
The model uses that generation's diagnostic-enabled llama library with the
repaired GGML library; public and private context layouts are unchanged.

The first dispatch repair's four model controls are terminal/pass, with exact
previous-generation streams and bounded RAM/staging/drain. Its all-API memcheck
reports zero errors. Those historical results are in `v2/model-audit.json`.
The final subnormal correction repeats that queue in `v3/run-model-controls.py`:
all four runs are terminal exit 0 and audited in `v3/model-audit.json`. The
full-model all-API memcheck reports zero errors. Each ordinary CPU/GPU teacher
passes 18 JSON lifecycle checks and three checkpoint controls, with 8,194,560
F32 teacher logits and baseline logits/routes byte-identical to the preceding
graph-update generation. RAM resident/staging peaks obey their separate caps;
leases, loads, reservations, staging and failures drain to zero in every run.
No throughput gain is measured here.

This does not complete full-model upstream/production-hybrid parity, general
position shifting, arbitrary interleaved serving, tensor-split attention,
long-context stress, memory budgeting or MTP/adaptive-depth qualification.
Production promotion, launcher integration and matched Strata benchmarks
remain separate roadmap gates.
