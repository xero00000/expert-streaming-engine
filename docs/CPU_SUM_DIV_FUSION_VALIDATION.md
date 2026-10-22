# CPU sum-and-divide fusion correctness

The CPU shortcut for `SUM_ROWS → DIV` now follows the ordinary operators.
Previously it always reduced dimension zero, treated every higher-dimensional
row as flat, used a different accumulator and reciprocal multiplication,
returned zeros for nonpositive sums, and left the sum output unwritten.
The shortcut was present in the checked-in base revision.

The corrected path reuses the ordinary row-sum implementation, rounds division
as division, addresses each dimension using its stride, and writes the sum.
It only fuses dimension-zero F32 reductions; other axes use the existing
operators. Zero denominators retain ordinary division behavior. This changes
CPU results where the shortcut previously changed the computation, including
small rounding differences in positive expert routing weights.

Evidence: `/home/xero/Desktop/test/results/ese-cpu-sumrows-div-20261012`.

- Six controls against the old frozen library demonstrate rounding, negative
  sums, the wrong reduction axis, strided-plane reads and missing sum output.
  With uniform strided input, the first wrong value is index 30: negative zero
  instead of 0.1. The corrected regression passes 240 complete-versus-staged
  graph cases across threads 1/2/8, widths 10/32, all four reduction axes,
  packed/strided layouts and five input patterns. Outputs, sums, inputs and
  storage gaps match; NaN comparisons require NaN on both sides.
- Thirteen related Release regressions and thirteen strict CPU ASan/UBSan/leak
  regressions pass. The production Qwen loader-gate test passes separately in
  both builds. Component checks do not establish full-model memory coverage.
- Actual original-model CPU traced and untraced runs both complete 33 tokens.
  Their 8,194,560 teacher logits and per-token routes are byte-identical after
  the correction. Before it, tracing changed 7,747,279 logit entries, with a
  maximum absolute difference of 1.52588e-5. The corrected traced control also
  matches the earlier ordinary-operator traced control exactly.
- The current 2080 SUPER control matches the prior frozen teacher logits and
  routes byte-for-byte. GPU arithmetic is unchanged by this CPU correction.

Full-model parity remains failed. Both corrected CPU runs have 64 failing
entries against the retained full-prefix reference and 59 against the cached
reference, at the unchanged `2e-5*(1+abs(reference))` bound. Turing retains one
failing entry against each reference. All 1,584 ordered expert selections and
33 top predictions match each reference in every current model control.
Model files and reference outputs are unchanged; normal Qwen loading remains
gated. There is no throughput claim for this correction.
