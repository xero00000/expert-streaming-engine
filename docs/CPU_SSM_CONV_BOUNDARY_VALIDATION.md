# CPU convolution graph boundaries

The CPU four-tap convolution shortcut now checks that both following nodes
belong to the current graph before considering a fused view and SiLU. The
previous condition compared the convolution's index with `n_nodes + 2`, so
it could inspect nodes beyond the current graph or graph view. The condition
was present in the checked-in base revision.

Terminal convolution graphs could crash while inspecting an uninitialized
node pointer. A short view of a larger graph could instead recognize and
execute a later SiLU outside the requested view, leaving the requested raw
convolution output unwritten. The corrected bound preserves the existing
eligible full-graph fusion and uses the ordinary convolution output when the
following pair is outside the graph.

Evidence: `/home/xero/Desktop/test/results/ese-cpu-ssm-conv-boundary-20261012`.

- Two old-library terminal controls exit with SIGSEGV. Two old-library prefix
  controls fail because the raw output still contains its initialization
  sentinel. Full-graph fusion and three-tap fallback controls pass against the
  same frozen library. A separate strict CPU UBSan control reports the invalid
  node-pointer access before the correction.
- `test-ssm-conv-boundary` passes 360 cases in both Release and strict CPU
  ASan/UBSan/leak builds: threads 1/2/8, channels 16/17/32, token counts
  1/4/33/64, three- and four-tap kernels, terminal convolution/view graphs,
  complete fused graphs and two prefix boundaries. Outputs meet an independent
  scalar convolution/SiLU oracle at `2e-5*(1+abs(reference))`; updated histories
  and input bytes match exactly. Prefix cases also require the later SiLU
  buffer to remain untouched until its graph portion executes.
- Sixteen focused CTests pass in each build, including recurrent operators,
  graph-boundary, normalization, transpose-copy, state ownership, server
  recovery and the normal Qwen loader-gate check.
- The original-model untraced CPU 33-token control completes successfully.
  All 8,194,560 teacher logits and per-token expert routes match the frozen
  pre-correction control byte-for-byte. Model-shard identities are unchanged.

The actual-model numerical gate remains failed: 64 entries exceed the fixed
bound against the retained full-prefix reference and 59 against the cached
reference. All 1,584 ordered expert selections and 33 top predictions match
each reference. This correction does not change weight files, reference
outputs, tolerance or GPU kernels, and does not establish full-model memory
or throughput qualification. Normal Qwen loading remains gated.

An initial sanitizer control using a preloaded old library also broke the
symbolizer subprocess and ended with SIGPIPE. That receipt is retained and
excluded from qualification; the ordinary unpatched sanitizer executable
provided the clean, symbolized negative control.
