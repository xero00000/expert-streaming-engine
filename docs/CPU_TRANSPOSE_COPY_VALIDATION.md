# CPU transpose copies into strided destinations — 2026-10-12

A CPU F32 transpose-copy fast path wrote destination planes as tightly packed
rows without checking the destination's actual row stride. Non-Flash-Attention
value-cache views have gaps between rows. Copying into those views misplaced
values and overwrote padding or other cached data.

The fast path now requires matching source/destination shapes and packed
destination rows. Strided or reshaped destinations use the existing general
copy, which respects their strides. The packed-plane optimization remains
available, including planes separated by outer-dimension gaps. Weight bytes,
projection arithmetic and reference tolerances are unchanged.

Evidence directory:
`/home/xero/Desktop/test/results/ese-cpu-split-diagnostic-20261012`.

## Operator and model checks

`test-copy-transpose-layout` checks 640 CPU copies. It varies thread counts
1/2/4/8, token widths 1/4/8/12/33, channels 8/32, plane counts 1/4 and batch
counts 1/2, plus packed/strided rows and matching/reshaped destinations.
It checks every value in destination storage, including offsets and gaps that
must retain their sentinel bytes. All cases pass in Release and CPU ASAN/UBSAN
with leak checking. Against the frozen old core, the same executable fails at
the first overwritten gap, with terminal exit 1.

A generated three-layer F32 Llama model also exposed the bug through the public
decoder: a single 12-token prompt and an 8+4 split differed by up to 0.187703 in
their final logits. The discrepancy reproduced without server code, with and
without an idle shared sequence or an intermediate output request.

Traces preserve the corresponding untraced output bytes. Before the repair,
layer0 embeddings, normalized inputs, Q/K/V projections and attention
probabilities matched between prompt shapes. The copied V-cache values did not
match their source projection: 372/384 entries in the full prompt, 248/256 in
the first split chunk, and 124/128 in the second chunk. Attention multiplication
matched a separate F64 calculation using those damaged cache values; the copy
layout was the identified error.

After the repair, all of those copied/read V-cache values match their source.
All compared traced tensors and the final logits match byte-for-byte between
full and split prompts. The persistent generated-model recovery test includes
an output-free first chunk and requires this exact split comparison, with and
without a shared system prefix.

Twelve focused regression tests pass in both Release and CPU ASAN/UBSAN/leak
builds. They include copy layout, real graph-abort recovery, response destruction,
decode status/capacity distinction, sequence/checkpoint state copying, hybrid
cache retiering, output ownership, attention layout and prefill backoff.

Seven HTTP controls also pass with the corrected core. Four use real CPU graph
abort callbacks; simulated -2 and positive 2 verify fatal-error handling. A
simulated capacity-1 result now retries the smaller batch and matches its clean
teacher exactly. Its idle slot's saved prefix remains byte-identical and its
cached continuation matches clean recomputation.

## Scope

This repair qualifies the tested CPU F32 copy layouts and synthetic Llama
integration. It does not change CUDA copy kernels or establish numerical
equivalence for every model, cache format or batch shape. No end-to-end speed
claim is made. The original Qwen 33-token numerical gate and normal architecture
gate remain closed.

```sh
cmake --build build_cuda_release --target test-copy-transpose-layout test-server-decode-recovery --parallel 4
ctest --test-dir build_cuda_release --output-on-failure -R '^(test-copy-transpose-layout|test-server-decode-recovery)$'
```
