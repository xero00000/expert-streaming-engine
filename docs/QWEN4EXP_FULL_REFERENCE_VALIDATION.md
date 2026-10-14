# Streaming full-model upstream reference and router precision

Evidence: `/home/xero/Desktop/test/results/ese-qwen-upstream-full-20261011`.
Full upstream model parity remains open. This stage adds the missing reference
runner and repairs the first demonstrated router precision discrepancy.

## Reference scope

`tools/qwen4exp_full_reference.py` executes the pinned Transformers decoder,
attention, sparse MoE, PLE and recurrent forward methods. The source must hash
to `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`.
Its configuration uses the existing hash-verified model configuration.
The reference reconstructs original quantized GGUF weights to F32, materializes
one layer at a time, decodes only selected experts, reads only requested PLE
rows, and projects the output vocabulary in bounded row chunks.

The coder artifact retains 256 experts instead of the source model's 512;
the reference explicitly adopts that artifact geometry. It reverses the
previously established recurrent value-head conversion permutation and maps
GGUF effective RMSNorm gamma to the upstream parameter convention. PLE hash
multipliers, table sizes and offsets must match the upstream implementation.
The split indexer projections are joined in the upstream query/key order.
No weight files are changed. This tests the quantized/pruned artifact under
the pinned architecture, rather than claiming equality to the unquantized,
unpruned checkpoint or its quality.

The first-token reference executes all 48 layers and all 248,320 logits,
using token ID 58 from the native probe's newly saved input-token artifact.
A follow-up emits 110 convolution, recurrent, attention and indexer state
artifacts. Its logits are byte-identical to the first reference generation.
State artifacts have not yet been compared to native serialized state.
The initial reference's peak RSS is 6,566,180 KiB, including mapped pages.
This establishes feasibility on the current host; reference timings are
correctness-workload timings, not an engine throughput benchmark.

## First discrepancy and scoped correction

At layer zero, native/upstream attention input, attention output and FFN input
maximum errors are respectively `9.54e-7`, `4.17e-7` and `1.43e-6`.
The original BF16 router rounds scores to BF16 precision: experts 44 and 193
both have score `-4.6875`. On the same captured F32 activation the independent
F32 calculation ranks expert 193 above 44. This changes the tenth expert.

The existing experimental `ESE_QWEN4EXP_EXPERT_F32=1` policy now requests
explicit precision for the router as well as expert projections. Across all
48 layers, captured new router outputs are byte-exact to independently
accumulated F64 original-BF16-weight dots rounded to F32. All 48 selected
router-position ID lists match that same-input oracle. The old runtime
retains its reproducible failing cutoff control. Other architectures and
default policy behavior remain unchanged by the new request.

Against the complete first-token upstream pass, matching expert sets increase
from 42/48 to 48/48; exact ordered ID lists increase from 18/48 to 45/48.
Complete logit maximum error decreases from `0.188531` to `0.158851`.
This is still a full-model discrepancy, not a parity pass.

The next isolated boundary is the shared expert. On the exact same captured
layer-zero FFN input, routed output maximum error is `4.47e-8`, shared-expert
gate error `5.96e-8`, but ungated shared-expert output error `0.00293624`.
The complete FFN output error is `0.00121531`. The shared quantized projection
path still needs explicit activation-preserving arithmetic and qualification.

## Validation status

The reference reconstruction/cache/source-pin tests and existing PLE/recurrent
Python regressions pass: 25 tests. Four focused native CTests pass, including
normal loader rejection and explicit projections. CPU-expert and hybrid
native controls produce byte-identical nine-token logits and routes under
the new router policy. The old native trace reproduces the preceding frozen
runtime's first nine teacher steps exactly.

The new hybrid all-API memory check passes with zero errors and zero leaked
bytes; its nine-token teachers and routes match the native controls exactly.
RAM/staging caps and final lease/load/reservation drain pass. Matched old/new
default-policy hybrid controls produce byte-identical teachers and routes.
The initial default control fails an inappropriate explicit-precision harness
assertion; its correction still uses settings mismatched to the historical
baseline, so that comparison is rejected. Both attempts are preserved. Final
matched controls use the same physical Ampere pair, budgets and tracing policy.
`qualification-final.json` records accepted checks and retained control failures.
These controls do not qualify full-model upstream parity, long-context or
serving/resource stress, broader kernels, or a matched Strata speed ratio.
