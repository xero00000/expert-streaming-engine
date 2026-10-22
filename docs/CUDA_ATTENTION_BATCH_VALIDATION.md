# Short-batch CUDA attention with scalar-equivalent arithmetic

`ESE_CUDA_ATTENTION_STABLE_BATCH=1` is an opt-in control for the existing
F16 MMA attention path with at most 32 queries. It selects the scalar decode
query tile and gives each query tile its own scalar-equivalent Stream-K
partition in the second CUDA grid dimension. Fixup metadata and outputs are
independent per query tile. Queries run in parallel in one attention launch
and one fixup launch; there is no host loop launching attention per token.

Matching only the query tile repaired the actual four-query Qwen fixture but
failed longer-cache scalar/batch comparisons. Matching the block partition as
well passes those comparisons. The rejected tile-only control and immutable
library/source are retained in the evidence directory; its temporary source
knob was removed from the final implementation.

The audit also repaired three ordinary-path defects: value rows and heads used
key strides; MMA support reporting compared K with itself instead of V; and
the Stream-K workspace passed a byte count to a float2 element allocator.
K/V row and head strides are now independent, shape checking uses V, and
workspace allocation uses the actual float2 element count. Same-layout
single-token arithmetic is preserved. The previous different-head-stride
fixture fails the CPU oracle (max error about 0.0813); the repaired cases pass.

`tests/test-attention-batch.cpp` covers 54 cases per GPU: head widths
64/80/96/112/128/192/256, 256/512/1024/4096 keys, query counts up to31,
padded/permuted Q, independently padded K/V rows and capacities, causal masks
with holes, and grouped-head ratios2/4/12/24. It requires bit-identical
batch versus independent single-query results and checks an independent double
CPU formula with a DEFAULT half-kernel error bound (2e-3*(1+abs(expected))).
This does not assert F32 precision; that contract has its separate regression.

All 54 native cases and unsuppressed all-API CUDA memcheck pass on the RTX3060Ti,
RTX2080SUPER and RTX3080. Ten focused normal CTests pass; the normal server
builds. With the final library and the frozen gated LLAMA library, default-path
GPU/CPU checkpoint runs each pass 21 cases, preserve all8,194,560 original
teacher logits bit-for-bit, stay within expert RAM/staging caps and drain
active loads/leases/reservations. These are short lifecycle/resource checks,
not whole-model upstream parity or long-context/concurrent serving qualification.

Actual Qwen controlled verifier runs use CPU experts in both paths and the
existing dense-vector projection diagnostic. At prefix32, depths1/2/4/8 now
match every logit, MoE route, complete saved state and next-token continuation.
At prefix8, depths1/2/4 pass; depth8 remains unqualified. An all-layer bounded
trace preserves the same teacher and verifier results. Its attention and FFN
inputs are exact through layer19; tiny shared gate/up differences at earlier
layers are rounded away by their down projections. Layer19 is the first
observed differing FFN output. Actual IQ4_XS/Q6_K shared-expert weights for
layers7/19 are saved as bounded immutable fixtures for the next investigation.
This does not qualify a production verifier or sampler.

A warm actual four-query/256-key operand fixture measures synchronized graph
dispatch plus GPU work, seven alternating 100-iteration repeats. Median
corrected batch versus four separate queries: CUDA0 31.02/76.64us,
CUDA1 49.48/159.93us, CUDA2 28.96/73.70us. The unqualified old batch has
37.47/66.94/35.10us medians on the same respective devices. These are small
operator measurements with warm data and uncontrolled clocks/OS scheduling;
no whole-model speedup or accepted-token rate is established.

Reproduce from the source directory:

```sh
cmake --build build_cuda_release --target test-attention-batch -j 8
CUDA_DEVICE_ORDER=PCI_BUS_ID ESE_CUDA_ATTENTION_STABLE_BATCH=1 build_cuda_release/bin/test-attention-batch 0
CUDA_DEVICE_ORDER=PCI_BUS_ID ESE_CUDA_ATTENTION_STABLE_BATCH=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --report-api-errors all --error-exitcode 97 build_cuda_release/bin/test-attention-batch 0
```

Repeat devices1/2. CPU-only builds report a CUDA-specific skip. Qualification
covers the existing MMA family and listed shapes; separate vector/new-MMA,
SWA, indexed DSA, quantized KV, larger batches and other devices require their
own qualification. The control stays opt-in; launcher defaults are unchanged.

Final GGML SHA256:
`72ac637e015e9455e869d23d0aec71247f9aca7c252f8a67d89fad5374f40bf4`.
Evidence, raw operands/traces, rejected controls, immutable generations,
commands/logs/exits and the next-boundary report are under
`/home/xero/Desktop/test/results/ese-attention-canonical-20261010`.
