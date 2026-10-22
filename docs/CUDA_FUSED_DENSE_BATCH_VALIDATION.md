# Parallel fused shared gate/up batches

The Qwen verifier's remaining prefix8/depth8 discrepancy was reproduced with
actual layer7 IQ4_XS and layer19 Q6_K gate/up weights, IQ4_NL down weights and
identical captured FFN inputs. Original batch8 gate/up outputs reproduce the
captured model values exactly. At layer19, 4,245 gate/up scalars differ from
independent single-token calls; one token's2,560 down-projection scalars differ.
This isolated the shared FFN boundary independently of attention and routing.

`ESE_CUDA_FUSED_BATCH_VECTOR=1` now maps contiguous, unbiased dense batches of
2..8 tokens to separate vectors in the second CUDA grid dimension. The weight
stride is zero, the quantized-input stride is one padded Q8_1 row and the output
stride is one token column. The existing fused single-vector reduction runs for
every token in parallel. Input quantization runs once across the token batch;
there is one fused gate/up launch, with no per-token host launch loop.

The first candidate let MMVQ's expert-channel heuristic reduce four warps to
one when there were multiple vectors. It fixed the sampled down outputs but
still changed gate/up bits, so it was rejected. Broadcast-weight fused vectors
now retain the single-token warp configuration. Nsight confirms the Q6_K
one-column/four-warp kernel with gridY1/2/4/8. Indexed expert launches keep their
existing selection rule. The first profile wrapper accidentally set the new
control to0; that negative run is retained and the corrected unique run sets1.

All actual-weight gate/up and down fixture values match their independent
single-token outputs at depths1/2/4/8 on all three GPUs. The generic regression
`tests/test-fused-dense-batch.cpp` covers216 cases per GPU: Q4_0/Q4_K/Q6_K/IQ3_S/
IQ4_NL/IQ4_XS, widths512/2560, output rows31/640, tokens1/2/3/4/5/8 and SiLU/
ReLU/GELU. Inputs include ordinary, zero, tiny and large values. Every result
must match scalar bits; edge/middle rows are also checked against a dequantized
double oracle with an explicit activation-quantization error bound. This is
not a claim of ideal F32 dot products. The old library fails the regression at
Q4_0 width512/rows31/tokens2/SiLU while its oracle bound still passes.

All216 cases pass native and unsuppressed all-API CUDA memcheck on the RTX3060Ti,
RTX2080SUPER and RTX3080; zero errors. Eleven focused normal CTests and the
normal server build pass. Both final GPU/CPU checkpoint runs pass21 cases and
preserve all8,194,560 original teacher logits exactly, with bounded expert RAM/
staging and clean load/lease/reservation drain.

With CPU experts in both paths, the dense-vector projection diagnostic and the
qualified attention control, real-model prefixes8/32 now pass depths1/2/4/8:
every target logit, MoE route, complete saved state and continuation matches.
Both ordinary teacher streams are preserved. This closes these controlled
batch arithmetic matrices; it does not qualify the production hybrid path,
accepted-prefix recovery, a sampler or a speculative speedup. Padded inputs,
biased/other gated operations, larger batches, custom quantizations and other
devices require separate qualification; the control remains opt-in.

Warm actual eight-token gate/up timing: seven alternating100-iteration repeats,
synchronized graph dispatch plus GPU work, excluding down projection. The
corrected parallel path is faster than separate single-token calls but slower
than the unqualified old batch. CUDA device0=3060Ti,1=2080SUPER,2=3080.

| Device | Layer | Correct batch us | Eight serial us | Old batch us (unqualified) |
| --- | --- | ---: | ---: | ---: |
| 0 | 7 | 41.74 | 61.51 | 31.93 |
| 0 | 19 | 57.35 | 77.34 | 37.38 |
| 1 | 7 | 40.37 | 64.80 | 36.45 |
| 1 | 19 | 49.98 | 74.30 | 36.60 |
| 2 | 7 | 26.38 | 48.53 | 24.14 |
| 2 | 19 | 34.79 | 58.10 | 24.61 |

These small operator measurements use warm data and uncontrolled clocks/OS
scheduling. They do not establish whole-model or accepted-token throughput.

Reproduce from the source directory:

```sh
cmake --build build_cuda_release --target test-fused-dense-batch -j 8
CUDA_DEVICE_ORDER=PCI_BUS_ID ESE_CUDA_FUSED_BATCH_VECTOR=1 build_cuda_release/bin/test-fused-dense-batch 0
CUDA_DEVICE_ORDER=PCI_BUS_ID ESE_CUDA_FUSED_BATCH_VECTOR=1 /usr/local/cuda/bin/compute-sanitizer --tool memcheck --report-api-errors all --error-exitcode 97 build_cuda_release/bin/test-fused-dense-batch 0
```

Repeat devices1/2. CPU-only builds report a CUDA-specific skip.
Final GGML SHA256:
`c13e871cf10ffeafb963050132238fb1d3cc0fec75ca39d1adf0e4cab7ae63b9`.
Full source/evidence/controls, actual-weight fixtures, profile, native/memcheck,
checkpoint/verifier runs and timings are under
`/home/xero/Desktop/test/results/ese-shared-fused-batch-20261010`.
