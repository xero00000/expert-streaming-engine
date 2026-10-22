# CUDA RMS normalization across layouts

The Qwen batch verifier exposed a general CUDA arithmetic inconsistency:
contiguous rows below width 1024 used a 256-thread RMS reduction, while strided
rows used one warp. Permuting the token/head axes therefore changed normalized
Q/K even when their input values were identical. Later quantization amplified
those small differences.

`ggml/src/ggml-cuda/norm.cu` now uses the contiguous reduction order for strided
F32 rows as well. Inactive shared-reduction lanes contribute defined zeros.
Contiguous rows and widths at least 1024 keep their previous arithmetic. Strided
small-row rounding intentionally changes to match the contiguous calculation;
the real-artifact ordinary decoding references below remain exact.

`tests/test-rms-layout.cpp` reproduces interleaved heads, padded token columns and
noncontiguous token/head views. It compares each batch row bit-for-bit with a
contiguous independent token and checks the result against a double CPU RMS
formula. Coverage is widths 32/64/128/255/256/511/1023/1024/2560, depths 1/2/4/8,
and ordinary/large/tiny/zero inputs: 144 cases per GPU. The frozen old library
fails at width64/depth2 with 191 differing scalars. The repaired kernel passes on
the RTX 3060 Ti, RTX 2080 SUPER and RTX 3080. Unsuppressed memcheck with
`--report-api-errors all` reports zero errors on each device.

Reproduction from the ESE source directory:

```sh
cmake --build build_cuda_release --target test-rms-layout -j 8
CUDA_DEVICE_ORDER=PCI_BUS_ID build_cuda_release/bin/test-rms-layout 0
CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/local/cuda/bin/compute-sanitizer --tool memcheck --report-api-errors all --error-exitcode 97 build_cuda_release/bin/test-rms-layout 0
```

Repeat with devices 1 and 2. The test activates a retained primary CUDA driver
context before backend discovery. A CPU build reports a CUDA-specific skip;
CPU formula/architecture checks are covered separately by the operator test.

Eight focused normal CTests pass with the production architecture opt-in unset.
The normal server builds. Operator, hybrid-state-copy and speculative-isolation
checks also pass strict ASan/UBSan/leak checking without suppressions.

Frozen GGML SHA256:
`336d80d1a2cbdd94e9b548efcdfea8f20990ca7130b7e538b0895dceca9e09ca`.
The real IQ1_M checkpoint runs using this library and traced gated LLAMA
`c75e5613ef8999c351ce129ccccdaa19549b7158535b6e53f3a0748321f3b9b9`
each pass 21 cases in GPU/CPU modes and reproduce all 8,194,560 original teacher
logits exactly. RAM/staging caps and cache drain are checked. The final optional
PLE diagnostic overload preserves the original result/function ABI. Final LLAMA
`d88a2e4d3cfb441961f825086d478d15c2d72a0636ab07c2bf57608aaeb0fb74`
also passes separate GPU/CPU 21-case reruns and both complete teacher comparisons.
Its verifier/regression evidence is recorded in the worklog.

With CPU experts in both paths and the existing dense-vector diagnostic control,
depth2 verification becomes exact after prefixes8/32; depth4 is also exact after
prefix32. Other depths still fail. This is partial verifier qualification, not a
speculative sampler, hybrid placement or speed claim. The next measured
divergence is in flash attention. A batched cuBLAS candidate for PLE projections
failed the arithmetic comparison on all three GPUs and was not adopted.

Full evidence and immutable generations are under
`/home/xero/Desktop/test/results/ese-ple-batch-arithmetic-20261010`.
