# CUDA flash-attention precision and batch arithmetic

The real Qwen verifier capture uses DEFAULT precision, F32 queries and F16
keys/values. Nsight records different half-precision MMA tiles for one and four
queries. Replaying the captured operands reproduces the batch output exactly;
the first differing token has maximum scalar/batch error 4.76837158e-7.
This DEFAULT arithmetic boundary remains open.

A separate backend defect ignored explicit F32 precision when there was more
than one query. The dispatch now selects the F32 vector kernel for supported
regular F16 key/value inputs at every query count. Unsupported regular F32
head sizes are rejected by backend capability reporting rather than advertised
as a supported half-precision path. Indexed DSA and quantized KV require
separate qualification.

The precision regression also found F16 softcap specializations advertised as
supported while their device bodies were stubs. F16 vector softcap variants are
now enabled, and symmetric head64 DEFAULT softcap uses this implemented path.
Qwen's DEFAULT head256/no-softcap dispatch remains unchanged.

`tests/test-attention-precision.cpp` checks 219 cases per GPU against an
independent double CPU oracle and independent single-query GPU evaluations.
It covers head pairs 64/64, 128/128, 256/256, 512/512, 192/128 and 576/512;
256/512/1024 keys; 1/2/4/8/17/33 queries; softcap 0/2; grouped-query heads;
strided K/V caches; padded causal masks and masked holes. Three unsupported
F32 head sizes (80/96/112) are negative capability checks. This is a precision
accuracy test, not a universal promise of bit-identical reductions.

All 219 cases pass on the RTX 3060 Ti, RTX 2080 SUPER and RTX 3080. Unsuppressed
CUDA memcheck with `--report-api-errors all` reports zero errors on each.
Nine focused normal CTests pass. The normal server builds. Real-artifact
GPU/CPU checkpoint runs each pass 21 cases and preserve all 8,194,560 original
teacher logits bit-for-bit with the final library. RAM/staging caps and clean
cache drain are verified. These are short lifecycle checks, not full serving
or long-context qualification.

The actual four-query operand fixture also passes with explicit F32 on all
three devices: all four outputs match their independent single-query outputs
bit-for-bit. On CUDA0 its CPU-oracle maximum error is at most 6.03e-7, versus
approximately 0.001625 with the original DEFAULT half kernel. DEFAULT fixture
outputs remain bit-identical to their frozen control on every device and retain
the measured scalar/batch discrepancy. Changing the Qwen graph's requested
precision requires its own whole-model quality and performance qualification.

Reproduction from the ESE source directory:

```sh
cmake --build build_cuda_release --target test-attention-precision -j 8
CUDA_DEVICE_ORDER=PCI_BUS_ID build_cuda_release/bin/test-attention-precision 0
CUDA_DEVICE_ORDER=PCI_BUS_ID /usr/local/cuda/bin/compute-sanitizer --tool memcheck --report-api-errors all --error-exitcode 97 build_cuda_release/bin/test-attention-precision 0
```

Repeat for devices 1 and 2. The test retains a primary CUDA driver context.
CPU-only builds report a CUDA-specific skip. Larger-batch F32 performance and
the full verifier/sampler remain open; this repair does not establish a speedup.

Frozen GGML SHA256:
`5fcf2667c9ab2b5556a26851316823ad9a6f526bf37d2cff06ce3341e96aee15`.
Captured raw operands, source-before snapshots, controls, native/memcheck logs,
replay runners and the Nsight trace are under
`/home/xero/Desktop/test/results/ese-attention-batch-arithmetic-20261010`.
