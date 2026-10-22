# Native regression audit — 2026-10-13

The complete registered target sets were rebuilt after the common library and
native context changes. This audit found and repaired chat-template integer
undefined behavior and corrected an inaccurate Q2_0 precision test. The full
regression gate remains incomplete because two initialized DeepSeek reference
tests still fail.

Evidence: `/home/xero/Desktop/test/results/ese-release-regression-20261013`.
The subsequent [HC normalization qualification](DEEPSEEK_HC_NORMALIZATION_VALIDATION.md)
corrects epsilon placement in CPU/CUDA and passes the token reference. Its
complete suites pass 95/96 CUDA and 91/92 CPU Release and CPU sanitizer tests;
only the initialized block reference remains failing. The audit below records
the earlier results before that correction.

## Builds and suites

- Full CUDA Release and CPU Debug ASan/UBSan builds pass, including previously
  missing tests and examples. Final builds rebuild dependents after the fixes.
  CPU sanitizer runs enable leak detection and halt on sanitizer errors.
- Final CUDA CTest passes 94 of 96 tests; CPU sanitizer CTest passes 90 of 92.
  Only `test-deepseek-v41-token` and `test-deepseek-v41-block` fail in either
  final suite. The initial CUDA run also failed Q2_0; the initial sanitizer run
  also reported an undefined float-to-integer conversion in Jinja arithmetic.
- Launcher/CI unittest discovery passes 87 tests. Final pytest passes 153 tests
  and 122 subtests, including the unittest cases; these are overlapping suites,
  not 240 distinct tests. All 15 Studio utility tests pass.
- Standalone Turbo KV reference, Python compilation, individual shell syntax
  checks, launcher help/doctor and both native help-policy checks pass.
  These are local checks; Windows packaging and complete release readiness
  are not established by them.

## Chat-template numeric safety

Integer arithmetic now uses checked signed 64-bit operations instead of
rounding through `double`. Modulo by zero and overflowing integer results
produce template errors; the minimum integer modulo minus one returns zero
without invoking undefined signed arithmetic. Integer construction checks the
upper double boundary before a narrowing cast. Large finite float values can
be constructed safely, and out-of-range integer conversion is rejected.

Permanent cases cover both integer endpoints, exact values above the double
integer range, fitting boundary arithmetic, overflow in each sign combination,
zero modulo and large finite float conversions. All three focused Jinja,
Python-Jinja and Q2_0 CTests pass in both builds. Existing template comparisons
with Python remain green. This changes template arithmetic, not model math.

## Q2_0 precision test

The old test compared a full F32 activation oracle with default CUDA dispatch,
which may quantize activations or produce F16 output. Its claim that a global
activation maximum made every Q8_1 block exact was incorrect.

The test now explicitly requests F32 for dense and routed operations. The
original activations, packed bytes, scalar oracle and 0.02 tolerance are
unchanged. A private control passed before integration. CPU and all three GPUs
(RTX 3060 Ti, RTX 2080 SUPER and RTX 3080) pass; an Ampere-pair CUDA memcheck
reports zero errors and leaked bytes. This is a test-contract correction; no
Q2_0 runtime kernel or default precision behavior changed.

## Remaining initialized DeepSeek failures

At the time of this audit, the native block test first differs at token 4, layer 1, element 70:
`-1.1640625` instead of `-1.171875`. A pinned-source replay with the saved
weights and inputs reproduces all 80 reference output blocks exactly.
Retaining every native graph intermediate reproduces the same failure.
Private controls changing Engram arithmetic or HC collapse/projection order
also reproduce it and are not integrated.

At that first failure, attention and FFN outputs are exact. The final residual
expansion has small HC combination-coefficient differences before BF16 rounding.
Native Sinkhorn adds epsilon before summation; the source adds it afterward.
A private source-order control moves the block failure to token 7, layer 7,
element 82, while its 10-token whole-model logits pass the unchanged tolerance.
The block check still fails. That kernel candidate remains private; the saved
references and all tolerances remain unchanged.

This is an initialized small-model diagnostic, not trained DeepSeek quality
evidence or closure of the original Qwen numerical gate. Complete Qwen parity,
64K execution, concurrent serving, matched benchmarks, platform packaging and
release promotion remain unqualified. Normal Qwen loading stays gated.
