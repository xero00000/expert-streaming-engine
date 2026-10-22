# Additional prompt coverage and longer-prefix discrepancy

Evidence: `/home/xero/Desktop/test/results/ese-qwen-prompt-coverage-20261011`
and `/home/xero/Desktop/test/results/ese-qwen-reference33-20261011`.
The first directory's `qualification-final.json` records two passing cases.
The longer-prefix numerical comparison remains failed; no tolerance is changed.

The diagnostic lifecycle probe now accepts `ESE_LIFECYCLE_PROMPT`. With the
variable absent it reproduces the previous nine-token teachers, routes, input
IDs and prefix/continuation state byte-for-byte. This is a probe option;
production inference and loader policy are unchanged.

The code prompt asks for a Python function summing even numbers. The multilingual
prompt begins with a Chinese explanation request and continues in English.
Each case compares its actual first 17 token IDs with the pinned upstream model,
using original quantized weights, F32 K/V/indexer, non-Flash attention and the
existing explicit arithmetic policy on the selected Ampere pair.

Each case passes 4,221,440 logits, 816 ordered expert-route rows, 4,080 named
layer boundaries and 109 original saved-state comparisons. The fixed bound is
`2e-5 * (1 + abs(reference))`; maximum logit errors are 1.33514e-5 for code and
1.95503e-5 for multilingual text. RAM/staging caps and complete drain pass.
These are short teacher-forced prefixes, not generated answers or performance
benchmarks. They do not qualify selective long-context QSA, other devices/cache
types, sequence transforms, vision or MTP.

The 33-token diagnostic retains a small failure at token 13. All expert routes
match, but 14 logits exceed the fixed bound (maximum error 3.30210e-5).
Layer boundaries first fail at attention input 39; three original K/V/indexer
state cases in that layer also fail. On identical native input, the unchanged
upstream residual-mixing module agrees within 5.36442e-7 at layer 39 and
1.84774e-6 at layer 42. This rules out that local operation as the demonstrated
cause and points to accumulated earlier numerical variation.

A separate reference experiment supports cached decoder chunks to compare
single-token and full-prefix upstream execution without replacing the existing
goldens. It preserves the pinned forward methods and records the chunk size.
Its first attempt passed only current-position RoPE to QSA, which also requires
historical positions, and failed in the harness. The corrected cached33 experiment completes, and the defaultfull-prefix
control reproduces all545original trace files byte-for-byte. Native versus
cachedsource still fails20logit entries at tokens13/23. Sourcefull-prefix
versus cachedexecution itself exceeds the samebound in sixlogit entries.
The retained cached artifact processes decoder layers one token at a time but
runs its final output mixer over the complete 33-token batch. That execution
detail affects Float32 rounding. Neither comparison replaces the originalgolden
or changes the bound.
Sixteenexisting Pythonreference/trace tests pass. An independent stored-decay
round-trip study finds21Float32 discrepancies;20have no exact exponential
preimage within four adjacent Float32ULPs, so no speculative parameter repair
was applied. All96same-input attention/FFN module comparisons now pass the unchanged
bound. No failing local operator was found; the remaining end-to-end
discrepancy is retained as an open gate rather than patched by changing
a passing operator or loosening the comparison.
No longer-prefix or release parity is claimed by this document.

Focused retained-trace attribution now also checks the numerical replay runtime.
The locally installed PyTorch 2.10.0 reproduces the original layer-38 attention
and input-mixer reference outputs exactly in full-prefix and cached modes;
PyTorch 2.11.0 introduces small replay differences with the same pinned source
and weight payloads. Eighteen matched-runtime cases separate local arithmetic
from propagated input differences at the token-13 discrepancy. These are module
and counterfactual diagnostics, with no end-to-end repair or altered golden.
Evidence: `/home/xero/Desktop/test/results/ese-qwen-incoming-attribution-20261012`.
New reference qualification records include the Python executable/version,
PyTorch version/path/configuration digest, NumPy version, thread count and
Float32 matmul precision. The original records are retained unchanged;
their missing runtime fields are not backfilled from assumptions. Sixteen
existing reference tests pass with PyTorch 2.10.0.

The reference generator now keeps the final output mixer on the requested
decoder chunk boundaries and records that chunk size explicitly. Six runner
regressions cover whole-prefix, single-token, partial-tail and larger-than-input
chunks. The saved old runner fails the single-token regression; all 22 current
reference/trace tests pass. The initial new test fixture omitted its layer-type
metadata; that failed attempt is retained and excluded from qualification.

On the retained cached layer-47 activations, the old final mixer and head replay
all 8,194,560 original logits byte-for-byte. Applying consistent single-token
mixing changes logits by at most 2.86103e-6, with no reference-to-reference
fixed-bound failures. Native comparisons still fail: CPU 60, Ampere 21 and
Turing 1 logits against this additional artifact. Original references and their
existing failures remain unchanged. This corrects reference execution semantics
without repairing the engine's numerical discrepancy.

The same investigation checks all 97 original residual mixers, including the
output mixer and PLE-adjusted input. All 582 mixed-input and 576 injection
comparisons against identical captured native inputs pass the fixed bound:
49,167,360 and 194,641,920 values respectively. The original module replays
reproduce 194 retained cases byte-for-byte, honoring the retained artifact's
actual final-mixer batching. Six layer-36 sensitivity cases and twelve local
versus incoming-input splits retain explicit fixed sibling outputs; they are
counterfactual diagnostics rather than whole-model replacements.
Evidence: `/home/xero/Desktop/test/results/ese-qwen-layer36-attribution-20261012`.

The corrected complete 48-layer cached reference also finishes successfully.
All 543 earlier intermediate files, trace geometry and weight-access records
are byte-identical to the retained cached reference. Only the final mixer and
logits change, exactly matching the separate saved-layer-47 replay. All 1,584
ordered expert lists and 33 top-token choices match each native profile.
The additional artifact still has the 60/21/1 failed-logit counts above.

An additional single-RTX-3080 nine-token control passes all 2,234,880 logits
against the original short-prefix reference and matches all 432 ordered expert
lists. It uses the explicit F32 policy, F32 K/V and indexer, non-Flash attention,
eight CPU threads and a 64 MiB RAM expert cache. Selecting five or nine GPU
experts per route with a 512 MiB GPU expert cache preserves the CPU control's
captured logits and route IDs byte-for-byte. Each configuration also completes
three reset rounds with exact native replays. These are private capture and
placement diagnostics; they do not qualify generated throughput, equal-memory
performance or the failed 33-token numerical gate.
Evidence: `/home/xero/Desktop/test/results/ese-strata-native-parity-20261012`.

The same single-RTX-3080 control now also covers the retained 33-token input.
Nine GPU experts per route and the existing path that computes all experts on
the GPU both preserve all 8,194,560 captured CPU-control logits, all route IDs
and 66 reset-replay token outputs exactly. The all-GPU path uses a 512 MiB
expert cache; cache misses are loaded before GPU computation. This validates
placement consistency for this input, including the transition past token 31.
Each profile still has 30 failed logits against the original full reference
and 31 against each cached reference, with all 33 top-token choices and 1,584
ordered expert lists matching. The normal architecture gate remains closed.

Increasing the nine-GPU-expert profile's RAM cache from 64 MiB to 1 GiB also
preserves captured outputs. Across three reset rounds it records 54,859 hits
in 142,560 component acquisitions (38.5%), reducing logical `pread` bytes from
97,244,928,000 to 59,934,745,600 (38.4%). Its warm reset timing does not improve
in this run. These counts support cache reuse and read-volume conclusions;
they do not measure physical SSD traffic or establish a speed benefit.

The private soak diagnostic accepts `ESE_LIFECYCLE_SOAK_TOKENS`; its default
remains nine and reproduces the retained capture byte-for-byte. Invalid count
or mode settings reject before model loading; the requested token count must
also fit the supplied prompt and context. The numerical backend is unchanged.
Evidence: `/home/xero/Desktop/test/results/ese-qwen-placement33-20261012`.

The full-GPU expert path also completes the single-RTX-3080 33-token control
under CUDA memcheck with all API errors reported and full leak checking.
All 99 decoded tokens and 66 reset-replay outputs pass; the checker reports
zero errors and zero bytes leaked. Its 8,194,560 captured logits and all route
IDs match the native run byte-for-byte. Tracked native inventory remains
fixed after warmup, and compute-pool and expert leases drain. This does not
establish complete physical-memory accounting or an aggregate memory limit.

Separate native controls on the RTX 3060 Ti and RTX 2080 SUPER also preserve
every captured logit and route ID when moving experts from CPU to GPU, with
exact reset replays. Their strict full/original-cached/corrected-cached source
failure counts are 408/311/309 and 1/1/1 respectively, unchanged by placement.
All 33 top-token choices and 1,584 ordered expert lists match each source.
The all-GPU path is faster in the 3060 Ti diagnostic and slower in the 2080
SUPER diagnostic; this supports device-specific evaluation rather than a
universal placement choice. These two controls have one process per profile
and have not completed this full-GPU 33-token memory-check scope.
Evidence: `/home/xero/Desktop/test/results/ese-qwen-gpu-experts-memcheck-20261012`.

These results do not qualify broader lifecycle/resource safety, long-context
execution, general kernel performance, speculation or a matched Strata benchmark.
