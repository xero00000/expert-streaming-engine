# Preserve target sampler state during speculative recovery

The target sampler samples and accepts only through the first rejected draft,
or through the bonus token when all drafts match. Its state already represents
exactly the returned output list. Model-state recovery must preserve that sampler.

## Reproduced failure and repair

Evidence: `/home/xero/Desktop/test/results/ese-sampler-recovery-20261010`.
The frozen old common archive and actual shared Qwen artifact reproduce a
first-rejection failure at prefix8/depth2 in CPU checkpoint mode. Scalar and
batched verification, complete recovered sequence bytes, logits and accepted
history are exact, but the old common helper rewinds the RNG. Twenty-nine of
32 subsequent draws differ from an untouched post-verification sampler.

The old helper clones the pre-draft sampler and calls acceptance again for the
returned IDs. Acceptance does not reproduce random draws or sampling-time
adaptive updates. Both direct restoration and base replay now preserve the live
verified sampler. Pre-draft checkpoint creation also stops constructing/cloning
an unnecessary sampler. Public signatures and optional legacy-owner cleanup
remain compatible; the restoration contract is documented in `speculative.h`.

The identical negative-control regression passes with the repaired archive:
complete state/logits, RNG, accepted history and all32 subsequent draws match.
The normal server builds. Three freshly linked focused normal tests and two
freshly linked ASan/UBSan tests pass, with leak detection and no suppressions.
Those native tests cover existing checkpoint-owner and MTP transaction behavior;
the artifact-dependent sampler regression supplies the direct sampling evidence.

## Wider controlled qualification

`qwen4exp-sampler-probe` drives actual target sampling, pre-draft checkpoint
creation and `common_speculative_commit`. Drafts follow a scalar target oracle
until a forced rejection; rejected future inputs include EOS. Full acceptance
also exercises the commit path that does not restore the checkpoint. The probe
checks scalar/true-batch output IDs, pre-recovery sampler agreement, complete
recovered sequence bytes, last logits, RNG, Mirostat mu, adaptive state and
accepted history. It requires checkpoint discard and no new sampler snapshot.

The completed commit matrix covers prefixes8/32, CPU/GPU-fallback modes,
depths2/4/8, every returned output count1..depth, and greedy, seeded top-k,
adaptive, grammar-constrained and Mirostat2 policies:280cases. All four runs
exit0 and every case passes. Run logs/exits and the final evidence audit are
authoritative.
The corrected identical-control executable, full commit executable and their
sources are preserved separately.

Expert RAM/staging limits and final leases/loads/reservations drain pass the
audit in all four runs.

For greedy/seeded/adaptive/grammar policies,32 future draws use identical target
logits and an untouched cloned sampler. Mirostat2 samples through the context's
RNG rather than the common sampler's RNG: this probe checks scalar verification
and preservation of mu but omits that cloned future-draw comparison. Direct
per-step architecture journals, actual MTP drafting, reasoning/expiring-bias
policies, long context, concurrent serving and accepted-token throughput need
their own qualification.

All real-model runs use controlled CPU expert placement, explicit dense-vector
projection and the qualified opt-in short-batch attention/fused controls. These
tests do not qualify production hybrid arithmetic or a speculation speed claim.
