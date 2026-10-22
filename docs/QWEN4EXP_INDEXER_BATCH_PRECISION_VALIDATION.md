# Qwen indexer precision and exact non-Flash batch replay

Evidence: `/home/xero/Desktop/test/results/ese-qwen-indexer-f32-20261011`.
The accepted frozen generation is `policy-runtime`; `qualification-final.json`
records its checks. The original upstream weights and goldens are preserved.
Full release qualification is still incomplete.

The existing experimental `ESE_QWEN4EXP_EXPERT_F32=1` now requests explicit
indexer key/query projection precision and an F32 indexer cache. The cache
selection happens in context initialization so allocation, retiering and
reported policy use the same type. At context 4096 the indexer occupies
24 MiB rather than the previous 12 MiB F16 representation. Other architectures
and the default Qwen policy retain their choices; the production gate stays closed.

The same policy also requests explicit precision for non-Flash value
aggregation. This uses existing CUDA arithmetic, without a new weight layout
or full-matrix scratch allocation.

## Evidence for the corrections

On the exact captured native input, the old indexer cache differs by as much
as 0.0468874 from original-weight/F64 projection, and also differs from its
F16-rounded result. Cache rounding alone does not explain the error.
The corrected runtime passes 216 original-weight key/query projection cases;
its maximum same-input projection error is 7.15256e-7.

The retained batch trace compares only visible keys, excluding future masked
positions. In full-attention layer 3, attention scores and softmax match
single-step execution byte-for-byte; the first meaningful difference is
value aggregation, with maximum error 2.38419e-7. That difference propagates
into saved state and continuation. The previous batch runner returns 3.
Explicit value aggregation now preserves exact batch replay at depths one/eight
and ubatch four: nine token checks and two complete state/continuation checks.

## Accepted scope

One nine-token prompt executes all 48 layers with the original quantized
weights, F32 K/V/indexer, non-Flash attention and the explicit expert policy.
The unchanged elementwise bound is `2e-5 * (1 + abs(reference))`.

- All 2,234,880 logits and 2,160 named layer boundaries pass; maximum logit
  error is 1.60336e-5.
- All 432 ordered expert-route rows match the upstream reference.
- All 109 original recurrent, convolution/PLE, K/V and indexer state cases
  pass. The parser consumes the entire serialized continuation state.
- CPU-expert, hybrid, memory and batch controls reproduce all teacher logits,
  routes, and prefix/continuation state files exactly.
- The hybrid unsuppressed all-API memory run reports zero errors and leaks;
  RAM/staging budgets and complete drain pass.
- The matched default control reproduces the preceding frozen default logits
  and routes byte-for-byte. Thirteen focused normal CTests pass, followed by
  three normal gate/schema checks after centralizing the cache policy.

Full-model checks use only the selected RTX 3060 Ti/RTX 3080 pair. This does
not qualify full-model Turing, tensor split, other cache representations,
selective long-context QSA, multiple prompts or sequence transforms.
Intermediate parameter reporting incorrectly labelled the 24 MiB allocation
F16; centralizing the context policy fixes this, and those earlier runtimes
remain separately frozen. A projection-auditor assumption about unpadded
initial blocks also failed and is retained as a harness diagnostic.

Next: extend pinned reference/native prefix and prompt coverage, then complete
the outstanding lifecycle, resource and long-context gates. All remaining
kernel/cache/speculation, fair Strata, serving, README rebrand and
GitHub publication work stays on the original roadmap. No throughput or
production-promotion claim follows from this numerical milestone.
