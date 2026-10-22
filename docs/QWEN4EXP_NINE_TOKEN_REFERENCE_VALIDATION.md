# Nine-token upstream comparison with F32 K/V

Evidence: `/home/xero/Desktop/test/results/ese-qwen-multitoken-reference-20261011`.
`qualification-scoped.json` distinguishes passing sequential controls from
sparse-indexer and exact batch-replay failures. Full upstream parity remains open.

The streamed pinned upstream reference executes all 48 decoder layers over the
nine actual token IDs, reading both original GGUF shards without changing them.
It emits 545 trace files containing 45,305,282 finite values, including recurrent,
convolution, PLE, K/V and indexer state. Its first token agrees with the earlier
single-token reference within the unchanged elementwise bound.

## Passing scope

With the existing explicit arithmetic policy, F32 K/V and Flash Attention off,
all 2,234,880 logits and 2,160 named layer boundaries satisfy
`2e-5 * (1 + abs(reference))`. All 432 ordered MoE route rows match exactly
(4,320 selected expert IDs). Maximum per-token logit error is 1.81794e-5.
This is one short prompt on the selected RTX 3060 Ti/RTX 3080 pair.

The loader previously rejected an F32 value cache without Flash Attention with
“V cache quantization requires flash_attn”. Its type check now admits F32
alongside F16/BF16; quantized cache restrictions remain. The real context/model
run and three focused normal CTests pass, including production Qwen rejection.

CPU-expert, hybrid and hybrid all-API memory controls reproduce the complete
nine-token teachers/routes and prefix/continuation sequence-state files exactly.
The memory run reports zero errors and leaks; RAM/staging caps and complete
drain pass. These checks use the frozen `f32-cache-runtime`.

The sequence-state parser consumes all 118,510,820 bytes and verifies positions
and PLE history. It applies the established value-head permutation and native
[value,key] → source [key,value] recurrent-state orientation. Original-state
comparisons pass for 36 recurrent, 37 convolution/PLE and 12 K/V layer pairs.
Maximum errors: recurrence 2.26498e-6; convolution/PLE 1.69277e-5;
K 6.95884e-6; V 9.53674e-6. Parser mistakes are retained as labelled diagnostics.

## Remaining failures

The sparse-indexer cache fails the original F32 state comparison: maximum
error 0.0468941 across the twelve full-attention layers. A separately labelled
F16-rounded reference also fails, so cache representation alone does not
explain it. The model builder still requests default precision for indexer
key/query projections. The next check is a same-input original-weight study
before correcting that policy.

Exact batch replay also fails at depth eight/ubatch four: maximum teacher
logit difference 1.40667e-5, with unchanged expert selections and top-1 choices,
but serialized state and continuation are not byte-exact. The runner returns
3 and reports unqualified. This is retained as a failure; the upstream
numerical tolerance is not substituted for the exact speculation invariant.
Next, locate the first batch-versus-single-step attention difference and
qualify deterministic F32 attention arithmetic.

Default Flash Attention/F16 and F32-cache/default-Flash configurations do not
pass the original upstream model comparison. They remain distinct from this
F32/non-Flash diagnostic. Neither a benchmark speedup nor release promotion
follows from this short numerical milestone. Selective long-context QSA,
broader prompts/devices/cache types, sequence transforms, memory/failure
stress, remaining kernels/cache/speculation, fair Strata comparison,
serving, README rebrand and GitHub publication remain required.
