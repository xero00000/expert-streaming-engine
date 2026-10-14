# Qwen short-prefix execution on Turing — 2026-10-12

The original two-shard Qwen GGUF completes three nine-token decode/reset rounds
on the 2080 SUPER, with CPU experts and all dense layers on that GPU. Eighteen
replayed tokens reproduce baseline logits and all captured routes bit-for-bit.
The nine-token baseline passes all 2234880 logits at the existing
`2e-5 * (1 + abs(reference))` bound and all 432 ordered expert selections against
the pinned independent reference. Maximum absolute logit difference is
1.5735626220703125e-5. The original 33-token numerical gate remains failed.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-turing-model-20261012`,
`qualification-final.json`. The frozen runtime is shared with the distinct
[graph-abort-status validation](DECODE_ABORT_STATUS_VALIDATION.md); normal
Qwen loading remains gated. Both external model shards stay read-only, with
unchanged size/inode/device/mtime identities before and after execution.

The control uses context capacity 4096, F32 KV, Flash Attention off, explicit
expert arithmetic, 64 MiB expert RAM and 4 MiB staging. The 2080 SUPER is the
only visible CUDA device. Sampled native RSS peaks at 26488651776 bytes and
owned process VRAM at 4938792960 bytes. Device-total usage is recorded separately.
These 50 ms target samples do not capture every possible shorter allocation peak.

The same three-round workload passes an isolated all-API/full-leak CUDA memory
check, terminal exit 0: zero errors and zero leaked bytes. Instrumented baseline
logits, routes and input IDs are byte-identical to the native Turing control.
Instrumented owned VRAM peaks at 4942987264 bytes. The sampler's instrumented
RSS value covers the sanitizer parent only and is not a model-RAM measurement.

All baseline route records match the selected Ampere control exactly. Baseline
logit bytes differ between devices while both pass the fixed numerical bound;
cross-device bit identity is not claimed. Within the Turing context, reset
replays are exact. This covers whole-model short-prefix execution with CPU
experts; it does not qualify GPU experts, long contexts, concurrent serving,
full-model upstream numerical parity or an end-to-end speed improvement.

A separate 33-token Turing diagnostic still fails the numerical gate: one logit
is outside the unchanged bound against each preserved full-prefix and cached
reference. The failing output row differs between references. All 1584 ordered
expert selections and all 33 top-token choices match both references. A traced
rerun reproduces the untraced Turing logits/routes exactly. Cross-device trace
comparisons first differ at the layer-zero gated-residual block's mixed input,
including token zero. An isolated replay of the original block reproduces each
device's output exactly; its normalization and first projection are identical
across devices, while the second projection's summation differs. This does not
establish the cause of every later numerical failure. Two controlled changes
to residual projection arithmetic worsen the full-model comparison and are
rejected as fixes. No reference, tolerance, weights or production arithmetic
was changed.
