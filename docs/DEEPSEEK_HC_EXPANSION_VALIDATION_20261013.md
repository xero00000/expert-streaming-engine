# DeepSeek V4.1 HC expansion precision validation

Date: 2026-10-13. Evidence: `results/ese-deepseek-block-gate-20261013`.

## Reproduced failure and correction

The frozen CPU release library reproduces the initialized block-reference
failure at token 7, layer 7, element 82: `-1.40625` instead of `-1.4140625`.
The previous same-input study isolated an attention residual at a BF16 rounding
midpoint. Computing coefficients in FP64 and rounding them back to FP32 before
expansion does not resolve that midpoint.

A private CPU prototype retained FP64 combination and post coefficients
through the residual expression. It passed the complete original block fixture:
96 layer evaluations, including abort and injected-failure replays, with maximum
absolute error `9.5367431640625e-7`. No saved reference, tolerance, weight or
input was changed.

The native correction adds `ggml_hc_post_precise`. It computes the small
Sinkhorn matrix and post coefficients from the original FP32 projected values,
scale and bias, keeps them in FP64 through expansion, and emits FP32 output.
The original inputs are explicit graph dependencies; the allocator therefore
keeps them alive until expansion finishes. The four-stream residual ordering
and source/destination indexing remain unchanged.

CPU workers use bounded automatic arrays. CUDA computes the coefficients once
per token into 576 bytes of shared memory, then distributes hidden elements
across the block. It makes no temporary device allocation and has no CPU
fallback. The existing HC_PRE primitive and its exact CPU normalization oracle
retain their FP32 behavior. Ordinary HC_POST calls retain their existing mode.
The additive C API has an isolated installed header, `ggml-hc.h`.

## Validation

| Configuration | Complete registered suite |
| --- | ---: |
| Ubuntu CPU Release | 92/92 |
| Ubuntu CUDA Release | 100/100 |
| Ubuntu CPU ASan/UBSan, leak checking and halt on error | 92/92 |

Independent long-double expansion checks cover 1–8 streams, 2 and 20
normalization iterations, three tokens and 65 hidden elements. The existing
normalization-order checks, original primitive fixtures, block fixture and
whole-model token fixture all remain active.

Separate primitive, complete block and complete token checks pass on RTX 3080,
RTX 3060 Ti and RTX 2080 SUPER. Model tests explicitly require precise HC
expansion and matrix multiplication to stay on the selected CUDA backend.
Strict per-card primitive memory checks report zero errors and zero leaked
bytes. The complete block runs retain 96 layer evaluations and both replay
checks on every card.

The full sanitizer suite also exposed an uninitialized `llama_split_mode` in a
synthetic model fixture. Four synthetic DeepSeek tests now value-initialize
their model objects. This changes no production loader policy or fixture data.
The final full suite passes with the original strict sanitizer settings.

## Evidence limits

These are initialized small-model fixtures, not a trained-model quality or
throughput qualification. The ordinary Qwen gate remains closed. Original
Qwen numerical parity, filled 64K inference, concurrent serving, matched Strata
benchmarks, current native Windows installers and signing are not established
by this validation. Earlier installer candidates predate this native change.

The evidence retains the reproduced baseline failure, private prototype setup
failures, the missing Python dependency invocation, the intentionally stopped
broad rebuilds and the sanitizer fixture failure separately from successful
qualification runs.
