# IQ3_XXS vector-dot fractional scale

The CUDA IQ3_XXS/Q8_1 vector dot now applies the stored fractional scale
`(2*ls + 1)/4` in floating point. The old integer expression
`(ls*sumi + sumi/2)/2` discarded fractional contributions, including negative
ones. Weight bytes, activation quantization and dispatch policy are unchanged.

Evidence: `/home/xero/Desktop/test/results/ese-iq3-dot-scale-20261011`.

## Independent component qualification

`tests/test-iq3-cuda-dot.cu` calls the production header's device function and
compares each 32-element partial dot against independently dequantized
IQ3_XXS weights and F64 host accumulation. Its 524,288 systematic cases cover
every combination of 256 grid entries, 16 scales and 128 sign codes, across
all eight partial positions. Another 8,192 deterministic dense cases vary
grids, signs, scales and signed activations. Power-of-two weight/input scales
make the expected results exactly representable: no tolerance is needed.
Each GPU checks 4,259,840 partial dots using bounded batches of 4,096 blocks.

The preserved old header fails 375,521 comparisons on the 2080 SUPER, with
maximum error 16. The corrected header has zero mismatches on 3060 Ti,
2080 SUPER and 3080. All three unsuppressed all-API memchecks pass, with zero
errors and zero leaked allocations. Commands and exit codes are retained in
`component-runs.json`; `run-component.py` reproduces the checks.

The first harness omitted ggml's FP16 table initialization and produced an
invalid host oracle. That failed run is retained as excluded evidence.
The final harness initializes ggml before using its public conversion callback.
Initial compiler/linker and CUDA-directory-scope configuration failures are also
retained; they are not qualification results.

## Integration status and limits

The production library rebuild is complete. Ten focused CTests pass, including
the current CUDA regression on all three cards. The rebuilt GGML is frozen with
its hash in `integration-library.json`.

The retained real layer-20/expert-10 replay passes CPU and all three GPUs, plus
unsuppressed all-API GPU memchecks with zero errors/leaks. Old-library replays
reproduce the archived CPU/GPU hidden and down outputs exactly. Corrected GPU
hidden output differs from the independent fractional-scale arithmetic model
by at most `9.251701e-8`, under the unchanged `1e-6` bound. All three GPU hidden
and down outputs agree byte-for-byte. The hidden correction versus the old GPU
result is at most `1.269579e-5`. CPU outputs, ordinary standalone MMQ up/gate,
and Q2 down with fixed captured input stay byte-identical. `expert-audit.json`
records the individual comparisons.

All four full-model controls are terminal with process exit 0. Native and GPU
memcheck sequence-bank focus preserve the preceding generation's captured
streams exactly, with zero sanitizer errors. Each ordinary verifier mode passes
18 JSON checks and three isolation/bank/pending-copy checks; RAM/staging caps
and lease/load/reservation drain pass. The two ordinary modes agree byte-for-byte
with each other. These are lifecycle checks against each run's own teacher.

**The old hybrid numerical baseline gate fails.** Each ordinary teacher contains
8,194,560 values; maximum difference from the preceding generation is
`4.25720584` (relative L2 `0.12892576`). The final baseline logits differ by up to
`1.95406151`. Among the 48 MoE streams, 35 change ordering, 20 change selected
sets, and ten change the first-five set used by the hybrid partition. Four of
the 12 QSA streams change ordering. `model-numerical-delta.json` separates these
streams; `model-lifecycle-audit.json` explicitly records the failed numerical
baseline gate. No assertion is relaxed in the original strict audit.

An ablation rebuilds only the IQ3_XXS MMVQ instantiation with the preserved old
device function, then links it with every other current object and uses the same
model library, inputs and settings. That control restores all 8,194,560 teacher
values, baseline logits and routes byte-exactly to the preceding generation.
`ablation-audit.json` and `run-old-iq3-ablation.py` retain the proof and commands.
This isolates the observed change to the corrected IQ3 arithmetic. It does not
prove the full-model change improves accuracy against an upstream model oracle.

The first model audit read the sanitizer summary from only stderr; this run
wrote it to stdout. A stream-corrected strict audit then exposed the real hybrid
baseline change. Both failures are retained. A separate lifecycle audit reports
internal recovery/memory results without relabelling the failed numerical gate.
The normal build keeps the Qwen architecture gate closed.

The prior actual expert study identified a larger CPU/GPU activation-format
difference: CPU IQ3 uses Q8_K, fused GPU IQ3 uses Q8_1 with half scales; CPU
Q2 down consumes F32 input while GPU Q2 down quantizes it. Correcting this
fractional scale alone cannot establish cross-placement equality or qualify
increased GPU residency. The measured corrected CPU/GPU hidden difference is
still `0.0102238655`. No throughput improvement is claimed.

Next: qualify a consistent CPU/GPU expert arithmetic policy against original
payload references, then integrate and revalidate placement and whole-model
logits/routes. This correction is component-qualified; hybrid numerical
promotion and the wider release roadmap remain open.
