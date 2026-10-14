# PLE and full-attention projection precision

Evidence: `/home/xero/Desktop/test/results/ese-qwen-ple-precision-20261011`.
`qualification-final.json` records the accepted integration checks and the
remaining upstream mismatches. Full-model parity remains incomplete.

The existing experimental `ESE_QWEN4EXP_EXPERT_F32=1` policy now preserves
F32 activations/results in PLE key/value and gated full-attention query,
key, value and output projections. These requests use existing explicit
projection implementations. Default Qwen and other architecture builder
behavior is retained. The production loader gate stays closed.

The PLE correction is applied in the Qwen model builder through its existing
projection trace tensors. The standalone PLE helper's default contract is
unchanged. Attention precision is scoped to Qwen by architecture and the
same opt-in, rather than by tensor-name matching.

## Same-input original-weight comparisons

The pinned upstream module consumes the exact captured native activation and
original reconstructed GGUF weights. No model payload or existing golden is
changed. The PLE embedding remains byte-exact to its independent row source.

| Boundary | Before maximum error | Corrected maximum error |
| --- | ---: | ---: |
| PLE key projection | 0.000888526 | 5.96046e-8 |
| PLE value projection | 0.000182755 | 1.76951e-8 |
| PLE normalized key | 0.0582542 | 1.52588e-5 |
| PLE residual | 0.000210755 | 7.45058e-8 |
| Attention query/gate projection | 0.00427574 | 4.76837e-7 |
| Attention key projection | 0.00120671 | 5.96046e-7 |
| Attention value projection | 0.00182207 | 2.14577e-6 |

The PLE-only generation brings the first three layers' compared boundaries
within approximately 1.5e-6 of the full upstream first-token reference.
After the attention correction, first full-attention output error decreases
from 0.00792846 to 0.000204176 against that same-input F32 reference.

An additional, explicitly labelled reference rounds stored K/V to F16 to
inspect the configured cache representation. Its output error is 4.40478e-5.
The F32 reference is preserved; this diagnostic is not a parity waiver.
Small projection differences can cross F16 rounding boundaries. Wider cache
and attention arithmetic qualification remains required.

## Integration qualification

CPU-expert and hybrid controls match all 2,234,880 nine-token teacher values
and per-token routes byte-for-byte. The hybrid all-API memory check passes
with zero errors and zero leaked bytes and reproduces those same teachers
and routes. RAM residency/staging remain bounded and leases, loads,
reservations and staging drain completely. Model memory checks expose only
the selected RTX 3060 Ti/RTX 3080 pair; this does not qualify full-model Turing.

Verification at depths one/eight matches nine token outputs/routes, complete
serialized sequence state and next-token continuation. Matched default
teachers/routes remain byte-identical to the preceding frozen runtime.
Twelve focused normal CTests pass, including production loader rejection.

## Next numerical boundary

Complete first-token logit maximum error is still 0.129172. An independent
F64 dot study of the original Q6_K vocabulary weights on the exact native
output-head input reproduces an error of 0.129748 across 248,320 logits.
The existing default head path still quantizes activations. This is now the
next concrete correction, rather than an inferred upstream hidden-state bug.

Simply dequantizing the full vocabulary matrix to F32 would require about
2.54 GB of additional device storage. The next kernel should consume packed
original weights and F32 input directly, retain bounded scratch, and qualify
accuracy, graph replay and memory before model integration.

Full upstream multi-token/state/long-context parity, resource/failure stress,
remaining kernel/cache/speculation work, GLM/Maya, matched Strata benchmarks,
serving and GitHub/README publication remain on the roadmap. No speedup or
production promotion is claimed by this correctness stage.
