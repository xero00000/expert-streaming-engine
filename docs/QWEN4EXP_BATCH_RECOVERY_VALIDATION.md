# Qwen4Exp true-batch accepted-prefix recovery

Result: all 228 required cases pass. Each ordinary teacher stream remains
bit-identical to the frozen prior run; expert RAM/staging caps and final drain
also pass the evidence audit.

This diagnostic checks recovery of the model state after a real verifier batch
has processed inputs that will be rejected. It does not qualify target sampling,
MTP, production hybrid placement or an inference speed improvement.

## Configuration and coverage

Evidence directory: `/home/xero/Desktop/test/results/ese-batch-recovery-20261010`.
The shared two-shard coder IQ1_M artifact is read-only. Runs use CPU experts,
the explicit dense-vector diagnostic, stable short-batch attention and the fused
short-batch gate/up control. The frozen GGML library is
`c13e871cf10ffeafb963050132238fb1d3cc0fec75ca39d1adf0e4cab7ae63b9`;
the gated LLAMA library is
`d88a2e4d3cfb441961f825086d478d15c2d72a0636ab07c2bf57608aaeb0fb74`.

The matrix has 228 required cases: prefixes 8 and 32, CPU and GPU-fallback
checkpoint modes, verifier depths 1/2/4/8, every accepted input count from zero
through the full depth, and three replay policies. The replay policies are
independent scalar calls, one batch requesting every output, and one batch
requesting only the last output. Every rejected suffix is changed, with EOS as
the last rejected input. This tests causal independence from future inputs and
history rollback at EOS.

For each case, the probe independently obtains the accepted prefix by scalar
execution, saves its complete canonical sequence bytes and retains its expected
continuation. It then verifies the poisoned batch, restores the checkpoint,
replays accepted inputs and compares:

- Every accepted logit bit and all 48 MoE route selections before recovery.
- The replay batch's last output against the scalar result, when applicable.
- Complete sequence bytes after recovery, including recurrent and PLE history.
- Continuation logits and route captures after recovery.
- Discarded checkpoint rejection, ordinary teacher-stream preservation, expert
  RAM/staging limits and final lease/load/reservation drain.

These checkpoints return `BASE_REPLAY_REQUIRED`: they restore the original
window, after which the accepted inputs must be decoded again. A successful
restore is not direct restoration of the accepted state. The accepted count in
this diagnostic counts processed input tokens. Serving's sampled output list
also includes its corrective/bonus output; that last output has not been decoded.

## Reproduction and interpretation

Each completed run saves its exact `argv.json`, `env.json`, output logs and exit
status. `finalize-evidence.py` checks the entire required matrix, unchanged
teacher files and resource counters before producing `checkpoint.json` and an
evidence manifest. Its check is required before treating the matrix as qualified.
The initial missing executable/link attempt is retained separately from its
successful retry.

The probe prints verification, restoration and replay durations, but callbacks,
state serialization and comparison work instrument these runs. They are not an
accepted-tokens-per-millisecond benchmark. CPU mode reserves two bounded snapshot
slots; GPU fallback reserves recurrent shadow rows. Reference and recovered
comparison buffers are held for one accepted-prefix case at a time.

Normal diagnostic targets compile with the production loader gate closed. The
normal sampler probe's negative control rejects `qwen4exp` as an unsupported
architecture. Long contexts, interleaved serving, cancellation/allocation
failures, full peak memory, actual sampling/MTP, parallel dense projections and
production hybrid execution remain separate roadmap requirements.
