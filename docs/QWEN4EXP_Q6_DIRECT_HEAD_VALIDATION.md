# Direct packed Q6_K output projection

Evidence: `/home/xero/Desktop/test/results/ese-q6-direct-head-20261011`.
`qualification-final.json` records the accepted guarded checks. The original
and intermediate failures remain in that directory; they are not passing evidence.

The original Q6_K vocabulary weights are read directly on CUDA with F32
activations, F64 dot accumulation and a fixed warp reduction. The kernel does
not allocate a dequantized vocabulary matrix or quantized activation scratch.
The complete 2560 × 248320 packed head occupies 521,472,000 bytes; materializing
its F32 weights would require 2,542,796,800 additional bytes.

`ESE_CUDA_Q6_DIRECT_F32=1` selects this arithmetic for supported regular Q6_K
matrix products requesting `GGML_PREC_F32`. The existing experimental
`ESE_QWEN4EXP_EXPERT_F32=1` also selects it and requests explicit precision in
both output-builder overloads. Environment selection is fixed for the process.
Unsupported layouts and CUDA split weights retain the existing dispatch.
The normal Qwen loader gate stays closed.

## Numerical and lifecycle evidence

The independent decoder/F64 oracle uses the original read-only GGUF bytes.
Fixtures include eight model-derived activation vectors, 7/257/248320 output
rows, 1/2/8 columns, two input allocations, three graph replays, precision
transitions, and padded small weight/input/output layouts with canaries.
Each explicit profile checks 16,441,392 oracle values including replay.
The printed total 16,476,240 also includes 34,848 default-policy comparisons.
These counts do not establish a universal floating-point identity theorem.

All five guarded CPU/CUDA numerical profiles have byte-exact explicit oracle
results, reproducing the initial generation. All three physical GPU component
runs pass unsuppressed all-API memory checks with zero errors and leaks. The
strict CPU ASAN/UBSAN/leak run also passes and its output artifact is byte-exact
to the guarded native CPU fixture.

On the exact captured model head input, all 248,320 updated logits are
byte-exact to the independent original-weight/F64 reference. The prior head
has maximum error 0.129748. Full first-token error against the pinned upstream
model decreases from 0.129172 to 0.00123310; full-model parity is still false.
The next failing boundary is full-attention layer 3 output, including the
configured F32-cache/default-Flash-Attention diagnostic.

CPU-expert/hybrid/all-API model controls match all 2,234,880 teacher values
and nine per-token route records exactly. The model memory check reports
zero errors/leaks; RAM/staging caps and complete drain pass. Batch depths
one/eight match nine token checks and two complete state/continuation checks.
All head calls request explicit precision and execute on CUDA. Five component
default-policy pairs and the model default control are byte-exact to their
previous frozen controls. Thirteen focused normal CTests pass. Full-model
checks use only the selected Ampere pair; component coverage includes Turing.

## Retained integration failures

The first integration changed an output-builder overload that this model does
not call. The actual overload now requests the precision, and the probe
asserts the output operation's requested precision and actual CUDA execution.

The initial unguarded backend also changed existing Q6_K projections that
already requested F32 accumulation, so default-model teachers/routes differed.
The explicit process opt-in repairs that regression; its matched default
teacher and route files are byte-exact to the previous frozen runtime.

Legacy/disabled-direct negative component attempts fail with an unwritten
output sentinel after a graph transition, rather than measuring vocabulary
arithmetic error. They are retained diagnostics and are not used as numerical
negative proof. The independent same-input head study supplies that proof.

## Remaining scope

2D layouts are the fixture scope. Four-dimensional broadcasting, tensor split,
broader view families, full-model Turing, other cache representations and
long-context/shared-model stress remain unqualified. This stage does not
claim end-to-end throughput improvement or production promotion.

A separate nine-token streamed upstream reference is now complete in
`results/ese-qwen-multitoken-reference-20261011`. It exposes larger downstream
model divergence beginning at position 3, despite the accurate head. Full
upstream state/sequence qualification, resources, cache/speculation/kernels,
GLM/Maya, fair Strata comparisons, serving, README rebrand and GitHub publication
remain on the original roadmap.
