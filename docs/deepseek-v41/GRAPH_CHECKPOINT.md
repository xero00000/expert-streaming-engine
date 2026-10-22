# Native graph checkpoint

September 10, 2026. Full-model support remains incomplete. This checkpoint
implements and validates graph components needed by the dedicated V4.1 model
graph. `deepseek41` is now recognized for model loading, but its forward graph
still raises an explicit incomplete-integration error; it is not runnable yet.

## Implemented graph components

`src/llama-deepseek-v41-ops.*` now builds:

- Engram key/value projection, per-stream normalization, signed square-root
  sigmoid gating and masked residual injection.
- Hyper-connection coefficient projection, Sinkhorn mixing, stream collapse
  and expansion. Attention consumes the preceding FFN's pre-mix; FFN consumes
  the current attention pre-mix. The expansion matrix orientation matches
  the existing HC_POST kernel without a transpose.
- Ratio-one and ratio-two compression with per-channel gate softmax, RMS
  normalization, complete-group output and retained incomplete tails.
- Adjacent-pair rotation of only the attention tail, forward and inverse,
  with ordinary or YaRN frequencies. GGML's extra YaRN magnitude scaling is
  cancelled to match the DeepSeek reference's frequency-only transformation.

`src/llama-deepseek-v41-layout.*` gives every main layer explicit ownership of
its KV, index keys, top-k output and candidate pool. It validates source order,
compression ratios and candidate-bank compatibility. The production config
resolves to 2 window-only layers, 4 Full, 4 Reindex and 30 Reuse layers. Index
keys remain attached to their owning KV bank even when no new compressed row
is published; a mutable "last index key" reference is not sufficient.

The payload calculator derives the model card's figure from the actual layout:
main KV rows are 288 bytes (FP4 values plus E4M3 scales), index-key rows are
68 bytes (FP4 values plus E8M0 scales). Three half-rate banks and one full-rate
bank require 890 bytes per token when the token count is even. At one million
tokens this is 890,000,000 bytes. The configured 1,048,576 tokens require
933,232,640 bytes. A separately calculated FP8 sliding window adds 2,703,360
bytes. These are logical future compressed payloads, not current allocations;
scratch, partial-group state, alignment and other runtime memory are additional.

## Correctness fixes found by the new checks

1. **Multithreaded CPU clamp skipped rows.** Its worker-zero-only early return
   conflicted with a loop advancing by the worker count. With four workers,
   three out of every four rows were not clamped. The zero-input Engram case
   exposed a 0.5 gate where the reference requires approximately 0.50025.
   Removing the early return makes all workers process their assigned rows.
   The same reference test fails before the fix and passes after it.

2. **CUDA graph reuse ignored most operation parameters.** Compatibility checked
   op_params only for SCALE and LATENT_ATTN. Two graphs with identical addresses
   and shapes but different RoPE/YaRN parameters reused captured launches from
   the first rotation. The second fixture differed by more than 4 in some
   elements despite forward/inverse round-trip tests passing. Compatibility
   now compares operation parameters for every node. This preserves reuse for
   unchanged parameters and updates captured launches when semantics change.

Existing CUDA ABS and STEP kernels are now reachable through the dispatcher
and advertised for contiguous FP32 inputs, enabling the Engram gate to stay on
the GPU. No new quantization format or unverified model graph is advertised.

## Evidence and scope

`tools/deepseek_v41_ops_reference.py` extracts unchanged Engram, HC, compressor
and rotation methods from the pinned official model source. It uses a portable
Torch transcription of the released TileLang Sinkhorn kernel. Diagnostic linear
maps stand in for attention/FFN in the HC block harness; this is explicitly not
a complete-model reference. Compression traces cover initial prefill, incomplete
groups and subsequent one-token steps. Rotation traces cover positions through
128 with and without YaRN; they do not establish million-token execution.

The fixture has 17 cases. Negative controls demonstrate sensitivity to feeding
a sublayer its own new pre-mix and transposing the mixing matrix. Masked and
zero-input cases exercise details that random inputs alone missed.

Validation completed:

- 19 Python checks pass, including fixture/source hashes and negative controls.
- CPU Debug: new Engram, graph and layout tests plus the existing Qwen graph
  test pass. Maximum observed graph-component absolute error is 8.35e-7.
- ASAN/UBSAN: all three native V4.1 tests pass with leak detection and
  halt-on-undefined-behavior enabled.
- CUDA Release: all 17 graph cases pass separately on the 3060 Ti, 2080 SUPER
  and 3080. Maximum observed absolute error is 2.65e-5, within the unchanged
  combined tolerance of 2e-5 absolute plus 2e-4 relative.
- Existing expert-cache tests, including compact CUDA parity and cache
  transactions, pass after the CUDA graph compatibility change.

`ops-reference.json` and `ops-results.json` hold the fixture and numerical
results. Detailed logs are under
`/home/xero/Desktop/test/results/deepseek-v41-20260910/graph-*.log`.

## Weight staging and next work

### Activation quantization checkpoint

`ggml/src/ggml-qat.*` and `ggml/src/ggml-cuda/qat.*` implement the three
released activation transforms: FP8 E4M3 values with power-of-two scales in
groups of 32, FP4 E2M1 values with power-of-two scales in groups of 32, and
FP4 E2M1 values with E4M3 scales in groups of 16. These preserve the released
amax floors and round-to-nearest-even rules. They produce dequantized FP32;
they do not yet pack physical cache storage or reproduce intervening BF16
activation casts. Existing MSE-optimized FP4 weight quantizers cannot replace
these scaling rules.

The operation uses a registered custom callback recognized specifically by
the CUDA dispatcher; unrelated custom operations remain unsupported on CUDA.
`tools/deepseek_v41_qat_reference.py` builds an independent portable Torch
oracle using native E4M3 casts and a nearest-codebook E2M1 implementation. It
transcribes the released formulas and does not execute the TileLang kernel.
All nine cases (random groups spanning magnitudes, zero groups, signed rounding
ties and adjacent floats) match exactly on CPU and each of the three GPUs.
No numerical tolerance is allowed for these fixtures. Results are recorded in
`qat-results.json`; CTest registers `test-deepseek-v41-qat`. The existing 17
graph fixtures also still pass on CPU and the 3060 Ti after this addition.

### Attention arithmetic checkpoint

`build_attention_core` consumes a query, the union of window/compressed latent
rows, an additive selection mask, and per-head sink logits. K and V use the
same whole latent vector. It pads KV to the CUDA kernel's 256-row requirement,
masks padding out, and selects the existing generic CPU path for sink support.
`build_attention_output` applies the grouped block-diagonal low-rank projection
and the final output projection. Its input must already have inverse RoPE.

The six Torch-reference cases cover 64/512-dimensional heads, 1/3/17 queries,
noncontiguous visible rows, and exact-zero sink-only queries. Every graph node
is checked for backend support; there is no CPU fallback in GPU tests. All
cases pass on CPU and all three GPUs. Removing sinks or mixing output groups
fails negative controls. The fixture includes an explicit F16 KV conversion;
this initial cache representation is temporary and is not packed FP4 storage.

These checks initially used the FP32 primitive tolerance and failed at about
1e-3: generic CPU FA accumulates F16 values in F16, and CUDA MMA rounds
intermediate operands. The attention-only tolerance is therefore explicitly
1e-3 absolute plus 1e-3 relative; the FP32 primitive and exact QAT tolerances
are unchanged. Observed maximum CPU absolute error is 0.001257, and device-zero
error is 0.000739. Sink-only outputs must still be exactly zero regardless of
that tolerance. Fixtures/results are `attention-reference.json` and
`attention-results.json`; CTest registers `test-deepseek-v41-attention`.

The sanitizer run exposed a preexisting typed-function-pointer mismatch for
`ggml_vec_dot_f16` in generic FA and `ggml_vec_dot_f32` in the grouped
projection. Their signatures now match `ggml_vec_dot_t` directly, with typed pointers
introduced inside each function; the incompatible function-pointer casts were
removed. This changes no numerical arithmetic.

ASAN/UBSAN passes the attention, FP32 primitive and QAT tests after those fixes.
The attention fixture supplies selected rows directly. It does not yet verify
indexer selection, runtime cache writes, projection-to-attention composition,
full-model loading, generation, or long-context memory/throughput.

### Indexer arithmetic and selection checkpoint

`build_indexer_scores` performs query/key dot products, rectifies each head's
score, multiplies by the already-scaled head weights, and sums over heads.
Its inputs must already have RoPE and FP4 activation quantization. The two
reference cases include the real 128-dimensional index heads, negative head
weights, multiple queries, and unequal query/key extents. Both pass CPU and
each of the three GPUs within the unchanged FP32 primitive tolerance.

`llama-deepseek-v41-index.*` supplies host candidate-block and final position
selection. Candidate scores are block maxima; the newest reachable block is
pinned even when its score is low. Candidate masks include a block's unseen
suffix, so consumers independently apply causal visibility. Final positions
are sorted, offset into the attention row union, and padded with -1. NaNs and
positive infinite input scores are rejected. Exact ties prefer lower positions;
the released Torch top-k does not specify tied-position order.

`tools/deepseek_v41_index_reference.py` executes the unchanged candidate-block
function extracted from the hash-verified official source. Exact position
comparisons use distinct scores. Additional native checks cover deterministic
ties, pinned partial blocks, no visible rows, and invalid scores. The CPU suite
now passes seven V4.1 tests; index scoring and selection also pass ASAN/UBSAN.
Fixtures/results are `index-reference.json` and `index-results.json`.

Selection currently runs on the host; the scoring graph runs on CPU or CUDA.
These components are not yet connected to model loading, cache publication,
or end-to-end generation. This checkpoint does not establish indexer throughput
or a GPU-resident long-context selection path.

### Packed KV state checkpoint

The QAT module now packs and unpacks canonical row bytes: FP8 or low-nibble-first
FP4 values, followed by per-group scale bytes. All nine numerical QAT fixtures
round-trip through the packed representation with exact reference equality.
An independent known-byte fixture checks FP4 code order, signs (including
negative zero), and the E4M3 scale encoding. Invalid geometry, nonfinite input,
and unrepresentable scales fail instead of publishing a corrupt row.

`llama-deepseek-v41-cache.*` implements one sequence's host KV state using those
rows. Main/global, index/global and window rows occupy 288, 68 and 528 bytes
at production dimensions. Global banks reserve capacity once and append paired
main/index rows atomically; readers resolve their layer's explicit owner.
Sliding-window rings return chronological rows. Begin/commit/abort protects
partial forward chunks: global appends truncate on abort, while a bounded
window snapshot restores overwritten ring slots. Commit rejects missing layer
or global-bank publications. Snapshot bytes are included in reserved-payload
accounting while a transaction is active.

Native checks use all 40 production layer owners and dimensions, a short window
to force wraps, and 16 tokens. They verify the actual final packed byte count
against the layout calculator, ring chronology, shared-owner reads, an odd
half-rate step with no publication, abort after overwrite, failed paired
publication, and incomplete-commit rejection. CPU and ASAN/UBSAN pass.

This is real packed host storage but not yet the running model's cache. It
does not establish million-token execution, GPU packed-cache reads, arbitrary
historical rewind, or compressor/Engram state transactions. Those states and
the model graph still need runtime integration. Allocator/object overhead and
temporary decoded rows are additional to packed payload bytes.

### Native model binding checkpoint

`llama-deepseek-v41-model.*` parses the explicit V4.1 source configuration and
derives all text-backbone tensor shapes. It checks routing semantics, main-layer
count, compression ownership, matrix geometry, rotary parameters and Engram
geometry. Binding accepts canonical tensor stems with or without `.weight` or
`.bias`, rejects ambiguous aliases and wrong dimensions/types, and requires
the six Engram projection/norm tensors for complete text-model binding.

The real mixed-Q2 inventory has 1,038 tensors, all accounted for by the derived
backbone bindings, including optional image-routing biases. Its unsuffixed names
are classified correctly for expert residency. The inventory contains no Engram
projection/norm tensors; a complete package must add those from the staged
Engram shards as well as external row stores. Metadata-only synthetic Engram
tensors are used only to check the completed binding contract in the test.

The `deepseek41` architecture has a dedicated hparams case and tensor creator.
GGUF metadata must include `deepseek41.config` with the source JSON and matching
ordinary architecture-prefixed dimensions. It selects global top-six routing,
normalized sqrt-softplus scores, the configured shared expert, and the dedicated
KV path; it does not allocate ordinary per-layer KV planes or treat DSpark as
main layers. Common tensor-name helpers reuse backbone spellings only. The
existing `deepseek4` execution path remains separate.

`test-deepseek-v41-model` checks all inventory bindings, required Engram tensors,
bad shapes, duplicate aliases, wrong routing and appended DSpark blocks. It also
writes a small metadata-only GGUF and exercises ESE's real architecture/hparams
loader, checking the resulting routing and cache semantics. It does not load
the full weight payload. CPU and sanitizer checks pass, and CPU/CUDA CLI builds
include the new loader path.

The staged source GGUF still needs canonical metadata and Engram packaging.
Forward graph dispatch currently stops with an explicit incomplete-integration
error. Cache/Engram lifecycle hooks and the combined forward graph remain the
next required work before any generation or tuning claim.

### Integrated attention forward checkpoint

`llama-deepseek-v41-forward.*` now connects projection, normalization, tail RoPE,
partial compressor state, packed cache publication, index scoring, candidate
selection, row gathering, attention sinks, inverse RoPE and grouped output
projection. It processes one token per layer. Graph construction is read-only;
CPU custom nodes perform state operations during execution. Matrix operations
run on the selected accelerator. A begin/commit/abort transaction covers both
the packed cache and incomplete compressor groups. Callback failures are
recorded and reject commit; finite placeholder outputs let execution finish
without throwing across a worker boundary. Rotary/cache positions are checked.

The production-precision branch preserves BF16 boundaries and FP8 activation
quantization before the projections that originally have quantized weights.
Compressor projections remain unquantized: ratio-two projections and pooling
run in FP32, followed by BF16 conversion before normalization. Index scoring
preserves the released BF16 intermediate boundaries. The alternative FP32 mode
is for initialized reference comparisons.

`tools/deepseek_v41_forward_reference.py` extracts the unchanged released
attention/compressor/indexer methods, with portable QAT and sparse-attention
implementations and initialized weights. It explicitly restores the owning
index-key bank before a layer, fixing the released shared-slot odd-step issue.
This is an initialized attention oracle, not a trained-weight/full-model oracle.
Eight layers cover window-only, owner, reindex and reuse behavior over ten decode
positions. The harness also replays an aborted complete step and a failed step
with an inconsistent position: 96 layer evaluations per backend and fixture.

Both FP32 and BF16 fixtures pass CPU and all three GPUs. BF16 outputs match
exactly on every backend; FP32 maximum absolute error is 0.00018302 on CPU
(its K-dot path converts Q to F16) and 2.99e-7 on GPU. Both retain the original
1e-3 absolute plus 1e-3 relative tolerance; no relaxation was needed. Each GPU
run verifies all 673 matrix operations stay on GPU, while state nodes use CPU.
Both fixtures pass ASAN/UBSAN, and the existing CUDA expert-cache test passes.
`forward-results.json` records the results. The separate attention fixture
also exercises the new 512-wide CUDA FP32 decode kernel on all three cards.

Two additional engine issues were exposed and fixed:

1. CPU buffers reported alignment headroom as usable capacity. A slightly
   larger scheduler graph could reuse a buffer whose aligned end exceeded
   its malloc allocation, corrupting heap metadata. The integrated GPU/CPU
   run reproduced the failure with CUDA graphs both enabled and disabled.
   A temporary allocation guard confirmed the overwrite. CPU allocation now
   reports only requested usable bytes, and buffer clearing starts at the
   aligned base. Temporary guard instrumentation was removed after validation.
2. Explicit FP32 attention precision did not control generic CPU value
   accumulation or CUDA's GQA decode choice. FP16 accumulation introduced
   differences that crossed subsequent QAT rounding boundaries. CPU generic
   attention now honors FP32 accumulation. CUDA selects its FP32 vector path
   for supported single-query F16-cache cases, including a new 512-wide
   specialization. These changes made the BF16 fixture exact, without changing
   tolerance. Typed BF16/F16 conversion callbacks and BF16 dot dispatch also
   now use matching generic function signatures, as required by UBSAN.

The model's outer Engram/HC/MoE graph and context lifecycle hooks still need to
be connected. The architecture execution guard remains. The source GGUF lacks
tokenizer metadata as well as the Engram projections, so canonical packaging
must include the separately supplied tokenizer and Engram assets. All five
mixed-Q2 backbone shards have now passed published SHA-256 verification; the
two Engram shards are still downloading.

A resumable HDD download is running for the pinned mixed-Q2 GGUF and the two
quantized Engram shards, under `/mnt/games_sdb2/ai/ese-models/deepseek-v41`.
The staging script verifies each completed file against its published LFS
SHA-256. Its status and script live in the results directory's `downloads/`
folder. A download still in progress is not a verified complete checkpoint.
The original 510 GB source checkpoint has not been downloaded.

Next: complete attention projection/indexing and shared cache read/write,
including the model's quantization-aware cache transforms; integrate these with
the dedicated V4.1 loader/graph; run an initialized complete-model reference;
then load the staged quant, generate text and measure real request/cache
behavior. The active goal remains full-model execution and tuning.
