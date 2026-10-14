# Qwen4Exp sequence lifecycle validation

The October 10 investigation repaired state lifecycle errors that must be fixed
before speculative decoding is qualified. The production Qwen4Exp loader gate
remains closed. These are correctness checks, not speed measurements.

## Reproduced errors and repairs

- A cached hybrid graph retained a sequence-zero state view when the next batch
  explicitly requested sequence one. The reuse key now includes every token's
  recurrent sequence slot and position-zero reset flag. Ordinary continuation
  still reuses its graph.
- Deferred state copying reshaped all layers to the base delta-net width. The
  PLE layer stores additional convolution state; copying now uses the entire
  stored row. Attention token positions also remain unchanged during a hybrid
  recurrent-row copy.
- The CPU backend reported cancellation, but decode ignored its graph status and
  committed pending PLE history. Decode now returns the failure before history
  and output commit. Recovery requires the caller's checkpoint restore; this does
  not make partially executed recurrent inference transactional by itself.
- Restoring an empty sequence inspected a zero-token metadata batch. It now
  removes the destination's cells and restores its recurrent row without
  allocating or indexing that batch.
- Generic per-step speculative checkpoints save delta-net and convolution state
  but omit Qwen's PLE accepted-prefix journal. Qwen declines this incomplete path.
  Automatic selection tries complete-row shadows, then serialized replay; an
  explicit per-step request fails. A qualified per-step implementation remains
  roadmap work.

The PLE extension is in `s_l`. Compressed-attention raw indexer keys are stored
separately in `kr_l`; the complete public sequence serializer already carries
them.

## Reproducible checks

`test-hybrid-state-copy` needs no model files. It executes the real copy graph
against three different row widths, simultaneous permutations, cloning and
identity copies; checks token-position preservation, empty sequence round trips
and the architecture-specific checkpoint selection. Normal CUDA-build CTests
and strict ASan/UBSan/leak checking pass. The frozen pre-repair library fails
the token-position regression.

`qwen4exp-lifecycle-probe` is an artifact-dependent diagnostic for the 48-layer
coder IQ1_M model. It uses the regular CLI model/cache arguments and forces two
sequence slots. Set `ESE_LIFECYCLE_DIRECTORY` to an owned output directory for
temporary session files, and optionally `ESE_LIFECYCLE_PREFIX` (8–256, default
32). The probe itself never opens the loader gate; it requires the separate
diagnostic build and per-run opt-in documented in the roadmap.

The probe checks full/sequence restore, migration, copying, removing/keeping
the copy source/destination, clear/remove replay, truncated-state rollback,
cancellation followed by explicit restore, CPU speculative rollback/replay,
whole and per-sequence session files, and an independent sequence. Every
continuation compares all 248,320 finite logits by their bits. Identically
placed continuations compare all 60 route tensors. Independent placement
compares the 48 MoE routes and omits QSA's physical block IDs, which change with
cache location. It does not claim long-context sparse-selection equivalence.

Optional `ESE_LIFECYCLE_TEACHER` writes the prefix and continuation logits, while
`ESE_LIFECYCLE_BASELINE` writes final logits/routes. Set
`ESE_LIFECYCLE_BASELINE_ONLY=1` for the shorter graph-reuse comparison. Repaired
reuse and reuse disabled match all 8,194,560 tested logits and the final routes
over the same 32-token varied prefix and continuation.

Evidence and frozen library/binary hashes live in
`/home/xero/Desktop/test/results/ese-lifecycle-20261010`. That run uses CUDA0/2,
fixed five-GPU/five-CPU expert placement, 2,680 MiB expert VRAM per device,
64 MiB expert RAM and 4 MiB staging. This validates a short text lifecycle on
that configuration, not the complete B4 or production release gate. QSA pruning
beyond the model's 2,048-token selection budget, interleaved requests, GPU-shadow
and accepted-prefix checkpoints, broader failure injection and 64K/resource
stress still require qualification.


## October 10 checkpoint isolation continuation

The subsequent investigation in `results/ese-spec-state-20261010` reproduced
three further problems: GPU base restore retained future PLE tokens; whole-slot
GPU snapshots rewound another sequence's state and metadata; malformed CPU
checkpoint restoration reported success after atomic rollback failed to reserve
contiguous token cells.

GPU fallback now snapshots only its sequence row on each backend. Restoring it
preserves other rows and uses the public tail-removal operation to remove both
KV cells and PLE history. Deferred sequence copies are materialized before save.
Shadow allocations for this two-slot artifact decrease from 125.39/99.75 MiB on
CUDA0/2 to 62.70/49.88 MiB. These are snapshot-buffer sizes, not total peak VRAM
or a throughput measurement.

A partial hybrid restore consumes serialized token metadata without moving live
attention cells: no attention weights were serialized in that format. This also
allows failed partial restoration to recover the original state while preserving
another sequence's interleaved KV cells. CPU restore reports failure if parsing
or rollback fails. CPU/GPU snapshots reject a restore for the wrong owner and
invalidate ownership on discard.

At that checkpoint the runtime still had **one active generic checkpoint per context**.
Saving a second window changed the owner. The following bank continuation replaces
that limitation for generic GPU/CPU fallback; per-step and architecture-private
journals remain separate capabilities.

`test-speculative-sequence-isolation` checks both owners, full row boundaries,
other-sequence KV/PLE preservation, invalid ownership, empty checkpoint rejection,
malformed-payload failure and atomic partial-state rollback. It passes CPU and
all three CUDA devices; strict ASan/UBSan/leak checks and five focused normal
CTests pass. Unsuppressed memcheck with all API errors enabled passes on all
three GPUs. The GPU test explicitly retains and activates a primary CUDA
context before GGML initialization; the first harness attempt's lazy startup
reported three no-current-context driver probes and is retained as failed
validation. No errors are suppressed.

The real artifact passes GPU and CPU replay for zero through four known accepted
tokens. Full sequence snapshots remain byte-identical when the other sequence
advances during verification, and a freshly copied sequence can be checkpointed
before its first decode. These checks compare complete KV/indexer/recurrent/PLE
state, not only logits. They do not prove arbitrary interleaved generation or
long-context sparse-selection correctness.

The reusable lifecycle probe now accepts `ESE_LIFECYCLE_SPEC_MODE=2` for GPU
fallback or `3` for CPU replay (default), and includes the accepted-prefix matrix
and two full-sequence isolation cases. The expanded GPU run passes eighteen
248,320-logit/route comparisons plus the two complete-state cases; its original
33-record teacher stream still matches all 8,194,560 logits and final routes
from graph reuse disabled. Separate CPU matrix/isolation runs also pass. These
are correctness tests with prescribed accepted tokens and **single-token verify
and replay calls**, not speculative acceptance-rate or batched-verifier speed
measurements. Fixed-depth timing, additional expert bytes, sampling qualification
and multi-window banking remain open.


## October 10 bounded checkpoint bank continuation

`results/ese-spec-bank-20261010` retains the failing two-save control and the
qualified replacement. Generic GPU and CPU fallback now keep one snapshot per
configured sequence slot. GPU shadows reserve that many complete rows at startup;
CPU slots reserve their partial serialized state plus bounded KV metadata and
PLE history headroom. Saves reject a payload beyond its startup limit. Slots and
buffers are reused, rather than added as new requests arrive.

The new public `llama_spec_ckpt_discard_seq` and the common speculative helper
remove only their owner's fallback snapshot. Global discard still clears all
windows. Restore checks its saved base position, and prefix removal, copying onto
a destination, position changes, clear/keep and complete state installation
invalidate affected snapshots. Generic per-step journals have no sequence axis;
auto selects the fallback bank for multi-slot contexts, and explicit per-step
selection fails. Architecture-private journals retain their existing single-window
semantics. This is interleaved use by the context's normal scheduler; it does not
make concurrent calls into one context thread-safe.

Both CPU-snapshot and GPU-snapshot runs on the actual coder model pass eighteen
full-logit/route comparisons and three complete-state cases. Two windows at
positions 32 and 33 restore complete sequence bytes exactly (118,926,000 and
118,953,668 bytes); discarding the first preserves the second. Each run's original
33-record stream still matches all 8,194,560 no-reuse teacher logits by bits.
The frozen gated library is
`8e0915e1cdad11fb415225ea51d02b7ac23b66dcb0218cb4e08673414c1eb902`.

For two live windows, GPU shadows reserve 125.39/99.75 MiB on CUDA0/2, while
CPU mode reserves 225.27 MiB across its two slots. The preceding single-window
allocation saving does not apply to this two-window capacity. These are checkpoint
reservations, not total peak memory; serialized restore also has temporary rollback
storage. Full resource accounting and failure-injection stress remain open.

The strengthened unit test covers both owners, unused-row canaries, two simultaneous
saves, reinitialization, common-layer independent discard, invalid base positions,
malformed payloads, payload-cap rejection and prefix/clear invalidation. CPU and
all three CUDA devices pass; strict unsuppressed ASan/UBSan/leak checks pass;
memcheck with all CUDA API errors reported passes all three devices with zero
errors. Five focused checks pass in both normal CUDA and CPU builds with production
opt-in absent; the normal server compiles. The new common helper uses a separate
translation unit because existing common/private headers collide on split metadata
enum names. An initial harness compile failure and the older mock context's missing
sequence-capacity setup are retained alongside the corrected results.

The production gate remains closed. Long-context sparse attention, serving stress,
allocation/cancellation failures and actual batched target verification/sampling
are still broader roadmap work. Single-token accepted-prefix replay is not evidence
of a qualified batched verifier or a speculative throughput gain.


## Actual verifier batch diagnostic (unqualified)

Set `ESE_LIFECYCLE_BATCH_DIAGNOSTIC=1` to compare real API batches of 1/2/4/8
tokens to single-token calls at the same saved prefix. This separate opt-in path
records logits, normalized RMS error, MoE route positions, complete state and
continuation. It returns 3 if qualification fails. It does not run acceptance
sampling or measure accepted-token throughput.

In `results/ese-batch-verifier-20261010`, the depth-1 control matches exactly.
Depths 2/4/8 change logits, routes and state, with worst normalized RMS logit
errors 4.285%/6.157%/6.570%. The same top token in these examples does not establish
sampling equivalence. Whole-call decode policy and one-token hybrid admission
change placement together with batch kernel shapes; a controlled isolation remains
necessary. The one-token microbatch variant aborts on an unsupported CPU REDUCE.
A tested admission-guard trial also failed and was reverted. These failures remain
open correctness/resource work, not a qualified speculation implementation.


## Microbatch policy and scalar-gate kernel continuation

`results/ese-microbatch-scheduler-20261010` records a CPU REDUCE assignment
(destination/source backends 2/2,2) and a debugger-caught SIGABRT in CPU graph
planning. Context and scheduler previously disagreed about policy during
reservation; one-token microbatches could still build a hybrid graph while the
scheduler used prefill. Both sides now change together. Hybrid reservation uses
decode policy, worst-case prefill uses prefill policy, and dynamic reservation
restores the caller's policy. Hybrid admission checks that policy. The actual
microbatch diagnostic now completes depths 1/2/4/8 instead of aborting.

Controlled CPU-expert runs separate the remaining effects. One-token microbatches
match serial CPU-expert decoding exactly at all four depths. True multi-token
batches still differ with CPU experts in both paths. This control isolates a
batch computation issue; it is not a replacement for performant verification.

Bounded activation tracing identifies a shape-dependent BF16 path in the shared
expert gate. Merely requesting F32 picked a different scalar reduction and changed
single-token results, so that trial is retained as unqualified. The final scalar-row
kernel instead performs the existing 256-thread FP32 reduction independently for
each token, preserving padded input/output strides. Qwen shared gates request this
precision, including split gates and LoRA base/factor projections. Other calls
retain their default precision.

The new scalar-row test first fails the prior reduction-order control, then passes
24 cases on each of the three GPUs across F32/F16/BF16 weights, odd/production
widths, depths 1/2/4/8 and padded rows. Unsuppressed all-API memcheck reports zero
errors on each GPU. The existing explicit-projection test passes 1,296 exact cases.
Six focused production-gate-closed checks and strict ASan/UBSan/leak checks pass.
Both final GPU and CPU checkpoint matrices pass all 21 cases and independently
match all 8,194,560 original no-reuse teacher logits exactly. Both drain cache
leases, loads, reservations and staging within the measured RAM/staging caps.
The first-four-layer trace also preserves the uninstrumented nine-record teacher
stream exactly. Layer-0 token-zero output agrees to max absolute 4.47035e-8 and
normalized RMS 1.82301e-7; the next substantial difference appears in layer-1 PLE
gated value (max absolute 4.83925e-5, normalized RMS 0.234564%). Its normalization
amplifies the difference. Key/value projection precision remains a hypothesis;
tracing those inner operations is the next diagnostic. Final states after one
reference token and a two-token batch are excluded from token-zero comparisons.

The combined scalar-row changes preserve both tested nine-record single-token
streams, but genuine batches remain unqualified. Neither the remaining verifier
mismatch nor sampling/acceptance correctness is resolved by the kernel test.
No throughput improvement or production promotion is claimed.

## PLE projection and RMS layout isolation

The next continuation confirms that identical PLE embeddings produce different
BF16 DEFAULT key/value projections across batch shapes. The existing dense-vector
control makes that branch exact. An optional PLE intermediate trace overload
preserves the original result/function ABI; the bounded callback also captures
the consumed embedding and F16 QSA masks as F32 diagnostic values. Optional
`ESE_LIFECYCLE_BATCH_LOGITS_DIRECTORY` saves both full reference and batch logits.

Strided small-row CUDA RMS used a different reduction from contiguous rows. The
repair and its 432 native cases, three all-API memchecks, CPU oracle, normal and
strict sanitizer results are documented in [CUDA RMS layout validation](CUDA_RMS_LAYOUT_VALIDATION.md).
Both 21-case checkpoint runs with the repaired GGML reproduce the original
8,194,560-logit teacher stream exactly and drain caches within their measured
RAM/staging caps. Exact frozen library generations are recorded in the worklog.

Controlled CPU-expert placement plus dense-vector arithmetic now verifies depth2
exactly after prefixes8/32, including terminal sequence bytes and continuation;
depth4 also passes at prefix32. Depth4 at prefix8 and depth8 still fail. A boundary
trace preserves all reference/batch logit bytes relative to the uninstrumented
run. The first token1 discrepancy appears in `fa-3`, with identical visible mask
positions0–9. This shifts the next investigation from a hypothesized QSA tail
error to the actual flash-attention precision/kernel dispatch. No overall
verifier, sampler, speed or production qualification is implied.
