# Qwen metadata rejection and state geometry

Malformed expert settings previously reached assertions in the shared
hyperparameter loader. A frozen old-library control with four experts and
five active experts aborts. Zero recurrent-group or PLE convolution dimensions
were accepted by metadata loading and could reach invalid divisions or state
geometry later. The same final fixture executable now rejects all three
inputs through ordinary exceptions, without changing the production gate.

The shared loader validates the non-vocabulary-only layer count before using
per-layer accessors and reports invalid expert counts instead of asserting.
The Qwen branch checks positive text/FFN/recurrent dimensions, divisible
recurrent head grouping, the equal key/value head widths used by its current
square recurrent-state representation, and sparse-attention geometry for the
layers that use it. PLE convolution and hyper-connection geometry are checked
before computing state widths. Products and combined recurrent/PLE rows must
fit the existing UINT32 width API; this is a representation limit, not the
device-memory budget.

Evidence is under
`/home/xero/Desktop/test/results/ese-qwen-loader-geometry-20261011`:

- The fixture's five existing array/range/PLE cases and 33 new scalar metadata
  cases pass without an abort in normal and strict ASan/UBSan/leak builds.
  Scalar cases include zero and excessive counts, invalid head grouping,
  mismatched widths, zero sparse/indexer dimensions, and state-width overflow.
- Six normal and six strict sanitizer regressions pass, including split-file
  construction cleanup, unsupported GGUF formats, output-ownership sessions,
  the pinned schema and production architecture rejection. The final fixture
  source additionally passes both builds after adding metadata-only artifact
  inspection modes.
- Paired final executable controls retain the original expert-count abort
  and accepted zero-group/zero-PLE-kernel failures. All three candidate
  executions return successful rejection.
- Read-only real Qwen metadata still loads: 48 layers, 256 experts/top 10,
  recurrent row width 817,152 values. Dense stories15M and conventional
  TinyMoE metadata also load, with zero experts and eight experts/top two,
  respectively. These are metadata checks and do not execute model inference.

An intermediate test exposed another existing abort: `block_count=0` reached
the generic attention-head accessor before the Qwen branch could reject it.
The final shared validation now rejects it before that access. Failed attempts
are preserved separately from passing evidence. Vocabulary-only loading keeps
its earlier early-return behavior.

## RMS epsilon and PLE history contracts

The 2026-10-12 follow-up rejects nonfinite or nonpositive RMS epsilon before
graph construction. NaN, positive/negative infinity, zero and negative epsilon
previously passed metadata loading. It also rejects PLE convolution kernels
smaller than two: kernel one produces an empty history, while the current PLE
state views and accepted-prefix checkpoints require a nonempty history.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-loader-contracts-20261012`.
Six paired controls use the same final fixture executable with the old frozen
libraries and the current libraries. Each old-library control demonstrates
accepted invalid metadata; every current-library control rejects it normally.
Two positive controls retain finite epsilon and the smallest supported history.
The fixture covers 44 malformed metadata cases in total.

Six focused Release tests and six strict CPU ASan/UBSan/leak tests pass. Original
Qwen metadata, dense stories15M metadata and conventional TinyMoE metadata load
in both builds. These are metadata checks; they do not execute inference.

Complete tensor-family/shape failure coverage, full graph/upstream reference
parity, aggregate resource budgets, long-context and serving are unqualified.
Independent GPU recovery checks use older frozen runtime generations. This
follow-up changes metadata admission only; arithmetic kernels, model files,
launcher configuration and the production architecture gate are unchanged.

## PLE head-count multiplication

The head-table count check now widens `heads_per_ngram` to `uint64_t` before
multiplying by `ngram_size - 1`. Previously, `heads_per_ngram=2,147,483,656`
with n-gram size three wrapped the 32-bit product to 16 and incorrectly matched
the fixture's 16 head offsets. The widened product is 4,294,967,312, so the same
metadata is rejected normally.

The permanent fixture adds this case, bringing its malformed metadata total
to 45. Normal Release and strict CPU ASan/UBSan/leak builds pass four focused
CTest executions and 18 targeted metadata controls. Ordinary malformed counts,
valid original Qwen metadata and its 1,224 tensors, and dense/MoE metadata keep
byte-identical output against the archived controls. The old acceptance is
preserved as failing evidence. The normal fixture still confirms the closed
production architecture gate.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-head-count-integration-20261013`.
The integration contains only the multiplication widening and one fixture
entry. These checks qualify metadata admission; they do not requalify model
inference or the longer-prompt numerical gate.

## Original artifact tensor inventory and fault controls

`test-qwen4exp-fixture --artifact-tensors MODEL.gguf` drives the actual tensor
creation path after metadata admission. The original coder artifact accounts
for all 1,224 tensors across 48 layers in both Release and CPU sanitizer builds.
No weight buffers are allocated and no weight payloads are loaded.

`--artifact-tensor-faults MODEL.gguf` additionally changes a separate loader's
metadata in memory before admission and tensor creation. Release and strict CPU
ASan/UBSan/leak controls both reject 239 cases, with byte-identical control output:
one missing tensor and four dimension faults for each of 47
distinct tensor families, plus four surplus-family cases. These include QSA
tensors on a recurrent layer, recurrent tensors on a QSA layer, PLE tensors on
a layer without PLE, and an unused tensor. The original files remain read-only.
This exercises tensor construction; it does not generate malformed GGUF files
or run inference. File parsing and split-file rejection have separate evidence.

The first fault harness incorrectly treated an extra PLE table row as an
invalid shape. The corrected control makes the table too small to cover its
head ranges and injects faults before metadata admission. The failed attempt
and a test compile failure are retained and excluded from passing results.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-tensor-families-20261012`.
Six focused regressions also pass in each final build. Full-model numerical
parity and complete loader metadata coverage remain unqualified.

## Serialized tensor faults

The same tensor contracts now have serialized GGUF coverage. Sparse metadata
fixtures contain the original shard headers and hole-only bodies; they are
never used for inference. Unchanged headers create all 1,224 tensors in Release
and strict CPU ASan/UBSan/leak builds, with zero weight buffers. The restored
positive controls reproduce the initial control output byte-for-byte.

Each build rejects 239 serialized cases through the intended tensor contract:
47 missing tensors, 188 dimension faults and four surplus tensors, covering
the same 47 families. Quantized row-width changes preserve block alignment;
split inventory counts are adjusted for missing and extra tensors. Every
declared tensor extent fits the sparse file, so parser errors or truncated
payloads are not counted as shape-validation evidence. Implicit trailing
dimensions are explicitly serialized when testing those dimensions.

No sanitizer errors, assertions or payload-bounds failures occur in the final
matrix. Original weight files remain read-only and their payloads are not
copied or loaded. The first fixture generator stopped before execution when
removing the only tensor in a shard; that attempt is retained and excluded.
The corrected matrix reruns every case in both builds.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-serialized-tensors-20261012`.
These checks qualify tensor construction from serialized headers, not complete
metadata validation, weight loading or inference. The production gate remains
closed.
