# V4.1 continuation: native implementation and trained validation

September 10, 2026. **Native trained-weight text generation is working.**
The initial runs cover five scoped quality checks. The subsequent speed sweep
adds repaired Q8 dense weights, frequency retention and bounded microbatches.
The latest local profile reaches roughly 2 tokens/s on a short explanation and
1.41–1.48 tokens/s on a separate 96-token prefix; see
[speed4-validation.json](speed4-validation.json) and
[PERFORMANCE_EXPERIMENTS.md](PERFORMANCE_EXPERIMENTS.md). It places Q8 dense
projections on CUDA0 and experts on CUDA2, uses a 512 MiB pinned RAM cache,
and preserves the previously verified Q8 logits. A later experimental greedy
DSpark executable now supports accepted-prefix restoration and target-feature
capture; see [DSPARK.md](DSPARK.md). Early-router I/O forecasting is now
implemented and validated, but remains disabled because its tested policies
did not improve end-to-end speed reliably.
See [RUN_LOCAL.md](RUN_LOCAL.md) and [trained-validation.json](trained-validation.json)
for current commands, measurements and limits. The sections below record the
implementation history; later checkpoints supersede earlier status statements.

## Quant selection

The published smaller artifacts need their complete component inventories,
not just a headline quantization name:

| Candidate, pinned revision | Observed storage | Decision |
| --- | --- | --- |
| [apetersson mixed-Q2 GGUF](https://huggingface.co/apetersson/DeepSeek-V4.1-Flash-MixedQ2-GGUF/tree/af983b61ee7903f414e8152bf1ece8f6ef2a1020) | 168,929,929,664 file bytes, 5/5 shards, 1,038 tensors | Preferred expert/backbone candidate for ESE; Engram absent |
| [Vontra MLX 2-bit](https://huggingface.co/Vontra/DeepSeek-V4.1-Flash-MLX-2bit-MTP/tree/802f1a00982705d81b79ad1c83aa0ccc0b863ebc) | 238,812,072,408 file bytes across 48 shards | Complete smaller alternative, but MLX affine matrices need a separate ESE execution/conversion path |
| Vontra's two Engram embedding tables alone | 61,441,828,000 payload bytes | 80-byte affine-2/group64 rows; native row decoder now implemented |
| [vcruz305 Q2_K](https://huggingface.co/vcruz305/DeepSeek-V4.1-Flash-GGUF/tree/87d2fa308df9cd354f0473ebe5c79e92cd311770) | Only five files of the declared seven-shard set at inspection | Incomplete at the pinned revision; not selected |

Mixed-Q2 GGUF plus only the affine-2 Engram table payloads is
**230,371,757,664 bytes (230.372 GB)** before small sidecar metadata. It fits the
approximately 339 GB available on NVMe with room for scratch and cache files.
It is a proposed mixed recipe, not an already published/validated combined model.
Avoid expanding the Engram tables to BF16 or duplicating complete input/output
checkpoints on NVMe. Source staging for custom quants is authorized on HDD;
`/mnt/games_sdb2` has approximately 2.2 TiB free, versus 901 GiB on the other HDD.

The mixed-Q2 pack's real metadata confirms 80 IQ2_XXS expert tensors, 40 Q2_K,
558 BF16 and 360 F32. Routed payload is **152,882,380,800 bytes**, versus
288,777,830,400 in the source. At batch one with every selection missing,
the routed traffic estimate falls from **4.512 GB/token to 2.389 GB/token**.
This is a traffic estimate; cache hit rates and actual throughput are unmeasured.

`mixedq2-gguf-inventory.json` comes from bounded header reads of every shard,
checking extents, alignment, names and split coverage. Published file bytes
differ from the README's stale table; the inventory uses the API blob sizes.
Both candidate configs have exactly the same `text_config`, and both tokenizer
files have identical SHA-256s. All 48 official source-weight LFS digests match
between the pack's declared source revision `2bc89ac...` and our pinned
`dba1be0...`; see `source-revision-comparison.json`.

**Remaining artifact issues:** the GGUF advertises `deepseek4`, omits several
V4.1 cache-sharing fields, and carries group-router metadata (8 groups/4 selected)
that must not be blindly used for the reference's global top-6 route. Its own
card declares unit importance, no imatrix calibration, and no demonstrated
end-to-end runtime. Use explicit V4.1 metadata/config binding and a dedicated
graph; never load it through the old V4 graph on the strength of that label.
The combined quant recipe needs task-quality evaluation, especially 2-bit Engram.

## Native implementation

`src/llama-deepseek-v41-engram.{h,cpp}` now contains:

- Exact 2/3/4-gram hashing over the exported compressed tokenizer map, two
  layer-specific multiplier sets and 24 prime buckets per layer.
- Image/dead-token boundaries, ordinary pad/EOS treatment, sequence isolation,
  contiguous-position validation, suffix rewind, copy and snapshot/restore.
- Checked table and multiplier bounds, avoiding signed overflow.
- FP8 E4M3FN/E8M0 264-byte row decoding and affine-2/BF16/group64 80-byte
  row decoding, yielding FP32 before the model's BF16 cast.

The hasher retains four bytes of compressed history per position so speculative
rollback can reconstruct the exact prior state. A context owns and serializes
its state; independent contexts do not share mutable history. Persisted snapshots
must be restored under the same model/token-map identity.

`row_store` in `llama-qwen4exp-ple-storage.*` gained a model-independent
`fixed_row_layout` constructor. It reuses the existing bounded block cache,
asynchronous reads, duplicate coalescing, leases and owner cancellation.
Each Engram layer gets a separate store/key space. Raw batches expose bytes
for the native Engram decoder; the old Qwen IQ4_NL dequantization convenience
method remains Qwen-specific. Existing Qwen constructor behavior is preserved.
The code remains in the historical Qwen storage namespace to avoid an unrelated
rename of active local work; no Qwen hashing is used by this path.

`engram-hash-reference.json` and `engram-token-map.u32le` are generated by the
pinned official `inference/engram.py`, not a hand-transcribed approximation.
The normalized vocab size is exactly 99,092 from 129,280 original token IDs.
Five reference sequences cover 173 positions (8,304 row IDs), including Unicode,
whitespace, special tokens, image boundaries and seeded random tokens. Official
chunk sizes 1/3/7 match full prefill. Native tests replay every suffix, interleave
two sequences, fork, restore and check rejected operations preserve history.

The affine-2 layout was checked against MLX's
[CPU quantized implementation](https://github.com/ml-explore/mlx/blob/81ba1c6a0e50a9268b931579c2d4f1158b9aab5a/mlx/backend/cpu/quantized.cpp).
The two production row fixtures are also checked via independent uint32
unpacking and Torch arithmetic. This is byte/decoder validation, not proof that
the lossy table retains the original model's quality.

## Validation completed

- **18 Python tests pass**, including all FP8 weight/scale encodings, real
  original/affine row decoding, hashes and metadata-parser failures.
- **Five native tests pass** in CPU Debug, ASAN/UBSAN with leak detection and
  halt-on-UB, and CUDA Release: new Engram plus existing Qwen row/page/cache/I/O.
- New raw-row tests cover 80/264-byte rows, unaligned page-crossing extents,
  duplicates, cache hits, pinned-capacity failure, eviction, owner cancellation,
  invalid rows, arithmetic bounds and invalid layouts.
- **Sixteen real expert-kernel cases pass**: IQ2_XXS and Q2_K, batches 1 and 8,
  CPU plus CUDA devices 0/1/2. No CPU fallback is permitted in GPU mode.

The expert test executes `ggml_mul_mat_id` with real layer-0 expert-0 data
(6,912,000 bytes total), compares against row dequantization plus a scalar
double-accumulation dot product, and requires relative RMS <=1%, cosine >=0.999.
Observed relative RMS is 0.405–0.796% on the GPUs and 0.427–0.435% on CPU.
Minimum observed cosine is about 0.999968. This isolates kernel accuracy on the
same quantized weights; it does not measure quantization loss versus the original
weights, routed multi-expert accumulation, the complete graph or throughput.
Results are in `expert-kernel-results.json`; payload hashes are in
`expert-fixture-manifest.json`. Native model support is not inferred from them.

## KV footprint

The card's optimized global KV figure is **890 bytes/token**:

- 1,000,000 tokens: 890,000,000 bytes = 890 MB, about 849 MiB.
- Configured 1,048,576 tokens: 933,232,640 bytes = 933 MB, exactly 890 MiB.

That requires the actual CSA2 sharing and FP4 encoding. It excludes SWA state,
working buffers, and weights, and it is not the allocation size of the current
unimplemented graph. Begin numerical graph checks at short context with a
simple higher-precision cache, then implement/test compression and shared cache
lifetimes before promoting these memory figures to the runtime planner.

## Next implementation boundary

Bind V4.1 metadata and the Engram map/sidecar identity explicitly; implement the
dedicated graph with shifted mHC, shared KV/indexers and Engram residual gating.
Validate on a small initialized complete reference model before loading the full
quant. Whole-checkpoint downloads are not needed for the completed tests and
have not started. HDD source staging is available if custom quantization or
broader original-weight comparison becomes necessary.

To reproduce from the physical ESE worktree:

```bash
python3 -m unittest discover -s tests -p 'test_deepseek_v41*.py' -v
cmake --build build_cpu_avx2_debug --target test-deepseek-v41-engram -j4
ctest --test-dir build_cpu_avx2_debug -R '^test-deepseek-v41-engram$' --output-on-failure
python3 tools/deepseek_v41_hash_reference.py \
  /home/xero/Desktop/test/results/deepseek-v41-20260910/reference \
  --output-dir docs/deepseek-v41
python3 tools/deepseek_v41_expert_fixture.py docs/deepseek-v41/mixedq2-gguf-inventory.json \
  --output-dir /home/xero/Desktop/test/results/deepseek-v41-20260910/expert-fixtures
cmake --build build_cuda_release --target test-deepseek-v41-expert -j4
build_cuda_release/bin/test-deepseek-v41-expert \
  /home/xero/Desktop/test/results/deepseek-v41-20260910/expert-fixtures 0
```

Use `cpu`, `1`, or `2` in place of `0` for the other tested backends.

### Routed-expert production arithmetic

The `LLM_ARCH_DEEPSEEK41` branch in `llm_build_moe_ffn` now preserves FP8 input
QAT, BF16 gate/up projection rounding, asymmetric SwiGLU clamps, routing-weight
multiplication before down-input rounding/QAT, and BF16 down-output rounding
before FP32 expert summation. Routing normalization includes the source's
`1e-20` denominator term. Other architectures retain their existing paths.

`tools/deepseek_v41_moe_reference.py` extracts the unchanged, hash-verified
released `Gate` and `Expert` classes and substitutes portable FP8 QAT linears.
`tests/test-deepseek-v41-moe.cpp` exercises the real graph builder with the
resulting initialized fixture, checking selected IDs, routing weights, and
outputs for two single-token cases and a five-token case. The high-amplitude
case exercises clamps. All three pass CPU at the fixed 1e-3 absolute plus
1e-3 relative tolerance (maximum absolute output difference 0.00848388672).
The test also passes ASAN/UBSAN with leak detection; all 12 registered V4.1 CPU
tests pass. This new MoE fixture has not yet been tested on CUDA or with ESE
hybrid/cache partitioning. It uses FP32 containers holding BF16 expert weights;
actual mixed-Q2 kernel tests remain separate.

Still required: shared expert, complete outer block/model graph and runtime
state lifecycle, initialized whole-model parity, tokenizer-aware canonical
model packaging, trained-weight generation, and hardware tuning. The forward
architecture guard remains until full graph integration is ready.

### Shared expert and production Engram update

`build_shared_expert` now implements the shared FP8-activation SwiGLU with
BF16 projection and intermediate rounding. The MoE oracle includes this expert
and the final BF16 cast after adding the routed FP32 sum. It also extracts the
unchanged released Engram forward method and tests one/five-token cases with
production embedding QAT, projection rounding, and residual-output rounding.
`build_engram_injection(..., production_precision=true)` performs these steps;
the default remains the existing FP32 primitive contract.

The expanded `test-deepseek-v41-moe` runs directly on CPU or a specified CUDA
backend, rejects unsupported operations, and passes on CPU and all three local
GPUs. Maximum error across the expanded fixture is 1.1920929e-7 CPU and
1.49011612e-7 GPU. The expanded fixture supersedes the earlier seeded fixture
because shared-expert initialization changes RNG consumption. ASAN/UBSAN with
leak detection also passes. This does not establish full-model or trained-weight
quality, nor ESE hybrid/cache correctness for the new arithmetic.

The candidate downloader completed successfully: the five mixed-Q2 backbone
shards and both affine2 Engram shards are downloaded and checksum-verified on
HDD. Canonical packaging and NVMe staging have not yet been performed.

### Full transformer block and initialized token-to-logit graph

`attention_forward_state::build_block` now joins production Engram, shifted
HC mixing, norms, attention, routed and shared experts, and final residual
rounding. `build_token` adds token embedding, initial stream replication and
one-hot pre-mix, all blocks, final collapse/norm, and the FP32 vocabulary head.
The caller still owns beginning/committing/aborting each token and supplying
validated decoded Engram rows. These methods are not yet attached to ordinary
`llama_context` decode; the architecture guard remains.

The full-block option of `deepseek_v41_forward_reference.py` extracts released
Block/Gate/Expert/MoE methods alongside attention, and creates an eight-layer
initialized token-to-logit fixture with ten positions. The test supports both
separate block graphs carrying native outputs forward, and one complete graph
per token (`--whole-model`). Both include complete abort/replay and injected
position-error replay. Real BF16 weight buffers are used. Whole-token CPU and
all three GPU outputs agree within 2.384185791015625e-7 maximum absolute logit
error; all 1,573 matrix operations per GPU run stay on that GPU. ASAN/UBSAN
with leak detection passes. All 14 registered V4.1 CPU tests pass.

Two integration findings matter:

* Index score ties have unspecified Torch top-k IDs. At position 7 the scores
  for rows 2 and 6 are both zero. Torch CPU chose 6, while Torch CUDA on all
  three GPUs and the native deterministic selector chose 2. The full-block
  oracle now specifies ascending-position ties for index/candidate tensors
  and records every changed selection in `index_tie_audit`. Selected score
  values must remain identical; tolerances have not been relaxed. Source
  methods are unchanged, with portable top-k policy supplied externally.
* The generic HC_POST kernel accumulates the sublayer contribution before
  the residual terms, unlike the released expression. Clang sanitizer and
  GCC builds differed at BF16 rounding boundaries. The V4.1 helper now sums
  separately materialized residual contributions first and adds the sublayer
  contribution last. This fixes the full-token sanitizer mismatch without
  changing the shared generic HC_POST kernel.

`token-results.json` records results and limitations. The oracle has a reduced
Engram lookup width and supplied initialized rows, four experts, eight layers,
and a 128-token vocabulary. It validates graph arithmetic and state ordering;
it does not validate the external model package, row-store integration, trained
weights, generation quality, or throughput. Next: attach this graph to native
runtime state/hash/row-store lifecycle, package the verified downloaded shards
with tokenizer and Engram metadata, then run actual generation and tuning.

### Transactional Engram/runtime state

`llama-deepseek-v41-runtime.{h,cpp}` now owns the hash history, both bounded
Engram row stores, decoded token inputs, and attention forward state for one
append-only text sequence. Begin submits both sets of 24 row requests, decodes
all rows in hash-column order, and returns only when they are ready. Failed
reads and failed graph executions can abort both hash and attention state.
Clear resets compressor tails, cache positions, and hash history while retaining
allocated cache capacity. A discovered empty-commit bug was fixed: committing
attention now requires that the token's position was actually published.

`load_runtime_assets` accepts version-1 embedded JSON metadata containing the
compressed token map, hash parameters, and two model-relative raw row files.
It validates layer/row geometry, encoding, file extent and declared size, and
opens the existing bounded file-source backend. It supports affine2 and FP8 row
formats. Full asset checksums remain a packaging/staging responsibility; the
loader uses the file source's bounded identity checks and validates extents.
The canonical package has not yet been written.

`test-deepseek-v41-runtime` covers actual temporary files and metadata rejection,
injected source failures, all 24 decoded rows per layer, rejecting an unexecuted
graph commit, full-width (6144-value) Engram inputs to the complete token graph,
complete abort replay, graph-error rollback, sequence binding, active clear,
and identical logits after clear/replay. It uses initialized weights and a
small synthetic row source, not trained-weight quality comparisons. CPU and
ASAN/UBSAN with leak detection pass; all 15 registered V4.1 CPU tests pass.

Still required: load the embedded asset metadata into the model, initialize
runtime state with context parameters, wire `llama_decode` begin/input/commit/
abort and native KV lifecycle hooks, generate the canonical package on NVMe,
and test actual model generation and hardware tuning. No architecture guard
has been removed and normal V4.1 decoding is still unavailable.

### Public model loading and native decoding

The V4.1 architecture now dispatches to the complete token graph. Model loading
retains embedded `deepseek41.assets` JSON and the model directory; context
creation opens those assets and initializes the runtime. Prompt batches execute
as one-token microbatches, with graph reuse disabled because cache/compressor
shapes depend on position. Reservation builds at the configured final position
without executing or publishing state.

`llama_decode` begins the runtime token before reserving its ordinary KV cell,
sets decoded Engram inputs, checks graph submission status, synchronizes, then
commits. A scope guard aborts private state and removes the just-reserved cell
on failure. The public clear and full-sequence deletion APIs reset private
state too. Historical suffix removal returns false; copying, shifting, position
scaling, external KV transactions, speculative checkpoints, embedding mode and
session serialization are not yet supported. Context creation admits one text
sequence; use `--no-context-shift` for actual CLI runs until replay/rewind support
is implemented. Unsupported operations must not silently rewrite private state.

`llama_hparams::has_kv` incorrectly interpreted an explicit zero KV-layer count
as all layers. Its condition now honors the documented nonnegative semantics:
-1 means all layers, 0 means none. The native test asserts that V4.1 has no
ordinary K/V tensors. This prevents a second, dense cache from defeating its
packed-cache memory budget.

The expanded runtime test writes an initialized GGUF with metadata, tokenizer,
all model tensors, and file-backed full-width Engram rows. It loads through
`llama_model_load_from_file`, creates a normal context and calls `llama_decode`
with a ten-token prompt, requesting every token's logits. Public decoding
matches the explicit runtime graph exactly on the same backend on CPU and all
three GPUs. Cancellation/retry, full deletion/clear/replay, and rejecting
unsupported suffix/session/speculative operations pass. ASAN/UBSAN with leak
detection passes. All 15 V4.1 CPU tests plus expert-cache regression pass;
CUDA expert-cache regression also passes. CPU/CUDA CLI targets are rebuilt.

The synthetic runtime row fixture showed CPU-versus-GPU logit differences
(up to 0.0597856641 in the initial cross-backend comparison); this test now
compares public and direct execution on the same backend to isolate runtime
integration. It is not a cross-backend numerical oracle. The separate BF16
initialized token-reference test remains the numerical reference check; trained
model quality and numerical sensitivity remain to be assessed on real weights.

Next: create tokenizer-correct canonical shards and raw Engram sidecars from the
verified downloads, stage them on NVMe, then test real generation and tune ESE
residency/cache/offload settings. The original end goal is not yet complete.

### Real package staging started

The released tokenizer exports using `deepseek-v3` preprocessing. Native tests
match all 1,498 cases, including every added token, embedded special tokens,
Unicode/code examples and deterministic randomized text. The test also passes
against tokenizer metadata extracted from the first staged native shard.
`deepseek_v41_tokenizer.py` generates the vocabulary GGUF and reference cases;
`test-deepseek-v41-tokenizer` is an optional native parity runner.

`deepseek_v41_package.py` stages five canonical backbone shards with unchanged
tensor bodies, two raw 80-byte affine2 Engram row files, and a sixth GGUF for
Engram projections/norms. It checks source backbone SHA256 values and rereads
destination tensor bodies before publication. Row packing copies the three
original planes without requantization and checks deterministic source rows.
The three packaging tests cover tensor-body preservation, byte-exact row
packing, and FP8-to-BF16 expansion against Torch.

Original FP8 Engram projections and original BF16 norms were fetched as bounded
ranges from the pinned official shards (about 315 MB). Their manifest is under
`results/deepseek-v41-20260910/engram-projections-fp8`. This avoids using the
MLX package's 2-bit projections; only its large row tables remain affine2.
Eight sampled real 32-row projection blocks also match Torch's BF16 expansion
exactly. The sixth GGUF will contain about 629 MB of expanded projections.

Destination: `/home/xero/Desktop/test/models/deepseek-v41-native-mixedq2`.
The live stager writes `package.json` there and logs to
`/home/xero/Desktop/test/results/deepseek-v41-20260910/package-stage.log`.
At this checkpoint the first backbone shard is fully copied/verified and the
second is in progress. Do not load the package until its manifest is complete.
Staging retains resumable partial files; check the live process before restarting.

Actual backbone tensor sizes: routed experts 142.383 GiB, other dense tensors
12.480 GiB, output head 1.233 GiB, token embedding 1.233 GiB. CPU-expert placement
needed a fix: V4.1's unsuffixed names did not match the built-in `ncmoe` regexes.
V4.1 now accepts bare names and weight/scale suffixes. The native GPU load test
asserts all routed experts stay on host while shared experts remain on GPU.
All 15 V4.1 CPU regressions pass after this change.

An initial real-model trial should use bounded expert RAM/VRAM caches, deferred
expert residency and no whole-layer prefill staging. Suggested starting point
(to be measured, not a verified performance configuration): 41 GPU layers,
layer split 10,10,6 to leave room for the output head on the last GPU, 8 GiB RAM
expert cache, 1 GiB VRAM expert cache per GPU, 768 MiB reserve per GPU, context
1024 and one-token microbatches. Use `--no-warmup --no-context-shift` and explicit
raw prompt formatting because the released tokenizer config has no chat template.
Trained-weight generation and tuning have not yet been run.

### First real load and padding correction

The initial stage completed all eight files (231,007,190,336 bytes), and the
full six-shard package passed the 1,498-case native tokenizer test. The first
real CLI run (`native-smoke-01`) then failed before model allocation because
an expert descriptor's source offset violated its declared alignment.

Inspection found that the published backbone GGUFs advertise 32-byte alignment
but contain tensor offsets aligned only to four bytes. The initial byte-for-byte
body copy preserved that defect. The packager now preserves individual tensor
payloads while inserting proper padding and rewriting offsets. The expert
alignment validator has not been relaxed.

`deepseek_v41_realign.py` is correcting the already staged NVMe copies, one
shard at a time, with a recovery journal. It verifies the old file checksum and
the unchanged tensor payloads, then atomically replaces each native shard and
updates the package manifest. Original HDD downloads are untouched. The package
is marked `realigning` until all five backbones are repaired; the first CLI
retry must wait for `complete`. A fourth packaging test covers this malformed
alignment case and resumable copying; all four packaging tests pass.

The smoke runner uses the pinned official chat encoder, not a guessed template.
It records the command, prompt, output/error logs and memory measurements, and
uses a 32 GiB observed-RSS stop threshold for the initial check. The next attempt
should use a new output directory (`native-smoke-02`) to preserve the first
failure log. Generation has not succeeded yet.

### Real generation and dense-weight conversion defect

Alignment repair completed: 1,044 tensors pass offset/bounds checks, package size
231,007,191,936 bytes. `native-smoke-02` exposed the CPU-placement override's
pinned allocation behavior; the runner now uses `GGML_CUDA_NO_MODEL_PINNED=1`,
matching ESE's existing launcher, so deferred model mappings remain enabled.

`native-smoke-03` successfully loaded 15.53 GiB of dense tensors, deferred
142.38 GiB of experts, and generated 32 tokens. Its output was incoherent;
0.42 tokens/s is a diagnostic measurement, not a usable-model benchmark.
Wall time 128.03 s; peak RSS 24,123,744,256 bytes (22.47 GiB). A second run with
zero GPU weight layers and zero VRAM expert cache also produced an incoherent
first token (`native-smoke-04-cpu`, 82.03 s, peak RSS 27,837,480,960 bytes).
That CUDA binary can still offload operations, so this is not a pure-CPU oracle.

`deepseek_v41_dense_audit.py` compares sampled trained tensors with bounded
ranges from the pinned original checkpoint, checking each source shard header's
SHA256. Embedding/head/norm, router and HC samples match exactly. All five
layer-0 FP8-derived attention projections fail: for example native wq_a begins
1.4210855e-14 where the correctly expanded original is 0.03515625. Sample
relative RMS error is approximately 1.0, far beyond BF16 rounding. The weight
artifact's claimed dense FP8 conversion is not correct. The original tensor
payload-preservation checks faithfully preserved this upstream defect.

Next: fetch the original 328 non-Engram FP8 dense tensors and scales (~6.53 GB),
expand with the tested 32x32 E8M0 decoder, patch the native package with a recovery
journal and verified payloads, then rehash affected shards and rerun the audit
and generation. Routed expert and affine2 Engram quantization quality remain
unverified; this discovery does not establish that every remaining weight is good.

Dense correction is now complete. `deepseek_v41_dense_repair.py` fetched
6,528,512,000 bytes of original FP8 weights/scales in 160 bounded ranges, expanded
all 328 affected tensors (13,044,285,440 BF16 bytes), verified every replacement,
and rehashed all five backbone shards. `dense-repair.json` journals completed
writes and retains the pre-repair manifest records. The native package manifest
includes the correction provenance; the smoke runner now refuses this artifact
without it. All five packaging/repair tests pass. The repeated dense sample audit
matches the original exactly (`dense-audit-corrected.json`).

A separate original-MXFP4 comparison (`expert-original-fidelity.json`) covers
three 32-row samples of each layer-0 expert-0 projection. IQ2_XXS gate/up cosine
0.94029–0.94085, relative RMS 0.35217–0.35368; Q2_K down cosine
0.96305–0.96329, relative RMS 0.27023–0.27105. This is ordinary quantization loss
and agrees with the artifact's reported range, unlike the dense conversion bug.
It does not establish full-model quality. `native-smoke-05-repaired` is the first
trained generation run against corrected dense weights.

### Trained text validation and initial cache tuning completed

The corrected model now answers `4` to the arithmetic prompt and stops normally.
Seven subsequent successful runs cover five distinct checks: arithmetic, a
coherent two-sentence explanation of the blue sky, recall of `COBALT` after
168 prompt tokens (beyond the 128-token sliding window), a Python `square(n)`
function verified on -3/0/4, and the factual answer `Paris`.

The explanation's output is byte-identical with 8/12/16 GiB expert RAM and
1/1.25/1.25 GiB VRAM per GPU. Decode rates are 0.709/0.735/0.735 tokens/s;
peak host RSS is 22.44/25.62/26.78 GiB. The larger caches reduce file reads but
show little overall speed gain. These were sequential probes with uncontrolled
OS page cache, not a controlled benchmark. Retain 8 GiB RAM / 1 GiB per GPU
as the conservative starting configuration. Long repeated-text recall prefill
was 1.31 tokens/s; do not generalize that rate to arbitrary prompts.

`trained-validation.json` contains commands' derived settings, outputs, timing,
memory and cache statistics. Full commands and prompt encodings remain in each
run directory. `RUN_LOCAL.md` provides the exact invocation. This completes the
initial native text enablement and measured tuning milestone. Larger trained
contexts, optimized batched prefill, DSpark, vision and chat serving are still
future work; nothing here establishes million-token serving performance.
