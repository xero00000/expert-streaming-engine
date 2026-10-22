# Sequence-owned Qwen attention: integration qualification in progress

October 11, 2026. The experimental model graph now orders attention and QSA
cache reads by each sequence's retained token positions. Production Qwen
loading remains gated. This is a correctness change; no throughput or complete
upstream-model parity result is claimed.

Previously, placing identical logical keys/values in different physical KV
slots changed flash-attention reduction arithmetic. A two-sequence prefix-bank
test failed when four target inputs were accepted: the other sequence's saved
state was unchanged, but its next logits and expert routes differed. The
captured operator regression and CUDA gather repair are documented in
`QWEN4EXP_ATTENTION_CACHE_ORDER_VALIDATION.md`.

The graph now uses sequence-owned row maps, position-sorted keys and causal
masks. Its padded attention length depends on that sequence's retained key
count. Unrelated sequences' physical cache extent cannot enlarge that window.
Masked padding repeats a valid owned row instead of reading unwritten cache
storage. Contiguous query groups have separate attention outputs, combined in
original query order before the output projection. Existing hybrid batch
splitting rules are unchanged.

QSA indexer keys use the same ordered reads. Pooling, block-to-token maps and
tail selection operate in logical row order; block RoPE uses the retained
token position at each block's start. The graph reuse guard checks sequence,
query grouping and padded key length before reusing these inputs. Runtime
input filling refreshes physical row IDs, positions, masks and tails.

A host regression covers reversed physical placement, different sequence
lengths, unrelated growth, copied shared cells, removed positions, implicit
sequence IDs, FA/non-FA padding, reservation shapes and duplicate-position
rejection. Its expanded version also checks causal and padded masks, QSA block
positions/maps, per-head row IDs and finite owned padding. It caught an edge
case during integration: unused tail slots must repeat a valid tail row, not
logical key zero. That correction has its own frozen v2 generation.

Evidence is under
`/home/xero/Desktop/test/results/ese-qwen-sequence-attention-20261011`:

- First frozen generation: the original target0/accepted4 failure passes;
  both continuation streams contain 248,320 byte-identical logits and exact
  MoE routes. All 30 bank cases pass across chunk sizes 1/2 at prefix8 and
  chunk size4 at prefix32, covering both target sequences and accepted counts
  0..4. All 19 single-sequence accepted-prefix cases also pass.
- Both ordinary checkpoint-mode controls pass their 18 JSON and three plain
  lifecycle checks. Teacher streams contain 8,194,560 F32 values each and are
  byte-identical to the preceding selected-backend generation, as are the
  baseline logits and routes. RAM/staging caps and final drain are audited.
  These checks qualify that first generation, before the partial-tail fix.
- Refined v2 generation: the expanded host input test passes normally and
  under strict ASan/UBSan/leak detection. Four focused strict sanitizer tests
  pass, as do ten final focused normal CTests. All 30 bank cases at chunk
  sizes1/2 and prefix8 plus chunk size4 at prefix32 pass with exact
  complete states, logits, MoE routes **and logical QSA block IDs**; RAM and
  staging caps/drain pass. That generation's isolated all-API model memcheck
  is terminal exit99 with twelve CUDA graph-update errors, although numerical
  checks pass; its ordinary controls were not launched. The subsequent graph
  cache repair passes the same focused model memcheck with zero errors and
  all30native bank cases. Both subsequent ordinary controls are terminal0 with
  byte-identical teachers; qualification is recorded separately in
  `CUDA_GRAPH_UPDATE_VALIDATION.md`.

The broader first-generation normal suite passed 14 of 16 tests. Two existing
DeepSeek fixtures failed: whole-model logits at position4/element0 and a block
at position4/layer1/element70 (-1.164062 versus -1.171875). A separate probe
compiled against the preceding private context header and loaded with the
preceding immutable libraries reproduces both failures exactly. These remain
open shared-path findings; the suite must not be reported as wholly green.
The first-generation five focused strict sanitizer tests pass.

Bounds still need broader qualification: multi-sequence mixed decode calls,
sequence shifts, selection-limited long-context QSA, padding-length boundaries
in batched verification, cancellation/failure recovery and complete peak
memory. The gather currently supports F16/F32 cache storage; other cache
formats and the tensor-split attention branch remain outside this integration
qualification. Bounded sparse KV tiers should avoid gathering a full 64K
window per sequence. Component tensor sizes are recorded separately; they are
not a whole-process memory guarantee.

Observe the existing runner and its actual child PID before launching any
additional model job. The refined queue is
`v2/run-model-qualification.py`, with `v2/active-model-job.json` and terminal
per-run exit records. A timeout is not a failed or stopped job. No launcher,
model artifact or published release was changed.
