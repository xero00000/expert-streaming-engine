# Experimental direct accepted-prefix recovery

`ESE_QWEN4EXP_PREFIX_JOURNAL=1` enables Qwen's complete-row checkpoint path
when explicitly requested through per-step checkpoint mode. It preallocates
prefix tensors on the same backend as each recurrent layer, complete base
shadows, and base logits. The initial qualified candidate permits one sequence,
one contiguous all-logit verification batch and at most eight input rows.
Seven drafts plus their leading input fit this capacity. Other layouts, MTP
hidden-state contexts and larger capacities decline this experimental path.
The normal Qwen architecture gate remains closed.

Every row contains convolution, SSM and applicable PLE state. Restoring an
accepted row copies the whole cache row, selects its logits and removes the
rejected attention/indexer tail and PLE token history. Zero acceptance restores
the base row and logits. Target sampling remains authoritative; recovery does
not rewind the verified sampler. The complete-state primitive retains the
DeltaNet result so state-copy fusion cannot redirect the final tail away from
the journal. Ordinary calls keep their existing fusion and allocate no journal.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-direct-prefix-20261011`.
All140model cases at prefixes8/32 pass, covering greedy, seeded, adaptive,
grammar and Mirostat2 policies, depths2/4/8 and all rejection/full-accept counts.
Decisions, sampler state, complete sequence bytes, last logits and applicable
32future draws match exactly. Mirostat2 context-RNG future-clone comparison is
omitted; mu and scalar verification are checked. Bounded expert RAM/staging and
final lease/load/reservation drain pass. Separate38cases at depths1/2/4/8
cover every accepted-input count0..depth, poisoned future/EOS tokens, invalid
restore requests, discarded windows and exact continuation/MoE routes. None
replays accepted inputs.

Both ordinary CPU/GPU-fallback lifecycle runs pass21checks apiece. Each stream's
8194560F32 teacher logits, baseline logits and routes remain byte-identical to
the frozen preceding generation. The complete-state copy-consumer regression
passes CPU and all three GPU native/all-API memory checks. Refreshed15kernel/
sequence regressions,14focused normal checks and4ASan/UBSan/leak checks pass.
These model comparisons use controlled CPU experts; they do not establish
production hybrid placement or an upstream full-model oracle.

Initial controls are retained: an incorrect test view stride, a stale private-
ABI test executable, a ccache argument placement mistake, a nine-input common
initialization request and a lifecycle probe missing mode reselection after
discard. Their repaired sources/builds are distinguished from passing evidence.

The measured reservation is1014.08MiB at eight rows for this artifact; the
candidate has a2GiB buffer/logit cap. Graph scratch, transient per-step buffers,
metadata and other caches are separate budgets. Seeded rejection commit phase
medians are0.51ms/0.48ms for prefixes8/32. Forced drafts, scalar-oracle cache
warming and overlapping local checks prevent end-to-end accepted-token or
matched fallback speed claims. Sequence banks, split batches, failure/stress
coverage, resource planning, actual MTP and adaptive-depth performance remain
open. Split-batch work is preserved as a subsequent immutable generation.
