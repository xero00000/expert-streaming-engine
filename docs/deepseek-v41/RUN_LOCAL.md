# Running DeepSeek V4.1 Flash in the local ESE build

The native text path now runs trained weights on this machine's RTX 3060 Ti,
RTX 2080 SUPER and RTX 3080. The working checkout is
`/home/xero/Desktop/test/ESE-v0.2.1-qwen4exp-plan` (`ESE-current` points here).

The verified native package is
`/home/xero/Desktop/test/models/deepseek-v41-native-mixedq2`: six GGUF shards
plus two affine2 Engram row files, totaling 231,007,191,936 bytes. Its manifest
records complete alignment and dense-weight repairs. The original MixedQ2
download has malformed tensor alignment and incorrect FP8-to-BF16 dense
conversion; use the repaired native package.

## Desktop and terminal launcher option

Both local launchers now offer **DeepSeek V4.1 Flash — 1M text chat, KV on
RTX 3080**. Reopen the terminal launcher or ESE Studio to load the new entry.
In the terminal model picker, search for `deepseek`; its slot key is
`deepseek-v41-flash`. ESE Studio shows the explicitly named Q8-dense profile.

This launcher profile allocates the declared **1,048,576-token context** for
one sequence. `ESE_V41_CACHE_DEVICE=CUDA2` stores the packed sliding windows,
compressed main KV and index-key banks on the RTX 3080, with no permanent CPU
copy of those banks. Cache reads still use temporary host buffers. Their full
capacity allocation is **935,936,000 bytes (892.58 MiB)**; workspaces, weights,
rollback windows and the expert cache require additional memory.

For the larger context, indexer weights and indexing work remain on CUDA2.
The expert cache reserves 1,536 MiB of GPU headroom and clamps its payload to
available VRAM. This avoids the CUDA0 workspace exhaustion encountered when
using the earlier short-context placement at the maximum setting. The small
512 MiB pinned expert RAM tier and eight loading workers are retained.

The shared configuration lives in `/home/xero/Desktop/test/model_slots.sh`.
Studio uses `/home/xero/Desktop/test/llama_install/scripts/ese_studio_profiles.py`
to resolve that same slot; other models continue through its installed ESE
launcher. The Studio model entry is in `/home/xero/.config/ese/studio.toml`.

Validation: full-capacity GPU allocation and byte I/O, ring wrap and rollback,
three chat requests including a follow-up, streaming completion, and exact
agreement of every logit at 54 teacher-forced positions. All 15 native tests
passed on the CUDA build and under ASAN/UBSAN. **A filled million-token prompt
has not been evaluated**; the capacity and short-request checks do not establish
full-length quality or throughput. Results are recorded in
`/home/xero/Desktop/test/results/deepseek-v41-20260910/launcher-validation.json`.

This option provides non-thinking text chat and V4.1 DSML tool calling.
The chat UI may attach tools even to a greeting; the template now accepts
these requests, and the server parses V4.1's leading-space DSML tag names.
Vision, multiple concurrent sequences, context shifting and saved sessions are not configured.
Its chat template matches the pinned V4.1 reference encoder in the checked
single-turn, system-message and multi-turn cases.
Tool parser and live API checks are recorded in
`/home/xero/Desktop/test/results/deepseek-v41-20260910/launcher-tools-api-check.json`
and `launcher-tools-all-parser-tests.log` in the same directory.

## Speed profile on this machine

The measured interactive profile uses the **Q8 dense variant**, frequency-based
expert-cache retention, and microbatches of eight. Q8 dense projections and
adjacent pointwise operations run on the RTX 3060 Ti (CUDA0). Sensitive
nonquantized projections and routed experts stay on the RTX 3080 (CUDA2);
the output head runs on the RTX 2080 SUPER (CUDA1).
Routed experts retain the original mixed-Q2 quantization, and Engram retains
affine2. The repaired BF16-dense package above remains available separately.

The latest profile uses a **512 MiB pinned expert RAM cache**, eight loading
workers and about 7 GiB of expert VRAM on CUDA2. The smaller explicit RAM tier
leaves more memory for the OS file cache. It measured **1.93–2.03 tokens/s** on
the short sky prompt and **1.41–1.48 tokens/s** on a separate 96-token
code-generation prefix. The corresponding longer checks reached 1.37 with
512 MiB of pageable RAM and 1.32 with an 8 GiB pageable RAM cache.

An earlier small-pageable-cache pair reached 2.05–2.10 tokens/s, followed by
slower controls as storage/cache conditions changed. The OS cache was not
reset and desktop GPU memory use varied. These measurements support the
smaller RAM profile; they are not a universal rate or a proven hardware limit.
See [PERFORMANCE_EXPERIMENTS.md](PERFORMANCE_EXPERIMENTS.md) and
[speed4-validation.json](speed4-validation.json) for the complete comparison.

The runner enables `ESE_CUDA_MOE_BATCH_VECTOR=1` and
`ESE_CUDA_DENSE_BATCH_VECTOR=1` for microbatches larger than one. These preserve
the single-token CUDA arithmetic while batching expert transfers. A fixed
54-position teacher-forced comparison reproduced every baseline Q8 logit
exactly; ordinary batched CUDA arithmetic did not. Set both variables yourself
when invoking `llama-cli` directly with `-ub` greater than one.

The extra dense Q8 quantization is approximate: against the repaired BF16-dense
model, the 40 answer positions had mean KL 0.0188 and 95% argmax agreement.
This is a scoped numerical/quality check, not broad model certification.
The direct BF16 vocabulary kernel accumulates in F32 and avoids a 2.5 GiB F32
weight copy on CUDA1. Its maximum difference from the preceding F32-copy head
was 7.63e-6 across the 54-position comparison, with every argmax unchanged.
The latest cache, placement and I/O changes reproduce the subsequent verified
Q8 profile's logits bit for bit. See [speed-validation.json](speed-validation.json)
for earlier quantization measurements and paths.

DSpark is now available in a separate experimental greedy executable, with
bounded accepted-prefix rollback and CUDA target-feature capture. Its tested
one-proposal configuration matched ordinary greedy token IDs but did not give
a meaningful speedup. It is not enabled in this profile. See
[DSPARK.md](DSPARK.md) for its package, invocation and measured limits.

## Ask a question

Run from the local checkout, selecting an output directory for the run:

```bash
cd /home/xero/Desktop/test/ESE-v0.2.1-qwen4exp-plan
python3 tools/deepseek_v41_smoke.py \
  --package /home/xero/Desktop/test/models/deepseek-v41-native-q8dense \
  --binary build_cuda_release/bin/llama-cli \
  --tensor-split 0,0,1 --tensor-override '^output$=CUDA1' \
  --tensor-override '^blk\.[0-9]+\.(attn_(norm|q_a_norm|kv_a_norm|sinks|q_a|q_b|kv|output_a|output_b)|ffn_(norm|gate_shexp|up_shexp|down_shexp)|hc_(attn|ffn)_(base|scale)|exp_probs_b|indexer\.(attn_q_b|k_norm)|attn_compressor_norm)$=CUDA0' \
  --expert-device CUDA2 --expert-vram-cache-mib 7168 \
  --ubatch 8 --cache-retention frequency --ram-retention frequency \
  --expert-ram-cache-mib 512 --pinned-ram --load-workers 8 --direct-bf16-head \
  --encoder /home/xero/Desktop/test/results/deepseek-v41-20260910/reference/encoding/encoding.py \
  --output /home/xero/Desktop/test/results/deepseek-v41-local-question \
  --question 'Explain why the sky is blue in two short sentences.' \
  --tokens 128
cat /home/xero/Desktop/test/results/deepseek-v41-local-question/stdout.txt
```

`--question-file /absolute/path.txt` accepts a longer question. The runner uses
the pinned official chat encoder and records the exact prompt, native command,
environment, output, timings and sampled memory in the output directory.

The command above uses explicit dense-weight placement, a separate CUDA1
output head, 512 MiB of pinned expert RAM, up to 7 GiB of expert VRAM,
256 MiB staging, eight loading workers, 768 MiB GPU reserve, context 1024, and
eight-token microbatches. Realized expert VRAM is clamped to available memory.
Only the small RAM arena is pinned. Model weights
remain memory-mapped with `GGML_CUDA_NO_MODEL_PINNED=1`; bounded expert leases
provide the active weights. A 32 GiB observed-RSS threshold stops an oversized
run. This is a diagnostic one-shot launcher, not an interactive chat server.

## Initial measurements before the speed profile

The same 40-token explanation was generated with identical output in all three
cache configurations. The OS page cache was not reset between runs, so these
are initial tuning probes rather than a controlled benchmark.

| Expert RAM | Expert VRAM per GPU | Prefill tokens/s | Decode tokens/s | Peak host RSS |
|---|---|---|---|---|
| 8 GiB | 1 GiB | 0.68 | 0.71 | 22.44 GiB |
| 12 GiB | 1.25 GiB | 0.66 | 0.74 | 25.62 GiB |
| 16 GiB | 1.25 GiB | 0.59 | 0.73 | 26.78 GiB |

These earlier BF16-dense runs used the original three-device split. Larger
caches reduced expert file reads but provided little overall speed improvement.
The larger-cache runs sampled total GPU usage of 6,989 / 6,650 / 8,624 MiB.
The baseline snapshot was 6,733 / 6,394 / 8,368 MiB. Device totals include the
desktop and other processes. The 12 GiB run reached its first output in
34.6 seconds including model startup.

Five checks passed: arithmetic (`2 + 2` → `4`), a coherent two-sentence
explanation, code-word recall after 168 prompt tokens, a Python `square(n)`
function tested on three inputs, and a factual response (`Paris`). The recall
prompt crosses the 128-token sliding window. These are scoped checks, not a
capability evaluation. See [trained-validation.json](trained-validation.json)
and the individual run logs under
`/home/xero/Desktop/test/results/deepseek-v41-20260910`.

## Current limits and next tuning work

The validated path supports one text sequence and microbatches up to eight.
The batch tests cover causal compressed-cache access, repeated window wraps,
whole-batch abort/retry, output selection, and remote expert routing. Explicit
checkpoints can restore any accepted prefix of up to eight verified tokens;
arbitrary suffix removal without a checkpoint remains unsupported. Million-token
context, vision and chat-server integration are not verified by these runs.
The remaining speed limit is predominantly expert loading and transfers.
Early-router storage advice is implemented as an opt-in experiment and is
disabled in the recommended profile because it has not demonstrated a reliable
speedup. Read [PERFORMANCE_EXPERIMENTS.md](PERFORMANCE_EXPERIMENTS.md) for its
measured prediction quality and limits.

For package reproduction, use `deepseek_v41_package.py`, followed by
`deepseek_v41_dense_repair.py --apply` with the pinned reference. The latter
fetches original dense weights, expands 328 tensors, journals replacements and
rehashes the package. The source tensor samples and expert quantization-error
reports are in this directory; full repair provenance remains beside the model.

To reproduce the Q8 dense variant, run `deepseek_v41_dense_q8.py` against the
repaired native package and its `dense-originals/prepared.json` journal. It
requires reflink support, preserves the routed/Engram payloads, and records
per-tensor quantization errors plus updated hashes. Never substitute the
unrepaired downloaded GGUF files.
