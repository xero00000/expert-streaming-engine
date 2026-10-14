# Qwen4-Exp reproducibility evidence

This directory contains immutable evidence used to gate experimental
`qwen4exp` support. Runtime support must remain disabled until the applicable
correctness and resource gates in `docs/V0_2_1_QWEN4EXP_PLAN.md` pass.

## Pinned upstream revisions

- Official model/config: `Qwen/Qwen3.8-Flash-Next` at
  `f5d08274bafd880402bd16f5e3e6c514136ec06c`.
- Qwen technical report repository: `QwenLM/Qwen3.8-Flash-Next` at
  `513aa6e18a335296fc13e538232a8735b230877d`; `tech_report.pdf` Git blob
  `f26e9dd6ea957f375eafee1af515a3b6b7c86ae3` (2,366,146 bytes).
- Transformers `qwen4_exp` reference: `huggingface/transformers` commit
  `fc5c5bde8e656dad91cbf34e61940d984b1c7b91`.
- GGUF fixture repository: `unsloth/Qwen3.8-Flash-Next-GGUF` at
  `d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249`.
- ESE base: `30642cdcbe4246aaddb0cbf7d079122c443ff5e2`.

`upstream-artifacts.json` records SHA-256 digests for the official config,
tokenizer, chat template, license, technical report, Transformers reference,
and quantization model card. Both model cards declare
`qwen-community-1.0`; the pinned GGUF repository has no `LICENSE` file, so
that provenance gap remains explicit rather than being silently treated as a
complete license fixture.

These revisions were resolved on 2026-08-26. The checked-in production
inventory is generated from the maintainer's `UD-IQ1_S` three-shard model;
it is evidence for local development, not a CI fixture.

## Regenerating the production inventory

```bash
python3 tools/qwen4exp_manifest.py /path/to/model-directory \
  --official-revision f5d08274bafd880402bd16f5e3e6c514136ec06c \
  --gguf-revision d3bc75ee6ccef3efc1e228ec00a6cc2cdb1e2249 \
  --output docs/qwen4exp/ud-iq1-s.manifest.json
```

The manifest stores shard basenames, sizes and SHA-256 hashes, all tensor
names/shapes/types/extents, selected architecture metadata, and the canonical
tensor-inventory hash. It deliberately does not store an absolute local path.

## Replaying the PLE reference trace

`ple-reference-trace.json` was produced by executing the PLE class extracted
from the content-verified, pinned Transformers source on a meta device. The
large embedding is replaced only after construction with a recorder that
captures the official row IDs, so no model weights are downloaded or
allocated.

```bash
python3 tools/qwen4exp_reference_trace.py \
  --verify docs/qwen4exp/ple-reference-trace.json
```

The script fetches only immutable commit URLs and rejects either source if its
SHA-256 differs from `upstream-artifacts.json`. For an offline integrity check:

```bash
python3 tools/qwen4exp_reference_trace.py \
  --check-hash-only docs/qwen4exp/ple-reference-trace.json
```

`ple-iq4nl-row-fixture.json` contains one production 90-byte PLE row plus all
160 dequantized float values. It lets CPU and accelerator dequantizers prove
value parity without accessing the 28.8 GB tensor. Regenerate it from a
verified local model with:

```bash
python3 tools/qwen4exp_row_fixture.py /path/to/model-directory \
  docs/qwen4exp/ud-iq1-s.manifest.json --row 16121432 \
  --output docs/qwen4exp/ple-iq4nl-row-fixture.json
```

## Storage prototype status

The CPU prototype now has a multi-request asynchronous reader and a fixed-arena
block cache suitable for compact-row and page modes. Exact concurrent reads are coalesced across
request owners, storage work runs on worker threads, and cancellation leaves
other owners' shared reads valid. Queue depth, logical/source-request bytes, read
counts, failures, coalescing, cancellations, and P50/P95/P99
submit-to-ready latency are reported separately.

The compact-row store maps global PLE IDs through checked extents, stages only
the selected 90-byte rows, retains them in the bounded cache, and dequantizes
the ordered batch to `16 × 160` floats. Cache slots are never evicted while a
lease is active, and the arena allocation is rounded down to the configured
physical capacity.

`ple-storage-warm-baseline.json` records the first production-shard replay.
It is explicitly a warm OS-page-cache baseline: the model had already been
hashed and inspected, so it is not evidence for the cold-cache gate. In this
run mmap had substantially lower submit-to-ready and token latency than
worker-thread pread. The benchmark reports source-request bytes separately;
it does not mislabel them as physical SSD traffic.

`ple-storage-cold-baseline.json` records a controlled cold page-store replay.
After validating the file identity, the benchmark passed the exact 28.8 GB PLE
tensor range to `POSIX_FADV_DONTNEED`; mmap runs additionally use
`MADV_DONTNEED`. The evidence records whether that kernel control succeeded,
but does not claim to flush or measure the NVMe controller's internal cache.

The row store separately bounds resident row-cache bytes and asynchronous
staging bytes. Its current constructor gives each arena the requested cache
budget. Benchmark telemetry reports staging capacity, current and peak
residency, and rejected batches, so a future whole-runtime budget can split
its allowance between the two arenas explicitly.

The controlled page store caches fixed 4 KiB blocks and reconstructs rows that
cross page boundaries while their page leases are pinned. The two-tier mode
places the compact 90-byte row cache above that page cache; the standalone
benchmark splits `--cache-mib` evenly between the tiers. Use
`tools/qwen4exp_ple_benchmark_matrix.py` to replay all requested storage modes,
backends, cache states, and 256 MiB–2 GiB budgets into content-addressed JSON.
`ple-storage-budget-matrix.json` is the checked-in 36-run warm matrix for
`pread`, mmap, and native depth-64 `io_uring`. Its short reference workload proves parity, concurrency,
bounded capacity, and accounting at each budget, but explicitly makes no
cache-pressure or production hit-rate claim.

For an eviction-producing workload, `--pressure-tokens N` replaces the short
trace replay with independent deterministic token histories for every session;
`--pressure-seed` makes the run reproducible. For example:

```bash
build_cpu_avx2_debug/bin/qwen4exp-ple-bench \
  --shard /path/to/Qwen3.8-Flash-Next-UD-IQ1_S-00002-of-00003.gguf \
  --manifest docs/qwen4exp/ud-iq1-s.manifest.json \
  --trace docs/qwen4exp/ple-reference-trace.json \
  --backend io_uring --storage two-tier --cache-state cold \
  --cache-mib 256 --workers 8 --iterations 1 --sessions 4 \
  --pressure-tokens 4096 --pressure-seed 1251732529
```

Unlike the coalescing trace, each session requests different rows. Periodic
PLE-EOS tokens exercise segment resets, while random rows drive cache churn
under fixed capacities. `ple-storage-pressure.json` records the example run:
its 266,528 source reads caused 233,760 page evictions with zero failures or
staging rejections. The compact-row tier reached only 23.6 MiB of its 128 MiB
capacity, so this run is page-pressure evidence and does not claim row-tier
saturation.

Submission is non-blocking, so an accepted token can enqueue its PLE rows
before layer 0. The benchmark's optional `--overlap-us` window models work in
layers 0 and 1 and reports `ple_wait_ns` separately from total token latency;
this is an overlap feasibility measurement, not a substitute for the eventual
end-to-end graph stall gate.

The production runtime now accepts `--ese-ple-mode auto|mmap|async|ram`,
`--ese-ple-cache-mib N`, and `--ese-ple-queue-depth N`. `auto` probes native
`io_uring` and falls back to the portable two-tier `pread` profile; `mmap` and
`async` select exact mmap and native `io_uring` sources, and explicit `async`
fails closed if `io_uring` is unavailable. Native `io_uring` and `ram` use the
compact selected-row store because production accesses showed negligible 4 KiB
page reuse; each source request is the exact 90-byte row. `ram` does not preload
the complete 26.8 GiB table. Portable two-tier profiles reserve one quarter of
the requested bound for 4 KiB pages and three quarters for compact rows.
Defaults remain the measured-safe 256 MiB total, four workers, and source batch
width 32.
The native resource planner accounts the full selected PLE cache as a fixed
RAM allocation, and resolved-plan JSON reports it separately as
`qwen4exp_ple_ram_bytes`. Sweep-bench table metadata and JSONL rows include the
PLE mode, cache MiB, and queue depth so externally generated tuning matrices
remain self-describing.

Context teardown emits one aggregate `qwen4exp_ple_stats` JSON record. It
contains logical and physical reads/bytes, source batch width, coalescing,
failures, cancellation, queue depth, submit-to-ready P50/P95/P99, measured
input-staging wait, and page/row capacity, residency, hits, misses, and
evictions. It contains no prompts, token-derived hashes, row IDs, filenames,
or local paths.

## CPU operator status

The architecture gate remains closed. The first dormant graph primitives cover
Qwen4Exp's four-stream gated-residual read and injection operations, with a
deterministic CPU scalar-parity test. The GGUF stores effective RMSNorm gamma
values directly: although pinned Transformers evaluates `(1 + raw_weight)`,
the production hyper-connection, PLE, and attention norm tensors are centered
near 1.0 and must be consumed as ordinary GGUF RMSNorm weights. The graph must
not add one a second time.

The recurrent wrapper now exposes the same projection, causal convolution,
delta-rule state update, gated RMSNorm, and output projection used by the
tested Qwen3-Next kernel, but intentionally omits its conventional pre-norm and
residual add. Qwen4Exp's surrounding gated-residual operators own those steps.
The PLE operator test additionally covers key/value projection, grouped
key/query/conv normalization, signed-square-root stream gating, and the
four-tap dilation-3 causal depthwise convolution with its explicit nine-token
state. QSA coverage now includes complete-block mean pooling, effective-gamma
key normalization, summed ReLU scores across indexer heads, causal block
masking, top-block selection, block-to-token expansion, and the exact union of
selected complete blocks with the incomplete causal tail. A tested raw-key
cache primitive writes at a nonzero KV head and reads the complete used prefix;
Qwen4Exp cache allocation, defragmentation, state save/restore, and graph-reuse
offset metadata use the existing `kr_l` infrastructure. The production builder
registers that write, applies block-start and query RoPE, constructs host-filled
I32 block geometry without unsupported CPU casts, and passes the sparse mask to
the standard attention path. F16 indexer-cache rows are promoted to F32 before
block pooling because the CPU backend's row reduction is F32-only; focused
operator coverage exercises that boundary.

PLE convolution state is reserved in the recurrent cache row, so generic
recurrent copy/reset/serialization carries its nine-token history. The context
also owns position-tagged PLE token history and public KV clear, sequence
remove, copy, and keep operations update it in lockstep. Unit coverage proves
suffix rollback reconstruction and token-history copy/restore. The
architecture-specific KV payload tail serializes either all histories or one
destination-remapped sequence, with bounded restore validation and the normal
transactional rollback path. Runtime input setup now hashes each accepted
token, submits its 16 global rows to a bounded two-tier PLE store, dequantizes
directly into the host graph input, and publishes the candidate token history
only after graph submission. The production-model runtime state round-trip
described below exercises both the full-context payload and destination-remapped
per-sequence payload. This wiring is dormant and is not evidence for an
end-to-end parity gate.

`graphs/build_qwen4exp.cpp` now compiles the reference layer order through
initial four-stream expansion, PLE injection and recurrent-row writeback,
DeltaNet layers, both GR injection sites, normalized top-10/512 softmax MoE,
sigmoid-gated shared experts, and the final no-injection stream mixer. The
shared attention machinery also recognizes Qwen4Exp's interleaved query/gate
projection on single-device and split-attention paths. The default loader still
rejects Qwen4Exp unconditionally. Dormant dispatch exists only so explicitly
instrumented local probes can exercise graph construction while the checked-in
safety gate remains closed.

## Production runtime smoke evidence

On 2026-08-26, an explicit temporary probe gate loaded the three-shard
`UD-IQ1_S` production model, validated all 67 metadata entries and 1,224 tensor
records, skipped only the 28,800,138,240-byte PLE allocation, and opened that
tensor's loader-validated shard path and byte extent through a bounded 64 MiB
page plus 192 MiB row store. A one-token `Hi` prompt then executed the complete
48-layer CPU graph successfully in 1,081.64 ms on the AVX2 debug build. The
reserved graph reported 6,604 nodes, one split, and a 6.40 MiB compute buffer;
the concrete one-token graph reallocated that buffer to 6.52 MiB.

The same model also passed the existing `llama-save-load-state` runtime test
with `-c 128 -b 1 -ub 1 -np 2 -p Hi -n 1`. It serialized and restored a
237,106,835-byte full-context state, copied a 118,068,292-byte sequence-0
payload, cleared the cache, restored that payload into sequence 1, and produced
the identical one-token continuation (`5`) in the original, full-state restore,
and remapped-sequence restore runs. The test exited successfully.

A production one-token run also exercised the runtime controls with
`--ese-ple-mode auto --ese-ple-cache-mib 32 --ese-ple-queue-depth 2`. The
resulting telemetry reported 16 reads / 65,536 physical bytes for 16 rows /
1,440 logical bytes, 2,305,805 ns of total PLE input staging, an exact 8 MiB
page tier, and a 25,165,800-byte row tier. The configured width-2 bound split
those reads into eight source batches and telemetry reported a peak source
batch width of exactly 2. Separate context-construction probes
successfully selected mmap, native `io_uring`, and compact-row RAM modes with
their requested 1 MiB bounds and queue depths.

A fixed-seed greedy 33-call sequential runtime probe (one prompt evaluation
plus 32 continuation evaluations) then exercised evolving PLE history with the
default 256 MiB/four-worker profile. It staged 528 rows / 47,520 logical bytes
through 542 page reads / 2,220,032 physical bytes with zero failures; peak
source batch width was 18 under the configured bound of 32. Aggregate
PLE input wait was 159.83 ms over 20,161.60 ms total runtime (0.79%); source
submit-to-ready P95 was 5.05 ms and the maximum per-call input wait was 5.66 ms.
The PLE caches were cold for this context, but the operating-system page cache
was not explicitly evicted, so this is a warm-source runtime stall result and
not the documented cold-NVMe result.

## CUDA reference evidence

On 2026-08-26, a CUDA 13.2 debug build targeting both `sm_75` and `sm_86`
passed all nine focused `qwen4exp|kda-delta-net` CTest cases. The PLE reference
test stages sixteen contiguous IQ4_NL rows into backend memory, executes the
backend's IQ4_NL-to-F32 graph, copies the result back, and requires exact
scalar-reference equality. It iterated over every visible CUDA device and
passed on device 0 (RTX 3060 Ti, compute capability 8.6), device 1 (RTX 2080
SUPER, 7.5), and device 2 (RTX 3080, 8.6) with driver 595.84. This is direct
CUDA gather/dequant/stage parity evidence for both Turing and Ampere; it is not
full-model CUDA logit, route, multi-GPU execution, or soak evidence.

The gated-residual operator test also has a CUDA-only subgraph mode. After
replacing a 3D gamma broadcast with the equivalent portable flattened 1D
broadcast, its scalar-reference parity case passed independently on all three
devices. The complete CPU operator graph remains the reference for operations
such as QSA selection whose portable CUDA fallback is scheduler-driven rather
than supported by one monolithic CUDA graph.

A temporary one-layer production-model CUDA smoke initially exposed an
unassigned QSA raw-key cache view with KV offload and an illegal access in the
layer-47 GR gamma `MUL` without KV offload. The common cause was tensor
placement: `split-mode=none` initialized only the main CUDA backend but still
selected secondary-device buffer types for some layer tensors. Tensor
placement now pins every offloaded repeating/output-layer buffer to the
selected main GPU in this mode, including custom `--device` ordering. The
selected-device memory calculation and partial-offload output buffer use the
same device mapping.

After that correction, one-layer `UD-IQ1_S` runs with normal KV offload and
with `--no-kv-offload` both completed in eager CUDA execution. Graph-enabled
normal-KV runs then completed independently on CUDA0 (RTX 3060 Ti), CUDA1 (RTX
2080 SUPER), and CUDA2 (RTX 3080). Each device held the complete 1,268.47 MiB
offloaded model buffer without allocating model tensors on another GPU,
reserved a 6,404-node/111-split graph, captured 61 CUDA graphs, and emitted the
same fixed-seed greedy continuation (`//`). Prompt evaluation was 756.29 ms,
680.44 ms, and 622.76 ms respectively. The current CPU graph emitted the same
continuation in 573.37 ms for the identical command line. This is short-path
production token parity and heterogeneous CPU-fallback scheduling evidence;
it is not full-logit, router-decision, multi-GPU layer-split, or soak evidence.

A true layer-split smoke then offloaded layers 45, 46, and 47 independently to
CUDA0, CUDA1, and CUDA2. The model buffers were 925.89 MiB, 925.89 MiB, and
920.78 MiB; all three backend contexts captured graphs (57/1/3), the one-token
prompt completed in 599.85 ms, and its `//` continuation matched the CPU and
single-device results.

A deeper 18-layer split also completed, with 5,145.11 MiB, 4,324.34 MiB, and
6,171.01 MiB model buffers on the three GPUs and a 501.74 ms first-token prompt
evaluation. The same split successfully allocated a 65,536-token context:
1,536 MiB of K/V cache plus a 192 MiB QSA indexer cache were distributed as
1,080.05 MiB host and 300.47/156.47/303.59 MiB device buffers. Graph capture
and the first token completed in 494.64 ms with the same continuation. This is
64K capacity-and-first-token evidence only; the cache was not filled to 64K and
the long-context completion gate remains open.

For a wider parity check, `llama-perplexity` captured seven evaluated logit
distributions from a 16-token batch in the CUDA-configured binary's `-ngl 0`
path and compared them with the three-layer/three-GPU path. KL divergence was
0.00067, all seven top tokens agreed, and RMS probability delta rounded below
0.001%. The declared tolerance for this short evidence tier is KL at most
0.001, 100% top-token agreement, and at most 0.25% selected expert-ID set
substitutions.

The `qwen4exp-route-probe` callback captures all 48 `ffn_moe_topk` tensors and
12 QSA top-block tensors without changing the graph. In the same 16-token
comparison, 756/768 token-layer MoE route sets were exact. The other twelve
sets each substituted one near-boundary expert: 12 of 7,680 selected expert
IDs differed (0.15625%), within the declared tolerance. All QSA selected-block
sets were exact; layer 47 differed only in ordering because this short padded
graph selected the complete 64-block candidate set. Per-tensor and aggregate
FNV-1a digests make the comparison reproducible, and setting
`QWEN4EXP_ROUTE_VALUES` to `1` or a comma-separated tensor-name list emits the
underlying IDs for mismatch accounting.

An intentionally separate comparison between the standalone AVX2 build and
the CUDA-configured binary at `-ngl 0` produced KL 0.32445 with 28.571% top
agreement. Both used AVX2/FMA/F16C, but different host compiler toolchains.
That result is retained as a build-reproducibility warning and is not folded
into the same-binary CPU/CUDA tolerance above; release packaging still needs a
single controlled toolchain matrix before promotion.

## Decode soak and placement exploration

A deterministic three-GPU `-ngl 18 -sm layer` decode ran for 1,801.35 seconds
at an 8,192-token context before an intentional SIGINT. It completed 3,990
decode tokens at 2.31 tokens/s with a 43,103,556 KiB peak RSS. Sampled device
memory remained flat at 5,427/4,520/7,218 MiB and temperatures and power stayed
bounded. There were no CUDA execution, PLE-read, or allocation failures. The
periodic scheduler reserve message and one CUDA graph-update failure were
handled by the existing graph re-reservation/reinstantiation paths. Device
memory returned to the 29/5/669 MiB idle baseline after exit.

That soak exposed a CLI cleanup bug: the SIGINT handler called `_exit(130)`, so
normal model destruction and final PLE telemetry were skipped. The handler now
performs only a bounded `sig_atomic_t` update; the generation loop handles the
request on the main thread, preserves the first-Ctrl-C interactive behavior,
runs normal timing/log/model/backend teardown, and returns 130. A separate
90-second real-model cancellation probe decoded 65 tokens at 2.33 tokens/s,
returned 130, emitted the complete 66-call `qwen4exp_ple_stats` record, destroyed
all three CUDA contexts, and returned device memory to idle.

Engine placement experiments found substantial but currently unpromotable
headroom. Full `--fit` placement kept 21.22 GiB of experts in host memory,
offloaded all dense layer work, and reached 7.29 tokens/s over 32 decode tokens
(3.13x the 2.33-token/s baseline). Its one-token route probe diverged broadly
after the first CUDA-resident layers, however, so it is outside the declared
short-route tolerance. A whole-layer 25-layer 8/8/9 split using
`-ts 0.320,0.320,0.360 --fit-margin 256` was reliable and reached 2.80 tokens/s
(20% faster), but changed nine of 480 selected expert IDs (1.875%); QSA sets
remained exact and two differed only in order. A 26-layer split fit the model
buffers but left too little headroom for a cuBLAS handle.

The initial 5/5 CPU/GPU hybrid expert experiment exposed bounded compact-cache
exhaustion at `blk.39.ffn_down_exps.weight`. Cache realization previously ran
after policy-sized compact graph copies had consumed VRAM, so the live cache
was clamped below its planned capacity and component buffers could fail lazily
after the full-tensor fallback was gone. Graph assignment now realizes every
catalogued component first and sizes compact copies from the resulting
per-device slot count. The formerly failing command then realized 302 slots and
512 MiB on each GPU, completed with zero forced fallbacks, and reached 7.31
tokens/s. The cache fix is retained, but this hybrid profile is not promoted:
its one-token comparison changed 464/480 canonical selected expert IDs after
layer zero. Disabling active-expert GPU migration with `-no-ooae` reached 7.91
tokens/s and produced the same output prefix as full `--fit`, localizing the
large drift to full-depth placement rather than cache compaction. The route
probe now excludes hybrid `_gpu`/`_cpu` partition tensors from its canonical
48-layer digest. These faster profiles remain optimization evidence and engine
follow-up targets, not recommended defaults.

## Upstream comparison and shared-artifact benchmark readiness

The upstream Qwen3.8-Flash-Next (`qwen4exp`) architecture and converter first
landed in llama.cpp PR [#27742](https://github.com/ggml-org/llama.cpp/pull/27742)
on 2026-08-27; this is merged support, not a draft. The important correctness
follow-up, PR [#27941](https://github.com/ggml-org/llama.cpp/pull/27941), also
landed (commit `36b1015`, 2026-09-01): it fixes indexer state on sequence copy,
separates QSA blocks by sequence set, handles image-token rank order, rejects
bad metadata cleanly, and fixes a CUDA launch limit at 262K context. Upstream
has continued changing this path since then, including Qwen4Exp MTP (PR #29761)
and QSA graph/memory optimizations. For reproducible parity work, pin an exact
upstream commit; the current master observed on 2026-10-04 was
`a7fb71fab83b474a0892b9a05aaa3a8ddca2729b`.

Treat upstream as the architecture/correctness reference, not as a drop-in for
ESE's storage design. ESE adds its own bounded PLE row storage and indexed IQ
expert path. ESE's normal model loader still rejects `qwen4exp`; the local
experimental graph and operator tests do not remove that production safety
gate. The CUDA/logit and route measurements above are useful focused evidence,
not proof that the gated path is ready for general generation.

The installed Strata Coder artifact is a possible common benchmark input:
`Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M`, split across a 29.61 GB model shard and a
28.80 GB PLE shard. Its metadata is `qwen4exp`, 48 layers and 256 experts (the
coder-pruned model, not the full 512-expert checkpoint); its PLE tensor is the
expected IQ4_NL, 160-value, 90-byte-row table. The model shard uses GGML type 42
(`Q2_0`) for nine tensors. The current ESE source registers this type and the
`test-gguf-unsupported-type` regression explicitly accepts it, correcting the
older Oct 2 screening report that described type 42 as unsupported. The
launcher-selected ESE server binary predates that later parser test/build, so
its behavior has not been revalidated against this artifact. More importantly,

the current ESE model-load path still intentionally throws
`unsupported architecture: qwen4exp`; Qwen architecture enablement and
numerical parity—not the GGUF type ID alone—remain the blockers. Do not report
an ESE-vs-Strata speed ratio until the exact artifact loads and passes output
parity on ESE.

On 2026-10-04 the current source tree's GGUF reader was validated directly
against the real shards with `test-qwen4exp-artifact` (read-only metadata
probe, built but not auto-registered because it needs shard paths):

```bash
build_cuda_release/bin/test-qwen4exp-artifact \
  /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \
  /path/to/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf
```

Shard 1 parsed with the exact architecture metadata (`qwen4exp`, 48 layers,
256 experts, top-10), 1,223 tensor records, and exactly nine Q2_0 tensors;
the mixed-quant histogram (484 bf16, 292 f32, 129 q6_K, 86 IQ4_NL, 47 q4_K,
45 IQ4_XS, 40 IQ2_S, 35 q5_K, 34 IQ3_XXS, 20 IQ3_S, one each f16/q8_0) is
consistent with an IQ1_M-class artifact. Shard 2 holds exactly one tensor,
`per_layer_token_embd.weight` as IQ4_NL with the documented 160 ×
320,001,536 shape and 28,800,138,240-byte payload, and its file size brackets
the payload within 192 bytes. Split metadata reads `split.no=0`,
`split.count=2`, `split.tensors.count=1224`. The probe allocates no tensor
data and does not open the architecture gate; the installed launcher binary
remains unvalidated until it is rebuilt from this worktree.

The artifact shards currently live under Strata's data directory rather than
ESE's configured model directory. A benchmark runner can point both engines at
the same read-only shard paths; do not duplicate the roughly 58.4 GB artifact
on the nearly full model volume.

The current Strata profile enables MTP/speculative decoding (spec 4), while
ESE's UI advertises MTP unsupported. For a baseline comparison, disable Strata
speculation or publish separate baseline and MTP rows. Match prompt/token IDs,
context, KV settings, batch sizes, GPU placement, and sampling, and record both
engine revisions and shard hashes.

### Planned: bounded resident-window KV streaming

Strata's Coder profile uses `--kv int8 --kv-resident 32768` at a 131,072-token
context: the full KV history is host-resident while only a 32K-cell window per
QSA layer is kept on each GPU. ESE currently has Q8 KV support and can place KV
in host memory, but it does not implement this bounded GPU-resident window /
streaming policy. This was a material mismatch in the 2026-10-04 ESE probe: a
full-context Q8 host KV could be allocated, but CUDA0 ran out of room reserving
the 1,024-token microbatch compute buffer. Track a future ESE implementation
that supports a configurable resident window (initial target: 32,768 cells),
streams the matching KV ranges for QSA attention across multi-GPU layer splits,
and verifies that streamed and fully resident KV produce equivalent logits.
Benchmark it against Strata with identical context, KV precision, window,
prefill chunk, prompt, and GPU placement. Keep this distinct from the separate
n-gram/PLE NVMe row streaming path above; PLE is already read on demand from
NVMe and is not what Strata's `--kv-resident` controls.

The nearest ESE smoke attempted the same 131,072 context, Q8 KV, 1,024 logical
batch and microbatch, CUDA0+CUDA2 split, and PLE row streaming, with MTP disabled
because ESE has no matching implementation. It failed during compute-buffer
allocation on CUDA0; it did not produce throughput. ESE also has no
resident-window control, so that attempt is not a completed apples-to-apples
benchmark.

The earlier Strata-vs-ESE screen found that ESE could not load this model and
contained no ESE throughput result. The later, explicitly opt-in local probe
build got far enough to run a short decode, but it is still not a head-to-head
benchmark: its matching MTP/KV-window controls are absent, it required a smaller
microbatch, and its prompt path differed from Strata's chat template.

## 2026-10-04 local probe against the Strata launcher profile

A dedicated `ESE_QWEN4EXP_PROBE_GATE=ON` build plus the per-run
`ESE_QWEN4EXP_PROBE_GATE=1` opt-in was used; normal builds remain fail-closed.
The shard paths were shared read-only. With 131,072 context, explicit `q8_0`
K/V, CUDA0+CUDA2 and logical batch 1,024, the ESE context allocation failed
when `ubatch=1,024`: CUDA0 could not allocate a 2,285 MiB compute buffer. The
same settings with `ubatch=128` succeeded; logged KV was 1,632 MiB Q8 plus a
384 MiB F16 QSA-indexer cache, CUDA0/CUDA2 compute buffers were 324/372 MiB,
and weight buffers were 5,855/6,411 MiB. On raw prompt `Hi`, ESE completed both
a 5-token smoke (6.60 generated tok/s) and a 128-token run (5.78 generated
tok/s; 172.96 ms/token). The longer run is more representative but still only
a local diagnostic, not a parity-qualified release benchmark. The PLE table
was read on demand through `io_uring` from the NVMe-backed second shard (for
128 tokens: 2,048 logical rows, 1,960 source reads / 176,400 bytes, 88 row
cache hits, zero failed reads), not loaded as a 28.8 GB tensor.

For a matched raw-prompt Strata reference, the saved launcher config ran on
CUDA0+CUDA2 with `STRATA_SPLIT_OWN=1`, auto layer split, prefill 1,024, spec 4,
MTP, 131,072 context, int8 KV, 32,768 resident cells, and `spec-min-p 0.5`.
Strata's tokenizer maps `Hi` to token ID 12675, which was passed directly to
the server engine, avoiding chat-template differences. At 128 generated tokens
Strata reported 2,944.2 ms decode (43.5 tok/s), accepting 82 of 83 drafts; it
reported 98.49% of KV block reads hitting VRAM. ESE used the same raw text,
output count, context, Q8 KV type, GPU pair and logical batch, but required
ubatch 128 rather than Strata's 1,024 prefill chunk, kept the full KV in host
memory rather than the 32K GPU window, and had no matching MTP or GPU expert
cache. Thus the 43.5 vs 5.78 tok/s results are useful evidence of the current
gap, but not a controlled apples-to-apples engine benchmark; ESE has also not
passed whole-model logits/routes parity. The exact 1,024 ESE ubatch failed to
allocate, while its 128-token ubatch run exited 0 after all 128 generated
tokens. Strata's five-token API sample (different, 53-token chat-template
prompt) is retained only as an earlier setup check, not the matched result.

## Why Strata was faster in the matched diagnostic

### What was measured

This is a same-host, same-artifact, same-raw-input smoke comparison—not a
controlled engine shootout. Both runs used the coder-pruned IQ1_M Qwen3.8-
Flash-Next shards, physical GPUs 0 and 2, 131,072 context, greedy decoding,
raw prompt token ID `12675` (`Hi`), and 128 generated tokens. Strata was engine
0.1.34; ESE used the dedicated opt-in Qwen4Exp probe build. The useful headline
is the measured gap; it is not an intrinsic engine-speed multiplier.

| Run | Decode result | Important runtime evidence |
| --- | ---: | --- |
| Strata saved profile | 2,944.2 ms for 128 tokens; 43.5 tok/s (about 23.0 ms/output token) | 82/83 MTP drafts accepted; 38 suffix-draft windows accepted 76/76; 30,833/49,227 expert lookups hit GPU cache (62.6%); 2,082/2,114 KV block reads hit VRAM (98.49%), with 0.1 MiB read from RAM |
| ESE local probe | 22,138.8 ms for 128 tokens; 5.78 tok/s (172.96 ms/output token) | ubatch 128; Q8 KV with a full host-resident history; PLE source counters show 1,960 reads / 176,400 bytes, 88 row-cache hits and 128 input waits |

The observed throughput ratio is about 7.5× (43.5 / 5.78). That ratio mixes
several deliberate capability/configuration differences. ESE's requested
ubatch 1,024 failed before inference because CUDA0 could not allocate a 2,285
MiB compute buffer; ubatch 128 completed the run. Strata used a 1,024-token
prefill chunk, `--spec 4` with MTP, int8 KV and a 32,768-cell GPU-resident QSA
KV window. ESE had no matching MTP or bounded resident KV-window control, and
this Qwen4Exp probe did not have Strata's per-layer GPU expert cache enabled.
ESE's Qwen graph has not passed whole-model logits/routes parity, either. The
configurations therefore differ in precisely the systems features most likely
to affect decode. Do not read 7.5× as an apples-to-apples engine result or as a
prediction of the speedup from any single feature.

Prompt timings are not compared here: ESE's one-token prefill logged 702 ms,
while Strata's server logged roughly 1.5 ms for its one-token prompt. A
one-token prompt is too short and too sensitive to each implementation's
setup/dispatch accounting to characterize prefill. The ESE 1,024 ubatch
allocation failure is a concrete memory-capacity result, not a completed
throughput measurement at that ubatch.

### Directly supported reasons for the decode gap

1. **Speculative decoding amortizes target-model work.** Strata's config enabled
   MTP speculation (`--spec 4`). Its verify window accepted 82 of 83 drafted
   tokens over the output, in addition to 76/76 suffix drafts across 38
   windows. A successful verify pass can advance several output positions
   while running the target model over a window, so the reported output-token
   rate is not the cost of 128 independent target-model decode steps. ESE did
   not implement a corresponding MTP path in this probe: each output token
   paid for the ESE target graph. This is the clearest directly measured
   algorithmic advantage in the run. It does not imply Strata's target model
   is intrinsically 7.5× faster, and speculative acceptance is prompt- and
   sampling-dependent.

2. **Strata keeps a selected QSA KV working set on the GPUs.** At 131K context,
   it retained 32K KV cells per QSA layer on device and stored the rest in
   pinned host memory. In this decode it reported a 98.49% VRAM hit rate across
   2,114 block reads, with only 0.1 MiB read from RAM. Since QSA attends to
   selected blocks rather than scanning every historical cell, that hot
   window can satisfy most accesses without transferring the entire history.
   ESE allocated Q8 K/V but had no equivalent 32K resident-window policy in
   the probe; its history was host-placed. This is strong evidence of a
   favorable Strata KV residency outcome for this prompt, not proof that all
   future QSA selections will fit in the same window or that the KV difference
   alone explains the speed ratio.

3. **A substantial share of routed experts execute from Strata's GPU caches.**
   Strata logged a 1,024-slot cache (about 1.82 GiB) and a separate 609-slot
   cache on the second layer stage (about 1.18 GiB); the auto-split profile
   chose the per-stage placement for routed-mass coverage. The active decode
   log counts 30,833 GPU-cache hits among 49,227 lookups, or 62.6%. Its cache
   is per-layer/per-stage, and the GPU computes resident experts while a
   15-worker CPU expert pool handles work not satisfied by GPU residency.   This both avoids some CPU expert
   compute and allows CPU work to overlap with GPU work. Some misses can also
   be delegated to the GPU according to the configured PCIe share. The hit-rate
   log is direct evidence; the amount of overlap and its net contribution were
   not separately timed in this run. ESE's
   general codebase has expert RAM-cache,
   prefetch and resource-planning infrastructure, but those are not equivalent
   to a tuned, Qwen4Exp-integrated GPU expert residency policy, and such a
   policy was not active in this probe.

4. **Strata's layer split is chosen for this pair, not divided evenly.** The
   launcher profile split the 48 layers at K=4: four layers on the first,
   slower GPU and the remaining 44 plus output head on the faster GPU. Its
   auto-split log estimates device layer times and reports 45.8 ms per decode
   window and 92.6% of routed mass covered by the planned caches. The process
   log calls the second GPU `CUDA1`; this is the second card in the saved
   physical-GPU-0,2 selection. Strata's layer-parallel design hands off once
   per verification window, rather than transferring activations at every
   layer. This is implementation/log evidence for a topology-aware plan. ESE
   used the same physical pair, but its graph partition and transfer costs
   were not instrumented in a comparable way, so no isolated split-speedup can
   be claimed.

### Likely secondary contributors (not isolated by this comparison)

- **Pinned expert arena and concurrent CPU/GPU work.** Strata logged a
  23.42-GiB expert arena, page-locked for GPU transfer, 15 expert-pool workers
  plus the host thread, and a 0.55 PCIe share for the second stage. The first
  device's probed PCIe setting was 0. These mechanisms help keep expert misses
  from becoming a single synchronous CPU→GPU bottleneck. ESE's weights and
  compute scheduling are organized differently; the benchmark did not measure
  per-expert wait time side by side.
- **Captured verify and draft execution.** Strata logged captured verify
  windows (up to six tokens) and separate captured MTP draft work. Its session
  replays layer graphs and performs the target verification as a window. ESE
  reported graph reuse enabled, but also 6,470 graph nodes and 213 graph
  splits. That suggests an opportunity to reduce scheduling/dispatch overhead
  in ESE; node and split counts alone do not show how much time they cost, and
  the two graph systems are not directly comparable.
- **Large, tuned prefill chunks and specialized kernels.** Strata's 1,024
  prefill chunk is backed by a purpose-built streaming path, pinned staging
  rings, fused/quantized expert operations, and layer-ahead transfers. Its
  published measurements show that chunking and expert streaming materially
  improve prompt throughput. ESE could not allocate the same requested ubatch
  on this device, which exposes a memory-efficiency gap. However, prompt
  chunking is primarily a prefill result; this 128-token decode run does not
  quantify its independent effect on decode speed.
- **Prompt-lookup drafting.** The Strata request log shows 76 of 76 suffix
  drafts accepted in 38 windows, in addition to the MTP acceptance count. It
  is another measured source of draft tokens for this particular continuation;
  its value depends on whether generated text repeats a suffix of the context.

### What did *not* explain most of this run

PLE is not Strata's KV streaming, and neither is a substitute for the other:
PLE is the model's n-gram embedding table; QSA KV holds attention history. ESE
read PLE rows on demand from the second shard using its `io_uring` backend, but
the 176,400-byte source-read count is not proof of physical-media reads on each
request (the OS may satisfy them from cache). ESE reported 164.8 ms total PLE
input-wait time during a 22.14-second decode—under 1% of that elapsed decode
by simple ratio—so this run does not support blaming PLE as the main cause of
the measured gap. PLE still deserves async-overlap work and cold/warm
measurement, especially for prompt batches and NVMe pressure.

The run also did not isolate kernel math, layer handoff, graph replay, CPU
threading, or memory bandwidth with feature toggles. These are plausible
contributors, not additive terms that can be summed to explain 7.5×. A fair
attribution needs ablations on one implementation, and a fair engine comparison
first needs whole-model parity and matched MTP, KV residency, batch/microbatch,
expert-cache budget, layer split, prompt, and sampling settings.

## ESE design to surpass Strata feature by feature

This is a roadmap, not a claim that ESE already wins. The principle is to
beat selected Strata baselines through joint, workload-aware optimization
across resources, while preserving parity; demonstrate any advantage on matched
hardware rather than assuming adaptation is automatically faster.
The existing Qwen4Exp support gate stays closed throughout this work.

| Capability | Strata evidence / current ESE status | ESE design target to beat it | Proof required before claiming a win |
| --- | --- | --- | --- |
| **Speculation / MTP** | Strata accepted 82/83 MTP drafts and 76/76 suffix drafts in this run. ESE Qwen4Exp has no MTP implementation. | First port the target model's MTP semantics and exact verifier; then choose draft depth by observed acceptance, draft cost, entropy/confidence and remaining context. Combine MTP with suffix lookup only when its marginal accepted tokens exceed its overhead. Share QSA/indexer work only after proving state reuse is valid. | Same target logits and greedy decisions as non-speculative ESE; deterministic rejection/rollback tests; accepted tokens per target pass, draft cost, and tokens/s reported by prompt family. Beat Strata's throughput on matched prompts without output-quality regression. |
| **MoE expert residency and execution** | 1,024 profile-ranked GPU slots; 62.6% observed lookup hits; CPU pool runs misses concurrently. ESE has reusable RAM-cache, expert-prefetch and resource-planner components, but the probe had no tuned Qwen GPU cache. | Integrate Qwen routing into ESE's existing cache/planner. Use per-layer, per-device route frequency plus recency and working-set changes; reserve high-value experts on each GPU, predict only high-confidence next-layer routes, and avoid duplicate CPU/GPU execution. Adapt cache quotas when context/KV pressure changes. Keep a CPU/reference fallback. | Route IDs/weights and logits match the oracle within declared tolerances; zero stale cache entries under concurrency; report hit rate, miss compute time, bytes moved and cache churn. At equal VRAM/RAM and same prompts, improve hit rate and end-to-end decode throughput over Strata's 62.6%/43.5 tok/s run. |
| **Sparse-aware QSA KV** | Fixed 32K-cell resident window; this run hit VRAM on 98.49% of block reads. ESE supports Q8 KV and host placement but not that bounded resident window. | Implement a two/three-tier cache keyed by sequence, layer and complete QSA block: GPU hot set for recent tail plus repeatedly selected blocks; pinned-host backing for warm history; optional explicit NVMe spill for cold history. Use prior indexer selections and reuse/cost statistics as hints, but resolve the current query's selected block IDs before attention and fetch every required block. Allocate each layer's hot set based on observed reuse and transfer cost, not a single global cell count. Preserve exact selected-block order, tail, positions, scales, sequence-copy and rollback semantics. | Streamed and fully resident logits/routes match at short/long contexts; no missing or reordered blocks; bounded GPU/pinned RAM/I/O under alternating old/new selections. At identical Q8/context, meet or beat Strata's decode rate and p95 wait while using no more VRAM; publish hit rate, bytes and read amplification. |
| **PLE / NVMe row access** | ESE already has an async `io_uring` row source and bounded row cache; the matched run recorded 1,960 source reads, 88 row hits and 164.8 ms aggregate input wait. These counters are source operations, not physical-media reads. | Keep exact 90-byte quantized-row storage, but make lookup batches nonblocking end-to-end: dedupe in-flight requests across sessions, use storage-page-aware grouping/cache without assuming rows are adjacent, prefetch known prompt rows ahead of layer 2, and overlap reads with layers 0–1 / other sessions. Use admission control so PLE cannot evict expert/KV working sets. Keep mmap as a measured fallback if async I/O loses on the target drive. | Exact row IDs/dequantized values, cancellation and lifetime safety; cold and warm runs on the real NVMe. Report device-vs-OS-cache reads, bytes/read amplification and p50/p95/p99 stalls. Target PLE p95 stall below 1% of token latency after warmup without an unbounded cache. |
| **Prompt processing / chunking** | Strata uses a 1,024-token chunk in this profile and a measured overlapped expert-streaming path. ESE requested ubatch 1,024 but failed allocation; 128 succeeded. | Add a memory-budgeted chunk selector that accounts for per-GPU graph scratch, KV/indexer, active experts, PLE buffers and multi-GPU staging before allocating. Fuse Qwen's PLE and expert work where parity allows; pipeline next-layer data and double-buffer inputs. Back off chunk size predictably rather than failing after partial setup. | Prompt logits match for all chunk sizes; no memory overcommit; compare TTFT and prompt tokens/s at 1K/4K/32K/128K prompts, cold/warm PLE, and fixed memory budget. Beat the 1,024-chunk Strata result or provide higher throughput at lower peak VRAM. |
| **Multi-GPU placement** | Strata's auto K=4 split follows estimated layer time, free VRAM, cache coverage and a once-per-window handoff. | Use ESE's device/resource planner as the foundation for a Qwen-aware partitioner. Optimize for predicted end-to-end window time including heterogeneous layer kernels, expert hit probability, KV/PLE traffic, peer/host link bandwidth, graph scratch and draft head placement. Re-plan only at safe request/session boundaries; expose the reason for each split. | Per-device utilization and transfer timings; parity across splits and fallback paths; no hidden CPU-only layer. Sweep single GPU, pair order, split point, and concurrent loads; exceed the best Strata split on matched GPUs, not merely the auto choice. |
| **Decode graph and kernels** | Strata captures verify/draft windows and uses model-specific native kernels. ESE reports graph reuse but the probe graph had 6,470 nodes / 213 splits; no direct timing attribution exists. | First reduce graph fragmentation and host dispatch; then capture stable shapes (including a small set of verify-window sizes), fuse GR/QSA/MoE/PLE steps only where the oracle agrees, and specialize QSA block gather/attention and quantized expert kernels to the actual IQ1_M tensor mix. Keep a simple reference path for each fused operation. | CPU/CUDA operator and whole-model parity, sanitizer/stress coverage, cold graph-build and warm replay timings. Compare kernel time, launch count, p50/p95 token latency and energy/token; retain a fusion only when end-to-end measurement improves. |
| **Memory budgeting and adaptation** | Strata has cache budgets, per-stage expert slots and KV streaming controls; ESE already has a transactional resource planner, but Qwen needs one integrated plan. | Create one per-device governor for dense weights, expert cache, hot KV, PLE staging/cache, graph scratch and safety reserve. Have it optimize predicted saved milliseconds per byte, with hard RAM/VRAM/pinned-memory ceilings. Adapt at safe boundaries when context grows or other sessions arrive; reserve before publication and roll back atomically on allocation failure. | Accounting agrees with allocator telemetry; forced-OOM and cancellation tests leave no partial state; stable 30-minute and multi-session runs. Compare throughput and peak memory at the same budgets, not only unconstrained best cases. |
| **Telemetry, controls and reproducibility** | Strata logs expert hits, KV block hits, MTP/suffix acceptance, split planning and prompt/decode time. ESE has PLE/expert/resource counters but they are not yet a unified Qwen report. | Emit a per-request cost ledger: target passes/tokens, draft acceptance, expert hits/misses by device, QSA block hits/transfers, PLE source/cache state, queue waits, graph dispatch, per-GPU memory, and total timing. Record engine revision, artifact hashes, tokenizer/input IDs, sampling, clocks and cold/warm state. Provide safe auto mode plus opt-in overrides. | A replay bundle reproduces the request and all counters; ablations change one policy at a time. Publish median and tail across repeated runs, correctness first, and no speed claim based on unlike prompts/configs. |
| **Vision and capability coverage** | Strata's product supports vision; this matched local text run disabled vision. ESE's Qwen4Exp plan keeps vision as a separate promotion gate. | Complete text correctness/performance first, then implement multimodal positions, image encoder and cache accounting as a separately advertised capability. Share the global GPU budget so image buffers do not silently displace KV or expert residency. | Image/text reference parity, lifecycle and multi-GPU tests, memory-pressure benchmark. Claim support only when runtime capability detection and tests pass. |

### Deferred experiments supplied by the user (2026-10-09)

These five experiments are queued for later. Continue the current decode
profiling, bounded cost-aware GPU quota and placement qualification, then ordinary
CPU cross-expert scheduling with bounded RAM leases and inspection of the existing
concurrent loader. Investigate **#1 first**, fold **#2 into validated speculation
work**, and treat **#3 as the deeper execution experiment**; #4 and #5 remain
secondary screens. None of the five experiments below has been completed by this
checkpoint. Reuse shared ESE components and preserve the other model adapters.

1. **Retain experts by waiting time saved per byte.** This extends D1/D8 and the
   saved-critical-path-time residency proposal below. Current code has opt-in
   age-decayed frequency retention, byte-cost accounting, wait telemetry,
   residency-aware placement within a fixed GPU quota, and layout packing. It
   does not establish separate latency-value retention policies for RAM and VRAM.
   Score reuse probability against the loading/upload delay that cannot overlap
   useful work, including the CPU execution alternative. A frequent cheap CPU
   expert may be worth less VRAM than a rarer expert whose upload stalls the GPU.
   First replay fixed routes with a future-informed cache at the same byte budget,
   starting state, admission constraints and loading concurrency. Variable-sized,
   weighted experts require a stated optimizer or upper bound; ordinary
   farthest-next-use eviction is not automatically optimal for saved time.
   Report milliseconds saved and requested bytes alongside hit rate; reject a
   complicated predictor if this screen leaves little latency headroom. The
   latest packing screens had expert RAM cache capacity zero, so investigate RAM
   retention separately with an explicit bounded RAM budget and measured storage
   reads rather than assuming host accesses were SSD misses.

2. **Choose speculative depth by accepted tokens per millisecond and expert
   traffic.** The general speculation tuner already rewards complete-step
   throughput (`accepted drafts + one target token` divided by elapsed step time),
   including drafting and verification, and supports a target-only arm. See
   [`../PHASE3_TRANSIENT_SPECULATION.md`](../PHASE3_TRANSIENT_SPECULATION.md).
   The additional experiment is to associate each depth with verification time,
   accepted tokens and incremental expert bytes, including rejected-draft
   traffic. Sweep fixed depths first and compare achievable adaptive performance
   with the best fixed setting on held-out workloads. Count route prediction and
   selector overhead in total step time; add route/traffic-aware selection only
   if there is meaningful headroom. General tuner availability does not qualify
   Qwen4Exp MTP: its verifier, sampling acceptance and recurrent/PLE/QSA rollback
   still need their own correctness evidence (B4/B6, D7).

3. **Compute an expert while its weights arrive.** Cache packing and tiled CUDA
   matmul do not implement this storage-to-compute pipeline. Store matching
   intermediate-channel tiles of gate rows, up rows and down-projection columns;
   load each tile, compute its output contribution and accumulate while later
   tiles arrive, using bounded tile buffers. First benchmark one expert using
   actual production quantization across several tile sizes. Compare total
   load-plus-compute time with the existing overlapped path at the same memory
   budget. Include I/O, staging, kernel launches and output accumulation. Reject
   a design whose small-kernel or down-layout costs consume its overlap benefit.
   Check quantization block alignment, weight reconstruction and output error:
   unchanged weight bytes do not guarantee identical floating-point results when
   reduction order changes (D6, C1/C3).

4. **Compile SSD placement for actual RAM cold misses.** Existing extent
   descriptors, batched reads, cache indexes and loader machinery are useful
   foundations; they are not evidence of a layout learned from co-missing expert
   requests. Trace misses after RAM retention decisions and place frequently
   co-missing experts near one another so useful adjacent requests can merge.
   Request only needed expert bytes rather than entire groups. Train layouts on
   one trace set and replay held-out traces at fixed cache size, queue depth and
   requested bytes; report physical read amplification, request count, waits and
   throughput. Check whether existing large reads and concurrency already
   saturate the SSD, and whether a coding-trained layout transfers to other
   workloads. Reject layout complexity without measurable storage headroom.

5. **Screen exact shared bases plus lossless residuals.** No validated
   cross-expert base/residual storage path was found in this inspection. First
   run an offline compression study on representative aligned quantized blocks;
   reconstruct the original quantized representation before computation and
   require byte-exact reconstruction. Count base storage, residuals, metadata,
   random-access amplification, decompression time, staging memory and extra
   memory traffic. Low-bit weights may offer little useful redundancy even when
   experts appear similar. Reject this quickly if reconstruction costs exceed
   the transfer savings; integration depends on passing that screen.

For all five, preserve selected expert IDs/weights, declare arithmetic changes,
and separate feasibility screens from repeated parity-qualified engine speedups.
The production architecture gate and existing release checklist remain in force.

### Source-reviewed gap priorities and shared-model ownership (2026-10-05)

The goal is independent ESE improvements, not copying Strata's implementation
or replacing ESE's multi-model engine with a Qwen-only engine. Source inspection
confirms the mechanisms below; their individual speed contributions still
need matched ablations after correctness. Existing general ESE capabilities
must not be described as absent merely because this Qwen probe does not use them.

| Priority / source evidence | Actual gap | Independent ESE improvement and acceptance |
| --- | --- | --- |
| Target-work amortization: Strata `src/core/mtp.cpp` captures draft steps; `verify.cpp` checks windows and requires the profile-filled expert tier. | ESE has general speculation, but no validated Qwen4Exp draft/verifier/state rollback integration. | Reuse ESE's verifier/checkpoint infrastructure; choose depth using accepted tokens per total draft+verify cost. Disable stages that lose time. Gate on identical target decisions and complete recurrent/PLE/QSA rollback (B4/B6, D7). |
| Expert execution: Strata `session.cpp:706–727` enqueues GPU hits before CPU misses and combines the same layer's outputs. `expert_cache.cpp` has profile-ranked/no-eviction residency paths. | ESE already has adaptive GPU cache, RAM leases, async loaders and calibrated hybrid routing; the Qwen diagnostic cache is disabled. | Integrate existing cache rather than build a second one. Rank saved critical-path nanoseconds per resident byte, accounting for admission/upload cost and workload drift; protect in-flight entries and use hysteresis. Prove routes/logits and no duplicate or stale work; measure p95 miss waits, churn and end-to-end throughput, not hit rate alone (D1). |
| QSA memory: Strata `layer.cpp:495–528,627–681` implements a bounded GPU working set with authoritative pinned-host backing. | ESE's Qwen graph builds a history-width sparse mask and calls standard attention; sparse selection is not evidence of compact selected-block work or bounded residency. | Independently implement ordered selected-block gather/attention plus per-layer hot blocks and exact host backing. Preserve tail, position, mask, sequence and rollback semantics. Optimize bytes actually read/transferred; adversarial old-block access must remain correct (B2/B4, D2/D6). |
| Host dispatch: Strata `session.cpp` has captured pre/post and token graphs, measured host-idle intervals and overlap controls. | ESE already reuses CUDA graphs, but this Qwen probe has many graph splits; split count alone does not establish cost. | Instrument host enqueue gaps and critical-path waits first. Fuse/capture only measured costly boundaries, with reference fallbacks and graph/non-graph parity; no cross-layer stale-output shortcut (D6). |
| Placement/prefill: Strata logs a 4/44 layer split and once-per-verify-window handoff; prefill has separate staging. | ESE diagnostic uses equal layer split and ubatch 128; dense/expert/KV/scratch budgets interact. | Extend the existing calibrated resource planner with layer/window costs, link traffic and cache coverage; reserve scratch before publication, choose safe chunk sizes and reconsider only at safe boundaries (C1/C2, D4/D5). |

**Preserve the other models:** `../PORT_ROADMAP.md`,
`../V2_HARDWARE_ADAPTIVE_MOE.md` and `../DEEPSEEK_V41_SUPPORT_PLAN.md` retain
ownership of their existing capabilities. Shared cache/planner/scheduler changes
must run their existing regression suites alongside Qwen tests. Report dense
attention, conventional MoE, recurrent/hybrid and DeepSeek/DSA/MLA/MTP paths
separately: unsupported/unavailable is not passed. Real-model coverage must use
the launcher's actual installed artifacts; do not invent model availability or
silently replace slots. Qwen-specific GR/PLE/QSA/MTP math stays in its adapter
and behind its architecture gate. Do not change other model defaults to improve
a single Qwen benchmark.

**Latest diagnostic evidence, not a speed-win claim:** durable ESE bundles
`baseline-ese-decode-warm-20261005.json` and
`baseline-ese-decode-warm-run2-20261005.json` record 7.66/7.95 tok/s with host KV;
`baseline-ese-decode-warm-kvresident-20261005.json` records 15.12 tok/s with full
GPU KV. These use host MoE and no adaptive expert cache. The resident result is
one sample, output logits remain incorrect/unqualified, and Strata's historic
43.5 tok/s used a different policy plus speculation. They establish a residency
investigation priority, not a correctness-qualified speedup. PLE's recorded
input wait remains a small part of these short-prompt runs, so it is not the
first decode bottleneck to optimize.

**Benchmark correction:** removing Strata's `--kv-resident` selects fully
GPU-resident KV (`kv_plan` returns its default when the resident count is zero).
It does not select host-only KV. The harness now refuses the unsupported Strata
host-only toggle instead of making a false matched-policy comparison. The
interrupted Strata benchmark left an owned service running with only 176 MiB
free; it was stopped after process identity verification. No completed request
or matched Strata baseline was recovered. A6 remains open until the replay
harness, policy labels and durable end-to-end evidence are validated.

### Implementation order and definition of “one-up”

1. **Freeze correctness and baseline:** pin the source/model/tokenizer revisions;
   finish CPU/CUDA full-model parity for logits, PLE row IDs, QSA block IDs,
   MoE routes, recurrent state and sequence lifecycle. Keep the normal loader
   rejection in place. Re-run the current 128-token diagnostic only as a
   historical reference, not as a passing parity baseline.
2. **Remove the immediate resource gap:** make ESE's 1,024-token prefill
   request either fit through planned scratch/lifetime reductions or degrade
   safely to the largest measured chunk. Establish matched ESE decode and
   prefill baselines with identical context, Q8 KV, GPU split and expert
   budgets.
3. **Integrate high-value tiers:** unify the resource ledger, add adaptive
   expert residency and the bounded sparse-aware GPU↔pinned-host KV tier, then
   overlap the existing PLE row I/O with independent layer work. Benchmark
   each feature alone and in combination; prevent one cache from stealing
   another's guaranteed working set.
4. **Recover algorithmic throughput:** port MTP and suffix drafting behind an
   exact verifier, then tune acceptance-aware draft depth. This is an optional
   correctness-sensitive feature, never a shortcut around target-model parity.
5. **Tune the whole system:** improve dynamic prefill sizing, graph/kernel
   fusion and topology-aware multi-GPU placement using per-request telemetry.
   Add vision only as its own gated milestone.

“One-up” should mean a demonstrated Pareto improvement, not more flags: equal
or better output correctness/quality and reliability, then better decode or
prompt throughput at the same resource budget—or the same throughput with
lower p95 latency, VRAM/RAM use or NVMe traffic. For any feature-level win,
compare cold and warm states and disclose regressions elsewhere. The end-to-end
target is to exceed Strata's 43.5 tok/s on this exact 128-token diagnostic
*after* matched feature settings and parity; it is not a promised result.

The existing release and parity gates in
[`../V0_2_1_QWEN4EXP_PLAN.md`](../V0_2_1_QWEN4EXP_PLAN.md) remain authoritative.
All designs above are proposals; none opens the production architecture gate.

### Launcher integration target (the Strata slot)

The launcher is the real acceptance surface, so its Strata slot defines the
contract an ESE Qwen4Exp backend must meet to be a drop-in. From
`/home/xero/Desktop/test/model_slots.sh` and `strata/strata_service.sh`,
`strata-coder-iq1m` is a *native service* (not a llama-server GGUF profile):

- `NATIVE_SERVICE[strata-coder-iq1m] = strata/strata_service.sh`, base URL
  `http://127.0.0.1:8080/v1`, empty API key;
- lifecycle verbs `start|stop|restart|status`, with `start` blocking until
  `/health` is ready (engine plus arena load can take minutes);
- `GET /health` returns
  `{"status":"ok","model":"qwen3.8-flash-next-coder-iq1_m","max_context":N,"loaded":true,"service":"strata"}`,
  and the wrapper treats the session as ready only when `status==ok` *and* the
  model id matches *and* `loaded` is true;
- the server exposes OpenAI `/v1/chat/completions` **and** Anthropic
  `/v1/messages` (plus `/v1/models`, `/v1/status`, `/models`, `/props`,
  `/slots`, `/mcp`) on one port, which is what lets launcher option 2 point
  Claude Code at Strata with no key;
- launcher slot label: "Strata Coder-Next 125B-A0.8B IQ1_M (native engine,
  128K, OpenAI+Anthropic on :8080)".

ESE's serving surface is already capable: `examples/server` implements both
OpenAI chat/completions and Anthropic `/v1/messages` (with Anthropic thinking
blocks), so the wire protocol is not the gap. The gaps are (1) the Qwen4Exp
graph is not production-loadable and its logits are known-incorrect, so the
launcher deliberately exposes only Strata for this model and ESE Studio warns
"The Qwen4Exp NVMe test profile is validation-only; its current output logits
are incorrect"; and (2) there is no native-service wrapper/slot that presents
ESE as `strata-coder-iq1m`'s peer.

Concrete drop-in acceptance criteria (to build once correctness lands):

- a service wrapper with the same `start|stop|restart|status` verbs and a
  readiness gate that reports ready only when the model is actually loaded;
- a `/health` (or equivalent) JSON a launcher can verify (`status`, `model`,
  `max_context`, `loaded`);
- OpenAI plus Anthropic on a single port with an empty/placeholder key for
  local use;
- a launcher slot declared as a `NATIVE_SERVICE` entry (with its own marker
  GGUF), so `switch_model.sh`, the picker and the profile syncers treat it like
  any other backend.

This is an integration requirement, not a performance claim, and it does not
open the production architecture gate.

Historical note: the upstream PR #27742 snapshot added no new GGML/CUDA
operation and used the stock MoE path. That snapshot observation does not
describe current upstream kernel support. The Aug 26 ESE comparison measured a
fused indexed IQ MoE gate/up/down path in this fork; broadcast gate/up
activation deduplication then left all 48 MoE and 12 QSA route digests unchanged
and measured 7.68-7.71 tokens/s versus 7.67 without it. Keep those dated
measurements as historical evidence, not as a claim about current upstream.

## Next correctness work

Before changing ESE's hard architecture gate, pin the upstream reference commit
and compare the experimental ESE graph on identical inputs. The upstream
history includes correctness-sensitive changes after the initial architecture
merge (notably #27941 and later Qwen4Exp graph/MTP work). The first ESE checks
should compare per-token logits, QSA selected-block IDs, MoE expert IDs and
weights, and recurrent state across prefill, single-token decode, chunked
prefill, sequence copy/remove, and cache save/restore. Include the production
IQ1_M artifact plus an independent fixture. Keep vision and MTP out of the
initial gate. Record any mismatches as failing parity evidence; do not expose
the local-test bypass through the normal launcher or Studio UI.

A first, intentionally narrow upstream comparison now covers synthetic Gated
Residual, PLE, QSA-indexer, MoE-routing, Gated DeltaNet recurrence, and
output-head logits. `tools/qwen4exp_upstream_parity.py` executes the RMSNorm,
Gated Residual, PLE layer, QSA indexer, top-k router / sparse MoE block, and
Gated DeltaNet recurrence functions extracted from the pinned Transformers
source, after checking its SHA-256, and writes
`operator-reference-fixture.json` with an integrity digest. The `moe` section
records the router's expert IDs and normalized routing weights (plus the block
output), `recurrent` records the chunked and single-step Gated DeltaNet
outputs and final recurrent state, and `logits` records the output-head
projection per token — the pinned `Qwen4ExpForCausalLM` applies `lm_head` to
the final hidden state with no extra normalization. These are module-level
goldens; end-to-end prompt logits over a real prompt still need runnable
weights and are matched under the B3 gate. The QSA
case gives the indexer an identity projection, ESE's effective key-norm gamma,
and identity rotary tables, and records the indexer's internal top-k through an
observation-only `Tensor.topk` wrapper; generation is seeded and byte-reproducible.
Regenerate and verify it with:

```bash
python3 tools/qwen4exp_upstream_parity.py \
  --output docs/qwen4exp/operator-reference-fixture.json \
  --verify docs/qwen4exp/operator-reference-fixture.json
```

The `test-qwen4exp-ops` CTest consumes that fixture and compares ESE's CPU GR
mixed input, injection, block residual output, final stream mixer output, PLE
gated value, normalized value, combined PLE output, and next convolution
state. For the QSA indexer it additionally compares per-block scores of every
complete block, the exact selected-block rank order (query 1 selects blocks
2 then 1, so ordering is non-trivial), the selected-token sequences including
the causal tail, and the selected-token mask visibility pattern against ESE's
sparse causal mask. Independent scalar checks remain for pooled keys, the raw-key
cache, mask unions, and QSA details the fixture does not pin.

Three scope limits are explicit. The fixture uses identity rotary tables, so it
pins pooling, normalization, scoring, top-k, tail handling, and mask
construction but not the RoPE rotation itself, which is standard GGML machinery
exercised by the production path and upstream tests. Visibility is a contiguous
causal prefix; pinned Transformers pools visible-order tokens into blocks, so
non-contiguous padding masks would diverge from ESE's aligned block pooling and
are deliberately out of scope here. And the comparison runs on the CPU
operator graph; the CUDA subgraph mode still covers the GR case only. This
test uses small deterministic synthetic tensors; it does not exercise
loader/model integration or establish full-model logits, route,
cache-lifecycle, CUDA, or quality parity. Keep the architecture rejection and
all broader correctness gates in force.

Once parity is clean, rebuild the ESE server from this exact worktree, verify
that it parses the shared Q2_0-containing GGUF, then stage the two existing
Strata shards read-only as the ESE input. Run matched cold/warm baseline tests
before comparing optimizations. The architecture gate remains closed until
those results and the existing resource, long-context, and recovery gates pass.

Production prompt fills showed that native io_uring should also use exact-row
storage. In a 511-token fill it reduced PLE wait from 1.132 s to 0.756 s and
raised prompt throughput from 24.70 to 25.52 tokens/s. In an 8,191-token fill,
source traffic fell from 546,861,056 to 11,794,410 bytes, PLE wait fell from
136.041 to 91.942 seconds, and prompt throughput rose from 19.26 to 21.47
tokens/s. The fill plus terminal decode completed with zero failed reads,
cancelled batches, or swaps and 43,451,784 KiB peak RSS. The page profile had
only 284 page hits and 117,127 evictions, while both profiles had just 22 row
hits; this is measured random-row behavior, not a synthetic locality
assumption. Queue width 64 and eight workers were separately rejected because
neither reduced wait or improved throughput.

The same batched-prompt work exposed a correctness defect in generic indexed
MoE CUDA execution. Routes owned by another backend were left as sentinels in
the compact map, and the quantizer could read them as row indices. The compact
maps are now initialized and the broadcast quantizer skips inactive routes.
Large indexed IQ tiles then failed CUDA function configuration on the mixed
`UD-IQ1_S` model's IQ2_XXS tensors, so prompt batches larger than eight use the
existing per-expert CUDA path while small batches and decode retain indexed
MMQ. Normal graph-enabled 63-, 511-, and 8,191-token fills all completed after
that policy; the prior 8K illegal-access failure is no longer reproducible.

GPU sampling made the utilization report more specific. The full-depth profile
creates 104 backend graph splits per token, and layer-split execution serializes
the three devices around host-resident expert work. Individual assigned device
segments reached 63-100% SM utilization while the other devices waited; low
aggregate utilization therefore does not mean that all three GPUs have idle
compute available for the same dependent token step. This points to eliminating
GPU/host expert boundaries or overlapping bounded expert staging as the larger
engine target.

One inherited CUDA policy also lagged current upstream: this fork enabled CUDA
Graphs only on Ampere and newer, while upstream permits Volta and newer. Lowering
the cutoff enabled graph capture on the RTX 2080 SUPER. A same-binary opt-out
(`GGML_CUDA_GRAPHS_AMPERE_ONLY=1`) retained the old policy for controlled A/B.
Four recurrent decode steps produced identical hashes for every MoE and QSA
route tensor with the old and new cutoffs. At a safe 1 GiB fit margin, a
128-token comparison improved from 7.21 to 7.25 tokens/s; after subtracting
measured PLE wait, the engine-only rates were 7.372 and 7.406 tokens/s. The gain
is valid but modest because only the Turing segment changes. A 256 MiB fit
margin exhausted device memory when callback-heavy repeated probes accumulated
CUDA Graph instances, so the lower margin is not promoted for this profile.

The graph-instance follow-up found that this fork keyed CUDA Graphs by the
first scheduler-node pointer and retained every key for the context lifetime.
An 8,191-token prompt therefore ended with 1,609 entries on GPU 0, mostly from
one-shot backend splits across successive ubatches. Current upstream has both
stable-shape warmup and timed eviction. Stable-shape warmup was rejected on
this engine because it reduced the 511-token prompt from 25.52 to 24.26
tokens/s. Porting only the five-second sweep / ten-second expiry policy, plus
correct destruction of each graph's copy-indirection allocation, preserved
one-shot capture while bounding stale entries. The same 8,191-token prompt
ended with 497 entries and improved from 21.47 to 23.06 tokens/s; peak RSS fell
from 43,451,784 to 43,219,964 KiB. A 32-token decode improved from 1.37 to 1.58
tokens/s and retained 42 rather than 83 GPU-0 executables. The same-binary
opt-out `GGML_CUDA_NO_GRAPH_EVICTION=1` preserves the old policy. Four
recurrent evaluations produced identical hashes for all 48 MoE and 12 QSA
tensors (16,224 route values), and both modes completed without read, CUDA, or
swap failures.

A subsequent 32K fill was intentionally stopped after 28:56 at the user's
request because generation remained near the documented 1-2 tokens/s range.
It had 43,335,416 KiB peak RSS and zero swaps when interrupted, but produced no
final prompt timing or teardown telemetry and is therefore not completion
evidence. The filled-32K and 64K gates remain open.

## Studio capability presentation

Studio discovery now derives a runtime-support descriptor from inspected GGUF
architecture metadata rather than from the filename. A `qwen4exp` model is
grouped under `Qwen 3.8 · Flash Next`, carries an explicit `Experimental`
badge, reports `Text validation pending`, `Vision unsupported`, and
`MTP unsupported`, and cannot start, enter the sweeper, or advertise a live
endpoint while the production loader gate is closed. Status is conveyed in
text as well as color, and native disabled controls enforce the presentation.

Split-GGUF discovery also recognizes the `-00001-of-00003` convention. Only
the first shard is a launchable model entry and the shard suffix is removed
from its display name. A Rust discovery test against the maintainer's real
three-shard directory proves it produces one `qwen4exp` profile with the
262,144-token metadata context and the gated capability policy. The complete
27-test Rust suite, 15 frontend behavior tests, TypeScript compilation, and
the Vite production build pass with the staged CUDA runtime.

The gate-restored CUDA and AVX2 builds each passed all ten focused Qwen4Exp,
expert-cache, and resource-planner CTests; the six content-addressed Python
reference-trace tests also passed. No temporary scheduler/runtime debug markers
remain in the source.

This proves production tensor validation, authoritative PLE shard staging,
one-token CPU/CUDA graph executability on Turing and Ampere, short-batch
same-build logit/route tolerance on a three-GPU layer split, deterministic
full-state and per-sequence save/restore for the documented short cases, and a
30-minute bounded-resource decode soak with separately verified graceful
cancellation. It does **not** prove generation quality, a filled long-context
run, general prompt-cache compatibility, or a tolerance-clean higher-offload
profile. The temporary probe-only loader changes were removed immediately
afterward; the normal unsupported-architecture probe continues to assert
deterministic rejection.
