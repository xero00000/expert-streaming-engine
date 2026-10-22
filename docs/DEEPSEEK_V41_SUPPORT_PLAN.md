# DeepSeek V4.1 Flash: local ESE support and tuning plan

Started September 10, 2026. **Native trained-weight text generation now works
on the three local GPUs.** Seven successful runs cover five scoped quality
checks and an initial cache sweep. The corrected mixed-Q2/affine2 package runs
at about 0.7 decode tokens/s on the tested explanation prompt, using about
22.4 GiB peak host RSS with the conservative cache settings.
See [the local run guide](deepseek-v41/RUN_LOCAL.md) for commands and limits,
and [the continuation record](deepseek-v41/CONTINUATION.md) for the implementation
history, validation and upstream weight-conversion repairs. The stages below
describe the support plan; larger-context tuning, batched prefill, DSpark,
vision and chat-server integration remain future work.

## Local baseline

Use `/home/xero/Desktop/test/ESE-current`, resolving to
`/home/xero/Desktop/test/ESE-v0.2.1-qwen4exp-plan`. HEAD is `ee86cae61`,
one commit ahead of the locally recorded `origin/main`, with substantial
uncommitted runtime/CUDA/storage work. No remote refresh was needed to select
this checkout; this is not a claim about today's live GitHub HEAD.
`/home/xero/Desktop/test/ESE` is a different, older desktop base with Android work.

The existing Release build is `build_cuda_release` with CUDA architectures
75 and 86. Hardware observed: RTX 3060 Ti 8 GiB, RTX 2080 SUPER 8 GiB,
RTX 3080 10 GiB, approximately 46 GiB system RAM. Aggregate VRAM is not one
contiguous allocation. NVMe is `/dev/nvme0n1p3`; rotational models remain archival.
The preexisting launcher and expert-cache work was retained.

Reuse the local retained expert slots, reserve/load/publish cache transactions,
bounded load queue, mixed-layout correctness fixes, and Qwen PLE row/page-store
infrastructure. Do not discard these changes by starting from public main.

## Measured checkpoint and bottleneck

Official repository: [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/tree/dba1be0a40aa45a94ad051997016db3960a90277).
Pinned revision: `dba1be0a40aa45a94ad051997016db3960a90277`.
`deepseek-v41/provenance.json` records hashes of the config, tensor index, and
reference source. All 48 safetensors headers were read with bounded HTTP ranges;
96,085 tensor extents agree exactly with the index's total_size.

| Component | Stored bytes (decimal GB) | Initial treatment |
| --- | ---: | --- |
| Routed expert weights and scales | 288.778 | Preserve source FP4 semantics; stream expert bundles |
| Engram weights and scales | 202.758 | Keep row-addressable on NVMe; bounded row/page caches |
| Other backbone tensors | 9.847 | Inventory placement and conversion expansion before allocation |
| DSpark | 7.933 | Defer speculative execution until ordinary decode is correct |
| Vision/aligner | 0.971 | Defer image execution for first text-only milestone |
| Total | 510.286 | Exceeds current NVMe free space |

At inspection, free space was 338.948 decimal GB (315.670 GiB): the full source
payload alone needs another 171.338 GB, plus headroom. Excluding vision and
DSpark still leaves approximately 501.383 GB. No files were deleted or large
weights downloaded. A full NVMe run needs more space or a separately validated
smaller representation. Avoid a blind source-plus-GGUF conversion that doubles
storage, and never expand the entire Engram table to FP16.

**Smaller-quant update:** a structurally verified mixed-Q2 GGUF is 168.930 GB,
but omits Engram. Pairing it with existing affine-2 Engram rows (61.442 GB)
would total **230.372 GB**, which fits current NVMe space. This mixed recipe
is a candidate, not a validated complete model. Its IQ2_XXS/Q2_K expert kernels
and 80-byte Engram row decoder now have local checks. Source-precision storage
numbers below remain the original baseline, not the only deployment option.

The user also authorized full source weights on HDD for custom quantization
when needed. `/mnt/games_sdb2` has about 2.2 TiB free; the other mounted HDD has
901 GiB. Prefer the larger drive for source-plus-conversion staging. Full weights
have not been downloaded; this milestone needed only small bounded samples.

Each of 15,360 routed experts occupies 18,800,640 bytes including scales.
Six selections across each of 40 main layers imply **4,512,153,600 bytes/token**
when all selected experts miss. This is a batch-one traffic scenario, not a
prediction of route locality or throughput. Batch unions, CPU/GPU placement,
shared experts, activation work and prefill have different costs.

Each Engram row is 256 E4M3FN bytes plus eight E8M0 scale bytes, one scale per
32 values. Two layers each select 24 rows: only 12,672 logical bytes/token.
The source stores weights and scales separately. Co-pack them losslessly into
264-byte rows in a sidecar to avoid two independent page reads per lookup.
Handle cross-page rows explicitly. Random-page amplification and latency matter
more than logical byte bandwidth here.

## Initial storage tuning evidence

`deepseek-v41/io-probe.json` records three repeats per configuration over a
private 512 MiB incompressible NVMe fixture, removed on completion. The probe
uses buffered pread and per-file DONTNEED advice, not global cache dropping.
Process physical read bytes matched logical bytes for these runs; device cache
is uncontrolled. The probe does not execute ESE, GPU transfers, or model routes.

| Read workload | Workers | Median GB/s | Median per-run p95 read service ms |
| --- | ---: | ---: | ---: |
| 18.8 MB expert bundle | 1 | 1.326 | 14.504 |
| 18.8 MB expert bundle | 2 | 1.816 | 20.939 |
| 18.8 MB expert bundle | 4 | 1.856 | 42.227 |
| 4 KiB random page | 1 | 0.049 | 0.081 |
| 4 KiB random page | 2 | 0.092 | 0.084 |
| 4 KiB random page | 4 | 0.169 | 0.093 |

Start expert-load experiments at two workers; compare against the current
synchronous default on the real model before changing any default. Four workers
added little bulk throughput and increased service latency. Test an independent
four-worker small-read queue for Engram; avoid its requests waiting behind
18.8 MB expert reads. Shared-device contention still requires a mixed test.

At 1.816 GB/s, 4.512 GB of misses would take about 2.48 seconds in this simple
storage-only calculation. This is not measured token latency. Cache locality
must be measured before claiming this model is practical on these GPUs.
Record GPU hits and host hits separately: a host hit avoids NVMe, not PCIe.

## Implementation order and acceptance criteria

1. **Inventory and primitive parity — started here.**
   `tools/deepseek_v41_preflight.py` inventories headers offline after an optional
   bounded fetch. It rejects missing/mismapped tensors, bad extents, mismatched
   totals and wrong revisions. `tools/deepseek_v41_engram.py` fetches individual
   real rows and decodes source FP8 plus scales. Twelve tests pass, including
   all 65,536 weight/scale combinations against Torch, signed zero/NaN handling,
   real rows from both tables, and refusal to read a non-range HTTP body.
   This is converter/reference groundwork; no native graph was registered.

2. **Exact Engram addressing and storage adapter.**
   Generate the compressed tokenizer map using pinned `inference/engram.py`
   normalization; assert the configured 99,092 compressed vocabulary size.
   Export hash multipliers and prime bucket offsets instead of assuming C++ RNG
   compatibility with NumPy. Trace 2/3/4-grams across chunked prefill, decode,
   sequence reset, image boundaries and rollback. Reuse checked extents and
   bounded leases from `src/llama-qwen4exp-ple-storage.*`, extracting shared
   infrastructure without inheriting Qwen hash semantics. Gate on exact row-ID,
   row-byte and decoded-value parity, cancellation and multi-sequence tests.
   Prefetch known prompt rows early and each generated token's rows before layer 1.

3. **V4.1 metadata and bounded converter.**
   Introduce a distinct architecture only with explicit experimental gating.
   Update `gguf-py/gguf/constants.py`, tensor mappings, `convert_hf_to_gguf.py`,
   `src/llama-arch.*`, hparams and tensor loading together. Parse nested
   `text_config`; preserve source/index layer IDs, candidate settings, Engram
   layout, epsilon, clamp, routing and shifted mHC metadata. Main layers are 40;
   the 43 compression entries include three DSpark stages, not 43 main layers.
   Reuse the V4 converter's `_pack_mxfp4_blocks` after independent nibble/scale
   parity tests. Process one expert/tensor at a time; preserve FP4 storage.
   For FP8 dense weights, use a supported execution representation with explicit
   memory accounting on SM75/86; do not assume native FP8/FP4 tensor-core support.
   Gate on a tiny synthetic complete fixture plus selected production tensors.

4. **Correct text graph before optimized attention.**
   Create a separate `src/graphs/build_deepseek41.cpp`, using
   `build_deepseek4.cpp` and `llama-dsv4.*` as references. Implement single-pass
   shifted mHC (pre_mix carries across sublayers), Engram residual gating,
   shared KV sources at layers 2/8/14/20, index sources at
   2/8/14/20/24/28/32/36, and candidate masking sourced at layer 20.
   Preserve sqrtsoftplus routing, unbiased mixture weights, 1.5 scale, and
   asymmetric SwiGLU clamps. Start with simple higher-precision KV at short
   context. Trace intermediate activations/logits against the pinned reference
   on initialized tiny weights, with layer-wise tolerances declared in advance.
   The downloaded reference loops all 40 blocks during forward; do not infer
   an encoder-only prefill scheduler from the card's 8B claim. Reconcile the
   optimized CED schedule with reference outputs before enabling that shortcut.

5. **Streaming correctness and first complete run.**
   Connect native expert cache transactions and Engram stores; compare resident,
   synchronous cache and queued-cache execution on a small complete fixture.
   Require teacher-forced logits and route parity through eviction, chunked
   prefill, SWA wrap, resets and multiple sequences. Stage full model only when
   the NVMe capacity plan is satisfied. Start at batch 1, 2K context, no DSpark,
   and one GPU for correctness; then validate two and three GPUs. Full dense
   residency of a small fixture is the parity control, not a claim that the full
   510 GB checkpoint fits RAM. Layer placement is the initial multi-GPU candidate;
   benchmark it against tensor splitting on this heterogeneous set.

6. **Tune against real requests.**
   Use explicit per-device budgets, first leaving at least 1 GiB free per GPU
   for scratch/KV/driver allocations and expanding reserves if measured peaks
   require it. Fit converted non-expert weights before assigning expert slots.
   Start host expert-cache sweeps at 8/16/24 GiB with 256/512 MiB Engram caches,
   subject to total host RSS, page cache and pinned-buffer budgets. Do not treat
   all free RAM as cache or pin the whole model. Compare expert workers 0/1/2/4,
   Engram queues separately, and bounded pinned staging. Record TTFT, p50/p95
   inter-token latency, exact paired output, route reuse distance, NVMe bytes,
   H2D bytes, host/GPU hits, stalls and peak per-device memory. Use five prompts
   and repeated paired runs with warm and file-evicted sources reported separately.

7. **Long context, optimized CED and KV compression.**
   Implement actual Full/Reindex/Reuse lifetimes and hierarchical candidate
   indexing, then FP4 global KV and bounded SWA replay. Require parity at
   compression and 128-token SWA boundaries and after prefix save/restore.
   Sweep 2K/8K/32K before larger contexts. The card's 890 bytes/token describes
   its optimized global KV; it is not the memory budget of the initial graph
   and excludes other state. Do not advertise million-token support from metadata.

8. **DSpark and vision.**
   Add these after ordinary text decoding is correct and profiled. The released
   reference supplies a DSpark forward path but not its speculative sampling
   loop. Three draft stages/block size five need their own verification and
   transactional rollback; current V4 single-predictor MTP is not interchangeable.
   Vision adds image embeddings, image-specific route bias and Engram boundaries.

## Reproduce this checkpoint

Run from the physical ESE worktree. Reference files live at
`/home/xero/Desktop/test/results/deepseek-v41-20260910/reference` and are pinned
by `deepseek-v41/provenance.json`. Full headers/source remain local research
artifacts; small reports and two row fixtures are included with this plan.

```bash
python3 tools/deepseek_v41_preflight.py \
  /home/xero/Desktop/test/results/deepseek-v41-20260910/reference
python3 -m unittest discover -s tests -p 'test_deepseek_v41*.py' -v
python3 tools/deepseek_v41_io_probe.py \
  --directory /home/xero/Desktop/test/results/deepseek-v41-20260910 \
  --preflight docs/deepseek-v41/preflight.json \
  --output /home/xero/Desktop/test/results/deepseek-v41-20260910/io-probe-repeat.json
```

The continuation now includes the native text graph, runtime, Engram storage,
tokenizer integration and a repaired trained-weight package. Actual generation
works across all three local GPUs. Arithmetic, explanation, code generation and
recall across the sliding-window boundary pass; an initial 8/12/16 GiB expert
RAM sweep produced identical text at about 0.71–0.74 decode tokens/s. The source
quant artifact required tensor realignment and replacement of 328 incorrectly
converted dense tensors from the original checkpoint. See
[the local run guide](deepseek-v41/RUN_LOCAL.md) for reproducible commands and
measured limits, and [the continuation record](deepseek-v41/CONTINUATION.md)
for implementation and validation details. Larger-context performance,
batched prefill, DSpark, vision and chat-server integration remain future work.
