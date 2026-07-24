# Expert Streaming Engine

**Run huge Mixture-of-Experts models on consumer dual-GPU boxes** by combining deferred mmap expert weights, hybrid GPU-resident MoE layers, and route-aware page-cache prefetch — without re-quantizing the model.

This repository is a focused fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (itself a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork). The goal is not a new quant format. It is **making F16 / MXFP4 MoE weights that do not fit in VRAM usable at interactive speeds** on low-VRAM multi-GPU hosts.

**License:** [MIT](LICENSE) · **Branch:** `chunked-gdn-port` · **Headline:** GPT-OSS 120B F16 · ~140 tok/s prefill · ~11.5 tok/s decode

---

## Headline result

**Model:** [GPT-OSS 120B](https://huggingface.co/openai/gpt-oss-120b) F16 GGUF (~61 GiB on disk)  
**Hardware:** dual Ampere (8 GiB + 10 GiB) · Ryzen 9 5950X · ~47 GiB DDR4 · NVMe  
**Daily-driver slot:** 64K context, Q8 K/V, hybrid CPU-MoE + 6 GPU-resident MoE layers

| Metric | Value | Notes |
| --- | ---: | --- |
| Warm prefill | **~139–141 tok/s** | 4.7K–22K token prompts; batch 1024 / ubatch 512 / 32 batch threads |
| Short-context decode | **~11.5 tok/s** | 48-token generations, multi-prompt median class |
| Long-context decode | **~8.9 tok/s** | At ~22K filled context |
| Needle-in-haystack | **PASS** | Exact retrieval at 27K tokens, 40% depth |
| Startup (deferred mmap) | **~13–16 s** | With `GGML_CUDA_NO_PINNED=1` (was ~505 s when pinned path forced) |

That is a **~61 GiB F16 MoE** running on **18 GiB combined VRAM**, with prefill competitive enough for agent workloads (Forge / coding agents) and decode in the low double digits.

---

## Why this exists

Standard “load the whole model into VRAM” fails here: GPT-OSS 120B F16 is ~61 GiB, and this host has 18 GiB of GPU memory. Full CPU offload of every expert is too slow for interactive decode. Re-quantizing to Q3 is blocked for this GGUF’s MXFP4 expert tensors without force-requantize quality risk.

**Expert streaming** treats each selected expert’s three projections (gate / up / down, ~12.6 MiB) as the paging unit:

```text
NVMe GGUF  ──mmap──►  host page cache / staging  ──►  GPU (dense + hot MoE tail)
                         ▲
                         │  router top-k → MADV_WILLNEED on selected slices
```

Only **4 of 128 experts per layer** are active per token. The engine:

1. Keeps dense weights + a small **GPU-resident MoE tail** on the two cards  
2. Maps the remaining MoE blocks with **`--defer-experts`** (mmap, not a 50+ GiB pinned allocation)  
3. Prefetches the **current router top-4** expert slices into the OS page cache  
4. Runs large-batch **CPU prefill** (PCIe weight-streaming prefill is far slower on this box)

---

## Benchmarks (measured)

GPT-OSS numbers below are from the isolated `:8014` harness and the live `:8000` launcher on the same dual-Ampere host. Full lab log: local `expert_streaming_lab/RESULTS.md` (methodology, raw JSON paths, rejected ablations). **Other models** on this host are summarized in [Fleet benchmarks](#fleet-benchmarks-same-dual-gpu-host).

### Hardware

| Component | Spec |
| --- | --- |
| GPU 0 | ~8 GiB Ampere |
| GPU 1 | ~10 GiB Ampere |
| CPU | AMD Ryzen 9 5950X (16C/32T) |
| System RAM | ~47 GiB usable DDR4 |
| Model store | NVMe (`gpt-oss-120b-F16.gguf`) |
| Combined VRAM | **18 GiB** |

### 1. Cold path vs warm path (why residency matters)

Early deferred-expert smoke (`--cpu-moe --defer-experts`, dense split across both GPUs):

| Request | Decode rate | Notes |
| --- | ---: | --- |
| First 5-token completion | **0.40 tok/s** | Cold expert pages |
| Identical second request | **26.7 tok/s** | Warm pages · **~67×** |

Storage alone: reading the 1.77 GiB of expert slices for one uncached token is **~1.2 s** on NVMe O_DIRECT — so pure AirLLM-style per-token streaming without a hot set is not interactive.

### 2. Deferred mmap + route prefetch (end-to-end)

Fixed 23-token prompt, `cache_prompt: false`, 48-token decode, `temperature: 0`:

| Configuration | Startup | 48-token decode | vs no-prefetch |
| --- | ---: | ---: | ---: |
| All 36 MoE layers deferred, no prefetch | 13 s | 1.31 tok/s | 1.0× |
| + current-route `MADV_WILLNEED` control | 13 s | **4.04 tok/s** | **3.1×** |
| First 30 deferred + 6 GPU-resident + route control | 13 s | **7.62 tok/s** | **5.8×** |
| 28 CPU-MoE layers + **8 GPU-resident** + route control (46/54) | 15 s | **9.41 tok/s** | **7.2×** |

Key environment fix: **`GGML_CUDA_NO_PINNED=1`**. Without it, CUDA’s pinned-buffer preference turns `--cpu-moe` into a ~58–59 GiB host allocation and readiness jumps from **~13 s → ~505 s**.

### 3. Multi-prompt decode (promoted hybrid)

Eight GPU-resident MoE blocks, 28 deferred (`--n-cpu-moe 28`), `--tensor-split 46,54`, current-route prefetch, `--scheduler-async`:

| Workload (48 tok, no prompt reuse) | Decode tok/s |
| --- | ---: |
| Coding | 7.23 |
| Reasoning | 7.30 |
| Support | 8.03 |
| **Workload mean** | **7.52** |
| Fixed long decode (async) | ~9.2–9.5 |

Async scheduling did not change the fixed-prompt peak much, but raised the **unseen multi-prompt mean** (~7.08 → **7.52** tok/s).

### 4. K/V cache sweep (1K fast slot) — Q8 wins

Same hybrid placement; repeated coding / reasoning / support medians:

| Profile | Workload median (runs) | Worst workload | Decision |
| --- | ---: | ---: | --- |
| F16 K/V, 46/54, 8 threads | 11.74 / 12.19 tok/s | 11.19 | Baseline |
| **Q8 K/V, 46/54, 8 threads** | **12.46 / 13.16 tok/s** | 11.11 | **Promoted** |
| Q4 K/V, 47/53 | 12.22 / 11.98 | 11.45 | Slower repeat |
| Q8 K/V, 47/53 | 12.31 / 11.41 | 9.94 | Worse low tail |

**Promoted Q8 average ~12.81 tok/s** vs **~11.96 tok/s** F16 K/V → **+7.1%**.  
Live validation (1K Q8 slot): warm short decode **15.8 tok/s**; coding / reasoning / support long decodes **9.5–11.5 tok/s**.

Nine resident MoE layers **do not fit** (10 GiB card needs ~10.2+ GiB for model storage alone).

### 5. Prefill sweep (64K agent slot) — **2.4×** from batch/thread flags

Geometry: 64K Q8 K/V, `--n-cpu-moe 30` (six GPU-resident MoE layers), warm 4,747-token prompt:

| Batch / ubatch / batch-threads | Warm prefill |
| --- | ---: |
| 128 / 64 / 8 *(old slot)* | 57.8 tok/s |
| 256 / 128 / 8 | 72.2 |
| 512 / 256 / 8 | 85.8 |
| 512 / 512 / 8 | 98.1 |
| 512 / 256 / 16 | 115.5 |
| 1024 / 512 / 16 | 120.1 |
| **1024 / 512 / 32** | **139.0 tok/s** |
| GPU weight-stream prefill (`offload-batch-size=8`) | 23.6–38.4 |

**Takeaways:**

- Prefill on this model is **CPU MoE** unless ubatch is huge enough for the CUDA offload heuristic; large ubatch amortizes expert weight passes.  
- Forced **GPU streaming prefill is ~6× slower** than CPU prefill (PCIe wall) — do not chase.  
- Decode threads stay at 8; batch threads only help prefill.  
- Validated 64K slot: **141.1 tok/s** sustained prefill over 22,438 tokens; **8.87 tok/s** decode at 22K depth; needle test passed.

### Improvement summary

| Change | Effect |
| --- | --- |
| Deferred mmap + `GGML_CUDA_NO_PINNED=1` | Startup **505 s → ~13 s**; enables true expert page release |
| Current-route expert prefetch | Decode **~1.3 → ~4.0 tok/s** on full-deferred path (**3.1×**) |
| Hybrid 8 GPU-resident MoE layers | Decode **~4.0 → ~9.4 tok/s** on fixed prompt (**~7×** vs bare deferred) |
| Async graph scheduler | Multi-prompt mean **+~0.4 tok/s** |
| Q8 K/V vs F16 K/V | **+7.1%** workload median (~12.8 vs ~12.0 tok/s) |
| Prefill batch/ubatch/tb flags | Warm prefill **57.8 → 139 tok/s** (**2.4×**) |
| GPU streaming prefill | **Rejected** (~6× slower) |
| Learned layer-0 route head in executor | **Rejected for default** (~1.36 tok/s; near no-prefetch) |
| Generic n-gram speculation | **Rejected** (6.12 tok/s vs ~9.4 hybrid) |
| Standard Q3_K_M conversion | **Blocked** (MXFP4 expert rows incompatible without force-requantize) |

---

## Fleet benchmarks (same dual-GPU host)

The numbers below are **measured on the same workstation** (dual ~8+10 GiB Ampere, Ryzen 9 5950X, ~47 GiB DDR4). Most daily drivers run on **ik_llama.cpp** with hybrid CPU-MoE / mixed quants; GPT-OSS uses **this expert-streaming fork**; Bonsai uses a Prism 1-bit fork; Gemma-4 uses a TurboQuant gemma4 fork. Decode rates are warm-path short generations unless noted. Prefill is listed when recorded in the launcher notes.

**Takeaway:** hybrid placement + good quants put **35B-A3B MoE models in the ~50–120 tok/s decode band** on 18 GiB VRAM; **120B F16** is interactive but slower (~9–13 tok/s) because it is much larger and still mostly CPU-MoE.

### At a glance

| Model family | Best daily decode | Context | Engine | Notes |
| --- | ---: | ---: | --- | --- |
| **Qwen3.6 35B-A3B VRAM13** (all-GPU quant) | **~122 t/s** | 256K | ik | Fastest 35B class (stronger compression) |
| **Qwen3.6 + transplanted MTP** | **~90–93 t/s** | 128K | ik | +15–19% vs mixed base |
| **Qwopus-Coder native MTP** | **~91–98 t/s** (code) | 128K | ik | Prose ~break-even (~76) |
| **Mixed q2k+imatrix fleet** (Qwen / Ornith / Agents / AgentWorld / Qwopus) | **~78–80 t/s** | **256K** | ik | Default daily class |
| **IQ4_XS uniform** | ~77 / ~65 / ~51 t/s | 64K / 128K / 256K | ik | Full-size quality path |
| **Bonsai-27B 1-bit** | **~53 t/s** fresh · **~10 @256K depth** | **256K all-VRAM** | prism | Only full-VRAM 256K dense option |
| **Bonsai-27B Q1_0 FAST** (1-bit) | **1283 t/s pp4K / 1095 t/s pp32K · 70.64 fresh / 36.62 @64K / 24.52 @128K** | **128K** | prism | `bonsai-128k-fast` · RTX 3080 only · no-thinking · ~2.54 GiB VRAM free · exact/tool gates pass |
| **Bonsai-27B Q1_0 3060 Ti** (1-bit) | **648.27 t/s pp4K / 565.09 t/s pp32K · 39.41 fresh / 18.37 @64K / 14.41 @98K / 11.92 @128K** | **128K** | prism | `bonsai-128k-3060ti` · RTX 3060 Ti only · no-thinking · ~0.99 GiB free at max-depth bench |
| **Q8_0 35B fleet** (max quality) | **~32–33 t/s** | 64K | ik | Heavy `n-cpu-moe 26` |
| **Gemma-4 26B-A4B Q8** | **~22 t/s** decode · **~273 pf** | 256K | tq | Needs `--no-op-offload` |
| **Laguna-S-2.1 IQ4_XS + Q4 DFlash** | **~13.48 code / ~11.45 tool t/s** | 8K | this fork | 3-GPU · 81% code draft acceptance · Poolside standard DFlash GGUF |
| **GPT-OSS 120B F16** (this fork) | **~11.5–13 t/s** · **~140 pf** | 1K–64K | expert-streaming | Huge F16 on 18 GiB VRAM |

### GPT-OSS 120B F16 (expert-streaming engine)

| Slot | Ctx | K/V | Placement | Prefill | Decode |
| --- | ---: | --- | --- | ---: | ---: |
| `gpt-oss-expert-streaming` | 1K | Q8 | 28 CPU-MoE + 8 GPU MoE · 46/54 | — | **~12.8 t/s** workload median · warm short **~15.8** |
| `gpt-oss-expert-streaming-8k` | 8K | Q4 | same 8 resident | — | **~5.6–9 t/s** class |
| `gpt-oss-expert-streaming-64k` | **64K** | Q8 | 30 CPU-MoE + 6 GPU MoE | **~139–141 t/s** | **~11.5** short · **~8.9 @22K** |

### Qwen3.6 35B-A3B (base)

| Slot / quant | Ctx | Offload | Prefill | Decode | Notes |
| --- | ---: | --- | ---: | ---: | --- |
| UD-IQ4_XS (`qwen36`) | 16K | `n-cpu-moe 4` | — | **~80 t/s** | Light hybrid |
| UD-IQ4_XS (`qwen36-64k`) | 64K | `n-cpu-moe 4` | **~1203 t/s** | **~77 t/s** | Max-speed IQ4 |
| UD-IQ4_XS (`qwen36-128k`) | 128K | `n-cpu-moe 8` | **~881 t/s** | **~64.7 t/s** | Was ~358/60 on Q3_K_XL |
| UD-IQ4_XS (`qwen36-256k`) | 256K | `n-cpu-moe 14` | **~626 t/s** | **~51 t/s** | Full window IQ4 |
| **Mixed q2k+imatrix** (`qwen36-mixed`) | **256K** | `-ot` layers 13–26 → CPU | — | **~77–78 t/s** | **Default** · PPL 2.4195 vs Q8 2.4053 |
| Mixed tight (`qwen36-mixed-max`) | 256K | fewer CPU layers | — | **~80 t/s** | Solo use · tight GPU1 |
| **VRAM13 q2ex+imat** (`qwen36-vram13`) | **256K** | **all-VRAM** | — | **~122 t/s** (was ~129 @short) | ~5% drop at full 256K KV |
| Phone ufs-q2ex-q40 | 64K | all-VRAM | — | all-VRAM speed class | Harsher quant; parity tests |
| Q8_0 (`qwen36-q8`) | 64K | `n-cpu-moe 26` | **~533 t/s** | **~33 t/s** | Lossless · not daily |
| Mixed + transplanted MTP (`qwen36-mtp`) | 128K | `-ot` 13–29 + `-mtp` | — | **~93 code / ~90 prose / ~91 thinking** | **+15–19%** vs ~78 base |

### Qwen3.6 abliterated (uncensored)

| Slot | Ctx | Decode | Notes |
| --- | ---: | ---: | --- |
| Abliterated mixed q2k (`qwen36-ablit-mixed`) | 256K | **~78 t/s** | Replaces slower Huihui Q4_K_M (~66–72) |
| Abliterated + transplanted MTP | 128K | **~87 code / ~83 prose** | **+6–11%** vs mixed |

Older uniform-Q4 abliterated harness (2026-06-24 `all_model_bench`): 128K **708 pf / 76 dec**, 256K **639 pf / 68 dec** (browser-stress PASS).

### Coding / agent fine-tunes (same 35B-A3B geometry)

| Model | Slot class | Ctx | Decode | Notes |
| --- | --- | ---: | ---: | --- |
| **Qwopus-Coder** mixed | daily coding | 256K | **~78 t/s** | Thinking-off |
| Qwopus native MTP | `qwopus-mtp` | 128K | **~91–98 code** · ~76 prose | Self-spec; verification-lossless |
| Qwopus MTP corpus-tuned | `qwopus-mtp-tuned` | 128K | (A/B vs native) | Head val top-1 62.17% |
| Qwopus VRAM13 + MTP | all-VRAM | 64K | all-VRAM + MTP class | Forge-safe ctx (128K FA+MTP OOM risk) |
| Qwopus Q8_0 | max quality | 64K | **~32 t/s** | `n-cpu-moe 26` |
| **Ornith-1.0** mixed | RL reasoning | 256K | **~78 t/s** | Was Q4_K_M ~61 · PPL mixed 2.3676 / Q8 2.3434 |
| Ornith + MTP | transplanted | 128K | **~83 code / ~80 prose** | +3–7% (modest) |
| Ornith Q8_0 | max quality | 64K | **~33 t/s** | Max quality |
| **Agents-A1** mixed | agentic thinking | 256K | **~78 t/s** | IFEval 94.8 / Seal-0 56.4 (upstream) |
| Agents-A1 native MTP | native head | 128K | **~83 code / ~78 prose / ~82 thinking** | +5–7% only |
| Agents-A1 Q8_0 | max quality | 64K | **~32 t/s** | Official headless Q8 |
| **AgentWorld** mixed | world-model / tools | 256K | **~79 t/s** | AgentWorldBench 56.4 class (upstream) |
| AgentWorld + MTP | transplanted | 128K | **~82 code / ~79 prose** | +2–5% marginal |
| AgentWorld Q8_0 | max quality | 64K | **~32 t/s** | Max quality |

### Dense / other architectures

| Model | Engine | Ctx | Prefill | Decode | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| **Bonsai-27B Q1_0** (1-bit) | prism | **256K all-VRAM** | **~476 t/s** (pp262144) | **~53 fresh / ~10.2 @256K depth** | ~13.5 GiB across both GPUs · RAM free |
| **Huihui Gemma-4 26B-A4B** Q8 abliterated | turboquant gemma4 | 256K | **~273 t/s** | **~22 t/s** | `n-cpu-moe 23` + **`--no-op-offload`** (op-offload was 94 pf) |
| Qwen2.5-0.5B Fable-5 FT | ik | 8K | — | very fast (tiny) | Experimental · needs anti-loop sampling |

### Quant / placement improvements (cross-model)

| Change | Typical effect on this host |
| --- | --- |
| Uniform IQ4_XS @256K → **mixed q2k+imatrix + `-ot` CPU mid-layers** | Decode **~51 → ~78 t/s** at full 256K (**~1.5×**) while keeping quality near Q8 (PPL +0.6%) |
| Mixed base → **transplanted / native MTP** | **+3% to +26%** decode depending on head match (best on Qwen/Qwopus code) |
| Uniform Q4 / Q3 abliterated → **mixed abliterated** | ~66–72 → **~78 t/s** @256K |
| All-VRAM purpose quant (**VRAM13**) | **~122 t/s** decode @256K — speed king, stronger compression |
| Q8 max-quality path | **~32–33 t/s** @64K — ~2.4× slower than mixed, best fidelity |
| ik free flags `-sas` + `-rtr` on CPU-MoE slots | **~+2–3%** + **~+1.7%** decode (measured on qwen36-mixed) |
| Gemma-4: disable op-offload | Prefill **94 → 273 t/s** (**~2.9×**) |

### Historical harness snapshots

Automated multi-model runs under `all_model_bench_*` (older configs, useful as floor references):

| Date | Config | Prefill | Decode |
| --- | --- | ---: | ---: |
| 2026-06-21 | Qwen3.6 UD-Q3_K_XL 16K | 1786 | 102 |
| 2026-06-24 | Ablit Q3_K_M 256K | ~700 | ~78 |
| 2026-06-24 | Ablit Q4_K_M browser-safe 128K | 708 | 76 |
| 2026-06-24 | Ablit Q4_K_M browser-safe 256K | 639 | 68 |
| 2026-06-24 | Q4 offload sweep 64K (`n-cpu-moe` 12→10) | 690→777 | 75→82 |

---

## Features in this fork

| Feature | Status |
| --- | --- |
| `--defer-experts` + mmap-backed CPU MoE | Production path |
| Hybrid GPU-resident MoE tail (`--n-cpu-moe`) | Production path |
| Route-aware page-cache prefetch (`LLAMA_EXPERT_PREFETCH` / router top-k → `MADV_WILLNEED`) | Production default on GPT-OSS slots |
| Router logit-tail staging (`LLAMA_EXPERT_PREFETCH_TAIL=N` — stage the next *N* near-miss experts per CPU-MoE layer) | Experimental; lab replay shows the real tail covers ~19–21% of the future experts history misses |
| Dual-GPU layer split (`--tensor-split 46,54`) | Measured fit for 8+10 GiB |
| Q8 / Q4 K/V on hybrid placement | Q8 promoted for 1K/64K; Q4 for 8K fit |
| Chunked GDN port work | Experimental branch history (`chunked-gdn-port`) |
| Poolside standard DFlash GGUFs (`general.architecture=dflash`) | Validated with Laguna-S-2.1 target + Q4_K_M DFlash draft |
| Learned cross-layer route head | Lab-only; not launcher default |
| Persistent CUDA expert cache | Dead-end experiment (kept for reference) |

Inherited from **ik_llama.cpp**: fused MoE, FlashMLA, IQK quants, tensor overrides, server API, etc. See upstream docs for general llama tooling.

---

## Quick start

### Build (CUDA)

```bash
git clone https://github.com/xero00000/expert-streaming-engine.git
cd expert-streaming-engine

cmake -B build-expert-streaming -DGGML_NATIVE=ON -DGGML_CUDA=ON
cmake --build build-expert-streaming --config Release -j"$(nproc)"
```

Binary: `build-expert-streaming/bin/llama-server`.

### Run GPT-OSS 120B hybrid (example)

Adjust model path, GPU split, and CPU-MoE count for your VRAM. The values below match the measured dual 8+10 GiB daily driver (64K):

```bash
export GGML_CUDA_NO_PINNED=1
export LLAMA_EXPERT_PREFETCH=1   # if your build uses the lab control; else use --prefetch-experts where available

./build-expert-streaming/bin/llama-server \
  -m /path/to/gpt-oss-120b-F16.gguf \
  --jinja --port 8000 \
  --ctx-size 65536 --parallel 1 --n-gpu-layers 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --n-cpu-moe 30 --defer-experts \
  --tensor-split 46,54 --split-mode layer --main-gpu 1 \
  --batch-size 1024 --ubatch-size 512 \
  --threads 8 --threads-batch 32 \
  --flash-attn on --scheduler-async \
  --no-warmup
```

**1K max-speed decode slot** (eight resident MoE layers, Q8 K/V):

```bash
# same env; n-cpu-moe 28, ctx 1024, batch 128 / ubatch 64, threads 8
```

**8K fit** (same eight resident layers, Q4 K/V to save VRAM):

```bash
# --ctx-size 8192 --cache-type-k q4_0 --cache-type-v q4_0 --n-cpu-moe 28
```

### Critical flags

| Flag / env | Why |
| --- | --- |
| `GGML_CUDA_NO_PINNED=1` | Keeps deferred experts on GGUF mmap instead of a giant pinned host buffer |
| `--defer-experts` | Releases expert page residency until touched |
| `--n-cpu-moe N` | First *N* MoE blocks CPU-mapped; remainder GPU-resident (tune to VRAM) |
| `--tensor-split A,B` | Balance dense + resident MoE across cards |
| Large `-b` / `-ub` + high `--threads-batch` | Prefill throughput on CPU MoE path |
| Do **not** force GPU MoE prefill via tiny `offload-batch-size` on PCIe dual-GPU | Measured ~6× slower here |

---

## Recommended configs (this hardware)

| Slot | Context | K/V | CPU-MoE | Resident MoE | Prefill | Decode (short) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Fast | 1K | Q8 | 28 | 8 | n/a focus | **~12–16 tok/s** warm |
| Agent | **64K** | Q8 | 30 | 6 | **~140 tok/s** | **~11.5 tok/s** |
| Mid-ctx | 8K | Q4 | 28 | 8 | — | **~5–9 tok/s** class |

On a different GPU pair, re-sweep `--n-cpu-moe` and `--tensor-split` until both cards stay under allocation limits with headroom for desktop / compute buffers.

---

## What did *not* work

1. **Naive full-expert streaming from disk** — cold 1.77 GiB/token class I/O.  
2. **Pinned ~58 GiB expert host mapping** on a 31–47 GiB RAM machine — slow or impossible; use mmap + `GGML_CUDA_NO_PINNED`.  
3. **Blind multi-expert prefetch** — can more than double NVMe traffic for modest hit gains.  
4. **Learned layer-0 → future-layer head as default** — ~38% precision in simulation; executor speed stayed near the no-prefetch floor.  
5. **N-gram speculative decoding** on this GGUF — slower than the hybrid path.  
6. **Qwen draft models** — incompatible vocab (201k vs 248k); GPT-OSS has no MTP tensors.  
7. **Forced GPU weight-streaming prefill** — PCIe bound, much slower than CPU MoE prefill.  
8. **Standard Q3_K_M of MXFP4 experts** — row-size reject without `--allow-requantize`.

---

## Status & limitations

- Optimized and measured for **GPT-OSS 120B F16** on **dual low-VRAM Ampere + large host RAM**. Other MoEs may work via the same flags but need their own sweeps.  
- Decode remains **host-RAM / page-cache sensitive**: first request after reboot or reclaim is slower; warm routes are faster.  
- 64K is a **hardware capacity** result (full KV still allocated for SWA layers in this engine path); the model metadata advertises larger YaRN context.  
- Multi-prompt quality equivalence under Q8 K/V is **runtime-validated** (HTTP 200, non-empty completions), not a full blind eval suite.  
- This is an engineering fork; expect divergence from upstream `ik_llama.cpp` / mainline `llama.cpp`.

---

## Development

General ik/llama build and parameter docs still apply:

- [docs/build.md](docs/build.md)  
- [docs/parameters.md](docs/parameters.md)  
- [docs/docker.md](docs/docker.md)  

Local measurement harnesses used for the tables above live outside this tree in the companion **expert streaming lab** (router traces, staging ring, `run_expert_streaming_benchmark.sh`, sweep scripts). Re-run claims on your hardware before trusting absolute tok/s numbers.

---

## Credits

- [ggerganov / ggml-org — llama.cpp](https://github.com/ggml-org/llama.cpp)  
- [ikawrakow — ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (CPU/CUDA MoE performance, quants, server features)  
- Upstream `--defer-experts` and community MoE mmap prefetch work that this hybrid stack builds on  

## License

MIT — see [LICENSE](LICENSE).
