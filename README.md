# Expert Streaming Engine

**Run huge Mixture-of-Experts models on consumer dual-GPU boxes** by combining deferred mmap expert weights, hybrid GPU-resident MoE layers, and route-aware page-cache prefetch — without re-quantizing the model.

This repository is a focused fork of [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (itself a [llama.cpp](https://github.com/ggml-org/llama.cpp) fork). The goal is not a new quant format. It is **making F16 / MXFP4 MoE weights that do not fit in VRAM usable at interactive speeds** on low-VRAM multi-GPU hosts.

| Badge | |
| --- | --- |
| License | [MIT](LICENSE) |
| Branch | `chunked-gdn-port` |
| Headline result | **GPT-OSS 120B F16** · **~140 tok/s prefill** · **~11.5 tok/s decode** |

---

## Headline result

**Model:** [GPT-OSS 120B](https://huggingface.co/) F16 GGUF (~61 GiB on disk)  
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

All numbers below are from the isolated `:8014` harness and the live `:8000` launcher on the same dual-Ampere host. Full lab log: local `expert_streaming_lab/RESULTS.md` (methodology, raw JSON paths, rejected ablations).

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

## Features in this fork

| Feature | Status |
| --- | --- |
| `--defer-experts` + mmap-backed CPU MoE | Production path |
| Hybrid GPU-resident MoE tail (`--n-cpu-moe`) | Production path |
| Route-aware page-cache prefetch (`LLAMA_EXPERT_PREFETCH` / router top-k → `MADV_WILLNEED`) | Production default on GPT-OSS slots |
| Dual-GPU layer split (`--tensor-split 46,54`) | Measured fit for 8+10 GiB |
| Q8 / Q4 K/V on hybrid placement | Q8 promoted for 1K/64K; Q4 for 8K fit |
| Chunked GDN port work | Experimental branch history (`chunked-gdn-port`) |
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

## What did *not* work (so you do not re-learn it)

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
