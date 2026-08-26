# ESE Reference Benchmarks

These measurements are retained as engineering evidence for promoted settings. They are not normalized across projects and are not promises for other hardware.

## v0.1.0 candidate: Qwen3.6 35B-A3B MoE

Measured on 2026-08-21 from commit `6c4d7d9db7eff21a1faa9a548486343743638b98`.
ESE identified 40 blocks, 256 experts, and the `qwen35moe` architecture. The
18,178,317,280-byte mixed-Q2K imatrix GGUF contains 34,660,610,688 total
parameters with about 3B active parameters per token. The runtime reports its
quantization class as `Q4_K - Medium`.

| Component | Configuration |
| --- | --- |
| Model | Qwen3.6 35B-A3B mixed-Q2K imatrix GGUF |
| Model SHA-256 | `9b9bdc97b9a4c84dfbd92a9dca78a2905f9181bbb49e0e2d4d1f26b30ff9438c` |
| CPU / RAM | Ryzen 9 5950X, 16 cores / 32 threads; 46 GiB RAM |
| GPUs | RTX 3060 Ti + RTX 2080 SUPER + RTX 3080; layer split `31/31/38` |
| Driver / CUDA | NVIDIA 595.84; CUDA 13.2 build toolkit |
| Model storage | ST4000VX016-3CV104 SATA, Btrfs |
| Build | GCC/G++ 12.4, CUDA architectures `75;86`, ESE build 4817 |

Command (the absolute model path is shortened to `$MODEL`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 ./llama-bench \
  -m "$MODEL" -p 512,2048 -n 128 -b 1024 -ub 512 \
  -ctk q8_0 -ctv q8_0 -t 8 -ngl 999 \
  -sm layer -ts 31/31/38 -fa 1 -r 5 -w 1 -o json
```

All 40 layers were GPU-resident with fused MoE enabled. Five measured
repetitions followed one warmup:

| Workload | Mean | Median | Minimum | Std. dev. |
| --- | ---: | ---: | ---: | ---: |
| 512-token prompt processing | 1,504.41 tok/s | 1,612.46 tok/s | 1,063.13 tok/s | 246.75 |
| 2,048-token prompt processing | 1,581.07 tok/s | 1,583.32 tok/s | 1,568.04 tok/s | 9.30 |
| 128-token generation | 118.02 tok/s | 116.59 tok/s | 114.96 tok/s | 3.13 |

The first measured 512-token repetition was the low outlier; the other four
were 1,607–1,623 tok/s. Startup from the external SATA model store is excluded
from these steady-state measurements. `ese plan` selected `resident` because
the 16.93 GiB file fit within the safe share of 24.22 GiB detected free VRAM.

## v0.1.0 candidate: Qwen3.5 27B dense

Measured on 2026-08-21 from commit `6c4d7d9db7eff21a1faa9a548486343743638b98`
(clean native sources; release-document changes did not affect the benchmark binary).
This is a dense-model, all-GPU throughput baseline for the consolidated release,
not an expert-streaming result.

| Component | Configuration |
| --- | --- |
| Model | Qwen3.5 27B Q4_K_M, 26,895,998,464 parameters, 16,536,406,016 bytes |
| Model SHA-256 | `3445102e9cde5d562508642c100a2f5ac3368a5a3f748442811d7a95daee3bec` |
| CPU / RAM | Ryzen 9 5950X, 16 cores / 32 threads; 46 GiB RAM |
| GPU 0 | RTX 3060 Ti, 8 GiB, `sm_86` |
| GPU 1 | RTX 2080 SUPER, 8 GiB, `sm_75` |
| GPU 2 | RTX 3080, 10 GiB, `sm_86` |
| Driver / CUDA | NVIDIA 595.84; CUDA 13.2 build toolkit |
| Storage | WDC WDS100T2B0C-00PXH0 NVMe |
| Build | GCC/G++ 12.4, CUDA architectures `75;86`, ESE build 4817 |

Command (the absolute model path is shortened to `$MODEL`):

```bash
CUDA_VISIBLE_DEVICES=0,1,2 ./llama-bench \
  -m "$MODEL" -p 512,2048 -n 128 -b 1024 -ub 512 \
  -ctk q8_0 -ctv q8_0 -t 8 -ngl 999 \
  -sm layer -ts 31/31/38 -fa 1 -r 5 -w 1 -o json
```

Five measured repetitions followed one warmup:

| Workload | Mean | Median | Minimum | Std. dev. |
| --- | ---: | ---: | ---: | ---: |
| 512-token prompt processing | 660.60 tok/s | 672.64 tok/s | 609.80 tok/s | 28.42 |
| 2,048-token prompt processing | 666.03 tok/s | 666.75 tok/s | 663.65 tok/s | 1.58 |
| 128-token generation | 26.78 tok/s | 26.74 tok/s | 26.72 tok/s | 0.08 |

### Live resource-controller check

The same model was started through `ese serve` with a 65,536-token context,
one slot, F16 KV, all 66 layers on the GPUs, and an explicit 1 GiB reserve on
each device. The native controller selected the same plan on repeated input,
the server became ready in about 12 seconds, and `/props` plus `/metrics`
reported the selected plan. After the model, 4 GiB KV cache, and compute buffers
were allocated, `nvidia-smi` reported:

| Device | VRAM used | VRAM free | Required reserve |
| --- | ---: | ---: | ---: |
| RTX 3060 Ti | 6,586 MiB | 1,257 MiB | 1,024 MiB |
| RTX 2080 SUPER | 6,636 MiB | 1,151 MiB | 1,024 MiB |
| RTX 3080 | 8,202 MiB | 1,673 MiB | 1,024 MiB |

This verifies the release controller's reserve invariant on the available
Turing and Ampere hardware. It does not claim Ada-or-newer coverage.

## GPT-OSS 120B F16

Reference machine:

| Component | Configuration |
| --- | --- |
| Model | GPT-OSS 120B F16 GGUF, about 61 GiB |
| GPU | dual Ampere, about 8 GiB + 10 GiB |
| CPU | Ryzen 9 5950X, 16 cores / 32 threads |
| RAM | about 47 GiB usable DDR4 |
| Storage | NVMe |
| Main slot | 64K context, Q8 K/V |
| Placement | 30 CPU/deferred MoE layers + 6 GPU-resident MoE layers |

Recorded promoted results:

| Metric | Result |
| --- | ---: |
| Warm prefill, 4.7K–22K prompts | about 139–141 tok/s |
| Short-context decode | about 11.5 tok/s |
| Decode at about 22K filled context | about 8.9 tok/s |
| Deferred-mmap startup | about 13–16 s |
| Needle retrieval, 27K / 40% depth | pass |

## Memory-path ablations

Fixed 23-token prompt, 48-token deterministic decode:

| Configuration | Startup | Decode |
| --- | ---: | ---: |
| All 36 MoE layers deferred, no prefetch | about 13 s | 1.31 tok/s |
| Current-route `MADV_WILLNEED` | about 13 s | 4.04 tok/s |
| 30 deferred + 6 GPU-resident MoE layers | about 13 s | 7.62 tok/s |
| 28 CPU/deferred + 8 GPU-resident MoE layers | about 15 s | 9.41 tok/s |

The cold/warm difference demonstrates why a bounded hot set matters:

| Request | Decode | Observation |
| --- | ---: | --- |
| First five-token completion | 0.40 tok/s | cold expert pages |
| Identical second request | 26.7 tok/s | warm page cache |

A direct read of roughly 1.77 GiB of expert slices for one uncached token measured around 1.2 seconds on the reference NVMe. Pure per-token storage streaming without locality or prefetch is therefore not an interactive design.

## Startup safeguard

The promoted stream path sets:

```text
GGML_CUDA_NO_PINNED=1
```

On the reference host, allowing CUDA's pinned-buffer preference could turn CPU MoE into an approximately 58–59 GiB host allocation and increase readiness from roughly 13 seconds to roughly 505 seconds. Deferred paging only works as intended when the runtime does not recreate a whole-model pinned allocation.

## Prefill sweep

64K Q8 K/V, six GPU-resident MoE layers, warm 4,747-token prompt:

| Batch / ubatch / batch threads | Prefill |
| --- | ---: |
| 128 / 64 / 8 | 57.8 tok/s |
| 256 / 128 / 8 | 72.2 tok/s |
| 512 / 256 / 8 | 85.8 tok/s |
| 512 / 512 / 8 | 98.1 tok/s |
| 512 / 256 / 16 | 115.5 tok/s |
| 1024 / 512 / 16 | 120.1 tok/s |
| 1024 / 512 / 32 | 139.0 tok/s |

Forced PCIe GPU weight-streaming prefill measured about 23.6–38.4 tok/s on this host and was rejected in favor of CPU-MoE prefill.

## KV sweep

On the one-thousand-token fast slot, Q8 K/V improved the documented repeated workload average from roughly 11.96 tok/s with F16 K/V to roughly 12.81 tok/s, about a 7.1% gain. Q4 did not win that particular sweep.

This result is one reason the unified launcher defaults to `q8_0` rather than assuming lower KV precision is always faster.

## v2 candidate: Kimi Linear 48B-A3B

This gate used `Kimi-Linear-48B-A3B-Instruct.MXFP4_MOE.gguf` (SHA-256
`3c3a000f566e68dfccd7925e49cf16602be830b72aecf844c15b6ae840e72a04`), a
27-layer, 49.123B-parameter model with 20 KDA layers, seven MLA layers, 256
experts, and top-8 routing. The file contains 610 tensors and is 25.331 GiB.

The tested host had 46 GiB RAM, an RTX 3060 Ti 8 GiB (`sm_86`), RTX 2080 SUPER
8 GiB (`sm_75`), and RTX 3080 10 GiB (`sm_86`). The promoted short-prompt
decode profile intentionally used only the two Ampere GPUs:

```bash
GGML_CUDA_NO_PINNED=1 CUDA_VISIBLE_DEVICES=0,2 llama-cli \
  -m Kimi-Linear-48B-A3B-Instruct.MXFP4_MOE.gguf \
  -c 65536 -b 128 -ub 32 -ngl 99 -sm layer -ts 4,22 \
  -cmoe --defer-experts \
  --expert-ram-cache-mib 8192 \
  --expert-vram-cache-mib 2048 \
  --expert-vram-reserve-mib 768 \
  --expert-storage-backend pread --expert-sidecar-only \
  --expert-cache-min-observations 1 \
  --no-warmup --temp 0 --seed 1234 -n 32 \
  -p 'The capital of France is'
```

| Measurement | Result |
| --- | ---: |
| Context allocated | 65,536 tokens |
| Five-token prompt processing | 4.69 tok/s |
| 32-token generation | 9.29 tok/s |
| Expert-cache slots | 570 per GPU |
| Aggregate cache hits / misses | 3,229 / 3,219 |
| Forced fallbacks / rejected admissions | 0 / 0 |

The generated continuation was deterministic across repeated runs:
`The capital of France is Paris. The capital of Italy is Rome. The capital of
Germany is Berlin.` Both runs had SHA-256
`f9af01b0e54b872d6cb619331565f86e5986288164f76dfac0f309996f6dd196`.
CPU and CUDA KDA unit outputs agreed within `4.47035e-08` on all three GPUs,
and ESE's first token matched the independent upstream reference output.

This is a 64K allocation and short-prompt decode gate, not a fully populated
64K prompt benchmark. KDA prefill still uses the correctness-first sequential
kernel; improving long-prompt KDA throughput remains separate performance
work. Increasing the cache from the former accidental 64-slot limit eliminated
short-run evictions, while the 32-token workload exercises sustained bounded
eviction and sidecar reload behavior.

## Benchmark acceptance template

New performance claims should record:

- exact commit and dirty state;
- model SHA or exact source revision;
- quantization and shard layout;
- CPU, RAM, storage, GPU, driver, and interconnect;
- complete command and relevant environment;
- cold and warm status;
- context fill and generated-token count;
- prompt-processing and decode separately;
- median plus low tail across repeated prompts;
- model output or parity evidence;
- peak RAM and per-device VRAM;
- cache hits, misses, evictions, and bytes read;
- whether the intended native path was proven active.

Lossless engine changes require parity evidence. Lossy KV changes additionally require KLD, perplexity, and task-quality comparisons against an F16 reference.
