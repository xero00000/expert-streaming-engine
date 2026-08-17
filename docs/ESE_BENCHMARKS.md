# ESE Reference Benchmarks

These measurements are retained as engineering evidence for promoted settings. They are not normalized across projects and are not promises for other hardware.

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
