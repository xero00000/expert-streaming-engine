# Expert Streaming Engine

Run sparse Mixture-of-Experts models on machines where the full model does not fit comfortably in VRAM—and, with deferred experts, where the expert set can exceed the safe system-RAM budget.

ESE is a focused `ik_llama.cpp`/`llama.cpp` fork with one transparent launcher and three native execution policies:

- **resident** — normal GPU offload when the model fits;
- **hybrid** — dense tensors on GPU, routed experts on CPU, with an optional MoE tail retained on GPU;
- **stream** — hybrid execution plus deferred mmap expert residency and route-aware page-cache prefetch.

The launcher never replaces the native runtime. `ese plan` prints the exact environment and `llama-server` command before anything runs.

## Start here

Requirements: Linux, Python 3.10+, CMake, a C++ compiler, and optionally CUDA.

```bash
git clone https://github.com/xero00000/expert-streaming-engine
cd expert-streaming-engine

./ese doctor
./ese build
./ese serve /models/model.gguf
```

Inspect a launch without starting the server:

```bash
./ese plan /models/model.gguf
```

Force a policy:

```bash
./ese serve /models/model.gguf --policy resident
./ese serve /models/model.gguf --policy hybrid
./ese serve /models/model.gguf --policy stream
```

Pass any native option after `--`:

```bash
./ese serve /models/model.gguf -c 131072 -- --jinja --metrics
```

The default API address is `http://127.0.0.1:8080`.

## What auto mode does

`ese` reads scalar GGUF metadata, validates and totals split shards, checks available RAM and NVIDIA VRAM, and selects the least complicated safe path.

| Policy | Selected when | Native mechanism |
| --- | --- | --- |
| `resident` | The model fits safely in free VRAM, or is dense | `-ngl 99`, with native `--fit` when needed |
| `hybrid` | A MoE fits RAM but not VRAM | `--cpu-moe`, optionally `--n-cpu-moe N` |
| `stream` | A MoE exceeds the safe RAM budget | `--defer-experts` plus CPU MoE and route prefetch |

On multiple NVIDIA GPUs, the launcher derives `--tensor-split` from currently free VRAM. Every decision includes a readable reason and can be emitted as JSON.

```bash
./ese plan /models/model.gguf --json
```

## Useful commands

```bash
./ese doctor
./ese build
./ese build --backend cuda --clean
./ese plan MODEL.gguf --json
./ese serve MODEL.gguf --dry-run
./ese serve MODEL.gguf --kv q4_0 -c 262144
./ese serve MODEL.gguf --tensor-split 46,54
./ese serve MODEL.gguf --policy hybrid --gpu-resident-moe 6
./ese serve MODEL.gguf --policy stream --gpu-resident-moe 6
```

Run `./ese <command> --help` for the complete interface.

## Architecture

```text
                              ESE
                               │
                    GGUF metadata + hardware
                               │
                         memory planner
              ┌────────────────┼───────────────┐
              │                │                │
           resident          hybrid           stream
              │                │                │
       GPU/native fit    CPU MoE + GPU     deferred mmap
                         dense/tail         + route prefetch
              └────────────────┴───────────────┘
                               │
                     llama-server / OpenAI API
```

The current engine has static hybrid placement and a specialized disk-backed stream path. A unified bounded `NVMe → RAM cache → VRAM cache` controller is the next native phase; it is not mislabeled as already complete.

See [Architecture](docs/ESE_ARCHITECTURE.md) for invariants and component boundaries.

## Current native capabilities

- deferred-mmap expert storage;
- route-aware top-k page-cache prefetch and optional router-logit-tail staging;
- CPU MoE with a selected final MoE tail resident on GPU;
- multi-GPU tensor splitting;
- DeepSeek V4 Flash integration work;
- DSpark, native MTP, and model-specific speculative paths on the integration line;
- Maple/DeepGrove architecture and native TQ2_0 CPU/CUDA paths;
- CUDA sparse/grouped MoE work, MXFP4 support, Flash Attention, and existing quantized KV types;
- standard `llama-server`, CLI, conversion, and API compatibility inherited upstream.

The adaptive MoE cache, expert-parallel cache distribution, VBR, Turbo KV, and TCQ ideas from `buun-llama-cpp` are tracked as native ports with explicit parity, memory, quality, and lifecycle gates. They are **not** advertised as completed flags in this branch. See [Port roadmap](docs/PORT_ROADMAP.md).

## Verified reference result

The retained ESE benchmark record includes GPT-OSS 120B F16 GGUF (about 61 GiB) on dual Ampere GPUs with 18 GiB combined VRAM, a Ryzen 9 5950X, about 47 GiB usable DDR4, and NVMe storage:

| Workload | Recorded result |
| --- | ---: |
| Warm prefill | about 139–141 tok/s |
| Short-context decode | about 11.5 tok/s |
| Decode near 22K filled context | about 8.9 tok/s |
| Deferred-mmap startup | about 13–16 s |
| 27K needle retrieval | pass |

These are machine- and configuration-specific measurements, not universal promises. Method and promoted settings are summarized in [Benchmarks](docs/ESE_BENCHMARKS.md).

## Documentation

- [Profiles and tuning](docs/ESE_PROFILES.md)
- [Architecture](docs/ESE_ARCHITECTURE.md)
- [Benchmarks](docs/ESE_BENCHMARKS.md)
- [Port roadmap and acceptance gates](docs/PORT_ROADMAP.md)
- [Branch policy](docs/BRANCH_POLICY.md)
- [Native build documentation](docs/build.md)
- [Native parameter reference](docs/parameters.md)
- [Speculative decoding](docs/speculative.md)

## Branch policy

`ese-unified` is the proposed consolidated core. It is based on the newer DeepSeek/Maple integration line rather than the older public default. Android/QNN remains a separate platform port until it passes the same parity and lifecycle gates. Historical experiment branches are retained as source references until their unique commits are tagged.

## License and lineage

MIT licensed. ESE is derived from `ik_llama.cpp`, which is derived from `llama.cpp`. Imported or adapted features must retain their original attribution and license notices. The roadmap pins the analyzed `buun-llama-cpp` revision before any code-level port.
