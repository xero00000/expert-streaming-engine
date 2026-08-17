# Expert Streaming Engine

Run models that do not fit comfortably in VRAM—and, for sparse MoE models, can exceed available system RAM—through one inspectable launcher.

ESE combines three execution policies behind a single command:

- **resident** — normal GPU offload when the model fits;
- **cache** — routed experts remain in host RAM while spare VRAM becomes an adaptive hot-expert cache;
- **stream** — expert storage is deferred and paged from the GGUF through the OS page cache, with route-aware prefetch.

The native engine is a focused `ik_llama.cpp`/`llama.cpp` fork. The `ese` launcher does not hide the native runtime: `ese plan` prints the exact environment and `llama-server` command before execution.

## Start here

Requirements: Linux, Python 3.10+, CMake, a C++ compiler, and optionally CUDA.

```bash
git clone https://github.com/xero00000/expert-streaming-engine
cd expert-streaming-engine

./ese doctor
./ese build
./ese serve /models/model.gguf
```

Inspect the plan without starting the server:

```bash
./ese plan /models/model.gguf
```

Force a policy when auto-selection is not what you want:

```bash
./ese serve /models/model.gguf --policy resident
./ese serve /models/model.gguf --policy cache
./ese serve /models/model.gguf --policy stream
```

Pass any native option after `--`:

```bash
./ese serve /models/model.gguf -c 131072 -- --jinja --metrics
```

The default API address is `http://127.0.0.1:8080`.

## What auto mode does

`ese` reads scalar GGUF metadata, totals all split shards, checks available RAM and VRAM, then chooses the least complicated safe path.

| Policy | Selected when | Native mechanism |
| --- | --- | --- |
| `resident` | The model fits safely in detected free VRAM | Conventional `-ngl 99` |
| `cache` | MoE experts fit in RAM but not VRAM | `exps=CPU` plus adaptive `--moe-cache` |
| `stream` | A sparse model exceeds the safe RAM budget | `--defer-experts`, CPU MoE, route prefetch |

On multiple NVIDIA GPUs, the launcher derives `--tensor-split` from currently free VRAM. For cached MoE inference it enables expert-parallel dispatch; on one GPU it uses the single-device cache path.

Auto mode is conservative. Every decision includes a human-readable reason:

```text
Policy : stream
Reason : MoE size 61.00 GiB exceeds 90% of available RAM (47.00 GiB);
         use deferred disk-backed experts
```

## Useful commands

```bash
./ese doctor                         # hardware and build prerequisites
./ese build                          # auto-select CUDA or CPU
./ese build --backend cuda --clean
./ese plan MODEL --json              # machine-readable memory plan
./ese serve MODEL --dry-run
./ese serve MODEL --kv q4_0 -c 262144
./ese serve MODEL --tensor-split 46,54
./ese serve MODEL --policy stream --gpu-resident-moe 6
```

Run `./ese <command> --help` for the complete interface.

## Architecture

```text
                              ESE
                               │
                    GGUF metadata + hardware
                               │
                         memory planner
              ┌────────────────┼────────────────┐
              │                │                │
           resident          cache            stream
              │                │                │
       GPU model path    experts in RAM   experts on NVMe
                               │                │
                         adaptive VRAM     bounded page cache
                           hot cache       + route prefetch
              └────────────────┴────────────────┘
                               │
                     llama-server / OpenAI API
```

The intended end state is one bounded hierarchy:

```text
NVMe → RAM expert cache → VRAM expert cache
                    ↘ KV / scratch / draft budget
```

See [Architecture](docs/ESE_ARCHITECTURE.md) for invariants and component boundaries.

## Current engine capabilities

- deferred-mmap expert storage and route-aware page-cache prefetch;
- CPU MoE with selected MoE layers resident on GPU;
- adaptive VRAM MoE caching for RAM-backed experts;
- multi-GPU expert-parallel cache dispatch;
- DeepSeek V4 Flash and DSpark paths on the integration line;
- native MTP and model-specific speculative paths inherited from the engine;
- CUDA sparse/grouped MoE work, including MXFP4 and TQ2_0 paths;
- Flash Attention, quantized KV types already supported by the native runtime;
- standard `llama-server`, CLI, conversion, and API compatibility inherited upstream.

VBR and TCQ from `buun-llama-cpp` are **not labeled complete** in this branch. Their port is isolated behind explicit quality and lifecycle acceptance gates rather than copied into the public path without validation. See [Port roadmap](docs/PORT_ROADMAP.md).

## Verified reference result

The existing ESE benchmark record includes GPT-OSS 120B F16 GGUF (about 61 GiB) on dual Ampere GPUs with 18 GiB combined VRAM, a Ryzen 9 5950X, about 47 GiB usable DDR4, and NVMe storage:

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

`ese-unified` is the proposed consolidated core. It is based on the newer DeepSeek/expert-cache integration line rather than the older public default. Android/QNN remains a separate platform port until its backend can pass the same parity and lifecycle gates. Historical experiment branches are retained only as source references.

## License and lineage

MIT licensed. ESE is derived from `ik_llama.cpp`, which is derived from `llama.cpp`. Individual imported or adapted features must retain their original attribution and license notices. The roadmap tracks `buun-llama-cpp` ideas by source commit before any code-level port.
