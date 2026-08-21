# Expert Streaming Engine

[![CI](https://github.com/xero00000/expert-streaming-engine/actions/workflows/ese-ci.yml/badge.svg?branch=main)](https://github.com/xero00000/expert-streaming-engine/actions/workflows/ese-ci.yml)
[![Latest release](https://img.shields.io/github/v/release/xero00000/expert-streaming-engine)](https://github.com/xero00000/expert-streaming-engine/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Run sparse Mixture-of-Experts GGUF models when the full model does not fit in
VRAM—and, with deferred experts, when it cannot safely remain resident in RAM.

Expert Streaming Engine (ESE) is a Linux-focused `ik_llama.cpp`/`llama.cpp`
fork. It combines a transparent `ese` launcher with a native global resource
controller and a bounded `NVMe → RAM → VRAM` expert hierarchy. The result is a
normal `llama-server` and OpenAI-compatible API with explicit memory limits,
observable decisions, and no hidden backend or precision fallback.

## Quick start

Requirements: Linux, Python 3.10+, CMake, a C++ compiler, and optionally an
NVIDIA CUDA toolchain.

```bash
git clone https://github.com/xero00000/expert-streaming-engine.git
cd expert-streaming-engine

./ese doctor
./ese build                    # auto-detect CUDA; use --backend cpu to force CPU
./ese plan /models/model.gguf  # inspect the complete command without running it
./ese serve /models/model.gguf
```

The server listens on `http://127.0.0.1:8080` by default. Check it with:

```bash
curl http://127.0.0.1:8080/health
```

Split GGUFs are supported: pass any correctly named shard and ESE validates the
complete set before launch.

## Four execution policies

`auto` is recommended. ESE inspects GGUF metadata, all model shards, available
host RAM, and current per-GPU VRAM before choosing a native policy.

| Policy | Use it when | Execution path |
| --- | --- | --- |
| `resident` | The model fits safely in VRAM, or is dense | Normal GPU offload and native fit |
| `hybrid` | You want static CPU/GPU MoE placement | Dense tensors on GPU; experts on CPU; optional GPU MoE tail |
| `cache` | Sparse weights fit RAM but not VRAM | Bounded RAM leases feeding adaptive per-device VRAM caches |
| `stream` | Sparse weights exceed the safe RAM budget | Deferred expert storage feeding the same bounded cache hierarchy |

```bash
./ese serve MODEL.gguf --policy auto
./ese serve MODEL.gguf --policy hybrid --gpu-resident-moe 6
./ese serve MODEL.gguf --policy cache --expert-ram-cache 4GiB
./ese serve MODEL.gguf --policy stream --expert-storage-backend pread
```

Use `./ese plan MODEL.gguf --json` for machine-readable launcher output. Pass
native `llama-server` options after `--`:

```bash
./ese serve MODEL.gguf -c 131072 -- --jinja --metrics
```

## Native memory controller

The launcher discovers hardware and supplies limits; the native runtime makes
the final allocation decision from real model geometry. Direct native launches
can use the same interface:

```bash
build/bin/llama-server -m MODEL.gguf \
  --memory-policy auto \
  --max-ram 40GiB \
  --reserve-vram 1GiB \
  --min-kv-quality turbo4 \
  --max-context 128K \
  --metrics
```

Before accepting requests, the runtime prints one deterministic JSON plan that
accounts for:

- dense and routed-expert weights;
- bounded expert RAM and per-device VRAM caches;
- KV quality, context, slots, batch, and graph workspace;
- prompt cache and disk-I/O staging;
- MTP/draft and multimodal transient modules;
- a safety reserve on every selected GPU.

The same object is returned by `/props`; `/metrics` reports the selected
context, planned RAM, transient capacity, and per-device headroom. Requested
storage backends and KV quality floors fail closed instead of silently falling
back. Plan transitions use prepare/commit/rollback semantics.

## How data moves

```text
GGUF shards on NVMe
        │
        ▼
checked expert descriptors ──► bounded RAM leases
                                      │
                                      ▼
                              adaptive GPU caches
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          ▼                           ▼                           ▼
      resident                    cache/stream                 transient
    model tensors              routed MoE experts           MTP / mmproj
          └───────────────────────────┬───────────────────────────┘
                                      ▼
                         llama-server / OpenAI API
```

Every cache has a configured capacity. Expert uploads use dedicated CUDA
transfer streams and event-scoped readiness; in-flight leases cannot be
evicted. Multi-GPU placement and reserves are accounted per device.

## Useful commands

```bash
./ese doctor --json
./ese build --backend cuda --clean
./ese plan MODEL.gguf --json
./ese serve MODEL.gguf --dry-run
./ese serve MODEL.gguf --slots 2 -c 131072
./ese serve MODEL.gguf --tensor-split 46,54
./ese serve MODEL.gguf --reserve-vram 2GiB
```

Run `./ese <command> --help` for the complete stable launcher interface. The
full native option reference remains available through
`build/bin/llama-server --help`.

## Validation status

The merged desktop path is covered by Python surface tests, native release
tests, ASAN/UBSAN lifecycle tests, CPU server builds, and model-backed CUDA
validation.

| Area | Current evidence |
| --- | --- |
| Expert hierarchy | mmap/pread parity, forced eviction, sanitizer coverage, and 1/2/3-GPU execution |
| CUDA hardware | RTX 2080 SUPER (`sm_75`), RTX 3060 Ti (`sm_86`), and RTX 3080 (`sm_86`) |
| Global controller | CPU plus real Turing+Ampere three-GPU model load with explicit 1 GiB reserves |
| Transient/speculation | CPU, Turing, Ampere, and model-backed image→text module swapping |
| Turbo KV foundation | CPU/CUDA codecs, direct attention paths, lifecycle tests, and quality sweeps |

Ada-or-newer runtime coverage is not claimed: suitable hardware is unavailable
to the solo maintainer. Those architecture-specific gates remain future work;
the runtime must still fail closed rather than inventing a synthetic pass.

The Android/QNN port remains a separate draft until it receives physical-device
build, parity, memory, and thermal evidence.

## Reference performance

On the consolidated v0.1.0 candidate, a Qwen3.6 35B-A3B MoE reached median
throughput of 1,612.46 tok/s for 512-token prompt processing, 1,583.32 tok/s
for 2,048-token prompt processing, and 116.59 tok/s for 128-token generation
across an RTX 3060 Ti, RTX 2080 SUPER, and RTX 3080.

The consolidated v0.1.0 candidate ran a Qwen3.5 27B Q4_K_M dense model across
an RTX 3060 Ti, RTX 2080 SUPER, and RTX 3080. Five-run averages were 660.60
tok/s for 512-token prompt processing, 666.03 tok/s for 2,048-token prompt
processing, and 26.78 tok/s for 128-token generation. A live 65,536-context
server also retained more than its declared 1 GiB reserve on every GPU.

The earlier GPT-OSS 120B F16 expert-streaming record remains available: about
139–141 tok/s warm prefill and 11.5 tok/s short-context decode on the same CPU
class and NVMe storage. These are machine- and configuration-specific
engineering records, not universal performance promises. See
[reference benchmarks](docs/ESE_BENCHMARKS.md) for exact commands, model hash,
hardware, repetition statistics, cold/warm behavior, and ablations.

## Documentation

- [Profiles and tuning](docs/ESE_PROFILES.md)
- [Global resource controller](docs/PHASE4_GLOBAL_RESOURCE_CONTROLLER.md)
- [Architecture and invariants](docs/ESE_ARCHITECTURE.md)
- [Expert-cache validation](docs/PHASE2_EXPERT_CACHE_VALIDATION.md)
- [Turbo KV Phase 1 validation](docs/TURBO_KV_PHASE1_VALIDATION.md)
- [Reference benchmarks](docs/ESE_BENCHMARKS.md)
- [Port roadmap](docs/PORT_ROADMAP.md)
- [Native build guide](docs/build.md)
- [Native parameter reference](docs/parameters.md)
- [Contributing](CONTRIBUTING.md)

## Scope and lineage

ESE prioritizes bounded memory, reproducibility, and correctness over a single
best benchmark. Experimental formats remain internal until their quality and
lifecycle gates pass. General desktop work targets `main`; platform and
research branches remain isolated until their own validation is complete.

ESE is MIT licensed. It derives from `ik_llama.cpp`, which derives from
`llama.cpp`; imported work retains its original attribution and license
notices.
