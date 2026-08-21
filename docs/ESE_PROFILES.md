# ESE Profiles and Tuning

The recommended entry point is:

```bash
./ese serve MODEL.gguf
```

Use `./ese plan MODEL.gguf` first. It shows the selected policy, model and shard size, RAM/VRAM, GGUF architecture metadata, environment variables, and exact native command.

## Resident policy

Use when weights fit safely in VRAM or for a normal dense-model launch:

```bash
./ese serve MODEL.gguf --policy resident
```

Core behavior:

```text
-ngl 99
-fa on
-ctk TYPE -ctv TYPE
```

For a dense model above the conservative free-VRAM threshold, the launcher adds native `--fit` so the engine calculates offload rather than forcing a full allocation.

## Hybrid policy

Use when you explicitly want static CPU/GPU expert placement:

```bash
./ese serve MODEL.gguf --policy hybrid
```

Safe default:

```text
-ngl 99
--cpu-moe
```

Dense and non-expert tensors use normal GPU offload while routed experts remain in CPU memory.
Unlike `cache`, this policy does not adapt expert residency from observed routes.

Retain a measured final MoE tail on GPU:

```bash
./ese serve MODEL.gguf \
  --policy hybrid \
  --gpu-resident-moe 6
```

For a 36-block GGUF this produces:

```text
--n-cpu-moe 30
```

The launcher requires a readable GGUF block count before translating this option. It does not guess layer geometry.

## Cache policy

Use when a sparse model fits the safe host-RAM budget but its experts do not
fit VRAM:

```bash
./ese serve MODEL.gguf --policy cache
```

The launcher enables the unified bounded expert hierarchy:

```text
NVMe/model shards -> RAM leases -> per-device VRAM cache
```

The default storage backend is `mmap`. RAM and per-device VRAM capacities are
calculated conservatively, or can be set explicitly:

```bash
./ese serve MODEL.gguf \
  --policy cache \
  --expert-ram-cache 4GiB \
  --expert-ram-staging 64MiB \
  --expert-vram-cache 1GiB \
  --expert-cache-min-observations 2
```

The RAM limit is a hard arena bound. The VRAM limit is applied independently
to every participating GPU and is reduced as needed to retain the configured
free-memory reserve.

## Stream policy

Use when a sparse model exceeds the safe host-RAM budget:

```bash
./ese serve MODEL.gguf --policy stream
```

The launcher sets:

```text
GGML_CUDA_NO_PINNED=1
```

and adds the same bounded controller used by `cache`, with `pread` as the
default storage backend:

```text
--defer-experts
--cpu-moe
```

Keep a known-good GPU tail when capacity permits:

```bash
./ese serve /models/gpt-oss-120b-F16.gguf \
  --policy stream \
  --gpu-resident-moe 6 \
  -c 65536 \
  --kv q8_0
```

Route-aware page-cache prefetch remains opt-in and requires mmap storage:

```bash
./ese serve MODEL.gguf \
  --policy stream \
  --expert-storage-backend mmap \
  --prefetch-tail 4
```

This sets `LLAMA_EXPERT_PREFETCH_TAIL=4`. The fired top-k stays first; the extra near-miss experts are advisory asynchronous work.

## KV selection

Automatic selection deliberately uses only existing native types:

- `q8_0` for the normal path;
- `q4_0` when context exceeds 131K or detected usable VRAM is very low.

Override directly:

```bash
./ese serve MODEL.gguf --kv f16
./ese serve MODEL.gguf --kv q8_0
./ese serve MODEL.gguf --kv q4_0
```

VBR/Turbo/TCQ is not exposed until its kernels, quality measurements, and lifecycle tests pass.

## Multi-GPU split

With two or more NVIDIA GPUs, auto mode calculates `--tensor-split` from currently free memory, not nominal card size.

```text
GPU 0 free: 7 GiB
GPU 1 free: 9 GiB
derived split: 44,56
```

Override after measuring a better model-specific split:

```bash
./ese serve MODEL.gguf --tensor-split 46,54
```

Each participating device has its own capacity and reserve. Expert uploads use
dedicated transfer streams and event dependencies, while placement follows the
normal graph/tensor split. The Phase 2 acceptance matrix exercises one, two,
and three GPUs; the implementation remains validation-pending until that
runtime evidence is attached.

## Context and reserve

The default context is 65,536 and the planner retains a 1 GiB VRAM reserve when selecting an automatic KV type.

```bash
./ese serve MODEL.gguf -c 131072 --reserve-vram 2GiB
```

The native runtime remains the authority on actual allocations.

## Threads and batching

Defaults:

- decode threads: up to 8, approximately half detected logical CPUs;
- batch threads: up to 32;
- hybrid/cache/stream batch: 1024;
- hybrid/cache/stream ubatch: 512.

Override after measuring your host:

```bash
./ese serve MODEL.gguf \
  --threads 8 \
  --batch-threads 32 \
  --batch-size 1024 \
  --ubatch-size 512
```

Decode and prefill often want different thread counts. Large CPU-MoE prefill can benefit from wide batch threads and large ubatches; decode can regress when every logical core is used.

## Native options

Everything after `--` is appended verbatim:

```bash
./ese serve MODEL.gguf -- \
  --jinja \
  --metrics \
  --log-verbosity 1
```

Stable ESE policy options belong before `--`; experimental native flags remain available after it.

## JSON plans

```bash
./ese plan MODEL.gguf --json
```

The JSON includes policy and reason, shards and total size, architecture metadata, RAM/per-GPU VRAM, environment, argument vector, and a shell-rendered command.

## Troubleshooting

### Startup allocates most host RAM

Verify the cache/stream plan contains `GGML_CUDA_NO_PINNED=1` and a non-zero
`--expert-ram-cache-mib` bound. A `stream` plan must additionally contain
`--defer-experts`.

### Hybrid mode is too slow

Measure a small GPU tail with `--gpu-resident-moe N`. Increase gradually; a tail that does not fit can prevent startup.

For route-adaptive placement instead, use `--policy cache` and inspect the
per-device hit, admission, eviction, and transfer telemetry.

### Stream prefetch is rejected

`--prefetch-tail N` is a page-cache hint and therefore requires
`--expert-storage-backend mmap`. The normal `stream` default is `pread` and
does not enable speculative prefetch.

### Auto chose the wrong policy

Force `--policy` and attach `./ese plan MODEL --json` plus the server startup log to the issue.

### Split GGUF is incomplete

Point `ese` at any correctly named shard. It validates all `-00001-of-000NN.gguf` files before launch.

### Server binary is missing

```bash
./ese build
```

or:

```bash
ESE_SERVER=/path/to/llama-server ./ese serve MODEL.gguf
```
