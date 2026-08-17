# ESE Profiles and Tuning

The recommended entry point is:

```bash
./ese serve MODEL.gguf
```

Use `./ese plan MODEL.gguf` before a large run. The plan shows the selected policy, model/shard size, available RAM/VRAM, GGUF architecture metadata, environment variables, and exact native command.

## Resident policy

Use when model weights fit safely in aggregate free VRAM.

```bash
./ese serve MODEL.gguf --policy resident
```

Generated core flags:

```text
-ngl 99 -fa on -ctk TYPE -ctv TYPE
```

This policy does not enable expert deferral or an adaptive expert cache. Leave headroom for KV, graph workspace, and transient allocations.

## Cache policy

Use for sparse models whose experts fit in host RAM but not VRAM.

```bash
./ese serve MODEL.gguf --policy cache
```

Core behavior:

```text
-ngl 99
-sm layer
-ot exps=CPU
--moe-cache on                         # one NVIDIA GPU
--moe-cache auto
--moe-cache-expert-parallel auto       # two or more NVIDIA GPUs
```

The cache path makes otherwise unused VRAM useful while preserving CPU execution for misses. Layer splitting is preferred for this workload. Multi-GPU fanout, thread count, reserve size, and cache admission are hardware-specific; benchmark rather than assuming every GPU should participate.

## Stream policy

Use when a sparse model exceeds the safe host-RAM budget.

```bash
./ese serve MODEL.gguf --policy stream
```

The launcher sets:

```text
GGML_CUDA_NO_PINNED=1
LLAMA_EXPERT_PREFETCH=1
```

and adds:

```text
--defer-experts
--cpu-moe
```

For GPT-OSS models with a readable block count, `--gpu-resident-moe N` is converted to `--n-cpu-moe BLOCKS-N`. This keeps a small MoE tail on GPU while the remaining expert set is deferred.

Example matching the documented six-resident-layer 64K shape:

```bash
./ese serve /models/gpt-oss-120b-F16.gguf \
  --policy stream \
  --gpu-resident-moe 6 \
  -c 65536 \
  --kv q8_0
```

Route-logit-tail staging is available but disabled by default because it should be promoted only after cold-cache A/B validation:

```bash
./ese serve MODEL.gguf --policy stream --prefetch-tail 4
```

## KV selection

Current automatic selection is intentionally limited to types supported by the native engine:

- `q8_0` for the normal path;
- `q4_0` when context exceeds 131K or detected usable VRAM is very low.

Override it directly:

```bash
./ese serve MODEL.gguf --kv f16
./ese serve MODEL.gguf --kv q8_0
./ese serve MODEL.gguf --kv q4_0
```

VBR/Turbo/TCQ is tracked separately and is not presented as available until its kernels, quality measurements, and lifecycle tests pass.

## Multi-GPU split

With two or more NVIDIA GPUs, auto mode calculates `--tensor-split` from currently free memory, not nominal card size.

Example:

```text
GPU 0 free: 7 GiB
GPU 1 free: 9 GiB
derived split: 44,56
```

Override when a measured model-specific split is better:

```bash
./ese serve MODEL.gguf --tensor-split 46,54
```

The split is visible in every plan.

## Context and VRAM reserve

The default context is 65,536 tokens and the planner reserves 1 GiB of VRAM when choosing an automatic KV type.

```bash
./ese serve MODEL.gguf -c 131072 --reserve-vram 2GiB
```

The reserve currently guides planning; the native runtime remains the authority on actual allocations.

## Threads and batching

Defaults:

- decode threads: up to 8, approximately half the detected logical CPUs;
- batch threads: up to 32;
- cache policy: batch 1024, ubatch 512;
- stream policy: batch 1024, ubatch 512.

Override after measuring your host:

```bash
./ese serve MODEL.gguf \
  --threads 8 \
  --batch-threads 32 \
  --batch-size 1024 \
  --ubatch-size 512
```

Do not assume decode and prefill want the same thread count. Large CPU-MoE prefill often benefits from a wider batch thread pool and large ubatches; decode can regress when every logical core is used.

## Native options

Everything after `--` is appended verbatim to `llama-server`:

```bash
./ese serve MODEL.gguf -- \
  --jinja \
  --metrics \
  --log-verbosity 1
```

Use this escape hatch for experimental flags. Stable ESE policy options belong before `--`.

## JSON plans

Automation can consume the planner without parsing terminal text:

```bash
./ese plan MODEL.gguf --json
```

The JSON includes:

- selected policy and reason;
- all shards and total size;
- GGUF architecture, expert count, and block count when available;
- detected RAM and per-GPU memory;
- environment variables;
- argument vector;
- shell-rendered command.

## Troubleshooting

### Startup allocates most host RAM

Confirm the stream command includes `GGML_CUDA_NO_PINNED=1` and `--defer-experts`. Use `./ese plan` to verify.

### Cache mode leaves one GPU mostly idle

Check whether expert-parallel cache dispatch is active. On two or more detected NVIDIA GPUs, the launcher adds `--moe-cache-expert-parallel auto`; explicit native flags after `--` can override behavior.

### Auto chose the wrong policy

Force it with `--policy`. Include the JSON plan and server startup log in a bug report.

### Split GGUF is incomplete

Point `ese` at any correctly named shard. It validates that all `-00001-of-000NN.gguf` files are present before launching.

### The server binary is missing

Run:

```bash
./ese build
```

or set a custom binary:

```bash
ESE_SERVER=/path/to/llama-server ./ese serve MODEL.gguf
```
