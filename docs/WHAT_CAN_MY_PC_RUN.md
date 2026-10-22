# What can my PC run?

Start with the PC you have. The complete model file does not always need to fit
in your GPU’s memory: supported sparse models can keep a bounded set of expert
weights in RAM/VRAM and read others from SSD. That saves resident weight memory;
it can also increase latency. Dense models do not gain the same expert-streaming
advantage.

## Check your hardware before downloading

Run `ese doctor` (Windows: `ese.exe doctor`) to see available RAM and GPUs. Use
available memory, not just the installed capacity: the desktop and other apps
already consume some. Make sure the SSD can hold **every model shard**, plus
room for downloads and normal system use.

In Studio’s Model hub, recommendations are memory-fit estimates. A model being
selectable or estimated to fit does not mean its 64K workload has been tested.

## Choose the runtime

| Your PC | Package | What to check |
| --- | --- | --- |
| CPU-only machine | Linux `cpu`, or the Windows installer with its separate CPU runtime | Model architecture, available RAM, SSD read speed and measured CPU throughput |
| NVIDIA gaming GPU with limited VRAM | Linux `cuda`, or the Windows installer | NVIDIA driver, free VRAM, host RAM and SSD; use an exact tested profile when available |
| Larger NVIDIA GPU or several GPUs | CUDA runtime | Per-card headroom and the actual selected devices; adding VRAM does not guarantee proportional speed |

AMD/Intel accelerator execution is not qualified by this release’s hardware
checks. Linux CPU operation remains an option on those PCs.

## Inspect a 64K plan

After downloading a supported GGUF, inspect it without starting the model:

```bash
ese plan /models/model.gguf --context 65536
```

For split GGUFs, keep all correctly named shards together. Pass one shard and
ESE validates the complete set. Check the selected policy, context, KV type,
RAM expert cache, GPU cache and device list before launching.

`resident` uses ordinary model placement when it fits. `cache` retains expert
weights within RAM/VRAM budgets. `stream` defers expert weights to storage.
These are implementation choices, not speed ratings. A lower-bit model may
consume fewer bytes but produce different answers; record the exact quant.

## Separate a limit from a measured result

A context setting of 65,536 means capacity was requested. To demonstrate a useful
64K workload, record the actual prompt-token count, generation after that fill,
time to first token, sustained speed and a retrieval check at several depths.
Longer context also needs cache/workspace memory and can slow attention.

A RAM-cache setting is not total process RAM. Read peak RSS, the OS/cgroup memory
measurement and GPU process memory separately. A benchmark on a 48 GiB host
with a 16 GiB process limit is a constrained test; it is not a physical 16 GiB
PC test, and shared page-cache effects must be disclosed.

## Tested configurations

| Configuration | Completed verification | Limit |
| --- | --- | --- |
| Linux CPU AppImage/DEB/RPM | Loader/launcher, driverless small-model API and cache controls; Ubuntu/Fedora package installation | Small generated test model; no large-model CPU speed claim |
| Linux CUDA AppImage/DEB/RPM | Native dependencies, small-model API/cache controls, NVIDIA GPU identity and package installation | Exact recorded controls; no universal model or filled-64K claim |
| Windows MSI/NSIS | Actual installation, CPU fallback, runtime identities and short reference-output checks; signed NSIS installed-update checks | No Authenticode signatures; physical Windows GPU inference is deferred and unqualified |
| New low-memory flagship | Pending | Publish the exact model and hardware row only after actual filled-64K and sustained checks pass |

See the [launch demo](LAUNCH_DEMO.md) for the reproducible flagship record,
[installation](install.md) for package choices, and
[reference benchmarks](ESE_BENCHMARKS.md) for historical measurements.
