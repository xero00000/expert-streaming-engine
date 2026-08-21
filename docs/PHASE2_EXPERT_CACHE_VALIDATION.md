# Phase 2 expert-cache validation

This document records the acceptance evidence for the bounded native
`storage → RAM → VRAM` expert hierarchy. The machine-readable runner is
`scripts/validate-phase2-expert-cache.py`.

## CPU and storage evidence

Validation model: `FlameF0X/TinyMoE-100m-2x8-retrained`, converted to a local
F16 GGUF with 10 MoE layers, 8 experts per layer, and 240 checked component
descriptors.

The deterministic prompt `The answer is`, seed 4242, temperature zero produced
the same output with the original tensors and the asserted lease-only cache:

```text
let's let us they new together
```

These storage/RAM reference runs explicitly use `-ngl 0`, even when the
acceptance binary has CUDA enabled. GPU visibility and offload are scoped only
inside the numbered GPU lanes, preventing an unrelated automatic multi-GPU
placement from contaminating the CPU baseline.

Both `mmap` and `pread` passed exact output parity while forcing 514 evictions.
The configured RAM capacity was 8,388,608 bytes and peak logical residency was
8,257,536 bytes. Each backend read 311,427,072 bytes through its claimed path.
Resident payloads come from one preallocated 64-byte-aligned arena whose size
is exactly the configured capacity, so promotion cannot transiently allocate
past the bound; alignment gaps reduce usable capacity instead of escaping it.
The source expert mappings were marked for deferred residency during these
runs, so correctness did not rely on retaining the original expert pages.

Ten cold/warm repetitions produced these distributions:

| Measurement | Minimum | Median | p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Cold prompt | 14.30 ms | 15.71 ms | 16.45 ms | 16.45 ms |
| Warm decode, per token | 3.58 ms | 3.93 ms | 4.11 ms | 4.11 ms |

Native `io_uring` output parity and 311,427,072 claimed-path bytes were proven
in an unrestricted local run. The managed validation sandbox denies
`io_uring_setup`; in that environment the backend fails explicitly with
`Operation not permitted` and never falls back to `pread`.

The checked descriptor/RAM-cache unit test passes with GCC and with Clang
ASAN+UBSAN, and the sanitizer configuration is a dedicated CI job. It covers
bounds, overflow, overlap, alignment, source identity,
canonical descriptor identity across cache clears, quantized geometry,
deterministic eviction, pin lifetime, concurrent fills,
64-byte lease alignment, exact CPU `MUL_MAT_ID` lease parity after the original
selected expert tensors are deliberately corrupted, and (when CUDA is
available) bit-exact full-tensor versus compact/remapped `MUL_MAT_ID` and fused
MoE outputs at TinyMoE's real matrix dimensions. LeakSanitizer
also passed in an unrestricted run; the managed sandbox cannot run LSAN under
its ptrace policy.

## GPU matrix

The CUDA build compiles for both locally available architecture families:

| GPU | Compute capability | Family |
| --- | ---: | --- |
| GeForce RTX 2080 SUPER | 7.5 | Turing |
| GeForce RTX 3060 Ti | 8.6 | Ampere |
| GeForce RTX 3080 | 8.6 | Ampere |

Run the cache-on/cache-off parity and hard-bound matrix with:

```bash
python3 scripts/validate-phase2-expert-cache.py \
  --cli build-cuda/bin/llama-cli \
  --model TinyMoE-100m-2x8-fixed-f16.gguf \
  --runs 10 \
  --gpu-counts 1,2,3
```

For every GPU count the runner requires exact repeated output under identical
batching/offload settings, per-device residency no greater than capacity,
non-zero lease-backed uploads, structured telemetry, and zero original-tensor
fallbacks in asserted sidecar-only mode. CPU storage lanes separately require
exact cache-on/cache-off token and output parity. CUDA selected-intermediate
and output parity is enforced in `test-expert-cache` by comparing full expert
tensors with the exact compact/remapped representation on the same backend;
this avoids treating expected CPU-versus-CUDA rounding differences in the
small, low-margin TinyMoE fixture as cache corruption. The two- and three-GPU
lanes use layer split with equal shares because TinyMoE's one-token graph split
places every routed activation on device 0. They require every requested
device to own an active cache. Capacity includes the route-remap arena.
Missing CUDA devices, per-device participation, or telemetry is a hard failure.
The report inventories GPU UUIDs, names, compute capabilities, memory, and
driver version through `nvidia-smi`, then selects devices by UUID so the
evidence is not ambiguous under CUDA device reordering. It also records the
Git commit and SHA-256 hashes of the CLI and model. GPU acceptance refuses a
dirty source tree, so the result identifies the exact candidate under test.

The cache uses one dedicated CUDA transfer stream per participating device.
Upload events make new slots visible to compute; compute-completion events make
prior slot reads visible to the transfer stream before reuse. Neither edge
globally synchronizes the compute stream. Decode misses are lease uploads,
while larger prompt graphs remain on the established CPU-MoE path where the
graph scheduler can overlap independent GPU work; this is the only workload
shape where CPU miss computation is treated as profitable.

The clean consolidated candidate passed the 1/2/3-GPU matrix on the inventory
above: each lane reported all
requested devices, 420 lease-backed uploads, zero forced fallbacks, and bounded
RAM/VRAM residency. `mmap`, `pread`, and native `io_uring` storage lanes also
passed. The machine-readable report was generated by the command above and is
attached to the pull request evidence.

## Maintainer hardware exception

Ada-or-newer runtime coverage is waived for this phase because the solo
maintainer has Turing and Ampere hardware only and cannot reasonably acquire a
newer GPU for this gate. This is a coverage exception, not permission for a
silent fallback: CUDA compilation, Turing/Ampere runtime evidence, bounds,
events, and sidecar-only assertions remain mandatory. The machine-readable
report records the exception as
`architecture_coverage.ada_or_newer = maintainer-waived-unavailable-hardware`;
it does not attempt to turn unavailable hardware into a synthetic pass.
