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

## October 9 continuation: bounded CPU lease groups

`ESE_CPU_EXPERT_LEASE_CHUNKING=1` enables ordinary CPU `MUL_MAT_ID` scheduling
across groups of at most four leased expert components when IQK and expert
chunking are compiled in. One thread acquires the group; workers share its
weights until every output-row chunk completes. Admission pressure reduces the
group, including to one expert, and all leases are released before advancing.
This avoids the old per-thread acquisition pattern in which workers can advance
to different experts and exhaust a tiny cache. Required leases still prohibit
using the original model tensor as a fallback. The option is experimental and
does not change launcher defaults.

The option now rejects graph execution when either required build feature is
compiled out. `ESE_CPU_EXPERT_LEASE_TRACE=1` records actual grouped calls for
qualification. The existing Qwen diagnostic backend has chunking OFF; its first
timing sweep therefore did not exercise this feature and is excluded. The
enabled lane uses the normal Release GGML backend as a per-run preload into the
gated probe executable/libllama. Both ablation modes use that same backend. No
existing build setting or production architecture gate was changed.

The candidate preserves the leased path's row partitions and its IQK/GEMV/vec-dot
arithmetic. Independent tests poison the original weights and check exact output
with one- and four-component RAM capacities, repeated nodes, forced eviction,
padding and broadcast activations. They cover F32 and production-shaped IQ3_S,
IQ4_NL and decode-only Q2_0. Successful acquisition counts fall from per-worker
requests to one per active expert; this alone is not a throughput claim.

A separate correctness fix replaces the unleased chunk path's negative work
counter sentinel with an independent atomic failure flag. Remaining workers
cannot increment away failure, and resetting the counter cannot change the
verdict observed by slower threads. The old code hung in sanitizer and Release
fallback checks; the final CPU, three-GPU cache suites and strict ASAN/UBSAN
checks pass.

Matched real-model probes preserve all 3,973,120 captured logits and MoE/QSA
routes, both for the ordinary control and for fixed 5/5 hybrid execution with a
64 MiB RAM cache and 4 MiB staging allowance. This is a bounded 16-step diagnostic,
not upstream model or long-context qualification. Evidence, rejected GPU-quota
trial and timing details: `/home/xero/Desktop/test/results/ese-resident-quota-20261009/`.

The enabled-backend ablation uses three repeats each of raw Hi (32 predictions)
and Logs256 (16 predictions), fixed 5/5 placement, the same 64 MiB RAM / 4 MiB
staging and 2680 MiB expert VRAM per GPU. Shared leases improve median decode
`3.252 -> 4.093` and `2.736 -> 3.024` tokens/s, with all six paired outputs
matching. Both modes have identical RAM misses and logical source/GPU bytes;
duplicate worker acquisition hits decrease. OS reuse and clocks are uncontrolled,
and these short local results do not promote a new general default. The enabled
TinyMoE CPU-only lane also matches 32 generated tokens and traces actual calls.

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
