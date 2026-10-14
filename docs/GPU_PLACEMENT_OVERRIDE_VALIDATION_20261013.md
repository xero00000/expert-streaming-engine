# Manual GPU placement validation — 2026-10-13

## Scope

The launcher and native resource plan now follow explicit CUDA placement before
choosing budgets, split weights and calibration identity. This validation uses
Ubuntu 24.04 builds, a generated three-layer Llama model and three GPUs: RTX
3080, RTX 3060 Ti and RTX 2080 SUPER. Production Qwen support remains gated off.

Evidence is archived outside the source tree at
`/home/xero/Desktop/test/results/ese-manual-gpu-placement-20261013`.

## Reproduced failures

The previous launcher counted every CUDA-visible card even when `--device`
selected one card or disabled GPU execution. Reordered device lists retained
split weights in the original order. Forwarded native GPU-layer settings did
not participate in automatic planning.

The previous native automatic fit changed `main_gpu` even in single-device
mode. Actual requests for main GPU 0, main GPU 1, and reordered CUDA2/CUDA0 main
GPU 0 initialized the wrong backend and crashed on completion. Main GPU 2
completed but its resize preparation rejected extra accelerators in the plan.
A CUDA2-only launch with a declared 1 MiB transient reservation also failed:
its selected-list index 0 was incorrectly used as physical resource ID 0.

Strict CPU UBSan separately found an uninitialized result boolean during an
error-response copy. A permanent regression fills result storage with 0xa5,
then checks direct and queued copies. It fails against the archived old header
with invalid bool value 165, and passes after scalar defaults are initialized.

## Changes

- Recognized native device, split mode, main GPU, GPU-layer and tensor-split
  arguments are read before planning. Supported aliases and underscore forms
  follow the native parser. Arguments consumed as prompt, filename or other
  values are preserved literally.
- CUDA budgets and automatic split weights use the requested device order.
  Single-device mode resolves the main index in that list. Explicit zero GPU
  layers disables automatic GPU budgets; existing resident-policy restrictions
  on fixed layer counts remain enforced.
- Tensor split values must be finite, nonnegative and match selected CUDA
  devices. Duplicate/unavailable devices and invalid main indices are rejected.
- Calibration identity follows selected physical UUID order. CPU execution
  avoids GPU calibration; incompatible graph/attention layouts disable
  automatic mixed calibration.
- Native planning maps the main selected-list index to its actual resource
  device ID and includes only the main device for single-device execution.
  Automatic fit preserves that requested main index.
- Server result flags, counters and timing values have defined defaults.
  Existing unknown prompt/prediction count sentinels remain -1.

## Completed checks

| Check | Result |
| --- | --- |
| Python launcher/profile suite on host and Ubuntu | 62/62 each |
| Studio package/staging controls | 28/28 |
| CUDA focused native suite | 7/7 |
| CPU focused native suite | 6/6 |
| Strict CPU ASan/UBSan/leak suite | 6/6 |
| CUDA selected-context memcheck, all three cards visible | No API errors, memory errors or leaked bytes |
| Current release server placement matrix | 13/13 |
| Strict sanitizer server resize | Exact same-build output, no sanitizer errors |

Seven native resource-plan cases check first/last cards, both orders of a
selected pair, both main indices and CPU-only selection. Each checks a declared
1 MiB transient reservation, actual backend identity and successful decode.
This is reservation metadata; no MTP model is loaded.

The 13 release server cases cover CUDA2, GPU-disabled execution, each main GPU,
reordered pairs, explicit 25/75 split, forwarded zero GPU layers, numeric CUDA
visibility and UUID visibility, plus a CPU runtime. Each retains saved content
and all 32 probabilities exactly against the independently archived CPU or
same-physical-GPU release response. Twelve real-plan cases preserve exact
output through 512→256→512 with dry-run and committed resize. Fixed manual
zero-layer execution rejects resize without a native plan as expected.

The CPU sanitizer server uses its own initial 512-capacity response as the
teacher for exact shrink/regrow comparison. A preliminary attempt incorrectly
compared the instrumented build with an optimized release build; that failed
observer is retained and excluded. It establishes no cross-build equivalence.

Libraries loaded by servers are checked against the frozen runtime; build and
CUDA SDK paths are excluded. CUDA vendor library bytes remain unchanged.

## Limits

CUDA global NCCL initialization still allocates on all visible cards even when
model execution selects one card or CPU. These controls verify selected native
backends and output, not exclusive GPU allocation. Non-CUDA accelerator device
arguments retain legacy passthrough and have no hardware qualification here.

No kernels, model weights, external references, numerical tolerances or
production architecture gates changed. Original Qwen numerical, full-model
64K, concurrent serving, matched performance and complete release gates remain
open. Linux candidate packages predate these source changes and require a new
build before publication. Candidates are unsigned and unpublished; native
Windows and signing validation remain incomplete.
