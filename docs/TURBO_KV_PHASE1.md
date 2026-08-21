# Phase 1 — Turbo KV foundation

This branch starts the first staged port from `buun-llama-cpp` without exposing an incomplete cache type to normal ESE serving.

## Pinned source

- Repository: `spiritbuun/buun-llama-cpp`
- Commit: `799e3995cd4f19aa9f6a3fa9fb5b4674422bf0ee`
- Tree: `a2bd802d81936bab8a066cbf789a427776fb4839`
- Initial source file: `ggml/src/ggml-turbo-quant.c`
- License: MIT

The commit and tree are recorded separately. A Git tree SHA is not a source revision by itself.

## What the fixed-codec foundation implements

The native foundation includes complete deterministic ESE CPU references for
all four fixed Turbo formats:

| Format | Values/block | Bytes/block | Exact bits/value | Status |
| --- | ---: | ---: | ---: | --- |
| Turbo2 | 128 | 40 | 2.5 | CPU reference + native CUDA row codec |
| Turbo3 | 128 | 56 | 3.5 | CPU reference + native CUDA row codec |
| Turbo4 | 128 | 66 | 4.125 | CPU reference + native CUDA row codec |
| Turbo8 | 128 | 130 | 8.125 | CPU reference + native CUDA row codec |

The implementation preserves the pinned design:

1. deterministic Gaussian rotation generated from seed `42`;
2. modified Gram-Schmidt orthogonalization;
3. extraction of the 128-value L2 norm;
4. rotation of the normalized block;
5. nearest Lloyd–Max centroid assignment;
6. reconstructed-norm correction stored as FP16;
7. inverse rotation during decode.

ESE wraps the reference in a small checked C ABI. It validates alignment, sizes, overflow, null arguments, and non-finite source values. Rotation initialization uses a C++ function-local immutable object, avoiding the unsynchronized mutable lazy globals in the source revision.

## Internal core integration

The fixed formats have pinned internal GGML/GGUF numeric IDs matching the
source revision:

| Format | Internal type | Numeric ID |
| --- | --- | ---: |
| Turbo3 | `GGML_TYPE_TURBO3_0` | 43 |
| Turbo4 | `GGML_TYPE_TURBO4_0` | 44 |
| Turbo2 | `GGML_TYPE_TURBO2_0` | 45 |
| Turbo8 | `GGML_TYPE_TURBO8_0` | 48 |

Core traits report each format's exact 128-value block geometry and storage
size. `ggml_quantize_chunk` routes all four formats through the same checked
deterministic CPU references used by the standalone tests. The integration
test proves core output is byte-identical to the reference codec.

## Native CUDA row integration

CUDA `SET_ROWS` quantizes F32 rows directly into fixed Turbo storage, and
CUDA `GET_ROWS` reconstructs F32 rows directly from that storage. Both paths
use the exact accepted reference rotation and centroid tables; they do not
stage through host memory or silently fall back to the CPU backend.

The CUDA graph test proves:

- exact encoded-byte parity with the CPU reference;
- decoded-value parity within `2e-6`;
- indexed row placement and untouched-row preservation;
- backend support for both graph operations;
- independent table initialization on every visible CUDA device.

## CUDA Flash Attention integration

CUDA Flash Attention can now consume internal fixed Turbo K and V tensors
without a host round trip. Decode batches of up to eight queries use a native
kernel that reads compressed codes directly, rotates Q into the compressed
domain, performs online softmax, accumulates V in the compressed domain, and
inverse-rotates the result. It supports all 16 fixed-tier K/V pairings,
masks, GQA, softcap, ALiBi slopes, multiple sequences, and attention sinks.
This path allocates no cache-sized temporary buffer.

Larger prompt-processing batches retain the compatibility route: each
compressed tensor is dequantized into stream-ordered F16 device staging before
the established CUDA Flash Attention implementation runs. This preserves the
existing prefill dispatch while a tiled direct prefill reader is developed.

The prefill temporary device allocation is explicitly bounded at two bytes per staged
element: at most `2 * (K elements + V elements)` bytes when both tensors are
compressed. Storage comes from the CUDA stream pool and follows its stream
lifetime; there is no CPU allocation, copy, or backend fallback.

The CUDA graph test requires the native path and compares Turbo attention against the decoded CPU reference
for a padded 256-token K/V extent. It exercises an F16 additive mask, two-query
GQA, logit softcap, all 16 K/V format pairings, and every visible CUDA device,
with a maximum absolute-error bound of `1e-3`. Setting
`GGML_TURBO_KV_REQUIRE_NATIVE_FATTN=1` makes an unsupported shape fail instead
of silently using staging; the CUDA test enables this guard.

## Per-head padding for unusual dimensions

Turbo storage blocks contain 128 values, so a logical K or V head that is not
128-aligned must never share a block with the next head. ESE now rounds each
Turbo head independently, zero-pads K, V, and Q before Flash Attention, and
crops the attention result back to the model's logical V width. Non-Turbo cache
geometry is unchanged.

The physical padded geometry is used consistently by whole-cache and split-
cache allocation, cache writes, attention views, defragmentation, K shift,
memory accounting, and state serialization. Turbo cache allocation is rejected
when Flash Attention is disabled because the legacy transposed-V layout is not
a valid Turbo execution path.

The CUDA graph test covers both a naturally aligned 128-wide head and a
96-wide head padded to 128 for every fixed format on every visible GPU. The odd-
width case retains the mask, two-query GQA, softcap, and `1e-3` error gate.

## Deliberately not exposed yet

This slice does **not** add `turbo2`, `turbo3`, `turbo4`, or `turbo8` to:

- `--cache-type-k` or `--cache-type-v`;
- `tools/ese.py`;
- ROCm, Metal, or Vulkan;
- a claim that the full serving cache lifecycle has passed validation.

The native KV parser remains an explicit whitelist, and a source-level regression test prevents the internal type names from being added accidentally. A registered storage type is still not a serving feature until backend execution, lifecycle behavior, and quality gates all pass.

## Why the pinned Turbo2/Turbo3 encoders were replaced

At the pinned revision:

- the Turbo2 CPU encoder records a norm and zero-fills codes;
- the Turbo3 CPU encoder is explicitly marked simplified/stub;
- Turbo3-TCQ and Turbo2-TCQ CPU paths are explicitly zero-fill stubs;
- Turbo1-TCQ CPU decode is explicitly a stub.

Those paths are useful as layout and backend integration references, but they
are not acceptable ESE correctness or fallback references. ESE therefore
retains the pinned fixed-format storage geometry and Lloyd–Max centroids while
providing complete Turbo2/Turbo3 quantize and dequantize references. Their core,
CUDA row, padding, and direct decode-attention paths are covered by the same
gates as Turbo4/Turbo8. TCQ remains unadvertised until its complete reference
and native backend paths exist.

## Tests

Run:

```bash
bash scripts/test-turbo-kv.sh
```

The test compiles the reference module independently and verifies:

- exact block size and bits/value accounting;
- aligned-size calculations;
- invalid/unsupported/buffer error paths;
- non-finite input rejection;
- exact zero round-trip;
- deterministic byte encoding;
- pinned FNV-1a hashes for seeded-random encoded reference vectors;
- finite reconstruction;
- norm preservation;
- conservative normalized-MSE limits over impulse, ramp, sinusoid, and seeded-random vectors.

For the native CUDA row-codec test, configure with `GGML_CUDA=ON`, build
`test-turbo-kv-cuda`, and run the resulting executable. The test runs all fixed
formats and all 16 K/V pairings on every visible CUDA device. The Turbo-specific pre-merge wrapper
does this automatically when passed `--require-cuda` (or when
`ESE_TURBO_REQUIRE_CUDA=1`).

## Promotion gates

### Gate A — core type integration — complete

- complete `ggml_type_traits`;
- checked tensor and row sizing;
- checked quantize/dequantize dispatch;
- deterministic core-vs-reference byte parity;
- pinned numeric IDs 43, 44, 45, and 48;
- compile-time block ABI assertions;
- native and launcher visibility guards.

The types remain internal until the later gates pass.

### Gate B — CUDA and Flash Attention — in progress

Before accepting the types as KV cache options:

- native encode/decode row kernels — complete;
- device-native Flash Attention compatibility bridge — complete;
- fused direct Turbo decode reads — complete for batches up to eight;
- tiled direct Turbo prefill reads — pending;
- odd head-dimension padding — complete at allocation and graph level;
- no host fallback;
- available NVIDIA hardware (the solo-maintainer Phase 1 release explicitly
  waives runtime coverage newer than Ampere);
- prompt-processing and decode measurements separately;
- bounded temporary memory.

Current hardware evidence covers Turing sm_75 and two Ampere sm_86 devices. It
verifies the row codecs and native masked, soft-capped GQA decode at aligned and
odd logical head widths. Runtime coverage newer than Ampere is unavailable and
is recorded as a maintainer-approved exception, not presented as tested. The
final promotion gate still requires tiled direct prefill reads and separate
prompt/decode measurements.

### Gate C — quality and lifecycle

Before making any fixed type stable:

- F16 reference;
- perplexity;
- KLD distribution;
- task-quality panel;
- context-depth sweep;
- one- and multi-slot serving;
- resize/defrag/shift behavior;
- save/restore metadata;
- failure-atomic allocation and rollback.

### Gate D — VBR

Only after fixed formats are stable:

- explicit per-layer/per-side tier map;
- model sensitivity ordering;
- deterministic VRAM/context/quality solver;
- observable current tier;
- retiering rollback;
- quality floor that cannot be silently crossed.

## Current user-facing status

The supported launcher cache choices remain:

```text
auto
f16
q8_0
q4_0
```

Turbo and TCQ remain implementation work, not user-facing claims.
