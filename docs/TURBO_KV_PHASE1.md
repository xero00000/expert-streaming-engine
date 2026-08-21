# Phase 1 — Turbo KV foundation

This branch starts the first staged port from `buun-llama-cpp` without exposing an incomplete cache type to normal ESE serving.

## Pinned source

- Repository: `spiritbuun/buun-llama-cpp`
- Commit: `799e3995cd4f19aa9f6a3fa9fb5b4674422bf0ee`
- Tree: `a2bd802d81936bab8a066cbf789a427776fb4839`
- Initial source file: `ggml/src/ggml-turbo-quant.c`
- License: MIT

The commit and tree are recorded separately. A Git tree SHA is not a source revision by itself.

## What this first slice implements

The initial native foundation includes the two formats whose pinned CPU reference paths are substantive:

| Format | Values/block | Bytes/block | Exact bits/value | Status |
| --- | ---: | ---: | ---: | --- |
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

Turbo4 and Turbo8 now have pinned internal GGML/GGUF numeric IDs matching the source revision:

| Format | Internal type | Numeric ID |
| --- | --- | ---: |
| Turbo4 | `GGML_TYPE_TURBO4_0` | 44 |
| Turbo8 | `GGML_TYPE_TURBO8_0` | 48 |

Core traits report the exact 128-value block geometry and 66/130-byte storage sizes. `ggml_quantize_chunk` routes both formats through the same checked deterministic CPU reference used by the standalone tests. The integration test proves core output is byte-identical to the reference codec.

## Native CUDA row integration

CUDA `SET_ROWS` quantizes F32 rows directly into Turbo4/Turbo8 storage, and
CUDA `GET_ROWS` reconstructs F32 rows directly from that storage. Both paths
use the exact accepted reference rotation and centroid tables; they do not
stage through host memory or silently fall back to the CPU backend.

The CUDA graph test proves:

- exact encoded-byte parity with the CPU reference;
- decoded-value parity within `2e-6`;
- indexed row placement and untouched-row preservation;
- backend support for both graph operations;
- independent table initialization on every visible CUDA device.

## CUDA Flash Attention compatibility bridge

CUDA Flash Attention can now consume internal Turbo4/Turbo8 K and V tensors
without a host round trip. The bridge dequantizes each compressed tensor into
stream-ordered F16 device staging, then invokes the established CUDA Flash
Attention implementation. This preserves that implementation's mask, GQA,
softcap, and precision dispatch semantics while the direct fused Turbo tile
readers are still being developed.

The temporary device allocation is explicitly bounded at two bytes per staged
element: at most `2 * (K elements + V elements)` bytes when both tensors are
compressed. Storage comes from the CUDA stream pool and follows its stream
lifetime; there is no CPU allocation, copy, or backend fallback.

The CUDA graph test compares Turbo attention against the decoded CPU reference
for a padded 256-token K/V extent. It exercises an F16 additive mask, two-query
GQA, and logit softcap for both formats on every visible CUDA device, with a
maximum absolute-error bound of `1e-3`.

## Deliberately not exposed yet

This slice does **not** add `turbo4` or `turbo8` to:

- `--cache-type-k` or `--cache-type-v`;
- `tools/ese.py`;
- ROCm, Metal, or Vulkan;
- server save/restore or cache lifecycle operations.

The native KV parser remains an explicit whitelist, and a source-level regression test prevents the internal type names from being added accidentally. A registered storage type is still not a serving feature until backend execution, lifecycle behavior, and quality gates all pass.

## Why Turbo2, Turbo3, and TCQ are not copied in this slice

At the pinned revision:

- the Turbo2 CPU encoder records a norm and zero-fills codes;
- the Turbo3 CPU encoder is explicitly marked simplified/stub;
- Turbo3-TCQ and Turbo2-TCQ CPU paths are explicitly zero-fill stubs;
- Turbo1-TCQ CPU decode is explicitly a stub.

Those paths are useful as backend integration references, but they are not acceptable ESE correctness or fallback references. ESE will not advertise them until a complete reference and a proven native backend path exist.

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
- finite reconstruction;
- norm preservation;
- conservative normalized-MSE limits over impulse, ramp, sinusoid, and seeded-random vectors.

For the native CUDA row-codec test, configure with `GGML_CUDA=ON`, build
`test-turbo-kv-cuda`, and run the resulting executable. The test runs Turbo4
and Turbo8 on every visible CUDA device. The Turbo-specific pre-merge wrapper
does this automatically when passed `--require-cuda` (or when
`ESE_TURBO_REQUIRE_CUDA=1`).

## Promotion gates

### Gate A — core type integration — complete

- complete `ggml_type_traits`;
- checked tensor and row sizing;
- checked quantize/dequantize dispatch;
- deterministic core-vs-reference byte parity;
- pinned numeric IDs 44 and 48;
- compile-time block ABI assertions;
- native and launcher visibility guards.

The types remain internal until the later gates pass.

### Gate B — CUDA and Flash Attention — in progress

Before accepting the types as KV cache options:

- native encode/decode row kernels — complete;
- device-native Flash Attention compatibility bridge — complete;
- fused direct Turbo reads — pending;
- odd head-dimension padding;
- no host fallback;
- Ampere plus one newer NVIDIA architecture;
- prompt-processing and decode measurements separately;
- bounded temporary memory.

Current hardware evidence covers Turing sm_75 and Ampere sm_86. It verifies the
row codecs and the masked, soft-capped GQA compatibility bridge. The final
promotion gate still requires direct fused reads, odd-dimension padding,
prompt/decode measurements, and Ampere plus one newer NVIDIA architecture.

### Gate C — quality and lifecycle

Before making either type stable:

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
