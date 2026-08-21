# Turbo KV Phase 1 validation record

This record captures the pre-merge evidence for Phase 1 issue #4. Turbo and
TCQ remain internal experimental types; `ese --kv auto` does not select them.

## Environment and scope

- CUDA toolkit: 13.2
- NVIDIA driver: 595.84
- Turing: GeForce RTX 2080 SUPER, compute capability 7.5
- Ampere: GeForce RTX 3060 Ti, compute capability 8.6
- Additional available Ampere device: GeForce RTX 3080, compute capability 8.6
- Newer-than-Ampere runtime coverage: unavailable to the solo maintainer and
  explicitly waived for this release. This is an exception, not a claim of
  Ada, Hopper, or Blackwell validation.

The native CUDA test passed independently on the Turing and Ampere devices.
It requires native Turbo Flash Attention, checks F32-to-Turbo graph-copy
support, compares every encoded byte with the CPU reference, covers I32 and
I64 row indices, and exercises all fixed/TCQ tiers and mixed K/V pairings.
Maximum observed direct-attention error was below `2.1e-6`.

## Model fixtures

| Purpose | Model | SHA-256 |
| --- | --- | --- |
| F16 quality, depth, and speed | Qwen3 4B Instruct AWQ Q4_K_M | `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5` |
| Conventional lifecycle and ASAN | stories15M Q4_0 | `66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739` |
| Recurrent fail-closed behavior | mamba-130m-hf Q4_0 | `10bab040654e02f3eb28ad0ba343ede957450796cef20387cd675f582a365aa3` |

The small stories fixture is intentionally not used as quality evidence: its
out-of-distribution F16 perplexity is unstable enough that even ordinary Q8
appears to improve perplexity by roughly fourfold. The 4B model produces a
credible F16 baseline and deterministic tier comparisons.

## F16-reference quality and performance

The harness uses a 32-token prompt, 64-token teacher-forced continuation, and
KLD samples after cache depths 31, 47, 63, 79, and 95. Prompt processing is
chunked in batches of eight so the claimed direct native reader executes.
KLD percentiles below are over the five depth samples.

Quality values were byte-for-byte reproducible on Turing and Ampere:

| Tier | PPL | PPL / F16 | KLD mean | KLD p50 | KLD max |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 | 3.7553 | 1.0000 | 0 | 0 | 0 |
| Q8_0 control | 3.6992 | 0.9851 | 0.000349 | 0.000001 | 0.001618 |
| Turbo8 | 3.7700 | 1.0039 | 0.000199 | 0.000001 | 0.000827 |
| Turbo4 | 4.1012 | 1.0921 | 0.032393 | 0.000618 | 0.126646 |
| Turbo3 | 4.6977 | 1.2509 | 0.060841 | 0.007670 | 0.255727 |
| Turbo2 | 139.8253 | 37.2339 | 3.190069 | 1.592695 | 7.682060 |
| Turbo3-TCQ | 4.8501 | 1.2915 | 0.048897 | 0.009113 | 0.209137 |
| Turbo2-TCQ | 14.6934 | 3.9127 | 3.065564 | 1.012689 | 12.586538 |
| Turbo1-TCQ | 2654.9982 | 706.9957 | 7.193900 | 8.518781 | 10.137912 |

The results deliberately prevent an unsafe promotion: Turbo8 is the only tier
near the F16 reference on this model, while the lower tiers require an explicit
model-specific quality budget and sensitivity order. The VBR solver treats
quality cost as a hard ceiling and fails rather than crossing it.

Separate prompt/decode throughput (tokens/second):

| Tier | Turing prompt | Turing decode | Ampere prompt | Ampere decode |
| --- | ---: | ---: | ---: | ---: |
| F16 | 264.4 | 100.1 | 253.5 | 109.0 |
| Q8_0 control | 314.4 | 97.2 | 317.0 | 105.8 |
| Turbo8 | 95.5 | 24.0 | 80.7 | 18.8 |
| Turbo4 | 103.6 | 26.4 | 89.9 | 20.3 |
| Turbo3 | 104.6 | 27.1 | 90.6 | 20.4 |
| Turbo2 | 104.9 | 27.6 | 91.7 | 20.8 |
| Turbo3-TCQ | 185.8 | 41.7 | 184.4 | 41.2 |
| Turbo2-TCQ | 216.9 | 48.8 | 213.3 | 48.4 |
| Turbo1-TCQ | 220.4 | 50.4 | 213.3 | 49.0 |

These are validation measurements, not speedup claims. The scalar-reference-
compatible Gaussian rotation in the fixed CUDA kernels is currently slower
than F16; the types therefore remain explicit and experimental.

## Lifecycle and bounded memory

The conventional lifecycle test covers two active slots, sequence copy,
shift, remove, defrag, incompatible checkpoint rollback, representation-aware
save/restore, injected retier failure, injected resize failure, shrink, expand,
and F16-to-Q8-to-F16 migration. Release and AddressSanitizer/LeakSanitizer runs
pass. Migration reports a maximum per-head host staging allocation of 0.60 KiB
for the fixture and never allocates a cache-sized host temporary.

The Mamba fixture successfully decodes before a Q8 retier request is rejected.
The layer representation remains unchanged, proving recurrent caches fail
closed rather than being interpreted as attention KV rows.

Unsupported live layouts (hybrid, transposed-V, MLA, and split caches) also
fail before allocation or mutation. A replacement cache is published only
after all occupied rows have converted successfully.

## Reproduction

Build the normal CPU and CUDA test targets, then run:

```bash
scripts/test-turbo-kv.sh
test-turbo-kv-core
test-turbo-kv-policy
test-turbo-kv-cuda
test-turbo-kv-lifecycle stories15M-q4_0.gguf
test-turbo-kv-recurrent mamba-130m-hf.Q4_0.gguf
test-turbo-kv-quality Qwen3-4B-Q4_K_M.gguf
```

For architecture isolation, set `CUDA_VISIBLE_DEVICES` before the CUDA and
quality executables. Set `ESE_TURBO_CUDA_DEVICE` when multiple devices remain
visible.
