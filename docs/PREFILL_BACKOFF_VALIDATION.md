# Budget-backed prefill reservation — 2026-10-11–12

Context initialization applies per-backend graph arena budgets before reservation.
The appended native `llama_context_params.compute_buffer_budget_bytes` parameter
sets a common per-backend cap. Zero queries current free CUDA memory and preserves
the configured `expert_vram_reserve_bytes`; CPU and other backends remain
unlimited unless explicitly capped. The cap includes live and replacement graph
arenas. Internal scheduler hooks forward admission and statistics to gallocr.
The new header is private to these consumers, avoiding another rebuild of every
CUDA kernel through a shared backend header.

When reservation fails, initialization retains the existing retry without
pipeline parallelism first. It then halves a causal microbatch, down to one
token, resetting scheduler assignments between attempts. Successful backoff
persists the effective `n_ubatch` and clamps the worst-output reservation bound.
Non-causal attention and detected encoder contexts retain full-batch reservation;
they fail when that reservation cannot fit. No expert selection, weight bytes,
cache format or comparison tolerance is changed.

The initial test exposed a separate CPU-only loader bug: split mode NONE indexed
`model.devices[0]` while the device list was empty. The retained GDB backtrace
locates the access in tensor placement. Zero-offload selection now uses the CPU
buffer type before considering any GPU lookup. Initial ASAN execution aborted
and the initial Release suite reported a segfault; these attempts are excluded
from final qualification.

Evidence: `/home/xero/Desktop/test/results/ese-prefill-backoff-20261011`.

## Qualified component scope

A generated one-layer Llama GGUF contains deterministic, nonzero F32 weights.
Tests load the real model, initialize real contexts and decode actual prompts.
They compare an uncapped control with admitted bounded execution using the
unchanged `2e-5 * (1 + abs(reference))` bound.

| Execution | Requested chunk | Admitted chunk | Per-backend cap | Largest tracked arena peak |
| --- | ---: | ---: | ---: | ---: |
| CPU | 16 | 4 | 12,138 bytes | 10,848 bytes |
| Each isolated GPU | 16 | 4 | 42,602 bytes | 42,080 bytes |
| CPU, 1024-token context | 1024 | 64 | 204,180 bytes | 172,800 bytes |
| Each isolated GPU, 1024-token context | 1024 | 64 | 310,316 bytes | 279,296 bytes |

The 1024-request case decodes 128 tokens, exercising two admitted chunks.
Every output passes the unchanged bound. In the 16-token case CPU and Turing
results are exact; the Ampere maximum error is 1.78813934e-7. Peaks in the table
are the largest individual scheduler arena, including CPU arenas in GPU runs;
they are not total process RAM or total-device VRAM measurements.

- Current core suites pass 25 Release and 18 CPU ASAN/UBSAN/leak regressions.
  Subsequent focused runs qualify the strengthened 128-token prompt and automatic
  reserve checks without changing the core runtime source.
- An impossible one-byte cap and a non-causal request fail cleanly. A pre-existing
  context's complete serialized session remains byte-exact after failures and
  independent context creation. The effective chunk/output bounds remain valid.
- Automatic CUDA admission rejects a reserve that leaves no graph room. The old
  session remains exact. This tests the free-memory/reserve branch separately
  from the explicit-cap cases.
- Native and unsuppressed all-API/leak memcheck pass on the 3060 Ti, 2080 SUPER
  and 3080 under per-device UUID isolation: all terminal 0, errors 0, leaked bytes
  0. Each test reports its physical UUID in the execution manifest.
- A freshly rebuilt device-selection fixture passes single GPU, noncontiguous
  CUDA0/CUDA2 selection, last-device-only selection, CPU-only and mapped CPU
  modes. Its native pass is not an all-visible GPU sanitizer qualification.
- A same-ABI control retains budget admission but disables causal retries. The
  current fixture fails with exit 1 at bounded context creation. Positive tests
  pass. Both positive and negative CPU execution enable leak checking.

## Limits

This qualifies small-model reservation/backoff and component output recovery.
It does not establish the original Qwen artifact's microbatch-1024 result,
whole-device peak VRAM, 64K context or general model/encoder/multi-GPU pipeline
equivalence. Actual pipeline-disable fallback and encoder-family execution are
not exercised by this fixture.

The cap covers graph arenas. Fixed model/KV/output allocations, graph metadata,
CUDA private pools, vendor/driver overhead and aggregate context limits remain
separate. Host metadata OOM still uses GGML assertions. CUDA free memory is a
snapshot; other processes can change it after admission. Runtime graph growth
retains the admitted cap, but this stage implements initialization backoff,
not a complete dynamic resource controller.

Appending the context parameter changes the native parameter-structure ABI:
rebuild callers and libraries together. Normal Qwen loading stays gated, the
original 33-token numerical comparison remains failed, and no whole-model
speed or release promotion is claimed.
