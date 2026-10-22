# CUDA workspace-aware prefill admission — 2026-10-12

Context initialization prepares known explicit-F32 CUDA matrix workspaces
before committing graph arenas. It refreshes memory headroom after pool backing
and cuBLAS handle creation, then uses the existing causal microbatch backoff.
This fixes the retained original-model pressure case that previously admitted
an arena size whose execution workspace could not fit.

Evidence: `/home/xero/Desktop/test/results/ese-workspace-prefill-plan-20261012`.
Normal Qwen loading remains gated; both external model shards are read-only.

## Implementation scope

The private scheduler interface separates graph placement from arena reservation.
The CUDA estimator examines the same dispatch predicates used by execution.
It recognizes explicit-F32, single-device, contiguous two-dimensional matrix
weight materialization, and excludes eligible scalar, small, direct Q6 and
opt-in direct BF16-head paths that do not need that materialization. Unknown
layouts and precision modes return unplanned; the estimator does not claim
coverage for every CUDA operator.

For each selected backend, initialization caches backing for its largest known
materialization. Active workspace leases prevent this preparation. Temporary
pool admission uses the existing physical cap and shared graph/pool ledger.
The cuBLAS handle is created before refreshing free-memory admission. No model
arithmetic, reference weights, tolerances or expert choices were changed.

## Actual pressure-case execution

The original quantized Qwen artifact uses the selected 3060 Ti/3080 pair, host
Q8 KV, Flash Attention, 131072-token context capacity, CPU experts, a 64 MiB
expert RAM cache, 4 MiB staging and a deliberately raised 4500 MiB VRAM reserve.
The input contains 128 tokens; logical batch and requested microbatch are 1024.

| Source generation | Admitted microbatch | Execution |
| --- | ---: | --- |
| Preserved arena-only admission | 256 | Workspace rejection; decode -2 |
| Current workspace preparation | 128 | Completes 128 tokens; position 127 |
| Preserved explicit-128 control | 128 | Completes 128 tokens; position 127 |

Current initialization prepares 120 MiB per CUDA backend, then rejects graph
reservations for 1024, 512 and 256 before admitting 128. Pool backing stays
125829120 bytes per backend, active bytes zero, across execution. Graph backing
is 910550272 and 908715264 bytes; combined reservations are 1036379392 and
1034544384 bytes. Shared-ledger rejections are zero; graph rejection counts
record the three intentional backoff attempts.

All 248320 final-token logits and all 6144 ordered layer/token expert lists are
byte-identical to the preserved explicit-128 control; input IDs also match.
This establishes before/after consistency at the same effective chunk size.
Q8/Flash numerical quality against the F32 reference remains unqualified.

Native process VRAM samples peak at 3202351104 and 3498049536 bytes. Sampled RSS
peaks at 29527916544 bytes. The run takes 135.35 seconds including load, capture,
artifact extraction and teardown; this is not a throughput benchmark. The
context contains 128 tokens, not 64K or 128K filled context.

The same automatic pressure profile completes under the selected Ampere-pair
all-API/leak memcheck: terminal exit 0, all 128 tokens executed, zero errors and
zero bytes leaked. Input IDs, final logits and all ordered routes are byte-exact
to native execution. Instrumented process VRAM samples peak at 3235905536 and
3537895424 bytes. Its 868.69-second wall time includes instrumentation and is
not throughput evidence. Its RSS sample describes the sanitizer parent, not
the model child, and is excluded from model-RAM claims.

## Numerical and component checks

A fresh F32/non-Flash, host-KV nine-token batch at context capacity 4096 passes
all 2234880 logits at the unchanged `2e-5 * (1 + abs(reference))` bound against
the pinned reference and preserved sequential control. All 432 ordered expert
lists match. This scope does not include hidden boundaries or saved states.

Native VMM and forced-legacy pool fixtures pass on all three GPUs. The explicit
workspace test checks a rejected preparation before backing, successful cache
preparation, repeat reuse and exact projection output. Fresh processes cover
the direct BF16 head enabled and disabled, its planned workspace difference,
and exact output. Unknown precision leaves the caller's estimate unchanged.
The legacy diagnostic differs from the production source only in its pool
selection predicate; transformation and frozen binary identities are retained.
Six isolated all-API/leak component memchecks cover VMM and legacy pools on
all three GPUs, each terminating with zero errors and zero leaked bytes.

The current Release regression group passes 27 tests. The earlier completed
CPU ASAN/UBSAN/leak group passes 19 tests for the scheduler/context changes;
the final subsequent edits refactor CUDA head predicates and extend GPU tests.
These counts are separate source-generation evidence, not a release gate total.

## Limits

This estimator does not cover split weights, arbitrary batched or remote
layouts, every quantization/precision path, concurrent temporary lifetimes or
all vendor allocations. Pool caching/fragmentation differs by allocator.
Existing runtime admission still handles unplanned growth; complete session
rollback after an earlier graph node writes state is not established.

There is no complete aggregate memory budget, sustained concurrency or filled
64K-context qualification. The original 33-token numerical gate remains failed.
Whole-model evidence here is limited to the selected Ampere pair; all-three-GPU
component coverage does not establish Turing whole-model coverage. No speed
claim or production promotion follows from this record.
