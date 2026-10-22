# Original Qwen prefill execution — 2026-10-12

This record retains the arena-only failure and explicit-128 control from its
source generation. The subsequent [workspace-aware admission check](
CUDA_WORKSPACE_PREFILL_VALIDATION.md) resolves this specific automatic pressure
case and preserves the original failed evidence.

The freshly rebuilt gated diagnostic executes actual prompts with the current
shared graph/workspace admission code. It records input IDs, logits, ordered
expert routes, graph/pool/ledger snapshots and 50 ms target external memory
samples. The normal loader gate remains closed. Both original model shards
remain read-only; size/inode/device/modification-time identities match across
each run. This stage does not repeat full payload hashing.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-prefill-execution-20261012`.

## Nine-token numerical check

F32 KV, Flash Attention off, host KV, 4096-token context capacity and a requested
1024-token chunk pass nine-token batched prefill. All 2,234,880 logits satisfy
the unchanged `2e-5 * (1 + abs(reference))` bound against the pinned reference
and preserved sequential output. All 432 layer/token ordered MoE route lists
match. Fresh sequential execution under the same profile agrees with the batch;
its maximum absolute logit difference is 1.239776611328125e-5.

The selected 3060 Ti/3080 pair also passes an unsuppressed all-API/leak memcheck
of that nine-token prefill: terminal exit 0, zero errors and zero leaked bytes.
Instrumented logits and route artifacts are byte-exact with the native batch.
This checks short full-model execution on the Ampere pair, not Turing or every
model path. The current normal architecture rejection CTest also passes.

This diagnostic captures logits and routes; it does not establish all hidden
boundaries, saved-state equivalence, sampling or long-context behavior. The
original 33-token gate remains failed.

## Memory-pressure execution gap

The Q8/Flash Attention resource profile uses host KV, an empty initial
131072-token capacity, CPU experts, 64 MiB expert RAM cache and 4 MiB staging.
It requests a 1024 logical batch with a deliberately raised 4500 MiB reserve.

| Requested chunk | Initialized chunk | Attempted prompt | Execution result |
| --- | ---: | ---: | --- |
| 1024 | 256 | 128 tokens | Pool admission rejects; decode returns -2 |
| 128 | 128 | 128 tokens | Completes; final position 127 |

The first run fits graph arenas after rejecting 1024 and 512 reservations, but
has only 46,012,672 bytes of shared headroom on CUDA0 at its initialization
snapshot. Execution rejects a pool allocation during CUDA capture before any
expert loads. The probe catches the failure and tears down; no completed
prefill is qualified by this attempt. Initialization backoff alone therefore
does not resolve execution under this pressure profile.

Explicit chunk 128 completes at the same reserve. Each CUDA pool ends at
125,829,120 bytes (120 MiB), active bytes zero. Graph arenas end at 910,550,272
and 908,715,264 bytes; combined graph/pool reservations are 1,036,379,392 and
1,034,544,384 bytes with no shared-ledger rejection. Sampled owned process VRAM
peaks are 3,202,351,104 and 3,498,049,536 bytes. Native sampled RSS peaks at
29,485,428,736 bytes. It fills 128 tokens, not 64K/128K context.

Header inspection shows a 100 MiB F32 materialization for quantized recurrent
QKV matrices, consistent with the missing execution headroom. That calculation
is an inference from tensor geometry and the cuBLAS path, not a directly logged
request identifying the rejected allocation. The observed maximum pool backing
is larger than that single matrix.

The failed runner's initial receipt incorrectly labels the attempted 128 tokens
as filled context and records a hold that did not run. Its retained
`interpretation.json` corrects that scope without overwriting raw evidence.
Subsequent runner receipts report no qualified filled count after failure.

## Cache-policy comparisons and limits

Initial Q8/Flash Attention output differs from the preserved F32/non-Flash
reference by up to about 2.14. Both batch and fresh sequential Q8 runs show that
cross-profile difference; it is not an isolated code-regression comparison.
Q8 batch versus sequential also differs within that profile. The passing F32
check does not qualify Q8 quality or exact batch equivalence. No cache-policy
differences are hidden and no tolerance/golden is changed.

External peaks are sampled and may miss transient allocations. Whole-device
used memory includes other processes; reported process peaks are separate.
Sanitizer RSS samples cover its parent rather than model child and are excluded
from model-host-memory conclusions. Decode timings include diagnostic route
capture and artifact writes, use different output counts/profiles, and are not
qualified throughput comparisons.

Automatic graph/workspace-aware chunk selection remains incomplete, as shown
by the failing 1024-request execution. Full aggregate admission, ordinary
partial-graph session rollback, sustained concurrency, filled long contexts,
the original numerical gate and matched Strata benchmarking remain unqualified.
No release promotion or speed claim is made.
