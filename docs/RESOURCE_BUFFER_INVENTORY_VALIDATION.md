# Tracked resource buffer inventory — 2026-10-11

`llama_resource_get_memory_pools` reports tracked native storage by category
and scheduler backend. Host storage uses backend -1; storage that cannot be
assigned to a scheduler backend uses -2. This additive API leaves the existing
resource snapshot structs unchanged.

Tracked categories include model weights, mapped weight spans, KV/indexer and
recurrent buffers, checkpoint banks, scheduler graph arenas, output buffers,
prepared/rollback cache replacements, selected host metadata and journals,
and allocated expert cache/prefill storage. Explicit buffers and metadata
contexts are deduplicated. A count query allocates no model/backend storage;
an undersized result array remains untouched and receives the required count.
Call from the context owner thread. The function does not synchronize GPU work.

## Verification

Evidence directory:
`/home/xero/Desktop/test/results/ese-resource-buffer-inventory-20261011`.

- CPU hybrid fixture passes in CUDA Release and CPU ASAN/UBSAN with leak
  checking: exact KV/checkpoint storage, buffer deduplication, insufficient
  caller capacity, and replacement accounting before publish, after publish,
  and after rollback/free.
- Selected-device tiny Llama model fixture checks inventory before and after
  actual decode in five configurations: first GPU, first/last GPU together,
  allocated CPU weights, last GPU, and mapped CPU weights. Counts agree with
  live weight/KV/output buffers and scheduler arena capacities; two consecutive
  inventory calls preserve the same capacities and locations.
- Native execution with all three physical GPUs visible passes, including
  noncontiguous CUDA0/CUDA2 selection.
- The final selected Ampere-pair GPU memcheck passes all five configurations
  with API error reporting enabled, zero errors and zero leaked bytes. Its
  two-GPU mode selects both visible GPUs.
- The first all-visible GPU sanitizer run failed with 12 API errors in
  `cuLibraryLoadData` through cuBLAS heuristics during decode, while inventory
  assertions passed and leaks were zero. The failed log remains retained;
  all-visible/Turing GPU sanitization is not qualified by this stage. Earlier
  selected-pair results are retained separately from the final two-GPU fixture.
- Initial fixture compile errors (helper declaration order and batch API
  arguments) were corrected before execution; failed build logs are retained.

## Limits

Mapped weight bytes describe virtual file-backed spans, not resident or
allocated RAM. Graph bytes describe scheduler arenas, not total CUDA memory.
The original checkpoint excludes OS page cache, backend-private workspaces, CUDA driver
and provider heaps, full PLE/cache bookkeeping and some host container storage.
It therefore cannot yet authorize allocations against a complete memory budget.
Model weights can be shared by several contexts; inventories must not be summed
across contexts as if each owns a separate model allocation.

C1 and the wider memory/long-context gates remain open. Separate validation
checkpoints cover expanded PLE/transaction bookkeeping, graph admission/backoff
and CUDA pool backing; none establishes a complete aggregate memory budget.
No performance, production promotion or complete-memory claim is made here.

## Server preparation uses tracked accelerator floors — 2026-10-13

The server resources response now includes the deduplicated native pool
capacities, category IDs and explicit resource-plan device IDs. Scheduler
backend IDs retain their original inventory meaning. An additive native
translation API maps host/CPU storage to -1 and CUDA storage to its physical
device ID; unsupported/unknown locations leave the output untouched.
Mapped weight spans remain labelled virtual spans rather than resident RAM.

Runtime resize preparation raises both current and target accelerator weight
and fixed-allocation forecasts to their live tracked floors. Larger configured
forecasts are preserved. Fixed allocations include separate graph and CUDA-pool
backing, output, checkpoint and handoff buffers; KV, expert caches and transient
module owners retain their existing distinct accounting. Read-only reconciliation
rejects overflow, stale planned bytes and inconsistent device topology without
changing either preparation plan. The serialized policy itself is unchanged.

The owner thread rechecks these fixed-allocation limits before preparing a
replacement. Growth beyond the validated limits rejects with a fresh-snapshot
retry response. This closes the interval between HTTP inspection and dispatch
for tracked accelerator weights/fixed allocations; complete concurrent ownership
and external process/vendor allocations are not covered.

Evidence: `/home/xero/Desktop/test/results/ese-live-device-memory-20261013/qualification-final.json`.

- Four focused Release and three strict CPU ASan/UBSan/leak CTests pass.
- A controlled planner case originally admits a 96 MiB preparation peak;
  accounting for live weights and workspace raises it to 144 MiB and correctly
  rejects against the 120 MiB usable budget. More capacity admits the peak.
  Repeated readings preserve totals without adding the same floor twice.
- Five HTTP controls include saved old/current runtimes on CUDA2 and CUDA0/2,
  plus a CPU sanitizer server. With a zero graph forecast, the CUDA2 fixture
  exposes 127,872 B of omitted weights and 18,560 B of fixed allocations.
  Current preparation includes all 146,432 B. Split-device allocations are
  charged to their respective physical IDs. Shrink/regrowth and generated
  content/all 32 token probabilities remain exact.
- The permanent server regression supplies stale zero limits against nonzero
  GPU weights. Rejection preserves saved sequence bytes, tracked capacities
  and replay logits. The CPU build checks host translation and invalid limits.
  Actual competing request growth is not forced by this fixture.
- Two selected Ampere-pair CUDA memory/API/leak checks pass with zero errors
  and zero leaked bytes, covering device selection and server recovery/recheck.
  Native CUDA2 and noncontiguous selection controls run with all three GPUs
  visible. Turing execution is outside this stage's sanitizer scope.

The CPU HTTP audit separately measures 29,473,656 B of graph/output/host metadata
that a zero graph forecast omits. Host accounting is not repaired by this
accelerator-floor change. It remains a demonstrated integration gap, alongside
complete original-model/concurrent/64K admission and provider/OS allocations.
Production Qwen gates remain closed. Initial fixture/build failures are retained
and excluded; preliminary HTTP controls are superseded by final owner checks.

## Tracked host floors join RAM preparation — 2026-10-13

Runtime preparation now includes live host graph/output, checkpoint/handoff,
metadata/journal and PLE staging capacity above its existing graph forecast.
Allocated host weights and PLE cache backing contribute only when they exceed
their separate RAM reservations. Virtual mapped weight spans do not enter the
allocated-weight input. Expert/auxiliary/IO reservations remain distinct.
This applies to CPU plans and to accelerator plans with host buffers.

The owner thread rechecks host weights, fixed buffers and PLE cache limits
before preparing replacements. A stale limit rejects without changing sequence
state or allocations. The original six-argument preparation overload retains
its source-level behavior; a snapshot-aware overload and extended internal
C++ report provide the host floors. The native C API and kernels are unchanged.

Evidence: `/home/xero/Desktop/test/results/ese-live-host-memory-20261013/qualification-final.json`.

- Four Release and three strict CPU ASan/UBSan/leak CTests pass. Host tests
  cover reservation reuse, repeated queries, overflow/failure atomicity,
  immutable reservation checks, exact byte boundaries and unbounded RAM.
- Fourteen HTTP controls cover old/current mapped and allocated CPU weights,
  CUDA2 and CUDA0/2, plus a CPU sanitizer server. The CPU fixture adds exactly
  29,473,656 B of omitted fixed capacity. Its current reservation is 29,802,232 B
  and replacement estimate 29,900,536 B. A 29,851,384 B budget permits the
  current context but rejects the replacement. Old requests incorrectly admit
  it; current dry-run and committed requests reject with unchanged plan, KV,
  devices, expert RAM and tracked inventory. Generous budgets admit shrink
  and regrowth. Content/all 32 probabilities remain exact throughout.
- Native CPU and CUDA server regressions supply stale host limits against
  actual allocations and preserve saved sequence bytes, inventory and replay.
  Two selected Ampere-pair memory/API/leak checks pass with zero errors/leaks.
- A private read-only observer of the existing hybrid retier regression
  measures another 10,176 B of retained preparation metadata in both builds.
  Its native migration log reports about 1 MiB of temporary staging. The
  original test source is unchanged by the observer, and original rollback/
  buffer assertions remain active.

These are live tracked floors rather than complete transaction-memory admission.
The separate preparation metadata and temporary migration storage are not yet
reserved by this change. Native target placement/allocation overhead, startup
admission, OS/vendor/provider heaps, growing/concurrent contexts and filled
original-model 64K operation remain unqualified. Normal Qwen gates remain closed.
