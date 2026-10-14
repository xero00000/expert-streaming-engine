# Graph arena growth and budget checks — 2026-10-11

Graph reservation now prepares layouts and replacement arenas before committing
either. Failed device allocation or budget admission releases candidates while
preserving the existing arenas and graph layout. Scheduler split allocation now
propagates a failed reservation rather than trying to use its uncommitted plan.

The additive graph allocator query reports current capacity, transient peak,
last measured requirement, allocation failures and budget rejections. Budgets
default to unlimited and include the old arena plus its replacement. Identical
buffer types share one arena and the strictest quota among their aliases. The
stats query returns the canonical arena for each alias; those values must not
be summed. The existing buffer-size query still returns zero for duplicate
aliases, preserving inventory deduplication.

Evidence: `/home/xero/Desktop/test/results/ese-graph-arena-budget-20261011`.

- Final CPU ASAN/UBSAN/leak regressions: 17 distinct tests pass (15 scoped tests
  plus expert-cache/resource-planner). Controlled allocators
  execute real CPU graphs, fail the second arena allocation after preparing the
  first, and verify cleanup, old pointers, old bytes and reusable committed
  layouts. Quota rejection precedes any arena allocation. Growth retries,
  old-plus-new peaks, shared aliases and complete teardown pass.
- A same-ABI reconstruction of the old reservation implementation fails the
  ownership check with exit 1 after its second allocation fails. The final
  positive test still passes. Leak checking is disabled only for the negative
  control because its deliberate early failure leaves raw fixture handles;
  positive regression runs enable leak checking.
- Final Release regressions: 24 distinct tests pass (22 scoped tests plus
  expert-cache/resource-planner). The earlier `components-release` attempt ran
  before the executable existed and is excluded. The interrupted build stopped
  without a completion receipt; the resumed build completed with exit 0 before
  running tests. A subsequent freshness build reports no work to do.
- `test-graph-arena-budget --cuda` passes on the 3060 Ti, 2080 SUPER and 3080,
  with independent/shared arenas on each: quota rejection, preserved old graph
  replay, successful growth and exact results. Each grown arena reports live
  1024 bytes and transient peak 1280 bytes. Duplicate aliases remain excluded
  from the existing size inventory. New-generation handoff tests pass for all
  three physical GPU pairs, including growth, quota/retry, graph reuse and views.
- Both GPU suites pass unsuppressed all-API/leak compute-sanitizer: terminal
  exit 0, zero reported errors and zero leaked bytes. These are allocator and
  scheduler component checks, not whole-model or throughput qualification.

Host metadata allocation failure still uses GGML assertions; this is not a
complete host-OOM recovery mechanism. Fresh preparation metadata can add host
allocation cost during reservation; performance qualification remains open.
This stage does not close aggregate context budgeting, CUDA private-pool
accounting, safe prefill chunk selection/backoff, whole-model or long-context
gates. The original 33-token numerical gate remains failed. No speed or release
promotion claim is made.
