# Handoff buffer allocation and inventory — 2026-10-11

The scheduler's private split-mode handoff arenas now have explicit native
inventory (additive `HANDOFF` category13) and per-backend live/peak, quota,
allocation-failure and budget-rejection queries. The quota cannot be lowered
below live use and includes both old and replacement requests during growth.
Defaults preserve the unlimited quota and existing extra-allocation policy.

Previously, growth freed the current arena before trying its replacement and
then accessed the returned buffer without a null check. Preparation now allocates
all backend replacements before changing any arena or tensor binding. On failure
it releases only candidates and returns allocation failure. Successful retirement
waits for old work before freeing old arenas; reuse adds no extra global barrier.
Checked byte arithmetic rejects overflow and an individual split larger than the
configured extra-allocation limit. The native inventory reports real buffer
capacities, independently of graph allocator arenas.

Evidence: `/home/xero/Desktop/test/results/ese-handoff-allocation-20261011`.

- A controlled host-backed device adapter runs actual CPU graphs with distinct
  non-host buffer types. First-allocation and growth failures propagate; failure
  on the second device releases the first candidate and preserves both old arenas.
  Budget rejection precedes allocation; smaller retries, later growth, graph
  reuse, offset views and native inventory remain exact. Teardown releases all.
- The reconstructed old preparation, with the same current scheduler ABI, reaches
  the real null-buffer member access. Strict UBSAN exits1. The initial symbolizer
  child inherited preload and failed, causing SIGPIPE(-13); that exit is excluded.
  The corrected control disables symbolization and verifies the fault directly.
- Initial compile scope error was corrected. The first fixture incorrectly used
  graph_compute's automatic reset after setting device assignments; explicit
  alloc_graph fixes the fixture and matches llama's execution contract. Failed
  attempts remain recorded; no production arithmetic or tolerance changed.
- Final21 Release and14 CPU ASAN/UBSAN/leak regressions pass, including shared
  Engram, resource/history/provider quotas and lifecycle/cache checks.
- Real transfers pass for all3 physical GPU pairs (3060Ti/2080SUPER/3080): warm,
  growth, quota rejection, exact smaller/grown retries, reused graphs and offset
  views. Each arena reports live256 then512bytes and growth peak768bytes.
- All-API/leak compute-sanitizer repeats those pairs: exit0, errors0, leaks0.
  These are scheduler/component checks, not whole-model or throughput evidence.

C1/C3 remain incomplete. The subsequent
[`GRAPH_ARENA_BUDGET_VALIDATION.md`](GRAPH_ARENA_BUDGET_VALIDATION.md) stage
qualifies transactional graph arena/layout growth independently and repeats
these handoff GPU cases. Safe prefill backoff remains unqualified. CUDA pools
distinguish physically committed/cached
bytes from a32GiB VMM address reservation and remain outside this inventory.
Global context budgets, driver/allocator overhead, whole-model/long-context/
concurrency stress and performance qualification remain open. The original33-token
numerical gate remains failed; normal Qwen loading stays closed.
No speed, promotion or publication claim is made.
