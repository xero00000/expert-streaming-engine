# Native cache forecasts

The native resource controller queries cache geometry at each candidate context
capacity. It previously multiplied an estimate at 4,096 tokens by the requested
capacity. That scaling also multiplied fixed recurrent state: it understated
small caches and overstated large ones. Runtime resize forecasts now use the
same metadata query, preserving fixed and stepped allocations.

An immutable metadata snapshot keeps queries valid after the planning probe is
freed. It retains no weights or backend buffers. Native plans identify this
sizing mode in JSON; the HTTP rebalance handler binds a fresh query from the
loaded model. A native plan without its query rejects a resize. Legacy/manual
plans retain their supplied per-token estimates.

Runtime host admission also combines the RAM weight/cache reservation with host
KV, graph and transient forecasts. This prevents a resize from spending the
same declared RAM capacity twice.

Dense-weight forecasts also exclude the skipped Qwen PLE storage extent while
retaining its separate bounded RAM-cache reservation; see [streamed PLE RAM
forecast checks](PLE_RESIDENT_FORECAST_VALIDATION.md).

Replacement admission also includes the separate RAM reservation in the
temporary old-plus-new cache peak; see [combined RAM preparation
checks](HOST_CACHE_PREPARATION_VALIDATION.md). These checks can reject a resize
that fits its final budget but lacks room to retain the original cache.

Evidence: `/home/xero/Desktop/test/results/ese-native-cache-forecast-20261012`.

- Four paired old-algorithm controls fail in both Release and strict CPU
  ASan/UBSan/leak builds: long-context overestimation, short-context
  underestimation, lost fixed state on shrink, and stepped storage on shrink.
  A separate intermediate-version control reproduces the combined host-budget
  admission defect. Current planner tests pass all five cases, unavailable
  queries, overflow, slot floors and conservative device-share rounding.
- Eighteen focused CTests pass in each build, including normal loader rejection,
  cache/state lifecycle, resource planning, graph/shared budgets and server
  recovery. Snapshot lifetime checks remain exact after the source model is
  changed and freed.
- Cache-sizing checks pass 540 synthetic and 144 original-metadata cases per
  build. The original-metadata result tables match byte-for-byte. For the
  original model with one slot and F16 K/V/indexer storage, native payloads are
  231,284,736 bytes at 4K and 1,929,977,856 bytes at 64K. Metadata-only planning,
  shrink and regrowth match those sizes; insufficient short-context and
  simultaneous old/replacement-cache budgets are rejected.
- Original-model CPU controller startup at 64K capacity, two slots and F16
  K/V/indexer storage forecasts 2,048,016,384 bytes. That exactly matches the
  initialized live cache tensor payload. External shard identities are
  unchanged. This initializes an empty cache; it does not fill 64K tokens.
- Unmodified Release and CPU sanitizer servers complete native-plan JSON
  queries, dry-run shrink, committed shrink from 512 to 256, and regrowth to
  512. A deterministic three-layer Llama fixture, two slots and CPU Flash
  Attention are used. Generated content and all returned token probabilities
  remain exact. CPU sanitizer shutdown reports no sanitizer or leak error.
- The original-model CPU 33-token control preserves all 8,194,560 logits and
  expert routes byte-for-byte versus the previous runtime. Against unchanged
  full-prefix and cached references, 64 and 59 logits respectively still exceed
  the fixed bound. All 1,584 ordered expert lists and 33 top-token choices match
  each reference. The numerical gate remains closed.

Initial build integration errors and HTTP attempts with a missing fixture path,
an insufficient harness budget, or an unsupported non-Flash resize layout are
retained separately and excluded from successful qualification.

These are cache tensor-payload forecasts. Per-device shares remain proportional
estimates, with conservative rounding on resize; they do not establish actual
layer ownership, split replicas or mixed per-layer formats. Allocator overhead
and other memory owners require separate accounting. Complete physical memory
admission, filled 64K execution, concurrent Qwen serving and throughput remain
unqualified. Model arithmetic, weights, goldens and tolerances are unchanged.
