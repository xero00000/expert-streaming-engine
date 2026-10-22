# Combined RAM admission during cache replacement

Runtime cache replacement now checks the combined RAM preparation peak before
allocating a replacement. That peak includes the existing RAM weight/cache
reservation, host workspace and live cache, and complete replacement pools.
Host dense weights are counted once. The check preserves the host reserve and
uses the smaller of the declared RAM limit and usable host-device capacity.

The previous device-only peak omitted the separate RAM-cache reservation. For
example, 128 MiB of host weights, a 512 MiB RAM cache, 64 MiB of workspace and
256 MiB of live KV fit a 1 GiB budget. Shrinking KV to 192 MiB also fits, but
preparing it while retaining the original KV needs 1,152 MiB. That transaction
now rejects before publication. Host transient-owner replacements also include
the incoming owner while the old owner remains available for rollback.

Preparation reports expose combined RAM current, target, peak, usable capacity
and headroom. RAM capacity and reservations remain immutable during these
transactions. An unspecified RAM budget retains existing per-device checks.
Admission uses declared reservations even when current cache occupancy is low;
smaller configured RAM caches can leave room for replacements.

Evidence: `/home/xero/Desktop/test/results/ese-host-cache-peak-20261012`.

- Four old-planner negative controls reproduce the omitted cache and transient
  peaks in Release and strict CPU ASan/UBSan/leak builds.
- Planner regressions cover 18 cache/reserve/one-byte boundary combinations,
  exact combined sizes without duplicate weights, overflow, immutable budgets,
  and GPU cache replacements that leave host reservations unchanged. Rejected
  operations preserve the caller's report and current plan. Host transient
  controls qualify accounting only.
- Eighteen focused CTests pass in each build, including the ordinary Qwen
  loader rejection, resource planning, cache/state lifecycle, graph/shared
  budgets and server recovery.
- Paired actual HTTP controls use a deterministic three-layer Llama fixture,
  two slots, F16 cache storage, CPU Flash Attention and a 1 GiB RAM budget.
  Both old servers return success for dry-run and committed 512-to-256 shrink
  although the independently calculated peak is 1,073,840,128 bytes against
  1,073,741,824 bytes available. Both current servers reject these requests with
  HTTP 400; plan, KV, device inventory and runtime resource state are unchanged.
  Generated content and all 32 returned token probabilities remain exact.
- With prompt-cache reservation disabled explicitly, both current servers
  complete dry-run, shrink and regrowth. Their reported shrink peak equals the
  independent 268,862,336-byte calculation. Content and probabilities remain
  exact. CPU sanitizer HTTP controls shut down without sanitizer or leak errors.

An initial Release HTTP attempt failed its frozen-library provenance assertion;
it is retained and excluded. The successful old control explicitly preloads its
frozen libraries. Initial build attempts named an unavailable target and are
also retained separately.

This establishes combined admission for the represented reservations and
replacement pools. Process RSS, untracked owners, allocator overhead, concurrent
Qwen serving and complete physical memory limits remain unqualified. Execution
support for host transient modules is unchanged. Model arithmetic, weights,
reference outputs and numerical bounds are unchanged; the original numerical
gate remains incomplete and ordinary Qwen loading remains gated.
