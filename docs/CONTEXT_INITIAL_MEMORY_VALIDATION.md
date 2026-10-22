# Native context host memory admission

Contexts created with an explicit planner RAM budget now check tracked host
storage before allocating initial KV/output buffers. The check combines native
KV placement, cache metadata, graph metadata, initial outputs, output IDs and
observed model/context owners with the existing weight, PLE and other planner
reservations. Mapped weight spans remain excluded from allocated RAM. Larger
observed values replace their reservations without a second charge.

The additive `llama_init_from_model_with_host_limits` API accepts a separate
limits structure; existing public C structures keep their layouts. The ordinary
initializer retains its unrestricted behavior. Context/model C++ layouts are
unchanged. The common initializer passes limits when the plan has an explicit
RAM capacity.

CPU graph backing receives a shared arena quota for the remaining capacity.
Existing microbatch backoff can fit a smaller graph, and retained plus incoming
graph backing remains charged during replacement. Output growth checks old and
incoming storage before allocation and audits actual prepared buffer size before
publication. A rejected replacement retains the previous output buffer.

## Evidence

Evidence: `/home/xero/Desktop/test/results/ese-context-initial-memory-20261013`.

- Six focused Release and five strict CPU ASan/UBSan/leak CTests pass. Native
  tests exercise steady capacity and one byte on either side, graph backoff,
  excessive/overflowed reservations, failed output growth with exact retained
  allocation/KV/logit state, and fitting output growth.
- Twelve final CPU startup/API controls cover old/current/sanitizer builds and
  mapped/allocated weights. At a 29,705,440 B budget, the saved old allocated
  server becomes ready with 29,802,232 B tracked host storage. The old mapped
  server has 29,670,264 B allocated storage, but its conservative weight
  reservation raises reported RAM to 29,802,232 B. Both subsequently reject a
  no-op rebalance for exceeding combined RAM. Current Release/sanitizer starts
  reject inside native context preflight, before KV/output buffer allocation.
- At a 1 GiB budget, current CPU allocated storage is 29,861,068 B. The increase
  includes newly inventoried model/context owners; it is not all a new physical
  allocation. Mapped storage is 29,729,108 B with 29,861,076 B reported RAM after
  conservative weight reconciliation. All 32 reported probabilities and content
  remain exact through 512-to-256-to-512 resizing. Release output matches the
  saved old binary; sanitizer controls use their own teacher.
- Six old/current GPU HTTP controls cover CUDA2, split CUDA0/2 and mixed
  CPU/CUDA2 placement. Exact target KV receipts and all 32 probabilities survive
  shrink/regrowth. Reported RAM reflects the additional tracked host owners.
- Native metadata checks pass 540 synthetic and 144 canonical original-metadata
  cases per build. Original cases allocate neither weights nor an inference
  context. Two Ampere-pair CUDA memchecks report zero errors and leaked bytes.

The evidence retains preliminary failures from incorrect test expectations and
runner arguments; only the named final runs qualify this change. Source
before/after, the owned patch, frozen binaries and a verified artifact manifest
are retained separately from the heavily modified worktree.

## Scope

This admits the tracked storage of one model/context and its specified planner
reservations. Context construction, model loading, admission-query temporary
storage, untracked containers, allocator overhead, backend/vendor/OS memory,
other contexts and concurrent application owners are not comprehensively
admitted. KV replacement continues to use its separate owner receipt checks;
this addition does not establish a complete physical RAM cap.

Original-model numerical parity, complete 64K inference, concurrent serving and
release readiness remain unqualified. Normal Qwen loading stays gated. Model
arithmetic, weights, references and tolerances are unchanged.
