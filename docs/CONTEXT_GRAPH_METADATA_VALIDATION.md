# Computation graph host metadata admission

Startup resource planning now reserves the native computation graph/tensor
metadata arena before context creation. `host_graph_metadata_bytes` is a
separate immutable RAM reservation in the serialized plan. Runtime preparation
reuses it when reconciling tracked host fixed allocations, keeping the arena
charged once while retaining larger observed limits.

The additive `llama_model_graph_metadata_bytes` API computes a conservative
arena bound without creating a context or allocating data buffers. It shares
native graph-node sizing, applies context/batch normalization, and includes the
iterative MLA bound before placement is fitted. Invalid inputs leave outputs
unchanged. Existing public C struct layouts are unchanged; internal common C++
plan/input structures require dependent source rebuilds. Older plan JSON can
omit the new field; its default reservation is zero.

## Evidence

Evidence: `/home/xero/Desktop/test/results/ese-context-host-admission-20261013`.

- Six focused Release and five strict CPU ASan/UBSan/leak CTests pass. Planner
  checks cover the exact RAM boundary and one byte on either side, overflow,
  immutable metadata reservations and reuse of the observed host fixed floor.
- The native query matches the allocated graph metadata capacity in all five
  CPU/CUDA context-selection modes. Cache-sizing tests check the query and
  untouched invalid-input outputs in 540 synthetic cases per build. Another
  144 canonical original-metadata cases per build pass without weights or
  context execution.
- Twelve startup/API controls cover saved old/current binaries and strict CPU
  sanitizers with both mapped and allocated CPU weights. With a 1,048,576 B RAM
  budget, the old server becomes ready with 29,670,264 B of tracked allocated
  host memory for mapped weights, or 29,802,232 B for allocated weights.
  Current Release and sanitizer servers reject startup during resource planning
  before entering context initialization.
- Generous controls reserve the exact 29,376,864 B native graph metadata arena.
  Planned RAM rises from 131,968 B to 29,508,832 B. Runtime host reconciliation
  then needs 96,792 B extra instead of 29,473,656 B. Reported current RAM stays
  29,802,232 B in old/current/sanitizer controls, proving single charging.
  Shrink/regrowth preserves content and all 32 probabilities exactly. Release
  output also matches the saved old binary; sanitizers use their own teacher.
- Six GPU HTTP controls cover CUDA2, CUDA0/2 and mixed CPU/GPU placement in
  saved old/current binaries. Native target KV placement and combined RAM
  preparation remain exact; resizing preserves output/probabilities.
- Two Ampere-pair CUDA memchecks cover selected-context sizing/allocation and
  owner-task revalidation. Both report zero errors and zero leaked bytes.

## Scope

This reserves the graph metadata arena. Other model/context metadata,
scheduler bookkeeping, output growth, graph tensor data, temporary allocation
peaks, allocator/vendor/OS overhead and complete startup/physical memory
admission remain separate. The remaining observed host floor is retained in
runtime preparation; it is not completely reserved at startup by this change.

Original-model numerical parity, concurrent serving and complete 64K inference
remain unqualified. Normal Qwen loading remains gated. Model arithmetic,
weights, reference outputs and tolerances are unchanged.
