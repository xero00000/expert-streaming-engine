# Native KV replacement preparation memory

Supported live KV replacements now forecast the actual backend buffer sizes,
tracked alternate metadata and retained migration scratch before allocation.
The server uses this receipt for RAM/device preparation admission, checks it
again on the inference owner thread, and compares the prepared transaction
with it before publication. Current KV placement comes from the live inventory.
Host KV remains charged to RAM when the serialized plan has only GPU rows.

The additive `llama_kv_cache_resize_memory` and
`llama_kv_cache_retier_memory` APIs build bounded tensor metadata without
allocating replacement KV buffers or migration payloads. The GGML sizing helper
sums backend allocation sizes and alignment, including contexts that require
multiple buffers. `llama_kv_cache_transaction_memory` reports tracked prepared
capacities. Queries follow the context's owner-thread contract. Unsupported
layouts, invalid formats and malformed inputs fail without publishing partial
outputs; an insufficient row capacity reports the required count only.
Existing public C struct layouts are unchanged. The internal common C++
preparation report has additional fields; source users must rebuild dependents.

Migration reserves its four scratch arrays once. The forecast includes the
maximum retained byte arrays and float transcode arrays, including the bounded
1 MiB recurrent-state copy chunk. Reports expose
`prepared_host_metadata_bytes` and `prepared_host_staging_bytes`.

## Verification

Evidence: `/home/xero/Desktop/test/results/ese-kv-preparation-forecast-20261013`.

- Five focused Release CTests and four strict CPU ASan/UBSan/leak CTests pass.
  The sizing helper allocates no buffers, handles alignment/splitting, and
  rejects overflow and oversized tensors atomically. Planner checks cover
  exact byte boundaries, duplicate/unknown devices, overflow and host reuse.
- The hybrid retier fixture's forecast matches 4,222,976 B of backend storage,
  10,176 B of tracked prepared metadata and 1,048,960 B of retained scratch.
  Unsupported K/V formats preserve output sentinels and live state.
- Native selected-device tests compare the forecast with actual prepared
  buffers in all five CPU/CUDA selection modes. Server tests reject a stale
  current KV inventory or a changed sizing receipt before allocation, with
  unchanged sequence bytes, allocation inventory and replay logits.
- Twenty-two HTTP controls cover saved old/current CPU binaries, strict CPU
  sanitizer execution, CUDA2, CUDA0/2 and mixed CPU/GPU placement. CPU controls
  include mapped and allocated weights. The old CPU replacement estimate is
  29,900,536 B; the new estimate is 29,930,208 B, including 29,640 B of metadata
  and 32 B of scratch. With a 29,915,372 B budget, the old binary admits dry-run
  and commit; current Release and sanitizer servers reject both while the
  29,802,232 B live session fits. Rejections preserve resource state.
  Generous shrink/regrowth preserves content and all 32 probabilities exactly.
  Sanitizer and Release runs use their own numerical teachers; old/current
  Release controls are also exact across binaries.
- GPU HTTP target buffers match native placement. For the split case, CUDA0
  prepares 32,768 B and CUDA2 prepares 65,536 B. The mixed case prepares
  65,536 B on the host and 32,768 B on CUDA2. The new RAM peak matches live
  RAM plus the actual incoming host KV, prepared metadata and scratch.
- A private cache-only fixture loads the canonical original model metadata,
  creates no weight buffers and checks targets 256, 4,096 and 65,536. Strict
  CPU sanitizer, CUDA2 and mixed CPU/CUDA0/CUDA2 controls match native candidate
  sizes, tracked metadata and scratch exactly. Seeded recurrent and occupied
  attention/indexer bytes survive publication and rollback. At 64K:

  | Placement | KV bytes | Tracked prepared metadata | Scratch |
  | --- | ---: | ---: | ---: |
  | CPU or CUDA2 | 1,929,977,856 | 4,881,024 | 1,049,088 |
  | CPU/CUDA0/CUDA2 | 1,929,977,856 | 4,881,064 | 1,049,088 |

  The mixed 64K buffers are 643,203,072 B on the host, 52,297,728 B on CUDA0
  and 1,234,477,056 B on CUDA2. Both external shards retain their pinned hashes.
- Three Ampere-pair CUDA memchecks cover selected-device preparation, server
  revalidation and original-metadata mixed cache preparation through 64K.
  Each reports zero errors and zero leaked bytes. Visible-device remapping is
  recorded; the instrumented mixed run uses device IDs 0/1 for physical 0/2.

Preliminary harness/build failures are retained and excluded: the first HTTP
harness incorrectly required bit identity across sanitizer/Release builds;
two new test variables collided with existing names; private CUDA probe
compilation/linking needed the local backend signature and driver link path.
The initial three-job CUDA build was interrupted and resumed with more compiler
jobs. Final builds/checks and final cache-only probe controls pass.

## Limits

The receipt covers requested backend storage, tracked vector/cell/tensor
metadata and retained migration arrays. It excludes generic allocator overhead,
untracked container nodes, temporary sizing-query bookkeeping, vendor/context
memory, OS page residency and allocations caused by settling pending updates.
A prepared buffer that exceeds its receipt is rejected before publication;
this does not prove preallocation control of arbitrary backend over-allocation.

The original-metadata fixture does not load model weights, run inference,
exercise concurrent serving or qualify complete 64K model execution. Complete
startup/aggregate physical memory admission and the original numerical gate
remain incomplete. Normal Qwen loading remains gated. No model arithmetic,
weights, reference outputs or numerical bounds changed.
