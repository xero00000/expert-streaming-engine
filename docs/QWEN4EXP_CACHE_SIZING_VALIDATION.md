# Qwen4Exp cache payload sizing

The native per-layer cache estimate now includes the raw indexer-key tensor
retained by each Qwen4Exp full-attention layer. Ordinary K/V tensors and fixed
recurrent/PLE state were already included. The missing indexer contribution
affected both native layer-placement estimates and `llama_model_kv_bytes`.

For the original 48-layer artifact at 65,536-token capacity, its twelve
128-wide indexer tensors occupy 201,326,592 bytes with F16 storage, 402,653,184
bytes with F32, or 106,954,752 bytes with Q8_0. These are payload sizes, excluding
allocator alignment and other context storage.

Evidence: `/home/xero/Desktop/test/results/ese-qwen-cache-sizing-20261012`.

- Old synthetic F32 and F16 controls reproduce underestimated attention-layer
  storage. The original metadata control reproduces the omission at layer 3.
  A dense Llama control is unchanged. The ordinary unpatched CPU sanitizer
  executable reproduces the same estimate mismatch without a sanitizer error.
- `test-qwen4exp-cache-sizing` passes 540 synthetic dense/hybrid cases in
  Release and strict CPU ASan/UBSan/leak builds: F32/F16/Q8_0 K/V and indexer
  storage, five context capacities up to 65,536, three slot counts and Flash
  Attention settings. Per-layer and public total predictions exactly equal
  independently constructed cache-tensor metadata sizes.
- Both builds pass 144 additional cases using the original model metadata.
  Their result tables match byte-for-byte. These checks allocate no weight
  buffers and execute no model context.
- Twelve focused CTests pass in each build, including loader admission,
  planning, graph/shared budgets, prefill backoff and cache/state regressions.
- Actual original-model CPU initialization checks two slots with F32 K/V and
  F16 indexer storage. At 4K capacity, the old estimate is 437,403,648 bytes
  versus 449,986,560 bytes of live cache tensors: a missing 12,582,912 bytes.
  The corrected estimate equals the live tensor payload exactly. At 64K
  capacity, the corrected estimate and live payload both equal 3,658,629,120
  bytes, including 201,326,592 bytes of indexer keys. External shard identities
  are unchanged in all three controls.

The actual-context controls initialize empty caches. They do not fill or decode
64K tokens, run GPU contexts, verify logits or qualify model-memory sanitizers.
The CUDA-enabled CPU control uses its existing host-buffer fallback with GPUs
hidden. Quantized-indexer checks establish size calculations; execution and
quality of those formats remain unqualified.

This is cache tensor-payload accounting, not a complete physical memory budget.
It excludes graph/output/checkpoint/provider storage, allocator overhead,
split replicas, mixed per-layer layouts and other owners. The resource
controller now queries candidate-context metadata instead of scaling its old
4K sample; see [native cache forecasts](NATIVE_CACHE_FORECAST_VALIDATION.md).
Aggregate admission, full-model parity and filled 64K qualification
remain incomplete. Model arithmetic, weights, reference outputs and tolerance
are unchanged; normal Qwen loading remains gated. No throughput claim follows
from these checks.
