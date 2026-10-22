# CUDA graph layout and temporary-workspace updates

October 11, 2026. Native CUDA components, focused model memory checks and both
final ordinary teacher controls pass for this frozen generation.

The preceding sequence-attention model passed numerical recovery but failed
unsuppressed all-API memcheck with twelve errors. The rejected calls were
cudaGraphExecUpdate and its driver/error-query counterparts. An independent
observer on the unchanged native model identifies five rejected updates of
640-byte memset nodes into temporary workspace. It returns the original API
status and does not suppress errors. Numerical continuations still match.

NVIDIA documents restrictions on changing memory mappings, allocation contexts
and node configuration during graph updates. See the
[CUDA graph API reference](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__GRAPH.html).
Ordinary GGML allocation IDs do not describe every temporary pool/VMM mapping.

The repaired native CUDA backend inspects a newly captured graph before
updating an existing executable. A recapture containing non-kernel/non-empty
nodes gets a fresh executable. This avoids deliberately trying an update
across unsupported memory/workspace changes. Unchanged captures replay;
kernel-only recaptures retain parameter updates. A newly instantiated graph
also no longer receives a redundant immediate update. The existing checked
fallback remains for other unexpected update failures. HIP/MUSA retain their
existing recapture policy; they are not qualified by these CUDA runs.

Another real cache defect appeared in inspection: the comparison recorded
output geometry but not source types, shapes or strides. The new regression
reuses the exact graph, allocations and addresses while changing a matrix
leaf's logical width from64 to32. The old executable incorrectly replays the
64-column dot product (8 instead of4). The identical final executable fails
with the frozen old GGML library and passes with the candidate.

Private side metadata now records output and source presence, type, dimensions,
strides and buffer kind. Structural changes recreate the executable rather
than assuming compatible topology; source address/operation-parameter changes
still use the existing comparison/update path. Side metadata is erased on
graph eviction/backend destruction, alongside allocation identities. Public
buffer and private backend-context layouts are unchanged.

Qualification evidence:

- 27 matrix shape/type/strided-input/replay checks and9 analytical
  pointer-update/replay checks pass on the3060Ti,2080SUPER and3080. The same
  graph/output addresses are reused and excluded matrix columns remain
  nonzero, so stale captures cannot pass accidentally.
- An observer records two successful kernel-only update calls, zero rejected
  calls, on **each GPU's native run**. This proves the update path remains
  active. Counter observations under sanitizer interposition are not used for
  that claim.
- All three component runs under unsuppressed all-API memcheck pass with
  zero errors. The old-library stale-width failure remains retained separately.
- The isolated3080 full-model target0/accepted4 case now passes all complete
  state/logit/MoE/logical-QSA checks and unsuppressed all-API memcheck with
  **zero errors**, versus twelve previously. RAM/staging caps and final drain
  pass. Both native reference/actual continuation streams remain byte-exact;
  native teacher values also match the preceding diagnostic generation.
- All30 bank cases pass again across ub1/2prefix8 and ub4prefix32 with exact
  states, logits, MoE routes and logical QSA block IDs. RAM/staging caps and
  final drain pass.
- Both final ordinary checkpoint-mode controls pass all21lifecycle checks.
  Each8,194,560-F32 teacher stream and its baseline logits/routes remain
  byte-identical to the preceding selected-backend baseline. RAM/staging caps
  and final drain pass; rejected-update diagnostics are absent.
- Eleven final focused normal CTests pass. The two previously reproduced
  DeepSeek fixture failures are outside this passing subset and remain open.

Evidence and immutable libraries/probes are under
`/home/xero/Desktop/test/results/ese-cuda-graph-update-20261011`.
`completed-qualification-audit.json` records verified completed runs;
`final-components` retains the expanded regression and update counters.
The model runner is terminal exit0; its seven per-run exit records and
`ordinary-teacher-audit.json` bind the completed results. No owned inference
job remains from this qualification queue.

This is correctness/resource evidence, not a throughput result. Recapture
instantiation cost, graph metadata peak memory, complete upstream parity,
long-context selection-limited QSA, failure injection and serving/concurrency
remain open. The earlier reproduced DeepSeek fixtures remain separate open
findings; no broad all-tests-green claim is made.
