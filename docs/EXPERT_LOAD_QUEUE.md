# Bounded GPU-route load queue: development checkpoint

September 4, 2026. Implemented on the dirty local desktop worktree based on `ee86cae61`, following `CACHE_CONCURRENCY_VALIDATION.md`. This is an opt-in implementation and correctness checkpoint; no inference-speed promotion has been made.

## Behavior

`--expert-load-workers N` selects 0 (the unchanged synchronous default) or 1–8 persistent loader threads for GPU expert-cache misses. A nonzero value requires the native RAM cache. The queue uses N fixed request slots and the existing RAM staging byte limit. Queued, running, and untaken completed requests all count against the request-byte limit. It allocates no second payload arena. Thread stacks and small queue metadata remain additional process overhead and must be included in later RSS measurements.

The GPU scheduler submits only experts whose requested component payload is not ready in its GPU cache. The model maps those IDs to immutable descriptors and submits as much as the queue admits. The scheduler consumes each completed lease independently, uploads it to the corresponding GPU slot, and starts more admitted reads as capacity becomes available. A later request can complete ahead of an earlier slow request. Already-ready RAM payloads bypass worker dispatch. A single missing expert uses the existing synchronous path.

Completed payloads retain the existing lease/event protection through H2D completion. The route keeps the original selected experts and weights; only load/upload order changes. Failed reads return errors, and required-lease execution remains fail-closed. A partially consumed batch drains queued results before another route borrows the model's pool. Pool destruction joins running reads before freeing their state. Blocking source reads are not cancelled by freeing their destination.

The model currently serializes ownership of its queue between GPU route batches/contexts. This avoids mixing completion IDs across contexts. CPU users can still call the underlying concurrent RAM cache. This is not a general multi-context scheduler redesign. Existing prefill staging and the synchronous per-source io_uring mutex remain separate follow-up work.

The internal callback interface is `ggml/src/ggml-expert-loader.h`. The new model parameter is populated by the CLI and recorded in its YAML parameter dump. Rebuild native callers after changing this parameter structure. No installed engine, Studio default, or production launcher was switched.

## Verification

- Controlled source tests prove independent completion, fixed request/byte admission, ready-hit bypass while both workers are blocked, duplicate-ID rejection, failure recovery, lease protection after taking results, and safe shutdown during a held read.
- The GPU cache test deliberately returns completions in reverse order and requires exact output equality with synchronous staging, while confirming both component batches were used.
- CPU focused tests: 10/10 after rebuilding the affected targets. CUDA expert-cache test passed, including reversed completion. AddressSanitizer/UndefinedBehaviorSanitizer passed. ThreadSanitizer passed five repetitions with `ignore_noninstrumented_modules=1`; see the earlier checkpoint for the archived noninstrumented runtime-library issue and standalone reproducer. This is not an unsuppressed TSAN claim.
- TinyMoE fixed F16: worker counts 0 and 2, each with pread and io_uring, on one/two/three CUDA devices, requesting 256 generated tokens at the same seed/context/precision/memory settings. All twelve initial runs match the corresponding synchronous output exactly. All cache modes performed uploads and eviction with zero forced fallback; queue/cache byte bounds held and final outstanding requests, fills, leases, staging and reservations were zero. Worker runs recorded `peak_active_loads=2`; serial runs recorded 1.
- A final matrix repeats those checks after removing unnecessary allocations from the synchronous path. All twelve final checks passed with exact serial/worker output equality; results are in `model-queue.json`. The final matrix overlaps the separate Release compilation, so its timings are not performance evidence.
- Malformed worker values -1, 9 and `2junk` are rejected.

## Performance decision

Keep the queue opt-in. The initial small, OS-cached fixture was slower with two workers in the Debug build despite demonstrating real overlapping fills. These exploratory runs were not a sustained, randomized production acceptance benchmark. The existing pread path already benefits from the operating-system page cache; extra worker dispatch and contention can cost more than any hidden read time.

A new `build_cuda_release` is compiling for CUDA architectures 75 and 86 using the same working CUDA toolchain. Complete that build, verify it, then compare worker counts on workloads with measured exposed storage wait as well as warm/resident controls. The local Qwen3 4B MoE, Qwen3.6 35B Q8, and GPT-OSS 120B files provide candidates; do not infer a speed gain from queue depth or the earlier synthetic delayed-source result. Tune worker count and staging within fixed total memory/quality/context budgets. Promote only a measured end-to-end improvement.

## Existing cross-mode numerical difference

A teacher-forced diagnostic kept 128-token histories identical across placements. Relative to the one-GPU resident run, the two-GPU resident run changed its raw top-token choice only at step 83, where the reference's top two logits tie exactly. The competing configuration separated them by 0.00390625. Its maximum absolute logit difference was 0.01171875. Disabling CUDA graphs reproduced the same diagnostic outputs.

The callback trace also found a near-tied expert boundary at layer 9, step 24: second/third router scores differed by about 0.00019, and a small numerical change swapped the selected second expert. Some cached comparisons have larger resulting logit differences. However, enabling the callback itself changes results, so the trace does not prove an exact first-difference location in uninstrumented execution. No tolerance was relaxed and no full-model quality gate was declared passed. These findings explain sensitivity in this fixture and narrow the investigation; Qwen4 placement correctness remains an independent open gate.

## Artifacts

From the repository root, `../results/benchmarks/ese-load-queue-20260904/` holds source backups, `load-queue-only.patch`, `checkpoint.json`, initial/final model matrices, native/sanitizer logs, argument checks and Release build commands/logs. The preceding `ese-gpu-parity-20260904/` directory holds the diagnostic probe source, forced token IDs, raw F32 logits, route traces and comparison scripts. Its reference binaries predate the worker parameter, so rebuild that probe against current headers/libraries before reusing it.
