# Desktop validation and overhead checkpoint — 2026-10-02

Android and publication excluded. Objective remains correct, bounded desktop execution with lower startup/TTFT/decode cost. This checkpoint does not claim a general speedup or full engine certification.

## New repair

Native resource planning copied `expert_sidecar_only` and `expert_load_workers` into a dry-run metadata probe while explicitly disabling its RAM cache. A valid strict-sidecar streaming invocation failed with `--expert-sidecar-only requires a non-zero expert RAM cache`, before planning completed. The metadata probe now clears these two runtime-only controls; the real load keeps the user's strict requirements. A real TinyMoE CPU invocation with native stream policy, strict sidecar and two configured workers passes after the change. The existing phase-2 acceptance runner now checks this combination and exact baseline output/RAM bounds. This validates probe/CPU execution, not GPU worker activity.

## Small overhead reductions

- io_uring batch completion identities use a bounded 64-bit mask instead of a heap-allocated vector<bool>. The ring requests 64 entries, and batches are limited to that ring capacity.
- Empty-pending staging allocation returns offset zero directly rather than constructing/sorting an empty interval vector.

These remove unnecessary host work without changing admission or storage semantics. No end-to-end speed claim is established. The file-source test now covers 129 reads across multiple ring batches, including bit 63, as well as prior single/batch error recovery. CPU, CUDA cache and strict ASAN/UBSAN tests pass after the changes.

## Verified now

- 18 focused Release CTests pass with RTX 2080 SUPER selected: cache, resource planner, Turbo KV core/policy, transient manager, spec tuner, MTP transaction, mapped draft, KDA DeltaNet, Qwen4Exp schema/operators/PLE storage/reference and architecture rejection, GGUF parser regression.
- Strict ASAN/UBSAN (halt-on-error, leak detection) expert-cache and GGUF parser tests pass; cache suite was rerun after the overhead changes.
- TinyMoE CPU mmap/pread/native io_uring runs have exact generated output against the baseline, bounded RAM/staging, zero outstanding loads/leases/reservations at teardown.
- TinyMoE single-GPU prefill acceptance passes three requests each for established/staged paths: exact output hashes, two staging lanes using 28,311,552 bytes under a 64 MiB limit, H2D and persistent-cache D2D traffic, zero correctness/global-sync fallbacks. Observed median prompt ratio 1.109 is a small same-prompt screen, not promotion evidence; this is an existing feature, not a newly measured optimization.
- TinyMoE live resource acceptance passes on the 2080: KV shrink/grow, busy rejection, KV rollback, expert-cache resize/rollback and combined transaction injected-failure stages.
- 85 Python tests pass.
- Studio TypeScript no-emit check and 15 frontend/package/report tests pass. No full Rust/UI interaction suite was run.
- Release server/CLI rebuild and git diff --check pass.

Raw model acceptance artifacts are under ../strata/benchmarks/: ese-prefill-validation-20261002.json, ese-storage-validation-fixed-20261002.json, ese-rebalance-tinymoe-20261002.json. These are feature screens, not large-model performance results.

## Failed/inconclusive screens retained honestly

The stories15M Q4 dense fixture failed the resize output check. A control shows cold-prompt generation already differs from an ordinary cached repeat before any transaction; subsequent repeat/shrink/grow outputs matched each other. This is an unresolved prompt-cache/batched-versus-single-token numerical investigation, not evidence that resizing caused the difference. The validator was not weakened or warmed artificially to conceal it.

An attempted 12-run worker timing matrix had identical output, but VRAM uploads and queue submissions were zero. Its promotion assertions rejected it and no worker-matrix report was saved. Follow-up CLI probes with CPU-MoE overrides also showed zero GPU route activity; native stream planning chose a resident fit for the tiny fixture. These runs cannot verify GPU worker throughput. Corrected workload/placement with nonzero submissions/uploads is needed. Existing controlled queue and reversed-completion CUDA unit tests pass but do not replace that end-to-end gate.

The checked-in Qwen4Exp PLE reference test intentionally fails in a CUDA build with GPUs hidden; it passes on the 2080. Do not mislabel the hidden-device result as an engine failure or relax its hardware requirement.

## Next gates

1. Complete fault-injected io_uring submit/partial-submit/wait handling with explicit ownership-safe draining. Previous review's syscall-error-path concern remains open.
2. Obtain a supported large-model baseline on verified NVMe (models/nvme-bench is currently empty in this inspection), with profiles proving exposed host/storage costs. Do not infer gains from archival scripts or compare placements as an optimization ratio.
3. Reproduce real GPU worker submissions and exact synchronous/worker outputs, then extend to 1/2/3 GPUs under fixed budgets.
4. Diagnose cold/cached dense-fixture drift with controlled batch geometry/logits and other supported fixtures.
5. Run full-model KV quality, MTP generation/rollback, transient multimodal ownership, cancellation and long-context soaks. Current unit checks do not certify those capabilities on all models.

Strata's tuned Ampere pair was left running; ESE GPU feature checks used the separate 2080 SUPER. This avoids model service interruption, but shared CPU/RAM activity means those observations are not clean production performance measurements. No defaults, model weights, architecture gates, commit, or push were changed.
