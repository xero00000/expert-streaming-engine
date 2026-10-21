# ESE release work log — started 20 October 2026

Working branch: `review/ese-consolidated-20261020`.
Starting commit: `332a0af11`.

Work proceeds one task at a time. Finish the task's available checks and record
its evidence before advancing. An implementation pass does not imply package,
hardware or release qualification. The user's performance/publication hold
remains active: no push, remote CI dispatch, public tag or publication until
that hold is explicitly lifted.

GLM/expert-execution, dense-to-MoE and Android work remain on their separate
review branches. Existing launcher CUDA worktrees and `ESE-current` stay intact.

## Baseline already completed

- Source review and consolidation; recoverable source/history archives.
- Removal of 11 redundant worktrees and 11 obsolete local branches.
- Clean Release CPU build and all 94 registered native CPU regressions.
- 128 Python unittest checks; 33 reference/replay pytest checks.
- 32 Studio packaging/runtime JavaScript checks.
- Studio runtime staging, TypeScript and Vite production build; Rust formatting.
- Documentation/version checks. Public Qwen admission remains OFF.

See [the consolidation review](CONSOLIDATION_REVIEW_20261020.md) for exact scope
and `/home/xero/Desktop/test/results/ese-consolidation-review-20261020` for logs.

## Ordered tasks

### 1. Align the release package workflow — LOCAL CHECKS PASS; FULL BUILD PENDING

- [x] Define one versioned artifact naming contract for Linux CPU/CUDA
  DEB/RPM/AppImage packages and Windows dual-runtime MSI/NSIS installers.
- [x] Implement separate Linux CPU and CUDA matrix jobs with the production Qwen
  probe gate OFF; retain the separate CPU fallback inside Windows installers.
- [x] Make asset collection reject missing, stale, duplicate and mismatched
  bundles/signatures without replacing existing output.
- [x] Keep CPU and CUDA update targets distinct so an update preserves the
  installed runtime variant; update URLs, checksums and metadata together.
- [x] Add focused regression coverage and check workflow syntax, documentation
  consistency and the complete eight-package contract locally.
- [x] Record implementation results separately from actual package production.

Acceptance: automation consistently produces the expected eight packages with
matching names, signatures/checksums and safe updater targets. Actual package
builds/installations remain task 6; hosted workflow execution stays on hold.
The local implementation is checked. The full acceptance criterion is not yet
claimed: no new eight-package production build or installed-update trial has
been run. Continue with task 2's clean-source validation, retaining the full
package acceptance check for tasks 6 and 7.

### 2. Finish validation of the consolidated source — COMPLETE (LOCAL SOURCE CHECKS)

- [x] Pin the package-manager toolchain and prove a frozen-lockfile build without
  modifying dependencies in another worktree.
- [x] Build the consolidated CUDA runtime and run relevant CPU/CUDA regressions on
  RTX 3060 Ti, RTX 3080 and RTX 2080 SUPER, retaining device/source identities.
- [x] Run Rust tests and Clippy; repair failures without bypassing checks.

Acceptance: required clean builds/checks pass against one recorded source state.

### 3. Verify inference correctness and recovery — IN PROGRESS

- Compare supported resident/cache/stream paths against pinned references.
- Exercise forced eviction, cancellation, allocation failure and session
  recovery; retain logits/routes/output and ownership/accounting evidence.
- Keep experimental Qwen admission disabled unless its independent numerical
  acceptance gates are resolved; do not promote research based on compilation.

Acceptance: reproducible correctness and recovery evidence for every claimed
supported execution path.

### 4. Resolve the performance hold — PENDING

- Define the exact model, prompt, resource budget, policies and speed target.
  A target has not yet been agreed; do not invent a passing threshold.
- Collect matched cold/warm startup, prefill, first-response and sustained
  decode timings; compare like-for-like ESE/Strata conditions where possible.
- Profile and improve measured bottlenecks, preserving correctness and limits.

Acceptance: repeated measurements meet the agreed target. Performance evidence
does not itself lift the user's publication hold.

### 5. Qualify realistic workloads — PENDING

- Fill at least 65,536 actual prompt tokens with generation headroom and no
  silent truncation; check retrieval near the start, middle and end.
- Record RAM/VRAM peaks, swap/page-cache conditions and sustained behavior.
- Exercise supported concurrent sessions and reject unsupported combinations.

Acceptance: sustained runs without corruption, silent truncation or progressive
memory growth, with exact commands and failed trials preserved.

### 6. Rebuild and install the final packages — PENDING

- Rebuild all eight packages from the recorded final source.
- Test Linux CPU/CUDA installation, dependencies, startup and inference.
- Test actual Windows MSI/NSIS installation, CPU fallback and NVIDIA inference.

Acceptance: installed packages reproduce the validated behavior; hosted PE
checks do not substitute for physical Windows GPU evidence.

### 7. Finish signing and update handling — PENDING

- Verify signing-key configuration without exposing key contents.
- Check signature verification, variant selection, installation and update
  failure/recovery. Record Windows installer signing status accurately.

Acceptance: an update selects the correct runtime and verifies successfully;
unsigned installer limitations are explicit where applicable.

### 8. Prepare the release — PENDING

- Freeze source and artifact hashes; finish release notes and evidence links.
- Complete available final checks and prepare the PR/tag content locally.
- Keep push, hosted CI and publication pending while the user's hold applies.

Acceptance: a concrete, reviewable release candidate with all claimed checks
recorded and no outstanding supported-scope release blockers.

## Activity

### Task 1 — initial inspection

- Confirmed the branch is clean at the starting commit.
- The current tag workflow builds only one Linux CPU package set.
- Its collector/updater use older normalized names, while documentation and
  prepared candidates use explicit platform/runtime names.
- Studio currently has one Linux updater endpoint with no installed-variant
  distinction. Variant preservation must be part of task 1.
- No workflow has been dispatched and no publishing action has been taken.

### Task 1 — implementation and local results

- Added [the package contract](RELEASE_PACKAGE_CONTRACT.md) and shared Python
  helpers for collection, per-job source/hash provenance, complete-set
  validation, checksums and static update feeds.
- Added Linux CPU/CUDA release matrix jobs. The CUDA build installs pinned
  CUDA 12.8.0 components and NCCL 2.25.1 with byte-count/SHA-256 checks; it
  retains CUDA 12.8 runtime/BLAS versions from existing Linux notices. Windows
  retains its CUDA 12.4 build and separate CPU fallback. All release native
  builds explicitly keep the production Qwen probe gate OFF.
- Verified the four CUDA component entries against NVIDIA's official
  redistribution manifest. Independently streamed the NCCL archive and
  verified its retained size/hash without keeping a duplicate download.
  The complete CUDA installer has not been executed here.
- Linux packaging derives its updater configuration from the selected native
  build. Explicit missing or mismatched build requests fail before replacing
  staging. Added separate CPU/CUDA feeds and a CPU-compatible legacy feed;
  Windows feeds all use the dual-runtime installer.
- Added negative checks for tampered payloads/signatures, wrong or mixed source
  commits, relabelled variants, missing/stale/duplicate bundles, existing output,
  runtime mismatch and unsafe archive extraction.
- **143 Python unittest checks passed**, including 22 release-asset checks and
  four CUDA installer controls. **35 Studio checks passed**.
- **Actionlint 1.7.7 passed** for the changed release workflow. Documentation,
  Python compilation and Git whitespace checks passed.
- A CLI fixture exercised both Linux collections, Windows collection and final
  preparation: **eight package entries, eleven verified checksums**, distinct
  CPU/CUDA AppImage URLs and one dual-runtime Windows target. These were test
  payloads, not real installers or cryptographically verified updates.
- The actual consolidated CPU runtime staged successfully with a `cpu` package
  marker and passed collector runtime verification.
- Evidence is under
  `/home/xero/Desktop/test/results/ese-release-work-20261020`, particularly
  `task1-python-tests.log`, `task1-studio-tests.log`, `task1-actionlint.log`,
  `task1-cli-contract.json`, `task1-real-cpu-staging.json` and
  `task1-nvidia-lock-validation.json`.
- No remote dispatch, push, tag, new production installer build, installed-update
  trial or publication occurred. Those checks remain explicitly pending.

### Task 2 — toolchain isolation and validation in progress

- Pinned Node.js 22.22.2, pnpm 11.19.0 and Rust 1.96.0 in source and
  the Studio/release workflows. Rust CI tests and Clippy now use `--locked`.
- Removed only the consolidated checkout's `node_modules` symlink. Its old
  target was preserved; a new local frozen-lockfile installation and the full
  `pnpm build` passed without changing the frontend lockfile.
- Installed the checksum-pinned CUDA 12.8/NCCL 2.25.1 SDK locally. The first
  compiler ABI probe exposed nvcc's `lib64` search picking up an older Conda
  CUDA runtime, because the redistributables use `lib`. Added a repeatable
  `lib64 -> lib` alias and regression checks; the fresh ABI probe now passes.
  The interrupted first build/configure logs are retained as failed-attempt
  evidence. The corrected full native CUDA build is in progress.
- Rust tests passed (27 tests, zero ignored). Clippy and GPU qualification are
  still pending; task 2 remains in progress.

### Task 2 — local build and numerical fixture repairs

- The full CUDA build completed with the pinned SDK. Local Conda GCC 12.4
  needs its host `lib` directory during executable linking; the SDK is first
  in that build search path. Runtime validation uses only the SDK environment,
  and linkage confirms CUDA/cuBLAS/NCCL come from the pinned SDK. Hosted
  Ubuntu GCC 12 has not been exercised remotely.
- Strict locked Clippy passed without suppressions. Rust tests (27), Studio
  tests (35), Python unittests (146), reference/replay pytest checks (42),
  telemetry tests (4), type checks and a local Worker dry-run bundle passed.
- Documentation validation initially entered installed third-party READMEs.
  It now prunes dependency/build trees; a regression check still rejects
  missing links in maintained ESE documents. The repaired check passed.
- The initial RTX 3060 Ti suite exposed an expert-cache router fixture failure:
  actual `0.3002929688` versus an exact `0.30078125`. An independent uncached
  F16 operation reproduced the same result. Exact weights/products did not
  guarantee exact 256-term half-precision partial sums.
- Kept the same strict tolerance, poison, routes, eviction, cache-hit and lease
  checks. Changed the input to 16 nonzero terms of `1/16` plus zero padding,
  making every partial sum representable for these fixture weights. The
  uncached oracle and isolated cache-router check now agree exactly. Production
  arithmetic is unchanged. The complete final GPU rerun is still pending.
- CUDA Studio staging, frontend production build, runtime version, backend API
  discovery of all three GPUs and vendor linkage without `LD_LIBRARY_PATH`
  passed. A discarded smoke command used an unsupported `--list-devices` CLI
  option; device discovery was checked with the actual staged backend C API.

### Task 2 — final acceptance results

Validated source: `f27f6462c6d71dd686108a91d6a524446e9d2681`. Subsequent
work-log/report edits only record these results; they do not change runtime
code. Both native runtime version strings identify that source commit.

- Complete CPU and CUDA Release builds passed. CPU uses local GCC 16.2.1;
  CUDA uses GCC 12.4.0, the pinned CUDA 12.8 SDK and NCCL 2.25.1, with real
  SM 75/86 code plus SM 86 PTX. `GGML_NATIVE` and the production Qwen gate
  remain OFF. Native production sources are unchanged by task 2.
- **CPU: 94/94 CTest checks passed.**
- **RTX 3060 Ti, RTX 2080 SUPER and RTX 3080: each has 99 CTest passes, zero
  failures and one expected single-device NCCL skip.** The BF16 batch test
  reports an internal skip under default CTest settings; it was separately
  enabled explicitly and **all 64 cases passed on each card**, including
  scalar-bit equality, the quantization-aware double oracle and graph replay.
- **30 additional explicit CUDA reference checks passed**, ten per GPU,
  covering expert IQ formats and DeepSeek operator/index/QAT/attention/MoE,
  block, forward, BF16-forward and whole-model synthetic fixtures. These
  fixtures are regression evidence, not real-model acceptance.
- **Five multi-GPU checks passed.** All-three-device NCCL initialized and
  produced exact F32/F16/BF16 reductions across two concurrent contexts. The
  two-of-three case correctly used the exact reduction fallback. Device
  selection, asynchronous copies and primary-context isolation also passed.
- Repeated final-source checks: **146 Python unittests, 42 reference/replay
  pytest checks, 35 Studio JavaScript checks, 27 Rust tests and four telemetry
  tests passed**. Locked strict Clippy, Rust formatting, frozen frontend
  installation/build, telemetry type checks/dry-run, documentation and both
  workflow lints passed. The three dependency lockfiles remain unchanged.
- Final CUDA staging passed its variant marker, source version, backend API
  discovery of all three GPUs and vendor-library linkage checks with no
  `LD_LIBRARY_PATH` or `LD_PRELOAD` override.
- Failed initial attempts, the uncached F16 reproduction, the corrected
  fixture checks and all final commands/logs are retained. Start with
  `/home/xero/Desktop/test/results/ese-release-work-20261020/task2-summary.json`;
  final per-device logs and hashes are in `task2-gpu-final/`. Exact compiler,
  SDK, device UUID and source identities are in
  `task2-source-environment-final.json`.

Task 2's local acceptance criterion is met. Task 3 is next: supported-path
inference correctness and recovery with pinned model references. Performance,
filled-64K/concurrency, eight real installers, signing/updates and physical
Windows evidence remain open. No push, remote workflow, tag or publication
occurred; the user's hold is unchanged.


### Task 3 — pinned fixture and execution plan

- Reuse the retained TinyMoE F16 fixture, SHA-256
  `c559eaabd8a56589b8abe09d678fbb56c673a3dcc28907e193d27b25ecd60684`.
  Compare resident, storage/cache and native stream paths using the consolidated
  runtime, then check worker queues and bounded prefill staging on the three
  available GPUs. Resident execution is the path reference; this does not
  establish upstream full-model numerical acceptance.
- Retain raw outputs, forced-token routes/logits, resource accounting and
  source/runtime identities. Check eviction and multi-device rollback without
  relaxing existing numerical assertions.
- Rebuild the retained HTTP abort shim against the consolidated API. Distinguish
  a real CPU graph abort from injected decode statuses and allocation admission
  failures. Check client disconnect cancellation and subsequent session use.
- Small fixtures cannot qualify large-model performance, filled-64K workloads,
  experimental Qwen/MTP/multimodal paths or actual driver out-of-memory recovery.
  Those limits remain explicit; public Qwen admission and publication stay OFF.


### Task 3 — cache matrix and two numerical defects

- The initial cache acceptance run stopped at its obsolete 4 MiB fixture budget.
  Padding headroom now leaves only one usable slot, so top-2 routing correctly
  stays on CPU. Commit `d249dd3e2` uses 6 MiB for two GPU slots and checks the
  smaller budget independently against CPU-expert execution. The corrected
  one/two/three-GPU matrix passed, including mmap/pread/io_uring CPU parity.
- Before numerical repairs, 24 sustained 128-token worker/backend trials passed
  with bounded, drained ownership. Bounded prefill passed on one/two GPUs. The
  equal-share three-GPU launcher run retained matching outputs but used expert
  staging on only two devices; repeat with explicit placement that assigns MoE
  layers to all three. Do not report that initial three-GPU trial as passed.
- Seven fresh HTTP abort/status trials passed against the consolidated runtime:
  four actual CPU graph aborts (with/without a system prefix), injected -2/+2
  statuses and recoverable +1 capacity pressure. These are not physical OOM
  or experimental-model acceptance.
- New real-disconnect checks found repeat-request corruption before cancellation.
  A CPU softmax/router adjacency shortcut independently failed a poisoned-output
  probability/route oracle. `c69fd21d8` restores normal softmax semantics, including
  masks/scaling and live intermediate output, with a permanent regression.
- A separate BOS/timing heuristic selected all experts for ordinary first-token
  microbatches. Replace it with explicit warmup state, invalidate graphs on
  transitions and preserve allocation-only reservation. The new generated MoE
  regression failed on the old behavior. With the repair, retained TinyMoE
  native cold replays match all 32,000 logits exactly and three real CPU-stream
  disconnects drain resources and preserve retry probabilities. Final rebuilds
  and affected-matrix reruns are still in progress.
- A strict 4 GiB automatic-plan startup also rejected its host context: the
  selected caches leave insufficient tracked context metadata/output headroom.
  Retain this as an open planning defect; bounded explicit cache configuration
  can test transaction recovery independently. Do not weaken host admission.
- Evidence, including unsuccessful controls, is retained under
  `/home/xero/Desktop/test/results/ese-release-work-20261020/task3/`.
  No performance threshold, full-model acceptance or release readiness is claimed.
