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

### 2. Finish validation of the consolidated source — PENDING

- Pin the package-manager toolchain and prove a frozen-lockfile build without
  modifying dependencies in another worktree.
- Build the consolidated CUDA runtime and run relevant CPU/CUDA regressions on
  RTX 3060 Ti, RTX 3080 and RTX 2080 SUPER, retaining device/source identities.
- Run Rust tests and Clippy; repair failures without bypassing checks.

Acceptance: required clean builds/checks pass against one recorded source state.

### 3. Verify inference correctness and recovery — PENDING

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
