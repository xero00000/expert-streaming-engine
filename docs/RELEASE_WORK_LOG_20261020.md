# ESE release work log — started 20 October 2026

Working branch: `review/ese-consolidated-20261020`.
Starting commit: `332a0af11`.

Work proceeds one task at a time. Finish the task's available checks and record
its evidence before advancing. An implementation pass does not imply package,
hardware or release qualification. The user's publication hold
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

### 3. Verify inference correctness and recovery — COMPLETE (LOCAL FIXTURE ACCEPTANCE)

- Compare supported resident/cache/stream paths against pinned references.
- Exercise forced eviction, cancellation, allocation failure and session
  recovery; retain logits/routes/output and ownership/accounting evidence.
- Keep experimental Qwen admission disabled unless its independent numerical
  acceptance gates are resolved; do not promote research based on compilation.

Acceptance: reproducible correctness and recovery evidence for every claimed
supported execution path.

### 4. Resolve the performance hold — COMPLETE (CURRENT SPEED ACCEPTED BY USER)

The [performance qualification protocol](PERFORMANCE_QUALIFICATION_20261020.md)
is prepared. The initial speed target was to match Strata; the user later
set a model-specific minimum of **80% of Strata**, as recorded below.

On 21 October the user clarified that qualification depends on the model and
requested testing **the Strata profile in the existing launcher**. That profile
is now the next reference; the earlier 35B comparison does not qualify a
Flash-Next launcher model. See the continuation entry below for profile selection
and compatibility evidence.

The user subsequently revised the required speed to **at least 80% of Strata**
and requested restoring the branch that loaded this model. Selected reference:
**Flash-Next IQ2_XS, 1M experimental**, using the saved
`strata-iq2_xs-1m.json` profile across all three GPUs. Its measured short-request
decode medians are 51.5 tok/s overall and 52.3 on repeats; the corresponding
ESE thresholds are **41.2 overall and 41.84 tok/s on repeats**. ESE currently
rejects this model architecture in its default build, so the current work
restores and tests the existing dedicated loading path before qualification.

On continuation, the user accepted the current speed: **“ill take 19 tokens
for now we can continue to improve speeds in later updates.”** This supersedes
the 80% speed requirement for this version. Further Strata-speed optimization
is deferred; the original baseline and failed comparisons remain historical
evidence. The accepted one-start/three-request diagnostic reaches 13.968 /
19.841 / 19.664 tok/s, with unchanged native output hashes. This is acceptance
of the observed speed, not a three-start qualification or a claim of Strata
parity. Native arithmetic, long-context/lifecycle acceptance and the production
admission gate remain open; this is not GitHub/release readiness.

- Define the exact model, prompt, resource budget, policies and speed target.
  Use Strata's measured sustained decode rate as the reference; retain startup,
  prefill and first-response timings separately.
- Collect matched cold/warm startup, prefill, first-response and sustained
  decode timings; compare like-for-like ESE/Strata conditions where possible.
- Profile and improve measured bottlenecks, preserving correctness and limits.

Acceptance: the user accepts the observed approximately 19 tok/s for this
version. Record the selected configuration and keep correctness/resource gates
separate. The speed acceptance does not lift the publication hold.

### 5. Qualify realistic workloads — SHORT CHECKS PASS; LONG PASSES DEFERRED BY USER

- Fill at least 65,536 actual prompt tokens with generation headroom and no
  silent truncation; check retrieval near the start, middle and end.
- Record RAM/VRAM peaks, swap/page-cache conditions and sustained behavior.
- Exercise supported concurrent sessions and reject unsupported combinations.

Acceptance: sustained runs without corruption, silent truncation or progressive
memory growth, with exact commands and failed trials preserved.

### 6. Rebuild and install the final packages — LINUX CHECKS PASS; WINDOWS PENDING

- Rebuild all eight packages from the recorded final source.
- Test Linux CPU/CUDA installation, dependencies, startup and inference.
- Test actual Windows MSI/NSIS installation, CPU fallback and NVIDIA inference.

Acceptance: installed packages reproduce the validated behavior; hosted PE
checks do not substitute for physical Windows GPU evidence.

### 7. Finish signing and update handling — LINUX CHECKS PASS; WINDOWS PENDING

- Verify signing-key configuration without exposing key contents.
- Check signature verification, variant selection, installation and update
  failure/recovery. Record Windows installer signing status accurately.

Acceptance: an update selects the correct runtime and verifies successfully;
unsigned installer limitations are explicit where applicable.

### 8. Prepare the release — LOCAL REVIEW DRAFT PREPARED; FULL MATRIX PENDING

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


### Task 3 — numerical and recovery validation completed; planning repair next

- Native candidate `9e0d3b037` rebuilds successfully for CPU/CUDA. CPU CTest
  passes 94/94; each of the three GPUs has 99 passes and one expected
  single-device NCCL skip. Both targeted strict ASAN/UBSAN/leak suites pass.
- Forced-token resident/cache/stream comparisons preserve all 512,000 logits
  and 320 route IDs per configuration. Uninstrumented eviction/storage checks,
  24 sustained worker trials and bounded prefill on one/two/three GPUs pass.
  The three-device prefill split is `1,1,8`, assigning MoE layers to every card.
- CPU and one/three-GPU KV/expert transactions, injected prepare/publish failures
  and exact continuation pass. The old HTTP verifier compared physical native
  receipts to serialized forecasts. `60ab8fdf2` reconciles live memory and a
  separate KV-only receipt, with six positive/negative regressions. Exact byte
  and peak arithmetic assertions remain mandatory.
- All 152 Python tests pass. Twenty-one real HTTP disconnects and seven final
  graph-abort/decode-status controls pass. The refreshed staged Studio CUDA
  runtime identifies the candidate and passes real inference/cancellation with
  no library-path/preload override. No new installer was built or installed.
- Details and limits are in
  [the inference/recovery validation](INFERENCE_RECOVERY_VALIDATION_20261020.md).
  Start the evidence review at
  `/home/xero/Desktop/test/results/ese-release-work-20261020/task3/task3-summary.json`.
- **Task 3 remains open.** The current CUDA candidate still rejects an automatic
  4 GiB RAM plan: 4,295,429,216 tracked bytes are required against 4,294,967,296
  usable. The CPU-only startup control fits; GPU placement leaves no host graph
  reserve to absorb omitted context metadata/output. Explicit bounded caches
  permit independent recovery testing but do not repair automatic planning.

Next, stay within task 3 and repair the automatic host forecast: account for
context/KV metadata, outputs and index maps before filling caches; preserve
actual capacity/rounding contracts; add a reproducing startup/boundary check;
then repeat the strict-budget GPU startup and affected recovery checks. Do not
weaken native host admission. Full-model, physical driver OOM and task 4 onward
remain unqualified. The performance/publication hold is unchanged.


### Task 3 continuation — native host startup reservation repair

The automatic GPU failure is being repaired by reconciling the provisional
plan with the native preflight receipt after model placement/context
normalization, before KV/output/graph payload allocation. The finalized plan
reserves model/context/KV bookkeeping and initial outputs separately from
graph metadata, reduces optional prompt-cache capacity to fitting whole MiB,
and retains the reservation in immutable runtime rebalance accounting.
The RAM ceiling and ordinary native admission checks remain enforced.

Focused CPU planner/recovery regressions and automatic 4 GiB CPU startup,
two cold inference requests and exact probability replay pass in a working
build. Final committed-source CPU/CUDA/sanitizer checks and the original GPU
reproducer are next; task 3 remains open until those results are recorded.


The first repair (`90cd89815`) clears the missing-context preflight rejection
but the original GPU trial then exposed missing **host graph backing** in GPU
plans. That failed trial remains in `task3/host-plan-fix/automatic-2080/`.
The follow-up measures the allocator's actual native arena layout without
allocating buffers or altering committed layouts, reserves host arena peaks
separately from fixed metadata, and reconciles optional cache capacity during
startup. No arbitrary margin or relaxed memory admission is introduced.


### Task 3 completed — native host planning and final regression qualification

Native source `9e55c492525def32759f53aee73798ad2d1f7f7d` completes the
context-storage and measured host-graph reservation repair. Optional prompt
cache capacity is finalized in whole MiB before application cache allocation;
startup reconciles native KV/output metadata, measured arena replacement peaks
and final tracked storage. Host admission and the immutable RAM ceiling remain
strict. The internal measurement hook preserves committed layouts and allocates
no backend buffers; boundary, overflow, rejection and old-output regressions pass.

The original RTX 2080 SUPER automatic `--max-ram 4GiB` case now starts, generates
twice with exact cold probability replay, and exits normally. Its finalized plan
reserves 461,928 bytes beyond graph metadata and 3,414,016 bytes of host graph
backing, with 4,202,692,608 bytes of optional prompt cache. The tracked admission
total after inference is 4,294,099,912 bytes, below the 4,294,967,296-byte ceiling.
Eight automatic-budget controls pass: CPU with/without warmup, each GPU,
three GPUs, GPU warmup and the staged runtime without library-path overrides.

Final-source checks pass: 94 CPU tests; 99 CUDA passes and one expected NCCL
skip on each of three GPUs; four strict ASAN/UBSAN/leak targets; 152 Python
unittests; exact route/logit and forced-eviction matrices; 24 sustained worker
trials; bounded prefill on one/two/three GPUs; CPU/three-GPU rebalance faults;
21 real disconnect/retry trials; and seven HTTP decode-failure controls.
Ada/newer GPUs, real driver exhaustion and upstream full-model numerical
qualification are not claimed by these fixture results.

Current evidence entry point:
`/home/xero/Desktop/test/results/ese-release-work-20261020/task3/host-plan-qualified/summary.json`.
It retains source/binary/compiler/device identities, raw outputs, commands,
responses, receipts, failure trials and checksums. Earlier evidence and failed
trials remain intact. Task 3's local fixture acceptance is complete; release
readiness still requires the remaining work-log tasks. No push, remote workflow,
tag or publication occurred.

Task 4 has begun with a concrete measurement protocol and local model/hardware
inventory. Its acceptance contract remains open pending the selected model,
prompt/resource budget and required speed target; no threshold was invented.

### Task 4 continuation — Strata target and first direct baseline

The user has now set the speed target: **match Strata**. Both engines complete
three fresh processes with three 512-prompt/128-generation requests each on the
same local IQ4_XS qwen35moe model, RTX 3080 and eight CPU threads. A separate
benchmark copy adds only Strata's required explicit zero-MTP metadata; every
one of its 733 tensor hashes matches the original. Original model files,
Strata service configurations and the experimental Qwen4Exp gate are preserved.

Strata's median decode is 45.29 tok/s; ESE's mapped-cache control is 31.18 tok/s
(0.688x). Repeated-request medians are 45.75 versus 31.38 tok/s. Both independently
repeat the exact generated output across all nine requests, preserve full token
counts and exit cleanly. Shared desktop load, differing actual resource use,
Strata swap and ESE's deferred first-request cache loading remain explicit.
The initial ESE pinned-model control was aborted above 35 GiB process RAM;
disabling only model-weight pinning eliminates a duplicate 14.63 GiB host copy,
reduces startup to 1.53–5.91 s and keeps observed process swap at zero.

Reusable native/Strata benchmark runners retain commands, binary/model/runner
identities, raw responses, receipts and sampled memory. Four measurement-integrity
tests and documentation checks pass. Fused-reduction tuning preserves output
but has no demonstrated speedup; cache/upload tuning is next. Native source
and binaries remain `9e55c4925`, so no new native qualification is claimed.

Task 4 remains open: the recorded comparison **does not match Strata**. Do not
advance to task 5 or lift the publication hold on these results. Detailed values,
limits and evidence paths are in the
[performance qualification record](PERFORMANCE_QUALIFICATION_20261020.md).

The larger-cache pilot exposed a second-request prefill abort at the live GPU
reserve boundary; that configuration is rejected and retained for admission/
recovery repair. Model-backed hybrid calibration independently exposed an AVX2
IQ4_XS CPU saturation error (12.8% versus scalar reference; the direct row dot
has 0.4% error). A synthetic regression fails before the narrow repair and
passes after it; final-source builds and affected checks are next. Calibration
thresholds and the mixed-routing validation gate remain enforced.

### Task 4 continuation — CPU correctness and workload measurement repair

Committed native source `f1c381c5f` fixes the AVX2 IQ4_XS saturating pair
products using signed lookup values. All 27 synthetic geometry/sign cases
agree with scalar dequantization and between one/four threads. Model-backed
CPU oracle errors are now 0.00377–0.00400 for Q6_K, IQ3_S and IQ4_XS, below
the unchanged 0.08 limit. Full calibration passes and produces a planner-ready
profile. The staged Studio runtime has not been rebuilt from this source.

Final-source checks pass: 95 CPU tests, 100 CUDA passes and one expected NCCL
skip on each physical GPU, and five strict ASAN/UBSAN/leak targets. Evidence:
`task4/iq4-fix-f1c381c5f/`, `task4/cpu-reference-diagnostic/` and
`task4/3080-iq4-hardware-profile-f1c381c5f.json` under the existing results root.
These checks do not replace upstream full-model numerical qualification.

The first mixed-routing A/B attempt is invalid: the existing validator appended
trial text after the supplied assistant-generation prefix, causing immediate
EOS. It accepted one generated token as a 128-token timing sample (a meaningless
1,000,000 tok/s). The run was stopped, its raw logs retained, and no verification
pass saved. New regressions reproduce the defect before repair. The validator
now preserves the exact supplied prompt, requests `ignore_eos`, requires full
generation and consistent prompt/timing counts, rejects truncated or invalid
durations, and checks throughput against counts/duration. Verification format
version 4 invalidates earlier receipts. Optional `--evidence-dir` retains native
logs even when validation fails. All 160 Python tests pass after the repair.

The Strata speed target remains unmet. A valid mixed-routing A/B trial and the
live prefill-reserve recovery repair are next; publication remains on hold.

The repaired full-token mixed A/B trial finishes with exact output parity, but
fails acceptance: measured medians are 29.16 tok/s for its established-path
control and 26.13 tok/s for the candidate after fallback (0.896x). The live
guard revokes the candidate during its first decode at 6.108x predicted upload
cost; it remains disabled. Its first prefill takes 197.13 s and first decode
78.14 s, retained separately from the three measured repeats. Evidence:
`task4/3080-iq4-hybrid-verification-full128-f1c381c5f.json`, the command record,
and `task4/hybrid-gate-full128-f1c381c5f-raw/`. The native runner now also checks
reported throughput against count/duration arithmetic; its regressions pass.

Native source `8cd98995a` repairs request-boundary prefill reserve loss. Strict
sidecar mode checks the actual CUDA free memory on staging backends before
launching any graph split or parallel barrier. Insufficient reserve returns
`GGML_STATUS_ALLOC_FAILED` and drains pass leases/redirects while retaining
admitted lanes for retry. No host-tensor fallback or reduced reserve is enabled.
A tiny staged CUDA graph reproduces the original SIGABRT before repair. After
repair it returns the allocation error without leases or output mutation, then
retries with exact output and unchanged staging allocation after the reserve is
restored. The test raises the configured reserve without exhausting the driver.
CPU (95), Python (160) and five strict sanitizer checks pass. Final CUDA checks
on each card and the original full-model recovery control are underway.
External allocations racing after the boundary check remain outside this repair.

The committed reserve repair now passes the complete local native suite: 95 CPU
tests, 100 CUDA passes plus one expected NCCL skip on each of the three cards,
five strict ASAN/UBSAN/leak targets, and 160 Python tests. Source/binary/library
identities and raw evidence are retained in `task4/prefill-reserve-fix/`.
Its full-model 4.75 GiB control finishes three 512/128 requests with exact
independent replay, safe idle resources, zero active leases and normal exit.
In this live memory state, component preflight rejects the GPU cache and uses
CPU execution; the old reserve abort is **not** reproduced. That control is
neither GPU-cache performance acceptance nor proof of a real driver-boundary
recovery. The deterministic staged CUDA regression supplies the reserve-loss
rejection/retry evidence. The oversized cache remains a rejected tuning setting.
The next pilot packs heterogeneous expert layouts within the original 4 GiB
cache, with the frozen workload and benchmark protocol unchanged.

The packing pilot completes with 2,900 slots in 4 GiB and 32.85/33.15 tok/s
on repeated requests, below Strata's target. Its own output repeats exactly;
its hash differs from the earlier native source's control. That difference is
unresolved until the repaired source is rerun without packing; no numerical
parity or packing-only gain is accepted yet. The initial request has 203.76 s
TTFT with only 21.9% initial file residency on the FUSE volume. A separate
byte-identical model copy is being prepared on local NVMe/btrfs to separate
storage costs from cache tuning. Both engines will be rerun on the same new
copy; all previous models, manifests and comparison results are preserved.

### Task 4 continuation — 21 October NVMe controls and resident profile

The NVMe copy verifies byte-for-byte against the comparison SHA; both engines
are rerun on that copy. Strata completes nine requests at a 45.51 tok/s median.
The repaired ESE uniform-cache control repeats the original output exactly,
at 30.60 tok/s overall / 30.93 on repeats. NVMe cuts its initial TTFT to 15.29 s
in this run, without closing the steady decode gap.

Packing's output change is attributable to its automatic stable-reduction
mode: the unpacked stable-mode full-model control matches packing exactly.
All 36 added geometry cases pass in that mode on all three GPUs; disabling
CUDA graphs preserves the packed output. The initially suspected packing
parity failure is not supported by the controls with the same reduction mode.
Legacy reduction is sensitive to physical cache geometry. The new CTest
explicitly enables stable reduction. Frequency retention is slower and rejected.

A residency index reuses observation nodes, preserving output and cache counts,
but its pilot shows no gain. Commit `9c25eee77` is reverted by `d3bd3fa48`.
Current native production source content is identical to the fully qualified
`8cd98995a` implementation; the additional geometry test passes on every card.

The fully resident **three-GPU** profile completes three fresh starts with
three requests each: 81.03 tok/s median (81.24 on repeats), exact independent
output replay, full 512/128 counts and normal shutdown. It exceeds Strata's
single-GPU rate by 1.78x using a different allocation. It does not close the
same-GPU gap. These results did not settle acceptance. The subsequent launcher
profile selection below defines the active reference. Resource/latency limits
and every pilot are recorded in the performance qualification document and
the results root.
Studio staging, later work-log tasks and the publication hold remain unchanged.

The 4.375 GiB packed-cache pilot retains the 1 GiB reserve and completes all
three requests with stable-mode exact replay and normal drain/shutdown, but
repeated decode is only 33.37/33.47 tok/s. Single-3080 acceptance remains open.
The corrected nine-run resident median is 81.02568 tok/s; the earlier 81.46
figure was an individual sample and has been corrected in user-facing status.

### Task 4 continuation — existing Strata launcher profile

The user requested the launcher profile rather than choosing an allocation for
the 35B comparison. `model_slots.sh` maps three Strata entries: Coder IQ1_M
128K, Flash-Next IQ2_XS 256K daily, and Flash-Next IQ2_XS 1M experimental.
The daily 256K profile was the initial working assumption while the user's
specific profile choice was pending. Its config is `strata-iq2_xs.json`: GPUs
`1,0,2`, layer boundaries `6,14`, K8V4 KV, resident experts, two pipeline
windows, speculative MTP depth four, vision reservation and the existing
runtime control vector. These differ materially from the earlier 35B test.

Both IQ2_XS GGUF shards are present (39,225,954,592 and 28,800,138,432 bytes).
Metadata identifies `Qwen3.8-Flash-Next GSQ-RCO`, architecture `qwen4exp`,
48 layers and 512 experts with ten selected per token. The Coder profile's
configured GGUF paths are missing; that profile cannot currently start as saved.
All three original configurations and their file inventory are retained in
`task4/launcher-profile-comparison/` without editing the launcher.

The current release CUDA binary is tested directly against the IQ2_XS model
and rejects it with `unsupported architecture: qwen4exp` (exit 1). Its compiled
probe gate remains OFF. This is an actual model-support blocker, so no ESE
speed or Strata parity can be claimed for this profile. The initial Strata
baseline used an isolated copy of the daily profile,
changing only the local service port and engine-log destination. Original
models, settings, production services and publication hold are preserved.
The standalone upload microbenchmark was deferred when the user changed the
reference; no transfer-buffer implementation was added.

The user subsequently selected **Flash-Next IQ2_XS, 1M experimental** explicitly.
The partial daily run is preserved: its first process completed three full
552/128 requests; the second startup was interrupted on selection. It is not
the selected acceptance baseline. `selection.json` records this change.
The selected `strata-iq2_xs-1m.json` retains the same models, GPU order,
layer boundaries, control vector, MTP and vision, but requests 1,048,576
context cells, Q4 KV with 32,768 resident cells, and YaRN scaling of four.
Three fresh starts with three short requests each have completed.
This tests short-request speed under the saved 1M allocation, not a filled
million-token prompt. Both full shard hashes were obtained before the daily
run; their file identities are rechecked for the selected run.

The selected profile completes all nine full 552/128 requests and three clean
service shutdowns with no remaining owned processes. Decode medians are
**51.5 tok/s overall, 52.3 on repeats, and 45.4 on first requests**. Fresh
startup takes 107.06–121.92 s. Repeats reuse 547 of the 552 prompt tokens;
their speed and prefill are not represented as full fresh-prompt work.
The 128 generated tokens include reasoning. This is short-request speed under
the 1M configuration, not filled-context or answer-quality qualification.
All nine output sequences differ despite the requested zero temperature and
fixed seed; the cause is not established, and deterministic replay is not
accepted. Largest monitored-process HWM is 33.52 GiB and sampled process swap
reaches 1.42 GiB. These resource conditions remain part of the reference.

The ESE load is also repeated with the selected 1,048,576 context request:
the same architecture rejection occurs before KV, YaRN or speculation
admission. The probe gate remains OFF. Config hashes verify that all three
original launcher profiles are unchanged. Summary, strict count/timing checks,
raw SSE, process/GPU samples and failure records are retained in
`task4/launcher-profile-comparison/selected-profile-summary.json` and its
referenced runs. No ESE/Strata speed ratio is eligible for this model.
`auxiliary-inputs.json` retains post-run hashes for pack/tokenizer files,
MTP artifacts, routing profile, control vector, vision and service code.

Required work for this selected model, in order:

1. Revalidate the current runtime against the recovered independent references
   for this exact 512-expert IQ2_XS artifact. The initial reference inventory
   missed `ese-iq2-xs-selected-model-20261013`; its full and cached 33-token
   references and six passing historical GPU comparisons are now recovered.
2. Resolve the remaining model/cache/speculation numerical gates and qualify
   its selected control-vector, YaRN and KV behavior before release admission.
   Existing gated research remains separate; simply opening the gate is not a fix.
3. Run ESE under the agreed model/profile workload and resource conditions,
   then compare fresh and repeated rates against this saved Strata reference.

Task 4 and the publication hold remain open. No engine source, launcher,
installed Studio runtime or package was changed by this profile test.

### Task 4 continuation — restore loading and reach 80% of Strata

The user explicitly requested getting this model working from the old loading
branch and set the minimum speed to 80% of Strata. The preserved
`plan/v0.2.1-qwen4exp` worktree has `build_qwen4exp_probe` with its configure-time
gate ON; the standard build keeps it OFF. The consolidated source already
contains the same Qwen model graph/operations. A dedicated local loading build
will use the current candidate's objects plus the probe-enabled loader, then
test the selected 512-expert IQ2_XS artifact. This avoids importing unrelated
GLM/dispatch research or changing the installed launcher during diagnosis.
Required speed is frozen at 41.2 tok/s overall and 41.84 on repeats for the
saved short-request reference. Model correctness and memory checks remain
required; loading by itself does not establish release readiness.

The recovered selected-model evidence has the same two complete GGUF hashes
as the new launcher baseline. It includes passing 33-token comparisons against
both pinned upstream references on all three cards. This corrects the earlier
incomplete statement that reference evidence existed only for the Coder
artifact. Those old results are bounded evidence, not current-build or 1M
qualification. Current dedicated loading succeeds and reproduces the saved
eight-token raw `Hi` control exactly with the explicit precision policy.
The release runtime and installed launcher remain untouched. A current
33-token reference replay and normal-policy performance diagnosis follow.


#### Restored loading and first decode optimizations

The dedicated current loader now passes the recovered 33-token selected-model
references: 8,194,560 logits against each of the full/cached upstream controls
within the unchanged `2e-5 * (1 + abs(reference))` bound, all 1,584 ordered
expert-route rows and all 33 top-one choices. The lifecycle observer needed to
recognize the selected IQ4_XS vocabulary head as well as Q6_K; the builder
already used explicit F32 precision for both. The 1M serving profile and normal
quantized arithmetic are not qualified by this bounded F32-cache replay.

Two isolated CUDA changes follow the profiling evidence:

- Route top-10 Q2_0 down projections through the existing indexed MMVQ kernel.
  The old eight-column check treated expert activation rows as matrix batch
  columns and fell back to host route readback plus ten launches. Expansion is
  limited to contiguous Q2_0 per-expert inputs. Other formats retain the old
  limit: an initial broader experiment exposed different reduction orders and
  was narrowed before further measurement. Dense and sparse top-10 controls at
  640-by-64 and the actual 640-by-2560 shape match ten independent calls exactly
  on the 2080 SUPER, 3060 Ti and 3080.
- Replace the explicit F32-to-F64 conversion's zero/subnormal `ldexp` branch
  with PTX conversion, preserving the arithmetic. On every card an independent
  integer oracle checks all 16,777,216 signed zero/subnormal patterns plus
  16,777,216 spread normal/special patterns (NaN payloads excluded). All pass.
  Eighteen real IQ2_XXS/IQ1_M expert, graph-update/replay and cache regression
  runs pass across the three cards, including bitwise explicit CPU parity.
  The current 33-token logits, teacher values and routes remain byte-identical
  to the passing pre-conversion current-build control.

Unprofiled three-GPU raw-Hi/eight-token diagnostics used 4K context and 3 GiB
expert cache per GPU. Normal arithmetic repeats improved from 11.49 to
13.31/13.26 tok/s after the Q2_0 dispatch change. Explicit arithmetic repeats
improved from 5.58/5.60 to 7.09/7.40 tok/s after conversion. These small controls
are not the selected 552-token/128-generation launcher comparison and do not
meet the 80% requirement. Nsight traces were collected separately from those
measurements. Raw logs, commands, library hashes and retained failed attempts
are under `task4/qwen4exp-iq2-restore/`.

The first 1M admission attempt failed allocating GPU KV storage. Inspection
shows the memory planner replaced requested Q4 K/V with Q8, while the explicit
policy requires F32 indexer storage. A host-KV control also failed compute
reservation. Explicit Q4 admission is being isolated without that planner;
these failures do not qualify 1M support. Task 4 remains open, the production
loader gate remains OFF, and the installed launcher/publication hold is intact.

The explicit-Q4 retry allocates 6,912 MiB K/V plus 6,144 MiB F32 indexer on
the three cards, then fails compute reservation even after reducing the
microbatch to one. The existing QSA path gathers all logical K/V rows before
applying its sparse mask. Its full-history intermediates are the next memory
boundary to isolate. Failed reservations now log per-backend required/live/
budget bytes and rejection counters. Five focused conversion, indexed-vector,
planner and graph/shared-budget CTests pass on each of the three cards.


#### 1M Q4 admission and bounded attention gather

The restored runtime now reaches readiness with 1,048,576 context cells,
Q4_0 K/V and YaRN scale four, then completes two raw-Hi/eight-token requests
and exits zero. This is configured capacity and short-request serving, not
filled-1M model correctness or the selected 552/128 throughput qualification.
The production gate and installed launcher remain unchanged.

Failed-reservation diagnostics isolated approximately 5.2 GB of temporary
compute per GPU. The QSA graph gathered every logical K/V row into F32 before
applying its sparse mask. The explicit Qwen arithmetic path now ranks the
zero/-infinity mask by logical row and gathers at most 2,304 padded rows per
query (2,048 selected tokens plus at most 31 tail tokens). It retains every
visible logical row once, preserves visible row order, and retains masked
padding. Small histories and other architectures retain their existing path.
The GPU-KV server's reserved compute drops to 841.53 / 904.44 / 841.63 MiB.
Persistent K/V is 6,912 MiB Q4 plus 6,144 MiB F32 indexer storage.

The new million-row test also caught CUDA integer repeat reinterpreting I32
payloads as half values. NaN canonicalization corrupted IDs such as 31,750.
Integer repeat now copies uint32 payloads directly, including strided inputs,
and its support query advertises that implementation. Tests cover negative
IDs, half-NaN bit patterns, tiled dimensions and per-head million-row maps.

Dense/compact attention controls retain the fixed `2e-5 * (1 + abs(reference))`
bound for actual 256-wide / 24-query-head / two-KV-head geometry, using F32,
Q4_0 and F16 caches. Gathered F32 rows use ordinary explicit F32 KQ/softmax/VQ:
the existing CUDA Flash Attention dispatch could otherwise convert those rows
to half despite the precision request. F16 rows retain explicit F32 Flash
Attention. Eleven focused operator, cache, planner, expert and memory-budget
CTests pass on each of the three GPUs.

A host-KV option is being qualified to leave more GPU memory for experts.
Host-backed rows are gathered before their device handoff. An initial 6 GiB
expert-cache request without a reserve aborted with CUDA OOM at graph launch;
that failed trial is retained. A 4 GiB cache / 1 GiB reserve retry serves and
exits cleanly. Its small diagnostic repeat is only 3.60 tok/s, versus 6.02 for
the 1M GPU-KV control, and is not a performance acceptance result.

The first 33-token host/GPU Q4 placement comparison agreed through token 23,
then diverged: CPU IQK and CUDA Q4 encoders use different rounding/reduction
rules. The explicit host-Q4 path now encodes on the layer's GPU and transfers
the packed rows unchanged. Both placements then reproduce all 8,194,560
logits and 1,584 ordered route rows byte-for-byte at the same 1M allocation
(`q4-1m-host-gpu-prefill33-v3-comparison.json`). This is a placement control,
not an upstream Q4-cache or filled-context qualification. Failed controls,
commands, source/library snapshots and successful runs remain under
`task4/qwen4exp-iq2-restore/`.

Remaining work is ordered: retain the upstream F32-cache replay after this
integration; reduce empty host-KV memory residency; run the actual 552-token /
128-generation workload; profile and optimize its remaining bottleneck; then
repeat the complete matched comparison against the frozen 80% thresholds.
MTP, control-vector projection, prefix reuse and other launcher-specific
features remain separate unqualified differences. Task 4 and publication stay
open until the selected profile's requirements actually pass.

The final current-source F32-cache/4K upstream replay also remains
byte-identical to the passing pre-change teacher, logits and route controls
(`canonical-host-reference-33/bitwise-control-comparison.json`). The private
runtime snapshots and full selected-model inputs are preserved; no production
architecture admission or publication is implied by this checkpoint.


#### Empty host-KV residency

The same 1M host-KV server previously reached readiness at 18,248,409,088
RSS bytes, including 14,634,127,360 anonymous bytes, with no prompt submitted.
Large owned CPU allocations now use private anonymous mappings on Linux.
A zero clear discards their pages; subsequent reads still return zero. Nonzero
clears retain memset, and externally supplied/file-backed buffers retain their
existing clear/ownership interface. Unsupported mapping/discard operations
fall back to the prior allocation/clear behavior. Logical buffer sizes and
allocator budget accounting remain unchanged.

After this isolated change, the same readiness control uses 4,438,892,544 RSS
bytes and 823,869,440 anonymous bytes, with no swap. These are sampled readiness
measurements, not process-peak or throughput claims. Startup in these controls
is 10.02 versus 4.26 seconds; no storage-cold claim follows. The complete Q4
1M/33-token host placement replay remains byte-identical for all 8,194,560
logits, ordered routes and input IDs.

Five focused CPU clear, graph/shared-budget, handoff and compact-QSA CTests
pass. The new clear test writes the owned buffer directly before clearing,
checks zero/nonzero clears and its unaligned tail, verifies page release where
available, and proves that clearing an external private file mapping overwrites
its contents while preserving its underlying file and ownership. Evidence:
`host-kv-residency-before/`, `host-kv-residency-after/`, `lazy-host-prefill33/`
and `lazy-host-kv-tests.log` under the selected-model restoration directory.


#### Mixed host attention / GPU recurrent-state allocation

The private Qwen4Exp path can retain recurrent state on its layer GPU while
placing attention and indexer KV in host buffers (`ESE_QWEN4EXP_HOST_ATTENTION_KV=1`,
with KV offload otherwise enabled). The opt-in applies only to this architecture
and rejects graph/attention splitting. At 1M allocation, host payload is 13,056 MiB
and GPU recurrent payload totals 112.57 MiB. The 33-token sequential replay matches
the prior GPU-placement control byte-for-byte for every logit and ordered route.
Enabling a 4 GiB expert cache preserves that exact result. Focused fixture and
cache-sizing tests pass. Commands and artifacts are in `hybrid-kv-prefill33/`,
`hybrid-kv-cache33/` and `hybrid-kv-focused-tests.log`.

Changing GPU layer placement, or batching prefill, introduces small early
arithmetic differences which cross a Q4 rounding boundary at token 24 and then
fail the existing full-logit/route comparison. Those runs are retained as failed
controls (`hybrid-kv-placement33/`, `hybrid-kv-actual-batched33/`); the folder named
`hybrid-kv-launcher-batched33/` actually used sequential decoding because the
probe checks environment-variable presence. Its recorded config states this.
The first complete 552-token / 128-generation diagnostic therefore uses the
validated placement and serial prefill. It is a manual-budget measurement with
launcher feature differences, not speed acceptance.


#### Strata source comparison and draft ablation

The complete ESE serial diagnostic finishes both 552/128 requests and exits
cleanly: 3.39 then 4.67 tok/s, first-token times 370.47 and 327.75 seconds.
Its GPU expert cache is active; these results do not meet the speed requirement.
The fixture/memory success must not be reported as performance readiness.

Read-only inspection of the local Strata source and frozen engine logs finds:
quantized Q8 activation dot products; grouped CPU/GPU expert computation; a
profile-filled/adaptive GPU cache and a resident RAM complement; 32K resident
KV streaming; and overlapping speculative windows. The saved baseline's PCIe
probes are 0.7, 2.8 and 20.6 GB/s in selected device order, making its 6/14
layer split material. ESE's balanced placement spreads uploads across slow links.
Strata retains 24.24 GiB of complementary expert weights in pinned RAM and
reports zero expert blob file reads during these requests.

An isolated copy disables MTP, suffix/lookup drafts and pipelined windows.
A first attempt with `--spec 1` is rejected by the native-pack admission rule;
`--spec 2` without a drafter is the valid serial path. The three completed
requests have zero proposed/accepted drafts and measure 35.9 / 42.1 / 43.6 tok/s.
The original launcher profile is unchanged. Disabling MTP also frees its buffers
and changes cache capacity, and outputs vary; this is a diagnostic ablation,
not a pure fixed-memory causal estimate or replacement qualification baseline.
Evidence: `strata-no-spec-ablation/`, `strata-no-mtp-ablation/` and their copied
profiles/commands under the restoration directory.

Next, exercise native quantized experts behind an opt-in while retaining
explicit dense/router/indexer precision. Compare against the appropriate native
quantized operator controls; differences from the original-input/F64 reference
must remain explicit. First measure the base decode with the launcher's layer
placement and adequate RAM/GPU residency; add MTP only after that bottleneck is
understood. The original tight-reference path remains an independent control.


The isolated checked-dot optimization retains the original-input control.
A compensated F32 dot is accepted only when its conservative error enclosure
fits strictly within one F32 rounding cell; ambiguous and unsafe ranges fall
back to the prior F64 dot. Fused gate/up remains unchanged. All three GPUs pass
the conversion oracle and checked-dot cancellation/halfway/subnormal/extreme
controls (2,805 accepted and 4,363 fallback dots each, no bitwise errors). All
18 real IQ2_XXS/IQ1_M expert/graph/cache checks pass. The current 33-token upstream
replay retains byte-identical teacher, logits, routes and ordered token routes.

A separate same-command Hi/32-token, three-request A/B measures repeated decode
5.45 / 5.49 before and 7.32 / 7.29 tok/s after. First requests differ in file-cache
state and are not attributed as optimization gains. Outputs repeat within each
run; this small diagnostic does not meet or replace the launcher workload.
Artifacts: `checked-dot-property-tests.json`, `checked-dot-expert-regression/`,
`checked-dot-reference33/`, `checked-dot-speed-{baseline,checked}/`.


#### Physical reserve preflight and Strata-style runtime experiments

The native quantized expert opt-in with the launcher split, 16 GiB RAM cache
and 6 GiB requested GPU cache gives 10.58 / 10.53 tok/s on Hi/32 repeats.
Enabling native packed projections also exposes component preflight fallback.
These are experimental arithmetic modes, not tight-reference qualifications.
The original explicit policy is retained independently. No quantization error
is accepted by relaxing its numerical bound.

A 128 MiB prefill staging limit cannot hold this model's full expert layers.
Two lanes actually require 1,475 MiB/device. The corrected 1,536 MiB limit
enables GPU staging, but a 552/128 run still measures 2.76 / 3.53 tok/s because
the 3080 cache falls back. First-token times improve to 136.63 / 85.70 seconds.
A separate native-projection trial was stopped before a complete request when
its insufficient staging and host-memory pressure became clear; its abort and
failed receipt are retained.

An opt-in initial 32K compute reservation leaves the 1M KV capacity intact;
runtime graph construction continues to use actual full history. This allows
a 128-token prefill batch, with 529 / 601 / 596 MiB initial GPU graph buffers
versus the prior smaller batch's roughly 900 MiB. A first reservation probe
incorrectly retained a 1M reservation write head and aborted an assertion;
the write head now follows the reserved rows. The corrected probe retains
byte-identical 33-token Q4 logits/routes/IDs. It does not qualify a filled 1M
context. With the cache allocation defect still present, the full 128-batch
run improves first-token times to 41.54 / 39.40 seconds but measures only
2.60 / 2.62 tok/s: all three expert caches fall back. A metadata inspection
overlaps that diagnostic's first prefill, so its prefill timing is not an
isolated acceptance measurement.

Allocation diagnostics isolate physical reserve shortfalls of a few MiB when
logical cache capacity fills the free-minus-reserve allowance. Page rounding
and backend state consume more physical device memory than the reported
logical buffer sizes. Component preflight now releases an unpublished cache
and retries up to twice with a smaller capacity, covering the measured
shortfall plus a modest allocation margin. It never reduces the reserve,
never retries unrelated layout failures, and checks physical free memory
after completing the allocation before publishing it.

Cache/alias/replacement/rollback and top-10 vector CTests pass on all three
GPUs. A selected-model clamped-cache replay triggers a physical retry on
every card, retains all 8,194,560 logits, 1,584 ordered routes and input IDs
byte-for-byte, reports all three active GPU caches and zero forced fallbacks,
and drains leases at exit. Evidence: `cache-preflight-diagnostic-hi/`,
`cache-physical-retry-tests-*.log`, `cache-physical-retry-reference33/` and
private library/source identities. The full-workload retry is a separate
measurement; the 80% speed requirement remains open.


### Task 4 — active caches, repeat reserve failure, and resident placement

- `workload-1m-cache-fixed128` activates all three GPU caches and completes its
  first 552/128 request at 5.585 tok/s. The second request fails the physical
  reserve guard (1,459,159,040 bytes free versus 1,468,006,400 required).
  Subsequent workspace growth consumes headroom beyond initial preflight.
  Its failed report and first receipt are retained; this is not acceptance.
- Strata's saved H2D probes identify especially slow first and second GPU
  links. A diagnostic places all expert weights for layers 0–5 on the 2080
  SUPER and 6–12 on the 3060 Ti, then stages/caches the remaining 35 layers
  on the 3080 (`-ts 6,7,36`). A strict F32-cache 33-token placement control
  matches the all-CPU-expert control bit for bit across 8,194,560 logits,
  1,584 route records and input IDs. This validates placement for that
  arithmetic/control only, not native quantized arithmetic or full context.
- `workload-1m-resident13` uses native quantized expert/projection experiments,
  bounded initial graph reservation, mixed host attention/GPU recurrent KV,
  1M Q4 allocation, 128-token microbatches, 1 GiB pinned RAM cache, 2.5 GiB
  device cache, 1.5 GiB prefill staging and a 1,400 MiB physical reserve.
  Both 552/128 requests finish cleanly at 7.535 / 8.000 tok/s with exact
  self replay, zero forced expert fallbacks and zero sampled process swap.
  Prompt rates are 23.57 / 25.17 tok/s; TTFT is 23.44 / 21.94 seconds.
  This remains below 41.2 overall / 41.84 repeat tok/s. Routing uses original
  weights; the opt-in native activation arithmetic remains unqualified
  against the original-input numerical reference.
- Runtime counters show 32.32 GB of decode expert uploads across two requests
  and 9.20 seconds of RAM-cache storage reads overall, motivating direct
  mapped-source and grouped CPU/GPU execution diagnostics. The first direct
  mapped-source attempt rejects startup because load workers require a RAM
  cache. It is preserved separately and rerun with workers explicitly zero.

Commands, immutable library identities, strict placement comparisons, raw SSE,
resources and failed trials are retained beneath
`task4/qwen4exp-iq2-restore/`. Original launcher/model files and publication hold
remain unchanged. No current result meets the speed requirement.


### Task 4 — mapped sources, CPU waits and recurrent projection policy

- Direct mapped expert sources, without the additional bounded RAM-cache copy,
  complete the same resident-13 552/128 diagnostic at 9.942 / 10.146 tok/s and
  17.59 / 15.67 seconds TTFT. The harness `resident` label here suppresses its
  RAM-lease defaults only: the final 35 layers still use a 2.5 GiB GPU cache
  and mapped host tensors, so this is not full model residency.
- A separate short-request Nsight trace and CPU sampling run identify CPU
  OpenMP barrier spinning, CPU matrix operations and full-capacity KV metadata
  scans. Profiling is excluded from performance results. CPU sampling with
  explicit io_uring PLE initially fails source allocation; a retained second
  diagnostic uses PLE mmap and succeeds. This I/O change is diagnostic only.
- Native packed projection policy was being overwritten after construction
  in recurrent Q/K/V, gate and output projections. The opt-in policy now
  survives those overrides; BF16/F16/F32 weights and default strict arithmetic
  keep their previous precision. The strict 4K/F32 placement control remains
  bit identical across 8,194,560 logits, 1,584 routes and all input IDs.
- With that correction, one decode worker completes 552/128 twice at 7.917 /
  8.087 tok/s (20.79 / 9.40 seconds TTFT). Four decode workers, 16 batch
  workers, `OMP_WAIT_POLICY=PASSIVE` and `GOMP_SPINCOUNT=0` measure 10.664 /
  10.737 tok/s (11.27 / 9.50 seconds TTFT). These combined changes are
  configuration diagnostics; no isolated causal speedup is claimed for the
  recurrent arithmetic change. Both remain below the agreed thresholds.
- New ordered-Qwen metadata bounds stop after visiting all occupied physical
  cells, retaining holes, shared owners and distant/wrapped/restored rows.
  Slot chronology, maximum extent, sequence position queries and token counts
  avoid scanning the empty tail of a 1M allocation for each short decode.
  Ordinary architectures keep their existing scan extent. Three focused
  schema/cache-sizing/attention-layout CTests pass, including new occupancy
  cases. The strict Q4/1M 33-token replay also remains bit identical across all
  8,194,560 logits, 1,584 routes and IDs; full workload and lifecycle checks
  follow before this change is considered complete.

`native-delta-runtime-identity.json` and `occupied-kv-runtime-identity.json`
record immutable private libraries and source diffs; each workload preserves
its exact argv/environment and raw receipts. The native arithmetic and bounded
initial graph reservation remain explicit experiments. Neither 1M allocation
nor a short exact replay qualifies filled-1M context. Publication remains held.


The occupied-cell workload completes at 11.799 / 11.787 tok/s with exactly the
same 128-token output hashes as the preceding passive-four-worker runtime.
Repeat TTFT is 9.10 seconds. The additional strict lifecycle control passes
shift, divide and partial-shift full snapshots and sequence migration exactly.
Its shared-prefix transform check fails because the other sequence snapshot
changes. The identical command on the immutable pre-optimization runtime
fails at the same check, establishing an existing experimental-Qwen lifecycle
blocker rather than a regression introduced by occupied-cell scans. Both
failures are retained (`occupied-kv-position-lifecycle` and
`pre-occupied-kv-position-lifecycle`); shared-prefix isolation is not qualified.

A one-token scheduler dump identifies host K/V gathers pulling ordinary
KQ/softmax/VQ attention onto the CPU. The next opt-in experiment explicitly
places those computations on each layer GPU while retaining bounded host row
gathers. It has not yet been built or qualified in this entry. All speed and
publication holds remain active.


### Task 4 — GPU attention and native top-10 CUDA capture

- Explicit layer-GPU placement of KQ, softmax, VQ and their merged output
  retains host row gathering. With the opt-in disabled, behavior is unchanged.
  `gpu-attention-strict-reference33` reproduces all 8,194,560 logits, 1,584
  routes and IDs exactly against the strict Q4/1M occupied-cell control.
- `workload-1m-gpu-attention-passive4` completes 552/128 twice at 12.535 /
  13.672 tok/s and 12.06 / 8.25 seconds TTFT. Outputs replay exactly within
  this runtime, but their hash differs from the CPU-attention full workload.
  The short strict replay does not qualify the longer native-policy output.
  This remains well below the agreed speed thresholds.
- A separate short Nsight run identifies substantial launch/synchronization
  overhead. In its approximate final-request window, BF16-to-F32 weight
  conversion consumes 130.7 ms across 4,096 kernels; explicit BF16 projections
  consume 78.4 ms. The runtime makes 37,457 individual kernel launch calls and
  4,720 stream synchronizations. These are trace diagnostics, not acceptance
  timings. Commands and the window query result are retained in
  `nsight-gpu-attention/`.
- An old indexed-node capture restriction rejects top-10 routes even though
  the recently optimized single-token Q2_0 path now routes entirely on-device.
  An opt-in capture exception checks that exact geometry, input contiguity,
  types, original precision policy, CUDA buffer ownership and same-device
  operands. Other indexed geometries keep the old capture restriction.
- The top-10 oracle now changes both inputs and IDs through six replays for
  each of two output dimensions (including the actual 640→2560 projection).
  All twelve rounds pass exactly against independent single-route calls on
  each of the 2080 SUPER, 3060 Ti and 3080. Sparse negative IDs also remain
  exactly zero. Each backend reports an instantiated cached CUDA graph.
  The matched full-model workload follows these checks separately.

Runtime identities, source diffs, raw SSE and commands remain under
`task4/qwen4exp-iq2-restore/`. Native activation arithmetic, GPU attention and
indexed capture remain explicit experiments. No result qualifies release.


The initial Q2_0-only capture diagnostic completes 552/128 twice at 13.035 /
13.773 tok/s and preserves both output hashes from the preceding GPU-attention
runtime. Its first TTFT is 21.73 seconds and repeat TTFT 8.37 seconds. The small
speed difference is not a qualification result. Inspection finds a second
legacy restriction on the broadcast IQ up/gate nodes, which can disable the
same graph before reaching its Q2_0 down projection. The next bounded capture
exception covers only the selected IQ2_S/IQ2_XXS/IQ1_M broadcast inputs, with
matching real-shape replay controls; it does not change their arithmetic or
extend the per-route activation dispatch to those formats.


The extended capture controls pass on all three GPUs: 30 changing-input/route
rounds are byte identical to the same indexed operations on a second backend
with CUDA graphs disabled, and all 12 Q2_0 rounds additionally retain exact
single-route parity. Initial IQ up/gate comparisons to a differently tiled
single-route operation fail both with capture enabled and disabled; those
controls are retained. The capture oracle consequently compares the same
pre-existing indexed geometry with/without capture, without changing the
Q2_0 oracle or any model-reference tolerance.

`workload-1m-capture-native-passive4` then completes 552/128 twice at 13.569 /
13.940 tok/s, with 10.38 / 8.30 seconds TTFT. Both output hashes match the
preceding GPU-attention and Q2_0-only capture workloads exactly. The effect is
modest, and the 41.2 / 41.84 tok/s thresholds remain unmet. The bounded capture
exception stays opt-in (`ESE_CUDA_CAPTURE_INDEXED_MMVQ=1`). The selected native
arithmetic and long-context policies still require independent qualification.


### Task 4 — direct BF16 gated-residual experiment (NOT NUMERICALLY QUALIFIED)

- An opt-in direct projection (`ESE_QWEN4EXP_NATIVE_GR=1`) reads original BF16
  GR weights and F32 inputs directly, with native F32 accumulation. It is
  limited to selected GR matrix shapes/names, contiguous tensors, 1–8 input
  columns and same-device CUDA storage. Default precision behavior remains
  unchanged; larger prefill batches retain their previous implementation.
- Operator controls cover 10240→320, 320→10240 and 10240→4 at B1/B5/B8 on
  all three cards. They pass the fixed `2e-5*(1+abs(reference))` GR bound
  against F32 calculations and sampled independent double dots. Existing
  head controls retain their previous bound; maximum observed combined error
  is 8.59e-6. No tolerance is relaxed.
- The isolated 33-token strict model comparison FAILS: 2,229,381 of 8,194,560
  logits exceed the original bound, maximum absolute difference 0.77685,
  and routes differ. Tokens 0–23 remain within the fixed bound; the first
  route-order difference is token 24, layer 23. All 33 argmax IDs agree.
  `native-gr-strict-reference33` retains all logits/routes and per-token
  divergence, rather than treating finite output or argmax agreement as a
  numerical pass. This path remains experimental and must not replace the
  strict arithmetic control.
- A separately labeled speed diagnostic completes 552/128 twice at 14.507 /
  15.620 tok/s, with 11.16 / 8.92 seconds TTFT. Native-policy output repeats
  exactly within that run, but changes from the preceding implementation.
  The repeat reads zero storage bytes and samples zero process swap. These
  are potential-speed measurements only, not qualification. They remain
  below 41.2 overall / 41.84 repeat tok/s even without the numerical blocker.

`native-gr-operator-tests.*`, `native-gr-runtime-identity.json` and
`workload-1m-native-gr-passive4/` preserve commands, binaries, failures, raw SSE
and resources. Further work must address numerical qualification and remaining
host launch/synchronization costs; no supported behavior or release is claimed.


### Task 4 — graph replay diagnosis and bounded CPU upload ownership

`graph-update-diagnostic` uses a bounded opt-in update/rejection logger
(`ESE_CUDA_GRAPH_UPDATE_TRACE=1`). No updates/rejections are emitted in the
short Hi run: all 11 / 18 / 214 cached graphs have executable instances at
shutdown. The suspicion that repeated updates disable the warm graphs is not
supported by this run.

`nsight-capture-native-mmap` collects CUDA and process CPU samples after the
selected native capture improvement. The async PLE attempt fails because
io_uring cannot allocate under the profiler; its original failed trace is
retained in `nsight-capture-native`. Only the diagnostic retry uses mmap PLE.
These short traced requests are excluded from speed qualification. In the
approximate last request window, individual kernel launches drop from 37,457
in the preceding uncaptured trace to 191, while 1,952 graph launches remain.
The window has 4,720 stream synchronizations (639 ms API time) and 8,907
asynchronous memcpy calls (222 ms API time). GPU kernel/copy intervals and
host API times overlap and must not be added as independent elapsed costs.
The prior graph-capture change addressed most individual launch overhead;
CPU/GPU boundary ordering and transfer ownership remain significant.

An opt-in CPU->CUDA copy path (`ESE_CUDA_STAGED_CPU_UPLOADS=1`) takes an
immediate CPU copy into backend-owned pinned slots, then queues H2D on the
destination CUDA stream. Four 8 MiB slots bound staging to 32 MiB per CUDA
backend context. Per-slot completion events prevent reuse before transfer
completion; backend teardown drains pending copies. Other backend pairs,
unsupported storage, copies larger than 8 MiB, active capture and failed
pinned allocation retain the existing fallback. No arithmetic is changed.
This addresses the generic CPU->CUDA fallback's destination-wide host wait
without retaining pointers into mutable CPU tensor storage.

`test-cuda-staged-upload` passes on all three cards: 288 staged copies and
72 oversize fallback copies, with pageable and pinned sources, immediate
source overwrite/release, repeated four-slot wrap, and three pending-copy
backend teardown cases. `staged-upload-strict-reference33` then matches
`gpu-attention-strict-reference33` bitwise for all 8,194,560 logits, all
ordered routes and input IDs; zero fixed-bound violations.

The failed native GR experiment is removed from working source after its
runtime, complete source diff, operator tests, model-reference failure and
potential-speed results were archived. Its 15.62 tok/s result remains an
unqualified experiment, not the current implementation. The staging
experiment and subsequent workloads keep that GR path disabled.


`workload-1m-staged-upload-passive4` completes both full 552/128 requests
with the preceding output hash unchanged, but slows to 12.556 / 13.009 tok/s
(TTFT 19.08 / 8.83 s) versus 13.569 / 13.940. This is no demonstrated speed
improvement. The staging implementation and its fixture are shelved from
working source; immutable runtime, complete patch and test source remain in
the result directory. The bounded graph diagnostics remain available. The
next isolated experiment pins a capped prefix of existing mapped CPU model
buffers, following Strata's resident pinned-source design without a second
large RAM copy. It keeps a 6 GiB host-availability reserve and is private,
explicitly opt-in, limited to 32 GiB and Qwen4exp.


`mapped-pin24-hi-diagnostic` reports CUDA `operation not supported` when
registering its read-only mapping; zero bytes were registered and it falls
back cleanly. Device attribute 113 (`HostRegisterReadOnlySupported`) is zero
on all three GPUs. The installed CUDA 12.8 headers document that this flag
requires that attribute; the mapped-source experiment is removed after its
source patch/runtime and failed-registration logs are archived. Its short
17.45 / 17.67 tok/s repeats are ordinary fallback diagnostics, not pinned
source results or speed qualification. Strata's `PinnedArena` owns writable
storage. ESE's existing writable pinned RAM arena is therefore the next
comparison, with a 32 GiB experimental maximum and an explicit 6 GiB host
availability check for capacities above its prior 1 GiB limit.


### Task 4 — resident pinned expert arena and first-fit fill cost

`workload-1m-pinned32-passive4` uses the existing writable expert arena with
32 GiB pinned capacity and frequency retention. It completes 552/128 twice
at 8.568 / 15.001 tok/s; output hashes match the preceding native workloads
exactly. Startup takes 32.93 seconds and TTFT is 60.46 / 7.23 seconds. The
first request reads 29.0 GB of storage and the repeat 114 MB. Only 18.13 GB
(16.89 GiB) of expert payload becomes resident, with zero RAM evictions.
The excessive arena capacity displaces useful file cache: process swap peaks
at 8.57 GB during startup/population and finishes near 150 MB. A 32 GiB
allocation is therefore not selected as the practical profile despite the
repeat improvement. This remains below 41.2 / 41.84 tok/s.

The RAM allocation planner previously walked the entire ordered resident
index for every admission, including dense initial fills. A new shortcut
uses the existing resident-byte total and final occupied end to recognize a
fully occupied prefix, merges only the bounded pending reservations, and
appends only when no earlier fit exists. Holes, partially published entries,
eviction and ambiguous reservations retain the original planner. A 1024-entry
fixture checks exact payload/placement and then requires reuse of an earlier
erased hole despite ample unused tail capacity. The full expert-cache suite
passes on the RTX 3080 and again with all three selected GPUs visible,
including concurrent fills, lease lifetime, failure recovery and multi-GPU
transaction publication. Logs are `dense-arena-cache-tests*.log`.

`pinned20-dense-strict-reference33` uses a 20 GiB pinned arena and the new
planner. It matches all 8,194,560 reference logits, ordered routes and input
IDs bitwise against `gpu-attention-strict-reference33`, with zero fixed-bound
violations. The following full workload uses 20 GiB, enough for the observed
16.89 GiB expert set with additional capacity, while leaving more host memory
for the model's file-backed PLE and other data. Private runtime hashes and
source differences remain in `pinned20-dense-runtime-identity.json`.


`workload-1m-pinned20-dense-passive4` completes 552/128 twice at 11.819 /
15.213 tok/s with unchanged native-policy output hashes. Startup takes 22.80
seconds; TTFT is 44.48 / 6.49 seconds. The first request reads 28.14 GB of
storage and the repeat reads zero bytes. Every sampled process-swap value is
zero, improving the memory behavior over the excessive 32 GiB arena. This
combines a smaller arena and the dense-fill shortcut, so these two effects
are not individually attributed. It remains a one-start diagnostic and is
not the three-start/nine-request qualification. The speed thresholds remain
unmet, and native arithmetic, long-context behavior, model features and
publication are still held.

The CUDA trace now points to the next architectural comparison: Strata keeps
a 32K KV window resident on GPU while the ESE 1M profile keeps ordinary
attention/indexer KV on the host. GPU attention alone still requires host
cache encoding/gathers and transfer boundaries. Further work must preserve
full logical history and lifecycle correctness when introducing any resident
window; reducing the logical cache to 32K would not qualify the 1M profile.


### Task 4 — packed host attention-cache handoff (in qualification)

A new opt-in `ESE_QWEN4EXP_PACKED_HOST_KV=1` path gathers selected Q4_0
rows byte-for-byte on CPU, transfers the packed result, and expands to F32 on
the layer's CUDA backend. It requires the existing explicit arithmetic and
KQV offload policy. Unsupported types/backends retain the preceding gather.
Persistent storage, logical context, chronological selection and cache
ownership are unchanged. Both ordered and selection-limited attention paths
use the helper. Packed rows occupy 18 bytes per 32 values versus 128 bytes
for F32, reducing this handoff's data volume by 7.11 times.

`test-qwen4exp-cache-order` now covers 16 packed host-handoff geometries x
three replays on each of the three cards and CPU. It checks stored Q4 bytes
and expanded F32 bits against the existing CPU F32 gather, including padded
and interleaved heads/rows, nonzero block offsets, strided IDs, changed data
and IDs, repeated rows and 2304-row compaction geometry. All pass, as do
existing gather fixtures. The initial CPU fixture incorrectly created two
CPU backend contexts and failed its placement assertion; the corrected CPU
fixture aliases one backend. Failure and corrected logs are retained, and
`packed-host-gather-tests.json` identifies the valid evidence explicitly.

`packed-host-strict-reference33` matches all logits/routes/IDs bitwise at
ubatch 16. The first full ubatch-128 workload fails at constructor reservation
with the existing `GGML_SCHED_MAX_SPLITS` assertion, before requests run.
`workload-1m-packed-host-passive4` preserves that failure. The implementation
is revised to expand the packed CPU gathers into the graph ahead of device
conversion/attention, so K and V gathers can share a handoff. This keeps the
scheduler limit fixed and F32 scratch per query. The revised ubatch-128
strict control constructs 43,666 nodes and 3,246 splits successfully;
its final numerical check and full workload follow.


The revised `packed-host-grouped-strict-reference33` uses ubatch 128 and
matches all 8,194,560 logits, routes and input IDs bitwise. The matching full
`workload-1m-packed-host-grouped-passive4` then completes 552/128 twice at
12.491 / 16.364 tok/s (TTFT 44.31 / 6.88 seconds), with the preceding native
output hash unchanged. Startup is 16.95 seconds. The repeat is 7.6% above the
preceding 15.213 tok/s diagnostic, but this remains one start/two requests,
not the three-start qualification. The full model prompt has fewer than
2304 keys: larger selection-limited model execution and long-context
lifecycle remain unqualified, despite the 2304-row operator fixtures and
successful worst-case constructor reservation. Keep the path opt-in and the
release gate closed. The 41.2 / 41.84 tok/s thresholds are still unmet.


### Task 4 — Q4 expansion launch geometry

`nsight-packed-host-fullprompt` traces three frozen 552-token requests with
32 generated tokens, separately from all builds. Its approximate final
decode window excludes prompt/startup time (last GPU-kernel end minus engine
prediction time). The window contains 31 actual GPU decode steps for the
32 output tokens. BF16-to-F32 conversion consumes 253 ms of summed kernel
time, small explicit BF16 projections 150 ms, and Q4-to-F32 copies 205 ms
across 744 copies. API waits overlap GPU work and are not additional costs.
Raw traces, reports and `warm-decode-summary.json` are retained; traced
throughput is excluded from qualification.

The Q4 copy launch uses `num_blocks = ne` and one thread per CUDA block,
while each thread handles QK4_0=32 values and immediately returns beyond
`ne`. This dispatch launches roughly 32 times too many single-thread blocks.
The revised launch uses 128 threads and ceil((ne/32)/128) blocks. The exact
same per-thread packed-block routine, strides, output arithmetic, graph-copy
function identity and destination-pointer indirection are retained. No
precision or capture guard is relaxed.

The cache-order suite now also changes a captured CPY destination within its
existing allocation across three replays. Tests cover width 32 / one row
(partial launch), width 256 / 17 and 2304 rows / two heads, padded/interleaved
source and destination strides, exact independent CPU Q4 dequantization and
untouched guards. These pass on CPU and all three cards, together with all
packed-handoff and preceding gather cases (`q4-launch-copy-tests*.log/json`).
The strict ubatch-128 / 1M model control and full workload follow.


`q4-copy-launch-strict-reference33` matches all 8,194,560 logits, routes
and IDs bitwise with the ubatch-128 / 1M reference. The separately timed
`workload-1m-q4-copy-launch-passive4` completes 552/128 twice at 13.630 /
18.408 tok/s, versus the preceding 12.491 / 16.364. Both output hashes remain
unchanged. This remains a one-start/two-request diagnostic; the three-start
qualification and speed thresholds remain open. The launch fix preserves
existing precision and applies to Q4_0->F32 copies without an experiment
flag; the host-cache handoff and native model policies remain opt-in.


### Task 4 — Strata-style native CPU expert arithmetic (experimental)

The selected model's four real expert geometries were calibrated separately
with four CPU threads and all three devices. The original Q2_0 down dot uses
scalar F32 activations: its CUDA2-contended cost is 309,866 ns per component,
versus 40,470–54,995 ns for each up/gate component. Strata uses integer Q8
activation dots for these weights. The new process-start opt-in
`ESE_CPU_Q2_0_Q8_DOT=1` uses Q8_0 activations and an AVX2 Q2/Q8 dot, retaining
the original F32 dot by default. CPU execution and workspace planning consult
the same activation policy. This changes native arithmetic and does not
establish full-model numerical qualification.

The integer dot passes 5,376 independent packed-code/extreme/width cases,
including -128 signed activations. Indexed-matrix fixtures cover actual
640->2560 down geometry, broadcast/per-route input, masked routes, changed
inputs/routes, one/four threads, and two-expert bounded leases with poisoned
backing storage. Native and default policies pass normal and ASan/UBSan
builds. The initial fixture incorrectly repeated experts within one token's
route list; AddressSanitizer localized an existing generic route-mapping
overflow under that geometry, also present with the default dot. The
corrected fixture uses unique top-k routes as produced by the model. Raw
failure and corrected logs are retained. Duplicate-per-token native route
inputs remain outside this validated fixture and the generic mapper still
assumes at most one occurrence per expert/token; no claim of general
repeated-ID safety is made here.

Actual selected-model Q2 weights pass the existing dequantized scalar
reference (NRMSE 0.003646, unchanged 0.08 calibration limit). The first
21-iteration opt-in calibration is not planner-ready because one contention
confidence is below the existing cutoff; it is not promoted. A separate
49-iteration run passes all existing confidence checks and is planner-ready.
Its CUDA2 Q2 cost is 51,155 ns, about 6.1 times faster than the preceding
309,866 ns diagnostic. Both calibration files remain in the evidence folder.
Transfer costs include a bounded RAM refill, not a resident pinned-hit
measurement. They must not be represented as pinned-cache-hit costs.

Hybrid graph construction now requires all expert weight components to be
host-backed before splitting a layer. Fully GPU-resident layers remain on
GPU; a CPU branch there would copy whole expert tensors back to the host.
A four-GPU/six-CPU route diagnostic uses residency-aware selection and early
CPU input transfer, with conservative whole-expert costs derived from the
49-iteration CUDA2 measurements and unchanged live drift/fallback guards.
Its full workload result follows. The opt-in CPU policy with explicit F32
experts still matches all 8,194,560 strict-reference logits, routes and IDs
bitwise at ubatch 128 / 1M context (`native-q2-cpu-strict-reference33`). This
control bypasses native expert arithmetic and does not qualify it.

The first `workload-1m-native-q2-hybrid4-passive4` is excluded: the existing
upload-drift guard revoked hybrid execution after 70 CPU layer calls, with
16.25 ms of lease acquisition across seven misses. Requests reach 10.484 /
16.679 tok/s and have different hashes as execution changes after revocation.
The warm-only upload estimate does not cover cold storage refills.

A second diagnostic uses the actual cold-refill component medians from the
same 49-iteration calibration (CUDA2 0.835–1.015 ms; each confidence >0.99).
The conservative maximum whole-expert refill cost is 2,950,006 ns. The drift
multiplier remains 4x. It assigns eight expert positions to GPU and two to
CPU, with active four-thread OpenMP to avoid passive worker wakeups during
CPU expert computation. This run stays within the guard, reports 8,890 CPU
layer calls (35 host-backed layers x 127 decode steps x two requests), and
has no forced fallback. Its 12.550 / 15.202 tok/s is slower than the preceding
GPU-only diagnostic, and the requests' output hashes differ. The runner
correctly marks both mixed runs failed. No change to output-repeatability or
numerical acceptance is made. Keep the native CPU option off in the selected
profile and retain the prior 18.408 tok/s GPU diagnostic as the best tested
repeat. CPU dot acceleration alone does not establish useful overlap or
full-model qualification. Task 4 and the release gate remain open.


A further cache-routing experiment keeps every complete cached expert on
its actual output GPU, beyond the configured fixed-position minimum. Mask
fixtures pass for complete hits, missing gate components, wrong layouts,
wrong devices, padding and exclusive/exhaustive CPU/GPU routes. The full
`workload-1m-all-resident-hits-hybrid8-active4` nevertheless revokes on the
unchanged CPU-drift guard after 70 CPU layer calls (85.996 ms versus the
calibrated 4x limit of 82.653 ms). Its 13.636 / 17.146 tok/s and differing
hashes do not qualify. The patch/test/runtime are archived outside the repo;
the working source is restored to the committed fixed-budget implementation.
This policy needs more useful CPU/GPU overlap before adoption. The selected
runtime links return to the preceding best Q4 launch implementation.


### Task 4 — aligned BF16-to-F32 conversion

The CUDA trace identifies repeated BF16-to-F32 weight expansion as a large
summed kernel cost. The aligned F32 destination path now moves four adjacent
BF16 values per thread with uint2/float4 loads/stores. Each value still uses
the original `__bfloat162float` instruction. Unaligned inputs/destinations,
other destination types and other CUDA-compatible vendors retain the scalar
path; short tails use the scalar instruction without reading past the end.

`test-bf16-convert` passes on all three cards: eleven shapes, all four
source/destination alignment combinations, three changed-input graph
replays, output guards, every BF16 bit pattern (including NaNs), exact scalar
CUDA comparison and an independent finite-value bit oracle. A 3,276,800-value
component diagnostic reports scalar/selected 49.158/49.257 us on the 2080,
52.274/50.053 us on the 3060 Ti, and 40.396/30.884 us on the 3080. These are
component timings, not end-to-end speed evidence. The separate ubatch-128 /
1M strict model control matches all 8,194,560 logits, routes and IDs bitwise.

The unchanged GPU-only 552/128 workload completes at 13.941 / 18.417 tok/s,
with both preceding native output hashes unchanged. This is effectively
neutral versus 13.630 / 18.408 for the preceding run; no end-to-end speedup
is claimed. CPU Q2/hybrid/all-hit options remain off. Evidence is retained as
`bf16-vector-*` and `workload-1m-bf16-vector-passive4`. The 41.2 / 41.84 tok/s
thresholds remain unmet. GPU conversion optimization alone does not remove
the host scheduling/transfer boundaries identified in the trace.


### Task 4 — existing autotuner and launcher layer-boundary diagnostics

The existing `ESE_CUDA_EXPLICIT_AUTOTUNE=1` option chooses 1/2/4 warps for
small explicit BF16 projections after its unchanged exactness checks.
`workload-1m-bf16-vector-autotune-passive4` completes at 13.435 / 18.694 tok/s
with both original native hashes unchanged. This is only a 1.5% repeat gain
in one start/two requests, not qualification or a resolved bottleneck. A
separate strict model control is still required before adopting the option
in a selected profile.

`workload-1m-resident14-bf16-vector-passive4` tests ratios 6,8,35 with the CPU
expert override beginning at layer 14, approaching the selected Strata
6/14 boundaries. CUDA1 model storage increases from 5,117.28 to 5,812.25 MiB
and CUDA2 decreases from 2,430.37 to 2,310.41 MiB; startup and both requests
finish without exhausting the existing reserves. It reaches 14.319 /
18.241 tok/s, below the preceding 13-resident-layer diagnostic. Its two
hashes agree with each other but differ from the 13-layer native output.
It is not promoted: neither a speed benefit nor native arithmetic parity
for the changed placement is established. No launcher file is changed.

`workload-1m-bf16-vector-passive1` retains 16 prompt threads but uses one
decode CPU thread, after packed host KV/gathers reduce host work. It completes
at 14.340 / 18.901 tok/s with both original native hashes unchanged. The
repeat is 2.6% above the four-thread vector diagnostic; this is one start/two
requests, not qualification. A short read-only SQLite summary of an older
trace ran during this diagnostic's first prompt; first-prompt latency is
therefore not a clean comparison. It did not overlap either timed decode
window or the repeat prompt. Further tracing uses a separate run.


### Task 4 — fresh CPU/GPU trace and graph-validation allocation work

`nsight-bf16-vector-fullprompt-cpu` traces three frozen 552-token requests
with 32 generated tokens, one decode CPU thread and the latest packed-Q4 /
vector-BF16 runtime. PLE uses mmap to avoid the previously observed sampler /
io_uring allocation failure; this is not the async benchmark profile. The
approximate final decode window is 2,077 ms and contains 31 GPU decode steps.
API and GPU times overlap and must not be added as elapsed time. The window
has 9,145 stream synchronizations, 3,782 graph launches, 17,922 H2D copies
(5.838 GB), and 2,416 D2H copies (65 MB). Summed GPU BF16 conversion is
234 ms; small explicit BF16 projections 152 ms; Q2 down projections 96 ms.
The main CUDA host thread has 1,857 CPU samples; graph validation owns 61
self samples and route preparation 29. Full SQL summaries, raw trace and
CPU samples are retained. Traced throughput is excluded from qualification.

Graph validation now compares saved layouts directly with current tensor
metadata and refreshes snapshots only when they change. Temporary allocation
ID vector capacity is reused per host thread, and the saved binding vector
is copied only on changes. Every allocation ID, source presence/type/buffer,
shape/stride, pointer, precision and operation parameter check remains in
place. Stable copy/view destination updates retain their existing indirection.
No graph-validity or eviction guard is disabled.

Graph leaf layout/type/stride and pointer mutation tests, packed Q4 copy /
gather replay, and actual IQ2_XXS/IQ1_M plus Q2 precision-mode switch fixtures
pass on each of the three cards (12 commands). The full expert-cache suite
also passes. A separate ubatch-128 / 1M 33-token model control includes the
existing projection autotuner and one decode thread and matches all
8,194,560 strict-reference logits, routes and IDs bitwise. As before, this
F32 control does not qualify native quantized expert arithmetic.

The isolated GPU-only `workload-1m-graph-guard-passive1` completes at 14.398 /
18.755 tok/s with both original native output hashes unchanged, versus the
preceding 14.340 / 18.901. Treat the full-workload timing as neutral: no
end-to-end speedup is claimed. Task 4 and the 80% speed gate remain open.

The existing GPU `ESE_EXPERT_CACHE_RETENTION=frequency` option is tested
separately from the already-selected RAM frequency policy. Its unchanged
552/128 requests complete at 13.947 / 18.099 tok/s with the same native hashes,
below the preceding LRU diagnostic. It is not adopted. Testing additional
GPU cache capacity follows, preserving the existing 1400 MiB physical reserve
and all arithmetic/context settings.


The requested 3328 MiB GPU cache diagnostic completes at 14.712 / 19.269
tok/s with unchanged native hashes. Combining that request with the already
validated autotuner completes three 552/128 requests at 13.968 / 19.841 /
19.664 tok/s. All hashes remain identical. The runtime admits 3,327,270,912
bytes (about 3.10 GiB), below the requested 3.25 GiB, under its existing
physical reserve clamp. Peak sampled 3080 usage is 8462 MiB, leaving 1778 MiB
at one-second sample points; no process swap is sampled. This does not prove
an instantaneous free-memory minimum. The 1400 MiB reserve and all live
guards remain enabled. These are one-start diagnostics and do not satisfy
the three-start qualification or the 80% speed requirement.


### Task 4 — duplicate native CPU route mapping safety

The earlier ASan failure exposed a real generic-operator bug, rather than a
public API prohibition on repeated routes. Model top-k lists are unique, but
`ggml_mul_mat_id` also accepts repeated expert IDs. The old native CPU mapper
reserved only one row per expert/token, so repeated IDs could overwrite the
next expert's mappings or the heap. This affects the default scalar path as
well as the opt-in Q2/Q8 path. The earlier unique-top-k fixture did not cover
that accepted generic geometry; this follow-up closes that gap.

Both native indexed matmul and fused up/gate now count routes, prefix-sum
per-expert offsets, and store one mapping per valid route. Expert grouping
and within-expert token/slot order are preserved. Workspace planning reserves
counts, offsets and the full route count, aligning the activation region
once rather than doubling it. Mapping storage scales with actual routed
positions instead of experts x tokens. For 512 experts / 10 routes / 128
tokens, mapping metadata is 18,432 bytes versus the prior 528,384 bytes
(excluding activation storage). No expert, route, or numerical threshold is
removed to make the case fit.

The Q2 native/default regressions now include the original repeated-ID
geometry, three-token batches, changed routes/inputs, negative and positive
out-of-range masks, broadcast/per-route input, one/four threads, and bounded
leases with poisoned backing storage. They pass normal and ASan/UBSan
builds. A separate self-contained IQ4_XS fused regression covers separate and
merged gate/up, constant-weight analytical outputs at the unchanged
2e-5*(1+abs(reference)) bound, repeated/masked/batched routes, changed inputs
and exact thread parity. It passes both builds. The expert-cache suite also
passes in the ASan CPU build and in the CUDA build with all three cards
visible. Evidence: `compact-route-*`; the original ASan report remains
available. The separate strict 33-token / 1M model control matches all
8,194,560 logits, routes and IDs bitwise with the preceding control.


### Task 4 — user acceptance of the current speed

The user accepted approximately 19 tok/s for this version and deferred further
speed improvement to later updates. The previous 80%-of-Strata criterion is
superseded for this version, rather than reported as achieved. The retained
best configuration and all three raw samples remain under
`task4/qwen4exp-iq2-restore/`: command
`run-workload-graph-guard-vram3328-autotune-passive1.json`, measurements
`workload-1m-graph-guard-vram3328-autotune-passive1/`, resource summary
`best-cache-autotune-diagnostic-summary.json`. Task 4's speed decision is
resolved by explicit user acceptance. Stop performance experiments and advance
to task 5's correctness, lifecycle and actual filled-context checks. Production
Qwen admission, package/install checks and publication remain independent.


### Task 5 — YaRN position transforms preserve other sessions

Before larger-context qualification, the selected model's shared-position
probe reproduced corruption of the untouched sequence at YaRN scale four.
The before/after snapshots differ in 37,845 bytes, confined to the first key
payload region. Sampled nonzero F32 keys are multiplied by 1.13862943649, the
YaRN attention gain for this scale. The shift graph was applying this gain
again, including rows with zero position delta. Stored keys already contain
their original magnitude; a delta rotation must preserve it.

Qwen4Exp's shift graph now cancels YaRN's internal gain when extrapolation is
enabled and uses unit magnitude when disabled. Other architectures retain
their existing shift factors. The lifecycle diagnostic saves both snapshots
before asserting equality, preserving failure evidence rather than losing it.

On all three selected GPUs together, strict F32 expert/projection-policy runs
pass with prefixes 8 and 33, plus prefix 33 with Q4_0 K/V and Flash Attention.
Each run passes 14 full-vocabulary continuation comparisons with exact logits
and ordered MoE routes: full/sequence migration after shift/divide/partial
shift, private/shared selected-owner equivalence, untouched-owner continuation,
and overlapping prefix copies. All three untouched-owner sequence snapshots
are byte-identical before/after their transformations in every run. Prefix 33
crosses the first compressed-attention block. The original unshifted prefix-8
snapshot also remains byte-identical to the pre-fix runtime. Six focused native
state/cache/production-rejection CTests pass after rebuilding their targets.

Evidence: `task4/qwen4exp-iq2-restore/unit-shift-validation-summary.json`,
`unit-shift-focused-{build,ctest}.log`, `shared-position-before-unit-rotation/`,
and the three `shared-position-unit-rotation*` passing directories. The initial
Q4 test with Flash Attention off rejects before model load because quantized V
requires it; that failed configuration is retained and the corrected FA-on run
passes. This is bounded lifecycle validation at scale four/4K allocated context,
not filled-64K evidence or native quantized-arithmetic qualification. The public
model gate remains closed. Proceed to actual prompt-length/resource checks.


### Task 5 — exact filled-context retrieval runner and pilot

Added `scripts/qualify-native-context.py` and usage/scope documentation in
`CONTEXT_QUALIFICATION_20261021.md`. It freezes server-tokenized prompts at
exact requested lengths, retaining the complete chat suffix and three records
near the beginning, middle and end. Raw streams/final receipts are preserved;
actual prompt/output counts, no truncation, finite consistent timings, correctly
labelled retrieval and exact repeated output are mandatory. It never starts or
changes an existing server and refuses to overwrite evidence. Three regression
cases cover exact 65,536-token fixture construction, complete record/suffix
preservation, wrong/missing/contradictory assignments and invalid templates.
All 19 native benchmark/context/surface Python checks pass.

The selected-model 2,048/128 pilot completes inference with no context
truncation, but fails the retrieval assertion: its reasoning contains all three
correct codes, while the fixed 128-token generation limit cuts the final answer
after `GAMMA=84507`. This failed result is retained at
`task5/flash-next-context-pilot/`; it is not a retrieval pass. Server exit is
normal, leases/queues/staging drain, and no process swap is sampled. Increase
the correctness workload's explicit generation headroom to 256 tokens and rerun
before advancing to larger lengths. This changes only the test output budget;
it is independent of the accepted 552/128 speed observation. Original model,
launcher and production admission remain unchanged.


The 256-token-headroom retry is retained at `task5/flash-next-context-256/`.
Both 2,048-token requests pass full receipt counts, three-position retrieval
and exact generated-text replay. The first 8,192-token request also passes
actual counts, no truncation and all three retrieval assignments; its repeat
and the 65,536-token stage are still running/pending. The first 8K request
uses 449.48 s for prefill and 30.43 s for 256 generated tokens (8.41 tok/s).
The accepted approximately 19 tok/s remains scoped to the earlier frozen
552-token workload, not a speed floor at larger filled contexts. These
correctness tests do not reopen performance optimization. Process swap is
sampled in this retry and will be reported with the completed resource audit;
no zero-swap claim applies. Both shards' exact size/mtime seals match the
previously completed SHA-256 identities. No filled-64K, concurrency, native
arithmetic or release acceptance is claimed by these intermediate results.


### Task 5 — bounded verification; multi-hour passes deferred

The user stated **“we dont have time for multi hour passes.”** The active
context runner was interrupted immediately and its owned server shut down
normally (server exit 0). No 65,536-token pass started. Preserve the two complete
2,048/256 requests with exact replay and the first complete 8,192/256 retrieval
pass; the second 8K request is interrupted, not a pass or an inference failure.
Final evidence and sampled resource peaks are in
`task5/flash-next-context-256/qualification-summary.json`, with the original
report, user-stop record, raw streams, identities and logs retained. Larger
filled-context/concurrency/native-arithmetic claims remain unqualified.

The probe now defaults to 2K only and a hard 120-second total wall-time budget.
Its POSIX interval timer interrupts even a server that never sends HTTP headers;
operator cancellation and budget exhaustion have distinct report states.
Four context regressions plus the existing benchmark/surface checks pass
(**20 total**), including a real local HTTP server that deliberately withholds
its response. Larger lengths/longer budgets require explicit arguments; do not
launch multi-hour passes as routine follow-up. Continue only the available
short release-readiness checks. No publication authorization or production
Qwen admission is implied by the reduced validation scope.


### Task 6 — six real Linux packages from a frozen source snapshot

Frozen package source: `e4b5739fc9380f401be3c548200a499a7f84ba34` in an
independent shared clone. Original source/build caches remain unchanged;
borrowed Git objects are mounted read-only. Source status is clean after the
frozen-lockfile frontend install and both package builds. Qualification-only
log updates after this point do not change that recorded package source.

The host CPU binary requires GLIBC 2.43 / GLIBCXX 3.4.35, so it is unsuitable
for the Ubuntu release baseline. Rebuilt the CPU runtime under cached Ubuntu
24.04 (GLIBC 2.39, GCC 13.3), with `GGML_NATIVE=OFF`, production Qwen admission
OFF and Release optimization: 134.64 seconds. The refreshed consolidated CUDA
runtime uses GCC 12.4, the selected 75/86 CUDA targets and lower symbol versions;
it loads and reports the frozen source/version inside Ubuntu 24.04. No full
CUDA recompilation is needed. An attempted `--list-devices` preflight is retained
as a rejected command: this fork does not expose that option. It is not counted
as a GPU inference check; packaged inference remains pending.

Built **six actual 0.2.1 installers**: CPU and CUDA AppImage/DEB/RPM. The CPU
package pass takes 113.49 seconds and CUDA 129.07 seconds; collection takes
1.01 / 22.43 seconds. Both variant collectors pass against the same frozen
source, runtime markers and separate updater overlays. Artifacts and indices:
`task6/ubuntu-build/assets-cpu/` and `task6/ubuntu-build/assets-cuda/`.
`task6/task6-progress.json` records exact names, sizes, commands and evidence.
Windows MSI/NSIS and the complete eight-package acceptance remain pending.

The initial offline dependency attempt stops at its 30-second budget because
pnpm's supply-chain validation requests uncached metadata. Retain that failed
attempt; the corrected network-enabled **frozen-lockfile** install passes its
policies in 29.15 seconds, without a lockfile/source change. Rust builds remain
offline against the existing registry cache. No remote CI or publication occurs.

The existing updater private key is present with mode 0600, matches the compiled
public key and successfully signs a local preflight payload. Verification uses
the updater's `minisign-verify 0.2.5` implementation: the valid payload passes,
while altered payload bytes and an altered trusted comment fail. Both real
collected AppImage signatures also verify against that configured public key.
No private key contents are logged or placed in container configuration. These
checks establish signing configuration and actual AppImage signatures, not an
installed-update acceptance pass. Native DEB/RPM package-manager signing and
Windows Authenticode are not claimed.

The system hypervisor has an ESE Windows test VM (`win10`, stopped, no physical
host-device passthrough) and a separate stopped VR VM with USB/GPU passthrough.
Only their configuration was inspected; neither VM nor its disk was changed.
The ESE bootstrap ISO's authorized key matches an available local key. The
original Win10 qcow2 is not directly user-readable but is managed through the
active `images` libvirt pool. If used, test an isolated overlay rather than
modifying the original VM/disk. Do not start the VR VM or detach the desktop
RTX 3080. All remaining inference checks keep short explicit time budgets.


### Task 6 — bounded Linux installation and packaged inference pass

All six frozen-source Linux candidates pass extraction, version/dependency,
license, desktop-entry and ELF dependency checks. CPU packages require no NVIDIA
devices or CUDA libraries. CUDA packages contain the four pinned vendor payloads
with their original hashes and exclude NVIDIA driver libraries. All **24 project
native payloads** match the corresponding frozen unpackaged builds after the
declared `$ORIGIN` RPATH relocation.

Actually installed both DEBs in separate Ubuntu 24.04 test containers and both
RPMs in separate Fedora 44 test containers. Package-manager verification, native
payload hashes, runtime variant markers, launcher doctor and native version all
pass. The Fedora base already contains a different candidate with version
0.2.1; the initial refusal is retained. Reinstalling with `--replacepkgs` installs
the current artifact, and its payload hashes and `rpm -V` then pass. An initial
DEB supervisor PATH omission also remains recorded; restoring the standard sbin
paths resolves it without changing the package.

**25 short inference cases pass** across extracted DEB/RPM/AppImage and installed
DEB/RPM runtimes: CPU builds without NVIDIA devices; CUDA builds with devices
hidden and with each of the RTX 3060 Ti, RTX 2080 SUPER and RTX 3080 selected
individually by UUID. Each case checks the 12-token prompt receipt, one generated
token, exact text and top-32 probabilities against its corresponding unpackaged
build, native/vendor library paths, owned GPU identity, resource enumeration,
unsupported resident rebalance rejection and normal server shutdown. These are
package-preservation checks, not full-vocabulary numerical or speed qualification.

The first inference attempt referenced an archived 32-vocabulary toy while
loading the current 32,000-vocabulary TinyMoE fixture. Preserve that failed
harness attempt. Corrected references use the exact current model SHA-256
`c559eaabd8a56589b8abe09d678fbb56c673a3dcc28907e193d27b25ecd60684`,
the frozen source and the matching CPU/CUDA unpackaged runtime. Exact output
assertions remain unchanged; no numerical tolerance is introduced.

All **six GUI startups** pass as an unprivileged user with isolated XDG data and
Xvfb displays: actual installed DEB/RPM and actual AppImage for both variants.
Each renders an ESE Studio window for at least 12 seconds; screenshots are
retained and representative installed-DEB/AppImage captures visually checked.
No model server or GUI probe remains running. Inference and GUI probes have
internal 90/100-second budgets plus outer timeouts below two minutes. No large
context or sustained throughput run is added.

Evidence: `task6/linux-inspection/linux-acceptance-summary.json`, with per-case
responses, exact fixture/reference identities, installation logs, GUI captures,
native hashes and helper provenance. Linux packaging acceptance is complete for
this short-check scope. Windows MSI/NSIS builds and installation, installed-update
and failure/recovery trials, and final release preparation remain pending.
The accepted approximately 19 tok/s scope and production Qwen gate remain as
previously recorded. No push, tag, hosted CI or publication has occurred.


### Task 6 — isolated Windows preflight

Created a new qcow2 backing overlay through libvirt and an independent test VM
(`ese-review-win-20261021`) with its own UUID, MAC, NVRAM and virtual TPM. No
physical device passthrough is assigned. The original `win10` configuration
matches the saved preflight XML exactly; the original disk remains the read-only
backing image. The VR VM and desktop GPU are untouched.

The clone boots Windows 10 Pro 19045 and accepts the already-authorized test SSH
key for its existing `ok` account. No key or account is changed. Node, pnpm, Rust,
MSVC, CMake/Ninja and installer tooling are present, but only an older CPU native
build is cached (`GGML_NATIVE=ON`). No CUDA toolkit or CUDA build cache is
present. Those older runtimes/installers are not current-source package evidence.
Current dual-runtime MSI/NSIS, physical Windows NVIDIA inference and Windows
update qualification remain pending; do not replace them with PE inspection.

The clone completes a normal guest shutdown. A managed-save request races with
that completion and reports that the domain is not running; final `domstate`
confirms **shut off**. No forced power-off or GPU detachment occurs. Retain the
overlay and inventory for further work at
`task6/windows-isolated/preflight-summary.json`.


### Task 7 — signed Linux update installation and recovery pass

Linked a local integration probe against the **same unmodified production
`tauri-plugin-updater 2.10.1`** library retained by the candidate build. It uses
the configured public key and the actual signed CPU/CUDA AppImages. Only the
test context uses a loopback HTTP feed. Production HTTPS endpoints/configuration
are unchanged. All six packaged Studio executables contain their corresponding
`latest-cpu.json` or `latest-cuda.json` URL and the expected public key, without
the opposite variant URL.

Eight bounded cases pass: default version comparison rejects the same-version
offer; HTTP 503, truncated transfer, corrupted signature and corrupted payload
are rejected; a forced file-size-limit write failure rolls back to the original
bytes; valid CPU and CUDA retries install their exact signed candidate hashes.
Tests run as UID 1000 on temporary files in the container filesystem and retain
0755 permissions. Installation starts from archived distinct 0.2.1 AppImage
builds; a **test-only comparator permits same-version replacement**. This is
not a previous-public-version migration or public-channel test. The forced
write error is `EFBIG`; rollback preserves the original bytes and executable
mode. Real valid payloads then replace the temporary installed files.

Both updated AppImages also render ESE Studio for at least 12 seconds as the
unprivileged user; the updated CUDA capture is visually checked. The complete
integration run has a hard 110-second budget; valid CPU/CUDA install cases take
1.15 / 14.08 seconds, plus separate short post-install GUI checks. All owned
probe processes stop. Evidence, raw failures, requests, exact before/after
hashes and helper identities are retained in
`task7/linux-update-qualification.json` and
`task6/ubuntu-build/update-checks/`.

Linux signing and local updater integration are now checked. Native DEB/RPM
package-manager signatures, Windows Authenticode and actual Windows update/
failure-recovery acceptance are not established by those results. Publication
and the public updater channel remain on hold.


### Task 8 — current local review draft and Linux hashes

Prepared [the current candidate review](RELEASE_CANDIDATE_REVIEW_20261021.md)
with the frozen package source, all six Linux payload hashes, evidence links,
accepted speed scope and explicit Windows/full-matrix gaps. The private
`task6/ubuntu-build/LINUX-CANDIDATE-SHA256SUMS.txt` includes six Linux packages
and their two collected AppImage signatures. It is not the complete release
`SHA256SUMS` or an eight-package acceptance pass.

Corrected stale statements that the current Linux AppImages are unsigned or
that old eight-package builds describe the current source. Installation notes,
Studio documentation and draft release notes now distinguish checked Linux
signatures from the pending Windows builds and public channel. No binary or
product configuration changes follow the frozen package source; later commits
record documentation and qualification. Complete release metadata generation
and final approval remain pending the actual current Windows artifacts and
their required qualification. No push, remote CI dispatch, PR, tag or release
is created.

Documentation/link invariants, all version fields and Git whitespace checks
pass. All five owned Linux build/test containers are stopped after their probes
finish; the Windows clone and both original VMs are shut off. The publication
hold and the active remaining Windows/full-matrix work remain unchanged.


### Task 6 — current Windows builds and Microsoft runtime staging repair

Staged the exact `e4b5739fc...` source archive in the isolated VM and verified
its SHA-256 on both hosts. Downloaded six Windows CUDA 12.4.1 component archives
(563,232,009 bytes total), verified every size/SHA-256 against NVIDIA's official
manifest on Linux and again in Windows, and extracted them into a private SDK
directory. NVCC 12.4.131 and its MSVC 19.44 compiler probe pass with the required
75-real/86-real/86-virtual targets and production Qwen admission OFF.

Pinned portable Node 22.22.2 and pnpm 11.19.0, installed Rust 1.96.0 and
PyInstaller 6.16.0, and passed the frozen-lockfile supply-chain checks. TypeScript,
Vite and the standalone `ese.exe --version` pass. The full Studio Release Rust
build passes in 269.07 seconds. Build commands have explicit limits; these are
builds, not new sustained inference/performance passes.

The inherited VM topology exposed eight sockets, but Windows reported only two
one-core processors. Reconfigured only the clone to one eight-core socket and
24 GiB RAM; Windows then sees eight processors. A new empty 32 GiB scratch
volume, identified by its unique serial, was initialized as S: for Cargo output.
Live SATA attachment was rejected without changing the VM; an emulated USB disk
was used instead. No physical host USB or GPU device was assigned. The original
VMs/disks remain unchanged and the base disk is opened read-only by QEMU.

The CPU native build completes after retaining two budget-limited attempts and
reusing their completed objects. The matching c559... fixture serves the exact
12-token input and one generated token. An initial shutdown harness inherited
OpenSSH's Ctrl+C-ignore setting; a separate Python console control proves this.
Clearing that inherited flag yields a normal server exit 0 without changing
product code. This single-token check does not qualify throughput.

Actual PE inspection exposed a packaging gap: MSVCP, VCRUNTIME and VCOMP imports
were satisfied by the build VM's system DLLs but were missing from the staged
native runtime. Added transitive release-x64 MSVC redistributable staging from
Visual Studio's desktop CRT/OpenMP folders, per-directory size/hash inventories,
and strict verification. Debug, OneCore, missing, altered and unlisted runtime
payloads are rejected. Studio's CRT files are mapped beside the application via
a Windows-only Tauri overlay. CI carries both runtime resource directories into
the separate MSI/NSIS jobs and verifies Studio's imports after those builds;
release verification checks the native and Studio closures together.

The real unfixed Windows stage fails the new dependency check. After repair,
the native and Studio import closures pass. The CPU server loads all six MSVC/
OpenMP DLLs from its staging directory, preserves exact text/top-32 probabilities
against the unpackaged Windows build and exits normally. The four native
project files remain byte-identical. Eleven focused Python controls pass on
Linux and Windows; all 35 Studio checks, workflow actionlint 1.7.7, documentation
invariants and whitespace checks pass. Microsoft runtime notices now describe
the app-local deployment and servicing scope.

The initial PyInstaller pin incorrectly forced a pefile version excluded by
PyInstaller; retain that resolver rejection. The corrected pinned PyInstaller
install resolves pefile 2023.2.7 successfully. A PowerShell process-wrapper exit
code observation bug also remains recorded; SDK/tool installation itself passes,
and subsequent .NET/Python process supervisors capture actual exit codes and
terminate only their owned process trees on budget exhaustion.

The first CUDA build reaches 91/416 steps before its 600-second build limit;
retain completed objects and the terminal budget record. No current dual-runtime
MSI/NSIS or Windows GPU acceptance is claimed. Evidence is under
`task6/windows-isolated/`, including `cpu-staged-msvc-proof.json`,
`msvc-staging-qualification.json`, `studio-rust-build-job.json`, SDK/source
identities and all failure/control logs. The Windows staging/notice changes
require a new final eight-package set; the earlier signed Linux candidates stay
identified by their original frozen source. Publication remains on hold.


### Time-bounded checkpoint — current source repair and refreshed CPU packages

Committed the Windows runtime repair as product/package source freeze
`1888775e987ba48465e4b48479d0dd1692d081ab`. Eleven focused Python controls,
workflow actionlint, version/documentation invariants and whitespace checks
pass. CPU staging with a valid CUDA SDK present also passes: unused CUDA DLLs
are excluded and the actual MSVC native/GUI import closures remain complete.

Applied only the changed tracked files to the Windows source tree, preserving
unchanged native input timestamps and compiled objects. Verified all 3,333
tracked source files against Git blob hashes. The first source-seal harness
incorrectly hashed five symbolic-link targets as regular file contents; retain
that failure. The corrected check validates link text separately. No native
source changes occurred since the earlier frozen archive.

Only the isolated clone was changed to one 16-core socket and 32 GiB RAM, and
Windows reports 16 CPUs. The resumed eight-job CUDA build adds 38/325 completed
steps. It is deliberately stopped after 289.75 seconds because the remaining
cold compilation exceeds a short step; it does not pass or exhaust its planned
600-second limit. The explicit checkpoint record distinguishes this stop from
a compiler failure. A separate Tauri CLI/custom-protocol prebuild reaches its
300-second budget after staging/frontend compilation and rebuilding the Tauri
feature-dependent crates. It does not pass; the earlier 269.07-second raw Cargo
build remains a separate passing result. All completed objects are preserved.
Do not automatically chain further cold build windows under the user's time
constraint. The clone has no NVIDIA display driver; no driver setup or physical
GPU passthrough was attempted in this step.

Reused the Linux native caches; refreshing Git build identifiers needs only
four incremental build steps per variant. The initial host CUDA relink lacks
its build-time vendor library search path and fails; the explicit cached CUDA/
compiler library path resolves that relink without changing source or flags.
Three real current-source Linux CPU packages build in 110.06 seconds and collect
in 1.53 seconds under `task6/ubuntu-build/final-1888775e9-assets-cpu/`.
Their four-file checksum list includes the three packages and AppImage signature,
not the complete release checksum set. The actual AppImage signature verifies.

The CPU DEB, RPM and AppImage inspections pass in 0.79, 0.98 and 2.47 seconds.
All 12 relocated native files match the refreshed unpackaged build exactly;
versions, ELF closure, driver exclusion, desktop files and current notices pass.
Retain two inspection harness failures: the clean image's inherited Python
wrapper references an unavailable old mount, and native `--version` writes its
banner to stderr. Explicit system-Python PATH and checking the captured banner
stream resolve both without modifying the packages. Successful evidence is
`task6/linux-inspection/final-1888775e9/banner-stderr-control/summary.json`.

The refreshed Linux CUDA packaging attempt hits its 180-second limit during DEB
bundling, after producing a partial AppImage build. It is not collected or
treated as a completed package set. The earlier six-package Linux qualification
remains tied to `e4b5739fc...`; do not mix it with the refreshed CPU source index.
No new inference throughput or extended-context pass runs. Approximately 19
tok/s remains the user-accepted experimental baseline.

Copied and hash-verified the terminal Windows source/build/checkpoint archive
(`09912027dd191298c1a4815611b093a5c9c12e612e301859a4bfafb0a7e6294f`).
Windows process inspection confirms no compiler/linker/build children remain.
The clone shuts down normally; both original VMs remain shut off. The owned
Linux build container is stopped after its bounded command; its sleeping PID 1
requires container SIGKILL after a two-second stop timeout. All caches, disks,
artifacts and failure evidence remain intact. Actual Windows installers,
Windows install/update/GPU checks, refreshed CUDA collection and final full-set
metadata remain pending. No push, hosted CI, PR, tag or publication occurs.


### Task 7 continuation — cached Linux bundles and package-format update policy

The user authorizes continuing the Windows build in the background while other
work proceeds. Resume the cached CUDA build with eight jobs and a 2,700-second
limit; do not resume long inference qualification. The original VMs/disks and
desktop GPU remain untouched.

Finish the missing Linux CUDA DEB/RPM bundles through Tauri's bundle-only
command. The interrupted prior run left the build executable's Tauri bundle
initializer set to DEB. A first bundle-only attempt warns and would retain that
wrong initializer in RPM; do not accept those bundles. Inspection of the pinned
upstream bundler confirms its normal behavior: replace the UNK initializer for
each format, then restore the unpatched binary after successful bundling.
Preserve the interrupted executable, resolve its actual mutable initializer
through the ELF symbol and relocation, and reverse exactly the three changed
marker bytes. Bundle-only then completes DEB/RPM without warnings in 55.55
seconds and retains their correct per-format initializer. No application or
native compilation occurs. A signing CLI environment/path argument conflict is
retained; using the existing key environment signs the unchanged prior AppImage
in 2.44 seconds. Collection completes in 36.93 seconds.

All six `1888775e9` Linux packages now pass static inspection, all 24 native
payload identities, current notices, dependency/version checks, exact compiled
CPU/CUDA feeds, public keys and actual ELF bundle-type initializers. Both real
AppImage signatures verify. Extracting CUDA packages directly onto the evidence
archive first hits a 65-second bound; using local container scratch completes
the DEB/RPM/AppImage checks in 16.89/29.96/22.70 seconds. Preserve the slow attempt.
Eight real updater API cases also pass against these new signed payloads,
including default same-version rejection, corrupt/truncated downloads,
write-failure rollback and valid CPU/CUDA replacement with exact bytes/mode/UID.
This remains local integration with a test-only reinstall comparator, not a
public-version upgrade. A prior absolute-host-path test symlink is corrected to
a relative sibling link visible inside the build container; retain that failed
test setup separately.

Review then exposes a remaining supported-flow defect: the GUI offers its
AppImage automatic-update action inside DEB/RPM installations even though the
package contract specifies manual package-manager updates. Add a bundle-type
guard using the already-pinned Tauri API and its default core permission.
DEB/RPM show manual update instructions and make no updater request. Unidentified
formats also avoid guessing a download type; AppImage, MSI and NSIS preserve
automatic signed-update checks. Five focused behavior tests cover zero network
requests for manual/failed detection, the three automatic formats and preserved
no-update/error results. This frontend change requires final packages from a
new recorded source; the checked `1888775e9` artifacts stay identified as such.

Read-only host preflight also establishes a physical Windows GPU limitation:
both spare GPUs belong to IOMMU group 15 alongside the mounted-storage SATA
controller, USB and Ethernet. The Linux VFIO ownership rules prohibit sharing
that group between native host drivers and the guest. No device is detached,
driver downloaded, host rebooted or original VM changed. Continue Windows
build/install/update qualification independently; physical Windows NVIDIA
inference remains blocked by the current host isolation. Evidence is
`task6/windows-isolated/physical-gpu-iommu-preflight.json`, with the kernel
documentation reference and actual group membership. Publication stays on hold.
