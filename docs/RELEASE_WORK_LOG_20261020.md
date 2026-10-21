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

### 3. Verify inference correctness and recovery — COMPLETE (LOCAL FIXTURE ACCEPTANCE)

- Compare supported resident/cache/stream paths against pinned references.
- Exercise forced eviction, cancellation, allocation failure and session
  recovery; retain logits/routes/output and ownership/accounting evidence.
- Keep experimental Qwen admission disabled unless its independent numerical
  acceptance gates are resolved; do not promote research based on compilation.

Acceptance: reproducible correctness and recovery evidence for every claimed
supported execution path.

### 4. Resolve the performance hold — IN PROGRESS (TARGET: AT LEAST 80% OF STRATA)

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

- Define the exact model, prompt, resource budget, policies and speed target.
  Use Strata's measured sustained decode rate as the reference; retain startup,
  prefill and first-response timings separately.
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
