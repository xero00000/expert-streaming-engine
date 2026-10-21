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

### 4. Resolve the performance hold — IN PROGRESS (TARGET: MATCH STRATA)

The [performance qualification protocol](PERFORMANCE_QUALIFICATION_20261020.md)
is prepared. The user has set the speed target: **match Strata**. Establish a
same-machine, same-model reference before applying that target.

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
