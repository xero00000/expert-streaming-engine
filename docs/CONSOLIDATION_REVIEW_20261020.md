# ESE source consolidation review — 20 October 2026

The release candidate, experimental inference work and Android draft now have
separate local review branches. Initial consolidation preserved every original
worktree; the subsequent authorized cleanup below removed redundant copies.
Nothing was pushed or published. `ESE-current` still points to
`ESE-v0.2.1-qwen4exp-plan` so existing launcher/build paths continue to work.

This review covers source provenance, preservation, integration boundaries,
documentation consistency and the fresh checks below. It does not qualify the
experimental GPU paths or remove the outstanding release gates.

## Where to continue

| Purpose | Local branch | Worktree |
| --- | --- | --- |
| Consolidated v0.2.1 candidate | `review/ese-consolidated-20261020` | `/home/xero/Desktop/test/ESE-review-consolidated` |
| GLM, expert execution, placement and dense-to-MoE research | `review/ese-research-20261020` | `/home/xero/Desktop/test/ESE-review-research` |
| Android application and embeddable native API draft | `review/ese-android-20261020` | `/home/xero/Desktop/test/ESE-review-android` |

The candidate starts from public `main` at
`eccbaff3c5fdbcc020e61a414ff36d01cfef7f94`. It replays the previously prepared
release commits `a5b07084d` and `9784567c4`, preserving the public benchmark
updates through October 19. It then incorporates the newer bounded
`--expert-load-workers` launcher option, its boundary/CLI regression checks,
and the documentation/ignore fixes described below.

The research branch retains 128 source/documentation changes over that
candidate baseline. Research native option names are also added to the
launcher's value-argument table. The Android branch records 53 changed or new
paths against its original `bc9df1596` base; it deliberately has not been
rebased into the newer desktop runtime.

## Preservation and lineage

The inventory covers 14 original ESE worktrees, including the prepared release
checkout under `results`. The historical combined-transaction, compact parity,
diagnostic, Windows-fix, concurrency and v0.2.0 work is retained. Several old
heads are public ancestors. Others have equivalent public patches despite
different ancestry. The expert-transaction branch has one unmatched patch ID;
its transaction test is byte-identical to the candidate and its native
prepare/commit/rollback/finalize APIs are present. Its original head is still
retained rather than assuming the branch is disposable.

Each original head has a local `refs/archive/ese-review-20261020/…` reference.
Tracked edit patches, per-worktree inventories and source hashes are retained
under:

`/home/xero/Desktop/test/results/ese-consolidation-review-20261020`

`preserved-source-blobs.tar.gz` contains 681 distinct payloads, deduplicated by
SHA-256, totalling 502,329,947 uncompressed bytes. Every archived payload was
read back and verified. `preservation-manifest.json` maps each original path,
mode, deletion and base commit to its payload. A final check of 1,287 original
overlay files found no content changes caused by this work.

The archive excludes 5,520 generated CMake/Gradle cache entries. Those entries
and all original build trees remain in place. Scratch source, retained Android
evidence and the large untracked research captures are preserved in the
archive. The root `ESE_OPTIMIZATION_PLAN.md` is copied alongside it. The
Windows bootstrap ISO remains an external artifact. An index identifies 160
related evidence directories; their datasets, models and binaries were not
copied into source history.

## Review findings addressed

- Most of the old worktrees' apparent pending source backlog is already in the
  October 14 release commit. Reusing that baseline avoids reintroducing older
  public documentation and benchmark data.
- The newer launcher worker option was present only as a dirty edit. It is now
  committed with checks for cache/staging limits, the zero-worker default,
  policy/count rejection and CLI-to-native propagation.
- The documentation checker expected obsolete installer names and text, the
  changelog lacked v0.2.1, and a research README linked to an excluded internal
  plan. Candidate documentation and its checker now agree. Unpublished
  installers are identified as candidates rather than existing latest assets.
- The research launcher omitted five native value flags: the three dense-MoE
  options, `--device-h2d-rate` and `--placement-policy`. This failed an existing
  parser guard and could misinterpret their values as device placement flags.
  The corrected research branch passes that guard.
- Local build directories, scratch probes and the raw native-recurrence
  capture directory now have explicit ignore rules on the candidate/research
  branches. A 171,498,259-byte fixture is preserved externally instead of being
  committed; it exceeds GitHub's 100 MiB per-file limit. The inherited
  83,359,037-byte operator reference remains tracked because existing
  regression coverage uses it. No review branch contains a blob over 100 MiB.

## Fresh validation

| Check | Result |
| --- | --- |
| Complete clean Release CPU build | Passed, all 480 build steps, 225 seconds |
| Registered native CPU CTest suite | 94/94 passed, 17 seconds |
| Candidate Python unittest suite | 128/128 passed |
| Qwen reference adapter and benchmark replay pytest checks | 33/33 passed |
| Studio packaging/runtime JavaScript tests | 32/32 passed |
| Runtime staging, TypeScript and Vite production build | Passed |
| Rust formatting | Passed |
| Documentation links/version checks and six-surface release-version guard | Passed |
| Research Python unittest suite and documentation check | 128/128 and passed |
| Candidate/research/Android Git whitespace checks | Passed |

The native CPU build used `GGML_CUDA=OFF`, `GGML_NATIVE=OFF` and
`ESE_QWEN4EXP_PROBE_GATE=OFF`. Its old retained DeepSeek block failure did not
reproduce: the complete fresh registered suite passes. The server help and
`ese 0.2.1` CLI smoke checks also pass.

The installed pnpm attempted an automatic dependency migration and aborted
before changing the shared dependency directory. The production build was
therefore run through the same three declared script stages directly with the
existing dependencies: runtime staging, TypeScript and Vite. Both the failed
pnpm attempt and successful direct build are retained in the evidence folder.

No new CUDA build, GPU model qualification, Rust test/Clippy run, Android
build/device test, package installation or Windows execution was performed in
this consolidation. Existing package evidence applies to the earlier prepared
artifacts; installers have not been rebuilt with the new launcher option.

## Findings still requiring work

1. **Release automation does not reproduce the documented candidate set.**
   `.github/workflows/release.yml` builds one Linux CPU variant; the prepared
   candidates include separate CPU and CUDA variants. Its collector and updater
   manifest still use `ese-studio_<version>_amd64…` names, whereas the prepared
   artifacts use `ESE-Studio_<version>_linux-x86_64_cpu/cuda…` and
   `ESE-Studio_<version>_windows-x64…`. Resolve the build matrix, naming,
   signatures and updater targets together before tagging.
2. **Research remains a separate qualification effort.** GLM/dense streaming,
   new expert dispatch and placement paths have not been promoted into the
   candidate. Several native research targets are intentionally manual rather
   than registered CTest gates. Sparse routing quality, full accounting,
   lifecycle/capture and matched performance need their own retained evidence.
3. **Android needs a focused integration and portability review.** Its draft
   remains based on the older engine. In `ese-core/include/ese/ese.h`, Windows
   static/client builds fall through to a GNU visibility attribute unless
   `ESE_CORE_SHARED` is defined; the CMake target defines that macro privately.
   Review export definitions for MSVC consumers before desktop integration.
   This has not been checked with an MSVC build here.
4. **Existing public-release gates remain open.** Full-model Qwen numerical
   acceptance, filled-64K/concurrent-session qualification, signing and Windows
   GPU/installer execution are still incomplete. Public Qwen admission remains
   disabled. The source cleanup does not establish release readiness.

The next release task is to reconcile and validate the package matrix. The
research and Android branches can be reviewed independently without changing
the candidate or the working launcher symlink.

## Subsequent authorized cleanup

After consolidation, the user requested removal of unneeded old branches and
worktrees. The following 11 redundant checkouts and their disposable build
outputs were removed:

- `ESE-combined-peak`, `ESE-combined-txn`, `ESE-compact-parity-gate`;
- `ESE-expert-cache-diag`, `ESE-expert-txn-api`, `ESE-kv-txn-api`;
- `ESE-main-winfix`, `ESE-mtp-owner-txn`, `ESE-transient-txn`;
- `ESE-v0.2.0-public`;
- `results/ese-v0.2.1-release-preparation-20261014/source`.

Their recorded HEADs and source-archive checksum were verified first. Fresh
status checks found only disposable generated files and no new source edits.
The model-file inventory found no untracked model weights beyond CMake
compiler-probe binaries. A process/file-descriptor check found no use of the
removed paths, and launcher/configuration searches found only historical
archive references to them.

Eleven obsolete local branch names were deleted, including the unmounted
`v2/concurrent-local-sessions` and `v2/transient-resource-transaction` branches.
Their exact heads remain recoverable under
`refs/archive/ese-cleanup-20261020/…`; the earlier review archive refs and
source payload archive also remain. No remote branch was deleted.

The cleanup freed 5,588,590,592 bytes (about 5.2 GiB) on the home filesystem,
plus the separate prepared-source checkout on the external results disk. Git
connectivity checks passed afterward. The two launcher-selected CUDA servers
and the consolidated CPU server still produce valid help and resolve their
shared-library dependencies.

`ESE-glm5-next` and `ESE-v0.2.1-qwen4exp-plan` remain because `model_slots.sh`
uses their CUDA builds. The primary `ESE` directory owns the shared Git
database and retains the original Android build workspace. The three new
review worktrees, `main`, and `llama-kimi-reference` remain available. The
launcher symlink was not changed.

Exact deletion records and recovery references are in `cleanup-result.json`
alongside the source archive and inventories.
