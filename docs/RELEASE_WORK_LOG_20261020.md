# ESE 0.2.1 release work log

Started 20 October 2026. Final release preparation: 22 October 2026 (UTC).
Working branch: `review/ese-consolidated-20261020`.

## Current authorization

After reviewing the GitHub presentation, the user requested merging the update
and publishing it, with clear update logs focused on ESE. This supersedes the
earlier publication pause. Merge through the protected `main` workflow, retain
qualified build provenance, and verify the uploaded release before finishing.

Physical Windows NVIDIA inference was explicitly deferred by the user and
remains unqualified. Extended context/concurrency and multi-hour inference
passes are also deferred. Production Qwen admission stays OFF. The user accepted
the recorded approximately 19 tok/s experimental baseline for this version;
it is not a supported-model speed guarantee.

## Completed work

| Task | Completed result |
| --- | --- |
| 1. Package contract | Eight actual packages; original per-platform source indices, 11 package/signature digests and correctly routed updater feeds |
| 2. Consolidated source | CPU/CUDA builds, scoped native/Python checks, Rust tests/Clippy and Studio production checks; detailed results retained in validation records |
| 3. Correctness/recovery | Pinned model/output references, supported cache/stream paths, eviction, cancellation and allocation/recovery controls |
| 4. Performance decision | Current experimental baseline accepted by user; further optimization deferred |
| 5. Workload scope | Bounded 2K/8K prompt retrieval/replay checks pass; long context/concurrency tests deferred |
| 6. Packages/installations | Linux CPU/CUDA AppImage/DEB/RPM and Windows MSI/NSIS software checks complete; physical Windows GPU inference deferred |
| 7. Signing/updates | Real AppImage/NSIS signatures and installed-update checks pass; DEB/RPM manual-update behavior verified |
| 8. Release preparation | Source freeze, package hashes, release notes, beginner presentation and local PR/tag drafts prepared |

[Candidate evidence and package hashes](RELEASE_CANDIDATE_REVIEW_20261021.md) ·
[Package contract](RELEASE_PACKAGE_CONTRACT.md) ·
[Detailed validation record](RELEASE_NOTES_V0_2_1.md)

## Build provenance

All eight packages were built from
`2ff6389928fb701ad9331275682529cf5896645b`. Subsequent changes are documentation
and GitHub presentation only. Publication must retain that original commit in
the manifest and source indices, record the final integration commit separately,
and prove product inputs did not change. No old package is relabelled as a new
build and no unnecessary rebuild is needed for release copy changes.

## Linux qualification

Six actual Linux packages pass native payload, dependency, notices, bundle-type,
configured updater-feed and public-key checks. Four Ubuntu/Fedora DEB/RPM
installations and ten short inference cases pass on `e991c3932`; the final six
GUI binaries, both native executable text sections and native shared libraries
match that installed/inference-qualified predecessor. Final package collection
and signatures are checked separately against the package source freeze.

Both final AppImage signed replacements preserve their exact hashes, mode 0755
and UID 1000, and their updated Settings views were visually checked. Eight
production-updater API controls cover same-version rejection, transport and
corruption failures, write-failure rollback and valid retries. DEB/RPM update
controls use package-manager instructions and do not contact an AppImage feed.

These are fixture and package-preservation controls. They do not establish
full-vocabulary arithmetic, general large-model performance or filled-1M
context acceptance.

## Windows qualification

The owned Windows clone builds CPU/CUDA runtimes, standalone launcher and
Studio. The native CUDA build completes in 543.31 seconds. MSI uses three
bounded embedded cabinets; NSIS packaging completes in 808.71 seconds. App-local
Microsoft C++/OpenMP dependencies are included and inventoried.

Actual MSI installation/uninstallation and NSIS installation pass. Both retain
all 33 expected runtime files, select CPU fallback without an NVIDIA driver,
and match the short fixture's reference top-32 output. NSIS cryptographic
signatures verify on host and guest. Seven updater download/format controls
and a real signed passive-mode reinstall pass, including installer exit 0,
deliberate license-file repair, runtime checks and post-update CPU inference.
The final interactive Settings capture was visually verified. Earlier captures
occluded by a Windows upgrade prompt remain identified in the private evidence.

Windows Authenticode signing is not claimed. No physical GPU was detached and
no Windows NVIDIA driver was installed. The host isolation constraint remains;
physical Windows GPU inference is deferred, not passed.

## Cleanup and evidence

All five owned Linux build/install containers are stopped with empty cgroups.
The Windows clone and both original VMs are shut off normally; user models,
launchers, caches and original disks remain intact. The actual Windows evidence
archive is 28,180,480 bytes, SHA-256
`a6beb807f9944ad78049cd295412d74029568bcc7ad4bf08bf99fb960d765353`.

Detailed engineering trials, failures and raw execution logs are retained in the
maintainer's local evidence archive. Public validation documents retain their
original versions and limits. Main evidence locations under
`/home/xero/Desktop/test/results/ese-release-work-20261020/`:

- `final-2ff638992-release-assets/`: eight packages, three source indices and
  generated checksums/update metadata.
- `task6/linux-inspection/final-2ff638992/qualification-index.json`: Linux
  identity, unchanged-code, installed behavior and GUI evidence.
- `task6/windows-isolated/final-2ff638992-evidence/`: actual install, inference,
  update and screenshot records.
- `final-2ff638992-linux-process-cleanup.json`: authoritative container cleanup.

## GitHub presentation

The newcomer-focused README leads with **Bigger AI. The PC you already own.**,
original banner artwork, actual Studio images and a three-step start. Detailed
commands and historical measurements remain in the engineering guide. The
beginner guide covers the first conversation and common terms.

The consolidated branch was pushed, initially with presentation commit
`1809c8b108df95566f4e90e866afffb1d46d50d6` and evidence commit
`a5470db79be7109f86fbcce80d7a57e906dbb9ca`. About text and topics are updated.
Actual hosted desktop and 320-pixel mobile banners loaded correctly and were
visually checked. Local light/dark, narrow text reflow and keyboard checks also
pass. The social preview PNG is prepared; its repository setting is unchanged.

## Final release sequence

1. Review the final update logs and pass documentation/version checks.
2. Open the consolidated PR, inspect checks and merge into current `main`.
3. Prove the integrated product inputs match the qualified package source.
4. Stage all eight original packages, signatures, checksums and provenance in
   the versioned GitHub release; verify the uploaded assets before publishing.
5. Verify the public release, download links and updater feeds; record the
   exact merge/tag/release identities and any remaining qualification limits.
