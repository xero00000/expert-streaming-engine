# ESE 0.2.1 candidate review — 21 October 2026

Unpublished local candidate. The eight-package release contract is incomplete.
Push, hosted CI, tag creation and publication remain on hold.

Retained initial Linux package source:
`e4b5739fc9380f401be3c548200a499a7f84ba34` on `review/ese-consolidated-20261020`.
The working branch now also repairs missing Windows Microsoft runtime staging.
The final eight-package set must include that repair; earlier Linux artifacts
and older Windows assets are not relabelled as the final source.

## Current results

| Area | Verified scope |
| --- | --- |
| Consolidation and source | Tasks 1–3 in the release work log; source checks and scoped inference/recovery acceptance pass |
| Selected Flash-Next speed | User accepted approximately 19 tok/s; further speed tuning deferred |
| Filled context | Two 2K/256 retrieval/replay requests and one 8K/256 request pass; longer passes deferred by user |
| Linux packages | Six actual CPU/CUDA AppImage/DEB/RPM; Ubuntu DEB/Fedora RPM installs; 24 exact relocated native payloads; 25 short inference cases; six GUI startups |
| AppImage signing/update | Configured key matches; two real signatures verify; eight updater API integration cases and both updated GUI startups pass |
| Windows | Current CPU native build, standalone launcher, frontend and Studio Rust build pass; CUDA 12.4 tools are installed and the CUDA build is partial; actual MSI/NSIS and physical GPU acceptance pending |
| Windows runtime dependencies | Missing app-local MSVC/OpenMP imports reproduced and repaired; six native DLL dependencies load from staging; exact top-32 output and normal shutdown pass; Studio CRT import closure passes |

Inference package controls compare one generated token and top-32 probabilities
against the corresponding unpackaged build with the same pinned model. They do
not establish full-vocabulary arithmetic or large-model/context qualification.

Updater tests use the unmodified production `tauri-plugin-updater 2.10.1`, the
configured public key and actual signed candidate AppImages. A test-only local
HTTP feed and comparator permit replacement of an older build with the same
0.2.1 version. Default version comparison separately rejects that same-version
update. Corrupted signature/payload, truncated transfer, HTTP failure and forced
write failure preserve the prior bytes and mode. A valid retry replaces them
with the exact candidate hash, retains mode 0755/UID 1000 and starts the GUI.
This is local API integration, not a published-channel or previous-public-
version migration trial. All six compiled app configurations contain only
their own CPU/CUDA feed plus the configured public key. Windows Authenticode
and native DEB/RPM package-manager signing are not claimed.

## Retained initial Linux artifact hashes

The private checksum file is
`task6/ubuntu-build/LINUX-CANDIDATE-SHA256SUMS.txt`; it includes the six Linux
packages and their two collected AppImage signature files. Full eight-package
metadata generation remains pending Windows. Package SHA-256 values:

| Package | SHA-256 |
| --- | --- |
| `ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage` | `36857f2f6da8519613b846773c5bdbcd04233f0282a5844f59234195c2573ed8` |
| `ESE-Studio_0.2.1_linux-x86_64_cpu.deb` | `a90054a818e9c790613871267fd5accbaceeddd709bf668a40fd2cfd7cd3269e` |
| `ESE-Studio_0.2.1_linux-x86_64_cpu.rpm` | `e8376ce14681cdcfb3eebfde0a0cda9457dabbfb6e9233e3ed8237e5ba5b3c1a` |
| `ESE-Studio_0.2.1_linux-x86_64_cuda.AppImage` | `6bc489866b25391076d56e18a28d75fbcc22919a898dcfe9d845201ee29c1389` |
| `ESE-Studio_0.2.1_linux-x86_64_cuda.deb` | `f42eb8b5d744d10437ec234f7b4f4c79eec7d4d32bedb773eeb1ba7e2ed78699` |
| `ESE-Studio_0.2.1_linux-x86_64_cuda.rpm` | `fb75b6facec3da295c6800a6d2fb7ce7a81c16b1698703680aa23df81311ce56` |

## Remaining work before release approval

1. Build current-source dual-runtime Windows MSI/NSIS, install both, and qualify
   CPU fallback, physical NVIDIA inference and Windows update handling.
2. Complete the eight-package collector/checksum/updater metadata acceptance
   against the final release source. Preserve the frozen-source distinction
   if final documentation/source changes require a new candidate build.
3. Reconcile final release notes and claims with the accepted short-check scope;
   retain experimental admission restrictions and deferred long workloads.
4. Present the complete local candidate for review; retain the user's publication
   hold until explicit approval. No publication command has been executed.

## Evidence

All paths below are relative to the private evidence root
`/home/xero/Desktop/test/results/ese-release-work-20261020/`:

- `task6/task6-progress.json`: frozen source and six real package builds.
- `task6/linux-inspection/linux-acceptance-summary.json`: package and installed-runtime checks.
- `task6/windows-isolated/preflight-summary.json`: isolated Windows inventory and stopped VM.
- `task7/linux-update-qualification.json`: signing, updater and post-install startup proofs.
- `task6/ubuntu-build/update-checks/`: raw updater requests, responses, failures, hashes and helpers.

The [release work log](RELEASE_WORK_LOG_20261020.md) retains failures and earlier
results in order; the [package contract](RELEASE_PACKAGE_CONTRACT.md) defines the
complete deliverable. Prepared local PR summary: consolidate ESE source and
release packaging, repair cache/recovery correctness, preserve CPU/CUDA update
variants, and document the measured/accepted performance and remaining platform
gates. The PR and tag have not been created.
