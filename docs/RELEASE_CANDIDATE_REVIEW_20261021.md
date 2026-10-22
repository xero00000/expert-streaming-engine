# ESE 0.2.1 candidate review — 21 October 2026

The user authorized merging and publishing ESE 0.2.1 after the release-log
review. All eight packages, source indices, checksums and updater feeds pass
local software qualification. Physical Windows NVIDIA inference is explicitly
deferred and remains unqualified. The release retains original package build
identities and records the integration commit separately.

Product/package source freeze:
`2ff6389928fb701ad9331275682529cf5896645b` on
`review/ese-consolidated-20261020`. It includes the Windows app-local Microsoft
runtime repair, the DEB/RPM manual-update guard and bounded MSZIP cabinets.
Later qualification documentation does not change the package source freeze.
Earlier candidates retain their original source indices and evidence.

## Current results

| Area | Verified scope |
| --- | --- |
| Consolidation/source | Tasks 1–3 in the work log; scoped source, inference and recovery acceptance pass |
| Selected Flash-Next speed | User accepted approximately 19 tok/s; further optimization deferred |
| Workload scope | Two 2K/256 and one 8K/256 retrieval/replay requests pass; long passes deferred |
| Final Linux package set | Six actual CPU/CUDA AppImage/DEB/RPM packages; all 24 native payload identities, dependency/notices checks, bundle-type initializers, variant feeds and keys pass |
| Linux install behavior | Four actual Ubuntu DEB/Fedora RPM installations, ten short inference cases and six Settings views pass on `e991c3932`; the final six GUI binaries, both server text sections and all native shared libraries match that qualified build exactly |
| Final AppImage updates | Both real signatures verify; final CPU/CUDA signed replacements preserve exact payload hashes, mode 0755 and UID 1000; both updated GUIs and enabled update actions are visually checked |
| Update failures | Eight production-updater integration controls on `e991c3932` cover same-version rejection, transport/corruption failures, write-failure rollback and valid retries; final-payload replacements are checked separately |
| Windows builds | CPU, CUDA 12.4, standalone launcher, frontend and Tauri CLI/custom-protocol Studio builds pass; native CUDA finishes in 543.31 seconds |
| Windows MSI | Real 1,048,719,360-byte package has three embedded cabinets; installation, 33 exact runtime files, GUI CRT inventory, automatic CPU selection, exact top-32 CPU inference and normal uninstall pass |
| Windows GUI | Actual MSI application opens Settings in the clone's interactive session with community sharing off and its bundled launcher selected; UI Automation confirms the update action. Its first screenshot is occluded by a Windows upgrade prompt; the final post-update NSIS Settings view is visually verified after declining that prompt only in the owned clone |
| Windows NSIS | Real 654,163,835-byte installer completes in 808.71 seconds; actual install returns 0 in 66.80 seconds; 33 exact runtime files, CPU fallback and exact top-32 inference pass |
| Windows signed update | Seven download/format controls plus real signed passive-mode reinstall pass; installer exit 0, deliberate license-file repair, all 33 runtime identities, CPU inference and post-update Settings startup verified |
| Windows physical NVIDIA | Deferred by the user for this candidate; inference remains unqualified. The host IOMMU constraint is unchanged; no GPU is detached or Windows NVIDIA driver installed |

Inference controls use the pinned TinyMoE fixture, twelve prompt IDs, one
prediction and top-32 probabilities against the matching unpackaged runtime.
They do not establish full-vocabulary arithmetic, large-model qualification,
filled 1M context or sustained throughput. Approximately 19 tok/s remains the
accepted experimental Flash-Next baseline. Production Qwen admission stays OFF.

The final Linux refresh changes the server build banner; its executable code
and shared libraries match the installed/inference-qualified predecessor.
Current packages are independently rebuilt, collected, hash checked and
signature verified. No old artifact index is relabelled as the current source.
DEB/RPM Settings show manual package-manager instructions and contact no
AppImage feed; AppImage and Windows retain signed automatic update checks.

Updater API tests use unmodified `tauri-plugin-updater 2.10.1`, the configured
public key, local HTTP feeds and actual signed payloads. A test-only comparator
permits same-version candidate replacement; the default comparator separately
rejects it. These tests do not claim a published-channel or earlier-public-key
migration. Windows Authenticode and native DEB/RPM package-manager signing are
not claimed.

## Current retained package hashes

| Package | SHA-256 |
| --- | --- |
| `ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage` | `75f8efa70bfa61ba2c51aa9b45bfa45caf488559b4ec09484ac5e9d7451253a9` |
| `ESE-Studio_0.2.1_linux-x86_64_cpu.deb` | `87996828b1f841f7b3b71ef60bb440f76cc8f8d505a6232ea2be62bc40ce1eb0` |
| `ESE-Studio_0.2.1_linux-x86_64_cpu.rpm` | `6d723a630280f1f8c162d025788bff1f184ed3320af68cec659b846ca81d69aa` |
| `ESE-Studio_0.2.1_linux-x86_64_cuda.AppImage` | `20ad4f68f60d127c62d54c9d504b442dee2642b0e8790a08feedc72e1a7872de` |
| `ESE-Studio_0.2.1_linux-x86_64_cuda.deb` | `9ca80374d1d45d9b2caa2f23f276f82a77768b23357a3ea36c7acc373d1b78e6` |
| `ESE-Studio_0.2.1_linux-x86_64_cuda.rpm` | `c8ac133b83ddfb4d517d58c7aea42145a19429cc22bcaa5e91f84b29e50847cb` |
| `ESE-Studio_0.2.1_windows-x64-setup.exe` | `6044551b526240ef26908d4aff744de2293b9121069a78e45b3f5540f1a90634` |
| `ESE-Studio_0.2.1_windows-x64.msi` | `efd3ec48fe93ec9a37ec8fa87ffa7b1e3004262cc8122ae9e2e876baaf20769f` |

The complete `SHA256SUMS` covers eight packages and three signature files.
`release-manifest.json`, both variant feeds, the legacy CPU feed and Windows
checksums are in `final-2ff638992-release-assets/`. All three artifact-set indices
identify the same full source freeze. The original local staging directory is retained; release publication uses its
verified bytes and unchanged source indices.

## Publication authorization and deferred qualification

The user explicitly instructed **“Lets skip that test for now”** in response
to the physical Windows NVIDIA test question. It is deferred for this candidate
and remains unqualified. The spare GPUs still share IOMMU group 15 with mounted
storage, USB and Ethernet; no passthrough or host device change was attempted.
This decision supersedes the earlier unresolved scope question.

All agreed local qualification tasks are complete. The user subsequently
authorized PR integration and publication. [PR #23](https://github.com/xero00000/expert-streaming-engine/pull/23)
records the merge checks. The lengthy hosted Windows rebuild is redundant with
the actual local package/install/update qualification and is cancelled under
the user's instruction to avoid multi-hour repeat passes; it is not claimed
as a hosted pass. Shorter merge checks are recorded separately. Long workload
checks remain deferred and production Qwen admission stays OFF.

## Evidence

Paths are relative to the private evidence root
`/home/xero/Desktop/test/results/ese-release-work-20261020/`:

- `task6/linux-inspection/final-2ff638992/qualification-index.json`: current
  Linux static, native identity, unchanged behavior and updated-GUI evidence.
- `task6/ubuntu-build/final-2ff638992-assets-{cpu,cuda}/`: final Linux packages
  and original per-variant source/hash indices.
- `task6/final-2ff638992-linux-signatures.json`: both final signatures and all
  eight package/signature digests.
- `task6/ubuntu-build/update-checks-final-2ff638992/`: actual final CPU/CUDA
  signed replacement, byte/mode/UID checks and post-update Settings captures.
- `task6/linux-inspection/final-e991c3932/installed-summary.json`: actual
  Ubuntu/Fedora installation and short CPU/GPU inference matrix.
- `task6/ubuntu-build/update-checks-final-e991c3932/`: complete eight-case
  failure/retry updater API controls, raw requests, helpers and logs.
- `task6/windows-isolated/build-cuda-background-03.execution.json`: successful
  bounded Windows CUDA build.
- `task6/windows-isolated/source-seal-2ff638992.json`: all 3,335 tracked source
  files, including symbolic-link identities, verified in the isolated clone.
- `task6/windows-isolated/transfer-final-msi.json`: retained actual MSI hash.
- `task6/windows-isolated/final-2ff638992-evidence/`: interactive MSI evidence;
  screenshot occlusion is recorded explicitly.
- `task6/windows-isolated/physical-gpu-iommu-preflight.json`: actual IOMMU group
  membership and the Linux VFIO ownership constraint.
- `task6/windows-isolated/final-2ff638992-evidence/`: copied Windows install,
  inference, signed-update and process evidence; the final NSIS Settings capture
  is visually verified. Both earlier occluded captures remain identified.
- `final-2ff638992-linux-process-cleanup.json`: all five owned container
  cgroups are empty after final cleanup; detached DBus children are covered.
- `task6/windows-isolated/final-nsis-signature.json`: real NSIS cryptographic
  verification; only its public signature is sent back to the clone.
- `final-2ff638992-release-assets/`: complete eight-package staging, three
  original source indices, 11 package/signature digests and six metadata files.
  Original asset identities remain unchanged through release upload.

See the [work log](RELEASE_WORK_LOG_20261020.md) for retained failures and the
[package contract](RELEASE_PACKAGE_CONTRACT.md) for the required deliverable.

## Local PR and tag drafts

Prepared PR title: **Prepare ESE 0.2.1 consolidated runtime and package release**

Prepared PR body:

> Consolidate the reviewed ESE changes, repair cache and recovery correctness,
> and produce Linux CPU/CUDA packages plus Windows installers with app-local
> Microsoft runtimes. DEB/RPM updates use the package manager; signed automatic
> updates preserve the AppImage runtime variant and update Windows NSIS installs.
>
> Local source, scoped inference/recovery, all eight packages and signed-update
> checks pass as documented above. The user accepts approximately 19 tok/s for
> the experimental Flash-Next profile. Long passes and physical Windows NVIDIA
> inference are deferred; Windows GPU inference remains unqualified and
> production Qwen admission stays OFF. Hosted release CI has not run.

Prepared tag name: `v0.2.1`. Product/package source freeze:
`2ff6389928fb701ad9331275682529cf5896645b`. Later commits contain qualification
documentation only. The final tag target must be the reviewed integration commit
that retains this product freeze and the final scope documentation.

Prepared tag message:

> ESE 0.2.1: consolidated runtime, CPU/CUDA Linux packages and dual-runtime
> Windows installers. Local installation and signed-update checks pass. Physical
> Windows NVIDIA inference and extended workload qualification are deferred;
> experimental Qwen admission remains disabled in production builds.

The user authorized using the reviewed content for PR integration and release
publication. The work log records the actual PR, tag and release outcome.
