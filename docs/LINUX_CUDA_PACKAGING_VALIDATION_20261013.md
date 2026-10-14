# Linux CPU and CUDA desktop packaging validation — 2026-10-13

## Scope and result

The current unsigned x86_64 ESE Studio 0.2.1 CPU and CUDA candidates qualify
for the installation, loader, launcher, small-model API/cache, and desktop
startup checks below. Both variants were built on Ubuntu 24.04. Actual DEB
installation was checked on Ubuntu 24.04 and actual RPM installation on
Fedora 44. AppImages ran using `--appimage-extract-and-run`.

Evidence is preserved under
`results/ese-linux-cuda-packaging-20261013`. The prior sealed stages remain
unchanged. The original compressed RPM candidates are superseded because
independent inspection found incorrect raw-payload checksums; extraction and
startup in the earlier stages were insufficient to qualify RPM integrity.

These checks use the generated three-layer, 32-token-vocabulary model and
saved CPU or same-physical-GPU controls. They do not establish original Qwen
model correctness, long-context behavior, throughput, or release-wide readiness.
The normal Qwen architecture gate remains OFF.

## Reproduced packaging defects

1. The initial CUDA DEB lacked `libcudart.so.12`, `libcublas.so.12`,
   `libcublasLt.so.12`, and `libnccl.so.2`. Its native server exited 127 in the
   isolated Ubuntu environment after the corresponding SDK packages were
   removed. A developer machine with the toolkit installed concealed this gap.
2. Initial AppImage dependency collection copied the NVIDIA driver into the
   package and modified vendor-library RPATHs. The first exclusion adapter
   also missed the GTK plugin's second dependency-collection invocation.
3. DEB/RPM metadata did not explicitly require the Python launcher and OpenMP
   runtime. The current DEB requires Python 3.10 or later and `libgomp1`; the
   RPM requires `python3` and `libgomp`.
4. The pinned Tauri 2.11.4 bundler uses rpm 0.16.0. Its compressed-payload
   SHA writer hashes the whole supplied buffer before the underlying writer
   reports how many bytes it consumed. Partial writes therefore hash repeated
   bytes and produce an invalid raw-CPIO digest. A standalone reproducer and
   the retained old CPU RPM independently demonstrate this defect. The current
   RPM configuration uses uncompressed payloads; both payload digests now match.

The diagnosed initial CUDA RPM build was explicitly terminated after the
checksum defect was reproduced. Its incomplete output is not a candidate.
All negative artifacts and observer setup errors are retained separately from
successful qualification evidence.

## Current packaging behavior

Linux staging resolves the CUDA runtime dependencies and copies the four
vendor shared libraries under their SONAME filenames. Symlinks are resolved
to ordinary files. Only project ELFs receive `$ORIGIN` RPATHs. Vendor files
remain byte-identical to their installed source libraries.

The Linux packaging entry point creates an isolated Tauri tool cache, checks
the pinned linuxdeploy executable hash, and routes both its outer invocation
and the GTK plugin callback through driver/runtime exclusions. A patchelf
adapter preserves vendor libraries and rejects other vendor modifications.
Actual artifact inspection verifies these controls; source tests alone are
not treated as proof of the resulting package contents.

The CUDA packages contain cudart 12.8.57, cuBLAS/cuBLASLt 12.8.3.14, and
NCCL 2.25.1. CUDA, NCCL and NVTX notices/licenses are included alongside the
ESE license and existing Windows CUDA 12.4 notice. License provenance is
preserved from the official versioned NVIDIA redistribution manifest/archive
and versioned NCCL source. No NVIDIA driver is redistributed.

CPU packages contain no CUDA, NCCL, or NVIDIA driver libraries and operate
without NVIDIA device bindings. CUDA packages require a compatible host
NVIDIA driver. Their CUDA-disabled CPU control was checked with the host
driver present; use the CPU package on systems without that driver.

Linux packages use the system Python launcher. AppImages also require host
Python 3.10 or later, OpenMP, and compatible graphics/WebKit system support.
The RPM payloads are larger because compression is disabled to avoid the
pinned bundler's checksum defect.

## Qualified artifact identities

| Variant | Format | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| CPU | deb | 23515462 | `1187e64c9f4c4871e595ececfb2931e843e7bdf2b2c6f6d05bf0c276c5ff7994` |
| CPU | rpm | 67384093 | `e14038b376fd9824e3e87685524e06c354be70575b77cab1255efa95a4b2820c` |
| CPU | AppImage | 97405432 | `ffcee4a3708e843a12ed3c6f5355a13670ee0c62aca9e0a1dfdaeb0ea14b0167` |
| CUDA | deb | 1281588378 | `0b12305f329db61b561d0dc9911efcf3fe3f11e18cd42fa4aad19223ba784cbc` |
| CUDA | rpm | 1724163529 | `056a0fcbb9614149f78b948ab23d059bf1c7051e3d5cebb9b95061213f0a6c85` |
| CUDA | AppImage | 1311881720 | `9d58534ee2b84223f42b7731025ca9f417cb0b8d9352361abf983074dfee0438` |

## Installation and loader checks

- All six artifact versions are 0.2.1; desktop entries validate.
- All project ELF dependencies resolve from the package or ordinary system
  libraries. No mutable build-output or CUDA SDK search path is needed.
- Each CUDA format contains exactly the intended vendor-library copies in the
  native runtime directory. Their four hashes match the unmodified originals.
  No NVIDIA driver library is present anywhere in an extracted artifact.
- All CPU formats have `cuda: false` runtime markers and pass doctor/launcher
  version/native version checks in Ubuntu with the toolkit, driver libraries,
  and NVIDIA device bindings absent.
- DEB installation succeeds for CPU and CUDA in separate owned Ubuntu
  containers. Fedora RPM installation succeeds for both variants; `rpm -V`
  reports no changed installed files.
- RPM header, payload, and raw-payload integrity checks pass. An independent
  streaming parser also verifies both CPU and CUDA payload SHA-256 values.
  Both rpm2archive and tar extraction complete successfully.

## Packaged runtime matrix

| Environment and artifact | Placements | Resident checks | Cache shrink/regrow checks |
| --- | --- | ---: | ---: |
| Ubuntu CUDA DEB/RPM/AppImage | CUDA disabled; RTX 3060 Ti; RTX 2080 SUPER; RTX 3080 | 12 | 12 |
| Ubuntu CPU DEB/RPM/AppImage, no NVIDIA driver | CPU | 3 | 3 |
| Fedora installed CUDA RPM | CUDA disabled; each of the same three GPUs | 4 | 4 |
| Fedora installed CPU RPM, no NVIDIA driver | CPU | 1 | 1 |
| Total | | 20 | 20 |

Every resident response retains all 32 saved probabilities and content exactly.
GPU selection is checked using the owned server PID, physical GPU UUID and
native resource device ID. Loaded project/vendor paths resolve to the selected
package. CPU controls expose no GPU resources. Resident launches without a
native resource plan correctly reject resource rebalance with HTTP 501.

Separate native cache-policy launches establish a real resource plan with a
nonzero host-graph metadata reservation. Their 512 → 256 → 512 cache-capacity
transitions preserve all 32 probabilities and content at each step. Forecasts,
resource snapshots, responses, process identities and terminal status are
preserved for each case.

## Desktop startup and source controls

The installed Ubuntu DEBs, extracted Ubuntu RPM payloads, actual AppImage files,
and both installed Fedora RPMs render ESE Studio as UID 1000 using isolated XDG
directories and an Xvfb display. Each remains alive for at least 12 seconds.
All eight screenshots show the expected application, a ready runtime and the
optional sharing setting OFF. These are startup checks, not a qualification
of every desktop interaction or a native Wayland session.

All 28 current Studio Node controls pass, including Linux dependency metadata,
CUDA runtime staging, vendor-protection rules, the GTK callback and CPU staging.
The packaging edits do not change native C++ or numerical source, model inputs,
references, tolerances, or architecture gates. Source manifests and an owned
patch preserve the scope independently of the existing dirty worktree.

## Limits

The candidates are unsigned and unpublished. Updater signing, native Windows
installation, original-model and release-wide qualification are incomplete.
Only Ubuntu 24.04 and Fedora 44 environments and the three named NVIDIA GPUs
are covered here. MIG, four-GPU operation, distribution-wide compatibility,
native Wayland, complete desktop workflows, and full-model performance are
not qualified by these checks.
