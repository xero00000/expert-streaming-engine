# Ubuntu candidate validation — 2026-10-13

Unsigned ESE Studio `0.2.1` CPU candidates build and execute in an isolated
Ubuntu 24.04 container. This qualifies the tested container environment and
package behavior. GitHub workflow execution, signed updates and the complete
release gate remain unqualified.

Evidence: `/home/xero/Desktop/test/results/ese-ubuntu-release-20261013`.
All 3,375 recorded source inputs and both native binary directories are
preserved separately from the mutable build directories.

## Environment and builds

The container uses the pinned `nvidia/cuda:12.8.0-devel-ubuntu24.04` image,
Ubuntu GCC 13, Node 22.22.2, pnpm 10.34.7 and Rust 1.96.0. Frontend dependencies
use the frozen lockfile. Rust dependencies use a separate Cargo home, and the
Rust build directory starts empty. No host packages or user configuration are
changed.

Complete CPU and CUDA Release builds pass. The native suites pass 91/92 CPU
tests and 95/96 CUDA tests. Both retain only the initialized DeepSeek block
comparison failure. CUDA targets include Turing and Ampere. Normal Qwen
loading remains gated; these results do not establish original Qwen parity.

The frontend production build, all 19 frontend tests, Rust formatting, all
27 Rust tests and Clippy with warnings denied pass in Ubuntu.

## Packaging and execution

The build produces DEB, RPM and AppImage CPU candidates. Every candidate
passes version, notices, desktop-file, launcher/doctor and native server
checks. Native libraries load from the package, with no build or Conda path
fallback. HTTP inference matches the retained synthetic teacher exactly,
including all 32 probabilities at context 512, after shrinking to 256, and
after restoring 512.

The DEB installs successfully inside the container. Every package-owned
installed file matches the extracted candidate. The installed application,
extracted RPM application and actual AppImage entry point each remain live
for at least 12 seconds on an isolated virtual display. Captures show
onboarding with sharing off and the bundled runtime ready. Configuration,
data and cache directories are isolated. The DEB is removed after validation.

Launcher/native/API and all 32 probabilities with shrink/regrow also pass as
Ubuntu's ordinary UID 1000 account for all three formats. The DEB and RPM
checks use protected installed/extracted payloads; AppImage uses that user's
own extraction. The installed DEB, RPM and actual AppImage entry points also
pass GUI startup as this user. A root smoke run's private temporary AppImage
extraction initially prevented the other user from reusing that temporary
path; removing the owned smoke residue resolves that observer collision.

The minimal container exposed missing AppImage build dependencies: `xdg-open`
and `file`. Installing `xdg-utils` and `file` resolves those failures; release
packaging now declares both packages. The container has no FUSE device, so
build tools use `APPIMAGE_EXTRACT_AND_RUN=1`. The existing `NO_STRIP=1` setting
is retained. These settings affect build tooling, with no application change.

RPM header and payload SHA256 checks pass. Ubuntu's RPM 4.18.2 `rpm2cpio`
emits the complete payload but exits 1 because this generated package omits
the archive-size tag it compares against. This behavior follows the
[RPM 4.18.2 converter source](https://github.com/rpm-software-management/rpm/blob/rpm-4.18.2-release/rpm2cpio.c).
The recommended `rpm2archive` converter exits 0, and its extracted application
passes all native/API/GUI checks. RPM's maintainers
[recommend that converter](https://github.com/rpm-software-management/rpm/discussions/2761).
RPM installation on a target RPM distribution is outside this Ubuntu check.
Failed preliminary builds and observer runs remain recorded and excluded.

## CUDA memory checks

Primitive controls pass on the RTX 3080, RTX 3060 Ti and RTX 2080 SUPER.
CUDA's enumerated order in this container differs from the NVML listing;
device names and UUIDs identify the hardware used by the checks.

With all three GPUs visible, NCCL 2.25.1 initializes process-wide main and
pair communicators. All three memory-check runs report retained NCCL
allocations at process exit. Source inspection finds no communicator teardown.
The same primitive checks with each individual GPU visible report zero errors
and zero leaked bytes. This isolates the observed shutdown leak to the
multiple-GPU NCCL path; that memory gate remains failed.

## Qualification limits

The candidates are unsigned and contain the CPU backend. Linux CUDA desktop
package execution, Windows CPU/CUDA installers, signed updater artifacts,
GitHub execution and publication are not qualified by this report. The
DeepSeek block comparison and NCCL shutdown memory gates remain failed.
Original Qwen numerical parity, full 64K execution, concurrent serving and
matched Strata evidence remain open. No model arithmetic, reference weights
or numerical tolerances changed in this validation.
