# Release package contract

The source of package names and release metadata is
`tools/release_assets.py`. For version `VERSION`, the public package set is:

| Platform/runtime | Package filenames |
| --- | --- |
| Linux CPU | `ESE-Studio_VERSION_linux-x86_64_cpu.AppImage`, `.deb`, `.rpm` |
| Linux CUDA | `ESE-Studio_VERSION_linux-x86_64_cuda.AppImage`, `.deb`, `.rpm` |
| Windows, CUDA plus separate CPU fallback | `ESE-Studio_VERSION_windows-x64-setup.exe`, `ESE-Studio_VERSION_windows-x64.msi` |

The Linux jobs build independent native runtimes with `GGML_NATIVE=OFF` and
`ESE_QWEN4EXP_PROBE_GATE=OFF`. The CUDA job uses checksum-pinned CUDA 12.8.0
components and NCCL 2.25.1 from NVIDIA, including the runtime versions retained
in the existing Linux packaging notices. The Windows job retains its CUDA 12.4
toolchain and separately builds the driver-independent CPU runtime.

Windows staging also resolves the native binaries' transitive Microsoft C++
and OpenMP imports from the installed Visual Studio desktop x64 release redist.
It bundles those unmodified files beside each native server and places Studio's
CRT resources beside its executable through `tauri.windows.conf.json`. Each
directory has an `ese-msvc-runtime.json` size/hash inventory. Missing imports
or altered inventory files fail verification; System32, debug and OneCore
directories are not redistribution sources. The build uses Python/pefile
(provided by the pinned PyInstaller dependency) to inspect PE files; the
installed launcher does not require a separate Python installation.

`ESE_MSVC_REDIST_DIR` can select a release redist version directory or its `x64`
subdirectory. Otherwise staging uses the developer-prompt redist directory or
the latest installed Visual Studio C++ toolchain. App-local Microsoft runtime
updates must be shipped with an ESE package update. See Microsoft's
[deployment methods](https://learn.microsoft.com/en-us/cpp/windows/choosing-a-deployment-method)
and [redistribution documentation](https://learn.microsoft.com/en-us/cpp/windows/redistributing-visual-cpp-files).

## Collection and release preparation

`collect-release-assets.py` selects only the requested version from each
bundler output tree. Linux requires `--variant cpu` or `--variant cuda`;
Windows contains both runtimes and takes no variant flag. All invocations
provide the full source commit and staged runtime directory. A mismatched
runtime marker, ambiguous bundle or missing/empty AppImage/NSIS signature is
rejected before copying. Existing nonempty output is never replaced.

Each job emits an `artifact-set-….json` index with version, platform, variant,
source commit, package/signature sizes and hashes. `prepare-release-assets.py`
checks all three indices, the complete eight-package set and every retained
digest against the requested source commit. Missing, altered, stale or
unexpected assets fail before release metadata is written. Signature
sidecar matching/presence is checked here; cryptographic and installed-update
qualification is a separate release task.

The generated metadata consists of `SHA256SUMS`, `SHA256SUMS-windows.txt`,
`release-manifest.json`, `latest-cpu.json`, `latest-cuda.json` and `latest.json`.
The checksum lists include both package payloads and updater signature files.

## Update routing

Linux release packages are built through `pnpm package:linux`, which selects a
Tauri configuration overlay from the native build's CUDA setting. Both variants
retain the same application identity, configuration paths and updater key.
The CPU overlay uses `latest-cpu.json`; the CUDA overlay uses
`latest-cuda.json`. Both manifest URLs point to the corresponding AppImage.
Windows update entries point to the dual-runtime NSIS installer in every feed.

The legacy `latest.json` retains CPU behavior for earlier Linux CPU releases.
Previously prepared unsigned CUDA packages are not claimed to have a qualified
automatic migration path. DEB/RPM updates remain package-manager/manual
installations of the matching variant. Signing and actual installed-update
tests remain required before enabling a release channel.

Studio reads Tauri's actual bundle type before checking for an automatic
update. DEB/RPM installations show package-manager instructions and do not
contact the AppImage feed. An unidentified format also uses manual updates;
AppImage and both Windows installer formats retain signed automatic updates.

Tauri documents [static updater metadata](https://v2.tauri.app/plugin/updater/#static-json-file)
and [configuration overlays](https://v2.tauri.app/develop/configuration-files/#extending-the-configuration).
The toolchain component lock records its authoritative NVIDIA manifest; the
archive installation approach follows NVIDIA's
[binary archive documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#tarball-and-zip-archive-deliverables).

No publication occurs merely by preparing these files locally. The user's
publication hold remains active.
