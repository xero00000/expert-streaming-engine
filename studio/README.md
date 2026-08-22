# ESE Studio

ESE Studio is the Linux and Windows desktop control center for Expert Streaming Engine. It replaces the legacy launcher presentation while keeping ESE as the model-planning and serving authority.

This foundation includes:

- automatic GGUF discovery from user-configured folders;
- a streaming local-model chat with stop, regenerate, clear, and persistent
  on-device conversation history;
- Hugging Face GGUF search, hardware-fit quant recommendations, and
  revision-pinned resumable downloads with speed and ETA;
- available models in the primary view and missing manual profiles in a collapsed **Unavailable profiles** section;
- user-configurable CLI app profiles with persistent embedded PTY terminals;
- Quick, Standard, and Exhaustive measured sweeps with a safe 90% context promotion default;
- per-trial checkpoints, cancellation/resume, exclusive-GPU safety, and active-model restoration;
- one-click promotion of verified context, KV, and batch settings into future model launches;
- optional, off-by-default sharing of sanitized verified-sweep summaries through **Help improve ESE**;
- a bundled ESE launcher/native runtime and signed in-app updates with visible progress;
- portable TOML configuration under `~/.config/ese/studio.toml`.

Sweep previews only describe the search matrix and never count as evidence. A verified result is shown only after Studio has launched the real ESE-planned `llama-server`, passed its health check, and measured completion throughput. Interrupted matching sweeps resume from their last per-trial checkpoint.

Before a measured sweep starts, Studio stops its managed model and checks for
unmanaged `llama-server` instances and external CUDA processes using at least
512 MiB in aggregate. A blocked preflight identifies the process and PID rather
than recording performance while another workload owns substantial VRAM.

## Hugging Face model hub

Search results are limited to repositories tagged GGUF. Studio reads repository
file metadata, excludes auxiliary `mmproj`/imatrix files, combines split model
shards, and labels each recognized quant as **Resident**, **Hybrid/cache**, or
**Streamed** using the local ESE hardware report. The recommendation is the
highest quality that preserves conservative GPU headroom, then the best option
within currently available RAM; advanced users can choose any listed quant.

Downloads may target any configured model folder. Studio checks a 5% free-space
margin, pins the repository revision, rejects paths escaping the model folder,
refuses symlink replacement, verifies expected file sizes, and retains `.part`
files when cancelled so the next attempt can resume. Existing complete files
are recognized rather than overwritten. Use `HF_TOKEN` or
`HUGGING_FACE_HUB_TOKEN` for gated/private repositories.

## Linux install

The v0.1.1 AppImage, DEB, and RPM packages contain both Studio and ESE. Use the
AppImage for signed in-app updating, or use the distribution package and apply
future package updates through APT/DNF.

For a local source build from the repository root, run:

```bash
./studio/scripts/install-local.sh
```

The installer checks commands and system libraries first. When required
packages are missing on a supported distribution, it prints them and asks
before using `sudo`; it never installs packages silently. It builds the ESE
runtime with CUDA when detected (CPU otherwise), installs it beside Studio,
and writes `ese-studio` plus `ese` launchers to `~/.local/bin`.

To run only the non-mutating preflight, use `./install.sh --check`.

For development:

```bash
pnpm install
pnpm tauri dev
```

## Windows source install

Open PowerShell in `studio` and run:

```powershell
.\install.ps1 -Check
.\install.ps1
```

The preflight checks Python, CMake, Node.js, Rust, the Visual Studio C++ Build
Tools, WebView2, and pnpm. When required software is missing, the installer
shows the complete list and asks before using `winget`; nothing is installed
silently. It separately asks before installing the PyInstaller build dependency,
then builds a standalone `ese.exe` and native runtime before producing
signed-ready NSIS (`.exe`) and MSI installers under
`src-tauri\target\release\bundle\`.

## Updates

Open **Settings → Updates** and choose **Check for updates**. Studio uses the
public release manifest on GitHub and accepts only artifacts signed by the ESE
updater key. The download reports progress, is verified before installation,
and restarts Studio when complete. Windows NSIS and Linux AppImage installs
support the full in-app replacement flow; DEB/RPM users should install the next
native package with their package manager.

## Configuration

Studio creates no configuration file until settings are saved. Linux defaults
discover `~/models` and `~/.local/share/ese/models`; Windows defaults discover
`%USERPROFILE%\models` and `%USERPROFILE%\Documents\ESE\models`. Installed
agent CLIs are discovered from platform-appropriate PATH, Cargo, Bun, npm, NVM,
and Scoop locations.

The first Studio launch presents one optional **Help improve ESE** switch. It
can be changed later in Settings. Turning it off also removes locally queued,
not-yet-uploaded benchmark summaries. Shared payloads exclude prompts,
responses, usernames, hostnames, local paths, raw logs, and raw error text.
Turning sharing off cannot retract an anonymous result already accepted by the
private collector or a grouped statistic already published on GitHub.

The TOML format is versioned and designed to remain editable outside the GUI. Both camelCase and snake_case field names are accepted when reading. Studio writes camelCase. Commands and paths live in TOML; secrets must use environment-variable references or the operating system credential manager rather than plaintext values.

```toml
version = 1
onboardingComplete = true
helpImproveEse = false
modelRoots = ["/home/me/models", "/run/media/me/model-drive"]
safeContextMargin = 0.9

[[apps]]
id = "codex"
name = "Codex"
command = "codex"
args = []
endpointAware = true
```

## Supported platforms

Linux packages are produced as DEB and RPM files. Windows packages are produced
as NSIS and MSI installers. The same GUI, configuration format, app discovery,
embedded terminals, model hub, and sweep workflow are used on both platforms.
The same local chat interface is included on both platforms and connects only
to the model server started by Studio.

## Release validation

```bash
pnpm install --frozen-lockfile
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml
pnpm tauri build --bundles deb,rpm --config src-tauri/tauri.unsigned.conf.json       # Linux source package
pnpm tauri build --bundles nsis,msi --config src-tauri/tauri.unsigned.conf.json      # Windows source package
```

GitHub Actions repeats these checks on Ubuntu and Windows and publishes the DEB,
RPM, NSIS, and MSI packages as CI artifacts. The protected tagged-release
workflow additionally builds the Linux AppImage, signs updater artifacts with
`TAURI_SIGNING_PRIVATE_KEY`, generates `latest.json`, and attaches all packages,
signatures, and platform checksum files to the GitHub release. The maintainer's
recoverable private-key backup is stored outside the repository with mode
`0600`; only the public key is committed.
