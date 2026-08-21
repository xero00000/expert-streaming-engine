# ESE Studio

ESE Studio is the Linux desktop control center for Expert Streaming Engine. It replaces the legacy launcher presentation while keeping ESE as the model-planning and serving authority.

This foundation includes:

- automatic GGUF discovery from user-configured folders;
- Hugging Face GGUF search, hardware-fit quant recommendations, and
  revision-pinned resumable downloads with speed and ETA;
- available models in the primary view and missing manual profiles in a collapsed **Unavailable profiles** section;
- user-configurable CLI app profiles with persistent embedded PTY terminals;
- Quick, Standard, and Exhaustive measured sweeps with a safe 90% context promotion default;
- per-trial checkpoints, cancellation/resume, exclusive-GPU safety, and active-model restoration;
- one-click promotion of verified context, KV, and batch settings into future model launches;
- optional, off-by-default sharing of sanitized verified-sweep summaries through **Help improve ESE**;
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

## Linux source install

Run:

```bash
./install.sh
```

The installer checks commands and system libraries first. When required packages are missing on a supported distribution, it prints them and asks before using `sudo`; it never installs packages silently. NVIDIA tools and the `ese` launcher are reported separately because the desktop shell can build without them. The local release build produces DEB and RPM packages; an AppImage should be built in a compatible release container because newer Fedora/Nobara RELR libraries exceed the `linuxdeploy` strip tool used by Tauri.

To run only the non-mutating preflight, use `./install.sh --check`.

For development:

```bash
pnpm install
pnpm tauri dev
```

To install a release build for the current user, including an application-menu entry and Wayland-safe launcher wrapper:

```bash
./scripts/install-local.sh
```

## Configuration

Studio creates no configuration file until settings are saved. Its defaults discover `~/models` and `~/.local/share/ese/models` when those directories exist, and expose installed Codex, Claude Code, and OpenCode commands as app templates.

The first Studio launch presents one optional **Help improve ESE** switch. It
can be changed later in Settings. Turning it off also removes locally queued,
not-yet-uploaded benchmark summaries. Shared payloads exclude prompts,
responses, usernames, hostnames, local paths, raw logs, and raw error text.
Turning sharing off cannot retract an anonymous result already accepted by the
private collector or a grouped statistic already published on GitHub.

The TOML format is versioned and designed to remain editable outside the GUI. Both camelCase and snake_case field names are accepted when reading. Studio writes camelCase. Commands and paths live in TOML; secrets must use environment-variable references or the Linux keyring rather than plaintext values.

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

## Supported platform

Linux is the first supported platform. The React UI and Rust backend are cross-platform by construction, but Windows packaging and process behavior are deferred until the Linux model/app/sweep workflows are stable.

## Release validation

```bash
pnpm install --frozen-lockfile
pnpm build
cargo fmt --manifest-path src-tauri/Cargo.toml --all -- --check
cargo clippy --manifest-path src-tauri/Cargo.toml --all-targets --all-features -- -D warnings
cargo test --manifest-path src-tauri/Cargo.toml
pnpm tauri build
```

GitHub Actions repeats these checks on Ubuntu and publishes the DEB and RPM as
CI artifacts. The tagged release workflow attaches both packages and a
`SHA256SUMS` file to the GitHub release.
