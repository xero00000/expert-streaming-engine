# ESE Studio architecture

ESE Studio is a Tauri 2 desktop application in `studio/`. React owns presentation and Rust owns filesystem discovery, portable configuration, PTYs, process lifecycle, Hugging Face access/downloads, and sweep execution.

## Boundary with ESE

The GUI does not duplicate model-placement policy. ESE remains the control plane for hardware inspection, deterministic planning, and `llama-server` startup. Studio invokes ESE planning and serving, retains the exact logs and measured sweep checkpoints, and presents them in the UI.

The old Bash launcher is migration input, not the new backend. App definitions and manually named model profiles move into versioned TOML. This avoids carrying its hard-coded paths and interactive shell presentation into the desktop application.

## Process model

- Each CLI app runs in its own native PTY and xterm.js tab.
- Terminal sessions stay alive when the selected model endpoint changes.
- CLI sessions are independent from the managed model process and are not killed when the selected model changes.
- Model serving is supervised through a dedicated PTY with visible plan and server logs.
- Sweeps require confirmed exclusive GPU access. Studio stops its active model, refuses to interfere with unmanaged `llama-server` processes, checkpoints every completed trial, and restores the prior model afterward.
- The exclusive preflight aggregates `nvidia-smi` compute allocations by PID and refuses external CUDA workloads using 512 MiB or more. Small compositor/browser allocations remain allowed.

## Model hub and downloads

The Rust backend calls the public Hugging Face Hub model API for GGUF-tagged
search and blob metadata. Quant recommendations use the same `ese doctor --json`
hardware inventory that informs ESE planning: aggregate VRAM receives a 1 GiB
reserve per GPU and an additional conservative fit margin, while hybrid fit is
bounded by currently available RAM. This is a download recommendation, not a
replacement for the authoritative launch-time ESE plan.

Download requests are restricted to configured, canonical model roots. The
backend pins the returned repository revision, validates every relative path,
checks disk headroom, refuses symlink replacement and conflicting completed
files, verifies expected byte sizes, and atomically promotes `.part` files only
after completion. Range requests resume partial transfers. A single managed
download emits throttled progress, transfer speed, and ETA events and can be
cancelled without discarding resumable data. Authentication is read only from
`HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`.

## Sweep policy

Simple mode optimizes in this order:

1. find the verified maximum stable context;
2. promote 90% of that maximum as the safe default;
3. optimize stable throughput at the promoted context.

Advanced mode exposes selectable objectives while sweep depth controls the KV, batch, and repetition search space. Exhaustive work is resumable: a matching cancelled or failed checkpoint is loaded and completed trials are skipped.

Preview and measured states are intentionally distinct. A preview is a generated trial matrix and can never be promoted as benchmark evidence.

## Community benchmark privacy boundary

Community sharing is explicit opt-in, off by default, and offered during the
first Studio launch with the same **Help improve ESE** switch in Settings. Only
a sweep in the verified `complete` state may enter the local outbox. Turning
the switch off deletes queued, not-yet-uploaded summaries.

Sanitization happens in the Rust client before HTTPS upload. Payloads contain
hardware/model categories, tested settings, stable throughput, and reduced
failure classes; they exclude prompts, responses, usernames, hostnames, local
paths, environment variables, tokens, logs, and raw errors. A random local
installation identifier is one-way hashed with a private server salt before
storage.

The Cloudflare Worker exposes a write-only submission route backed by a private
D1 database. Its read route requires a secret shared only with GitHub Actions
and returns grouped statistics rather than rows. Groups with fewer than three
verified submissions are suppressed. The scheduled GitHub publisher can
therefore update `COMMUNITY_BENCHMARKS.md` and `benchmarks/community.json`
without possessing a raw-data export capability.

## Configuration and secrets

The versioned user config is `studio.toml` under the operating system's
application-config directory. On Linux this normally resolves below
`$XDG_CONFIG_HOME/ese`, or `~/.config/ese` when `XDG_CONFIG_HOME` is unset;
Windows uses the user's roaming AppData directory. The Settings screen exposes
the exact resolved path. Model roots are auto-scanned; manual profiles override
discovered metadata. Missing profiles are preserved for removable drives but
hidden in a collapsed unavailable section by default.

Configuration is portable. Secrets are not: app profiles should reference
environment variables or the operating system's credential store rather than
storing plaintext tokens.

## Current desktop foundation

- Desktop shell, recursive discovery, hidden missing profiles, native PTYs, and typed portable config.
- Installed-agent discovery across PATH and platform package locations,
  including Linux user directories and Windows npm/Scoop paths, with
  non-destructive profile merging.
- Active-model handoff to endpoint-aware apps through OpenAI-compatible and ESE-specific environment variables; Hermes also receives persistent provider synchronization.
- ESE plan/serve supervision and real completion-based benchmark trials.
- Stability gates, checkpoint/resume, cancellation, active-model restoration, and verified-profile promotion.
- Unified Linux and Windows packaging: Studio carries the ESE launcher, native
  server, and required ESE libraries as one versioned runtime payload.
- Hardware-aware Hugging Face GGUF browsing and safe resumable downloads.
- Off-by-default community benchmark sharing with a private raw collector and privacy-thresholded public aggregates.
- Linux AppImage/DEB/RPM and Windows NSIS/MSI builds, with signed updater
  artifacts generated only by the protected tag-release workflow.
- Manual update checks, visible download progress, signature verification, and
  restart installation through the stable GitHub release manifest. The full
  in-app replacement flow targets Linux AppImage and Windows NSIS installs;
  native Linux packages continue through APT/DNF.
