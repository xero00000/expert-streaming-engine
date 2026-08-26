# ESE Studio release validation record

This append-only record separates automated package/source evidence from
physical model-backed validation. A checked item requires either CI evidence or
a retained model-backed artifact; compilation alone is not inference evidence.

## v0.1.0 Linux foundation — 2026-08-21

### Automated gates

- [x] Frontend TypeScript and production Vite build.
- [x] Rust formatting and Clippy with warnings denied.
- [x] Rust backend tests, including model handoff, app discovery, sweep
  checkpointing, telemetry sanitization, and Hugging Face path safety.
- [x] Cloudflare binding generation and TypeScript checks.
- [x] Worker/D1 tests for bounded ingestion, schema validation, identity
  hashing, duplicate handling, authenticated export, and the three-sample
  privacy threshold.
- [x] Worker deployment dry run without publishing.
- [x] DEB and RPM production builds.
- [x] CI definition for clean Ubuntu DEB installation, headless startup,
  desktop-file validation, RPM content inspection, and package artifacts.
- [x] Tag-driven release definition for normalized package names and
  `SHA256SUMS`.

### UI regression — 2026-08-21

- [x] Models, Model hub, Apps, Config sweeper, and Settings open correctly.
- [x] Navigation icons and labels share a consistent alignment column.
- [x] Model families start collapsed, expand on demand, and expand while
  searching.
- [x] Missing profiles remain outside the primary available-model list.
- [x] Custom-app editor exposes name, command, arguments, and working directory.
- [x] Advanced sweep controls and measured-plan preview are reachable.
- [x] Settings uses two columns at 1280×720 and one column at the supported
  980×640 minimum.
- [x] Every view has visible keyboard focus and no horizontal overflow at both
  tested sizes.
- [x] The terminal drawer has pointer, double-click, and keyboard resizing plus
  explicit collapse and restore controls.

### Native and model-backed evidence

- [x] Recursive discovery is active across an internal model folder and a
  removable-drive model folder.
- [x] Codex, Claude Code, OpenCode, and Hermes were discovered from real Linux
  installations and persisted without replacing customized profiles.
- [x] Endpoint-aware handoff includes URL, key, model/GGUF identity, context,
  architecture, quantization, KV type, batch, ubatch, and resource-plan data.
- [x] A real 27B GGUF standard sweep completed 14/14 trials, rejected an
  unstable high context, verified 65,092 tokens, promoted 58,368 tokens, and
  selected q8_0/256 at 3.98 tok/s.
- [x] A separate real 27B sweep was cancelled after five trials and retained a
  resumable checkpoint, proving cancellation does not discard completed work.
- [x] Dense and sparse-MoE reference benchmarks, including multi-GPU Turing and
  Ampere results, are retained in `docs/ESE_BENCHMARKS.md`.
- [x] Production telemetry health, authenticated aggregate export, and cleanup
  of the smoke row were verified; production began with zero retained rows.

### Release limitations

- Linux x86-64 is the supported Studio target for v0.1.0.
- Packages are checksummed but not cryptographically signed.
- AppImage, automatic updates, Windows, Android/QNN, and Ada-or-newer physical
  CUDA validation are not claimed by this release.
- The Android/QNN work remains an isolated draft and is not part of v0.1.0.

## v0.1.1 maintenance release

- [x] Studio, launcher, and native server versions report `0.1.1`.
- [x] Extracted RPM contains the launcher, planner, native server, and required
  shared libraries; the native help surface runs from the extracted package.
- [x] Linux package runtime uses relocatable `$ORIGIN` library paths.
- [x] Tauri updater artifacts are mandatory-signed in the tag workflow and the
  recoverable private key is stored outside the repository with mode `0600`.
- [x] Stable GitHub `latest.json` generation covers Linux x86-64 AppImage and
  Windows x86-64 installer assets.
- [x] Settings exposes manual check, progress, verified install, failure
  recovery, and restart states without changing the existing visual system.
- [x] User-local Nobara installation reports `ese 0.1.1`, finds the installed
  server, resolves bundled ESE libraries, and starts Studio successfully.
- [x] Ubuntu and Windows packaging jobs pass on the consolidated pull request.

## v0.1.2 Windows GPU hotfix

- [x] Studio, launcher, and native server manifests report `0.1.2`.
- [x] Windows package staging contains the CUDA-enabled server, CUDA runtime,
  and cuBLAS redistributable libraries while retaining CPU fallback.
- [x] Packaged-runtime hardware discovery succeeds without requiring CMake or a
  compiler; CI runs `ese.exe doctor --json` from the staged runtime.
- [ ] A real NVIDIA Windows sweep starts the server, passes health, and records
  throughput. Hosted Windows CI has no physical NVIDIA GPU, so PE/dependency and
  no-driver checks do not satisfy this gate.
- [x] Failed sweeps show the native diagnostic and offer a privacy-safe GitHub
  bug report draft.

Publishing remains gated by a successful tag workflow that builds both Linux
and Windows assets, generates checksums and signed updater metadata, and creates
the GitHub release. That workflow status is operational release evidence, not a
substitute for the unchecked physical NVIDIA sweep above.

## v0.2.0 hardware-adaptive feature release

- [x] Studio, launcher, frontend fallback, Cargo manifest/lock, and package
  metadata report `0.2.0` locally.
- [x] Hardware calibration, mixed CPU/GPU MoE, bounded prefill streaming, and
  reversible KV/expert/transient rebalancing passed their retained CPU,
  sanitizer, Turing, Ampere, and heterogeneous three-GPU gates.
- [x] Dense concurrent sessions and real Qwen text/image/text transient-owner
  handoffs pass model-backed lifecycle and shutdown validation.
- [x] Kimi Linear 48B-A3B loads and produces deterministic output through its
  native KDA/MLA and bounded top-8 expert-cache paths.
- [x] A 65,536-context Qwen3.8 27B profile survived 1,306- and 4,786-token
  workloads after the verified `32,32,36` tensor split replaced the stale
  `30,25,45` profile.
- [ ] The consolidated private pull request is green, reviewed, and merged to
  public `main` without missing public-main commits.
- [ ] The `v0.2.0` tag workflow publishes signed Linux/Windows updater assets,
  installers, checksums, and release notes.
