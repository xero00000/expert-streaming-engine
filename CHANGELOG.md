# Changelog

All notable user-visible changes to Expert Streaming Engine are recorded here.
Release dates use UTC and the release notes under `docs/releases/` remain the
authoritative historical record for each tag.

## [Unreleased]

### Changed

- Documentation now distinguishes the supported `main` branch from deleted
  historical research lines.
- Build, installation, container, Android, and Studio configuration guidance
  now describes ESE rather than inherited upstream projects.
- Documentation CI checks local links, release-version consistency, supported
  branch references, and accidental upstream installation instructions.

## [0.1.2] - 2026-08-24

### Fixed

- Windows Studio packages include the CUDA-enabled native runtime and required
  CUDA/cuBLAS redistributable libraries while retaining CPU fallback.
- Installed Windows hardware discovery no longer requires source-build tools.
- Failed configuration sweeps expose the native server diagnostic.
- Visible Studio failures offer a privacy-scrubbed, editable GitHub bug-report
  draft.
- The tag workflow allows enough time for CUDA, MSI, and NSIS packaging to
  complete on the hosted Windows runner.

See [the v0.1.2 release notes](docs/releases/v0.1.2.md).

## [0.1.1] - 2026-08-22

### Added

- Unified Studio installers containing the matching ESE launcher and native
  `llama-server` runtime.
- Signed AppImage and NSIS update artifacts, visible update progress, verified
  installation, recovery, and restart from Studio Settings.
- Linux and Windows source-install dependency preflight.

### Changed

- Studio now uses its bundled version-matched ESE runtime instead of silently
  preferring an unrelated launcher on `PATH`.

See [the v0.1.1 release notes](docs/releases/v0.1.1.md).

## [0.1.0] - 2026-08-21

### Added

- `ese` front door with `doctor`, `build`, `plan`, and `serve` commands.
- GGUF metadata inspection, split-shard validation, JSON plans, native-option
  passthrough, and hardware-aware multi-GPU planning.
- Automatic `resident`, `hybrid`, `cache`, and `stream` policy selection.
- Native global resource controller with deterministic RAM, VRAM, context, KV,
  expert-cache, transient-module, workspace, staging, and reserve accounting.
- Bounded NVMe to RAM to per-device VRAM expert hierarchy with mmap, pread, and
  io_uring sources, adaptive admission, route-aware prefetch, and observable
  eviction/lifetime behavior.
- Failure-atomic transient MTP/mmproj management, adaptive speculative depth,
  and mapped draft-vocabulary support.
- Turbo KV, TCQ, and VBR foundations. These codecs remain internal or
  experimental until their remaining quality and lifecycle gates pass.
- ESE Studio for Linux with GGUF discovery, app terminals, Hugging Face model
  downloads, measured configuration sweeps, and optional privacy-filtered
  community benchmark sharing.
- Deterministic launcher tests, native parser-surface guards, sanitizer tests,
  CPU builds, and model-backed CUDA validation.

See [the v0.1.0 release notes](docs/releases/v0.1.0.md).

[Unreleased]: https://github.com/xero00000/expert-streaming-engine/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/xero00000/expert-streaming-engine/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/xero00000/expert-streaming-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/xero00000/expert-streaming-engine/releases/tag/v0.1.0
