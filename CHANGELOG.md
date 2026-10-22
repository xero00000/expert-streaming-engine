# Changelog

All notable user-visible changes to Expert Streaming Engine are recorded here.
Release dates use UTC and the release notes under `docs/releases/` remain the
authoritative historical record for each tag.

## [Unreleased]

## [0.2.1] - 2026-10-22

### Improved

- Reuse compatible expert-cache payloads across layer layouts and overlap
  independent RAM-cache reads within their configured limits.
- Improve memory accounting, allocation-failure recovery and server request
  handling so failed work leaves the runtime ready for the next request.
- Refresh the GitHub introduction and add a beginner guide covering model
  discovery, a first local conversation and basic troubleshooting.

### Fixed

- Bundle the Microsoft C++ and OpenMP runtime dependencies needed by Windows
  Studio and its native servers.
- Select the separate Windows CPU runtime when NVIDIA hardware is unavailable.
- Keep DEB/RPM update checks with the package manager instead of offering an
  AppImage replacement.
- Use bounded MSI cabinets so the complete CUDA payload can be packaged.

### Packaging and updates

- Provide separate Linux CPU/CUDA AppImage, DEB and RPM packages, plus Windows
  MSI and setup EXE installers containing both native runtimes.
- Verify AppImage and NSIS updater signatures and installed-update behavior;
  Linux update feeds preserve the installed CPU/CUDA variant.
- Keep synchronous expert loading as the default, with bounded parallel loading
  available through `--expert-load-workers` for advanced configurations.

### Qualification

- Local source checks, scoped correctness/recovery tests, Linux package
  installation, Windows installer/CPU fallback and signed-update checks pass.
- Physical Windows NVIDIA inference and extended context/concurrency checks are
  deferred and remain unqualified. Experimental Qwen admission stays disabled
  in production builds. Windows installers lack Authenticode signatures.

See [the release notes](docs/releases/v0.2.1.md) and
[detailed validation scope](docs/RELEASE_NOTES_V0_2_1.md).

## [0.2.0] - 2026-08-25

### Added

- Topology-bound, model-backed hardware calibration and deterministic workload
  verification for adaptive CPU/GPU MoE routing.
- A one-way native runtime guard that revokes mixed expert routing when live
  CPU, upload, or fallback telemetry contradicts the verified profile.
- Idle-safe runtime KV, expert-cache, and transient MTP/projector rebalancing
  with dry-run admission, coherent resource reporting, and reversible
  multi-pool publication.
- Configurable 1/2/4-session local serving and concurrency-focused sweeps for
  dense resident models in ESE Studio.
- Native `kimi-linear` GGUF execution for Kimi Linear 48B-A3B, including its
  hybrid KDA/MLA graph, 256-expert top-8 routing, and bounded sidecar-backed
  expert caching on CPU, Turing, and Ampere CUDA paths.

### Changed

- MoE prefill and decode can use calibrated heterogeneous CPU/GPU work,
  bounded double-buffered expert streaming, and per-device adaptive caches
  without requiring the complete expert set in RAM or VRAM.
- Multi-session launch remains fail-closed for MoE, hybrid, cache, and stream
  plans until those modes have equivalent multi-sequence parity evidence.
- Documentation now distinguishes the supported `main` branch from deleted
  historical research lines.
- Build, installation, container, Android, and Studio configuration guidance
  now describes ESE rather than inherited upstream projects.
- Documentation CI checks local links, release-version consistency, supported
  branch references, and accidental upstream installation instructions.

### Fixed

- KV, expert-cache, and transient-owner replacement failures now restore the
  exact prior physical owners and logical resource plan before inference
  resumes.
- Concurrent request cancellation, deferred transient-residency work, and
  protocol stop reasons no longer interfere across local sessions.
- Graceful shutdown no longer schedules a decode continuation after task-queue
  admission closes; active and deferred transient leases are drained without
  an uncaught queue exception.
- Byte-sized expert caches are no longer silently limited to 64 physical
  `(layer, expert)` slots, and free-VRAM-clamped caches retain allocation
  headroom for backend quantization padding instead of aborting during decode.
- Configuration sweeps now perform inference during capacity checks and apply
  the exact tested multi-GPU tensor split instead of retaining a stale profile.

See [the v0.2.0 release notes](docs/releases/v0.2.0.md).

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

[Unreleased]: https://github.com/xero00000/expert-streaming-engine/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/xero00000/expert-streaming-engine/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/xero00000/expert-streaming-engine/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/xero00000/expert-streaming-engine/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/xero00000/expert-streaming-engine/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/xero00000/expert-streaming-engine/releases/tag/v0.1.0
