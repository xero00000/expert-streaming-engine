# ESE Studio v0.1.0 release checklist

This checklist records the Linux-first release gate. A checked item requires
either automated CI evidence or a retained model-backed artifact; compilation
alone is not accepted as inference evidence.

## Automated gates

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

## UI regression — 2026-08-21

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

## Native and model-backed evidence

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

## Release limitations

- Linux x86-64 is the supported Studio target for v0.1.0.
- Packages are checksummed but not cryptographically signed.
- AppImage, automatic updates, Windows, Android/QNN, and Ada-or-newer physical
  CUDA validation are not claimed by this release.
- The Android/QNN work remains an isolated draft and is not part of v0.1.0.
