# Changelog

## Unreleased — unified engine

### Fixed

- Windows Studio packages now include the CUDA-enabled native runtime and its
  redistributable CUDA libraries while retaining CPU fallback.
- Hardware discovery no longer fails merely because an installed user does not
  have source-build tools, and failed sweeps surface the native server error.
- Visible Studio failures offer a privacy-scrubbed, prefilled GitHub bug report
  that the user can review before submitting.

### Added

- `ese` front door with `doctor`, `build`, `plan`, and `serve`.
- GGUF scalar metadata reader with split-shard validation and efficient array skipping.
- Automatic `resident`, `hybrid`, and `stream` policy selection.
- Hardware-aware multi-GPU tensor-split planning.
- Conservative CPU-MoE placement with optional GPU-resident MoE tail.
- Stream safeguards for deferred experts and route-aware prefetch.
- JSON plans and native-argument passthrough.
- Deterministic launcher tests, native parser-surface guard, and CPU-build CI.
- Focused architecture, profile, benchmark, branch, and port-roadmap docs.

### Changed

- Consolidation base moved from the older public default to the newer DeepSeek/Maple integration line.
- README reduced to one supported path; experimental detail moved into focused documents.
- Branch roles and retirement order are explicit.
- The intermediate `cache` policy was corrected to `hybrid`; the branch does not generate unsupported buun-style `--moe-cache` flags.

### Not yet claimed complete

- Adaptive RAM/VRAM expert cache and expert-parallel distribution.
- VBR/Turbo/TCQ KV codecs.
- Dynamic native global resource controller.
- Generalized MTP/mmproj transient swapping.
- Adaptive MTP depth and mapped draft vocabulary.
