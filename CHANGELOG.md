# Changelog

## Unreleased — unified engine

### Added

- `ese` front door with `doctor`, `build`, `plan`, and `serve`.
- GGUF scalar metadata reader and split-shard validation.
- Automatic `resident`, `cache`, and `stream` policy selection.
- Hardware-aware multi-GPU tensor-split planning.
- Automatic single- versus multi-GPU MoE-cache selection.
- Stream safeguards for deferred experts and route prefetch.
- JSON memory plans and native-argument passthrough.
- Deterministic launcher unit tests and unified CI.
- Focused architecture, profile, benchmark, branch, and port-roadmap docs.

### Changed

- Consolidation base moved from the older public default to the newer DeepSeek/expert-cache integration line.
- README reduced to one supported path and moved experimental detail into focused documents.
- Branch roles are explicitly documented.

### Not yet claimed complete

- VBR/Turbo/TCQ KV codecs.
- Dynamic native global memory controller.
- Unified bounded RAM cache below the VRAM expert cache.
- Generalized MTP/mmproj transient swapping.
- Adaptive MTP depth and mapped draft vocabulary.
