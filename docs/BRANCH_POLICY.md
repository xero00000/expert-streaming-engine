# Branch Policy

The repository accumulated useful work on several long-lived branches. The unified line makes their roles explicit instead of asking users to guess which branch contains the usable engine.

## Proposed core

### `ese-unified`

Candidate for the public default branch.

Base: `deepseek4-expert-streaming`.

Contains:

- the current DeepSeek V4 integration line;
- deferred expert streaming and opt-in route-aware mmap prefetch;
- CPU-MoE hybrid placement with an optional GPU-resident tail;
- the bounded descriptor/storage/RAM expert tier and adaptive per-device VRAM
  tier implemented for Phase 2, subject to its recorded hardware matrix;
- the Phase 1 Turbo KV, TCQ, VBR, and lifecycle foundation released as
  `v0.1.0`;
- DeepSeek V4, DSpark/MTP, Maple/TQ2_0, and recent CUDA work already present on that line;
- the `ese` launcher, focused docs, tests, and CI.

General supported work should target this branch through reviewable pull requests.

## Integration source

### `deepseek4-expert-streaming`

Selected because it is materially newer than the old public default and contains the broadest relevant native model/kernel set. Once the unified line is promoted, this branch should stop receiving unrelated direct commits.

It is not described as already containing buun's adaptive VRAM MoE cache or VBR/TCQ. Those are tracked ports.

## Platform R&D

### `android-s25-qnn`

Keep separate until the Android/QNN backend passes the same model-load, parity, lifecycle, and memory-bound gates as the desktop engine. A future merge should preserve one common model/storage/control layer with platform-specific backends below it.

## Historical port candidates

### `fa-q4-d256-longctx`

Contains older long-context Flash Attention/Q4 work. Re-port individual commits only after checking whether newer integration-line kernels supersede them.

### `expert-gpu-cache-experiment`

Historical source for a bounded GPU-cache experiment and route-prefetch work. The experiment recorded a structural miss-rate/PCIe dead end on its tested GPT-OSS geometry; do not present it as the newer adaptive cache design. Retain it for provenance until unique commits are tagged.

## Previous default

### `chunked-gdn-port`

The old public default remains untouched until the unified pull request passes CI and review. Before changing the default, tag its last commit, for example `legacy/chunked-gdn-port-2026-08`.

## Cleanup sequence

1. Review and merge the unified PR.
2. Change the repository default to the unified line or rename it to `main`.
3. Tag final tips of branches that will be retired.
4. Update issues and links to the new default.
5. Close or move obsolete experiment issues.
6. Delete only branches whose unique commits are tagged and documented.
7. Keep Android as an active platform branch.
8. Protect the default branch and require unified CI.

This sequence is non-destructive: a concise repository should not erase experimental evidence or authorship history.
