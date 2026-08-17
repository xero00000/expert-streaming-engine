# Branch Policy

The repository accumulated useful work on several long-lived branches. The unified line makes their roles explicit instead of asking users to guess which branch contains the real engine.

## Proposed core

### `ese-unified`

Candidate for the public default branch.

Base: `deepseek4-expert-streaming`.

Contains:

- current integration-line engine code;
- deferred expert streaming;
- adaptive MoE cache and expert-parallel work;
- DeepSeek V4, DSpark, MTP, and recent CUDA work already on that line;
- the `ese` launcher;
- focused ESE documentation and CI.

New generally supported work should target this branch through reviewable pull requests.

## Integration source

### `deepseek4-expert-streaming`

This was selected as the consolidation base because it is materially newer than the old default and already contains the broadest relevant native feature set. Once `ese-unified` is promoted, this branch should stop receiving unrelated direct commits.

## Platform R&D

### `android-s25-qnn`

Keep separate until the Android/QNN backend can pass the same model-load, parity, lifecycle, and memory-bound gates as the desktop engine. Platform build files and QNN kernels should not be mixed into the core branch merely to reduce the branch count.

A future Android merge should preserve one common model/storage/control layer with platform-specific backends below it.

## Historical port candidates

### `fa-q4-d256-longctx`

Contains older long-context Flash Attention/Q4 work. Re-port individual commits onto the unified line only after checking whether newer upstream kernels supersede them. Do not merge the branch wholesale.

### `expert-gpu-cache-experiment`

Historical source for route-prefetch and GPU-cache experiments. The useful production route-prefetch work is already represented on the newer line. Retain temporarily for archaeology, then tag and remove after commit provenance is documented.

## Previous default

### `chunked-gdn-port`

The old public default remains untouched until the unified pull request passes CI and review. After promotion, tag its last commit—for example `legacy/chunked-gdn-port-2026-08`—before changing the default branch.

## Cleanup sequence

1. Merge the unified pull request.
2. Change the repository default to `ese-unified` or rename it to `main`.
3. Tag the final tips of branches that will be retired.
4. Update open issues and links to the new default.
5. Close or move obsolete experiment issues.
6. Delete only branches whose unique commits are tagged and documented.
7. Keep Android as an active platform branch.
8. Protect the default branch and require unified CI.

This sequence is intentionally non-destructive. A concise repository should not lose experimental evidence or authorship history.
