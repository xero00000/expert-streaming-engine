# Branch Policy

The supported desktop engine lives on `main`. General fixes and features start
from `main`, pass unified CI in a pull request, and return to `main` by squash
merge. Release tags are cut only from a clean, validated `main` commit.

## Active branches

### `main`

The public default and release branch. It contains the unified launcher, the
native global resource controller, the bounded expert hierarchy, Turbo KV/VBR
foundation, transient MTP/mmproj management, adaptive speculation, and the
current DeepSeek/Maple integration line.

### `android-s25-qnn`

An active platform draft. It stays separate until the Android/QNN backend has
physical-device build, parity, lifecycle, bounded-memory, and thermal evidence.
Its open pull request must be moved to `main` before the former integration base
can be retired.

## Retained research lines

### `deepseek4-expert-streaming`

The historical integration source for current desktop work. Do not add general
changes here; preserve it only while unique research commits still need a
traceable reference.

### `fa-q4-d256-longctx`

Older long-context Flash Attention/Q4 work. Re-port individual changes only
after checking whether the current `main` kernels supersede them.

### `expert-gpu-cache-experiment`

A recorded GPU-cache experiment that found a miss-rate/PCIe dead end on its
tested GPT-OSS geometry. It is not the adaptive cache now on `main`. Retain or
archive its unique tip for provenance; do not present it as a supported line.

### `chunked-gdn-port`

The former public default. It remains temporarily because the Android draft
still targets it. Retarget/rebase that pull request before archiving and
deleting this branch.

## Cleanup rules

- Delete merged pull-request heads after their merge and CI evidence are
  recorded.
- Never infer that a squash-merged branch is unmerged from ancestry alone;
  verify its pull-request state.
- Preserve active platform work and unique research until it is merged,
  superseded with evidence, or tagged for archival.
- Keep local model fixtures before removing disposable validation worktrees.
- Protect `main`; never use it as an experiment scratch branch.
