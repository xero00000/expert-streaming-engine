# Branch Policy

The supported engine lives on `main`. General fixes and features start from
`main`, pass CI in a pull request, and return to `main` by squash merge. Release
tags are cut only from a clean, validated `main` commit.

## Supported branches

- `main` — the public default, integration, and release branch.

No other long-lived remote branch is currently supported. Platform experiments
and research should use a short-lived pull-request branch and must not be
described as available until that branch actually exists on the remote.

## Historical branch lineage

The following names occur in old discussions, commits, or planning records but
are not current remote branches:

- `android-s25-qnn` — an unmerged Android/QNN draft. Android/QNN remains future
  work and has no supported branch or runtime on `main`.
- `deepseek4-expert-streaming` — the former DeepSeek/Maple integration source.
- `fa-q4-d256-longctx` — an older long-context Flash Attention/Q4 experiment.
- `expert-gpu-cache-experiment` — an earlier cache experiment, distinct from
  the bounded adaptive cache now on `main`.
- `chunked-gdn-port` — the former public default before consolidation.

Use merged pull requests, release tags, and immutable commit IDs when citing
historical work. A historical branch name is not a reproducible reference after
the branch has been deleted.

## Working-branch rules

- Create a focused branch from current `main` for each independently testable
  change.
- Delete merged pull-request heads after their CI evidence and squash commit
  are recorded.
- Never infer that a squash-merged branch is unmerged from ancestry alone;
  verify its pull-request state.
- Preserve unique research with an immutable tag or commit ID before deleting
  its working branch.
- Keep local model fixtures before removing disposable validation worktrees.
- Protect `main`; never use it as an experiment scratch branch.
