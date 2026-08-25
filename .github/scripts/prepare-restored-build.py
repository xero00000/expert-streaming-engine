#!/usr/bin/env python3
"""Make a restored Ninja build directory safely incremental after checkout."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path


MARKER_NAME = ".ese-source-sha"
UNCHANGED_MTIME = 946_684_800  # 2000-01-01T00:00:00Z


def git_bytes(*args: str) -> bytes:
    return subprocess.check_output(["git", *args])


def valid_commit(value: str) -> bool:
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", value):
        return False
    return subprocess.run(
        ["git", "cat-file", "-e", f"{value}^{{commit}}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def is_ancestor(base: str) -> bool:
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", base, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def nul_paths(data: bytes) -> list[Path]:
    return [Path(os.fsdecode(value)) for value in data.split(b"\0") if value]


def source_base(build_root: Path, fallback_base: str) -> str | None:
    marker = build_root / MARKER_NAME
    if marker.is_file():
        value = marker.read_text(encoding="utf-8").strip()
        if valid_commit(value):
            return value
        print(f"Ignoring invalid restored build marker: {value!r}")
        return None
    value = fallback_base.strip()
    if value and valid_commit(value):
        print(f"Restored build predates source markers; using event base {value}")
        return value
    return None


def prepare(build_root: Path, fallback_base: str) -> int:
    if not build_root.is_dir():
        print("No restored build directory; a clean build will run")
        return 0

    base = source_base(build_root, fallback_base)
    if base is None or not is_ancestor(base):
        print("Restored build source is unknown; retaining checkout timestamps for a full rebuild")
        return 0

    tracked = nul_paths(git_bytes("ls-files", "-z"))
    changed = set(
        nul_paths(
            git_bytes(
                "diff",
                "--name-only",
                "--diff-filter=ACMRT",
                "-z",
                base,
                "HEAD",
                "--",
            )
        )
    )
    for path in tracked:
        if path.is_file() and not path.is_symlink():
            os.utime(path, (UNCHANGED_MTIME, UNCHANGED_MTIME))

    changed_mtime = max(UNCHANGED_MTIME + 1, int(time.time()))
    changed_existing = 0
    for path in changed:
        if path.is_file() and not path.is_symlink():
            os.utime(path, (changed_mtime, changed_mtime))
            changed_existing += 1

    print(
        f"Prepared restored build from {base}: "
        f"{len(tracked)} tracked files normalized, {changed_existing} changed files refreshed"
    )
    return 0


def record(build_root: Path) -> int:
    build_root.mkdir(parents=True, exist_ok=True)
    head = git_bytes("rev-parse", "HEAD").decode("ascii").strip()
    marker = build_root / MARKER_NAME
    marker.write_text(f"{head}\n", encoding="utf-8")
    print(f"Recorded restored-build source {head}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-root", type=Path, required=True)
    parser.add_argument("--fallback-base", default="")
    parser.add_argument("--record", action="store_true")
    args = parser.parse_args()
    if args.record:
        return record(args.build_root)
    return prepare(args.build_root, args.fallback_base)


if __name__ == "__main__":
    sys.exit(main())
