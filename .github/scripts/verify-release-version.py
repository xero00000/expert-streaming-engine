#!/usr/bin/env python3
"""Fail a release when any user-visible ESE version surface disagrees."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def cargo_lock_version() -> str:
    packages = tomllib.loads(
        (ROOT / "studio/src-tauri/Cargo.lock").read_text(encoding="utf-8")
    )["package"]
    matches = [package["version"] for package in packages if package["name"] == "ese-studio"]
    if len(matches) != 1:
        raise ValueError(f"expected one ese-studio Cargo.lock package, found {len(matches)}")
    return matches[0]


def regex_version(path: str, pattern: str) -> str:
    text = (ROOT / path).read_text(encoding="utf-8")
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"version pattern was not found in {path}")
    return match.group(1)


def versions() -> dict[str, str]:
    package = json.loads((ROOT / "studio/package.json").read_text(encoding="utf-8"))
    tauri = json.loads(
        (ROOT / "studio/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    )
    cargo = tomllib.loads(
        (ROOT / "studio/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    )
    return {
        "studio/package.json": package["version"],
        "studio/src-tauri/tauri.conf.json": tauri["version"],
        "studio/src-tauri/Cargo.toml": cargo["package"]["version"],
        "studio/src-tauri/Cargo.lock": cargo_lock_version(),
        "tools/ese.py": regex_version(
            "tools/ese.py", r'version="ese ([0-9]+\.[0-9]+\.[0-9]+)"'
        ),
        "studio/src/App.tsx": regex_version(
            "studio/src/App.tsx",
            r'appVersion, setAppVersion\] = useState\("([0-9]+\.[0-9]+\.[0-9]+)"\)',
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()
    observed = versions()
    mismatches = {path: value for path, value in observed.items() if value != args.expected}
    if mismatches:
        for path, value in mismatches.items():
            print(f"{path}: expected {args.expected}, found {value}", file=sys.stderr)
        return 1
    for path in observed:
        print(f"{path}: {args.expected}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
