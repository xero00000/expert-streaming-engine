#!/usr/bin/env python3
"""Collect only the requested version from a potentially restored bundle tree."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def collect(platform: str, version: str, bundle_root: Path, output: Path) -> list[Path]:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError(f"invalid release version: {version}")
    if platform == "linux":
        definitions = [
            ("deb", f"*_{version}_amd64.deb", f"ese-studio_{version}_amd64.deb", False),
            ("rpm", f"*-{version}-*.x86_64.rpm", f"ese-studio-{version}-1.x86_64.rpm", False),
            ("appimage", f"*_{version}_amd64.AppImage", f"ese-studio_{version}_amd64.AppImage", True),
        ]
    elif platform == "windows":
        definitions = [
            ("nsis", f"*_{version}_x64-setup.exe", f"ese-studio_{version}_x64-setup.exe", True),
            ("msi", f"*_{version}_x64*.msi", f"ese-studio_{version}_x64.msi", False),
        ]
    else:
        raise ValueError(f"unsupported release platform: {platform}")
    selected: list[tuple[Path, str]] = []
    for directory, pattern, destination, signed in definitions:
        candidates = sorted(p for p in (bundle_root / directory).glob(pattern) if p.is_file())
        if len(candidates) != 1:
            raise ValueError(f"expected one {version} {directory} bundle, found {len(candidates)}")
        source = candidates[0]
        if source.stat().st_size == 0:
            raise ValueError(f"empty release bundle: {source}")
        selected.append((source, destination))
        if signed:
            signature = source.with_name(source.name + ".sig")
            if not signature.is_file() or signature.stat().st_size == 0:
                raise ValueError(f"missing or empty matching signature: {signature}")
            selected.append((signature, destination + ".sig"))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"release asset directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for source, destination in selected:
        target = output / destination
        shutil.copy2(source, target)
        assets.append(target)
    return assets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", required=True, choices=["linux", "windows"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--bundle-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        for asset in collect(args.platform, args.version, args.bundle_root, args.output):
            print(asset)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Release asset collection failed: {error}\n")


if __name__ == "__main__":
    main()
