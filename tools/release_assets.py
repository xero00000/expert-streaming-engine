"""Release package names, provenance, checksums and variant-preserving feeds."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


def validate_version(version: str) -> None:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?", version):
        raise ValueError(f"invalid release version: {version}")


def definitions(platform: str, version: str, variant: str | None = None) -> list[tuple[str, str, str, bool]]:
    validate_version(version)
    if platform == "linux":
        if variant not in ("cpu", "cuda"):
            raise ValueError("Linux release assets require an explicit cpu or cuda variant")
        prefix = f"ESE-Studio_{version}_linux-x86_64_{variant}"
        return [
            ("deb", f"*_{version}_amd64.deb", prefix + ".deb", False),
            ("rpm", f"*-{version}-*.x86_64.rpm", prefix + ".rpm", False),
            ("appimage", f"*_{version}_amd64.AppImage", prefix + ".AppImage", True),
        ]
    if platform == "windows":
        if variant is not None:
            raise ValueError("Windows packages contain both runtimes; do not select a variant")
        prefix = f"ESE-Studio_{version}_windows-x64"
        return [
            ("nsis", f"*_{version}_x64-setup.exe", prefix + "-setup.exe", True),
            ("msi", f"*_{version}_x64*.msi", prefix + ".msi", False),
        ]
    raise ValueError(f"unsupported release platform: {platform}")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        value = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
        return value.hexdigest()


def verify_runtime(platform: str, variant: str | None, runtime_root: Path) -> None:
    def marker(relative: str) -> dict:
        return json.loads((runtime_root / relative / "ese-runtime.json").read_text())

    primary = marker("build/bin")
    if platform == "linux":
        if primary.get("cuda") is not (variant == "cuda") or primary.get("package_variant") != variant:
            raise ValueError("staged Linux runtime does not match the requested package variant")
    elif primary.get("cuda") is not True or marker("build/cpu/bin").get("cuda") is not False:
        raise ValueError("Windows release requires both CUDA and driver-independent CPU runtimes")


def collect(platform: str, version: str, bundle_root: Path, output: Path, *,
            variant: str | None = None, source_commit: str = "",
            runtime_root: Path | None = None) -> list[Path]:
    expected = definitions(platform, version, variant)
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise ValueError("source commit must be a full 40-character Git commit ID")
    if runtime_root is not None:
        verify_runtime(platform, variant, runtime_root)
    selected = []
    for directory, pattern, destination, signed in expected:
        candidates = sorted(p for p in (bundle_root / directory).glob(pattern) if p.is_file())
        if len(candidates) != 1:
            raise ValueError(f"expected one {version} {directory} bundle, found {len(candidates)}")
        source = candidates[0]
        if source.stat().st_size == 0:
            raise ValueError(f"empty release bundle: {source}")
        selected.append((source, destination))
        if signed:
            signature = source.with_name(source.name + ".sig")
            if not signature.is_file() or not signature.read_text(encoding="utf-8").strip():
                raise ValueError(f"missing or empty matching signature: {signature}")
            selected.append((signature, destination + ".sig"))
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"release asset directory must be empty: {output}")
    files = {name: {"bytes": source.stat().st_size, "sha256": digest(source)}
             for source, name in selected}
    group = f"{platform}-{variant}" if variant else platform
    metadata = {"version": version, "platform": platform, "variant": variant,
                "source_commit": source_commit, "files": files}
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for source, destination in selected:
        target = output / destination
        shutil.copy2(source, target)
        assets.append(target)
    index = output / f"artifact-set-{group}.json"
    index.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return assets + [index]


def prepare_release(version: str, repository: str, assets: Path, *, source_commit: str | None = None) -> list[Path]:
    validate_version(version)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repository):
        raise ValueError("repository must be an owner/name GitHub repository")
    groups = [("linux", "cpu"), ("linux", "cuda"), ("windows", None)]
    files, indices, commits = {}, [], set()
    for platform, variant in groups:
        group = f"{platform}-{variant}" if variant else platform
        index = assets / f"artifact-set-{group}.json"
        data = json.loads(index.read_text(encoding="utf-8"))
        if (data.get("version"), data.get("platform"), data.get("variant")) != (version, platform, variant):
            raise ValueError(f"artifact-set identity mismatch: {index.name}")
        commit = data.get("source_commit", "")
        if not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"missing source provenance: {index.name}")
        commits.add(commit)
        expected_names = set()
        for _, _, name, signed in definitions(platform, version, variant):
            expected_names.add(name)
            if signed:
                expected_names.add(name + ".sig")
        if set(data.get("files", {})) != expected_names:
            raise ValueError(f"incomplete or unexpected package set: {index.name}")
        for name in sorted(expected_names):
            path = assets / name
            info = {"bytes": path.stat().st_size, "sha256": digest(path)}
            if info["bytes"] == 0 or info != data["files"][name]:
                raise ValueError(f"asset digest or size mismatch: {name}")
            if name.endswith(".sig") and not path.read_text(encoding="utf-8").strip():
                raise ValueError(f"empty matching signature: {name}")
            files[name] = info
        indices.append(index.name)
    if len(commits) != 1:
        raise ValueError("release packages were built from different source commits")
    if source_commit is not None and commits != {source_commit}:
        raise ValueError("release packages do not match the requested source commit")
    if set(p.name for p in assets.iterdir()) != set(files) | set(indices):
        raise ValueError("release asset directory contains unexpected or previously generated files")
    base = f"https://github.com/{repository}/releases/download/v{version}/"
    windows = f"ESE-Studio_{version}_windows-x64-setup.exe"

    def target(name: str) -> dict:
        return {"url": base + name, "signature": (assets / (name + ".sig")).read_text(encoding="utf-8").strip()}

    generated = {}
    for variant in ("cpu", "cuda"):
        update = {"version": version, "notes": f"Expert Streaming Engine v{version}",
                  "pub_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                  "platforms": {"linux-x86_64": target(f"ESE-Studio_{version}_linux-x86_64_{variant}.AppImage"),
                                "windows-x86_64": target(windows)}}
        generated[f"latest-{variant}.json"] = json.dumps(update, indent=2) + "\n"
    # Legacy Linux releases used CPU packages. Windows always receives both runtimes.
    generated["latest.json"] = generated["latest-cpu.json"]
    generated["SHA256SUMS"] = "".join(f"{files[name]['sha256']}  {name}\n" for name in sorted(files))
    generated["SHA256SUMS-windows.txt"] = "".join(
        f"{files[name]['sha256']}  {name}\n" for name in sorted(files) if "_windows-x64" in name)
    generated["release-manifest.json"] = json.dumps({
        "version": version, "source_commit": next(iter(commits)), "files": files,
        "package_count": sum(not name.endswith(".sig") for name in files),
        "updater_feeds": {"cpu": "latest-cpu.json", "cuda": "latest-cuda.json", "legacy": "latest.json"},
        "signature_checks": "matching nonempty sidecars; cryptographic verification is a separate qualification",
    }, indent=2) + "\n"
    paths = []
    for name, content in generated.items():
        path = assets / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(content)
        paths.append(path)
    return paths


def collect_main() -> None:
    parser = argparse.ArgumentParser(description="Collect one versioned release package set.")
    parser.add_argument("--platform", required=True, choices=["linux", "windows"])
    parser.add_argument("--variant", choices=["cpu", "cuda"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        paths = collect(args.platform, args.version, args.bundle_root, args.output,
                        variant=args.variant, source_commit=args.source_commit, runtime_root=args.runtime_root)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Release asset collection failed: {error}\n")
    for path in paths:
        print(path)


def prepare_main() -> None:
    parser = argparse.ArgumentParser(description="Validate all eight packages and generate release metadata.")
    parser.add_argument("--version", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    try:
        paths = prepare_release(args.version, args.repository, args.assets, source_commit=args.source_commit)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Release preparation failed: {error}\n")
    for path in paths:
        print(path)
