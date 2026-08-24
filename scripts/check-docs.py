#!/usr/bin/env python3
"""Validate ESE's maintained Markdown and documentation invariants."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SUPPORTED_BRANCH = re.compile(r"^- `([^`]+)`(?:\s|$)")


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def maintained_markdown() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if ".git" not in path.parts and "github-data" not in path.parts
    )


def check_links(errors: list[str]) -> None:
    for document in maintained_markdown():
        in_fence = False
        for line_number, line in enumerate(
            document.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            stripped = line.lstrip()
            if stripped.startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            for raw_target in MARKDOWN_LINK.findall(line):
                target = raw_target.strip()
                if target.startswith("<") and target.endswith(">"):
                    target = target[1:-1]
                elif any(character.isspace() for character in target):
                    # Markdown permits an optional quoted title after a target.
                    target = target.split(maxsplit=1)[0]
                if not target or target.startswith(
                    ("#", "http://", "https://", "mailto:", "data:", "/")
                ):
                    continue

                relative = unquote(target.split("#", 1)[0].split("?", 1)[0])
                if not relative:
                    continue

                resolved = (document.parent / relative).resolve()
                try:
                    resolved.relative_to(ROOT)
                except ValueError:
                    fail(
                        errors,
                        f"{document.relative_to(ROOT)}:{line_number}: "
                        f"link escapes the repository: {target}",
                    )
                    continue
                if not resolved.exists():
                    fail(
                        errors,
                        f"{document.relative_to(ROOT)}:{line_number}: "
                        f"missing local link target: {target}",
                    )


def extract_version(errors: list[str]) -> str | None:
    package_version = json.loads(
        (ROOT / "studio/package.json").read_text(encoding="utf-8")
    )["version"]
    tauri_version = json.loads(
        (ROOT / "studio/src-tauri/tauri.conf.json").read_text(encoding="utf-8")
    )["version"]
    cargo_text = (ROOT / "studio/src-tauri/Cargo.toml").read_text(encoding="utf-8")
    cargo_match = re.search(r'^version = "([^"]+)"', cargo_text, re.MULTILINE)
    launcher_text = (ROOT / "tools/ese.py").read_text(encoding="utf-8")
    launcher_match = re.search(r'version="ese ([^"]+)"', launcher_text)

    if not cargo_match:
        fail(errors, "studio/src-tauri/Cargo.toml: package version not found")
        return None
    if not launcher_match:
        fail(errors, "tools/ese.py: launcher version not found")
        return None

    versions = {
        "studio/package.json": package_version,
        "studio/src-tauri/tauri.conf.json": tauri_version,
        "studio/src-tauri/Cargo.toml": cargo_match.group(1),
        "tools/ese.py": launcher_match.group(1),
    }
    if len(set(versions.values())) != 1:
        fail(
            errors,
            "release versions disagree: "
            + ", ".join(f"{path}={version}" for path, version in versions.items()),
        )
        return None
    return package_version


def require_text(errors: list[str], path: str, required: list[str]) -> None:
    content = (ROOT / path).read_text(encoding="utf-8")
    for token in required:
        if token not in content:
            fail(errors, f"{path}: missing current-release token: {token}")


def check_versions(errors: list[str]) -> None:
    version = extract_version(errors)
    if version is None:
        return

    require_text(
        errors,
        "README.md",
        [
            f"v{version} desktop installers",
            f"ese-studio_{version}_amd64.AppImage",
            f"ese-studio_{version}_x64-setup.exe",
        ],
    )
    require_text(errors, "studio/README.md", [f"v{version} AppImage"])
    require_text(
        errors,
        "docs/install.md",
        [f"examples below use `{version}`", f"ese-studio_{version}_amd64.deb"],
    )
    require_text(errors, "CHANGELOG.md", [f"## [{version}] -"])
    if not (ROOT / f"docs/releases/v{version}.md").is_file():
        fail(errors, f"docs/releases/v{version}.md: current release notes are missing")


def check_upstream_misdirection(errors: list[str]) -> None:
    canonical = [
        "README.md",
        "studio/README.md",
        "docs/build.md",
        "docs/install.md",
        "docs/docker.md",
        "docs/android.md",
        "docker/README.md",
    ]
    forbidden = [
        re.compile(
            r"git clone https://github\.com/(?:ikawrakow/ik_llama\.cpp|"
            r"ggerganov/llama\.cpp)"
        ),
        re.compile(r"ghcr\.io/ggerganov/llama\.cpp"),
        re.compile(r"brew install llama\.cpp"),
    ]
    for relative in canonical:
        content = (ROOT / relative).read_text(encoding="utf-8")
        for pattern in forbidden:
            if pattern.search(content):
                fail(
                    errors,
                    f"{relative}: canonical ESE guidance matches forbidden "
                    f"upstream-install pattern {pattern.pattern!r}",
                )


def supported_branches() -> list[str]:
    lines = (ROOT / "docs/BRANCH_POLICY.md").read_text(encoding="utf-8").splitlines()
    in_section = False
    branches: list[str] = []
    for line in lines:
        if line == "## Supported branches":
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and (match := SUPPORTED_BRANCH.match(line)):
            branches.append(match.group(1))
    return branches


def check_remote_branches(errors: list[str]) -> None:
    documented = supported_branches()
    if not documented:
        fail(errors, "docs/BRANCH_POLICY.md: no supported branches documented")
        return
    try:
        output = subprocess.run(
            ["git", "ls-remote", "--heads", "origin"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        fail(errors, f"could not query remote branches: {exc}")
        return
    remote = {
        line.rsplit("refs/heads/", 1)[1]
        for line in output.splitlines()
        if "refs/heads/" in line
    }
    for branch in documented:
        if branch not in remote:
            fail(
                errors,
                f"docs/BRANCH_POLICY.md: supported branch does not exist on origin: {branch}",
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-remote-branches",
        action="store_true",
        help="verify documented supported branches with git ls-remote",
    )
    args = parser.parse_args()

    errors: list[str] = []
    check_links(errors)
    check_versions(errors)
    check_upstream_misdirection(errors)
    if args.check_remote_branches:
        check_remote_branches(errors)

    if errors:
        print("Documentation validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    branch_note = " and remote branches" if args.check_remote_branches else ""
    print(f"Documentation validation passed{branch_note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
