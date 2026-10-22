"""Stage and verify app-local, release x64 MSVC redistributables."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


MSVC_DLL = re.compile(r"(?:msvcp|msvcr|vcruntime|vcomp|concrt|vccorlib)\d+[a-z0-9_]*\.dll", re.I)
MANIFEST = "ese-msvc-runtime.json"


def debug_runtime(name: str) -> bool:
    return bool(re.search(r"\d+d(?:_|\.dll$)", name, re.I))


def pe_imports(path: Path, *, include_delay: bool = True) -> set[str]:
    import pefile

    pe = pefile.PE(str(path), fast_load=True)
    try:
        if pe.FILE_HEADER.Machine != 0x8664:
            raise ValueError(f"Expected an x64 PE image: {path}")
        tables = [pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        if include_delay:
            tables.append(pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"])
        pe.parse_data_directories(directories=tables)
        return {
            entry.dll.decode("ascii").lower()
            for table in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT")
            for entry in getattr(pe, table, [])
        }
    finally:
        pe.close()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def redist_directory(explicit: Path | None = None) -> Path:
    """Use Visual Studio's release desktop x64 redist, never System32/debug."""
    value = explicit or os.environ.get("ESE_MSVC_REDIST_DIR") or os.environ.get("VCToolsRedistDir")
    if value:
        root = Path(value)
        root = root if root.name.lower() == "x64" else root / "x64"
        if not (root / "Microsoft.VC143.CRT").is_dir():
            raise ValueError(f"MSVC desktop x64 redistributables not found: {root}")
        return root

    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
    result = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    install = result.stdout.strip()
    if not install:
        raise ValueError("No MSVC installation found; set ESE_MSVC_REDIST_DIR")
    base = Path(install) / "VC/Redist/MSVC"
    versions = sorted(
        (p for p in base.iterdir() if p.is_dir() and re.fullmatch(r"\d+(?:\.\d+)+", p.name)),
        key=lambda p: tuple(map(int, p.name.split("."))), reverse=True,
    )
    for version in versions:
        root = version / "x64"
        if (root / "Microsoft.VC143.CRT").is_dir():
            return root
    raise ValueError("No release desktop x64 MSVC redist found; set ESE_MSVC_REDIST_DIR")


def redist_files(root: Path) -> dict[str, Path]:
    if root.name.lower() != "x64" or root.parent.name.lower() == "onecore" or "debug_nonredist" in (p.lower() for p in root.parts):
        raise ValueError("MSVC redistributables must use the release desktop x64 directory")
    files = {}
    for folder in ("Microsoft.VC143.CRT", "Microsoft.VC143.OpenMP"):
        directory = root / folder
        if directory.is_dir():
            for file in directory.iterdir():
                if file.is_file() and MSVC_DLL.fullmatch(file.name):
                    if debug_runtime(file.name):
                        raise ValueError(f"Debug MSVC runtime is not redistributable: {file.name}")
                    files[file.name.lower()] = file
    return files


def required_files(images: list[Path], candidates: dict[str, Path], reader=pe_imports) -> dict[str, Path]:
    pending = list(images)
    visited = set()
    required = {}
    while pending:
        image = pending.pop()
        if image in visited:
            continue
        visited.add(image)
        for name in reader(image, include_delay=True):
            name = name.lower()
            if not MSVC_DLL.fullmatch(name):
                continue
            if debug_runtime(name):
                raise ValueError(f"Debug MSVC runtime is not redistributable: {name}")
            if name not in candidates:
                raise ValueError(f"Missing release x64 MSVC redistributable {name}, imported by {image.name}")
            if name not in required:
                required[name] = candidates[name]
                pending.append(candidates[name])
    return required


def write_manifest(directory: Path, files: dict[str, Path], version: str) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    for existing in directory.iterdir():
        if existing.is_file() and MSVC_DLL.fullmatch(existing.name) and existing.name.lower() not in files:
            existing.unlink()
    rows = []
    for name, source in sorted(files.items()):
        target = directory / name
        shutil.copy2(source, target)
        rows.append({"name": name, "bytes": target.stat().st_size, "sha256": digest(target)})
    result = {"schema_version": 1, "redist_version": version, "files": rows}
    (directory / MANIFEST).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def stage(runtime_bins: list[Path], gui_directory: Path, redist: Path, reader=pe_imports) -> dict:
    candidates = redist_files(redist)
    plans = {}
    for directory in runtime_bins:
        images = [p for p in directory.iterdir() if p.suffix.lower() in (".exe", ".dll")]
        plans[directory] = required_files(images, candidates, reader)
    # Studio is compiled after beforeBuildCommand runs. Its CRT files must
    # already be available to the bundler; native OpenMP is isolated below.
    crt = {name: p for name, p in candidates.items() if p.parent.name == "Microsoft.VC143.CRT"}
    if not crt:
        raise ValueError("The MSVC release CRT directory is empty")
    crt.update(required_files(list(crt.values()), candidates, reader))
    plans[gui_directory] = crt
    # Resolve every transitive import before copying any redistributable.
    return {
        str(directory): write_manifest(directory, files, redist.parent.name)
        for directory, files in plans.items()
    }


def verify(directory: Path, reader=pe_imports, extra_images: list[Path] | None = None) -> None:
    files = {p.name.lower(): p for p in directory.iterdir() if p.is_file()}
    images = [p for p in files.values() if p.suffix.lower() in (".exe", ".dll")]
    required = required_files(images + (extra_images or []), files, reader)
    present = {name for name in files if MSVC_DLL.fullmatch(name)}
    if not required and not present:
        return
    manifest = json.loads((directory / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Invalid MSVC redistributable manifest")
    rows = manifest.get("files", [])
    names = [row["name"] for row in rows]
    if len(names) != len(set(names)) or not set(required).issubset(names) or set(names) != present:
        raise ValueError("Incomplete MSVC redistributable manifest")
    for row in rows:
        name = row["name"]
        if not MSVC_DLL.fullmatch(name) or name not in files:
            raise ValueError(f"Invalid MSVC redistributable entry: {name}")
        file = files[name]
        if file.stat().st_size != row["bytes"] or digest(file) != row["sha256"]:
            raise ValueError(f"MSVC redistributable bytes changed: {name}")
