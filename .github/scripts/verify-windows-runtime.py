#!/usr/bin/env python3
"""Verify the staged Windows CUDA runtime without loading the NVIDIA driver."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pefile


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Required runtime file is missing: {path}")
    return path


def pe_imports(path: Path) -> set[str]:
    try:
        pe = pefile.PE(str(path), fast_load=True)
        pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]],
        )
        return {
            entry.dll.decode("ascii").lower()
            for entry in getattr(pe, "DIRECTORY_ENTRY_IMPORT", [])
        }
    except (OSError, pefile.PEFormatError, UnicodeDecodeError) as error:
        raise SystemExit(f"Could not inspect PE imports for {path}: {error}") from error


def require_import(path: Path, dependency: str) -> None:
    imports = pe_imports(path)
    if dependency.lower() not in imports:
        found = ", ".join(sorted(imports)) or "none"
        raise SystemExit(f"{path.name} does not import {dependency}; found: {found}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    build_root = args.build_root.resolve()
    runtime_bin = runtime_root / "build" / "bin"

    launcher = require_file(runtime_root / "ese.exe")
    server = require_file(runtime_bin / "llama-server.exe")
    llama = require_file(runtime_bin / "llama.dll")
    ggml = require_file(runtime_bin / "ggml.dll")
    require_file(runtime_bin / "mtmd.dll")

    for pattern, description in (
        ("cudart64_*.dll", "CUDA runtime"),
        ("cublas64_*.dll", "cuBLAS runtime"),
        ("cublasLt64_*.dll", "cuBLAS Lt runtime"),
    ):
        if not any(runtime_bin.glob(pattern)):
            raise SystemExit(f"{description} was not staged in {runtime_bin}")

    manifest = json.loads(require_file(runtime_bin / "ese-runtime.json").read_text(encoding="utf-8"))
    if manifest.get("cuda") is not True:
        raise SystemExit("CUDA runtime manifest was not staged")

    cache = require_file(build_root / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^GGML_CUDA:BOOL=ON\s*$", cache, re.MULTILINE):
        raise SystemExit("Runtime was not configured with CUDA")

    # The ESE build is monolithic: the CUDA backend is compiled into ggml.dll.
    # Verify the complete PE chain and its CUDA/driver imports without executing
    # a driver-linked binary on GitHub's GPU-less Windows runner.
    require_import(server, "llama.dll")
    require_import(server, "ggml.dll")
    require_import(llama, "ggml.dll")
    for dependency in ("cudart64_12.dll", "cublas64_12.dll", "nvcuda.dll"):
        require_import(ggml, dependency)

    print(f"Verified staged CUDA runtime and PE dependency chain from {launcher}")


if __name__ == "__main__":
    main()
