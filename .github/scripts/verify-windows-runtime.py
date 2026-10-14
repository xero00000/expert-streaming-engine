#!/usr/bin/env python3
"""Verify the staged Windows CUDA runtime without loading the NVIDIA driver."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import pefile


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise SystemExit(f"Required runtime file is missing: {path}")
    return path


def pe_imports(path: Path, *, include_delay: bool = False) -> set[str]:
    try:
        pe = pefile.PE(str(path), fast_load=True)
        directories = [pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_IMPORT"]]
        if include_delay:
            directories.append(pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT"])
        pe.parse_data_directories(directories=directories)
        entries = list(getattr(pe, "DIRECTORY_ENTRY_IMPORT", []))
        if include_delay:
            entries += list(getattr(pe, "DIRECTORY_ENTRY_DELAY_IMPORT", []))
        return {
            entry.dll.decode("ascii").lower()
            for entry in entries
        }
    except (OSError, pefile.PEFormatError, UnicodeDecodeError) as error:
        raise SystemExit(f"Could not inspect PE imports for {path}: {error}") from error


def require_import(path: Path, dependency: str) -> None:
    imports = pe_imports(path)
    if dependency.lower() not in imports:
        found = ", ".join(sorted(imports)) or "none"
        raise SystemExit(f"{path.name} does not import {dependency}; found: {found}")


def verify_cpu_runtime(runtime_root: Path, build_root: Path, launcher: Path) -> None:
    cpu_bin = runtime_root / "build" / "cpu" / "bin"
    server = require_file(cpu_bin / "llama-server.exe")
    for name in ("ggml.dll", "llama.dll", "mtmd.dll"):
        require_file(cpu_bin / name)
    marker = json.loads(require_file(cpu_bin / "ese-runtime.json").read_text(encoding="utf-8"))
    if marker.get("cuda") is not False:
        raise SystemExit("CPU fallback runtime marker must disable CUDA")
    cache = require_file(build_root / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
    if not re.search(r"^GGML_CUDA:BOOL=OFF\s*$", cache, re.MULTILINE):
        raise SystemExit("CPU fallback build must disable CUDA")
    if not re.search(r"^ESE_QWEN4EXP_PROBE_GATE:BOOL=OFF\s*$", cache, re.MULTILINE):
        raise SystemExit("CPU fallback must keep the normal Qwen loader gate closed")
    for path in cpu_bin.iterdir():
        if path.suffix.lower() in (".exe", ".dll"):
            cuda = sorted(dep for dep in pe_imports(path, include_delay=True)
                          if re.match(r"(?:nvcuda|cudart|cublas|ggml-cuda)", dep))
            if cuda:
                raise SystemExit(f"CPU fallback {path.name} imports CUDA dependencies: {cuda}")
    environment = os.environ.copy()
    for name in ("ESE_SERVER", "ESE_SERVER_BACKEND"):
        environment.pop(name, None)
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run([str(server), "--version"], env=environment, check=True,
                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    # The server returns 1 after printing requested usage (server.cpp main).
    # Require usage text as well, so loader failures cannot pass this smoke.
    usage = subprocess.run([str(server), "--help"], env=environment, check=False,
                           capture_output=True, text=True, timeout=60)
    output = usage.stdout + (usage.stderr or "")
    if usage.returncode not in (0, 1) or not all(option in output for option in ("usage:", "--model", "--port")):
        raise SystemExit(f"CPU server help smoke failed with exit {usage.returncode}: {output[:400]}")
    doctor = subprocess.run([str(launcher), "doctor", "--json"], env=environment,
                            check=True, capture_output=True, text=True, timeout=60)
    report = json.loads(doctor.stdout)
    if Path(report["server"]).resolve() != server.resolve() or report.get("server_cuda") is not False:
        raise SystemExit("Packaged launcher did not select the separate CPU runtime without visible CUDA GPUs")
    print(f"Verified driver-independent CPU PE imports, executable smoke, and automatic launcher selection from {server}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--build-root", required=True, type=Path)
    parser.add_argument("--cpu-build-root", type=Path,
                        help="Also require, execute and verify the separately staged CPU fallback")
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
    if args.cpu_build_root is not None:
        verify_cpu_runtime(runtime_root, args.cpu_build_root.resolve(), launcher)


if __name__ == "__main__":
    main()
