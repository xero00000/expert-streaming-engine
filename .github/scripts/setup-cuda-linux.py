#!/usr/bin/env python3
"""Install pinned CUDA/NCCL build components without installing a GPU driver."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from urllib.request import urlopen

LOCK = Path(__file__).with_name("cuda-linux-12.8.lock.json")


def install_archive(archive: Path, root: Path, expected_hash: str, expected_bytes: int) -> None:
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive.stat().st_size != expected_bytes or digest != expected_hash:
        raise ValueError(f"CUDA archive size/hash mismatch: {archive.name}")
    with tempfile.TemporaryDirectory(prefix="ese-cuda-extract-") as directory:
        destination = Path(directory)
        with tarfile.open(archive, "r:xz") as package:
            package.extractall(destination, filter="data")
        entries = list(destination.iterdir())
        if len(entries) != 1 or not entries[0].is_dir() or entries[0].is_symlink():
            raise ValueError("CUDA archive must have one package root")
        shutil.copytree(entries[0], root, dirs_exist_ok=True, symlinks=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if "\n" in str(root) or "\r" in str(root):
        raise ValueError("CUDA root must be a single path")
    lock = json.loads(LOCK.read_text())
    for item in lock["packages"]:
        print(f"Installing pinned {item['name']}", flush=True)
        with tempfile.TemporaryDirectory(prefix="ese-cuda-download-") as directory:
            archive = Path(directory) / "package.tar.xz"
            with urlopen(item["url"], timeout=120) as response, archive.open("wb") as target:
                shutil.copyfileobj(response, target, length=1024 * 1024)
            install_archive(archive, root, item["sha256"], item["bytes"])
    for relative in ["bin/nvcc", "include/cuda_runtime.h", "include/nccl.h", "lib/libnccl.so", "lib/libcublas.so"]:
        if not (root / relative).is_file():
            raise ValueError(f"Incomplete CUDA/NCCL toolchain: {relative}")
    # These release builds link cuBLAS/NCCL dynamically; discard unused static
    # copies to leave room for Rust builds and all three package formats in CI.
    for pattern in ["lib/libcublas*_static.a", "lib/libnccl_static.a"]:
        for path in root.glob(pattern):
            path.unlink()
    subprocess.run([str(root / "bin/nvcc"), "--version"], check=True)
    environment = {"CUDA_PATH": str(root), "CUDAToolkit_ROOT": str(root),
                   "CUDACXX": str(root / "bin/nvcc"), "NCCL_INCLUDE_DIR": str(root / "include"),
                   "NCCL_LIB_DIR": str(root / "lib"),
                   "LD_LIBRARY_PATH": str(root / "lib") + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")}
    if os.environ.get("GITHUB_ENV"):
        with Path(os.environ["GITHUB_ENV"]).open("a") as target:
            target.write("".join(f"{key}={value}\n" for key, value in environment.items()))
        with Path(os.environ["GITHUB_PATH"]).open("a") as target:
            target.write(str(root / "bin") + "\n")
    else:
        print(json.dumps(environment, indent=2))


if __name__ == "__main__":
    main()
