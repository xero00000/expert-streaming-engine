"""Check the pinned build installer without downloading a toolchain in tests."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("setup_cuda_linux", ROOT / ".github/scripts/setup-cuda-linux.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CudaLinuxSetupTests(unittest.TestCase):
    def fixture(self, entries):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        archive = root / "fixture.tar.xz"
        with tarfile.open(archive, "w:xz") as package:
            for name, content in entries.items():
                item = tarfile.TarInfo(name)
                item.size = len(content)
                package.addfile(item, io.BytesIO(content))
        return root, archive, hashlib.sha256(archive.read_bytes()).hexdigest()

    def test_component_archives_merge_without_discarding_previous_components(self):
        root, archive, digest = self.fixture({"component/include/cuda.h": b"header"})
        target = root / "installed"
        target.mkdir()
        (target / "previous").write_bytes(b"preserved")
        MODULE.install_archive(archive, target, digest, archive.stat().st_size)
        self.assertEqual((target / "include/cuda.h").read_bytes(), b"header")
        self.assertEqual((target / "previous").read_bytes(), b"preserved")

    def test_hash_or_size_mismatch_does_not_modify_installation(self):
        root, archive, digest = self.fixture({"component/header": b"header"})
        for expected_hash, size in [("0" * 64, archive.stat().st_size), (digest, 1)]:
            with self.subTest(hash=expected_hash, size=size), self.assertRaisesRegex(ValueError, "size/hash mismatch"):
                MODULE.install_archive(archive, root / "installed", expected_hash, size)
            self.assertFalse((root / "installed").exists())

    def test_archive_path_traversal_does_not_modify_installation(self):
        root, archive, digest = self.fixture({"../escape": b"unsafe"})
        with self.assertRaises(tarfile.FilterError):
            MODULE.install_archive(archive, root / "installed", digest, archive.stat().st_size)
        self.assertFalse((root / "installed").exists())

    def test_lock_has_all_required_pinned_components(self):
        lock = json.loads(MODULE.LOCK.read_text())
        self.assertEqual(lock["cuda_version"], "12.8.0")
        self.assertEqual({item["name"] for item in lock["packages"]},
                         {"cuda_cudart", "cuda_nvcc", "cuda_cccl", "libcublas", "nccl"})
        for item in lock["packages"]:
            self.assertTrue(item["url"].startswith("https://developer.download.nvidia.com/compute/"))
            self.assertRegex(item["sha256"], "^[0-9a-f]{64}$")
            self.assertGreater(item["bytes"], 0)
