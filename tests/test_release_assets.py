import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.release_assets import prepare_release, verify_runtime


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "collect_release_assets", ROOT / ".github/scripts/collect-release-assets.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReleaseAssetsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bundle = self.root / "bundle with spaces"
        self.output = self.root / "assets"

    def put(self, directory, name, content=b"current"):
        path = self.bundle / directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def linux(self, version="0.2.1"):
        self.put("deb", f"ESE Studio_{version}_amd64.deb")
        self.put("rpm", f"ESE Studio-{version}-1.x86_64.rpm")
        self.put("appimage", f"ESE Studio_{version}_amd64.AppImage")
        self.put("appimage", f"ESE Studio_{version}_amd64.AppImage.sig", b"signature")

    def windows(self, version="0.2.1"):
        self.put("nsis", f"ESE Studio_{version}_x64-setup.exe")
        self.put("nsis", f"ESE Studio_{version}_x64-setup.exe.sig", b"signature")
        self.put("msi", f"ESE Studio_{version}_x64_en-US.msi")

    def collect(self, platform, variant=None):
        return MODULE.collect(platform, "0.2.1", self.bundle, self.output,
                              variant=variant if platform == "linux" else None,
                              source_commit="a" * 40)

    def test_linux_ignores_restored_old_and_adjacent_versions(self):
        self.linux("0.2.0")
        self.linux("0.2.10")
        self.linux()
        result = self.collect("linux", "cpu")
        self.assertEqual(len(result), 5)
        self.assertEqual(
            {p.name for p in result},
            {"ESE-Studio_0.2.1_linux-x86_64_cpu.deb", "ESE-Studio_0.2.1_linux-x86_64_cpu.rpm",
             "ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage", "ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage.sig",
             "artifact-set-linux-cpu.json"},
        )
        self.assertEqual((self.bundle / "deb/ESE Studio_0.2.0_amd64.deb").read_bytes(), b"current")

    def test_windows_uses_matching_current_installer_and_signature(self):
        self.windows("0.2.0")
        self.windows()
        result = self.collect("windows")
        self.assertEqual(len(result), 4)
        self.assertEqual(
            {p.name for p in result},
            {"ESE-Studio_0.2.1_windows-x64-setup.exe", "ESE-Studio_0.2.1_windows-x64-setup.exe.sig",
             "ESE-Studio_0.2.1_windows-x64.msi", "artifact-set-windows.json"},
        )
        self.assertEqual((self.output / "ESE-Studio_0.2.1_windows-x64-setup.exe.sig").read_bytes(), b"signature")

    def test_old_installer_cannot_be_renamed_as_current(self):
        self.windows("0.2.0")
        with self.assertRaisesRegex(ValueError, "expected one 0.2.1"):
            self.collect("windows")
        self.assertFalse(self.output.exists())

    def test_adjacent_version_does_not_match(self):
        self.linux("0.2.10")
        with self.assertRaisesRegex(ValueError, "expected one 0.2.1"):
            self.collect("linux", "cpu")

    def test_missing_signature_rejects_before_copying(self):
        self.windows("0.2.0")
        self.windows()
        (self.bundle / "nsis/ESE Studio_0.2.1_x64-setup.exe.sig").unlink()
        with self.assertRaisesRegex(ValueError, "matching signature"):
            self.collect("windows")
        self.assertFalse(self.output.exists())

    def test_empty_signature_rejects(self):
        self.windows()
        self.put("nsis", "ESE Studio_0.2.1_x64-setup.exe.sig", b"")
        with self.assertRaisesRegex(ValueError, "matching signature"):
            self.collect("windows")

    def test_ambiguous_current_bundle_rejects_before_copying(self):
        self.windows()
        self.put("msi", "Another Studio_0.2.1_x64_en-US.msi")
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.collect("windows")
        self.assertFalse(self.output.exists())

    def test_empty_bundle_rejects(self):
        self.linux()
        self.put("deb", "ESE Studio_0.2.1_amd64.deb", b"")
        with self.assertRaisesRegex(ValueError, "empty release bundle"):
            self.collect("linux", "cpu")

    def test_existing_output_is_preserved(self):
        self.linux()
        self.output.mkdir()
        retained = self.output / "retained"
        retained.write_bytes(b"saved")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.collect("linux", "cpu")
        self.assertEqual(retained.read_bytes(), b"saved")
        self.assertEqual(list(self.output.iterdir()), [retained])

    def test_invalid_version_rejects(self):
        with self.assertRaisesRegex(ValueError, "invalid release version"):
            MODULE.collect("linux", "../0.2.1", self.bundle, self.output)

    def test_unknown_platform_rejects(self):
        with self.assertRaisesRegex(ValueError, "unsupported release platform"):
            self.collect("other")

    def complete_release(self):
        self.linux()
        self.windows()
        merged = self.root / "merged assets"
        merged.mkdir()
        for platform, variant in [("linux", "cpu"), ("linux", "cuda"), ("windows", None)]:
            output = self.root / f"{platform}-{variant}"
            MODULE.collect(platform, "0.2.1", self.bundle, output,
                           variant=variant, source_commit="a" * 40)
            for path in output.iterdir():
                shutil.copy2(path, merged / path.name)
            shutil.rmtree(output)
        return merged

    def test_linux_variant_is_required(self):
        self.linux()
        with self.assertRaisesRegex(ValueError, "explicit cpu or cuda"):
            self.collect("linux")
        self.assertFalse(self.output.exists())

    def test_runtime_variant_mismatch_rejects_before_copying(self):
        self.linux()
        runtime = self.root / "runtime"
        marker = runtime / "build/bin/ese-runtime.json"
        marker.parent.mkdir(parents=True)
        marker.write_text(json.dumps({"cuda": False, "package_variant": "cpu"}))
        with self.assertRaisesRegex(ValueError, "does not match"):
            MODULE.collect("linux", "0.2.1", self.bundle, self.output, variant="cuda",
                           source_commit="a" * 40, runtime_root=runtime)
        self.assertFalse(self.output.exists())

    def test_windows_release_requires_separate_cpu_fallback(self):
        runtime = self.root / "runtime"
        for path, cuda in [("build/bin", True), ("build/cpu/bin", True)]:
            marker = runtime / path / "ese-runtime.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"cuda": cuda}))
        with self.assertRaisesRegex(ValueError, "both CUDA and driver-independent CPU"):
            verify_runtime("windows", None, runtime)

    def test_all_eight_packages_have_verified_checksums_and_separate_update_feeds(self):
        assets = self.complete_release()
        prepare_release("0.2.1", "owner/repository", assets)
        manifest = json.loads((assets / "release-manifest.json").read_text())
        self.assertEqual(manifest["package_count"], 8)
        self.assertEqual(manifest["source_commit"], "a" * 40)
        cpu = json.loads((assets / "latest-cpu.json").read_text())
        cuda = json.loads((assets / "latest-cuda.json").read_text())
        self.assertTrue(cpu["platforms"]["linux-x86_64"]["url"].endswith("_cpu.AppImage"))
        self.assertTrue(cuda["platforms"]["linux-x86_64"]["url"].endswith("_cuda.AppImage"))
        self.assertEqual(cpu["platforms"]["windows-x86_64"], cuda["platforms"]["windows-x86_64"])
        self.assertEqual(cpu["platforms"]["linux-x86_64"]["signature"], "signature")
        self.assertEqual((assets / "latest.json").read_bytes(), (assets / "latest-cpu.json").read_bytes())
        self.assertEqual(len((assets / "SHA256SUMS").read_text().splitlines()), 11)
        self.assertEqual(len((assets / "SHA256SUMS-windows.txt").read_text().splitlines()), 3)

    def test_tampered_package_or_signature_rejects_before_metadata_is_written(self):
        for suffix in (".AppImage", ".AppImage.sig"):
            with self.subTest(suffix=suffix):
                assets = self.complete_release()
                path = assets / ("ESE-Studio_0.2.1_linux-x86_64_cuda" + suffix)
                path.write_bytes(b"replacement")
                with self.assertRaisesRegex(ValueError, "digest or size mismatch"):
                    prepare_release("0.2.1", "owner/repository", assets)
                self.assertFalse((assets / "latest.json").exists())
                shutil.rmtree(assets)

    def test_missing_package_rejects_before_metadata_is_written(self):
        assets = self.complete_release()
        (assets / "ESE-Studio_0.2.1_linux-x86_64_cuda.rpm").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_release("0.2.1", "owner/repository", assets)
        self.assertFalse((assets / "SHA256SUMS").exists())

    def test_mixed_source_commits_reject(self):
        assets = self.complete_release()
        index = assets / "artifact-set-windows.json"
        data = json.loads(index.read_text())
        data["source_commit"] = "b" * 40
        index.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "different source commits"):
            prepare_release("0.2.1", "owner/repository", assets)
        self.assertFalse((assets / "latest.json").exists())

    def test_old_or_unexpected_assets_reject(self):
        assets = self.complete_release()
        (assets / "ese-studio_0.2.0_amd64.deb").write_bytes(b"stale")
        with self.assertRaisesRegex(ValueError, "unexpected or previously generated"):
            prepare_release("0.2.1", "owner/repository", assets)
        self.assertFalse((assets / "latest.json").exists())

    def test_artifact_set_cannot_be_relabelled_as_another_variant(self):
        assets = self.complete_release()
        index = assets / "artifact-set-linux-cuda.json"
        data = json.loads(index.read_text())
        data["variant"] = "cpu"
        index.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            prepare_release("0.2.1", "owner/repository", assets)
        self.assertFalse((assets / "latest-cuda.json").exists())

    def test_consistent_but_wrong_source_commit_is_rejected(self):
        assets = self.complete_release()
        with self.assertRaisesRegex(ValueError, "requested source commit"):
            prepare_release("0.2.1", "owner/repository", assets, source_commit="b" * 40)
        self.assertFalse((assets / "latest.json").exists())

    def test_previously_prepared_outputs_are_preserved(self):
        assets = self.complete_release()
        prepare_release("0.2.1", "owner/repository", assets)
        saved = {p.name: p.read_bytes() for p in assets.iterdir()}
        with self.assertRaisesRegex(ValueError, "unexpected or previously generated"):
            prepare_release("0.2.1", "owner/repository", assets)
        self.assertEqual(saved, {p.name: p.read_bytes() for p in assets.iterdir()})


if __name__ == "__main__":
    unittest.main()
