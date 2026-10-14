import importlib.util
import tempfile
import unittest
from pathlib import Path


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

    def collect(self, platform):
        return MODULE.collect(platform, "0.2.1", self.bundle, self.output)

    def test_linux_ignores_restored_old_and_adjacent_versions(self):
        self.linux("0.2.0")
        self.linux("0.2.10")
        self.linux()
        result = self.collect("linux")
        self.assertEqual(len(result), 4)
        self.assertEqual(
            {p.name for p in result},
            {"ese-studio_0.2.1_amd64.deb", "ese-studio-0.2.1-1.x86_64.rpm",
             "ese-studio_0.2.1_amd64.AppImage", "ese-studio_0.2.1_amd64.AppImage.sig"},
        )
        self.assertEqual((self.bundle / "deb/ESE Studio_0.2.0_amd64.deb").read_bytes(), b"current")

    def test_windows_uses_matching_current_installer_and_signature(self):
        self.windows("0.2.0")
        self.windows()
        result = self.collect("windows")
        self.assertEqual(len(result), 3)
        self.assertEqual(
            {p.name for p in result},
            {"ese-studio_0.2.1_x64-setup.exe", "ese-studio_0.2.1_x64-setup.exe.sig", "ese-studio_0.2.1_x64.msi"},
        )
        self.assertEqual((self.output / "ese-studio_0.2.1_x64-setup.exe.sig").read_bytes(), b"signature")

    def test_old_installer_cannot_be_renamed_as_current(self):
        self.windows("0.2.0")
        with self.assertRaisesRegex(ValueError, "expected one 0.2.1"):
            self.collect("windows")
        self.assertFalse(self.output.exists())

    def test_adjacent_version_does_not_match(self):
        self.linux("0.2.10")
        with self.assertRaisesRegex(ValueError, "expected one 0.2.1"):
            self.collect("linux")

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
            self.collect("linux")

    def test_existing_output_is_preserved(self):
        self.linux()
        self.output.mkdir()
        retained = self.output / "retained"
        retained.write_bytes(b"saved")
        with self.assertRaisesRegex(ValueError, "must be empty"):
            self.collect("linux")
        self.assertEqual(retained.read_bytes(), b"saved")
        self.assertEqual(list(self.output.iterdir()), [retained])

    def test_invalid_version_rejects(self):
        with self.assertRaisesRegex(ValueError, "invalid release version"):
            MODULE.collect("linux", "../0.2.1", self.bundle, self.output)

    def test_unknown_platform_rejects(self):
        with self.assertRaisesRegex(ValueError, "unsupported release platform"):
            self.collect("other")


if __name__ == "__main__":
    unittest.main()
