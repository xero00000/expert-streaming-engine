import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / ".github" / "scripts" / "prepare-restored-build.py"
UNCHANGED_MTIME = 946_684_800


class PrepareRestoredBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=self.root, check=True)
        subprocess.run(["git", "config", "user.name", "ESE test"], cwd=self.root, check=True)
        (self.root / "changed.txt").write_text("before\n", encoding="utf-8")
        (self.root / "unchanged.txt").write_text("same\n", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=self.root, check=True)
        self.base = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()

    def tearDown(self):
        self.temporary.cleanup()

    def run_script(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPT), *args],
            cwd=self.root,
            text=True,
            capture_output=True,
            check=True,
        )

    def advance_head(self):
        (self.root / "changed.txt").write_text("after\n", encoding="utf-8")
        subprocess.run(["git", "add", "changed.txt"], cwd=self.root, check=True)
        subprocess.run(["git", "commit", "-qm", "change"], cwd=self.root, check=True)

    def test_marker_preserves_changed_and_ages_unchanged_sources(self):
        self.advance_head()
        build = self.root / "build"
        build.mkdir()
        (build / ".ese-source-sha").write_text(f"{self.base}\n", encoding="utf-8")

        result = self.run_script("--build-root", str(build))

        self.assertEqual(os.stat(self.root / "unchanged.txt").st_mtime, UNCHANGED_MTIME)
        self.assertGreater(os.stat(self.root / "changed.txt").st_mtime, UNCHANGED_MTIME)
        self.assertIn("1 changed files refreshed", result.stdout)

    def test_fallback_base_migrates_an_unmarked_cache(self):
        self.advance_head()
        build = self.root / "build"
        build.mkdir()

        result = self.run_script(
            "--build-root", str(build), "--fallback-base", self.base
        )

        self.assertEqual(os.stat(self.root / "unchanged.txt").st_mtime, UNCHANGED_MTIME)
        self.assertGreater(os.stat(self.root / "changed.txt").st_mtime, UNCHANGED_MTIME)
        self.assertIn("predates source markers", result.stdout)

    def test_unknown_cache_source_keeps_checkout_timestamps(self):
        build = self.root / "build"
        build.mkdir()
        before = os.stat(self.root / "unchanged.txt").st_mtime

        result = self.run_script(
            "--build-root", str(build), "--fallback-base", "not-a-commit"
        )

        self.assertEqual(os.stat(self.root / "unchanged.txt").st_mtime, before)
        self.assertIn("retaining checkout timestamps", result.stdout)

    def test_record_writes_the_exact_head(self):
        build = self.root / "build"

        self.run_script("--build-root", str(build), "--record")

        marker = (build / ".ese-source-sha").read_text(encoding="utf-8").strip()
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=self.root, text=True
        ).strip()
        self.assertEqual(marker, head)


if __name__ == "__main__":
    unittest.main()
