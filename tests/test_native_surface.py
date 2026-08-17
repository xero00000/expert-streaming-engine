from __future__ import annotations

import unittest
from pathlib import Path


class NativeSurfaceTests(unittest.TestCase):
    def test_policy_flags_exist_in_native_parser(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "common" / "common.cpp").read_text(encoding="utf-8")
        for flag in ("--fit", "--cpu-moe", "--n-cpu-moe", "--defer-experts"):
            with self.subTest(flag=flag):
                self.assertIn(flag, source)

    def test_launcher_does_not_reference_unported_cache_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "tools" / "ese.py").read_text(encoding="utf-8")
        self.assertNotIn('"--moe-cache"', launcher)
        self.assertNotIn('"--moe-cache-expert-parallel"', launcher)


if __name__ == "__main__":
    unittest.main()
