from __future__ import annotations

import unittest
from pathlib import Path


class NativeSurfaceTests(unittest.TestCase):
    def test_policy_flags_exist_in_native_parser(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "common" / "common.cpp").read_text(encoding="utf-8")
        for flag in (
            "--fit",
            "--cpu-moe",
            "--n-cpu-moe",
            "--defer-experts",
            "--expert-ram-cache-mib",
            "--expert-ram-staging-mib",
            "--expert-storage-backend",
            "--expert-sidecar-only",
            "--expert-vram-cache-mib",
            "--expert-vram-reserve-mib",
            "--expert-cache-min-observations",
            "--memory-policy",
            "--max-ram",
            "--reserve-vram",
            "--min-kv-quality",
            "--max-context",
            "--resource-preference",
            "--resource-plan-json",
        ):
            with self.subTest(flag=flag):
                self.assertIn(flag, source)

    def test_launcher_does_not_reference_unported_cache_cli(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (root / "tools" / "ese.py").read_text(encoding="utf-8")
        self.assertNotIn('"--moe-cache"', launcher)
        self.assertNotIn('"--moe-cache-expert-parallel"', launcher)


if __name__ == "__main__":
    unittest.main()
