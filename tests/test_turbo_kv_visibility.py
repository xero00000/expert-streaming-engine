# SPDX-License-Identifier: MIT
"""Guard the staged Turbo KV rollout from accidental CLI exposure."""

from pathlib import Path
import re
import unittest


ROOT = Path.cwd()


class TurboKVVisibilityTests(unittest.TestCase):
    def test_native_kv_parser_does_not_expose_internal_turbo_types(self) -> None:
        source = (ROOT / "common/common.cpp").read_text(encoding="utf-8")
        match = re.search(
            r"static ggml_type kv_cache_type_from_str\(const std::string & s\) \{"
            r"(?P<body>.*?)"
            r"\n\}",
            source,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "native KV-cache parser was not found")
        body = match.group("body")
        for tier in ("turbo2", "turbo3", "turbo4", "turbo8"):
            self.assertNotIn(tier, body)

    def test_launcher_cache_choices_remain_stable(self) -> None:
        source = (ROOT / "tools/ese.py").read_text(encoding="utf-8")
        for tier in ("turbo2", "turbo3", "turbo4", "turbo8"):
            self.assertNotIn(f'"{tier}"', source)


if __name__ == "__main__":
    unittest.main()
