"""Documentation validation must be independent of installed dependencies."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "check_docs", Path(__file__).resolve().parents[1] / "scripts/check-docs.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DocumentationTests(unittest.TestCase):
    def test_dependency_readmes_are_excluded_but_maintained_links_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ["studio/node_modules/pkg/README.md", "studio/src-tauri/target/README.md"]:
                path = root / relative
                path.parent.mkdir(parents=True)
                path.write_text("[upstream source](missing-file.md)\n")
            (root / "docs").mkdir()
            document = root / "docs/guide.md"
            document.write_text("[missing ESE doc](missing.md)\n")
            with patch.object(MODULE, "ROOT", root):
                self.assertEqual(MODULE.maintained_markdown(), [document])
                errors = []
                MODULE.check_links(errors)
                self.assertEqual(len(errors), 1)
                self.assertIn("docs/guide.md:1: missing local link target: missing.md", errors[0])


if __name__ == "__main__":
    unittest.main()
