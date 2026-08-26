from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate-phase-d-rebalance.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("ese_phase_d_validator", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Phase D validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PhaseDValidatorDispatchTests(unittest.TestCase):
    def test_transient_only_skips_conventional_kv_and_combined_matrices(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                name: root / name
                for name in ("server", "model.gguf", "mmproj.gguf", "image.png")
            }
            for path in paths.values():
                path.write_bytes(b"fixture")
            argv = [
                str(SCRIPT),
                "--server", str(paths["server"]),
                "--model", str(paths["model.gguf"]),
                "--context", "512",
                "--target-context", "256",
                "--predict", "4",
                "--busy-predict", "64",
                "--gpu-layers", "99",
                "--expert-initial-mib", "64",
                "--expert-target-mib", "32",
                "--transient-only",
                "--mmproj", str(paths["mmproj.gguf"]),
                "--image", str(paths["image.png"]),
                "--transient-mtp-mib", "1024",
                "--transient-mmproj-mib", "2048",
            ]
            output = StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    validator,
                    "validate_transient_success",
                    return_value={"committed": True},
                ) as transient_success,
                mock.patch.object(
                    validator,
                    "validate_transient_shutdown",
                    return_value={"clean_exit": True},
                ) as transient_shutdown,
                mock.patch.object(
                    validator,
                    "validate_transient_failure",
                    side_effect=lambda _args, stage, _offset: {"stage": stage},
                ) as transient_failure,
                mock.patch.object(validator, "validate_success") as kv_success,
                mock.patch.object(validator, "validate_failure") as kv_failure,
                mock.patch.object(validator, "validate_expert_success") as expert_success,
                mock.patch.object(
                    validator, "validate_transient_combined_failure"
                ) as combined_failure,
                mock.patch.object(
                    validator, "sha256_file", side_effect=lambda path: path.name
                ),
                redirect_stdout(output),
            ):
                self.assertEqual(validator.main(), 0)

            report = json.loads(output.getvalue())
            self.assertIsNone(report["committed_path"])
            self.assertIsNone(report["injected_failure"])
            self.assertIsNone(report["expert_cache"])
            self.assertEqual(report["transient"]["evidence_scope"], "transient-only")
            self.assertEqual(
                set(report["transient"]["injected_failures"]),
                set(validator.TRANSIENT_FAILURE_STAGES),
            )
            transient_success.assert_called_once()
            transient_shutdown.assert_called_once()
            self.assertTrue(report["transient"]["graceful_shutdown"]["clean_exit"])
            self.assertEqual(transient_failure.call_count, 2)
            kv_success.assert_not_called()
            kv_failure.assert_not_called()
            expert_success.assert_not_called()
            combined_failure.assert_not_called()

    def test_combined_mode_preserves_schema_v3_failure_field(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                name: root / name
                for name in ("server", "model.gguf", "mmproj.gguf", "image.png")
            }
            for path in paths.values():
                path.write_bytes(b"fixture")
            argv = [
                str(SCRIPT),
                "--server", str(paths["server"]),
                "--model", str(paths["model.gguf"]),
                "--context", "512",
                "--target-context", "256",
                "--predict", "4",
                "--busy-predict", "64",
                "--gpu-layers", "99",
                "--expert-initial-mib", "64",
                "--expert-target-mib", "32",
                "--mmproj", str(paths["mmproj.gguf"]),
                "--image", str(paths["image.png"]),
                "--transient-mtp-mib", "1024",
                "--transient-mmproj-mib", "2048",
            ]
            output = StringIO()
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(validator, "validate_success", return_value={"kv": True}),
                mock.patch.object(validator, "validate_failure", return_value={"kv": "failure"}),
                mock.patch.object(
                    validator,
                    "validate_expert_success",
                    return_value={"accelerator_count": 1},
                ),
                mock.patch.object(
                    validator,
                    "validate_expert_failure",
                    return_value={"expert": "failure"},
                ),
                mock.patch.object(
                    validator,
                    "validate_combined_failure",
                    side_effect=lambda _args, stage, _offset: {"stage": stage},
                ),
                mock.patch.object(
                    validator,
                    "validate_transient_success",
                    return_value={"transient": True},
                ),
                mock.patch.object(
                    validator,
                    "validate_transient_shutdown",
                    return_value={"clean_exit": True},
                ),
                mock.patch.object(
                    validator,
                    "validate_transient_combined_failure",
                    side_effect=lambda _args, stage, _offset: {"stage": stage},
                ),
                mock.patch.object(
                    validator, "sha256_file", side_effect=lambda path: path.name
                ),
                redirect_stdout(output),
            ):
                self.assertEqual(validator.main(), 0)

            report = json.loads(output.getvalue())
            transient = report["transient"]
            self.assertEqual(transient["evidence_scope"], "combined-three-pool")
            self.assertIn("combined_injected_failures", transient)
            self.assertNotIn("injected_failures", transient)
            self.assertEqual(
                set(transient["combined_injected_failures"]),
                set(validator.COMBINED_FAILURE_STAGES + validator.TRANSIENT_FAILURE_STAGES),
            )


if __name__ == "__main__":
    unittest.main()
