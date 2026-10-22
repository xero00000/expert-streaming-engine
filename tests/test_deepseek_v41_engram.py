"""Numerical cross-checks against Torch's independent float8 decoders."""
import hashlib
import json
import math
from pathlib import Path
import unittest

from tools.deepseek_v41_engram import decode_affine2_row, decode_row, e4m3fn, e8m0


class EngramTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Torch is required for independent FP8 parity checks")
        cls.torch = torch

    def test_all_e4m3_encodings(self):
        torch = self.torch
        expected = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float().tolist()
        for byte, reference in enumerate(expected):
            actual = e4m3fn(byte)
            if math.isnan(reference):
                self.assertTrue(math.isnan(actual))
            else:
                self.assertEqual(actual, reference)
                self.assertEqual(math.copysign(1, actual), math.copysign(1, reference))

    def test_all_scale_encodings(self):
        torch = self.torch
        expected = torch.arange(256, dtype=torch.uint8).view(torch.float8_e8m0fnu).float().tolist()
        for byte, reference in enumerate(expected):
            if math.isnan(reference):
                self.assertTrue(math.isnan(e8m0(byte)))
            else:
                self.assertEqual(e8m0(byte), reference)

    def test_production_rows(self):
        torch = self.torch
        root = Path(__file__).resolve().parents[1] / "docs" / "deepseek-v41"
        for layer in (1, 14):
            fixture = json.loads((root / f"engram-layer{layer}-row12345.json").read_text())
            weights, scales = bytes.fromhex(fixture["weight_hex"]), bytes.fromhex(fixture["scale_hex"])
            self.assertEqual(hashlib.sha256(weights + scales).hexdigest(), fixture["payload_sha256"])
            values = torch.tensor(list(weights), dtype=torch.uint8).view(torch.float8_e4m3fn).float()
            factors = torch.tensor(list(scales), dtype=torch.uint8).view(torch.float8_e8m0fnu).float()
            expected = (values.reshape(8, 32) * factors[:, None]).flatten()
            actual = torch.tensor(decode_row(weights, scales), dtype=torch.float32)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(actual.bfloat16(), expected.bfloat16()))
            self.assertEqual(actual.tolist(), fixture["decoded_before_bf16_cast"])

    def test_all_weight_scale_combinations(self):
        torch = self.torch
        weights = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
        scales = torch.arange(256, dtype=torch.uint8).view(torch.float8_e8m0fnu).float()
        for scale in range(256):
            expected = weights * scales[scale]
            actual = torch.tensor(decode_row(bytes(range(256)), bytes([scale] * 8)))
            self.assertTrue(torch.all((actual == expected) | (actual.isnan() & expected.isnan())).item())

    def test_truncated_row_rejected(self):
        with self.assertRaises(ValueError):
            decode_row(bytes(255), bytes(8))

    def test_affine2_production_rows(self):
        torch = self.torch
        root = Path(__file__).resolve().parents[1] / "docs" / "deepseek-v41"
        for layer in (1, 14):
            fixture = json.loads((root / f"engram-affine2-layer{layer}-row12345.json").read_text())
            weights, scales, biases = (bytes.fromhex(fixture[k]) for k in ("weight_hex", "scale_hex", "bias_hex"))
            self.assertEqual(hashlib.sha256(weights + scales + biases).hexdigest(), fixture["payload_sha256"])
            # Independent uint32 unpacking instead of the production byte-wise decoder.
            import struct
            words = torch.tensor(struct.unpack("<16I", weights), dtype=torch.int64)
            codes = ((words[:, None] >> (2 * torch.arange(16))) & 3).flatten().float()
            factors = torch.tensor(list(scales), dtype=torch.uint8).view(torch.bfloat16).float()
            offsets = torch.tensor(list(biases), dtype=torch.uint8).view(torch.bfloat16).float()
            expected = (codes.reshape(4, 64) * factors[:, None] + offsets[:, None]).flatten()
            actual = torch.tensor(decode_affine2_row(weights, scales, biases))
            self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(actual.tolist(), fixture["decoded_before_bf16_cast"])


class HashFixtureTest(unittest.TestCase):
    def test_token_map_and_reference_identity(self):
        root = Path(__file__).resolve().parents[1] / "docs" / "deepseek-v41"
        trace = json.loads((root / "engram-hash-reference.json").read_text())
        expected_hash = trace.pop("trace_sha256")
        self.assertEqual(hashlib.sha256(json.dumps(trace, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), expected_hash)
        raw_map = (root / trace["config"]["token_map_file"]).read_bytes()
        self.assertEqual(hashlib.sha256(raw_map).hexdigest(), trace["config"]["token_map_sha256"])
        self.assertEqual(len(raw_map), trace["config"]["token_map_count"] * 4)
        self.assertEqual(trace["config"]["compressed_vocab_size"], 99092)

    def test_graph_reference_identity_and_negative_controls(self):
        root = Path(__file__).resolve().parents[1] / "docs" / "deepseek-v41"
        fixture = json.loads((root / "ops-reference.json").read_text())
        expected = fixture.pop("fixture_sha256")
        self.assertEqual(hashlib.sha256(json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), expected)
        self.assertEqual(len(fixture["cases"]), 17)
        for case in fixture["cases"]:
            if case["name"].startswith("random"):
                self.assertGreater(min(case["negative_controls"].values()), .001)
        provenance = json.loads((root / "provenance.json").read_text())
        self.assertEqual(hashlib.sha256((root / "config.json").read_bytes()).hexdigest(), provenance["files"]["config.json"]["sha256"])


if __name__ == "__main__":
    unittest.main()
