"""Offline guards for the metadata-only V4.1 inventory."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.deepseek_v41_preflight import inventory, read_range


class PreflightTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "headers").mkdir()
        self.config = {"model_type": "deepseek_v41", "text_config": {
            "num_hidden_layers": 1, "n_routed_experts": 2, "num_experts_per_tok": 1,
            "engram_max_ngram_size": 3, "engram_n_heads": 2}}
        self.provenance = {"revision": "a" * 40, "repository": "test/model"}
        names = ["layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.1.w1.weight",
                 "layers.0.engram.embed.weight", "layers.0.engram.embed.scale"]
        self.index = {"metadata": {"total_size": 80}, "weight_map": {n: "one.safetensors" for n in names}}
        self.header = {"revision": "a" * 40, "shard": "one.safetensors", "header": {
            n: {"data_offsets": [i * 20, (i + 1) * 20], "shape": [4, 5], "dtype": "I8"}
            for i, n in enumerate(names)}}

    def run_inventory(self):
        (self.root / "headers" / "one.safetensors.json").write_text(json.dumps(self.header))
        return inventory(self.root, self.config, self.index, self.provenance)

    def test_scales_included_and_runtime_not_advertised(self):
        result = self.run_inventory()
        self.assertFalse(result["runtime_supported"])
        self.assertEqual(result["tensor_bytes"], 80)
        self.assertEqual(result["decode_batch1_all_main_layers"]["engram_logical_bytes"], 40)
        self.assertEqual(result["decode_batch1_all_main_layers"]["routed_bytes_if_all_miss"], 20)

    def test_wrong_revision_rejected(self):
        self.header["revision"] = "b" * 40
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.run_inventory()

    def test_missing_tensor_rejected(self):
        self.header["header"].pop("layers.0.engram.embed.scale")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.run_inventory()

    def test_overlap_rejected(self):
        self.header["header"]["layers.0.engram.embed.scale"]["data_offsets"] = [59, 79]
        with self.assertRaisesRegex(ValueError, "overlapping"):
            self.run_inventory()

    def test_wrong_total_rejected(self):
        self.index["metadata"]["total_size"] += 1
        with self.assertRaisesRegex(ValueError, "sizes disagree"):
            self.run_inventory()

    def test_missing_expert_rejected(self):
        self.config["text_config"]["n_routed_experts"] = 3
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.run_inventory()

    def test_server_ignoring_range_never_reads_body(self):
        with patch("urllib.request.urlopen") as request:
            response = request.return_value.__enter__.return_value
            response.status = 200
            with self.assertRaisesRegex(ValueError, "refusing weight download"):
                read_range("https://example.test/weight", 0, 8)
            response.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()
