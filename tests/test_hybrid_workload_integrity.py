from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import ese


class HybridWorkloadIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.prompt = "<|im_start|>user\nWrite a checklist.<|im_end|>\n<|im_start|>assistant\n"
        self.receipt = {
            "content": "A complete deterministic response.",
            "tokens_predicted": 128,
            "tokens_evaluated": 512,
            "truncated": False,
            "timings": {
                "prompt_n": 512, "predicted_n": 128,
                "predicted_ms": 2000.0, "predicted_per_second": 64.0,
            },
        }

    def complete(self, receipt):
        with mock.patch.object(ese, "_http_json", return_value=receipt) as request:
            result = ese._completion_result(8080, self.prompt, 128, 4242, 60.0)
        return result, request.call_args.args[2]

    def test_exact_workload_and_receipt(self):
        (digest, speed), payload = self.complete(self.receipt)
        self.assertEqual(digest, hashlib.sha256(self.receipt["content"].encode()).hexdigest())
        self.assertEqual(speed, 64.0)
        self.assertEqual(payload["prompt"], self.prompt)
        self.assertEqual(payload["n_predict"], 128)
        self.assertTrue(payload["ignore_eos"])
        self.assertFalse(payload["cache_prompt"])

    def test_rejects_incomplete_or_inconsistent_receipts(self):
        for key, value in (("tokens_predicted", 1), ("tokens_predicted", True),
                           ("tokens_evaluated", 511), ("truncated", True)):
            receipt = copy.deepcopy(self.receipt)
            receipt[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ese.ESEError):
                self.complete(receipt)
        for key, value in (("predicted_n", 1), ("predicted_ms", 0),
                           ("predicted_ms", float("nan")),
                           ("predicted_per_second", 1000000.0),
                           ("predicted_per_second", float("inf"))):
            receipt = copy.deepcopy(self.receipt)
            receipt["timings"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ese.ESEError):
                self.complete(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["tokens_predicted"] = receipt["timings"]["predicted_n"] = 1
        receipt["timings"]["predicted_ms"] = 0.001
        receipt["timings"]["predicted_per_second"] = 1000000.0
        with self.assertRaises(ese.ESEError):
            self.complete(receipt)

    def test_derives_missing_speed_from_verified_counts_and_time(self):
        del self.receipt["timings"]["predicted_per_second"]
        (digest, speed), _ = self.complete(self.receipt)
        self.assertTrue(digest)
        self.assertEqual(speed, 64.0)

    def test_all_trials_preserve_chat_generation_prefix(self):
        process = mock.Mock()
        process.poll.return_value = None
        plan = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, \
             mock.patch.object(ese, "_plan_with_port", return_value=plan), \
             mock.patch.object(ese, "_execution_environment", return_value={}), \
             mock.patch.object(ese.subprocess, "Popen", return_value=process), \
             mock.patch.object(ese, "_http_json", return_value={}), \
             mock.patch.object(ese, "_parse_expert_cache_telemetry", return_value={}), \
             mock.patch.object(ese, "_completion_result", return_value=("digest", 64.0)) as completion:
            result = ese._run_hybrid_validation_server(
                plan, label="test", prompt=self.prompt, tokens=128, samples=3, warmups=1,
                startup_timeout=10.0, request_timeout=60.0,
                log_path=Path(temporary) / "server.log",
            )
        self.assertEqual(len(completion.call_args_list), 4)
        for call in completion.call_args_list:
            self.assertEqual(call.args[1], self.prompt)
            self.assertEqual(call.args[2], 128)
        self.assertEqual(result["tokens_per_second_samples"], [64.0] * 3)
        process.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
