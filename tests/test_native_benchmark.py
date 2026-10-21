"""Reject incomplete benchmark receipts rather than reporting misleading speed."""
import copy
import importlib.util
import io
import json
from pathlib import Path
import time
import unittest

spec = importlib.util.spec_from_file_location('benchmark_native', Path(__file__).parents[1] / 'scripts/benchmark-native-inference.py')
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
strata_spec = importlib.util.spec_from_file_location('benchmark_strata', Path(__file__).parents[1] / 'scripts/benchmark-strata-inference.py')
strata = importlib.util.module_from_spec(strata_spec)
strata_spec.loader.exec_module(strata)


class MeasurementIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.receipt = dict(stop=True, truncated=False, tokens_evaluated=512, tokens_predicted=128,
                            timings=dict(prompt_n=512, predicted_n=128, prompt_ms=1000,
                                         predicted_ms=2000, prompt_per_second=512, predicted_per_second=64))

    def test_actual_counts_and_finite_timings_required(self):
        benchmark.validate_completion(self.receipt, 512, 128)
        for field in ('tokens_evaluated', 'tokens_predicted'):
            wrong = copy.deepcopy(self.receipt)
            wrong[field] -= 1
            with self.assertRaises(RuntimeError): benchmark.validate_completion(wrong, 512, 128)
        for field, value in [('prompt_n', 511), ('predicted_n', 127), ('prompt_ms', 0),
                             ('predicted_ms', -1), ('prompt_per_second', float('nan')),
                             ('predicted_per_second', float('inf'))]:
            wrong = copy.deepcopy(self.receipt)
            wrong['timings'][field] = value
            with self.assertRaises(RuntimeError): benchmark.validate_completion(wrong, 512, 128)
        for field, value in [('stop', False), ('truncated', True)]:
            wrong = dict(self.receipt, **{field: value})
            with self.assertRaises(RuntimeError): benchmark.validate_completion(wrong, 512, 128)

    def stream(self, values):
        wire = b': heartbeat\n\n' + b''.join(b'data: '+json.dumps(value).encode()+b'\n\n' for value in values)
        raw = io.BytesIO()
        result = benchmark.consume_stream(io.BytesIO(wire+b'data: [DONE]\n\n'), time.monotonic(), raw)
        self.assertEqual(raw.getvalue(), wire+b'data: [DONE]\n\n')
        return result

    def test_empty_token_is_distinguished_from_first_visible_text(self):
        content, final, first, visible = self.stream([
            dict(content='', tokens_predicted=1, stop=False),
            dict(content='hello', tokens_predicted=2, stop=False), self.receipt])
        self.assertEqual(content, 'hello')
        self.assertEqual(final, self.receipt)
        self.assertLessEqual(first, visible)

    def test_error_or_missing_token_or_final_rejected(self):
        for values in [[dict(error='decode failed')], [self.receipt], [dict(content='partial', stop=False)]]:
            with self.assertRaises(RuntimeError): self.stream(values)

    def test_strata_rejects_reuse_early_stop_and_wrong_receipt_layout(self):
        receipt = 'DONE 128 512 1000.0 2000.0 length 0 0 0 0 0 0 0 0.0 512 0'
        self.assertEqual(strata.parse_receipt(receipt, [1]*128, 512, 128)['predicted_per_second'], 64)
        for index, value in [(1, '127'), (2, '511'), (3, 'nan'), (4, '0'),
                             (5, 'stop'), (8, '1'), (14, '511')]:
            fields = receipt.split();fields[index] = value
            with self.assertRaises(RuntimeError): strata.parse_receipt(' '.join(fields), [1]*128, 512, 128)
        with self.assertRaises(RuntimeError): strata.parse_receipt(receipt+' 0', [1]*128, 512, 128)
        with self.assertRaises(RuntimeError): strata.parse_receipt(receipt, [1]*127, 512, 128)


if __name__ == '__main__': unittest.main()
