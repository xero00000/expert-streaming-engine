"""The HTTP validator must distinguish plan forecasts from native receipts."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('rebalance_validation', Path(__file__).parents[1] / 'scripts/validate-phase-d-rebalance.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class NativePreparationPeakTests(unittest.TestCase):
    def setUp(self):
        current = dict(id=0, dense_bytes=50, graph_bytes=100, kv_bytes=53,
                       expert_cache_bytes=32, expert_prefill_staging_bytes=3,
                       transient_bytes=0, planned_bytes=238,
                       capacity_bytes=1024, reserve_bytes=16)
        target = dict(current, kv_bytes=25, expert_cache_bytes=10, planned_bytes=188)
        self.report = dict(current_plan=dict(context=1024, devices=[current]),
                           target_plan=dict(context=512, devices=[target]),
                           preparation_peak=dict(prepares_kv=True, prepares_expert_cache=True, devices=[dict(
                               id=0, current_live_bytes=267, target_live_bytes=233,
                               prepared_kv_bytes=40, prepared_expert_cache_bytes=10,
                               prepared_transient_bytes=0, peak_bytes=317,
                               peak_headroom_bytes=691, capacity_bytes=1024, reserve_bytes=16)]))
        self.snapshot = dict(devices=[dict(id=0, expert_cache=dict(allocated_bytes=20))],
                             memory_inventory=dict(pools=[dict(device_id=0, category=c, bytes=b)
                                                          for c, b in [(0, 60), (4, 80), (14, 40), (2, 64)]]))
        self.kv_reference = dict(preparation_peak=dict(prepares_kv=True, prepares_expert_cache=False,
                                                     devices=[dict(id=0, prepared_kv_bytes=40)]))

    def verify(self):
        with patch.object(validation, 'resources', return_value=self.snapshot), patch.object(
                validation, 'rebalance', return_value=self.kv_reference) as query:
            validation.require_combined_preparation_peak(self.report, 18083)
            query.assert_called_once_with(18083, 512, True)

    def test_native_receipts_can_differ_from_planning_estimates(self):
        before = copy.deepcopy(self.report)
        self.verify()
        self.assertEqual(before, self.report)

    def test_each_underreported_pool_is_rejected(self):
        for field in ['current_live_bytes', 'target_live_bytes', 'prepared_kv_bytes',
                      'prepared_expert_cache_bytes', 'capacity_bytes', 'reserve_bytes']:
            with self.subTest(field=field):
                item = self.report['preparation_peak']['devices'][0]
                item[field] -= 1
                with self.assertRaisesRegex(RuntimeError, 'inconsistent'):
                    self.verify()
                item[field] += 1

    def test_bad_peak_and_headroom_arithmetic_is_rejected(self):
        for field in ['peak_bytes', 'peak_headroom_bytes']:
            item = self.report['preparation_peak']['devices'][0]
            item[field] += 1
            with self.assertRaisesRegex(RuntimeError, 'arithmetic'):
                self.verify()
            item[field] -= 1

    def test_missing_native_device_is_rejected(self):
        self.kv_reference['preparation_peak']['devices'] = []
        with self.assertRaisesRegex(RuntimeError, 'topology'):
            self.verify()

    def test_live_receipt_cannot_contain_negative_bytes(self):
        self.snapshot['memory_inventory']['pools'][0]['bytes'] = -1
        with self.assertRaisesRegex(RuntimeError, 'invalid byte'):
            self.verify()

    def test_reference_must_only_prepare_kv(self):
        self.kv_reference['preparation_peak']['prepares_expert_cache'] = True
        with self.assertRaisesRegex(RuntimeError, 'isolate'):
            self.verify()


if __name__ == '__main__':
    unittest.main()
