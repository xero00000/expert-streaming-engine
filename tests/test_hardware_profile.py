from __future__ import annotations

import json
import os
import tempfile
import unittest
import argparse
import io
from pathlib import Path
from unittest import mock

from tools.hardware_profile import (
    HardwareProfileError,
    build_hardware_profile,
    collect_hardware_identity,
    hardware_fingerprint,
    load_hardware_profile,
    planner_profile_reasons,
    save_hardware_profile,
    stale_profile_reasons,
)
from tools.ese import _calibrate, _hardware_profile_status


def identity() -> dict[str, object]:
    return {
        "os": {"system": "Linux", "release": "test"},
        "cpu": {
            "architecture": "x86_64",
            "model": "Test CPU",
            "logical_cores": 16,
            "physical_cores": 8,
            "sockets": 1,
            "numa_nodes": 1,
        },
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-test",
                "name": "Test GPU",
                "pci_bus_id": "0000:01:00.0",
                "compute_capability": "8.6",
                "driver_version": "999.1",
                "total_memory_mib": 8192,
            }
        ],
        "nvidia_topology": "GPU0 CPU Affinity 0-7 NUMA 0",
    }


class HardwareProfileTests(unittest.TestCase):
    def test_identity_is_stable_and_ignores_volatile_free_memory(self) -> None:
        first = identity()
        second = json.loads(json.dumps(first))
        self.assertEqual(hardware_fingerprint(first), hardware_fingerprint(second))

    def test_profile_round_trip_is_atomic_and_private(self) -> None:
        profile = build_hardware_profile(
            identity(),
            {"host_memory": {"copy_gbps": 42.0}},
            benchmark_source={"binary": "ese-hardware-bench", "build": "test"},
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "nested" / "hardware-profile.json"
            self.assertEqual(save_hardware_profile(profile, target), target)
            self.assertEqual(load_hardware_profile(target), profile)
            self.assertEqual(os.stat(target).st_mode & 0o777, 0o600)

    def test_tampering_is_rejected(self) -> None:
        profile = build_hardware_profile(
            identity(), {"host_memory": {"copy_gbps": 42.0}}, benchmark_source={}
        )
        profile["hardware"]["cpu"]["model"] = "Changed"
        with self.assertRaises(HardwareProfileError):
            save_hardware_profile(profile, Path("unused"))

    def test_stale_reasons_name_changed_components(self) -> None:
        profile = build_hardware_profile(
            identity(), {"host_memory": {"copy_gbps": 42.0}}, benchmark_source={}
        )
        changed = json.loads(json.dumps(identity()))
        changed["gpus"][0]["driver_version"] = "1000.0"
        changed["cpu"]["numa_nodes"] = 2
        reasons = stale_profile_reasons(profile, changed)
        self.assertIn("CPU or NUMA topology changed", reasons)
        self.assertIn("GPU, driver, PCIe identity, or device order changed", reasons)

    def test_collect_identity_normalizes_nvidia_output(self) -> None:
        def capture(command: tuple[str, ...]) -> str | None:
            if command[:2] == ("lscpu", "--json"):
                return json.dumps(
                    {
                        "lscpu": [
                            {"field": "Architecture:", "data": "x86_64"},
                            {"field": "CPU(s):", "data": "16"},
                            {"field": "Core(s) per socket:", "data": "8"},
                            {"field": "Socket(s):", "data": "1"},
                            {"field": "NUMA node(s):", "data": "1"},
                        ]
                    }
                )
            if command[0] == "nvidia-smi" and command[1].startswith("--query-gpu="):
                return "0, GPU-b, Card, 0000:02:00.0, 8.6, 999.1, 8192"
            if command == ("nvidia-smi", "topo", "-m"):
                return "GPU0    X    PHB\nGPU0 CPU Affinity 0-7 NUMA 0"
            return None

        with mock.patch("tools.hardware_profile._linux_cpu_model", return_value="Test CPU"):
            detected = collect_hardware_identity(capture)
        self.assertEqual(detected["cpu"]["physical_cores"], 8)
        self.assertEqual(detected["gpus"][0]["index"], 0)
        self.assertEqual(detected["gpus"][0]["uuid"], "GPU-b")
        self.assertEqual(detected["nvidia_topology"], "GPU0 X PHB\nGPU0 CPU Affinity 0-7 NUMA 0")

    def test_calibrate_wraps_native_measurements_with_current_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            benchmark = root / "ese-hardware-bench"
            benchmark.write_text("placeholder", encoding="utf-8")
            target = root / "profile.json"
            args = argparse.Namespace(
                benchmark=str(benchmark),
                iterations=7,
                bytes=256 * 1024**2,
                model=None,
                output=target,
                json=True,
            )
            completed = subprocess_result = mock.Mock(
                returncode=0,
                stdout=json.dumps(
                    {
                        "benchmark_source": {"build": "test"},
                        "measurements": {"host_memory": {"copy_gbps": 42.0}},
                    }
                ),
                stderr="",
            )
            with (
                mock.patch("tools.ese.subprocess.run", return_value=completed),
                mock.patch("tools.ese.collect_hardware_identity", return_value=identity()),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(_calibrate(args), 0)
            saved = load_hardware_profile(target)
            self.assertEqual(saved["measurements"]["host_memory"]["copy_gbps"], 42.0)
            self.assertEqual(saved["benchmark_source"]["build"], "test")
            self.assertIn("--threads", saved["benchmark_source"]["command"])
            self.assertIn("8", saved["benchmark_source"]["command"])
            self.assertIs(subprocess_result, completed)

    def test_profile_status_returns_two_when_topology_is_stale(self) -> None:
        profile = build_hardware_profile(
            identity(), {"host_memory": {"copy_gbps": 42.0}}, benchmark_source={}
        )
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "profile.json"
            save_hardware_profile(profile, target)
            changed = json.loads(json.dumps(identity()))
            changed["gpus"][0]["uuid"] = "GPU-replaced"
            args = argparse.Namespace(profile=target, json=True)
            with (
                mock.patch("tools.ese.collect_hardware_identity", return_value=changed),
                mock.patch("sys.stdout", new_callable=io.StringIO),
            ):
                self.assertEqual(_hardware_profile_status(args), 2)

    def test_baseline_profile_is_rejected_by_planner_gate(self) -> None:
        profile = build_hardware_profile(
            identity(),
            {
                "cpu_moe": {"status": "measured"},
                "expert_cache_upload": {"status": "measured"},
                "cpu_cache_contention": {"status": "measured"},
            },
            benchmark_source={"planner_ready": False},
        )
        self.assertEqual(
            planner_profile_reasons(profile, identity()),
            ["calibration is baseline-only and not approved for planner decisions"],
        )


if __name__ == "__main__":
    unittest.main()
