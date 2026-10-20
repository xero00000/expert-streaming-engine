from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from tools.ese import (
    ESEError,
    GIB,
    GPUInfo,
    HardwareInfo,
    HostMemory,
    ModelInfo,
    auto_tensor_split,
    detect_nvidia_gpus,
    build_launch_plan,
    discover_model_shards,
    inspect_expert_geometry,
    inspect_expert_geometries,
    inspect_expert_layer_formats,
    parse_size,
    read_gguf_metadata,
    read_gguf_index,
    select_policy,
    _execution_environment,
    _baseline_hybrid_plan,
    _parse_expert_cache_telemetry,
    _plan_from_args,
    _make_parser,
    _doctor,
    _hardware_for_server,
    _server_supports_cuda,
    _select_packaged_windows_binary,
    _resolve_binary,
    _native_placement,
    _placement_hardware_identity,
    _NATIVE_VALUE_FLAGS,
    _repo_root,
    _save_hybrid_verification,
    _solve_calibrated_hybrid,
    _validated_hybrid_telemetry,
    _validated_calibration_drift,
    _validated_cpu_calibration_drift,
    hybrid_verification_reason,
    model_fingerprint,
)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_minimal_gguf(path: Path, metadata: dict[str, object]) -> None:
    values = []
    for key, value in metadata.items():
        if isinstance(value, str):
            values.append(_gguf_string(key) + struct.pack("<I", 8) + _gguf_string(value))
        elif isinstance(value, int):
            values.append(_gguf_string(key) + struct.pack("<I", 4) + struct.pack("<I", value))
        else:
            raise TypeError(value)
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, len(values)) + b"".join(values))


def write_tensor_gguf(path: Path) -> None:
    metadata = _gguf_string("general.alignment") + struct.pack("<II", 4, 32)
    tensors = b"".join(
        (
            _gguf_string("blk.0.ffn_gate_exps.weight")
            + struct.pack("<IQQQIQ", 3, 16, 32, 2, 1, 0),
            _gguf_string("output.weight") + struct.pack("<IQQIQ", 2, 16, 16, 1, 1024),
        )
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 2, 1) + metadata + tensors
    data_offset = (len(header) + 31) // 32 * 32
    path.write_bytes(header + bytes(data_offset - len(header)) + bytes(2048))


def moe_model(size: int = 40 * GIB) -> ModelInfo:
    return ModelInfo(
        requested_path=Path("/model.gguf"),
        shards=(Path("/model.gguf"),),
        total_bytes=size,
        metadata={
            "general.architecture": "gpt-oss",
            "gpt-oss.expert_count": 128,
            "gpt-oss.expert_used_count": 4,
            "gpt-oss.block_count": 36,
        },
    )


def dense_model(size: int = 4 * GIB) -> ModelInfo:
    return ModelInfo(
        requested_path=Path("/dense.gguf"),
        shards=(Path("/dense.gguf"),),
        total_bytes=size,
        metadata={
            "general.architecture": "llama",
            "llama.block_count": 32,
        },
    )


def hardware(*free_gib: int, ram_available: int = 100) -> HardwareInfo:
    return HardwareInfo(
        host=HostMemory(128 * GIB, ram_available * GIB),
        gpus=tuple(
            GPUInfo(index, f"GPU {index}", max(free + 2, 8) * GIB, free * GIB)
            for index, free in enumerate(free_gib)
        ),
        logical_cpus=32,
    )


def verified_telemetry_summary() -> dict[str, int]:
    return {
        "layers": 4,
        "misses": 8,
        "route_positions": 16,
        "gpu_route_positions": 8,
        "forced_fallbacks": 0,
        "predicted_upload_ns_per_expert": 100,
        "upload_calibration_drift_ppm": 1_000_000,
        "cpu_compute_ns": 1_000,
        "cpu_compute_calls": 8,
        "predicted_cpu_ns_per_expert": 100,
        "cpu_calibration_drift_ppm": 1_250_000,
    }


class LauncherTests(unittest.TestCase):
    def test_native_device_order_controls_hardware_and_automatic_split(self) -> None:
        for device, expected, split in [
            ("CUDA2,CUDA0", [2, 0], "75,25"),
            ("CUDA2", [2], None),
            ("none", [], None),
            ("CPU", [], None),
        ]:
            with self.subTest(device=device):
                plan = build_launch_plan(
                    model=dense_model(), hardware=hardware(8, 16, 24),
                    binary=Path("/server"), policy="resident",
                    extra_args=("--device", device),
                )
                self.assertEqual([gpu.index for gpu in plan.hardware.gpus], expected)
                self.assertEqual(plan.hardware.free_vram, sum(hardware(8, 16, 24).gpus[i].free_bytes for i in expected))
                if split is None:
                    self.assertNotIn("--tensor-split", plan.arguments)
                else:
                    self.assertEqual(plan.arguments[plan.arguments.index("--tensor-split") + 1], split)

    def test_native_single_gpu_uses_local_index_in_selected_order(self) -> None:
        plan = build_launch_plan(
            model=dense_model(), hardware=hardware(8, 16, 24),
            binary=Path("/server"), policy="resident",
            extra_args=("-dev", "CUDA2,CUDA0", "-sm", "none", "-mg", "1"),
        )
        self.assertEqual([gpu.index for gpu in plan.hardware.gpus], [0])
        self.assertEqual(plan.hardware.free_vram, 8 * GIB)
        self.assertNotIn("--tensor-split", plan.arguments)
        self.assertEqual(plan.arguments[-6:], ("--device", "CUDA2,CUDA0", "--split-mode", "none", "--main-gpu", "1"))

    def test_non_cuda_device_passthrough_keeps_nvidia_budgets_empty(self) -> None:
        for device in ("Metal", "Vulkan0", "ROCm0", "localhost:9000|0"):
            with self.subTest(device=device):
                selected = _native_placement(hardware(), ("--device", device))
                self.assertEqual(selected.hardware.gpus, ())
                self.assertEqual(selected.extra_args, ("--device", device))

    def test_native_gpu_layers_and_split_override_launcher_values(self) -> None:
        for alias in ("-ngl", "--gpu-layers", "--n-gpu-layers", "--n_gpu_layers"):
            with self.subTest(alias=alias):
                plan = build_launch_plan(
                    model=dense_model(), hardware=hardware(8, 16, 24),
                    binary=Path("/server"), policy="resident", gpu_layers=18,
                    tensor_split="1,1,1", extra_args=(alias, "0", "-ts", "0/0/100"),
                )
                self.assertEqual(plan.arguments.count("-ngl"), 1)
                self.assertEqual(plan.arguments[plan.arguments.index("-ngl") + 1], "0")
                self.assertEqual(plan.hardware.gpus, ())
                self.assertNotIn("--memory-policy", plan.arguments)
                self.assertNotIn("--fit", plan.arguments)
                self.assertEqual(plan.arguments.count("--tensor-split"), 1)
                self.assertEqual(plan.arguments[plan.arguments.index("--tensor-split") + 1], "0/0/100")
        automatic = _native_placement(hardware(8), ("-ngl", "-1"))
        self.assertIsNone(automatic.gpu_layers)
        cpu = build_launch_plan(
            model=dense_model(), hardware=hardware(2, 24), binary=Path("/server"),
            policy="resident", context=4096, extra_args=("--device", "CUDA0", "-ngl", "0"),
        )
        self.assertEqual(cpu.arguments[cpu.arguments.index("-ctk") + 1], "q8_0")

    def test_native_placement_rejects_unavailable_or_invalid_overrides(self) -> None:
        for extra, reason in [
            (("--device", "CUDA3"), "unavailable CUDA"),
            (("--device", "CUDA0,CUDA0"), "duplicate"),
            (("--device", "CUDA0,"), "unavailable CUDA"),
            (("--device", "CUDA0", "--main-gpu", "1"), "outside"),
            (("--main-gpu", "-1"), "negative"),
            (("-ngl", "NaN"), "integer"),
            (("--split-mode", "row"), "split-mode"),
            (("--tensor-split", "nan,1"), "finite"),
            (("--tensor-split", "inf,1"), "finite"),
            (("--tensor-split", "-1,1"), "nonnegative"),
            (("--tensor-split", "1,1,1,1"), "more weights"),
            (("--tensor-split", "1,"), "numbers"),
            (("--device",), "requires a value"),
        ]:
            with self.subTest(extra=extra), self.assertRaisesRegex(ESEError, reason):
                _native_placement(hardware(8, 16, 24), extra)
        with self.assertRaisesRegex(ESEError, "unavailable CUDA"):
            _native_placement(hardware(), ("--device", "CUDA0"))
        zero = _native_placement(hardware(8, 16), ("--tensor-split", "0,0"))
        self.assertEqual(zero.tensor_split, "0,0")

    def test_native_value_arguments_do_not_become_placement_flags(self) -> None:
        placement_flags = {"--device", "-dev", "--main-gpu", "-mg", "--split-mode", "-sm", "--tensor-split", "-ts", "--gpu-layers", "--n-gpu-layers", "-ngl"}
        for option in sorted(_NATIVE_VALUE_FLAGS - placement_flags):
            with self.subTest(option=option):
                selected = _native_placement(hardware(8, 16, 24), (option, "--device", "--no-mmap", "--device", "CUDA2"))
                self.assertEqual([gpu.index for gpu in selected.hardware.gpus], [2])
                self.assertEqual(selected.extra_args[:3], (option, "--device", "--no-mmap"))

    def test_native_value_argument_table_matches_the_current_parser(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "common/common.cpp").read_text()
        mask = re.sub(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'', lambda match: " " * len(match.group()), source, flags=re.S)
        flags = set()
        for match in re.finditer(r"\bif\s*\(\s*arg\s*==", mask):
            start = mask.find("{", match.end())
            end, depth = start + 1, 1
            while depth and end < len(mask):
                depth += (mask[end] == "{") - (mask[end] == "}")
                end += 1
            if "CHECK_ARG" in mask[start:end]:
                flags.update(re.findall(r'arg\s*==\s*"([^"\n]+)"', source[match.start():start]))
        self.assertEqual(_NATIVE_VALUE_FLAGS, flags)

    def test_native_selected_uuid_order_binds_calibration_identity(self) -> None:
        detected = dataclasses.replace(hardware(8, 16, 24), gpus=tuple(
            dataclasses.replace(gpu, uuid=f"GPU-{gpu.index}") for gpu in hardware(8, 16, 24).gpus
        ))
        original = {"cuda_devices": ["GPU-0", "GPU-1", "GPU-2"], "cpu": {}, "gpus": []}
        selected = _native_placement(detected, ("--device", "CUDA2,CUDA0"))
        identity = _placement_hardware_identity(selected.hardware, original)
        self.assertEqual(identity["cuda_devices"], ["GPU-2", "GPU-0"])
        self.assertEqual(original["cuda_devices"], ["GPU-0", "GPU-1", "GPU-2"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            write_minimal_gguf(path, {"general.architecture": "gpt-oss", "gpt-oss.expert_count": 8, "gpt-oss.expert_used_count": 3, "gpt-oss.block_count": 4})
            args = _make_parser().parse_args(["plan", str(path), "--policy", "cache"])
            args.extra = ("--device", "CUDA2,CUDA0")
            with (
                mock.patch("tools.ese._resolve_binary", return_value=Path("/server")),
                mock.patch("tools.ese._hardware_for_server", return_value=detected),
                mock.patch("tools.ese.collect_hardware_identity", return_value=original),
                mock.patch("tools.ese.calibrated_hybrid_gpu_experts", return_value=(0, "automatic hybrid routing disabled: fixture")) as calibrate,
            ):
                plan = _plan_from_args(args)
            self.assertEqual(calibrate.call_args.args[2]["cuda_devices"], ["GPU-2", "GPU-0"])
            self.assertEqual([gpu.index for gpu in plan.hardware.gpus], [2, 0])

    def test_native_cpu_or_component_split_skips_incompatible_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            write_minimal_gguf(path, {"general.architecture": "gpt-oss", "gpt-oss.expert_count": 8, "gpt-oss.expert_used_count": 3, "gpt-oss.block_count": 4})
            for extra, reason in [
                (("--device", "none"), "no CUDA device"),
                (("--split-mode", "graph"), "component layouts"),
                (("--split-mode", "attn"), "component layouts"),
            ]:
                with self.subTest(extra=extra):
                    args = _make_parser().parse_args(["plan", str(path), "--policy", "cache"])
                    args.extra = extra
                    with (
                        mock.patch("tools.ese._resolve_binary", return_value=Path("/server")),
                        mock.patch("tools.ese._hardware_for_server", return_value=hardware(8, 16, 24)),
                        mock.patch("tools.ese.collect_hardware_identity") as identity,
                        mock.patch("tools.ese.calibrated_hybrid_gpu_experts") as calibrate,
                    ):
                        plan = _plan_from_args(args)
                    self.assertEqual(plan.hybrid_gpu_experts, 0)
                    self.assertIn(reason, plan.hybrid_selection)
                    identity.assert_not_called()
                    calibrate.assert_not_called()

    def test_gpu_detection_matches_cuda_visible_uuid_order(self) -> None:
        def capture(command):
            if "uuid" in command[1]:
                return "0, GPU-a, Card A, 8192, 6144\n1, GPU-b, Card B, 10240, 9216\n2, GPU-c, Card C, 8192, 7168"
            return "0, Card A, 8192, 6144\n1, Card B, 10240, 9216\n2, Card C, 8192, 7168"

        for visible, names, nvml_indices, split in [
            (("GPU-b", "GPU-a", "GPU-c"), ["Card B", "Card A", "Card C"], [1, 0, 2], "41,27,32"),
            (("GPU-b", "GPU-a"), ["Card B", "Card A"], [1, 0], "60,40"),
            (("GPU-c",), ["Card C"], [2], None),
            ((), [], [], None),
        ]:
            with (
                self.subTest(visible=visible),
                mock.patch("tools.ese._run_capture", side_effect=capture),
                mock.patch("tools.ese.cuda_device_uuids", return_value=visible, create=True),
            ):
                detected = detect_nvidia_gpus()
                self.assertEqual([gpu.name for gpu in detected], names)
                self.assertEqual([gpu.index for gpu in detected], list(range(len(visible))))
                self.assertEqual([gpu.nvml_index for gpu in detected], nvml_indices)
                self.assertEqual([gpu.uuid for gpu in detected], list(visible))
                self.assertEqual(auto_tensor_split(detected), split)

    def test_gpu_detection_rejects_unknown_cuda_mapping(self) -> None:
        for visible, reason in [(None, "cannot discover CUDA"), (("GPU-missing",), "memory telemetry")]:
            with (
                self.subTest(visible=visible),
                mock.patch("tools.ese._run_capture", return_value="0, GPU-a, Card A, 8192, 6144"),
                mock.patch("tools.ese.cuda_device_uuids", return_value=visible),
            ):
                with self.assertRaisesRegex(ESEError, reason):
                    detect_nvidia_gpus()

    def test_gpu_detection_rejects_duplicate_telemetry_uuid(self) -> None:
        with mock.patch("tools.ese._run_capture", return_value=(
            "0, GPU-a, Card A, 8192, 6144\n1, GPU-a, Card A, 8192, 6144"
        )):
            with self.assertRaisesRegex(ESEError, "duplicate GPU identities"):
                detect_nvidia_gpus()

    def test_gpu_detection_without_telemetry_remains_empty(self) -> None:
        with (
            mock.patch("tools.ese._run_capture", return_value=None),
            mock.patch("tools.ese.cuda_device_uuids") as discover,
        ):
            self.assertEqual(detect_nvidia_gpus(), ())
            discover.assert_not_called()

    def test_cpu_runtime_does_not_require_cuda_discovery(self) -> None:
        with (
            mock.patch("tools.ese._server_supports_cuda", return_value=False),
            mock.patch("tools.ese.detect_hardware", side_effect=ESEError("bad CUDA driver")) as detect,
            mock.patch("tools.ese.detect_host_memory", return_value=HostMemory(128 * GIB, 100 * GIB)),
        ):
            detected = _hardware_for_server(Path("/cpu-server"))
            self.assertEqual(detected.gpus, ())
            self.assertEqual(detected.host.available_bytes, 100 * GIB)
            detect.assert_not_called()

    def test_calibrated_cpu_endpoint_disables_decode_gpu_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "model.gguf"
            write_minimal_gguf(model_path, {
                "general.architecture": "gpt-oss", "gpt-oss.expert_count": 8,
                "gpt-oss.expert_used_count": 3, "gpt-oss.block_count": 4,
            })
            key = {
                "ggml_type_id": 1, "input_width": 32,
                "expert_width": 32, "bytes_per_expert_component": 64,
            }
            for policy, cache_args in [
                ("cache", []), ("cache", ["--expert-vram-cache", "256MiB"]),
                ("stream", []), ("stream", ["--expert-vram-cache", "256MiB"]),
            ]:
                args = _make_parser().parse_args([
                    "plan", str(model_path), "--policy", policy,
                    "--expert-storage-backend", "pread", *cache_args,
                ])
                args.extra = ()
                for cpu_cost, upload_cost, expected_cache in [
                    (1.0, 1_000_000.0, False),
                    (1_000_000.0, 1.0, True),
                ]:
                    with self.subTest(policy=policy, cache_args=cache_args, cpu_cost=cpu_cost):
                        profile = {"measurements": {"cpu_cache_contention": {"devices": [{
                            "backend": "CUDA0", "profiles": [{
                                **key, "cpu_ns_per_expert_component": cpu_cost,
                                "upload_ns_per_expert_component": upload_cost,
                            }],
                        }]}}}
                        choice = _solve_calibrated_hybrid(profile, [(tuple(key.values()),)], 3)
                        self.assertEqual(choice[0], 0)
                        with (
                            mock.patch("tools.ese._resolve_binary", return_value=Path("/server")),
                            mock.patch("tools.ese._hardware_for_server", return_value=hardware(20)),
                            mock.patch("tools.ese.collect_hardware_identity", return_value={"cpu": {}, "gpus": []}),
                            mock.patch("tools.ese.calibrated_hybrid_gpu_experts", return_value=choice),
                        ):
                            plan = _plan_from_args(args)
                        self.assertEqual("--expert-vram-cache-mib" in plan.arguments, expected_cache)
                        self.assertIn("--cpu-moe", plan.arguments)
                        self.assertEqual(plan.hybrid_gpu_experts, 0)
                        self.assertEqual(plan.hybrid_selection, choice[1])

    def test_cpu_endpoint_allows_explicit_mixed_candidate_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "model.gguf"
            write_minimal_gguf(model_path, {
                "general.architecture": "gpt-oss", "gpt-oss.expert_count": 8,
                "gpt-oss.expert_used_count": 3, "gpt-oss.block_count": 4,
            })
            args = _make_parser().parse_args([
                "plan", str(model_path), "--policy", "cache",
                "--expert-storage-backend", "pread",
                "--expert-vram-cache", "256MiB", "--hybrid-candidate", "1",
            ])
            args.extra = ()
            with (
                mock.patch("tools.ese._resolve_binary", return_value=Path("/server")),
                mock.patch("tools.ese._hardware_for_server", return_value=hardware(20)),
                mock.patch("tools.ese.collect_hardware_identity", return_value={"cpu": {}, "gpus": []}),
                mock.patch("tools.ese.calibrated_hybrid_gpu_experts", return_value=(
                    0, "calibration selected the established all-CPU expert path",
                )),
                mock.patch("tools.ese._calibrated_expert_cost_bounds", return_value=(100.0, 200.0)),
            ):
                plan = _plan_from_args(args, require_hybrid_verification=False)
            self.assertEqual(plan.hybrid_gpu_experts, 1)
            position = plan.arguments.index("--expert-vram-cache-mib")
            self.assertEqual(plan.arguments[position + 1], "256")
            position = plan.arguments.index("--expert-hybrid-gpu-experts")
            self.assertEqual(plan.arguments[position + 1], "1")

    def test_automatic_plan_requires_matching_workload_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_path = root / "model.gguf"
            write_minimal_gguf(model_path, {
                "general.architecture": "gpt-oss", "gpt-oss.expert_count": 8,
                "gpt-oss.expert_used_count": 3, "gpt-oss.block_count": 4,
            })
            evidence = root / "hybrid.json"
            args = argparse.Namespace(
                model=model_path, context=4096, slots=1, port=8080, policy="cache",
                no_auto_hybrid=False, hardware_profile=root / "profile.json",
                hybrid_candidate=None,
                hybrid_verification=evidence, binary="/server", host="127.0.0.1",
                threads=4, batch_threads=8, batch_size=64, ubatch_size=32,
                kv="q8_0", reserve_vram=GIB, gpu_resident_moe=None,
                tensor_split=None, prefetch_tail=0, expert_ram_cache=256 * 1024**2,
                expert_ram_staging=32 * 1024**2, expert_vram_cache=256 * 1024**2,
                expert_storage_backend="pread", expert_cache_min_observations=1,
                extra=(),
            )
            identity = {"cpu": {"model": "test"}, "gpus": [{"uuid": "GPU-test"}]}
            with (
                mock.patch("tools.ese._hardware_for_server", return_value=hardware(20)),
                mock.patch("tools.ese.collect_hardware_identity", return_value=identity),
                mock.patch(
                    "tools.ese.calibrated_hybrid_gpu_experts",
                    return_value=(1, "calibrated"),
                ),
                mock.patch(
                    "tools.ese._calibrated_expert_cost_bounds",
                    return_value=(100.0, 200.0),
                ),
            ):
                blocked = _plan_from_args(args)
                self.assertEqual(blocked.hybrid_gpu_experts, 0)
                self.assertIn("validate-hybrid", blocked.hybrid_selection)

                candidate = _plan_from_args(args, require_hybrid_verification=False)
                _save_hybrid_verification(evidence, candidate, identity, {
                    "passed": True, "output_parity": True, "speedup": 1.2,
                    "minimum_speedup": 1.02, "telemetry_valid": True,
                    "telemetry_summary": verified_telemetry_summary(),
                })
                allowed = _plan_from_args(args)
                self.assertEqual(allowed.hybrid_gpu_experts, 1)

                args.hybrid_candidate = 2
                blocked_candidate = _plan_from_args(args)
                self.assertEqual(blocked_candidate.hybrid_gpu_experts, 0)
                candidate_two = _plan_from_args(args, require_hybrid_verification=False)
                self.assertEqual(candidate_two.hybrid_gpu_experts, 2)
                _save_hybrid_verification(evidence, candidate_two, identity, {
                    "passed": True, "output_parity": True, "speedup": 1.3,
                    "minimum_speedup": 1.02, "telemetry_valid": True,
                    "telemetry_summary": verified_telemetry_summary(),
                })
                allowed_candidate = _plan_from_args(args)
                self.assertEqual(allowed_candidate.hybrid_gpu_experts, 2)

                args.policy = "resident"
                with self.assertRaisesRegex(ESEError, "cache or stream"):
                    _plan_from_args(args)

    def test_hybrid_workload_evidence_is_model_hardware_and_plan_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_path = root / "model.gguf"
            write_minimal_gguf(
                model_path,
                {
                    "general.architecture": "gpt-oss",
                    "gpt-oss.expert_count": 8,
                    "gpt-oss.expert_used_count": 2,
                    "gpt-oss.block_count": 4,
                },
            )
            model = ModelInfo(
                requested_path=model_path,
                shards=(model_path,),
                total_bytes=model_path.stat().st_size,
                metadata={
                    "general.architecture": "gpt-oss",
                    "gpt-oss.expert_count": 8,
                    "gpt-oss.expert_used_count": 2,
                    "gpt-oss.block_count": 4,
                },
            )
            plan = build_launch_plan(
                model=model,
                hardware=hardware(20),
                binary=Path("/server"),
                policy="cache",
                context=4096,
                expert_storage_backend="pread",
                expert_vram_cache=256 * 1024**2,
                hybrid_gpu_experts=1,
                hybrid_cpu_ns_per_expert=100,
                hybrid_upload_ns_per_expert=200,
            )
            evidence = root / "hybrid.json"
            identity = {"cpu": {"model": "test"}, "gpus": [{"uuid": "GPU-test"}]}
            result = {
                "passed": True,
                "output_parity": True,
                "speedup": 1.25,
                "minimum_speedup": 1.02,
                "telemetry_valid": True,
                "telemetry_summary": verified_telemetry_summary(),
            }
            _save_hybrid_verification(evidence, plan, identity, result)
            self.assertIsNone(hybrid_verification_reason(plan, identity, evidence))
            self.assertEqual(os.stat(evidence).st_mode & 0o777, 0o600)

            changed_identity = {"cpu": {"model": "other"}, "gpus": [{"uuid": "GPU-test"}]}
            self.assertIn(
                "no matching workload A/B verification",
                hybrid_verification_reason(plan, changed_identity, evidence) or "",
            )
            changed_workload = build_launch_plan(
                model=model,
                hardware=hardware(20),
                binary=Path("/server"),
                policy="cache",
                context=8192,
                expert_storage_backend="pread",
                expert_vram_cache=256 * 1024**2,
                hybrid_gpu_experts=1,
                hybrid_cpu_ns_per_expert=100,
                hybrid_upload_ns_per_expert=200,
            )
            self.assertIn(
                "no matching workload A/B verification",
                hybrid_verification_reason(changed_workload, identity, evidence) or "",
            )
            saved = json.loads(evidence.read_text(encoding="utf-8"))
            recorded_model_id = next(iter(saved["entries"].values()))["model_fingerprint"]
            model_path.write_bytes(model_path.read_bytes() + b"changed")
            self.assertNotEqual(model_fingerprint(model), recorded_model_id)

    def test_losing_hybrid_evidence_fails_closed_and_baseline_removes_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            model_path = Path(temp) / "model.gguf"
            write_minimal_gguf(model_path, {"general.architecture": "gpt-oss"})
            model = ModelInfo(model_path, (model_path,), model_path.stat().st_size, {
                "general.architecture": "gpt-oss", "gpt-oss.expert_count": 8,
                "gpt-oss.expert_used_count": 2, "gpt-oss.block_count": 4,
            })
            plan = build_launch_plan(
                model=model, hardware=hardware(20), binary=Path("/server"), policy="cache",
                expert_storage_backend="pread", expert_vram_cache=256 * 1024**2,
                hybrid_gpu_experts=1,
                hybrid_cpu_ns_per_expert=100, hybrid_upload_ns_per_expert=200,
            )
            baseline = _baseline_hybrid_plan(plan)
            self.assertNotIn("--expert-hybrid-gpu-experts", baseline.arguments)
            self.assertNotIn("--expert-hybrid-cpu-ns-per-expert", baseline.arguments)
            self.assertNotIn("--expert-hybrid-upload-ns-per-expert", baseline.arguments)
            identity = {"cpu": {}, "gpus": []}
            evidence = Path(temp) / "hybrid.json"
            _save_hybrid_verification(evidence, plan, identity, {
                "passed": True, "output_parity": True, "speedup": 1.2,
                "minimum_speedup": 1.02,
            })
            self.assertIn(
                "failed live hybrid telemetry checks",
                hybrid_verification_reason(plan, identity, evidence) or "",
            )
            _save_hybrid_verification(evidence, plan, identity, {
                "passed": False, "output_parity": True, "speedup": 0.95,
                "minimum_speedup": 1.02, "telemetry_valid": True,
                "telemetry_summary": verified_telemetry_summary(),
            })
            self.assertIn(
                "did not beat",
                hybrid_verification_reason(plan, identity, evidence) or "",
            )
            _save_hybrid_verification(evidence, plan, identity, {
                "passed": True, "output_parity": True, "speedup": float("nan"),
                "minimum_speedup": 1.02, "telemetry_valid": True,
                "telemetry_summary": verified_telemetry_summary(),
            })
            self.assertIn(
                "invalid performance evidence",
                hybrid_verification_reason(plan, identity, evidence) or "",
            )

    def test_hybrid_telemetry_must_reconcile_and_prove_mixed_routing(self) -> None:
        layer = {
            "level": "vram-layer", "layer": 4, "routes": 6,
            "route_positions": 12, "gpu_route_positions": 6,
            "route_readback_ns": 100, "hits": 3, "misses": 3,
            "lease_acquire_ns": 200, "lease_uploads": 9,
            "transfer_submit_ns": 300, "transfer_wait_ns": 400,
            "load_bytes": 500, "cpu_compute_ns": 600,
            "cpu_compute_calls": 6,
        }
        total = {
            "level": "vram-total", "hits": 3, "misses": 3,
            "lease_uploads": 9, "transfer_submit_ns": 300,
            "transfer_wait_ns": 400, "load_bytes": 500,
            "forced_fallbacks": 0, "cpu_compute_ns": 600,
            "cpu_compute_calls": 6,
        }
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "server.log"
            log.write_text(
                "noise\nexpert_cache_stats: " + json.dumps(layer) +
                "\nexpert_cache_stats: " + json.dumps(total) + "\n",
                encoding="utf-8",
            )
            telemetry = _parse_expert_cache_telemetry(log)
        summary = _validated_hybrid_telemetry(telemetry)
        self.assertEqual(summary["layers"], 1)
        self.assertEqual(summary["gpu_route_positions"], 6)
        self.assertLess(_validated_calibration_drift(summary, 400.0), 4.0)
        self.assertLess(_validated_cpu_calibration_drift(summary, 100.0, 2), 4.0)

        broken = json.loads(json.dumps(telemetry))
        broken["totals"][0]["load_bytes"] = 499
        with self.assertRaisesRegex(ESEError, "does not reconcile"):
            _validated_hybrid_telemetry(broken)
        fallback = json.loads(json.dumps(telemetry))
        fallback["totals"][0]["forced_fallbacks"] = 1
        with self.assertRaisesRegex(ESEError, "forbidden host-tensor fallback"):
            _validated_hybrid_telemetry(fallback)
        revoked = json.loads(json.dumps(telemetry))
        revoked["guard_failures"] = [{"status": 3}]
        with self.assertRaisesRegex(ESEError, "runtime guard revoked"):
            _validated_hybrid_telemetry(revoked)
        with self.assertRaisesRegex(ESEError, "contradicts calibration"):
            _validated_calibration_drift(summary, 10.0)
        with self.assertRaisesRegex(ESEError, "CPU-branch timing contradicts"):
            _validated_cpu_calibration_drift(summary, 1.0, 2)

    def test_calibrated_hybrid_solver_is_conservative_across_devices(self) -> None:
        key = {
            "ggml_type_id": 16,
            "input_width": 4096,
            "expert_width": 2048,
            "bytes_per_expert_component": 2162688,
        }
        profile = {
            "measurements": {
                "cpu_cache_contention": {
                    "devices": [
                        {
                            "backend": "CUDA0",
                            "profiles": [{
                                **key,
                                "cpu_ns_per_expert_component": 120000.0,
                                "upload_ns_per_expert_component": 90000.0,
                            }],
                        },
                        {
                            "backend": "CUDA1",
                            "profiles": [{
                                **key,
                                "cpu_ns_per_expert_component": 120000.0,
                                "upload_ns_per_expert_component": 150000.0,
                            }],
                        },
                    ]
                }
            }
        }
        uploads, reason = _solve_calibrated_hybrid(
            profile, [(tuple(key.values()), tuple(key.values()))], 7
        )
        self.assertEqual(uploads, 3)
        self.assertIn("3 GPU and 4 CPU", reason)

        single_device = {
            "measurements": {
                "cpu_cache_contention": {
                    "devices": profile["measurements"]["cpu_cache_contention"]["devices"][:1]
                }
            }
        }
        uploads, _ = _solve_calibrated_hybrid(
            single_device, [(tuple(key.values()), tuple(key.values()))], 7
        )
        self.assertEqual(uploads, 4)  # same golden case as the native solver

        missing = dict(key)
        missing["ggml_type_id"] = 19
        uploads, reason = _solve_calibrated_hybrid(
            profile, [(tuple(missing.values()),)], 7
        )
        self.assertEqual(uploads, 0)
        self.assertIn("no exact component match", reason)

        for cpu_cost, upload_cost, expected in (
            (1.0, 1_000_000.0, "all-CPU"),
            (1_000_000.0, 1.0, "all-GPU"),
        ):
            extreme = {
                "measurements": {
                    "cpu_cache_contention": {
                        "devices": [{
                            "backend": "CUDA0",
                            "profiles": [{
                                **key,
                                "cpu_ns_per_expert_component": cpu_cost,
                                "upload_ns_per_expert_component": upload_cost,
                            }],
                        }]
                    }
                }
            }
            uploads, reason = _solve_calibrated_hybrid(
                extreme, [(tuple(key.values()),)], 7
            )
            self.assertEqual(uploads, 0)
            self.assertIn(expected, reason)

    def test_cache_plan_applies_a_verified_hybrid_split(self) -> None:
        plan = build_launch_plan(
            model=moe_model(),
            hardware=hardware(20),
            binary=Path("/server"),
            policy="cache",
            expert_vram_cache=256 * 1024**2,
            expert_storage_backend="pread",
            hybrid_gpu_experts=2,
            hybrid_cpu_ns_per_expert=100,
            hybrid_upload_ns_per_expert=200,
            hybrid_selection="verified test profile",
        )
        position = plan.arguments.index("--expert-hybrid-gpu-experts")
        self.assertEqual(plan.arguments[position + 1], "2")
        cpu_guard = plan.arguments.index("--expert-hybrid-cpu-ns-per-expert")
        upload_guard = plan.arguments.index("--expert-hybrid-upload-ns-per-expert")
        self.assertEqual(plan.arguments[cpu_guard + 1], "100")
        self.assertEqual(plan.arguments[upload_guard + 1], "200")
        self.assertEqual(plan.hybrid_gpu_experts, 2)
        self.assertEqual(plan.as_dict()["hybrid_routing"]["selection"], "verified test profile")

        mmap_plan = build_launch_plan(
            model=moe_model(),
            hardware=hardware(20),
            binary=Path("/server"),
            policy="cache",
            expert_vram_cache=256 * 1024**2,
            hybrid_gpu_experts=2,
            hybrid_cpu_ns_per_expert=100,
            hybrid_upload_ns_per_expert=200,
        )
        self.assertNotIn("--expert-hybrid-gpu-experts", mmap_plan.arguments)
        self.assertIn("pread bounded-lease", mmap_plan.hybrid_selection)
    def test_cpu_runtime_ignores_driver_visible_gpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "build" / "bin" / "llama-server.exe"
            binary.parent.mkdir(parents=True)
            binary.touch()
            with mock.patch("tools.ese.detect_hardware", return_value=hardware(12, 20)):
                detected = _hardware_for_server(binary)
            self.assertEqual(detected.gpus, ())

    def test_runtime_marker_enables_cuda_planning(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            binary = Path(temp) / "llama-server.exe"
            binary.touch()
            binary.with_name("ese-runtime.json").write_text('{"cuda": true}')
            self.assertTrue(_server_supports_cuda(binary))

    def test_packaged_windows_cpu_selection_before_cuda_binary_load(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "build"
            cuda = root / "bin" / "llama-server.exe"
            cpu = root / "cpu" / "bin" / "llama-server.exe"
            for path, enabled in ((cuda, True), (cpu, False)):
                path.parent.mkdir(parents=True)
                path.touch()
                path.with_name("ese-runtime.json").write_text(json.dumps({"cuda": enabled}))
            with mock.patch.dict(os.environ, {}, clear=True):
                for devices in (None, (), ("GPU-visible",)):
                    with self.subTest(devices=devices), mock.patch("tools.ese.cuda_device_uuids", return_value=devices):
                        self.assertEqual(_select_packaged_windows_binary(cuda, root), cuda if devices else cpu.resolve())
                with mock.patch.dict(os.environ, {"ESE_SERVER_BACKEND": "cpu"}), mock.patch("tools.ese.cuda_device_uuids", side_effect=AssertionError("CPU must not initialize CUDA")):
                    self.assertEqual(_select_packaged_windows_binary(cuda, root), cpu.resolve())
                with mock.patch.dict(os.environ, {"ESE_SERVER_BACKEND": "cuda"}), mock.patch("tools.ese.cuda_device_uuids", side_effect=AssertionError("CUDA override remains explicit")):
                    self.assertEqual(_select_packaged_windows_binary(cuda, root), cuda)
                cpu.unlink()
                with mock.patch("tools.ese.cuda_device_uuids", side_effect=AssertionError("No CPU bundle means unchanged behavior")):
                    self.assertEqual(_select_packaged_windows_binary(cuda, root), cuda)

    def test_packaged_windows_cpu_selection_preserves_explicit_binary_overrides(self) -> None:
        from types import SimpleNamespace
        windows_os = SimpleNamespace(name="nt", environ=os.environ)
        with tempfile.TemporaryDirectory() as temp, mock.patch("tools.ese.os", windows_os), mock.patch("tools.ese._select_packaged_windows_binary", side_effect=AssertionError("Explicit binary must win")):
            explicit = Path(temp) / "explicit-server.exe"
            environment = Path(temp) / "environment-server.exe"
            self.assertEqual(_resolve_binary(str(explicit)), explicit.resolve())
            with mock.patch.dict(os.environ, {"ESE_SERVER": str(environment)}):
                self.assertEqual(_resolve_binary(None), environment.resolve())

    def test_packaged_windows_cpu_selection_rejects_cuda_payload_in_cpu_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "build"
            cpu = root / "cpu" / "bin" / "llama-server.exe"
            cpu.parent.mkdir(parents=True)
            cpu.touch()
            cpu.with_name("ese-runtime.json").write_text('{"cuda": true}')
            with self.assertRaisesRegex(ESEError, "CPU-only runtime marker"):
                _select_packaged_windows_binary(root / "bin" / "llama-server.exe", root)

    def test_json_doctor_is_a_runtime_probe_without_build_tools(self) -> None:
        output = StringIO()
        with (
            mock.patch("tools.ese.detect_hardware", return_value=hardware(12)),
            mock.patch("tools.ese.shutil.which", return_value=None),
            mock.patch("tools.ese._windows_msvc_installation", return_value=None),
            redirect_stdout(output),
        ):
            self.assertEqual(_doctor(True), 0)
        report = __import__("json").loads(output.getvalue())
        self.assertFalse(report["build_ready"])
        self.assertEqual(report["gpus"][0]["free_bytes"], 12 * GIB)

    def test_frozen_launcher_uses_executable_directory_as_runtime_root(self) -> None:
        with (
            mock.patch.object(sys, "frozen", True, create=True),
            mock.patch.object(sys, "executable", "/opt/ese/ese.exe"),
        ):
            self.assertEqual(_repo_root(), Path("/opt/ese"))

    def test_execution_environment_prefers_bundled_native_libraries(self) -> None:
        plan = build_launch_plan(
            model=moe_model(),
            hardware=hardware(20),
            binary=Path("/opt/ese/build/bin/llama-server"),
            policy="cache",
        )
        with mock.patch.dict("os.environ", {"LD_LIBRARY_PATH": "/system/libs"}, clear=True):
            environment = _execution_environment(plan)
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/opt/ese/build/bin:/system/libs")

    def test_parse_size(self) -> None:
        self.assertEqual(parse_size("8GiB"), 8 * GIB)
        self.assertEqual(parse_size("1.5G"), 1_500_000_000)
        self.assertEqual(parse_size("512M"), 512_000_000)

    def test_read_minimal_gguf(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            write_minimal_gguf(
                path,
                {
                    "general.architecture": "gpt-oss",
                    "general.name": "GPT-OSS Test",
                    "gpt-oss.block_count": 36,
                    "gpt-oss.expert_count": 128,
                },
            )
            metadata = read_gguf_metadata(path)
            self.assertEqual(metadata["general.architecture"], "gpt-oss")
            self.assertEqual(metadata["gpt-oss.block_count"], 36)

    def test_expert_layer_formats_preserve_component_multiplicity(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            write_tensor_gguf(path)
            self.assertEqual(
                inspect_expert_layer_formats(path),
                (((1, 16, 32, 512),),),
            )

    def test_read_tensor_index_and_expert_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            write_tensor_gguf(path)
            metadata, tensors = read_gguf_index(path)
            self.assertEqual(metadata["general.alignment"], 32)
            self.assertEqual(tensors[0]["dimensions"], (16, 32, 2))
            self.assertEqual(tensors[0]["span_bytes"], 1024)
            expert = inspect_expert_geometry(path)
            self.assertIsNotNone(expert)
            self.assertEqual(expert["ggml_type"], 1)
            self.assertEqual(expert["expert_component_bytes"], 512)
            self.assertEqual(len(inspect_expert_geometries(path)), 1)

    def test_discover_split_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = [root / f"model-{part:05d}-of-00003.gguf" for part in (1, 2, 3)]
            for path in paths:
                path.write_bytes(b"x")
            self.assertEqual(discover_model_shards(paths[0]), tuple(p.resolve() for p in paths))

    def test_stream_policy_when_model_exceeds_ram(self) -> None:
        policy, _ = select_policy(moe_model(70 * GIB), hardware(20, ram_available=48))
        self.assertEqual(policy, "stream")

    def test_auto_uses_bounded_cache_when_moe_exceeds_vram(self) -> None:
        policy, _ = select_policy(moe_model(), hardware(20))
        self.assertEqual(policy, "cache")

    def test_launcher_delegates_limits_to_native_controller(self) -> None:
        plan = build_launch_plan(
            model=moe_model(),
            hardware=hardware(7, 9, ram_available=47),
            binary=Path("/server"),
            policy="auto",
            context=128 * 1024,
            reserve_vram=GIB,
        )
        expected = {
            "--memory-policy": "auto",
            "--max-ram": f"{47 * GIB}B",
            "--reserve-vram": f"{GIB}B",
            "--min-kv-quality": "turbo4",
            "--max-context": str(128 * 1024),
            "--resource-preference": "balanced",
        }
        for flag, value in expected.items():
            position = plan.arguments.index(flag)
            self.assertEqual(plan.arguments[position + 1], value)

    def test_hybrid_policy_uses_supported_cpu_moe_flag(self) -> None:
        plan = build_launch_plan(
            model=moe_model(),
            hardware=hardware(20),
            binary=Path("/server"),
            policy="hybrid",
        )
        self.assertIn("--cpu-moe", plan.arguments)
        self.assertNotIn("--moe-cache", plan.arguments)
        self.assertNotIn("--moe-cache-expert-parallel", plan.arguments)

    def test_hybrid_gpu_tail_maps_to_n_cpu_moe(self) -> None:
        plan = build_launch_plan(
            model=moe_model(),
            hardware=hardware(7, 9),
            binary=Path("/server"),
            policy="hybrid",
            gpu_resident_moe=6,
        )
        position = plan.arguments.index("--n-cpu-moe")
        self.assertEqual(plan.arguments[position + 1], "30")
        self.assertEqual(auto_tensor_split(plan.hardware.gpus), "44,56")

    def test_stream_plan_sets_deferred_expert_environment(self) -> None:
        plan = build_launch_plan(
            model=moe_model(61 * GIB),
            hardware=hardware(7, 9, ram_available=47),
            binary=Path("/server"),
            policy="stream",
            gpu_resident_moe=6,
            prefetch_tail=4,
            expert_storage_backend="mmap",
        )
        self.assertEqual(plan.environment["GGML_CUDA_NO_MODEL_PINNED"], "1")
        self.assertNotIn("GGML_CUDA_NO_PINNED", plan.environment)
        self.assertEqual(plan.environment["LLAMA_EXPERT_PREFETCH"], "1")
        self.assertEqual(plan.environment["LLAMA_EXPERT_PREFETCH_TAIL"], "4")
        self.assertIn("--defer-experts", plan.arguments)
        self.assertIn("--expert-ram-cache-mib", plan.arguments)
        self.assertIn("--expert-vram-cache-mib", plan.arguments)
        self.assertIn("--expert-vram-reserve-mib", plan.arguments)
        backend = plan.arguments.index("--expert-storage-backend")
        self.assertEqual(plan.arguments[backend + 1], "mmap")
        position = plan.arguments.index("--n-cpu-moe")
        self.assertEqual(plan.arguments[position + 1], "30")

    def test_cache_and_stream_share_bounded_native_hierarchy(self) -> None:
        cache = build_launch_plan(
            model=moe_model(),
            hardware=hardware(7, 9, ram_available=47),
            binary=Path("/server"),
            policy="cache",
            expert_ram_cache=768 * 1024**2,
            expert_ram_staging=32 * 1024**2,
            expert_vram_cache=256 * 1024**2,
        )
        self.assertNotIn("--defer-experts", cache.arguments)
        self.assertIn("--cpu-moe", cache.arguments)
        expected = {
            "--expert-ram-cache-mib": "768",
            "--expert-ram-staging-mib": "32",
            "--expert-vram-cache-mib": "256",
            "--expert-vram-reserve-mib": "1024",
            "--expert-storage-backend": "mmap",
            "--expert-cache-min-observations": "2",
        }
        for flag, value in expected.items():
            position = cache.arguments.index(flag)
            self.assertEqual(cache.arguments[position + 1], value)

        stream = build_launch_plan(
            model=moe_model(70 * GIB),
            hardware=hardware(20, ram_available=48),
            binary=Path("/server"),
            policy="stream",
        )
        backend = stream.arguments.index("--expert-storage-backend")
        self.assertEqual(stream.arguments[backend + 1], "pread")
        self.assertNotIn("LLAMA_EXPERT_PREFETCH", stream.environment)

        with self.assertRaises(ESEError):
            build_launch_plan(
                model=moe_model(),
                hardware=hardware(20),
                binary=Path("/server"),
                policy="stream",
                prefetch_tail=1,
            )

    def test_expert_load_workers_preserve_bounded_cache_plan(self) -> None:
        for policy in ("cache", "stream"):
            for workers in (0, 1, 4, 8):
                with self.subTest(policy=policy, workers=workers):
                    plan = build_launch_plan(
                        model=moe_model(), hardware=hardware(20),
                        binary=Path("/server"), policy=policy,
                        expert_load_workers=workers,
                        expert_ram_cache=768 * 1024**2,
                        expert_ram_staging=32 * 1024**2,
                    )
                    for flag, value in (("--expert-ram-cache-mib", "768"),
                                        ("--expert-ram-staging-mib", "32")):
                        self.assertEqual(plan.arguments[plan.arguments.index(flag) + 1], value)
                    if workers:
                        self.assertEqual(plan.arguments.count("--expert-load-workers"), 1)
                        self.assertEqual(
                            plan.arguments[plan.arguments.index("--expert-load-workers") + 1],
                            str(workers),
                        )
                    else:
                        self.assertNotIn("--expert-load-workers", plan.arguments)
                    self.assertEqual("--defer-experts" in plan.arguments, policy == "stream")

    def test_expert_load_workers_reject_unavailable_policies_and_counts(self) -> None:
        for policy, workers in (("resident", 1), ("hybrid", 4), ("cache", -1), ("stream", 9)):
            with self.subTest(policy=policy, workers=workers), self.assertRaisesRegex(
                ESEError, "--expert-load-workers"
            ):
                build_launch_plan(
                    model=moe_model(), hardware=hardware(20),
                    binary=Path("/server"), policy=policy, expert_load_workers=workers,
                )

    def test_expert_load_worker_cli_option_reaches_native_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.gguf"
            write_minimal_gguf(path, {
                "general.architecture": "gpt-oss", "gpt-oss.expert_count": 128,
                "gpt-oss.expert_used_count": 4, "gpt-oss.block_count": 36,
            })
            args = _make_parser().parse_args([
                "plan", str(path), "--policy", "stream", "--expert-load-workers", "4",
                "--no-auto-hybrid",
            ])
            args.extra = ()
            with mock.patch("tools.ese._hardware_for_server", return_value=hardware(20)), \
                    mock.patch("tools.ese._resolve_binary", return_value=Path("/server")):
                plan = _plan_from_args(args)
            self.assertEqual(plan.arguments[plan.arguments.index("--expert-load-workers") + 1], "4")

    def test_parallel_slots_are_enabled_for_resident_models(self) -> None:
        plan = build_launch_plan(
            model=dense_model(),
            hardware=hardware(7, 9, ram_available=47),
            binary=Path("/server"),
            policy="resident",
            context=8192,
            slots=2,
        )

        parallel = plan.arguments.index("-np")
        self.assertEqual(plan.arguments[parallel + 1], "2")
        self.assertNotIn("--expert-vram-cache-mib", plan.arguments)
        self.assertNotIn("--expert-prefill-staging-mib", plan.arguments)
        self.assertNotIn("-no-mmad", plan.arguments)
        self.assertNotIn("-nkvo", plan.arguments)
        self.assertEqual(plan.as_dict()["concurrency"], {
            "slots": 2,
            "total_context": 8192,
            "context_per_slot": 4096,
            "adaptive_expert_cache": False,
            "kqv_offload": True,
        })

    def test_parallel_slots_reject_expert_offload_until_parity_is_proven(self) -> None:
        for policy in ("hybrid", "cache", "stream"):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ESEError, "concurrent sessions currently require a dense resident model"
            ):
                build_launch_plan(
                    model=moe_model(),
                    hardware=hardware(7, 9),
                    binary=Path("/server"),
                    policy=policy,
                    context=8192,
                    slots=2,
                )

        with self.assertRaisesRegex(
            ESEError, "concurrent sessions currently require a dense resident model"
        ):
            build_launch_plan(
                model=moe_model(),
                hardware=hardware(7, 9),
                binary=Path("/server"),
                policy="resident",
                context=8192,
                slots=2,
            )

    def test_parallel_slots_require_even_total_context(self) -> None:
        with self.assertRaisesRegex(ESEError, "divide evenly"):
            build_launch_plan(
                model=moe_model(),
                hardware=hardware(20),
                binary=Path("/server"),
                policy="cache",
                context=4097,
                slots=2,
            )

    def test_prefill_staging_is_auto_by_default_and_explicitly_controllable(self) -> None:
        automatic = build_launch_plan(
            model=moe_model(),
            hardware=hardware(20),
            binary=Path("/server"),
            policy="cache",
        )
        self.assertNotIn("--expert-prefill-staging-mib", automatic.arguments)

        explicit = build_launch_plan(
            model=moe_model(),
            hardware=hardware(20),
            binary=Path("/server"),
            policy="stream",
            expert_prefill_staging=513 * 1024**2,
        )
        position = explicit.arguments.index("--expert-prefill-staging-mib")
        self.assertEqual(explicit.arguments[position + 1], "513")

        with self.assertRaisesRegex(ESEError, "cache or stream"):
            build_launch_plan(
                model=moe_model(),
                hardware=hardware(20),
                binary=Path("/server"),
                policy="resident",
                expert_prefill_staging=64 * 1024**2,
            )

    def test_dense_oversized_resident_plan_uses_native_fit(self) -> None:
        model = ModelInfo(
            requested_path=Path("/dense.gguf"),
            shards=(Path("/dense.gguf"),),
            total_bytes=30 * GIB,
            metadata={"general.architecture": "llama", "llama.block_count": 40},
        )
        plan = build_launch_plan(
            model=model,
            hardware=hardware(20),
            binary=Path("/server"),
            policy="resident",
        )
        self.assertIn("--fit", plan.arguments)

    def test_fixed_resident_gpu_layers_disable_native_fit(self) -> None:
        plan = build_launch_plan(
            model=dense_model(40 * GIB),
            hardware=hardware(20),
            binary=Path("/server"),
            policy="resident",
            gpu_layers=18,
        )
        self.assertNotIn("--fit", plan.arguments)
        self.assertNotIn("--memory-policy", plan.arguments)
        self.assertNotIn("--max-ram", plan.arguments)
        self.assertNotIn("--resource-preference", plan.arguments)
        index = plan.arguments.index("-ngl")
        self.assertEqual(plan.arguments[index + 1], "18")


if __name__ == "__main__":
    unittest.main()
