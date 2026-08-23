from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
import unittest
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
                mock.patch("tools.ese.detect_hardware", return_value=hardware(20)),
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
        self.assertEqual(plan.environment["GGML_CUDA_NO_PINNED"], "1")
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


if __name__ == "__main__":
    unittest.main()
