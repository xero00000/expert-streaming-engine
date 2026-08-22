from __future__ import annotations

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
    parse_size,
    read_gguf_metadata,
    read_gguf_index,
    select_policy,
    _execution_environment,
    _repo_root,
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


class LauncherTests(unittest.TestCase):
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
