from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from tools.ese import (
    GIB,
    GPUInfo,
    HardwareInfo,
    HostMemory,
    ModelInfo,
    auto_tensor_split,
    build_launch_plan,
    discover_model_shards,
    parse_size,
    read_gguf_metadata,
    select_policy,
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
        )
        self.assertEqual(plan.environment["GGML_CUDA_NO_PINNED"], "1")
        self.assertEqual(plan.environment["LLAMA_EXPERT_PREFETCH"], "1")
        self.assertEqual(plan.environment["LLAMA_EXPERT_PREFETCH_TAIL"], "4")
        self.assertIn("--defer-experts", plan.arguments)
        position = plan.arguments.index("--n-cpu-moe")
        self.assertEqual(plan.arguments[position + 1], "30")

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
