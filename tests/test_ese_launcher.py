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
    body = b"GGUF" + struct.pack("<IQQ", 3, 0, len(values)) + b"".join(values)
    path.write_bytes(body)


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
            paths = [
                root / "model-00001-of-00003.gguf",
                root / "model-00002-of-00003.gguf",
                root / "model-00003-of-00003.gguf",
            ]
            for path in paths:
                path.write_bytes(b"x")
            self.assertEqual(discover_model_shards(paths[0]), tuple(p.resolve() for p in paths))

    def test_stream_policy_when_model_exceeds_ram(self) -> None:
        model = ModelInfo(
            requested_path=Path("/model.gguf"),
            shards=(Path("/model.gguf"),),
            total_bytes=70 * GIB,
            metadata={
                "general.architecture": "gpt-oss",
                "gpt-oss.expert_count": 128,
                "gpt-oss.block_count": 36,
            },
        )
        hardware = HardwareInfo(
            host=HostMemory(total_bytes=64 * GIB, available_bytes=48 * GIB),
            gpus=(GPUInfo(0, "GPU", 24 * GIB, 20 * GIB),),
            logical_cpus=32,
        )
        policy, _ = select_policy(model, hardware)
        self.assertEqual(policy, "stream")

    def test_single_gpu_cache_uses_on(self) -> None:
        model = ModelInfo(
            requested_path=Path("/model.gguf"),
            shards=(Path("/model.gguf"),),
            total_bytes=40 * GIB,
            metadata={"general.architecture": "deepseek", "deepseek.expert_count": 64},
        )
        hardware = HardwareInfo(
            host=HostMemory(128 * GIB, 100 * GIB),
            gpus=(GPUInfo(0, "GPU", 24 * GIB, 20 * GIB),),
            logical_cpus=32,
        )
        plan = build_launch_plan(
            model=model,
            hardware=hardware,
            binary=Path("/server"),
            policy="cache",
        )
        self.assertIn("--moe-cache", plan.arguments)
        position = plan.arguments.index("--moe-cache")
        self.assertEqual(plan.arguments[position + 1], "on")
        self.assertNotIn("--moe-cache-expert-parallel", plan.arguments)

    def test_multi_gpu_cache_uses_expert_parallel(self) -> None:
        model = ModelInfo(
            requested_path=Path("/model.gguf"),
            shards=(Path("/model.gguf"),),
            total_bytes=40 * GIB,
            metadata={"general.architecture": "deepseek", "deepseek.expert_count": 64},
        )
        hardware = HardwareInfo(
            host=HostMemory(128 * GIB, 100 * GIB),
            gpus=(
                GPUInfo(0, "A", 24 * GIB, 20 * GIB),
                GPUInfo(1, "B", 24 * GIB, 10 * GIB),
            ),
            logical_cpus=32,
        )
        plan = build_launch_plan(
            model=model,
            hardware=hardware,
            binary=Path("/server"),
            policy="cache",
        )
        self.assertIn("--moe-cache-expert-parallel", plan.arguments)
        self.assertEqual(auto_tensor_split(hardware.gpus), "67,33")

    def test_stream_plan_sets_bounded_host_environment(self) -> None:
        model = ModelInfo(
            requested_path=Path("/model.gguf"),
            shards=(Path("/model.gguf"),),
            total_bytes=61 * GIB,
            metadata={
                "general.architecture": "gpt-oss",
                "gpt-oss.expert_count": 128,
                "gpt-oss.block_count": 36,
            },
        )
        hardware = HardwareInfo(
            host=HostMemory(64 * GIB, 47 * GIB),
            gpus=(
                GPUInfo(0, "A", 8 * GIB, 7 * GIB),
                GPUInfo(1, "B", 10 * GIB, 9 * GIB),
            ),
            logical_cpus=32,
        )
        plan = build_launch_plan(
            model=model,
            hardware=hardware,
            binary=Path("/server"),
            policy="stream",
            resident_moe_layers=6,
        )
        self.assertEqual(plan.environment["GGML_CUDA_NO_PINNED"], "1")
        self.assertEqual(plan.environment["LLAMA_EXPERT_PREFETCH"], "1")
        self.assertIn("--defer-experts", plan.arguments)
        position = plan.arguments.index("--n-cpu-moe")
        self.assertEqual(plan.arguments[position + 1], "30")


if __name__ == "__main__":
    unittest.main()
