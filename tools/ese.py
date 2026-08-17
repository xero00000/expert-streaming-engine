#!/usr/bin/env python3
"""Unified launcher and memory planner for Expert Streaming Engine.

The launcher is intentionally standard-library only. It does not replace the
native runtime; it selects a conservative, inspectable set of native flags for
one of three policies:

* resident: the model is expected to fit in aggregate VRAM;
* cache: experts stay in host RAM while hot rows/tensors use spare VRAM;
* stream: expert storage is deferred and paged from the model file.

Every generated command can be inspected with ``ese plan`` or ``--dry-run``.
Unknown native options can be appended after ``--``.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import glob
import json
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

GIB = 1024 ** 3
MIB = 1024 ** 2
DEFAULT_SERVER = Path("build/bin/llama-server")
KNOWN_KV_TYPES = ("auto", "f16", "q8_0", "q4_0")


class ESEError(RuntimeError):
    """Actionable launcher error."""


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int

    @property
    def free_fraction(self) -> float:
        return self.free_bytes / self.total_bytes if self.total_bytes else 0.0


@dataclasses.dataclass(frozen=True)
class HostMemory:
    total_bytes: int
    available_bytes: int


@dataclasses.dataclass(frozen=True)
class ModelInfo:
    requested_path: Path
    shards: tuple[Path, ...]
    total_bytes: int
    metadata: dict[str, Any]

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", "")).lower()

    @property
    def name(self) -> str:
        return str(
            self.metadata.get("general.name")
            or self.metadata.get("general.basename")
            or self.requested_path.stem
        )

    @property
    def block_count(self) -> int | None:
        candidates = (
            f"{self.architecture}.block_count",
            "llama.block_count",
            "general.block_count",
        )
        return _first_positive_int(self.metadata, candidates)

    @property
    def expert_count(self) -> int | None:
        candidates = (
            f"{self.architecture}.expert_count",
            f"{self.architecture}.expert_used_count",
            "llama.expert_count",
            "general.expert_count",
        )
        return _first_positive_int(self.metadata, candidates)

    @property
    def is_moe(self) -> bool:
        if (self.expert_count or 0) > 1:
            return True
        text = f"{self.architecture} {self.name}".lower()
        return any(token in text for token in ("moe", "mixtral", "gpt-oss", "deepseek"))


@dataclasses.dataclass(frozen=True)
class HardwareInfo:
    host: HostMemory
    gpus: tuple[GPUInfo, ...]
    logical_cpus: int

    @property
    def total_vram(self) -> int:
        return sum(g.total_bytes for g in self.gpus)

    @property
    def free_vram(self) -> int:
        return sum(g.free_bytes for g in self.gpus)


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    policy: str
    reason: str
    model: ModelInfo
    hardware: HardwareInfo
    binary: Path
    environment: dict[str, str]
    arguments: tuple[str, ...]

    def command(self) -> tuple[str, ...]:
        return (str(self.binary), *self.arguments)

    def shell_command(self) -> str:
        env = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in sorted(self.environment.items())
        )
        cmd = shlex.join(self.command())
        return f"{env} {cmd}".strip()

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "reason": self.reason,
            "model": {
                "path": str(self.model.requested_path),
                "shards": [str(p) for p in self.model.shards],
                "total_bytes": self.model.total_bytes,
                "total_human": human_bytes(self.model.total_bytes),
                "name": self.model.name,
                "architecture": self.model.architecture or None,
                "block_count": self.model.block_count,
                "expert_count": self.model.expert_count,
                "is_moe": self.model.is_moe,
            },
            "hardware": {
                "ram_total_bytes": self.hardware.host.total_bytes,
                "ram_available_bytes": self.hardware.host.available_bytes,
                "ram_available_human": human_bytes(self.hardware.host.available_bytes),
                "logical_cpus": self.hardware.logical_cpus,
                "gpus": [dataclasses.asdict(g) for g in self.hardware.gpus],
                "vram_total_bytes": self.hardware.total_vram,
                "vram_free_bytes": self.hardware.free_vram,
                "vram_free_human": human_bytes(self.hardware.free_vram),
            },
            "environment": dict(self.environment),
            "arguments": list(self.arguments),
            "command": list(self.command()),
            "shell_command": self.shell_command(),
        }


def _first_positive_int(metadata: dict[str, Any], keys: Iterable[str]) -> int | None:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def human_bytes(value: int) -> str:
    number = float(max(0, value))
    for suffix in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if number < 1024.0 or suffix == "PiB":
            if suffix == "B":
                return f"{int(number)} {suffix}"
            return f"{number:.2f} {suffix}"
        number /= 1024.0
    raise AssertionError("unreachable")


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp](?:i?b)?|b)?\s*$", re.IGNORECASE)


def parse_size(value: str) -> int:
    match = _SIZE_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}; examples: 1024M, 8GiB, 1.5T"
        )
    number = float(match.group(1))
    suffix = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "kib": 1024,
        "m": 1000 ** 2,
        "mb": 1000 ** 2,
        "mib": 1024 ** 2,
        "g": 1000 ** 3,
        "gb": 1000 ** 3,
        "gib": 1024 ** 3,
        "t": 1000 ** 4,
        "tb": 1000 ** 4,
        "tib": 1024 ** 4,
        "p": 1000 ** 5,
        "pb": 1000 ** 5,
        "pib": 1024 ** 5,
    }
    return int(number * multipliers[suffix])


def _run_capture(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def detect_nvidia_gpus() -> tuple[GPUInfo, ...]:
    output = _run_capture(
        (
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        )
    )
    if not output:
        return ()
    result: list[GPUInfo] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            result.append(
                GPUInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    total_bytes=int(float(parts[2]) * MIB),
                    free_bytes=int(float(parts[3]) * MIB),
                )
            )
        except ValueError:
            continue
    return tuple(sorted(result, key=lambda gpu: gpu.index))


def detect_host_memory() -> HostMemory:
    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                if ":" not in line:
                    continue
                key, raw = line.split(":", 1)
                amount = raw.strip().split()[0]
                values[key] = int(amount) * 1024
        except (OSError, ValueError, IndexError):
            pass
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", values.get("MemFree", 0))
        if total:
            return HostMemory(total_bytes=total, available_bytes=available)

    if sys.platform == "darwin":
        total_raw = _run_capture(("sysctl", "-n", "hw.memsize"))
        if total_raw and total_raw.isdigit():
            total = int(total_raw)
            return HostMemory(total_bytes=total, available_bytes=total)

    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return HostMemory(
                total_bytes=int(status.ullTotalPhys),
                available_bytes=int(status.ullAvailPhys),
            )

    return HostMemory(total_bytes=0, available_bytes=0)


def detect_hardware() -> HardwareInfo:
    return HardwareInfo(
        host=detect_host_memory(),
        gpus=detect_nvidia_gpus(),
        logical_cpus=max(1, os.cpu_count() or 1),
    )


_SPLIT_RE = re.compile(
    r"^(?P<prefix>.+)-(?P<part>\d{5})-of-(?P<count>\d{5})\.gguf$",
    re.IGNORECASE,
)


def discover_model_shards(path: Path) -> tuple[Path, ...]:
    requested = path.expanduser().resolve()
    if not requested.is_file():
        raise ESEError(f"model file does not exist: {requested}")

    match = _SPLIT_RE.match(requested.name)
    if not match:
        return (requested,)

    expected = int(match.group("count"))
    pattern = str(requested.with_name(f"{match.group('prefix')}-?????-of-{expected:05d}.gguf"))
    candidates = sorted(Path(item).resolve() for item in glob.glob(pattern))
    if len(candidates) != expected:
        raise ESEError(
            f"split GGUF is incomplete: expected {expected} shards, found {len(candidates)}"
        )
    return tuple(candidates)


_GGUF_U8 = 0
_GGUF_I8 = 1
_GGUF_U16 = 2
_GGUF_I16 = 3
_GGUF_U32 = 4
_GGUF_I32 = 5
_GGUF_F32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_U64 = 10
_GGUF_I64 = 11
_GGUF_F64 = 12
_GGUF_SCALARS: dict[int, str] = {
    _GGUF_U8: "<B",
    _GGUF_I8: "<b",
    _GGUF_U16: "<H",
    _GGUF_I16: "<h",
    _GGUF_U32: "<I",
    _GGUF_I32: "<i",
    _GGUF_F32: "<f",
    _GGUF_BOOL: "<?",
    _GGUF_U64: "<Q",
    _GGUF_I64: "<q",
    _GGUF_F64: "<d",
}


def _read_exact(handle: BinaryIO, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ESEError("truncated GGUF metadata")
    return data


def _read_u32(handle: BinaryIO) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_u64(handle: BinaryIO) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_gguf_string(handle: BinaryIO) -> str:
    length = _read_u64(handle)
    if length > 64 * MIB:
        raise ESEError(f"unreasonable GGUF string length: {length}")
    return _read_exact(handle, length).decode("utf-8", errors="replace")


def _read_gguf_value(
    handle: BinaryIO,
    value_type: int,
    *,
    materialize_arrays: bool = False,
    depth: int = 0,
) -> Any:
    if depth > 4:
        raise ESEError("GGUF metadata nesting is too deep")
    scalar_format = _GGUF_SCALARS.get(value_type)
    if scalar_format:
        return struct.unpack(scalar_format, _read_exact(handle, struct.calcsize(scalar_format)))[0]
    if value_type == _GGUF_STRING:
        return _read_gguf_string(handle)
    if value_type == _GGUF_ARRAY:
        element_type = _read_u32(handle)
        length = _read_u64(handle)
        if length > 100_000_000:
            raise ESEError(f"unreasonable GGUF array length: {length}")
        if materialize_arrays and length <= 4096:
            return [
                _read_gguf_value(
                    handle,
                    element_type,
                    materialize_arrays=materialize_arrays,
                    depth=depth + 1,
                )
                for _ in range(length)
            ]
        for _ in range(length):
            _read_gguf_value(
                handle,
                element_type,
                materialize_arrays=False,
                depth=depth + 1,
            )
        return {"type": element_type, "length": length}
    raise ESEError(f"unsupported GGUF metadata value type: {value_type}")


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read scalar/string metadata without loading tensor data or large arrays."""
    with path.open("rb") as handle:
        magic = _read_exact(handle, 4)
        if magic != b"GGUF":
            raise ESEError(f"not a GGUF file: {path}")
        version = _read_u32(handle)
        if version not in (2, 3):
            raise ESEError(f"unsupported GGUF version {version}; expected version 2 or 3")
        _tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)
        if metadata_count > 1_000_000:
            raise ESEError(f"unreasonable GGUF metadata count: {metadata_count}")
        metadata: dict[str, Any] = {"_gguf.version": version}
        for _ in range(metadata_count):
            key = _read_gguf_string(handle)
            value_type = _read_u32(handle)
            metadata[key] = _read_gguf_value(handle, value_type)
        return metadata


def inspect_model(path: Path) -> ModelInfo:
    shards = discover_model_shards(path)
    metadata: dict[str, Any]
    try:
        metadata = read_gguf_metadata(shards[0])
    except ESEError as exc:
        metadata = {"_ese.metadata_warning": str(exc)}
    return ModelInfo(
        requested_path=shards[0],
        shards=shards,
        total_bytes=sum(item.stat().st_size for item in shards),
        metadata=metadata,
    )


def _model_text(model: ModelInfo) -> str:
    return f"{model.architecture} {model.name} {model.requested_path.name}".lower()


def select_policy(
    model: ModelInfo,
    hardware: HardwareInfo,
    requested: str = "auto",
) -> tuple[str, str]:
    if requested != "auto":
        return requested, f"selected explicitly with --policy {requested}"

    free_vram = hardware.free_vram
    available_ram = hardware.host.available_bytes
    text = _model_text(model)

    if model.is_moe and available_ram and model.total_bytes > int(available_ram * 0.90):
        return (
            "stream",
            f"MoE size {human_bytes(model.total_bytes)} exceeds 90% of available RAM "
            f"({human_bytes(available_ram)}); use deferred disk-backed experts",
        )

    if "gpt-oss" in text and model.total_bytes > max(free_vram, int(available_ram * 0.70)):
        return (
            "stream",
            "GPT-OSS model exceeds the safe resident memory budget; use ESE deferred experts",
        )

    if model.is_moe and (not free_vram or model.total_bytes > int(free_vram * 0.85)):
        return (
            "cache",
            "MoE weights do not fit safely in free VRAM but can use host-backed adaptive expert caching",
        )

    if free_vram and model.total_bytes <= int(free_vram * 0.85):
        return (
            "resident",
            f"model fits within 85% of detected free VRAM ({human_bytes(free_vram)})",
        )

    if model.is_moe:
        return (
            "cache",
            "GPU capacity was not detected; host-backed MoE cache is the conservative default",
        )

    return (
        "resident",
        "dense/non-MoE model and no stronger memory-pressure signal was detected",
    )


def auto_tensor_split(gpus: Sequence[GPUInfo]) -> str | None:
    if len(gpus) < 2:
        return None
    weights = [max(1, gpu.free_bytes) for gpu in gpus]
    total = sum(weights)
    percentages = [round(value * 100 / total) for value in weights]
    difference = 100 - sum(percentages)
    percentages[-1] += difference
    return ",".join(str(max(1, value)) for value in percentages)


def choose_kv_type(
    requested: str,
    context: int,
    hardware: HardwareInfo,
    reserve_vram: int,
) -> str:
    if requested != "auto":
        return requested
    usable_vram = max(0, hardware.free_vram - reserve_vram)
    if context > 131_072 or (hardware.gpus and usable_vram < 4 * GIB):
        return "q4_0"
    return "q8_0"


def _default_threads(logical_cpus: int) -> tuple[int, int]:
    decode = max(1, min(8, logical_cpus // 2 if logical_cpus > 2 else logical_cpus))
    batch = max(decode, min(32, logical_cpus))
    return decode, batch


def _resident_moe_layers(hardware: HardwareInfo, requested: int | None) -> int:
    if requested is not None:
        return max(0, requested)
    return max(0, min(8, int(hardware.free_vram / (1500 * MIB))))


def build_launch_plan(
    *,
    model: ModelInfo,
    hardware: HardwareInfo,
    binary: Path,
    policy: str = "auto",
    context: int = 65_536,
    slots: int = 1,
    host: str = "127.0.0.1",
    port: int = 8080,
    threads: int | None = None,
    batch_threads: int | None = None,
    batch_size: int | None = None,
    ubatch_size: int | None = None,
    kv_type: str = "auto",
    reserve_vram: int = GIB,
    resident_moe_layers: int | None = None,
    tensor_split: str | None = None,
    prefetch_tail: int = 0,
    extra_args: Sequence[str] = (),
) -> LaunchPlan:
    selected_policy, reason = select_policy(model, hardware, policy)
    detected_threads, detected_batch_threads = _default_threads(hardware.logical_cpus)
    threads = threads or detected_threads
    batch_threads = batch_threads or detected_batch_threads
    kv = choose_kv_type(kv_type, context, hardware, reserve_vram)
    split = tensor_split or auto_tensor_split(hardware.gpus)

    args: list[str] = [
        "-m",
        str(model.requested_path),
        "-c",
        str(context),
        "-np",
        str(slots),
        "--host",
        host,
        "--port",
        str(port),
        "-t",
        str(threads),
        "-tb",
        str(batch_threads),
        "-fa",
        "on",
        "-ctk",
        kv,
        "-ctv",
        kv,
    ]
    env: dict[str, str] = {}

    if split:
        args.extend(("--tensor-split", split))

    if selected_policy == "resident":
        args.extend(("-ngl", "99"))
        if batch_size is not None:
            args.extend(("-b", str(batch_size)))
        if ubatch_size is not None:
            args.extend(("-ub", str(ubatch_size)))

    elif selected_policy == "cache":
        args.extend(("-ngl", "99", "-sm", "layer", "-ot", "exps=CPU"))
        if len(hardware.gpus) >= 2:
            args.extend(("--moe-cache", "auto", "--moe-cache-expert-parallel", "auto"))
        else:
            args.extend(("--moe-cache", "on"))
        args.extend(("-b", str(batch_size or 1024), "-ub", str(ubatch_size or 512)))

    elif selected_policy == "stream":
        env.update(
            {
                "GGML_CUDA_NO_PINNED": "1",
                "LLAMA_EXPERT_PREFETCH": "1",
            }
        )
        if prefetch_tail > 0:
            env["LLAMA_EXPERT_PREFETCH_TAIL"] = str(min(prefetch_tail, 32))
        args.extend(("-ngl", "99", "--defer-experts"))
        text = _model_text(model)
        blocks = model.block_count
        keep_gpu = _resident_moe_layers(hardware, resident_moe_layers)
        if "gpt-oss" in text and blocks and keep_gpu < blocks:
            args.extend(("--n-cpu-moe", str(max(1, blocks - keep_gpu))))
        else:
            args.append("--cpu-moe")
        args.extend(("-b", str(batch_size or 1024), "-ub", str(ubatch_size or 512)))

    else:
        raise ESEError(f"unknown policy: {selected_policy}")

    cleaned_extra = list(extra_args)
    if cleaned_extra and cleaned_extra[0] == "--":
        cleaned_extra.pop(0)
    args.extend(cleaned_extra)

    return LaunchPlan(
        policy=selected_policy,
        reason=reason,
        model=model,
        hardware=hardware,
        binary=binary,
        environment=env,
        arguments=tuple(args),
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_binary(value: str | None, build_dir: Path | None = None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    env_binary = os.environ.get("ESE_SERVER")
    if env_binary:
        return Path(env_binary).expanduser().resolve()
    root = _repo_root()
    if build_dir:
        return (build_dir / "bin" / "llama-server").resolve()
    return (root / DEFAULT_SERVER).resolve()


def _print_plan(plan: LaunchPlan, as_json: bool) -> None:
    if as_json:
        print(json.dumps(plan.as_dict(), indent=2, sort_keys=True))
        return
    print(f"Policy : {plan.policy}")
    print(f"Reason : {plan.reason}")
    print(
        f"Model  : {plan.model.name} | {human_bytes(plan.model.total_bytes)} | "
        f"{len(plan.model.shards)} shard(s)"
    )
    architecture = plan.model.architecture or "unknown"
    print(
        f"GGUF   : arch={architecture}, blocks={plan.model.block_count or 'unknown'}, "
        f"experts={plan.model.expert_count or 'unknown'}"
    )
    print(
        f"Host   : RAM available {human_bytes(plan.hardware.host.available_bytes)}; "
        f"VRAM free {human_bytes(plan.hardware.free_vram)} across "
        f"{len(plan.hardware.gpus)} NVIDIA GPU(s)"
    )
    print("Command:")
    print(f"  {plan.shell_command()}")


def _doctor(as_json: bool) -> int:
    hardware = detect_hardware()
    root = _repo_root()
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    server = _resolve_binary(None)
    checks = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "repo_root": str(root),
        "cmake": cmake,
        "ninja": ninja,
        "compiler": compiler,
        "server": str(server),
        "server_exists": server.is_file(),
        "ram_total": hardware.host.total_bytes,
        "ram_available": hardware.host.available_bytes,
        "gpus": [dataclasses.asdict(gpu) for gpu in hardware.gpus],
    }
    if as_json:
        print(json.dumps(checks, indent=2, sort_keys=True))
    else:
        print("Expert Streaming Engine doctor")
        print(f"  Platform : {checks['platform']}")
        print(f"  Python   : {checks['python']}")
        print(f"  CMake    : {cmake or 'missing'}")
        print(f"  Ninja    : {ninja or 'optional/not found'}")
        print(f"  Compiler : {compiler or 'missing'}")
        print(f"  Server   : {server} ({'ready' if server.is_file() else 'not built'})")
        print(
            f"  RAM      : {human_bytes(hardware.host.available_bytes)} available / "
            f"{human_bytes(hardware.host.total_bytes)} total"
        )
        if hardware.gpus:
            for gpu in hardware.gpus:
                print(
                    f"  GPU {gpu.index:<2}: {gpu.name} — "
                    f"{human_bytes(gpu.free_bytes)} free / {human_bytes(gpu.total_bytes)}"
                )
        else:
            print("  NVIDIA GPU: not detected")
    return 0 if cmake and compiler else 2


def _build(args: argparse.Namespace) -> int:
    root = _repo_root()
    build_dir = Path(args.build_dir).expanduser().resolve()
    if args.clean and build_dir.exists():
        shutil.rmtree(build_dir)

    backend = args.backend
    if backend == "auto":
        backend = "cuda" if shutil.which("nvidia-smi") or shutil.which("nvcc") else "cpu"

    configure = [
        "cmake",
        "-S",
        str(root),
        "-B",
        str(build_dir),
        f"-DCMAKE_BUILD_TYPE={args.build_type}",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_SERVER=ON",
    ]
    if backend == "cuda":
        configure.append("-DGGML_CUDA=ON")
    elif backend == "cpu":
        configure.append("-DGGML_CUDA=OFF")
    else:
        raise ESEError(f"unsupported build backend: {backend}")

    if shutil.which("ninja"):
        configure.extend(("-G", "Ninja"))

    print("+", shlex.join(configure))
    subprocess.run(configure, cwd=root, check=True)
    build = [
        "cmake",
        "--build",
        str(build_dir),
        "--config",
        args.build_type,
        "--target",
        "llama-server",
        "-j",
        str(args.jobs),
    ]
    print("+", shlex.join(build))
    subprocess.run(build, cwd=root, check=True)
    binary = _resolve_binary(None, build_dir)
    print(f"Built: {binary}")
    return 0


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", type=Path, help="GGUF file or first shard")
    parser.add_argument(
        "--policy",
        choices=("auto", "resident", "cache", "stream"),
        default="auto",
        help="memory policy; auto is recommended",
    )
    parser.add_argument("--binary", help="path to llama-server (or set ESE_SERVER)")
    parser.add_argument("-c", "--context", type=int, default=65_536)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--batch-threads", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--ubatch-size", type=int)
    parser.add_argument("--kv", choices=KNOWN_KV_TYPES, default="auto")
    parser.add_argument(
        "--reserve-vram",
        type=parse_size,
        default=GIB,
        metavar="SIZE",
        help="VRAM safety reserve used by the planner (default: 1GiB)",
    )
    parser.add_argument(
        "--gpu-resident-moe",
        type=int,
        help="stream policy: requested number of MoE layers kept on GPU",
    )
    parser.add_argument(
        "--tensor-split",
        help="manual split such as 46,54; default is proportional to free VRAM",
    )
    parser.add_argument(
        "--prefetch-tail",
        type=int,
        default=0,
        help="stream policy: prefetch N router-logit tail experts (0-32)",
    )
    parser.add_argument("--json", action="store_true")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ese",
        description="One front door for resident, cached, and disk-streamed ESE inference.",
    )
    parser.add_argument("--version", action="version", version="ese unified launcher 0.1")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect build tools, RAM, and GPUs")
    doctor.add_argument("--json", action="store_true")

    build = commands.add_parser("build", help="configure and build llama-server")
    build.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    build.add_argument("--build-dir", default=str(_repo_root() / "build"))
    build.add_argument("--build-type", default="Release")
    build.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    build.add_argument("--clean", action="store_true")

    plan = commands.add_parser("plan", help="inspect the selected policy and native command")
    _add_plan_arguments(plan)

    serve = commands.add_parser("serve", help="plan and execute llama-server")
    _add_plan_arguments(serve)
    serve.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan without executing the server",
    )
    return parser


def _plan_from_args(args: argparse.Namespace) -> LaunchPlan:
    if args.context <= 0:
        raise ESEError("--context must be positive")
    if args.slots <= 0:
        raise ESEError("--slots must be positive")
    if not 1 <= args.port <= 65535:
        raise ESEError("--port must be between 1 and 65535")
    model = inspect_model(args.model)
    hardware = detect_hardware()
    binary = _resolve_binary(args.binary)
    return build_launch_plan(
        model=model,
        hardware=hardware,
        binary=binary,
        policy=args.policy,
        context=args.context,
        slots=args.slots,
        host=args.host,
        port=args.port,
        threads=args.threads,
        batch_threads=args.batch_threads,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        kv_type=args.kv,
        reserve_vram=args.reserve_vram,
        resident_moe_layers=args.gpu_resident_moe,
        tensor_split=args.tensor_split,
        prefetch_tail=args.prefetch_tail,
        extra_args=args.extra,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    native_args: list[str] = []
    if "--" in raw:
        separator = raw.index("--")
        native_args = raw[separator + 1 :]
        raw = raw[:separator]
    args = parser.parse_args(raw)
    if args.command in ("plan", "serve"):
        args.extra = native_args
    elif native_args:
        parser.error("native arguments after -- are supported only by plan and serve")
    try:
        if args.command == "doctor":
            return _doctor(args.json)
        if args.command == "build":
            return _build(args)

        plan = _plan_from_args(args)
        _print_plan(plan, args.json)
        if args.command == "plan" or args.dry_run:
            return 0

        if not plan.binary.is_file():
            raise ESEError(
                f"server binary not found: {plan.binary}; run './ese build' "
                "or pass --binary"
            )
        environment = os.environ.copy()
        environment.update(plan.environment)
        os.execvpe(str(plan.binary), list(plan.command()), environment)
        return 0
    except ESEError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
