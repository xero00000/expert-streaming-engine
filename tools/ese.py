#!/usr/bin/env python3
"""Unified launcher and memory planner for Expert Streaming Engine.

The launcher is intentionally standard-library only. It chooses one of the
native execution paths that this branch actually exposes:

* resident: ordinary GPU offload / native auto-fit;
* hybrid: dense tensors on GPU with routed experts on CPU, optionally keeping
  a user-selected MoE tail on GPU;
* cache: bounded RAM expert leases feeding the adaptive per-device VRAM cache;
* stream: the same hierarchy with deferred source residency and explicit
  storage I/O.

``ese plan`` always prints the exact environment and native command. Native
llama-server options can be appended after ``--``.
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

from tools.hardware_profile import (
    HardwareProfileError,
    build_hardware_profile,
    collect_hardware_identity,
    default_profile_path,
    load_hardware_profile,
    save_hardware_profile,
    stale_profile_reasons,
)

GIB = 1024**3
MIB = 1024**2
DEFAULT_SERVER = Path("build/bin") / ("llama-server.exe" if os.name == "nt" else "llama-server")
DEFAULT_HARDWARE_BENCH = Path("build/bin") / (
    "ese-hardware-bench.exe" if os.name == "nt" else "ese-hardware-bench"
)
KNOWN_KV_TYPES = ("auto", "f16", "q8_0", "q4_0")
POLICIES = ("auto", "resident", "hybrid", "cache", "stream")


class ESEError(RuntimeError):
    """Actionable launcher error."""


@dataclasses.dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    total_bytes: int
    free_bytes: int


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
        return _first_positive_int(
            self.metadata,
            (
                f"{self.architecture}.block_count",
                "llama.block_count",
                "general.block_count",
            ),
        )

    @property
    def expert_count(self) -> int | None:
        return _first_positive_int(
            self.metadata,
            (
                f"{self.architecture}.expert_count",
                f"{self.architecture}.expert_used_count",
                "llama.expert_count",
                "general.expert_count",
            ),
        )

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
        return f"{env} {shlex.join(self.command())}".strip()

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
            return f"{int(number)} {suffix}" if suffix == "B" else f"{number:.2f} {suffix}"
        number /= 1024.0
    raise AssertionError("unreachable")


_SIZE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgtp](?:i?b)?|b)?\s*$", re.IGNORECASE)


def parse_size(value: str) -> int:
    match = _SIZE_RE.match(value)
    if not match:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}; examples: 1024M, 8GiB, 1.5T"
        )
    suffix = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "kib": 1024,
        "m": 1000**2,
        "mb": 1000**2,
        "mib": 1024**2,
        "g": 1000**3,
        "gb": 1000**3,
        "gib": 1024**3,
        "t": 1000**4,
        "tb": 1000**4,
        "tib": 1024**4,
        "p": 1000**5,
        "pb": 1000**5,
        "pib": 1024**5,
    }
    return int(float(match.group(1)) * multipliers[suffix])


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
                values[key] = int(raw.strip().split()[0]) * 1024
        except (OSError, ValueError, IndexError):
            pass
        if values.get("MemTotal"):
            return HostMemory(
                total_bytes=values["MemTotal"],
                available_bytes=values.get("MemAvailable", values.get("MemFree", 0)),
            )

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
            return HostMemory(int(status.ullTotalPhys), int(status.ullAvailPhys))

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


_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_SCALARS: dict[int, str] = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
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


def _read_gguf_string(handle: BinaryIO, *, decode: bool = True) -> str:
    length = _read_u64(handle)
    if length > 64 * MIB:
        raise ESEError(f"unreasonable GGUF string length: {length}")
    raw = _read_exact(handle, length)
    return raw.decode("utf-8", errors="replace") if decode else ""


def _read_gguf_value(handle: BinaryIO, value_type: int, *, depth: int = 0) -> Any:
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
        scalar_format = _GGUF_SCALARS.get(element_type)
        if scalar_format:
            handle.seek(struct.calcsize(scalar_format) * length, os.SEEK_CUR)
        elif element_type == _GGUF_STRING:
            for _ in range(length):
                _read_gguf_string(handle, decode=False)
        else:
            for _ in range(length):
                _read_gguf_value(handle, element_type, depth=depth + 1)
        return {"type": element_type, "length": length}
    raise ESEError(f"unsupported GGUF metadata value type: {value_type}")


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    """Read GGUF metadata without loading tensor payloads or materializing arrays."""
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ESEError(f"not a GGUF file: {path}")
        version = _read_u32(handle)
        if version not in (2, 3):
            raise ESEError(f"unsupported GGUF version {version}; expected 2 or 3")
        _tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)
        if metadata_count > 1_000_000:
            raise ESEError(f"unreasonable GGUF metadata count: {metadata_count}")
        metadata: dict[str, Any] = {"_gguf.version": version}
        for _ in range(metadata_count):
            key = _read_gguf_string(handle)
            metadata[key] = _read_gguf_value(handle, _read_u32(handle))
        return metadata


def inspect_model(path: Path) -> ModelInfo:
    shards = discover_model_shards(path)
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
    if "gpt-oss" in text and available_ram and model.total_bytes > max(
        free_vram, int(available_ram * 0.70)
    ):
        return (
            "stream",
            "GPT-OSS exceeds the safe resident-memory budget; use deferred experts",
        )
    if model.is_moe and (not free_vram or model.total_bytes > int(free_vram * 0.85)):
        return (
            "cache",
            "MoE weights do not fit safely in free VRAM; use bounded RAM leases and adaptive VRAM residency",
        )
    if free_vram and model.total_bytes <= int(free_vram * 0.85):
        return (
            "resident",
            f"model fits within 85% of detected free VRAM ({human_bytes(free_vram)})",
        )
    if model.is_moe:
        return "cache", "GPU capacity was not detected; use the bounded RAM expert tier on CPU"
    return "resident", "dense/non-MoE model; use the native resident/auto-fit path"


def auto_tensor_split(gpus: Sequence[GPUInfo]) -> str | None:
    if len(gpus) < 2:
        return None
    weights = [max(1, gpu.free_bytes) for gpu in gpus]
    total = sum(weights)
    percentages = [round(value * 100 / total) for value in weights]
    percentages[-1] += 100 - sum(percentages)
    return ",".join(str(max(1, value)) for value in percentages)


def _mib_ceil(value: int, option: str) -> str:
    if value < MIB:
        raise ESEError(f"{option} must be at least 1MiB")
    return str((value + MIB - 1) // MIB)


def _mib_nonnegative(value: int, option: str) -> str:
    if value < 0:
        raise ESEError(f"{option} cannot be negative")
    return str((value + MIB - 1) // MIB)


def _default_expert_ram_cache(hardware: HardwareInfo) -> int:
    available = hardware.host.available_bytes
    if available <= 0:
        return 512 * MIB
    return max(MIB, min(4 * GIB, available // 8))


def _default_expert_vram_cache(hardware: HardwareInfo, reserve_vram: int) -> int:
    if not hardware.gpus:
        return 0
    usable_per_device = min(max(0, gpu.free_bytes - reserve_vram) for gpu in hardware.gpus)
    candidate = min(2 * GIB, usable_per_device // 4)
    return candidate if candidate >= 64 * MIB else 0


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


def _moe_placement_args(
    model: ModelInfo,
    gpu_resident_moe: int | None,
    *,
    require_cpu_layer: bool,
) -> list[str]:
    keep = 0 if gpu_resident_moe is None else gpu_resident_moe
    if keep < 0:
        raise ESEError("--gpu-resident-moe cannot be negative")
    blocks = model.block_count
    if keep == 0:
        return ["--cpu-moe"]
    if blocks is None:
        raise ESEError(
            "--gpu-resident-moe requires a readable GGUF block_count; "
            "use --cpu-moe after -- as a conservative fallback"
        )
    if keep > blocks or (require_cpu_layer and keep >= blocks):
        limit = blocks - 1 if require_cpu_layer else blocks
        raise ESEError(f"--gpu-resident-moe must be at most {limit} for this model")
    if keep == blocks:
        return []
    return ["--n-cpu-moe", str(blocks - keep)]


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
    gpu_resident_moe: int | None = None,
    tensor_split: str | None = None,
    prefetch_tail: int = 0,
    expert_ram_cache: int | None = None,
    expert_ram_staging: int | None = None,
    expert_vram_cache: int | None = None,
    expert_storage_backend: str | None = None,
    expert_cache_min_observations: int = 2,
    extra_args: Sequence[str] = (),
) -> LaunchPlan:
    if reserve_vram < 0:
        raise ESEError("--reserve-vram cannot be negative")
    if not 0 <= prefetch_tail <= 32:
        raise ESEError("--prefetch-tail must be between 0 and 32")
    selected_policy, reason = select_policy(model, hardware, policy)
    decode_default, batch_default = _default_threads(hardware.logical_cpus)
    threads = threads or decode_default
    batch_threads = batch_threads or batch_default
    kv = choose_kv_type(kv_type, context, hardware, reserve_vram)
    split = tensor_split or auto_tensor_split(hardware.gpus)
    if expert_storage_backend not in (None, "mmap", "pread", "io_uring"):
        raise ESEError("--expert-storage-backend must be mmap, pread, or io_uring")
    if expert_cache_min_observations < 1:
        raise ESEError("--expert-cache-min-observations must be at least 1")

    args: list[str] = [
        "-m",
        str(model.requested_path),
        "--memory-policy",
        policy,
        "--max-ram",
        f"{hardware.host.available_bytes}B",
        "--reserve-vram",
        f"{reserve_vram}B",
        "--min-kv-quality",
        "turbo4",
        "--max-context",
        str(context),
        "--resource-preference",
        "balanced",
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
        if hardware.gpus and model.total_bytes > int(hardware.free_vram * 0.85):
            args.append("--fit")
        if batch_size is not None:
            args.extend(("-b", str(batch_size)))
        if ubatch_size is not None:
            args.extend(("-ub", str(ubatch_size)))

    elif selected_policy == "hybrid":
        if not model.is_moe:
            raise ESEError("--policy hybrid requires a sparse MoE model")
        args.extend(("-ngl", "99"))
        args.extend(
            _moe_placement_args(model, gpu_resident_moe, require_cpu_layer=False)
        )
        args.extend(("-b", str(batch_size or 1024), "-ub", str(ubatch_size or 512)))

    elif selected_policy in ("cache", "stream"):
        if not model.is_moe:
            raise ESEError(f"--policy {selected_policy} requires a sparse MoE model")
        env["GGML_CUDA_NO_PINNED"] = "1"
        backend = expert_storage_backend or ("pread" if selected_policy == "stream" else "mmap")
        if prefetch_tail > 0:
            if backend != "mmap":
                raise ESEError("--prefetch-tail requires --expert-storage-backend mmap")
            env["LLAMA_EXPERT_PREFETCH"] = "1"
            env["LLAMA_EXPERT_PREFETCH_TAIL"] = str(min(prefetch_tail, 32))

        ram_cache = (
            _default_expert_ram_cache(hardware)
            if expert_ram_cache is None
            else expert_ram_cache
        )
        vram_cache = (
            expert_vram_cache
            if expert_vram_cache is not None
            else _default_expert_vram_cache(hardware, reserve_vram)
        )
        args.extend(("-ngl", "99"))
        if selected_policy == "stream":
            args.append("--defer-experts")
        args.extend(
            _moe_placement_args(model, gpu_resident_moe, require_cpu_layer=True)
        )
        args.extend(
            (
                "--expert-ram-cache-mib",
                _mib_ceil(ram_cache, "--expert-ram-cache"),
                "--expert-storage-backend",
                backend,
                "--expert-cache-min-observations",
                str(expert_cache_min_observations),
            )
        )
        if expert_ram_staging is not None:
            if expert_ram_staging > ram_cache:
                raise ESEError("--expert-ram-staging cannot exceed --expert-ram-cache")
            args.extend(
                (
                    "--expert-ram-staging-mib",
                    _mib_ceil(expert_ram_staging, "--expert-ram-staging"),
                )
            )
        if vram_cache:
            args.extend(
                (
                    "--expert-vram-cache-mib",
                    _mib_ceil(vram_cache, "--expert-vram-cache"),
                    "--expert-vram-reserve-mib",
                    _mib_nonnegative(reserve_vram, "--reserve-vram"),
                )
            )
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
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def _execution_environment(plan: LaunchPlan) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(plan.environment)
    if os.name != "nt":
        runtime_library_dir = str(plan.binary.parent)
        inherited_library_path = environment.get("LD_LIBRARY_PATH")
        environment["LD_LIBRARY_PATH"] = (
            f"{runtime_library_dir}{os.pathsep}{inherited_library_path}"
            if inherited_library_path
            else runtime_library_dir
        )
    return environment


def _resolve_binary(value: str | None, build_dir: Path | None = None) -> Path:
    server_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    if value:
        requested = Path(value).expanduser()
        if os.name == "nt" and not requested.suffix:
            requested = requested.with_suffix(".exe")
        return requested.resolve()
    if os.environ.get("ESE_SERVER"):
        requested = Path(os.environ["ESE_SERVER"]).expanduser()
        if os.name == "nt" and not requested.suffix:
            requested = requested.with_suffix(".exe")
        return requested.resolve()
    root = _repo_root()
    binary_root = build_dir if build_dir else root / "build"
    candidates = [
        binary_root / "bin" / server_name,
        binary_root / "bin" / "Release" / server_name,
        binary_root / "bin" / "RelWithDebInfo" / server_name,
        binary_root / "bin" / "Debug" / server_name,
    ]
    return next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())


def _windows_msvc_installation() -> str | None:
    if os.name != "nt":
        return None
    program_files = os.environ.get("ProgramFiles(x86)") or os.environ.get("ProgramFiles")
    if not program_files:
        return None
    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        return None
    return _run_capture(
        (
            str(vswhere),
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        )
    )


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
    print(
        f"GGUF   : arch={plan.model.architecture or 'unknown'}, "
        f"blocks={plan.model.block_count or 'unknown'}, "
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
    compiler = (
        shutil.which("c++")
        or shutil.which("g++")
        or shutil.which("clang++")
        or _windows_msvc_installation()
    )
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
        "-DLLAMA_BUILD_EXAMPLES=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        f"-DGGML_CUDA={'ON' if backend == 'cuda' else 'OFF'}",
    ]
    if os.name == "nt":
        configure.extend(("-A", "x64"))
    elif shutil.which("ninja"):
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
        "ese-hardware-bench",
        "-j",
        str(args.jobs),
    ]
    print("+", shlex.join(build))
    subprocess.run(build, cwd=root, check=True)
    print(f"Built server: {_resolve_binary(None, build_dir)}")
    print(f"Built calibration tool: {build_dir / 'bin' / 'ese-hardware-bench'}")
    return 0


def _calibrate(args: argparse.Namespace) -> int:
    benchmark = Path(args.benchmark).expanduser().resolve()
    if not benchmark.is_file():
        raise ESEError(
            f"hardware benchmark not found: {benchmark}; build the "
            "ese-hardware-bench target or pass --benchmark"
        )
    if args.model is not None and not args.model.expanduser().is_file():
        raise ESEError(f"calibration model not found: {args.model.expanduser()}")
    identity = collect_hardware_identity()
    command = [
        str(benchmark),
        "--json",
        "--iterations",
        str(args.iterations),
        "--bytes",
        str(args.bytes),
    ]
    physical_cores = identity.get("cpu", {}).get("physical_cores", 0)
    if isinstance(physical_cores, int) and physical_cores > 0:
        command.extend(("--threads", str(physical_cores)))
    if args.model is not None:
        command.extend(("--model", str(args.model.expanduser().resolve())))
    completed = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise ESEError(f"hardware calibration failed: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise ESEError("hardware benchmark did not return valid JSON") from exc
    if not isinstance(result, dict) or not isinstance(result.get("measurements"), dict):
        raise ESEError("hardware benchmark JSON is missing measurements")
    source = result.get("benchmark_source")
    if not isinstance(source, dict):
        source = {}
    source = {
        **source,
        "binary": str(benchmark),
        "command": command,
    }
    try:
        profile = build_hardware_profile(
            identity, result["measurements"], benchmark_source=source
        )
        target = save_hardware_profile(profile, args.output)
    except HardwareProfileError as exc:
        raise ESEError(str(exc)) from exc
    payload = {
        "profile": str(target),
        "hardware_fingerprint": profile["hardware_fingerprint"],
        "measurements": profile["measurements"],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Hardware calibration saved: {target}")
        print(f"Fingerprint: {profile['hardware_fingerprint']}")
    return 0


def _hardware_profile_status(args: argparse.Namespace) -> int:
    try:
        profile = load_hardware_profile(args.profile)
        reasons = stale_profile_reasons(profile, collect_hardware_identity())
    except HardwareProfileError as exc:
        raise ESEError(str(exc)) from exc
    payload = {
        "profile": str(args.profile),
        "valid": not reasons,
        "stale_reasons": reasons,
        "created_at": profile["created_at"],
        "hardware_fingerprint": profile["hardware_fingerprint"],
        "benchmark_source": profile["benchmark_source"],
        "measurements": profile["measurements"],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"Hardware profile: {args.profile}")
        print(f"Status: {'ready' if not reasons else 'stale'}")
        for reason in reasons:
            print(f"  - {reason}")
    return 0 if not reasons else 2


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", type=Path, help="GGUF file or first shard")
    parser.add_argument(
        "--policy",
        choices=POLICIES,
        default="auto",
        help="resident, static hybrid, bounded cache, or deferred stream; auto is recommended",
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
        help="VRAM safety reserve used by KV planning (default: 1GiB)",
    )
    parser.add_argument(
        "--gpu-resident-moe",
        type=int,
        help="hybrid/stream: number of final MoE layers retained on GPU",
    )
    parser.add_argument(
        "--tensor-split",
        help="manual split such as 46,54; default is proportional to free VRAM",
    )
    parser.add_argument(
        "--prefetch-tail",
        type=int,
        default=0,
        help="mmap-backed cache/stream: stage N router-logit near-miss experts (0-32)",
    )
    parser.add_argument(
        "--expert-ram-cache",
        type=parse_size,
        help="cache/stream RAM expert capacity (default: one eighth of available RAM, capped at 4GiB)",
    )
    parser.add_argument(
        "--expert-ram-staging",
        type=parse_size,
        help=(
            "cache/stream reusable I/O staging bound "
            "(default: max(largest expert component, min(cache, 64MiB)))"
        ),
    )
    parser.add_argument(
        "--expert-vram-cache",
        type=parse_size,
        help="adaptive expert cache per GPU (default: one quarter of free VRAM after reserve, capped at 2GiB)",
    )
    parser.add_argument(
        "--expert-storage-backend",
        choices=("mmap", "pread", "io_uring"),
        help="exact expert source backend (default: mmap for cache, pread for stream)",
    )
    parser.add_argument(
        "--expert-cache-min-observations",
        type=int,
        default=2,
        help="minimum routes before adaptive VRAM admission (default: 2)",
    )
    parser.add_argument("--json", action="store_true")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ese",
        description="One front door for resident, hybrid, cached, and disk-streamed ESE inference.",
    )
    parser.add_argument("--version", action="version", version="ese 0.1.1")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="inspect build tools, RAM, and GPUs")
    doctor.add_argument("--json", action="store_true")

    build = commands.add_parser("build", help="configure and build llama-server")
    build.add_argument("--backend", choices=("auto", "cuda", "cpu"), default="auto")
    build.add_argument("--build-dir", default=str(_repo_root() / "build"))
    build.add_argument("--build-type", default="Release")
    build.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    build.add_argument("--clean", action="store_true")

    calibrate = commands.add_parser(
        "calibrate", help="measure this machine and save a topology-bound performance profile"
    )
    calibrate.add_argument(
        "--benchmark",
        default=str(_repo_root() / DEFAULT_HARDWARE_BENCH),
        help="path to the native ESE hardware benchmark",
    )
    calibrate.add_argument("--model", type=Path, help="optional GGUF for real expert-kernel trials")
    calibrate.add_argument("--iterations", type=int, default=7)
    calibrate.add_argument("--bytes", type=parse_size, default=256 * MIB, metavar="SIZE")
    calibrate.add_argument("--output", type=Path, default=default_profile_path())
    calibrate.add_argument("--json", action="store_true")

    profile = commands.add_parser(
        "hardware-profile", help="inspect profile validity for the current topology"
    )
    profile.add_argument("--profile", type=Path, default=default_profile_path())
    profile.add_argument("--json", action="store_true")

    plan = commands.add_parser("plan", help="inspect the policy and native command")
    _add_plan_arguments(plan)

    serve = commands.add_parser("serve", help="plan and execute llama-server")
    _add_plan_arguments(serve)
    serve.add_argument("--dry-run", action="store_true")
    return parser


def _plan_from_args(args: argparse.Namespace) -> LaunchPlan:
    if args.context <= 0:
        raise ESEError("--context must be positive")
    if args.slots <= 0:
        raise ESEError("--slots must be positive")
    if not 1 <= args.port <= 65535:
        raise ESEError("--port must be between 1 and 65535")
    return build_launch_plan(
        model=inspect_model(args.model),
        hardware=detect_hardware(),
        binary=_resolve_binary(args.binary),
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
        gpu_resident_moe=args.gpu_resident_moe,
        tensor_split=args.tensor_split,
        prefetch_tail=args.prefetch_tail,
        expert_ram_cache=args.expert_ram_cache,
        expert_ram_staging=args.expert_ram_staging,
        expert_vram_cache=args.expert_vram_cache,
        expert_storage_backend=args.expert_storage_backend,
        expert_cache_min_observations=args.expert_cache_min_observations,
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
        if args.command == "calibrate":
            if args.iterations < 3:
                raise ESEError("--iterations must be at least 3")
            if args.bytes < MIB:
                raise ESEError("--bytes must be at least 1MiB")
            return _calibrate(args)
        if args.command == "hardware-profile":
            return _hardware_profile_status(args)
        plan = _plan_from_args(args)
        _print_plan(plan, args.json)
        if args.command == "plan" or args.dry_run:
            return 0
        if not plan.binary.is_file():
            raise ESEError(
                f"server binary not found: {plan.binary}; run './ese build' or pass --binary"
            )
        environment = _execution_environment(plan)
        if os.name == "nt":
            return subprocess.run(list(plan.command()), env=environment, check=False).returncode
        os.execvpe(str(plan.binary), list(plan.command()), environment)
        return 0
    except ESEError as exc:
        parser.error(str(exc))
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode or 1)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
