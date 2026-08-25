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
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Sequence

from tools.hardware_profile import (
    HardwareProfileError,
    build_hardware_profile,
    collect_hardware_identity,
    default_profile_path,
    hardware_fingerprint,
    load_hardware_profile,
    planner_profile_reasons,
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
HYBRID_VERIFICATION_VERSION = 3
HYBRID_MAX_CALIBRATION_DRIFT = 4.0


def default_hybrid_verification_path() -> Path:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_root / "ese" / "hybrid-verifications.json"


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
    def expert_used_count(self) -> int | None:
        return _first_positive_int(
            self.metadata,
            (
                f"{self.architecture}.expert_used_count",
                "llama.expert_used_count",
                "general.expert_used_count",
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
    context: int = 0
    slots: int = 1
    hybrid_gpu_experts: int = 0
    hybrid_selection: str = "automatic hybrid routing was not evaluated"

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
            "hybrid_routing": {
                "gpu_experts": self.hybrid_gpu_experts,
                "selection": self.hybrid_selection,
            },
            "concurrency": {
                "slots": self.slots,
                "total_context": self.context,
                "context_per_slot": self.context // self.slots,
                "adaptive_expert_cache": "--expert-vram-cache-mib" in self.arguments,
                "kqv_offload": "-nkvo" not in self.arguments,
            },
            "arguments": list(self.arguments),
            "command": list(self.command()),
            "shell_command": self.shell_command(),
        }


def model_fingerprint(model: ModelInfo) -> str:
    """Bind verification to shard contents without hashing a multi-hundred-GB model."""
    digest = hashlib.sha256()
    sample_bytes = 64 * 1024
    for index, shard in enumerate(model.shards):
        size = shard.stat().st_size
        digest.update(struct.pack("<IQ", index, size))
        with shard.open("rb") as handle:
            first = handle.read(min(sample_bytes, size))
            digest.update(struct.pack("<Q", len(first)))
            digest.update(first)
            if size > sample_bytes:
                handle.seek(max(0, size - sample_bytes))
                last = handle.read(sample_bytes)
                digest.update(struct.pack("<Q", len(last)))
                digest.update(last)
    return digest.hexdigest()


def _hybrid_plan_signature(plan: LaunchPlan) -> str:
    # Model contents are bound separately. The listener address does not affect
    # decode, but context, KV, batching, threading, cache bounds, and native
    # overrides all remain part of the workload evidence.
    ignored_with_value = {
        "-m", "--host", "--port", "--max-ram",
    }
    relevant: list[str] = []
    arguments = list(plan.arguments)
    index = 0
    while index < len(arguments):
        value = arguments[index]
        if value in ignored_with_value and index + 1 < len(arguments):
            index += 2
            continue
        relevant.append(value)
        index += 1
    payload = {
        "policy": plan.policy,
        "arguments": relevant,
        "environment": dict(sorted(plan.environment.items())),
        "hybrid_gpu_experts": plan.hybrid_gpu_experts,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _hybrid_verification_key(
    plan: LaunchPlan, current_identity: dict[str, Any]
) -> tuple[str, str, str, str]:
    model_id = model_fingerprint(plan.model)
    hardware_id = hardware_fingerprint(current_identity)
    plan_id = _hybrid_plan_signature(plan)
    key = hashlib.sha256(f"{model_id}:{hardware_id}:{plan_id}".encode("ascii")).hexdigest()
    return key, model_id, hardware_id, plan_id


def _load_hybrid_verifications(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": HYBRID_VERIFICATION_VERSION, "entries": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ESEError(f"cannot read hybrid verification evidence: {exc}") from exc
    if (
        not isinstance(data, dict)
        or data.get("version") != HYBRID_VERIFICATION_VERSION
        or not isinstance(data.get("entries"), dict)
    ):
        raise ESEError("hybrid verification evidence has an unsupported schema")
    return data


def _recorded_hybrid_telemetry_is_valid(record: dict[str, Any]) -> bool:
    summary = record.get("telemetry_summary")
    if not isinstance(summary, dict):
        return False
    required_nonnegative = (
        "layers",
        "misses",
        "route_positions",
        "gpu_route_positions",
        "forced_fallbacks",
        "predicted_upload_ns_per_expert",
        "upload_calibration_drift_ppm",
        "cpu_compute_ns",
        "cpu_compute_calls",
        "predicted_cpu_ns_per_expert",
        "cpu_calibration_drift_ppm",
    )
    for field in required_nonnegative:
        value = summary.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
    return (
        summary["layers"] > 0
        and summary["misses"] >= 3
        and 0 < summary["gpu_route_positions"] < summary["route_positions"]
        and summary["forced_fallbacks"] == 0
        and summary["predicted_upload_ns_per_expert"] > 0
        and summary["cpu_compute_ns"] > 0
        and summary["cpu_compute_calls"] >= 3
        and summary["predicted_cpu_ns_per_expert"] > 0
        and summary["upload_calibration_drift_ppm"]
        <= round(HYBRID_MAX_CALIBRATION_DRIFT * 1_000_000)
        and summary["cpu_calibration_drift_ppm"]
        <= round(HYBRID_MAX_CALIBRATION_DRIFT * 1_000_000)
    )


def hybrid_verification_reason(
    plan: LaunchPlan,
    current_identity: dict[str, Any],
    path: Path,
) -> str | None:
    try:
        evidence = _load_hybrid_verifications(path)
        key, model_id, hardware_id, plan_id = _hybrid_verification_key(plan, current_identity)
    except (ESEError, OSError) as exc:
        return str(exc)
    record = evidence["entries"].get(key)
    if not isinstance(record, dict):
        return "no matching workload A/B verification is recorded"
    if (
        record.get("model_fingerprint") != model_id
        or record.get("hardware_fingerprint") != hardware_id
        or record.get("plan_signature") != plan_id
    ):
        return "workload A/B verification does not match the current model, hardware, or plan"
    if record.get("output_parity") is not True:
        return "workload A/B verification failed deterministic output parity"
    if record.get("telemetry_valid") is not True or not _recorded_hybrid_telemetry_is_valid(record):
        return "workload A/B verification failed live hybrid telemetry checks"
    if record.get("passed") is not True:
        return "measured hybrid workload did not beat the established path"
    speedup = record.get("speedup")
    minimum = record.get("minimum_speedup")
    if (
        isinstance(speedup, bool)
        or isinstance(minimum, bool)
        or not isinstance(speedup, (int, float))
        or not isinstance(minimum, (int, float))
        or not math.isfinite(float(speedup))
        or not math.isfinite(float(minimum))
        or minimum <= 1.0
        or speedup < minimum
    ):
        return "workload A/B verification has invalid performance evidence"
    return None


def _save_hybrid_verification(
    path: Path,
    plan: LaunchPlan,
    current_identity: dict[str, Any],
    result: dict[str, Any],
) -> None:
    evidence = _load_hybrid_verifications(path)
    key, model_id, hardware_id, plan_id = _hybrid_verification_key(plan, current_identity)
    record = {
        **result,
        "model_fingerprint": model_id,
        "hardware_fingerprint": hardware_id,
        "plan_signature": plan_id,
        "gpu_experts": plan.hybrid_gpu_experts,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    evidence["entries"][key] = record
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(evidence, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


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


def _server_supports_cuda(binary: Path) -> bool:
    override = os.environ.get("ESE_SERVER_BACKEND", "").strip().lower()
    if override in {"cuda", "cpu"}:
        return override == "cuda"

    marker = binary.parent / "ese-runtime.json"
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        if isinstance(value.get("cuda"), bool):
            return value["cuda"]
    except (OSError, ValueError, AttributeError):
        pass

    cache = binary.parent.parent / "CMakeCache.txt"
    try:
        return bool(
            re.search(r"^GGML_CUDA:BOOL=ON\s*$", cache.read_text(encoding="utf-8"), re.MULTILINE)
        )
    except OSError:
        return False


def _hardware_for_server(binary: Path) -> HardwareInfo:
    hardware = detect_hardware()
    if hardware.gpus and not _server_supports_cuda(binary):
        return dataclasses.replace(hardware, gpus=())
    return hardware


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


def read_gguf_index(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read GGUF metadata and tensor descriptors without loading tensor payloads."""
    with path.open("rb") as handle:
        if _read_exact(handle, 4) != b"GGUF":
            raise ESEError(f"not a GGUF file: {path}")
        version = _read_u32(handle)
        if version not in (2, 3):
            raise ESEError(f"unsupported GGUF version {version}; expected 2 or 3")
        tensor_count = _read_u64(handle)
        metadata_count = _read_u64(handle)
        if tensor_count > 10_000_000:
            raise ESEError(f"unreasonable GGUF tensor count: {tensor_count}")
        if metadata_count > 1_000_000:
            raise ESEError(f"unreasonable GGUF metadata count: {metadata_count}")
        metadata: dict[str, Any] = {"_gguf.version": version}
        for _ in range(metadata_count):
            key = _read_gguf_string(handle)
            metadata[key] = _read_gguf_value(handle, _read_u32(handle))
        tensors: list[dict[str, Any]] = []
        for _ in range(tensor_count):
            name = _read_gguf_string(handle)
            dimensions_count = _read_u32(handle)
            if not 1 <= dimensions_count <= 4:
                raise ESEError(f"invalid GGUF tensor rank {dimensions_count}: {name}")
            dimensions = tuple(_read_u64(handle) for _ in range(dimensions_count))
            ggml_type = _read_u32(handle)
            offset = _read_u64(handle)
            tensors.append(
                {"name": name, "dimensions": dimensions, "ggml_type": ggml_type, "offset": offset}
            )
        alignment = int(metadata.get("general.alignment", 32))
        if alignment <= 0 or alignment > MIB or alignment & (alignment - 1):
            raise ESEError(f"invalid GGUF alignment: {alignment}")
        data_offset = (handle.tell() + alignment - 1) // alignment * alignment
        payload_bytes = max(0, path.stat().st_size - data_offset)
        offsets = sorted({int(tensor["offset"]) for tensor in tensors})
        if offsets and (offsets[0] < 0 or offsets[-1] > payload_bytes):
            raise ESEError(f"GGUF tensor offset is outside the data payload: {path}")
        next_offset = {
            offset: offsets[index + 1] if index + 1 < len(offsets) else payload_bytes
            for index, offset in enumerate(offsets)
        }
        for tensor in tensors:
            tensor["span_bytes"] = max(0, next_offset[int(tensor["offset"])] - int(tensor["offset"]))
            tensor["data_offset"] = data_offset + int(tensor["offset"])
        return metadata, tensors


def read_gguf_metadata(path: Path) -> dict[str, Any]:
    return read_gguf_index(path)[0]


_EXPERT_TENSOR_RE = re.compile(r"\.ffn_(?:gate|up|down|gate_up)_exps(?:\.weight)?$", re.I)
_EXPERT_COMPONENT_RE = re.compile(
    r"^(?P<layer>.+)\.ffn_(?P<component>gate|up|down|gate_up)_exps(?:\.weight)?$",
    re.I,
)


def inspect_expert_geometries(path: Path) -> list[dict[str, Any]]:
    """Return unique real expert component formats/geometries from all shards."""
    candidates: list[dict[str, Any]] = []
    for shard in discover_model_shards(path):
        _, tensors = read_gguf_index(shard)
        for tensor in tensors:
            dimensions = tensor["dimensions"]
            if not _EXPERT_TENSOR_RE.search(tensor["name"]) or len(dimensions) < 3:
                continue
            expert_count = int(dimensions[-1])
            if expert_count <= 0 or tensor["span_bytes"] <= 0:
                continue
            candidates.append(
                {
                    **tensor,
                    "expert_count": expert_count,
                    "expert_component_bytes": int(tensor["span_bytes"]) // expert_count,
                    "shard": str(shard),
                }
            )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in candidates:
        key = (
            item["ggml_type"], tuple(item["dimensions"]),
            item["expert_component_bytes"], item["expert_count"],
        )
        unique.setdefault(key, item)
    return sorted(
        unique.values(),
        key=lambda item: (item["ggml_type"], tuple(item["dimensions"]), item["expert_component_bytes"]),
    )


def inspect_expert_geometry(path: Path) -> dict[str, Any] | None:
    geometries = inspect_expert_geometries(path)
    return max(geometries, key=lambda item: item["expert_component_bytes"]) if geometries else None


def inspect_expert_layer_formats(path: Path) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Return unique per-layer expert component multisets used by decode."""
    layers: dict[str, list[tuple[int, int, int, int]]] = {}
    for shard in discover_model_shards(path):
        _, tensors = read_gguf_index(shard)
        for tensor in tensors:
            match = _EXPERT_COMPONENT_RE.match(tensor["name"])
            dimensions = tensor["dimensions"]
            if not match or len(dimensions) < 3:
                continue
            expert_count = int(dimensions[-1])
            span_bytes = int(tensor["span_bytes"])
            if expert_count <= 0 or span_bytes <= 0:
                continue
            layers.setdefault(match.group("layer"), []).append(
                (
                    int(tensor["ggml_type"]),
                    int(dimensions[0]),
                    int(dimensions[1]),
                    span_bytes // expert_count,
                )
            )
    return tuple(sorted({tuple(sorted(components)) for components in layers.values()}))


def _solve_calibrated_hybrid(
    profile: dict[str, Any],
    layer_formats: Sequence[Sequence[tuple[int, int, int, int]]],
    expert_used: int,
) -> tuple[int, str]:
    """Mirror the native integer split solver and fail closed on missing formats."""
    if expert_used < 2:
        return 0, "model does not expose a mixed top-k expert route"
    contention = profile.get("measurements", {}).get("cpu_cache_contention", {})
    devices = contention.get("devices", []) if isinstance(contention, dict) else []
    if not layer_formats or not isinstance(devices, list) or not devices:
        return 0, "profile or model has no calibrated expert layout"

    solved: list[int] = []
    for device in devices:
        if not isinstance(device, dict) or not isinstance(device.get("backend"), str):
            return 0, "profile has an invalid calibrated backend"
        entries = device.get("profiles", [])
        if not isinstance(entries, list):
            return 0, f"profile has no calibrated formats for {device['backend']}"
        costs: dict[tuple[int, int, int, int], tuple[float, float]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            try:
                key = tuple(int(entry[name]) for name in (
                    "ggml_type_id", "input_width", "expert_width",
                    "bytes_per_expert_component",
                ))
                costs[key] = (
                    float(entry["cpu_ns_per_expert_component"]),
                    float(entry["upload_ns_per_expert_component"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        for components in layer_formats:
            try:
                cpu_ns = sum(costs[tuple(component)][0] for component in components)
                upload_ns = sum(costs[tuple(component)][1] for component in components)
            except KeyError:
                return 0, f"profile has no exact component match for {device['backend']}"
            if not cpu_ns > 0 or not upload_ns > 0:
                return 0, f"profile contains invalid costs for {device['backend']}"
            best_uploads = min(
                range(expert_used + 1),
                key=lambda uploads: max(
                    (expert_used - uploads) * cpu_ns,
                    uploads * upload_ns,
                ),
            )
            solved.append(best_uploads)

    conservative = min(solved, default=0)
    if conservative == 0:
        return 0, "calibration selected the established all-CPU expert path"
    if conservative >= expert_used:
        return 0, "calibration selected the established all-GPU cache path"
    return conservative, (
        f"planner-ready calibration selected {conservative} GPU and "
        f"{expert_used - conservative} CPU route positions"
    )


def calibrated_hybrid_gpu_experts(
    model: ModelInfo,
    profile_path: Path,
    current_identity: dict[str, Any],
) -> tuple[int, str]:
    try:
        profile = load_hardware_profile(profile_path)
    except HardwareProfileError as exc:
        return 0, f"automatic hybrid routing disabled: {exc}"
    reasons = planner_profile_reasons(profile, current_identity)
    if reasons:
        return 0, f"automatic hybrid routing disabled: {reasons[0]}"
    expert_used = model.expert_used_count
    if expert_used is None:
        return 0, "automatic hybrid routing disabled: model top-k metadata is unavailable"
    try:
        layouts = inspect_expert_layer_formats(model.requested_path)
    except ESEError as exc:
        return 0, f"automatic hybrid routing disabled: {exc}"
    return _solve_calibrated_hybrid(profile, layouts, expert_used)


def _calibrated_expert_cost_bounds(
    model: ModelInfo, profile_path: Path
) -> tuple[float, float]:
    try:
        profile = load_hardware_profile(profile_path)
    except HardwareProfileError as exc:
        raise ESEError(f"cannot load calibrated upload costs: {exc}") from exc
    layouts = inspect_expert_layer_formats(model.requested_path)
    contention = profile.get("measurements", {}).get("cpu_cache_contention", {})
    devices = contention.get("devices", []) if isinstance(contention, dict) else []
    if not layouts or not isinstance(devices, list) or not devices:
        raise ESEError("calibration has no model upload-cost matrix")
    predicted_cpu: list[float] = []
    predicted_upload: list[float] = []
    for device in devices:
        if not isinstance(device, dict) or not isinstance(device.get("profiles"), list):
            raise ESEError("calibration has an invalid device upload-cost matrix")
        costs: dict[tuple[int, int, int, int], tuple[float, float]] = {}
        for entry in device["profiles"]:
            if not isinstance(entry, dict):
                continue
            try:
                key = tuple(int(entry[name]) for name in (
                    "ggml_type_id",
                    "input_width",
                    "expert_width",
                    "bytes_per_expert_component",
                ))
                cpu_value = float(entry["cpu_ns_per_expert_component"])
                upload_value = float(entry["upload_ns_per_expert_component"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                math.isfinite(cpu_value)
                and cpu_value > 0
                and math.isfinite(upload_value)
                and upload_value > 0
            ):
                costs[key] = (cpu_value, upload_value)
        for components in layouts:
            try:
                component_costs = [costs[tuple(component)] for component in components]
            except KeyError as exc:
                raise ESEError("calibration has no exact cost for a model component") from exc
            predicted_cpu.append(sum(cost[0] for cost in component_costs))
            predicted_upload.append(sum(cost[1] for cost in component_costs))
    if (
        not predicted_cpu
        or not predicted_upload
        or not all(math.isfinite(value) and value > 0 for value in predicted_cpu)
        or not all(math.isfinite(value) and value > 0 for value in predicted_upload)
    ):
        raise ESEError("calibration produced no valid expert cost bounds")
    return max(predicted_cpu), max(predicted_upload)


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
    expert_prefill_staging: int | None = None,
    expert_storage_backend: str | None = None,
    expert_cache_min_observations: int = 2,
    hybrid_gpu_experts: int = 0,
    hybrid_cpu_ns_per_expert: int = 0,
    hybrid_upload_ns_per_expert: int = 0,
    hybrid_maximum_drift_ppm: int = 4_000_000,
    hybrid_minimum_cpu_calls: int = 64,
    hybrid_selection: str = "automatic hybrid routing was not evaluated",
    extra_args: Sequence[str] = (),
) -> LaunchPlan:
    if slots < 1:
        raise ESEError("--slots must be positive")
    if context < slots or context % slots:
        raise ESEError("--context must divide evenly across --slots")
    if reserve_vram < 0:
        raise ESEError("--reserve-vram cannot be negative")
    if not 0 <= prefetch_tail <= 32:
        raise ESEError("--prefetch-tail must be between 0 and 32")
    selected_policy, reason = select_policy(model, hardware, policy)
    if slots > 1 and (selected_policy != "resident" or model.is_moe):
        raise ESEError(
            "concurrent sessions currently require a dense resident model; MoE, "
            "hybrid, cache, and stream execution are limited to one session until "
            "their multi-sequence output parity gate passes"
        )
    decode_default, batch_default = _default_threads(hardware.logical_cpus)
    threads = threads or decode_default
    batch_threads = batch_threads or batch_default
    kv = choose_kv_type(kv_type, context, hardware, reserve_vram)
    split = tensor_split or auto_tensor_split(hardware.gpus)
    if expert_storage_backend not in (None, "mmap", "pread", "io_uring"):
        raise ESEError("--expert-storage-backend must be mmap, pread, or io_uring")
    if expert_cache_min_observations < 1:
        raise ESEError("--expert-cache-min-observations must be at least 1")
    if expert_prefill_staging is not None and expert_prefill_staging < 0:
        raise ESEError("--expert-prefill-staging cannot be negative")
    if expert_prefill_staging and selected_policy not in ("cache", "stream"):
        raise ESEError("--expert-prefill-staging requires cache or stream policy")
    if hybrid_gpu_experts < 0 or hybrid_gpu_experts > 64:
        raise ESEError("hybrid GPU experts must be between 0 and 64")
    if hybrid_gpu_experts and (
        hybrid_cpu_ns_per_expert <= 0 or hybrid_upload_ns_per_expert <= 0
    ):
        raise ESEError("hybrid routing requires positive calibrated runtime guard costs")
    if hybrid_maximum_drift_ppm < 1_000_000:
        raise ESEError("hybrid runtime guard drift must be at least 1000000 ppm")
    if hybrid_minimum_cpu_calls < 1:
        raise ESEError("hybrid runtime guard requires at least one CPU call")

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
    if expert_prefill_staging is not None:
        args.extend((
            "--expert-prefill-staging-mib",
            _mib_nonnegative(expert_prefill_staging, "--expert-prefill-staging"),
        ))

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
        # Streaming must not pin the model's potentially enormous CPU tensor
        # storage. Keep bounded exact CUDA staging available independently;
        # GGML_CUDA_NO_PINNED remains the user's global emergency kill switch.
        env["GGML_CUDA_NO_MODEL_PINNED"] = "1"
        backend = expert_storage_backend or ("pread" if selected_policy == "stream" else "mmap")
        if hybrid_gpu_experts and backend != "pread":
            hybrid_gpu_experts = 0
            hybrid_selection = (
                "automatic hybrid routing disabled: planner calibration covers "
                "the pread bounded-lease path"
            )
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
            if hybrid_gpu_experts:
                args.extend(
                    (
                        "--expert-hybrid-gpu-experts",
                        str(hybrid_gpu_experts),
                        "--expert-hybrid-cpu-ns-per-expert",
                        str(hybrid_cpu_ns_per_expert),
                        "--expert-hybrid-upload-ns-per-expert",
                        str(hybrid_upload_ns_per_expert),
                        "--expert-hybrid-maximum-drift-ppm",
                        str(hybrid_maximum_drift_ppm),
                        "--expert-hybrid-minimum-cpu-calls",
                        str(hybrid_minimum_cpu_calls),
                    )
                )
        elif hybrid_gpu_experts:
            hybrid_gpu_experts = 0
            hybrid_selection = "automatic hybrid routing disabled: expert VRAM cache is unavailable"
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
        context=context,
        slots=slots,
        hybrid_gpu_experts=hybrid_gpu_experts if selected_policy in ("cache", "stream") else 0,
        hybrid_selection=(
            hybrid_selection
            if selected_policy in ("cache", "stream")
            else "automatic hybrid routing is not used by the selected policy"
        ),
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


def _plan_with_port(plan: LaunchPlan, port: int) -> LaunchPlan:
    arguments = list(plan.arguments)
    try:
        position = arguments.index("--port")
        arguments[position + 1] = str(port)
    except (ValueError, IndexError) as exc:
        raise ESEError("planned server command has no valid --port option") from exc
    return dataclasses.replace(plan, arguments=tuple(arguments))


def _baseline_hybrid_plan(plan: LaunchPlan) -> LaunchPlan:
    arguments = list(plan.arguments)
    if "--expert-hybrid-gpu-experts" not in arguments:
        raise ESEError("calibration did not produce a mixed hybrid plan to validate")
    for option in (
        "--expert-hybrid-gpu-experts",
        "--expert-hybrid-cpu-ns-per-expert",
        "--expert-hybrid-upload-ns-per-expert",
        "--expert-hybrid-maximum-drift-ppm",
        "--expert-hybrid-minimum-cpu-calls",
    ):
        try:
            position = arguments.index(option)
        except ValueError:
            continue
        del arguments[position : position + 2]
    return dataclasses.replace(
        plan,
        arguments=tuple(arguments),
        hybrid_gpu_experts=0,
        hybrid_selection="workload A/B established-path control",
    )


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _http_json(
    port: int,
    path: str,
    payload: dict[str, Any] | None,
    timeout: float,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            parsed = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ESEError(f"server request {path} failed: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ESEError(f"server request {path} returned non-object JSON")
    return parsed


def _parse_expert_cache_telemetry(log_path: Path) -> dict[str, Any]:
    prefix = "expert_cache_stats: "
    guard_prefix = "expert_hybrid_guard: "
    layers: list[dict[str, Any]] = []
    totals: list[dict[str, Any]] = []
    guard_failures: list[dict[str, Any]] = []
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ESEError(f"cannot read server telemetry: {exc}") from exc
    for line in lines:
        guard_position = line.find(guard_prefix)
        if guard_position >= 0:
            try:
                guard = json.loads(line[guard_position + len(guard_prefix) :])
            except json.JSONDecodeError as exc:
                raise ESEError(f"server emitted malformed hybrid-guard telemetry: {exc}") from exc
            if not isinstance(guard, dict):
                raise ESEError("server emitted non-object hybrid-guard telemetry")
            guard_failures.append(guard)
        position = line.find(prefix)
        if position < 0:
            continue
        try:
            item = json.loads(line[position + len(prefix) :])
        except json.JSONDecodeError as exc:
            raise ESEError(f"server emitted malformed expert-cache telemetry: {exc}") from exc
        if not isinstance(item, dict):
            raise ESEError("server emitted non-object expert-cache telemetry")
        if item.get("level") == "vram-layer":
            layers.append(item)
        elif item.get("level") == "vram-total":
            totals.append(item)
    return {"layers": layers, "totals": totals, "guard_failures": guard_failures}


def _validated_hybrid_telemetry(telemetry: dict[str, Any]) -> dict[str, int]:
    layers = telemetry.get("layers")
    totals = telemetry.get("totals")
    guard_failures = telemetry.get("guard_failures")
    if not isinstance(guard_failures, list):
        raise ESEError("hybrid server omitted runtime-guard telemetry state")
    if guard_failures:
        first = guard_failures[0]
        reason = first.get("reason", "unknown") if isinstance(first, dict) else "unknown"
        detail = ""
        if isinstance(first, dict) and reason == "cpu-drift":
            observed = first.get("cpu_compute_ns")
            calls = first.get("cpu_compute_calls")
            predicted = first.get("cpu_ns_per_expert")
            positions = first.get("cpu_route_positions")
            if all(isinstance(value, int) and value > 0 for value in (observed, calls, predicted, positions)):
                drift = observed / (calls * predicted * positions)
                detail = f", observed/predicted={drift:.3f}x"
        elif isinstance(first, dict) and reason == "upload-drift":
            misses = first.get("misses")
            predicted = first.get("upload_ns_per_expert")
            observed_values = [first.get(name) for name in (
                "lease_acquire_ns", "transfer_submit_ns", "transfer_wait_ns"
            )]
            if (
                isinstance(misses, int) and misses > 0
                and isinstance(predicted, int) and predicted > 0
                and all(isinstance(value, int) and value >= 0 for value in observed_values)
            ):
                drift = sum(observed_values) / (misses * predicted)
                detail = f", observed/predicted={drift:.3f}x"
        raise ESEError(
            f"hybrid runtime guard revoked the mixed route during validation ({reason}{detail})"
        )
    if not isinstance(layers, list) or not layers:
        raise ESEError("hybrid server emitted no per-layer expert-cache telemetry")
    if not isinstance(totals, list) or len(totals) != 1 or not isinstance(totals[0], dict):
        raise ESEError("hybrid server did not emit exactly one aggregate expert-cache record")
    total = totals[0]
    layer_fields = (
        "routes",
        "route_positions",
        "gpu_route_positions",
        "hits",
        "misses",
        "lease_acquire_ns",
        "lease_uploads",
        "transfer_submit_ns",
        "transfer_wait_ns",
        "load_bytes",
        "cpu_compute_ns",
        "cpu_compute_calls",
    )
    sums = {field: 0 for field in layer_fields}
    seen_layers: set[int] = set()
    for item in layers:
        if not isinstance(item, dict):
            raise ESEError("hybrid server emitted a non-object per-layer telemetry record")
        layer = item.get("layer")
        if isinstance(layer, bool) or not isinstance(layer, int) or layer < 0:
            raise ESEError("hybrid server emitted an invalid telemetry layer index")
        if layer in seen_layers:
            raise ESEError(f"hybrid server emitted duplicate telemetry for layer {layer}")
        seen_layers.add(layer)
        for field in layer_fields:
            value = item.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ESEError(f"hybrid layer telemetry has invalid {field}")
            sums[field] += value

    for field in (
        "hits",
        "misses",
        "lease_uploads",
        "transfer_submit_ns",
        "transfer_wait_ns",
        "load_bytes",
        "cpu_compute_ns",
        "cpu_compute_calls",
    ):
        value = total.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ESEError(f"hybrid aggregate telemetry has invalid {field}")
        if value != sums[field]:
            raise ESEError(f"hybrid per-layer {field} does not reconcile with its aggregate")
    forced_fallbacks = total.get("forced_fallbacks")
    if (
        isinstance(forced_fallbacks, bool)
        or not isinstance(forced_fallbacks, int)
        or forced_fallbacks < 0
    ):
        raise ESEError("hybrid aggregate telemetry has invalid forced_fallbacks")
    if forced_fallbacks != 0:
        raise ESEError("hybrid execution used a forbidden host-tensor fallback")
    if sums["routes"] <= 0 or sums["route_positions"] <= 0:
        raise ESEError("hybrid telemetry did not observe routed expert work")
    if not 0 < sums["gpu_route_positions"] < sums["route_positions"]:
        raise ESEError("hybrid telemetry did not prove both CPU and GPU route partitions")
    return {
        "layers": len(seen_layers),
        **sums,
        "forced_fallbacks": forced_fallbacks,
    }


def _validated_calibration_drift(
    telemetry_summary: dict[str, int], predicted_upload_ns_per_expert: float
) -> float:
    if not math.isfinite(predicted_upload_ns_per_expert) or predicted_upload_ns_per_expert <= 0:
        raise ESEError("calibrated expert upload prediction is invalid")
    misses = telemetry_summary.get("misses", 0)
    if misses < 3:
        raise ESEError("hybrid workload observed too few cache misses to validate calibration")
    observed_ns = sum(
        telemetry_summary.get(field, 0)
        for field in ("lease_acquire_ns", "transfer_submit_ns", "transfer_wait_ns")
    )
    if observed_ns <= 0:
        raise ESEError("hybrid workload reported no measured upload-path time")
    drift = observed_ns / (predicted_upload_ns_per_expert * misses)
    if not math.isfinite(drift) or drift > HYBRID_MAX_CALIBRATION_DRIFT:
        raise ESEError(
            "live upload-path timing contradicts calibration "
            f"({drift:.3f}x observed/predicted; limit {HYBRID_MAX_CALIBRATION_DRIFT:.1f}x)"
        )
    return drift


def _validated_cpu_calibration_drift(
    telemetry_summary: dict[str, int],
    predicted_cpu_ns_per_expert: float,
    cpu_route_positions: int,
) -> float:
    if not math.isfinite(predicted_cpu_ns_per_expert) or predicted_cpu_ns_per_expert <= 0:
        raise ESEError("calibrated expert CPU prediction is invalid")
    if cpu_route_positions <= 0:
        raise ESEError("hybrid plan has no CPU route positions")
    calls = telemetry_summary.get("cpu_compute_calls", 0)
    observed_ns = telemetry_summary.get("cpu_compute_ns", 0)
    if calls < 3 or observed_ns <= 0:
        raise ESEError("hybrid workload observed too little CPU-branch timing data")
    predicted_ns = predicted_cpu_ns_per_expert * cpu_route_positions * calls
    drift = observed_ns / predicted_ns
    if not math.isfinite(drift) or drift > HYBRID_MAX_CALIBRATION_DRIFT:
        raise ESEError(
            "live CPU-branch timing contradicts calibration "
            f"({drift:.3f}x observed/predicted; limit {HYBRID_MAX_CALIBRATION_DRIFT:.1f}x)"
        )
    return drift


def _completion_result(
    port: int, prompt: str, tokens: int, seed: int, timeout: float
) -> tuple[str, float]:
    response = _http_json(
        port,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": tokens,
            "temperature": 0,
            "seed": seed,
            "cache_prompt": False,
        },
        timeout,
    )
    content = response.get("content")
    timings = response.get("timings")
    if not isinstance(content, str) or not isinstance(timings, dict):
        raise ESEError("completion response omitted content or timings")
    speed = timings.get("predicted_per_second")
    if (
        isinstance(speed, bool)
        or not isinstance(speed, (int, float))
        or not math.isfinite(float(speed))
        or speed <= 0
    ):
        predicted_n = timings.get("predicted_n")
        predicted_ms = timings.get("predicted_ms")
        if (
            not isinstance(predicted_n, bool)
            and not isinstance(predicted_ms, bool)
            and isinstance(predicted_n, (int, float))
            and isinstance(predicted_ms, (int, float))
            and math.isfinite(float(predicted_n))
            and math.isfinite(float(predicted_ms))
            and predicted_n > 0
            and predicted_ms > 0
        ):
            speed = predicted_n * 1000.0 / predicted_ms
        else:
            raise ESEError("completion response omitted valid decode throughput")
    return hashlib.sha256(content.encode("utf-8")).hexdigest(), float(speed)


def _run_hybrid_validation_server(
    plan: LaunchPlan,
    *,
    label: str,
    prompt: str,
    tokens: int,
    samples: int,
    warmups: int,
    startup_timeout: float,
    request_timeout: float,
    log_path: Path,
) -> dict[str, Any]:
    port = _free_tcp_port()
    plan = _plan_with_port(plan, port)
    result: dict[str, Any] | None = None
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            plan.command(),
            env=_execution_environment(plan),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + startup_timeout
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise ESEError(
                        f"{label} server exited during startup with code {process.returncode}"
                    )
                try:
                    _http_json(port, "/health", None, 2.0)
                    break
                except ESEError:
                    time.sleep(0.25)
            else:
                raise ESEError(
                    f"{label} server did not become healthy within {startup_timeout:.0f}s"
                )

            for index in range(warmups):
                _completion_result(
                    port, f"{prompt}\nWarmup {index}.", tokens, 1234 + index, request_timeout
                )
            hashes: list[str] = []
            speeds: list[float] = []
            for index in range(samples):
                output_hash, speed = _completion_result(
                    port,
                    f"{prompt}\nMeasured trial {index}.",
                    tokens,
                    4321 + index,
                    request_timeout,
                )
                hashes.append(output_hash)
                speeds.append(speed)
            result = {
                "output_hashes": hashes,
                "tokens_per_second_samples": speeds,
                "tokens_per_second_median": statistics.median(speeds),
            }
        except ESEError as exc:
            log.flush()
            try:
                tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            except OSError:
                tail = []
            detail = "\n".join(tail)
            raise ESEError(f"{exc}\n{detail}" if detail else str(exc)) from exc
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    if result is None:
        raise ESEError(f"{label} server produced no validation result")
    result["expert_cache_telemetry"] = _parse_expert_cache_telemetry(log_path)
    return result


def _validate_hybrid(args: argparse.Namespace) -> int:
    if args.samples < 3:
        raise ESEError("--samples must be at least 3")
    if args.tokens < 2:
        raise ESEError("--tokens must be at least 2")
    if args.warmups < 0:
        raise ESEError("--warmups cannot be negative")
    if not math.isfinite(args.minimum_speedup) or args.minimum_speedup <= 1.0:
        raise ESEError("--minimum-speedup must be greater than 1.0")
    if not math.isfinite(args.startup_timeout) or args.startup_timeout <= 0:
        raise ESEError("--startup-timeout must be positive")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ESEError("--request-timeout must be positive")

    candidate = _plan_from_args(args, require_hybrid_verification=False)
    if candidate.hybrid_gpu_experts <= 0:
        raise ESEError(candidate.hybrid_selection)
    baseline = _baseline_hybrid_plan(candidate)
    current_identity = collect_hardware_identity()
    evidence_path = args.hybrid_verification.expanduser().resolve()
    prompt = args.prompt or (
        "A careful local ESE benchmark validates deterministic expert routing, "
        "output parity, and stable decode throughput. " * 32
    )
    with tempfile.TemporaryDirectory(prefix="ese-hybrid-validation-") as temporary:
        root = Path(temporary)
        baseline_result = _run_hybrid_validation_server(
            baseline,
            label="established-path",
            prompt=prompt,
            tokens=args.tokens,
            samples=args.samples,
            warmups=args.warmups,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            log_path=root / "baseline.log",
        )
        hybrid_result = _run_hybrid_validation_server(
            candidate,
            label="hybrid",
            prompt=prompt,
            tokens=args.tokens,
            samples=args.samples,
            warmups=args.warmups,
            startup_timeout=args.startup_timeout,
            request_timeout=args.request_timeout,
            log_path=root / "hybrid.log",
        )

    output_parity = baseline_result["output_hashes"] == hybrid_result["output_hashes"]
    baseline_speed = float(baseline_result["tokens_per_second_median"])
    hybrid_speed = float(hybrid_result["tokens_per_second_median"])
    speedup = hybrid_speed / baseline_speed
    telemetry_reason = None
    try:
        telemetry_summary = _validated_hybrid_telemetry(
            hybrid_result["expert_cache_telemetry"]
        )
        predicted_cpu_ns, predicted_upload_ns = _calibrated_expert_cost_bounds(
            candidate.model, args.hardware_profile
        )
        telemetry_summary["predicted_upload_ns_per_expert"] = round(predicted_upload_ns)
        telemetry_summary["upload_calibration_drift_ppm"] = round(
            _validated_calibration_drift(telemetry_summary, predicted_upload_ns) * 1_000_000
        )
        telemetry_summary["predicted_cpu_ns_per_expert"] = round(predicted_cpu_ns)
        cpu_positions = (candidate.model.expert_used_count or 0) - candidate.hybrid_gpu_experts
        telemetry_summary["cpu_calibration_drift_ppm"] = round(
            _validated_cpu_calibration_drift(
                telemetry_summary, predicted_cpu_ns, cpu_positions
            )
            * 1_000_000
        )
        telemetry_valid = True
    except ESEError as exc:
        telemetry_summary = {}
        telemetry_valid = False
        telemetry_reason = str(exc)
    passed = output_parity and telemetry_valid and speedup >= args.minimum_speedup
    result = {
        "passed": passed,
        "output_parity": output_parity,
        "output_hashes": hybrid_result["output_hashes"] if output_parity else [],
        "telemetry_valid": telemetry_valid,
        "telemetry_reason": telemetry_reason,
        "telemetry_summary": telemetry_summary,
        "baseline_tokens_per_second": baseline_speed,
        "hybrid_tokens_per_second": hybrid_speed,
        "speedup": speedup,
        "minimum_speedup": args.minimum_speedup,
        "samples": args.samples,
        "warmups": args.warmups,
        "tokens_per_sample": args.tokens,
    }
    _save_hybrid_verification(evidence_path, candidate, current_identity, result)
    output = {**result, "evidence": str(evidence_path), "gpu_experts": candidate.hybrid_gpu_experts}
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print(f"Hybrid workload verification: {'PASS' if passed else 'FAIL'}")
        print(f"Output parity: {'exact' if output_parity else 'failed'}")
        print(f"Live telemetry: {'valid' if telemetry_valid else telemetry_reason}")
        print(f"Established path: {baseline_speed:.3f} tok/s")
        print(f"Hybrid path:      {hybrid_speed:.3f} tok/s ({speedup:.3f}x)")
        print(f"Evidence: {evidence_path}")
    return 0 if passed else 2


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
    print(f"Hybrid : {plan.hybrid_selection}")
    print(
        f"Slots  : {plan.slots} concurrent; "
        f"{plan.context // plan.slots} context each ({plan.context} total)"
    )
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
        "server_cuda": _server_supports_cuda(server),
        "build_ready": bool(cmake and compiler),
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
    # ``doctor`` is also the stable machine-readable hardware probe used by
    # Studio.  A packaged runtime does not need CMake or a compiler, so their
    # absence must not turn a valid hardware report into a failed probe.
    # Callers that care about source-build readiness can inspect build_ready.
    return 0


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
    model_path: Path | None = None
    expert_geometries: list[dict[str, Any]] = []
    if args.model is not None:
        requested_model = args.model.expanduser().resolve()
        if requested_model.is_dir():
            models = sorted(requested_model.glob("*.gguf"))
            if not models:
                raise ESEError(f"calibration model directory contains no GGUF files: {requested_model}")
            model_path = models[0]
        elif requested_model.is_file():
            model_path = requested_model
        else:
            raise ESEError(f"calibration model not found: {requested_model}")
        expert_geometries = inspect_expert_geometries(model_path)
        if not expert_geometries:
            raise ESEError(f"no routed expert tensors found in calibration model: {model_path}")
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
    if model_path is not None:
        command.extend(("--model", str(model_path)))
        for geometry in expert_geometries:
            dimensions = geometry["dimensions"]
            command.extend(
                (
                    "--model-expert-spec",
                    "|".join(
                        str(value) for value in (
                            geometry["ggml_type"], dimensions[0], dimensions[1],
                            geometry["expert_count"], geometry["expert_component_bytes"],
                            geometry["data_offset"], geometry["shard"],
                        )
                    ),
                )
            )
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
        identity = collect_hardware_identity()
        reasons = stale_profile_reasons(profile, identity)
        planner_reasons = planner_profile_reasons(profile, identity)
    except HardwareProfileError as exc:
        raise ESEError(str(exc)) from exc
    payload = {
        "profile": str(args.profile),
        "valid": not reasons,
        "stale_reasons": reasons,
        "planner_ready": not planner_reasons,
        "planner_reasons": planner_reasons,
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
        print(f"Planner: {'ready' if not planner_reasons else 'not ready'}")
        for reason in planner_reasons:
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
        "--expert-prefill-staging",
        type=parse_size,
        help=(
            "total two-lane whole-layer prefill staging per GPU; omitted lets "
            "the native planner size it automatically, 0 disables it"
        ),
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
    parser.add_argument(
        "--hardware-profile",
        type=Path,
        default=default_profile_path(),
        help="topology-bound calibration profile used for automatic hybrid routing",
    )
    parser.add_argument(
        "--no-auto-hybrid",
        action="store_true",
        help="disable calibrated CPU/GPU expert route splitting",
    )
    parser.add_argument(
        "--hybrid-candidate",
        type=int,
        help=(
            "advanced: evaluate a mixed GPU route-position count; it remains disabled "
            "until this exact candidate passes validate-hybrid"
        ),
    )
    parser.add_argument(
        "--hybrid-verification",
        type=Path,
        default=default_hybrid_verification_path(),
        help="hardware/model/config-bound workload A/B verification evidence",
    )
    parser.add_argument("--json", action="store_true")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ese",
        description="One front door for resident, hybrid, cached, and disk-streamed ESE inference.",
    )
    parser.add_argument("--version", action="version", version="ese 0.2.0")
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
    calibrate.add_argument(
        "--model",
        type=Path,
        help="GGUF file, first shard, or directory for model-backed expert calibration",
    )
    calibrate.add_argument("--iterations", type=int, default=21)
    calibrate.add_argument("--bytes", type=parse_size, default=256 * MIB, metavar="SIZE")
    calibrate.add_argument("--output", type=Path, default=default_profile_path())
    calibrate.add_argument("--json", action="store_true")

    profile = commands.add_parser(
        "hardware-profile", help="inspect profile validity for the current topology"
    )
    profile.add_argument("--profile", type=Path, default=default_profile_path())
    profile.add_argument("--json", action="store_true")

    validate_hybrid = commands.add_parser(
        "validate-hybrid",
        help="A/B test calibrated hybrid decode before allowing automatic activation",
    )
    _add_plan_arguments(validate_hybrid)
    validate_hybrid.add_argument("--samples", type=int, default=5)
    validate_hybrid.add_argument("--warmups", type=int, default=1)
    validate_hybrid.add_argument("--tokens", type=int, default=16)
    validate_hybrid.add_argument("--minimum-speedup", type=float, default=1.02)
    validate_hybrid.add_argument("--prompt")
    validate_hybrid.add_argument("--startup-timeout", type=float, default=600.0)
    validate_hybrid.add_argument("--request-timeout", type=float, default=600.0)

    plan = commands.add_parser("plan", help="inspect the policy and native command")
    _add_plan_arguments(plan)

    serve = commands.add_parser("serve", help="plan and execute llama-server")
    _add_plan_arguments(serve)
    serve.add_argument("--dry-run", action="store_true")
    return parser


def _plan_from_args(
    args: argparse.Namespace, *, require_hybrid_verification: bool = True
) -> LaunchPlan:
    if args.context <= 0:
        raise ESEError("--context must be positive")
    if args.slots <= 0:
        raise ESEError("--slots must be positive")
    if not 1 <= args.port <= 65535:
        raise ESEError("--port must be between 1 and 65535")
    model = inspect_model(args.model)
    binary = _resolve_binary(args.binary)
    # A CPU-only packaged server must not inherit driver-visible GPUs from the
    # host.  Runtime-aware detection also lets the Windows CUDA marker select
    # GPU planning without requiring build tools in the installed package.
    hardware = _hardware_for_server(binary)
    hybrid_gpu_experts = 0
    current_identity: dict[str, Any] | None = None
    selected_policy, _ = select_policy(model, hardware, args.policy)
    hybrid_selection = "automatic hybrid routing is not used by the selected policy"
    hybrid_candidate = getattr(args, "hybrid_candidate", None)
    if hybrid_candidate is not None and selected_policy not in ("cache", "stream"):
        raise ESEError("--hybrid-candidate requires the cache or stream policy")
    if args.no_auto_hybrid and hybrid_candidate is not None:
        raise ESEError("--hybrid-candidate cannot be combined with --no-auto-hybrid")
    if selected_policy in ("cache", "stream") and args.no_auto_hybrid:
        hybrid_selection = "automatic hybrid routing disabled by --no-auto-hybrid"
    elif selected_policy in ("cache", "stream"):
        current_identity = collect_hardware_identity()
        hybrid_gpu_experts, hybrid_selection = calibrated_hybrid_gpu_experts(
            model, args.hardware_profile, current_identity
        )
        if hybrid_candidate is not None:
            expert_used = model.expert_used_count
            if expert_used is None or not 1 <= hybrid_candidate < expert_used:
                raise ESEError(
                    "--hybrid-candidate must be between 1 and one less than model top-k"
                )
            conservative_endpoint = hybrid_selection in (
                "calibration selected the established all-CPU expert path",
                "calibration selected the established all-GPU cache path",
            )
            if hybrid_gpu_experts <= 0 and not conservative_endpoint:
                raise ESEError(
                    f"--hybrid-candidate requires valid exact calibration: {hybrid_selection}"
                )
            hybrid_gpu_experts = hybrid_candidate
            hybrid_selection = (
                f"advanced evidence-gated candidate selected {hybrid_candidate} GPU and "
                f"{expert_used - hybrid_candidate} CPU route positions"
            )
    hybrid_cpu_ns_per_expert = 0
    hybrid_upload_ns_per_expert = 0
    if hybrid_gpu_experts > 0:
        predicted_cpu_ns, predicted_upload_ns = _calibrated_expert_cost_bounds(
            model, args.hardware_profile
        )
        hybrid_cpu_ns_per_expert = max(1, round(predicted_cpu_ns))
        hybrid_upload_ns_per_expert = max(1, round(predicted_upload_ns))
    plan_arguments = dict(
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
        gpu_resident_moe=args.gpu_resident_moe,
        tensor_split=args.tensor_split,
        prefetch_tail=args.prefetch_tail,
        expert_ram_cache=args.expert_ram_cache,
        expert_ram_staging=args.expert_ram_staging,
        expert_vram_cache=args.expert_vram_cache,
        expert_prefill_staging=getattr(args, "expert_prefill_staging", None),
        expert_storage_backend=args.expert_storage_backend,
        expert_cache_min_observations=args.expert_cache_min_observations,
        hybrid_gpu_experts=hybrid_gpu_experts,
        hybrid_cpu_ns_per_expert=hybrid_cpu_ns_per_expert,
        hybrid_upload_ns_per_expert=hybrid_upload_ns_per_expert,
        hybrid_selection=hybrid_selection,
        extra_args=args.extra,
    )
    plan = build_launch_plan(**plan_arguments)
    if require_hybrid_verification and plan.hybrid_gpu_experts > 0:
        if current_identity is None:
            current_identity = collect_hardware_identity()
        reason = hybrid_verification_reason(
            plan, current_identity, args.hybrid_verification.expanduser().resolve()
        )
        if reason:
            plan_arguments["hybrid_gpu_experts"] = 0
            plan_arguments["hybrid_selection"] = (
                f"automatic hybrid routing disabled: {reason}; run `ese validate-hybrid`"
            )
            plan = build_launch_plan(**plan_arguments)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = _make_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    native_args: list[str] = []
    if "--" in raw:
        separator = raw.index("--")
        native_args = raw[separator + 1 :]
        raw = raw[:separator]
    args = parser.parse_args(raw)
    if args.command in ("plan", "serve", "validate-hybrid"):
        args.extra = native_args
    elif native_args:
        parser.error(
            "native arguments after -- are supported only by plan, serve, and validate-hybrid"
        )

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
        if args.command == "validate-hybrid":
            return _validate_hybrid(args)
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
