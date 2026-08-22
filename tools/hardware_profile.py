"""Versioned, topology-bound hardware calibration profiles for ESE."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = 1
PROFILE_FILENAME = "hardware-profile.json"


class HardwareProfileError(RuntimeError):
    """A profile is malformed, unsafe to reuse, or cannot be persisted."""


def default_profile_path() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache) if cache else Path.home() / ".cache"
    return root / "ese" / PROFILE_FILENAME


def _capture(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _linux_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _positive_int(value: object, default: int = 0) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _nonnegative_int(value: object, default: int = -1) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _cpu_topology(capture: Callable[[Sequence[str]], str | None]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "architecture": platform.machine(),
        "model": _linux_cpu_model(),
        "logical_cores": os.cpu_count() or 1,
        "physical_cores": 0,
        "sockets": 0,
        "numa_nodes": 0,
    }
    raw = capture(("lscpu", "--json"))
    if not raw:
        return result
    try:
        fields = {
            str(item.get("field", "")).rstrip(":").strip(): item.get("data")
            for item in json.loads(raw).get("lscpu", [])
        }
    except (AttributeError, json.JSONDecodeError, TypeError):
        return result
    result["architecture"] = str(fields.get("Architecture") or result["architecture"])
    result["model"] = str(fields.get("Model name") or result["model"])
    result["logical_cores"] = _positive_int(fields.get("CPU(s)"), result["logical_cores"])
    cores_per_socket = _positive_int(fields.get("Core(s) per socket"))
    sockets = _positive_int(fields.get("Socket(s)"))
    result["sockets"] = sockets
    result["physical_cores"] = cores_per_socket * sockets
    result["numa_nodes"] = _positive_int(fields.get("NUMA node(s)"))
    return result


def _gpu_identity(capture: Callable[[Sequence[str]], str | None]) -> list[dict[str, Any]]:
    query = "index,uuid,name,pci.bus_id,compute_cap,driver_version,memory.total"
    raw = capture(("nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"))
    if not raw:
        return []
    devices: list[dict[str, Any]] = []
    for line in raw.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        devices.append(
            {
                "index": _nonnegative_int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "pci_bus_id": parts[3].lower(),
                "compute_capability": parts[4],
                "driver_version": parts[5],
                "total_memory_mib": _positive_int(parts[6]),
            }
        )
    return sorted(devices, key=lambda device: (device["uuid"], device["pci_bus_id"]))


def collect_hardware_identity(
    capture: Callable[[Sequence[str]], str | None] = _capture,
) -> dict[str, Any]:
    identity = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
        },
        "cpu": _cpu_topology(capture),
        "gpus": _gpu_identity(capture),
    }
    topology = capture(("nvidia-smi", "topo", "-m"))
    if topology:
        # Normalize whitespace while preserving the topology matrix and NUMA affinity.
        identity["nvidia_topology"] = "\n".join(
            re.sub(r"\s+", " ", line.strip()) for line in topology.splitlines() if line.strip()
        )
    return identity


def hardware_fingerprint(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_hardware_profile(
    identity: Mapping[str, Any],
    measurements: Mapping[str, Any],
    *,
    benchmark_source: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(measurements, Mapping) or not measurements:
        raise HardwareProfileError("calibration produced no measurements")
    profile = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hardware_fingerprint": hardware_fingerprint(identity),
        "hardware": dict(identity),
        "benchmark_source": dict(benchmark_source),
        "measurements": dict(measurements),
    }
    validate_hardware_profile(profile)
    return profile


def validate_hardware_profile(profile: Mapping[str, Any]) -> None:
    if profile.get("schema_version") != SCHEMA_VERSION:
        raise HardwareProfileError(
            f"unsupported hardware profile schema {profile.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    hardware = profile.get("hardware")
    if not isinstance(hardware, Mapping):
        raise HardwareProfileError("hardware profile is missing its hardware identity")
    expected = hardware_fingerprint(hardware)
    if profile.get("hardware_fingerprint") != expected:
        raise HardwareProfileError("hardware profile fingerprint does not match its identity")
    if not isinstance(profile.get("created_at"), str) or not profile["created_at"]:
        raise HardwareProfileError("hardware profile is missing its creation time")
    if not isinstance(profile.get("measurements"), Mapping) or not profile["measurements"]:
        raise HardwareProfileError("hardware profile contains no measurements")
    if not isinstance(profile.get("benchmark_source"), Mapping):
        raise HardwareProfileError("hardware profile is missing benchmark provenance")


def stale_profile_reasons(
    profile: Mapping[str, Any], current_identity: Mapping[str, Any]
) -> list[str]:
    reasons: list[str] = []
    try:
        validate_hardware_profile(profile)
    except HardwareProfileError as exc:
        return [str(exc)]
    if profile["hardware_fingerprint"] != hardware_fingerprint(current_identity):
        old_cpu = profile["hardware"].get("cpu", {})
        new_cpu = current_identity.get("cpu", {})
        if old_cpu != new_cpu:
            reasons.append("CPU or NUMA topology changed")
        old_gpus = profile["hardware"].get("gpus", [])
        new_gpus = current_identity.get("gpus", [])
        if old_gpus != new_gpus:
            reasons.append("GPU, driver, PCIe identity, or device order changed")
        if profile["hardware"].get("os") != current_identity.get("os"):
            reasons.append("operating system or kernel changed")
        if profile["hardware"].get("nvidia_topology") != current_identity.get("nvidia_topology"):
            reasons.append("GPU/NUMA interconnect topology changed")
        if not reasons:
            reasons.append("hardware identity changed")
    return reasons


def planner_profile_reasons(
    profile: Mapping[str, Any], current_identity: Mapping[str, Any]
) -> list[str]:
    reasons = stale_profile_reasons(profile, current_identity)
    if reasons:
        return reasons
    source = profile.get("benchmark_source", {})
    if not isinstance(source, Mapping) or source.get("planner_ready") is not True:
        reasons.append("calibration is baseline-only and not approved for planner decisions")
    measurements = profile.get("measurements", {})
    for name in ("cpu_moe", "expert_cache_upload", "cpu_cache_contention"):
        measurement = measurements.get(name, {}) if isinstance(measurements, Mapping) else {}
        if not isinstance(measurement, Mapping) or measurement.get("status") != "measured":
            reasons.append(f"required planner measurement is unavailable: {name}")
    return reasons


def load_hardware_profile(path: Path | None = None) -> dict[str, Any]:
    target = path or default_profile_path()
    try:
        loaded = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HardwareProfileError(f"hardware profile does not exist: {target}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HardwareProfileError(f"could not read hardware profile {target}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise HardwareProfileError("hardware profile root must be an object")
    validate_hardware_profile(loaded)
    return loaded


def save_hardware_profile(profile: Mapping[str, Any], path: Path | None = None) -> Path:
    validate_hardware_profile(profile)
    target = path or default_profile_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
