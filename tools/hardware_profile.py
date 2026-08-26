"""Versioned, topology-bound hardware calibration profiles for ESE."""

from __future__ import annotations

import hashlib
import json
import math
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
    if isinstance(source, Mapping) and source.get("planner_ready") is True:
        required_usage = {
            "expert_payload_cpu_moe",
            "bounded_ram_lease_upload",
            "bounded_ram_lease_contention",
            "per_device_contention",
        }
        usage = source.get("model_usage", [])
        if source.get("calibration_level") != "planner" or not isinstance(usage, list) or not required_usage.issubset(usage):
            reasons.append("planner provenance does not prove every production calibration path")

        def finite_number(value: object, *, minimum: float | None = None, maximum: float | None = None) -> bool:
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                return False
            return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)

        def format_key(item: object) -> tuple[int, int, int, int] | None:
            if not isinstance(item, Mapping):
                return None
            values = tuple(item.get(name) for name in (
                "ggml_type_id", "input_width", "expert_width", "bytes_per_expert_component"
            ))
            if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in values):
                return None
            return values  # type: ignore[return-value]

        cpu_measurement = measurements.get("cpu_moe", {})
        cpu_profiles = cpu_measurement.get("model_profiles", []) if isinstance(cpu_measurement, Mapping) else []
        expected_formats: set[tuple[int, int, int, int]] = set()
        if not isinstance(cpu_profiles, list) or not cpu_profiles:
            reasons.append("planner profile has no model-backed CPU expert measurements")
        else:
            for item in cpu_profiles:
                key = format_key(item)
                if key is None or key in expected_formats:
                    reasons.append("planner CPU formats are invalid or duplicated")
                    continue
                expected_formats.add(key)
                if (
                    item.get("correctness") != "single-thread-and-dequantized-scalar-reference"
                    or not finite_number(item.get("independent_reference_nrmse"), minimum=0, maximum=0.08)
                    or not finite_number(item.get("confidence"), minimum=0.80)
                    or not finite_number(item.get("relative_standard_error"), minimum=0)
                    or not finite_number(item.get("sample_count"), minimum=7)
                ):
                    reasons.append(f"planner CPU correctness or confidence is incomplete for type {key[0]}")

        contention = measurements.get("cpu_cache_contention", {})
        contention_devices = contention.get("devices", []) if isinstance(contention, Mapping) else []
        if not isinstance(contention, Mapping) or (
            contention.get("upload_path") != "pread_to_bounded_ram_lease_to_async_upload"
            or contention.get("distribution") != "warm-steady-state"
        ):
            reasons.append("planner contention did not use the labeled bounded lease path")
        contention_backends: set[str] = set()
        if not isinstance(contention_devices, list) or not contention_devices:
            reasons.append("planner profile has no per-device contention measurements")
        else:
            for device in contention_devices:
                backend = device.get("backend", "unknown") if isinstance(device, Mapping) else "unknown"
                if not isinstance(backend, str) or not backend or backend in contention_backends:
                    reasons.append("planner contention devices are invalid or duplicated")
                    continue
                contention_backends.add(backend)
                profiles = device.get("profiles", []) if isinstance(device, Mapping) else []
                if not isinstance(profiles, list) or not profiles:
                    reasons.append(f"planner profile has no contention formats for {backend}")
                    continue
                actual_formats = {key for item in profiles if (key := format_key(item)) is not None}
                if len(actual_formats) != len(profiles) or actual_formats != expected_formats:
                    reasons.append(f"planner contention formats are incomplete for {backend}")
                for measurement in profiles:
                    valid_confidence = isinstance(measurement, Mapping) and all((
                        finite_number(measurement.get("cpu_ns_per_expert_component"), minimum=0),
                        finite_number(measurement.get("upload_ns_per_expert_component"), minimum=0),
                        finite_number(measurement.get("cpu_confidence"), minimum=0.80),
                        finite_number(measurement.get("upload_confidence"), minimum=0.80),
                        finite_number(measurement.get("cpu_relative_standard_error"), minimum=0),
                        finite_number(measurement.get("upload_relative_standard_error"), minimum=0),
                        finite_number(measurement.get("cpu_sample_count"), minimum=7),
                        finite_number(measurement.get("upload_sample_count"), minimum=7),
                    ))
                    valid_confidence = valid_confidence and (
                        measurement.get("cpu_ns_per_expert_component", 0) > 0
                        and measurement.get("upload_ns_per_expert_component", 0) > 0
                    )
                    if not valid_confidence:
                        reasons.append(f"planner confidence is below 0.80 for {backend}")
                        break

        upload_measurement = measurements.get("expert_cache_upload", {})
        lease_profiles = upload_measurement.get("lease_upload_profiles", []) if isinstance(upload_measurement, Mapping) else []
        if not isinstance(lease_profiles, list) or not lease_profiles:
            reasons.append("planner profile has no exact leased-upload measurements")
        else:
            lease_pairs: set[tuple[str, tuple[int, int, int, int]]] = set()
            for item in lease_profiles:
                key = format_key(item)
                backend = item.get("backend") if isinstance(item, Mapping) else None
                pair = (backend, key) if isinstance(backend, str) and key is not None else None
                if pair is None or pair in lease_pairs:
                    reasons.append("planner leased-upload formats are invalid or duplicated")
                    continue
                lease_pairs.add(pair)
                if (
                    item.get("storage_backend") != "pread"
                    or item.get("distribution") != "warm-steady-state"
                    or not finite_number(item.get("confidence"), minimum=0.80)
                    or not finite_number(item.get("relative_standard_error"), minimum=0)
                    or not finite_number(item.get("sample_count"), minimum=7)
                ):
                    reasons.append(f"planner leased-upload evidence is incomplete for {backend}")
            expected_pairs = {
                (backend, key) for backend in contention_backends for key in expected_formats
            }
            if lease_pairs != expected_pairs:
                reasons.append("planner leased-upload matrix does not match contention devices and formats")
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
