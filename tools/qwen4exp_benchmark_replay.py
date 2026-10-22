#!/usr/bin/env python3
"""Build and (optionally) run the canonical Qwen4Exp Strata-comparable baseline.

Roadmap A6 harness. It turns ``docs/qwen4exp/benchmark-workload.json`` plus one
value per feature toggle into each engine's exact command line, so a baseline
can be replayed and a feature ablation changes exactly one variable.

The Strata path cannot use the launcher's ``strata_service.sh`` for this: that
wrapper always reads the on-disk config, so a toggle could never reach the
engine. The harness instead writes a config derived from the on-disk one and
launches ``serve/server.py`` directly on a dedicated port, then stops the whole
process group.

Diagnostic only. The ESE path needs the dedicated probe build *and* an explicit
``--allow-probe-gate`` flag, and it only ever sets the runtime opt-in on the
child process it spawns. The production loader gate stays closed; this harness
must never be wired into a normal launch path.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "docs" / "qwen4exp" / "benchmark-workload.json"


class Unsupported(ValueError):
    """A toggle asks an engine for a capability it does not implement."""


def canonical_hash(value: dict[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    expected = spec.get("workload_sha256")
    payload = {key: value for key, value in spec.items() if key != "workload_sha256"}
    actual = canonical_hash(payload)
    if expected != actual:
        raise ValueError(f"workload spec hash mismatch: expected {expected}, got {actual}")
    return spec


def n_predict_for(spec: dict[str, Any], phase: str) -> int:
    if phase == "prefill":
        return 1
    if phase == "decode":
        return int(spec["workload"]["n_predict"])
    raise Unsupported(f"unknown phase: {phase}")


def build_ese_argv(spec: dict[str, Any], *, phase: str, kv_policy: str, mtp: str) -> list[str]:
    if mtp == "on":
        raise Unsupported("ESE Qwen4Exp has no MTP/speculative path (see README capability matrix)")
    if kv_policy not in ("resident", "host"):
        raise Unsupported(f"unknown kv_policy: {kv_policy}")
    wl = spec["workload"]
    ese = spec["ese"]
    moe_placement = ese.get("moe_placement", "gpu")
    if moe_placement not in ("cpu", "gpu"):
        raise Unsupported(f"unknown ese.moe_placement: {moe_placement}")
    argv = [
        ese["binary"],
        "-m", ese["model"],
        "-c", str(wl["context_length"]),
        "-b", str(wl["logical_batch"]),
        "-ub", str(wl["ubatch"]),
        "-ngl", "99",
        "-dev", ",".join(wl["devices"]),
        "-sm", wl["split_mode"],
        "-ts", wl["tensor_split"],
        "-ctk", wl["kv_precision_ese"],
        "-ctv", wl["kv_precision_ese"],
        "-fa", "on",
        "--ese-ple-mode", ese["ple_mode"],
        "--ese-ple-cache-mib", str(ese["ple_cache_mib"]),
        "--ese-ple-queue-depth", str(ese["ple_queue_depth"]),
        "-p", wl["prompt_text"],
        "-n", str(n_predict_for(spec, phase)),
        "--temp", str(wl["temperature"]),
        "-s", str(wl["seed"]),
        "--no-warmup",
        "--simple-io",
    ]
    if moe_placement == "cpu":
        # 24.3 GiB of experts cannot fit the CUDA0+CUDA2 budget; keep them host-resident.
        argv += ["--cpu-moe"]
    if ese.get("expert_vram_cache_mib"):
        argv += ["--expert-vram-cache-mib", str(ese["expert_vram_cache_mib"])]
    if kv_policy == "host":
        # No bounded resident QSA-KV window yet: keep the whole history host-side.
        argv += ["-nkvo"]
    return argv


def build_strata_plan(spec: dict[str, Any], *, phase: str, kv_policy: str, mtp: str) -> dict[str, Any]:
    if kv_policy not in ("resident", "host"):
        raise Unsupported(f"unknown kv_policy: {kv_policy}")
    wl = spec["workload"]
    strata = spec["strata"]
    config = json.loads(Path(strata["config"]).read_text(encoding="utf-8"))
    args = list(config["args"])
    if kv_policy == "host":
        # Strata kv_plan() returns a fully GPU-resident plan when --kv-resident is
        # absent/zero. Removing the flag is NOT a host-only ablation.
        raise Unsupported("Strata host-only KV is not exposed by this config; removing "
                          "--kv-resident makes KV fully GPU-resident, not host-resident")
    if mtp == "off":
        args = _strip_option(_strip_option(args, "--spec", takes_value=True), "--mtp", takes_value=True)
    request = {
        "model": config.get("model_name", "local"),
        "messages": [{"role": "user", "content": wl["prompt_text"]}],
        "max_tokens": n_predict_for(spec, phase),
        "temperature": wl["temperature"],
        "seed": wl["seed"],
        "stream": False,
    }
    port = int(strata["benchmark_port"])
    return {
        # The launcher's own wrapper always reads the on-disk config, so it cannot express a
        # toggle. The benchmark therefore launches serve/server.py directly with a config
        # generated from `config_template` + `engine_args` (see strata_engine_config).
        "service_command": [strata["service"], "start"],
        "service_stop": [strata["service"], "stop"],
        "health_url": f"http://127.0.0.1:{port}/health",
        "request_url": f"http://127.0.0.1:{port}/v1/chat/completions",
        "runtime": {
            "python": strata["python"],
            "server": strata["server"],
            "port": port,
            "cwd": config.get("cwd"),
            "model_name": config.get("model_name"),
        },
        "config_template": config,
        "engine_args": args,
        "request": request,
    }


def strata_engine_config(plan: dict[str, Any], *, log_path: Path | None = None) -> dict[str, Any]:
    """The exact config the benchmark hands to serve/server.py for this toggle set."""
    config = dict(plan["config_template"])
    config["args"] = list(plan["engine_args"])
    config["port"] = plan["runtime"]["port"]
    if log_path is not None:
        config["log"] = str(log_path)
    return config


def _strip_option(args: list[str], flag: str, *, takes_value: bool) -> list[str]:
    out: list[str] = []
    index = 0
    while index < len(args):
        if args[index] == flag:
            index += 2 if takes_value else 1
            continue
        out.append(args[index])
        index += 1
    return out


def drop_os_cache(files: list[str]) -> dict[str, Any]:
    """Evict the OS page cache for these files. Does not touch device caches."""
    dropped, failed = [], []
    for name in files:
        try:
            fd = os.open(name, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            dropped.append(name)
        except OSError as error:
            failed.append({"file": name, "error": str(error)})
    return {
        "method": "posix_fadvise_dontneed",
        "physical_device_cache_claim": False,
        "dropped": dropped,
        "failed": failed,
    }


LLAMA_EVAL = re.compile(
    r"(?<!prompt )eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:runs|tokens)\s*\(.*?([0-9.]+)\s*tokens per second",
    re.S,
)
LLAMA_PROMPT = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*(?:tokens|runs).*?([0-9.]+)\s*tokens per second", re.S)


def parse_llama_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    if (match := LLAMA_PROMPT.search(text)):
        metrics["prompt_eval_ms"] = float(match.group(1))
        metrics["prompt_tokens"] = int(match.group(2))
        metrics["prompt_tokens_per_s"] = float(match.group(3))
    if (match := LLAMA_EVAL.search(text)):
        metrics["decode_ms"] = float(match.group(1))
        metrics["decode_tokens"] = int(match.group(2))
        # llama-cli prints a sentinel 1,000,000 tok/s when no decode was timed.
        # Preserve the duration/count but do not expose that as real throughput.
        if metrics["decode_ms"] > 0 and metrics["decode_tokens"] > 0:
            metrics["decode_tokens_per_s"] = float(match.group(3))
            metrics["ms_per_token"] = metrics["decode_ms"] / metrics["decode_tokens"]
        else:
            metrics["decode_timing_available"] = False
    return metrics


def decode_child_output(raw: bytes) -> str:
    """Raw model output is arbitrary BPE bytes, so it is not valid UTF-8 in general."""
    return raw.decode("utf-8", errors="replace")


def run_ese(spec: dict[str, Any], argv: list[str], *, allow_probe_gate: bool, timeout: int) -> dict[str, Any]:
    if not allow_probe_gate:
        raise Unsupported(
            "refusing to run the ESE probe without --allow-probe-gate; the production "
            "loader gate must stay closed in normal paths"
        )
    env = dict(os.environ)
    env[spec["ese"]["probe_gate_env"]] = "1"
    started = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env=env, start_new_session=True)
    try:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_session(proc)
            stdout, stderr = proc.communicate(timeout=5)
    finally:
        _terminate_session(proc)
    wall_ms = (time.monotonic() - started) * 1000.0
    return {
        "exit_code": 124 if timed_out else proc.returncode,
        "timed_out": timed_out,
        "wall_ms": wall_ms,
        "stdout": decode_child_output(stdout),
        "stderr": decode_child_output(stderr),
        "stdout_bytes": stdout,
        "stderr_bytes": stderr,
        "env": {spec["ese"]["probe_gate_env"]: "1"},
    }


def _terminate_session(proc: subprocess.Popen) -> None:
    """Stop the server *and* the engine it spawned (own process group)."""
    # start_new_session=True makes the spawned PID the group ID. The parent
    # may already have exited while an engine/worker still holds pipes open.
    # Signal only this owned group, never a group obtained from a reused PID.
    group = proc.pid
    try:
        os.killpg(group, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        proc.poll()  # reap the leader while checking surviving descendants
        try:
            os.killpg(group, 0)
        except ProcessLookupError:
            proc.wait()
            return
        time.sleep(0.05)
    try:
        os.killpg(group, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait(timeout=5)


def strata_server_argv(plan: dict[str, Any], config_path: Path) -> list[str]:
    runtime = plan["runtime"]
    return [runtime["python"], runtime["server"], "--engine", "strata",
            "--config", str(config_path), "--port", str(runtime["port"])]


def prepare_strata_run(plan: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write the toggle-specific engine config. Paths are absolute: serve/server.py
    runs with the engine's own cwd, so a relative config path resolves wrongly."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Each run owns a fresh directory; repeated ablations never overwrite evidence.
    output_dir = Path(tempfile.mkdtemp(prefix="strata-run-", dir=output_dir))
    log_path = output_dir / "strata-engine.log"
    config_path = output_dir / "strata-run-config.json"
    config_path.write_text(json.dumps(strata_engine_config(plan, log_path=log_path), indent=2) + "\n",
                           encoding="utf-8")
    return {"output_dir": output_dir, "log_path": log_path, "config_path": config_path}


def run_strata(plan: dict[str, Any], *, output_dir: Path | None, timeout: int) -> dict[str, Any]:
    runtime = plan["runtime"]
    # Fail closed before launching: a matching health response on an occupied
    # port belongs to another process, not this run's owned server.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", int(runtime["port"])))
        except OSError as error:
            raise Unsupported(f"Strata benchmark port {runtime['port']} is unavailable; "
                              "refusing to use an existing service") from error
    started = time.monotonic()
    scratch = None
    if output_dir is None:
        scratch = tempfile.TemporaryDirectory(prefix="qwen4exp-strata-")
        output_dir = Path(scratch.name)
    paths = prepare_strata_run(plan, output_dir)
    log_path, config_path = paths["log_path"], paths["config_path"]

    argv = strata_server_argv(plan, config_path)
    ready, health, payload, code, failure = False, {}, None, 1, None
    # The engine opens its own log in append mode. Server narration must use a
    # different file or its non-append descriptor can overwrite engine records.
    server_log_path = paths["output_dir"] / "strata-server.log"
    engine_log = open(server_log_path, "wb")
    proc = subprocess.Popen(argv, cwd=runtime["cwd"], stdout=engine_log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                failure = f"strata server exited during startup (code {proc.returncode})"
                break
            try:
                with urllib.request.urlopen(plan["health_url"], timeout=5) as response:
                    health = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, OSError, ValueError):
                health = {}
            if (health.get("status") == "ok" and health.get("loaded")
                    and health.get("model") == runtime["model_name"]):
                ready = True
                break
            time.sleep(3)
        else:
            failure = f"strata /health never reported loaded within {timeout}s"
        if ready:
            body = json.dumps(plan["request"]).encode("utf-8")
            request = urllib.request.Request(plan["request_url"], data=body,
                                             headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read().decode("utf-8", errors="replace")
                code = 0
            except Exception as error:  # keep the failure visible in the bundle
                payload, failure = str(error), str(error)
    finally:
        _terminate_session(proc)
        engine_log.close()
        if scratch is not None:
            scratch.cleanup()
    return {
        "exit_code": code,
        "wall_ms": (time.monotonic() - started) * 1000.0,
        "stdout": payload or "",
        "stderr": failure or "",
        "env": {},
        "health": health,
        "ready": ready,
        "engine_log_path": str(log_path),
        "engine_config_path": str(config_path),
        "server_log_path": str(server_log_path),
    }


def parse_strata_metrics(payload: str | None) -> dict[str, Any]:
    """Strata answers OpenAI-style; keep usage and any llama-server style timings."""
    if not payload:
        return {}
    try:
        body = json.loads(payload)
    except ValueError:
        return {}
    metrics: dict[str, Any] = {}
    for key in ("usage", "timings"):
        if isinstance(body.get(key), dict):
            metrics[key] = body[key]
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--engine", choices=["ese", "strata"], default="ese")
    parser.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    parser.add_argument("--cache-state", choices=["cold", "warm"], default="warm")
    parser.add_argument("--mtp", choices=["on", "off"], default="off")
    parser.add_argument("--kv-policy", choices=["resident", "host"], default="host")
    parser.add_argument("--output", type=Path, help="write the evidence bundle here")
    parser.add_argument("--execute", action="store_true", help="actually run; default is a dry plan")
    parser.add_argument("--allow-probe-gate", action="store_true",
                        help="required for --engine ese --execute; sets the opt-in only on the child")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_spec(args.spec)
    toggles = {"phase": args.phase, "cache_state": args.cache_state,
               "mtp": args.mtp, "kv_policy": args.kv_policy}

    if args.engine == "ese":
        if args.mtp == "on":
            print(json.dumps({"unsupported": "ESE MTP", "toggles": toggles}, indent=2))
            return 2
        plan: dict[str, Any] = {"argv": build_ese_argv(spec, phase=args.phase,
                                                       kv_policy=args.kv_policy, mtp=args.mtp)}
        cache_files = spec["ese"]["cache_files"]
    else:
        try:
            plan = build_strata_plan(spec, phase=args.phase, kv_policy=args.kv_policy, mtp=args.mtp)
        except Unsupported as error:
            print(json.dumps({"unsupported": str(error), "toggles": toggles}, indent=2))
            return 2
        cache_files = spec["strata"]["cache_files"]

    bundle: dict[str, Any] = {
        "schema_version": 1,
        "date": dt.date.today().isoformat(),
        "architecture": "qwen4exp",
        "workload_sha256": spec["workload_sha256"],
        "engine": args.engine,
        "toggles": toggles,
        "plan": plan,
        "executed": bool(args.execute),
    }

    if not args.execute:
        bundle["scope"] = {"diagnostic_only": True, "parity_qualified": False,
                           "note": "dry plan only; no engine was run"}
        print(json.dumps(bundle, indent=2))
        return 0

    if args.output is None:
        print(json.dumps({"refused": "--execute requires --output for durable evidence"}, indent=2))
        return 2
    if args.engine == "ese" and not args.allow_probe_gate:
        print(json.dumps({"refused": "ESE execution requires --allow-probe-gate"}, indent=2))
        return 2
    if args.timeout <= 0:
        print(json.dumps({"refused": "--timeout must be positive"}, indent=2))
        return 2
    if args.output.exists():
        print(json.dumps({"refused": f"evidence already exists: {args.output}; use a new output path"}, indent=2))
        return 2

    if args.cache_state == "cold":
        bundle["cache_state"] = drop_os_cache(cache_files)
    else:
        bundle["cache_state"] = {"method": "none", "physical_device_cache_claim": False,
                                 "note": "warm: OS page cache left as-is"}

    try:
        outcome = (run_ese(spec, plan["argv"], allow_probe_gate=args.allow_probe_gate, timeout=args.timeout)
                   if args.engine == "ese"
                   else run_strata(plan, output_dir=args.output.parent if args.output else None,
                                   timeout=args.timeout))
    except Unsupported as error:
        print(json.dumps({"refused": str(error), "toggles": toggles}, indent=2))
        return 2

    stdout_path = stderr_path = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = args.output.with_suffix(args.output.suffix + ".stdout.log")
        stderr_path = args.output.with_suffix(args.output.suffix + ".stderr.log")
        stdout_raw = outcome.get("stdout_bytes", outcome["stdout"].encode("utf-8"))
        stderr_raw = outcome.get("stderr_bytes", outcome["stderr"].encode("utf-8"))
        stdout_path.write_bytes(stdout_raw)
        stderr_path.write_bytes(stderr_raw)

    metrics = (parse_llama_metrics(outcome["stdout"] + outcome["stderr"]) if args.engine == "ese"
               else parse_strata_metrics(outcome["stdout"]))
    result: dict[str, Any] = {
        "exit_code": outcome["exit_code"],
        "wall_ms": outcome["wall_ms"],
        "stdout_path": str(stdout_path) if stdout_path else None,
        "stderr_path": str(stderr_path) if stderr_path else None,
        "metrics": metrics,
    }
    if stdout_path is not None:
        result["stdout_sha256"] = hashlib.sha256(stdout_path.read_bytes()).hexdigest()
        result["stderr_sha256"] = hashlib.sha256(stderr_path.read_bytes()).hexdigest()
    for extra in ("timed_out", "ready", "health", "engine_log_path", "engine_config_path", "server_log_path"):
        if extra in outcome:
            result[extra] = outcome[extra]
    scope_note = ("ESE Qwen4Exp logits are not parity-verified; treat as a resource/timing baseline only"
                  if args.engine == "ese" else
                  "reference-engine timing only; the ESE side of this workload is not output-comparable yet")
    bundle.update({
        "env": outcome["env"],
        "result": result,
        "scope": {
            "diagnostic_only": True,
            "parity_qualified": False,
            "input_mode": "raw_text" if args.engine == "ese" else "templated_chat",
            "cross_engine_input_equivalence_verified": False,
            "note": scope_note,
        },
    })
    bundle["evidence_sha256"] = canonical_hash(bundle)

    if args.output:
        args.output.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(bundle, indent=2))
    return 0 if outcome["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
