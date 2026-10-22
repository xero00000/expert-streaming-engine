import copy
import importlib.util
import json
import os
import subprocess
import socket
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "tools" / "qwen4exp_benchmark_replay.py"


def _harness():
    spec = importlib.util.spec_from_file_location("qwen4exp_benchmark_replay", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def strata_spec(tmp_path_factory):
    """Replay planning tests use an owned config and never require a local service."""
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    tmp_path = tmp_path_factory.mktemp("strata-input")
    config = tmp_path / "strata-config.json"
    config.write_text(json.dumps({
        "exe": str(tmp_path / "strata"),
        "cwd": str(tmp_path),
        "model_name": "replay-fixture",
        "args": [
            "--pack", str(tmp_path / "pack"),
            "--native", str(tmp_path / "model.gguf"),
            "--prefill", str(spec["strata"]["prefill_chunk"]),
            "--spec", str(spec["strata"]["spec_depth"]),
            "--mtp", str(tmp_path / "mtp"),
            "--max-context", str(spec["workload"]["context_length"]),
            "--kv", spec["workload"]["kv_precision_strata"],
            "--kv-resident", str(spec["strata"]["kv_resident_cells"]),
        ],
    }))
    spec["strata"]["config"] = str(config)
    return spec


def test_workload_spec_is_content_addressed_and_canonical() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    assert spec["architecture"] == "qwen4exp"
    assert spec["workload"]["prompt_token_ids"] == [12675]
    assert spec["workload"]["context_length"] == 131072
    assert set(spec["toggles"]) == {"phase", "cache_state", "mtp", "kv_policy"}


def test_ese_kv_policy_toggle_changes_only_one_option() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    resident = harness.build_ese_argv(spec, phase="decode", kv_policy="resident", mtp="off")
    host = harness.build_ese_argv(spec, phase="decode", kv_policy="host", mtp="off")
    assert host == resident + ["-nkvo"]
    assert "-ctk" in resident and resident[resident.index("-ctk") + 1] == "q8_0"


def test_ese_phase_toggle_changes_only_predict_count() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    prefill = harness.build_ese_argv(spec, phase="prefill", kv_policy="host", mtp="off")
    decode = harness.build_ese_argv(spec, phase="decode", kv_policy="host", mtp="off")
    assert prefill.index("-n") == decode.index("-n")
    predict_index = prefill.index("-n") + 1
    assert prefill[predict_index] == "1"
    assert decode[predict_index] == "128"
    assert len(prefill) == len(decode)
    for index, (a, b) in enumerate(zip(prefill, decode)):
        if index != predict_index:
            assert a == b, f"phase toggle changed more than -n at index {index}"


def test_ese_moe_placement_is_one_explicit_option() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    assert spec["ese"]["moe_placement"] == "cpu"
    argv = harness.build_ese_argv(spec, phase="decode", kv_policy="host", mtp="off")
    assert argv.count("--cpu-moe") == 1

    gpu_spec = copy.deepcopy(spec)
    gpu_spec["ese"]["moe_placement"] = "gpu"
    gpu = harness.build_ese_argv(gpu_spec, phase="decode", kv_policy="host", mtp="off")
    assert "--cpu-moe" not in gpu
    assert [arg for arg in argv if arg != "--cpu-moe"] == gpu
    assert len(argv) == len(gpu) + 1

    bad_spec = copy.deepcopy(spec)
    bad_spec["ese"]["moe_placement"] = "auto"
    with pytest.raises(harness.Unsupported):
        harness.build_ese_argv(bad_spec, phase="decode", kv_policy="host", mtp="off")


def test_ese_mtp_is_reported_unsupported() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    assert spec["ese"]["supports_mtp"] is False
    with pytest.raises(harness.Unsupported):
        harness.build_ese_argv(spec, phase="decode", kv_policy="host", mtp="on")


def test_strata_kv_policy_and_mtp_toggles_strip_exact_options(strata_spec) -> None:
    harness = _harness()
    spec = strata_spec
    resident = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="on")["engine_args"]
    assert resident.count("--kv-resident") == 1 and str(spec["strata"]["kv_resident_cells"]) in resident
    with pytest.raises(harness.Unsupported, match="fully GPU-resident"):
        harness.build_strata_plan(spec, phase="decode", kv_policy="host", mtp="on")

    mtp_off = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="off")["engine_args"]
    assert "--spec" not in mtp_off and "--mtp" not in mtp_off
    assert len(resident) == len(mtp_off) + 4


def test_strata_generated_config_carries_the_toggles_and_leaves_the_source_alone(strata_spec) -> None:
    """strata_service.sh always reads the on-disk config, so the harness must launch
    serve/server.py with a derived config for a toggle to mean anything."""
    harness = _harness()
    spec = strata_spec
    source = Path(spec["strata"]["config"])
    before = source.read_bytes()

    plan = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="off")
    assert plan["runtime"]["python"] == spec["strata"]["python"]
    assert plan["runtime"]["server"] == spec["strata"]["server"]
    assert plan["runtime"]["port"] == spec["strata"]["benchmark_port"]
    assert plan["request_url"].endswith(f":{plan['runtime']['port']}/v1/chat/completions")

    config = harness.strata_engine_config(plan, log_path=Path("bundle/strata-engine.log"))
    assert config["args"] == plan["engine_args"]
    for stripped in ("--spec", "--mtp"):
        assert stripped not in config["args"]
    assert "--kv-resident" in config["args"]
    assert source.read_bytes() == before

    full = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="on")
    full_config = harness.strata_engine_config(full)
    assert full_config["args"] == list(json.loads(before)["args"])
    assert "--kv-resident" in full_config["args"]


def test_strata_run_paths_are_absolute(tmp_path, monkeypatch, strata_spec) -> None:
    """serve/server.py runs with the engine's cwd, so a relative --config path fails."""
    harness = _harness()
    spec = strata_spec
    plan = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="off")
    monkeypatch.chdir(tmp_path)
    paths = harness.prepare_strata_run(plan, Path("docs/qwen4exp"))
    assert paths["config_path"].is_absolute() and paths["log_path"].is_absolute()
    written = json.loads(paths["config_path"].read_text())
    assert written["args"] == plan["engine_args"]
    assert written["log"] == str(paths["log_path"])
    argv = harness.strata_server_argv(plan, paths["config_path"])
    assert argv[:3] == [spec["strata"]["python"], spec["strata"]["server"], "--engine"]
    assert argv[argv.index("--config") + 1] == str(paths["config_path"])
    assert argv[argv.index("--port") + 1] == str(spec["strata"]["benchmark_port"])


def test_strata_metrics_keep_usage_and_timings() -> None:
    harness = _harness()
    payload = json.dumps({"usage": {"completion_tokens": 128},
                          "timings": {"predicted_ms": 1234.5, "predicted_n": 128}})
    metrics = harness.parse_strata_metrics(payload)
    assert metrics["usage"]["completion_tokens"] == 128
    assert metrics["timings"]["predicted_ms"] == 1234.5
    assert harness.parse_strata_metrics("not json") == {}
    assert harness.parse_strata_metrics(None) == {}


def test_metrics_parser_accepts_real_llama_cli_token_wording() -> None:
    """Current llama-cli prints `/ N tokens` for decode, not `/ N runs`."""
    harness = _harness()
    text = (
        "main: prompt eval time =     479.96 ms /     1 tokens (  479.96 ms per token,     2.08 tokens per second)\n"
        "main:        eval time =   15996.29 ms /   128 tokens (  124.97 ms per token,     8.00 tokens per second)\n"
        "main:       total time =   16476.26 ms /   129 tokens\n"
    )
    metrics = harness.parse_llama_metrics(text)
    assert metrics["prompt_tokens_per_s"] == pytest.approx(2.08)
    assert metrics["decode_tokens"] == 128
    assert metrics["decode_tokens_per_s"] == pytest.approx(8.00)
    assert metrics["ms_per_token"] == pytest.approx(124.97, rel=1e-3)


def test_metrics_parser_does_not_confuse_prompt_and_decode() -> None:
    harness = _harness()
    text = (
        "llama_perf_context_print:        prompt eval time =     702.18 ms /     1 tokens "
        "(  702.18 ms per token,     1.42 tokens per second)\n"
        "llama_perf_context_print:        eval time =   22138.80 ms /   128 runs   "
        "(  172.96 ms per token,     5.78 tokens per second)\n"
    )
    metrics = harness.parse_llama_metrics(text)
    assert metrics["prompt_eval_ms"] == pytest.approx(702.18)
    assert metrics["decode_ms"] == pytest.approx(22138.80)
    assert metrics["decode_tokens"] == 128
    assert metrics["decode_tokens_per_s"] == pytest.approx(5.78)
    assert metrics["ms_per_token"] == pytest.approx(172.96, rel=1e-3)


def test_execute_tolerates_raw_non_utf8_model_output() -> None:
    """llama-cli prints decoded piece bytes, which are not always valid UTF-8."""
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    script = (
        "import sys; sys.stdout.buffer.write(b'\\xff\\xfe' + b'eval time =   22138.80 ms /' "
        "b'   128 runs   (  172.96 ms per token,     5.78 tokens per second)\\n')"
    )
    outcome = harness.run_ese(spec, [sys.executable, "-c", script], allow_probe_gate=True, timeout=30)
    assert outcome["exit_code"] == 0
    assert "\ufffd" in outcome["stdout"]
    assert harness.parse_llama_metrics(outcome["stdout"])["decode_tokens_per_s"] == pytest.approx(5.78)


def test_strata_replays_never_overwrite_run_config(tmp_path, strata_spec) -> None:
    harness = _harness()
    spec = strata_spec
    on = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="on")
    off = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="off")
    first = harness.prepare_strata_run(on, tmp_path)
    original = first["config_path"].read_bytes()
    second = harness.prepare_strata_run(off, tmp_path)
    assert first["output_dir"] != second["output_dir"]
    assert first["config_path"].read_bytes() == original
    assert "--mtp" in json.loads(original)["args"]
    assert "--mtp" not in json.loads(second["config_path"].read_bytes())["args"]


def test_ese_timeout_preserves_partial_raw_output_and_reaps_child() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    script = "import os,time; os.write(1,b'raw\\xff'); os.write(2,b'partial'); time.sleep(30)"
    started = time.monotonic()
    outcome = harness.run_ese(spec, [sys.executable, "-c", script], allow_probe_gate=True, timeout=0.2)
    assert time.monotonic() - started < 5
    assert outcome["exit_code"] == 124 and outcome["timed_out"] is True
    assert outcome["stdout_bytes"] == b"raw\xff"
    assert outcome["stderr_bytes"] == b"partial"


def test_owned_group_cleanup_after_leader_exits(tmp_path) -> None:
    harness = _harness()
    marker = tmp_path / "survived"
    script = (
        "import subprocess,sys; subprocess.Popen([sys.executable,'-c',"
        f"{('import time,pathlib; time.sleep(0.6); pathlib.Path(' + repr(str(marker)) + ').touch(); time.sleep(30)')!r}"
        "])"
    )
    proc = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    proc.wait(timeout=5)
    harness._terminate_session(proc)
    time.sleep(0.7)
    assert not marker.exists(), "descendant survived its exited group leader"


def test_strata_startup_failure_preserves_separate_server_and_engine_logs(tmp_path, strata_spec) -> None:
    harness = _harness()
    spec = strata_spec
    plan = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="off")
    server = tmp_path / "failing-server.py"
    server.write_text(
        "import json,sys,pathlib\n"
        "cfg=json.loads(pathlib.Path(sys.argv[sys.argv.index('--config')+1]).read_text())\n"
        "pathlib.Path(cfg['log']).write_bytes(b'engine evidence')\n"
        "print('server evidence',flush=True)\n"
        "sys.exit(7)\n"
    )
    plan["runtime"]["python"] = sys.executable
    plan["runtime"]["server"] = str(server)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    plan["runtime"]["port"] = port
    plan["health_url"] = f"http://127.0.0.1:{port}/health"
    plan["request_url"] = f"http://127.0.0.1:{port}/v1/chat/completions"
    outcome = harness.run_strata(plan, output_dir=tmp_path, timeout=5)
    assert outcome["exit_code"] == 1 and outcome["ready"] is False
    assert "code 7" in outcome["stderr"]
    assert Path(outcome["engine_log_path"]).read_bytes() == b"engine evidence"
    assert Path(outcome["server_log_path"]).read_bytes() == b"server evidence\n"
    assert outcome["engine_log_path"] != outcome["server_log_path"]


def test_strata_refuses_occupied_port_before_launch(tmp_path, monkeypatch, strata_spec) -> None:
    harness = _harness()
    spec = strata_spec
    plan = harness.build_strata_plan(spec, phase="decode", kv_policy="resident", mtp="off")
    def unexpected_launch(*args, **kwargs):
        pytest.fail("occupied-port refusal must happen before process launch")
    monkeypatch.setattr(harness.subprocess, "Popen", unexpected_launch)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        plan["runtime"]["port"] = listener.getsockname()[1]
        with pytest.raises(harness.Unsupported, match="refusing to use an existing service"):
            harness.run_strata(plan, output_dir=tmp_path, timeout=5)
    assert list(tmp_path.iterdir()) == []


def test_zero_duration_decode_is_not_throughput() -> None:
    harness = _harness()
    metrics = harness.parse_llama_metrics(
        "main: eval time = 0.00 ms / 1 tokens (0.00 ms per token, 1000000.00 tokens per second)"
    )
    assert metrics["decode_ms"] == 0
    assert metrics["decode_timing_available"] is False
    assert "decode_tokens_per_s" not in metrics


def test_cli_timeout_writes_verifiable_raw_evidence(tmp_path) -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    binary = tmp_path / "fixture-cli"
    binary.write_text(
        f"#!{sys.executable}\nimport os,time\nos.write(1,b'raw\\xff')\n"
        "os.write(2,b'partial')\ntime.sleep(30)\n"
    )
    binary.chmod(0o700)
    spec["ese"]["binary"] = str(binary)
    spec.pop("workload_sha256")
    spec["workload_sha256"] = harness.canonical_hash(spec)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    output = tmp_path / "evidence.json"
    argv = [sys.executable, str(HARNESS), "--spec", str(spec_path), "--execute",
            "--allow-probe-gate", "--timeout", "1", "--output", str(output)]
    completed = subprocess.run(argv, capture_output=True, timeout=10)
    assert completed.returncode == 1, completed.stderr
    bundle = json.loads(output.read_bytes())
    assert bundle["result"]["exit_code"] == 124
    raw = Path(bundle["result"]["stdout_path"]).read_bytes()
    assert raw == b"raw\xff"
    assert bundle["result"]["stdout_sha256"] == harness.hashlib.sha256(raw).hexdigest()
    expected = bundle.pop("evidence_sha256")
    assert harness.canonical_hash(bundle) == expected
    before = output.read_bytes()
    repeated = subprocess.run(argv, capture_output=True, timeout=5)
    assert repeated.returncode == 2
    assert output.read_bytes() == before


def test_execute_refuses_without_explicit_probe_gate() -> None:
    harness = _harness()
    spec = harness.load_spec(harness.DEFAULT_SPEC)
    with pytest.raises(harness.Unsupported):
        harness.run_ese(spec, ["/bin/true"], allow_probe_gate=False, timeout=5)
