import json
from pathlib import Path

import numpy as np
import pytest

from tools import qwen4exp_layer0_real as real

import torch

EVIDENCE = Path(__file__).resolve().parents[1] / "docs/qwen4exp/native-recurrence-20261007"


def test_capture_reader_refuses_missing_duplicate_nonfinite_and_wrong_shapes(tmp_path):
    p = tmp_path / "capture.log"
    def write(records):
        p.write_text("\n".join("qwen4exp_float_values: " + json.dumps(record) for record in records))
    inp = {"name": "qwen4exp_attn_input-0", "values": [0.] * real.HIDDEN}
    out = {"name": "qwen4exp_attn_output-0", "values": [0.] * real.HIDDEN}
    write([inp, out])
    tensors, provenance = real.read_capture(p)
    assert provenance["tokens"] == 1 and len(tensors) == 2
    for records, message in (([inp], "must include"), ([inp, out, inp], "duplicate"),
                             ([inp, {**out, "values": [float("nan")]}], "invalid captured"),
                             ([inp, {**out, "values": [0.]}], "violates bounded")):
        write(records)
        with pytest.raises(ValueError, match=message):
            real.read_capture(p)


def test_captured_artifact_binds_log_and_measures_failures_without_waiving_tolerance():
    fixture = json.loads((EVIDENCE / "layer0-captured-fixture.json").read_text())
    assert fixture["fixture_sha256"] == real.canonical_hash({k: v for k, v in fixture.items() if k != "fixture_sha256"})
    captured, provenance = real.read_capture(EVIDENCE / "fullgraph-layer0-qualified.stdout.log")
    assert provenance["sha256"] == fixture["capture"]["sha256"]
    assert fixture["geometry"]["tokens"] == provenance["tokens"] == 1
    np.testing.assert_array_equal(captured["qwen4exp_attn_input-0"], fixture["input"])
    output_error = np.max(np.abs(captured["qwen4exp_attn_output-0"] - np.asarray(fixture["expected"]["output"], dtype=np.float32)))
    assert output_error == fixture["captured_graph_comparison"]["output"]["max_abs"]
    for comparison in fixture["captured_graph_comparison"].values():
        assert comparison["passes_2e_5"] == (comparison["max_abs"] < 2e-5)
    assert "does not establish" in fixture["scope"]
    assert fixture["captured_input_a_roundtrip_gate_max_abs"] < 2e-5


def test_repaired_graph_recurrence_and_real_gated_residual_pass_strict_tolerance():
    fixture = json.loads((EVIDENCE / "layer0-gr-precision-fixture.json").read_text())
    assert fixture["fixture_sha256"] == real.canonical_hash({k: v for k, v in fixture.items() if k != "fixture_sha256"})
    captured, provenance = real.read_capture(EVIDENCE / "precision-gr.stdout.log")
    assert fixture["capture"]["sha256"] == provenance["sha256"]
    assert len(fixture["captured_graph_comparison"]) == 7
    for comparison in fixture["captured_graph_comparison"].values():
        assert comparison["max_abs"] < 2e-5
        assert comparison["passes_2e_5"]
    gr = fixture["gated_residual_comparison"]
    assert set(gr["comparisons"]) == {"qwen4exp_attn_input-0", "qwen4exp_attn_residual-0", "qwen4exp_ffn_input-0"}
    assert len(gr["tensor_payload_sha256"]) == 8
    for comparison in gr["comparisons"].values():
        assert comparison["max_abs"] < 2e-5
        assert comparison["passes_2e_5"]
    np.testing.assert_array_equal(captured["qwen4exp_attn_input-0"], fixture["input"])


def test_cyclic_order_and_invalid_geometry():
    assert real.cyclic_order(2, 6).tolist() == [0, 3, 1, 4, 2, 5]
    for key_heads, heads in ((0, 6), (2, 5), (2, 0)):
        with pytest.raises(ValueError, match="invalid grouped-head geometry"):
            real.cyclic_order(key_heads, heads)


def test_weight_canonicalization_is_inverse(monkeypatch):
    for name, value in (("HEADS", 6), ("KEY_HEADS", 2), ("WIDTH", 2)):
        monkeypatch.setattr(real, name, value)
    order = real.cyclic_order(2, 6)
    canonical = np.arange(24, dtype=np.float32).reshape(12, 2)
    stored = canonical.reshape(6, 2, 2)[order].reshape(12, 2)
    np.testing.assert_array_equal(real.canonicalize("in_proj_z", stored, order), canonical)
    prefix = np.arange(16, dtype=np.float32).reshape(8, 2)
    np.testing.assert_array_equal(real.canonicalize("in_proj_qkv", np.concatenate((prefix, stored)), order),
                                  np.concatenate((prefix, canonical)))
    columns = canonical.T.copy()
    stored_columns = columns.reshape(2, 6, 2)[:, order].reshape(2, 12)
    np.testing.assert_array_equal(real.canonicalize("out_proj", stored_columns, order), columns)
    np.testing.assert_array_equal(real.canonicalize("norm_weight", prefix, order), prefix)


def test_tokens_are_bounded_before_io():
    for tokens in (0, -1, 9, 100000):
        with pytest.raises(ValueError, match="tokens must be"):
            real.build(Path("does-not-exist"), "does-not-exist", tokens)


def test_cache_preserves_history_and_copies_recurrent_state():
    cache = real.LayerCache()
    assert not cache.has_previous_state(0)
    first = torch.arange(2, dtype=torch.float32).reshape(1, 1, 2)
    cache.update_conv_state(first, 0, real.KERNEL)
    torch.testing.assert_close(cache.layers[0].conv_states[0], torch.tensor([[[0., 0., 1.]]]))
    second = torch.tensor([[[2.]]])
    joined = cache.update_conv_state(second, 0, real.KERNEL)
    torch.testing.assert_close(joined, torch.tensor([[[0., 0., 1., 2.]]]))
    torch.testing.assert_close(cache.layers[0].conv_states[0], torch.tensor([[[0., 1., 2.]]]))
    state = torch.tensor([1.])
    cache.update_recurrent_state(state, 0)
    state.zero_()
    assert cache.has_previous_state(0)
    torch.testing.assert_close(cache.layers[0].recurrent_states[0], torch.tensor([1.]))
    with pytest.raises(ValueError, match="kernel disagrees"):
        cache.update_conv_state(second, 0, 99)


def test_decode_fixture_integrity_shapes_and_batch_consistency():
    fixture = json.loads((EVIDENCE / "layer0-real-decode-fixture.json").read_text())
    assert fixture["schema_version"] == 2
    assert fixture["fixture_sha256"] == real.canonical_hash({k: v for k, v in fixture.items() if k != "fixture_sha256"})
    assert fixture["expected_layout"] == "cyclic_gguf"
    assert "persistent-cache" in fixture["decode_contract"]
    tokens = fixture["geometry"]["tokens"]
    assert tokens > fixture["geometry"]["kernel"], "fixture must exercise convolution-history rollover"
    assert len(fixture["steps"]) == tokens
    batch = fixture["expected"]
    for step in fixture["steps"]:
        assert step.keys() == batch.keys()
        for name, data in step.items():
            expected_size = len(batch[name]) if name in ("state", "conv_state") else len(batch[name]) // tokens
            assert len(data) == expected_size, name
            assert np.isfinite(data).all(), name
    output_error = np.max(np.abs(np.array(batch["output"]) - np.concatenate([step["output"] for step in fixture["steps"]])))
    state_error = np.max(np.abs(np.array(batch["state"]) - fixture["steps"][-1]["state"]))
    conv_error = np.max(np.abs(np.array(batch["conv_state"]) - fixture["steps"][-1]["conv_state"]))
    measured = dict(output_max_abs=output_error, state_max_abs=state_error, conv_state_max_abs=conv_error)
    for name, error in measured.items():
        assert error < 2e-5
        assert error == fixture["batch_decode_consistency"][name]
    # Retained snapshots must not alias the cache mutated by later tokens.
    assert not np.array_equal(fixture["steps"][0]["conv_state"], fixture["steps"][-1]["conv_state"])
    assert not np.array_equal(fixture["steps"][0]["state"], fixture["steps"][-1]["state"])


def test_saved_real_fixture_integrity_and_scope():
    path = EVIDENCE / "layer0-real-fixture.json"
    fixture = json.loads(path.read_text())
    assert fixture["fixture_sha256"] == real.canonical_hash({k: v for k, v in fixture.items() if k != "fixture_sha256"})
    assert fixture["expected_layout"] == "cyclic_gguf"
    assert fixture["repeat_type"] == 1
    assert fixture["reference"]["modeling_sha256"] == real.TRANSFORMERS_SHA256
    assert fixture["a_exp_log_roundtrip_max_abs"] < 4e-6
    assert fixture["identity_order_control"]["output_max_abs_delta"] > 0.4
    assert "synthetic input" in fixture["scope"]
    tokens = fixture["geometry"]["tokens"]
    sizes = {"output": tokens * real.HIDDEN, "state": real.HEADS * real.WIDTH ** 2,
             "conv_state": real.CHANNELS * (real.KERNEL - 1), "in_proj_qkv": tokens * real.CHANNELS,
             "core_output": tokens * real.HEADS * real.WIDTH}
    for name, size in sizes.items():
        assert len(fixture["expected"][name]) == size
        assert np.isfinite(fixture["expected"][name]).all()
    assert len(fixture["tensors"]) == 9
    for record in fixture["tensors"].values():
        assert len(record["payload_sha256"]) == 64
