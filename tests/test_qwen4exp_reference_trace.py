import hashlib
import json
from pathlib import Path

from tools.qwen4exp_reference_trace import verify_trace_hash


REPO_ROOT = Path(__file__).resolve().parents[1]
TRACE = REPO_ROOT / "docs" / "qwen4exp" / "ple-reference-trace.json"
ROW_FIXTURE = REPO_ROOT / "docs" / "qwen4exp" / "ple-iq4nl-row-fixture.json"
WARM_BASELINE = REPO_ROOT / "docs" / "qwen4exp" / "ple-storage-warm-baseline.json"
COLD_BASELINE = REPO_ROOT / "docs" / "qwen4exp" / "ple-storage-cold-baseline.json"
BUDGET_MATRIX = REPO_ROOT / "docs" / "qwen4exp" / "ple-storage-budget-matrix.json"
PRESSURE_EVIDENCE = REPO_ROOT / "docs" / "qwen4exp" / "ple-storage-pressure.json"
OPERATOR_FIXTURE = REPO_ROOT / "docs" / "qwen4exp" / "operator-reference-fixture.json"


def test_qwen4exp_operator_fixture_is_content_addressed_and_pinned() -> None:
    fixture = json.loads(OPERATOR_FIXTURE.read_text(encoding="utf-8"))
    expected = fixture.pop("fixture_sha256")
    encoded = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert fixture["architecture"] == "qwen4exp"
    assert fixture["reference"]["revision"] == "fc5c5bde8e656dad91cbf34e61940d984b1c7b91"
    assert fixture["reference"]["modeling_sha256"] == "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"
    assert set(fixture["outputs"]) == {
        "gated_residual_mixed", "gated_residual_injection", "gated_residual_output",
        "gated_residual_final_mixed", "ple_gated_value", "ple_gated_value_normed",
        "ple_output", "ple_next_state",
    }
    qsa = fixture["qsa"]
    assert qsa["identity_rope"] is True
    assert len(qsa["per_query"]) == qsa["queries"] == 2
    for per_query in qsa["per_query"]:
        assert len(per_query["block_order"]) == qsa["top_blocks"] == 2
        assert len(per_query["selected_tokens"]) == qsa["top_blocks"] * qsa["compress_ratio"] + 1
        assert len(per_query["mask_row"]) == qsa["kv_length"] == 8
        assert len(per_query["scores"]) <= qsa["blocks"]

    # A5 expert routing golden: the pinned router's expert IDs and weights.
    moe = fixture["moe"]
    assert moe["num_experts"] == 6 and moe["top_k"] == 2 and moe["norm_topk_prob"] is True
    assert len(moe["router_logits"]) == moe["tokens"] * moe["num_experts"]
    assert len(moe["expert_ids"]) == len(moe["routing_weights"]) == moe["tokens"]
    for ids, weights in zip(moe["expert_ids"], moe["routing_weights"]):
        assert len(ids) == len(weights) == moe["top_k"]
        assert len(set(ids)) == moe["top_k"]
        assert all(0 <= i < moe["num_experts"] for i in ids)
        assert abs(sum(weights) - 1.0) < 1e-5
    assert len(moe["moe_output"]) == moe["tokens"] * moe["hidden"]

    # A5 recurrent-state golden: the pinned Gated DeltaNet recurrence.
    recurrent = fixture["recurrent"]
    assert recurrent["use_qk_l2norm"] is True
    state_size = recurrent["heads"] * recurrent["key_head_dim"] * recurrent["value_head_dim"]
    assert len(recurrent["chunked_final_state"]) == state_size
    assert len(recurrent["recurrent_final_state"]) == state_size
    assert len(recurrent["chunked_output"]) == (
        recurrent["sequence_length"] * recurrent["heads"] * recurrent["value_head_dim"]
    )
    assert len(recurrent["recurrent_output"]) == len(recurrent["chunked_output"])

    native = fixture["recurrent_native"]
    assert native["key_head_dim"] == native["value_head_dim"] == 128
    assert native["heads"] == 2 and native["sequence_length"] == 5
    assert native["state_layout"] == "head,key,value"
    assert len(native["single_step_states"]) == native["sequence_length"]
    assert all(len(state) == 2 * 128 * 128 for state in native["single_step_states"])
    assert max(abs(a - b) for a, b in zip(native["single_step_output"], native["recurrent_output"])) < 2e-5
    assert max(abs(a - b) for a, b in zip(native["single_step_states"][-1], native["recurrent_final_state"])) < 2e-5
    assert max(abs(a - b) for a, b in zip(native["chunked_output"], native["recurrent_output"])) < 2e-5

    grouped = fixture["recurrent_grouped"]
    assert grouped["heads"] == 6 and grouped["key_heads"] == 2
    assert grouped["repeat_type"] == 0 and grouped["head_expansion"] == "repeat_interleave"
    assert len(grouped["query"]) == 5 * 2 * 128
    assert len(grouped["value"]) == 5 * 6 * 128
    assert all(len(state) == 6 * 128 * 128 for state in grouped["single_step_states"])
    assert max(abs(a - b) for a, b in zip(grouped["single_step_output"], grouped["recurrent_output"])) < 2e-5
    assert max(abs(a - b) for a, b in zip(grouped["chunked_output"], grouped["recurrent_output"])) < 2e-5

    for name, width, heads in (("recurrent_scalar", 32, 2),
                               ("recurrent_scalar_low_magnitude", 32, 6),
                               ("recurrent_low_magnitude", 128, 2)):
        recurrence = fixture[name]
        assert recurrence["key_head_dim"] == recurrence["value_head_dim"] == width
        assert recurrence["heads"] == heads and recurrence["sequence_length"] == 5
        assert len(recurrence["single_step_states"]) == 5
        assert all(len(state) == heads * width * width for state in recurrence["single_step_states"])
        for output in ("single_step_output", "chunked_output"):
            assert len(recurrence[output]) == len(recurrence["recurrent_output"])
            assert max(abs(a - b) for a, b in zip(recurrence[output], recurrence["recurrent_output"])) < 2e-5
        assert max(abs(a - b) for a, b in zip(recurrence["single_step_states"][-1], recurrence["recurrent_final_state"])) < 2e-5
        if "low_magnitude" in name:
            row = width * recurrence.get("key_heads", heads)
            assert recurrence["query"][:row] == [0.0] * row
            assert 0 < max(abs(x) for x in recurrence["query"][row:2 * row]) < 1e-5
            assert max(abs(x) for x in recurrence["query"][3 * row:4 * row]) > 0.1

    l2 = fixture["recurrent_l2"]
    assert l2["width"] == 128 and l2["rows"] == 3 and l2["epsilon"] == 1e-6
    assert len(l2["input"]) == len(l2["output"]) == 384
    assert l2["output"][:128] == [0.0] * 128
    assert max(abs(x) for x in l2["output"][128:256]) < 0.05

    # A5 per-token logits golden: the pinned output-head path.
    logits = fixture["logits"]
    assert logits["final_norm_applied"] is False
    assert len(logits["logits"]) == logits["tokens"] * logits["vocab"]
    assert len(logits["lm_head_weight"]) == logits["vocab"] * logits["hidden"]
    assert len(logits["hidden_input"]) == logits["tokens"] * logits["hidden"]


def test_complete_recurrent_layer_batch_matches_cached_steps() -> None:
    fixture = json.loads(OPERATOR_FIXTURE.read_text(encoding="utf-8"))
    layer = fixture["recurrent_layer"]
    config_bytes = (REPO_ROOT / "docs/qwen4exp/pinned-model-config.json").read_bytes()
    assert hashlib.sha256(config_bytes).hexdigest() == "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
    configured_gate = json.loads(config_bytes)["text_config"]["output_gate_type"]
    assert configured_gate == "sigmoid"
    assert layer["output_gate_type"] == configured_gate
    assert fixture["recurrent_layer_gguf"]["output_gate_type"] == configured_gate
    qsa = fixture["qsa_rotary"]
    assert (qsa["width"], qsa["rotary_dims"], qsa["heads"]) == (128, 64, 4)
    assert qsa["full_width_control_max_abs"] > 1e-3
    for i, (before, after) in enumerate(zip(qsa["input"], qsa["expected"])):
        if i % qsa["width"] >= qsa["rotary_dims"]:
            assert before == after
    assert (layer["hidden"], layer["width"], layer["key_heads"], layer["heads"], layer["tokens"], layer["kernel"]) == (8, 128, 2, 6, 5, 4)
    assert len(layer["steps"]) == layer["tokens"]
    channels = layer["width"] * (2 * layer["key_heads"] + layer["heads"])
    expected_widths = {"in_proj_qkv": channels, "in_proj_z": 128 * 6,
                       "in_proj_a": 6, "in_proj_b": 6, "convolved": channels,
                       "query": 128 * 6, "key": 128 * 6, "value": 128 * 6,
                       "g": 6, "beta": 6, "core_output": 128 * 6,
                       "gated_output": 128 * 6, "output": 8}
    for name, width in expected_widths.items():
        batched = layer["batched"][name]
        steps = [value for step in layer["steps"] for value in step[name]]
        assert len(batched) == len(steps) == width * layer["tokens"]
        assert max(abs(a - b) for a, b in zip(batched, steps)) < 2e-5, name
    for name, size in (("state", 128 * 128 * 6), ("conv_state", channels * 3)):
        batched, final = layer["batched"][name], layer["steps"][-1][name]
        assert len(batched) == len(final) == size
        assert max(abs(a - b) for a, b in zip(batched, final)) < 2e-5, name
    assert any(abs(x) > 1e-4 for x in layer["batched"]["output"])
    assert any(abs(x) > 1e-4 for x in layer["steps"][0]["state"])


def test_gguf_value_head_permutation_preserves_complete_layer_output() -> None:
    fixture = json.loads(OPERATOR_FIXTURE.read_text(encoding="utf-8"))
    source, converted = fixture["recurrent_layer"], fixture["recurrent_layer_gguf"]
    assert converted["repeat_type"] == 1 and converted["head_order"] == "cyclic_gguf"
    assert converted["source_head_indices"] == [0, 3, 1, 4, 2, 5]
    assert source["batched"]["output"] == converted["batched"]["output"]
    assert source["weights"]["in_proj_a"] != converted["weights"]["in_proj_a"]
    for a, b in zip(source["steps"], converted["steps"]):
        assert a["output"] == b["output"]
    assert source["norm_weight"] == converted["norm_weight"]


def test_qwen4exp_ple_reference_trace_is_content_addressed() -> None:
    trace = json.loads(TRACE.read_text(encoding="utf-8"))
    verify_trace_hash(trace)
    assert trace["architecture"] == "qwen4exp"
    assert trace["ple"]["eos_token_id"] == 248044
    assert len(trace["ple"]["head_vocab_sizes"]) == 16
    for sequence in trace["sequences"]:
        assert len(sequence["token_ids"]) == len(sequence["row_ids"])
        assert all(len(row) == 16 for row in sequence["row_ids"])


def test_qwen4exp_ple_row_fixture_is_content_addressed() -> None:
    fixture = json.loads(ROW_FIXTURE.read_text(encoding="utf-8"))
    expected = fixture.pop("fixture_sha256")
    encoded = json.dumps(fixture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    quantized = bytes.fromhex(fixture["quantized_hex"])
    assert len(quantized) == 90
    assert hashlib.sha256(quantized).hexdigest() == fixture["quantized_sha256"]
    assert len(fixture["dequantized_values"]) == fixture["value_count"] == 160


def test_qwen4exp_ple_warm_baseline_is_content_addressed_and_honest() -> None:
    evidence = json.loads(WARM_BASELINE.read_text(encoding="utf-8"))
    expected = evidence.pop("evidence_sha256")
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert evidence["cache_state"]["cold_cache_claim"] is False
    assert {result["backend"] for result in evidence["results"]} == {"pread", "mmap"}
    assert all(result["reads"]["failed_reads"] == 0 for result in evidence["results"])


def test_qwen4exp_ple_cold_baseline_is_content_addressed_and_scoped() -> None:
    evidence = json.loads(COLD_BASELINE.read_text(encoding="utf-8"))
    expected = evidence.pop("evidence_sha256")
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert evidence["cache_state"]["cold_cache_control_succeeded"] is True
    assert evidence["cache_state"]["physical_device_cache_claim"] is False
    assert evidence["configuration"]["os_cache_control"] == "posix_fadvise_dontneed"
    assert evidence["configuration"]["backend"] == "io_uring"
    assert evidence["result"]["reads"]["failed_reads"] == 0
    assert evidence["result"]["reads"]["staging_resident_bytes"] == 0
    assert evidence["result"]["reads"]["source_batches"] == 16
    assert evidence["result"]["reads"]["peak_source_batch_width"] == 17


def test_qwen4exp_ple_budget_matrix_is_content_addressed_and_complete() -> None:
    evidence = json.loads(BUDGET_MATRIX.read_text(encoding="utf-8"))
    expected = evidence.pop("evidence_sha256")
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert evidence["scope"]["budget_pressure_claim"] is False
    results = evidence["results"]
    assert len(results) == 36
    assert {result["backend"] for result in results} == {"pread", "mmap", "io_uring"}
    assert {result["storage_mode"] for result in results} == {"row", "page", "two-tier"}
    assert {result["cache_mib"] for result in results} == {256, 512, 1024, 2048}
    for result in results:
        budget = result["cache_mib"] * 1024 * 1024
        assert result["reads"]["failed_reads"] == 0
        assert result["reads"]["staging_resident_bytes"] == 0
        assert result["reads"]["source_batches"] > 0
        assert result["reads"]["peak_source_batch_width"] > 1
        if result["storage_mode"] == "row":
            assert budget - 90 < result["row_cache"]["capacity_bytes"] <= budget
        elif result["storage_mode"] == "page":
            assert result["page_cache"]["capacity_bytes"] == budget
        else:
            combined = (
                result["row_cache"]["capacity_bytes"]
                + result["page_cache"]["capacity_bytes"]
            )
            assert budget - 90 < combined <= budget


def test_qwen4exp_ple_pressure_evidence_is_bounded_and_failure_free() -> None:
    evidence = json.loads(PRESSURE_EVIDENCE.read_text(encoding="utf-8"))
    expected = evidence.pop("evidence_sha256")
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert evidence["workload"] == "deterministic-pressure"
    assert evidence["backend"] == "io_uring"
    assert evidence["storage_mode"] == "two-tier"
    assert evidence["cache_state"] == "cold"
    assert evidence["cache_mib"] == 256
    assert evidence["sessions"] == 4
    assert evidence["pressure_tokens_per_session"] == 4096
    assert evidence["tokens"] == 16384
    assert evidence["reads"]["failed_reads"] == 0
    assert evidence["reads"]["staging_rejected_batches"] == 0
    assert evidence["reads"]["staging_resident_bytes"] == 0
    assert evidence["reads"]["staging_peak_resident_bytes"] <= evidence["reads"]["staging_capacity_bytes"]
    for cache_name in ("page_cache", "row_cache"):
        cache = evidence[cache_name]
        assert cache["resident_bytes"] <= cache["capacity_bytes"]
        assert cache["peak_resident_bytes"] <= cache["capacity_bytes"]
        assert cache["pinned_capacity_failures"] == 0
    assert evidence["page_cache"]["evictions"] > 0
