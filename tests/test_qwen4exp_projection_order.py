import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.qwen4exp_projection_order import (
    _pinned_header,
    cyclic_source_order,
    locate_heads,
    stored_head_for_source,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/qwen4exp/native-recurrence-20261007"
PROBE_MANIFEST = EVIDENCE / "layer0-projection-probes.json"
REPORT = EVIDENCE / "projection-order-report.json"
HEAD_ORDER_REPORT = EVIDENCE / "head-order-report.json"


def rows_fixture(value_heads, key_heads, width, columns, probe_heads, expectation, seed=0, noise=1e-3):
    """Source window plus a stored tensor arranged with `expectation`."""
    rng = np.random.default_rng(seed)
    blocks = rng.normal(size=(value_heads, width, columns)).astype(np.float32)
    order = cyclic_source_order(key_heads, value_heads) if expectation == "cyclic" else np.arange(value_heads)
    stored = np.concatenate([blocks[order[s]] for s in range(value_heads)], axis=0)
    stored = stored + rng.normal(scale=noise, size=stored.shape).astype(np.float32)
    source = np.concatenate([blocks[h] for h in range(len(probe_heads))], axis=0)
    return source, stored


def columns_fixture(value_heads, key_heads, width, rows, expectation, seed=1, noise=1e-3):
    """Full source rows plus a stored tensor arranged with `expectation`."""
    rng = np.random.default_rng(seed)
    blocks = rng.normal(size=(value_heads, rows, width)).astype(np.float32)
    order = cyclic_source_order(key_heads, value_heads) if expectation == "cyclic" else np.arange(value_heads)
    stored = np.concatenate([blocks[order[s]] for s in range(value_heads)], axis=1)
    stored = stored + rng.normal(scale=noise, size=stored.shape).astype(np.float32)
    source = np.concatenate([blocks[h] for h in range(value_heads)], axis=1)
    return source, stored


def test_source_head_order_is_the_documented_transpose():
    assert cyclic_source_order(16, 48).tolist() == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45,
                                                    1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
                                                    2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47]
    assert [stored_head_for_source(h, 16, 48) for h in range(6)] == [0, 16, 32, 1, 17, 33]
    # Round trip, and refusal to invent a head that is not a value head.
    for head in range(48):
        assert cyclic_source_order(16, 48)[stored_head_for_source(head, 16, 48)] == head
    with pytest.raises(ValueError, match="invalid grouped-head geometry"):
        cyclic_source_order(7, 48)
    with pytest.raises(ValueError, match="outside value-head range"):
        stored_head_for_source(48, 16, 48)


def test_locate_heads_separates_cyclic_from_identity():
    source, stored = rows_fixture(6, 2, 4, 3, [0, 1, 2], "cyclic")
    located = locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic")
    assert located["all_forward_located"] and located["all_reverse_located"]
    assert [r["expected_stored_head"] for r in located["forward"]] == [0, 2, 4]
    assert all(r["expected_rank"] == 0 and r["located_expected"] for r in located["forward"])
    # The moved heads are separated by a wide margin, and the identity
    # hypothesis is grossly wrong rather than merely slightly worse.
    assert located["forward"][0]["identity_max_error"] == located["forward"][0]["expected_max_error"]
    assert located["forward"][1]["identity_max_error"] > 10 * located["forward"][1]["expected_max_error"]
    assert min(r["match_margin"] for r in located["forward"]) > 5
    # The matched-pair residual is measured, not assumed away.
    assert located["quantization"]["matched_max_error"] == max(r["expected_max_error"] for r in located["forward"])
    assert located["quantization"]["matched_mean_error"] > 0
    assert all(r["expected_relative_error"] is not None for r in located["forward"])

    # Negative control: an identity-arranged tensor must not satisfy the cyclic
    # expectation, while it does satisfy the identity one.
    source, stored = rows_fixture(6, 2, 4, 3, [0, 1, 2], "identity")
    wrong = locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic")
    assert not wrong["all_forward_located"]
    # The identity arrangement still holds each source head at its own index, so
    # the un-moved head 0 matches and the moved heads are mislocated.
    assert [r["located_stored_head"] for r in wrong["forward"]] == [0, 1, 2]
    assert wrong["forward"][0]["expected_rank"] == 0
    assert all(r["expected_rank"] > 0 for r in wrong["forward"][1:])
    identity = locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "rows", "identity")
    assert identity["all_forward_located"] and identity["all_reverse_located"]


def test_locate_heads_handles_a_column_permutation():
    source, stored = columns_fixture(6, 2, 4, 3, "cyclic")
    located = locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "columns", "cyclic")
    assert located["all_forward_located"] and located["all_reverse_located"]
    assert [r["expected_stored_head"] for r in located["forward"]] == [0, 2, 4]
    assert all(r["expected_rank"] == 0 for r in located["forward"])
    # A column permutation below a stored head offset is meaningless.
    with pytest.raises(ValueError, match="no stored head offset"):
        locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "columns", "cyclic", 1)
    # A tensor that is not value-head grouped must be refused.
    with pytest.raises(ValueError, match="not value-head grouped"):
        locate_heads(source[:, :12], stored[:, :12], [0, 1, 2], 2, 6, 4, "columns", "cyclic")


def test_locate_heads_rejects_malformed_input():
    source, stored = rows_fixture(6, 2, 4, 3, [0, 1, 2], "cyclic")
    with pytest.raises(ValueError, match="window rank mismatch"):
        locate_heads(source[0], stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic")
    with pytest.raises(ValueError, match="column mismatch"):
        locate_heads(source, stored[:, :2], [0, 1, 2], 2, 6, 4, "rows", "cyclic")
    with pytest.raises(ValueError, match="non-finite"):
        locate_heads(source * np.nan, stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic")
    with pytest.raises(ValueError, match="invalid source head probe set"):
        locate_heads(source, stored, [0, 0, 1], 2, 6, 4, "rows", "cyclic")
    with pytest.raises(ValueError, match="unknown expectation"):
        locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "rows", "circular")
    with pytest.raises(ValueError, match="unknown permutation axis"):
        locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "diagonal", "cyclic")
    with pytest.raises(ValueError, match="does not hold exactly the probed heads"):
        locate_heads(source[:4], stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic")
    with pytest.raises(ValueError, match="too short for the value-head block"):
        locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic", 4)
    with pytest.raises(ValueError, match="negative stored head offset"):
        locate_heads(source, stored, [0, 1, 2], 2, 6, 4, "rows", "cyclic", -1)


def test_pinned_header_document_is_validated(tmp_path):
    good = {"repository": "Qwen/Qwen3.8-Flash-Next", "revision": "f5d08274bafd880402bd16f5e3e6c514136ec06c",
            "shard": "model-00001-of-00131.safetensors", "header_sha256": "00", "header_bytes": 39296}
    path = tmp_path / "header.json"
    path.write_text(json.dumps(good))
    assert _pinned_header(path)["revision"] == good["revision"]
    for mutation, message in (({"revision": "main"}, "revision must be pinned"),
                              ({"shard": "../model.safetensors"}, "unsafe source shard name"),
                              ({"shard": "model.txt"}, "unsafe source shard name")):
        path.write_text(json.dumps({**good, **mutation}))
        with pytest.raises(ValueError, match=message):
            _pinned_header(path)
    del good["header_bytes"]
    path.write_text(json.dumps(good))
    with pytest.raises(ValueError, match="missing header_bytes"):
        _pinned_header(path)


def test_probe_manifest_is_bounded_and_hashed():
    manifest = json.loads(PROBE_MANIFEST.read_text())
    assert manifest["repository"] == "Qwen/Qwen3.8-Flash-Next"
    assert manifest["revision"] == "f5d08274bafd880402bd16f5e3e6c514136ec06c"
    assert manifest["header_sha256"] == "bd0ba6edc3cffdcfb05456a7e1c3973ac7380a6c43fc47ddfd220a91a283fea5"
    assert (manifest["key_heads"], manifest["value_heads"], manifest["width"]) == (16, 48, 128)
    # Bounded probes only: the pinned shard is 35.9 GB, these reads total ~10 MiB.
    assert sum(p["size_bytes"] for p in manifest["probes"]) == 10616832
    for probe in manifest["probes"]:
        raw = (EVIDENCE / probe["file"]).read_bytes()
        assert len(raw) == probe["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == probe["sha256"]
        assert probe["row_count"] * probe["shape"][1] * 2 == probe["size_bytes"]
        assert probe["fetch_row_start"] % manifest["width"] == 0


def test_actual_projection_evidence_confirms_the_cyclic_value_head_order():
    report = json.loads(REPORT.read_text())
    digest = report.pop("report_sha256")
    assert hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == digest
    assert report["probe_manifest_sha256"] == hashlib.sha256(PROBE_MANIFEST.read_bytes()).hexdigest()
    assert (report["key_heads"], report["value_heads"], report["width"]) == (16, 48, 128)
    assert report["source_shard"] == "model-00001-of-00131.safetensors"

    results = {r["tag"]: r for r in report["results"]}
    assert set(results) == {"in_proj_qkv_qk", "in_proj_qkv_v", "in_proj_z", "out_proj"}
    assert results["in_proj_qkv_qk"]["expectation"] == "identity"
    assert results["in_proj_qkv_qk"]["permutation_axis"] == "rows"
    assert results["out_proj"]["permutation_axis"] == "columns"
    assert [results[t]["stored_quantization"] for t in ("in_proj_qkv_v", "in_proj_z", "out_proj")] == \
        ["Q6_K", "Q4_K", "Q6_K"]

    for tag, result in results.items():
        assert result["all_forward_located"] and result["all_reverse_located"], tag
        for record in result["forward"]:
            assert record["expected_rank"] == 0 and record["located_expected"], tag
            assert record["located_stored_head"] == record["expected_stored_head"], tag
            moved = record["expected_stored_head"] != record["source_head"]
            if moved:
                assert record["identity_max_error"] > 10 * record["expected_max_error"], tag
        for record in result["reverse"]:
            assert record["located_expected"], tag

    # Q/K are not permuted; the value block, the gate projection and the output
    # projection all use the cyclic order proven for the head tensors.
    cyclic_map = {0: 0, 1: 16, 2: 32, 3: 1, 4: 17, 5: 33}
    assert [r["expected_stored_head"] for r in results["in_proj_qkv_qk"]["forward"]] == [0, 1, 2]
    for tag in ("in_proj_qkv_v", "in_proj_z", "out_proj"):
        forward = results[tag]["forward"]
        assert [r["source_head"] for r in forward] == [0, 1, 2, 3, 4, 5]
        assert [r["expected_stored_head"] for r in forward] == [cyclic_map[h] for h in range(6)]
        # Quantization error stays small and every real match beats its runner-up.
        assert max(r["expected_max_error"] for r in forward) < 0.05
        assert min(r["match_margin"] for r in forward) > 10
        # The conversion residual is bounded and far below the cost of getting the
        # head order wrong, which is what makes the ordering claim meaningful.
        fidelity = results[tag]["quantization"]
        assert fidelity["matched_max_error"] < 0.05
        assert fidelity["matched_mean_error"] < 0.002
        assert fidelity["matched_max_relative_error"] < 0.1
        assert fidelity["moved_head_min_identity_error"] > 10 * fidelity["matched_max_error"]

    # The quantized projections use the same permutation as the unquantized head
    # tensors proven earlier.
    head_order = json.loads(HEAD_ORDER_REPORT.read_text())
    assert head_order["results"][0]["source_head_indices"] == cyclic_source_order(16, 48).tolist()


def test_quantized_projection_ordering_closes_the_head_order_report_gap():
    # The earlier report explicitly refuses to claim quantized projection parity;
    # the new evidence is what closes that gap, and it must stay layered on top
    # of the same pinned revision.
    head_order = json.loads(HEAD_ORDER_REPORT.read_text())
    assert "no full-model output or quantized projection parity claim" in head_order["scope"]
    report = json.loads(REPORT.read_text())
    assert report["source_repository"] == head_order["source_repository"]
    assert report["source_revision"] == head_order["source_revision"]
    assert "no full-model output" in report["scope"]
