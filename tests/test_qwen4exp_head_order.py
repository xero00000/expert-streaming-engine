import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.qwen4exp_head_order import compare_head_order

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/qwen4exp/native-recurrence-20261007"


def test_head_order_distinguishes_identity_and_cyclic():
    source = np.arange(6 * 8, dtype=np.float32).reshape(6, 8)
    stored = source[[0, 3, 1, 4, 2, 5]]
    result = compare_head_order(source, stored, 2, 6, 128, "head")
    assert result["cyclic_exact"] and result["cyclic_max_error"] == 0
    assert result["identity_max_error"] > 1
    assert result["source_head_indices"] == [0, 3, 1, 4, 2, 5]
    with pytest.raises(ValueError, match="geometry"):
        compare_head_order(source, stored, 4, 6, 128, "head")
    with pytest.raises(ValueError, match="shape/non-finite"):
        compare_head_order(source, stored * np.nan, 2, 6, 128, "head")
    with pytest.raises(ValueError, match="unknown tensor role"):
        compare_head_order(source, stored, 2, 6, 128, "bad")


def test_actual_layer0_evidence_is_hashed_and_discriminates_head_order():
    manifest_path = EVIDENCE / "layer0-source-tensors.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["repository"] == "Qwen/Qwen3.8-Flash-Next"
    assert manifest["revision"] == "f5d08274bafd880402bd16f5e3e6c514136ec06c"
    assert sum(r["size_bytes"] for r in manifest["records"]) == 573888
    for record in manifest["records"]:
        raw = (EVIDENCE / record["file"]).read_bytes()
        assert len(raw) == record["size_bytes"]
        assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    report = json.loads((EVIDENCE / "head-order-report.json").read_text())
    digest = report.pop("report_sha256")
    assert hashlib.sha256(json.dumps(report, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == digest
    assert report["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert (report["key_heads"], report["value_heads"], report["width"]) == (16, 48, 128)
    results = {r["gguf_name"]: r for r in report["results"]}
    for suffix in ("ssm_alpha.weight", "ssm_beta.weight", "ssm_conv1d.weight", "ssm_dt.bias"):
        r = results["blk.0." + suffix]
        assert r["cyclic_exact"] and r["cyclic_max_error"] == 0
        assert r["identity_max_error"] > 0.1
    assert results["blk.0.ssm_norm.weight"]["cyclic_exact"]
    # F32 exp differs slightly by implementation; require strict numeric
    # agreement while identity is grossly different. Do not claim exact bytes.
    assert results["blk.0.ssm_a"]["cyclic_max_error"] < 1e-5
    assert results["blk.0.ssm_a"]["identity_max_error"] > 100
