#!/usr/bin/env python3
"""Compare original checkpoint tensors with Qwen4Exp GGUF head permutations.

This reports evidence, not full-model correctness. Source tensor manifest must
contain hashes, dtype, shape and pinned repository/revision provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader, dequantize


def compare_head_order(source: np.ndarray, stored: np.ndarray, key_heads: int, heads: int, width: int, role: str) -> dict:
    if key_heads <= 0 or heads <= 0 or heads % key_heads or width <= 0:
        raise ValueError("invalid grouped-head geometry")
    if source.shape != stored.shape or not np.isfinite(source).all() or not np.isfinite(stored).all():
        raise ValueError("tensor shape/non-finite mismatch")
    order = np.arange(heads).reshape(key_heads, heads // key_heads).T.ravel()
    if role == "conv":
        offset = 2 * key_heads * width
        if source.shape[0] != offset + heads * width:
            raise ValueError("convolution geometry mismatch")
        predicted = np.concatenate((source[:offset], source[offset:].reshape(heads, width, -1)[order].reshape(source[offset:].shape)))
    elif role == "head":
        if source.shape[0] != heads:
            raise ValueError("head tensor geometry mismatch")
        predicted = source[order]
    elif role == "norm":
        predicted = source
    else:
        raise ValueError("unknown tensor role")
    return {"identity_max_error": float(np.max(np.abs(source - stored))),
            "cyclic_max_error": float(np.max(np.abs(predicted - stored))),
            "cyclic_exact": bool(np.array_equal(predicted, stored)),
            "source_head_indices": order.tolist()}


def inspect(model: Path, source_manifest: Path) -> dict:
    document = json.loads(source_manifest.read_text())
    if not re.fullmatch(r"[0-9a-f]{40}", document["revision"]):
        raise ValueError("source revision must be pinned")
    reader = GGUFReader(model, "r")
    if reader.fields["general.architecture"].contents() != "qwen4exp":
        raise ValueError("expected qwen4exp GGUF")
    tensors = {t.name: t for t in reader.tensors}
    key_heads = int(reader.fields["qwen4exp.ssm.group_count"].contents())
    heads = int(reader.fields["qwen4exp.ssm.time_step_rank"].contents())
    width = int(reader.fields["qwen4exp.ssm.state_size"].contents())
    mapping = {"in_proj_a.weight": ("ssm_alpha.weight", "head"),
               "in_proj_b.weight": ("ssm_beta.weight", "head"),
               "dt_bias": ("ssm_dt.bias", "head"), "A_log": ("ssm_a", "head"),
               "conv1d.weight": ("ssm_conv1d.weight", "conv"), "norm.weight": ("ssm_norm.weight", "norm")}
    results = []
    for record in document["records"]:
        suffix = record["name"].split("linear_attn.")[-1]
        if suffix not in mapping:
            continue
        if record["dtype"] != "BF16":
            raise ValueError("comparison requires BF16 source")
        filename = record["file"]
        if Path(filename).name != filename:
            raise ValueError("unsafe source tensor filename")
        raw = (source_manifest.parent / filename).read_bytes()
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError("source tensor hash/length mismatch")
        source = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(record["shape"]).squeeze()
        if suffix == "A_log":
            source = -np.exp(source)  # GGUF representation, rounded F32 exp
        target, role = mapping[suffix]
        tensor = tensors["blk.0." + target]
        stored = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape.tolist()))).squeeze()
        result = compare_head_order(source, stored, key_heads, heads, width, role)
        result.update(source_name=record["name"], gguf_name=tensor.name,
                      source_sha256=record["sha256"], gguf_payload_sha256=hashlib.sha256(tensor.data.tobytes()).hexdigest())
        results.append(result)
    if len(results) != len(mapping):
        raise ValueError("missing required source tensors")
    payload = {"schema_version": 1, "model": str(model.resolve()), "model_bytes": model.stat().st_size,
               "source_repository": document["repository"], "source_revision": document["revision"],
               "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
               "key_heads": key_heads, "value_heads": heads, "width": width, "results": results,
               "scope": "Layer-0 head-associated tensors only; no full-model output or quantized projection parity claim."}
    payload["report_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("source_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.model, args.source_manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
