#!/usr/bin/env python3
"""Prove the value-head order of the quantized Qwen4Exp projections.

`qwen4exp_head_order.py` proves the cyclic value-head permutation for the
unquantized layer-0 head tensors (alpha/beta/decay/conv/norm), where stored
payload bytes equal the source bytes exactly.  The three large projections that
consume the value dimension are quantized, so byte equality cannot hold.  This
tool instead fetches a bounded window of the pinned original checkpoint and
*locates* every fetched source value head among all value heads of the stored
GGUF tensor.  That is a tolerance-free nearest-neighbour statement: it does not
depend on any quantization error model.

Runtime assumption under test (`src/llama-delta-net.h` `qwen4exp_repeat_type`,
`src/llama-delta-net.cpp` `build_qkvz`/`build_qkv`/`build_gated_output`):
stored value head `s` holds source value head `order[s]`, the source
`repeat_interleave` layout transposed `[key_head, repeat] -> [repeat, key_head]`.

Only bounded range reads are performed; the full shard is never downloaded.

Usage:

    python3 tools/qwen4exp_projection_order.py --fetch \\
        --header docs/qwen4exp/native-recurrence-20261007/checkpoint-layer0-header.json \\
        --probe-dir docs/qwen4exp/native-recurrence-20261007 \\
        --output docs/qwen4exp/native-recurrence-20261007/layer0-projection-probes.json
    python3 tools/qwen4exp_projection_order.py <model.gguf> \\
        docs/qwen4exp/native-recurrence-20261007/layer0-projection-probes.json \\
        --output docs/qwen4exp/native-recurrence-20261007/projection-order-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf import GGUFReader, dequantize
from deepseek_v41_preflight import read_range

# Value heads fetched per row-permuted probe.  Six heads cover all three
# repeats for two key heads, which is what distinguishes the [key, repeat] and
# [repeat, key] transposes.  Full rows are fetched for the column-permuted probe.
PROBE_HEADS = 6
OUT_PROJ_ROWS = 64


def cyclic_source_order(key_heads: int, value_heads: int) -> np.ndarray:
    """Source value head held by each stored value head (GGUF cyclic order)."""
    if key_heads <= 0 or value_heads <= 0 or value_heads % key_heads:
        raise ValueError("invalid grouped-head geometry")
    return np.arange(value_heads).reshape(key_heads, value_heads // key_heads).T.ravel()


def stored_head_for_source(source_head: int, key_heads: int, value_heads: int) -> int:
    """Inverse of :func:`cyclic_source_order`."""
    if not 0 <= source_head < value_heads:
        raise ValueError("source head outside value-head range")
    hits = np.flatnonzero(cyclic_source_order(key_heads, value_heads) == source_head)
    if hits.size != 1:
        raise ValueError("source head is not a value head")
    return int(hits[0])


def default_probes(layer: int, key_heads: int, value_heads: int, width: int) -> list[dict]:
    key_dim = key_heads * width
    prefix = f"model.language_model.layers.{layer}.linear_attn"
    return [
        {
            "tag": "in_proj_qkv_qk",
            "name": f"{prefix}.in_proj_qkv.weight",
            "file": f"layer{layer}-in_proj_qkv-qk.bin",
            "gguf": f"blk.{layer}.attn_qkv.weight",
            "permutation_axis": "rows",
            "expectation": "identity",
            "fetch_row_start": 0,
            "row_count": 3 * width,
            "source_heads": [0, 1, 2],
        },
        {
            "tag": "in_proj_qkv_v",
            "name": f"{prefix}.in_proj_qkv.weight",
            "file": f"layer{layer}-in_proj_qkv-v.bin",
            "gguf": f"blk.{layer}.attn_qkv.weight",
            "permutation_axis": "rows",
            "expectation": "cyclic",
            "fetch_row_start": 2 * key_dim,
            "row_count": PROBE_HEADS * width,
            "source_heads": list(range(PROBE_HEADS)),
        },
        {
            "tag": "in_proj_z",
            "name": f"{prefix}.in_proj_z.weight",
            "file": f"layer{layer}-in_proj_z.bin",
            "gguf": f"blk.{layer}.attn_gate.weight",
            "permutation_axis": "rows",
            "expectation": "cyclic",
            "fetch_row_start": 0,
            "row_count": PROBE_HEADS * width,
            "source_heads": list(range(PROBE_HEADS)),
        },
        {
            "tag": "out_proj",
            "name": f"{prefix}.out_proj.weight",
            "file": f"layer{layer}-out_proj.bin",
            "gguf": f"blk.{layer}.ssm_out.weight",
            "permutation_axis": "columns",
            "expectation": "cyclic",
            "fetch_row_start": 0,
            "row_count": OUT_PROJ_ROWS,
            "source_heads": list(range(PROBE_HEADS)),
        },
    ]


def head_block(matrix: np.ndarray, head: int, width: int, axis: str) -> np.ndarray:
    """Flatten one width-wide head block of a value-head-grouped matrix."""
    if width <= 0 or head < 0:
        raise ValueError("invalid head block request")
    if axis == "rows":
        block = matrix[head * width:(head + 1) * width, :]
    elif axis == "columns":
        block = matrix[:, head * width:(head + 1) * width]
    else:
        raise ValueError("unknown permutation axis")
    return np.ascontiguousarray(block).reshape(-1)


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()


def locate_heads(source_window: np.ndarray, stored_window: np.ndarray, source_heads: list[int],
                 key_heads: int, value_heads: int, width: int, axis: str, expectation: str,
                 stored_head_offset: int = 0) -> dict:
    """Locate each fetched source head among all stored value heads.

    For a row-permuted tensor ``stored_window`` must hold every value head and
    ``stored_head_offset`` gives the width-wide block where the probed value
    section starts (the value block of a combined QKV tensor, otherwise 0);
    ``source_window`` then holds exactly the fetched source heads.  For a
    column-permuted tensor ``stored_window`` must be the stored tensor restricted
    to the same rows as the probe, with all value-head columns present.
    """
    if source_window.ndim != 2 or stored_window.ndim != 2:
        raise ValueError("probe window rank mismatch")
    if source_window.shape[1] != stored_window.shape[1]:
        raise ValueError("probe window column mismatch")
    if not np.isfinite(source_window).all() or not np.isfinite(stored_window).all():
        raise ValueError("non-finite probe values")
    if stored_head_offset < 0:
        raise ValueError("negative stored head offset")
    if len(set(source_heads)) != len(source_heads) or min(source_heads) < 0 or max(source_heads) >= value_heads:
        raise ValueError("invalid source head probe set")
    if expectation not in ("cyclic", "identity"):
        raise ValueError("unknown expectation")
    if axis == "rows":
        if source_window.shape[0] != len(source_heads) * width:
            raise ValueError("row window does not hold exactly the probed heads")
        if stored_window.shape[0] < (stored_head_offset + value_heads) * width:
            raise ValueError("stored window is too short for the value-head block")
        grouped = stored_window[stored_head_offset * width:(stored_head_offset + value_heads) * width, :]
    elif axis == "columns":
        if stored_head_offset:
            raise ValueError("column permutation has no stored head offset")
        if source_window.shape != stored_window.shape or stored_window.shape[1] != value_heads * width:
            raise ValueError("stored window is not value-head grouped")
        grouped = stored_window
    else:
        raise ValueError("unknown permutation axis")

    stored_blocks = [head_block(grouped, s, width, axis) for s in range(value_heads)]
    if any(block.shape != stored_blocks[0].shape for block in stored_blocks):
        raise ValueError("stored head blocks are not uniform")

    source_blocks = {head: head_block(source_window, head, width, axis) for head in source_heads}
    forward = []
    reverse = []
    for head in source_heads:
        src = source_blocks[head]
        expected = stored_head_for_source(head, key_heads, value_heads) if expectation == "cyclic" else head
        errors = np.array([float(np.max(np.abs(src - block))) for block in stored_blocks])
        ranking = np.argsort(errors, kind="stable")
        best = int(ranking[0])
        expected_error = float(errors[expected])
        runner_up = min(float(e) for i, e in enumerate(errors) if i != expected)
        delta = np.abs(src - stored_blocks[expected])
        magnitude = float(np.max(np.abs(src)))
        forward.append({
            "source_head": head,
            "expected_stored_head": expected,
            "located_stored_head": best,
            "located_expected": best == expected,
            "expected_rank": int(np.flatnonzero(ranking == expected)[0]),
            "expected_max_error": expected_error,
            "expected_mean_error": float(np.mean(delta)),
            "expected_relative_error": expected_error / magnitude if magnitude > 0 else None,
            "located_max_error": float(errors[best]),
            "identity_max_error": float(errors[head]),
            "runner_up_max_error": runner_up,
            "match_margin": (runner_up / expected_error) if expected_error > 0 else None,
            "source_block_sha256": _digest(src),
            "expected_stored_block_sha256": _digest(stored_blocks[expected]),
        })
        reverse.append({
            "stored_head": expected,
            "expected_source_head": head,
            "errors": {str(h): float(np.max(np.abs(source_blocks[h] - stored_blocks[expected]))) for h in source_heads},
        })
    for record in reverse:
        best = min(record["errors"], key=lambda h: (record["errors"][h], int(h)))
        record["located_source_head"] = int(best)
        record["located_expected"] = int(best) == record["expected_source_head"]
    # The matched (source head, stored head) pairs are, by construction, the
    # correct pairing; their residuals measure what the GGUF conversion did to
    # this tensor rather than whether the head order is right.
    relatives = [r["expected_relative_error"] for r in forward if r["expected_relative_error"] is not None]
    fidelity = {
        "matched_max_error": max(r["expected_max_error"] for r in forward),
        "matched_mean_error": float(np.mean([r["expected_mean_error"] for r in forward])),
        "matched_max_relative_error": max(relatives) if relatives else None,
        "moved_head_min_identity_error": min(
            (r["identity_max_error"] for r in forward if r["expected_stored_head"] != r["source_head"]),
            default=None),
    }
    return {
        "expectation": expectation,
        "permutation_axis": axis,
        "value_heads": value_heads,
        "quantization": fidelity,
        "forward": forward,
        "reverse": reverse,
        "all_forward_located": all(r["located_expected"] for r in forward),
        "all_reverse_located": all(r["located_expected"] for r in reverse),
    }


def _pinned_header(header_path: Path) -> dict:
    document = json.loads(header_path.read_text())
    for key in ("repository", "revision", "shard", "header_sha256", "header_bytes"):
        if key not in document:
            raise ValueError(f"source header document is missing {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", document["revision"]):
        raise ValueError("source revision must be pinned")
    if Path(document["shard"]).name != document["shard"] or not document["shard"].endswith(".safetensors"):
        raise ValueError("unsafe source shard name")
    return document


def fetch_probes(header_path: Path, probe_dir: Path, output: Path, layer: int,
                 key_heads: int, value_heads: int, width: int) -> dict:
    document = _pinned_header(header_path)
    base = f"https://huggingface.co/{document['repository']}/resolve/{document['revision']}/{document['shard']}"

    def fetch(start: int, size: int) -> bytes:
        for attempt in range(5):
            try:
                return read_range(base + f"?ese_projection_order={start}-{size}", start, size)
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    header_size = struct.unpack("<Q", fetch(0, 8))[0]
    if header_size != document["header_bytes"]:
        raise ValueError("pinned header size mismatch")
    raw_header = fetch(8, header_size)
    if hashlib.sha256(raw_header).hexdigest() != document["header_sha256"]:
        raise ValueError("pinned header hash mismatch")
    header = json.loads(raw_header)
    data_base = 8 + header_size

    probes = []
    for spec in default_probes(layer, key_heads, value_heads, width):
        if spec["name"] not in header:
            raise ValueError(f"source tensor not present in pinned header: {spec['name']}")
        entry = header[spec["name"]]
        dtype, shape = entry["dtype"], entry["shape"]
        start, end = entry["data_offsets"]
        if dtype != "BF16" or len(shape) != 2:
            raise ValueError(f"unexpected source dtype/rank for {spec['name']}: {dtype} {shape}")
        if spec["permutation_axis"] == "columns":
            if shape[1] != value_heads * width:
                raise ValueError(f"source matrix is not value-head grouped: {spec['name']} {shape}")
        elif shape[0] % width or shape[0] < (spec["fetch_row_start"] + spec["row_count"]):
            raise ValueError(f"source rows are not head aligned: {spec['name']} {shape}")
        row_bytes = shape[1] * 2
        if end - start != row_bytes * shape[0]:
            raise ValueError("source tensor byte size disagrees with its shape")
        offset = data_base + start + spec["fetch_row_start"] * row_bytes
        payload = fetch(offset, spec["row_count"] * row_bytes)
        (probe_dir / spec["file"]).write_bytes(payload)
        probes.append({
            **spec,
            "dtype": dtype,
            "shape": shape,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "source_offset": offset,
        })
    manifest = {
        "schema_version": 1,
        "repository": document["repository"],
        "revision": document["revision"],
        "shard": document["shard"],
        "header_file": header_path.name,
        "header_sha256": document["header_sha256"],
        "header_bytes": header_size,
        "layer": layer,
        "key_heads": key_heads,
        "value_heads": value_heads,
        "width": width,
        "probes": probes,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def inspect(model: Path, probe_manifest: Path) -> dict:
    document = json.loads(probe_manifest.read_text())
    _pinned_header(probe_manifest.parent / document["header_file"])
    key_heads = int(document["key_heads"])
    value_heads = int(document["value_heads"])
    width = int(document["width"])
    reader = GGUFReader(model, "r")
    if reader.fields["general.architecture"].contents() != "qwen4exp":
        raise ValueError("expected qwen4exp GGUF")
    if int(reader.fields["qwen4exp.ssm.group_count"].contents()) != key_heads:
        raise ValueError("GGUF key-head count disagrees with the probe manifest")
    if int(reader.fields["qwen4exp.ssm.time_step_rank"].contents()) != value_heads:
        raise ValueError("GGUF value-head count disagrees with the probe manifest")
    if int(reader.fields["qwen4exp.ssm.state_size"].contents()) != width:
        raise ValueError("GGUF head width disagrees with the probe manifest")
    tensors = {t.name: t for t in reader.tensors}

    results = []
    for probe in document["probes"]:
        raw = (probe_manifest.parent / probe["file"]).read_bytes()
        if len(raw) != probe["size_bytes"] or hashlib.sha256(raw).hexdigest() != probe["sha256"]:
            raise ValueError("probe payload hash/length mismatch")
        # BF16 payload: the stored 16 bits are the high half of the FP32 pattern.
        source = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32).reshape(
            probe["row_count"], probe["shape"][1])
        target = tensors[probe["gguf"]]
        stored = dequantize(target.data, target.tensor_type).reshape(tuple(reversed(target.shape.tolist())))
        if stored.shape != tuple(probe["shape"]):
            raise ValueError(f"stored shape {stored.shape} disagrees with source {probe['shape']}")
        rows = probe["row_count"]
        if probe["permutation_axis"] == "rows":
            if probe["fetch_row_start"] % width:
                raise ValueError("row probe is not head aligned")
            stored_head_offset = probe["fetch_row_start"] // width
            stored_window = stored
        else:
            stored_head_offset = 0
            stored_window = stored[:rows, :]
        analysis = locate_heads(source, stored_window, probe["source_heads"], key_heads, value_heads,
                                width, probe["permutation_axis"], probe["expectation"], stored_head_offset)
        analysis.update(tag=probe["tag"], source_name=probe["name"], gguf_name=target.name,
                        stored_quantization=target.tensor_type.name, probe_sha256=probe["sha256"],
                        fetch_row_start=probe["fetch_row_start"], stored_head_offset=stored_head_offset,
                        row_count=rows)
        results.append(analysis)

    payload = {
        "schema_version": 1,
        "model": str(model.resolve()),
        "model_bytes": model.stat().st_size,
        "source_repository": document["repository"],
        "source_revision": document["revision"],
        "source_shard": document["shard"],
        "source_header_sha256": document["header_sha256"],
        "probe_manifest_sha256": hashlib.sha256(probe_manifest.read_bytes()).hexdigest(),
        "key_heads": key_heads,
        "value_heads": value_heads,
        "width": width,
        "results": results,
        "scope": ("Layer-0 quantized projections only. Locates bounded source windows and "
                  "measures the conversion residual on those windows; no full-model output, "
                  "expert routing or logit parity claim."),
    }
    payload["report_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path, nargs="?")
    parser.add_argument("probe_manifest", type=Path, nargs="?")
    parser.add_argument("--fetch", action="store_true", help="fetch bounded source probes instead of comparing")
    parser.add_argument("--header", type=Path, help="pinned checkpoint header document (fetch mode)")
    parser.add_argument("--probe-dir", type=Path, help="directory that holds the probe payloads")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--key-heads", type=int, default=16)
    parser.add_argument("--value-heads", type=int, default=48)
    parser.add_argument("--width", type=int, default=128)
    args = parser.parse_args()

    if args.fetch:
        if args.header is None or args.probe_dir is None:
            parser.error("--fetch requires --header and --probe-dir")
        report = fetch_probes(args.header, args.probe_dir, args.output, args.layer,
                              args.key_heads, args.value_heads, args.width)
    else:
        if args.model is None or args.probe_manifest is None:
            parser.error("comparison mode requires <model.gguf> and <probe manifest>")
        report = inspect(args.model, args.probe_manifest)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
