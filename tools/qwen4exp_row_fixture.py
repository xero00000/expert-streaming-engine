#!/usr/bin/env python3
"""Extract a content-addressed IQ4_NL PLE row fixture from a model manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGMLQuantizationType, quants  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--row", type=int, required=True, help="global PLE table row")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-shard-hash", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    ple = manifest["ple"]
    if ple["type"] != "IQ4_NL" or ple["row_size_bytes"] != 90 or ple["shape"][0] != 160:
        raise ValueError("manifest does not describe the expected 160-value IQ4_NL PLE rows")
    if args.row < 0 or args.row >= ple["row_count"]:
        raise ValueError("requested PLE row is out of range")

    shard_entry = next(item for item in manifest["files"] if item["name"] == ple["shard"])
    shard_path = args.model_dir / shard_entry["name"]
    if shard_path.stat().st_size != shard_entry["size_bytes"]:
        raise ValueError("PLE shard size does not match the manifest")
    if not args.skip_shard_hash:
        actual_hash = sha256_file(shard_path)
        if actual_hash != shard_entry["sha256"]:
            raise ValueError("PLE shard SHA-256 does not match the manifest")

    file_offset = ple["data_offset"] + args.row * ple["row_size_bytes"]
    with shard_path.open("rb") as source:
        source.seek(file_offset)
        quantized = source.read(ple["row_size_bytes"])
    if len(quantized) != ple["row_size_bytes"]:
        raise ValueError("PLE row read completed short")

    data = np.frombuffer(quantized, dtype=np.uint8).reshape(1, -1)
    values = quants.dequantize(data, GGMLQuantizationType.IQ4_NL).reshape(-1).tolist()
    float_bytes = struct.pack(f"<{len(values)}f", *values)
    payload: dict[str, object] = {
        "schema_version": 1,
        "architecture": "qwen4exp",
        "source": {
            "shard": shard_entry["name"],
            "shard_sha256": shard_entry["sha256"],
            "tensor_inventory_sha256": manifest["tensor_inventory_sha256"],
        },
        "tensor": ple["name"],
        "global_row": args.row,
        "file_offset": file_offset,
        "quantized_type": "IQ4_NL",
        "quantized_sha256": hashlib.sha256(quantized).hexdigest(),
        "quantized_hex": quantized.hex(),
        "value_count": len(values),
        "dequantized_f32_le_sha256": hashlib.sha256(float_bytes).hexdigest(),
        "dequantized_values": values,
    }
    fixture = {**payload, "fixture_sha256": canonical_hash(payload)}
    rendered = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
