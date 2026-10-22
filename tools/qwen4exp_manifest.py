#!/usr/bin/env python3
"""Create a content-addressed qwen4exp GGUF inventory without loading weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "gguf-py"))

from gguf import GGUFReader  # noqa: E402


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list):
        return [json_value(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        return value.item()
    return value


def inspect_shards(paths: list[Path], include_hashes: bool) -> dict[str, Any]:
    if not paths:
        raise ValueError("no GGUF shards found")

    files: list[dict[str, Any]] = []
    tensors: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}

    for path in paths:
        reader = GGUFReader(path, "r")
        architecture = reader.fields.get("general.architecture")
        if architecture is not None and architecture.contents() != "qwen4exp":
            raise ValueError(f"{path.name}: expected qwen4exp architecture")

        if not model_metadata:
            for key, field in reader.fields.items():
                if key.startswith(("qwen4exp.", "general.")) or key in {
                    "general.architecture",
                    "general.name",
                    "general.description",
                    "general.file_type",
                    "general.quantization_version",
                    "general.license",
                    "general.source.huggingface.repository",
                    "general.source.url",
                    "tokenizer.ggml.model",
                    "tokenizer.ggml.bos_token_id",
                    "tokenizer.ggml.eos_token_id",
                    "tokenizer.ggml.padding_token_id",
                }:
                    model_metadata[key] = json_value(field.contents())

        file_entry: dict[str, Any] = {
            "name": path.name,
            "size_bytes": path.stat().st_size,
            "data_offset": int(reader.data_offset),
            "tensor_count": len(reader.tensors),
        }
        for key in ("split.no", "split.count", "split.tensors.count"):
            if field := reader.fields.get(key):
                file_entry[key] = json_value(field.contents())
        if include_hashes:
            file_entry["sha256"] = sha256_file(path)
        files.append(file_entry)

        for tensor in reader.tensors:
            tensors.append(
                {
                    "name": tensor.name,
                    "shape": [int(value) for value in tensor.shape],
                    "type": tensor.tensor_type.name,
                    "shard": path.name,
                    "data_offset": int(tensor.data_offset),
                    "size_bytes": int(tensor.n_bytes),
                }
            )

    expected_tensors = files[0].get("split.tensors.count")
    if expected_tensors is not None and expected_tensors != len(tensors):
        raise ValueError(
            f"split inventory declares {expected_tensors} tensors, found {len(tensors)}"
        )

    ple = next(
        (tensor for tensor in tensors if tensor["name"] == "per_layer_token_embd.weight"),
        None,
    )
    if ple is None:
        raise ValueError("missing per_layer_token_embd.weight")
    if len(ple["shape"]) != 2 or ple["shape"][1] == 0:
        raise ValueError("invalid PLE tensor shape")
    if ple["size_bytes"] % ple["shape"][1] != 0:
        raise ValueError("PLE tensor does not contain fixed-size rows")

    inventory_bytes = json.dumps(
        tensors, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "schema_version": 1,
        "architecture": "qwen4exp",
        "metadata": model_metadata,
        "files": files,
        "tensor_count": len(tensors),
        "tensor_inventory_sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "ple": {
            **ple,
            "row_count": ple["shape"][1],
            "row_size_bytes": ple["size_bytes"] // ple["shape"][1],
        },
        "tensors": tensors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="directory containing all GGUF shards")
    parser.add_argument("--output", type=Path, help="write JSON to this file instead of stdout")
    parser.add_argument(
        "--skip-sha256", action="store_true", help="skip full-file hashes for a quick inventory"
    )
    parser.add_argument("--official-revision", help="Qwen model repository commit")
    parser.add_argument("--gguf-revision", help="quantized model repository commit")
    parser.add_argument("--gguf-repository", help="explicit quantized-model repository; otherwise use GGUF metadata")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = sorted(args.model_dir.glob("*.gguf"))
    manifest = inspect_shards(paths, not args.skip_sha256)
    manifest["sources"] = {
        "official_model": {
            "repository": "Qwen/Qwen3.8-Flash-Next",
            "revision": args.official_revision,
        },
        "gguf": {
            "repository": args.gguf_repository or manifest["metadata"].get("general.source.huggingface.repository"),
            "url": manifest["metadata"].get("general.repo_url") or manifest["metadata"].get("general.source.url"),
            "revision": args.gguf_revision,
        },
    }
    rendered = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
