#!/usr/bin/env python3
"""Generate deterministic PLE row traces from pinned Transformers source."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any

TRANSFORMERS_REVISION = "fc5c5bde8e656dad91cbf34e61940d984b1c7b91"
TRANSFORMERS_SHA256 = "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"
TRANSFORMERS_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{TRANSFORMERS_REVISION}/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
)
MODEL_REVISION = "f5d08274bafd880402bd16f5e3e6c514136ec06c"
MODEL_CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
MODEL_CONFIG_URL = (
    "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/"
    f"{MODEL_REVISION}/config.json"
)

REFERENCE_SEQUENCES = {
    "eos_segment_reset": [1, 2, 3, 248044, 4, 5],
    "tokenizer_eos_is_not_ple_eos": [10, 248046, 11, 248044, 12],
    "consecutive_ple_eos": [248044, 248044, 1],
    "vocabulary_edges": [0, 248319, 0, 248319],
}


def read_source(location: str) -> bytes:
    if location.startswith(("https://", "http://")):
        with urllib.request.urlopen(location, timeout=30) as response:
            return response.read()
    return Path(location).read_bytes()


def checked_source(location: str, expected_sha256: str) -> bytes:
    content = read_source(location)
    actual = hashlib.sha256(content).hexdigest()
    if actual != expected_sha256:
        raise ValueError(
            f"source hash mismatch for {location}: expected {expected_sha256}, got {actual}"
        )
    return content


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_trace_hash(trace: dict[str, Any]) -> None:
    expected = trace.get("trace_sha256")
    payload = {key: value for key, value in trace.items() if key != "trace_sha256"}
    actual = canonical_hash(payload)
    if expected != actual:
        raise ValueError(f"trace hash mismatch: expected {expected}, got {actual}")


def extract_reference_namespace(source: bytes) -> dict[str, Any]:
    import torch

    tree = ast.parse(source, filename="pinned_modeling_qwen4_exp.py")
    definitions = {
        "_splitmix64",
        "_build_layer_multipliers",
        "_is_prime",
        "_find_nth_prime_after",
        "Qwen4ExpTextNGramEmbedding",
    }
    constants = {
        "_SPLITMIX_GAMMA",
        "_MASK64",
        "_SPLITMIX_M1",
        "_SPLITMIX_M2",
        "_PRIME_1",
    }
    selected: list[ast.stmt] = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in definitions:
            selected.append(node)
            found.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = {target.id for target in targets if isinstance(target, ast.Name)}
            if names & constants:
                selected.append(node)
                found.update(names & constants)
    missing = (definitions | constants) - found
    if missing:
        raise ValueError(f"pinned Transformers source is missing definitions: {sorted(missing)}")

    module = ast.Module(body=selected, type_ignores=[])
    namespace: dict[str, Any] = {"math": math, "nn": torch.nn, "torch": torch}
    exec(compile(module, "pinned_modeling_qwen4_exp.py", "exec"), namespace)
    return namespace


def generate_trace(modeling_source: bytes, config_source: bytes) -> dict[str, Any]:
    import torch

    namespace = extract_reference_namespace(modeling_source)
    text_config = json.loads(config_source)["text_config"]
    config = SimpleNamespace(
        ngram_size=text_config.get("ngram_size", 3),
        heads_per_ngram=text_config.get("heads_per_ngram", 8),
        vocab_size=text_config["vocab_size"],
        ngram_vocab_size_base=text_config.get("ngram_vocab_size_base", 20_000_000),
        seed=text_config.get("seed", 1234),
        eos_token_id=text_config["eos_token_id"],
        make_ngram_vocab_size_divisible_by=text_config.get(
            "make_ngram_vocab_size_divisible_by", 128
        ),
    )
    if config.seed is None:
        config.seed = 1234

    embedding_class = namespace["Qwen4ExpTextNGramEmbedding"]
    with torch.device("meta"):
        module = embedding_class(config, text_config["ple_embed_dim"], 1, 0)

    module.layer_multipliers = torch.nn.Buffer(
        namespace["_build_layer_multipliers"](
            config.vocab_size, config.ngram_size, 0, config.seed
        )
    )
    module.ngram_heads_vocab_sizes = torch.nn.Buffer(
        torch.tensor(module.head_vocab_sizes, dtype=torch.long)
    )
    module.ngram_heads_offsets = torch.nn.Buffer(
        torch.tensor(module.head_offsets, dtype=torch.long)
    )

    class RecordingEmbedding(torch.nn.Module):
        def __init__(self, head_dim: int):
            super().__init__()
            self.weight = torch.empty(0)
            self.head_dim = head_dim
            self.ids: Any = None

        def forward(self, indices: Any) -> Any:
            self.ids = indices.detach().clone()
            return torch.zeros((*indices.shape, self.head_dim), dtype=torch.float32)

    head_count = (config.ngram_size - 1) * config.heads_per_ngram
    recorder = RecordingEmbedding(text_config["ple_embed_dim"] // head_count)
    module.ngram_embedding = recorder

    sequences = []
    for name, token_ids in REFERENCE_SEQUENCES.items():
        module(torch.tensor([token_ids], dtype=torch.long), None)
        sequences.append(
            {
                "name": name,
                "token_ids": token_ids,
                "row_ids": recorder.ids.squeeze(0).tolist(),
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "architecture": "qwen4exp",
        "sources": {
            "official_model": {
                "repository": "Qwen/Qwen3.8-Flash-Next",
                "revision": MODEL_REVISION,
                "config_sha256": MODEL_CONFIG_SHA256,
            },
            "transformers": {
                "repository": "huggingface/transformers",
                "revision": TRANSFORMERS_REVISION,
                "modeling_sha256": TRANSFORMERS_SHA256,
            },
        },
        "ple": {
            "layer_id_one_indexed": text_config["ple_layer_ids"][0],
            "eos_token_id": config.eos_token_id,
            "layer_multipliers": module.layer_multipliers.tolist(),
            "head_vocab_sizes": module.head_vocab_sizes,
            "head_offsets": module.head_offsets,
        },
        "sequences": sequences,
    }
    return {**payload, "trace_sha256": canonical_hash(payload)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modeling-source", default=TRANSFORMERS_URL)
    parser.add_argument("--config-source", default=MODEL_CONFIG_URL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path, help="compare generated output with a trace file")
    parser.add_argument(
        "--check-hash-only", type=Path, help="verify a checked-in trace without fetching sources"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_hash_only:
        verify_trace_hash(json.loads(args.check_hash_only.read_text(encoding="utf-8")))
        return 0

    modeling_source = checked_source(args.modeling_source, TRANSFORMERS_SHA256)
    config_source = checked_source(args.config_source, MODEL_CONFIG_SHA256)
    trace = generate_trace(modeling_source, config_source)
    rendered = json.dumps(trace, indent=2, sort_keys=True) + "\n"

    if args.verify:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        verify_trace_hash(expected)
        if trace != expected:
            raise ValueError(f"generated trace does not match {args.verify}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    elif not args.verify:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
