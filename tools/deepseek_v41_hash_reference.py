#!/usr/bin/env python3
"""Export Engram constants, token map and exact traces from pinned DeepSeek code.

Uses the official tokenizer normalization and NumPy RNG once at export time.
Native execution consumes these constants; it does not emulate either library.
No model weights or accelerator kernels are loaded.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import struct
import sys
from types import SimpleNamespace


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import tokenizers
    import numpy
    import sympy

    provenance = json.loads((args.snapshot / "provenance.json").read_text())
    for name in ("config.json", "inference/engram.py"):
        if digest(args.snapshot / name) != provenance["files"][name]["sha256"]:
            raise ValueError(f"pinned source changed: {name}")
    spec = importlib.util.spec_from_file_location("deepseek_v41_reference_engram", args.snapshot / "inference/engram.py")
    official = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = official
    spec.loader.exec_module(official)
    backend = tokenizers.Tokenizer.from_file(str(args.snapshot / "tokenizer.json"))

    class Tokenizer:
        backend_tokenizer = backend

        def __len__(self):
            return backend.get_vocab_size(with_added_tokens=True)

    config = json.loads((args.snapshot / "config.json").read_text())["text_config"]
    model_args = SimpleNamespace(**{k: v for k, v in config.items() if k.startswith("engram_")},
                                 engram_pad_id=config["engram_pad_token_id"], max_batch_size=1, max_seq_len=512)
    layout = official.EngramLayout.from_args(model_args)
    state = official.NgramHashState(model_args, layout, Tokenizer())
    # Constructor asserts compressed vocab size against the official 99,092 value.
    token_map = state.token_map.tolist()
    raw_map = struct.pack(f"<{len(token_map)}I", *token_map)
    constants = {
        "compressed_vocab_size": config["engram_compressed_vocab_size"],
        "pad_compressed_id": state.pad_id,
        "layer_ids": list(layout.layer_ids),
        "num_embeddings": list(layout.num_embeddings),
        "multipliers": state.multipliers.tolist(),
        "primes": state.primes.flatten(1).tolist(),
        "offsets": state.offsets.tolist(),
        "token_map_file": "engram-token-map.u32le",
        "token_map_count": len(token_map),
        "token_map_sha256": hashlib.sha256(raw_map).hexdigest(),
    }
    prompts = [" The THE the café CAFÉ\n\t  hello world!", "def add(a, b):\n    return a + b\n", "你好世界。日本語と🙂 café"]
    cases = []
    for number, prompt in enumerate(prompts):
        ids = backend.encode(prompt, add_special_tokens=False).ids
        cases.append((f"text-{number}", ids, [True] * len(ids)))
    ids = [0, 2, 100, 129264, 129264, 900, 1, 2, 129264, 1000, 2, 500, 60]
    cases.append(("image-boundaries-and-pad", ids, [i not in (3, 4, 8) for i in range(len(ids))]))
    rng = random.Random(41001)
    ids = [rng.randrange(len(token_map)) for _ in range(128)]
    cases.append(("seeded-random", ids, [rng.random() > .15 for _ in ids]))
    traces = []
    for name, ids, mask in cases:
        tokens, enabled = torch.tensor([ids]), torch.tensor([mask])
        expected = state(tokens, 0, enabled)[0].tolist()
        # Verify that the released implementation agrees with itself when chunked.
        for width in (1, 3, 7):
            chunked = []
            for start in range(0, len(ids), width):
                chunked.extend(state(tokens[:, start:start + width], start, enabled[:, start:start + width])[0].tolist())
            if chunked != expected:
                raise ValueError(f"official chunk parity failed: {name}, width={width}")
        traces.append({"name": name, "token_ids": ids, "text_mask": mask, "row_ids": expected})
    output = {
        "repository": provenance["repository"], "revision": provenance["revision"],
        "source_sha256": digest(args.snapshot / "inference/engram.py"),
        "tokenizer_sha256": digest(args.snapshot / "tokenizer.json"),
        "versions": {"torch": torch.__version__, "tokenizers": tokenizers.__version__,
                     "numpy": numpy.__version__, "sympy": sympy.__version__},
        "config": constants, "cases": traces,
    }
    output["trace_sha256"] = hashlib.sha256(json.dumps(output, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / constants["token_map_file"]).write_bytes(raw_map)
    (args.output_dir / "engram-hash-reference.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"token_map_count": len(token_map), "compressed_vocab_size": config["engram_compressed_vocab_size"],
                      "cases": len(traces), "tokens": sum(len(case[1]) for case in cases), "trace_sha256": output["trace_sha256"]}))


if __name__ == "__main__":
    main()
