#!/usr/bin/env python3
"""Fetch one real gate/down expert from an inventoried GGUF for kernel validation."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .deepseek_v41_preflight import read_range
else:
    from deepseek_v41_preflight import read_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("blk.0.ffn_gate_exps", "blk.0.ffn_down_exps"):
        tensor = next(t for t in inventory["tensors"] if t["name"] == name)
        if tensor["type"] not in ("IQ2_XXS", "Q2_K") or tensor["shape"][2] != 384:
            raise ValueError("unexpected expert geometry")
        size = tensor["bytes"] // 384
        if size > 8 * 1024 * 1024 or size * 384 != tensor["bytes"]:
            raise ValueError("expert fixture exceeds size bound")
        url = f"https://huggingface.co/{inventory['repository']}/resolve/{inventory['revision']}/{tensor['shard']}"
        data = read_range(url, tensor["file_offset"], size)
        filename = name + ".expert0.bin"
        (args.output_dir / filename).write_bytes(data)
        records.append({"file": filename, "tensor": name, "expert": 0, "shape": tensor["shape"][:2],
                        "type": tensor["type"], "type_id": tensor["type_id"], "source_offset": tensor["file_offset"],
                        "bytes": size, "sha256": hashlib.sha256(data).hexdigest()})
    (args.output_dir / "fixture.json").write_text(json.dumps({"repository": inventory["repository"],
        "revision": inventory["revision"], "records": records}, indent=2) + "\n")
    print(f"Fetched {sum(r['bytes'] for r in records)} expert bytes for kernel tests")


if __name__ == "__main__":
    main()
