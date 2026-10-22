#!/usr/bin/env python3
"""Inspect pinned GGUF headers via bounded ranges without downloading tensor data."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import struct
import sys
import urllib.request

if __package__:
    from .deepseek_v41_preflight import read_range
else:
    from deepseek_v41_preflight import read_range

SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class PrefixReader:
    def __init__(self, data):
        self.data, self.offset = data, 0

    def take(self, count):
        if count < 0 or self.offset + count > len(self.data):
            raise EOFError("GGUF header exceeds fetched prefix")
        start = self.offset
        self.offset += count
        return self.data[start:self.offset]

    def scalar(self, kind):
        fmt = "<" + SCALARS[kind]
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.scalar(10)).decode("utf-8")

    def value(self, kind, keep=True, depth=0):
        if depth > 1:
            raise ValueError("nested GGUF arrays are unsupported")
        if kind in SCALARS:
            return self.scalar(kind)
        if kind == 8:
            return self.string()
        if kind != 9:
            raise ValueError("unknown GGUF value type")
        element, count = self.scalar(4), self.scalar(10)
        if count > 2000000:
            raise ValueError("oversized GGUF array")
        values = []
        for _ in range(count):
            value = self.value(element, keep, depth + 1)
            if keep:
                values.append(value)
        return values if keep else {"element_type": element, "count": count}


def parse_header(data):
    reader = PrefixReader(data)
    if reader.take(4) != b"GGUF" or reader.scalar(4) != 3:
        raise ValueError("expected GGUF v3")
    tensor_count, metadata_count = reader.scalar(10), reader.scalar(10)
    if tensor_count > 1000000 or metadata_count > 100000:
        raise ValueError("oversized GGUF header counts")
    metadata = {}
    for _ in range(metadata_count):
        key = reader.string()
        if key in metadata:
            raise ValueError("duplicate GGUF metadata key")
        metadata[key] = reader.value(reader.scalar(4), not key.startswith("tokenizer."))
    tensors = []
    names = set()
    for _ in range(tensor_count):
        name, dimensions = reader.string(), reader.scalar(4)
        if name in names or not 1 <= dimensions <= 4:
            raise ValueError("duplicate tensor or invalid dimensions")
        names.add(name)
        shape = [reader.scalar(10) for _ in range(dimensions)]
        if not all(shape):
            raise ValueError("empty GGUF tensor dimension")
        tensors.append({"name": name, "shape": shape, "type_id": reader.scalar(4), "relative_offset": reader.scalar(10)})
    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("invalid GGUF alignment")
    return metadata, tensors, (reader.offset + alignment - 1) // alignment * alignment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_info", type=Path, help="Hugging Face API model record with pinned sha and blob sizes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
    from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType
    info = json.loads(args.model_info.read_text())
    repository, revision = info["id"], info["sha"]
    shards = [f for f in info["siblings"] if f["rfilename"].endswith(".gguf")]
    manifests, inventory, split_ids = [], [], set()
    all_names = set()
    for file in shards:
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{file['rfilename']}"
        for count in (1024 * 1024, 4 * 1024 * 1024, 16 * 1024 * 1024):
            try:
                metadata, tensors, data_offset = parse_header(read_range(url, 0, min(count, file["size"])))
                break
            except EOFError:
                continue
        else:
            raise ValueError("GGUF header exceeds 16 MiB bound")
        if metadata.get("split.count", 1) != len(shards):
            raise ValueError("incomplete or mixed GGUF split set")
        split_id = metadata.get("split.no", 0)
        if split_id in split_ids:
            raise ValueError("duplicate GGUF split ID")
        split_ids.add(split_id)
        extents = []
        for tensor in tensors:
            name = tensor["name"]
            if name in all_names:
                raise ValueError("duplicate tensor across shards")
            all_names.add(name)
            kind = GGMLQuantizationType(tensor["type_id"])
            block, size = GGML_QUANT_SIZES[kind]
            if tensor["shape"][0] % block:
                raise ValueError("GGUF quantization block does not divide row")
            nbytes = math.prod(tensor["shape"]) // block * size
            offset = data_offset + tensor["relative_offset"]
            if offset + nbytes > file["size"]:
                raise ValueError("GGUF tensor exceeds published file size")
            extents.append((offset, offset + nbytes))
            tensor.update({"type": kind.name, "bytes": nbytes, "file_offset": offset, "shard": file["rfilename"]})
            inventory.append(tensor)
        previous_end = data_offset
        for begin, end in sorted(extents):
            if begin < previous_end:
                raise ValueError("overlapping GGUF tensor extents")
            previous_end = end
        manifests.append({"file": file["rfilename"], "bytes": file["size"], "lfs_sha256": file.get("lfs", {}).get("sha256"),
                          "metadata": {k: v for k, v in metadata.items() if not k.startswith("tokenizer.")},
                          "tensor_count": len(tensors), "data_offset": data_offset})
    if split_ids != set(range(len(shards))):
        raise ValueError("non-contiguous GGUF split set")
    for file in manifests:
        if file["metadata"].get("split.tensors.count", len(inventory)) != len(inventory):
            raise ValueError("GGUF split tensor count disagrees with inventory")
    experts = [t for t in inventory if any(name in t["name"] for name in ("ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"))]
    result = {"repository": repository, "revision": revision, "runtime_supported": False,
              "file_bytes": sum(f["bytes"] for f in manifests), "tensor_count": len(inventory),
              "types": dict(Counter(t["type"] for t in inventory)),
              "routed_expert_bytes": sum(t["bytes"] for t in experts),
              "engram_table_names": [t["name"] for t in inventory if "engram" in t["name"] and "emb" in t["name"]],
              "files": manifests, "tensors": inventory}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("files", "tensors")}, indent=2))


if __name__ == "__main__":
    main()
