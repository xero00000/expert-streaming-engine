#!/usr/bin/env python3
"""Inventory V4.1 checkpoint headers and estimate streaming traffic; never load weights.

Use --fetch-headers once with a pinned reference snapshot, then rerun offline.
This is an enablement diagnostic, not a declaration of runtime support.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import urllib.request

MAX_HEADER = 32 * 1024 * 1024
EXPERT = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.")


def read_range(url: str, start: int, size: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + size - 1}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        expected = f"bytes {start}-{start + size - 1}/"
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(expected):
            raise ValueError("server did not honor bounded range request; refusing weight download")
        data = response.read(size + 1)
    if len(data) != size:
        raise ValueError("incorrect range length")
    return data


def fetch_headers(snapshot: Path, index: dict, provenance: dict) -> None:
    revision = provenance["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("a full pinned Hugging Face revision is required")
    directory = snapshot / "headers"
    directory.mkdir(exist_ok=True)
    for shard in sorted(set(index["weight_map"].values())):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ValueError("invalid shard name")
        destination = directory / (shard + ".json")
        if destination.exists():
            continue
        url = f"https://huggingface.co/{provenance['repository']}/resolve/{revision}/{shard}"
        size = struct.unpack("<Q", read_range(url, 0, 8))[0]
        if not 2 <= size <= MAX_HEADER:
            raise ValueError(f"unsafe safetensors header size: {size}")
        raw = read_range(url, 8, size)
        header = json.loads(raw)
        destination.write_text(json.dumps({"revision": revision, "shard": shard,
            "header_sha256": hashlib.sha256(raw).hexdigest(), "header": header}) + "\n")


def category(name: str) -> str:
    if EXPERT.match(name):
        return "routed_experts"
    if ".engram.embed." in name:
        return "engram_tables"
    if name.startswith("mtp."):
        return "dspark"
    if name.startswith(("vision.", "aligner.")):
        return "vision"
    return "other_backbone"


def inventory(snapshot: Path, config: dict, index: dict, provenance: dict) -> dict:
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("expected deepseek_v41 config; do not alias to deepseek4")
    text = config["text_config"]
    totals: dict[str, int] = defaultdict(int)
    expert_sizes: dict[tuple[int, int], int] = defaultdict(int)
    tables = []
    seen = set()
    for shard in sorted(set(index["weight_map"].values())):
        document = json.loads((snapshot / "headers" / (shard + ".json")).read_text())
        if document["revision"] != provenance["revision"] or document["shard"] != shard:
            raise ValueError("header provenance mismatch")
        intervals = []
        for name, tensor in document["header"].items():
            if name == "__metadata__":
                continue
            if name in seen or index["weight_map"].get(name) != shard:
                raise ValueError(f"duplicate or incorrectly mapped tensor: {name}")
            seen.add(name)
            begin, end = tensor["data_offsets"]
            if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end <= begin:
                raise ValueError(f"invalid tensor offsets: {name}")
            intervals.append((begin, end))
            size = end - begin
            totals[category(name)] += size
            if match := EXPERT.match(name):
                expert_sizes[tuple(map(int, match.groups()))] += size
            if category(name) == "engram_tables":
                rows = tensor["shape"][0]
                if rows <= 0 or size % rows:
                    raise ValueError(f"non-row-addressable Engram tensor: {name}")
                tables.append({"name": name, "shape": tensor["shape"], "dtype": tensor["dtype"],
                    "bytes": size, "row_bytes": size // rows, "shard": shard})
        cursor = 0
        for begin, end in sorted(intervals):
            if begin != cursor:
                raise ValueError("safetensors data contains gaps or overlapping tensors")
            cursor = end
    if seen != set(index["weight_map"]):
        raise ValueError("incomplete tensor inventory")
    total = sum(totals.values())
    if total != index["metadata"]["total_size"]:
        raise ValueError("tensor sizes disagree with index")
    layers, experts, topk = (text[k] for k in ("num_hidden_layers", "n_routed_experts", "num_experts_per_tok"))
    if set(expert_sizes) != {(layer, expert) for layer in range(layers) for expert in range(experts)}:
        raise ValueError("routed expert coverage disagrees with config")
    sizes = sorted(set(expert_sizes.values()))
    if len(sizes) != 1:
        raise ValueError("heterogeneous expert sizes require route-specific traffic accounting")
    expert_bytes = sizes[0]
    # All main layers executed, batch=1, no cache reuse; excludes shared experts and dense weights.
    cold_bytes = expert_bytes * layers * topk
    lookups = (text["engram_max_ngram_size"] - 1) * text["engram_n_heads"]
    engram_bytes = sum(t["row_bytes"] for t in tables) * lookups
    return {"runtime_supported": False, "repository": provenance["repository"],
        "revision": provenance["revision"], "tensor_count": len(seen), "tensor_bytes": total,
        "bytes_by_category": dict(totals), "expert_count": len(expert_sizes),
        "expert_bundle_bytes_including_scales": expert_bytes, "engram_tensors": tables,
        "decode_batch1_all_main_layers": {"routed_expert_selections": layers * topk,
            "routed_bytes_if_all_miss": cold_bytes, "engram_logical_bytes": engram_bytes,
            "note": "Not measured throughput; ignores compute, other weights, page amplification and transfer overlap."},
        "transfer_scenarios": [{"expert_cache_hit_fraction": hit,
            "routed_bytes_per_token": cold_bytes * (1 - hit)} for hit in (0, .25, .5, .75, .9)]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path, help="directory with config, index and provenance.json")
    parser.add_argument("--fetch-headers", action="store_true")
    parser.add_argument("--storage-path", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    def read(name):
        return json.loads((args.snapshot / name).read_text())
    config, index, provenance = read("config.json"), read("model.safetensors.index.json"), read("provenance.json")
    for name in ("config.json", "model.safetensors.index.json"):
        if hashlib.sha256((args.snapshot / name).read_bytes()).hexdigest() != provenance["files"][name]["sha256"]:
            raise ValueError(f"snapshot hash mismatch: {name}")
    if args.fetch_headers:
        fetch_headers(args.snapshot, index, provenance)
    result = inventory(args.snapshot, config, index, provenance)
    free = shutil.disk_usage(args.storage_path).free
    result["storage"] = {"path": str(args.storage_path.resolve()), "free_bytes": free,
        "checkpoint_payload_fits": result["tensor_bytes"] <= free,
        "minimum_additional_bytes": max(0, result["tensor_bytes"] - free),
        "note": "Payload only; reserve extra space for headers, conversion outputs and filesystem headroom."}
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
