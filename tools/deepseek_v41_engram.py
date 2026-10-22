#!/usr/bin/env python3
"""Extract a bounded Engram row and decode its FP8 weights for parity fixtures.

Each row has 256 E4M3FN weights and eight E8M0 scales (32 weights/scale).
Output is before the reference model's BF16 cast; this is not a GGUF converter.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

if __package__:
    from .deepseek_v41_preflight import read_range
else:
    from deepseek_v41_preflight import read_range


def e4m3fn(byte: int) -> float:
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return math.nan
    value = math.ldexp(mantissa, -9) if exponent == 0 else math.ldexp(1 + mantissa / 8, exponent - 7)
    return -value if byte & 128 else value


def e8m0(byte: int) -> float:
    return math.nan if byte == 255 else math.ldexp(1.0, byte - 127)


def decode_row(weights: bytes, scales: bytes) -> list[float]:
    if len(weights) != 256 or len(scales) != 8:
        raise ValueError("expected exactly 256 weight bytes and 8 scale bytes")
    values = []
    for i, value in enumerate(weights):
        product = e4m3fn(value) * e8m0(scales[i // 32])
        # Match the reference float32 multiplication, including overflow, before BF16 casting.
        try:
            values.append(struct.unpack("<f", struct.pack("<f", product))[0])
        except OverflowError:
            values.append(math.copysign(math.inf, product))
    return values


def decode_affine2_row(weights: bytes, scales: bytes, biases: bytes) -> list[float]:
    """MLX little-endian uint32 packing, affine-2/group64, BF16 scale and bias."""
    if len(weights) != 64 or len(scales) != 8 or len(biases) != 8:
        raise ValueError("expected 64 weight, 8 scale and 8 bias bytes")
    def bf16(data):
        return [struct.unpack("<f", b"\0\0" + data[i:i + 2])[0] for i in range(0, 8, 2)]
    factors, offsets = bf16(scales), bf16(biases)
    values = []
    for i in range(256):
        code = (weights[i // 4] >> ((i % 4) * 2)) & 3
        value = code * factors[i // 64] + offsets[i // 64]
        try:
            values.append(struct.unpack("<f", struct.pack("<f", value))[0])
        except OverflowError:
            values.append(math.copysign(math.inf, value))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--layer", type=int, choices=(1, 14), required=True)
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--format", choices=("fp8", "affine2"), default="fp8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    provenance = json.loads((args.snapshot / "provenance.json").read_text())
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    config = json.loads((args.snapshot / "config.json").read_text())
    if args.format == "affine2":
        quant = config["quantization_config"]
        if (quant.get("bits"), quant.get("group_size"), quant.get("mode")) != (2, 64, "affine"):
            raise ValueError("affine2 requires MLX 2-bit/group64 configuration")
    root = f"layers.{args.layer}.engram.embed"
    payload = {}
    locations = []
    encoding = (("weight", "F8_E4M3", 256, 256), ("scale", "F8_E8M0", 8, 8)) if args.format == "fp8" else (
        ("weight", "U32", 16, 64), ("scales", "BF16", 4, 8), ("biases", "BF16", 4, 8))
    for suffix, dtype, columns, width in encoding:
        name = root + "." + suffix
        shard = index["weight_map"][name]
        document = json.loads((args.snapshot / "headers" / (shard + ".json")).read_text())
        if document["revision"] != provenance["revision"]:
            raise ValueError("revision mismatch")
        tensor = document["header"][name]
        if tensor["dtype"] != dtype or tensor["shape"][1:] != [columns]:
            raise ValueError("unsupported row encoding")
        if not 0 <= args.row < tensor["shape"][0]:
            raise ValueError("row out of range")
        url = f"https://huggingface.co/{provenance['repository']}/resolve/{provenance['revision']}/{shard}"
        header_size = struct.unpack("<Q", read_range(url, 0, 8))[0]
        offset = 8 + header_size + tensor["data_offsets"][0] + args.row * width
        payload[suffix] = read_range(url, offset, width)
        locations.append({"tensor": name, "shard": shard, "file_offset": offset, "bytes": width})
    values = decode_row(payload["weight"], payload["scale"]) if args.format == "fp8" else decode_affine2_row(
        payload["weight"], payload["scales"], payload["biases"])
    if not all(math.isfinite(v) for v in values):
        raise ValueError("nonfinite source row")
    encoded = {"weight_hex": payload["weight"].hex(), "scale_hex": payload["scale"].hex()} if args.format == "fp8" else {
        "weight_hex": payload["weight"].hex(), "scale_hex": payload["scales"].hex(), "bias_hex": payload["biases"].hex()}
    args.output.write_text(json.dumps({"repository": provenance["repository"], "format": args.format,
        "revision": provenance["revision"], "layer": args.layer, "row": args.row,
        "locations": locations, **encoded,
        "payload_sha256": hashlib.sha256(b"".join(payload.values())).hexdigest(),
        "decoded_before_bf16_cast": values}, indent=2) + "\n")


if __name__ == "__main__":
    main()
