#!/usr/bin/env python3
"""Bounded, read-only lossless-compression probe of native expert components."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import zstandard
from deepseek_v41_gguf_probe import parse_header

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    layers = (0, 7, 14, 21, 28, 35)
    records, blocks = [], Counter()
    for path in sorted(args.package.glob('*.gguf')):
        with path.open('rb') as source:
            for size in (1 << 20, 4 << 20, 16 << 20):
                source.seek(0)
                try:
                    _, tensors, start = parse_header(source.read(size))
                    break
                except EOFError:
                    if size == 16 << 20:
                        raise
            for tensor in tensors:
                fields = tensor['name'].split('.')
                if len(fields) != 3 or fields[0] != 'blk' or not fields[1].isdigit():
                    continue
                layer = int(fields[1])
                if layer not in layers or fields[2] not in ('ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps'):
                    continue
                shape = tensor['shape']
                if len(shape) != 3:
                    raise ValueError('expected packed expert dimension')
                kind = GGMLQuantizationType(tensor['type_id'])
                elements, block_bytes = GGML_QUANT_SIZES[kind]
                count = math.prod(shape[:2])
                assert count % elements == 0
                size = count // elements * block_bytes
                assert 0 < size <= 16 << 20
                expert = (layer * 13 + 7) % shape[2]
                offset = start + tensor['relative_offset'] + expert * size
                source.seek(offset)
                raw = source.read(size)
                if len(raw) != size:
                    raise IOError('short expert component read')
                for index in range(0, len(raw) - 4095, 4096):
                    blocks[hashlib.sha256(raw[index:index + 4096]).digest()] += 1
                for level in (1, 3):
                    before = time.perf_counter()
                    packed = zstandard.ZstdCompressor(level=level).compress(raw)
                    encode_seconds = time.perf_counter() - before
                    before = time.perf_counter()
                    decoded = zstandard.ZstdDecompressor().decompress(packed)
                    decode_seconds = time.perf_counter() - before
                    assert decoded == raw
                    records.append({'file': path.name, 'tensor': tensor['name'], 'expert': expert,
                        'offset': offset, 'type': kind.name, 'bytes': size, 'zstd_level': level,
                        'compressed_bytes': len(packed), 'ratio': len(packed) / size,
                        'encode_seconds': encode_seconds, 'decode_seconds': decode_seconds,
                        'sha256': hashlib.sha256(raw).hexdigest(), 'lossless_verified': True})
    if len(records) != 36:
        raise ValueError(f'expected 18 expert components at two levels; found {len(records)} records')
    summaries = {}
    for level in (1, 3):
        selected = [row for row in records if row['zstd_level'] == level]
        summaries[str(level)] = {
            'sample_bytes': sum(row['bytes'] for row in selected),
            'weighted_ratio': sum(row['compressed_bytes'] for row in selected) / sum(row['bytes'] for row in selected),
            'median_component_ratio': statistics.median(row['ratio'] for row in selected),
        }
    result = {'scope': '18 complete components, six layers, deterministic expert sample; not a whole-model deduplication scan or inference-speed measurement',
        'package': str(args.package.resolve()), 'summaries': summaries,
        'sample_4k_blocks': sum(blocks.values()), 'duplicate_4k_blocks': sum(count - 1 for count in blocks.values()),
        'records': records}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
