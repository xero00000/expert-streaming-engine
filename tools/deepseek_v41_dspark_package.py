#!/usr/bin/env python3
"""Build a native DSpark-only GGUF from the verified original range files."""
import argparse
import bisect
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct

import numpy as np

from deepseek_v41_package import expand_fp8, record, sha, string
from deepseek_v41_dense_repair import atomic


NAMES = {
    'attn.attn_sink': 'attn_sink', 'attn.wq_a.weight': 'attn_q_a',
    'attn.wq_b.weight': 'attn_q_b', 'attn.q_norm.weight': 'attn_q_a_norm',
    'attn.wkv.weight': 'attn_kv', 'attn.kv_norm.weight': 'attn_kv_a_norm',
    'attn.wo_a.weight': 'attn_output_a', 'attn.wo_b.weight': 'attn_output_b',
    'attn_norm.weight': 'attn_norm', 'ffn_norm.weight': 'ffn_norm',
    'ffn.gate.weight': 'ffn_gate_inp', 'ffn.gate.bias': 'exp_probs_b',
    'ffn.shared_experts.w1.weight': 'ffn_gate_shexp',
    'ffn.shared_experts.w2.weight': 'ffn_down_shexp',
    'ffn.shared_experts.w3.weight': 'ffn_up_shexp',
    'main_proj.weight': 'main_proj', 'main_norm.weight': 'main_norm',
    'norm.weight': 'output_norm', 'markov_head.embed.weight': 'markov_embd',
    'markov_head.head.weight': 'markov_head', 'confidence_head.proj.weight': 'confidence',
    **{f'hc_{part}_{suffix}': f'hc_{part}_{suffix}'
       for part in ('attn', 'ffn') for suffix in ('fn', 'scale', 'base')},
}


def repack_mxfp4(packed, scales):
    """Original adjacent nibbles -> GGML low-half/high-half 32-element blocks."""
    rows, packed_width = packed.shape
    codes = np.empty((rows, packed_width * 2), dtype=np.uint8)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    codes = codes.reshape(rows, -1, 32)
    assert scales.shape == codes.shape[:2]
    result = np.empty((*scales.shape, 17), dtype=np.uint8)
    result[:, :, 0] = scales
    result[:, :, 1:] = codes[:, :, :16] | (codes[:, :, 16:] << 4)
    return result.reshape(rows, -1)


class Ranges:
    def __init__(self, root, manifest):
        self.root, self.manifest = root, manifest
        self.shards = {}
        for name, item in manifest['ranges'].items():
            path = root / name
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
            self.shards.setdefault(item['shard'], []).append((item['offset'], item['bytes'], path))
        for entries in self.shards.values():
            entries.sort()

    def tensor(self, name):
        item = self.manifest['tensors'][name]
        start, end = [self.manifest['data_bases'][item['shard']] + x for x in item['data_offsets']]
        entries = self.shards[item['shard']]
        offsets = [x[0] for x in entries]
        parts = []
        while start < end:
            offset, size, path = entries[bisect.bisect_right(offsets, start) - 1]
            assert offset <= start < offset + size
            count = min(end - start, offset + size - start)
            with path.open('rb') as source:
                source.seek(start - offset)
                data = source.read(count)
                assert len(data) == count
                parts.append(data)
            start += count
        dtype = {'F32': '<f4', 'BF16': '<u2', 'F8_E4M3': 'u1', 'F8_E8M0': 'u1', 'I8': 'u1'}[item['dtype']]
        return np.frombuffer(b''.join(parts), dtype).reshape(item['shape'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--ggml-library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / 'manifest.json').read_text())
    assert manifest['status'] == 'complete'
    config = json.loads(args.config.read_text())
    config = config.get('text_config', config)
    assert config['num_nextn_predict_layers'] == 3 and config['dspark_n_routed_experts'] == 128
    lib = ctypes.CDLL(str(args.ggml_library.resolve()))
    dequant = lib.dequantize_row_mxfp4
    dequant.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    dequant.restype = None
    args.output.mkdir(parents=True, exist_ok=True)
    final = args.output / 'DeepSeek-V4.1-DSpark-MXFP4.gguf'
    journal = args.output / 'package.json'
    source_hash = sha(args.source / 'manifest.json')
    if final.exists():
        prior = json.loads(journal.read_text())
        assert prior['status'] == 'complete' and prior['source_manifest_sha256'] == source_hash
        assert sha(final) == prior['sha256']
        print('Existing DSpark package verified', flush=True)
        return

    print('Verifying original range hashes', flush=True)
    reader = Ranges(args.source, manifest)
    tensors, consumed = [], set()
    for name, item in sorted(manifest['tensors'].items()):
        if re.search(r'\.experts\.\d+\.', name) or name.endswith('.scale'):
            continue
        if name.endswith('.bias_vl'):
            consumed.add(name)  # Drafts are text; no vision routing bias.
            continue
        stage, suffix = name.split('.', 2)[1:]
        target = f'mtp.{stage}.{NAMES[suffix]}'
        shape = list(reversed(item['shape']))
        if suffix == 'attn.wo_a.weight':
            shape = [shape[0], config['o_lora_rank'], config['o_groups']]
        kind = 0 if item['dtype'] == 'F32' else 30
        tensors.append({'name': target, 'shape': shape, 'type': kind,
                        'bytes': int(np.prod(shape)) * (4 if kind == 0 else 2), 'sources': [name]})
        consumed.add(name)
        if item['dtype'] == 'F8_E4M3':
            consumed.add(name.removesuffix('.weight') + '.scale')
    for stage in range(3):
        for source, target in [('w1', 'ffn_gate_exps'), ('w3', 'ffn_up_exps'), ('w2', 'ffn_down_exps')]:
            names = [f'mtp.{stage}.ffn.experts.{expert}.{source}.weight' for expert in range(128)]
            rows, packed_width = manifest['tensors'][names[0]]['shape']
            shape = [packed_width * 2, rows, 128]
            tensors.append({'name': f'mtp.{stage}.{target}', 'shape': shape, 'type': 39,
                            'bytes': int(np.prod(shape)) // 32 * 17, 'sources': names})
            for name in names:
                consumed.update((name, name.removesuffix('.weight') + '.scale'))
    assert consumed == set(manifest['tensors']), set(manifest['tensors']) - consumed
    tensors.sort(key=lambda t: t['name'])
    offset = 0
    info = bytearray()
    for tensor in tensors:
        offset += -offset % 32
        tensor['offset'] = offset
        info += string(tensor['name']) + struct.pack('<I', len(tensor['shape']))
        info += b''.join(struct.pack('<Q', d) for d in tensor['shape'])
        info += struct.pack('<IQ', tensor['type'], offset)
        offset += tensor['bytes']
    metadata = [record('general.architecture', 8, 'deepseek41_dspark'),
                record('general.alignment', 4, 32), record('deepseek41_dspark.version', 4, 1),
                record('deepseek41_dspark.config', 8, json.dumps(config)),
                record('deepseek41_dspark.source_revision', 8, manifest['revision'])]
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata)) + b''.join(metadata) + info
    header += b'\0' * (-len(header) % 32)
    state = {'status': 'building', 'source_manifest_sha256': source_hash, 'revision': manifest['revision'],
             'format': 'original MXFP4 experts, FP8 dense expanded to BF16, shared target embedding/head',
             'tensors': [], 'native_dequant_rows_checked': 0}
    atomic(journal, state)
    levels = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    partial = final.with_suffix('.partial')
    with partial.open('wb') as output:
        output.write(header)
        for tensor in tensors:
            output.seek(len(header) + tensor['offset'])
            digest = hashlib.sha256()
            written = 0
            for name in tensor['sources']:
                values = reader.tensor(name)
                if tensor['type'] == 39:
                    scales = reader.tensor(name.removesuffix('.weight') + '.scale')
                    packed = repack_mxfp4(values, scales)
                    for row in (0, len(values) // 2, len(values) - 1):
                        decoded = np.empty(values.shape[1] * 2, dtype=np.float32)
                        dequant(packed[row].ctypes.data, decoded.ctypes.data, decoded.size)
                        codes = np.empty(decoded.size, np.uint8)
                        codes[0::2], codes[1::2] = values[row] & 15, values[row] >> 4
                        expected = levels[codes] * np.exp2(scales[row].astype(np.float32) - 127).repeat(32)
                        assert np.array_equal(decoded, expected), name
                        state['native_dequant_rows_checked'] += 1
                    data = packed.tobytes()
                elif manifest['tensors'][name]['dtype'] == 'F8_E4M3':
                    data = expand_fp8(values, reader.tensor(name.removesuffix('.weight') + '.scale')).tobytes()
                else:
                    data = values.tobytes()
                digest.update(data)
                written += output.write(data)
            assert written == tensor['bytes'], tensor['name']
            state['tensors'].append({**tensor, 'sha256': digest.hexdigest()})
            atomic(journal, state)
            print(f'{len(state["tensors"])}/{len(tensors)} {tensor["name"]}', flush=True)
        output.flush()
        os.fsync(output.fileno())
    state.update(status='complete', bytes=partial.stat().st_size, sha256=sha(partial))
    partial.replace(final)
    atomic(journal, state)
    print(f'Completed {final}: {state["bytes"]:,} bytes', flush=True)


if __name__ == '__main__':
    main()
