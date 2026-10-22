#!/usr/bin/env python3
"""Export the released tokenizer unchanged to GGUF and create native parity cases."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'gguf-py'))
import gguf
from tokenizers import Tokenizer


def add_tokenizer(writer, tokenizer_path, config_path):
    data = json.loads(Path(tokenizer_path).read_text())
    config = json.loads(Path(config_path).read_text())
    assert data['model']['type'] == 'BPE' and data['normalizer'] == {'type': 'Sequence', 'normalizers': []}
    ids = dict(data['model']['vocab'])
    for item in data['added_tokens']:
        assert not item['single_word'] and not item['lstrip'] and not item['rstrip']
        ids[item['content']] = item['id']
    size = max(ids.values()) + 1
    assert len(ids) == size
    tokens = [''] * size
    types = [1] * size
    for token, index in ids.items(): tokens[index] = token
    for item in data['added_tokens']: types[item['id']] = 3 if item['special'] else 4
    writer.add_tokenizer_model('gpt2')
    writer.add_tokenizer_pre('deepseek-v3')
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    merges = data['model']['merges']
    writer.add_token_merges([' '.join(m) if isinstance(m, list) else m for m in merges])
    writer.add_bos_token_id(ids[config['bos_token']['content']])
    writer.add_eos_token_id(ids[config['eos_token']['content']])
    writer.add_pad_token_id(ids['<｜▁pad▁｜>'])
    writer.add_add_bos_token(config['add_bos_token'])
    writer.add_add_eos_token(config['add_eos_token'])
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args()
    writer = gguf.GGUFWriter(args.gguf, 'llama')
    data = add_tokenizer(writer, args.directory / 'tokenizer.json', args.directory / 'tokenizer_config.json')
    writer.add_context_length(256); writer.add_embedding_length(32); writer.add_block_count(1)
    writer.add_feed_forward_length(32); writer.add_head_count(1); writer.add_head_count_kv(1)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.write_header_to_file(); writer.write_kv_data_to_file(); writer.write_tensors_to_file(); writer.close()
    tokenizer = Tokenizer.from_file(str(args.directory / 'tokenizer.json'))
    texts = ['', 'Hello world!', '1234567890', 'a b c', '日本語 hello 😀', '中文測試。 한국어 عربى हिन्दी русский',
             'def foo(x):\n    return x+1', '\t \r\n  \n', '\\p{N} \\s abc', 'A\u0301 café naïve',
             '<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜>']
    texts += [item['content'] for item in data['added_tokens']]
    texts += ['prefix ' + item['content'] + ' suffix' for item in data['added_tokens'][::17]]
    rng = random.Random(410)
    alphabet = 'abcXYZ0123456789 ,.!?\n\r\t_-/()[]{}\\é中文日本語한글🙂'
    texts += [''.join(rng.choice(alphabet) for _ in range(rng.randrange(1, 200))) for _ in range(128)]
    result = {'tokenizer_sha256': hashlib.sha256((args.directory / 'tokenizer.json').read_bytes()).hexdigest(),
              'cases': [{'text': text, 'ids': tokenizer.encode(text, add_special_tokens=False).ids} for text in texts]}
    args.reference.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(f'Wrote {len(texts)} tokenizer cases')

if __name__ == '__main__': main()
