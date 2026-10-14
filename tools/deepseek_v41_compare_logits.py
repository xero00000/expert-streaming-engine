#!/usr/bin/env python3
"""Compare teacher-forced logits, including a candidate that saves a subset."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    meta = json.loads(Path(str(path) + '.json').read_text())
    if meta['dtype'] != 'float32-native':
        raise ValueError('unsupported logit dtype')
    shape = (meta['rows'], meta['vocabulary'])
    if path.stat().st_size != int(np.prod(shape)) * 4:
        raise ValueError(f'logit file size mismatch: {path}')
    return meta, np.memmap(path, dtype=np.float32, mode='r', shape=shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--answer-start', type=int, default=0)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference, x = read(args.reference)
    candidate, y = read(args.candidate)
    if reference['vocabulary'] != candidate['vocabulary']:
        raise ValueError('vocabulary mismatch')
    lookup = {row['position']: (i, row) for i, row in enumerate(reference['positions'])}
    rows = []
    for index, row in enumerate(candidate['positions']):
        ref_index, ref = lookup[row['position']]
        if (row['input_token'], row['next_token']) != (ref['input_token'], ref['next_token']):
            raise ValueError(f'token mismatch at position {row["position"]}')
        a, b = np.asarray(x[ref_index], dtype=np.float64), np.asarray(y[index], dtype=np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('nonfinite logits')
        maximum = float(np.max(np.abs(a - b)))
        same = int(a.argmax()) == int(b.argmax())
        a -= a.max()
        b -= b.max()
        a -= np.log(np.exp(a).sum())
        b -= np.log(np.exp(b).sum())
        rows.append({'position': row['position'], 'max_abs_logit_difference': maximum,
                     'kl': max(0., float(np.sum(np.exp(a) * (a - b)))),
                     'same_argmax': same, 'reference_nll': float(-a[row['next_token']]),
                     'candidate_nll': float(-b[row['next_token']])})

    def summarize(selected):
        if not selected:
            return None
        return {'rows': len(selected),
                'max_abs_logit_difference': max(row['max_abs_logit_difference'] for row in selected),
                'mean_kl': float(np.mean([row['kl'] for row in selected])),
                'max_kl': max(row['kl'] for row in selected),
                'argmax_agreement': float(np.mean([row['same_argmax'] for row in selected])),
                'reference_mean_nll': float(np.mean([row['reference_nll'] for row in selected])),
                'candidate_mean_nll': float(np.mean([row['candidate_nll'] for row in selected]))}

    report = {'reference': str(args.reference.resolve()), 'candidate': str(args.candidate.resolve()),
              'all': summarize(rows),
              'answer': summarize([row for row in rows if row['position'] >= args.answer_start]),
              'positions': rows}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('all', 'answer')}, indent=2))


if __name__ == '__main__':
    main()
