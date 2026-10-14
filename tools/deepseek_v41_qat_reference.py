#!/usr/bin/env python3
"""Portable Torch oracle for released V4.1 activation scaling formulas.

E4M3 rounding uses Torch's native cast; E2M1 uses an independent codebook
nearest-distance selection with ties to even codes. Does not execute TileLang.
"""
import argparse
import json
from pathlib import Path

import torch


def quantize(x, mode):
    width = 16 if mode == 2 else 32
    groups = x.reshape(-1, width)
    maximum = groups.abs().amax(-1, keepdim=True)
    if mode == 2:
        scale = (maximum.clamp_min(6 * 2**-9) / 6).to(torch.float8_e4m3fn).float()
    else:
        minimum, divisor = (1e-4, 448) if mode == 0 else (6 * 2**-126, 6)
        scale = torch.exp2(torch.ceil(torch.log2(maximum.clamp_min(minimum) * (1 / divisor))))
    normalized = groups / scale
    if mode == 0:
        rounded = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    else:
        # Order even codes first so argmin's first-tie choice implements RNE.
        levels = torch.tensor([0, 1, 2, 4, .5, 1.5, 3, 6], dtype=torch.float32)
        indices = (normalized.abs().unsqueeze(-1) - levels).abs().argmin(-1)
        rounded = torch.copysign(levels[indices], normalized)
    return (rounded * scale).reshape_as(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(410)
    cases = []
    for mode in range(3):
        width = 16 if mode == 2 else 32
        samples = {'zero': torch.zeros(9, width),
                   'random': torch.randn(257, width) * torch.logspace(-4, 2, 257)[:, None]}
        # Force unit scale, then probe rounding midpoints and adjacent floats.
        midpoints = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5] if mode else
                                 [2**-10, 3 * 2**-10, 1.0625, 1.1875, 15.5, 248, 432])
        rows = []
        for midpoint in midpoints:
            for sign in [-1, 1]:
                for direction in [-float('inf'), 0, float('inf')]:
                    row = torch.zeros(width)
                    row[0] = 448 if mode == 0 else 6
                    value = midpoint if direction == 0 else torch.nextafter(midpoint, torch.tensor(direction))
                    row[1] = sign * value
                    rows.append(row)
        samples['ties'] = torch.stack(rows)
        for name, x in samples.items():
            expected = quantize(x, mode)
            assert torch.isfinite(expected).all()
            def tensor(value):
                return {'shape': list(reversed(value.shape)), 'values': value.flatten().tolist()}
            cases.append({'name': f'qat_{mode}_{name}', 'kind': 'qat', 'mode': mode,
                          'inputs': {'x': tensor(x)}, 'expected': {'output': tensor(expected)}})
    args.output.write_text(json.dumps({'absolute_tolerance': 0, 'relative_tolerance': 0,
                                      'oracle': __doc__, 'cases': cases}) + '\n')


if __name__ == '__main__':
    main()
