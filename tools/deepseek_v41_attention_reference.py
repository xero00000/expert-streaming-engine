#!/usr/bin/env python3
"""Torch latent attention/sink and grouped-output oracle (not full model).

The initial backend uses an explicit F16 cache/mask, included in this oracle.
Selected window/compressed rows are supplied by the fixture, not an indexer.
"""
import argparse
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(414)
    cases = []
    for dim in [64, 512]:
        for tokens in [1, 3, 17]:
            heads, groups, rank, hidden, rows = 8, 2, 7, 24, 64
            q = torch.randn(tokens, heads, dim) * .3
            kv = torch.randn(rows, dim)
            sinks = torch.linspace(-1, 2, heads)
            mask = torch.full(((tokens + 15) // 16 * 16, rows), -1e9)
            for token in range(tokens):
                mask[token, :token + 1] = 0
                mask[token, 32:35 + token] = 0
            # A sink-only query must yield exactly zero, including with large
            # sink logits. Other queries exercise noncontiguous selected rows.
            if tokens > 1:
                mask[0] = -1e9
            wo_a = torch.randn(groups, rank, dim * heads // groups) * .02
            wo_b = torch.randn(hidden, groups * rank) * .1
            scores = torch.einsum('thd,nd->thn', q, kv.half().float()) / dim**.5
            scores += mask[:tokens].half().float()[:, None, :]
            extended = torch.cat([scores, sinks.expand(tokens, -1)[:, :, None]], dim=-1)
            probs = extended.softmax(-1)[..., :-1]
            attended = torch.einsum('thn,nd->thd', probs, kv.half().float())
            projected = torch.einsum('tgd,grd->tgr', attended.reshape(tokens, groups, -1), wo_a)
            output = projected.flatten(1) @ wo_b.T
            without_sink = torch.einsum('thn,nd->thd', scores.softmax(-1), kv.half().float())
            sink_difference = (without_sink - attended).abs().nan_to_num().max().item()
            wrong_groups = torch.einsum('tgd,grd->tgr', attended.reshape(tokens, groups, -1), wo_a.flip(0))
            group_difference = (wrong_groups.flatten(1) @ wo_b.T - output).abs().max().item()
            assert sink_difference > .01 and group_difference > .01
            def encode(x):
                return {'shape': list(reversed(x.shape)), 'values': x.flatten().tolist()}
            cases.append({'name': f'attention-d{dim}-t{tokens}', 'kind': 'attention',
                          'zero_first_query': tokens > 1,
                          'negative_controls': {'omit_sink_max_error': sink_difference,
                                                'wrong_groups_max_error': group_difference},
                          'inputs': {name: encode(value) for name, value in
                                     dict(q=q, kv=kv, mask=mask, sinks=sinks, wo_a=wo_a, wo_b=wo_b).items()},
                          'expected': {'attended': encode(attended), 'output': encode(output)}})
    # Existing CPU FA accumulates F16 values in F16; CUDA MMA also rounds its
    # intermediate operands. This tolerance is specific to that backend path,
    # not the tighter FP32 graph primitive or exact QAT fixtures.
    args.output.write_text(json.dumps({'absolute_tolerance': 1e-3, 'relative_tolerance': 1e-3,
                                      'scope': __doc__, 'cases': cases}) + '\n')


if __name__ == '__main__':
    main()
