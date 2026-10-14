#!/usr/bin/env python3
"""Trace unchanged official candidate selection plus Torch index-head scoring."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    provenance = json.loads((args.snapshot / 'provenance.json').read_text())
    source = (args.snapshot / 'inference/model.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    assert digest == provenance['files']['inference/model.py']['sha256']
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'select_candidate_blocks')
    scope = {'torch': torch, 'F': F}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'official-candidate-selection', 'exec'), scope)
    select = scope['select_candidate_blocks']
    torch.manual_seed(415)
    torch.set_num_threads(1)
    cases, selections = [], []
    for dim, heads, tokens, rows in [(32, 4, 1, 19), (128, 8, 7, 129)]:
        q, k = torch.randn(tokens, heads, dim), torch.randn(rows, dim)
        weights = torch.randn(tokens, heads) / (dim * heads)**.5
        scores = (torch.einsum('thd,nd->thn', q, k).relu() * weights[..., None]).sum(1)
        def encode(x):
            return {'shape': list(reversed(x.shape)), 'values': x.flatten().tolist()}
        cases.append({'name': f'index-d{dim}-t{tokens}', 'kind': 'index_scores',
                      'inputs': {name: encode(x) for name, x in dict(q=q, k=k, weights=weights).items()},
                      'expected': {'scores': encode(scores)}})
    # Distinct scores make exact reference top-k comparisons meaningful;
    # Torch does not specify its ordering for tied scores.
    for width in [1, 7, 17, 129]:
        scores = torch.randperm(width).float() - width / 2
        for visible in sorted(set([0, 1, width // 2, width])):
            for top_blocks in [1, 3]:
                masked = scores.clone()
                masked[visible:] = -torch.inf
                candidates = select(masked, visible, top_blocks, 4)
                filtered = masked.masked_fill(~candidates, -torch.inf)
                topk, offset = min(3, width), 128
                picks = filtered.topk(topk).indices
                picks = picks[torch.isfinite(filtered[picks])].sort().values
                expected = (picks + offset).tolist() + [-1] * (topk - picks.numel())
                selections.append({'scores': scores.tolist(), 'visible': visible, 'block_size': 4,
                                   'top_blocks': top_blocks, 'top_k': topk, 'offset': offset,
                                   'candidates': candidates.int().tolist(), 'positions': expected})
    args.output.write_text(json.dumps({'model_source_sha256': digest, 'source_revision': provenance['revision'],
        'absolute_tolerance': 2e-5, 'relative_tolerance': 2e-4, 'cases': cases, 'selections': selections}) + '\n')


if __name__ == '__main__':
    main()
