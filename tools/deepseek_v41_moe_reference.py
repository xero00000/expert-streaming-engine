#!/usr/bin/env python3
"""Initialized official Gate/Expert oracle; not a trained model quality test."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from deepseek_v41_qat_reference import quantize

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('snapshot', type=Path)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
source = (a.snapshot / 'inference/model.py').read_bytes()
digest = hashlib.sha256(source).hexdigest()
assert digest == json.loads((a.snapshot / 'provenance.json').read_text())['files']['inference/model.py']['sha256']
class Linear(torch.nn.Linear):
    def __init__(self, inp, out, dtype=None):
        super().__init__(inp, out, bias=False, dtype=torch.bfloat16)
    def forward(self, x):
        return F.linear(quantize(x.float(), 0).to(x.dtype), self.weight)
scope = dict(torch=torch, nn=torch.nn, F=F, Linear=Linear, ModelArgs=SimpleNamespace, linear=F.linear)
body = [n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name in {'Gate', 'Expert'}]
exec(compile(ast.Module(body=body, type_ignores=[]), 'official-v41-moe', 'exec'), scope)
engram_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Engram')
engram_forward = next(n for n in engram_class.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
exec(compile(ast.Module(body=[engram_forward], type_ignores=[]), 'official-v41-engram', 'exec'), scope)
torch.manual_seed(4109)
torch.set_num_threads(1)
args = SimpleNamespace(dim=32, get_moe_config=lambda _: (4, 3), score_func='sqrtsoftplus', gate_temp=1,
                       norm_topk_prob=True, route_scale=1.5, vision_enabled=False)
gate = scope['Gate'](0, args)
experts = [scope['Expert'](32, 32, swiglu_limit=10) for _ in range(4)]
shared = scope['Expert'](32, 32, swiglu_limit=10)
def enc(x):
    return {'shape': list(reversed(x.shape)), 'values': x.detach().float().flatten().tolist()}
with torch.no_grad():
    gate.weight.copy_(torch.randn_like(gate.weight) * .1)
    gate.bias.copy_(torch.tensor([.2, -.4, .8, -.1]))
    for e in experts + [shared]:
        for w in e.parameters():
            w.copy_(torch.randn_like(w) * .35)
    weights = {'router': enc(gate.weight), 'bias': enc(gate.bias)}
    for name, attr in [('up', 'w3'), ('gate', 'w1'), ('down', 'w2')]:
        weights[name] = enc(torch.stack([getattr(e, attr).weight for e in experts]))
    for name, attr in [('shared_up', 'w3'), ('shared_gate', 'w1'), ('shared_down', 'w2')]:
        weights[name] = enc(getattr(shared, attr).weight)
    cases = []
    for count, scale in [(1, .3), (1, 12), (5, 3)]:
        x = (torch.randn(count, 32) * scale).bfloat16()
        w, ids = gate(x)
        y = torch.zeros(count, 32)
        for t in range(count):
            for k in range(3):
                y[t] += experts[int(ids[t, k])](x[t:t+1], w[t, k]).float()[0]
        shared_output = shared(x)
        cases.append(dict(input=enc(x), output=enc(y), ids=enc(ids), routing_weights=enc(w),
                          shared_output=enc(shared_output), combined_output=enc((y + shared_output.float()).bfloat16())))
engram_cases = []
with torch.no_grad():
    for tokens in [1, 5]:
        x = torch.randn(1, tokens, 4, 32).bfloat16()
        embeddings = torch.randn(1, tokens, 4, 8).bfloat16()
        projector = Linear(32, 160)
        projector.weight.copy_(torch.randn_like(projector.weight) * .1)
        q, k = torch.randn(4, 32).bfloat16(), torch.randn(4, 32).bfloat16()
        mask = torch.tensor([[i % 2 == 0 for i in range(tokens)]])
        owner = SimpleNamespace(dim=32, hc_mult=4, eps=1e-20, clamp_value=1e-6,
                                embed=lambda _: embeddings, wkv=projector, q_weight=q, k_weight=k)
        y = scope['forward'](owner, x, None, mask)
        engram_cases.append(dict(input=enc(x[0]), embeddings=enc(embeddings.flatten(-2)[0]),
            projection=enc(projector.weight), q=enc(q), k=enc(k), mask=enc(mask[0].view(tokens, 1, 1)), output=enc(y[0])))
a.output.write_text(json.dumps(dict(source_sha256=digest, weights=weights, cases=cases, engram_cases=engram_cases), indent=2) + '\n')
