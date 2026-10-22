#!/usr/bin/env python3
"""Official initialized attention methods with portable QAT/sparse attention.

The full-block option traces initialized token-to-logit computation, with
ascending-position index ties. This is not a trained-weight oracle. Restores each layer's
index-key owner before execution, fixing the released shared-slot odd-step bug.
"""
import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from deepseek_v41_qat_reference import quantize


@contextmanager
def index_tie_policy(enabled, audit):
    # Torch documents tied top-k IDs as unstable. CPU and CUDA choose different
    # rows for the same zero scores in this fixture. Canonicalize index/candidate
    # ties to ascending position, matching the native selector and local CUDA.
    original = torch.Tensor.topk
    def stable(self, k, dim=-1, largest=True, sorted=True):
        result = original(self, k, dim=dim, largest=largest, sorted=sorted)
        if self.ndim != 3:
            return result
        ids = torch.argsort(self, dim=dim, descending=largest, stable=True).narrow(dim, 0, k)
        values = self.gather(dim, ids)
        assert torch.equal(values.sort(dim).values, result.values.sort(dim).values)
        if not torch.equal(ids.sort(dim).values, result.indices.sort(dim).values):
            audit.append({'scores': self.float().flatten().tolist(),
                          'torch_cpu_ids': result.indices.flatten().tolist(), 'canonical_ids': ids.flatten().tolist()})
        return torch.return_types.topk((values, ids))
    if enabled: torch.Tensor.topk = stable
    try:
        yield
    finally:
        torch.Tensor.topk = original


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('snapshot', type=Path)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--production-precision', action='store_true')
    p.add_argument('--full-block', action='store_true')
    args = p.parse_args()
    if args.full_block: args.production_precision = True
    provenance = json.loads((args.snapshot / 'provenance.json').read_text())
    source = (args.snapshot / 'inference/model.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    assert digest == provenance['files']['inference/model.py']['sha256']
    config = json.loads(args.config.read_text())['text_config']
    config.update(hidden_size=64, num_hidden_layers=8, num_attention_heads=8, head_dim=64, qk_rope_head_dim=32,
                  q_lora_rank=32, o_lora_rank=16 if args.production_precision else 8, o_groups=2, index_n_heads=4, index_head_dim=32, index_topk=2,
                  sliding_window=4, compress_ratios=[0, 0, 2, 2, 2, 1, 1, 1], kv_source_layer_ids=[2, 5],
                  index_source_layer_ids=[2, 3, 5, 6], candidate_source_layer_id=5, candidate_topk_blocks=2,
                  candidate_block_size=2, engram_layer_ids=[1, 4], max_position_embeddings=16)
    params = SimpleNamespace(dim=64, n_heads=8, head_dim=64, rope_head_dim=32, q_lora_rank=32,
        o_lora_rank=config['o_lora_rank'], o_groups=2, index_n_heads=4, index_head_dim=32, index_topk=2, window_size=4,
        compress_ratios=config['compress_ratios'], kv_source_layers=[2, 5], index_source_layers=[2, 3, 5, 6],
        candidate_source_layer=5, candidate_topk_blocks=2, candidate_block_size=2, n_layers=8,
        norm_eps=config['rms_norm_eps'], max_batch_size=1, max_seq_len=16,
        original_seq_len=65536, rope_theta=10000, compress_rope_theta=160000, rope_factor=16, beta_fast=32, beta_slow=1)
    if args.full_block:
        config.update(vocab_size=128, n_routed_experts=4, num_experts_per_tok=3, moe_intermediate_size=32, n_shared_experts=1)
        params.__dict__.update(get_moe_config=lambda _: (4, 3), score_func='sqrtsoftplus', gate_temp=1,
            norm_topk_prob=True, route_scale=config['routed_scaling_factor'], vision_enabled=False,
            expert_dtype='fp4', moe_inter_dim=32, n_shared_experts=1, swiglu_limit=10,
            hc_mult=config['hc_mult'], hc_sinkhorn_iters=config['hc_sinkhorn_iters'], hc_eps=config['hc_eps'])
    class Linear(torch.nn.Linear):
        def __init__(self, inp, out, bias=False, dtype=None):
            self.original_quantized = dtype is None or dtype == torch.float4_e2m1fn_x2
            actual_dtype = (torch.float32 if dtype == torch.float32 else torch.bfloat16) if args.production_precision else torch.float32
            super().__init__(inp, out, bias=bias, dtype=actual_dtype)
        def forward(self, x):
            if args.production_precision and self.original_quantized:
                x = quantize(x.float(), 0).to(x.dtype)
            return F.linear(x, self.weight)
    def act_quant(x, *args):
        x.copy_(quantize(x.float(), 0)); return x
    def fp4_act_quant(x, block_size=32, inplace=True, scale_dtype=None):
        x.copy_(quantize(x.float(), 2 if block_size == 16 else 1)); return x
    def sparse(q, kv, sinks, indices, scale):
        result = torch.zeros_like(q)
        for token in range(q.shape[1]):
            picks = indices[0, token]
            values = kv[0, picks[picks >= 0].long()]
            scores = q[0, token].float() @ values.float().T * scale
            probs = torch.cat([scores, sinks[:, None]], -1).softmax(-1)[:, :-1]
            result[0, token] = probs @ values.float()
        return result
    def sinkhorn(mixes, scale, base, hc, iterations, eps):
        pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2*hc] * scale[1] + base[hc:2*hc])
        comb = (mixes[..., 2*hc:] * scale[2] + base[2*hc:]).unflatten(-1, (hc, hc)).softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(iterations-1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb
    shared = SimpleNamespace(compress_kv=None, index_k=None, topk_idxs=None, candidates=None)
    scope = dict(torch=torch, nn=torch.nn, F=F, math=math, lru_cache=lru_cache, ModelArgs=SimpleNamespace,
                 Linear=Linear, ColumnParallelLinear=Linear, RowParallelLinear=Linear, world_size=1,
                 shared_attn=shared, act_quant=act_quant, fp4_act_quant=fp4_act_quant, sparse_attn=sparse,
                 fp8_block_size=32, fp4_block_size=32, scale_fmt='ue8m0', scale_dtype=torch.float32, rank=0, linear=F.linear, contextmanager=contextmanager,
                 hc_split_sinkhorn=sinkhorn, EngramLayout=object)
    names = {'RMSNorm', 'Compressor', 'Indexer', 'Attention', 'precompute_freqs_cis', 'apply_rotary_emb',
             'get_window_topk_idxs', 'select_candidate_blocks'}
    if args.full_block: names.update({'Block', 'Expert', 'Gate', 'MoE', 'set_dtype'})
    body = [node for node in ast.parse(source).body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), 'official-v41-attention', 'exec'), scope)
    torch.set_num_threads(1); torch.manual_seed(416)
    if args.production_precision: torch.set_default_dtype(torch.bfloat16)
    layers = [scope['Block' if args.full_block else 'Attention'](i, params) for i in range(8)]
    weights = {}
    mapping = {'wq_a.weight': 'attn_q_a', 'q_norm.weight': 'attn_q_a_norm', 'wq_b.weight': 'attn_q_b',
        'wkv.weight': 'attn_kv', 'kv_norm.weight': 'attn_kv_a_norm', 'wo_a.weight': 'attn_output_a',
        'wo_b.weight': 'attn_output_b', 'attn_sink': 'attn_sinks', 'compressor.wkv.weight': 'attn_compressor_kv',
        'compressor.wgate.weight': 'attn_compressor_gate', 'compressor.norm.weight': 'attn_compressor_norm',
        'indexer.wk.weight': 'indexer.attn_k', 'indexer.k_norm.weight': 'indexer.k_norm',
        'indexer.wq_b.weight': 'indexer.attn_q_b', 'indexer.weights_proj.weight': 'indexer.proj'}
    def encode(value):
        return {'shape': list(reversed(value.shape)), 'values': value.detach().flatten().tolist(),
                'dtype': 'bf16' if value.dtype == torch.bfloat16 else 'f32'}
    if args.full_block:
        mapping = {'attn.' + key: value for key, value in mapping.items()}
        mapping.update({'attn_norm.weight': 'attn_norm', 'ffn_norm.weight': 'ffn_norm',
            'ffn.gate.weight': 'ffn_gate_inp', 'ffn.gate.bias': 'exp_probs_b',
            'ffn.shared_experts.w1.weight': 'ffn_gate_shexp', 'ffn.shared_experts.w2.weight': 'ffn_down_shexp',
            'ffn.shared_experts.w3.weight': 'ffn_up_shexp'})
        for sub in ['attn', 'ffn']:
            for suffix in ['fn', 'scale', 'base']: mapping[f'hc_{sub}_{suffix}'] = f'hc_{sub}_{suffix}'
    tie_audit = []
    with torch.no_grad(), index_tie_policy(args.full_block, tie_audit):
        for i, layer in enumerate(layers):
            for name, value in layer.named_parameters():
                value.copy_(torch.randn_like(value) * .1 + (1 if 'norm' in name else 0))
                if not name.startswith('ffn.experts.'):
                    weights[f'blk.{i}.{mapping[name]}'] = encode(value)
            if args.full_block:
                for attr, stem in [('w1', 'gate'), ('w2', 'down'), ('w3', 'up')]:
                    weights[f'blk.{i}.ffn_{stem}_exps'] = encode(torch.stack([getattr(e, attr).weight for e in layer.ffn.experts]))
        if args.full_block:
            cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'Engram')
            fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
            exec(compile(ast.Module(body=[fn], type_ignores=[]), 'official-engram-forward', 'exec'), scope)
            engrams = {}
            for i in config['engram_layer_ids']:
                projector = Linear(32, 64 * (params.hc_mult + 1))
                projector.weight.copy_(torch.randn_like(projector.weight) * .1)
                q, k = torch.randn(params.hc_mult, 64) * .1, torch.randn(params.hc_mult, 64) * .1
                engrams[i] = SimpleNamespace(dim=64, hc_mult=params.hc_mult, eps=params.norm_eps,
                    clamp_value=1e-6, wkv=projector, q_weight=q, k_weight=k)
                for stem, value in [('engram_wkv', projector.weight), ('engram_q_norm', q), ('engram_k_norm', k)]:
                    weights[f'blk.{i}.{stem}'] = encode(value)
        if args.full_block:
            embedding = torch.randn(128, 64) * .3
            final_norm = torch.randn(64) * .1 + 1
            final_head = (torch.randn(128, 64) * .1).float()
            for stem, value in [('token_embd', embedding), ('output_norm', final_norm), ('output', final_head)]:
                weights[stem] = encode(value)
        steps = []
        token_steps = []
        for position in range(10):
            step = []
            if args.full_block:
                token = (position * 17 + 3) % 128
                x = embedding[token].view(1, 1, 1, 64).repeat(1, 1, params.hc_mult, 1)
                pre = torch.zeros(1, 1, params.hc_mult, dtype=torch.float32); pre[..., 0] = 1
            for i, layer in enumerate(layers):
                owner = layers[2 if i < 5 else 5]
                if i >= 2: shared.index_k = (owner.attn if args.full_block else owner).indexer.k_cache
                if args.full_block:
                    record = {'input': encode(x[0]), 'incoming_pre': encode(pre[0])}
                    if i in engrams:
                        rows = torch.randn(1, 1, 4, 8) * .2
                        engrams[i].embed = lambda _: rows
                        x = scope['forward'](engrams[i], x, None, None)
                        record['engram_rows'] = encode(rows.flatten(-2)[0])
                    x, pre = layer(x, position, pre, None)
                    record.update(output=encode(x[0]), next_pre=encode(pre[0]))
                    step.append(record)
                else:
                    x = torch.randn(1, 1, 64) * .3
                    output = layer(x, position)
                    step.append({'input': encode(x[0]), 'output': encode(output[0])})
            steps.append(step)
            if args.full_block:
                hidden = layer.hc_pre(x, pre)
                hidden = (hidden.float() * torch.rsqrt(hidden.float().square().mean(-1, keepdim=True) + params.norm_eps) * final_norm.float()).bfloat16()
                logits = F.linear(hidden.float(), final_head)
                token_steps.append({'token': token, 'logits': encode(logits[0])})
    args.output.write_text(json.dumps({'config': config, 'model_source_sha256': digest, 'scope': __doc__,
        'index_tie_policy': 'ascending_position' if args.full_block else 'torch_unspecified',
        'index_tie_audit': tie_audit, 'production_precision': args.production_precision, 'full_block': args.full_block,
        'absolute_tolerance': 1e-3, 'relative_tolerance': 1e-3, 'weights': weights, 'steps': steps, 'token_steps': token_steps}) + '\n')


if __name__ == '__main__':
    main()
