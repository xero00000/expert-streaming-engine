#!/usr/bin/env python3
"""Trace V4.1 Engram and shifted-HC methods on small initialized tensors.

The model methods are extracted unchanged from the pinned official source.
Sinkhorn uses a portable Torch transcription of the released TileLang kernel.
Attention/FFN outputs here use diagnostic linear maps, not a full-model oracle.
"""
import argparse
import ast
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    provenance = json.loads((args.snapshot / "provenance.json").read_text())
    source = (args.snapshot / "inference/model.py").read_bytes()
    expected_hash = provenance["files"]["inference/model.py"]["sha256"]
    if hashlib.sha256(source).hexdigest() != expected_hash:
        raise ValueError("reference model source hash mismatch")

    def sinkhorn(mixes, scale, base, hc, iterations, eps):
        pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + eps
        post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * scale[1] + base[hc:2 * hc])
        comb = (mixes[..., 2 * hc:] * scale[2] + base[2 * hc:]).unflatten(-1, (hc, hc))
        comb = comb.softmax(-1) + eps
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
        for _ in range(iterations - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + eps)
            comb = comb / (comb.sum(-2, keepdim=True) + eps)
        return pre, post, comb

    tree = ast.parse(source)
    def method(class_name, name):
        body = tree.body if class_name is None else next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body
        fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
        scope = {"torch": torch, "F": F, "hc_split_sinkhorn": sinkhorn, "lru_cache": lru_cache, "math": math}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "pinned-deepseek-method", "exec"), scope)
        return scope[name]

    mixes_fn, pre_fn, post_fn = (method("Block", name) for name in ("hc_mixes", "hc_pre", "hc_post"))
    inject_fn = method("Engram", "forward")
    def encode(tensor):
        tensor = tensor.detach().contiguous().float()
        return {"shape": list(reversed(tensor.shape)), "values": tensor.flatten().tolist()}

    cases = []
    for seed, tokens, mode in ((4101, 1, "random"), (4102, 3, "random"), (4103, 8, "random"),
                              (4104, 3, "zero_gate"), (4105, 2, "all_masked")):
        torch.manual_seed(seed)
        hidden, hc, lookup_width = 64, 4, 32
        x = torch.randn(1, tokens, hc, hidden) * .3
        if mode == "zero_gate":
            x.zero_()
        embeddings = torch.randn(1, tokens, 4, 8) * .2
        projector = torch.nn.Linear(lookup_width, hidden * (hc + 1), bias=False)
        q_weight, k_weight = torch.randn(hc, hidden) * .5, torch.randn(hc, hidden) * .5
        mask = torch.tensor([[i % 3 != 1 for i in range(tokens)]])
        if mode == "all_masked":
            mask.zero_()
        owner = SimpleNamespace(dim=hidden, hc_mult=hc, eps=1e-20, clamp_value=1e-6,
                                embed=lambda _: embeddings, wkv=projector, q_weight=q_weight, k_weight=k_weight)
        injected = inject_fn(owner, x, None, mask)
        # Record the gate independently of output subtraction (zero values can hide an incorrect gate).
        kv = projector(embeddings.flatten(-2))
        key = kv[..., :hidden * hc].unflatten(-1, (hc, hidden))
        rstd = torch.rsqrt(x.square().mean(-1) + owner.eps) * torch.rsqrt(key.square().mean(-1) + owner.eps)
        dot = (x * q_weight * k_weight * key).sum(-1) * rstd * hidden**-.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot)).masked_fill(~mask.unsqueeze(-1), 0)
        block = SimpleNamespace(norm_eps=1e-20, hc_mult=hc, hc_sinkhorn_iters=20, hc_eps=1e-6)
        width = hc * (hc + 2)
        attn_fn = torch.randn(width, hidden * hc) * .04
        ffn_fn = torch.randn(width, hidden * hc) * .04
        attn_scale, ffn_scale = torch.tensor([.7, .8, 1.1]), torch.tensor([.6, .9, 1.2])
        attn_base, ffn_base = torch.randn(width) * .2, torch.randn(width) * .2
        prior = torch.rand(1, tokens, hc)
        attn_weight, ffn_weight = torch.randn(hidden, hidden) * .05, torch.randn(hidden, hidden) * .05
        ap, apo, ac = mixes_fn(block, injected, attn_fn, attn_scale, attn_base)
        attn_input = pre_fn(block, injected, prior)
        attn_output = F.linear(attn_input, attn_weight)
        after_attention = post_fn(block, attn_output, injected, apo, ac)
        fp, fpo, fc = mixes_fn(block, after_attention, ffn_fn, ffn_scale, ffn_base)
        ffn_input = pre_fn(block, after_attention, ap)
        output = post_fn(block, F.linear(ffn_input, ffn_weight), after_attention, fpo, fc)
        wrong_self_pre = pre_fn(block, injected, ap)
        wrong_comb = post_fn(block, attn_output, injected, apo, ac.transpose(-1, -2))
        inputs = {"x": x[0], "embeddings": embeddings.flatten(-2)[0], "key_value_projection": projector.weight,
                  "q_weight": q_weight, "k_weight": k_weight, "text_mask": mask[0].view(tokens, 1, 1),
                  "incoming_pre": prior[0], "attn_fn": attn_fn, "attn_scale": attn_scale, "attn_base": attn_base,
                  "ffn_fn": ffn_fn, "ffn_scale": ffn_scale, "ffn_base": ffn_base,
                  "attn_weight": attn_weight, "ffn_weight": ffn_weight}
        expected = {"engram_output": injected[0], "engram_gate": gate[0].unsqueeze(-1),
                    "attn_pre": ap[0], "attn_post": apo[0], "attn_comb": ac[0], "attn_input": attn_input[0],
                    "after_attention": after_attention[0], "ffn_input": ffn_input[0], "next_pre": fp[0], "block_output": output[0]}
        cases.append({"name": f"{mode}-{tokens}", "inputs": {k: encode(v) for k, v in inputs.items()},
                      "expected": {k: encode(v) for k, v in expected.items()},
                      "negative_controls": {"self_pre_max_difference": (wrong_self_pre - attn_input).abs().max().item(),
                                            "comb_transpose_max_difference": (wrong_comb - after_attention).abs().max().item()}})
    for case in cases:
        if case["name"].startswith("random") and min(case["negative_controls"].values()) < 1e-3:
            raise ValueError("reference fixture does not distinguish required HC semantics")

    compressor = method("Compressor", "forward")
    for ratio in (1, 2):
        torch.manual_seed(4200 + ratio)
        dim = 32
        norm_weight = torch.rand(dim) + .5
        layer = SimpleNamespace(compress_ratio=ratio, wkv=torch.nn.Linear(64, dim, bias=False),
                                wgate=torch.nn.Linear(64, dim, bias=False),
                                norm=lambda value: value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-20) * norm_weight,
                                kv_state=torch.zeros(1, ratio, dim), score_state=torch.full((1, ratio, dim), -torch.inf))
        for start, length in ((0, 1), (0, 5), (5, 1), (6, 1), (7, 1)):
            x = torch.randn(1, length, 64)
            previous = start % ratio if start else 0
            values = torch.cat([layer.kv_state[:, :previous], layer.wkv(x)], dim=1)
            scores = torch.cat([layer.score_state[:, :previous], layer.wgate(x)], dim=1)
            expected_latent = compressor(layer, x, start)
            complete = values.size(1) // ratio * ratio
            inputs = {"values": encode(values[0]), "norm_weight": encode(norm_weight)}
            if ratio > 1:
                inputs["scores"] = encode(scores[0])
            outputs = {}
            if expected_latent is not None:
                outputs["latent"] = encode(expected_latent[0])
            if complete < values.size(1):
                outputs["tail_values"] = encode(values[0, complete:])
                outputs["tail_scores"] = encode(scores[0, complete:])
            cases.append({"kind": "compression", "name": f"compression-r{ratio}-p{start}-n{length}",
                          "ratio": ratio, "inputs": inputs, "expected": outputs})

    frequencies = method(None, "precompute_freqs_cis")
    rotate = method(None, "apply_rotary_emb")
    for original_context, theta, factor in ((0, 10000.0, 1.0), (65536, 160000.0, 16.0)):
        torch.manual_seed(4300 + original_context)
        x = torch.randn(1, 4, 2, 64)
        positions = torch.tensor([0, 1, 127, 128], dtype=torch.int32)
        frequency_table = frequencies(32, 129, original_context, theta, factor, 32, 1)
        forward, inverse = x.clone(), x.clone()
        rotate(forward[..., -32:], frequency_table[positions.long()])
        rotate(inverse[..., -32:], frequency_table[positions.long()], True)
        restored = forward.clone()
        rotate(restored[..., -32:], frequency_table[positions.long()], True)
        cases.append({"kind": "rope", "name": f"rope-original-{original_context}",
                      "rope": {"rotary_dimensions": 32, "original_context": original_context, "theta": theta,
                               "factor": factor, "beta_fast": 32, "beta_slow": 1},
                      "inputs": {"x": encode(x[0]), "positions": {"shape": [4], "dtype": "i32", "values": positions.tolist()}},
                      "expected": {"forward": encode(forward[0]), "inverse": encode(inverse[0]), "restored": encode(restored[0])}})
    result = {"source_revision": provenance["revision"], "model_source_sha256": expected_hash,
              "scope": "Official Engram/HC methods, Torch Sinkhorn transcription; diagnostic sublayers, not full model",
              "norm_eps": 1e-20, "hc_eps": 1e-6, "sinkhorn_iterations": 20,
              "torch_version": torch.__version__, "absolute_tolerance": 2e-5, "relative_tolerance": 2e-4, "cases": cases}
    result["fixture_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    args.output.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"cases": len(cases), "fixture_sha256": result["fixture_sha256"],
                      "negative_controls": [c["negative_controls"] for c in cases if "negative_controls" in c]}, indent=2))


if __name__ == "__main__":
    main()
