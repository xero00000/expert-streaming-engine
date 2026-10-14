#!/usr/bin/env python3
"""Stream a pinned upstream text forward from the original quantized GGUF.

The upstream decoder, attention, MoE and recurrent forward methods execute
unchanged. Only weight access is replaced: one layer is materialized at a
time, experts are decoded on demand, and the output head is read in chunks.
This is an offline correctness oracle, not an inference performance engine.
The artifact's expert pruning and conversion permutations are explicit.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader, GGML_QUANT_SIZES, dequantize
from qwen4exp_upstream_parity import extract_reference, model_text_config, TRANSFORMERS_SHA256
from qwen4exp_reference_trace import extract_reference_namespace
from qwen4exp_layer0_real import canonicalize, cyclic_order


def full_reference(source: bytes) -> dict:
    if hashlib.sha256(source).hexdigest() != TRANSFORMERS_SHA256:
        raise ValueError("reference source hash mismatch")
    ref = extract_reference(source)
    ref.update(extract_reference_namespace(source))
    ref.update(GradientCheckpointingLayer=nn.Module, Callable=Callable,
               ALL_ATTENTION_FUNCTIONS=SimpleNamespace(get_interface=lambda _name, default: default))
    wanted = {"repeat_kv", "eager_attention_forward", "Qwen4ExpTextAttention", "Qwen4ExpTextDecoderLayer"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
    if {n.name for n in nodes} != wanted:
        raise ValueError("missing full decoder definitions")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned_full_decoder.py", "exec"), ref)
    return ref


def reference_execution_metadata() -> dict:
    """Bind reference results to the numerical runtime as well as source/weights."""
    return dict(
        python_executable=sys.executable,
        python_version=sys.version,
        torch_version=str(torch.__version__),
        torch_module_path=torch.__file__,
        torch_configuration_sha256=hashlib.sha256(torch.__config__.show().encode()).hexdigest(),
        numpy_version=np.__version__,
        torch_threads=torch.get_num_threads(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


def reference_output_mixer(module, hidden, chunk_tokens):
    """Keep the final mixer on the same execution chunks as the decoder."""
    if chunk_tokens < 0:
        raise ValueError("chunk tokens must be nonnegative")
    if chunk_tokens == 0:
        return module(hidden)
    return torch.cat([module(hidden[:, start:start+chunk_tokens])
                      for start in range(0, hidden.shape[1], chunk_tokens)], dim=1)


class Weights:
    def __init__(self, paths: list[Path]):
        self.readers = [GGUFReader(str(p), "r") for p in paths]
        self.tensors = {}
        for reader in self.readers:
            for t in reader.tensors:
                if t.name in self.tensors:
                    raise ValueError(f"duplicate weight {t.name}")
                self.tensors[t.name] = t
        self.accesses = []

    def meta(self, name):
        return self.readers[0].fields[name].contents()

    def read(self, name, rows=None, expert=None):
        t = self.tensors[name]
        shape = tuple(reversed(t.shape.tolist()))
        block, size = GGML_QUANT_SIZES[t.tensor_type]
        width = shape[-1]
        if width % block:
            raise ValueError(f"unaligned row for {name}")
        raw = t.data.view(np.uint8).reshape(*shape[:-1], width // block * size)
        if expert is not None:
            if len(shape) != 3 or not 0 <= expert < shape[0]:
                raise ValueError("expert index outside tensor")
            raw = raw[expert]
            shape = shape[1:]
        if rows is not None:
            raw = raw[rows]
            shape = (*raw.shape[:-1], width)
        # Preserve typed F32/F16 inputs to the public decoder. Quantized inputs
        # are byte rows; BF16 uses its registered decoder.
        if t.tensor_type.name == "F32":
            data = np.ascontiguousarray(raw).view(np.float32)
        elif t.tensor_type.name == "F16":
            data = np.ascontiguousarray(raw).view(np.float16).astype(np.float32)
        else:
            data = dequantize(np.ascontiguousarray(raw), t.tensor_type)
        result = torch.from_numpy(np.ascontiguousarray(data.reshape(shape)).copy())
        if not torch.isfinite(result).all():
            raise ValueError(f"nonfinite decoded weights {name}")
        self.accesses.append(dict(name=name, expert=expert, shape=list(result.shape),
                                  type=t.tensor_type.name, payload_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
        return result


def assign(module, attribute, value):
    setattr(module, attribute, nn.Parameter(value, requires_grad=False))


def linear(module, store, name, transform=None):
    value = store.read(name)
    if transform is not None:
        value = torch.from_numpy(canonicalize(transform, value.numpy(), cyclic_order(16, 48)).copy())
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if tuple(value.shape) != tuple(module.weight.shape):
        raise ValueError(f"linear shape mismatch {name}: {tuple(value.shape)} != {tuple(module.weight.shape)}")
    assign(module, "weight", value)


def norm(module, store, name):
    # GGUF stores effective gamma; upstream RMSNorm adds one to its parameter.
    value = store.read(name) - 1
    if tuple(value.shape) != tuple(module.weight.shape):
        raise ValueError(f"norm shape mismatch {name}")
    assign(module, "weight", value)


def gr(module, store, prefix):
    norm(module.hc_norm, store, prefix + "norm.weight")
    linear(module.input_mix_weight_down, store, prefix + "down.weight")
    linear(module.input_mix_weight_up, store, prefix + "up.weight")
    if module.block_inject_weight is not None:
        linear(module.block_inject_weight, store, prefix + "inject.weight")


class ExpertBank:
    def __init__(self, store, layer, roles, cache=False):
        self.store, self.layer, self.roles = store, layer, roles
        # Optional reference-only reuse, scoped to the currently loaded layer.
        self.cache = {} if cache else None

    def __getitem__(self, index):
        if self.cache is not None and int(index) in self.cache:
            return self.cache[int(index)]
        parts = [self.store.read(f"blk.{self.layer}.ffn_{role}_exps.weight", expert=int(index)) for role in self.roles]
        result = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        if self.cache is not None:
            self.cache[int(index)] = result
        return result


class RowEmbedding(nn.Module):
    def __init__(self, store, name):
        super().__init__()
        self.store, self.name = store, name
        self.register_buffer("weight", torch.empty(0))  # upstream queries only device
        self.last_ids = None

    def forward(self, ids):
        self.last_ids = ids.detach().clone()
        return self.store.read(self.name, rows=ids.cpu().numpy())


class SplitProjection(nn.Module):
    def __init__(self, q, k):
        super().__init__()
        self.q, self.k = q, k

    def forward(self, x):
        return torch.cat((nn.functional.linear(x, self.q), nn.functional.linear(x, self.k)), dim=-1)


class ReferenceCache:
    """Cache storage for a contiguous, unpadded text reference batch.

    No eviction, quantization or placement logic is borrowed from ESE.
    Convolution slot 0 keeps K-1 inputs, slot 1 the PLE dilated history,
    and slot 2 the n-gram token history.
    """
    def __init__(self, layers):
        self.layers = [SimpleNamespace(record_past=False, conv_states=[None]*3,
                                       recurrent_states=[None], keys=None, values=None,
                                       indexer=None) for _ in range(layers)]

    def has_previous_state(self, layer, state_idx=0):
        return self.layers[layer].conv_states[state_idx] is not None

    def update_conv_state(self, current, layer, conv_kernel_size, state_idx=0):
        prior = self.layers[layer].conv_states[state_idx]
        joined = current if prior is None else torch.cat((prior, current), dim=-1)
        length = conv_kernel_size - 1 if state_idx == 0 else conv_kernel_size
        stored = nn.functional.pad(joined, (max(0, length-joined.shape[-1]), 0))[..., -length:]
        self.layers[layer].conv_states[state_idx] = stored.detach().clone()
        return joined

    def update_recurrent_state(self, state, layer):
        self.layers[layer].recurrent_states[0] = state.detach().clone()

    def update_indexer(self, current, layer):
        previous = self.layers[layer].indexer
        self.layers[layer].indexer = current if previous is None else torch.cat((previous, current), dim=1)
        return self.layers[layer].indexer

    def update(self, keys, values, layer):
        entry = self.layers[layer]
        entry.keys = keys if entry.keys is None else torch.cat((entry.keys, keys), dim=2)
        entry.values = values if entry.values is None else torch.cat((entry.values, values), dim=2)
        return entry.keys, entry.values


def load_layer(ref, cfg, store, index, cache_experts=False):
    with torch.device("meta"):
        layer = ref["Qwen4ExpTextDecoderLayer"](cfg, index)
    p = f"blk.{index}."
    gr(layer.attn_hyper_connection, store, p + "hc_attn_")
    gr(layer.mlp_hyper_connection, store, p + "hc_ffn_")
    moe = layer.mlp
    assign(moe.gate, "weight", store.read(p + "ffn_gate_inp.weight"))
    for attribute, roles in (("gate_up_proj", ("gate", "up")), ("down_proj", ("down",))):
        delattr(moe.experts, attribute)
        setattr(moe.experts, attribute, ExpertBank(store, index, roles, cache=cache_experts))
    for role in ("gate", "up", "down"):
        linear(getattr(moe.shared_expert, role + "_proj"), store, p + f"ffn_{role}_shexp.weight")
    linear(moe.shared_expert_gate, store, p + "ffn_gate_inp_shexp.weight")
    if cfg.layer_types[index] == "linear_attention":
        a = layer.linear_attn
        mapping = {"in_proj_qkv": ("attn_qkv.weight", "in_proj_qkv"),
                   "in_proj_z": ("attn_gate.weight", "in_proj_z"),
                   "in_proj_a": ("ssm_alpha.weight", "in_proj_a"),
                   "in_proj_b": ("ssm_beta.weight", "in_proj_b"),
                   "out_proj": ("ssm_out.weight", "out_proj")}
        for role, (suffix, transform) in mapping.items():
            linear(getattr(a, role), store, p + suffix, transform)
        order = cyclic_order(16, 48)
        def canonical(role, suffix):
            return torch.from_numpy(canonicalize(role, store.read(p + suffix).numpy(), order).copy())
        assign(a.conv1d, "weight", canonical("conv1d", "ssm_conv1d.weight").unsqueeze(1))
        decay = canonical("a_log", "ssm_a")
        if not (decay < 0).all():
            raise ValueError("invalid stored decay")
        assign(a, "A_log", (-decay).log())
        assign(a, "dt_bias", canonical("dt_bias", "ssm_dt.bias"))
        assign(a.norm, "weight", store.read(p + "ssm_norm.weight"))
    else:
        a = layer.self_attn
        for role, suffix in (("q", "q"), ("k", "k"), ("v", "v"), ("o", "output")):
            linear(getattr(a, role + "_proj"), store, p + f"attn_{suffix}.weight")
        norm(a.q_norm, store, p + "attn_q_norm.weight")
        norm(a.k_norm, store, p + "attn_k_norm.weight")
        a.indexer.index_qk_proj = SplitProjection(store.read(p + "indexer.q_proj.weight"), store.read(p + "indexer.k_proj.weight"))
        norm(a.indexer.q_layernorm, store, p + "indexer.q_norm.weight")
        norm(a.indexer.k_layernorm, store, p + "indexer.k_norm.weight")
    if layer.ple is not None:
        ple = layer.ple
        linear(ple.key_proj, store, p + "ple_key.weight")
        linear(ple.value_proj, store, p + "ple_value.weight")
        for role in ("key", "query", "conv"):
            norm(getattr(ple, "norm_" + role), store, p + f"ple_norm_{role}.weight")
        assign(ple.conv1d, "weight", store.read(p + "ple_conv1d.weight").unsqueeze(1))
        emb = ple.ple_embedding
        emb.layer_multipliers = ref["_build_layer_multipliers"](cfg.vocab_size, cfg.ngram_size, 0, cfg.seed)
        emb.ngram_heads_vocab_sizes = torch.tensor(emb.head_vocab_sizes, dtype=torch.long)
        emb.ngram_heads_offsets = torch.tensor(emb.head_offsets, dtype=torch.long)
        for field, value in (("layer_multipliers", emb.layer_multipliers.tolist()),
                             ("head_vocab_sizes", emb.head_vocab_sizes), ("head_offsets", emb.head_offsets)):
            if value != store.meta("qwen4exp.ple." + field):
                raise ValueError(f"upstream hash geometry differs: {field}")
        emb.ngram_embedding = RowEmbedding(store, "per_layer_token_embd.weight")
    remaining = [name for name, value in layer.named_parameters() if value.is_meta]
    if remaining:
        raise ValueError(f"unloaded parameters: {remaining}")
    return layer.eval()


def run(args):
    torch.set_num_threads(args.threads)
    ref = full_reference(args.source.read_bytes())
    store = Weights(args.gguf)
    conf = model_text_config()
    # This coder artifact has 256 retained experts, rather than upstream's 512.
    conf.update(num_experts=int(store.meta("qwen4exp.expert_count")),
                norm_topk_prob=True, seed=conf.get("seed", 1234),
                intermediate_size=conf["shared_expert_intermediate_size"], _attn_implementation="eager")
    cfg = SimpleNamespace(**conf)
    ids = torch.tensor([json.loads(args.input_ids.read_text())[:args.tokens]], dtype=torch.long)
    if ids.shape[1] != args.tokens or args.tokens < 1:
        raise ValueError("input count mismatch")
    args.output.mkdir(exist_ok=False, parents=True)
    torch.set_grad_enabled(False)
    started = time.monotonic()
    hidden = store.read("token_embd.weight", rows=ids.numpy()).repeat(1, 1, cfg.hc_count)
    n = ids.shape[1]
    dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    phase = torch.arange(n).float()[:, None] / cfg.rope_parameters["rope_theta"] ** (torch.arange(0, dim, 2).float() / dim)
    phase = torch.cat((phase, phase), dim=-1)[None]
    position = (phase.cos(), phase.sin())
    mask = torch.where(torch.ones(n, n, dtype=torch.bool).tril(), 0.0, torch.finfo(torch.float32).min)[None, None]
    trace = []
    chunk_tokens = getattr(args, "chunk_tokens", 0)
    cache = ReferenceCache(cfg.num_hidden_layers)
    def save(name, value):
        value = value.detach().cpu().contiguous()
        if not torch.isfinite(value).all():
            raise ValueError(f"nonfinite reference activation {name}")
        value.numpy().tofile(args.output / (name + ".bin"))
        trace.append(dict(name=name, shape=list(value.shape), dtype=str(value.dtype)))
    save("token_embeddings", hidden[..., :cfg.hidden_size])
    for index in range(cfg.num_hidden_layers):
        layer = load_layer(ref, cfg, store, index, cache_experts=chunk_tokens > 0)
        observations = {}
        def observe(name, value):
            if chunk_tokens:
                observations.setdefault(name, []).append(value.detach())
            else:
                save(name, value)
        def observe_gr(_m, _a, out, name):
            observe(name, out[0])
        layer.attn_hyper_connection.register_forward_hook(lambda m,a,o,i=index: observe_gr(m,a,o,f"attn_input-{i}"))
        layer.mlp_hyper_connection.register_forward_hook(lambda m,a,o,i=index: observe_gr(m,a,o,f"ffn_input-{i}"))
        layer.mlp_hyper_connection.register_forward_pre_hook(lambda m,a,i=index: observe(f"attn_residual-{i}",a[0]))
        def observe_router(_m, _a, out, i=index):
            observe(f"route_logits-{i}", out[0])
            observe(f"route_ids-{i}", out[2])
            observe(f"route_weights-{i}", out[1])
        layer.mlp.gate.register_forward_hook(observe_router)
        attention = layer.linear_attn if cfg.layer_types[index] == "linear_attention" else layer.self_attn
        attention.register_forward_hook(lambda m,a,o,i=index: observe(f"attn_output-{i}",o if torch.is_tensor(o) else o[0]))
        layer.mlp.register_forward_hook(lambda m,a,o,i=index: observe(f"ffn_output-{i}",o))
        if chunk_tokens:
            parts = []
            for start in range(0, n, chunk_tokens):
                end = min(n, start + chunk_tokens)
                parts.append(layer(hidden[:, start:end],
                    # QSA rotates pooled historical keys too; its forward
                    # method slices the current query positions itself.
                    (position[0][:, :end], position[1][:, :end]),
                    attention_mask=mask[:, :, start:end, :end],
                    ple_input_ids=ids[:, start:end], past_key_values=cache))
            hidden = torch.cat(parts, dim=1)
            for name, values in observations.items():
                save(name, torch.cat(values, dim=1))
        else:
            hidden = layer(hidden, position, attention_mask=mask, ple_input_ids=ids, past_key_values=cache)
        save(f"layer_output-{index}", hidden)
        state = cache.layers[index]
        for slot, value in enumerate(state.conv_states):
            if value is not None: save(f"conv_state-{index}-{slot}", value)
        for slot, value in enumerate(state.recurrent_states):
            if value is not None: save(f"recurrent_state-{index}-{slot}", value)
        for role in ("keys", "values", "indexer"):
            value = getattr(state, role)
            if value is not None: save(f"{role}_state-{index}", value)
        del layer
        progress = dict(completed_layers=index+1, seconds=time.monotonic()-started,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.output / "progress.json").write_text(json.dumps(progress, indent=2))
        print(json.dumps(progress), flush=True)
    with torch.device("meta"):
        mixer = ref["Qwen4ExpTextGatedResidual"](cfg, use_combine=False)
    gr(mixer, store, "output_hc_")
    hidden = reference_output_mixer(mixer, hidden, chunk_tokens)
    save("output_input", hidden)
    logits = []
    for start in range(0, cfg.vocab_size, args.output_rows):
        weight = store.read("output.weight", rows=slice(start, min(start+args.output_rows, cfg.vocab_size)))
        if chunk_tokens:
            logits.append(torch.cat([nn.functional.linear(hidden[:, i:i+chunk_tokens], weight)
                                     for i in range(0, n, chunk_tokens)], dim=1))
        else:
            logits.append(nn.functional.linear(hidden, weight))
    save("logits", torch.cat(logits, dim=-1))
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2))
    (args.output / "weights.json").write_text(json.dumps(store.accesses, indent=2))
    (args.output / "qualification.json").write_text(json.dumps(dict(
        source_sha256=TRANSFORMERS_SHA256, input_ids=ids.tolist(), layers=cfg.num_hidden_layers,
        pruned_experts=cfg.num_experts, reference_complete=True, native_parity=False,
        chunk_tokens=chunk_tokens,
        output_mixer_chunk_tokens=chunk_tokens,
        execution=reference_execution_metadata(),
        seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, nargs="+", required=True)
    parser.add_argument("--input-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-rows", type=int, default=512)
    parser.add_argument("--chunk-tokens", type=int, default=0,
                        help="upstream forward chunk size; 0 uses the full prefix, 1 tests cached single-token execution")
    args = parser.parse_args()
    if args.output_rows <= 0:
        parser.error("output rows must be positive")
    if args.chunk_tokens < 0:
        parser.error("chunk tokens must be nonnegative")
    run(args)
