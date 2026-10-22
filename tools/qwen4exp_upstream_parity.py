#!/usr/bin/env python3
"""Generate small operator goldens by executing pinned Transformers Qwen4Exp classes.

Covers the Gated Residual, PLE, QSA indexer, top-k router / sparse MoE, the
Gated DeltaNet recurrence, and the output-head logits path, so reference
traces exist for every Qwen4Exp component without running the full model.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Unpack

import torch
from torch import nn
import torch.nn.functional as F

TRANSFORMERS_REVISION = "fc5c5bde8e656dad91cbf34e61940d984b1c7b91"
TRANSFORMERS_SHA256 = "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459"
TRANSFORMERS_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{TRANSFORMERS_REVISION}/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"
)
H, S, R, T, E, K, D = 5, 4, 3, 2, 6, 4, 3
EPS = 1.0e-6

def model_text_config() -> dict:
    # Pin configuration as well as modeling code: the class's SiLU fallback
    # differs from the shipped model's explicitly configured sigmoid gate.
    path = Path(__file__).resolve().parents[1] / "docs/qwen4exp/pinned-model-config.json"
    raw = path.read_bytes()
    expected = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("model configuration does not match the pinned SHA-256")
    return json.loads(raw)["text_config"]

def model_output_gate() -> str:
    return model_text_config()["output_gate_type"]

def generate_qsa_rotary_reference(ref: dict[str, Any]) -> dict[str, Any]:
    config = model_text_config()
    width, heads = 128, 4
    rotary = int(config["head_dim"] * config["rope_parameters"]["partial_rotary_factor"])
    base = config["rope_parameters"]["rope_theta"]
    positions = torch.tensor([32, 2051, 4096], dtype=torch.long)
    x = values(len(positions) * heads * width, 0.02, 0.31).reshape(1, len(positions), heads, width)
    def rotate(dim):
        frequency = positions.float()[:, None] / (base ** (torch.arange(0, dim, 2).float() / dim))
        phase = torch.cat((frequency, frequency), -1)[None]
        return ref["apply_rotary_pos_emb"](x, cos=phase.cos(), sin=phase.sin(), unsqueeze_dim=2)
    expected = rotate(rotary)
    wrong = rotate(width)
    assert float((expected - wrong).abs().max()) > 1e-3
    assert torch.equal(x[..., rotary:], expected[..., rotary:])
    return {"width": width, "heads": heads, "rotary_dims": rotary, "freq_base": base,
            "positions": positions.tolist(), "input": as_list(x), "expected": as_list(expected),
            "full_width_control_max_abs": float((expected - wrong).abs().max())}


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_source(location: str) -> bytes:
    if location.startswith(("https://", "http://")):
        with urllib.request.urlopen(location, timeout=30) as response:
            return response.read()
    return Path(location).read_bytes()


def extract_reference(source: bytes) -> dict[str, Any]:
    tree = ast.parse(source, filename="pinned_modeling_qwen4_exp.py")
    wanted = {
        "Qwen4ExpTextRMSNorm",
        "Qwen4ExpTextRMSNormGated",
        "Qwen4ExpTextGatedDeltaNet",
        "apply_mask_to_padding_states",
        "causal_conv1d_fn",
        "causal_conv1d_update",
        "Qwen4ExpTextGatedResidual",
        "Qwen4ExpTextPLELayer",
        "Qwen4ExpTextQSAIndexer",
        "Qwen4ExpTextMLP",
        "Qwen4ExpTextExperts",
        "Qwen4ExpTextTopKRouter",
        "Qwen4ExpTextSparseMoeBlock",
        "apply_rotary_pos_emb",
        "rotate_half",
        "l2norm",
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
    }
    selected = [
        node for node in tree.body
        if (isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in wanted)
    ]
    if {node.name for node in selected} != wanted:
        raise ValueError("pinned Transformers source lacks required Qwen4Exp definitions")
    module = ast.Module(body=selected, type_ignores=[])

    # The pinned module decorates kernels/experts with hub dispatch wrappers and
    # reads activations from a shared registry. None of that is relevant to a
    # CPU golden, so the wrappers become identity decorators and ACT2FN is a
    # small local table of the activations qwen4exp uses.
    def passthrough(*args: Any, **_kwargs: Any):
        # Handles both decorator forms in the pinned source: the bare
        # ``@use_experts_implementation`` (a single non-string argument that is
        # the class itself) and factories with lists of kernel functions or
        # ``@use_kernel_forward_from_hub("RMSNormGated")`` (string arguments).
        if len(args) == 1 and callable(args[0]):
            return args[0]

        def decorator(obj: Any) -> Any:
            return obj
        return decorator

    act2fn = {
        "silu": F.silu,
        "gelu": F.gelu,
        "gelu_pytorch_tanh": lambda x: F.gelu(x, approximate="tanh"),
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh,
    }
    namespace: dict[str, Any] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "math": math,
        "ACT2FN": act2fn,
        "Cache": Any,
        "Unpack": Unpack,
        "TransformersKwargs": dict,
        "Qwen4ExpTextConfig": Any,
        "use_experts_implementation": passthrough,
        "use_kernelized_func": passthrough,
        "use_kernel_func_from_hub_with_fallback": passthrough,
        "use_kernel_forward_from_hub": passthrough,
        "force_accelerate_hooks": passthrough,
    }
    exec(compile(module, "pinned_modeling_qwen4_exp.py", "exec"), namespace)
    return namespace


def values(count: int, scale: float, phase: float) -> torch.Tensor:
    i = torch.arange(count, dtype=torch.float32)
    return scale * (torch.sin(i * 0.37 + phase) + 0.4 * torch.cos(i * 0.19 - phase))


def as_list(tensor: torch.Tensor) -> list[float]:
    return tensor.detach().to(torch.float32).contiguous().view(-1).tolist()


class StaticEmbedding(nn.Module):
    def __init__(self, embeddings: torch.Tensor):
        super().__init__()
        self.register_buffer("embeddings", embeddings)

    def forward(self, _input_ids: torch.Tensor, _cache: Any) -> torch.Tensor:
        return self.embeddings


class ReferenceCache:
    def __init__(self, prior: torch.Tensor, state_length: int):
        self.prior = prior
        self.state_length = state_length
        self.next_state: torch.Tensor | None = None

    def update_conv_state(self, current: torch.Tensor, _layer: int, state_idx: int, conv_kernel_size: int):
        if state_idx != 1 or conv_kernel_size != self.state_length:
            raise ValueError("unexpected PLE convolution-state request")
        history = torch.cat((self.prior, current), dim=-1)
        self.next_state = history[..., -self.state_length :].clone()
        return history


class IndexerCache:
    """Stands in for the HF cache's indexer state with a fixed raw-key matrix."""

    def __init__(self, keys: torch.Tensor):
        self.keys = keys

    def update_indexer(self, _raw_keys: torch.Tensor, _layer: int) -> torch.Tensor:
        return self.keys


def generate_qsa_reference(ref: dict[str, Any]) -> dict[str, Any]:
    """Run the pinned QSA indexer and record scores, selection order, and mask."""
    heads, head_dim, ratio, top = 2, 3, 2, 2
    blocks, queries, kv_length = 4, 2, 8
    rows = blocks * ratio + 1
    mask_floor = -1.0e29  # anything below this is treated as blocked

    indexer_type = ref["Qwen4ExpTextQSAIndexer"]
    indexer = indexer_type(
        SimpleNamespace(
            indexer_n_heads=heads,
            indexer_kv_heads=1,
            indexer_head_dim=head_dim,
            indexer_compress_ratio=ratio,
            indexer_budget=top * ratio,
            rms_norm_eps=EPS,
            hidden_size=heads * head_dim + head_dim,
        ),
        0,
    )
    # Deterministic projection: identity mapping so the indexer's queries are
    # exactly the ESE test's raw q values (first heads*head_dim features per
    # token) and its token keys are the cached raw-key rows. The key norm gets
    # ESE's effective gamma (upstream stores raw weight; gamma = 1 + weight).
    with torch.no_grad():
        indexer.index_qk_proj.weight.copy_(
            torch.eye(heads * head_dim + head_dim)
        )
        indexer.k_layernorm.weight.copy_(values(head_dim, 0.09, 0.41))
    raw_keys = values(rows * head_dim, 0.45, -0.27).reshape(rows, head_dim)
    query_positions = [4, 6]
    query_raw = values(heads * head_dim * queries, 0.5, 0.33)
    hidden = torch.stack(
        [torch.cat((query_raw[heads * head_dim * t : heads * head_dim * (t + 1)], raw_keys[p]))
         for t, p in enumerate(query_positions)]
    ).unsqueeze(0)
    attention_mask = torch.full(
        (1, 1, queries, kv_length), torch.finfo(torch.float32).min
    )
    for t, p in enumerate(query_positions):
        attention_mask[0, 0, t, : p + 1] = 0.0
    # Identity rotary tables: the indexer's RoPE application becomes exact
    # identity, so this fixture pins pooling, norms, scoring, top-k, tail
    # handling, and mask construction but not the rotation itself.
    cos = torch.ones(1, kv_length, head_dim)
    sin = torch.zeros(1, kv_length, head_dim)

    recorded: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_topk = torch.Tensor.topk

    def recording_topk(tensor: torch.Tensor, k: int, *args: Any, **kwargs: Any):
        result = original_topk(tensor, k, *args, **kwargs)
        recorded.append((tensor.detach().clone(), result.indices.detach().clone()))
        return result

    torch.Tensor.topk = recording_topk  # type: ignore[method-assign]
    try:
        mask_out = indexer(
            hidden, (cos, sin), attention_mask, IndexerCache(raw_keys.unsqueeze(0))
        )
    finally:
        torch.Tensor.topk = original_topk  # type: ignore[method-assign]

    if tuple(mask_out.shape) != (1, 1, queries, kv_length):
        raise ValueError(f"unexpected pinned indexer mask shape {tuple(mask_out.shape)}")

    per_query = []
    for q in range(queries):
        row = mask_out[0, 0, q]
        selected = [int(i) for i in range(kv_length) if row[i].item() > mask_floor]
        scores, order = recorded[q]
        block_order = [int(i) for i in order.view(-1).tolist()]
        chosen = [
            b for b in range(blocks)
            if all(b * ratio + o in selected for o in range(ratio))
        ]
        if sorted(chosen) != sorted(block_order):
            raise ValueError("recorded top-k order disagrees with the selected-token mask")
        tail = sorted(set(selected) - {b * ratio + o for b in chosen for o in range(ratio)})
        ordered_tokens = [b * ratio + o for b in block_order for o in range(ratio)] + tail
        if sorted(ordered_tokens) != selected:
            raise ValueError("selected tokens are not exactly the chosen blocks plus tail")
        per_query.append(
            {
                "query_position": query_positions[q],
                "scores": scores.view(-1).tolist(),
                "block_order": block_order,
                "selected_tokens": ordered_tokens,
                "mask_row": row.view(-1).tolist(),
            }
        )

    return {
        "heads": heads,
        "kv_heads": 1,
        "head_dim": head_dim,
        "compress_ratio": ratio,
        "top_blocks": top,
        "blocks": blocks,
        "queries": queries,
        "kv_length": kv_length,
        "raw_key_rows": rows,
        "mask_floor": mask_floor,
        "identity_rope": True,
        "per_query": per_query,
    }


def generate_moe_reference(ref: dict[str, Any]) -> dict[str, Any]:
    """Run the pinned top-k router and sparse MoE block; record expert choice."""
    experts, top, hidden, intermediate, shared = 6, 2, 5, 4, 3
    tokens = 3
    config = SimpleNamespace(
        num_experts=experts,
        num_experts_per_tok=top,
        norm_topk_prob=True,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        shared_expert_intermediate_size=shared,
        hidden_act="silu",
    )
    router_weight = values(experts * hidden, 0.6, 0.15).reshape(experts, hidden)
    hidden_states = values(tokens * hidden, 0.5, -0.35).reshape(tokens, hidden)

    router = ref["Qwen4ExpTextTopKRouter"](config)
    with torch.no_grad():
        router.weight.copy_(router_weight)
    router_logits, routing_weights, expert_ids = router(hidden_states)

    block = ref["Qwen4ExpTextSparseMoeBlock"](config)
    with torch.no_grad():
        block.gate.weight.copy_(router_weight)
        block.experts.gate_up_proj.copy_(
            values(experts * 2 * intermediate * hidden, 0.2, 0.05).reshape(experts, 2 * intermediate, hidden)
        )
        block.experts.down_proj.copy_(
            values(experts * hidden * intermediate, 0.18, -0.12).reshape(experts, hidden, intermediate)
        )
        block.shared_expert.gate_proj.weight.copy_(
            values(shared * hidden, 0.1, 0.3).reshape(shared, hidden)
        )
        block.shared_expert.up_proj.weight.copy_(
            values(shared * hidden, 0.12, -0.2).reshape(shared, hidden)
        )
        block.shared_expert.down_proj.weight.copy_(
            values(hidden * shared, 0.14, 0.4).reshape(hidden, shared)
        )
        block.shared_expert_gate.weight.copy_(values(hidden, 0.09, 0.25).reshape(1, hidden))
    moe_output = block(hidden_states.unsqueeze(0))

    return {
        "num_experts": experts,
        "top_k": top,
        "norm_topk_prob": True,
        "tokens": tokens,
        "hidden": hidden,
        "input": as_list(hidden_states),
        "router_logits": as_list(router_logits),
        "expert_ids": [[int(i) for i in row] for row in expert_ids.tolist()],
        "routing_weights": [[float(w) for w in row] for row in routing_weights.tolist()],
        "moe_output": as_list(moe_output),
    }


def generate_recurrent_reference(ref: dict[str, Any], *, native: bool = False,
                                 grouped: bool = False, head_dim: int = 128,
                                 low_magnitude: bool = False) -> dict[str, Any]:
    """Run the pinned recurrence; retain the original rectangular oracle and add
    a native 128-wide square-state oracle supported by the CUDA kernel."""
    heads, k_head_dim, v_head_dim, seq = (2, head_dim, head_dim, 5) if native else (2, 3, 2, 5)
    key_heads = heads
    if grouped:
        if not native:
            raise ValueError("grouped fixture requires native geometry")
        heads = 6  # same 1:3 key/value-head ratio as the production 16/48 model
    q = values(seq * key_heads * k_head_dim, 0.5, 0.21).reshape(1, seq, key_heads, k_head_dim)
    k = values(seq * key_heads * k_head_dim, 0.45, -0.17).reshape(1, seq, key_heads, k_head_dim)
    if low_magnitude:
        # Include zero, tiny and ordinary rows; a second normalization must not
        # erase the pinned additive-epsilon behavior on small vectors.
        scales = torch.tensor([0.0, 1e-5, 1e-3, 1.0, 1e-5]).reshape(1, seq, 1, 1)
        q, k = q * scales, k * scales
    input_q, input_k = q, k
    if grouped:
        # Match Qwen4ExpTextGatedDeltaNet.forward's explicit repeat_interleave.
        q = q.repeat_interleave(heads // key_heads, dim=2)
        k = k.repeat_interleave(heads // key_heads, dim=2)
    v = values(seq * heads * v_head_dim, 0.55, 0.33).reshape(1, seq, heads, v_head_dim)
    beta = torch.sigmoid(values(seq * heads, 0.4, 0.1)).reshape(1, seq, heads)
    g = -F.softplus(values(seq * heads, 0.5, -0.2)).reshape(1, seq, heads)

    chunk = ref["torch_chunk_gated_delta_rule"]
    recurrent = ref["torch_recurrent_gated_delta_rule"]
    # Prefill recurrence and the decode-time single-step recurrence, each from a
    # zero initial state, so an ESE implementation can be compared step by step.
    chunk_out, chunk_state = chunk(
        q, k, v, g, beta, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    step_out, step_state = recurrent(
        q, k, v, g, beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    extra = {}
    if native:
        state = None
        outputs, states = [], []
        for token in range(seq):
            output, state = recurrent(
                q[:, token:token + 1], k[:, token:token + 1], v[:, token:token + 1],
                g[:, token:token + 1], beta[:, token:token + 1], initial_state=state,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
            outputs.extend(as_list(output))
            states.append(as_list(state))
        extra = {"single_step_output": outputs, "single_step_states": states,
                 "state_layout": "head,key,value", "input_layout": "token,head,dimension"}
        if grouped:
            extra.update(key_heads=key_heads, repeat_type=0, head_expansion="repeat_interleave")
    return {
        **extra,
        "heads": heads,
        "key_head_dim": k_head_dim,
        "value_head_dim": v_head_dim,
        "sequence_length": seq,
        "use_qk_l2norm": True,
        "query": as_list(input_q),
        "key": as_list(input_k),
        "value": as_list(v),
        "beta": as_list(beta),
        "g": as_list(g),
        "chunked_output": as_list(chunk_out),
        "chunked_final_state": as_list(chunk_state),
        "recurrent_output": as_list(step_out),
        "recurrent_final_state": as_list(step_state),
    }


def generate_recurrent_layer_reference(ref: dict[str, Any]) -> dict[str, Any]:
    """Execute the complete pinned layer with grouped heads and cached decode.

    Hooks observe the actual upstream forward boundaries; native tests consume
    these observations rather than a separately reconstructed Python layer.
    """
    hidden, width, key_heads, heads, tokens, kernel = 8, 128, 2, 6, 5, 4
    config = SimpleNamespace(hidden_size=hidden, linear_num_value_heads=heads,
        linear_num_key_heads=key_heads, linear_key_head_dim=width,
        linear_value_head_dim=width, linear_conv_kernel_dim=kernel,
        hidden_act="silu", rms_norm_eps=1e-6, output_gate_type=model_output_gate(),
        layer_types=["linear_attention"])
    layer = ref["Qwen4ExpTextGatedDeltaNet"](config, 0)
    with torch.no_grad():
        for index, name in enumerate(("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")):
            weight = getattr(layer, name).weight
            weight.copy_(values(weight.numel(), 0.07 + index * 0.01, index * 0.13).reshape_as(weight))
        layer.conv1d.weight.copy_(values(layer.conv1d.weight.numel(), 0.2, 0.11).reshape_as(layer.conv1d.weight))
        layer.dt_bias.copy_(values(heads, 0.3, 0.2))
        layer.A_log.copy_(values(heads, 0.2, -0.3))
        layer.norm.weight.copy_(1.0 + values(width, 0.15, 0.1))
    inputs = values(tokens * hidden, 0.7, 0.22).reshape(1, tokens, hidden)

    class LayerCache:
        def __init__(self):
            self.layers = [SimpleNamespace(record_past=False, conv_states=[None], recurrent_states=[None])]

        def has_previous_state(self, _layer_idx, state_idx=0):
            return self.layers[0].recurrent_states[state_idx] is not None

        def update_conv_state(self, mixed, _layer_idx, conv_kernel_size):
            prior = self.layers[0].conv_states[0]
            joined = mixed if prior is None else torch.cat((prior, mixed), dim=-1)
            self.layers[0].conv_states[0] = F.pad(joined, (max(0, kernel - 1 - joined.shape[-1]), 0))[:, :, -(kernel - 1):].clone()
            return joined

        def update_recurrent_state(self, state, _layer_idx):
            self.layers[0].recurrent_states[0] = state.detach().clone()

    captured = {}
    handles = []
    for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"):
        handles.append(getattr(layer, name).register_forward_hook(
            lambda _module, _args, output, name=name: captured.update({name: as_list(output)})))
    handles.append(layer.norm.register_forward_hook(
        lambda _module, args, output: captured.update(core_output=as_list(args[0]), gated_output=as_list(output))))
    original = {name: ref[name] for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update")}

    def observe_recurrence(function):
        def wrapper(q, k, v, *args, **kwargs):
            captured.update(query=as_list(q), key=as_list(k), value=as_list(v),
                            g=as_list(kwargs["g"]), beta=as_list(kwargs["beta"]))
            output, state = function(q, k, v, *args, **kwargs)
            captured["state"] = as_list(state)
            return output, state
        return wrapper

    def observe_conv(function):
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            captured["convolved"] = as_list(output.transpose(1, 2))
            return output
        return wrapper

    try:
        for name, function in original.items():
            ref[name] = observe_recurrence(function) if "delta_rule" in name else observe_conv(function)
        cache = LayerCache()
        batched = layer(inputs, cache)
        batch_trace = {**captured, "output": as_list(batched),
                       "conv_state": as_list(cache.layers[0].conv_states[0])}
        cache = LayerCache()
        steps = []
        for token in range(tokens):
            captured.clear()
            output = layer(inputs[:, token:token + 1], cache)
            steps.append({**captured, "output": as_list(output),
                          "conv_state": as_list(cache.layers[0].conv_states[0])})
    finally:
        ref.update(original)
        for handle in handles:
            handle.remove()
    return {"hidden": hidden, "width": width, "key_heads": key_heads,
            "heads": heads, "tokens": tokens, "kernel": kernel, "epsilon": 1e-6,
            "output_gate_type": config.output_gate_type,
            "input": as_list(inputs),
            "weights": {name: as_list(getattr(layer, name).weight) for name in
                        ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")},
            "conv_weight": as_list(layer.conv1d.weight), "dt_bias": as_list(layer.dt_bias),
            "a_log": as_list(layer.A_log), "norm_weight": as_list(layer.norm.weight),
            "batched": batch_trace, "steps": steps}


def gguf_recurrent_layer_reference(canonical: dict[str, Any]) -> dict[str, Any]:
    """Transform canonical value-head storage, not reference model semantics.

    This transpose is evidenced by exact original/GGUF alpha, beta and conv
    comparisons. The final layer output must remain the canonical oracle.
    """
    result = copy.deepcopy(canonical)
    kh, heads, width = result["key_heads"], result["heads"], result["width"]
    hidden, kernel = result["hidden"], result["kernel"]
    order = torch.arange(heads).reshape(kh, heads // kh).T.flatten()
    def reorder(data, trailing):
        return as_list(torch.tensor(data).reshape(heads, trailing)[order])
    weights = result["weights"]
    for name in ("in_proj_a", "in_proj_b"):
        weights[name] = reorder(weights[name], hidden)
    weights["in_proj_z"] = reorder(weights["in_proj_z"], width * hidden)
    qkv = weights["in_proj_qkv"]
    offset = 2 * kh * width * hidden
    weights["in_proj_qkv"] = qkv[:offset] + reorder(qkv[offset:], width * hidden)
    out = torch.tensor(weights["out_proj"]).reshape(hidden, heads, width)
    weights["out_proj"] = as_list(out[:, order])
    offset = 2 * kh * width * kernel
    conv = result["conv_weight"]
    result["conv_weight"] = conv[:offset] + reorder(conv[offset:], width * kernel)
    for name in ("dt_bias", "a_log"):
        result[name] = reorder(result[name], 1)
    for trace in (result["batched"], *result["steps"]):
        count = len(trace["output"]) // hidden
        for name, dim in (("in_proj_z", width), ("in_proj_a", 1), ("in_proj_b", 1),
                          ("g", 1), ("beta", 1), ("core_output", width),
                          ("gated_output", width), ("query", width), ("key", width), ("value", width)):
            trace[name] = as_list(torch.tensor(trace[name]).reshape(count, heads, dim)[:, order])
        for name in ("in_proj_qkv", "convolved"):
            data = torch.tensor(trace[name]).reshape(count, width * (2 * kh + heads))
            trace[name] = as_list(torch.cat((data[:, :2 * kh * width],
                data[:, 2 * kh * width:].reshape(count, heads, width)[:, order].reshape(count, heads * width)), dim=1))
        trace["state"] = reorder(trace["state"], width * width)
        conv = trace["conv_state"]
        offset = 2 * kh * width * (kernel - 1)
        trace["conv_state"] = conv[:offset] + reorder(conv[offset:], width * (kernel - 1))
    result.update(repeat_type=1, head_order="cyclic_gguf", source_head_indices=order.tolist())
    return result


def generate_l2_reference(ref: dict[str, Any]) -> dict[str, Any]:
    # Include zero and tiny vectors: ordinary-magnitude recurrence inputs hide
    # the distinction between additive squared-norm epsilon and a norm clamp.
    rows = torch.zeros((3, 128), dtype=torch.float32)
    rows[1, :4] = torch.tensor([1e-5, -2e-5, 3e-5, -4e-5])
    rows[2, :4] = torch.tensor([0.2, -0.4, 0.6, -0.8])
    return {"width": 128, "rows": 3, "epsilon": 1e-6,
            "input": as_list(rows), "output": as_list(ref["l2norm"](rows, eps=1e-6))}


def generate_logits_reference() -> dict[str, Any]:
    """Pin the output-head logits path applied to per-token hidden states.

    The pinned Qwen4ExpForCausalLM computes logits as ``lm_head(last_hidden)``
    with no final normalization of its own, so this records that exact
    projection per token. It is a head-path golden, not an end-to-end
    backbone forward (which requires runnable weights; see B3).
    """
    hidden, vocab, tokens = 6, 8, 4
    head = nn.Linear(hidden, vocab, bias=False)
    with torch.no_grad():
        head.weight.copy_(values(vocab * hidden, 0.35, 0.6).reshape(vocab, hidden))
    hidden_states = values(tokens * hidden, 0.7, -0.5).reshape(1, tokens, hidden)
    logits = head(hidden_states)
    return {
        "tokens": tokens,
        "hidden": hidden,
        "vocab": vocab,
        "final_norm_applied": False,
        "hidden_input": as_list(hidden_states),
        "lm_head_weight": as_list(head.weight),
        "logits": as_list(logits),
    }


def generate(source: bytes) -> dict[str, Any]:
    if hashlib.sha256(source).hexdigest() != TRANSFORMERS_SHA256:
        raise ValueError("pinned Transformers source SHA-256 mismatch")
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    ref = extract_reference(source)
    hs = H * S

    hyper = values(hs * T, 0.7, 0.2).reshape(1, T, hs)
    gamma = values(hs, 0.15, -0.1) + 1.0
    down = values(hs * R, 0.11, 0.5).reshape(R, hs)
    up = values(R * hs, 0.13, -0.3).reshape(hs, R)
    inject = values(hs * S, 0.09, 0.7).reshape(S, hs)
    gr_type = ref["Qwen4ExpTextGatedResidual"]
    gr = gr_type(SimpleNamespace(hc_count=S, hidden_size=H, rms_norm_eps=EPS, hc_lowrank=R))
    output_gr = gr_type(SimpleNamespace(hc_count=S, hidden_size=H, rms_norm_eps=EPS, hc_lowrank=R), use_combine=False)
    with torch.no_grad():
        gr.hc_norm.weight.copy_(gamma - 1.0)
        gr.input_mix_weight_down.weight.copy_(down)
        gr.input_mix_weight_up.weight.copy_(up)
        gr.block_inject_weight.weight.copy_(inject)
        output_gr.hc_norm.weight.copy_(gamma - 1.0)
        output_gr.input_mix_weight_down.weight.copy_(down)
        output_gr.input_mix_weight_up.weight.copy_(up)
    mixed, preserved_hyper, injection = gr(hyper)
    residual = preserved_hyper + (mixed.unsqueeze(-2) * injection.unsqueeze(-1)).flatten(-2)
    final_mixed = output_gr(residual)

    hidden = values(hs * T, 0.7, 0.2).reshape(1, T, hs)
    embeddings = values(E * T, 0.4, -0.6).reshape(1, T, E)
    key_projection = values(E * hs, 0.12, 0.9).reshape(hs, E)
    value_projection = values(E * H, 0.14, -0.8).reshape(H, E)
    key_gamma = values(hs, 0.08, 0.1) + 1.0
    query_gamma = values(hs, 0.07, 0.3) + 1.0
    conv_gamma = values(hs, 0.06, -0.4) + 1.0
    conv_weight = values(K * hs, 0.1, 0.25).reshape(hs, 1, K)
    state_length = (K - 1) * D
    # Match GGML's [channels, history] tensor layout (channel is contiguous).
    prior = values(hs * state_length, 0.3, -0.45).reshape(1, state_length, hs).transpose(1, 2).contiguous()

    ple_type = ref["Qwen4ExpTextPLELayer"]
    ple = ple_type.__new__(ple_type)
    nn.Module.__init__(ple)
    ple.layer_idx = 0
    ple.hidden_size = H
    ple.hc_count = S
    ple.short_conv_state_len = state_length
    ple.ple_embedding = StaticEmbedding(embeddings)
    ple.key_proj = nn.Linear(E, hs, bias=False)
    ple.value_proj = nn.Linear(E, H, bias=False)
    ple.norm_key = ref["Qwen4ExpTextRMSNorm"](hs, group_size=H, eps=EPS)
    ple.norm_query = ref["Qwen4ExpTextRMSNorm"](hs, group_size=H, eps=EPS)
    ple.norm_conv = ref["Qwen4ExpTextRMSNorm"](hs, group_size=H, eps=EPS)
    ple.conv1d = nn.Conv1d(hs, hs, kernel_size=K, groups=hs, dilation=D, bias=False)
    with torch.no_grad():
        ple.key_proj.weight.copy_(key_projection)
        ple.value_proj.weight.copy_(value_projection)
        ple.norm_key.weight.copy_(key_gamma - 1.0)
        ple.norm_query.weight.copy_(query_gamma - 1.0)
        ple.norm_conv.weight.copy_(conv_gamma - 1.0)
        ple.conv1d.weight.copy_(conv_weight)

    captured: dict[str, torch.Tensor] = {}
    ple.norm_conv.register_forward_hook(
        lambda _module, inputs, output: captured.update(norm_input=inputs[0].detach().clone(), normed=output.detach().clone())
    )
    cache = ReferenceCache(prior, state_length)
    ple_output = ple(hidden, torch.zeros((1, T), dtype=torch.long), cache)
    if cache.next_state is None:
        raise ValueError("pinned PLE reference did not update convolution state")

    payload = {
        "schema_version": 1,
        "architecture": "qwen4exp",
        "reference": {
            "repository": "huggingface/transformers",
            "revision": TRANSFORMERS_REVISION,
            "modeling_sha256": TRANSFORMERS_SHA256,
        },
        "fixture": {"hidden": H, "streams": S, "low_rank": R, "tokens": T, "embedding": E,
                    "conv_kernel": K, "conv_dilation": D, "rms_epsilon": EPS},
        "outputs": {
            "gated_residual_mixed": as_list(mixed),
            "gated_residual_injection": as_list(injection),
            "gated_residual_output": as_list(residual),
            "gated_residual_final_mixed": as_list(final_mixed),
            "ple_gated_value": as_list(captured["norm_input"]),
            "ple_gated_value_normed": as_list(captured["normed"]),
            "ple_output": as_list(ple_output),
            # ESE stores [channels, history] in GGML contiguous order.
            "ple_next_state": as_list(cache.next_state.transpose(1, 2)),
        },
        "qsa": generate_qsa_reference(ref),
        "qsa_rotary": generate_qsa_rotary_reference(ref),
        "moe": generate_moe_reference(ref),
        "recurrent": generate_recurrent_reference(ref),
        "recurrent_native": generate_recurrent_reference(ref, native=True),
        "recurrent_grouped": generate_recurrent_reference(ref, native=True, grouped=True),
        "recurrent_scalar": generate_recurrent_reference(ref, native=True, head_dim=32),
        "recurrent_scalar_low_magnitude": generate_recurrent_reference(
            ref, native=True, grouped=True, head_dim=32, low_magnitude=True),
        "recurrent_low_magnitude": generate_recurrent_reference(ref, native=True, low_magnitude=True),
        "recurrent_layer": generate_recurrent_layer_reference(ref),
        "recurrent_l2": generate_l2_reference(ref),
        "logits": generate_logits_reference(),
    }
    payload["recurrent_layer_gguf"] = gguf_recurrent_layer_reference(payload["recurrent_layer"])
    return {**payload, "fixture_sha256": canonical_hash(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=TRANSFORMERS_URL, help="pinned source URL or local file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", type=Path, help="require generated fixture to match this file")
    args = parser.parse_args()
    fixture = generate(load_source(args.source))
    rendered = json.dumps(fixture, indent=2, sort_keys=True) + "\n"
    if args.verify:
        expected = json.loads(args.verify.read_text(encoding="utf-8"))
        if fixture != expected:
            raise ValueError(f"generated upstream parity fixture differs from {args.verify}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
