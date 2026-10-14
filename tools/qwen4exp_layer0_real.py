#!/usr/bin/env python3
"""Run the pinned Qwen4Exp recurrent layer on real layer-0 GGUF weights.

Every existing layer oracle uses synthetic weights.  This tool closes that gap
for the recurrent block without needing a full-model run. Fully resident weights
exceed host memory; feasibility of the existing streaming paths remains open.

It dequantizes the real layer-0 recurrent tensors from the shared GGUF,
un-permutes the cyclic value-head storage back to canonical source order, loads
them into the pinned `Qwen4ExpTextGatedDeltaNet`, and records the reference
trace.  The canonicalization is the exact inverse of the permutation proven by
`qwen4exp_head_order.py` and `qwen4exp_projection_order.py`:

    canonical[order[s]] = stored[s]      (along the value-head axis)

with `order = arange(heads).reshape(key_heads, heads//key_heads).T.ravel()`.

The native side is expected to consume the *stored* GGUF order with the
production `qwen4exp_repeat_type = 1`, so this file records `repeat_type: 1` and
the GGUF payload digests that consumer must read back.

Usage:

    python3 tools/qwen4exp_layer0_real.py \\
        --gguf /home/xero/Desktop/strata/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf \\
        --source /home/xero/.cache/uv/archive-v0/R7qmBgNzLWKHiXSl/transformers/models/qwen4_exp/modeling_qwen4_exp.py \\
        --tokens 5 \\
        --output docs/qwen4exp/native-recurrence-20261007/layer0-real-decode-fixture.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf import GGUFReader, dequantize
from qwen4exp_upstream_parity import (
    TRANSFORMERS_REVISION,
    TRANSFORMERS_SHA256,
    TRANSFORMERS_URL,
    canonical_hash,
    extract_reference,
    load_source,
    model_output_gate,
    values,
)

LAYER = 0
TOKENS = 3
HIDDEN, WIDTH, KEY_HEADS, HEADS, KERNEL = 2560, 128, 16, 48, 4
CHANNELS = WIDTH * (2 * KEY_HEADS + HEADS)
EPSILON = 1.0e-6

# role -> GGUF tensor name, and how its value-head axis is stored.
STORED = {
    "in_proj_qkv": "attn_qkv.weight",
    "in_proj_z": "attn_gate.weight",
    "out_proj": "ssm_out.weight",
    "conv1d": "ssm_conv1d.weight",
    "in_proj_a": "ssm_alpha.weight",
    "in_proj_b": "ssm_beta.weight",
    "dt_bias": "ssm_dt.bias",
    "a_log": "ssm_a",
    "norm_weight": "ssm_norm.weight",
}
# Tensors whose value-head axis is permuted, and the axis within the
# dequantized (source-shaped) tensor.  `norm_weight` is width-indexed and stays.
PERMUTED = {
    "in_proj_qkv": "value-rows",
    "in_proj_z": "heads-first",
    "in_proj_a": "heads-first",
    "in_proj_b": "heads-first",
    "out_proj": "heads-column",
    "conv1d": "value-rows",
    "dt_bias": "heads-first",
    "a_log": "heads-first",
}


def cyclic_order(key_heads: int, heads: int) -> np.ndarray:
    """Source value head held by each stored value head."""
    if key_heads <= 0 or heads <= 0 or heads % key_heads:
        raise ValueError("invalid grouped-head geometry")
    return np.arange(heads).reshape(key_heads, heads // key_heads).T.ravel()


def _scatter(block: np.ndarray, order: np.ndarray, axis: int) -> np.ndarray:
    """canonical[order[s]] = stored[s] along ``axis``."""
    result = np.empty_like(block)
    for stored, source in enumerate(order):
        destination = [slice(None)] * block.ndim
        destination[axis] = int(source)
        source_index = [slice(None)] * block.ndim
        source_index[axis] = stored
        result[tuple(destination)] = block[tuple(source_index)]
    return np.ascontiguousarray(result)


def canonicalize(role: str, data: np.ndarray, order: np.ndarray) -> np.ndarray:
    """Map one stored layer-0 tensor to canonical source order."""
    kind = PERMUTED.get(role)
    if kind is None:
        return data
    if kind == "heads-column":
        return _scatter(data.reshape(data.shape[0], HEADS, WIDTH), order, 1).reshape(data.shape)
    if kind == "value-rows":
        offset = 2 * KEY_HEADS * WIDTH
        if data.shape[0] != offset + HEADS * WIDTH:
            raise ValueError(f"{role} is not a QKV/conv value-channel layout")
        value = data[offset:].reshape(HEADS, WIDTH, -1)
        return np.concatenate((data[:offset], _scatter(value, order, 0).reshape(data.shape[0] - offset, -1)))
    if kind == "heads-first":
        return _scatter(data.reshape(HEADS, -1), order, 0).reshape(data.shape)
    raise ValueError(f"unknown permutation kind {kind}")


def read_gguf_tensors(path: Path) -> tuple[dict, dict]:
    reader = GGUFReader(path, "r")
    if reader.fields["general.architecture"].contents() != "qwen4exp":
        raise ValueError("expected qwen4exp GGUF")
    for field, expected in (("qwen4exp.ssm.group_count", KEY_HEADS),
                            ("qwen4exp.ssm.time_step_rank", HEADS),
                            ("qwen4exp.ssm.state_size", WIDTH)):
        if int(reader.fields[field].contents()) != expected:
            raise ValueError(f"GGUF {field} disagrees with the pinned geometry")
    tensors = {t.name: t for t in reader.tensors}
    data, records = {}, {}
    for role, suffix in STORED.items():
        name = f"blk.{LAYER}.{suffix}"
        if name not in tensors:
            raise ValueError(f"missing GGUF tensor {name}")
        tensor = tensors[name]
        # GGUF stores ne0-major, i.e. the reverse of the source tensor shape.
        shaped = np.ascontiguousarray(
            dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape.tolist()))))
        data[role] = shaped
        records[role] = {"gguf_name": name, "gguf_type": tensor.tensor_type.name,
                         "source_shape": [int(v) for v in shaped.shape],
                         "payload_sha256": hashlib.sha256(tensor.data.tobytes()).hexdigest()}
    expected_shapes = {
        "in_proj_qkv": (HEADS * WIDTH + 2 * KEY_HEADS * WIDTH, HIDDEN),
        "in_proj_z": (HEADS * WIDTH, HIDDEN),
        "in_proj_a": (HEADS, HIDDEN),
        "in_proj_b": (HEADS, HIDDEN),
        "out_proj": (HIDDEN, HEADS * WIDTH),
        "conv1d": (CHANNELS, KERNEL),
        "dt_bias": (HEADS,),
        "a_log": (HEADS,),
        "norm_weight": (WIDTH,),
    }
    for role, shape in expected_shapes.items():
        if data[role].shape != shape:
            raise ValueError(f"{role} has shape {data[role].shape}, expected {shape}")
    return data, records


def build_layer(ref: dict):
    config = SimpleNamespace(hidden_size=HIDDEN, linear_num_value_heads=HEADS,
        linear_num_key_heads=KEY_HEADS, linear_key_head_dim=WIDTH, linear_value_head_dim=WIDTH,
        linear_conv_kernel_dim=KERNEL, hidden_act="silu", rms_norm_eps=EPSILON,
        output_gate_type=model_output_gate(), layer_types=["linear_attention"])
    return ref["Qwen4ExpTextGatedDeltaNet"](config, 0)


def _tensor(data: np.ndarray) -> torch.Tensor:
    """Copy into a writable float32 tensor: GGUF payload views are read-only."""
    return torch.from_numpy(np.array(data, dtype=np.float32, copy=True, order="C"))


def apply_weights(layer, weights: dict, norm_weight: np.ndarray) -> None:
    """Load source-ordered (canonical) tensors into the pinned module."""
    a_stored = weights["a_log"]
    if not np.all(a_stored < 0):
        raise ValueError("ssm_a must hold -exp(A_log) and therefore be negative")
    with torch.no_grad():
        layer.in_proj_qkv.weight.copy_(_tensor(weights["in_proj_qkv"]))
        layer.in_proj_z.weight.copy_(_tensor(weights["in_proj_z"]))
        layer.in_proj_a.weight.copy_(_tensor(weights["in_proj_a"]))
        layer.in_proj_b.weight.copy_(_tensor(weights["in_proj_b"]))
        layer.out_proj.weight.copy_(_tensor(weights["out_proj"]))
        layer.conv1d.weight.copy_(_tensor(weights["conv1d"]).reshape(CHANNELS, 1, KERNEL))
        # GGUF stores -exp(A_log); recover the parameter the pinned module expects.
        layer.A_log.copy_(_tensor(np.log(-a_stored.astype(np.float32))))
        layer.dt_bias.copy_(_tensor(weights["dt_bias"]))
        layer.norm.weight.copy_(_tensor(norm_weight))


class LayerCache:
    """One-layer cache preserving convolution and recurrence across decode calls."""

    def __init__(self):
        self.layers = [SimpleNamespace(record_past=False, conv_states=[None], recurrent_states=[None])]

    def has_previous_state(self, _layer_idx, state_idx=0):
        return self.layers[0].recurrent_states[state_idx] is not None

    def update_conv_state(self, mixed, _layer_idx, conv_kernel_size):
        if conv_kernel_size != KERNEL:
            raise ValueError("cache convolution kernel disagrees with layer geometry")
        prior = self.layers[0].conv_states[0]
        joined = mixed if prior is None else torch.cat((prior, mixed), dim=-1)
        self.layers[0].conv_states[0] = torch.nn.functional.pad(
            joined, (max(0, KERNEL - 1 - joined.shape[-1]), 0))[:, :, -(KERNEL - 1):].clone()
        return joined

    def update_recurrent_state(self, state, _layer_idx):
        self.layers[0].recurrent_states[0] = state.detach().clone()


def run_trace(ref: dict, layer, inputs: torch.Tensor, cache: LayerCache | None = None) -> dict:
    """Execute the pinned layer forward, observing every boundary that matters."""
    captured: dict = {}
    handles = [getattr(layer, name).register_forward_hook(
        lambda _m, _a, output, name=name: captured.update({name: output}))
        for name in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b")]
    handles.append(layer.norm.register_forward_hook(
        lambda _m, args, output: captured.update(core_output=args[0], gated_output=output)))
    originals = {name: ref[name] for name in
                 ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule",
                  "causal_conv1d_fn", "causal_conv1d_update")}

    def observe_recurrence(function):
        def wrapper(q, k, v, *args, **kwargs):
            captured.update(query=q, key=k, value=v, g=kwargs["g"], beta=kwargs["beta"])
            output, state = function(q, k, v, *args, **kwargs)
            captured["state"] = state
            return output, state
        return wrapper

    def observe_conv(function):
        def wrapper(*args, **kwargs):
            output = function(*args, **kwargs)
            captured["convolved"] = output.transpose(1, 2)
            return output
        return wrapper

    try:
        for name, function in originals.items():
            ref[name] = observe_recurrence(function) if "delta_rule" in name else observe_conv(function)
        if cache is None:
            cache = LayerCache()
        with torch.no_grad():
            output = layer(inputs, cache)
        captured["output"] = output
        captured["conv_state"] = cache.layers[0].conv_states[0]
    finally:
        ref.update(originals)
        for handle in handles:
            handle.remove()

    trace = {name: np.ascontiguousarray(value.detach().to(torch.float32).reshape(-1).numpy()).copy()
             for name, value in captured.items()}
    for name, array in trace.items():
        if not np.isfinite(array).all():
            raise ValueError(f"reference trace {name} contains non-finite values")
    return trace


def stored_trace(trace: dict, tokens: int, order: np.ndarray) -> dict:
    """Reorder canonical observations to the native GGUF value-head layout."""
    result = {name: value.copy() for name, value in trace.items()}
    for name, dim in (("in_proj_z", WIDTH), ("in_proj_a", 1), ("in_proj_b", 1),
                      ("g", 1), ("beta", 1), ("core_output", WIDTH),
                      ("gated_output", WIDTH), ("query", WIDTH), ("key", WIDTH), ("value", WIDTH)):
        result[name] = trace[name].reshape(tokens, HEADS, dim)[:, order].reshape(-1)
    offset = 2 * KEY_HEADS * WIDTH
    for name in ("in_proj_qkv", "convolved"):
        data = trace[name].reshape(tokens, CHANNELS)
        result[name] = np.concatenate((data[:, :offset],
            data[:, offset:].reshape(tokens, HEADS, WIDTH)[:, order].reshape(tokens, -1)), axis=1).reshape(-1)
    result["state"] = trace["state"].reshape(HEADS, WIDTH, WIDTH)[order].reshape(-1)
    data = trace["conv_state"].reshape(CHANNELS, KERNEL - 1)
    result["conv_state"] = np.concatenate((data[:offset],
        data[offset:].reshape(HEADS, WIDTH, KERNEL - 1)[order].reshape(-1, KERNEL - 1))).reshape(-1)
    return result


def read_capture(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    """Read one bounded, zero-state layer-0 evaluation from probe stdout."""
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("capture log exceeds 16 MiB diagnostic bound")
    raw = path.read_bytes()
    tensors = {}
    for line in raw.decode().splitlines():
        if not line.startswith("qwen4exp_float_values: "):
            continue
        record = json.loads(line.split(": ", 1)[1])
        name = record["name"]
        if name in tensors:
            raise ValueError(f"duplicate captured tensor {name}")
        data = np.asarray(record["values"], dtype=np.float32)
        if data.ndim != 1 or not np.isfinite(data).all():
            raise ValueError(f"invalid captured tensor {name}")
        tensors[name] = data
    input_name = "qwen4exp_attn_input-0"
    output_name = "qwen4exp_attn_output-0"
    if input_name not in tensors or output_name not in tensors:
        raise ValueError("capture must include layer-0 recurrent input and output")
    count = tensors[input_name].size
    if count % HIDDEN or not 1 <= count // HIDDEN <= 8 or tensors[output_name].size != count:
        raise ValueError("capture input/output violates bounded layer-0 geometry")
    return tensors, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                     "tokens": count // HIDDEN, "state_assumption": "single zero-state prefill evaluation"}


def compare_gated_residuals(gguf: Path, ref: dict, captured: dict, tokens: int) -> dict:
    """Execute pinned GR modules on captured embeddings/residuals with real weights."""
    if "qwen4exp_token_embeddings" not in captured:
        return {}
    reader = GGUFReader(gguf, "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    records, comparisons = {}, {}
    def weights(prefix, role):
        name = f"blk.0.hc_{prefix}_{role}.weight"
        tensor = tensors[name]
        records[name] = hashlib.sha256(tensor.data.tobytes()).hexdigest()
        return _tensor(dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape.tolist()))))
    def compare(name, expected):
        actual = captured[name]
        golden = expected.detach().reshape(-1).numpy()
        if actual.shape != golden.shape:
            raise ValueError(f"capture GR shape disagrees at {name}")
        error = np.abs(actual - golden)
        comparisons[name] = {"max_abs": float(error.max()), "mean_abs": float(error.mean()),
                             "passes_2e_5": bool(error.max() < 2e-5)}
    config = SimpleNamespace(hc_count=4, hidden_size=HIDDEN, hc_lowrank=320, rms_norm_eps=EPSILON)
    embeddings = _tensor(captured["qwen4exp_token_embeddings"]).reshape(1, tokens, HIDDEN)
    hyper = embeddings.repeat(1, 1, 4)
    for prefix, input_name, output_name in (("attn", None, "qwen4exp_attn_input-0"),
            ("ffn", "qwen4exp_attn_residual-0", "qwen4exp_ffn_input-0")):
        if output_name not in captured or (input_name is not None and input_name not in captured):
            continue
        layer = ref["Qwen4ExpTextGatedResidual"](config)
        with torch.no_grad():
            layer.hc_norm.weight.copy_(weights(prefix, "norm") - 1.)
            layer.input_mix_weight_down.weight.copy_(weights(prefix, "down"))
            layer.input_mix_weight_up.weight.copy_(weights(prefix, "up"))
            layer.block_inject_weight.weight.copy_(weights(prefix, "inject"))
            layer_input = hyper if input_name is None else _tensor(captured[input_name]).reshape(1, tokens, 4 * HIDDEN)
            mixed, _, injection = layer(layer_input)
            compare(output_name, mixed)
            if prefix == "attn" and "qwen4exp_attn_residual-0" in captured:
                block = _tensor(captured["qwen4exp_attn_output-0"]).reshape(1, tokens, HIDDEN).repeat(1, 1, 4)
                compare("qwen4exp_attn_residual-0", hyper + injection.repeat_interleave(HIDDEN, dim=-1) * block)
    return {"comparisons": comparisons, "tensor_payload_sha256": records,
            "scope": "Pinned GR on native embeddings/residuals; no MoE or upstream embedding-fidelity qualification."}


def build(gguf: Path, source: str | Path, tokens: int, capture: Path | None = None) -> dict:
    if not 1 <= tokens <= 8:
        raise ValueError("tokens must be in [1, 8] for bounded layer-0 diagnostics")
    captured, capture_record = read_capture(capture) if capture is not None else ({}, None)
    if capture_record is not None:
        tokens = capture_record["tokens"]
    torch.set_num_threads(8)
    torch.manual_seed(0)
    raw_source = load_source(str(source))
    if hashlib.sha256(raw_source).hexdigest() != TRANSFORMERS_SHA256:
        raise ValueError("modeling source does not match the pinned SHA-256")
    ref = extract_reference(raw_source)
    order = cyclic_order(KEY_HEADS, HEADS)

    data, records = read_gguf_tensors(gguf)
    canonical = {role: canonicalize(role, data[role], order) for role in PERMUTED}
    canonical["norm_weight"] = data["norm_weight"]
    for role in PERMUTED:
        records[role]["canonical_sha256"] = hashlib.sha256(
            np.ascontiguousarray(canonical[role]).tobytes()).hexdigest()

    inputs = (_tensor(captured["qwen4exp_attn_input-0"]).reshape(1, tokens, HIDDEN)
              if captured else values(tokens * HIDDEN, 0.5, 0.22).reshape(1, tokens, HIDDEN).to(torch.float32))
    layer = build_layer(ref)

    # 1. Canonical weights: the pinned reference oracle for this real layer.
    apply_weights(layer, canonical, canonical["norm_weight"])
    expected = run_trace(ref, layer, inputs)
    cache = LayerCache()
    steps = [run_trace(ref, layer, inputs[:, token:token + 1], cache) for token in range(tokens)]
    consistency = {
        "output_max_abs": float(np.max(np.abs(expected["output"] - np.concatenate([step["output"] for step in steps])))),
        "state_max_abs": float(np.max(np.abs(expected["state"] - steps[-1]["state"]))),
        "conv_state_max_abs": float(np.max(np.abs(expected["conv_state"] - steps[-1]["conv_state"]))),
    }
    if any(error >= 2e-5 for error in consistency.values()):
        raise ValueError(f"pinned batch/decode reference disagrees: {consistency}")

    # 2. Control: read the stored order as if it were canonical.  That is the
    #    plausible-wrong alternative the permutation evidence rules out, so its
    #    divergence measures what getting the head order wrong would cost.
    apply_weights(layer, data, data["norm_weight"])
    identity = run_trace(ref, layer, inputs)

    def delta(name: str) -> float:
        return float(np.max(np.abs(expected[name] - identity[name])))

    def relative(name: str) -> float | None:
        scale = float(np.max(np.abs(expected[name])))
        return delta(name) / scale if scale > 0 else None

    restored_a = -layer.A_log.float().exp().detach().numpy()
    a_roundtrip = float(np.max(np.abs(restored_a - data["a_log"])))
    native_expected = stored_trace(expected, tokens, order)
    payload = {
        "schema_version": 2,
        "architecture": "qwen4exp",
        "reference": {"revision": TRANSFORMERS_REVISION, "modeling_sha256": TRANSFORMERS_SHA256,
                      "transformers_url": TRANSFORMERS_URL},
        "gguf": {"path": str(gguf), "bytes": gguf.stat().st_size,
                 "identity_scope": "Per-tensor payload hashes below; the full shard is not re-hashed."},
        "layer": LAYER,
        "geometry": {"hidden": HIDDEN, "width": WIDTH, "key_heads": KEY_HEADS, "heads": HEADS,
                     "kernel": KERNEL, "tokens": tokens, "epsilon": EPSILON, "channels": CHANNELS},
        "head_order": "cyclic_gguf",
        "repeat_type": 1,
        "output_gate_type": model_output_gate(),
        "canonicalization": {"rule": "canonical[order[s]] = stored[s] along the value-head axis",
                             "order_sha256": canonical_hash({"order": order.tolist()}),
                             "source_head_indices": order.tolist()},
        "tensors": records,
        "input": inputs.reshape(-1).tolist(),
        "expected_layout": "cyclic_gguf",
        "a_exp_log_roundtrip_max_abs": a_roundtrip,
        "expected": {name: value.tolist() for name, value in native_expected.items()},
        "steps": [{name: value.tolist() for name, value in stored_trace(step, 1, order).items()}
                  for step in steps],
        "batch_decode_consistency": consistency,
        "decode_contract": "Zero-state first token followed by persistent-cache single-token decode; every convolution and recurrent state is recorded.",
        "identity_order_control": {
            "core_output_max_abs_delta": delta("core_output"),
            "core_output_relative_delta": relative("core_output"),
            "output_max_abs_delta": delta("output"),
            "output_relative_delta": relative("output"),
            "note": "Identical real weights with the stored order read as canonical.",
        },
        "scope": ("Layer-0 recurrent block on real quantized GGUF weights with a synthetic input. "
                  "Not full-model output, not the real layer-0 activation, and not a "
                  "quantization-fidelity bound."),
    }
    if capture_record is not None:
        mapping = {"linear_attn_qkv_mixed-0": "in_proj_qkv", "z-0": "in_proj_z",
                   "alpha-0": "in_proj_a", "beta-0": "in_proj_b", "gate-0": "g",
                   "conv_output_silu-0": "convolved", "attn_output-0": "core_output",
                   "final_output-0": "gated_output", "qwen4exp_attn_output-0": "output"}
        comparisons = {}
        for name, boundary in mapping.items():
            if name not in captured:
                continue
            actual, golden = captured[name], native_expected[boundary]
            if actual.shape != golden.shape:
                raise ValueError(f"capture shape disagrees at {name}")
            error = np.abs(actual - golden)
            comparisons[boundary] = {"max_abs": float(error.max()), "mean_abs": float(error.mean()),
                                     "reference_max_abs": float(np.abs(golden).max()),
                                     "passes_2e_5": bool(error.max() < 2e-5)}
        raw_a = canonical["a_log"].reshape(1, HEADS)
        restored_a = -torch.exp(_tensor(np.log(-canonical["a_log"]))).numpy().reshape(1, HEADS)
        softplus = torch.nn.functional.softplus(
            _tensor(expected["in_proj_a"].reshape(tokens, HEADS)) + _tensor(canonical["dt_bias"])).numpy()
        payload["captured_input_a_roundtrip_gate_max_abs"] = float(np.max(np.abs((raw_a - restored_a) * softplus)))
        payload["gated_residual_comparison"] = compare_gated_residuals(gguf, ref, captured, tokens)
        payload["capture"] = capture_record
        payload["captured_graph_comparison"] = comparisons
        payload["scope"] = ("Pinned recurrent layer on a captured native layer-0 activation and real dequantized weights. "
                            "Quantized native graph versus F32 reference errors are measured, not waived. "
                            "This does not establish upstream activation correctness or full-model parity.")
    payload["fixture_sha256"] = canonical_hash({k: v for k, v in payload.items() if k != "fixture_sha256"})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--source", default=TRANSFORMERS_URL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=TOKENS)
    parser.add_argument("--capture", type=Path, help="Separate probe stdout containing one zero-state layer-0 evaluation")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 8:
        parser.error("--tokens must be in [1, 8]")
    payload = build(args.gguf, args.source, args.tokens, args.capture)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
