"""Weight streaming must reconstruct the same matrices as resident decoding."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from qwen4exp_full_reference import Weights, ExpertBank, ReferenceCache, full_reference
import qwen4exp_full_reference as reference_module
from gguf import GGMLQuantizationType as Q, dequantize, quantize


def store(name, data, qtype, shape):
    result = Weights.__new__(Weights)
    result.tensors = {name: SimpleNamespace(data=data, tensor_type=qtype,
                                           shape=np.array(shape[::-1]))}
    result.accesses = []
    return result


@pytest.mark.parametrize("qtype", [Q.F32, Q.F16, Q.BF16, Q.Q4_0])
def test_arbitrary_embedding_rows_and_output_chunks_match_resident(qtype):
    matrix = (np.sin(np.arange(11 * 64) * .19) * .07).astype(np.float32).reshape(11, 64)
    encoded = quantize(matrix, qtype)
    weights = store("weight", encoded, qtype, matrix.shape)
    resident = torch.from_numpy(dequantize(encoded, qtype))
    ids = np.array([[7, 1, 7], [10, 0, 3]])
    assert torch.equal(weights.read("weight", rows=ids), resident[torch.from_numpy(ids)])
    chunks = torch.cat([weights.read("weight", rows=slice(i, min(i+3, 11))) for i in range(0, 11, 3)])
    assert torch.equal(chunks, resident)
    # Compare actual output projections as well as reconstructed weights.
    x = torch.arange(128, dtype=torch.float32).reshape(2, 64) / 64
    streamed = torch.cat([torch.nn.functional.linear(x, weights.read("weight", rows=slice(i, i+3)))
                          for i in range(0, 11, 3)], -1)
    torch.testing.assert_close(streamed, torch.nn.functional.linear(x, resident), rtol=1e-6, atol=1e-6)


def test_expert_bank_reads_only_requested_expert_in_gate_up_order():
    weights = Weights.__new__(Weights)
    weights.tensors, weights.accesses = {}, []
    expected = []
    for offset, role in enumerate(("gate", "up")):
        matrix = (np.arange(3*4*64, dtype=np.float32).reshape(3, 4, 64) + offset * 1000) / 100
        encoded = quantize(matrix, Q.Q4_0)
        name = f"blk.2.ffn_{role}_exps.weight"
        weights.tensors.update(store(name, encoded, Q.Q4_0, matrix.shape).tensors)
        expected.append(torch.from_numpy(dequantize(encoded, Q.Q4_0)[2]))
    bank = ExpertBank(weights, 2, ("gate", "up"))
    assert torch.equal(bank[torch.tensor(2)], torch.cat(expected))
    assert [a["expert"] for a in weights.accesses] == [2, 2]
    with pytest.raises(ValueError, match="outside tensor"):
        bank[3]
    with pytest.raises(ValueError, match="outside tensor"):
        bank[-1]


def test_full_reference_rejects_unpinned_source_before_execution():
    with pytest.raises(ValueError, match="source hash mismatch"):
        full_reference(b"raise RuntimeError('must not execute')")


def test_cache_preserves_independent_histories_and_correct_convolution_lengths():
    cache = ReferenceCache(2)
    first = torch.tensor([[[1., 2.]]])
    next_input = torch.tensor([[[3.]]])
    assert not cache.has_previous_state(0)
    cache.update_conv_state(first, 0, conv_kernel_size=4)
    assert torch.equal(cache.layers[0].conv_states[0], torch.tensor([[[0., 1., 2.]]]))
    joined = cache.update_conv_state(next_input, 0, conv_kernel_size=4)
    assert torch.equal(joined, torch.tensor([[[0., 1., 2., 3.]]]))
    assert torch.equal(cache.layers[0].conv_states[0], torch.tensor([[[1., 2., 3.]]]))
    cache.update_conv_state(first, 0, conv_kernel_size=9, state_idx=1)
    assert cache.layers[0].conv_states[1].shape[-1] == 9
    assert not cache.has_previous_state(1)
    cache.update_conv_state(torch.tensor([[11, 12]]), 1, conv_kernel_size=2, state_idx=2)
    assert not cache.has_previous_state(0, state_idx=2)
    assert cache.layers[1].conv_states[2].tolist() == [[11, 12]]
    cache.update_recurrent_state(first, 0)
    first.fill_(99)
    assert cache.layers[0].recurrent_states[0].tolist() == [[[1., 2.]]]


@pytest.mark.parametrize("chunk_tokens", [0, 1, 2, 4, 7, 9])
def test_run_keeps_final_mixer_on_decoder_chunks(tmp_path, monkeypatch, chunk_tokens):
    # Exercise the reference runner's actual call sites. A tiny stateless
    # decoder makes chunk boundaries observable without production weights.
    calls = []
    embedding = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    head = torch.arange(10, dtype=torch.float32).reshape(5, 2) / 4

    class TinyWeights:
        accesses = []

        def __init__(self, _paths):
            pass

        def meta(self, _name):
            return 2

        def read(self, name, rows):
            return (embedding if name == "token_embd.weight" else head)[rows]

    class TinyMixer(torch.nn.Module):
        def __init__(self, _cfg, use_combine=False):
            super().__init__()

        def forward(self, value):
            calls.append(value.shape[1])
            return value[..., :2]

    class TinyLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_hyper_connection = torch.nn.Identity()
            self.mlp_hyper_connection = torch.nn.Identity()
            self.linear_attn = torch.nn.Identity()
            self.mlp = torch.nn.Identity()
            self.mlp.gate = torch.nn.Identity()

        def forward(self, value, _position, **_kwargs):
            return value + .125

    cfg = dict(hidden_size=2, hc_count=2, num_hidden_layers=1, vocab_size=5,
               layer_types=["linear_attention"],
               shared_expert_intermediate_size=4, head_dim=2,
               rope_parameters=dict(partial_rotary_factor=1., rope_theta=10000.))
    monkeypatch.setattr(reference_module, "Weights", TinyWeights)
    monkeypatch.setattr(reference_module, "full_reference", lambda _source: {"Qwen4ExpTextGatedResidual": TinyMixer})
    monkeypatch.setattr(reference_module, "model_text_config", lambda: cfg)
    monkeypatch.setattr(reference_module, "load_layer", lambda *_args, **_kwargs: TinyLayer())
    monkeypatch.setattr(reference_module, "gr", lambda *_args: None)
    source = tmp_path / "source.py"
    source.write_bytes(b"# tiny runner fixture")
    ids = [0, 1, 2, 3, 4, 0, 1]
    input_ids = tmp_path / "ids.json"
    import json
    input_ids.write_text(json.dumps(ids))
    args = SimpleNamespace(threads=1, source=source, gguf=[], input_ids=input_ids,
                           output=tmp_path / "output", tokens=len(ids), chunk_tokens=chunk_tokens,
                           output_rows=2)
    reference_module.run(args)
    expected_calls = [7] if chunk_tokens == 0 else [min(chunk_tokens, 7-start)
                                                  for start in range(0, 7, chunk_tokens)]
    assert calls == expected_calls, "final mixer changed decoder chunk boundaries"
    expected_input = (embedding[ids] + .125).unsqueeze(0)
    output = np.fromfile(args.output / "output_input.bin", np.float32).reshape(1, 7, 2)
    assert np.array_equal(output, expected_input.numpy())
    logits = np.fromfile(args.output / "logits.bin", np.float32).reshape(1, 7, 5)
    assert np.array_equal(logits, torch.nn.functional.linear(expected_input, head).numpy())
    qualification = json.loads((args.output / "qualification.json").read_text())
    assert qualification["output_mixer_chunk_tokens"] == chunk_tokens
