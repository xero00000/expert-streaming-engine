import ctypes
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf


def packed_blocks():
    # Every packed byte, signed/zero scales, multiple tensor rows.
    blocks = np.zeros((32, 18), dtype=np.uint8)
    expected = np.zeros((32, 64), dtype=np.float32)
    for b in range(32):
        scale = np.float16((b % 7 - 3) * 0.125)
        blocks[b, :2] = np.frombuffer(scale.tobytes(), dtype=np.uint8)
        for j in range(16):
            byte = (b * 16 + j) % 256
            blocks[b, 2 + j] = byte
            for lane in range(4):
                expected[b, 4 * j + lane] = np.float32(scale) * (((byte >> (2 * lane)) & 3) - 1)
    return blocks, expected


def test_q2_0_geometry_and_dequantization():
    blocks, expected = packed_blocks()
    assert int(gguf.GGMLQuantizationType.Q2_0) == 42
    assert gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q2_0] == (64, 18)
    actual = gguf.dequantize(blocks.reshape(4, 144), gguf.GGMLQuantizationType.Q2_0)
    np.testing.assert_array_equal(actual, expected.reshape(4, 512))
    assert gguf.quant_shape_to_byte_shape((4, 512), gguf.GGMLQuantizationType.Q2_0) == (4, 144)
    with pytest.raises(NotImplementedError):
        gguf.quantize(expected, gguf.GGMLQuantizationType.Q2_0)
    with pytest.raises(ValueError):
        gguf.dequantize(np.zeros((17,), dtype=np.uint8), gguf.GGMLQuantizationType.Q2_0)


def test_q2_0_dequantization_matches_native():
    library = Path(os.environ.get("ESE_TEST_GGML_LIBRARY", ROOT / "build_cpu_avx2_debug" / "ggml" / "src" / "libggml.so"))
    # This parity gate requires the native build; absence is a failure, not a skip.
    assert library.is_file(), f"build libggml first or set ESE_TEST_GGML_LIBRARY: {library}"
    lib = ctypes.CDLL(str(library))
    class InitParams(ctypes.Structure):
        _fields_ = [("mem_size", ctypes.c_size_t), ("mem_buffer", ctypes.c_void_p), ("no_alloc", ctypes.c_bool)]
    lib.ggml_init.argtypes = [InitParams]
    lib.ggml_init.restype = ctypes.c_void_p
    lib.ggml_free.argtypes = [ctypes.c_void_p]
    lib.dequantize_row_q2_0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.dequantize_row_q2_0.restype = None
    ctx = lib.ggml_init(InitParams(1024 * 1024, None, True))
    assert ctx
    try:
        blocks, expected = packed_blocks()
        native = np.empty_like(expected)
        lib.dequantize_row_q2_0(blocks.ctypes.data, native.ctypes.data, native.size)
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(gguf.dequantize(blocks, gguf.GGMLQuantizationType.Q2_0), native)
    finally:
        lib.ggml_free(ctx)


def write_fixture(path):
    writer = gguf.GGUFWriter(path, "qwen4exp")
    writer.add_name("synthetic manifest")
    writer.add_string("general.source.huggingface.repository", "fixture-owner/fixture-model")
    writer.add_string("general.repo_url", "https://huggingface.co/fixture-owner/fixture-model")
    writer.add_string("general.quantized_by", "fixture quantizer")
    writer.add_uint32("qwen4exp.block_count", 4)
    writer.add_tensor("per_layer_token_embd.weight", np.zeros((80, 16), dtype=np.float32))
    block = np.frombuffer(struct.pack("<e", 0.5) + bytes([0b11100100] * 16), dtype=np.uint8).reshape(1, 18)
    writer.add_tensor("blk.0.ffn_down_exps.weight", block, raw_dtype=gguf.GGMLQuantizationType.Q2_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def test_manifest_cli_reads_q2_0_and_reports_actual_source(tmp_path):
    path = tmp_path / "fixture.gguf"
    write_fixture(path)
    output = tmp_path / "inventory.json"
    command = [sys.executable, str(ROOT / "tools/qwen4exp_manifest.py"), str(tmp_path),
               "--skip-sha256", "--output", str(output)]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text())
    assert manifest["sources"]["gguf"]["repository"] == "fixture-owner/fixture-model"
    assert manifest["sources"]["gguf"]["revision"] is None
    assert manifest["sources"]["gguf"]["url"] == "https://huggingface.co/fixture-owner/fixture-model"
    assert manifest["metadata"]["general.quantized_by"] == "fixture quantizer"
    tensor = next(t for t in manifest["tensors"] if t["type"] == "Q2_0")
    assert tensor["shape"] == [64, 1] and tensor["size_bytes"] == 18
    assert tensor["data_offset"] + tensor["size_bytes"] <= path.stat().st_size
    assert manifest["ple"]["row_count"] == 80 and manifest["ple"]["row_size_bytes"] == 64
    result = subprocess.run(command + ["--gguf-repository", "override-owner/override-model"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["sources"]["gguf"]["repository"] == "override-owner/override-model"
