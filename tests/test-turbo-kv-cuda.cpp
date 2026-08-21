// SPDX-License-Identifier: MIT

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml-turbo-kv.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

std::vector<float> make_updates() {
    std::vector<float> values(GGML_TURBO_KV_BLOCK_ELEMENTS * 4);
    uint64_t state = UINT64_C(0x243f6a8885a308d3);
    for (size_t i = 0; i < values.size(); ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t mixed = state * UINT64_C(2685821657736338717);
        const float random = static_cast<float>((mixed >> 40) / double(UINT64_C(1) << 24) * 2.0 - 1.0);
        values[i] = random + 0.35f * std::sin(float(i) * 0.071f);
    }
    values[0] = 7.0f;
    values[127] = -5.0f;
    values[256] = 0.0f;
    return values;
}

std::vector<float> make_attention_values(int64_t width, int rows, float phase) {
    std::vector<float> values(width * rows);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.7f * std::sin(float(i) * 0.037f + phase) +
            0.3f * std::cos(float(i) * 0.091f - phase);
    }
    return values;
}

void check_flash_attention(
        int device,
        ggml_type type_k,
        ggml_turbo_kv_format format_k,
        ggml_type type_v,
        ggml_turbo_kv_format format_v,
        const char * name,
        int64_t logical_width) {
    constexpr int64_t storage_width = GGML_TURBO_KV_BLOCK_ELEMENTS;
    constexpr int64_t key_rows = 256;
    constexpr int64_t query_rows = 2;
    constexpr int64_t query_heads = 2;
    constexpr int64_t mask_rows = GGML_PAD(query_rows, GGML_KQ_MASK_PAD);
    require(logical_width > 0 && logical_width <= storage_width, std::string(name) + ": logical head width");
    const float scale = 1.0f / std::sqrt(float(logical_width));
    const float softcap = 4.0f;

    const std::vector<float> logical_keys = make_attention_values(logical_width, key_rows, 0.2f);
    const std::vector<float> logical_values = make_attention_values(logical_width, key_rows, 1.1f);
    std::vector<float> keys(storage_width * key_rows, 0.0f);
    std::vector<float> values(storage_width * key_rows, 0.0f);
    for (int64_t row = 0; row < key_rows; ++row) {
        std::copy_n(logical_keys.begin() + row * logical_width, logical_width,
                keys.begin() + row * storage_width);
        std::copy_n(logical_values.begin() + row * logical_width, logical_width,
                values.begin() + row * storage_width);
    }
    std::vector<float> queries(logical_width * query_rows * query_heads);
    for (int64_t head = 0; head < query_heads; ++head) {
        const std::vector<float> head_queries = make_attention_values(
                logical_width, query_rows, -0.4f + 0.7f * head);
        std::copy(head_queries.begin(), head_queries.end(),
                queries.begin() + head * logical_width * query_rows);
    }
    std::vector<float> mask_f32(key_rows * mask_rows, 0.0f);
    for (int64_t query = 0; query < query_rows; ++query) {
        for (int64_t key = 0; key < key_rows; ++key) {
            mask_f32[query * key_rows + key] = -0.015f * float((key + 3 * query) % 17);
        }
    }
    std::vector<ggml_fp16_t> mask_f16(mask_f32.size());
    ggml_fp32_to_fp16_row(mask_f32.data(), mask_f16.data(), mask_f32.size());
    std::vector<uint8_t> encoded_keys(ggml_turbo_kv_encoded_size(format_k, keys.size()));
    std::vector<uint8_t> encoded_values(ggml_turbo_kv_encoded_size(format_v, values.size()));
    require(ggml_turbo_kv_quantize_reference(
                format_k, keys.data(), keys.size(), encoded_keys.data(), encoded_keys.size()) == GGML_TURBO_KV_STATUS_OK,
            std::string(name) + ": CPU key encode");
    require(ggml_turbo_kv_quantize_reference(
                format_v, values.data(), values.size(), encoded_values.data(), encoded_values.size()) == GGML_TURBO_KV_STATUS_OK,
            std::string(name) + ": CPU value encode");

    std::vector<float> decoded_keys(keys.size());
    std::vector<float> decoded_values(values.size());
    require(ggml_turbo_kv_dequantize_reference(
                format_k, encoded_keys.data(), encoded_keys.size(), decoded_keys.data(), decoded_keys.size()) == GGML_TURBO_KV_STATUS_OK,
            std::string(name) + ": CPU key decode");
    require(ggml_turbo_kv_dequantize_reference(
                format_v, encoded_values.data(), encoded_values.size(), decoded_values.data(), decoded_values.size()) == GGML_TURBO_KV_STATUS_OK,
            std::string(name) + ": CPU value decode");

    std::vector<float> expected(logical_width * query_rows * query_heads);
    for (int64_t head = 0; head < query_heads; ++head) {
        for (int64_t query = 0; query < query_rows; ++query) {
            float logits[key_rows];
            float max_logit = -INFINITY;
            for (int64_t key = 0; key < key_rows; ++key) {
                float dot = 0.0f;
                for (int64_t column = 0; column < logical_width; ++column) {
                    dot += queries[head * logical_width * query_rows + query * logical_width + column] *
                        decoded_keys[key * storage_width + column];
                }
                const float capped = softcap * std::tanh(dot * scale / softcap);
                logits[key] = capped + ggml_fp16_to_fp32(mask_f16[query * key_rows + key]);
                max_logit = std::max(max_logit, logits[key]);
            }
            float denominator = 0.0f;
            for (float & logit : logits) {
                logit = std::exp(logit - max_logit);
                denominator += logit;
            }
            for (int64_t column = 0; column < logical_width; ++column) {
                float output = 0.0f;
                for (int64_t key = 0; key < key_rows; ++key) {
                    output += logits[key] / denominator * decoded_values[key * storage_width + column];
                }
                expected[query * logical_width * query_heads + head * logical_width + column] = output;
            }
        }
    }

    ggml_init_params params = {
        /* .mem_size = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * context = ggml_init(params);
    require(context != nullptr, std::string(name) + ": attention context");
    ggml_tensor * q = ggml_new_tensor_4d(context, GGML_TYPE_F32, logical_width, query_rows, query_heads, 1);
    ggml_tensor * k = ggml_new_tensor_4d(context, type_k, storage_width, key_rows, 1, 1);
    ggml_tensor * v = ggml_new_tensor_4d(context, type_v, storage_width, key_rows, 1, 1);
    ggml_tensor * mask = ggml_new_tensor_4d(context, GGML_TYPE_F16, key_rows, mask_rows, 1, 1);
    ggml_tensor * q_padded = logical_width < storage_width
        ? ggml_pad(context, q, storage_width - logical_width, 0, 0, 0)
        : q;
    ggml_tensor * attention_padded = ggml_flash_attn_ext(
            context, q_padded, k, v, mask, scale, 0.0f, softcap);
    ggml_tensor * attention = attention_padded;
    if (logical_width < storage_width) {
        attention = ggml_view_4d(context, attention_padded,
                logical_width, attention_padded->ne[1], attention_padded->ne[2], attention_padded->ne[3],
                attention_padded->nb[1], attention_padded->nb[2], attention_padded->nb[3], 0);
        attention = ggml_cont(context, attention);
    }

    ggml_backend_t backend = ggml_backend_cuda_init(device, nullptr, nullptr);
    require(backend != nullptr, std::string(name) + ": attention CUDA backend");
    require(ggml_backend_supports_op(backend, attention_padded), std::string(name) + ": CUDA Flash Attention support");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(buffer != nullptr, std::string(name) + ": attention device allocation");
    ggml_backend_tensor_set(q, queries.data(), 0, queries.size() * sizeof(float));
    ggml_backend_tensor_set(k, encoded_keys.data(), 0, encoded_keys.size());
    ggml_backend_tensor_set(v, encoded_values.data(), 0, encoded_values.size());
    ggml_backend_tensor_set(mask, mask_f16.data(), 0, mask_f16.size() * sizeof(ggml_fp16_t));

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, attention);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            std::string(name) + ": CUDA Flash Attention graph compute");
    ggml_backend_synchronize(backend);

    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(attention, actual.data(), 0, actual.size() * sizeof(float));
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), std::string(name) + ": finite attention output");
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    require(max_error <= 1.0e-3f, std::string(name) + ": attention matches decoded CPU reference");
    std::cout << name << " attention device=" << device
              << " head_width=" << logical_width << " max_error=" << max_error << "\n";

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(context);
}

void check_format(
        int device,
        ggml_type type,
        ggml_turbo_kv_format format,
        ggml_type index_type,
        const char * name) {
    constexpr int64_t width = GGML_TURBO_KV_BLOCK_ELEMENTS * 2;
    constexpr int64_t base_rows = 3;
    constexpr int64_t update_rows = 2;

    const std::vector<float> updates = make_updates();
    const size_t encoded_row_bytes = ggml_turbo_kv_encoded_size(format, width);
    std::vector<uint8_t> expected(encoded_row_bytes * update_rows);
    require(ggml_turbo_kv_quantize_reference(
                format, updates.data(), updates.size(), expected.data(), expected.size()) == GGML_TURBO_KV_STATUS_OK,
            std::string(name) + ": CPU reference encode");

    ggml_init_params params = {
        /* .mem_size = */ 4 * 1024 * 1024,
        /* .mem_buffer = */ nullptr,
        /* .no_alloc = */ true,
    };
    ggml_context * context = ggml_init(params);
    require(context != nullptr, std::string(name) + ": context");

    ggml_tensor * base = ggml_new_tensor_2d(context, type, width, base_rows);
    ggml_tensor * source = ggml_new_tensor_2d(context, GGML_TYPE_F32, width, update_rows);
    ggml_tensor * indices = ggml_new_tensor_1d(context, index_type, update_rows);
    ggml_tensor * stored = ggml_set_rows(context, base, source, indices);
    ggml_tensor * decoded = index_type == GGML_TYPE_I32 ? ggml_get_rows(context, stored, indices) : nullptr;

    ggml_backend_t backend = ggml_backend_cuda_init(device, nullptr, nullptr);
    require(backend != nullptr, std::string(name) + ": CUDA backend");
    require(ggml_backend_supports_op(backend, stored), std::string(name) + ": CUDA SET_ROWS support");
    if (decoded != nullptr) {
        require(ggml_backend_supports_op(backend, decoded), std::string(name) + ": CUDA GET_ROWS support");
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(buffer != nullptr, std::string(name) + ": device allocation");

    std::vector<uint8_t> empty_base(ggml_nbytes(base), 0);
    const int32_t selected_rows_i32[update_rows] = {0, 2};
    const int64_t selected_rows_i64[update_rows] = {0, 2};
    ggml_backend_tensor_set(base, empty_base.data(), 0, empty_base.size());
    ggml_backend_tensor_set(source, updates.data(), 0, updates.size() * sizeof(float));
    if (index_type == GGML_TYPE_I32) {
        ggml_backend_tensor_set(indices, selected_rows_i32, 0, sizeof(selected_rows_i32));
    } else {
        ggml_backend_tensor_set(indices, selected_rows_i64, 0, sizeof(selected_rows_i64));
    }

    ggml_cgraph * graph = ggml_new_graph(context);
    ggml_build_forward_expand(graph, decoded != nullptr ? decoded : stored);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            std::string(name) + ": CUDA graph compute");
    ggml_backend_synchronize(backend);

    std::vector<uint8_t> encoded(ggml_nbytes(base));
    ggml_backend_tensor_get(base, encoded.data(), 0, encoded.size());
    require(std::memcmp(encoded.data(), expected.data(), encoded_row_bytes) == 0,
            std::string(name) + ": row 0 GPU bytes match CPU reference");
    require(std::all_of(
                encoded.begin() + encoded_row_bytes,
                encoded.begin() + 2 * encoded_row_bytes,
                [](uint8_t value) { return value == 0; }),
            std::string(name) + ": untouched row remains unchanged");
    require(std::memcmp(encoded.data() + 2 * encoded_row_bytes,
                        expected.data() + encoded_row_bytes,
                        encoded_row_bytes) == 0,
            std::string(name) + ": row 2 GPU bytes match CPU reference");

    float max_error = 0.0f;
    if (decoded != nullptr) {
        std::vector<float> gpu_decoded(updates.size());
        std::vector<float> cpu_decoded(updates.size());
        ggml_backend_tensor_get(decoded, gpu_decoded.data(), 0, gpu_decoded.size() * sizeof(float));
        require(ggml_turbo_kv_dequantize_reference(
                    format, expected.data(), expected.size(), cpu_decoded.data(), cpu_decoded.size()) == GGML_TURBO_KV_STATUS_OK,
                std::string(name) + ": CPU reference decode");
        for (size_t i = 0; i < gpu_decoded.size(); ++i) {
            require(std::isfinite(gpu_decoded[i]), std::string(name) + ": finite GPU decode");
            max_error = std::max(max_error, std::fabs(gpu_decoded[i] - cpu_decoded[i]));
        }
        require(max_error <= 2.0e-6f, std::string(name) + ": GPU reconstruction matches CPU reference");
    }

    std::cout << name << " device=" << device
              << " indices=" << (index_type == GGML_TYPE_I32 ? "i32" : "i64")
              << " encoded_bytes=" << expected.size();
    if (decoded != nullptr) {
        std::cout << " max_decode_error=" << max_error;
    }
    std::cout << "\n";

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(context);
}

} // namespace

int main() {
    require(setenv("GGML_TURBO_KV_REQUIRE_NATIVE_FATTN", "1", 1) == 0,
            "require native Turbo Flash Attention path");
    const int device_count = ggml_backend_cuda_get_device_count();
    require(device_count > 0, "CUDA device required");
    struct format_spec {
        ggml_type type;
        ggml_turbo_kv_format format;
        const char * name;
    };
    const format_spec formats[] = {
        {GGML_TYPE_TURBO2_0, GGML_TURBO_KV_FORMAT_TURBO2, "turbo2_0"},
        {GGML_TYPE_TURBO3_0, GGML_TURBO_KV_FORMAT_TURBO3, "turbo3_0"},
        {GGML_TYPE_TURBO4_0, GGML_TURBO_KV_FORMAT_TURBO4, "turbo4_0"},
        {GGML_TYPE_TURBO8_0, GGML_TURBO_KV_FORMAT_TURBO8, "turbo8_0"},
    };
    for (int device = 0; device < device_count; ++device) {
        for (const format_spec & spec : formats) {
            check_format(device, spec.type, spec.format, GGML_TYPE_I32, spec.name);
            check_format(device, spec.type, spec.format, GGML_TYPE_I64, spec.name);
        }
        for (const format_spec & key : formats) {
            for (const format_spec & value : formats) {
                const std::string name = std::string(key.name) + "-K/" + value.name + "-V";
                check_flash_attention(device,
                        key.type, key.format, value.type, value.format, name.c_str(), 128);
            }
        }
        for (const format_spec & spec : formats) {
            check_flash_attention(device,
                    spec.type, spec.format, spec.type, spec.format, spec.name, 96);
        }
    }
    std::cout << "PASS: native CUDA fixed Turbo row codecs and direct Flash Attention reads\n";
    return 0;
}
