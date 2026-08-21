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
    const int device_count = ggml_backend_cuda_get_device_count();
    require(device_count > 0, "CUDA device required");
    for (int device = 0; device < device_count; ++device) {
        check_format(device, GGML_TYPE_TURBO4_0, GGML_TURBO_KV_FORMAT_TURBO4, GGML_TYPE_I32, "turbo4_0");
        check_format(device, GGML_TYPE_TURBO4_0, GGML_TURBO_KV_FORMAT_TURBO4, GGML_TYPE_I64, "turbo4_0");
        check_format(device, GGML_TYPE_TURBO8_0, GGML_TURBO_KV_FORMAT_TURBO8, GGML_TYPE_I32, "turbo8_0");
        check_format(device, GGML_TYPE_TURBO8_0, GGML_TURBO_KV_FORMAT_TURBO8, GGML_TYPE_I64, "turbo8_0");
    }
    std::cout << "PASS: native CUDA Turbo4/Turbo8 row codecs\n";
    return 0;
}
