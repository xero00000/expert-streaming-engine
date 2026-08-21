// SPDX-License-Identifier: MIT

#include "ggml.h"
#include "ggml-turbo-kv.h"
#include "llama-kv-padding.h"

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

std::vector<float> make_source() {
    std::vector<float> values(GGML_TURBO_KV_BLOCK_ELEMENTS * 2);
    uint64_t state = UINT64_C(0x6a09e667f3bcc909);

    for (size_t i = 0; i < values.size(); ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t mixed = state * UINT64_C(2685821657736338717);
        const float random = static_cast<float>(
            (mixed >> 40) / static_cast<double>(UINT32_C(1) << 24) * 2.0 - 1.0);
        values[i] = random + 0.4f * std::sin(static_cast<float>(i) * 0.091f);
    }

    values[0] = 8.0f;
    values[127] = -6.0f;
    values[128] = 0.0f;
    return values;
}

double normalized_mse(const std::vector<float> & expected, const std::vector<float> & actual) {
    double error = 0.0;
    double signal = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        const double delta = static_cast<double>(expected[i]) - actual[i];
        error += delta * delta;
        signal += static_cast<double>(expected[i]) * expected[i];
    }
    return signal > 0.0 ? error / signal : error;
}

void check_type(
        enum ggml_type type,
        enum ggml_turbo_kv_format format,
        const char * expected_name,
        size_t expected_block_bytes,
        double max_nmse) {
    const std::vector<float> source = make_source();
    const size_t encoded_bytes = ggml_turbo_kv_encoded_size(format, source.size());

    require(std::strcmp(ggml_type_name(type), expected_name) == 0,
            std::string(expected_name) + ": type name");
    require(ggml_blck_size(type) == GGML_TURBO_KV_BLOCK_ELEMENTS,
            std::string(expected_name) + ": block size");
    require(ggml_type_size(type) == expected_block_bytes,
            std::string(expected_name) + ": type size");
    require(ggml_is_quantized(type),
            std::string(expected_name) + ": quantized trait");
    require(ggml_row_size(type, 128) == expected_block_bytes,
            std::string(expected_name) + ": one-block row size");
    require(ggml_row_size(type, 256) == expected_block_bytes * 2,
            std::string(expected_name) + ": two-block row size");
    require(encoded_bytes == expected_block_bytes * 2,
            std::string(expected_name) + ": standalone encoded size");

    std::vector<uint8_t> through_core(encoded_bytes);
    std::vector<uint8_t> through_reference(encoded_bytes);

    const size_t written = ggml_quantize_chunk(
        type,
        source.data(),
        through_core.data(),
        0,
        2,
        GGML_TURBO_KV_BLOCK_ELEMENTS,
        nullptr,
        nullptr);

    require(written == encoded_bytes,
            std::string(expected_name) + ": ggml_quantize_chunk byte count");
    require(ggml_turbo_kv_quantize_reference(
                format,
                source.data(),
                source.size(),
                through_reference.data(),
                through_reference.size()) == GGML_TURBO_KV_STATUS_OK,
            std::string(expected_name) + ": standalone reference quantize");
    require(through_core == through_reference,
            std::string(expected_name) + ": core dispatch is byte-identical to reference");

    std::vector<float> decoded(source.size());
    switch (format) {
        case GGML_TURBO_KV_FORMAT_TURBO2:
            ggml_turbo2_to_float(through_core.data(), decoded.data(), decoded.size());
            break;
        case GGML_TURBO_KV_FORMAT_TURBO3:
            ggml_turbo3_to_float(through_core.data(), decoded.data(), decoded.size());
            break;
        case GGML_TURBO_KV_FORMAT_TURBO4:
            ggml_turbo4_to_float(through_core.data(), decoded.data(), decoded.size());
            break;
        case GGML_TURBO_KV_FORMAT_TURBO8:
            ggml_turbo8_to_float(through_core.data(), decoded.data(), decoded.size());
            break;
    }

    require(std::all_of(decoded.begin(), decoded.end(), [](float value) {
                return std::isfinite(value);
            }),
            std::string(expected_name) + ": finite decode");
    require(normalized_mse(source, decoded) <= max_nmse,
            std::string(expected_name) + ": reconstruction quality");

    std::cout << expected_name
              << " id=" << static_cast<int>(type)
              << " bytes=" << written
              << " bits/value=" << ggml_turbo_kv_bits_per_value(format)
              << " nmse=" << normalized_mse(source, decoded)
              << "\n";
}

} // namespace

static_assert(GGML_TYPE_TURBO2_0 == 45, "Turbo2 GGML/GGUF numeric ID changed");
static_assert(GGML_TYPE_TURBO3_0 == 43, "Turbo3 GGML/GGUF numeric ID changed");
static_assert(GGML_TYPE_TURBO4_0 == 44, "Turbo4 GGML/GGUF numeric ID changed");
static_assert(GGML_TYPE_TURBO8_0 == 48, "Turbo8 GGML/GGUF numeric ID changed");
static_assert(sizeof(ggml_turbo2_block) == GGML_TURBO2_BLOCK_BYTES, "Turbo2 ABI changed");
static_assert(sizeof(ggml_turbo3_block) == GGML_TURBO3_BLOCK_BYTES, "Turbo3 ABI changed");
static_assert(sizeof(ggml_turbo4_block) == GGML_TURBO4_BLOCK_BYTES, "Turbo4 ABI changed");
static_assert(sizeof(ggml_turbo8_block) == GGML_TURBO8_BLOCK_BYTES, "Turbo8 ABI changed");

int main() {
    require(llama_kv_head_dim_for_type(GGML_TYPE_F16, 96) == 96,
            "F16 head geometry remains unchanged");
    require(llama_kv_head_dim_for_type(GGML_TYPE_TURBO2_0, 96) == 128,
            "Turbo2 pads a 96-wide head independently");
    require(llama_kv_head_dim_for_type(GGML_TYPE_TURBO3_0, 96) == 128,
            "Turbo3 pads a 96-wide head independently");
    require(llama_kv_head_dim_for_type(GGML_TYPE_TURBO4_0, 96) == 128,
            "Turbo4 pads a 96-wide head independently");
    require(llama_kv_head_dim_for_type(GGML_TYPE_TURBO8_0, 129) == 256,
            "Turbo8 rounds each head to the next 128-value block");
    require(llama_kv_gqa_dim_for_type(GGML_TYPE_TURBO4_0, 96, 3) == 384,
            "Turbo GQA padding does not cross head boundaries");

    check_type(
        GGML_TYPE_TURBO2_0,
        GGML_TURBO_KV_FORMAT_TURBO2,
        "turbo2_0",
        GGML_TURBO2_BLOCK_BYTES,
        0.25);
    check_type(
        GGML_TYPE_TURBO3_0,
        GGML_TURBO_KV_FORMAT_TURBO3,
        "turbo3_0",
        GGML_TURBO3_BLOCK_BYTES,
        0.08);
    check_type(
        GGML_TYPE_TURBO4_0,
        GGML_TURBO_KV_FORMAT_TURBO4,
        "turbo4_0",
        GGML_TURBO4_BLOCK_BYTES,
        0.03);
    check_type(
        GGML_TYPE_TURBO8_0,
        GGML_TURBO_KV_FORMAT_TURBO8,
        "turbo8_0",
        GGML_TURBO8_BLOCK_BYTES,
        0.0005);

    std::cout << "PASS: internal GGML fixed Turbo type integration\n";
    return 0;
}
