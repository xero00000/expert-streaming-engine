// SPDX-License-Identifier: MIT

#include "ggml-turbo-kv.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string & message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        std::exit(1);
    }
}

double l2_norm(const std::vector<float> & values) {
    double sum = 0.0;
    for (float value : values) {
        sum += static_cast<double>(value) * value;
    }
    return std::sqrt(sum);
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

std::vector<float> make_impulse() {
    std::vector<float> values(GGML_TURBO_KV_BLOCK_ELEMENTS, 0.0f);
    values[0] = 1.0f;
    values[17] = -0.25f;
    return values;
}

std::vector<float> make_ramp() {
    std::vector<float> values(GGML_TURBO_KV_BLOCK_ELEMENTS * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = -2.0f + 4.0f * static_cast<float>(i) /
            static_cast<float>(values.size() - 1);
    }
    return values;
}

std::vector<float> make_sinusoid() {
    std::vector<float> values(GGML_TURBO_KV_BLOCK_ELEMENTS * 2);
    for (size_t i = 0; i < values.size(); ++i) {
        const float phase = static_cast<float>(i) * 0.071f;
        values[i] = 1.7f * std::sin(phase) + 0.35f * std::cos(phase * 0.37f);
    }
    return values;
}

std::vector<float> make_seeded_random() {
    std::vector<float> values(GGML_TURBO_KV_BLOCK_ELEMENTS * 3);
    uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
    for (size_t i = 0; i < values.size(); ++i) {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        const uint64_t mixed = state * UINT64_C(2685821657736338717);
        values[i] = static_cast<float>(
            (mixed >> 40) / static_cast<double>(UINT32_C(1) << 24) * 6.0 - 3.0);
    }
    values[0] = 12.0f;
    values[127] = -9.0f;
    values[128] = 0.0f;
    return values;
}

void test_metadata() {
    static_assert(sizeof(ggml_turbo4_block) == GGML_TURBO4_BLOCK_BYTES, "Turbo4 size");
    static_assert(sizeof(ggml_turbo8_block) == GGML_TURBO8_BLOCK_BYTES, "Turbo8 size");

    require(ggml_turbo_kv_block_elements() == 128, "block element count");
    require(ggml_turbo_kv_block_bytes(GGML_TURBO_KV_FORMAT_TURBO4) == 66, "Turbo4 block bytes");
    require(ggml_turbo_kv_block_bytes(GGML_TURBO_KV_FORMAT_TURBO8) == 130, "Turbo8 block bytes");
    require(std::abs(ggml_turbo_kv_bits_per_value(GGML_TURBO_KV_FORMAT_TURBO4) - 4.125) < 1.0e-12,
            "Turbo4 exact bits/value");
    require(std::abs(ggml_turbo_kv_bits_per_value(GGML_TURBO_KV_FORMAT_TURBO8) - 8.125) < 1.0e-12,
            "Turbo8 exact bits/value");
    require(ggml_turbo_kv_encoded_size(GGML_TURBO_KV_FORMAT_TURBO4, 384) == 198,
            "Turbo4 encoded size");
    require(ggml_turbo_kv_encoded_size(GGML_TURBO_KV_FORMAT_TURBO8, 384) == 390,
            "Turbo8 encoded size");
    require(ggml_turbo_kv_encoded_size(GGML_TURBO_KV_FORMAT_TURBO4, 129) == 0,
            "misaligned encoded size rejected");
}

void test_errors() {
    std::array<float, 128> values{};
    std::array<uint8_t, 130> encoded{};
    std::array<float, 128> decoded{};

    require(ggml_turbo_kv_quantize_reference(
                static_cast<ggml_turbo_kv_format>(99), values.data(), values.size(),
                encoded.data(), encoded.size()) == GGML_TURBO_KV_STATUS_UNSUPPORTED_FORMAT,
            "unsupported quantize format");
    require(ggml_turbo_kv_quantize_reference(
                GGML_TURBO_KV_FORMAT_TURBO4, values.data(), 127,
                encoded.data(), encoded.size()) == GGML_TURBO_KV_STATUS_COUNT_NOT_ALIGNED,
            "misaligned quantize count");
    require(ggml_turbo_kv_quantize_reference(
                GGML_TURBO_KV_FORMAT_TURBO4, values.data(), values.size(),
                encoded.data(), 1) == GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL,
            "small output rejected");
    require(ggml_turbo_kv_dequantize_reference(
                GGML_TURBO_KV_FORMAT_TURBO8, encoded.data(), 1,
                decoded.data(), decoded.size()) == GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL,
            "small input rejected");
    require(ggml_turbo_kv_quantize_reference(
                GGML_TURBO_KV_FORMAT_TURBO4, nullptr, 0, nullptr, 0) ==
                GGML_TURBO_KV_STATUS_OK,
            "empty input is valid");

    values[5] = std::numeric_limits<float>::quiet_NaN();
    require(ggml_turbo_kv_quantize_reference(
                GGML_TURBO_KV_FORMAT_TURBO8, values.data(), values.size(),
                encoded.data(), encoded.size()) == GGML_TURBO_KV_STATUS_INVALID_ARGUMENT,
            "non-finite input rejected");
}

void test_zero(enum ggml_turbo_kv_format format) {
    std::vector<float> source(GGML_TURBO_KV_BLOCK_ELEMENTS * 2, 0.0f);
    std::vector<uint8_t> encoded(ggml_turbo_kv_encoded_size(format, source.size()));
    std::vector<float> decoded(source.size(), 1.0f);

    require(ggml_turbo_kv_quantize_reference(
                format, source.data(), source.size(), encoded.data(), encoded.size()) ==
                GGML_TURBO_KV_STATUS_OK,
            "zero quantize");
    require(ggml_turbo_kv_dequantize_reference(
                format, encoded.data(), encoded.size(), decoded.data(), decoded.size()) ==
                GGML_TURBO_KV_STATUS_OK,
            "zero dequantize");
    require(std::all_of(decoded.begin(), decoded.end(), [](float value) { return value == 0.0f; }),
            "zero round-trip is exactly zero");
}

void test_round_trip_case(
        enum ggml_turbo_kv_format format,
        const std::string & name,
        const std::vector<float> & source,
        double max_normalized_mse,
        double max_norm_error) {
    const size_t bytes = ggml_turbo_kv_encoded_size(format, source.size());
    std::vector<uint8_t> first(bytes);
    std::vector<uint8_t> second(bytes);
    std::vector<float> decoded(source.size());

    require(bytes != 0, name + ": encoded size");
    require(ggml_turbo_kv_quantize_reference(
                format, source.data(), source.size(), first.data(), first.size()) ==
                GGML_TURBO_KV_STATUS_OK,
            name + ": first deterministic quantize");
    require(ggml_turbo_kv_quantize_reference(
                format, source.data(), source.size(), second.data(), second.size()) ==
                GGML_TURBO_KV_STATUS_OK,
            name + ": second deterministic quantize");
    require(first == second, name + ": encoding is deterministic");

    require(ggml_turbo_kv_dequantize_reference(
                format, first.data(), first.size(), decoded.data(), decoded.size()) ==
                GGML_TURBO_KV_STATUS_OK,
            name + ": round-trip dequantize");
    require(std::all_of(decoded.begin(), decoded.end(), [](float value) { return std::isfinite(value); }),
            name + ": decoded values are finite");

    const double nmse = normalized_mse(source, decoded);
    const double source_norm = l2_norm(source);
    const double norm_error = source_norm > 0.0
        ? std::abs(l2_norm(decoded) / source_norm - 1.0)
        : l2_norm(decoded);

    std::cout << (format == GGML_TURBO_KV_FORMAT_TURBO4 ? "turbo4" : "turbo8")
              << " case=" << name
              << " bytes=" << bytes
              << " bits/value=" << ggml_turbo_kv_bits_per_value(format)
              << " nmse=" << nmse
              << " norm_error=" << norm_error << "\n";

    require(nmse <= max_normalized_mse, name + ": normalized MSE threshold");
    require(norm_error <= max_norm_error, name + ": norm preservation threshold");
}

void test_round_trips(enum ggml_turbo_kv_format format, double max_nmse) {
    test_round_trip_case(format, "impulse", make_impulse(), max_nmse, 0.003);
    test_round_trip_case(format, "ramp", make_ramp(), max_nmse, 0.003);
    test_round_trip_case(format, "sinusoid", make_sinusoid(), max_nmse, 0.003);
    test_round_trip_case(format, "seeded-random", make_seeded_random(), max_nmse, 0.003);
}

} // namespace

int main() {
    test_metadata();
    test_errors();
    test_zero(GGML_TURBO_KV_FORMAT_TURBO4);
    test_zero(GGML_TURBO_KV_FORMAT_TURBO8);
    test_round_trips(GGML_TURBO_KV_FORMAT_TURBO4, 0.03);
    test_round_trips(GGML_TURBO_KV_FORMAT_TURBO8, 0.0005);
    std::cout << "PASS: Turbo KV CPU reference foundation\n";
    return 0;
}
