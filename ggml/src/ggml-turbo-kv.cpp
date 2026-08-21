// SPDX-License-Identifier: MIT
/*
 * ESE Turbo KV CPU reference foundation.
 *
 * Algorithm and centroid tables are adapted from spiritbuun/buun-llama-cpp:
 *   commit 799e3995cd4f19aa9f6a3fa9fb5b4674422bf0ee
 *   ggml/src/ggml-turbo-quant.c
 *
 * The pinned Turbo2/Turbo3 encoders were incomplete. ESE retains their fixed
 * storage geometry and centroid tables while providing complete deterministic
 * CPU references for every fixed Turbo tier. TCQ and VBR remain gated until
 * their complete reference and backend paths exist.
 */

#include "ggml-turbo-kv.h"
#include "ggml-turbo-kv-internal.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {

constexpr size_t kBlock = GGML_TURBO_KV_BLOCK_ELEMENTS;
constexpr size_t kMatrixElements = kBlock * kBlock;
constexpr uint64_t kRotationSeed = 42;
constexpr double kPi = 3.141592653589793238462643383279502884;

static_assert(sizeof(ggml_turbo2_block) == GGML_TURBO2_BLOCK_BYTES, "unexpected Turbo2 block padding");
static_assert(sizeof(ggml_turbo3_block) == GGML_TURBO3_BLOCK_BYTES, "unexpected Turbo3 block padding");
static_assert(sizeof(ggml_turbo4_block) == GGML_TURBO4_BLOCK_BYTES, "unexpected Turbo4 block padding");
static_assert(sizeof(ggml_turbo8_block) == GGML_TURBO8_BLOCK_BYTES, "unexpected Turbo8 block padding");

constexpr std::array<float, 4> kCentroids2 = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f,
};

constexpr std::array<float, 8> kCentroids3 = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

constexpr std::array<float, 16> kCentroids4 = {
    -0.241556f, -0.182907f, -0.143047f, -0.111065f,
    -0.083317f, -0.058069f, -0.034311f, -0.011353f,
     0.011353f,  0.034311f,  0.058069f,  0.083317f,
     0.111065f,  0.143047f,  0.182907f,  0.241556f,
};

constexpr std::array<float, 256> kCentroids8 = {
    -0.34189706f, -0.29884648f, -0.27157635f, -0.25121314f, -0.23484838f, -0.22114745f, -0.20937878f, -0.19909346f,
    -0.19000012f, -0.18188631f, -0.17459596f, -0.16801505f, -0.16205003f, -0.15662252f, -0.15166721f, -0.14712896f,
    -0.14295785f, -0.13910911f, -0.13554055f, -0.13222068f, -0.12912571f, -0.12622918f, -0.12350196f, -0.12092574f,
    -0.11848717f, -0.11616507f, -0.11394363f, -0.11181760f, -0.10977627f, -0.10780649f, -0.10590563f, -0.10406567f,
    -0.10227606f, -0.10053416f, -0.09883731f, -0.09718554f, -0.09557616f, -0.09400389f, -0.09246874f, -0.09096270f,
    -0.08947787f, -0.08801425f, -0.08657184f, -0.08515064f, -0.08375065f, -0.08237188f, -0.08101431f, -0.07967795f,
    -0.07836547f, -0.07707417f, -0.07580144f, -0.07454460f, -0.07330103f, -0.07206804f, -0.07084568f, -0.06963391f,
    -0.06843275f, -0.06724219f, -0.06606225f, -0.06489290f, -0.06373417f, -0.06258603f, -0.06144851f, -0.06032158f,
    -0.05920527f, -0.05810222f, -0.05700977f, -0.05592792f, -0.05485668f, -0.05379604f, -0.05274602f, -0.05170659f,
    -0.05067778f, -0.04965957f, -0.04864930f, -0.04764700f, -0.04664999f, -0.04565829f, -0.04467189f, -0.04369080f,
    -0.04271500f, -0.04174452f, -0.04077933f, -0.03981945f, -0.03886486f, -0.03791293f, -0.03696631f, -0.03602498f,
    -0.03508631f, -0.03415029f, -0.03321957f, -0.03229416f, -0.03137139f, -0.03045128f, -0.02953382f, -0.02861901f,
    -0.02770950f, -0.02680265f, -0.02589845f, -0.02499689f, -0.02409534f, -0.02319644f, -0.02230284f, -0.02141190f,
    -0.02052095f, -0.01963266f, -0.01874701f, -0.01786137f, -0.01697838f, -0.01609804f, -0.01522035f, -0.01434266f,
    -0.01346497f, -0.01258993f, -0.01171755f, -0.01084516f, -0.00997278f, -0.00910304f, -0.00823331f, -0.00736357f,
    -0.00649649f, -0.00562941f, -0.00476233f, -0.00389524f, -0.00302816f, -0.00216373f, -0.00129930f, -0.00043487f,
     0.00043222f,  0.00129930f,  0.00216373f,  0.00302816f,  0.00389524f,  0.00476233f,  0.00562941f,  0.00649649f,
     0.00736357f,  0.00823331f,  0.00910304f,  0.00997278f,  0.01084516f,  0.01171755f,  0.01258993f,  0.01346497f,
     0.01434266f,  0.01522035f,  0.01609804f,  0.01697838f,  0.01786137f,  0.01874701f,  0.01963266f,  0.02052095f,
     0.02141190f,  0.02230284f,  0.02319644f,  0.02409534f,  0.02499689f,  0.02589845f,  0.02680265f,  0.02770950f,
     0.02861901f,  0.02953382f,  0.03045128f,  0.03137139f,  0.03229416f,  0.03321957f,  0.03415029f,  0.03508631f,
     0.03602498f,  0.03696631f,  0.03791293f,  0.03886486f,  0.03981945f,  0.04077933f,  0.04174452f,  0.04271500f,
     0.04369080f,  0.04467189f,  0.04565829f,  0.04664999f,  0.04764700f,  0.04864930f,  0.04965957f,  0.05067778f,
     0.05170659f,  0.05274602f,  0.05379604f,  0.05485668f,  0.05592792f,  0.05700977f,  0.05810222f,  0.05920527f,
     0.06032158f,  0.06144851f,  0.06258603f,  0.06373417f,  0.06489290f,  0.06606225f,  0.06724219f,  0.06843275f,
     0.06963391f,  0.07084568f,  0.07206804f,  0.07330103f,  0.07454460f,  0.07580144f,  0.07707417f,  0.07836547f,
     0.07967795f,  0.08101431f,  0.08237188f,  0.08375065f,  0.08515064f,  0.08657184f,  0.08801425f,  0.08947787f,
     0.09096270f,  0.09246874f,  0.09400389f,  0.09557616f,  0.09718554f,  0.09883731f,  0.10053416f,  0.10227606f,
     0.10406567f,  0.10590563f,  0.10780649f,  0.10977627f,  0.11181760f,  0.11394363f,  0.11616507f,  0.11848717f,
     0.12092574f,  0.12350196f,  0.12622918f,  0.12912571f,  0.13222068f,  0.13554055f,  0.13910911f,  0.14295785f,
     0.14712896f,  0.15166721f,  0.15662252f,  0.16205003f,  0.16801505f,  0.17459596f,  0.18188631f,  0.19000012f,
     0.19909346f,  0.20937878f,  0.22114745f,  0.23484838f,  0.25121314f,  0.27157635f,  0.29884648f,  0.34189706f,
};

struct RotationTables {
    std::array<float, kMatrixElements> forward{};
    std::array<float, kMatrixElements> inverse{};
};

uint64_t lcg_next(uint64_t & state) {
    state = state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
    return state;
}

double normal_sample(uint64_t & state) {
    const uint64_t a = lcg_next(state);
    double u1 = static_cast<double>(a >> 11) / static_cast<double>(UINT64_C(1) << 53);
    if (u1 < 1.0e-15) {
        u1 = 1.0e-15;
    }

    const uint64_t b = lcg_next(state);
    const double u2 = static_cast<double>(b >> 11) / static_cast<double>(UINT64_C(1) << 53);
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * kPi * u2);
}

RotationTables make_rotation_tables() {
    RotationTables tables;
    uint64_t state = kRotationSeed;

    for (float & value : tables.forward) {
        value = static_cast<float>(normal_sample(state));
    }

    // Modified Gram-Schmidt over columns, matching the pinned source.
    for (size_t column = 0; column < kBlock; ++column) {
        float norm_sq = 0.0f;
        for (size_t row = 0; row < kBlock; ++row) {
            const float value = tables.forward[row * kBlock + column];
            norm_sq += value * value;
        }

        const float norm = std::sqrt(norm_sq);
        if (norm > 1.0e-10f) {
            const float inv = 1.0f / norm;
            for (size_t row = 0; row < kBlock; ++row) {
                tables.forward[row * kBlock + column] *= inv;
            }
        }

        for (size_t other = column + 1; other < kBlock; ++other) {
            float dot = 0.0f;
            for (size_t row = 0; row < kBlock; ++row) {
                dot += tables.forward[row * kBlock + column] *
                       tables.forward[row * kBlock + other];
            }
            for (size_t row = 0; row < kBlock; ++row) {
                tables.forward[row * kBlock + other] -=
                    dot * tables.forward[row * kBlock + column];
            }
        }
    }

    for (size_t row = 0; row < kBlock; ++row) {
        for (size_t column = 0; column < kBlock; ++column) {
            tables.inverse[row * kBlock + column] =
                tables.forward[column * kBlock + row];
        }
    }

    return tables;
}

const RotationTables & rotation_tables() {
    static const RotationTables tables = make_rotation_tables();
    return tables;
}

void matvec(
        const std::array<float, kMatrixElements> & matrix,
        const std::array<float, kBlock> & input,
        std::array<float, kBlock> & output) {
    for (size_t row = 0; row < kBlock; ++row) {
        float sum = 0.0f;
        const size_t offset = row * kBlock;
        for (size_t column = 0; column < kBlock; ++column) {
            sum += matrix[offset + column] * input[column];
        }
        output[row] = sum;
    }
}

template <size_t N>
uint8_t nearest_centroid(float value, const std::array<float, N> & centroids) {
    size_t lo = 0;
    size_t hi = N - 1;
    while (lo < hi) {
        const size_t mid = (lo + hi) / 2;
        const float threshold = 0.5f * (centroids[mid] + centroids[mid + 1]);
        if (value < threshold) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return static_cast<uint8_t>(lo);
}

uint16_t fp32_to_fp16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));

    const uint32_t sign = (bits >> 16) & UINT32_C(0x8000);
    uint32_t mantissa = bits & UINT32_C(0x007fffff);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & UINT32_C(0xff)) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }

        mantissa |= UINT32_C(0x00800000);
        const uint32_t shift = static_cast<uint32_t>(14 - exponent);
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((UINT32_C(1) << shift) - 1);
        const uint32_t halfway = UINT32_C(1) << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) {
            ++half_mantissa;
        }
        return static_cast<uint16_t>(sign | half_mantissa);
    }

    if (exponent >= 31) {
        if ((bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000)) {
            uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
            if (payload == 0) {
                payload = 1;
            }
            return static_cast<uint16_t>(sign | UINT32_C(0x7c00) | payload);
        }
        return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
    }

    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & UINT32_C(0x1fff);
    if (remainder > UINT32_C(0x1000) ||
        (remainder == UINT32_C(0x1000) && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == UINT32_C(0x400)) {
            half_mantissa = 0;
            ++exponent;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | UINT32_C(0x7c00));
            }
        }
    }

    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | (half_mantissa & UINT32_C(0x3ff)));
}

float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = (static_cast<uint32_t>(value & UINT16_C(0x8000))) << 16;
    uint32_t exponent = (value >> 10) & UINT16_C(0x1f);
    uint32_t mantissa = value & UINT16_C(0x03ff);
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t e = -14;
            while ((mantissa & UINT32_C(0x0400)) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= UINT32_C(0x03ff);
            bits = sign |
                (static_cast<uint32_t>(e + 127) << 23) |
                (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | UINT32_C(0x7f800000) | (mantissa << 13);
    } else {
        bits = sign |
            ((exponent + static_cast<uint32_t>(127 - 15)) << 23) |
            (mantissa << 13);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool all_finite(const float * values, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (!std::isfinite(values[i])) {
            return false;
        }
    }
    return true;
}

template <size_t N>
void quantize_indices(
        const float * src,
        const std::array<float, N> & centroids,
        std::array<uint8_t, kBlock> & indices,
        float & scale) {
    float norm_sq = 0.0f;
    for (size_t i = 0; i < kBlock; ++i) {
        norm_sq += src[i] * src[i];
    }
    const float norm = std::sqrt(norm_sq);

    std::array<float, kBlock> normalized{};
    if (norm > 1.0e-10f) {
        const float inv_norm = 1.0f / norm;
        for (size_t i = 0; i < kBlock; ++i) {
            normalized[i] = src[i] * inv_norm;
        }
    }

    std::array<float, kBlock> rotated{};
    matvec(rotation_tables().forward, normalized, rotated);

    float reconstructed_norm_sq = 0.0f;
    for (size_t i = 0; i < kBlock; ++i) {
        indices[i] = nearest_centroid(rotated[i], centroids);
        const float reconstructed = centroids[indices[i]];
        reconstructed_norm_sq += reconstructed * reconstructed;
    }

    const float reconstructed_norm = std::sqrt(reconstructed_norm_sq);
    scale = reconstructed_norm > 1.0e-10f ? norm / reconstructed_norm : norm;
}

template <size_t N>
void dequantize_indices(
        const std::array<uint8_t, kBlock> & indices,
        float scale,
        const std::array<float, N> & centroids,
        float * dst) {
    std::array<float, kBlock> rotated{};
    for (size_t i = 0; i < kBlock; ++i) {
        rotated[i] = centroids[indices[i]];
    }

    std::array<float, kBlock> reconstructed{};
    matvec(rotation_tables().inverse, rotated, reconstructed);
    for (size_t i = 0; i < kBlock; ++i) {
        dst[i] = reconstructed[i] * scale;
    }
}

void quantize_turbo4(const float * src, ggml_turbo4_block & dst) {
    std::array<uint8_t, kBlock> indices{};
    float scale = 0.0f;
    quantize_indices(src, kCentroids4, indices, scale);
    dst.scale = fp32_to_fp16(scale);

    for (size_t i = 0; i < kBlock; i += 2) {
        dst.qs[i / 2] = static_cast<uint8_t>(
            (indices[i] & UINT8_C(0x0f)) | ((indices[i + 1] & UINT8_C(0x0f)) << 4));
    }
}

void dequantize_turbo4(const ggml_turbo4_block & src, float * dst) {
    std::array<uint8_t, kBlock> indices{};
    for (size_t i = 0; i < kBlock; i += 2) {
        indices[i] = src.qs[i / 2] & UINT8_C(0x0f);
        indices[i + 1] = (src.qs[i / 2] >> 4) & UINT8_C(0x0f);
    }
    dequantize_indices(indices, fp16_to_fp32(src.scale), kCentroids4, dst);
}

void quantize_turbo2(const float * src, ggml_turbo2_block & dst) {
    std::array<uint8_t, kBlock> indices{};
    float scale = 0.0f;
    quantize_indices(src, kCentroids2, indices, scale);
    const uint16_t scale_fp16 = fp32_to_fp16(scale);
    for (size_t subblock = 0; subblock < 4; ++subblock) {
        auto & output = dst.subblocks[subblock];
        output.scale = scale_fp16;
        const size_t offset = subblock * GGML_TURBO_KV_SUBBLOCK_ELEMENTS;
        for (size_t i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS; i += 4) {
            output.qs[i / 4] = static_cast<uint8_t>(
                indices[offset + i] |
                (indices[offset + i + 1] << 2) |
                (indices[offset + i + 2] << 4) |
                (indices[offset + i + 3] << 6));
        }
    }
}

void dequantize_turbo2(const ggml_turbo2_block & src, float * dst) {
    std::array<float, kBlock> rotated{};
    for (size_t subblock = 0; subblock < 4; ++subblock) {
        const auto & input = src.subblocks[subblock];
        const float scale = fp16_to_fp32(input.scale);
        const size_t offset = subblock * GGML_TURBO_KV_SUBBLOCK_ELEMENTS;
        for (size_t i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS; ++i) {
            const uint8_t index = (input.qs[i / 4] >> ((i % 4) * 2)) & UINT8_C(0x03);
            rotated[offset + i] = kCentroids2[index] * scale;
        }
    }
    std::array<float, kBlock> reconstructed{};
    matvec(rotation_tables().inverse, rotated, reconstructed);
    std::copy(reconstructed.begin(), reconstructed.end(), dst);
}

void quantize_turbo3(const float * src, ggml_turbo3_block & dst) {
    std::array<uint8_t, kBlock> indices{};
    float scale = 0.0f;
    quantize_indices(src, kCentroids3, indices, scale);
    const uint16_t scale_fp16 = fp32_to_fp16(scale);
    for (size_t subblock = 0; subblock < 4; ++subblock) {
        auto & output = dst.subblocks[subblock];
        output.scale = scale_fp16;
        std::memset(output.qs, 0, sizeof(output.qs));
        std::memset(output.high, 0, sizeof(output.high));
        const size_t offset = subblock * GGML_TURBO_KV_SUBBLOCK_ELEMENTS;
        for (size_t i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS; ++i) {
            const uint8_t index = indices[offset + i];
            output.qs[i / 4] |= static_cast<uint8_t>((index & UINT8_C(0x03)) << ((i % 4) * 2));
            output.high[i / 8] |= static_cast<uint8_t>(((index >> 2) & 1u) << (i % 8));
        }
    }
}

void dequantize_turbo3(const ggml_turbo3_block & src, float * dst) {
    std::array<float, kBlock> rotated{};
    for (size_t subblock = 0; subblock < 4; ++subblock) {
        const auto & input = src.subblocks[subblock];
        const float scale = fp16_to_fp32(input.scale);
        const size_t offset = subblock * GGML_TURBO_KV_SUBBLOCK_ELEMENTS;
        for (size_t i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS; ++i) {
            const uint8_t low = (input.qs[i / 4] >> ((i % 4) * 2)) & UINT8_C(0x03);
            const uint8_t high = (input.high[i / 8] >> (i % 8)) & 1u;
            rotated[offset + i] = kCentroids3[low | (high << 2)] * scale;
        }
    }
    std::array<float, kBlock> reconstructed{};
    matvec(rotation_tables().inverse, rotated, reconstructed);
    std::copy(reconstructed.begin(), reconstructed.end(), dst);
}

void quantize_turbo8(const float * src, ggml_turbo8_block & dst) {
    std::array<uint8_t, kBlock> indices{};
    float scale = 0.0f;
    quantize_indices(src, kCentroids8, indices, scale);
    dst.scale = fp32_to_fp16(scale);
    std::memcpy(dst.qs, indices.data(), indices.size());
}

void dequantize_turbo8(const ggml_turbo8_block & src, float * dst) {
    std::array<uint8_t, kBlock> indices{};
    std::memcpy(indices.data(), src.qs, indices.size());
    dequantize_indices(indices, fp16_to_fp32(src.scale), kCentroids8, dst);
}

bool multiplication_overflows(size_t left, size_t right) {
    return right != 0 && left > std::numeric_limits<size_t>::max() / right;
}

} // namespace

extern "C" {

size_t ggml_turbo_kv_block_elements(void) {
    return kBlock;
}

size_t ggml_turbo_kv_block_bytes(enum ggml_turbo_kv_format format) {
    switch (format) {
        case GGML_TURBO_KV_FORMAT_TURBO2:
            return sizeof(ggml_turbo2_block);
        case GGML_TURBO_KV_FORMAT_TURBO3:
            return sizeof(ggml_turbo3_block);
        case GGML_TURBO_KV_FORMAT_TURBO4:
            return sizeof(ggml_turbo4_block);
        case GGML_TURBO_KV_FORMAT_TURBO8:
            return sizeof(ggml_turbo8_block);
    }
    return 0;
}

double ggml_turbo_kv_bits_per_value(enum ggml_turbo_kv_format format) {
    const size_t bytes = ggml_turbo_kv_block_bytes(format);
    return bytes == 0 ? 0.0 : static_cast<double>(bytes * 8) / static_cast<double>(kBlock);
}

size_t ggml_turbo_kv_encoded_size(enum ggml_turbo_kv_format format, size_t value_count) {
    const size_t block_bytes = ggml_turbo_kv_block_bytes(format);
    if (block_bytes == 0 || value_count % kBlock != 0) {
        return 0;
    }

    const size_t blocks = value_count / kBlock;
    if (multiplication_overflows(blocks, block_bytes)) {
        return 0;
    }
    return blocks * block_bytes;
}

const float * ggml_turbo_kv_rotation_forward(void) {
    return rotation_tables().forward.data();
}

const float * ggml_turbo_kv_rotation_inverse(void) {
    return rotation_tables().inverse.data();
}

const float * ggml_turbo_kv_centroids4(void) {
    return kCentroids4.data();
}

const float * ggml_turbo_kv_centroids2(void) {
    return kCentroids2.data();
}

const float * ggml_turbo_kv_centroids3(void) {
    return kCentroids3.data();
}

const float * ggml_turbo_kv_centroids8(void) {
    return kCentroids8.data();
}

enum ggml_turbo_kv_status ggml_turbo_kv_quantize_reference(
        enum ggml_turbo_kv_format format,
        const float * src,
        size_t value_count,
        void * dst,
        size_t dst_size) {
    const size_t block_bytes = ggml_turbo_kv_block_bytes(format);
    if (block_bytes == 0) {
        return GGML_TURBO_KV_STATUS_UNSUPPORTED_FORMAT;
    }
    if (value_count % kBlock != 0) {
        return GGML_TURBO_KV_STATUS_COUNT_NOT_ALIGNED;
    }
    if (value_count == 0) {
        return GGML_TURBO_KV_STATUS_OK;
    }
    if (src == nullptr || dst == nullptr || !all_finite(src, value_count)) {
        return GGML_TURBO_KV_STATUS_INVALID_ARGUMENT;
    }

    const size_t required = ggml_turbo_kv_encoded_size(format, value_count);
    if (required == 0 || dst_size < required) {
        return GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t blocks = value_count / kBlock;
    if (format == GGML_TURBO_KV_FORMAT_TURBO2) {
        auto * output = static_cast<ggml_turbo2_block *>(dst);
        for (size_t block = 0; block < blocks; ++block) quantize_turbo2(src + block * kBlock, output[block]);
    } else if (format == GGML_TURBO_KV_FORMAT_TURBO3) {
        auto * output = static_cast<ggml_turbo3_block *>(dst);
        for (size_t block = 0; block < blocks; ++block) quantize_turbo3(src + block * kBlock, output[block]);
    } else if (format == GGML_TURBO_KV_FORMAT_TURBO4) {
        auto * output = static_cast<ggml_turbo4_block *>(dst);
        for (size_t block = 0; block < blocks; ++block) {
            quantize_turbo4(src + block * kBlock, output[block]);
        }
    } else {
        auto * output = static_cast<ggml_turbo8_block *>(dst);
        for (size_t block = 0; block < blocks; ++block) {
            quantize_turbo8(src + block * kBlock, output[block]);
        }
    }
    return GGML_TURBO_KV_STATUS_OK;
}

enum ggml_turbo_kv_status ggml_turbo_kv_dequantize_reference(
        enum ggml_turbo_kv_format format,
        const void * src,
        size_t src_size,
        float * dst,
        size_t value_count) {
    const size_t block_bytes = ggml_turbo_kv_block_bytes(format);
    if (block_bytes == 0) {
        return GGML_TURBO_KV_STATUS_UNSUPPORTED_FORMAT;
    }
    if (value_count % kBlock != 0) {
        return GGML_TURBO_KV_STATUS_COUNT_NOT_ALIGNED;
    }
    if (value_count == 0) {
        return GGML_TURBO_KV_STATUS_OK;
    }
    if (src == nullptr || dst == nullptr) {
        return GGML_TURBO_KV_STATUS_INVALID_ARGUMENT;
    }

    const size_t required = ggml_turbo_kv_encoded_size(format, value_count);
    if (required == 0 || src_size < required) {
        return GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL;
    }

    const size_t blocks = value_count / kBlock;
    if (format == GGML_TURBO_KV_FORMAT_TURBO2) {
        const auto * input = static_cast<const ggml_turbo2_block *>(src);
        for (size_t block = 0; block < blocks; ++block) dequantize_turbo2(input[block], dst + block * kBlock);
    } else if (format == GGML_TURBO_KV_FORMAT_TURBO3) {
        const auto * input = static_cast<const ggml_turbo3_block *>(src);
        for (size_t block = 0; block < blocks; ++block) dequantize_turbo3(input[block], dst + block * kBlock);
    } else if (format == GGML_TURBO_KV_FORMAT_TURBO4) {
        const auto * input = static_cast<const ggml_turbo4_block *>(src);
        for (size_t block = 0; block < blocks; ++block) {
            dequantize_turbo4(input[block], dst + block * kBlock);
        }
    } else {
        const auto * input = static_cast<const ggml_turbo8_block *>(src);
        for (size_t block = 0; block < blocks; ++block) {
            dequantize_turbo8(input[block], dst + block * kBlock);
        }
    }
    return GGML_TURBO_KV_STATUS_OK;
}


[[noreturn]] static void abort_adapter_failure() {
    std::abort();
}

static size_t checked_value_count(int64_t value_count) {
    if (value_count < 0) {
        abort_adapter_failure();
    }
    return static_cast<size_t>(value_count);
}

static size_t checked_total_value_count(int64_t nrows, int64_t n_per_row) {
    if (nrows < 0 || n_per_row < 0) {
        abort_adapter_failure();
    }

    const size_t rows = static_cast<size_t>(nrows);
    const size_t per_row = static_cast<size_t>(n_per_row);
    if (per_row != 0 && rows > std::numeric_limits<size_t>::max() / per_row) {
        abort_adapter_failure();
    }
    return rows * per_row;
}

static void require_adapter_status(enum ggml_turbo_kv_status status) {
    if (status != GGML_TURBO_KV_STATUS_OK) {
        abort_adapter_failure();
    }
}

static void adapter_to_float(
        enum ggml_turbo_kv_format format,
        const void * src,
        float * dst,
        int64_t value_count) {
    const size_t count = checked_value_count(value_count);
    const size_t bytes = ggml_turbo_kv_encoded_size(format, count);
    if (count != 0 && bytes == 0) {
        abort_adapter_failure();
    }

    require_adapter_status(
        ggml_turbo_kv_dequantize_reference(format, src, bytes, dst, count));
}

static void adapter_from_float(
        enum ggml_turbo_kv_format format,
        const float * src,
        void * dst,
        int64_t value_count) {
    const size_t count = checked_value_count(value_count);
    const size_t bytes = ggml_turbo_kv_encoded_size(format, count);
    if (count != 0 && bytes == 0) {
        abort_adapter_failure();
    }

    require_adapter_status(
        ggml_turbo_kv_quantize_reference(format, src, count, dst, bytes));
}

static size_t adapter_quantize_rows(
        enum ggml_turbo_kv_format format,
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    (void) imatrix;

    const size_t total = checked_total_value_count(nrows, n_per_row);
    if (n_per_row % static_cast<int64_t>(GGML_TURBO_KV_BLOCK_ELEMENTS) != 0) {
        abort_adapter_failure();
    }

    const size_t bytes = ggml_turbo_kv_encoded_size(format, total);
    if (total != 0 && bytes == 0) {
        abort_adapter_failure();
    }

    require_adapter_status(
        ggml_turbo_kv_quantize_reference(format, src, total, dst, bytes));
    return bytes;
}

void ggml_turbo4_to_float(const void * src, float * dst, int64_t value_count) {
    adapter_to_float(GGML_TURBO_KV_FORMAT_TURBO4, src, dst, value_count);
}

void ggml_turbo2_to_float(const void * src, float * dst, int64_t value_count) {
    adapter_to_float(GGML_TURBO_KV_FORMAT_TURBO2, src, dst, value_count);
}

void ggml_turbo3_to_float(const void * src, float * dst, int64_t value_count) {
    adapter_to_float(GGML_TURBO_KV_FORMAT_TURBO3, src, dst, value_count);
}

void ggml_turbo8_to_float(const void * src, float * dst, int64_t value_count) {
    adapter_to_float(GGML_TURBO_KV_FORMAT_TURBO8, src, dst, value_count);
}

void ggml_turbo4_from_float(const float * src, void * dst, int64_t value_count) {
    adapter_from_float(GGML_TURBO_KV_FORMAT_TURBO4, src, dst, value_count);
}

void ggml_turbo2_from_float(const float * src, void * dst, int64_t value_count) {
    adapter_from_float(GGML_TURBO_KV_FORMAT_TURBO2, src, dst, value_count);
}

void ggml_turbo3_from_float(const float * src, void * dst, int64_t value_count) {
    adapter_from_float(GGML_TURBO_KV_FORMAT_TURBO3, src, dst, value_count);
}

size_t ggml_turbo2_quantize_rows(
        const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return adapter_quantize_rows(GGML_TURBO_KV_FORMAT_TURBO2, src, dst, nrows, n_per_row, imatrix);
}

size_t ggml_turbo3_quantize_rows(
        const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix) {
    return adapter_quantize_rows(GGML_TURBO_KV_FORMAT_TURBO3, src, dst, nrows, n_per_row, imatrix);
}

void ggml_turbo8_from_float(const float * src, void * dst, int64_t value_count) {
    adapter_from_float(GGML_TURBO_KV_FORMAT_TURBO8, src, dst, value_count);
}

size_t ggml_turbo4_quantize_rows(
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    return adapter_quantize_rows(
        GGML_TURBO_KV_FORMAT_TURBO4, src, dst, nrows, n_per_row, imatrix);
}

size_t ggml_turbo8_quantize_rows(
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix) {
    return adapter_quantize_rows(
        GGML_TURBO_KV_FORMAT_TURBO8, src, dst, nrows, n_per_row, imatrix);
}

const char * ggml_turbo_kv_status_string(enum ggml_turbo_kv_status status) {
    switch (status) {
        case GGML_TURBO_KV_STATUS_OK:
            return "ok";
        case GGML_TURBO_KV_STATUS_INVALID_ARGUMENT:
            return "invalid argument";
        case GGML_TURBO_KV_STATUS_UNSUPPORTED_FORMAT:
            return "unsupported format";
        case GGML_TURBO_KV_STATUS_COUNT_NOT_ALIGNED:
            return "value count is not a multiple of 128";
        case GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL:
            return "buffer too small";
    }
    return "unknown status";
}

} // extern "C"
