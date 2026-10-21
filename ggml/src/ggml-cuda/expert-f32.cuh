// Original-weight/F32-input expert arithmetic and explicit indexed dispatch.
#pragma once

#include "common.cuh"

bool ggml_cuda_expert_f32_supported_layout(const ggml_tensor * op);
void ggml_cuda_expert_f32(ggml_tensor * op, cudaStream_t stream);

static __device__ __forceinline__ double expert_f32_to_double(float value) {
    // CUDA fast math may flush subnormal F32 inputs in an ordinary cast.
    // PTX conversion preserves them directly, including signed zero, without
    // calling general-purpose ldexp for every zero or subnormal operand.
    double output;
    asm("cvt.f64.f32 %0, %1;" : "=d"(output) : "f"(value));
    return output;
}

static __device__ __forceinline__ float expert_f64_to_float(double value) {
    // Explicit conversion preserves subnormal F32 outputs with fast math enabled.
    float output;
    asm("cvt.rn.f32.f64 %0, %1;" : "=f"(output) : "d"(value));
    return output;
}

template <ggml_type type>
static __device__ __forceinline__ float expert_f32_weight(const void * weights, int column) {
    if constexpr (type == GGML_TYPE_Q6_K) {
        const auto * block = (const block_q6_K *) weights + column / QK_K;
        const int within = column % QK_K;
        const int half = within / 128, group = (within % 128) / 32, lane = within % 32;
        const int low = block->ql[half * 64 + lane + (group % 2) * 32];
        const int high = (block->qh[half * 32 + lane] >> (2 * group)) & 3;
        const int code = ((group < 2 ? low & 15 : low >> 4) | (high << 4)) - 32;
        const float scale = __half2float(block->d) * block->scales[half * 8 + lane / 16 + 2 * group];
        return scale * code;
    } else if constexpr (type == GGML_TYPE_IQ1_M) {
        const auto * block = (const block_iq1_m *) weights + column / QK_K;
        const int within = column % QK_K;
        const int segment = within / 32;
        const int group = (within % 32) / 8;
        const int position = within % 8;
        const uint16_t * sc = (const uint16_t *) block->scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
                    ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        const int high = block->qh[2 * segment + group / 2] >> (4 * (group % 2));
        const int index = block->qs[4 * segment + group] | ((high & 7) << 8);
        const int grid = (iq1s_grid_gpu[index] >> (8 * (position % 4) + 4 * (position / 4))) & 15;
        const float delta = (high & 8) ? -IQ1M_DELTA : IQ1M_DELTA;
        const int code = (sc[segment / 2] >> (6 * (segment % 2) + 3 * (group / 2))) & 7;
        const float d = __half2float(scale.f16) * (2 * code + 1);
        return d * (float(grid - 1) + delta);
    } else if constexpr (type == GGML_TYPE_IQ2_XXS) {
        const auto * block = (const block_iq2_xxs *) weights + column / QK_K;
        const int within = column % QK_K;
        const int segment = within / 32;
        const int group = (within % 32) / 8;
        const int position = within % 8;
        const uint8_t * codes = (const uint8_t *) block->qs + segment * 8;
        const uint8_t * auxiliary = codes + 4;
        const uint32_t aux = uint32_t(auxiliary[0]) | (uint32_t(auxiliary[1]) << 8) |
                             (uint32_t(auxiliary[2]) << 16) | (uint32_t(auxiliary[3]) << 24);
        const int magnitude = (iq2xxs_grid[codes[group]] >> (8 * position)) & 255;
        const bool negative = ((ksigns64[(aux >> (7 * group)) & 127] >> (8 * position)) & 255) != 0;
        const float scale = __half2float(block->d) * (0.5f + float(aux >> 28)) * 0.25f;
        return scale * (negative ? -magnitude : magnitude);
    } else if constexpr (type == GGML_TYPE_IQ3_XXS) {
        const auto * block = (const block_iq3_xxs *) weights + column / QK_K;
        const int within = column % QK_K;
        const int segment = within / 32;
        const int group = (within % 32) / 8;
        const int position = within % 8;
        const uint8_t * auxiliary = block->qs + 64 + 4 * segment;
        const uint32_t aux = uint32_t(auxiliary[0]) | (uint32_t(auxiliary[1]) << 8) |
                             (uint32_t(auxiliary[2]) << 16) | (uint32_t(auxiliary[3]) << 24);
        const uint32_t grid = iq3xxs_grid[block->qs[segment * 8 + group * 2 + position / 4]];
        const int magnitude = (grid >> (8 * (position % 4))) & 255;
        const bool negative = ((ksigns64[(aux >> (7 * group)) & 127] >> (8 * position)) & 255) != 0;
        const float scale = __half2float(block->d) * (0.5f + float(aux >> 28)) * 0.5f;
        return scale * (negative ? -magnitude : magnitude);
    } else if constexpr (type == GGML_TYPE_IQ2_S) {
        const auto * block = (const block_iq2_s *) weights + column / QK_K;
        const int within = column % QK_K;
        const int segment = within / 32;
        const int group = (within % 32) / 8;
        const int position = within % 8;
        const int index = block->qs[segment * 4 + group] | (((block->qh[segment] >> (2 * group)) & 3) << 8);
        const int magnitude = (iq2s_grid[index] >> (8 * position)) & 255;
        const bool negative = (block->qs[32 + segment * 4 + group] >> position) & 1;
        const int scale_code = (block->scales[segment] >> (4 * (group / 2))) & 15;
        const float scale = __half2float(block->d) * (0.5f + float(scale_code)) * 0.25f;
        return scale * (negative ? -magnitude : magnitude);
    } else if constexpr (type == GGML_TYPE_IQ3_S) {
        const auto * block = (const block_iq3_s *) weights + column / QK_K;
        const int within = column % QK_K;
        const int segment = within / 32;
        const int group = (within % 32) / 8;
        const int position = within % 8;
        const int side = position / 4;
        const int index = block->qs[segment * 8 + group * 2 + side] |
                          (((block->qh[segment] >> (2 * group + side)) & 1) << 8);
        const int magnitude = (iq3s_grid[index] >> (8 * (position % 4))) & 255;
        const bool negative = (block->signs[segment * 4 + group] >> position) & 1;
        const int scale_code = (block->scales[segment / 2] >> (4 * (segment % 2))) & 15;
        return (__half2float(block->d) * (1 + 2 * scale_code)) * (negative ? -magnitude : magnitude);
    } else if constexpr (type == GGML_TYPE_IQ4_NL) {
        const auto * block = (const block_iq4_nl *) weights + column / QK4_NL;
        const int within = column % QK4_NL;
        const int code = (block->qs[within % 16] >> (4 * (within / 16))) & 15;
        return __half2float(block->d) * kvalues_iq4nl[code];
    } else if constexpr (type == GGML_TYPE_IQ4_XS) {
        const auto * block = (const block_iq4_xs *) weights + column / QK_K;
        const int within = column % QK_K;
        const int segment = within / 32;
        const int position = within % 32;
        const int scale_code = ((block->scales_l[segment / 2] >> (4 * (segment % 2))) & 15) |
                               (((block->scales_h >> (2 * segment)) & 3) << 4);
        const int code = (block->qs[segment * 16 + position % 16] >> (4 * (position / 16))) & 15;
        return (__half2float(block->d) * (scale_code - 32)) * kvalues_iq4nl[code];
    } else {
        static_assert(type == GGML_TYPE_Q2_0, "unsupported prototype weight type");
        const auto * block = (const block_q2_0 *) weights + column / QK2_0;
        const int within = column % QK2_0;
        const int code = (block->qs[within / 4] >> (2 * (within % 4))) & 3;
        return __half2float(block->d) * (code - 1);
    }
}

static __device__ __forceinline__ double expert_f32_reduce(double partial) {
    for (int stride = 16; stride > 0; stride /= 2) {
        partial += __shfl_down_sync(0xffffffffu, partial, stride);
    }
    return partial;
}

template <ggml_type type>
static __global__ void expert_f32_project(const void * weights, const float * input,
                                         float * output, int columns, int rows, size_t row_bytes) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const char * weight_row = (const char *) weights + row * row_bytes;
    double partial = 0;
    for (int column = lane; column < columns; column += 32) {
        partial += expert_f32_to_double(expert_f32_weight<type>(weight_row, column)) *
                   expert_f32_to_double(input[column]);
    }
    const double sum = expert_f32_reduce(partial);
    if (lane == 0) {
        output[row] = expert_f64_to_float(sum);
    }
}

template <ggml_type type = GGML_TYPE_IQ3_XXS>
static __global__ void expert_f32_up_gate(const void * up, const void * gate, const float * input,
                                         float * hidden, float * up_output, float * gate_output,
                                         int columns, int rows, size_t row_bytes) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const char * up_row = (const char *) up + row * row_bytes;
    const char * gate_row = (const char *) gate + row * row_bytes;
    double partial_up = 0, partial_gate = 0;
    for (int column = lane; column < columns; column += 32) {
        const double x = expert_f32_to_double(input[column]);
        partial_up += expert_f32_to_double(expert_f32_weight<type>(up_row, column)) * x;
        partial_gate += expert_f32_to_double(expert_f32_weight<type>(gate_row, column)) * x;
    }
    const double sum_up = expert_f32_reduce(partial_up);
    const double sum_gate = expert_f32_reduce(partial_gate);
    if (lane == 0) {
        const float u = expert_f64_to_float(sum_up);
        const float g = expert_f64_to_float(sum_gate);
        if (up_output) {
            up_output[row] = u;
        }
        if (gate_output) {
            gate_output[row] = g;
        }
        const double gd = expert_f32_to_double(g);
        hidden[row] = expert_f64_to_float((gd / (1.0 + exp(-gd))) * expert_f32_to_double(u));
    }
}
