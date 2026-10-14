// Bounded original-weight/F32-input expert arithmetic for explicit precision
// requests and reference probes. Default arithmetic remains separate.
#pragma once

#include "ggml.h"
#include <math.h>
#include <string.h>

static inline bool ggml_expert_f32_enabled(const struct ggml_tensor * op) {
    return (op->op == GGML_OP_MUL_MAT_ID && op->op_params[0] == GGML_PREC_F32) ||
           (op->op == GGML_OP_MOE_FUSED_UP_GATE && op->op_params[2] == GGML_PREC_F32);
}

static inline bool ggml_expert_f32_type(enum ggml_type type) {
    return type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_Q2_0 ||
           type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ||
           type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS;
}

static inline bool ggml_expert_f32_supported(const struct ggml_tensor * op) {
    if (!ggml_expert_f32_enabled(op)) return false;
    const bool fused = op->op == GGML_OP_MOE_FUSED_UP_GATE;
    const struct ggml_tensor * w = op->src[0];
    const struct ggml_tensor * g = fused ? op->src[1] : NULL;
    const struct ggml_tensor * x = op->src[fused ? 2 : 1];
    const struct ggml_tensor * ids = op->src[fused ? 3 : 2];
    if (!w || !x || !ids || !ggml_expert_f32_type(w->type) || x->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 || op->type != GGML_TYPE_F32) return false;
    if (w->ne[3] != 1 || x->ne[3] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        w->ne[0] <= 0 || w->ne[1] <= 0 || w->ne[2] <= 0 || ids->ne[0] <= 0 || ids->ne[1] <= 0 ||
        w->ne[0] % ggml_blck_size(w->type) != 0 ||
        w->ne[0] != x->ne[0] || ids->ne[1] != x->ne[2] || x->ne[1] <= 0 ||
        ids->ne[0] % x->ne[1] != 0 || x->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float) || w->view_offs % 2 != 0 ||
        w->nb[1] % 2 != 0 || w->nb[2] % 2 != 0) return false;
    if (w->nb[0] != ggml_type_size(w->type) ||
        w->nb[1] < ggml_row_size(w->type, w->ne[0]) ||
        w->nb[2] < w->nb[1] * w->ne[1]) return false;
    const int64_t rows = fused && !g ? w->ne[1] / 2 : w->ne[1];
    if (op->ne[0] != rows || op->ne[1] != ids->ne[0] || op->ne[2] != ids->ne[1] ||
        op->ne[3] != 1 || op->nb[1] < rows * sizeof(float) ||
        op->nb[2] < op->nb[1] * ids->ne[0]) return false;
    if (fused) {
        float limit;
        memcpy(&limit, op->op_params + 1, sizeof(limit));
        if (op->op_params[0] != GGML_UNARY_OP_SILU || limit != 0 || op->src[4] || op->src[5]) return false;
        if (!g && w->ne[1] % 2 != 0) return false;
        if (g && (g->type != w->type || !ggml_are_same_shape(w, g) ||
                  g->view_offs % 2 != 0 || g->nb[1] % 2 != 0 || g->nb[2] % 2 != 0 ||
                  g->nb[0] != w->nb[0] || g->nb[1] < ggml_row_size(g->type, g->ne[0]) ||
                  g->nb[2] < g->nb[1] * g->ne[1])) return false;
    }
    return true;
}

static inline double ggml_expert_f32_to_double(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0) {
        const double magnitude = ldexp((double) (bits & 0x007fffffu), -149);
        return bits & 0x80000000u ? -magnitude : magnitude;
    }
    return (double) value;
}

static inline float ggml_expert_f32_dot(enum ggml_type type, const void * weights,
                                      const float * input, int64_t columns) {
    const int64_t block_size = ggml_blck_size(type);
    const size_t bytes = ggml_type_size(type);
    const ggml_to_float_t decode = ggml_internal_get_type_traits(type).to_float;
    GGML_ASSERT(decode && block_size > 0 && block_size <= 256);
    GGML_ASSERT(columns % block_size == 0);
    float decoded[256];
    double partial[32] = {0};
    for (int64_t base = 0; base < columns; base += block_size) {
        decode((const char *) weights + (base / block_size) * bytes, decoded, block_size);
        for (int64_t j = 0; j < block_size; ++j) {
            partial[(base + j) % 32] += ggml_expert_f32_to_double(decoded[j]) *
                                      ggml_expert_f32_to_double(input[base + j]);
        }
    }
    for (int stride = 16; stride > 0; stride /= 2) {
        for (int lane = 0; lane < stride; ++lane) {
            partial[lane] += partial[lane + stride];
        }
    }
    return (float) partial[0];
}

// Input is prepared once and shared across rows. These formats decode
// nonzero weights from FP16 scales into normal F32 values, so the inner loop
// needs no per-element subnormal reconstruction and can use SIMD.
static inline float ggml_expert_f32_dot_prepared(enum ggml_type type, const void * weights,
                                               const double * input, int64_t columns) {
    GGML_ASSERT(type == GGML_TYPE_IQ1_M || type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_Q2_0 ||
                type == GGML_TYPE_IQ2_S || type == GGML_TYPE_IQ3_S ||
                type == GGML_TYPE_IQ4_NL || type == GGML_TYPE_IQ4_XS);
    const int64_t block_size = ggml_blck_size(type);
    const size_t bytes = ggml_type_size(type);
    const ggml_to_float_t decode = ggml_internal_get_type_traits(type).to_float;
    GGML_ASSERT(decode && block_size <= 256 && block_size % 32 == 0);
    GGML_ASSERT(columns % block_size == 0);
    float decoded[256];
    double partial[32] = {0};
    for (int64_t base = 0; base < columns; base += block_size) {
        decode((const char *) weights + (base / block_size) * bytes, decoded, block_size);
        for (int64_t group = 0; group < block_size; group += 32) {
            for (int lane = 0; lane < 32; ++lane) {
                partial[lane] += (double) decoded[group + lane] * input[base + group + lane];
            }
        }
    }
    for (int stride = 16; stride > 0; stride /= 2) {
        for (int lane = 0; lane < stride; ++lane) {
            partial[lane] += partial[lane + stride];
        }
    }
    return (float) partial[0];
}

static inline float ggml_expert_f32_silu_product(float gate, float up) {
    const double g = ggml_expert_f32_to_double(gate);
    return (float) ((g / (1.0 + exp(-g))) * ggml_expert_f32_to_double(up));
}
