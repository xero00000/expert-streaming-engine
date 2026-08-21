// SPDX-License-Identifier: MIT

#pragma once

#include "ggml.h"
#include "ggml-turbo-kv.h"

#include <cstdint>

static inline bool llama_kv_type_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO2_0 || type == GGML_TYPE_TURBO3_0 ||
           type == GGML_TYPE_TURBO4_0 || type == GGML_TYPE_TURBO8_0 ||
           type == GGML_TYPE_TURBO1_TCQ || type == GGML_TYPE_TURBO2_TCQ ||
           type == GGML_TYPE_TURBO3_TCQ;
}

static inline uint32_t llama_kv_head_dim_for_type(ggml_type type, uint32_t logical_head_dim) {
    if (llama_kv_type_is_turbo(type)) {
        return GGML_PAD(logical_head_dim, GGML_TURBO_KV_BLOCK_ELEMENTS);
    }
    // Quantized cache rows must not let a block straddle adjacent attention
    // heads. Apart from being ambiguous to decode, from_float implementations
    // may write a complete final block and overrun an unpadded short row.
    if (ggml_is_quantized(type)) {
        return GGML_PAD(logical_head_dim, ggml_blck_size(type));
    }
    return logical_head_dim;
}

static inline uint32_t llama_kv_gqa_dim_for_type(
        ggml_type type,
        uint32_t logical_head_dim,
        uint32_t head_count) {
    return llama_kv_head_dim_for_type(type, logical_head_dim) * head_count;
}
