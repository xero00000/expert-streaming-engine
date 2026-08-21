// SPDX-License-Identifier: MIT

#pragma once

#include "ggml.h"
#include "ggml-turbo-kv.h"

#include <cstdint>

static inline bool llama_kv_type_is_turbo(ggml_type type) {
    return type == GGML_TYPE_TURBO4_0 || type == GGML_TYPE_TURBO8_0;
}

static inline uint32_t llama_kv_head_dim_for_type(ggml_type type, uint32_t logical_head_dim) {
    return llama_kv_type_is_turbo(type)
        ? GGML_PAD(logical_head_dim, GGML_TURBO_KV_BLOCK_ELEMENTS)
        : logical_head_dim;
}

static inline uint32_t llama_kv_gqa_dim_for_type(
        ggml_type type,
        uint32_t logical_head_dim,
        uint32_t head_count) {
    return llama_kv_head_dim_for_type(type, logical_head_dim) * head_count;
}
