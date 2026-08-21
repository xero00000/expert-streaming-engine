#pragma once

#include "common.cuh"

static inline bool ggml_cuda_turbo_kv_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO2_0:
        case GGML_TYPE_TURBO3_0:
        case GGML_TYPE_TURBO4_0:
        case GGML_TYPE_TURBO8_0:
        case GGML_TYPE_TURBO1_TCQ:
        case GGML_TYPE_TURBO2_TCQ:
        case GGML_TYPE_TURBO3_TCQ:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_turbo_kv_get_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

void ggml_cuda_turbo_kv_set_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst);

void ggml_cuda_turbo_kv_quantize(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        ggml_tensor * dst);

void ggml_cuda_turbo_kv_dequantize_f16(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        void * dst);

bool ggml_cuda_turbo_kv_fattn_is_supported(const ggml_tensor * dst);

void ggml_cuda_turbo_kv_flash_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst);
