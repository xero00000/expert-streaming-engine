#pragma once

#include "common.cuh"

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
