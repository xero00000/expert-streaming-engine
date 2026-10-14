#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

enum ggml_qat_format {
    GGML_QAT_FP8_E8M0 = 0, // groups of 32, E4M3 values, ceil power-of-two scales
    GGML_QAT_FP4_E8M0 = 1, // groups of 32, E2M1 values, ceil power-of-two scales
    GGML_QAT_FP4_E4M3 = 2, // groups of 16, E2M1 values, rounded E4M3 scales
};

// Quantize/dequantize to FP32. This preserves the model's activation transform;
// it does not pack a physical cache or use the MSE-optimized weight quantizer.
GGML_API struct ggml_tensor * ggml_qat(struct ggml_context * ctx, struct ggml_tensor * input, enum ggml_qat_format format);

// Recognizes only this registered custom callback; all other custom operators
// remain ineligible for the dedicated accelerator kernel. Returns -1 otherwise.
GGML_API int ggml_qat_mode(const struct ggml_tensor * tensor);

// Canonical packed row: value bytes first (low nibble is the earlier FP4
// element), then one scale byte per group. Returns 0 for invalid geometry.
GGML_API size_t ggml_qat_row_size(int64_t width, enum ggml_qat_format format);
// False on nonfinite input/unrepresentable scale; destination is unspecified
// on failure. Callers must stage writes before publishing a cache row.
GGML_API bool ggml_qat_pack_row(const float * input, void * packed, int64_t width, enum ggml_qat_format format);
GGML_API bool ggml_qat_unpack_row(const void * packed, float * output, int64_t width, enum ggml_qat_format format);

#ifdef __cplusplus
}
#endif
