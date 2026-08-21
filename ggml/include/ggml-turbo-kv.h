// SPDX-License-Identifier: MIT
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TURBO_KV_BLOCK_ELEMENTS 128u
#define GGML_TURBO4_BLOCK_BYTES      66u
#define GGML_TURBO8_BLOCK_BYTES      130u

enum ggml_turbo_kv_format {
    GGML_TURBO_KV_FORMAT_TURBO4 = 4,
    GGML_TURBO_KV_FORMAT_TURBO8 = 8,
};

enum ggml_turbo_kv_status {
    GGML_TURBO_KV_STATUS_OK                  = 0,
    GGML_TURBO_KV_STATUS_INVALID_ARGUMENT    = 1,
    GGML_TURBO_KV_STATUS_UNSUPPORTED_FORMAT  = 2,
    GGML_TURBO_KV_STATUS_COUNT_NOT_ALIGNED   = 3,
    GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL    = 4,
};

struct ggml_turbo4_block {
    uint16_t scale;
    uint8_t  qs[GGML_TURBO_KV_BLOCK_ELEMENTS / 2];
};

struct ggml_turbo8_block {
    uint16_t scale;
    uint8_t  qs[GGML_TURBO_KV_BLOCK_ELEMENTS];
};

size_t ggml_turbo_kv_block_elements(void);
size_t ggml_turbo_kv_block_bytes(enum ggml_turbo_kv_format format);
double ggml_turbo_kv_bits_per_value(enum ggml_turbo_kv_format format);
size_t ggml_turbo_kv_encoded_size(enum ggml_turbo_kv_format format, size_t value_count);

enum ggml_turbo_kv_status ggml_turbo_kv_quantize_reference(
        enum ggml_turbo_kv_format format,
        const float * src,
        size_t value_count,
        void * dst,
        size_t dst_size);

enum ggml_turbo_kv_status ggml_turbo_kv_dequantize_reference(
        enum ggml_turbo_kv_format format,
        const void * src,
        size_t src_size,
        float * dst,
        size_t value_count);


/*
 * Internal GGML row adapters. These make the checked CPU references usable by
 * core type traits and ggml_quantize_chunk without exposing a server cache
 * option. The row width must be a multiple of 128.
 */
void ggml_turbo4_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo8_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo4_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo8_from_float(const float * src, void * dst, int64_t value_count);

size_t ggml_turbo4_quantize_rows(
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix);

size_t ggml_turbo8_quantize_rows(
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix);

const char * ggml_turbo_kv_status_string(enum ggml_turbo_kv_status status);

#ifdef __cplusplus
}
#endif
