// SPDX-License-Identifier: MIT
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_TURBO_KV_BLOCK_ELEMENTS 128u
#define GGML_TURBO_KV_SUBBLOCK_ELEMENTS 32u
#define GGML_TURBO2_BLOCK_BYTES      40u
#define GGML_TURBO3_BLOCK_BYTES      56u
#define GGML_TURBO4_BLOCK_BYTES      66u
#define GGML_TURBO8_BLOCK_BYTES      130u
#define GGML_TURBO1_TCQ_BLOCK_BYTES  20u
#define GGML_TURBO2_TCQ_BLOCK_BYTES  36u
#define GGML_TURBO3_TCQ_BLOCK_BYTES  52u

enum ggml_turbo_kv_format {
    GGML_TURBO_KV_FORMAT_TURBO2 = 2,
    GGML_TURBO_KV_FORMAT_TURBO3 = 3,
    GGML_TURBO_KV_FORMAT_TURBO4 = 4,
    GGML_TURBO_KV_FORMAT_TURBO8 = 8,
    GGML_TURBO_KV_FORMAT_TURBO1_TCQ = 11,
    GGML_TURBO_KV_FORMAT_TURBO2_TCQ = 12,
    GGML_TURBO_KV_FORMAT_TURBO3_TCQ = 13,
};

enum ggml_turbo_kv_status {
    GGML_TURBO_KV_STATUS_OK                  = 0,
    GGML_TURBO_KV_STATUS_INVALID_ARGUMENT    = 1,
    GGML_TURBO_KV_STATUS_UNSUPPORTED_FORMAT  = 2,
    GGML_TURBO_KV_STATUS_COUNT_NOT_ALIGNED   = 3,
    GGML_TURBO_KV_STATUS_BUFFER_TOO_SMALL    = 4,
};

struct ggml_turbo2_subblock {
    uint16_t scale;
    uint8_t  qs[GGML_TURBO_KV_SUBBLOCK_ELEMENTS / 4];
};

struct ggml_turbo2_block {
    struct ggml_turbo2_subblock subblocks[GGML_TURBO_KV_BLOCK_ELEMENTS / GGML_TURBO_KV_SUBBLOCK_ELEMENTS];
};

struct ggml_turbo3_subblock {
    uint16_t scale;
    uint8_t  qs[GGML_TURBO_KV_SUBBLOCK_ELEMENTS / 4];
    uint8_t  high[GGML_TURBO_KV_SUBBLOCK_ELEMENTS / 8];
};

struct ggml_turbo3_block {
    struct ggml_turbo3_subblock subblocks[GGML_TURBO_KV_BLOCK_ELEMENTS / GGML_TURBO_KV_SUBBLOCK_ELEMENTS];
};

struct ggml_turbo4_block {
    uint16_t scale;
    uint8_t  qs[GGML_TURBO_KV_BLOCK_ELEMENTS / 2];
};

struct ggml_turbo8_block {
    uint16_t scale;
    uint8_t  qs[GGML_TURBO_KV_BLOCK_ELEMENTS];
};

/*
 * TCQ bitstreams are little-endian sliding windows. The prefix contains the
 * L-k surviving bits of the free initial state, followed by 128 k-bit output
 * symbols. State t is therefore decoded in O(1) by reading L bits at t*k.
 */
struct ggml_turbo1_tcq_block {
    uint16_t scale;
    uint8_t  qs[17];
    uint8_t  padding;
};

struct ggml_turbo2_tcq_block {
    uint16_t scale;
    uint8_t  qs[33];
    uint8_t  padding;
};

struct ggml_turbo3_tcq_block {
    uint16_t scale;
    uint8_t  qs[49];
    uint8_t  padding;
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
void ggml_turbo2_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo3_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo4_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo8_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo2_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo3_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo1_tcq_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo2_tcq_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo3_tcq_to_float(const void * src, float * dst, int64_t value_count);
void ggml_turbo1_tcq_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo2_tcq_from_float(const float * src, void * dst, int64_t value_count);
void ggml_turbo3_tcq_from_float(const float * src, void * dst, int64_t value_count);

size_t ggml_turbo2_quantize_rows(
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix);

size_t ggml_turbo3_quantize_rows(
        const float * src,
        void * dst,
        int64_t nrows,
        int64_t n_per_row,
        const float * imatrix);

size_t ggml_turbo1_tcq_quantize_rows(
        const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t ggml_turbo2_tcq_quantize_rows(
        const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
size_t ggml_turbo3_tcq_quantize_rows(
        const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

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
