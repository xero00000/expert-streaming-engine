// SPDX-License-Identifier: MIT
/*
 * Native CUDA row codecs for the checked ESE Turbo4/Turbo8 CPU references.
 *
 * The pinned buun CUDA kernels use a different FWHT rotation than the pinned
 * scalar reference. ESE deliberately uploads the exact accepted Gaussian
 * rotation and centroid tables instead, so CUDA storage and reconstruction
 * remain compatible with the CPU/GGUF contract established in Phase 1.
 */

#include "turbo-kv.cuh"

#include "ggml-turbo-kv.h"
#include "../ggml-turbo-kv-internal.h"

#include <cfloat>
#include <cstdint>
#include <mutex>

static __device__ float d_turbo_rotation_forward[GGML_TURBO_KV_BLOCK_ELEMENTS * GGML_TURBO_KV_BLOCK_ELEMENTS];
static __device__ float d_turbo_rotation_inverse[GGML_TURBO_KV_BLOCK_ELEMENTS * GGML_TURBO_KV_BLOCK_ELEMENTS];
static __constant__ float d_turbo_centroids2[4];
static __constant__ float d_turbo_centroids3[8];
static __constant__ float d_turbo_centroids4[16];
static __constant__ float d_turbo_centroids8[256];

static void ggml_cuda_turbo_kv_ensure_tables() {
    static std::mutex mutex;
    static bool initialized[GGML_CUDA_MAX_DEVICES] = {};

    const int device = ggml_cuda_get_device();
    GGML_ASSERT(device >= 0 && device < GGML_CUDA_MAX_DEVICES);

    std::lock_guard<std::mutex> lock(mutex);
    if (initialized[device]) {
        return;
    }

    constexpr size_t matrix_bytes =
        GGML_TURBO_KV_BLOCK_ELEMENTS * GGML_TURBO_KV_BLOCK_ELEMENTS * sizeof(float);
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_rotation_forward, ggml_turbo_kv_rotation_forward(), matrix_bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_rotation_inverse, ggml_turbo_kv_rotation_inverse(), matrix_bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_centroids2, ggml_turbo_kv_centroids2(), 4 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_centroids3, ggml_turbo_kv_centroids3(), 8 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_centroids4, ggml_turbo_kv_centroids4(), 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_centroids8, ggml_turbo_kv_centroids8(), 256 * sizeof(float)));
    initialized[device] = true;
}

template<int format>
static __device__ __forceinline__ float turbo_centroid(int index);

template<>
__device__ __forceinline__ float turbo_centroid<2>(int index) {
    return d_turbo_centroids2[index];
}

template<>
__device__ __forceinline__ float turbo_centroid<3>(int index) {
    return d_turbo_centroids3[index];
}

template<>
__device__ __forceinline__ float turbo_centroid<4>(int index) {
    return d_turbo_centroids4[index];
}

template<>
__device__ __forceinline__ float turbo_centroid<8>(int index) {
    return d_turbo_centroids8[index];
}

template<int format>
static __device__ __forceinline__ int turbo_index(const void * block, int element);

template<>
__device__ __forceinline__ int turbo_index<2>(const void * block, int element) {
    const auto * value = static_cast<const ggml_turbo2_block *>(block);
    const auto & subblock = value->subblocks[element / GGML_TURBO_KV_SUBBLOCK_ELEMENTS];
    const int local = element % GGML_TURBO_KV_SUBBLOCK_ELEMENTS;
    return (subblock.qs[local / 4] >> ((local % 4) * 2)) & 0x03;
}

template<>
__device__ __forceinline__ int turbo_index<3>(const void * block, int element) {
    const auto * value = static_cast<const ggml_turbo3_block *>(block);
    const auto & subblock = value->subblocks[element / GGML_TURBO_KV_SUBBLOCK_ELEMENTS];
    const int local = element % GGML_TURBO_KV_SUBBLOCK_ELEMENTS;
    const int low = (subblock.qs[local / 4] >> ((local % 4) * 2)) & 0x03;
    const int high = (subblock.high[local / 8] >> (local % 8)) & 0x01;
    return low | (high << 2);
}

template<>
__device__ __forceinline__ int turbo_index<4>(const void * block, int element) {
    const auto * value = static_cast<const ggml_turbo4_block *>(block);
    const uint8_t packed = value->qs[element / 2];
    return (element & 1) ? packed >> 4 : packed & 0x0f;
}

template<>
__device__ __forceinline__ int turbo_index<8>(const void * block, int element) {
    return static_cast<const ggml_turbo8_block *>(block)->qs[element];
}

template<int format>
static __device__ __forceinline__ float turbo_scale(const void * block);

template<>
__device__ __forceinline__ float turbo_scale<2>(const void * block) {
    return __half2float(__ushort_as_half(static_cast<const ggml_turbo2_block *>(block)->subblocks[0].scale));
}

template<>
__device__ __forceinline__ float turbo_scale<3>(const void * block) {
    return __half2float(__ushort_as_half(static_cast<const ggml_turbo3_block *>(block)->subblocks[0].scale));
}

template<>
__device__ __forceinline__ float turbo_scale<4>(const void * block) {
    return __half2float(__ushort_as_half(static_cast<const ggml_turbo4_block *>(block)->scale));
}

template<>
__device__ __forceinline__ float turbo_scale<8>(const void * block) {
    return __half2float(__ushort_as_half(static_cast<const ggml_turbo8_block *>(block)->scale));
}

template<int format>
static __device__ __forceinline__ size_t turbo_block_bytes();

template<>
__device__ __forceinline__ size_t turbo_block_bytes<2>() { return GGML_TURBO2_BLOCK_BYTES; }

template<>
__device__ __forceinline__ size_t turbo_block_bytes<3>() { return GGML_TURBO3_BLOCK_BYTES; }

template<>
__device__ __forceinline__ size_t turbo_block_bytes<4>() { return GGML_TURBO4_BLOCK_BYTES; }

template<>
__device__ __forceinline__ size_t turbo_block_bytes<8>() { return GGML_TURBO8_BLOCK_BYTES; }

template<int format>
static __device__ __forceinline__ int turbo_nearest(float value);

template<>
__device__ __forceinline__ int turbo_nearest<2>(float value) {
    int lo = 0;
    int hi = 3;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (value < 0.5f * (d_turbo_centroids2[mid] + d_turbo_centroids2[mid + 1])) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

template<>
__device__ __forceinline__ int turbo_nearest<3>(float value) {
    int lo = 0;
    int hi = 7;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        if (value < 0.5f * (d_turbo_centroids3[mid] + d_turbo_centroids3[mid + 1])) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

template<>
__device__ __forceinline__ int turbo_nearest<4>(float value) {
    int lo = 0;
    int hi = 15;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        const float threshold = 0.5f * (d_turbo_centroids4[mid] + d_turbo_centroids4[mid + 1]);
        if (value < threshold) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

template<>
__device__ __forceinline__ int turbo_nearest<8>(float value) {
    int lo = 0;
    int hi = 255;
    while (lo < hi) {
        const int mid = (lo + hi) / 2;
        const float threshold = 0.5f * (d_turbo_centroids8[mid] + d_turbo_centroids8[mid + 1]);
        if (value < threshold) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

template<int format>
static __device__ __forceinline__ void turbo_store_block(
        const float * src,
        void * dst) {
    constexpr int qk = GGML_TURBO_KV_BLOCK_ELEMENTS;

    float norm_sq = 0.0f;
    for (int i = 0; i < qk; ++i) {
        norm_sq += src[i] * src[i];
    }
    const float norm = sqrtf(norm_sq);
    const float inv_norm = norm > 1.0e-10f ? 1.0f / norm : 0.0f;

    uint8_t indices[qk];
    float reconstructed_norm_sq = 0.0f;
    for (int row = 0; row < qk; ++row) {
        float rotated = 0.0f;
        for (int column = 0; column < qk; ++column) {
            const float normalized = src[column] * inv_norm;
            rotated += d_turbo_rotation_forward[row * qk + column] * normalized;
        }
        const int index = turbo_nearest<format>(rotated);
        indices[row] = static_cast<uint8_t>(index);
        const float reconstructed = turbo_centroid<format>(index);
        reconstructed_norm_sq += reconstructed * reconstructed;
    }

    const float reconstructed_norm = sqrtf(reconstructed_norm_sq);
    const float scale = reconstructed_norm > 1.0e-10f ? norm / reconstructed_norm : norm;
    const uint16_t scale_fp16 = __half_as_ushort(__float2half_rn(scale));

    if constexpr (format == 2) {
        auto * output = static_cast<ggml_turbo2_block *>(dst);
        for (int subblock = 0; subblock < 4; ++subblock) {
            output->subblocks[subblock].scale = scale_fp16;
            for (int i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS; i += 4) {
                const int offset = subblock * GGML_TURBO_KV_SUBBLOCK_ELEMENTS + i;
                output->subblocks[subblock].qs[i / 4] = indices[offset] |
                    (indices[offset + 1] << 2) | (indices[offset + 2] << 4) | (indices[offset + 3] << 6);
            }
        }
    } else if constexpr (format == 3) {
        auto * output = static_cast<ggml_turbo3_block *>(dst);
        for (int subblock = 0; subblock < 4; ++subblock) {
            output->subblocks[subblock].scale = scale_fp16;
            for (int i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS / 4; ++i) output->subblocks[subblock].qs[i] = 0;
            for (int i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS / 8; ++i) output->subblocks[subblock].high[i] = 0;
            for (int i = 0; i < GGML_TURBO_KV_SUBBLOCK_ELEMENTS; ++i) {
                const int index = indices[subblock * GGML_TURBO_KV_SUBBLOCK_ELEMENTS + i];
                output->subblocks[subblock].qs[i / 4] |= (index & 0x03) << ((i % 4) * 2);
                output->subblocks[subblock].high[i / 8] |= ((index >> 2) & 0x01) << (i % 8);
            }
        }
    } else if constexpr (format == 4) {
        auto * output = static_cast<ggml_turbo4_block *>(dst);
        output->scale = scale_fp16;
        for (int i = 0; i < qk; i += 2) {
            output->qs[i / 2] = indices[i] | (indices[i + 1] << 4);
        }
    } else {
        auto * output = static_cast<ggml_turbo8_block *>(dst);
        output->scale = scale_fp16;
        for (int i = 0; i < qk; ++i) {
            output->qs[i] = indices[i];
        }
    }
}

template<int format, typename index_t>
static __global__ void k_turbo_set_rows(
        const float * src0,
        const index_t * src1,
        void * dst,
        int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
        int64_t ne11, int64_t ne12,
        int64_t s01, int64_t s02, int64_t s03,
        int64_t s10, int64_t s11, int64_t s12,
        int64_t nb1, int64_t nb2, int64_t nb3) {
    constexpr int qk = GGML_TURBO_KV_BLOCK_ELEMENTS;
    const int64_t block_index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total_blocks = ne00 * ne01 * ne02 * ne03 / qk;
    if (block_index >= total_blocks) return;

    const int64_t base = block_index * qk;
    const int64_t i03 = base / (ne00 * ne01 * ne02);
    const int64_t i02 = (base / (ne00 * ne01)) % ne02;
    const int64_t i01 = (base / ne00) % ne01;
    const int64_t i00 = base % ne00;
    const int64_t i12 = i03 % ne12;
    const int64_t i11 = i02 % ne11;
    const int64_t dst_row = src1[i01 * s10 + i11 * s11 + i12 * s12];
    if (dst_row < 0) return;

    const float * src_block = src0 + i01 * s01 + i02 * s02 + i03 * s03 + i00;
    char * dst_row_ptr = static_cast<char *>(dst) + dst_row * nb1 + i02 * nb2 + i03 * nb3;
    void * dst_block = dst_row_ptr + (i00 / qk) * turbo_block_bytes<format>();
    turbo_store_block<format>(src_block, dst_block);
}

template<int format>
static __global__ void k_turbo_get_rows(
        const void * src0,
        const int32_t * src1,
        float * dst,
        int64_t ne00, int64_t ne01, int64_t ne10, int64_t ne11, int64_t ne12,
        size_t nb01, size_t nb02, size_t nb03,
        size_t s1, size_t s2, size_t s3,
        size_t s10, size_t s11, size_t s12) {
    constexpr int qk = GGML_TURBO_KV_BLOCK_ELEMENTS;
    const int64_t i00 = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t i10 = blockIdx.y;
    const int64_t flat = blockIdx.z;
    const int64_t i11 = flat / ne12;
    const int64_t i12 = flat % ne12;
    if (i00 >= ne00 || i10 >= ne10 || i11 >= ne11) return;

    const int32_t i01 = src1[i10 * s10 + i11 * s11 + i12 * s12];
    float * dst_row = dst + i10 * s1 + i11 * s2 + i12 * s3;
    if (i01 < 0 || i01 >= ne01) {
        dst_row[i00] = 0.0f;
        return;
    }

    const char * src_row = static_cast<const char *>(src0) + i01 * nb01 + i11 * nb02 + i12 * nb03;
    const int64_t block_index = i00 / qk;
    const int element = i00 % qk;
    const void * block = src_row + block_index * turbo_block_bytes<format>();

    float reconstructed = 0.0f;
    for (int column = 0; column < qk; ++column) {
        reconstructed += d_turbo_rotation_inverse[element * qk + column] *
            turbo_centroid<format>(turbo_index<format>(block, column));
    }
    dst_row[i00] = reconstructed * turbo_scale<format>(block);
}

template<int format, typename index_t>
static void launch_set_rows(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    constexpr int qk = GGML_TURBO_KV_BLOCK_ELEMENTS;
    const int64_t total_blocks = ne00 * ne01 * ne02 * ne03 / qk;
    constexpr int threads = 64;
    const int blocks = int((total_blocks + threads - 1) / threads);
    if (blocks == 0) return;

    k_turbo_set_rows<format, index_t><<<blocks, threads, 0, stream>>>(
        static_cast<const float *>(src0->data), static_cast<const index_t *>(src1->data), dst->data,
        ne00, ne01, ne02, ne03, ne11, ne12,
        nb01 / sizeof(float), nb02 / sizeof(float), nb03 / sizeof(float),
        nb10 / sizeof(index_t), nb11 / sizeof(index_t), nb12 / sizeof(index_t),
        nb1, nb2, nb3);
}

static int turbo_format_for_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_TURBO2_0: return 2;
        case GGML_TYPE_TURBO3_0: return 3;
        case GGML_TYPE_TURBO4_0: return 4;
        case GGML_TYPE_TURBO8_0: return 8;
        default: return 0;
    }
}

void ggml_cuda_turbo_kv_set_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    ggml_cuda_turbo_kv_ensure_tables();
    const int format = turbo_format_for_type(dst->type);
    GGML_ASSERT(format != 0);

    if (src1->type == GGML_TYPE_I64) {
        if (format == 2) launch_set_rows<2, int64_t>(src0, src1, dst, ctx.stream());
        else if (format == 3) launch_set_rows<3, int64_t>(src0, src1, dst, ctx.stream());
        else if (format == 4) launch_set_rows<4, int64_t>(src0, src1, dst, ctx.stream());
        else launch_set_rows<8, int64_t>(src0, src1, dst, ctx.stream());
    } else {
        if (format == 2) launch_set_rows<2, int32_t>(src0, src1, dst, ctx.stream());
        else if (format == 3) launch_set_rows<3, int32_t>(src0, src1, dst, ctx.stream());
        else if (format == 4) launch_set_rows<4, int32_t>(src0, src1, dst, ctx.stream());
        else launch_set_rows<8, int32_t>(src0, src1, dst, ctx.stream());
    }
}

template<int format>
static void launch_get_rows(
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst,
        cudaStream_t stream) {
    GGML_TENSOR_BINARY_OP_LOCALS
    constexpr int threads = 128;
    const dim3 blocks((ne00 + threads - 1) / threads, ne10, ne11 * ne12);
    k_turbo_get_rows<format><<<blocks, threads, 0, stream>>>(
        src0->data, static_cast<const int32_t *>(src1->data), static_cast<float *>(dst->data),
        ne00, ne01, ne10, ne11, ne12, nb01, nb02, nb03,
        nb1 / sizeof(float), nb2 / sizeof(float), nb3 / sizeof(float),
        nb10 / sizeof(int32_t), nb11 / sizeof(int32_t), nb12 / sizeof(int32_t));
}

void ggml_cuda_turbo_kv_get_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    ggml_cuda_turbo_kv_ensure_tables();
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    const int format = turbo_format_for_type(src0->type);
    GGML_ASSERT(format != 0);
    if (format == 2) launch_get_rows<2>(src0, src1, dst, ctx.stream());
    else if (format == 3) launch_get_rows<3>(src0, src1, dst, ctx.stream());
    else if (format == 4) launch_get_rows<4>(src0, src1, dst, ctx.stream());
    else launch_get_rows<8>(src0, src1, dst, ctx.stream());
}

template<int format>
static __global__ void k_turbo_dequantize_f16(
        const void * src,
        half * dst,
        int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3) {
    constexpr int qk = GGML_TURBO_KV_BLOCK_ELEMENTS;
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = ne0 * ne1 * ne2 * ne3;
    if (index >= total) return;

    const int64_t i3 = index / (ne0 * ne1 * ne2);
    const int64_t i2 = (index / (ne0 * ne1)) % ne2;
    const int64_t i1 = (index / ne0) % ne1;
    const int64_t i0 = index % ne0;
    const char * row = static_cast<const char *>(src) + i1 * nb1 + i2 * nb2 + i3 * nb3;
    const void * block = row + (i0 / qk) * turbo_block_bytes<format>();

    float value = 0.0f;
    for (int column = 0; column < qk; ++column) {
        value += d_turbo_rotation_inverse[(i0 % qk) * qk + column] *
            turbo_centroid<format>(turbo_index<format>(block, column));
    }
    dst[index] = __float2half(value * turbo_scale<format>(block));
}

void ggml_cuda_turbo_kv_dequantize_f16(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src,
        void * dst) {
    ggml_cuda_turbo_kv_ensure_tables();
    const int format = turbo_format_for_type(src->type);
    GGML_ASSERT(format != 0);
    GGML_ASSERT(src->ne[0] % GGML_TURBO_KV_BLOCK_ELEMENTS == 0);

    const int64_t total = ggml_nelements(src);
    constexpr int threads = 128;
    const int blocks = int((total + threads - 1) / threads);
    if (format == 2) {
        k_turbo_dequantize_f16<2><<<blocks, threads, 0, ctx.stream()>>>(
            src->data, static_cast<half *>(dst),
            src->ne[0], src->ne[1], src->ne[2], src->ne[3], src->nb[1], src->nb[2], src->nb[3]);
    } else if (format == 3) {
        k_turbo_dequantize_f16<3><<<blocks, threads, 0, ctx.stream()>>>(
            src->data, static_cast<half *>(dst),
            src->ne[0], src->ne[1], src->ne[2], src->ne[3], src->nb[1], src->nb[2], src->nb[3]);
    } else if (format == 4) {
        k_turbo_dequantize_f16<4><<<blocks, threads, 0, ctx.stream()>>>(
            src->data, static_cast<half *>(dst),
            src->ne[0], src->ne[1], src->ne[2], src->ne[3],
            src->nb[1], src->nb[2], src->nb[3]);
    } else {
        k_turbo_dequantize_f16<8><<<blocks, threads, 0, ctx.stream()>>>(
            src->data, static_cast<half *>(dst),
            src->ne[0], src->ne[1], src->ne[2], src->ne[3],
            src->nb[1], src->nb[2], src->nb[3]);
    }
}

template<int format_k, int format_v>
static __global__ void k_turbo_flash_attn_direct(
        const char * __restrict__ q_data,
        const char * __restrict__ k_data,
        const char * __restrict__ v_data,
        const char * __restrict__ mask_data,
        const float * __restrict__ sinks,
        float * __restrict__ dst,
        float scale,
        float max_bias,
        float m0,
        float m1,
        uint32_t n_head_log2,
        float logit_softcap,
        int32_t ne01,
        int32_t ne02,
        int32_t ne03,
        size_t nb01,
        size_t nb02,
        size_t nb03,
        int32_t ne11,
        int32_t ne12,
        size_t nb11,
        size_t nb12,
        size_t nb13,
        size_t nb21,
        size_t nb22,
        size_t nb23,
        int32_t ne33,
        size_t nb31,
        size_t nb33) {
    constexpr int qk = GGML_TURBO_KV_BLOCK_ELEMENTS;
    const int element = threadIdx.x;
    const int query = blockIdx.x;
    const int sequence = blockIdx.y / ne02;
    const int head = blockIdx.y % ne02;
    if (element >= qk || query >= ne01 || sequence >= ne03) return;

    const int gqa_ratio = ne02 / ne12;
    const int kv_head = head / gqa_ratio;
    const float * q = reinterpret_cast<const float *>(
        q_data + sequence * nb03 + head * nb02 + query * nb01);
    const char * k_head = k_data + sequence * nb13 + kv_head * nb12;
    const char * v_head = v_data + sequence * nb23 + kv_head * nb22;

    __shared__ float q_rot[qk];
    __shared__ float products[qk];
    __shared__ float v_acc[qk];
    __shared__ float state[4]; // max, denominator, old-output rescale, new weight

    float transformed_q = 0.0f;
    for (int original = 0; original < qk; ++original) {
        transformed_q += d_turbo_rotation_inverse[original * qk + element] * q[original];
    }
    q_rot[element] = transformed_q;
    v_acc[element] = 0.0f;
    if (element == 0) {
        state[0] = -FLT_MAX / 2.0f;
        state[1] = 0.0f;
    }
    __syncthreads();

    const uint32_t h = static_cast<uint32_t>(head);
    const float slope = max_bias <= 0.0f ? 1.0f : powf(
        h < n_head_log2 ? m0 : m1,
        static_cast<float>(h < n_head_log2 ? h + 1 : 2 * (h - n_head_log2) + 1));
    const half * mask = mask_data ? reinterpret_cast<const half *>(
        mask_data + (sequence % ne33) * nb33 + query * nb31) : nullptr;

    for (int key = 0; key < ne11; ++key) {
        const void * k_block = k_head + key * nb11;
        const float k_value = turbo_centroid<format_k>(turbo_index<format_k>(k_block, element)) *
            turbo_scale<format_k>(k_block);
        const float partial = warp_reduce_sum(q_rot[element] * k_value);
        if ((element & (WARP_SIZE - 1)) == 0) {
            products[element / WARP_SIZE] = partial;
        }
        __syncthreads();

        if (element == 0) {
            float score = 0.0f;
            for (int warp = 0; warp < qk / WARP_SIZE; ++warp) score += products[warp];
            score *= scale;
            if (logit_softcap != 0.0f) score = logit_softcap * tanhf(score / logit_softcap);
            if (mask) score += slope * __half2float(mask[key]);

            const float new_max = fmaxf(state[0], score);
            state[2] = expf(state[0] - new_max);
            state[3] = expf(score - new_max);
            state[0] = new_max;
            state[1] = state[1] * state[2] + state[3];
        }
        __syncthreads();

        const void * v_block = v_head + key * nb21;
        const float v_value = turbo_centroid<format_v>(turbo_index<format_v>(v_block, element)) *
            turbo_scale<format_v>(v_block);
        v_acc[element] = v_acc[element] * state[2] + v_value * state[3];
        __syncthreads();
    }

    if (sinks) {
        if (element == 0) {
            const float new_max = fmaxf(state[0], sinks[head]);
            state[2] = expf(state[0] - new_max);
            state[3] = expf(sinks[head] - new_max);
            state[0] = new_max;
            state[1] = state[1] * state[2] + state[3];
        }
        __syncthreads();
        v_acc[element] *= state[2];
        __syncthreads();
    }

    products[element] = state[1] > 0.0f ? v_acc[element] / state[1] : 0.0f;
    __syncthreads();

    float output = 0.0f;
    for (int rotated = 0; rotated < qk; ++rotated) {
        output += d_turbo_rotation_inverse[element * qk + rotated] * products[rotated];
    }
    dst[((sequence * ne01 + query) * ne02 + head) * qk + element] = output;
}

bool ggml_cuda_turbo_kv_fattn_is_supported(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    return q->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        q->ne[0] == GGML_TURBO_KV_BLOCK_ELEMENTS &&
        q->ne[1] <= 8 &&
        k->ne[0] == GGML_TURBO_KV_BLOCK_ELEMENTS &&
        v->ne[0] == GGML_TURBO_KV_BLOCK_ELEMENTS &&
        turbo_format_for_type(k->type) != 0 && turbo_format_for_type(v->type) != 0 &&
        k->ne[1] == v->ne[1] && k->ne[2] == v->ne[2] && k->ne[3] == v->ne[3] &&
        q->ne[2] % k->ne[2] == 0 &&
        (!mask || (mask->type == GGML_TYPE_F16 &&
            mask->ne[0] >= k->ne[1] && mask->ne[1] >= GGML_PAD(q->ne[1], 16)));
}

template<int format_k, int format_v>
static void launch_turbo_flash_attn_direct(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        float max_bias,
        float m0,
        float m1,
        uint32_t n_head_log2,
        float logit_softcap) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const dim3 blocks(q->ne[1], q->ne[2] * q->ne[3], 1);
    k_turbo_flash_attn_direct<format_k, format_v><<<blocks, GGML_TURBO_KV_BLOCK_ELEMENTS, 0, ctx.stream()>>>(
        static_cast<const char *>(q->data),
        static_cast<const char *>(k->data),
        static_cast<const char *>(v->data),
        mask ? static_cast<const char *>(mask->data) : nullptr,
        sinks ? static_cast<const float *>(sinks->data) : nullptr,
        static_cast<float *>(dst->data),
        scale, max_bias, m0, m1, n_head_log2, logit_softcap,
        q->ne[1], q->ne[2], q->ne[3], q->nb[1], q->nb[2], q->nb[3],
        k->ne[1], k->ne[2], k->nb[1], k->nb[2], k->nb[3],
        v->nb[1], v->nb[2], v->nb[3],
        mask ? mask->ne[3] : 1, mask ? mask->nb[1] : 0, mask ? mask->nb[3] : 0);
    CUDA_CHECK(cudaGetLastError());
}

template<int format_k>
static void launch_turbo_flash_attn_for_v(
        int format_v,
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst,
        float scale,
        float max_bias,
        float m0,
        float m1,
        uint32_t n_head_log2,
        float logit_softcap) {
    if (format_v == 2) launch_turbo_flash_attn_direct<format_k, 2>(ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    else if (format_v == 3) launch_turbo_flash_attn_direct<format_k, 3>(ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    else if (format_v == 4) launch_turbo_flash_attn_direct<format_k, 4>(ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    else launch_turbo_flash_attn_direct<format_k, 8>(ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
}

void ggml_cuda_turbo_kv_flash_attn(
        ggml_backend_cuda_context & ctx,
        ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_turbo_kv_fattn_is_supported(dst));
    ggml_cuda_turbo_kv_ensure_tables();

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    const uint32_t n_head = dst->src[0]->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -max_bias / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int format_k = turbo_format_for_type(dst->src[1]->type);
    const int format_v = turbo_format_for_type(dst->src[2]->type);
    if (format_k == 2) launch_turbo_flash_attn_for_v<2>(format_v, ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    else if (format_k == 3) launch_turbo_flash_attn_for_v<3>(format_v, ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    else if (format_k == 4) launch_turbo_flash_attn_for_v<4>(format_v, ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
    else launch_turbo_flash_attn_for_v<8>(format_v, ctx, dst, scale, max_bias, m0, m1, n_head_log2, logit_softcap);
}
