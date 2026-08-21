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

#include <mutex>

static __device__ float d_turbo_rotation_forward[GGML_TURBO_KV_BLOCK_ELEMENTS * GGML_TURBO_KV_BLOCK_ELEMENTS];
static __device__ float d_turbo_rotation_inverse[GGML_TURBO_KV_BLOCK_ELEMENTS * GGML_TURBO_KV_BLOCK_ELEMENTS];
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
        d_turbo_centroids4, ggml_turbo_kv_centroids4(), 16 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(
        d_turbo_centroids8, ggml_turbo_kv_centroids8(), 256 * sizeof(float)));
    initialized[device] = true;
}

template<int format>
static __device__ __forceinline__ float turbo_centroid(int index);

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
__device__ __forceinline__ size_t turbo_block_bytes<4>() { return GGML_TURBO4_BLOCK_BYTES; }

template<>
__device__ __forceinline__ size_t turbo_block_bytes<8>() { return GGML_TURBO8_BLOCK_BYTES; }

template<int format>
static __device__ __forceinline__ int turbo_nearest(float value);

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

    if constexpr (format == 4) {
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

void ggml_cuda_turbo_kv_set_rows(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {
    ggml_cuda_turbo_kv_ensure_tables();
    const int format = dst->type == GGML_TYPE_TURBO4_0 ? 4 : 8;
    GGML_ASSERT(dst->type == GGML_TYPE_TURBO4_0 || dst->type == GGML_TYPE_TURBO8_0);

    if (src1->type == GGML_TYPE_I64) {
        if (format == 4) launch_set_rows<4, int64_t>(src0, src1, dst, ctx.stream());
        else launch_set_rows<8, int64_t>(src0, src1, dst, ctx.stream());
    } else {
        if (format == 4) launch_set_rows<4, int32_t>(src0, src1, dst, ctx.stream());
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
    if (src0->type == GGML_TYPE_TURBO4_0) {
        launch_get_rows<4>(src0, src1, dst, ctx.stream());
    } else {
        GGML_ASSERT(src0->type == GGML_TYPE_TURBO8_0);
        launch_get_rows<8>(src0, src1, dst, ctx.stream());
    }
}
