#include "common.cuh"

static __device__ __forceinline__ void dequantize_tq2_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_tq2_0 * x = (const block_tq2_0 *) vx;
    const float d = __half2float(x[ib].d);

    // Packing: element e -> byte = 32*(e/128) + (e%32), lane = (e/32)%4.
    const int e0 = (int) iqs;
    const int e1 = e0 + 1;
    const int b0 = 32 * (e0 >> 7) + (e0 & 31);
    const int l0 = (e0 >> 5) & 3;
    const int b1 = 32 * (e1 >> 7) + (e1 & 31);
    const int l1 = (e1 >> 5) & 3;
    const int c0 = (x[ib].qs[b0] >> (2 * l0)) & 0x3;
    const int c1 = (x[ib].qs[b1] >> (2 * l1)) & 0x3;
#ifdef GGML_CUDA_F16
    v.x = __float2half((c0 - 1) * d);
    v.y = __float2half((c1 - 1) * d);
#else
    v.x = (c0 - 1) * d;
    v.y = (c1 - 1) * d;
#endif
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = kvalues_iq4nl[vui & 0xF];
    v.y = kvalues_iq4nl[vui >>  4];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x = v.x * d;
    v.y = v.y * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q6_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q6_0 * x = (const block_q6_0 *) vx;

    const dfloat d = x[ib].d;

    const uint8_t h = x[ib].qh[iqs%8] >> 4*(iqs/8);
    v.x = ((x[ib].qs[iqs] & 0xf) | ((h & 0x3) << 4));
    v.y = ((x[ib].qs[iqs] >>  4) | ((h & 0xc) << 2));

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {32.0f, 32.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 32.0f) * d;
    v.y = (v.y - 32.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}
