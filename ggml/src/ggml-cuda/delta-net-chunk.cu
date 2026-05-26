#include "common.cuh"
#include "delta-net.cuh"
#include "solve_tri.cuh"
#include "ggml-cuda/common.cuh"

#include <mma.h>

// 2026-05-26: ik_llama.cpp doesn't define these AMD/Moore-Threads detection
// macros (mainline llama.cpp does). They only affect calculate_smem() which
// boosts shared-memory limits for those backends. On NVIDIA (our only target,
// CUDA-only build) GGML_CUDA_CC_IS_AMD(cc) is always false so this whole
// branch is dead code — stubbing the inner macros to false is safe.
#ifndef GGML_CUDA_CC_IS_RDNA3
#define GGML_CUDA_CC_IS_RDNA3(cc) false
#endif
#ifndef GGML_CUDA_CC_IS_RDNA4
#define GGML_CUDA_CC_IS_RDNA4(cc) false
#endif
#ifndef GGML_CUDA_CC_IS_MTHREADS
#define GGML_CUDA_CC_IS_MTHREADS(cc) false
#endif

static constexpr int WM = 16, WN = 16, WK = 8;

template<int M, int N, int N_WARPS>
static __host__ __device__ constexpr int dot_max_tpw() {
    return ((M / WM) * (N / WN) + N_WARPS - 1) / N_WARPS;
}

template<int M, int N, int N_WARPS>
__device__ __forceinline__ void dot_init(
    int warp_id,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> acc[],
    int my_tiles[],
    int & n_my
) {
    using namespace nvcuda::wmma;
    constexpr int total = (M / WM) * (N / WN);
    n_my = 0;
#pragma unroll
    for (int tt = warp_id; tt < total; tt += N_WARPS) {
        fill_fragment(acc[n_my], 0.0f);
        my_tiles[n_my] = tt;
        n_my++;
    }
}

template<int M, int N, int K_DIM,
         typename A_LAYOUT, typename B_LAYOUT>
__device__ __forceinline__ void dot_mma(
    const float * smem_a, int lda,
    const float * smem_b, int ldb,
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, WM, WN, WK, float> acc[],
    const int my_tiles[],
    int n_my
) {
    using namespace nvcuda::wmma;
    constexpr int tiles_n = N / WN;

    for (int kk = 0; kk < K_DIM; kk += WK) {
#pragma unroll
        for (int tt = 0; tt < n_my; tt++) {
            const int tm = my_tiles[tt] / tiles_n;
            const int tn = my_tiles[tt] % tiles_n;

            fragment<matrix_a, WM, WN, WK, precision::tf32, A_LAYOUT> a_frag;
            fragment<matrix_b, WM, WN, WK, precision::tf32, B_LAYOUT> b_frag;

            const float * a_ptr = std::is_same_v<A_LAYOUT, col_major>
                ? smem_a + tm * WM + kk * lda
                : smem_a + kk + tm * WM * lda;

            const float * b_ptr = std::is_same_v<B_LAYOUT, col_major>
                ? smem_b + kk + tn * WN * ldb
                : smem_b + tn * WN + kk * ldb;

            load_matrix_sync(a_frag, a_ptr, lda);
            load_matrix_sync(b_frag, b_ptr, ldb);
            mma_sync(acc[tt], a_frag, b_frag, acc[tt]);
        }
    }
}

template <int S_v, bool KDA>
__global__ void __launch_bounds__(S_v, 1)
gated_delta_net_cuda(const float * q,
                     const float * k,
                     const float * v,
                     const float * g,
                     const float * beta,
                     const float * curr_state,
                     float *       dst,
                     const int64_t H,
                     const int64_t n_tokens,
                     const int64_t n_seqs,
                     const int64_t sq1,
                     const int64_t sq2,
                     const int64_t sq3,
                     const int64_t sv1,
                     const int64_t sv2,
                     const int64_t sv3,
                     const int64_t sb1,
                     const int64_t sb2,
                     const int64_t sb3,
                     const int64_t rq1,
                     const int64_t rq3,
                     const float   scale,
                     const int64_t n_tokens_dst,
                     const int64_t t_offset) {
    const int64_t h_idx    = blockIdx.x;
    const int64_t sequence = blockIdx.y;
    const int     col      = threadIdx.x;  // each thread owns one column

    const int64_t iq1 = h_idx / rq1;
    const int64_t iq3 = sequence / rq3;

    const int64_t attn_score_elems = S_v * H * n_tokens_dst * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset;
    attn_data += (sequence * n_tokens_dst * H + t_offset * H + h_idx) * S_v;

    // GCN and CDNA devices spill registers, we use shared mem for them. See https://github.com/ggml-org/llama.cpp/pull/20282#issuecomment-4025770229
    // TODO: check optimal path for RDNA1 and RDNA2 devices.
#if (defined(GGML_USE_HIP) && !defined(RDNA3) && !defined(RDNA4)) || defined(GGML_USE_MUSA)
    extern __shared__ float s_shared[];
    float * s = s_shared + col * S_v;
#else
    float s[S_v];
#endif
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        s[i] = curr_state[i * S_v + col];
    }

    for (int t = 0; t < n_tokens; t++) {
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                kv_col += s[i] * k_t[i];
            }

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                s[i] = g_val * s[i] + k_t[i] * delta_col;
                attn_col += s[i] * q_t[i];
            }

            attn_data[col] = attn_col * scale;
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                kv_col += expf(g_t[i]) * s[i] * k_t[i];
            }

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_col = 0.0f;
#pragma unroll
            for (int i = 0; i < S_v; i++) {
                s[i] = expf(g_t[i]) * s[i] + k_t[i] * delta_col;
                attn_col += s[i] * q_t[i];
            }

            attn_data[col] = attn_col * scale;
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory
#pragma unroll
    for (int i = 0; i < S_v; i++) {
        state[i * S_v + col] = s[i];
    }
}

static size_t calculate_smem(const int sv, int cc)
{
    size_t smem = 0;
    if ((GGML_CUDA_CC_IS_AMD(cc) && !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_RDNA4(cc)) || GGML_CUDA_CC_IS_MTHREADS(cc)) {
        smem = sv * sv * sizeof(float);
    }
    return smem;
}

template <bool KDA>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        int64_t S_v, int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t rq1, int64_t rq3,
        float scale, cudaStream_t stream,
        int64_t n_tokens_dst = 0, int64_t t_offset = 0) {

    if (n_tokens_dst == 0) {
        n_tokens_dst = n_tokens;
    }

    dim3 grid_dims(H, n_seqs, 1);
    dim3 block_dims(S_v, 1, 1);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 32: {
            constexpr int sv = 32;
            size_t smem = calculate_smem(sv, cc);
            gated_delta_net_cuda<sv, KDA><<<grid_dims, block_dims, smem, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, rq1, rq3, scale, n_tokens_dst, t_offset);
            break;
        }
        case 64: {
            constexpr int sv = 64;
            size_t smem = calculate_smem(sv, cc);
            gated_delta_net_cuda<sv, KDA><<<grid_dims, block_dims, smem, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, rq1, rq3, scale, n_tokens_dst, t_offset);
            break;
        }
        case 128: {
            constexpr int sv = 128;
            size_t smem = calculate_smem(sv, cc);
            gated_delta_net_cuda<sv, KDA><<<grid_dims, block_dims, smem, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, rq1, rq3, scale, n_tokens_dst, t_offset);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

//TODO: optimize
template<size_t CS>
static __global__ void cumsum_kernel_cuda(float * g_cs, const float * g,
        const int64_t nb1, const int64_t nb2, const int64_t nb3,
        const int64_t ng1, const int64_t ng2, const int64_t ng3) {

    const int head_idx  = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    const int seq_idx   = blockIdx.z;

    g    += head_idx * nb1 + (chunk_idx * CS) * nb2 + seq_idx * nb3;
    g_cs += head_idx * ng1 + (chunk_idx * CS) * ng2 + seq_idx * ng3;

    //TODO: optimize
    g_cs[0] = g[0];
    for (size_t i = 1 ; i < CS; ++i) {
        g_cs[i] = g_cs[i-1] + g[i * nb2];
    }
}

template<size_t CS, size_t S_v>
static __global__ void compute_kkt_cuda(float * Akk, float * Akq, const float * k, const float * q, const float * g_cs, const float * beta,
    int64_t sk1, int64_t sk2, int64_t sk3,
    int64_t sg1, int64_t sg2, int64_t sg3,
    int64_t sb1, int64_t sb2, int64_t sb3,
    float scale) {

    const int h_idx = blockIdx.x;
    const int c_idx = blockIdx.y;
    const int s_idx = blockIdx.z;

    k    += h_idx * sk1 + (CS * c_idx) * sk2 + s_idx * sk3;
    q    += h_idx * sk1 + (CS * c_idx) * sk2 + s_idx * sk3;
    g_cs += h_idx * sg1 + (CS * c_idx) * sg2 + s_idx * sg3;
    beta += h_idx * sb1 + (CS * c_idx) * sb2 + s_idx * sb3;

    Akk += ((int64_t)s_idx * gridDim.x * gridDim.y + h_idx * gridDim.y + c_idx) * CS * CS;
    Akq += ((int64_t)s_idx * gridDim.x * gridDim.y + h_idx * gridDim.y + c_idx) * CS * CS;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    {
        using namespace nvcuda::wmma;
        constexpr int NW  = 256 / 32;
        constexpr int TPW = dot_max_tpw<CS, CS, NW>();
        constexpr int KP  = S_v + 1;

        extern __shared__ float s_dyn[];
        float * smem_k   = s_dyn;
        float * smem_q   = s_dyn + KP * CS;
        float * smem_g   = s_dyn + 2 * KP * CS;
        float * smem_b   = s_dyn + 2 * KP * CS + CS;
        float * smem_tmp = s_dyn + 2 * KP * CS + 2 * CS;

        const int warp_id = tid / 32;
        const int lane_id = tid % 32;

        for (int i = tid; i < (int)S_v * (int)CS; i += blockDim.x * blockDim.y) {
            const int d = i % (int)S_v;
            const int t = i / (int)S_v;
            smem_k[d + t * KP] = k[t * sk2 + d];
            smem_q[d + t * KP] = q[t * sk2 + d];
        }
        if (tid < (int)CS) {
            smem_g[tid] = g_cs[tid];
            smem_b[tid] = beta[tid * sb2];
        }
        __syncthreads();

        fragment<accumulator, WM, WN, WK, float> acc[TPW];
        int tiles[TPW];
        int n_tiles;
        dot_init<CS, CS, NW>(warp_id, acc, tiles, n_tiles);
        dot_mma<CS, CS, S_v, row_major, col_major>(smem_k, KP, smem_k, KP, acc, tiles, n_tiles);

        constexpr int tn_dim = CS / WN;
#pragma unroll
        for (int tt = 0; tt < n_tiles; tt++) {
            const int tm = tiles[tt] / tn_dim;
            const int tn = tiles[tt] % tn_dim;
            const int i_off = tm * WM;
            const int j_off = tn * WN;

            store_matrix_sync(smem_tmp + warp_id * WM * WN, acc[tt], WN, mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < WM * WN; idx += 32) {
                const int di = idx / WN;
                const int dj = idx % WN;
                const int i = i_off + di;
                const int j = j_off + dj;
                Akk[i + j * CS] = (i <= j)
                    ? smem_tmp[warp_id * WM * WN + idx] * smem_b[j] * expf(smem_g[j] - smem_g[i])
                    : 0.0f;
            }
        }

        dot_init<CS, CS, NW>(warp_id, acc, tiles, n_tiles);
        dot_mma<CS, CS, S_v, row_major, col_major>(smem_k, KP, smem_q, KP, acc, tiles, n_tiles);

#pragma unroll
        for (int tt = 0; tt < n_tiles; tt++) {
            const int tm = tiles[tt] / tn_dim;
            const int tn = tiles[tt] % tn_dim;
            const int i_off = tm * WM;
            const int j_off = tn * WN;

            store_matrix_sync(smem_tmp + warp_id * WM * WN, acc[tt], WN, mem_row_major);
            __syncwarp();

#pragma unroll
            for (int idx = lane_id; idx < WM * WN; idx += 32) {
                const int di = idx / WN;
                const int dj = idx % WN;
                const int i = i_off + di;
                const int j = j_off + dj;
                Akq[i + j * CS] = (i <= j)
                    ? smem_tmp[warp_id * WM * WN + idx] * expf(smem_g[j] - smem_g[i]) * scale
                    : 0.0f;
            }
        }
    }
}

template<size_t CS>
static __global__ void prepare_solve_tri_cuda(const float * A, float * lhs, float * rhs) {
    const int64_t offset = (int64_t)blockIdx.x * CS * CS;
    const float * A_b   = A   + offset;
    float *       lhs_b = lhs + offset;
    float *       rhs_b = rhs + offset;

    for (int idx = threadIdx.x; idx < CS * CS; idx += blockDim.x) {
        const int i = idx % CS;  // row
        const int j = idx / CS;  // col
        const float val = A_b[idx];

        if (i < j) {
            lhs_b[idx] = val;
            rhs_b[idx] = -val;
        } else if (i == j) {
            lhs_b[idx] = 1.0f;
            rhs_b[idx] = 0.0f;
        } else {
            lhs_b[idx] = 0.0f;
            rhs_b[idx] = 0.0f;
        }
    }
}

template<size_t CS>
static __global__ void add_identity_cuda(float * X) {
    const int64_t offset = (int64_t)blockIdx.x * CS * CS;
    float * X_b = X + offset;
    const int tid = threadIdx.x;
    if (tid < CS) {
        X_b[tid + tid * CS] += 1.0f;
    }
}

template<size_t S_v, size_t CS>
static __global__ void compute_wu_cuda(
    const float * k, const float * v, const float * A,
    const float * beta, const float * gate_cs,
    float * k_cd, float * v_new,
    const int64_t sk1, const int64_t sk2, const int64_t sk3,
    const int64_t sv1, const int64_t sv2, const int64_t sv3,
    const int64_t sa1, const int64_t sa2, const int64_t sa3,
    const int64_t sb1, const int64_t sb2, const int64_t sb3,
    const int64_t sg1, const int64_t sg2, const int64_t sg3,
    const int64_t sd1, const int64_t sd2, const int64_t sd3) {

    const int h_idx = blockIdx.x;
    const int c_idx = blockIdx.y;
    const int s_idx = blockIdx.z;

    k       += h_idx * sk1 + (CS * c_idx) * sk2 + s_idx * sk3;
    v       += h_idx * sv1 + (CS * c_idx) * sv2 + s_idx * sv3;
    beta    += h_idx * sb1 + (CS * c_idx) * sb2 + s_idx * sb3;
    gate_cs += h_idx * sg1 + (CS * c_idx) * sg2 + s_idx * sg3;

    A       += s_idx * sa3 + h_idx * sa2 + c_idx * sa1;

    k_cd    += h_idx * sd1 + c_idx * sd2 + s_idx * sd3;
    v_new   += h_idx * sd1 + c_idx * sd2 + s_idx * sd3;

    __shared__ float A_s[CS * CS];
    __shared__ float sc_kcd[CS];
    __shared__ float sc_vnew[CS];

    const int tid = threadIdx.x;

    static_assert(CS * CS % 4 == 0, "CS*CS must be divisible by 4");
    for (int i = tid; i < (int)(CS * CS) / 4; i += blockDim.x) {
        reinterpret_cast<float4 *>(A_s)[i] = reinterpret_cast<const float4 *>(A)[i];
    }

    if (tid < CS) {
        float b = beta[tid * sb2];
        sc_vnew[tid] = b;
        sc_kcd[tid]  = b * expf(gate_cs[tid]);
    }

    __syncthreads();

    {
        using namespace nvcuda::wmma;
        constexpr int NW  = 256 / 32;
        constexpr int TPW = dot_max_tpw<S_v, CS, NW>();
        constexpr int KP  = CS + 1;

        extern __shared__ float s_dyn[];
        float * s_buf = s_dyn;

        const int warp_id = tid / 32;

        for (int i = tid; i < (int)CS * (int)S_v; i += blockDim.x) {
            const int kk = i % (int)CS;
            const int d  = i / (int)CS;
            s_buf[kk + d * KP] = k[kk * sk2 + d] * sc_kcd[kk];
        }
        __syncthreads();

        fragment<accumulator, WM, WN, WK, float> acc[TPW];
        int tiles[TPW];
        int n_tiles;
        dot_init<S_v, CS, NW>(warp_id, acc, tiles, n_tiles);
        dot_mma<S_v, CS, CS, row_major, col_major>(s_buf, KP, A_s, CS, acc, tiles, n_tiles);

        constexpr int tn_dim = CS / WN;
#pragma unroll
        for (int tt = 0; tt < n_tiles; tt++) {
            const int tm = tiles[tt] / tn_dim;
            const int tn = tiles[tt] % tn_dim;
            store_matrix_sync(k_cd + tm * WM + tn * WN * S_v, acc[tt], S_v, mem_col_major);
        }
        __syncthreads();

        for (int i = tid; i < (int)CS * (int)S_v; i += blockDim.x) {
            const int kk = i % (int)CS;
            const int d  = i / (int)CS;
            s_buf[kk + d * KP] = v[kk * sv2 + d] * sc_vnew[kk];
        }
        __syncthreads();

        dot_init<S_v, CS, NW>(warp_id, acc, tiles, n_tiles);
        dot_mma<S_v, CS, CS, row_major, col_major>(s_buf, KP, A_s, CS, acc, tiles, n_tiles);

#pragma unroll
        for (int tt = 0; tt < n_tiles; tt++) {
            const int tm = tiles[tt] / tn_dim;
            const int tn = tiles[tt] % tn_dim;
            store_matrix_sync(v_new + tm * WM + tn * WN * S_v, acc[tt], S_v, mem_col_major);
        }
    }
}

template<int K, int BV, int CS>
static __global__ void compute_state_fused_cuda(
    const float * __restrict__ k_raw,
    float * __restrict__ v_new,
    const float * __restrict__ k_cd,
    const float * __restrict__ g_cs,
    const float * __restrict__ state_in,
    float * __restrict__ states_out,
    float * __restrict__ state_final,
    int64_t sk1, int64_t sk2, int64_t sk3,
    int64_t ng1, int64_t ng3,
    int64_t sd1, int64_t sd2, int64_t sd3,
    int64_t n_chunks, int64_t H) {

    constexpr int SP = BV + 1;
    constexpr int BP = K  + 1;
    constexpr int VP = BV + 1;

    extern __shared__ float smem[];
    float * smem_state = smem;
    float * smem_buf   = smem + K * SP;
    float * smem_vbuf  = smem + K * SP + BP * CS;
    float * smem_tmp   = smem + K * SP + BP * CS + VP * CS;

    const int v_block = blockIdx.x;
    const int h_idx   = blockIdx.y;
    const int s_idx   = blockIdx.z;
    const int tid     = threadIdx.x;
    const int64_t hs  = (int64_t)s_idx * H + h_idx;
    const int v_start = v_block * BV;

    const float * s_in = state_in + hs * K * K;
#pragma unroll
    for (int i = 0; i < K * BV; i += blockDim.x) {
        const int i0 = i + tid;
        const int m = i0 % BV;
        const int k = i0 / BV;
        smem_state[k * SP + m] = s_in[(v_start + m) + k * K];
    }
    __syncthreads();

    for (int c = 0; c < (int)n_chunks; c++) {
        float * s_out = states_out + (hs * n_chunks + c) * K * K;
#pragma unroll
        for (int i = 0; i < K * BV; i += blockDim.x) {
            const int i0 = i + tid;
            const int m = i0 % BV;
            const int k = i0 / BV;
            s_out[(v_start + m) + k * K] = smem_state[k * SP + m];
        }

        const float * kcd_ptr = k_cd + h_idx * sd1 + c * sd2 + s_idx * sd3;
        static_assert(K % 4 == 0, "K must be divisible by 4");
        for (int i = tid; i < (K / 4) * (int)CS; i += blockDim.x) {
            const int k4 = i % (K / 4);
            const int t  = i / (K / 4);
            float4 val = reinterpret_cast<const float4 *>(kcd_ptr + t * K)[k4];
            const int k = k4 * 4;
            smem_buf[(k + 0) + t * BP] = val.x;
            smem_buf[(k + 1) + t * BP] = val.y;
            smem_buf[(k + 2) + t * BP] = val.z;
            smem_buf[(k + 3) + t * BP] = val.w;
        }
        __syncthreads();

        {
            using namespace nvcuda::wmma;
            constexpr int NW  = 256 / 32;
            constexpr int TPW = dot_max_tpw<BV, CS, NW>();

            fragment<accumulator, WM, WN, WK, float> acc[TPW];
            int tiles[TPW];
            int n_tiles;
            dot_init<BV, CS, NW>(tid / 32, acc, tiles, n_tiles);
            dot_mma<BV, CS, K, col_major, col_major>(smem_state, SP, smem_buf, BP, acc, tiles, n_tiles);

            constexpr int tn_dim = CS / WN;
#pragma unroll
            for (int tt = 0; tt < n_tiles; tt++) {
                const int tm = tiles[tt] / tn_dim;
                const int tn = tiles[tt] % tn_dim;
                store_matrix_sync(smem_vbuf + tm * WM + tn * WN * VP, acc[tt], VP, mem_col_major);
            }
        }
        __syncthreads();

        float * vnew_ptr = v_new + h_idx * sd1 + c * sd2 + s_idx * sd3;
#pragma unroll
        for (int i = 0; i < BV * CS; i += blockDim.x) {
            const int i0 = i + tid;
            const int m = i0 % BV;
            const int t = i0 / BV;
            const float vd = vnew_ptr[(v_start + m) + t * K] - smem_vbuf[m + t * VP];
            smem_vbuf[m + t * VP] = vd;
            vnew_ptr[(v_start + m) + t * K] = vd;
        }

        const float * k_ch = k_raw + h_idx * sk1 + s_idx * sk3 + c * (int64_t)CS * sk2;
        const float * g_ch = g_cs + h_idx * ng1 + s_idx * ng3 + c * CS;
        const float g_last = g_ch[CS - 1];
#pragma unroll
        for (int i = 0; i < K * CS; i += blockDim.x) {
            const int i0 = i + tid;
            const int k = i0 % K;
            const int t = i0 / K;
            smem_buf[k + t * BP] = k_ch[k + t * sk2] * expf(g_last - g_ch[t]);
        }
        __syncthreads();

        const float gate = expf(g_last);
        {
            using namespace nvcuda::wmma;
            constexpr int NW  = 256 / 32;
            constexpr int TPW = dot_max_tpw<K, BV, NW>();

            const int warp_id = tid / 32;
            const int lane_id = tid % 32;

            fragment<accumulator, WM, WN, WK, float> acc[TPW];
            int tiles[TPW];
            int n_tiles;
            dot_init<K, BV, NW>(warp_id, acc, tiles, n_tiles);
            dot_mma<K, BV, CS, col_major, row_major>(smem_buf, BP, smem_vbuf, VP, acc, tiles, n_tiles);

            constexpr int tn_dim = BV / WN;
#pragma unroll
            for (int tt = 0; tt < n_tiles; tt++) {
                const int tm = tiles[tt] / tn_dim;
                const int tn = tiles[tt] % tn_dim;
                const int k_off = tm * WM;
                const int m_off = tn * WN;

                store_matrix_sync(smem_tmp + warp_id * WM * WN, acc[tt], WN, mem_row_major);
                __syncwarp();

#pragma unroll
                for (int i = lane_id; i < WM * WN; i += 32) {
                    const int dk = i / WN;
                    const int dm = i % WN;
                    smem_state[(k_off + dk) * SP + (m_off + dm)] =
                        gate * smem_state[(k_off + dk) * SP + (m_off + dm)]
                            + smem_tmp[warp_id * WM * WN + i];
                }
                __syncwarp();
            }
        }
        __syncthreads();
    }

    float * s_final = state_final + hs * K * K;
#pragma unroll
    for (int i = 0; i < K * BV; i += blockDim.x) {
        const int i0 = i + tid;
        const int m = i0 % BV;
        const int k = i0 / BV;
        s_final[(v_start + m) + k * K] = smem_state[k * SP + m];
    }
}

template<size_t CS, size_t S_v>
static __global__ void compute_output_cuda(
    const float * __restrict__ q,
    const float * __restrict__ v_delta,
    const float * __restrict__ Akq,
    const float * __restrict__ states,
    const float * __restrict__ g_cs,
    float * __restrict__ output,
    int64_t sq1, int64_t sq2, int64_t sq3,
    int64_t ng1, int64_t ng3,
    int64_t sd1, int64_t sd2, int64_t sd3,
    int64_t n_chunks, int64_t H, int64_t n_tokens,
    float scale) {

    const int h_idx = blockIdx.x;
    const int s_idx = blockIdx.y;
    const int c_idx = blockIdx.z;
    const int tid   = threadIdx.x;
    const int64_t hs = (int64_t)s_idx * H + h_idx;

    const float * chunk_state = states + (hs * n_chunks + c_idx) * S_v * S_v;
    const float * chunk_vd = v_delta + h_idx * sd1 + c_idx * sd2 + s_idx * sd3;
    const float * chunk_akq = Akq + ((int64_t)s_idx * H * n_chunks + h_idx * n_chunks + c_idx) * CS * CS;
    const float * chunk_g = g_cs + h_idx * ng1 + s_idx * ng3 + c_idx * CS;
    // TODO: handle GQA
    const float * q_base = q + h_idx * sq1 + s_idx * sq3 + c_idx * (int64_t)CS * sq2;
    float * out_base = output + h_idx * S_v + s_idx * n_tokens * S_v * H + c_idx * (int64_t)CS * S_v * H;

    __shared__ float s_akq[CS * CS];
    __shared__ float s_gexp[CS];

    extern __shared__ float s_dyn[];
    float * s_state = s_dyn;

    static_assert(CS * CS % 4 == 0 && S_v * S_v % 4 == 0, "sizes must be divisible by 4 for float4 loads");
    for (int i = tid; i < (int)(CS * CS) / 4; i += blockDim.x) {
        reinterpret_cast<float4 *>(s_akq)[i] = reinterpret_cast<const float4 *>(chunk_akq)[i];
    }
    if (tid < (int)CS) {
        s_gexp[tid] = expf(chunk_g[tid]);
    }
    for (int i = tid; i < (int)(S_v * S_v) / 4; i += blockDim.x) {
        reinterpret_cast<float4 *>(s_state)[i] = reinterpret_cast<const float4 *>(chunk_state)[i];
    }
    __syncthreads();

    {
        using namespace nvcuda::wmma;
        constexpr int SW  = 32;
        constexpr int SWP = SW + 1;
        constexpr int VDP = S_v + 1;

        float * s_tile = s_dyn + S_v * S_v;

        const int warp_id = tid / 32;
        const int lane_id = tid % 32;
        constexpr int NW  = 256 / 32;
        constexpr int TPW = dot_max_tpw<S_v, CS, NW>();

        fragment<accumulator, WM, WN, WK, float> acc[TPW];
        int tiles[TPW]; int n_tiles;
        dot_init<S_v, CS, NW>(warp_id, acc, tiles, n_tiles);

        for (int kk = 0; kk < (int)S_v; kk += SW) {
            for (int i = tid; i < SW * (int)CS; i += blockDim.x) {
                const int j = i % SW;
                const int t = i / SW;
                s_tile[j + t * SWP] = q_base[(kk + j) + t * sq2] * s_gexp[t] * scale;
            }
            __syncthreads();

            dot_mma<S_v, CS, SW, col_major, col_major>(
                s_state + kk * S_v, S_v,
                s_tile, SWP,
                acc, tiles, n_tiles);
            __syncthreads();
        }

        float * s_vd = s_state;
        for (int i = tid; i < (int)(S_v / 4) * (int)CS; i += blockDim.x) {
            const int d4 = i % (S_v / 4);
            const int t  = i / (S_v / 4);
            float4 val = reinterpret_cast<const float4 *>(chunk_vd + t * S_v)[d4];
            const int d = d4 * 4;
            s_vd[(d + 0) + t * VDP] = val.x;
            s_vd[(d + 1) + t * VDP] = val.y;
            s_vd[(d + 2) + t * VDP] = val.z;
            s_vd[(d + 3) + t * VDP] = val.w;
        }
        __syncthreads();

        dot_mma<S_v, CS, CS, col_major, col_major>(
            s_vd, VDP,
            s_akq, CS,
            acc, tiles, n_tiles);

        float * s_wtmp = s_tile;
        constexpr int tiles_n = CS / WN;
#pragma unroll
        for (int tt = 0; tt < n_tiles; tt++) {
            const int tm = tiles[tt] / tiles_n;
            const int tn = tiles[tt] % tiles_n;
            const int d_off = tm * WM;
            const int t_off = tn * WN;

            store_matrix_sync(s_wtmp + warp_id * WM * WN, acc[tt], WN, mem_row_major);
            __syncwarp();

#pragma unroll
            for (int i = lane_id; i < WM * WN; i += 32) {
                const int dd = i / WN;
                const int dt = i % WN;
                out_base[(d_off + dd) + (t_off + dt) * S_v * H] = s_wtmp[warp_id * WM * WN + i];
            }
            __syncwarp();
        }
    }
}

// 2026-05-26: non-static so the dispatch in delta-net.cu can call us. All
// other helpers in this file stay file-local (static).
void delta_net_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t, nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t, nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t, nbg, src_g, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const int64_t rq1 = nev1 / neq1;
    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    // 2026-05-26 PATH C: g comes in with metadata override matching beta exactly
    // (set in delta-net.cu's dispatch). Underlying data is in shared "heads
    // adjacent in memory" layout — beta's nb[] correctly describes both. We
    // no longer require ggml_is_contiguous: the data isn't standard-contiguous
    // by ggml's definition (heads/tokens are not in ne[] order), but the kernel
    // uses nbb1/2/3 derived from beta's nb to index it correctly. Same applies
    // to beta. The same_stride check is also redundant now (we forced it via
    // metadata override) but kept as a sanity belt-and-suspenders.
    auto _same_meaningful = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->type != b->type) return false;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (a->ne[i] != b->ne[i]) return false;
            if (a->ne[i] > 1 && a->nb[i] != b->nb[i]) return false;
        }
        return true;
    };
    GGML_ASSERT(_same_meaningful(src_g, src_beta));
    // ggml_is_contiguous asserts removed: Path C override gives g and beta
    // non-standard-contiguous strides, but kernel indexing is correct.
    GGML_ASSERT(ggml_is_contiguous(src_state));

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sg1 = nbg1 / sizeof(float);
    const int64_t sg2 = nbg2 / sizeof(float);
    const int64_t sg3 = nbg3 / sizeof(float);

    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    static constexpr int CS = 64;

    const int64_t n_full   = (n_tokens / CS) * CS;
    const int64_t n_tail   = n_tokens - n_full;
    const int     n_chunks = n_full / CS;

    if (n_chunks == 0) {
        launch_gated_delta_net<false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, rq1, rq3, scale, ctx.stream());
        return;
    }

    // TODO: KDA
    // g_cs output layout: [CS, n_chunks, H, n_seqs] contiguous
    const int64_t ng1 = CS * n_chunks;      // head stride
    const int64_t ng2 = 1;                  // token stride (consecutive within chunk)
    const int64_t ng3 = CS * n_chunks * H;  // seq stride

    ggml_cuda_pool_alloc<float> gate_csum(ctx.pool(), CS * n_chunks * H * n_seqs);
    ggml_cuda_pool_alloc<float> Akk(ctx.pool(), CS * CS * n_chunks * H * n_seqs);
    ggml_cuda_pool_alloc<float> Akq(ctx.pool(), CS * CS * n_chunks * H * n_seqs);

    {
        dim3 grid(H, n_chunks, n_seqs);
        cumsum_kernel_cuda<CS><<<grid, 1, 0, ctx.stream()>>>(
            gate_csum.get(), g_d, sg1, sg2, sg3, ng1, ng2, ng3);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 grid(H, n_chunks, n_seqs);
        constexpr int KP = 128 + 1;
        constexpr int kkt_smem_bytes = (2 * KP * CS + 2 * CS + 8 * 16 * 16) * sizeof(float);
        dim3 block(256, 1, 1);
        CUDA_SET_SHARED_MEMORY_LIMIT((compute_kkt_cuda<CS, 128>), kkt_smem_bytes);
        switch (S_v) {
            case 128:
                compute_kkt_cuda<CS, 128><<<grid, block, kkt_smem_bytes, ctx.stream()>>>(Akk.get(), Akq.get(), k_d, q_d, gate_csum.get(), b_d, sq1, sq2, sq3, ng1, ng2, ng3, sg1, sg2, sg3, scale);
                break;
            default:
                GGML_ABORT("Fatal error");
                break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // Triangular solve: compute (I + L)^{-1} where L = strictly_lower(kb). TODO: Fuse into a kernel which computes (I + A)^-1 directly
    {
        const int64_t n_batches = n_chunks * H * n_seqs;
        ggml_cuda_pool_alloc<float> lhs(ctx.pool(), CS * CS * n_batches);
        ggml_cuda_pool_alloc<float> rhs(ctx.pool(), CS * CS * n_batches);

        // Split Akk into lhs (I + L) and rhs (-L)
        prepare_solve_tri_cuda<CS><<<n_batches, 256, 0, ctx.stream()>>>(
            Akk.get(), lhs.get(), rhs.get());
        CUDA_CHECK(cudaGetLastError());

        // Solve
        const size_t bs = CS * CS;
        ggml_cuda_solve_tri(ctx, lhs.get(), rhs.get(), Akk.get(), CS, CS,
            n_batches, 1,
            bs, bs * n_batches,
            bs, bs * n_batches,
            bs, bs * n_batches);

        // Add identity: result = Akk + I = (I+L)^{-1}
        add_identity_cuda<CS><<<n_batches, CS, 0, ctx.stream()>>>(Akk.get());
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t sd1 = S_v * CS * n_chunks;      // head stride
    const int64_t sd2 = S_v * CS;                 // chunk stride
    const int64_t sd3 = S_v * CS * n_chunks * H;  // seq stride

    ggml_cuda_pool_alloc<float> k_cd(ctx.pool(), S_v * CS * n_chunks * H * n_seqs);
    ggml_cuda_pool_alloc<float> v_new(ctx.pool(), S_v * CS * n_chunks * H * n_seqs);
    {
        int64_t sa1 = CS * CS;
        int64_t sa2 = CS * CS * n_chunks;
        int64_t sa3 = CS * CS * n_chunks * H;

        dim3 grid(H, n_chunks, n_seqs);
        dim3 block(256, 1, 1);
        constexpr int wu_smem_bytes = (CS + 1) * 128 * (int)sizeof(float);
        CUDA_SET_SHARED_MEMORY_LIMIT((compute_wu_cuda<128, CS>), wu_smem_bytes);
        compute_wu_cuda<128, CS><<<grid, block, wu_smem_bytes, ctx.stream()>>>(k_d, v_d, Akk.get(), b_d, gate_csum.get(), k_cd.get(), v_new.get(),
            sq1, sq2, sq3,
            sv1, sv2, sv3,
            sa1, sa2, sa3,
            sb1, sb2, sb3,
            ng1, ng2, ng3,
            sd1, sd2, sd3);
    }

    constexpr int BV = 32;
    ggml_cuda_pool_alloc<float> states(ctx.pool(), S_v * S_v * n_chunks * H * n_seqs);
    ggml_cuda_pool_alloc<float> state_out(ctx.pool(), S_v * S_v * H * n_seqs);

    {
        constexpr int smem_bytes = (128 * (BV + 1) + (128 + 1) * CS + (BV + 1) * CS + 8 * 16 * 16) * sizeof(float);
        CUDA_SET_SHARED_MEMORY_LIMIT((compute_state_fused_cuda<128, BV, CS>), smem_bytes);

        dim3 grid(128 / BV, H, n_seqs);  // (2, H, n_seqs)
        compute_state_fused_cuda<128, BV, CS><<<grid, 256, smem_bytes, ctx.stream()>>>(
            k_d, v_new.get(), k_cd.get(), gate_csum.get(),
            s_d,
            states.get(), state_out.get(),
            sq1, sq2, sq3,
            ng1, ng3,
            sd1, sd2, sd3,
            n_chunks, H);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        constexpr int output_smem_bytes = (128 * 128 + 33 * 64) * (int)sizeof(float);
        CUDA_SET_SHARED_MEMORY_LIMIT((compute_output_cuda<CS, 128>), output_smem_bytes);

        dim3 grid(H, n_seqs, n_chunks);
        compute_output_cuda<CS, 128><<<grid, 256, output_smem_bytes, ctx.stream()>>>(
            q_d, v_new.get(), Akq.get(), states.get(), gate_csum.get(),
            dst_d,
            sq1, sq2, sq3,
            ng1, ng3,
            sd1, sd2, sd3,
            n_chunks, H, n_tokens,
            scale);
        CUDA_CHECK(cudaGetLastError());
    }

    if (n_tail == 0) {
        const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
        float * state_dst = dst_d + attn_score_elems;
        CUDA_CHECK(cudaMemcpyAsync(state_dst, state_out.get(),
            S_v * S_v * H * n_seqs * sizeof(float),
            cudaMemcpyDeviceToDevice, ctx.stream()));
    } else {
        const float * q_tail = q_d + n_full * sq2;
        const float * k_tail = k_d + n_full * sq2;
        const float * v_tail = v_d + n_full * sv2;
        const float * g_tail = g_d + n_full * sg2;
        const float * b_tail = b_d + n_full * sb2;

        launch_gated_delta_net<false>(q_tail, k_tail, v_tail, g_tail, b_tail,
            state_out.get(), dst_d,
            S_v, H, n_tail, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, rq1, rq3, scale, ctx.stream(),
            n_tokens, n_full);
    }
}

