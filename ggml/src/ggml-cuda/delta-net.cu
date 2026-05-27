#include "common.cuh"
#include "delta-net.cuh"
#include <vector>
#include <cmath>
// Note: cstdlib/cstring are transitively included via common.cuh -> cuda_runtime.h.
// Do NOT add explicit includes here — conda nvcc's host compiler picks up a
// mismatched glibc sysroot that defines _Float32/64/128 types incompatible
// with this nvcc version, causing 100+ errors on stdlib.h / wchar.h.

// Delta Net Linear Attention Kernel for Qwen3-Next (HEAD_DIM=128)
// State layout: [S_v, S_v*H_v, 1, n_seqs] (column-major)

__device__ __forceinline__ float sigmoid_f(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <int block_size>
__device__ __forceinline__ float reduce_sum(float x, float * s) {
    x = warp_reduce_sum(x);
    if constexpr (block_size > WARP_SIZE) {
        //__shared__ float s[block_size/WARP_SIZE];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s[warp_id] = x;
        }
        __syncthreads();
        x = lane_id < block_size/WARP_SIZE ? s[lane_id] : 0.0f;
        x = warp_reduce_sum(x);
    }
    return x;
}

template <int HEAD_DIM, int block_size>
__global__ void delta_net_recurrent_f32(
    const float * __restrict__ q,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ k,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ v,         // [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const float * __restrict__ g,         // [n_tokens, 1, n_heads, n_seqs]
    const float * __restrict__ beta_in,   // [1, n_tokens, n_heads, n_seqs]
    const float * __restrict__ state_in,  // [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    float * __restrict__ dst,             // output + new_state(s) concatenated
    float * __restrict__ saved_states,
    const int64_t n_heads,
    const int64_t gqa_ratio,
    const int repeat_type,
    const int64_t n_tokens,
    const int64_t n_seqs,
    const int64_t output_offset,          // offset where state starts in output
    size_t vnb1, size_t vnb2, size_t vnb3) {
    constexpr int warps_per_head = HEAD_DIM/WARP_SIZE;
    const int batch_idx = blockIdx.x / (warps_per_head*n_heads);
    const int sub_head_idx  = blockIdx.x % (warps_per_head*n_heads);
    const int head_idx = sub_head_idx / warps_per_head;
    const int sub_idx  = sub_head_idx % warps_per_head;
    const int head_idx_kq = repeat_type == 0 ? head_idx / gqa_ratio : head_idx % (n_heads/gqa_ratio);
    const int tid = threadIdx.x;

    // Strides for input tensors (column-major)
    // Q/K/V: [HEAD_DIM, n_tokens, n_heads, n_seqs]
    const int64_t qkv_stride_token = HEAD_DIM;
    const int64_t qkv_stride_head = HEAD_DIM * n_tokens;
    const int64_t qkv_stride_batch = HEAD_DIM * n_tokens * n_heads;
    const int64_t qkv_stride_batch_kq = qkv_stride_batch / gqa_ratio;

    // G/Beta: [n_tokens, 1, n_heads, n_seqs] / [1, n_tokens, n_heads, n_seqs]
    //const int64_t g_stride_head = n_tokens;
    const int64_t g_stride_batch = n_tokens * n_heads;

    // State: [HEAD_DIM, HEAD_DIM*n_heads, 1, n_seqs]
    // For head h: columns h*HEAD_DIM to (h+1)*HEAD_DIM
    // state[row, col] for head h = state[row, h*HEAD_DIM + col]
    // Linear index: row + (h*HEAD_DIM + col) * HEAD_DIM = row + h*HEAD_DIM^2 + col*HEAD_DIM
    const int64_t state_head_offset = head_idx * HEAD_DIM * HEAD_DIM;
    const int64_t state_batch_stride = HEAD_DIM * HEAD_DIM * n_heads;

    // State step stride for save_all_states: HEAD_DIM^2 * n_heads * n_seqs
    const int64_t state_step_stride = HEAD_DIM * HEAD_DIM * n_heads * n_seqs;

    // Pointers for this batch/head
    const float * q_ptr = q + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * k_ptr = k + batch_idx * qkv_stride_batch_kq + head_idx_kq * qkv_stride_head;
    const float * v_ptr = v + batch_idx * vnb3 + head_idx * vnb2;
    const float * g_ptr = g + batch_idx * g_stride_batch + head_idx;
    const float * beta_ptr = beta_in + batch_idx * g_stride_batch + head_idx;
    const float * state_src = state_in + batch_idx * state_batch_stride + state_head_offset;

    // Output layout: [head_v_dim, num_v_heads, n_seq_tokens, n_seqs]
    // For [dim, head, token, batch]: index = dim + head*S_v + token*S_v*H_v + batch*S_v*H_v*n_tokens
    float * out_base = dst + batch_idx * (HEAD_DIM * n_heads * n_tokens) + head_idx * HEAD_DIM;
    const int64_t out_token_stride = HEAD_DIM * n_heads;  // stride between tokens
    float * state_dst = dst + output_offset + batch_idx * state_batch_stride + state_head_offset;

    // Shared memory for current token's Q, K, V (normalized), and intermediate results
    extern __shared__ float smem[];
    float * sQ = smem;                          // HEAD_DIM
    float * sK = sQ + HEAD_DIM;                 // HEAD_DIM

    const float scale = rsqrtf((float)HEAD_DIM);

    __shared__ float sum_helper[block_size/WARP_SIZE];

    constexpr int num_warps = block_size/WARP_SIZE;
    const int row = tid % WARP_SIZE;
    const int col_idx_0 = tid / WARP_SIZE;
    const int row_out = row + sub_idx * WARP_SIZE;

    // Keep the state in registers, copy the final state to its destination at the end
    float state_local[HEAD_DIM/num_warps];
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_local[i] = state_src[col*HEAD_DIM + row_out];
    }

    constexpr int WARP_SIZE_S = WARP_SIZE + 1;
    constexpr int num_stored_rows = block_size/WARP_SIZE;
    __shared__ float all_sum[2*WARP_SIZE_S*num_stored_rows];
    auto all_sum1 = all_sum;
    auto all_sum2 = all_sum1 + WARP_SIZE_S*num_stored_rows;

    for (int64_t t = 0; t < n_tokens; t++) {
        float sum_kq = 0.0f;
        for (int i = tid; i < HEAD_DIM; i += block_size) {
            sQ[i] = q_ptr[t * qkv_stride_token + i] * scale;
            sK[i] = k_ptr[t * qkv_stride_token + i];
            sum_kq += sK[i] * sQ[i];
        }

        float attn_score = reduce_sum<block_size>(sum_kq, sum_helper);

        float beta_val = sigmoid_f(beta_ptr[t*n_heads]);
        float decay    = expf(fminf(g_ptr[t*n_heads], 50.0f));

        float sum1 = 0, sum2 = 0;
#pragma unroll
        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            sum1 += state_local[i] * sK[col];
            sum2 += state_local[i] * sQ[col];
        }
        all_sum1[col_idx_0*WARP_SIZE_S + row] = sum1;
        all_sum2[col_idx_0*WARP_SIZE_S + row] = sum2;

        __syncthreads();

        sum1 = sum2 = 0;
#pragma unroll
        for (int i = 0; i < block_size/WARP_SIZE; ++i) {
            sum1 += all_sum1[i*WARP_SIZE_S + row];
            sum2 += all_sum2[i*WARP_SIZE_S + row];
        }

        //float sv_new = beta_val * (v_ptr[t * qkv_stride_token + row_out] - sum1 * decay);
        float sv_new = beta_val * (v_ptr[t * vnb1 + row_out] - sum1 * decay);
        if (col_idx_0 == 0) {
            out_base[t * out_token_stride + row_out] = sum2 * decay + sv_new * attn_score;
        }

        for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
            int col = num_warps*i + col_idx_0;
            float new_state_val = decay * state_local[i] + sv_new * sK[col];
            new_state_val = fminf(fmaxf(new_state_val, -1e6f), 1e6f);
            state_local[i] = new_state_val;
        }

        // Save per-step state if requested
        if (saved_states && t < n_tokens - 1) {
            float * state_step_dst = saved_states + batch_idx * state_batch_stride + state_head_offset + t * state_step_stride;
            for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
                int col = num_warps*i + col_idx_0;
                state_step_dst[col*HEAD_DIM + row_out] = state_local[i];
            }
        }

        // Barrier required: (a) sK reads in the state update above must complete
        // before next iteration overwrites sK at the top of the loop, and (b) this
        // single barrier also orders all_sum1/all_sum2 reads above vs. the next
        // iteration's writes — subsuming the prior barriers after the cross-warp
        // reduction and after the loop exit.
        __syncthreads();
    }
    // Copy the final state to its destination
    for (int i = 0; i < HEAD_DIM/num_warps; ++i) {
        int col = num_warps*i + col_idx_0;
        state_dst[col*HEAD_DIM + row_out] = state_local[i];
    }
}

static void delta_net_f32_cuda(
    const float * q,
    const float * k,
    const float * v,
    const float * g,
    const float * beta,
    const float * state_in,
    float * dst,
    float * saved_states,
    const int64_t head_dim,
    const int64_t n_tokens,
    const int64_t n_heads,
    const int64_t gqa_ratio,
    const int     repeat_type,
    const int64_t n_seqs,
    size_t vnb1, size_t vnb2, size_t vnb3,
    const int device_id,
    const int cc, // compute capability (e.g., 890 for SM 8.9, 1200 for SM 12.0)
    cudaStream_t stream) {
    GGML_UNUSED(device_id);
    GGML_UNUSED(cc);

    const int64_t output_offset = head_dim * n_tokens * n_heads * n_seqs;

    if (head_dim != 64 && head_dim != 128) {
        GGML_ABORT("Unsupported delta net head size");
    }

    GGML_ASSERT(head_dim % WARP_SIZE == 0);
    const int num_blocks = n_seqs * n_heads * (head_dim/WARP_SIZE);
    const size_t smem_size = 2 * head_dim * sizeof(float);

    if (n_tokens <= 8) {
        constexpr int threads_per_block = 256;
        if (head_dim == 64) {
            delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        } else {
            delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        }
    } else {
        constexpr int threads_per_block = 128;
        if (head_dim == 64) {
            delta_net_recurrent_f32<64, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        } else {
            delta_net_recurrent_f32<128, threads_per_block><<<num_blocks, threads_per_block, smem_size, stream>>>(
                    q, k, v, g, beta, state_in, dst, saved_states, n_heads, gqa_ratio, repeat_type, n_tokens, n_seqs, output_offset, vnb1, vnb2, vnb3);
        }
    }

    CUDA_CHECK(cudaGetLastError());

}

// PATH D1 helper: de-interleave v from ik_llama's interleaved SSM buffer.
// ik_llama's qwen3-next graph builder stores v interleaved with the z gate:
//   source layout: [S_v, n_tokens, H_v, n_seqs]  (dim1=n_tokens, dim2=H_v)
//   nb = [4, S_v*4, S_v*n_tokens*4, 2*S_v*n_tokens*H_v*4]  <- nb[3] is 2x tight
// Output layout: [S_v, H_v, n_tokens, n_seqs] contiguous (what chunked expects)
//   nb_tight = [4, S_v*4, S_v*H_v*4, S_v*H_v*n_tokens*4]
// One thread per output element. Total elements = S_v * H_v * n_tokens * n_seqs.
__global__ void extract_v_interleaved(
    const float * __restrict__ src,   // original v buffer (interleaved)
    float       * __restrict__ dst,   // tight output buffer
    int64_t S_v, int64_t H_v, int64_t n_tokens, int64_t n_seqs,
    int64_t src_nb1,   // token stride in floats  (= S_v)
    int64_t src_nb2,   // head stride in floats   (= S_v * n_tokens)
    int64_t src_nb3    // seq stride in floats    (= 2 * S_v * n_tokens * H_v, interleaved)
) {
    const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = S_v * H_v * n_tokens * n_seqs;
    if (idx >= total) return;

    // Decode flat index into (d, h, t, s) for the tight [S_v, H_v, n_tokens, n_seqs] layout
    const int64_t d = idx % S_v;
    const int64_t h = (idx / S_v) % H_v;
    const int64_t t = (idx / (S_v * H_v)) % n_tokens;
    const int64_t s = idx / (S_v * H_v * n_tokens);

    // Read from source using ik_llama's original strides:
    //   src layout: [S_v, n_tokens, H_v, n_seqs]
    //   index: s * src_nb3 + h * src_nb2 + t * src_nb1 + d
    const int64_t src_idx = s * src_nb3 + h * src_nb2 + t * src_nb1 + d;
    dst[idx] = src[src_idx];
}

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];  // q
    const ggml_tensor * src1 = dst->src[1];  // k
    const ggml_tensor * src2 = dst->src[2];  // v
    const ggml_tensor * src3 = dst->src[3];  // g
    const ggml_tensor * src4 = dst->src[4];  // beta
    const ggml_tensor * src5 = dst->src[5];  // state
    const ggml_tensor * src6 = dst->src[6];  // when not null, state for token 0...n_token-1

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t head_dim = src0->ne[0];
    const int64_t n_tokens = src0->ne[1];
    const int64_t n_heads = src2->ne[2];
    const int64_t n_heads_kq = src0->ne[2];
    const int64_t n_seqs = src0->ne[3];
    GGML_ASSERT(n_heads % n_heads_kq == 0);
    const int64_t gqa_ratio = n_heads / n_heads_kq;

    // Dimension validation
    // Q/K: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src1->ne[0] == head_dim && src1->ne[1] == n_tokens && src1->ne[2] == n_heads_kq && src1->ne[3] == n_seqs);
    // V: [head_dim, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src2->ne[0] == head_dim && src2->ne[1] == n_tokens && src2->ne[2] == n_heads && src2->ne[3] == n_seqs);
    // G: [n_tokens, 1, n_heads, n_seqs]
    // 2026-05-26: g accepted in either [n_tokens, 1, n_heads, n_seqs] (sequential
    // legacy) or [1, n_tokens, n_heads, n_seqs] (matches beta, unblocks chunked).
    // Memory layout is identical; only the ggml shape interpretation differs.
    // Both paths through this function use g_stride_batch = n_tokens * n_heads
    // and index via [t * n_heads] — that math is unaffected by which dim is 1.
    GGML_ASSERT(((src3->ne[0] == n_tokens && src3->ne[1] == 1) ||
                 (src3->ne[0] == 1 && src3->ne[1] == n_tokens)) &&
                src3->ne[2] == n_heads && src3->ne[3] == n_seqs);
    // Beta: [1, n_tokens, n_heads, n_seqs]
    GGML_ASSERT(src4->ne[0] == 1 && src4->ne[1] == n_tokens && src4->ne[2] == n_heads && src4->ne[3] == n_seqs);
    // State: [head_dim, head_dim*n_heads, 1, n_seqs]
    GGML_ASSERT(src5->ne[0] == head_dim && src5->ne[1] == head_dim * n_heads && src5->ne[2] == 1 && src5->ne[3] == n_seqs);

    // Verify output tensor size
    const int64_t output_size = head_dim * n_tokens * n_heads * n_seqs;
    const int64_t state_size = head_dim * head_dim * n_heads * n_seqs;

    int repeat_type = dst->op_params[0];
    if (src6) {
        GGML_ASSERT(src6->type == GGML_TYPE_F32);
        GGML_ASSERT(src6->ne[0] >= (n_tokens - 1)*state_size);
    }

    const int64_t expected_size = output_size + state_size;
    GGML_ASSERT(ggml_nelements(dst) == expected_size);

    GGML_ASSERT(head_dim <= 256);  // Reasonable limit for shared memory

    // 2026-05-26: chunked GDN dispatch (ported from mainline am17an branch
    // commit 8ea2990). The chunked kernel runs the GDN prefill in O(C^2)
    // per-chunk + O(C*N) reduction, much faster than the per-token sequential
    // recurrence for large n_tokens. Engages when ALL conditions hold:
    //   * not KDA (gate dim equals 1, not S_v)
    //   * n_tokens > 1 (it's a prefill, not single-token decode)
    //   * head_dim == 128 (only shape the chunked kernel targets)
    //   * src6 (saved_steps for speculative-decoding checkpoint) is null —
    //     the chunked kernel doesn't write per-step state.
    //   * g and beta have matching strides — ik_llama.cpp's qwen3next graph
    //     builder constructs g as [n_tokens, 1, n_heads, n_seqs] and beta as
    //     [1, n_tokens, n_heads, n_seqs] (different strides). The chunked
    //     kernel asserts same_stride(g, beta). Falling back keeps the model
    //     working but means chunked never fires until we either (a) align
    //     the graph builder, or (b) extend the kernel to handle ik_llama's
    //     layout. TODO: aligning g/beta in build_qwen3next.cpp is the next
    //     port step. For now, the dispatch compiles + the binary loads but
    //     chunked is dormant.
    // Otherwise falls through to the existing sequential per-token kernel.
    const bool kda = (src3->ne[0] == head_dim);
    // 2026-05-26: ggml_are_same_stride() checks ALL nb[i] including trivial
    // dims (size 1). After our permute alignment, g and beta have identical
    // strides EXCEPT nb[0] (which is 256 vs 128 in our shape [1,n_tokens,H_v,1])
    // — and ne[0]=1 means that stride can never actually be traversed. Use a
    // size-aware check that only requires strides match for dims with ne > 1.
    auto same_meaningful_stride = [](const ggml_tensor * a, const ggml_tensor * b) {
        if (a->type != b->type) return false;
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (a->ne[i] != b->ne[i]) return false;
            if (a->ne[i] > 1 && a->nb[i] != b->nb[i]) return false;
        }
        return true;
    };
    const bool same_g_beta_stride = same_meaningful_stride(src3, src4);
    // DIAG: log first prefill and first decode call separately
    static bool _logged_pref = false, _logged_dec = false;
    const bool will_chunk = !kda && n_tokens > 1 && head_dim == 128 && src6 == nullptr && same_g_beta_stride;
    if ((n_tokens > 1 && !_logged_pref) || (n_tokens == 1 && !_logged_dec)) {
        fprintf(stderr, "[CHUNKED-DISPATCH] kda=%d n_tokens=%ld head_dim=%ld src6=%p same_stride=%d -> %s\n",
                kda, n_tokens, head_dim, const_cast<void*>(static_cast<const void*>(src6)), same_g_beta_stride,
                will_chunk ? "CHUNKED" : "sequential");
        fprintf(stderr, "  g    ne=[%ld %ld %ld %ld] nb=[%ld %ld %ld %ld]\n",
                src3->ne[0], src3->ne[1], src3->ne[2], src3->ne[3],
                src3->nb[0], src3->nb[1], src3->nb[2], src3->nb[3]);
        fprintf(stderr, "  beta ne=[%ld %ld %ld %ld] nb=[%ld %ld %ld %ld]\n",
                src4->ne[0], src4->ne[1], src4->ne[2], src4->ne[3],
                src4->nb[0], src4->nb[1], src4->nb[2], src4->nb[3]);
        fflush(stderr);
        if (n_tokens > 1) _logged_pref = true; else _logged_dec = true;
    }
    // 2026-05-26 PATH C + D1:
    // Path C overrides q/k/v/g metadata so am17an's chunked kernel can read
    // them without a physical copy (ne[1]↔ne[2], nb[1]↔nb[2] swap).
    // Path D1 additionally de-interleaves v into a tight temp buffer before
    // calling delta_net_chunk(), because v's buffer is interleaved with the
    // SSM z gate: nb[3] = 2× tight size. The chunked kernel's sv2 stride
    // would otherwise span v+z, corrupting the computation.
    //
    // Gated behind GDN_CHUNK=1 env var so default behavior is unchanged.
    static const bool _enable_path_c = []() {
        const char * e = std::getenv("GDN_CHUNK");
        return e && *e && *e != '0';
    }();
    if (!_enable_path_c || kda || n_tokens < 64 || head_dim != 128 || src6 != nullptr) {
        // Path C+D1 disabled or doesn't meet preconditions; fall through to sequential.
    } else {
        ggml_tensor * src_q_mut    = dst->src[0];
        ggml_tensor * src_k_mut    = dst->src[1];
        ggml_tensor * src_v_mut    = dst->src[2];
        ggml_tensor * src_g_mut    = dst->src[3];
        ggml_tensor * src_beta_mut = dst->src[4];

        // Save original metadata for all 5 tensors (restored after the call).
        int64_t saved_q_ne[4], saved_k_ne[4], saved_v_ne[4], saved_g_ne[4], saved_b_ne[4];
        size_t  saved_q_nb[4], saved_k_nb[4], saved_v_nb[4], saved_g_nb[4], saved_b_nb[4];
        void *  saved_v_data = src_v_mut->data;
        for (int i = 0; i < 4; i++) {
            saved_q_ne[i] = src_q_mut->ne[i];    saved_q_nb[i] = src_q_mut->nb[i];
            saved_k_ne[i] = src_k_mut->ne[i];    saved_k_nb[i] = src_k_mut->nb[i];
            saved_v_ne[i] = src_v_mut->ne[i];    saved_v_nb[i] = src_v_mut->nb[i];
            saved_g_ne[i] = src_g_mut->ne[i];    saved_g_nb[i] = src_g_mut->nb[i];
            saved_b_ne[i] = src_beta_mut->ne[i]; saved_b_nb[i] = src_beta_mut->nb[i];
        }

        // q/k: swap ne[1]↔ne[2] and nb[1]↔nb[2] to reinterpret
        // ik_llama's [S, T, H, Sq] as am17an's expected [S, H, T, Sq].
        // Memory unchanged; the new strides correctly describe the per-head/
        // per-token access pattern needed by the chunked kernel.
        std::swap(src_q_mut->ne[1], src_q_mut->ne[2]);
        std::swap(src_q_mut->nb[1], src_q_mut->nb[2]);
        std::swap(src_k_mut->ne[1], src_k_mut->ne[2]);
        std::swap(src_k_mut->nb[1], src_k_mut->nb[2]);

        // g and beta: rewrite ne/nb to am17an's [1, H, T, S] tight-contig
        // layout. ik_llama's memory layout for both g (from build_beta_gate
        // alpha = ggml_cont_3d [H, T, S]) and beta (= ggml_cont_4d [H, 1, T, S])
        // is the SAME: heads at stride 1, tokens at stride H, seqs at stride H*T.
        // The post-permute ne/nb (set in llama-delta-net.cpp build_fused_delta_net)
        // describes a DIFFERENT shape interpretation that swaps token/head
        // stride positions — the chunked kernel reads sb1=nb[1]/4 expecting
        // head stride, but the post-permute beta has nb[1] = token stride.
        // Fix: relabel as [1, H, T, S] with nb[1]=4 (head stride, 1 float)
        // and nb[2]=4*H (token stride, H floats).
        const int64_t S_H_v = n_heads;        // 32 for qwen3-next
        const size_t  stride_h = sizeof(float);
        const size_t  stride_t = S_H_v * sizeof(float);
        const size_t  stride_s = S_H_v * n_tokens * sizeof(float);
        src_g_mut->ne[0] = 1;     src_g_mut->ne[1] = S_H_v;  src_g_mut->ne[2] = n_tokens;  src_g_mut->ne[3] = n_seqs;
        src_g_mut->nb[0] = 4;     src_g_mut->nb[1] = stride_h; src_g_mut->nb[2] = stride_t; src_g_mut->nb[3] = stride_s;
        src_beta_mut->ne[0] = 1;  src_beta_mut->ne[1] = S_H_v;  src_beta_mut->ne[2] = n_tokens;  src_beta_mut->ne[3] = n_seqs;
        src_beta_mut->nb[0] = 4;  src_beta_mut->nb[1] = stride_h; src_beta_mut->nb[2] = stride_t; src_beta_mut->nb[3] = stride_s;

        // PATH D1: extract v into a tight contiguous buffer.
        // v's physical buffer is interleaved with the SSM z gate:
        //   saved_v_nb[3] = 2 * S_v * H_v * n_tokens * sizeof(float)
        // After the ne[1]↔ne[2] swap, the chunked kernel would see sv2 =
        // nb[1]/sizeof(float) = (original nb[2])/4 = S_v * H_v * 2 (2x tight).
        // We copy v into a contiguous [S_v, H_v, n_tokens, n_seqs] buffer so
        // the chunked kernel sees sv1=S_v, sv2=S_v*H_v, sv3=S_v*H_v*n_tokens.
        const int64_t S_v    = head_dim;        // 128
        const int64_t H_v    = n_heads;         // 32 for qwen3-next
        const int64_t v_elems = S_v * H_v * n_tokens * n_seqs;
        ggml_cuda_pool_alloc<float> v_packed(ctx.pool(), v_elems);

        // Original strides in float units (before any swap).
        // src_v_mut currently still has original ne/nb (we haven't swapped v yet).
        const int64_t src_nb1 = (int64_t)(saved_v_nb[1] / sizeof(float)); // S_v (token stride)
        const int64_t src_nb2 = (int64_t)(saved_v_nb[2] / sizeof(float)); // S_v * n_tokens (ik_llama layout: dim1=n_tokens)
        const int64_t src_nb3 = (int64_t)(saved_v_nb[3] / sizeof(float)); // 2 * S_v * n_tokens * H_v (interleaved)
        // ik_llama v shape before swap: [S_v, n_tokens, H_v, n_seqs]
        //   ne = [S_v=128, n_tokens=2048, H_v=32, n_seqs=1]
        //   nb = [4, S_v*4, S_v*n_tokens*4, 2*S_v*n_tokens*H_v*4]
        //                      ↑dim1=n_tokens          ↑interleaved z
        // After D1 tight buffer: [S_v, H_v, n_tokens, n_seqs] contiguous
        //   nb_tight = [4, S_v*4, S_v*H_v*4, S_v*H_v*n_tokens*4]

        // Launch extract kernel: one thread per output element.
        // Grid covers all (d, h, t, s) independently.
        // Each thread: out[s*H_v*n_tokens*S_v + t*H_v*S_v + h*S_v + d]
        //            = in[s*src_nb3 + h*src_nb2 + t*src_nb1 + d*src_nb0]
        // Note: in ik_llama layout dim1=n_tokens, dim2=H_v, so:
        //   in[s*src_nb3 + h*src_nb2 + t*src_nb1 + d]
        // where src_nb2 = S_v*n_tokens (head stride), src_nb1 = S_v (token stride).
        {
            const float * v_src = (const float *)saved_v_data;
            float *       v_dst = v_packed.get();
            // Total elements: v_elems. Use a 1D grid of 256-thread blocks.
            const int threads = 256;
            const int blocks  = (v_elems + threads - 1) / threads;
            // Pass strides and dims to the kernel.
            // Kernel reads:  v_src[seq*src_nb3 + head*src_nb2 + tok*src_nb1 + dim]
            // Kernel writes: v_dst[seq*H_v*n_tokens*S_v + head*n_tokens*S_v + tok*S_v + dim]
            // (tight layout: [S_v, H_v, n_tokens, n_seqs] — same as what chunked expects)
            extract_v_interleaved<<<blocks, threads, 0, ctx.stream()>>>(
                v_src, v_dst,
                S_v, H_v, n_tokens, n_seqs,
                src_nb1, src_nb2, src_nb3);
            CUDA_CHECK(cudaGetLastError());
        }

        // Override v metadata to point at the tight buffer.
        // After swap, chunked kernel expects v as [S_v, H_v, n_tokens, n_seqs]:
        src_v_mut->data  = v_packed.get();
        src_v_mut->ne[0] = S_v;               src_v_mut->nb[0] = sizeof(float);
        src_v_mut->ne[1] = H_v;               src_v_mut->nb[1] = S_v * sizeof(float);
        src_v_mut->ne[2] = n_tokens;           src_v_mut->nb[2] = S_v * H_v * sizeof(float);
        src_v_mut->ne[3] = n_seqs;             src_v_mut->nb[3] = S_v * H_v * n_tokens * sizeof(float);

        // Log engagement once on first chunked call (no overhead after first hit).
        static bool _logged_once = false;
        if (!_logged_once) {
            fprintf(stderr, "[CHUNKED-PATH-C+D1] engaged at n_tokens=%ld, head_dim=%ld, H_v=%ld, H_k=%ld\n",
                n_tokens, head_dim, n_heads, n_heads_kq);
            fflush(stderr);
            _logged_once = true;
        }

        delta_net_chunk(ctx, dst);

        // Restore all metadata (v_packed auto-frees via RAII).
        for (int i = 0; i < 4; i++) {
            src_q_mut->ne[i]    = saved_q_ne[i]; src_q_mut->nb[i]    = saved_q_nb[i];
            src_k_mut->ne[i]    = saved_k_ne[i]; src_k_mut->nb[i]    = saved_k_nb[i];
            src_v_mut->ne[i]    = saved_v_ne[i]; src_v_mut->nb[i]    = saved_v_nb[i];
            src_g_mut->ne[i]    = saved_g_ne[i]; src_g_mut->nb[i]    = saved_g_nb[i];
            src_beta_mut->ne[i] = saved_b_ne[i]; src_beta_mut->nb[i] = saved_b_nb[i];
        }
        src_v_mut->data = saved_v_data;
        return;
    }

    // Get device info from ctx (avoids calling CUDA runtime APIs inside dispatch)
    const int device_id = ctx.device;
    const int cc = ggml_cuda_info().devices[device_id].cc;

    delta_net_f32_cuda(
        (const float *)src0->data,
        (const float *)src1->data,
        (const float *)src2->data,
        (const float *)src3->data,
        (const float *)src4->data,
        (const float *)src5->data,
        (float *)dst->data,
        src6 ? (float *)src6->data : nullptr,
        head_dim, n_tokens, n_heads, gqa_ratio, repeat_type, n_seqs,
        src2->nb[1]/sizeof(float), src2->nb[2]/sizeof(float), src2->nb[3]/sizeof(float),
        device_id, cc,
        ctx.stream());

}
