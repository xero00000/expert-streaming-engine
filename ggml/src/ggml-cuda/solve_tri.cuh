#include "common.cuh"

void ggml_cuda_op_solve_tri(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// 2026-05-26: ported from mainline am17an branch (commit 8ea2990) — used by
// the chunked Gated Delta Net kernel in delta-net-chunk.cu. Standalone
// solve-triangular call that takes raw pointers (vs. ggml_cuda_op_solve_tri
// which works on a tensor's src[0]/src[1]). Routes to the fast in-shared-mem
// kernel for small matrices, cuBLAS batched-TRSM otherwise.
void ggml_cuda_solve_tri(ggml_backend_cuda_context & ctx,
                         const float * A, const float * B, float * X,
                         int n, int k,
                         int64_t ne02, int64_t ne03,
                         size_t nb02, size_t nb03,
                         size_t nb12, size_t nb13,
                         size_t nb2,  size_t nb3);
