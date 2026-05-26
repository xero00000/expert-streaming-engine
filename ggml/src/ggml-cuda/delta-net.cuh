#include "common.cuh"

void ggml_cuda_op_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// 2026-05-26: chunked prefill path ported from mainline am17an branch
// (commit 8ea2990). Implemented in delta-net-chunk.cu. Called from
// ggml_cuda_op_delta_net() when the shape matches the chunked kernel's
// requirements (S_v == 128, n_tokens > 1, not KDA) AND saved_steps
// (src[6]) is null. Otherwise the existing sequential kernel runs.
void delta_net_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// 2026-05-26: chunked prefill path ported from mainline am17an branch
// (commit 8ea2990). Implemented in delta-net-chunk.cu. Called from
// ggml_cuda_op_delta_net() when the shape matches the chunked kernel's
// requirements (S_v == 128, n_tokens > 1, not KDA) AND saved_steps
// (src[6]) is null. Otherwise the existing sequential kernel runs.
void delta_net_chunk(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
