#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-quants.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static bool run(ggml_backend_t backend, const std::string & mode) {
    const bool with_id = mode != "dense";
    // ggml_init fills the fp16 lookup table the converters use, so it must come first.
    ggml_init_params params = {2 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    // Every possible packed byte, multiple scales, and block/row/expert boundaries.
    constexpr int n = 640, rows = 4, experts = 3, tokens = 9, selected = 2;
    std::vector<block_q2_0> packed(n / 64 * rows * experts);
    std::vector<float> decoded(n * rows * experts);
    for (size_t b = 0; b < packed.size(); ++b) {
        packed[b].d = ggml_fp32_to_fp16((int(b % 7) - 3) * 0.125f);
        for (int j = 0; j < 16; ++j) packed[b].qs[j] = uint8_t(b * 16 + j);
        for (int j = 0; j < 64; ++j) {
            const int code = (unsigned(packed[b].qs[j / 4]) / (1u << (2 * (j % 4)))) % 4;
            decoded[b * 64 + j] = ggml_fp16_to_fp32(packed[b].d) * (code - 1);
        }
    }
    std::vector<float> actual(decoded.size());
    std::fprintf(stderr, "%s: dequant\n", ggml_backend_name(backend));
    dequantize_row_q2_0(packed.data(), actual.data(), decoded.size());
    if (actual != decoded || ggml_row_size(GGML_TYPE_Q2_0, n) != 180) return false;

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_Q2_0, n, rows, experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n, selected, tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, selected, tokens);
    ggml_tensor * routed = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_tensor * dense_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_Q2_0, n, rows);
    ggml_tensor * dense_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, tokens);
    ggml_tensor * dense = ggml_mul_mat(ctx, dense_weights, dense_input);
    // The oracle uses original F32 activations and accumulates into F32.
    // Default CUDA dispatch may quantize activations or produce F16 output.
    ggml_mul_mat_set_prec(routed, GGML_PREC_F32);
    ggml_mul_mat_set_prec(dense, GGML_PREC_F32);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    if (with_id) ggml_build_forward_expand(graph, routed);
    if (mode != "id") ggml_build_forward_expand(graph, dense);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) { ggml_free(ctx); return false; }
    (void) 0;
    std::vector<float> x(n * selected * tokens);
    std::vector<int32_t> routes(selected * tokens);
    // Nontrivial integer activations across block, token and route boundaries.
    for (size_t i = 0; i < x.size(); ++i) x[i] = int(i % 255) - 127;
    for (size_t i = 0; i < routes.size(); ++i) routes[i] = i % experts;
    ggml_backend_tensor_set(weights, packed.data(), 0, ggml_nbytes(weights));
    ggml_backend_tensor_set(dense_weights, packed.data(), 0, ggml_nbytes(dense_weights));
    ggml_backend_tensor_set(input, x.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(dense_input, x.data(), 0, ggml_nbytes(dense_input));
    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
    std::fprintf(stderr, "%s: compute\n", ggml_backend_name(backend));
    bool passed = ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS;
    std::vector<float> result(rows * selected * tokens), dense_result(rows * tokens);
    ggml_backend_tensor_get(routed, result.data(), 0, ggml_nbytes(routed));
    ggml_backend_tensor_get(dense, dense_result.data(), 0, ggml_nbytes(dense));
    for (int t = 0; t < tokens; ++t) {
        for (int s = 0; s < (with_id ? selected : 0); ++s) {
            for (int r = 0; r < rows; ++r) {
                float ref = 0;
                for (int j = 0; j < n; ++j) ref += decoded[j + n * (r + rows * routes[s + selected * t])] * x[j + n * (s + selected * t)];
                const float got = result[r + rows * (s + selected * t)];
                if (std::fabs(got - ref) > 0.02f) {
                    std::fprintf(stderr, "%s routed mismatch t=%d s=%d r=%d: %g != %g\n", ggml_backend_name(backend), t, s, r, got, ref);
                    passed = false;
                }
            }
        }
        for (int r = 0; r < rows; ++r) {
            float ref = 0;
            for (int j = 0; j < n; ++j) ref += decoded[j + n * r] * x[j + n * t];
            if (std::fabs(dense_result[r + rows * t] - ref) > 0.02f) {
                std::fprintf(stderr, "%s dense mismatch t=%d r=%d: %g != %g\n", ggml_backend_name(backend), t, r, dense_result[r + rows * t], ref);
                passed = false;
            }
        }
    }
    std::fprintf(stderr, "%s: done passed=%d\n", ggml_backend_name(backend), int(passed));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return passed;
}

int main(int argc, char ** argv) {
    const std::string mode = argc < 2 ? "all" : argv[1];
    ggml_backend_t cpu = ggml_backend_cpu_init();
    bool passed = cpu && run(cpu, mode);
    ggml_backend_free(cpu);
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        ggml_backend_t gpu = ggml_backend_cuda_init(0, nullptr, nullptr);
        passed = gpu && run(gpu, mode) && passed;
        ggml_backend_free(gpu);
    } else {
        std::fprintf(stderr, "SKIP: CUDA unavailable\n");
    }
#endif
    return passed ? 0 : 1;
}
