#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int D = 64;
constexpr int T = 3;
constexpr int H = 2;

void check(bool value, const std::string & message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::vector<float> make_data(size_t count, float scale, float phase) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = scale * (std::sin(float(i) * 0.071f + phase) +
                             0.5f * std::cos(float(i) * 0.037f - phase));
    }
    return result;
}

std::vector<float> reference(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & g,
        const std::vector<float> & beta,
        const std::vector<float> & initial) {
    std::vector<float> out(D * T * H);
    std::vector<float> state = initial;
    std::vector<float> v_new(D);
    constexpr float eps = 1.0e-12f;
    const float scale = 1.0f / std::sqrt(float(D));

    for (int h = 0; h < H; ++h) {
        float * state_h = state.data() + h * D * D;
        for (int t = 0; t < T; ++t) {
            const float * q_t = q.data() + h * D * T + t * D;
            const float * k_t = k.data() + h * D * T + t * D;
            const float * v_t = v.data() + h * D * T + t * D;
            const float * g_t = g.data() + h * D * T + t * D;
            float q2 = 0.0f;
            float k2 = 0.0f;
            for (int i = 0; i < D; ++i) {
                q2 += q_t[i] * q_t[i];
                k2 += k_t[i] * k_t[i];
            }
            const float qi = 1.0f / std::sqrt(q2 + eps);
            const float ki = 1.0f / std::sqrt(k2 + eps);
            for (int col = 0; col < D; ++col) {
                const float decay = std::exp(std::min(g_t[col], 50.0f));
                for (int row = 0; row < D; ++row) {
                    state_h[col + row * D] *= decay;
                }
            }
            float score = 0.0f;
            for (int col = 0; col < D; ++col) {
                score += k_t[col] * ki * q_t[col] * qi * scale;
            }
            const float b = 1.0f / (1.0f + std::exp(-beta[h * T + t]));
            for (int row = 0; row < D; ++row) {
                float sk = 0.0f;
                float sq = 0.0f;
                for (int col = 0; col < D; ++col) {
                    sk += state_h[col + row * D] * k_t[col];
                    sq += state_h[col + row * D] * q_t[col];
                }
                v_new[row] = b * (v_t[row] - sk * ki);
                out[t * D * H + h * D + row] = sq * qi * scale + v_new[row] * score;
            }
            for (int col = 0; col < D; ++col) {
                for (int row = 0; row < D; ++row) {
                    state_h[col + row * D] += v_new[row] * k_t[col] * ki;
                }
            }
        }
    }
    out.insert(out.end(), state.begin(), state.end());
    return out;
}

std::vector<float> run(ggml_backend_t backend,
        const std::vector<float> & q_data,
        const std::vector<float> & k_data,
        const std::vector<float> & v_data,
        const std::vector<float> & g_data,
        const std::vector<float> & beta_data,
        const std::vector<float> & state_data) {
    ggml_init_params params = {4 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    check(ctx != nullptr, "context");
    auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, H, 1);
    auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, H, 1);
    auto v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, H, 1);
    auto g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, T, H, 1);
    auto b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, T, H, 1);
    auto s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, D, D * H, 1, 1);
    // The model graph normalizes Q/K before invoking the fused operator.  The
    // CUDA kernel relies on that contract; the CPU fallback defensively
    // normalizes again, which is numerically idempotent.
    auto q_norm = ggml_l2_norm(ctx, q, 1.0e-12f);
    auto k_norm = ggml_l2_norm(ctx, k, 1.0e-12f);
    auto result = ggml_delta_net(ctx, q_norm, k_norm, v, g, b, s, nullptr);
    result->op_params[0] = 1;

    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer != nullptr, "backend allocation");
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(b, beta_data.data(), 0, beta_data.size() * sizeof(float));
    ggml_backend_tensor_set(s, state_data.data(), 0, state_data.size() * sizeof(float));
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);
    check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
    ggml_backend_synchronize(backend);
    std::vector<float> actual(ggml_nelements(result));
    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return actual;
}

void compare(const std::vector<float> & actual, const std::vector<float> & expected,
        float tolerance, const char * label) {
    check(actual.size() == expected.size(), std::string(label) + " size");
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        check(std::isfinite(actual[i]), std::string(label) + " finite");
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    check(max_error <= tolerance, std::string(label) + " reference parity, max error=" + std::to_string(max_error));
    std::cout << label << " max_error=" << max_error << '\n';
}

} // namespace

int main() {
    const auto q = make_data(D * T * H, 0.7f, 0.1f);
    const auto k = make_data(D * T * H, 0.6f, 0.4f);
    const auto v = make_data(D * T * H, 0.5f, 0.8f);
    const auto g = make_data(D * T * H, 0.03f, -0.2f);
    const auto b = make_data(T * H, 0.4f, 0.7f);
    const auto s = make_data(D * D * H, 0.02f, 0.3f);
    const auto expected = reference(q, k, v, g, b, s);

    auto cpu = ggml_backend_cpu_init();
    check(cpu != nullptr, "CPU backend");
    compare(run(cpu, q, k, v, g, b, s), expected, 2.0e-5f, "CPU KDA");
    ggml_backend_free(cpu);

#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() > 0) {
        auto cuda = ggml_backend_cuda_init(0, nullptr, nullptr);
        check(cuda != nullptr, "CUDA backend");
        compare(run(cuda, q, k, v, g, b, s), expected, 3.0e-4f, "CUDA KDA");
        ggml_backend_free(cuda);
    }
#endif
    return 0;
}
