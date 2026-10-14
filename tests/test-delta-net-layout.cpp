#include "ggml.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include "llama-delta-net.h"
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
enum class decay_layout { legacy_scalar, aligned_scalar, per_key };

void require(bool value, const std::string & message) {
    if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}

void compare(ggml_tensor * tensor, const std::vector<float> & expected, const std::string & label) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    float maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), label + " nonfinite at " + std::to_string(i));
        maximum = std::max(maximum, std::fabs(actual[i] - expected[i]));
    }
    require(maximum < 2.0e-5f, label + " max error " + std::to_string(maximum));
}

// An independent recurrence oracle: nonzero initial state, unit Q/K, changing
// values/beta, and genuinely per-key KDA decay. Check state as well as output.
void run(ggml_backend_t backend, int d, int tokens, decay_layout layout, bool fused_copy,
         bool graph_builder = false) {
    const int heads = graph_builder ? d : 2;
    const bool per_key = layout == decay_layout::per_key;
    const bool checkpoints = !fused_copy && tokens == d;
    const size_t state_size = size_t(d) * d * heads;
    const std::string label = "D=" + std::to_string(d) + " T=" + std::to_string(tokens) +
        " layout=" + std::to_string(int(layout)) + " fused=" + std::to_string(fused_copy) +
        " builder=" + std::to_string(graph_builder);
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, label + " context");
    auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, tokens, heads, 1);
    auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, tokens, heads, 1);
    auto v_raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, tokens, 1);
    auto v = ggml_permute(ctx, v_raw, 0, 2, 1, 3);
    auto g_raw = per_key ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, tokens, heads, 1)
                        : ggml_new_tensor_3d(ctx, GGML_TYPE_F32, heads, tokens, 1);
    auto g = per_key ? g_raw : layout == decay_layout::legacy_scalar
        ? ggml_permute(ctx, g_raw, 2, 0, 3, 1) : ggml_permute(ctx, g_raw, 2, 1, 0, 3);
    auto beta_raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, heads, 1, tokens, 1);
    auto beta = ggml_permute(ctx, beta_raw, 2, 0, 1, 3);
    // Match the production KDA builder's contiguous beta rows per head.
    if (per_key) beta = ggml_cont(ctx, beta);
    auto prior = graph_builder ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d, heads, 1)
                              : ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d * heads, 1, 1);
    auto saved = checkpoints ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (tokens - 1) * state_size) : nullptr;
    ggml_tensor * result = nullptr;
    if (graph_builder) {
        const llm_build_cb callback = [](ggml_tensor *, const char *, int) {};
        auto built = delta_net::build_fused_delta_net(ctx, q, k, v_raw, g_raw, beta_raw, prior,
                                                     0, callback, 0, saved);
        result = built.first->view_src;
        require(result && result->op == GGML_OP_DELTA_NET, label + " builder recurrence");
    } else result = ggml_delta_net(ctx, q, k, v, g, beta, prior, saved);
    auto output = ggml_view_1d(ctx, result, d * heads * tokens, 0);
    auto final_state = ggml_view_1d(ctx, result, state_size, d * heads * tokens * sizeof(float));
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    if (fused_copy) {
        auto copy = ggml_cpy(ctx, ggml_reshape_4d(ctx, final_state, d, d * heads, 1, 1), prior);
        ggml_build_forward_expand(graph, copy);
    } else ggml_build_forward_expand(graph, result);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, label + " allocation");
    if (fused_copy) {
        bool eligible = false;
        for (int i = 0; i < graph->n_nodes; ++i)
            if (graph->nodes[i] == result) eligible = ggml_delta_net_find_state_cpy(graph, i) >= 0;
        require(eligible, label + " fused-copy eligibility");
    }

    std::vector<float> qs(d * tokens * heads, 0), ks(qs.size(), 0), vs(qs.size());
    std::vector<float> gs(per_key ? qs.size() : heads * tokens), bs(heads * tokens);
    std::vector<float> initial(state_size), expected(d * heads * tokens), expected_steps;
    for (size_t i = 0; i < initial.size(); ++i) initial[i] = 0.003f * std::sin(float(i) * 0.13f);
    for (int h = 0; h < heads; ++h) for (int t = 0; t < tokens; ++t) {
        const size_t offset = size_t(d) * (t + tokens * h);
        qs[offset] = 0.6f; qs[offset + 1] = 0.8f;
        ks[offset] = 0.8f; ks[offset + 1] = 0.6f;
        bs[h + heads * t] = -0.7f + 0.01f * (t % 17) + 0.1f * h;
        for (int j = 0; j < d; ++j) {
            vs[j + d * (h + heads * t)] = 0.08f * std::sin(0.1f * j + 0.07f * t + h);
            if (per_key) gs[j + d * (t + tokens * h)] = -0.015f - 0.001f * (j % 7) - 0.0003f * (t % 13);
        }
        if (!per_key) gs[h + heads * t] = -0.015f - 0.001f * h - 0.0003f * (t % 13);
    }
    auto state = initial;
    const float scale = 1.0f / std::sqrt(float(d));
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            const float b = 1.0f / (1.0f + std::exp(-bs[h + heads * t]));
            for (int row = 0; row < d; ++row) {
                double sk = 0;
                for (int col = 0; col < d; ++col) {
                    const float decay = std::exp(per_key ? gs[col + d * (t + tokens * h)] : gs[h + heads * t]);
                    auto & s = state[col + d * (row + d * h)];
                    s *= decay;
                    sk += double(s) * ks[col + d * (t + tokens * h)];
                }
                const float update = b * (vs[row + d * (h + heads * t)] - float(sk));
                double out = 0;
                for (int col = 0; col < d; ++col) {
                    auto & s = state[col + d * (row + d * h)];
                    s += update * ks[col + d * (t + tokens * h)];
                    out += double(s) * qs[col + d * (t + tokens * h)];
                }
                expected[row + d * (h + heads * t)] = float(out) * scale;
            }
        }
        if (checkpoints && t + 1 < tokens) expected_steps.insert(expected_steps.end(), state.begin(), state.end());
    }
    auto put = [](ggml_tensor * dst, const std::vector<float> & data) {
        ggml_backend_tensor_set(dst, data.data(), 0, data.size() * sizeof(float));
    };
    put(q, qs); put(k, ks); put(v_raw, vs); put(g_raw, gs); put(beta_raw, bs); put(prior, initial);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, label + " compute");
    compare(output, expected, label + " output");
    compare(fused_copy ? prior : final_state, state, label + " final state");
    if (saved) compare(saved, expected_steps, label + " checkpoints");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    std::cout << "PASS: " << label << '\n';
}
} // namespace

int main() {
    ggml_backend_t backend = nullptr;
    const char * device = std::getenv("DELTA_NET_TEST_CUDA_DEVICE");
#ifdef GGML_USE_CUDA
    if (device) backend = ggml_backend_cuda_init(std::stoi(device), nullptr, nullptr);
#else
    require(!device, "CUDA test requested from CPU-only build");
#endif
    if (!device) backend = ggml_backend_cpu_init();
    require(backend != nullptr, "requested backend");
    const std::vector<int> widths = device ? std::vector<int>{64, 128} : std::vector<int>{32, 64, 128};
    for (int d : widths) for (int tokens : {d - 1, d, d + 1})
        for (auto layout : {decay_layout::legacy_scalar, decay_layout::aligned_scalar, decay_layout::per_key})
            for (bool fused : {false, true}) run(backend, d, tokens, layout, fused);
    // Before permutation the scalar gate starts with the value-head count.
    // Equal value-head count/head width must not classify that as KDA either.
    run(backend, 64, 2, decay_layout::legacy_scalar, false, true);
    run(backend, 64, 2, decay_layout::per_key, false, true);
    ggml_backend_free(backend);
    return 0;
}
