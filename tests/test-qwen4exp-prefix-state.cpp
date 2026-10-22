#include "ggml.h"
#include "ggml-backend.h"
#include "llama-qwen4exp-ops.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
std::vector<float> read(ggml_tensor * value) {
    std::vector<float> result(ggml_nelements(value));
    ggml_backend_tensor_get(value, result.data(), 0, result.size() * sizeof(float));
    return result;
}
void put(ggml_tensor * value, const std::vector<float> & data) {
    check(size_t(ggml_nelements(value)) == data.size(), "input geometry");
    ggml_backend_tensor_set(value, data.data(), 0, data.size() * sizeof(float));
}
void exact(const float * a, const float * b, size_t count, const char * message) {
    if (count) check(std::memcmp(a, b, count * sizeof(float)) == 0, message);
}
struct shape {
    int width, key_heads, value_heads, conv_kernel, ple_channels, ple_kernel, dilation;
    int conv_channels() const { return width * (2 * key_heads + value_heads); }
    int conv_size() const { return conv_channels() * (conv_kernel - 1); }
    int ssm_size() const { return width * width * value_heads; }
    int ple_size() const { return ple_channels * (ple_kernel - 1) * dilation; }
    int state_size() const { return conv_size() + ssm_size() + ple_size(); }
};
struct result { std::vector<float> states, conv, delta, ple; };

// Each operator has a real producer. Expanding ONLY the complete journal
// catches missing graph dependencies for the operators' side-effect leaves.
result evaluate(ggml_backend_t backend, const shape & s, int tokens, int start,
                const std::vector<float> & initial, int poison_from = -1, bool copy_state = false) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    check(ctx, "context");
    const int d = s.width, hk = s.key_heads, hv = s.value_heads;
    const int c = s.conv_channels(), l = s.conv_kernel - 1;
    auto * conv_prior = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, l, c, 1);
    auto * conv_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c, tokens);
    auto * conv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.conv_kernel, c);
    auto * sequences = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, tokens);
    auto * conv_saved = tokens > 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, int64_t(s.conv_size()) * tokens) : nullptr;
    auto * conv = ggml_ssm_conv(ctx, conv_prior, conv_input, conv_weight, sequences, conv_saved);

    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, tokens, hk, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, tokens, hk, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, tokens, hv, 1);
    auto * gate_raw = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hv, tokens, 1);
    auto * gate = ggml_permute(ctx, gate_raw, 2, 0, 3, 1);
    auto * beta_raw = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, hv, 1, tokens, 1);
    auto * beta = ggml_permute(ctx, beta_raw, 2, 0, 1, 3);
    auto * ssm_prior = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d * hv, 1, 1);
    auto * ssm_saved = tokens > 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, int64_t(s.ssm_size()) * (tokens - 1)) : nullptr;
    auto * delta = ggml_delta_net(ctx, q, k, v, gate, beta, ssm_prior, ssm_saved);

    ggml_tensor * ple = nullptr, * ple_input = nullptr, * ple_prior = nullptr;
    ggml_tensor * ple_weights = nullptr, * columns = nullptr, * ple_output = nullptr;
    if (s.ple_channels) {
        const int history = (s.ple_kernel - 1) * s.dilation;
        ple_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.ple_channels, tokens);
        ple_prior = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.ple_channels, history);
        ple_weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, s.ple_kernel, s.ple_channels);
        columns = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, history * tokens);
        ple_output = qwen4exp_build_ple_dilated_conv(ctx, ple_input, ple_weights,
                ple_prior, s.ple_kernel, s.dilation, &ple, columns).output;
    }
    auto * states = qwen4exp_build_recurrent_prefix_states(ctx, conv, delta, ple);
    check(states->ne[0] == s.state_size() && states->ne[1] == tokens && ggml_is_contiguous(states), "complete state layout");
    auto * graph = ggml_new_graph_custom(ctx, 512, false);
    if (copy_state) {
        auto * final_ssm = ggml_view_4d(ctx, delta, d, d * hv, 1, 1,
                d * sizeof(float), d * d * hv * sizeof(float), d * d * hv * sizeof(float),
                size_t(d) * hv * tokens * sizeof(float));
        ggml_build_forward_expand(graph, ggml_cpy(ctx, final_ssm, ssm_prior));
    }
    ggml_build_forward_expand(graph, states);
    if (ple_output) ggml_build_forward_expand(graph, ple_output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer, "allocation");
    put(conv_prior, {initial.begin(), initial.begin() + s.conv_size()});
    put(ssm_prior, {initial.begin() + s.conv_size(), initial.begin() + s.conv_size() + s.ssm_size()});
    std::vector<float> inputs(size_t(c) * tokens), weights(size_t(s.conv_kernel) * c);
    std::vector<float> qs(size_t(d) * tokens * hk, 0.f), ks(qs.size(), 0.f);
    std::vector<float> vs(size_t(d) * tokens * hv), gs(size_t(hv) * tokens), bs(gs.size());
    for (int t = 0; t < tokens; ++t) {
        const bool poison = poison_from >= 0 && t >= poison_from;
        for (int j = 0; j < c; ++j) inputs[j + c * t] = poison ? 64.f : (j % 17 == 0 ? -0.f : .01f * float(j % 23 + start + t + 1));
        for (int h = 0; h < hk; ++h) {
            qs[d * (t + tokens * h)] = .6f;
            qs[1 + d * (t + tokens * h)] = .8f;
            ks[d * (t + tokens * h)] = .8f;
            ks[1 + d * (t + tokens * h)] = .6f;
        }
        for (int h = 0; h < hv; ++h) {
            gs[h + hv * t] = -.03f - .001f * h;
            bs[h + hv * t] = .4f + .003f * ((start + t) % 7);
            for (int j = 0; j < d; ++j) vs[j + d * (t + tokens * h)] = poison ? -32.f : .03f * std::sin(.13f * j + .17f * (start + t) + h);
        }
    }
    for (size_t i = 0; i < weights.size(); ++i) weights[i] = .05f * float(i % 5 + 1);
    put(conv_input, inputs); put(conv_weight, weights);
    put(q, qs); put(k, ks); put(v, vs); put(gate_raw, gs); put(beta_raw, bs);
    std::vector<int32_t> ids(tokens, 0);
    ggml_backend_tensor_set(sequences, ids.data(), 0, ids.size() * sizeof(int32_t));
    std::vector<float> ple_values;
    if (ple) {
        const int history = (s.ple_kernel - 1) * s.dilation;
        put(ple_prior, {initial.end() - s.ple_size(), initial.end()});
        ple_values.resize(size_t(s.ple_channels) * tokens);
        for (int t = 0; t < tokens; ++t) for (int j = 0; j < s.ple_channels; ++j)
            ple_values[j + s.ple_channels * t] = poison_from >= 0 && t >= poison_from ? 128.f : (j % 11 == 0 ? -0.f : .02f * float(j % 29 + start + t));
        put(ple_input, ple_values);
        std::vector<ggml_fp16_t> half(size_t(s.ple_kernel) * s.ple_channels);
        for (size_t i = 0; i < half.size(); ++i) half[i] = ggml_fp32_to_fp16(.07f * float(i % 3 + 1));
        ggml_backend_tensor_set(ple_weights, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
        std::vector<int32_t> indexes(size_t(history) * tokens);
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < history; ++h) indexes[h + history * t] = t + h + 1;
        ggml_backend_tensor_set(columns, indexes.data(), 0, indexes.size() * sizeof(int32_t));
    }
    for (int replay = 0; replay < 2; ++replay) {
        if (copy_state) put(ssm_prior, {initial.begin() + s.conv_size(), initial.begin() + s.conv_size() + s.ssm_size()});
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    }
    result out{read(states), read(conv), read(delta), ple_output ? read(ple_output) : std::vector<float>{}};
    const auto conv_steps = conv_saved ? read(conv_saved) : std::vector<float>{};
    const auto ssm_steps = ssm_saved ? read(ssm_saved) : std::vector<float>{};
    const auto ple_steps = ple ? read(ple) : std::vector<float>{};
    // Independent channel-major convolution FIFO and F64 recurrence oracle.
    std::vector<float> conv_fifo(initial.begin(), initial.begin() + s.conv_size());
    std::vector<float> ple_fifo(initial.end() - s.ple_size(), initial.end());
    std::vector<double> oracle(initial.begin() + s.conv_size(), initial.begin() + s.conv_size() + s.ssm_size());
    for (int t = 0; t < tokens; ++t) {
        auto * row = out.states.data() + size_t(t) * s.state_size();
        for (int j = 0; j < c; ++j) {
            for (int h = 0; h + 1 < l; ++h) conv_fifo[h + l * j] = conv_fifo[h + 1 + l * j];
            conv_fifo[l - 1 + l * j] = inputs[j + c * t];
        }
        exact(row, conv_fifo.data(), s.conv_size(), "conv FIFO bits");
        if (tokens > 1) exact(row, conv_steps.data() + size_t(t) * s.conv_size(), s.conv_size(), "conv journal bits");
        const auto * actual_ssm = row + s.conv_size();
        const auto * source_ssm = t + 1 < tokens ? ssm_steps.data() + size_t(t) * s.ssm_size() : out.delta.data() + size_t(d) * hv * tokens;
        exact(actual_ssm, source_ssm, s.ssm_size(), "SSM journal bits");
        for (int h = 0; h < hv; ++h) for (int value = 0; value < d; ++value) {
            const int kh = h / (hv / hk);
            double dot = 0;
            for (int key = 0; key < d; ++key) {
                auto & state = oracle[key + d * (value + d * h)];
                state *= std::exp(double(gs[h + hv * t]));
                dot += state * ks[key + d * (t + tokens * kh)];
            }
            const double update = (double(vs[value + d * (t + tokens * h)]) - dot) / (1.0 + std::exp(-double(bs[h + hv * t])));
            for (int key = 0; key < d; ++key) oracle[key + d * (value + d * h)] += update * ks[key + d * (t + tokens * kh)];
        }
        for (int j = 0; j < s.ssm_size(); ++j) check(std::fabs(double(actual_ssm[j]) - oracle[j]) < 2.e-5, "SSM F64 prefix oracle");
        if (ple) {
            std::move(ple_fifo.begin() + s.ple_channels, ple_fifo.end(), ple_fifo.begin());
            std::copy(ple_values.begin() + size_t(t) * s.ple_channels, ple_values.begin() + size_t(t + 1) * s.ple_channels,
                      ple_fifo.end() - s.ple_channels);
            exact(row + s.conv_size() + s.ssm_size(), ple_fifo.data(), s.ple_size(), "PLE FIFO bits");
            exact(row + s.conv_size() + s.ssm_size(), ple_steps.data() + size_t(t) * s.ple_size(), s.ple_size(), "PLE journal bits");
        }
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return out;
}
int run(ggml_backend_t backend, int & cuts) {
    int cases = 0;
    const std::vector<shape> shapes = {
        {64, 2, 2, 2, 0, 0, 0}, {64, 2, 6, 3, 3, 3, 2},
        {128, 2, 6, 4, 160, 3, 4}, {128, 16, 48, 4, 10240, 3, 4},
    };
    for (const auto & s : shapes) for (int tokens : {1, 2, 3, 4, 8}) {
        std::vector<float> initial(s.state_size());
        for (size_t i = 0; i < initial.size(); ++i) initial[i] = i % 13 == 0 ? -0.f : .002f * std::cos(.17f * float(i));
        const auto original = evaluate(backend, s, tokens, 0, initial);
        const auto copied = evaluate(backend, s, tokens, 0, initial, -1, true);
        exact(original.states.data(), copied.states.data(), original.states.size(), "state-copy consumer invalidates journal tail");
        // Seed three sequence rows with a stride larger than the live state.
        // Restore a prefix into the middle row through the backend copy op.
        for (int kept = 0; kept <= tokens; ++kept) {
            const auto poisoned = evaluate(backend, s, tokens, 0, initial, kept);
            exact(original.states.data(), poisoned.states.data(), size_t(kept) * s.state_size(), "poisoned suffix changes accepted state");
            std::vector<float> retained = initial;
            if (kept) std::copy_n(original.states.data() + size_t(kept - 1) * s.state_size(), s.state_size(), retained.data());
            auto * ctx = ggml_init({1024 * 1024, nullptr, true});
            auto * bank = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, s.state_size() + 7, 3);
            auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, s.state_size());
            auto * dst = ggml_view_1d(ctx, bank, s.state_size(), bank->nb[1]);
            auto * graph = ggml_new_graph_custom(ctx, 64, false);
            ggml_build_forward_expand(graph, ggml_cpy(ctx, input, dst));
            auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            check(buffer, "restore allocation");
            std::vector<float> guards(ggml_nelements(bank), -777.f);
            put(bank, guards); put(input, retained);
            check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "restore compute");
            auto restored_bank = read(bank);
            std::copy(retained.begin(), retained.end(), guards.begin() + s.state_size() + 7);
            exact(restored_bank.data(), guards.data(), guards.size(), "restore changes another sequence or padding");
            std::vector<float> restored(restored_bank.begin() + s.state_size() + 7, restored_bank.begin() + 2 * s.state_size() + 7);
            ggml_backend_buffer_free(buffer); ggml_free(ctx);
            const auto resumed = evaluate(backend, s, 1, kept, restored);
            const auto teacher = evaluate(backend, s, 1, kept, retained);
            exact(resumed.states.data(), teacher.states.data(), s.state_size(), "restored complete continuation state");
            exact(resumed.conv.data(), teacher.conv.data(), resumed.conv.size(), "restored conv continuation");
            exact(resumed.delta.data(), teacher.delta.data(), resumed.delta.size(), "restored recurrent continuation");
            exact(resumed.ple.data(), teacher.ple.data(), resumed.ple.size(), "restored PLE continuation");
            ++cuts;
        }
        ++cases;
    }
    return cases;
}
}
int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : -1;
#ifdef GGML_USE_CUDA
        CUdevice cuda_device; CUcontext primary = nullptr;
        if (device >= 0) {
            check(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&cuda_device, device) == CUDA_SUCCESS &&
                  cuDevicePrimaryCtxRetain(&primary, cuda_device) == CUDA_SUCCESS && cuCtxSetCurrent(primary) == CUDA_SUCCESS,
                  "CUDA primary context");
            backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        } else
#endif
        {
            check(device == -1, "CUDA unavailable");
            backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(backend, 2);
        }
        check(backend, "backend");
        int cuts = 0;
        const int cases = run(backend, cuts);
        ggml_backend_free(backend); backend = nullptr;
#ifdef GGML_USE_CUDA
        if (device >= 0) check(cuDevicePrimaryCtxRelease(cuda_device) == CUDA_SUCCESS, "primary release");
#endif
        std::printf("PASS: %d complete prefix-state cases, %d accepted cuts; independent FIFO/F64 oracles, producer dependencies, sequence guards, exact restored continuation\n", cases, cuts);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        if (backend) ggml_backend_free(backend);
        return 1;
    }
}
