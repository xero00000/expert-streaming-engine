#include "llama-qwen4exp-ops.h"
#include "llama-delta-net.h"

#include "ggml-backend.h"
#include "../ggml/src/ggml-impl.h"
#include "../ggml/src/ggml-quants.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
extern "C" {
#include "sha256/sha256.h"
}
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

constexpr int64_t H = 5;
constexpr int64_t S = 4;
constexpr int64_t R = 3;
constexpr int64_t T = 2;
constexpr int64_t E = 6;
constexpr int64_t K = 4;
constexpr int64_t D = 3;
constexpr float EPS = 1.0e-6f;

void require(bool value, const std::string & message) {
    if (!value) {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

std::vector<float> values(size_t count, float scale, float phase) {
    std::vector<float> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = scale * (std::sin(float(i) * 0.37f + phase) + 0.4f * std::cos(float(i) * 0.19f - phase));
    }
    return out;
}

struct reference_result {
    std::vector<float> mixed;
    std::vector<float> injection;
    std::vector<float> residual;
};

reference_result reference(
        const std::vector<float> & input,
        const std::vector<float> & norm_weight,
        const std::vector<float> & down,
        const std::vector<float> & up,
        const std::vector<float> & inject) {
    constexpr int64_t HS = H * S;
    std::vector<float> normed(HS * T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            float mean_square = 0.0f;
            for (int64_t h = 0; h < H; ++h) {
                const float x = input[h + H * s + HS * t];
                mean_square += x * x;
            }
            const float inv_rms = 1.0f / std::sqrt(mean_square / float(H) + EPS);
            for (int64_t h = 0; h < H; ++h) {
                const int64_t i = h + H * s;
                // GGUF stores Transformers' effective (1 + raw_weight) gamma.
                normed[i + HS * t] = input[i + HS * t] * inv_rms * norm_weight[i];
            }
        }
    }

    std::vector<float> low(R * T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t r = 0; r < R; ++r) {
            float sum = 0.0f;
            for (int64_t i = 0; i < HS; ++i) sum += down[i + HS * r] * normed[i + HS * t];
            const float x = sum / float(S);
            low[r + R * t] = x / (1.0f + std::exp(-x));
        }
    }

    reference_result result = {
        std::vector<float>(H * T),
        std::vector<float>(S * T),
        std::vector<float>(input.size()),
    };
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            for (int64_t h = 0; h < H; ++h) {
                const int64_t i = h + H * s;
                float projected = 0.0f;
                for (int64_t r = 0; r < R; ++r) projected += up[r + R * i] * low[r + R * t];
                const float weight = 1.0f / (1.0f + std::exp(-projected));
                result.mixed[h + H * t] += weight * normed[i + H * S * t] / float(S);
            }
            float projected = 0.0f;
            for (int64_t i = 0; i < HS; ++i) projected += inject[i + HS * s] * normed[i + HS * t];
            result.injection[s + S * t] = 2.0f / (1.0f + std::exp(-projected / float(S)));
        }
    }
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            for (int64_t h = 0; h < H; ++h) {
                const int64_t i = h + H * s + H * S * t;
                result.residual[i] = input[i] + result.mixed[h + H * t] * result.injection[s + S * t];
            }
        }
    }
    return result;
}

std::pair<std::vector<float>, std::vector<float>> ple_reference(
        const std::vector<float> & hidden,
        const std::vector<float> & embeddings,
        const std::vector<float> & key_projection,
        const std::vector<float> & value_projection,
        const std::vector<float> & key_norm,
        const std::vector<float> & query_norm,
        const std::vector<float> & conv_norm) {
    constexpr int64_t HS = H * S;
    auto grouped_norm = [](const std::vector<float> & input, const std::vector<float> & weight) {
        std::vector<float> output(input.size());
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t s = 0; s < S; ++s) {
                float mean_square = 0.0f;
                for (int64_t h = 0; h < H; ++h) {
                    const float x = input[h + H * s + HS * t];
                    mean_square += x * x;
                }
                const float inv_rms = 1.0f / std::sqrt(mean_square / float(H) + EPS);
                for (int64_t h = 0; h < H; ++h) {
                    const int64_t i = h + H * s;
                    output[i + HS * t] = input[i + HS * t] * inv_rms * weight[i];
                }
            }
        }
        return output;
    };

    std::vector<float> key(HS * T);
    std::vector<float> value(H * T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t i = 0; i < HS; ++i) {
            for (int64_t e = 0; e < E; ++e) key[i + HS * t] += key_projection[e + E * i] * embeddings[e + E * t];
        }
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t e = 0; e < E; ++e) value[h + H * t] += value_projection[e + E * h] * embeddings[e + E * t];
        }
    }
    key = grouped_norm(key, key_norm);
    const auto query = grouped_norm(hidden, query_norm);
    std::vector<float> gated(HS * T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            float dot = 0.0f;
            for (int64_t h = 0; h < H; ++h) {
                const int64_t i = h + H * s + HS * t;
                dot += key[i] * query[i];
            }
            dot /= std::sqrt(float(H));
            const float signed_root = std::copysign(std::sqrt(std::max(std::fabs(dot), 1.0e-6f)), dot);
            const float weight = 1.0f / (1.0f + std::exp(-signed_root));
            for (int64_t h = 0; h < H; ++h) gated[h + H * s + HS * t] = weight * value[h + H * t];
        }
    }
    return { gated, grouped_norm(gated, conv_norm) };
}

std::pair<std::vector<float>, std::vector<float>> ple_conv_reference(
        const std::vector<float> & input,
        const std::vector<float> & weight,
        const std::vector<float> & prior_state) {
    constexpr int64_t HS = H * S;
    constexpr int64_t L = (K - 1) * D;
    std::vector<float> history(HS * (L + T));
    std::copy(prior_state.begin(), prior_state.end(), history.begin());
    std::copy(input.begin(), input.end(), history.begin() + HS * L);
    std::vector<float> output(HS * T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t channel = 0; channel < HS; ++channel) {
            float sum = 0.0f;
            for (int64_t tap = 0; tap < K; ++tap) {
                sum += weight[tap + K * channel] * history[channel + HS * (t + tap * D)];
            }
            output[channel + HS * t] = sum / (1.0f + std::exp(-sum));
        }
    }
    std::vector<float> next_state(HS * L);
    std::copy(history.begin() + HS * T, history.begin() + HS * (T + L), next_state.begin());
    return { output, next_state };
}

void compare(ggml_tensor * tensor, const std::vector<float> & expected, const char * label) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    float max_error = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), std::string(label) + " produced a non-finite value");
        max_error = std::max(max_error, std::fabs(actual[i] - expected[i]));
    }
    if (std::getenv("QWEN4EXP_TEST_REPORT_ERRORS") != nullptr) {
        std::cout << label << " max_abs_error=" << max_error << '\n';
    }
    require(max_error < 2.0e-5f, std::string(label) + " parity, max error=" + std::to_string(max_error));
}

void compare_fixture(ggml_tensor * tensor, const nlohmann::json & expected, const char * label) {
    require(expected.size() == ggml_nelements(tensor), std::string(label) + " fixture tensor size");
    std::vector<float> values;
    values.reserve(expected.size());
    for (const auto & value : expected) values.push_back(value.get<float>());
    compare(tensor, values, label);
}

void compare_i32(ggml_tensor * tensor, const std::vector<int32_t> & expected, const char * label) {
    std::vector<int32_t> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(int32_t));
    require(actual == expected, std::string(label) + " index parity");
}

void compare_mask(ggml_tensor * tensor, const std::vector<float> & expected, const char * label) {
    std::vector<float> actual(expected.size());
    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < actual.size(); ++i) {
        const bool same_infinity = std::isinf(actual[i]) && std::isinf(expected[i]) &&
            std::signbit(actual[i]) == std::signbit(expected[i]);
        require(same_infinity || std::fabs(actual[i] - expected[i]) < 1.0e-6f,
                std::string(label) + " mismatch at " + std::to_string(i));
    }
}

void check_recurrence(ggml_backend_t backend, const nlohmann::json & ref, bool fused_copy,
                      bool checkpoints = true) {
    const int64_t d = ref.at("key_head_dim");
    const int64_t heads = ref.at("heads");
    const int64_t tokens = ref.at("sequence_length");
    const int64_t key_heads = ref.value("key_heads", heads);
    // Canonical source fixtures are not converted GGUF value-head storage.
    const int repeat_type = ref.value("repeat_type", 0);
    if (heads != key_heads) {
        require(repeat_type == ref.at("repeat_type").get<int>(), "Qwen4Exp head expansion matches pinned reference");
    }
    require((d == 128 || d == 32) && ref.at("value_head_dim").get<int64_t>() == d &&
            (heads == 2 || heads == 6) && key_heads == 2 && tokens == 5,
            "native recurrence supported geometry");
    const auto queries = ref.at("query").get<std::vector<float>>();
    const auto keys = ref.at("key").get<std::vector<float>>();
    const auto values = ref.at("value").get<std::vector<float>>();
    const auto decay = ref.at("g").get<std::vector<float>>();
    const auto probabilities = ref.at("beta").get<std::vector<float>>();
    const size_t state_size = d * d * heads;
    const auto state_order = [=](const nlohmann::json & state) {
        require(state.size() == state_size, "recurrent reference state size");
        std::vector<float> out(state_size);
        // Transformers [head,key,value] -> GGML key + value*D + head*D*D.
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t key = 0; key < d; ++key)
                for (int64_t value = 0; value < d; ++value)
                    out[key + d * value + d * d * h] = state.at(value + d * key + d * d * h).get<float>();
        return out;
    };
    const auto run = [&](int64_t start, int64_t count, const std::vector<float> & prior,
                         const nlohmann::json & expected_output, const std::vector<float> & expected_state) {
        ggml_init_params params = { 1024 * 1024, nullptr, true };
        auto ctx = ggml_init(params);
        require(ctx != nullptr, "recurrence context");
        // Match build_qkv/build_fused_delta_net's physical token-major inputs and
        // permutations, including non-contiguous V, G and beta views.
        auto q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, key_heads, count, 1);
        auto k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, key_heads, count, 1);
        auto v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, heads, count, 1);
        auto g = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, heads, count, 1);
        auto b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, heads, 1, count, 1);
        auto state = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, d * heads, 1, 1);
        auto q_norm = qwen4exp_build_l2_norm(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), 1.0e-6f);
        auto k_norm = qwen4exp_build_l2_norm(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), 1.0e-6f);
        auto saved = checkpoints && count > 1 ? ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (count - 1) * state_size) : nullptr;
        auto result = ggml_delta_net(ctx, q_norm, k_norm, ggml_permute(ctx, v, 0, 2, 1, 3),
                ggml_permute(ctx, g, 2, 0, 3, 1), ggml_permute(ctx, b, 2, 0, 1, 3), state, saved);
        result->op_params[0] = repeat_type;
        auto output = ggml_view_1d(ctx, result, d * heads * count, 0);
        auto final_state = ggml_view_1d(ctx, result, state_size, d * heads * count * sizeof(float));
        auto graph = ggml_new_graph_custom(ctx, 64, false);
        auto published_state = final_state;
        if (fused_copy) {
            auto copy = ggml_cpy(ctx, ggml_reshape_4d(ctx, final_state, d, d * heads, 1, 1), state);
            ggml_build_forward_expand(graph, copy);
            published_state = state;
        } else {
            ggml_build_forward_expand(graph, result);
        }
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "recurrence allocation");
        const size_t data_bytes = d * heads * count * sizeof(float);
        ggml_backend_tensor_set(q, queries.data() + d * key_heads * start, 0, d * key_heads * count * sizeof(float));
        ggml_backend_tensor_set(k, keys.data() + d * key_heads * start, 0, d * key_heads * count * sizeof(float));
        ggml_backend_tensor_set(v, values.data() + d * heads * start, 0, data_bytes);
        ggml_backend_tensor_set(g, decay.data() + heads * start, 0, heads * count * sizeof(float));
        std::vector<float> raw_beta(heads * count);
        for (size_t i = 0; i < raw_beta.size(); ++i) {
            const float p = probabilities.at(heads * start + i);
            raw_beta[i] = std::log(p / (1.0f - p)); // fused op applies sigmoid itself
        }
        ggml_backend_tensor_set(b, raw_beta.data(), 0, raw_beta.size() * sizeof(float));
        ggml_backend_tensor_set(state, prior.data(), 0, prior.size() * sizeof(float));
        if (fused_copy) {
            bool eligible = false;
            for (int i = 0; i < graph->n_nodes; ++i) {
                if (graph->nodes[i] == result) eligible = ggml_delta_net_find_state_cpy(graph, i) >= 0;
            }
            require(eligible, "recurrence copy must exercise backend fusion eligibility");
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "recurrence compute");
        compare_fixture(output, expected_output, "pinned Transformers recurrence output");
        compare(published_state, expected_state, "pinned Transformers recurrence state");
        if (saved != nullptr) {
            std::vector<float> expected_steps;
            for (int64_t step = 0; step < count - 1; ++step) {
                const auto canonical = state_order(ref.at("single_step_states").at(start + step));
                expected_steps.insert(expected_steps.end(), canonical.begin(), canonical.end());
            }
            compare(saved, expected_steps, "pinned Transformers recurrent checkpoints");
        }
        std::vector<float> actual_state(state_size);
        ggml_backend_tensor_get(published_state, actual_state.data(), 0, state_size * sizeof(float));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        return actual_state;
    };
    std::vector<float> state(state_size, 0.0f);
    run(0, tokens, state, ref.at("recurrent_output"), state_order(ref.at("recurrent_final_state")));
    const auto & step_outputs = ref.at("single_step_output");
    require(step_outputs.size() == size_t(tokens * heads * d), "single-step output size");
    require(ref.at("single_step_states").size() == size_t(tokens), "single-step state count");
    for (int64_t token = 0; token < tokens; ++token) {
        nlohmann::json expected = nlohmann::json::array();
        for (int64_t i = 0; i < heads * d; ++i) expected.push_back(step_outputs.at(token * heads * d + i));
        state = run(token, 1, state, expected, state_order(ref.at("single_step_states").at(token)));
    }
}

void check_softplus_precision(ggml_backend_t backend) {
    // Tiny negative inputs must not collapse to zero after rounding 1 + exp(x).
    const std::vector<float> input = {-80.f, -30.f, -20.f, -15.829147f, -14.171041f,
                                     -8.f, 0.f, 0.58845603f, 20.f, 21.f, 80.f};
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "softplus precision context");
    auto src = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, input.size());
    auto output = ggml_softplus(ctx, src);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "softplus precision allocation");
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "softplus precision compute");
    std::vector<float> actual(input.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    for (size_t i = 0; i < input.size(); ++i) {
        const double expected = input[i] > 20.f ? input[i] : std::log1p(std::exp(double(input[i])));
        require(std::isfinite(actual[i]) && actual[i] > 0.f, "softplus must remain finite and positive");
        require(std::abs(double(actual[i]) - expected) <= expected * 1e-6,
                "softplus precision against double reference, input=" + std::to_string(input[i]));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void check_explicit_projection_precision(ggml_backend_t backend, int64_t rows = 3, int64_t columns = 2560) {
    // A strided broadcast batch and cancellation-sensitive rows exercise the
    // opt-in floating projection path, including BF16 weights with F32 inputs.
    const int64_t tokens = rows > 256 ? 32 : 2;
    constexpr int64_t batches = 2;
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "explicit precision context");
    auto input_storage = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, columns + 8, tokens, batches);
    auto input = ggml_view_4d(ctx, input_storage, columns, tokens, batches, 1,
        input_storage->nb[1], input_storage->nb[2], input_storage->nb[3], 0);
    std::vector<float> input_data(ggml_nelements(input_storage), 0.f);
    for (int64_t b = 0; b < batches; ++b) for (int64_t t = 0; t < tokens; ++t)
        for (int64_t k = 0; k < columns; ++k)
            input_data[k + (columns + 8) * (t + tokens * b)] = rows > 256
                ? 0.00390625f + float((k * 13 + t * 7 + b) % 127) / 524288.f
                : 0.1f + float((k * 13 + t * 7 + b) % 127) / 91.f;
    auto f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, rows);
    auto bf16 = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, columns, rows);
    auto f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, columns, rows);
    std::vector<float> weights(columns * rows);
    std::vector<ggml_bf16_t> packed(weights.size());
    std::vector<ggml_fp16_t> half_packed(weights.size());
    for (int64_t r = 0; r < rows; ++r) for (int64_t k = 0; k < columns; ++k) {
        const float value = (k % 2 ? -1.f : 1.f) * (1.f + float((k + r * 3) % 17) / 16.f);
        packed[k + columns * r] = ggml_fp32_to_bf16(value);
        weights[k + columns * r] = ggml_bf16_to_fp32(packed[k + columns * r]);
        half_packed[k + columns * r] = ggml_fp32_to_fp16(weights[k + columns * r]);
    }
    auto out_f32 = ggml_mul_mat(ctx, f32, input);
    auto out_bf16 = ggml_mul_mat(ctx, bf16, input);
    auto out_f16 = ggml_mul_mat(ctx, f16, input);
    ggml_mul_mat_set_prec(out_f32, GGML_PREC_F32);
    ggml_mul_mat_set_prec(out_bf16, GGML_PREC_F32);
    ggml_mul_mat_set_prec(out_f16, GGML_PREC_F32);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, out_f32);
    ggml_build_forward_expand(graph, out_bf16);
    ggml_build_forward_expand(graph, out_f16);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "explicit precision allocation");
    ggml_backend_tensor_set(input_storage, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(f32, weights.data(), 0, weights.size() * sizeof(float));
    ggml_backend_tensor_set(bf16, packed.data(), 0, packed.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(f16, half_packed.data(), 0, half_packed.size() * sizeof(ggml_fp16_t));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "explicit precision compute");
    std::vector<float> expected(rows * tokens * batches);
    for (int64_t b = 0; b < batches; ++b) for (int64_t t = 0; t < tokens; ++t) for (int64_t r = 0; r < rows; ++r) {
        double sum = 0.;
        for (int64_t k = 0; k < columns; ++k)
            sum += double(weights[k + columns * r]) * double(input_data[k + (columns + 8) * (t + tokens * b)]);
        expected[r + rows * (t + tokens * b)] = float(sum);
    }
    compare(out_f32, expected, "explicit F32 projection double oracle");
    compare(out_bf16, expected, "explicit BF16 projection double oracle");
    compare(out_f16, expected, "explicit F16 projection double oracle");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void check_explicit_quantized_precision(ggml_backend_t backend, ggml_type quant_type) {
    require(quant_type == GGML_TYPE_Q6_K || quant_type == GGML_TYPE_Q4_K, "quantized precision family");
    const std::string family = quant_type == GGML_TYPE_Q6_K ? "Q6" : "Q4";
    auto decode = [&](const void * data, float * out, int64_t n) {
        if (quant_type == GGML_TYPE_Q6_K) dequantize_row_q6_K(static_cast<const block_q6_K *>(data), out, n);
        else dequantize_row_q4_K(static_cast<const block_q4_K *>(data), out, n);
    };
    const bool default_control = std::getenv("QWEN4EXP_TEST_Q6_DEFAULT_CONTROL") != nullptr;
    for (const auto & shape : std::vector<std::pair<int64_t, int64_t>>{{9, 256}, {321, 2560}, {129, 10240}}) {
        const auto rows = shape.first, columns = shape.second;
        for (int64_t tokens : {1, 2}) {
            auto * ctx = ggml_init({1024 * 1024, nullptr, true});
            require(ctx != nullptr, "Q6 precision context");
            auto * weights = ggml_new_tensor_2d(ctx, quant_type, columns, rows);
            auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, tokens);
            auto * output = ggml_mul_mat(ctx, weights, input);
            auto * default_output = ggml_mul_mat(ctx, weights, input);
            ggml_mul_mat_set_prec(output, GGML_PREC_F32);
            auto * graph = ggml_new_graph(ctx);
            ggml_build_forward_expand(graph, output);
            ggml_build_forward_expand(graph, default_output);
            auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
            require(buffer != nullptr, "Q6 precision allocation");
            std::vector<float> values(columns * rows), input_values(columns * tokens);
            for (size_t i = 0; i < values.size(); ++i)
                values[i] = float(int((i * 19 + i / columns * 7) % 127) - 63) / 4096.f;
            for (size_t i = 0; i < input_values.size(); ++i)
                input_values[i] = float(int((i * 17 + i / columns * 3) % 113) - 56) / 91.f;
            std::vector<uint8_t> packed(ggml_nbytes(weights));
            require(ggml_quantize_chunk(quant_type, values.data(), packed.data(), 0,
                    rows, columns, nullptr, nullptr) == packed.size(), "Q6 precision quantization");
            std::vector<float> dequantized(values.size());
            decode(packed.data(),
                    dequantized.data(), dequantized.size());
            std::vector<float> expected(rows * tokens);
            for (int64_t t = 0; t < tokens; ++t) for (int64_t r = 0; r < rows; ++r) {
                double sum = 0.;
                for (int64_t k = 0; k < columns; ++k)
                    sum += double(dequantized[r * columns + k]) * double(input_values[t * columns + k]);
                expected[t * rows + r] = float(sum);
            }
            ggml_backend_tensor_set(weights, packed.data(), 0, packed.size());
            ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size() * sizeof(float));
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Q6 precision compute");
            std::vector<float> defaults(expected.size());
            ggml_backend_tensor_get(default_output, defaults.data(), 0, defaults.size() * sizeof(float));
            unsigned char digest[SHA256_DIGEST_SIZE];
            sha256_hash(digest, reinterpret_cast<const unsigned char *>(defaults.data()), defaults.size() * sizeof(float));
            std::string hex;
            for (unsigned char byte : digest) {
                hex += "0123456789abcdef"[byte >> 4]; hex += "0123456789abcdef"[byte & 15];
            }
            std::cout << "quantized_default: type=" << family << " rows=" << rows << " columns=" << columns << " tokens=" << tokens
                      << " sha256=" << hex << '\n';
            if (!default_control) compare(output, expected, ("explicit " + family + " F32 activation/double dot oracle").c_str());
            ggml_backend_buffer_free(buffer);
            ggml_free(ctx);
        }
    }
    if (default_control) return; // Only the explicit oracle is bypassed for old-library dispatch controls.
    // Padded weight rows and activation strides, with broadcasting along both
    // outer axes. Poison activation padding so accidental reads are visible.
    constexpr int64_t columns = 2560, rows = 7, weight_batches = 2, batches = 4, outer = 2;
    for (int64_t weight_outer : {int64_t(1), outer}) for (int64_t tokens : {1, 2, 8, 128}) {
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        require(ctx != nullptr, "Q6 layout context");
        auto * weight_storage = ggml_new_tensor_4d(ctx, quant_type, columns, rows + 3, weight_batches, weight_outer);
        auto * weights = ggml_view_4d(ctx, weight_storage, columns, rows, weight_batches, weight_outer,
                weight_storage->nb[1], weight_storage->nb[2], weight_storage->nb[3], 0);
        auto * input_storage = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, columns + 13, tokens, batches, outer);
        auto * input = ggml_view_4d(ctx, input_storage, columns, tokens, batches, outer,
                input_storage->nb[1], input_storage->nb[2], input_storage->nb[3], 0);
        auto * output = ggml_mul_mat(ctx, weights, input);
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
        const bool supported = ggml_backend_supports_op(backend, output);
        const bool unsupported_direct = !supported && std::getenv("QWEN4EXP_TEST_Q6_UNSUPPORTED_DIRECT");
        auto compute = supported || unsupported_direct ? backend : ggml_backend_cpu_init();
        require(compute != nullptr, "Q6 layout fallback backend");
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, compute);
        require(buffer != nullptr, "Q6 layout allocation");
        std::vector<float> raw(ggml_nelements(weight_storage)), input_values(ggml_nelements(input_storage), NAN);
        for (int64_t wz = 0; wz < weight_outer; ++wz)
        for (int64_t b = 0; b < weight_batches; ++b) for (int64_t r = 0; r < rows + 3; ++r)
            for (int64_t k = 0; k < columns; ++k)
                raw[((wz * weight_batches + b) * (rows + 3) + r) * columns + k] = r >= rows ? 64.f :
                        float(int((k * 19 + r * 7 + b * 31 + wz * 17) % 127) - 63) / 4096.f;
        for (int64_t z = 0; z < outer; ++z) for (int64_t b = 0; b < batches; ++b)
            for (int64_t t = 0; t < tokens; ++t) for (int64_t k = 0; k < columns; ++k)
                input_values[((z * batches + b) * tokens + t) * (columns + 13) + k] =
                        float(int((k * 17 + t * 3 + b * 11 + z * 23) % 113) - 56) / 91.f;
        std::vector<uint8_t> packed(ggml_nbytes(weight_storage));
        require(ggml_quantize_chunk(quant_type, raw.data(), packed.data(), 0,
                weight_outer * weight_batches * (rows + 3), columns, nullptr, nullptr) == packed.size(), "Q6 layout quantization");
        std::vector<float> decoded(raw.size()), expected(rows * tokens * batches * outer);
        decode(packed.data(), decoded.data(), decoded.size());
        for (int64_t z = 0; z < outer; ++z) for (int64_t b = 0; b < batches; ++b)
            for (int64_t t = 0; t < tokens; ++t) for (int64_t r = 0; r < rows; ++r) {
                double sum = 0;
                for (int64_t k = 0; k < columns; ++k)
                    sum += double(decoded[(((z / (outer / weight_outer)) * weight_batches +
                            b / (batches / weight_batches)) * (rows + 3) + r) * columns + k]) *
                           double(input_values[((z * batches + b) * tokens + t) * (columns + 13) + k]);
                expected[((z * batches + b) * tokens + t) * rows + r] = float(sum);
            }
        ggml_backend_tensor_set(weight_storage, packed.data(), 0, packed.size());
        ggml_backend_tensor_set(input_storage, input_values.data(), 0, input_values.size() * sizeof(float));
        if (unsupported_direct) {
            std::vector<float> canary(expected.size(), 12345.f);
            ggml_backend_tensor_set(output, canary.data(), 0, canary.size() * sizeof(float));
            std::cout << "q6_unsupported_direct: support_query=false\n" << std::flush;
            const auto status = ggml_backend_graph_compute(compute, graph);
            std::vector<float> observed(canary.size());
            ggml_backend_tensor_get(output, observed.data(), 0, observed.size() * sizeof(float));
            std::cout << "q6_unsupported_direct: status=" << int(status) << " output_unchanged="
                      << (observed == canary ? "true" : "false") << '\n';
            ggml_backend_buffer_free(buffer); ggml_free(ctx);
            return;
        }
        require(ggml_backend_graph_compute(compute, graph) == GGML_STATUS_SUCCESS, "Q6 layout compute");
        compare(output, expected, ("explicit " + family + " strided/broadcast F32 double oracle").c_str());
        std::cout << "quantized_layout: type=" << family << " tokens=" << tokens << " weight_outer=" << weight_outer
                  << " backend=" << (supported ? "requested" : "cpu_fallback") << " padding_poisoned=true\n";
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        if (compute != backend) ggml_backend_free(compute);
    }
}

using layer_weights = std::vector<std::pair<std::string, std::vector<float>>>;

void check_qsa_partial_rope(ggml_backend_t backend, const nlohmann::json & ref) {
    const int width = ref.at("width"), heads = ref.at("heads"), rotary = ref.at("rotary_dims");
    const auto positions = ref.at("positions").get<std::vector<int32_t>>();
    const auto data = ref.at("input").get<std::vector<float>>();
    require(width == 128 && rotary == 64 && heads == 4, "configured partial indexer rotary geometry");
    auto ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "QSA rotary context");
    auto input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, heads, positions.size());
    auto pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, positions.size());
    auto output = qwen4exp_build_qsa_rope(ctx, input, pos, rotary, 262144,
            ref.at("freq_base"), 1.f, 0.f, 1.f, 32.f, 1.f);
    auto wrong = qwen4exp_build_qsa_rope(ctx, input, pos, width, 262144,
            ref.at("freq_base"), 1.f, 0.f, 1.f, 32.f, 1.f);
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, wrong);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "QSA rotary allocation");
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "QSA rotary compute");
    compare_fixture(output, ref.at("expected"), "pinned partial indexer RoPE");
    std::vector<float> actual(data.size()), control(data.size());
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
    ggml_backend_tensor_get(wrong, control.data(), 0, control.size() * sizeof(float));
    float difference = 0.f;
    for (size_t i = 0; i < data.size(); ++i) {
        if (i % width >= size_t(rotary)) require(actual[i] == data[i], "QSA nonrotary features preserved");
        difference = std::max(difference, std::fabs(actual[i] - control[i]));
    }
    require(difference > 1e-3f, "QSA golden rejects full-head rotation");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

struct packed_layer_weight {
    std::string role;
    ggml_type type;
    std::vector<char> payload;
};
using packed_layer_weights = std::vector<packed_layer_weight>;

layer_weights read_real_layer_weights(const char * path, const nlohmann::json & records,
                                     packed_layer_weights * packed_original = nullptr) {
    ggml_context * metadata = nullptr;
    auto file = gguf_init_from_file(path, {true, &metadata});
    require(file != nullptr && metadata != nullptr, "real layer GGUF metadata");
    std::ifstream input(path, std::ios::binary);
    require(bool(input), "real layer GGUF input");
    layer_weights result;
    for (auto entry = records.begin(); entry != records.end(); ++entry) {
        const auto name = entry.value().at("gguf_name").get<std::string>();
        const int index = gguf_find_tensor(file, name.c_str());
        auto tensor = ggml_get_tensor(metadata, name.c_str());
        require(index >= 0 && tensor != nullptr, "real layer tensor " + name);
        const auto shape = entry.value().at("source_shape").get<std::vector<int64_t>>();
        require(shape.size() == size_t(ggml_n_dims(tensor)), "real layer rank " + name);
        for (size_t axis = 0; axis < shape.size(); ++axis) {
            require(shape[shape.size() - 1 - axis] == tensor->ne[axis], "real layer shape " + name);
        }
        require(ggml_nbytes(tensor) <= 32 * 1024 * 1024, "bounded real layer payload");
        std::vector<char> payload(ggml_nbytes(tensor));
        input.seekg(gguf_get_data_offset(file) + gguf_get_tensor_offset(file, index));
        input.read(payload.data(), payload.size());
        require(input.gcount() == std::streamsize(payload.size()), "real layer payload read " + name);
        unsigned char digest[SHA256_DIGEST_SIZE];
        sha256_hash(digest, reinterpret_cast<const unsigned char *>(payload.data()), payload.size());
        std::string hex;
        for (unsigned char byte : digest) {
            hex += "0123456789abcdef"[byte >> 4];
            hex += "0123456789abcdef"[byte & 15];
        }
        require(hex == entry.value().at("payload_sha256").get<std::string>(), "real layer payload SHA-256 " + name);
        std::vector<float> values(ggml_nelements(tensor));
        if (tensor->type == GGML_TYPE_F32) {
            std::memcpy(values.data(), payload.data(), payload.size());
        } else {
            const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
            for (int64_t row = 0; row < ggml_nrows(tensor); ++row) {
                const void * src = payload.data() + row * row_bytes;
                float * dst = values.data() + row * tensor->ne[0];
                // Typed calls avoid the legacy traits table's incompatible function-pointer casts.
                switch (tensor->type) {
                    case GGML_TYPE_Q6_K: dequantize_row_q6_K(static_cast<const block_q6_K *>(src), dst, tensor->ne[0]); break;
                    case GGML_TYPE_Q4_K: dequantize_row_q4_K(static_cast<const block_q4_K *>(src), dst, tensor->ne[0]); break;
                    case GGML_TYPE_BF16: ggml_bf16_to_fp32_row(static_cast<const ggml_bf16_t *>(src), dst, tensor->ne[0]); break;
                    default: require(false, "unsupported real layer native dequantizer " + name);
                }
            }
        }
        result.emplace_back(entry.key(), std::move(values));
        if (packed_original) packed_original->push_back({entry.key(), tensor->type, std::move(payload)});
    }
    gguf_free(file);
    ggml_free(metadata);
    return result;
}

void check_real_quantized_precision(ggml_backend_t backend, const nlohmann::json & fixture,
                             const packed_layer_weights & packed) {
    const int64_t tokens = fixture.at("geometry").at("tokens");
    int projections = 0;
    for (const auto & item : packed) {
        const auto & role = item.role;
        if (role != "in_proj_qkv" && role != "in_proj_z" && role != "out_proj") continue;
        require(item.type == GGML_TYPE_Q6_K || item.type == GGML_TYPE_Q4_K, "real quantized projection type");
        ++projections;
        const auto shape = fixture.at("tensors").at(role).at("source_shape").get<std::vector<int64_t>>();
        require(shape.size() == 2, "real Q6 matrix rank");
        const int64_t rows = shape[0], columns = shape[1];
        const auto inputs = (role != "out_proj" ? fixture.at("input") :
                fixture.at("expected").at("gated_output")).get<std::vector<float>>();
        const auto expected = fixture.at("expected").at(role == "out_proj" ? "output" : role).get<std::vector<float>>();
        require(inputs.size() == size_t(columns * tokens) && expected.size() == size_t(rows * tokens),
                "real Q6 reference geometry");
        auto * ctx = ggml_init({1024 * 1024, nullptr, true});
        require(ctx != nullptr, "real Q6 context");
        auto * weight = ggml_new_tensor_2d(ctx, item.type, columns, rows);
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, columns, tokens);
        auto * output = ggml_mul_mat(ctx, weight, input);
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        require(ggml_backend_supports_op(backend, output), "real Q6 backend support");
        require(item.payload.size() == ggml_nbytes(weight), "original quantized byte count");
        auto * graph = ggml_new_graph(ctx); ggml_build_forward_expand(graph, output);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "real Q6 allocation");
        ggml_backend_tensor_set(weight, item.payload.data(), 0, item.payload.size());
        ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "real Q6 compute");
        compare(output, expected, ("original packed " + std::string(ggml_type_name(item.type)) + " " + role).c_str());
        std::cout << "real_quantized: type=" << ggml_type_name(item.type) << " role=" << role << " rows=" << rows << " columns=" << columns
                  << " tokens=" << tokens << " original_payload=true\n";
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    require(projections == 3, "three original quantized layer projections");
}

void check_real_alpha_precision(ggml_backend_t backend, const nlohmann::json & fixture,
                                const layer_weights & real_weights) {
    const int64_t hidden = fixture.at("geometry").at("hidden");
    const int64_t heads = fixture.at("geometry").at("heads");
    const int64_t tokens = fixture.at("geometry").at("tokens");
    require(fixture.at("tensors").at("in_proj_a").at("gguf_type") == "BF16", "real alpha storage contract");
    auto values = [&](const char * role) -> const std::vector<float> & {
        for (const auto & item : real_weights) if (item.first == role) return item.second;
        throw std::runtime_error(std::string("missing real alpha parameter ") + role);
    };
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "real alpha context");
    auto * f32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, heads);
    auto * bf16 = ggml_new_tensor_2d(ctx, GGML_TYPE_BF16, hidden, heads);
    auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    auto * bias = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * decay = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, heads);
    auto * alpha_f32 = ggml_mul_mat(ctx, f32, input);
    auto * alpha_bf16 = ggml_mul_mat(ctx, bf16, input);
    ggml_mul_mat_set_prec(alpha_f32, GGML_PREC_F32);
    ggml_mul_mat_set_prec(alpha_bf16, GGML_PREC_F32);
    // Diagnostic negative control: the historical F32 accumulation path.
    if (std::getenv("QWEN4EXP_TEST_ALPHA_DEFAULT")) ggml_mul_mat_set_prec(alpha_f32, GGML_PREC_DEFAULT);
    auto gate = [&](ggml_tensor * alpha) {
        return ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, alpha, bias)), decay);
    };
    auto * gate_f32 = gate(alpha_f32);
    auto * gate_bf16 = gate(alpha_bf16);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, gate_f32);
    ggml_build_forward_expand(graph, gate_bf16);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "real alpha allocation");
    const auto & weights = values("in_proj_a");
    require(weights.size() == size_t(hidden * heads), "real alpha weights");
    std::vector<ggml_bf16_t> packed(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        packed[i] = ggml_fp32_to_bf16(weights[i]);
        const float restored = ggml_bf16_to_fp32(packed[i]);
        require(std::memcmp(&restored, &weights[i], sizeof(float)) == 0, "real alpha BF16 roundtrip");
    }
    const auto inputs = fixture.at("input").get<std::vector<float>>();
    require(inputs.size() == size_t(hidden * tokens), "real alpha input");
    ggml_backend_tensor_set(f32, weights.data(), 0, weights.size() * sizeof(float));
    ggml_backend_tensor_set(bf16, packed.data(), 0, packed.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(float));
    ggml_backend_tensor_set(bias, values("dt_bias").data(), 0, heads * sizeof(float));
    ggml_backend_tensor_set(decay, values("a_log").data(), 0, heads * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "real alpha compute");
    const auto expected_alpha = fixture.at("expected").at("in_proj_a").get<std::vector<float>>();
    const auto expected_gate = fixture.at("expected").at("g").get<std::vector<float>>();
    require(expected_alpha.size() == size_t(heads * tokens) && expected_gate.size() == expected_alpha.size(),
            "real alpha reference shape");
    compare(alpha_bf16, expected_alpha, "real BF16 alpha projection");
    compare(gate_bf16, expected_gate, "real BF16 recurrent gate");
    compare(alpha_f32, expected_alpha, "real F32 alpha projection");
    compare(gate_f32, expected_gate, "real F32 recurrent gate");
    std::cout << "real_alpha: tokens=" << tokens << " heads=" << heads
              << " BF16_roundtrip_exact=true F32_and_BF16_gate_pass=true\n";
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void check_recurrent_layer(ggml_backend_t backend, const nlohmann::json & ref,
                           const layer_weights * real_weights = nullptr,
                           const packed_layer_weights * original = nullptr) {
    require(original == nullptr || real_weights != nullptr, "packed recurrent layer requires real weights");
    const int64_t hidden = ref.at("hidden"), width = ref.at("width");
    const int64_t kh = ref.at("key_heads"), heads = ref.at("heads");
    const int64_t tokens = ref.at("tokens"), kernel = ref.at("kernel");
    const int64_t channels = width * (2 * kh + heads), state_size = width * width * heads;
    const float epsilon = ref.at("epsilon");
    require((real_weights == nullptr && hidden == 8 && width == 128 && kh == 2 && heads == 6 && tokens == 5 && kernel == 4) ||
            (real_weights != nullptr && hidden == 2560 && width == 128 && kh == 16 && heads == 48 && tokens >= 1 && tokens <= 8 && kernel == 4),
            "complete recurrent layer geometry");
    auto real_data = [&](const char * name) -> const std::vector<float> & {
        for (const auto & entry : *real_weights) if (entry.first == name) return entry.second;
        require(false, std::string("missing real layer weight ") + name);
        std::abort();
    };
    const auto input_data = ref.at("input").get<std::vector<float>>();
    std::vector<float> prior_conv(channels * (kernel - 1), 0.0f), prior_state(state_size, 0.0f);
    auto run = [&](int64_t start, int64_t count, const nlohmann::json & expected) {
        ggml_init_params params = { 4 * 1024 * 1024, nullptr, true };
        auto ctx = ggml_init(params);
        require(ctx != nullptr, "layer context");
        std::vector<std::pair<ggml_tensor *, std::vector<float>>> inputs;
        std::vector<std::pair<ggml_tensor *, const std::vector<char> *>> byte_inputs;
        auto tensor = [&](int64_t rows, int64_t cols, std::vector<float> values) {
            auto result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rows, cols);
            require(values.size() == size_t(rows * cols), "layer tensor data size");
            inputs.emplace_back(result, std::move(values));
            return result;
        };
        nlohmann::json sliced = nlohmann::json::array();
        for (int64_t i = 0; i < hidden * count; ++i) sliced.push_back(input_data.at(start * hidden + i));
        auto input = tensor(hidden, count, sliced);
        auto weight = [&](const char * name) {
            return real_weights != nullptr ? real_data(name) : ref.at("weights").at(name).get<std::vector<float>>();
        };
        auto parameter = [&](const char * name, const char * role) {
            return real_weights != nullptr ? real_data(role) : ref.at(name).get<std::vector<float>>();
        };
        auto projection_weight = [&](const char * name, int64_t columns, int64_t rows) {
            if (original == nullptr) return tensor(columns, rows, weight(name));
            for (const auto & item : *original) if (item.role == name) {
                auto * result = ggml_new_tensor_2d(ctx, item.type, columns, rows);
                require(ggml_nbytes(result) == item.payload.size(), "packed projection byte geometry");
                byte_inputs.emplace_back(result, &item.payload);
                return result;
            }
            throw std::runtime_error(std::string("missing original projection ") + name);
        };
        auto project = [&](const char * name, int64_t outputs) {
            auto result = ggml_mul_mat(ctx, projection_weight(name, hidden, outputs), input);
            // Both weight representations preserve F32 activations under the
            // explicit precision contract; compare with the same pinned oracle.
            ggml_mul_mat_set_prec(result, GGML_PREC_F32);
            return result;
        };
        auto mixed = project("in_proj_qkv", channels);
        auto z = project("in_proj_z", width * heads);
        auto alpha = project("in_proj_a", heads);
        ggml_mul_mat_set_prec(alpha, GGML_PREC_F32);
        auto beta = project("in_proj_b", heads);
        auto dt = tensor(heads, 1, parameter("dt_bias", "dt_bias"));
        std::vector<float> a = parameter("a_log", "a_log");
        if (real_weights == nullptr) {
            for (float & value : a) value = -std::exp(value); // GGUF stores -exp(A_log)
        }
        auto alpha_biased = ggml_add(ctx, alpha, dt);
        auto alpha_softplus = ggml_softplus(ctx, alpha_biased);
        auto gate = ggml_mul(ctx, alpha_softplus, tensor(heads, 1, a));
        auto conv_weight = tensor(kernel, channels, parameter("conv_weight", "conv1d"));
        auto conv_state = ggml_reshape_3d(ctx, tensor(kernel - 1, channels, prior_conv), kernel - 1, channels, 1);
        auto seq = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, count);
        auto conv_raw = ggml_ssm_conv(ctx, conv_state, mixed, conv_weight, seq, nullptr);
        auto conv = ggml_silu(ctx, ggml_view_2d(ctx, conv_raw, channels, count, channels * sizeof(float), 0));
        auto next_conv = ggml_cont(ctx, ggml_view_2d(ctx, conv_raw, kernel - 1, channels,
                kernel * sizeof(float), (channels * count + 1) * sizeof(float)));
        auto split = [&](int64_t h, int64_t offset) {
            return ggml_view_4d(ctx, conv, width, h, count, 1, width * sizeof(float),
                               channels * sizeof(float), channels * count * sizeof(float), offset * sizeof(float));
        };
        auto q = split(kh, 0), k = split(kh, width * kh), v = split(heads, 2 * width * kh);
        auto q_norm = qwen4exp_build_l2_norm(ctx, ggml_permute(ctx, q, 0, 2, 1, 3), 1e-6f);
        auto k_norm = qwen4exp_build_l2_norm(ctx, ggml_permute(ctx, k, 0, 2, 1, 3), 1e-6f);
        auto state = ggml_reshape_4d(ctx, tensor(width, width * heads, prior_state), width, width, heads, 1);
        // Use the production adapter, not a duplicate op/permutation implementation.
        const llm_build_cb cb = [](ggml_tensor *, const char *, int) {};
        const int repeat_type = ref.value("repeat_type", 0);
        if (ref.contains("head_order")) {
            require(repeat_type == delta_net::qwen4exp_repeat_type, "production GGUF head order");
        }
        auto recurrence = delta_net::build_fused_delta_net(ctx, q_norm, k_norm, v,
            ggml_reshape_3d(ctx, gate, heads, count, 1), ggml_reshape_4d(ctx, beta, heads, 1, count, 1),
            state, 0, cb, repeat_type);
        auto core = recurrence.first;
        auto norm = ggml_mul(ctx, ggml_rms_norm(ctx, ggml_reshape_2d(ctx, core, width, heads * count), epsilon),
                             tensor(width, 1, parameter("norm_weight", "norm_weight")));
        require(ref.at("output_gate_type").get<std::string>() == "sigmoid", "configured recurrent output gate");
        auto gated = qwen4exp_build_recurrent_output_gate(ctx, norm, ggml_reshape_2d(ctx, z, width, heads * count));
        auto output = ggml_mul_mat(ctx, projection_weight("out_proj", width * heads, hidden),
                                   ggml_reshape_2d(ctx, gated, width * heads, count));
        if (original != nullptr) require(byte_inputs.size() == 5, "all five original projection tensors used");
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        auto graph = ggml_new_graph_custom(ctx, 256, false);
        ggml_build_forward_expand(graph, output);
        ggml_build_forward_expand(graph, next_conv);
        auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "layer allocation");
        for (const auto & entry : inputs) ggml_backend_tensor_set(entry.first, entry.second.data(), 0, entry.second.size() * sizeof(float));
        for (const auto & entry : byte_inputs)
            ggml_backend_tensor_set(entry.first, entry.second->data(), 0, entry.second->size());
        std::vector<int32_t> seq_ids(count, 0);
        ggml_backend_tensor_set(seq, seq_ids.data(), 0, seq_ids.size() * sizeof(int32_t));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "layer compute");
        if (real_weights != nullptr && std::getenv("QWEN4EXP_TEST_GATE_DIAGNOSTICS") != nullptr) {
            nlohmann::json diagnostic = {{"start", start}, {"count", count}, {"original_packed", original != nullptr}};
            for (const auto & boundary : std::vector<std::pair<ggml_tensor *, const char *>> {
                    {alpha, "alpha"}, {alpha_biased, "alpha_biased"},
                    {alpha_softplus, "softplus"}, {gate, "g"}}) {
                std::vector<float> data(ggml_nelements(boundary.first));
                ggml_backend_tensor_get(boundary.first, data.data(), 0, data.size() * sizeof(float));
                diagnostic[boundary.second] = data;
            }
            std::cout << "qwen4exp_gate_diagnostic: " << diagnostic.dump() << '\n';
        }
        for (const auto & boundary : std::vector<std::pair<ggml_tensor *, const char *>> {
                {mixed, "in_proj_qkv"}, {z, "in_proj_z"}, {alpha, "in_proj_a"}, {beta, "in_proj_b"},
                {gate, "g"}, {conv, "convolved"}, {core, "core_output"},
                {gated, "gated_output"}, {output, "output"}, {next_conv, "conv_state"}}) {
            compare_fixture(boundary.first, expected.at(boundary.second), boundary.second);
        }
        std::vector<float> expected_state(state_size);
        for (int64_t h = 0; h < heads; ++h)
            for (int64_t key = 0; key < width; ++key)
                for (int64_t value = 0; value < width; ++value)
                    expected_state[key + width * value + width * width * h] =
                        expected.at("state").at(value + width * key + width * width * h).get<float>();
        compare(recurrence.second, expected_state, "complete layer recurrent state");
        ggml_backend_tensor_get(next_conv, prior_conv.data(), 0, prior_conv.size() * sizeof(float));
        ggml_backend_tensor_get(recurrence.second, prior_state.data(), 0, prior_state.size() * sizeof(float));
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
    };
    if (real_weights != nullptr) std::cout << "Checking real layer batch\n";
    run(0, tokens, ref.at("batched"));
    std::fill(prior_conv.begin(), prior_conv.end(), 0.0f);
    std::fill(prior_state.begin(), prior_state.end(), 0.0f);
    // Schema 1 remains a batch-only compatibility fixture; schema 2 requires steps.
    if (real_weights == nullptr || ref.contains("steps")) {
        require(ref.at("steps").size() == size_t(tokens), "layer decode trace count");
        for (int64_t token = 0; token < tokens; ++token) {
            if (real_weights != nullptr) std::cout << "Checking real layer decode token " << token << '\n';
            run(token, 1, ref.at("steps").at(token));
        }
    }
}

void check_l2(ggml_backend_t backend, const nlohmann::json & ref) {
    ggml_init_params params = { 1024 * 1024, nullptr, true };
    auto ctx = ggml_init(params);
    require(ctx != nullptr, "L2 context");
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ref.at("width"), ref.at("rows"));
    auto output = qwen4exp_build_l2_norm(ctx, input, ref.at("epsilon"));
    auto legacy = ggml_l2_norm(ctx, input, ref.at("epsilon"));
    auto graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    ggml_build_forward_expand(graph, legacy);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "L2 allocation");
    const auto data = ref.at("input").get<std::vector<float>>();
    require(data.size() == size_t(ggml_nelements(input)), "L2 input size");
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "L2 compute");
    compare_fixture(output, ref.at("output"), "pinned Transformers zero/tiny/ordinary L2");
    std::vector<float> legacy_values(data.size());
    ggml_backend_tensor_get(legacy, legacy_values.data(), 0, legacy_values.size() * sizeof(float));
    float legacy_error = 0.0f;
    for (size_t i = 0; i < legacy_values.size(); ++i) {
        legacy_error = std::max(legacy_error, std::fabs(legacy_values[i] - ref.at("output").at(i).get<float>()));
    }
    require(legacy_error > 0.1f, "tiny-vector golden must reject the old clamp-based L2");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

void check_routing_and_head(ggml_backend_t backend, const nlohmann::json & fixture) {
    // These are the same GGML primitives used by llm_build_moe_ffn/build_output.
    // Isolate the numerical boundaries without opening the architecture loader gate.
    const auto & moe = fixture.at("moe");
    const int64_t hidden = moe.at("hidden");
    const int64_t experts = moe.at("num_experts");
    const int64_t top = moe.at("top_k");
    const int64_t tokens = moe.at("tokens");
    require(hidden > 0 && experts > 0 && top > 0 && top <= experts && tokens > 0,
            "routing fixture geometry");
    require(moe.at("norm_topk_prob").get<bool>(), "routing fixture normalized top-k");

    const auto & head = fixture.at("logits");
    const int64_t head_hidden = head.at("hidden");
    const int64_t vocab = head.at("vocab");
    const int64_t head_tokens = head.at("tokens");
    require(head_hidden > 0 && vocab > 0 && head_tokens > 0, "head fixture geometry");
    require(!head.at("final_norm_applied").get<bool>(), "head fixture has no extra norm");

    ggml_init_params params = { 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "routing/head context");
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, tokens);
    auto router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden, experts);
    auto router_logits = ggml_mul_mat(ctx, router, input);
    auto probabilities = ggml_soft_max(ctx, router_logits);
    // top_k is a strided view of a full expert-width argsort. Materialize it
    // before comparing flat bytes; production get_rows honors those strides.
    auto ids = ggml_cont(ctx, ggml_top_k(ctx, probabilities, top));
    auto selected = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probabilities, 1, experts, tokens), ids);
    auto selected_rows = ggml_reshape_2d(ctx, selected, top, tokens);
    auto normalized = ggml_div(ctx, selected_rows, ggml_sum_rows(ctx, selected_rows));

    auto head_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_hidden, head_tokens);
    auto head_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, head_hidden, vocab);
    auto logits = ggml_mul_mat(ctx, head_weight, head_input);
    auto graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, router_logits);
    ggml_build_forward_expand(graph, ids);
    ggml_build_forward_expand(graph, normalized);
    ggml_build_forward_expand(graph, logits);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "routing/head allocation");

    const auto put = [](ggml_tensor * tensor, const std::vector<float> & data) {
        require(data.size() == size_t(ggml_nelements(tensor)), "routing/head input size");
        ggml_backend_tensor_set(tensor, data.data(), 0, data.size() * sizeof(float));
    };
    put(input, moe.at("input").get<std::vector<float>>());
    put(router, values(hidden * experts, 0.6f, 0.15f));
    put(head_input, head.at("hidden_input").get<std::vector<float>>());
    put(head_weight, head.at("lm_head_weight").get<std::vector<float>>());
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            "routing/head graph compute");
    compare_fixture(router_logits, moe.at("router_logits"), "pinned Transformers router logits");
    std::vector<int32_t> expected_ids;
    std::vector<float> expected_weights;
    require(moe.at("expert_ids").size() == size_t(tokens) &&
            moe.at("routing_weights").size() == size_t(tokens), "routing fixture token count");
    for (int64_t token = 0; token < tokens; ++token) {
        const auto & row_ids = moe.at("expert_ids").at(token);
        const auto & row_weights = moe.at("routing_weights").at(token);
        require(row_ids.size() == size_t(top) && row_weights.size() == size_t(top),
                "routing fixture top-k width");
        for (int64_t rank = 0; rank < top; ++rank) {
            expected_ids.push_back(row_ids.at(rank).get<int32_t>());
            expected_weights.push_back(row_weights.at(rank).get<float>());
        }
    }
    compare_i32(ids, expected_ids, "pinned Transformers expert IDs");
    compare(normalized, expected_weights, "pinned Transformers normalized routing weights");
    compare_fixture(logits, head.at("logits"), "pinned Transformers output-head logits");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

} // namespace

int main(int argc, char ** argv) {
    const bool real_layer = argc == 4 && std::string(argv[1]) == "--real-layer0";
    if (argc != 2 && !real_layer) {
        std::cerr << "usage: " << argv[0] << " UPSTREAM_OPERATOR_FIXTURE.json\n"
                  << "       " << argv[0] << " --real-layer0 REAL_LAYER_FIXTURE.json SHARD1.gguf\n";
        return 1;
    }
    std::ifstream fixture_input(real_layer ? argv[2] : argv[1]);
    if (!fixture_input) {
        std::cerr << "failed to open pinned upstream operator fixture: " << argv[1] << '\n';
        return 1;
    }
    const auto upstream_fixture = nlohmann::json::parse(fixture_input);
    if (upstream_fixture.at("architecture") != "qwen4exp" ||
        upstream_fixture.at("reference").at("revision") != "fc5c5bde8e656dad91cbf34e61940d984b1c7b91" ||
        upstream_fixture.at("reference").at("modeling_sha256") !=
            "77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459") {
        std::cerr << "operator fixture is not from the pinned Transformers reference\n";
        return 1;
    }
    if (real_layer) {
        const int schema = upstream_fixture.at("schema_version");
        require((schema == 1 || schema == 2) && upstream_fixture.at("layer") == 0 &&
                upstream_fixture.at("expected_layout") == "cyclic_gguf" && upstream_fixture.at("repeat_type") == 1,
                "real layer fixture contract");
        auto ref = upstream_fixture.at("geometry");
        ref["input"] = upstream_fixture.at("input");
        ref["batched"] = upstream_fixture.at("expected");
        if (schema == 2) {
            require(upstream_fixture.at("steps").is_array(), "real layer decode traces");
            ref["steps"] = upstream_fixture.at("steps");
        }
        ref["head_order"] = upstream_fixture.at("head_order");
        ref["repeat_type"] = upstream_fixture.at("repeat_type");
        ref["output_gate_type"] = upstream_fixture.at("output_gate_type");
        packed_layer_weights packed_original;
        auto weights = read_real_layer_weights(argv[3], upstream_fixture.at("tensors"), &packed_original);
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        if (const char * text = std::getenv("QWEN4EXP_TEST_CUDA_DEVICE")) {
            const int device = std::stoi(text);
            require(device >= 0 && device < ggml_backend_cuda_get_device_count(), "real layer CUDA device");
            backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        }
#endif
        if (backend == nullptr) backend = ggml_backend_cpu_init();
        require(backend != nullptr, "real layer backend");
        check_softplus_precision(backend);
        check_explicit_projection_precision(backend);
        check_explicit_projection_precision(backend, 4, 10240);
        check_explicit_projection_precision(backend, 128, 10240);
        check_explicit_projection_precision(backend, 9, 2579);
        // More than 256 output rows bypass the small explicit projection kernel
        // and exercise the cuBLAS path, including on TF32-capable GPUs.
        check_explicit_projection_precision(backend, 320, 2560);
        check_explicit_quantized_precision(backend, GGML_TYPE_Q6_K);
        check_explicit_quantized_precision(backend, GGML_TYPE_Q4_K);
        check_real_alpha_precision(backend, upstream_fixture, weights);
        check_real_quantized_precision(backend, upstream_fixture, packed_original);
        check_recurrent_layer(backend, ref, &weights);
        std::cout << "Checking recurrent layer with original packed projections\n";
        check_recurrent_layer(backend, ref, &weights, &packed_original);
        std::cout << "packed_recurrent: batch_and_cached_decode=true original_projections=5\n";
        ggml_backend_free(backend);
        std::cout << "PASS: real GGUF layer-0 recurrent " << (schema == 2 ? "batch and cached decode" : "batch")
                  << " match pinned Transformers on the fixture input\n";
        return 0;
    }
    const auto & fixture = upstream_fixture.at("fixture");
    require(fixture.at("hidden").get<int64_t>() == H &&
            fixture.at("streams").get<int64_t>() == S &&
            fixture.at("low_rank").get<int64_t>() == R &&
            fixture.at("tokens").get<int64_t>() == T &&
            fixture.at("embedding").get<int64_t>() == E &&
            fixture.at("conv_kernel").get<int64_t>() == K &&
            fixture.at("conv_dilation").get<int64_t>() == D,
            "pinned operator fixture geometry");
    const auto & upstream = upstream_fixture.at("outputs");

    constexpr int64_t HS = H * S;
    const auto input_data = values(HS * T, 0.7f, 0.2f);
    auto norm_data = values(HS, 0.15f, -0.1f);
    for (float & value : norm_data) value += 1.0f;
    const auto down_data = values(HS * R, 0.11f, 0.5f);
    const auto up_data = values(R * HS, 0.13f, -0.3f);
    const auto inject_data = values(HS * S, 0.09f, 0.7f);
    const auto expected = reference(input_data, norm_data, down_data, up_data, inject_data);
    const auto expected_final_mixed = reference(expected.residual, norm_data, down_data, up_data, inject_data).mixed;
    const auto embeddings_data = values(E * T, 0.4f, -0.6f);
    const auto ple_key_data = values(E * HS, 0.12f, 0.9f);
    const auto ple_value_data = values(E * H, 0.14f, -0.8f);
    auto ple_key_norm_data = values(HS, 0.08f, 0.1f);
    auto ple_query_norm_data = values(HS, 0.07f, 0.3f);
    auto ple_conv_norm_data = values(HS, 0.06f, -0.4f);
    for (float & value : ple_key_norm_data) value += 1.0f;
    for (float & value : ple_query_norm_data) value += 1.0f;
    for (float & value : ple_conv_norm_data) value += 1.0f;
    const auto expected_ple = ple_reference(input_data, embeddings_data, ple_key_data, ple_value_data,
            ple_key_norm_data, ple_query_norm_data, ple_conv_norm_data);
    const auto ple_conv_data = values(K * HS, 0.1f, 0.25f);
    const auto ple_state_data = values(HS * (K - 1) * D, 0.3f, -0.45f);
    const auto expected_conv = ple_conv_reference(expected_ple.second, ple_conv_data, ple_state_data);
    const auto expected_conv_zero = ple_conv_reference(
            expected_ple.second, ple_conv_data, std::vector<float>(HS * (K - 1) * D));
    std::vector<float> expected_initial_streams(HS * T);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t s = 0; s < S; ++s) {
            std::copy_n(input_data.begin() + HS * t, H,
                    expected_initial_streams.begin() + H * s + HS * t);
        }
    }
    std::vector<float> expected_ple_output(expected_ple.first.size());
    for (size_t i = 0; i < expected_ple_output.size(); ++i) {
        expected_ple_output[i] = expected_ple.first[i] + expected_conv.first[i];
    }
    constexpr int64_t PLE_BASE_STATE = 7;
    constexpr int64_t PLE_STATE_WIDTH = PLE_BASE_STATE + HS * (K - 1) * D;
    auto ple_storage_data = values(PLE_STATE_WIDTH * 2, 0.21f, 0.73f);
    std::vector<float> expected_ple_prior(HS * (K - 1) * D);
    std::copy_n(ple_storage_data.begin() + PLE_STATE_WIDTH + PLE_BASE_STATE,
            expected_ple_prior.size(), expected_ple_prior.begin());
    auto expected_ple_storage = ple_storage_data;
    std::copy(expected_conv.second.begin(), expected_conv.second.end(),
            expected_ple_storage.begin() + PLE_STATE_WIDTH + PLE_BASE_STATE);

    constexpr int64_t QD = 3;
    constexpr int64_t QH = 2;
    constexpr int64_t QB = 4;
    constexpr int64_t QQ = 2;
    constexpr int64_t QTOP = 2;
    constexpr int64_t QRATIO = 2;
    auto qsa_query_data = values(QD * QH * QQ, 0.5f, 0.33f);
    const auto qsa_raw_key_data = values(QD * (QB * QRATIO + 1), 0.45f, -0.27f);
    auto qsa_key_norm_data = values(QD, 0.09f, 0.41f);
    for (float & value : qsa_key_norm_data) value += 1.0f;
    {
        // The pinned reference normalizes each indexer head (gamma 1) before
        // scoring, so feed this op the same post-q_layernorm queries.
        for (int64_t query_index = 0; query_index < QQ; ++query_index) {
            for (int64_t head = 0; head < QH; ++head) {
                float mean_square = 0.0f;
                for (int64_t dim = 0; dim < QD; ++dim) {
                    const float x = qsa_query_data[dim + QD * head + QD * QH * query_index];
                    mean_square += x * x;
                }
                const float inv_rms = 1.0f / std::sqrt(mean_square / float(QD) + EPS);
                for (int64_t dim = 0; dim < QD; ++dim) {
                    qsa_query_data[dim + QD * head + QD * QH * query_index] *= inv_rms;
                }
            }
        }
    }
    std::vector<float> qsa_key_data(QD * QB);
    for (int64_t block = 0; block < QB; ++block) {
        float mean_square = 0.0f;
        for (int64_t dim = 0; dim < QD; ++dim) {
            float pooled = 0.0f;
            for (int64_t offset = 0; offset < QRATIO; ++offset) {
                pooled += qsa_raw_key_data[dim + QD * (block * QRATIO + offset)];
            }
            pooled /= float(QRATIO);
            qsa_key_data[dim + QD * block] = pooled;
            mean_square += pooled * pooled;
        }
        const float inv_rms = 1.0f / std::sqrt(mean_square / float(QD) + EPS);
        for (int64_t dim = 0; dim < QD; ++dim) {
            qsa_key_data[dim + QD * block] *= inv_rms * qsa_key_norm_data[dim];
        }
    }
    // Complete-block visibility matching the pinned reference scenario:
    // query 0 (position 4) sees tokens 0..4, so blocks 2 and 3 are incomplete;
    // query 1 (position 6) sees tokens 0..6, so block 3 is incomplete.
    std::vector<float> qsa_mask_data(QB * QQ, 0.0f);
    qsa_mask_data[2] = -1.0e30f;
    qsa_mask_data[3] = -1.0e30f;
    qsa_mask_data[3 + QB] = -1.0e30f;
    std::vector<float> expected_qsa_scores(QB * QQ);
    std::vector<int32_t> expected_qsa_top(QTOP * QQ);
    for (int64_t query_index = 0; query_index < QQ; ++query_index) {
        std::vector<std::pair<float, int32_t>> ranked;
        for (int64_t block = 0; block < QB; ++block) {
            float score = 0.0f;
            for (int64_t head = 0; head < QH; ++head) {
                float dot = 0.0f;
                for (int64_t dim = 0; dim < QD; ++dim) {
                    dot += qsa_query_data[dim + QD * head + QD * QH * query_index] *
                        qsa_key_data[dim + QD * block];
                }
                score += std::max(dot, 0.0f);
            }
            score = score / std::sqrt(float(QD)) + qsa_mask_data[block + QB * query_index];
            expected_qsa_scores[block + QB * query_index] = score;
            ranked.emplace_back(score, int32_t(block));
        }
        std::stable_sort(ranked.begin(), ranked.end(), [](const auto & left, const auto & right) {
            return left.first > right.first;
        });
        for (int64_t rank = 0; rank < QTOP; ++rank) {
            expected_qsa_top[rank + QTOP * query_index] = ranked[rank].second;
        }
    }
    std::vector<int32_t> qsa_block_token_map(QRATIO * QB);
    for (int64_t block = 0; block < QB; ++block) {
        for (int64_t offset = 0; offset < QRATIO; ++offset) {
            qsa_block_token_map[offset + QRATIO * block] = int32_t(block * QRATIO + offset);
        }
    }
    const std::vector<int32_t> qsa_tail_data = {4, 6};
    std::vector<int32_t> expected_qsa_tokens((QRATIO * QTOP + 1) * QQ);
    for (int64_t query_index = 0; query_index < QQ; ++query_index) {
        for (int64_t rank = 0; rank < QTOP; ++rank) {
            const int32_t block = expected_qsa_top[rank + QTOP * query_index];
            for (int64_t offset = 0; offset < QRATIO; ++offset) {
                expected_qsa_tokens[offset + QRATIO * rank + (QRATIO * QTOP + 1) * query_index] =
                    block * QRATIO + int32_t(offset);
            }
        }
        expected_qsa_tokens[QRATIO * QTOP + (QRATIO * QTOP + 1) * query_index] =
            qsa_tail_data[query_index];
    }
    constexpr int64_t QKV = 8;
    std::vector<float> qsa_causal_data(QKV * QQ, -1.0e30f);
    std::vector<float> expected_qsa_mask(QKV * QQ, -INFINITY);
    for (int64_t query_index = 0; query_index < QQ; ++query_index) {
        const int64_t visible = qsa_tail_data[query_index] + 1;
        for (int64_t token = 0; token < visible; ++token) qsa_causal_data[token + QKV * query_index] = 0.0f;
        for (int64_t rank = 0; rank < QRATIO * QTOP + 1; ++rank) {
            const int32_t token = expected_qsa_tokens[rank + (QRATIO * QTOP + 1) * query_index];
            expected_qsa_mask[token + QKV * query_index] = qsa_causal_data[token + QKV * query_index];
        }
    }
    constexpr int64_t QCACHE = 7;
    constexpr int64_t QCACHE_HEAD = 3;
    constexpr int64_t QCACHE_BATCH = 2;
    constexpr int64_t QCACHE_USED = QCACHE_HEAD + QCACHE_BATCH;
    auto qsa_cache_initial = values(QD * QCACHE, 0.31f, -0.17f);
    const auto qsa_cache_batch_data = values(QD * QCACHE_BATCH, 0.52f, 0.61f);
    auto expected_qsa_cached = std::vector<float>(qsa_cache_initial.begin(),
            qsa_cache_initial.begin() + QD * QCACHE_USED);
    std::copy(qsa_cache_batch_data.begin(), qsa_cache_batch_data.end(),
            expected_qsa_cached.begin() + QD * QCACHE_HEAD);
    const std::vector<float> qsa_union_blocks = {
        0.0f, -INFINITY, 0.0f, -INFINITY,
        -INFINITY, 0.0f, -INFINITY, 0.0f,
    };
    const std::vector<float> qsa_union_tails = {
        -INFINITY, 0.0f, -INFINITY, -INFINITY,
        0.0f, -INFINITY, -INFINITY, -INFINITY,
    };
    const std::vector<float> qsa_union_enabled = {0.0f, 1.0f};
    const std::vector<float> expected_qsa_union = {
        0.0f, -INFINITY, 0.0f, -INFINITY,
        0.0f, 0.0f, -INFINITY, 0.0f,
    };

    ggml_init_params params = { 2 * 1024 * 1024, nullptr, true };
    ggml_context * ctx = ggml_init(params);
    require(ctx != nullptr, "ggml context");
    auto input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HS, T);
    auto norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HS);
    auto down = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HS, R);
    auto up = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, R, HS);
    auto inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HS, S);
    auto embeddings = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, T);
    auto ple_key = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, HS);
    auto ple_value = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, E, H);
    auto ple_key_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HS);
    auto ple_query_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HS);
    auto ple_conv_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, HS);
    auto ple_conv_weight = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, K, HS);
    auto ple_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HS, (K - 1) * D);
    auto ple_state_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, PLE_STATE_WIDTH, 2);
    auto token_embeddings = ggml_view_2d(ctx, input, H, T, input->nb[1], 0);
    auto qsa_query = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, QD, QH, QQ);
    auto qsa_raw_keys = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QD, QB * QRATIO + 1);
    auto qsa_raw_keys_f16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, QD, QB * QRATIO + 1);
    auto qsa_key_norm = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, QD);
    auto qsa_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QB, QQ);
    auto qsa_map = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, QRATIO, QB);
    auto qsa_tail = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, QQ);
    auto qsa_causal = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QKV, QQ);
    auto qsa_cache_batch = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QD, QCACHE_BATCH);
    auto qsa_key_cache = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, QD, QCACHE);
    auto qsa_union_block = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto qsa_union_tail = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
    auto qsa_union_enable = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, 2);
    const auto combined = qwen4exp_build_gated_residual(ctx, input, norm, down, up, inject, H, S, EPS);
    const auto output_only = qwen4exp_build_gated_residual(ctx, input, norm, down, up, nullptr, H, S, EPS);
    auto residual = qwen4exp_build_stream_injection(ctx, combined.hyper_input, combined.mixed_input,
            combined.injection_weights, H, S);
    const auto output_final = qwen4exp_build_gated_residual(ctx, residual, norm, down, up, nullptr, H, S, EPS);
    const auto ple = qwen4exp_build_ple_gate(ctx, input, embeddings, ple_key, ple_value,
            ple_key_norm, ple_query_norm, ple_conv_norm, H, S, EPS);
    const auto ple_conv = qwen4exp_build_ple_dilated_conv(
            ctx, ple.gated_value_normed, ple_conv_weight, ple_state, K, D);
    const auto ple_conv_zero = qwen4exp_build_ple_dilated_conv(
            ctx, ple.gated_value_normed, ple_conv_weight, nullptr, K, D);
    auto initial_streams = qwen4exp_build_initial_streams(ctx, token_embeddings, S);
    auto ple_output = qwen4exp_build_ple_output(ctx, ple.gated_value, ple_conv.output);
    const auto ple_persistent = qwen4exp_build_ple_state(ctx, ple_state_storage,
            PLE_BASE_STATE, 1, HS, (K - 1) * D, false);
    const auto ple_reset = qwen4exp_build_ple_state(ctx, ple_state_storage,
            PLE_BASE_STATE, 0, HS, (K - 1) * D, true);
    auto ple_prior_snapshot = ggml_dup(ctx, ple_persistent.prior_state);
    auto ple_reset_snapshot = ggml_dup(ctx, ple_reset.prior_state);
    auto ple_state_write = qwen4exp_build_ple_state_write(
            ctx, ple_conv.next_state, ple_persistent.destination);
    auto qsa_keys = qwen4exp_build_qsa_pooled_keys(
            ctx, qsa_raw_keys, qsa_key_norm, QRATIO, QB, EPS);
    auto qsa_keys_f16 = qwen4exp_build_qsa_pooled_keys(
            ctx, qsa_raw_keys_f16, qsa_key_norm, QRATIO, QB, EPS);
    require(qsa_keys_f16->type == GGML_TYPE_F32, "QSA F16 cache pools in F32");
    const auto qsa = qwen4exp_build_qsa_top_blocks(ctx, qsa_query, qsa_keys, qsa_mask, QTOP);
    auto qsa_tokens = qwen4exp_build_qsa_token_indices(ctx, qsa.top_blocks, qsa_map, qsa_tail);
    auto qsa_sparse_mask = qwen4exp_build_qsa_sparse_mask(ctx, qsa_causal, qsa_tokens);
    const auto qsa_cached = qwen4exp_build_qsa_cached_keys(
            ctx, qsa_cache_batch, qsa_key_cache, QCACHE_HEAD, QCACHE_USED);
    auto qsa_union = qwen4exp_build_qsa_mask_union(
            ctx, qsa_union_block, qsa_union_tail, qsa_union_enable);
    require(combined.hyper_input == input, "GR must preserve the residual input");
    require(combined.injection_weights != nullptr, "block GR injection output");
    require(output_only.injection_weights == nullptr, "output GR has no injection output");
    require(qsa_cached.write->op == GGML_OP_CPY, "QSA cache write node");
    require(qsa_cached.write->view_src == qsa_key_cache, "QSA cache write roots at key cache");
    require(qsa_cached.write->view_offs == size_t(QCACHE_HEAD) * qsa_key_cache->nb[1],
            "QSA cache write offset is reusable");

    const bool cuda_test = std::getenv("QWEN4EXP_TEST_CUDA_DEVICE") != nullptr;
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, combined.mixed_input);
    ggml_build_forward_expand(graph, combined.injection_weights);
    ggml_build_forward_expand(graph, output_only.mixed_input);
    ggml_build_forward_expand(graph, residual);
    ggml_build_forward_expand(graph, output_final.mixed_input);
    if (!cuda_test) {
        ggml_build_forward_expand(graph, ple.gated_value);
        ggml_build_forward_expand(graph, ple.gated_value_normed);
        ggml_build_forward_expand(graph, ple_conv.output);
        ggml_build_forward_expand(graph, ple_conv.next_state);
        ggml_build_forward_expand(graph, ple_conv_zero.output);
        ggml_build_forward_expand(graph, ple_conv_zero.next_state);
        ggml_build_forward_expand(graph, initial_streams);
        ggml_build_forward_expand(graph, ple_output);
        ggml_build_forward_expand(graph, ple_prior_snapshot);
        ggml_build_forward_expand(graph, ple_reset_snapshot);
        ggml_build_forward_expand(graph, ple_state_write);
        ggml_build_forward_expand(graph, qsa_cached.write);
        ggml_build_forward_expand(graph, qsa_keys_f16);
        ggml_build_forward_expand(graph, qsa.scores);
        ggml_build_forward_expand(graph, qsa.top_blocks);
        ggml_build_forward_expand(graph, qsa_tokens);
        ggml_build_forward_expand(graph, qsa_sparse_mask);
        ggml_build_forward_expand(graph, qsa_union);
    }
    ggml_backend_t backend = nullptr;
    const char * backend_name = "CPU";
#ifdef GGML_USE_CUDA
    if (const char * device_text = std::getenv("QWEN4EXP_TEST_CUDA_DEVICE")) {
        const int device = std::stoi(device_text);
        require(device >= 0 && device < ggml_backend_cuda_get_device_count(), "CUDA test device");
        backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        backend_name = "CUDA";
    }
#endif
    if (backend == nullptr) backend = ggml_backend_cpu_init();
    require(backend != nullptr, std::string(backend_name) + " backend");
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "backend allocation");
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(norm, norm_data.data(), 0, norm_data.size() * sizeof(float));
    ggml_backend_tensor_set(down, down_data.data(), 0, down_data.size() * sizeof(float));
    ggml_backend_tensor_set(up, up_data.data(), 0, up_data.size() * sizeof(float));
    ggml_backend_tensor_set(inject, inject_data.data(), 0, inject_data.size() * sizeof(float));
    ggml_backend_tensor_set(embeddings, embeddings_data.data(), 0, embeddings_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_key, ple_key_data.data(), 0, ple_key_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_value, ple_value_data.data(), 0, ple_value_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_key_norm, ple_key_norm_data.data(), 0, ple_key_norm_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_query_norm, ple_query_norm_data.data(), 0, ple_query_norm_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_conv_norm, ple_conv_norm_data.data(), 0, ple_conv_norm_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_conv_weight, ple_conv_data.data(), 0, ple_conv_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_state, ple_state_data.data(), 0, ple_state_data.size() * sizeof(float));
    ggml_backend_tensor_set(ple_state_storage, ple_storage_data.data(), 0,
            ple_storage_data.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_query, qsa_query_data.data(), 0, qsa_query_data.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_raw_keys, qsa_raw_key_data.data(), 0, qsa_raw_key_data.size() * sizeof(float));
    std::vector<ggml_fp16_t> qsa_raw_key_data_f16(qsa_raw_key_data.size());
    ggml_fp32_to_fp16_row(qsa_raw_key_data.data(), qsa_raw_key_data_f16.data(), qsa_raw_key_data.size());
    ggml_backend_tensor_set(qsa_raw_keys_f16, qsa_raw_key_data_f16.data(), 0,
            qsa_raw_key_data_f16.size() * sizeof(ggml_fp16_t));
    ggml_backend_tensor_set(qsa_key_norm, qsa_key_norm_data.data(), 0, qsa_key_norm_data.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_mask, qsa_mask_data.data(), 0, qsa_mask_data.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_map, qsa_block_token_map.data(), 0, qsa_block_token_map.size() * sizeof(int32_t));
    ggml_backend_tensor_set(qsa_tail, qsa_tail_data.data(), 0, qsa_tail_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(qsa_causal, qsa_causal_data.data(), 0, qsa_causal_data.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_cache_batch, qsa_cache_batch_data.data(), 0,
            qsa_cache_batch_data.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_key_cache, qsa_cache_initial.data(), 0,
            qsa_cache_initial.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_union_block, qsa_union_blocks.data(), 0,
            qsa_union_blocks.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_union_tail, qsa_union_tails.data(), 0,
            qsa_union_tails.size() * sizeof(float));
    ggml_backend_tensor_set(qsa_union_enable, qsa_union_enabled.data(), 0,
            qsa_union_enabled.size() * sizeof(float));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
            std::string(backend_name) + " graph compute");
    compare(combined.mixed_input, expected.mixed, "block GR mixed input");
    compare(combined.injection_weights, expected.injection, "block GR injection");
    compare(output_only.mixed_input, expected.mixed, "output GR mixed input");
    compare(residual, expected.residual, "GR stream injection");
    compare(output_final.mixed_input, expected_final_mixed, "final output GR mixed input");
    compare_fixture(output_final.mixed_input, upstream.at("gated_residual_final_mixed"),
            "pinned Transformers final GR mixed input");
    compare_fixture(combined.mixed_input, upstream.at("gated_residual_mixed"), "pinned Transformers GR mixed input");
    compare_fixture(combined.injection_weights, upstream.at("gated_residual_injection"), "pinned Transformers GR injection");
    compare_fixture(residual, upstream.at("gated_residual_output"), "pinned Transformers GR output");
    compare_fixture(output_final.mixed_input, upstream.at("gated_residual_final_mixed"),
            "pinned Transformers final GR mixed input");
    if (!cuda_test) {
        compare(ple.gated_value, expected_ple.first, "PLE gated value");
        compare(ple.gated_value_normed, expected_ple.second, "PLE gated value norm");
        compare(ple_conv.output, expected_conv.first, "PLE dilated convolution");
        compare(ple_conv.next_state, expected_conv.second, "PLE convolution state");
        compare(ple_conv_zero.output, expected_conv_zero.first, "PLE zero-state convolution");
        compare(ple_conv_zero.next_state, expected_conv_zero.second, "PLE zero-state update");
        compare(initial_streams, expected_initial_streams, "initial hyper-connection streams");
        compare(ple_output, expected_ple_output, "PLE gated-plus-convolution output");
        compare_fixture(ple.gated_value, upstream.at("ple_gated_value"), "pinned Transformers PLE gated value");
        compare_fixture(ple.gated_value_normed, upstream.at("ple_gated_value_normed"), "pinned Transformers PLE normalized value");
        compare_fixture(ple_output, upstream.at("ple_output"), "pinned Transformers PLE output");
        compare_fixture(ple_conv.next_state, upstream.at("ple_next_state"), "pinned Transformers PLE next state");
        compare(ple_prior_snapshot, expected_ple_prior, "PLE recurrent-row prior view");
        compare(ple_reset_snapshot, std::vector<float>(expected_ple_prior.size()), "PLE reset prior");
        compare(ple_state_storage, expected_ple_storage, "PLE recurrent-row writeback");
        compare(qsa.scores, expected_qsa_scores, "QSA block scores");
        compare(qsa_keys, qsa_key_data, "QSA pooled keys");
        compare_i32(qsa.top_blocks, expected_qsa_top, "QSA top blocks");
        compare_i32(qsa_tokens, expected_qsa_tokens, "QSA expanded blocks and tail");
        compare_mask(qsa_sparse_mask, expected_qsa_mask, "QSA sparse causal mask");
        compare(qsa_cached.cached_keys, expected_qsa_cached, "QSA persistent raw-key cache");
        compare_mask(qsa_union, expected_qsa_union, "QSA block-tail mask union");

        // Pinned Transformers QSA indexer parity: the fixture was produced by
        // executing the reference indexer and recording its internal top-k.
        const auto & qsa_reference = upstream_fixture.at("qsa");
        require(qsa_reference.at("heads").get<int64_t>() == QH &&
                qsa_reference.at("head_dim").get<int64_t>() == QD &&
                qsa_reference.at("compress_ratio").get<int64_t>() == QRATIO &&
                qsa_reference.at("top_blocks").get<int64_t>() == QTOP &&
                qsa_reference.at("blocks").get<int64_t>() == QB &&
                qsa_reference.at("queries").get<int64_t>() == QQ &&
                qsa_reference.at("kv_length").get<int64_t>() == QKV &&
                qsa_reference.at("raw_key_rows").get<int64_t>() == QB * QRATIO + 1,
                "pinned QSA fixture geometry");
        const double mask_floor = qsa_reference.at("mask_floor").get<double>();
        std::vector<float> qsa_score_actual(QB * QQ);
        ggml_backend_tensor_get(qsa.scores, qsa_score_actual.data(), 0,
                qsa_score_actual.size() * sizeof(float));
        std::vector<float> qsa_mask_actual(QKV * QQ);
        ggml_backend_tensor_get(qsa_sparse_mask, qsa_mask_actual.data(), 0,
                qsa_mask_actual.size() * sizeof(float));
        std::vector<int32_t> expected_order(QTOP * QQ);
        std::vector<int32_t> expected_selected((QRATIO * QTOP + 1) * QQ);
        size_t query_slot = 0;
        for (const auto & per_query : qsa_reference.at("per_query")) {
            require(query_slot < size_t(QQ), "pinned QSA fixture query count");
            require(per_query.at("query_position").get<int32_t>() == qsa_tail_data[query_slot],
                    "pinned QSA fixture query position");
            const auto & reference_scores = per_query.at("scores");
            // Reference scores cover exactly the complete blocks, in block order.
            for (size_t block = 0; block < reference_scores.size(); ++block) {
                require(block < size_t(QB), "pinned QSA complete-block count");
                const float actual = qsa_score_actual[block + QB * query_slot];
                const float expected = reference_scores.at(block).get<float>();
                require(std::isfinite(actual) && std::fabs(actual - expected) < 2.0e-5f,
                        "pinned Transformers QSA score, query " + std::to_string(query_slot) +
                        " block " + std::to_string(block));
            }
            const auto & block_order = per_query.at("block_order");
            require(block_order.size() == size_t(QTOP), "pinned QSA selected-block count");
            for (size_t rank = 0; rank < block_order.size(); ++rank) {
                expected_order[rank + QTOP * query_slot] = block_order.at(rank).get<int32_t>();
            }
            const auto & selected = per_query.at("selected_tokens");
            require(selected.size() == size_t(QRATIO * QTOP + 1), "pinned QSA token count");
            for (size_t rank = 0; rank < selected.size(); ++rank) {
                expected_selected[rank + (QRATIO * QTOP + 1) * query_slot] =
                    selected.at(rank).get<int32_t>();
            }
            const auto & mask_row = per_query.at("mask_row");
            require(mask_row.size() == size_t(QKV), "pinned QSA mask width");
            for (int64_t token = 0; token < QKV; ++token) {
                const bool reference_visible =
                    mask_row.at(token).get<double>() > mask_floor;
                const float actual = qsa_mask_actual[token + QKV * query_slot];
                const bool ese_visible = std::isfinite(actual) && actual > -1.0e29f;
                require(reference_visible == ese_visible,
                        "pinned Transformers QSA mask visibility, query " +
                        std::to_string(query_slot) + " token " + std::to_string(token));
            }
            ++query_slot;
        }
        require(query_slot == size_t(QQ), "pinned QSA fixture query count");
        compare_i32(qsa.top_blocks, expected_order, "pinned Transformers QSA block order");
        compare_i32(qsa_tokens, expected_selected, "pinned Transformers QSA selected tokens");
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    check_softplus_precision(backend);
    check_explicit_projection_precision(backend);
    check_explicit_projection_precision(backend, 4, 10240);
    check_explicit_projection_precision(backend, 128, 10240);
    check_explicit_projection_precision(backend, 9, 2579);
    check_explicit_projection_precision(backend, 320, 2560);
    check_explicit_quantized_precision(backend, GGML_TYPE_Q6_K);
    check_explicit_quantized_precision(backend, GGML_TYPE_Q4_K);
    check_qsa_partial_rope(backend, upstream_fixture.at("qsa_rotary"));
    check_l2(backend, upstream_fixture.at("recurrent_l2"));
    check_routing_and_head(backend, upstream_fixture);
    check_recurrent_layer(backend, upstream_fixture.at("recurrent_layer"));
    check_recurrent_layer(backend, upstream_fixture.at("recurrent_layer_gguf"));
    for (const char * section : { "recurrent_native", "recurrent_grouped",
                                 "recurrent_low_magnitude", "recurrent_scalar",
                                 "recurrent_scalar_low_magnitude" }) {
        // 32-wide cases deliberately exercise scalar CPU fallback. CUDA's
        // recurrent kernel only supports widths 64/128; this is not a CUDA gate.
        if (cuda_test && upstream_fixture.at(section).at("key_head_dim").get<int64_t>() == 32) continue;
        std::cout << "Checking " << section << '\n';
        check_recurrence(backend, upstream_fixture.at(section), false);
        check_recurrence(backend, upstream_fixture.at(section), true);
        // Without checkpoint storage CUDA can dispatch chunked prefill instead
        // of the sequential checkpoint kernel; retain the same oracle.
        check_recurrence(backend, upstream_fixture.at(section), false, false);
        check_recurrence(backend, upstream_fixture.at(section), true, false);
    }
    ggml_backend_free(backend);
    std::cout << "PASS: Qwen4Exp operators, routing, output head and recurrence match pinned Transformers synthetic goldens\n";
    return 0;
}
