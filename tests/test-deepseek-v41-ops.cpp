#include "llama-deepseek-v41-ops.h"
#include "ggml-backend.h"
#include "ggml-hc.h"
#include "../ggml/src/ggml-qat.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;
using namespace deepseek_v41;

static void require(bool condition, const std::string & message) {
    if (!condition) throw std::runtime_error(message);
}

static void check_hc_normalization_order(ggml_backend_t backend, bool cpu) {
    // The pinned hc_split_sinkhorn reduces first, then adds epsilon. A scalar
    // oracle catches moving epsilon into the accumulator even when ordinary
    // primitive tolerances hide the resulting float differences.
    for (int streams = 1; streams <= 8; ++streams) {
        for (int iterations : {2, 20}) {
            constexpr int tokens = 3;
            const int width = streams * (streams + 2);
            const float eps = 1e-6f;
            ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(8, false), nullptr, true};
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
            require(bool(ctx), "HC normalization context allocation failed");
            auto * mixes = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, tokens);
            auto * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
            auto * base = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, width);
            auto * coefficients = ggml_hc_pre(ctx.get(), mixes, scale, base, streams, iterations, eps);
            auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
            ggml_build_forward_expand(graph, coefficients);
            require(ggml_backend_supports_op(backend, coefficients), "HC normalization backend unsupported");
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
            require(bool(buffer), "HC normalization allocation failed");
            std::vector<float> input(width * tokens), bias(width, 0.0f);
            const std::array<float, 3> scaling = {1.0f, 1.0f, 1.0f};
            for (int token = 0; token < tokens; ++token) {
                for (int i = 0; i < width; ++i) input[token * width + i] = float(((i * 17 + token * 11) % 31) - 15) * 0.125f;
            }
            ggml_backend_tensor_set(mixes, input.data(), 0, input.size() * sizeof(float));
            ggml_backend_tensor_set(scale, scaling.data(), 0, sizeof(scaling));
            ggml_backend_tensor_set(base, bias.data(), 0, bias.size() * sizeof(float));
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "HC normalization execution failed");
            ggml_backend_synchronize(backend);
            std::vector<float> actual(width * tokens);
            ggml_backend_tensor_get(coefficients, actual.data(), 0, actual.size() * sizeof(float));
            for (int token = 0; token < tokens; ++token) {
                std::vector<float> expected(streams * streams);
                for (int row = 0; row < streams; ++row) {
                    float maximum = input[token * width + 2 * streams + row * streams];
                    for (int col = 1; col < streams; ++col) maximum = std::max(maximum, input[token * width + 2 * streams + row * streams + col]);
                    float sum = 0.0f;
                    for (int col = 0; col < streams; ++col) {
                        auto & value = expected[row * streams + col];
                        value = std::exp(input[token * width + 2 * streams + row * streams + col] - maximum);
                        sum += value;
                    }
                    for (int col = 0; col < streams; ++col) expected[row * streams + col] = expected[row * streams + col] / sum + eps;
                }
                auto normalize = [&](bool columns) {
                    for (int outer = 0; outer < streams; ++outer) {
                        float sum = 0.0f;
                        for (int inner = 0; inner < streams; ++inner) sum += expected[columns ? inner * streams + outer : outer * streams + inner];
                        sum += eps;
                        for (int inner = 0; inner < streams; ++inner) expected[columns ? inner * streams + outer : outer * streams + inner] /= sum;
                    }
                };
                normalize(true);
                for (int i = 1; i < iterations; ++i) { normalize(false); normalize(true); }
                for (int i = 0; i < streams * streams; ++i) {
                    const float value = actual[2 * streams * tokens + token * streams * streams + i];
                    const float error = std::abs(value - expected[i]);
                    require(std::isfinite(value) && (cpu ? error == 0.0f : error <= 3e-7f),
                        "HC normalization must add epsilon after reduction: streams " + std::to_string(streams) +
                        " iterations " + std::to_string(iterations) + " token " + std::to_string(token) +
                        " element " + std::to_string(i));
                }
            }
        }
    }
}

static void check_hc_precise_expansion(ggml_backend_t backend) {
    // Independent long-double oracle; use more than one token and a hidden
    // width crossing CUDA's usual warp width to exercise both index axes.
    for (int S = 1; S <= 8; ++S) for (int iterations : {2, 20}) {
        constexpr int H = 65, T = 3;
        const int W = S*(S+2);
        const float eps = 1e-6f;
        ggml_init_params params{ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        require(bool(ctx), "precise HC context allocation failed");
        auto * x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, H, T);
        auto * res = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, H, S, T);
        auto * p = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, W, T);
        auto * scale = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 3);
        auto * bias = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, W);
        auto * out = ggml_hc_post_precise(ctx.get(), x, res, p, scale, bias, iterations, eps);
        auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, out);
        for (int n = 0; n < graph->n_nodes; ++n)
            require(ggml_backend_supports_op(backend, graph->nodes[n]), "precise HC backend unsupported");
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
        require(bool(buffer), "precise HC tensor allocation failed");
        std::vector<float> xv(H*T), rv(H*S*T), pv(W*T), bv(W);
        const std::array<float, 3> sv = {0.75f, -0.375f, 1.25f};
        for (size_t i = 0; i < xv.size(); ++i) xv[i] = float(int(i*13%43)-21)*0.0625f;
        for (size_t i = 0; i < rv.size(); ++i) rv[i] = float(int(i*17%61)-30)*0.03125f;
        for (size_t i = 0; i < pv.size(); ++i) pv[i] = float(int(i*19%47)-23)*0.125f;
        for (size_t i = 0; i < bv.size(); ++i) bv[i] = float(int(i*7%13)-6)*0.0625f;
        ggml_backend_tensor_set(x, xv.data(), 0, xv.size()*sizeof(float));
        ggml_backend_tensor_set(res, rv.data(), 0, rv.size()*sizeof(float));
        ggml_backend_tensor_set(p, pv.data(), 0, pv.size()*sizeof(float));
        ggml_backend_tensor_set(scale, sv.data(), 0, sizeof(sv));
        ggml_backend_tensor_set(bias, bv.data(), 0, bv.size()*sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "precise HC compute failed");
        ggml_backend_synchronize(backend);
        std::vector<float> actual(H*S*T);
        ggml_backend_tensor_get(out, actual.data(), 0, actual.size()*sizeof(float));
        for (int t = 0; t < T; ++t) {
            std::vector<long double> matrix(S*S), post(S);
            for (int d = 0; d < S; ++d)
                post[d] = 2.0L / (1.0L + std::exp(-(static_cast<long double>(pv[t*W+S+d])*sv[1] + bv[S+d])));
            for (int i = 0; i < S*S; ++i) matrix[i] = static_cast<long double>(pv[t*W+2*S+i])*sv[2]+bv[2*S+i];
            for (int r = 0; r < S; ++r) {
                long double maximum = matrix[r*S], sum = 0;
                for (int c = 1; c < S; ++c) maximum = std::max(maximum, matrix[r*S+c]);
                for (int c = 0; c < S; ++c) { matrix[r*S+c] = std::exp(matrix[r*S+c]-maximum); sum += matrix[r*S+c]; }
                for (int c = 0; c < S; ++c) matrix[r*S+c] = matrix[r*S+c]/sum+eps;
            }
            auto normalize = [&](bool columns) {
                for (int outer = 0; outer < S; ++outer) {
                    long double sum = 0;
                    for (int inner = 0; inner < S; ++inner) sum += matrix[columns ? inner*S+outer : outer*S+inner];
                    sum += eps;
                    for (int inner = 0; inner < S; ++inner) matrix[columns ? inner*S+outer : outer*S+inner] /= sum;
                }
            };
            normalize(true);
            for (int i = 1; i < iterations; ++i) { normalize(false); normalize(true); }
            for (int d = 0; d < S; ++d) for (int h = 0; h < H; ++h) {
                long double sum = 0;
                for (int r = 0; r < S; ++r) sum += matrix[r*S+d]*rv[(t*S+r)*H+h];
                const float expected = float(sum + post[d]*xv[t*H+h]);
                const float v = actual[(t*S+d)*H+h];
                require(std::isfinite(v) && std::abs(v-expected) <= 2e-7f*(1+std::abs(expected)),
                        "precise HC expansion differs from long-double oracle");
            }
        }
    }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: test-deepseek-v41-ops FIXTURE.json cpu|CUDA_DEVICE_NUMBER");
        std::ifstream file(argv[1]);
        require(bool(file), "missing V4.1 ops reference");
        const auto fixture = json::parse(file);
        ggml_backend_t raw = nullptr;
        if (std::string(argv[2]) == "cpu") {
            raw = ggml_backend_cpu_init();
            if (raw) ggml_backend_cpu_set_n_threads(raw, 4);
        } else {
#ifdef GGML_USE_CUDA
            int device = std::stoi(argv[2]);
            require(device >= 0 && device < ggml_backend_cuda_get_device_count(), "invalid CUDA device");
            raw = ggml_backend_cuda_init(device, nullptr, nullptr);
#else
            throw std::runtime_error("CUDA backend not compiled; refusing CPU fallback");
#endif
        }
        require(raw != nullptr, "backend initialization failed");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw, ggml_backend_free);
        check_hc_normalization_order(backend.get(), std::string(argv[2]) == "cpu");
        check_hc_precise_expansion(backend.get());
        const float atol = fixture.at("absolute_tolerance"), rtol = fixture.at("relative_tolerance");
        json results = json::array();
        for (const auto & item : fixture.at("cases")) {
            ggml_init_params params{ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(1024, false), nullptr, true};
            std::unique_ptr<ggml_context, decltype(&ggml_free)> owner(ggml_init(params), ggml_free);
            auto * ctx = owner.get();
            require(ctx != nullptr, "context allocation failed");
            std::map<std::string, ggml_tensor *> inputs, outputs;
            for (auto entry = item.at("inputs").begin(); entry != item.at("inputs").end(); ++entry) {
                const auto shape = entry.value().at("shape").get<std::vector<int64_t>>();
                require(shape.size() > 0 && shape.size() <= 4, "invalid input rank");
                auto * tensor = ggml_new_tensor(ctx, entry.value().value("dtype", "f32") == "i32" ? GGML_TYPE_I32 : GGML_TYPE_F32,
                                               int(shape.size()), shape.data());
                ggml_set_name(tensor, entry.key().c_str());
                inputs[entry.key()] = tensor;
            }
            auto in = [&](const char * name) { return inputs.at(name); };
            const std::string kind = item.value("kind", "block");
            if (kind == "index_scores") {
                outputs["scores"] = build_indexer_scores(ctx, in("q"), in("k"), in("weights"));
            } else if (kind == "attention") {
                auto * attended = build_attention_core(ctx, in("q"), in("kv"), in("mask"), in("sinks"));
                outputs["attended"] = attended;
                outputs["output"] = build_attention_output(ctx, attended, in("wo_a"), in("wo_b"));
            } else if (kind == "qat") {
                outputs["output"] = ggml_qat(ctx, in("x"), (ggml_qat_format) item.at("mode").get<int>());
            } else if (kind == "compression") {
                auto result = build_compression(ctx, in("values"), inputs.count("scores") ? in("scores") : nullptr,
                                                in("norm_weight"), item.at("ratio"), fixture.at("norm_eps"));
                if (result.latent) outputs["latent"] = result.latent;
                if (result.tail_values) outputs["tail_values"] = result.tail_values;
                if (result.tail_scores) outputs["tail_scores"] = result.tail_scores;
            } else if (kind == "rope") {
                rope_config config;
                config.rotary_dimensions = item.at("rope").at("rotary_dimensions");
                config.original_context = item.at("rope").at("original_context");
                config.theta = item.at("rope").at("theta");
                config.factor = item.at("rope").at("factor");
                config.beta_fast = item.at("rope").at("beta_fast");
                config.beta_slow = item.at("rope").at("beta_slow");
                outputs["forward"] = build_rope_tail(ctx, in("x"), in("positions"), config);
                outputs["inverse"] = build_rope_tail(ctx, in("x"), in("positions"), config, true);
                outputs["restored"] = build_rope_tail(ctx, outputs["forward"], in("positions"), config, true);
            } else {
                require(kind == "block", "unknown fixture kind");
                auto injection = build_engram_injection(ctx, in("x"), in("embeddings"), in("key_value_projection"),
                    in("q_weight"), in("k_weight"), in("text_mask"), fixture.at("norm_eps"));
                auto attn = build_hc_mixes(ctx, injection.output, in("attn_fn"), in("attn_scale"), in("attn_base"),
                    fixture.at("norm_eps"), fixture.at("sinkhorn_iterations"), fixture.at("hc_eps"));
                auto * attn_input = build_hc_collapse(ctx, injection.output, in("incoming_pre"));
                auto * attn_value = ggml_mul_mat(ctx, in("attn_weight"), attn_input);
                ggml_mul_mat_set_prec(attn_value, GGML_PREC_F32);
                auto * after_attention = build_hc_expand(ctx, attn_value, injection.output, attn);
                auto ffn = build_hc_mixes(ctx, after_attention, in("ffn_fn"), in("ffn_scale"), in("ffn_base"),
                    fixture.at("norm_eps"), fixture.at("sinkhorn_iterations"), fixture.at("hc_eps"));
                auto * ffn_input = build_hc_collapse(ctx, after_attention, attn.pre);
                auto * ffn_value = ggml_mul_mat(ctx, in("ffn_weight"), ffn_input);
                ggml_mul_mat_set_prec(ffn_value, GGML_PREC_F32);
                auto * output = build_hc_expand(ctx, ffn_value, after_attention, ffn);
                outputs = {{"engram_output", injection.output}, {"engram_gate", injection.gate}, {"attn_pre", attn.pre},
                    {"attn_post", attn.post}, {"attn_comb", attn.comb}, {"attn_input", attn_input},
                    {"after_attention", after_attention}, {"ffn_input", ffn_input}, {"next_pre", ffn.pre}, {"block_output", output}};
            }
            require(outputs.size() == item.at("expected").size(), "missing expected graph outputs");
            auto * graph = ggml_new_graph_custom(ctx, 1024, false);
            for (auto & entry : outputs) {
                ggml_set_name(entry.second, entry.first.c_str());
                ggml_build_forward_expand(graph, entry.second);
            }
            for (int node = 0; node < graph->n_nodes; ++node) {
                require(ggml_backend_supports_op(backend.get(), graph->nodes[node]),
                    std::string("unsupported graph node: ") + ggml_op_name(graph->nodes[node]->op));
            }
            auto * allocation = ggml_backend_alloc_ctx_tensors(ctx, backend.get());
            require(allocation != nullptr, "graph allocation failed");
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(allocation, ggml_backend_buffer_free);
            for (const auto & entry : inputs) {
                if (entry.second->type == GGML_TYPE_I32) {
                    auto values = item.at("inputs").at(entry.first).at("values").get<std::vector<int32_t>>();
                    require(values.size() * sizeof(int32_t) == ggml_nbytes(entry.second), "fixture input extent mismatch");
                    ggml_backend_tensor_set(entry.second, values.data(), 0, values.size() * sizeof(int32_t));
                    continue;
                }
                auto values = item.at("inputs").at(entry.first).at("values").get<std::vector<float>>();
                require(values.size() * sizeof(float) == ggml_nbytes(entry.second), "fixture input extent mismatch");
                ggml_backend_tensor_set(entry.second, values.data(), 0, values.size() * sizeof(float));
            }
            require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "ops graph execution failed");
            ggml_backend_synchronize(backend.get());
            json errors;
            json mismatches = json::object();
            for (const auto & entry : outputs) {
                auto expected = item.at("expected").at(entry.first).at("values").get<std::vector<float>>();
                require(int64_t(expected.size()) == ggml_nelements(entry.second), "output extent mismatch");
                // All selected outputs are contiguous or packed views of the HC coefficients.
                require(ggml_is_contiguous(entry.second), "non-contiguous test output");
                std::vector<float> actual(expected.size());
                ggml_backend_tensor_get(entry.second, actual.data(), 0, actual.size() * sizeof(float));
                if (item.value("zero_first_query", false)) {
                    const size_t prefix = entry.first == "attended" ? entry.second->ne[0] * entry.second->ne[1] : entry.second->ne[0];
                    for (size_t i = 0; i < prefix; ++i) require(actual[i] == 0.0f, "sink-only query must produce exact zero");
                }
                double max_error = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    double error = std::abs(double(actual[i]) - expected[i]);
                    max_error = std::max(max_error, error);
                    if ((!std::isfinite(actual[i]) || error > atol + rtol * std::abs(expected[i])) && !mismatches.contains(entry.first)) {
                        mismatches[entry.first] = {{"index", i}, {"actual", actual[i]}, {"expected", expected[i]}};
                    }
                }
                errors[entry.first] = max_error;
            }
            results.push_back({{"case", item.at("name")}, {"max_absolute_errors", errors}});
            if (!mismatches.empty()) {
                std::fprintf(stderr, "%s\n", json{{"case", item.at("name")}, {"max_errors", errors}, {"first_mismatches", mismatches}}.dump(2).c_str());
                throw std::runtime_error("V4.1 graph primitive parity failed");
            }
        }
        std::printf("%s\n", json{{"backend", ggml_backend_name(backend.get())}, {"results", results}}.dump(2).c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
