#include "llama-deepseek-v41-forward.h"
#include "ggml-backend.h"
#include "llama-model.h"
#include "llama-context.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <nlohmann/json.hpp>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>

using namespace deepseek_v41;
static void check(bool ok, const std::string & message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char ** argv) {
    try {
        check(argc == 3 || (argc == 4 && std::string(argv[3]) == "--whole-model"), "usage: test-deepseek-v41-forward FIXTURE.json cpu|CUDA_DEVICE [--whole-model]");
        const bool whole_model = argc == 4;
        std::ifstream input(argv[1]); const auto fixture = nlohmann::json::parse(input);
        auto config = model_config::from_json(fixture.at("config").dump());
        attention_forward_state state(config, config.context);
        const bool full_block = fixture.value("full_block", false);
        llama_model model{};
        model.arch = LLM_ARCH_DEEPSEEK41;
        llama_context lctx(model);
        lctx.cparams = {};
        lctx.inp_tokens = nullptr; lctx.inp_embd = nullptr;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        check(bool(cpu), "CPU backend failed"); ggml_backend_cpu_set_n_threads(cpu.get(), 4);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu(nullptr, ggml_backend_free);
        if (std::string(argv[2]) != "cpu") {
#ifdef GGML_USE_CUDA
            gpu.reset(ggml_backend_cuda_init(std::stoi(argv[2]), nullptr, nullptr));
            check(bool(gpu), "CUDA backend failed");
#else
            throw std::runtime_error("CUDA not compiled");
#endif
        }
        auto primary = gpu ? gpu.get() : cpu.get();
        std::vector<ggml_backend_t> backends = {primary}; if (gpu) backends.push_back(cpu.get());
        ggml_init_params wp{ggml_tensor_overhead() * 1024, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> weight_ctx(ggml_init(wp), ggml_free);
        std::map<std::string, ggml_tensor *> weights;
        for (auto entry = fixture.at("weights").begin(); entry != fixture.at("weights").end(); ++entry) {
            const auto shape = entry.value().at("shape").get<std::vector<int64_t>>();
            auto * tensor = ggml_new_tensor(weight_ctx.get(), entry.value().value("dtype", "f32") == "bf16" ? GGML_TYPE_BF16 : GGML_TYPE_F32,
                                           shape.size(), shape.data());
            ggml_set_name(tensor, entry.key().c_str()); weights[entry.key()] = tensor;
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> weight_buffer(
            ggml_backend_alloc_ctx_tensors(weight_ctx.get(), primary), ggml_backend_buffer_free);
        check(bool(weight_buffer), "weight allocation failed");
        for (const auto & entry : weights) {
            const auto values = fixture.at("weights").at(entry.first).at("values").get<std::vector<float>>();
            if (entry.second->type == GGML_TYPE_BF16) {
                std::vector<ggml_bf16_t> bf16(values.size());
                ggml_fp32_to_bf16_row_ref(values.data(), bf16.data(), values.size());
                ggml_backend_tensor_set(entry.second, bf16.data(), 0, bf16.size() * sizeof(ggml_bf16_t));
            } else ggml_backend_tensor_set(entry.second, values.data(), 0, values.size() * sizeof(float));
        }
        std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched(
            ggml_backend_sched_new(backends.data(), nullptr, backends.size(), whole_model ? 16384 : 4096, false), ggml_backend_sched_free);
        check(bool(sched), "scheduler creation failed");
        double maximum_error = 0;
        size_t evaluated = 0, gpu_matmuls = 0;
        for (size_t position = 0; position < fixture.at("steps").size(); ++position) {
            const int attempts = position == 5 ? 3 : 1; // replay both a complete abort and an injected failure
            for (int attempt = 0; attempt < attempts; ++attempt) {
                state.begin(position);
                std::vector<float> carry, carry_pre;
                if (whole_model) {
                    check(full_block, "whole-model execution requires a full block fixture");
                    ggml_init_params gp{ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr, true};
                    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(gp), ggml_free);
                    struct reset_guard { ggml_backend_sched_t value; ~reset_guard() { ggml_backend_sched_reset(value); } } reset{sched.get()};
                    auto token = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
                    auto pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
                    ggml_set_input(token); ggml_set_input(pos);
                    std::map<uint32_t, ggml_tensor *> rows;
                    for (auto layer : config.engram_layers) {
                        int64_t width = fixture["steps"][position][layer]["engram_rows"]["shape"][0];
                        rows[layer] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
                        ggml_set_input(rows[layer]);
                    }
                    auto graph = ggml_new_graph_custom(ctx.get(), 16384, false);
                    auto output = state.build_token(ctx.get(), lctx, weights, token, pos, rows, graph);
                    check(ggml_backend_sched_alloc_graph(sched.get(), graph), "whole-model graph allocation failed");
                    if (gpu) for (int node = 0; node < graph->n_nodes; ++node) {
                        const auto op = graph->nodes[node]->op;
                        if (op == GGML_OP_HC_POST && graph->nodes[node]->op_params[0] == 1) {
                            check(ggml_backend_sched_get_tensor_backend(sched.get(), graph->nodes[node]) == gpu.get(),
                                  "whole-model precise HC expansion fell back to CPU");
                        }
                        if (op == GGML_OP_MUL_MAT || op == GGML_OP_MUL_MAT_ID) {
                            check(ggml_backend_sched_get_tensor_backend(sched.get(), graph->nodes[node]) == gpu.get(),
                                  "whole-model matrix multiplication fell back to CPU");
                            ++gpu_matmuls;
                        }
                    }
                    const bool fail = position == 5 && attempt == 1;
                    const int32_t tid = fixture["token_steps"][position]["token"];
                    const int32_t pid = position + (fail ? 1 : 0);
                    ggml_backend_tensor_set(token, &tid, 0, sizeof(tid));
                    ggml_backend_tensor_set(pos, &pid, 0, sizeof(pid));
                    for (auto row : rows) {
                        auto values = fixture["steps"][position][row.first]["engram_rows"]["values"].get<std::vector<float>>();
                        ggml_backend_tensor_set(row.second, values.data(), 0, values.size() * sizeof(float));
                    }
                    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "whole-model compute failed");
                    ggml_backend_sched_synchronize(sched.get());
                    if (fail) {
                        check(!state.error().empty(), "whole-model position failure not detected");
                        bool rejected = false;
                        try { state.commit(); } catch (const std::runtime_error &) { rejected = true; }
                        check(rejected, "whole-model failed step committed");
                    } else {
                        check(state.error().empty(), state.error());
                        auto expected = fixture["token_steps"][position]["logits"]["values"].get<std::vector<float>>();
                        std::vector<float> actual(expected.size());
                        ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                        for (size_t i = 0; i < actual.size(); ++i) {
                            const double error = std::abs(double(actual[i]) - expected[i]);
                            maximum_error = std::max(maximum_error, error);
                            check(std::isfinite(actual[i]) && error <= 1e-3 + 1e-3 * std::abs(expected[i]),
                                  "whole-model logit mismatch at position " + std::to_string(position) + " element " + std::to_string(i));
                        }
                    }
                    evaluated += config.attention.compress_ratios.size();
                    if (attempt + 1 < attempts) state.abort(); else state.commit();
                    continue;
                }
                for (size_t layer = 0; layer < config.attention.compress_ratios.size(); ++layer) {
                    ggml_init_params gp{ggml_tensor_overhead() * 1024 + ggml_graph_overhead_custom(4096, false), nullptr, true};
                    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(gp), ggml_free);
                    struct reset_guard { ggml_backend_sched_t value; ~reset_guard() { ggml_backend_sched_reset(value); } } reset{sched.get()};
                    auto * x = full_block ? ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, config.hidden, config.streams, 1) :
                            ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, config.hidden, 1);
                    auto * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
                    ggml_set_input(x); ggml_set_input(pos);
                    const auto & item = fixture.at("steps").at(position).at(layer);
                    auto * graph = ggml_new_graph_custom(ctx.get(), 4096, false);
                    ggml_tensor * output, * pre = nullptr, * next_pre = nullptr, * rows = nullptr, * mask = nullptr;
                    if (full_block) {
                        pre = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, config.streams, 1);
                        ggml_set_input(pre);
                        if (item.contains("engram_rows")) {
                            rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, item["engram_rows"]["shape"][0].get<int64_t>(), 1);
                            mask = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 1, 1);
                            ggml_set_input(rows); ggml_set_input(mask);
                        }
                        auto block = state.build_block(ctx.get(), lctx, weights, layer, x, pre, pos, graph, rows, mask);
                        output = block.streams; next_pre = block.next_pre;
                        ggml_set_output(next_pre);
                        ggml_build_forward_expand(graph, next_pre);
                    } else output = state.build(ctx.get(), weights, layer, x, pos, fixture.value("production_precision", false));
                    ggml_build_forward_expand(graph, output);
                    check(ggml_backend_sched_alloc_graph(sched.get(), graph), "forward graph allocation failed");
                    if (gpu) for (int node = 0; node < graph->n_nodes; ++node) {
                        if (graph->nodes[node]->op == GGML_OP_HC_POST && graph->nodes[node]->op_params[0] == 1) {
                            check(ggml_backend_sched_get_tensor_backend(sched.get(), graph->nodes[node]) == gpu.get(),
                                  "precise HC expansion fell back to CPU");
                        }
                    }
                    if (gpu) for (int node = 0; node < graph->n_nodes; ++node) if (graph->nodes[node]->op == GGML_OP_MUL_MAT || graph->nodes[node]->op == GGML_OP_MUL_MAT_ID) {
                        check(ggml_backend_sched_get_tensor_backend(sched.get(), graph->nodes[node]) == gpu.get(), "matrix multiplication fell back to CPU");
                        ++gpu_matmuls;
                    }
                    auto values = item.at("input").at("values").get<std::vector<float>>();
                    const bool inject_failure = position == 5 && attempt == 1 && layer + 1 == config.attention.compress_ratios.size();
                    const int32_t p = position + (inject_failure ? 1 : 0);
                    if (full_block) {
                        if (layer) values = carry;
                        auto pv = layer ? carry_pre : item["incoming_pre"]["values"].get<std::vector<float>>();
                        ggml_backend_tensor_set(pre, pv.data(), 0, pv.size() * sizeof(float));
                        if (rows) {
                            auto rv = item["engram_rows"]["values"].get<std::vector<float>>();
                            const float one = 1;
                            ggml_backend_tensor_set(rows, rv.data(), 0, rv.size() * sizeof(float));
                            ggml_backend_tensor_set(mask, &one, 0, sizeof(one));
                        }
                    }
                    ggml_backend_tensor_set(x, values.data(), 0, values.size() * sizeof(float));
                    ggml_backend_tensor_set(pos, &p, 0, sizeof(p));
                    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "attention forward execution failed");
                    ggml_backend_sched_synchronize(sched.get());
                    if (inject_failure) {
                        check(!state.error().empty(), "mismatched cache/rotary position did not record a forward failure");
                        bool rejected = false;
                        try { state.commit(); } catch (const std::runtime_error &) { rejected = true; }
                        check(rejected, "failed forward was committed");
                        ++evaluated;
                        continue;
                    }
                    check(state.error().empty(), state.error());
                    auto expected = item.at("output").at("values").get<std::vector<float>>();
                    std::vector<float> actual(expected.size());
                    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                    for (size_t i = 0; i < actual.size(); ++i) {
                        const double error = std::abs(double(actual[i]) - expected[i]);
                        maximum_error = std::max(maximum_error, error);
                        check(std::isfinite(actual[i]) && error <= fixture.at("absolute_tolerance").get<double>() +
                            fixture.at("relative_tolerance").get<double>() * std::abs(expected[i]),
                            "forward mismatch at position " + std::to_string(position) + " layer " + std::to_string(layer) +
                            " element " + std::to_string(i) + " actual " + std::to_string(actual[i]) + " expected " + std::to_string(expected[i]));
                    }
                    if (full_block) {
                        carry = actual;
                        carry_pre.resize(config.streams);
                        ggml_backend_tensor_get(next_pre, carry_pre.data(), 0, carry_pre.size() * sizeof(float));
                        auto expected_pre = item["next_pre"]["values"].get<std::vector<float>>();
                        for (size_t i = 0; i < carry_pre.size(); ++i) check(std::abs(carry_pre[i] - expected_pre[i]) < 1e-3f,
                                "full block next pre-mix mismatch");
                    }
                    ++evaluated;
                }
                if (attempt + 1 < attempts) state.abort(); else state.commit();
            }
        }
        std::printf("%s\n", nlohmann::json{{"backend", ggml_backend_name(primary)}, {"layer_evaluations", evaluated},
            {"max_absolute_error", maximum_error}, {"gpu_matmuls", gpu_matmuls}, {"aborted_step_replayed", true},
            {"failed_step_replayed", true}}.dump(2).c_str());
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
