#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-deepseek-v41-ops.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <map>
using nlohmann::json;
static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
int main(int argc, char ** argv) {
    try {
        check(argc == 2 || argc == 3, "usage: test-deepseek-v41-moe REFERENCE.json [cpu|CUDA_DEVICE]");
        std::ifstream file(argv[1]);
        auto fixture = json::parse(file);
        llama_model model{};
        model.arch = LLM_ARCH_DEEPSEEK41;
        llama_context lctx(model);
        lctx.cparams = {};
        lctx.inp_tokens = nullptr;
        lctx.inp_embd = nullptr;
        ggml_backend_t raw = nullptr;
        if (argc == 2 || std::string(argv[2]) == "cpu") {
            raw = ggml_backend_cpu_init();
            if (raw) ggml_backend_cpu_set_n_threads(raw, 4);
        } else {
#ifdef GGML_USE_CUDA
            int device = std::stoi(argv[2]);
            check(device >= 0 && device < ggml_backend_cuda_get_device_count(), "invalid CUDA device");
            raw = ggml_backend_cuda_init(device, nullptr, nullptr);
#else
            throw std::runtime_error("CUDA not compiled; refusing CPU fallback");
#endif
        }
        check(raw, "backend init failed");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw, ggml_backend_free);
        double max_error = 0;
        for (const auto & sample : fixture.at("cases")) {
            ggml_init_params params{ggml_tensor_overhead() * 512 + ggml_graph_overhead_custom(512, false), nullptr, true};
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
            std::map<ggml_tensor *, std::vector<float>> data;
            auto tensor = [&](const json & item) {
                auto shape = item.at("shape").get<std::vector<int64_t>>();
                auto t = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, shape.size(), shape.data());
                data[t] = item.at("values").get<std::vector<float>>();
                return t;
            };
            auto x = tensor(sample.at("input"));
            auto router = tensor(fixture["weights"]["router"]);
            auto bias = tensor(fixture["weights"]["bias"]);
            auto up = tensor(fixture["weights"]["up"]);
            auto gate = tensor(fixture["weights"]["gate"]);
            auto down = tensor(fixture["weights"]["down"]);
            std::map<std::string, ggml_tensor *> named;
            llm_build_cb cb = [&](ggml_tensor * t, const char * name, int) { named[name] = t; };
            auto graph = ggml_new_graph_custom(ctx.get(), 512, false);
            auto out = llm_build_context::llm_build_moe_ffn(ctx.get(), lctx, x, router, up, gate, down, bias,
                    4, 3, LLM_FFN_SILU, true, true, 1.5f, LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS, cb, 0, graph);
            auto shared = deepseek_v41::build_shared_expert(ctx.get(), x,
                    tensor(fixture["weights"]["shared_gate"]), tensor(fixture["weights"]["shared_up"]),
                    tensor(fixture["weights"]["shared_down"]));
            auto combined = ggml_cast(ctx.get(), ggml_cast(ctx.get(), ggml_add(ctx.get(), out, shared), GGML_TYPE_BF16), GGML_TYPE_F32);
            ggml_build_forward_expand(graph, combined);
            const auto & engram_case = fixture["engram_cases"][x->ne[1] == 1 ? 0 : 1];
            auto engram = deepseek_v41::build_engram_injection(ctx.get(), tensor(engram_case["input"]),
                    tensor(engram_case["embeddings"]), tensor(engram_case["projection"]),
                    tensor(engram_case["q"]), tensor(engram_case["k"]), tensor(engram_case["mask"]), 1e-20f, true);
            ggml_build_forward_expand(graph, engram.output);
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            check(bool(buffer), "buffer allocation failed");
            for (const auto & entry : data) ggml_backend_tensor_set(entry.first, entry.second.data(), 0, entry.second.size() * sizeof(float));
            for (int i = 0; i < graph->n_nodes; ++i) {
                check(ggml_backend_supports_op(backend.get(), graph->nodes[i]), "backend does not support MoE node");
            }
            check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "MoE compute failed");
            auto ids_expected = sample["ids"]["values"].get<std::vector<int32_t>>();
            std::vector<int32_t> ids(ids_expected.size());
            auto selected = named.at("ffn_moe_topk");
            for (int64_t t = 0; t < x->ne[1]; ++t) {
                ggml_backend_tensor_get(selected, ids.data() + 3*t, t * selected->nb[1], 3 * sizeof(int32_t));
            }
            if (ids != ids_expected) {
                for (size_t i = 0; i < ids.size(); ++i) std::fprintf(stderr, "route %zu: %d expected %d\n", i, ids[i], ids_expected[i]);
            }
            check(ids == ids_expected, "expert selection differs from official Gate");
            for (const auto & pair : std::vector<std::pair<ggml_tensor *, json>>{
                    {engram.output, engram_case["output"]}, {out, sample["output"]}, {shared, sample["shared_output"]}, {combined, sample["combined_output"]}, {named.at("ffn_moe_weights_scaled"), sample["routing_weights"]}}) {
                auto expected = pair.second["values"].get<std::vector<float>>();
                std::vector<float> actual(expected.size());
                ggml_backend_tensor_get(pair.first, actual.data(), 0, actual.size() * sizeof(float));
                for (size_t i = 0; i < actual.size(); ++i) {
                    max_error = std::max(max_error, double(std::abs(actual[i] - expected[i])));
                    check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= .001f + .001f * std::abs(expected[i]),
                          "MoE differs from official BF16 expert reference");
                }
            }
        }
        std::printf("V4.1 official MoE: %zu cases passed, max error %.9g\n", fixture["cases"].size(), max_error);
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
