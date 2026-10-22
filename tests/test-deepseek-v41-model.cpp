#include "llama-deepseek-v41-model.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <fstream>
#include <memory>
#include <filesystem>
#include <chrono>
#include <stdexcept>

using namespace deepseek_v41;
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template <typename F> static void rejects(F fn) {
    bool rejected = false; try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "invalid V4.1 model binding accepted");
}
static nlohmann::json read(const char * path) { std::ifstream input(path); return nlohmann::json::parse(input); }
int main(int argc, char ** argv) {
    try {
        check(argc == 3, "usage: test-deepseek-v41-model CONFIG.json GGUF_INVENTORY.json");
        const auto source = read(argv[1]);
        const auto config = model_config::from_json(source.dump());
        check(config.attention.compress_ratios.size() == 40 && config.active_experts == 6 && config.experts == 384 &&
              config.index_topk == 512 && config.output_groups == 8, "incorrect source model contract");
        // Exercise the real GGUF metadata loader without allocating weights.
        struct temporary {
            std::filesystem::path path;
            ~temporary() { std::error_code error; std::filesystem::remove(path, error); }
        } file{std::filesystem::temp_directory_path() / ("ese-v41-meta-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf")};
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
        gguf_set_val_str(meta.get(), "general.architecture", "deepseek41");
        gguf_set_val_str(meta.get(), "deepseek41.config", source.dump().c_str());
        const std::map<std::string, uint32_t> fields = {{"block_count", 40}, {"vocab_size", config.vocabulary},
            {"context_length", config.context}, {"embedding_length", config.hidden}, {"attention.head_count", config.heads},
            {"attention.head_count_kv", 1}, {"attention.key_length", 512}, {"attention.value_length", 512},
            {"rope.dimension_count", 64}, {"expert_count", 384}, {"expert_used_count", 6}};
        for (const auto & field : fields) gguf_set_val_u32(meta.get(), ("deepseek41." + field.first).c_str(), field.second);
        gguf_write_to_file(meta.get(), file.path.string().c_str(), false);
        {
            llama_model_loader loader(file.path.string(), 0, false, false, false, false, false, false, false, nullptr, nullptr);
            llama_model model{};
            llm_load_arch(loader, model); llm_load_hparams(loader, model, true);
            check(model.arch == LLM_ARCH_DEEPSEEK41 && model.hparams.n_expert_groups == 0 && model.hparams.n_group_used == 0 &&
                  model.hparams.expert_gating_func == LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS &&
                  model.hparams.n_layer_kv_from_start == 0 && !model.hparams.has_kv(0) && model.hparams.nextn_predict_layers == 0 &&
                  model.hparams.rope_type == LLAMA_ROPE_TYPE_NORM, "native GGUF metadata loader selected incorrect semantics");
        }
        check(llm_arch_from_string("deepseek41") == LLM_ARCH_DEEPSEEK41 &&
              llm_arch_from_string("deepseek4") != LLM_ARCH_DEEPSEEK41, "architecture identities conflated");
        check(llm_tensor_type(LLM_ARCH_DEEPSEEK41, "blk.0.ffn_gate_exps", 0) == LLM_TENSOR_FFN_GATE_EXPS,
              "unsuffixed expert tensor not classified for residency");
        ggml_init_params params{ggml_tensor_overhead() * 2200, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        std::map<std::string, ggml_tensor *> tensors;
        const auto inventory = read(argv[2]).at("tensors");
        for (const auto & item : inventory) {
            const auto shape = item.at("shape").get<std::vector<int64_t>>();
            auto * tensor = ggml_new_tensor(ctx.get(), ggml_type(item.at("type_id").get<int>()), shape.size(), shape.data());
            const auto name = item.at("name").get<std::string>();
            ggml_set_name(tensor, name.c_str()); tensors[name] = tensor;
        }
        auto lookup = [&](const std::string & name) { auto it = tensors.find(name); return it == tensors.end() ? nullptr : it->second; };
        const auto backbone = bind_model_tensors(config, lookup, false);
        check(backbone.size() == inventory.size(), "unbound backbone tensors remain");
        rejects([&] { bind_model_tensors(config, lookup); }); // real mixed-Q2 package omits Engram
        for (const auto & requirement : model_tensor_requirements(config)) if (requirement.engram) {
            auto * tensor = ggml_new_tensor(ctx.get(), GGML_TYPE_F16, requirement.shape.size(), requirement.shape.data());
            ggml_set_name(tensor, (requirement.name + ".weight").c_str());
            tensors[requirement.name + ".weight"] = tensor;
        }
        check(bind_model_tensors(config, lookup).size() == inventory.size() + 6, "complete text binding mismatch");
        auto * q = tensors.at("blk.0.attn_q_b");
        ++q->ne[1]; rejects([&] { bind_model_tensors(config, lookup); }); --q->ne[1];
        auto * duplicate = ggml_dup_tensor(ctx.get(), q);
        tensors["blk.0.attn_q_b.weight"] = duplicate;
        rejects([&] { bind_model_tensors(config, lookup); });
        tensors.erase("blk.0.attn_q_b.weight");
        auto invalid = source; invalid["text_config"]["topk_method"] = "group_limited_greedy";
        rejects([&] { model_config::from_json(invalid.dump()); });
        invalid = source; invalid["text_config"]["num_hidden_layers"] = 43;
        // This would incorrectly include DSpark as ordinary blocks; the three
        // appended ratios have no Engram/top-k issue, so tensor binding must fail.
        const auto wrong = model_config::from_json(invalid.dump());
        rejects([&] { bind_model_tensors(wrong, lookup); });
        std::printf("V4.1 model contract: all %zu backbone tensors bound; missing Engram, dimensions, aliases and routing checks pass\n", backbone.size());
    } catch (const std::exception & error) { std::fprintf(stderr, "%s\n", error.what()); return 1; }
}
