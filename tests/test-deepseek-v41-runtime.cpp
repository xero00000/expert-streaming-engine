#include "llama-deepseek-v41-runtime.h"
#include "llama-model.h"
#include "llama-context.h"
#include "llama-spec-features-dflash.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <nlohmann/json.hpp>
#include <atomic>
#include <filesystem>
#include <chrono>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <stdexcept>
using nlohmann::json;
using namespace deepseek_v41;
static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template <typename F> static void rejects(F f) { bool threw = false; try { f(); } catch (const std::exception &) { threw = true; } check(threw, "invalid operation accepted"); }
class memory_source : public llama_expert_cache::source {
public:
    std::atomic<bool> fail{false};
    uint32_t id() const override { return 41; }
    llama_expert_cache::source_identity identity() const override { return {41, 1}; }
    uint64_t size() const override { return 408 * 80; }
    llama_expert_cache::read_backend backend() const override { return llama_expert_cache::read_backend::pread; }
    void read(uint64_t offset, void * output, size_t bytes) const override {
        if (fail) throw std::runtime_error("injected Engram read failure");
        check(offset <= size() && bytes <= size() - offset, "read outside table");
        auto dst = static_cast<uint8_t *>(output);
        for (size_t i = 0; i < bytes; ++i) {
            const uint64_t at = offset + i, col = at % 80, row = at / 80;
            // Affine2 scales=1, biases=0; packed values vary with row ID.
            dst[i] = col < 64 ? uint8_t((row % 4) * 0x55) : col < 72 ? (col % 2 ? 0x3f : 0x80) : 0;
        }
    }
};
int main(int argc, char ** argv) {
    try {
        check(argc == 2 || argc == 3, "usage: test-deepseek-v41-runtime BLOCK_REFERENCE.json [CUDA_DEVICE]");
        // The V4.1 batch-preserving profile deliberately retains the vector
        // kernels; ordinary CUDA GEMM rounding can change BF16/QAT routing.
        if (argc == 3) {
#ifdef _WIN32
            _putenv_s("ESE_CUDA_MOE_BATCH_VECTOR", "1");
            _putenv_s("ESE_CUDA_DENSE_BATCH_VECTOR", "1");
#else
            setenv("ESE_CUDA_MOE_BATCH_VECTOR", "1", 1);
            setenv("ESE_CUDA_DENSE_BATCH_VECTOR", "1", 1);
#endif
        }
        std::ifstream input(argv[1]); auto fixture = json::parse(input);
        auto config = model_config::from_json(fixture["config"].dump());
        engram_hash_config hash; hash.compressed_vocab_size = config.vocabulary; hash.pad_compressed_id = 2;
        for (uint32_t i = 0; i < config.vocabulary; ++i) hash.token_map.push_back(i);
        auto source = std::make_shared<memory_source>();
        std::array<engram_asset, 2> assets;
        for (size_t i = 0; i < 2; ++i) {
            hash.layers[i].layer_id = config.engram_layers[i]; hash.layers[i].rows = 408;
            hash.layers[i].multipliers = {1, 3, 5, 7}; hash.layers[i].primes.fill(17);
            assets[i] = {source, {0, 408, 80}, engram_row_format::affine2};
        }
        struct temporary_directory {
            std::filesystem::path path;
            ~temporary_directory() { std::error_code error; std::filesystem::remove_all(path, error); }
        } directory{std::filesystem::temp_directory_path() / ("v41-runtime-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))};
        std::filesystem::create_directory(directory.path);
        json metadata = {{"version", 1}, {"hash", {{"compressed_vocab_size", hash.compressed_vocab_size},
            {"pad_compressed_id", hash.pad_compressed_id}, {"token_map", hash.token_map}, {"layers", json::array()}}}, {"rows", json::array()}};
        std::vector<uint8_t> file_data(source->size()); source->read(0, file_data.data(), file_data.size());
        for (size_t i = 0; i < 2; ++i) {
            const auto & h = hash.layers[i];
            const auto name = "rows" + std::to_string(i) + ".bin";
            std::ofstream file(directory.path / name, std::ios::binary);
            file.write(reinterpret_cast<const char *>(file_data.data()), file_data.size());
            check(bool(file), "fixture file write failed");
            metadata["hash"]["layers"].push_back({{"layer_id", h.layer_id}, {"rows", h.rows}, {"primes", h.primes}, {"multipliers", h.multipliers}});
            metadata["rows"].push_back({{"layer_id", h.layer_id}, {"row_count", h.rows}, {"file", name},
                {"encoding", "affine2-g64-bf16"}, {"data_offset", 0}, {"file_bytes", file_data.size()}});
        }
        auto loaded = load_runtime_assets(config, metadata.dump(), directory.path.string());
        {
            runtime_state files(config, config.context, loaded.hash, loaded.rows);
            files.begin(7, 0, 3);
            check(files.decoded_rows().at(config.engram_layers[0]).size() == 6144, "file-backed row decode failed");
            files.abort();
        }
        auto bad = metadata; bad["rows"][0]["file_bytes"] = file_data.size() + 1;
        rejects([&] { load_runtime_assets(config, bad.dump(), directory.path.string()); });
        bad = metadata; bad["rows"][0]["file"] = "../other.bin";
        rejects([&] { load_runtime_assets(config, bad.dump(), directory.path.string()); });
        bad = metadata; bad["hash"]["token_map"][0] = config.vocabulary;
        rejects([&] { load_runtime_assets(config, bad.dump(), directory.path.string()); });
        runtime_state state(config, config.context, hash, assets, 80 * 128);
        source->fail = true;
        rejects([&] { state.begin(7, 0, 3); });
        check(!state.active() && state.position() == 0, "read failure published state");
        source->fail = false;
        rejects([&] { state.begin(7, 1, 3); });
        state.begin(7, 0, 3);
        rejects([&] { state.commit(); }); // no layer has run
        state.abort();
        check(state.position() == 0, "incomplete commit corrupted position");
        llama_model model{}; model.arch = LLM_ARCH_DEEPSEEK41;
        llama_context lctx(model); lctx.cparams = {}; lctx.inp_tokens = nullptr; lctx.inp_embd = nullptr;
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
        ggml_backend_cpu_set_n_threads(cpu.get(), 4);
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu(nullptr, ggml_backend_free);
#ifdef GGML_USE_CUDA
        if (argc == 3) gpu.reset(ggml_backend_cuda_init(std::stoi(argv[2]), nullptr, nullptr));
#endif
        auto backend = gpu ? gpu.get() : cpu.get();
        ggml_init_params wp{ggml_tensor_overhead() * 1024, nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> wc(ggml_init(wp), ggml_free);
        std::map<std::string, ggml_tensor *> weights;
        std::map<ggml_tensor *, std::vector<float>> data;
        for (auto it = fixture["weights"].begin(); it != fixture["weights"].end(); ++it) {
            auto shape = it.value()["shape"].get<std::vector<int64_t>>();
            auto values = it.value()["values"].get<std::vector<float>>();
            if (it.key().find("engram_wkv") != std::string::npos) {
                std::vector<float> wide(6144 * shape[1], 0);
                for (int64_t row = 0; row < shape[1]; ++row) std::copy_n(values.begin() + row * shape[0], shape[0], wide.begin() + row * 6144);
                values = std::move(wide); shape[0] = 6144;
            }
            auto tensor = ggml_new_tensor(wc.get(), GGML_TYPE_F32, shape.size(), shape.data());
            ggml_set_name(tensor, it.key().c_str());
            weights[it.key()] = tensor; data[tensor] = std::move(values);
        }
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> wb(
                ggml_backend_alloc_ctx_tensors(wc.get(), backend), ggml_backend_buffer_free);
        check(bool(wb), "weight allocation failed");
        for (auto & item : data) ggml_backend_tensor_set(item.first, item.second.data(), 0, item.second.size() * sizeof(float));
        std::vector<ggml_backend_t> backends{backend};
        if (gpu) backends.push_back(cpu.get());
        std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched(
                ggml_backend_sched_new(backends.data(), nullptr, backends.size(), 16384, false), ggml_backend_sched_free);
        std::vector<std::vector<float>> first_pass;
        for (int pass = 0; pass < 2; ++pass) {
            state.clear();
            check(state.position() == 0, "clear failed");
            for (uint32_t position = 0; position < 10; ++position) {
                const int attempts = position == 5 ? 3 : 1;
                std::vector<float> before_abort;
                for (int attempt = 0; attempt < attempts; ++attempt) {
                    int32_t tid = fixture["token_steps"][position]["token"];
                    state.begin(7, position, tid);
                    engram_sequence_state oracle(hash);
                    for (uint32_t p = 0; p < position; ++p) oracle.accept(7, p, fixture["token_steps"][p]["token"]);
                    auto expected_rows = oracle.accept(7, position, tid);
                    for (size_t i = 0; i < 2; ++i) for (size_t row = 0; row < 24; ++row) {
                        const auto & values = state.decoded_rows().at(config.engram_layers[i]);
                        for (size_t col = 0; col < 256; ++col) check(values[row * 256 + col] == expected_rows[i][row] % 4,
                                "hash-to-row/decode order mismatch");
                    }
                    rejects([&] { state.begin(8, position, tid); });
                    ggml_init_params gp{ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr, true};
                    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(gp), ggml_free);
                    struct reset_guard { ggml_backend_sched_t sched; ~reset_guard() { ggml_backend_sched_reset(sched); } } reset{sched.get()};
                    auto token = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
                    auto pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1);
                    ggml_set_input(token); ggml_set_input(pos);
                    std::map<uint32_t, ggml_tensor *> row_inputs;
                    for (auto layer : config.engram_layers) { row_inputs[layer] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 6144, 1); ggml_set_input(row_inputs[layer]); }
                    auto graph = ggml_new_graph_custom(ctx.get(), 16384, false);
                    auto output = state.forward().build_token(ctx.get(), lctx, weights, token, pos, row_inputs, graph);
                    check(ggml_backend_sched_alloc_graph(sched.get(), graph), "graph allocation failed");
                    const bool failure = position == 5 && attempt == 1;
                    int32_t pid = position + (failure ? 1 : 0);
                    ggml_backend_tensor_set(token, &tid, 0, sizeof(tid)); ggml_backend_tensor_set(pos, &pid, 0, sizeof(pid));
                    for (auto item : row_inputs) { const auto & values = state.decoded_rows().at(item.first); ggml_backend_tensor_set(item.second, values.data(), 0, values.size() * sizeof(float)); }
                    check(ggml_backend_sched_graph_compute(sched.get(), graph) == GGML_STATUS_SUCCESS, "graph compute failed");
                    ggml_backend_sched_synchronize(sched.get());
                    if (failure) { rejects([&] { state.commit(); }); state.abort(); }
                    else {
                        std::vector<float> logits(config.vocabulary);
                        ggml_backend_tensor_get(output, logits.data(), 0, logits.size() * sizeof(float));
                        for (auto value : logits) check(std::isfinite(value), "nonfinite runtime logits");
                        if (attempt + 1 < attempts) { before_abort = logits; state.abort(); }
                        else {
                            if (position == 5) check(logits == before_abort, "abort changed replay logits");
                            if (pass) check(logits == first_pass[position], "clear/replay changed logits"); else first_pass.push_back(logits);
                            state.commit();
                            check(state.position() == position + 1 && !state.active(), "commit state mismatch");
                            rejects([&] { state.begin(8, position + 1, tid); });
                        }
                    }
                }
            }
        }
        check(state.row_stats(0).reads.source_reads > 0 && state.row_stats(1).reads.source_reads > 0, "row reads not exercised");
        state.begin(7, 10, 3); state.clear();
        check(!state.active() && state.position() == 0, "active clear failed");
        state.begin(8, 0, 3); state.abort();
        // Exercise the public model loader/context/decode API with these same
        // initialized weights and real file-backed Engram rows.
        std::unique_ptr<gguf_context, decltype(&gguf_free)> meta(gguf_init_empty(), gguf_free);
        gguf_set_val_str(meta.get(), "general.architecture", "deepseek41");
        auto native_config = fixture["config"]; native_config["max_position_embeddings"] = 256;
        gguf_set_val_str(meta.get(), "deepseek41.config", native_config.dump().c_str());
        gguf_set_val_str(meta.get(), "deepseek41.assets", metadata.dump().c_str());
        const std::map<std::string, uint32_t> fields = {{"block_count", 8}, {"vocab_size", config.vocabulary},
            {"context_length", 256}, {"embedding_length", config.hidden}, {"feed_forward_length", config.expert_hidden},
            {"attention.head_count", config.heads}, {"attention.head_count_kv", 1},
            {"attention.key_length", config.attention.head_dim}, {"attention.value_length", config.attention.head_dim},
            {"rope.dimension_count", config.rotary}, {"expert_count", config.experts}, {"expert_used_count", config.active_experts}};
        for (auto field : fields) gguf_set_val_u32(meta.get(), ("deepseek41." + field.first).c_str(), field.second);
        gguf_set_val_str(meta.get(), "tokenizer.ggml.model", "llama");
        std::vector<std::string> token_strings;
        std::vector<const char *> token_ptrs;
        for (uint32_t i = 0; i < config.vocabulary; ++i) token_strings.push_back("t" + std::to_string(i));
        for (auto & token : token_strings) token_ptrs.push_back(token.c_str());
        gguf_set_arr_str(meta.get(), "tokenizer.ggml.tokens", token_ptrs.data(), token_ptrs.size());
        std::vector<float> token_scores(config.vocabulary, 0);
        gguf_set_arr_data(meta.get(), "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, token_scores.data(), token_scores.size());
        gguf_set_val_u32(meta.get(), "tokenizer.ggml.unknown_token_id", 0);
        gguf_set_val_u32(meta.get(), "tokenizer.ggml.bos_token_id", 1);
        gguf_set_val_u32(meta.get(), "tokenizer.ggml.eos_token_id", 2);
        for (auto weight : weights) {
            gguf_add_tensor(meta.get(), weight.second);
            const auto & values = data.at(weight.second);
            gguf_set_tensor_data(meta.get(), weight.first.c_str(), values.data(), values.size() * sizeof(float));
        }
        auto model_path = (directory.path / "initialized.gguf").string();
        gguf_write_to_file(meta.get(), model_path.c_str(), false);
        llama_backend_init();
        auto mp = llama_model_default_params(); mp.n_gpu_layers = argc == 3 ? 99 : 0;
        if (argc == 3) {
#ifndef GGML_USE_CUDA
            throw std::runtime_error("CUDA not compiled; refusing GPU fallback");
#else
            mp.main_gpu = std::stoi(argv[2]); mp.split_mode = LLAMA_SPLIT_MODE_NONE;
#endif
        } mp.merge_up_gate_exps = false; mp.merge_qkv = false;
        std::unique_ptr<llama_model, decltype(&llama_free_model)> loaded_model(llama_model_load_from_file(model_path.c_str(), mp), llama_free_model);
        check(bool(loaded_model), "public V4.1 model load failed");
        if (argc == 3) {
            auto cpu_expert_params = mp; cpu_expert_params.ncmoe = config.attention.compress_ratios.size();
            std::unique_ptr<llama_model, decltype(&llama_free_model)> cpu_expert_model(
                    llama_model_load_from_file(model_path.c_str(), cpu_expert_params), llama_free_model);
            check(bool(cpu_expert_model), "CPU-expert V4.1 model load failed");
            for (const auto & layer : cpu_expert_model->layers) {
                for (auto tensor : {layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps})
                    check(ggml_backend_buffer_is_host(tensor->buffer), "unsuffixed V4.1 expert ignored CPU placement");
                check(!ggml_backend_buffer_is_host(layer.ffn_up_shexp->buffer), "CPU expert placement moved shared expert off GPU");
            }
        }
        auto cp = llama_context_default_params(); cp.n_ctx = 256; cp.n_batch = 16; cp.n_ubatch = 8; cp.n_threads = cp.n_threads_batch = 4;
        std::unique_ptr<llama_context, decltype(&llama_free)> native(llama_init_from_model(loaded_model.get(), cp), llama_free);
        check(bool(native), "public V4.1 context creation failed");
        for (auto tensor : native->kv_self.k_l) check(!tensor, "V4.1 allocated ordinary K storage");
        for (auto tensor : native->kv_self.v_l) check(!tensor, "V4.1 allocated ordinary V storage");
        std::vector<llama_token> tokens;
        for (auto step : fixture["token_steps"]) tokens.push_back(step["token"].get<llama_token>());
        for (int pass = 0; pass < 2; ++pass) {
            auto batch = llama_batch_get_one(tokens.data(), tokens.size(), 0, 0);
            std::vector<int8_t> all_logits(tokens.size(), 1); batch.logits = all_logits.data();
            check(llama_decode(native.get(), batch) == 0, "public V4.1 prompt decode failed");
            float native_error = 0;
            for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
                auto logits = llama_get_logits_ith(native.get(), token_index);
                float error = 0;
                for (size_t i = 0; i < config.vocabulary; ++i) error = std::max(error, std::abs(logits[i] - first_pass[token_index][i]));
                std::fprintf(stderr, "Native token %zu logit difference %.9g\n", token_index, error);
                native_error = std::max(native_error, error);
            }
            std::fprintf(stderr, "Native logit max difference: %.9g\n", native_error);
            check(native_error < 1e-5f, "public decode differs from explicit runtime graph");
            check(native->dsv41_runtime->position() == 10, "native prompt position mismatch");
            check(!llama_spec_ckpt_save(native.get(), 0), "unsupported speculative checkpoint accepted");
            check(!llama_spec_ckpt_restore(native.get(), 0, 10, 0), "unsupported speculative restore accepted");
            check(!llama_kv_cache_seq_rm(native.get(), 0, 4, -1), "unsupported suffix removal accepted");
            check(native->dsv41_runtime->position() == 10, "rejected suffix removal changed state");
            llama_token next = 3;
            llama_token aborted_tokens[] = {3, 4, 5, 6};
            llama_set_abort_callback(native.get(), [](void *) { return true; }, nullptr);
            check(llama_decode(native.get(), llama_batch_get_one(aborted_tokens, 4, 10, 0)) != 0, "aborted native graph succeeded");
            check(native->dsv41_runtime->position() == 10 && !native->dsv41_runtime->active(), "native graph abort changed state");
            llama_set_abort_callback(native.get(), nullptr, nullptr);
            check(llama_decode(native.get(), llama_batch_get_one(&next, 1, 10, 0)) == 0, "native abort retry failed");
            check(native->dsv41_runtime->position() == 11, "native token commit failed");
            check(llama_state_get_size(native.get()) == 0, "incomplete private session state advertised");
            check(llama_kv_cache_seq_rm(native.get(), 0, 0, -1), "native full sequence deletion failed");
            check(native->dsv41_runtime->position() == 0, "native full deletion left runtime state");
            llama_kv_cache_clear(native.get());
        }
        check(llama_decode(native.get(), llama_batch_get_one(tokens.data(), tokens.size(), 0, 0)) == 0,
              "last-logit-only prompt decode failed");
        float sparse_error = 0;
        auto sparse_logits = llama_get_logits_ith(native.get(), -1);
        for (size_t i = 0; i < config.vocabulary; ++i)
            sparse_error = std::max(sparse_error, std::abs(sparse_logits[i] - first_pass.back()[i]));
        std::fprintf(stderr, "Sparse output logit max difference: %.9g\n", sparse_error);
        check(sparse_error < 1e-5f, "output selection changed last prompt logits");
        llama_kv_cache_clear(native.get());
        // Cross many window wraps and compression boundaries with the same
        // token stream. Reused compressed rows must never expose future tokens.
        std::vector<llama_token> long_tokens(144);
        for (size_t i = 0; i < long_tokens.size(); ++i) long_tokens[i] = tokens[i % tokens.size()];
        std::vector<float> sequential(long_tokens.size() * config.vocabulary);
        for (size_t i = 0; i < long_tokens.size(); ++i) {
            check(llama_decode(native.get(), llama_batch_get_one(&long_tokens[i], 1, i, 0)) == 0, "long sequential decode failed");
            auto logits = llama_get_logits_ith(native.get(), -1);
            std::copy(logits, logits + config.vocabulary, sequential.data() + i * config.vocabulary);
        }
        llama_kv_cache_clear(native.get());
        float long_error = 0;
        for (size_t i = 0; i < long_tokens.size(); i += 8) {
            auto batch = llama_batch_get_one(long_tokens.data() + i, 8, i, 0);
            int8_t all_logits[8] = {1,1,1,1,1,1,1,1}; batch.logits = all_logits;
            check(llama_decode(native.get(), batch) == 0, "long batched decode failed");
            for (size_t j = 0; j < 8; ++j) {
                auto logits = llama_get_logits_ith(native.get(), j);
                float step_error = 0;
                for (size_t k = 0; k < config.vocabulary; ++k)
                    step_error = std::max(step_error, std::abs(logits[k] - sequential[(i+j)*config.vocabulary+k]));
                if (step_error > long_error) std::fprintf(stderr, "Long batch position %zu difference %.9g\n", i+j, step_error);
                long_error = std::max(long_error, step_error);
            }
        }
        std::fprintf(stderr, "Long batch logit max difference: %.9g\n", long_error);
        check(long_error < 1e-5f, "long batch differs across window/compression boundaries");
        llama_kv_cache_clear(native.get());
        int32_t capture_layers[] = {0, 2, 7};
        check(llama_set_dflash_capture_layers(native.get(), capture_layers, 3), "DSpark capture setup failed");
        std::vector<float> single_features;
        for (uint32_t position = 0; position < 16; ++position) {
            check(llama_decode(native.get(), llama_batch_get_one(long_tokens.data() + position, 1, position, 0)) == 0,
                  "DSpark single-token feature decode failed");
            std::vector<float> features;
            check(llama_spec_copy_dflash_rows_from_output_indices(native.get(), {0}, features) && features.size() == 3 * config.hidden,
                  "DSpark single-token feature shape mismatch");
            const auto & embedding = data.at(weights.at("token_embd"));
            for (uint32_t i = 0; i < config.hidden; ++i) {
                const float expected = ggml_bf16_to_fp32(ggml_fp32_to_bf16(embedding[long_tokens[position] * config.hidden + i]));
                check(features[i] == expected, "DSpark layer-zero feature is not the input stream mean");
            }
            single_features.insert(single_features.end(), features.begin(), features.end());
        }
        llama_kv_cache_clear(native.get());
        auto feature_batch = llama_batch_get_one(long_tokens.data(), 16, 0, 0);
        int8_t feature_logits[16]; std::fill_n(feature_logits, 16, 1); feature_batch.logits = feature_logits;
        check(llama_decode(native.get(), feature_batch) == 0, "DSpark feature batch decode failed");
        std::vector<int32_t> feature_rows(16);
        for (int i = 0; i < 16; ++i) feature_rows[i] = i;
        std::vector<float> batched_features;
        check(llama_spec_copy_dflash_rows_from_output_indices(native.get(), feature_rows, batched_features),
              "DSpark batched feature readback failed");
        check(batched_features == single_features, "DSpark feature rows changed under microbatching");
        for (int row = 0; row < 16; ++row) {
            const auto logits = llama_get_logits_ith(native.get(), row);
            for (uint32_t i = 0; i < config.vocabulary; ++i)
                check(std::abs(logits[i] - sequential[row * config.vocabulary + i]) < 1e-5f,
                      "DSpark feature capture changed target logits");
        }
        llama_clear_dflash_capture(native.get());
        llama_kv_cache_clear(native.get());
        std::puts("DSpark input-stream feature capture: exact rows and target-logit parity passed");
        // Every accepted prefix must match ordinary decoding, including the
        // half-finished compressor pair and the Engram n-gram history.
        for (uint32_t checkpoint_position : {0u, 5u, 127u}) {
            for (int accepted = 0; accepted <= 8; ++accepted) {
                if (checkpoint_position == 127 && accepted != 0 && accepted != 1 && accepted != 3 && accepted != 8) continue;
                llama_kv_cache_clear(native.get());
                for (uint32_t start = 0; start < checkpoint_position; start += 8) {
                    const uint32_t count = std::min(8u, checkpoint_position - start);
                    check(llama_decode(native.get(), llama_batch_get_one(long_tokens.data() + start, count, start, 0)) == 0,
                          "checkpoint prefix prefill failed");
                }
                check(llama_spec_ckpt_init(native.get(), LLAMA_SPEC_CKPT_AUTO, 8) == LLAMA_SPEC_CKPT_PER_STEP,
                      "V4.1 checkpoint initialization failed");
                check(llama_spec_ckpt_save(native.get(), 0), "V4.1 checkpoint save failed");
                auto batch = llama_batch_get_one(long_tokens.data() + checkpoint_position, 8, checkpoint_position, 0);
                int8_t all_logits[8] = {1,1,1,1,1,1,1,1}; batch.logits = all_logits;
                check(llama_decode(native.get(), batch) == 0, "V4.1 verification batch failed");
                check(!llama_spec_ckpt_restore(native.get(), 1, checkpoint_position, accepted - 1),
                      "checkpoint restore accepted a different sequence");
                check(llama_spec_ckpt_restore_ex(native.get(), 0, checkpoint_position, accepted - 1) == LLAMA_SPEC_CKPT_RESTORE_DIRECT,
                      "V4.1 accepted prefix restore failed");
                const uint32_t next_position = checkpoint_position + accepted;
                check(native->dsv41_runtime->position() == next_position, "checkpoint restored wrong runtime position");
                check(llama_decode(native.get(), llama_batch_get_one(long_tokens.data() + next_position, 1, next_position, 0)) == 0,
                      "decoding after accepted prefix failed");
                const auto logits = llama_get_logits_ith(native.get(), -1);
                float error = 0;
                for (size_t i = 0; i < config.vocabulary; ++i)
                    error = std::max(error, std::abs(logits[i] - sequential[next_position * config.vocabulary + i]));
                std::fprintf(stderr, "Checkpoint at %u accepting %d: next-logit difference %.9g\n", checkpoint_position, accepted, error);
                check(error < 1e-5f, "accepted-prefix restore changed next-token logits");
                llama_spec_ckpt_discard(native.get());
            }
        }
        llama_kv_cache_clear(native.get());
        // A head override must keep its F32 projection on that backend even
        // when the transformer runs on CUDA; otherwise the cast migrates back
        // to the last layer's device and consumes several GiB in real models.
        llama_model_tensor_buft_override head_overrides[] = {
            {"^output$", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        auto head_params = mp; head_params.tensor_buft_overrides = head_overrides;
        std::unique_ptr<llama_model, decltype(&llama_free_model)> head_model(
                llama_model_load_from_file(model_path.c_str(), head_params), llama_free_model);
        check(bool(head_model), "CPU-head V4.1 model load failed");
        struct head_observation { bool seen = false; bool host = true; } observed;
        auto head_cp = cp;
        head_cp.cb_eval_user_data = &observed;
        head_cp.cb_eval = [](ggml_tensor * tensor, bool ask, void * userdata) -> int {
            const bool head = std::strcmp(ggml_get_name(tensor), "result_output") == 0;
            if (ask) return head;
            if (head) {
                auto & result = *static_cast<head_observation *>(userdata);
                result.seen = true;
                result.host = result.host && tensor->buffer && ggml_backend_buffer_is_host(tensor->buffer);
            }
            return true;
        };
        std::unique_ptr<llama_context, decltype(&llama_free)> head_context(
                llama_init_from_model(head_model.get(), head_cp), llama_free);
        check(bool(head_context), "CPU-head context creation failed");
        auto head_batch = llama_batch_get_one(tokens.data(), tokens.size(), 0, 0);
        std::vector<int8_t> head_logits(tokens.size(), 1); head_batch.logits = head_logits.data();
        check(llama_decode(head_context.get(), head_batch) == 0, "CPU-head decode failed");
        check(observed.seen && observed.host, "output override did not retain head compute on CPU");
        for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
            auto logits = llama_get_logits_ith(head_context.get(), token_index);
            for (size_t i = 0; i < config.vocabulary; ++i)
                check(std::abs(logits[i] - first_pass[token_index][i]) < 1e-5f, "head placement changed logits");
        }
        std::puts("V4.1 runtime: row I/O failure, ordered decoding, commit, abort, failed graph, sequence binding, clear/replay and head placement passed");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
