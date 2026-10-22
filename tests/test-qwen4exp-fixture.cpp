// Tiny synthetic qwen4exp GGUF fixture (roadmap A4 / plan Phase 0.3).
//
// Purpose: prove the experimental qwen4exp metadata path can be exercised
// without the 58 GiB production quant. This test *generates* a small,
// self-consistent qwen4exp GGUF with the same operators/tensor families the
// production artifact declares, then drives the real loader metadata path
// (`llama_model_loader` + `llm_load_arch` + `llm_load_hparams`).
//
// It deliberately does NOT call the public `llama_model_load_from_file` to
// obtain a model: the production architecture gate must stay closed. The one
// public-API call below is a negative control that asserts the fixture is still
// rejected when the diagnostic opt-in is absent.
//
// Structural constants (PLE head tables, layer multipliers, the recurrent /
// full-attention schedule) are compiled into the runtime hparams, so a
// faithful fixture keeps them exact and only shrinks the free dimensions.

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-arch.h"
#include "llama-model.h"
#include "llama-model-loader.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- fixed structural constants (must match the compiled hparams) ----
constexpr uint32_t kLayers               = 4;
constexpr uint32_t kFullAttnInterval     = 4;   // layer (i+1)%4==0 is full attention
constexpr uint32_t kPleNgram             = 3;
constexpr uint32_t kPleHeadsPerNgram     = 8;
constexpr uint32_t kPleHeadTableEntries  = 16;  // heads_per_ngram * (ngram - 1)
constexpr uint32_t kRowsPerHead          = 5;
constexpr uint32_t kPleRows              = kPleHeadTableEntries * kRowsPerHead; // 80

// ---- tiny free geometry ----
constexpr uint32_t kNEmbd         = 64;
constexpr uint32_t kNEmbdPerLayer = 16;
constexpr uint32_t kNHeads        = 4;
constexpr uint32_t kNHeadsKv      = 2;
constexpr uint32_t kHeadDim       = 16;  // kNEmbd / kNHeads, and key/value length
constexpr uint32_t kNExpert       = 4;
constexpr uint32_t kNExpertUsed   = 2;
constexpr uint32_t kHcCount       = 2;
constexpr uint32_t kHcLowRank     = 8;

struct fixture_options {
    std::vector<uint32_t> ratios            = {0, 0, 0, 4};
    uint64_t              table_rows        = kPleRows;   // PLE table ne[1]
    bool                  contiguous_offsets = true;
    bool                  valid_ple_layer    = true;       // false => ple.layers out of range
    std::vector<std::pair<std::string, uint32_t>> overrides;
    std::vector<std::pair<std::string, float>> float_overrides;
};

void check(bool ok, const std::string & why) {
    if (!ok) throw std::runtime_error(why);
}

template <typename F>
bool rejects(F fn) {
    try {
        fn();
    } catch (const std::exception &) {
        return true;
    }
    return false;
}

struct scratch_directory {
    std::filesystem::path path;
    explicit scratch_directory(const std::string & tag) {
        path = std::filesystem::temp_directory_path() /
            (tag + "-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(path);
    }
    ~scratch_directory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

std::string write_fixture(const std::filesystem::path & dir, const fixture_options & opt) {
    ggml_init_params tensor_params{ggml_tensor_overhead() * 8, nullptr, /*no_alloc=*/true};
    ggml_context * tensors = ggml_init(tensor_params);
    check(tensors != nullptr, "ggml context for fixture tensors");
    ggml_tensor * ple_table = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, kNEmbdPerLayer,
            (int64_t) opt.table_rows);
    ggml_set_name(ple_table, "per_layer_token_embd.weight");

    gguf_context * meta = gguf_init_empty();
    check(meta != nullptr, "gguf context for fixture");

    gguf_set_val_str(meta, "general.architecture", "qwen4exp");
    gguf_set_val_str(meta, "general.name", "qwen4exp-synthetic-fixture");

    // generic geometry
    gguf_set_val_u32(meta, "qwen4exp.block_count",           kLayers);
    gguf_set_val_u32(meta, "qwen4exp.vocab_size",            128);
    gguf_set_val_u32(meta, "qwen4exp.context_length",        512);
    gguf_set_val_u32(meta, "qwen4exp.embedding_length",      kNEmbd);
    gguf_set_val_u32(meta, "qwen4exp.attention.head_count",  kNHeads);
    gguf_set_val_u32(meta, "qwen4exp.attention.head_count_kv", kNHeadsKv);
    gguf_set_val_u32(meta, "qwen4exp.attention.key_length",  kHeadDim);
    gguf_set_val_u32(meta, "qwen4exp.attention.value_length", kHeadDim);
    gguf_set_val_u32(meta, "qwen4exp.rope.dimension_count",  kHeadDim);
    gguf_set_val_u32(meta, "qwen4exp.expert_count",          kNExpert);
    gguf_set_val_u32(meta, "qwen4exp.expert_used_count",     kNExpertUsed);

    // model-specific geometry
    gguf_set_val_u32(meta, "qwen4exp.expert_feed_forward_length",        32);
    gguf_set_val_u32(meta, "qwen4exp.expert_shared_feed_forward_length", 16);
    gguf_set_val_f32(meta, "qwen4exp.attention.layer_norm_rms_epsilon",  1e-6f);

    const int32_t rope_sections[4] = {8, 4, 2, 2};
    gguf_set_arr_data(meta, "qwen4exp.rope.dimension_sections", GGUF_TYPE_INT32, rope_sections, 4);

    gguf_set_val_u32(meta, "qwen4exp.ssm.conv_kernel",    4);
    gguf_set_val_u32(meta, "qwen4exp.ssm.inner_size",     64);
    gguf_set_val_u32(meta, "qwen4exp.ssm.state_size",     8);
    gguf_set_val_u32(meta, "qwen4exp.ssm.time_step_rank", 8);
    gguf_set_val_u32(meta, "qwen4exp.ssm.group_count",    2);

    gguf_set_val_u32(meta, "qwen4exp.attention.indexer.head_count",  2);
    gguf_set_val_u32(meta, "qwen4exp.attention.indexer.key_length",  8);
    gguf_set_val_u32(meta, "qwen4exp.attention.indexer.top_k",       8);
    gguf_set_val_u32(meta, "qwen4exp.full_attention_interval",       kFullAttnInterval);
    gguf_set_val_u32(meta, "qwen4exp.hyper_connection.count",        kHcCount);
    gguf_set_val_u32(meta, "qwen4exp.hyper_connection.low_rank",     kHcLowRank);
    gguf_set_val_u32(meta, "qwen4exp.embedding_length_per_layer_input", kNEmbdPerLayer);

    gguf_set_arr_data(meta, "qwen4exp.attention.compress_ratios", GGUF_TYPE_UINT32,
            opt.ratios.data(), opt.ratios.size());

    gguf_set_val_u32(meta, "qwen4exp.ple.conv_kernel",     4);
    gguf_set_val_u32(meta, "qwen4exp.ple.eos_token_id",    248044);
    gguf_set_val_u32(meta, "qwen4exp.ple.image_token_id",  248042);
    gguf_set_val_u32(meta, "qwen4exp.ple.heads_per_ngram", kPleHeadsPerNgram);
    gguf_set_val_u32(meta, "qwen4exp.ple.ngram_size",      kPleNgram);

    uint64_t offsets[kPleHeadTableEntries] = {};
    uint64_t sizes[kPleHeadTableEntries]   = {};
    uint64_t populated = 0;
    for (uint32_t head = 0; head < kPleHeadTableEntries; ++head) {
        offsets[head] = populated;
        if (!opt.contiguous_offsets && head == 0) {
            offsets[head] = 1; // break the "offsets start at 0 and stay contiguous" rule
        }
        sizes[head] = kRowsPerHead;
        populated += kRowsPerHead;
    }
    gguf_set_arr_data(meta, "qwen4exp.ple.head_offsets",     GGUF_TYPE_UINT64, offsets, kPleHeadTableEntries);
    gguf_set_arr_data(meta, "qwen4exp.ple.head_vocab_sizes", GGUF_TYPE_UINT64, sizes,   kPleHeadTableEntries);
    const uint64_t multipliers[3] = {1, 2, 4};
    gguf_set_arr_data(meta, "qwen4exp.ple.layer_multipliers", GGUF_TYPE_UINT64, multipliers, 3);
    const uint32_t ple_layers[1] = {opt.valid_ple_layer ? 1u : kLayers};
    gguf_set_arr_data(meta, "qwen4exp.ple.layers", GGUF_TYPE_UINT32, ple_layers, 1);

    gguf_set_val_u32(meta, "tokenizer.ggml.eos_token_id", 248046);
    gguf_set_val_u32(meta, "tokenizer.ggml.bos_token_id", 248044);

    gguf_add_tensor(meta, ple_table);
    std::vector<float> table_data((size_t) kNEmbdPerLayer * opt.table_rows, 0.0f);
    for (size_t i = 0; i < table_data.size(); ++i) {
        table_data[i] = 0.001f * float(i % 97);
    }
    gguf_set_tensor_data(meta, "per_layer_token_embd.weight",
            table_data.data(), table_data.size() * sizeof(float));
    for (const auto & item : opt.overrides) gguf_set_val_u32(meta, item.first.c_str(), item.second);
    for (const auto & item : opt.float_overrides) gguf_set_val_f32(meta, item.first.c_str(), item.second);

    const auto path = (dir / "qwen4exp-fixture.gguf").string();
    gguf_write_to_file(meta, path.c_str(), /*only_meta=*/false);

    gguf_free(meta);
    ggml_free(tensors);
    return path;
}

void load_metadata(const std::string & path, llama_model & model) {
    llama_model_loader loader(path, 0, false, false, false, false, false, false, false, nullptr, nullptr);
    // The production rejection lives inside llm_load_arch, so a fixture test that
    // called it could only run in a probe-gate build. Set the architecture
    // directly and exercise the metadata/schema path, which is what a tiny
    // synthetic fixture is for; the gate itself is covered by
    // test-qwen4exp-probe. This keeps the test runnable in every build.
    model.arch = LLM_ARCH_QWEN4EXP;
    llm_load_hparams(loader, model, /*ignore_vocab=*/true);
}

// Faults affect only the in-memory metadata owned by this loader. The original
// GGUF files remain read-only and their weight payloads are never materialized.
int create_artifact_tensors(const std::string & path, const std::string & name = {},
        int dimension = -1, bool missing = false, bool extra = false) {
    llama_model model{};
    llama_model_loader loader(path, 0, false, false, false, false,
            false, false, false, nullptr, nullptr);
    model.arch = LLM_ARCH_QWEN4EXP;
    ggml_tensor added{};
    if (extra) {
        auto weight = loader.weights.front();
        added = *weight.tensor;
        ggml_set_name(&added, name.c_str());
        weight.tensor = &added;
        loader.weights.push_back(weight);
        std::sort(loader.weights.begin(), loader.weights.end(), [](const auto & a, const auto & b) {
            return std::string(ggml_get_name(a.tensor)) < ggml_get_name(b.tensor);
        });
        ++loader.n_tensors;
    } else if (!name.empty()) {
        auto found = std::find_if(loader.weights.begin(), loader.weights.end(), [&](const auto & weight) {
            return name == ggml_get_name(weight.tensor);
        });
        check(found != loader.weights.end(), "fault control tensor missing: " + name);
        if (missing) loader.weights.erase(found);
        else {
            check(dimension >= 0 && dimension < GGML_MAX_DIMS, "fault control dimension");
            if (name == "per_layer_token_embd.weight" && dimension == 1) {
                // Extra table rows are valid; make this table too small to
                // cover its configured head ranges instead.
                found->tensor->ne[dimension] = 1;
            } else ++found->tensor->ne[dimension];
        }
    }
    llm_load_hparams(loader, model, true);
    const auto cpu = ggml_backend_cpu_buffer_type();
    model.split_mode = LLAMA_SPLIT_MODE_LAYER;
    model.mtp = false;
    model.buft_input = llama_model::layer_buft(cpu);
    model.buft_output = llama_model::layer_buft(cpu);
    model.buft_layer.assign(model.hparams.n_layer, llama_model::layer_buft(cpu));
    auto helper = create_tensors_helper_interface::instance(loader, model);
    helper->create_tensors();
    loader.done_getting_tensors();
    check(model.bufs.empty(), "tensor-only check allocated weight buffers");
    check(model.qwen4exp_ple_source.valid, "tensor-only check lost PLE source");
    check(loader.n_created == loader.n_tensors, "tensor-only inventory mismatch");
    return loader.n_created;
}

void capture_log(enum ggml_log_level, const char * text, void * user_data) {
    static_cast<std::string *>(user_data)->append(text);
}

} // namespace

int main(int argc, char ** argv) {
    try {
        scratch_directory dir("ese-qwen4exp-fixture");
        if (argc == 3 && (std::string(argv[1]) == "--artifact-tensors" ||
                         std::string(argv[1]) == "--artifact-tensor-faults")) {
            // Exercise production tensor creation without allocating/loading
            // weights or bypassing the public architecture gate.
            llama_backend_init();
            const int count = create_artifact_tensors(argv[2]);
            std::printf("artifact_tensors: tensors=%d created=%d weight_buffers=0\n", count, count);
            if (std::string(argv[1]) == "--artifact-tensor-faults") {
                std::map<std::string, std::string> families;
                {
                    llama_model_loader loader(argv[2], 0, false, false, false, false,
                            false, false, false, nullptr, nullptr);
                    for (const auto & weight : loader.weights) {
                        const std::string name = ggml_get_name(weight.tensor);
                        const size_t start = name.rfind("blk.", 0) == 0 ? name.find('.', 4) + 1 : 0;
                        families.emplace(name.substr(start), name);
                    }
                }
                size_t failures = 0;
                for (const auto & family : families) {
                    for (int dimension = -1; dimension < GGML_MAX_DIMS; ++dimension) {
                        bool rejected = false;
                        try {
                            create_artifact_tensors(argv[2], family.second, dimension, dimension < 0);
                        } catch (const std::exception & error) {
                            const std::string message = error.what();
                            rejected = message.find(family.second) != std::string::npos &&
                                message.find(dimension < 0 ? "not found" : "wrong shape") != std::string::npos;
                            if (family.first == "per_layer_token_embd.weight") {
                                rejected = message.find("qwen4exp PLE tensor") != std::string::npos ||
                                    message.find("Qwen4Exp PLE table shape") != std::string::npos;
                            }
                            check(rejected, "unexpected tensor fault rejection: " + message);
                        }
                        check(rejected, "tensor fault accepted: " + family.second);
                        ++failures;
                        std::printf("rejected: %s %s %d\n", family.second.c_str(),
                                dimension < 0 ? "missing" : "dimension", dimension);
                    }
                }
                for (const char * name : {"blk.0.attn_q.weight", "blk.3.ssm_out.weight",
                        "blk.0.ple_key.weight", "unused.weight"}) {
                    bool rejected = false;
                    try { create_artifact_tensors(argv[2], name, -1, false, true); }
                    catch (const std::exception & error) {
                        rejected = std::string(error.what()).find("wrong number of tensors") != std::string::npos;
                        check(rejected, std::string("unexpected extra-family rejection: ") + error.what());
                    }
                    check(rejected, std::string("extra tensor family accepted: ") + name);
                    ++failures;
                    std::printf("rejected: extra %s\n", name);
                }
                std::printf("artifact_tensor_faults: families=%zu rejections=%zu original_tensors=%d\n",
                        families.size(), failures, count);
            }
            llama_backend_free();
            return 0;
        }
        if (argc == 3 && (std::string(argv[1]) == "--artifact-hparams" ||
                          std::string(argv[1]) == "--generic-hparams")) {
            llama_model model;
            if (std::string(argv[1]) == "--generic-hparams") {
                llama_model_loader loader(argv[2], 0, false, false, false, false, false, false, false, nullptr, nullptr);
                model.arch = llm_arch_from_string(loader.arch_name);
                llm_load_hparams(loader, model, true);
            } else {
                load_metadata(argv[2], model);
            }
            std::printf("artifact_hparams: layers=%u experts=%u used=%u recurrent_width=%u\n",
                        model.hparams.n_layer, model.hparams.n_expert, model.hparams.n_expert_used,
                        model.hparams.n_embd_v_s());
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--reject-metadata") {
            fixture_options opt;
            opt.overrides.emplace_back(argv[2], uint32_t(std::stoull(argv[3])));
            llama_model rejected;
            check(rejects([&] { load_metadata(write_fixture(dir.path, opt), rejected); }),
                  std::string("malformed metadata accepted: ") + argv[2]);
            std::puts("PASS: malformed metadata rejected without abort");
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--reject-float-metadata") {
            fixture_options opt;
            opt.float_overrides.emplace_back(argv[2], std::stof(argv[3]));
            llama_model rejected;
            check(rejects([&] { load_metadata(write_fixture(dir.path, opt), rejected); }),
                  std::string("malformed floating-point metadata accepted: ") + argv[2]);
            std::puts("PASS: malformed floating-point metadata rejected without abort");
            return 0;
        }

        // ---- the well-formed fixture loads through the real metadata path ----
        llama_model model;
        load_metadata(write_fixture(dir.path, {}), model);

        const auto & hparams = model.hparams;
        check(model.arch == LLM_ARCH_QWEN4EXP, "fixture did not select LLM_ARCH_QWEN4EXP");
        check(llm_arch_from_string("qwen4exp") == LLM_ARCH_QWEN4EXP,
                "architecture string mapping is broken");
        check(hparams.n_layer == kLayers, "fixture layer count mismatch");
        check(hparams.n_embd == kNEmbd, "fixture embedding length mismatch");
        check(hparams.n_head() == kNHeads, "fixture head count mismatch");
        check(hparams.n_head_kv(0) == kNHeadsKv, "fixture kv head count mismatch");
        check(hparams.n_expert == kNExpert && hparams.n_expert_used == kNExpertUsed,
                "fixture expert counts mismatch");
        check(hparams.n_ff_exp == 32 && hparams.n_ff_shexp == 16, "fixture expert ffn lengths mismatch");

        check(hparams.qwen4exp_full_attention_interval == kFullAttnInterval,
                "fixture full-attention interval mismatch");
        for (uint32_t layer = 0; layer < kLayers; ++layer) {
            const uint32_t expected = (layer + 1) % kFullAttnInterval == 0 ? 4u : 0u;
            check(hparams.qwen4exp_compress_ratios[layer] == expected,
                    "fixture compress ratio mismatch at layer " + std::to_string(layer));
            check(hparams.is_recurrent(layer) == (expected == 0),
                    "fixture recurrent schedule mismatch at layer " + std::to_string(layer));
        }
        check(hparams.qwen4exp_ple_layers[1] && !hparams.qwen4exp_ple_layers[0] &&
                !hparams.qwen4exp_ple_layers[2] && !hparams.qwen4exp_ple_layers[3],
                "fixture PLE layer selection mismatch");
        check(hparams.qwen4exp_ple_ngram_size == kPleNgram &&
                hparams.qwen4exp_ple_heads_per_ngram == kPleHeadsPerNgram,
                "fixture PLE n-gram geometry mismatch");
        check(hparams.qwen4exp_ple_head_offsets[0] == 0 &&
                hparams.qwen4exp_ple_head_offsets[15] == 15 * kRowsPerHead &&
                hparams.qwen4exp_ple_head_vocab_sizes[15] == kRowsPerHead,
                "fixture PLE head ranges mismatch");
        check(hparams.qwen4exp_hc_count == kHcCount && hparams.qwen4exp_hc_low_rank == kHcLowRank &&
                hparams.qwen4exp_embd_per_layer_input == kNEmbdPerLayer,
                "fixture hyper-connection geometry mismatch");
        check(hparams.qwen4exp_ple_conv_state_size(1) ==
                kNEmbd * kHcCount * (4 - 1) * kPleNgram,
                "fixture PLE convolution state geometry mismatch");
        check(hparams.qwen4exp_ple_conv_state_size(0) == 0,
                "fixture PLE state should be zero on non-PLE layers");

        std::printf("fixture loaded: %u layers, %u experts (top %u), PLE table %u x %u\n",
                hparams.n_layer, hparams.n_expert, hparams.n_expert_used, kNEmbdPerLayer, kPleRows);

        // ---- malformed fixtures must be rejected by the same path ----
        struct rejection_case {
            const char * name;
            fixture_options opt;
        };
        const std::vector<rejection_case> bad = {
            {"compress_ratios count",   {{0, 0, 0}, kPleRows, true, true}},
            {"compress_ratios value",   {{0, 0, 4, 0}, kPleRows, true, true}},
            {"PLE head offsets",        {{0, 0, 0, 4}, kPleRows, false, true}},
            {"PLE table too small",     {{0, 0, 0, 4}, 40, true, true}},
            {"PLE layer out of range",  {{0, 0, 0, 4}, kPleRows, true, false}},
        };
        for (const auto & test : bad) {
            const std::string path = write_fixture(dir.path, test.opt);
            llama_model rejected;
            check(rejects([&] { load_metadata(path, rejected); }),
                    std::string("malformed fixture accepted: ") + test.name);
        }
        std::printf("negative cases rejected: %zu\n", bad.size());
        const std::vector<std::pair<const char *, uint32_t>> invalid_metadata = {
            {"qwen4exp.expert_count", 513}, {"qwen4exp.expert_used_count", 5},
            {"qwen4exp.expert_count", 0}, {"qwen4exp.expert_used_count", 0},
            {"qwen4exp.block_count", 0}, {"qwen4exp.block_count", 513},
            {"qwen4exp.context_length", 0}, {"qwen4exp.embedding_length", 0},
            {"qwen4exp.vocab_size", 0}, {"qwen4exp.expert_feed_forward_length", 0},
            {"qwen4exp.expert_shared_feed_forward_length", 0},
            {"qwen4exp.ssm.conv_kernel", 0}, {"qwen4exp.ssm.inner_size", 0},
            {"qwen4exp.ssm.state_size", 0}, {"qwen4exp.ssm.time_step_rank", 0},
            {"qwen4exp.ssm.group_count", 0}, {"qwen4exp.ssm.time_step_rank", 3},
            {"qwen4exp.ssm.group_count", 3}, {"qwen4exp.ssm.inner_size", 65},
            {"qwen4exp.ssm.state_size", 16}, {"qwen4exp.ple.conv_kernel", 0},
            {"qwen4exp.ple.conv_kernel", 1},
            {"qwen4exp.attention.head_count", 0}, {"qwen4exp.attention.head_count_kv", 0},
            {"qwen4exp.attention.key_length", 0}, {"qwen4exp.attention.value_length", 0},
            {"qwen4exp.attention.indexer.head_count", 0},
            {"qwen4exp.attention.indexer.key_length", 0}, {"qwen4exp.attention.indexer.top_k", 0},
            {"qwen4exp.hyper_connection.count", 0}, {"qwen4exp.hyper_connection.low_rank", 0},
            {"qwen4exp.ple.heads_per_ngram", 0x80000008u}, // u32 product with 2 wraps to 16
            {"qwen4exp.ssm.conv_kernel", UINT32_MAX},
            {"qwen4exp.hyper_connection.count", UINT32_MAX},
            {"qwen4exp.ple.conv_kernel", UINT32_MAX},
        };
        for (const auto & item : invalid_metadata) {
            fixture_options opt;
            opt.overrides.push_back(item);
            llama_model rejected;
            check(rejects([&] { load_metadata(write_fixture(dir.path, opt), rejected); }),
                  std::string("malformed metadata accepted: ") + item.first + "=" + std::to_string(item.second));
        }
        std::printf("invalid scalar metadata rejected without abort: %zu\n", invalid_metadata.size());

        const char * rms_key = "qwen4exp.attention.layer_norm_rms_epsilon";
        for (float epsilon : {0.0f, -1.0f, std::numeric_limits<float>::infinity(),
                -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            fixture_options opt;
            opt.float_overrides.emplace_back(rms_key, epsilon);
            llama_model rejected;
            check(rejects([&] { load_metadata(write_fixture(dir.path, opt), rejected); }),
                  "invalid RMS epsilon accepted");
        }
        // Positive epsilon and the smallest supported PLE history remain valid.
        for (float epsilon : {1.0e-8f, 1.0e-3f}) {
            fixture_options opt;
            opt.float_overrides.emplace_back(rms_key, epsilon);
            opt.overrides.emplace_back("qwen4exp.ple.conv_kernel", 2);
            llama_model accepted;
            load_metadata(write_fixture(dir.path, opt), accepted);
            check(accepted.hparams.f_norm_rms_eps == epsilon &&
                    accepted.hparams.qwen4exp_ple_conv_state_size(1) ==
                        kNEmbd * kHcCount * kPleNgram,
                    "valid RMS epsilon or minimal PLE history changed");
        }
        std::puts("invalid RMS epsilon rejected: 5; valid epsilon/minimal-history controls: 2");

        // ---- the production gate stays closed on the same fixture ----
        if (std::getenv("ESE_QWEN4EXP_PROBE_GATE") == nullptr) {
            std::string logs;
            llama_log_set(capture_log, &logs);
            llama_model * loaded = llama_model_load_from_file(
                    (dir.path / "qwen4exp-fixture.gguf").string().c_str(),
                    llama_model_default_params());
            llama_log_set(nullptr, nullptr);
            check(loaded == nullptr, "production loader accepted the qwen4exp fixture");
            check(logs.find("unsupported architecture: qwen4exp") != std::string::npos,
                    "production loader did not report the qwen4exp rejection: " + logs);
            std::printf("production loader still rejects the fixture (gate closed)\n");
        }

        std::printf("PASS: synthetic qwen4exp fixture loads via the metadata path without the "
                "58 GiB quant and without opening the production gate\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
