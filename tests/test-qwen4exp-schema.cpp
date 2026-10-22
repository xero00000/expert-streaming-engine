#include "llama-arch.h"
#include "llama-model.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>

using json = nlohmann::json;

static int layer_from_name(const std::string & name) {
    if (name.rfind("blk.", 0) != 0) return -1;
    const size_t end = name.find('.', 4);
    if (end == std::string::npos) return -1;
    return std::stoi(name.substr(4, end - 4));
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s QWEN4EXP_MANIFEST.json\n", argv[0]);
        return 1;
    }
    std::ifstream input(argv[1]);
    if (!input) return 1;
    const json manifest = json::parse(input);
    const json & metadata = manifest.at("metadata");

    const LLM_KV kv(LLM_ARCH_QWEN4EXP);
    if (kv(LLM_KV_QWEN4EXP_PLE_EOS_TOKEN_ID) != "qwen4exp.ple.eos_token_id" ||
            kv(LLM_KV_HYPER_CONNECTION_LOW_RANK) != "qwen4exp.hyper_connection.low_rank" ||
            kv(LLM_KV_QWEN4EXP_EMBEDDING_LENGTH_PER_LAYER_INPUT) !=
                "qwen4exp.embedding_length_per_layer_input") {
        std::fprintf(stderr, "Qwen4Exp metadata registry is incomplete\n");
        return 1;
    }

    if (manifest.at("architecture") != "qwen4exp" || manifest.at("tensor_count") != 1224 ||
            metadata.at("qwen4exp.block_count") != 48 ||
            metadata.at("qwen4exp.context_length") != 262144 ||
            metadata.at("qwen4exp.embedding_length") != 2560 ||
            metadata.at("qwen4exp.embedding_length_per_layer_input") != 160 ||
            metadata.at("qwen4exp.attention.head_count") != 24 ||
            metadata.at("qwen4exp.attention.head_count_kv") != 2 ||
            metadata.at("qwen4exp.attention.key_length") != 256 ||
            metadata.at("qwen4exp.attention.value_length") != 256 ||
            metadata.at("qwen4exp.attention.indexer.head_count") != 4 ||
            metadata.at("qwen4exp.attention.indexer.key_length") != 128 ||
            metadata.at("qwen4exp.attention.indexer.top_k") != 2048 ||
            metadata.at("qwen4exp.expert_count") != 512 ||
            metadata.at("qwen4exp.expert_used_count") != 10 ||
            metadata.at("qwen4exp.hyper_connection.count") != 4 ||
            metadata.at("qwen4exp.hyper_connection.low_rank") != 320) {
        std::fprintf(stderr, "Qwen4Exp production metadata does not match the pinned schema\n");
        return 1;
    }

    const json & ratios = metadata.at("qwen4exp.attention.compress_ratios");
    if (ratios.size() != 48) return 1;
    for (size_t layer = 0; layer < ratios.size(); ++layer) {
        const uint32_t expected = (layer + 1) % 4 == 0 ? 4 : 0;
        if (ratios.at(layer) != expected) {
            std::fprintf(stderr, "Qwen4Exp attention schedule mismatch at layer %zu\n", layer);
            return 1;
        }
    }

    const json & sizes = metadata.at("qwen4exp.ple.head_vocab_sizes");
    const json & offsets = metadata.at("qwen4exp.ple.head_offsets");
    const json & multipliers = metadata.at("qwen4exp.ple.layer_multipliers");
    if (sizes.size() != 16 || offsets.size() != 16 || multipliers.size() != 3 ||
            metadata.at("qwen4exp.ple.layers") != json::array({1}) ||
            metadata.at("qwen4exp.ple.ngram_size") != 3 ||
            metadata.at("qwen4exp.ple.heads_per_ngram") != 8 ||
            metadata.at("qwen4exp.ple.eos_token_id") != 248044 ||
            metadata.at("tokenizer.ggml.eos_token_id") != 248046) {
        std::fprintf(stderr, "Qwen4Exp PLE metadata geometry is invalid\n");
        return 1;
    }
    uint64_t populated_rows = 0;
    for (size_t head = 0; head < sizes.size(); ++head) {
        if (offsets.at(head).get<uint64_t>() != populated_rows) return 1;
        populated_rows += sizes.at(head).get<uint64_t>();
    }
    if (populated_rows > manifest.at("ple").at("row_count").get<uint64_t>() ||
            manifest.at("ple").at("row_count").get<uint64_t>() - populated_rows != 90) {
        std::fprintf(stderr, "Qwen4Exp PLE populated range is inconsistent with tensor rows\n");
        return 1;
    }

    std::set<std::string> tensor_names;
    for (const json & tensor : manifest.at("tensors")) {
        const std::string name = tensor.at("name").get<std::string>();
        tensor_names.insert(name);
        const int layer = layer_from_name(name);
        if (llm_tensor_type(LLM_ARCH_QWEN4EXP, name, layer) == LLM_TENSOR_UNKNOWN) {
            std::fprintf(stderr, "unmapped Qwen4Exp tensor: %s\n", name.c_str());
            return 1;
        }
    }

    for (int layer = 0; layer < 48; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const bool full_attention = (layer + 1) % 4 == 0;
        const char * required_full[] = {
            "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
            "attn_q_norm.weight", "attn_k_norm.weight", "indexer.q_proj.weight",
            "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight",
        };
        const char * required_recurrent[] = {
            "attn_qkv.weight", "attn_gate.weight", "ssm_conv1d.weight", "ssm_dt.bias",
            "ssm_a", "ssm_alpha.weight", "ssm_beta.weight", "ssm_norm.weight", "ssm_out.weight",
        };
        auto require_family = [&](const auto & family) {
            for (const char * suffix : family) {
                if (tensor_names.count(prefix + suffix) == 0) {
                    std::fprintf(stderr, "missing scheduled Qwen4Exp tensor: %s%s\n",
                        prefix.c_str(), suffix);
                    return false;
                }
            }
            return true;
        };
        if (full_attention ? !require_family(required_full) : !require_family(required_recurrent)) return 1;
        if (full_attention ? tensor_names.count(prefix + "ssm_out.weight") != 0
                           : tensor_names.count(prefix + "indexer.q_proj.weight") != 0) {
            std::fprintf(stderr, "Qwen4Exp layer mixes full and recurrent tensor families: %d\n", layer);
            return 1;
        }
        const char * common[] = {
            "ffn_gate_inp.weight", "ffn_gate_exps.weight", "ffn_up_exps.weight",
            "ffn_down_exps.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
            "ffn_up_shexp.weight", "ffn_down_shexp.weight", "hc_attn_down.weight",
            "hc_attn_norm.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
            "hc_ffn_down.weight", "hc_ffn_norm.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
        };
        for (const char * suffix : common) {
            if (tensor_names.count(prefix + suffix) == 0) return 1;
        }
    }

    const LLM_TN names(LLM_ARCH_QWEN4EXP);
    if (names(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight") != "per_layer_token_embd.weight" ||
            names(LLM_TENSOR_PLE_VALUE, "weight", 1) != "blk.1.ple_value.weight" ||
            names(LLM_TENSOR_HC_OUTPUT_UP, "weight") != "output_hc_up.weight") {
        std::fprintf(stderr, "Qwen4Exp canonical tensor names are incorrect\n");
        return 1;
    }

    llama_hparams state_geometry;
    state_geometry.n_layer = 48;
    state_geometry.n_embd = 2560;
    state_geometry.qwen4exp_hc_count = 4;
    state_geometry.qwen4exp_ple_conv_kernel = 4;
    state_geometry.qwen4exp_ple_ngram_size = 3;
    state_geometry.qwen4exp_ple_layers[1] = true;
    if (state_geometry.qwen4exp_ple_conv_state_size(0) != 0 ||
            state_geometry.qwen4exp_ple_conv_state_size(1) != 9 * 4 * 2560) {
        std::fprintf(stderr, "Qwen4Exp PLE convolution state geometry is incorrect\n");
        return 1;
    }

    const llama_model_params defaults = llama_model_default_params();
    if (defaults.qwen4exp_ple_mode != LLAMA_QWEN4EXP_PLE_AUTO ||
            defaults.qwen4exp_ple_cache_bytes != 256ULL*1024ULL*1024ULL ||
            defaults.qwen4exp_ple_queue_depth != 32) {
        std::fprintf(stderr, "Qwen4Exp PLE runtime defaults are incorrect\n");
        return 1;
    }
    return 0;
}
