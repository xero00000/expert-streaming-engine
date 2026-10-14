#include "llama-deepseek-v41-model.h"
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace deepseek_v41 {
static void check(bool ok, const char * message) { if (!ok) throw std::invalid_argument(message); }

model_config model_config::from_json(const std::string & source) {
    const auto root = nlohmann::json::parse(source);
    const auto & j = root.contains("text_config") ? root.at("text_config") : root;
    check(j.at("model_type") == "deepseek_v41_text", "configuration is not DeepSeek V4.1 text");
    for (const char * key : {"vocab_size", "hidden_size", "num_attention_heads", "qk_rope_head_dim", "q_lora_rank",
         "o_lora_rank", "o_groups", "n_routed_experts", "num_experts_per_tok", "n_shared_experts", "moe_intermediate_size",
         "index_n_heads", "index_topk", "candidate_topk_blocks", "candidate_block_size", "hc_mult", "hc_sinkhorn_iters",
         "max_position_embeddings", "engram_n_heads", "engram_head_dim", "engram_max_ngram_size", "num_hidden_layers",
         "head_dim", "index_head_dim", "sliding_window"}) {
        check(j.at(key).is_number_integer() && j.at(key).get<int64_t>() > 0 &&
              j.at(key).get<int64_t>() <= INT32_MAX, "invalid V4.1 integer configuration");
    }
    model_config c;
    c.vocabulary = j.at("vocab_size"); c.hidden = j.at("hidden_size"); c.heads = j.at("num_attention_heads");
    c.rotary = j.at("qk_rope_head_dim"); c.query_rank = j.at("q_lora_rank");
    c.output_rank = j.at("o_lora_rank"); c.output_groups = j.at("o_groups");
    c.experts = j.at("n_routed_experts"); c.active_experts = j.at("num_experts_per_tok");
    c.shared_experts = j.at("n_shared_experts"); c.expert_hidden = j.at("moe_intermediate_size");
    c.index_heads = j.at("index_n_heads"); c.index_topk = j.at("index_topk");
    c.candidate_blocks = j.at("candidate_topk_blocks"); c.candidate_block_size = j.at("candidate_block_size");
    c.streams = j.at("hc_mult"); c.sinkhorn_iterations = j.at("hc_sinkhorn_iters");
    c.context = j.at("max_position_embeddings"); c.norm_eps = j.at("rms_norm_eps"); c.hc_eps = j.at("hc_eps");
    c.expert_scale = j.at("routed_scaling_factor"); c.swiglu_limit = j.at("swiglu_limit");
    c.rope_theta = j.at("rope_theta"); c.compressed_rope_theta = j.at("compress_rope_theta");
    const auto & rope = j.at("rope_scaling");
    check(rope.at("rope_type") == "yarn", "unsupported V4.1 rotary scaling");
    c.original_context = rope.at("original_max_position_embeddings"); c.rope_factor = rope.at("factor");
    c.beta_fast = rope.at("beta_fast"); c.beta_slow = rope.at("beta_slow");
    c.engram_layers = j.at("engram_layer_ids").get<std::vector<uint32_t>>();
    c.engram_heads = j.at("engram_n_heads"); c.engram_head_dim = j.at("engram_head_dim");
    c.engram_max_ngram = j.at("engram_max_ngram_size");
    const uint32_t layers = j.at("num_hidden_layers");
    c.attention.compress_ratios = j.at("compress_ratios").get<std::vector<uint32_t>>();
    check(layers > 0 && layers <= 512 && c.attention.compress_ratios.size() >= layers, "invalid V4.1 main layer count");
    c.attention.compress_ratios.resize(layers); // DSpark blocks do not become main layers
    for (auto ratio : c.attention.compress_ratios) check(ratio <= 2, "unsupported V4.1 compression ratio");
    c.attention.kv_source_layers = j.at("kv_source_layer_ids").get<std::vector<uint32_t>>();
    c.attention.index_source_layers = j.at("index_source_layer_ids").get<std::vector<uint32_t>>();
    c.attention.candidate_source_layer = j.at("candidate_source_layer_id");
    c.attention.head_dim = j.at("head_dim"); c.attention.index_head_dim = j.at("index_head_dim");
    c.attention.sliding_window = j.at("sliding_window");
    attention_layout validate(c.attention);
    check(j.at("scoring_func") == "sqrtsoftplus" && j.at("topk_method") == "noaux_tc" &&
          j.at("norm_topk_prob") == true && j.at("num_key_value_heads") == 1,
          "unsupported V4.1 routing/latent attention semantics");
    check(c.hidden && c.vocabulary && c.heads && c.query_rank && c.output_rank && c.output_groups &&
          c.heads % c.output_groups == 0 && c.experts && c.active_experts && c.active_experts <= c.experts &&
          c.shared_experts && c.expert_hidden && c.index_heads && c.index_topk && c.candidate_blocks &&
          c.candidate_block_size && c.streams > 0 && c.streams <= 8 && c.sinkhorn_iterations &&
          c.context && c.rotary && c.rotary % 2 == 0 && c.rotary <= c.attention.head_dim &&
          c.rotary <= c.attention.index_head_dim, "invalid V4.1 model dimensions");
    check(c.engram_heads == 8 && c.engram_head_dim == 256 && c.engram_max_ngram == 4 && c.engram_layers.size() == 2,
          "unsupported V4.1 Engram geometry");
    check(c.attention.head_dim % 32 == 0 && c.attention.index_head_dim % 32 == 0 && c.rope_factor >= 1 &&
          c.original_context > 0 && uint64_t(c.hidden) * c.streams <= INT32_MAX &&
          uint64_t(c.heads) * c.attention.head_dim <= INT32_MAX && uint64_t(c.output_rank) * c.output_groups <= INT32_MAX &&
          uint64_t(c.shared_experts) * c.expert_hidden <= INT32_MAX &&
          uint64_t(c.index_heads) * c.attention.index_head_dim <= INT32_MAX, "unsupported V4.1 packed/matrix geometry");
    std::set<uint32_t> unique;
    for (auto layer : c.engram_layers) check(layer < layers && unique.insert(layer).second, "invalid V4.1 Engram layer");
    for (float value : {c.norm_eps, c.hc_eps, c.expert_scale, c.swiglu_limit, c.rope_theta,
                        c.compressed_rope_theta, c.rope_factor, c.beta_fast, c.beta_slow})
        check(std::isfinite(value) && value > 0, "invalid V4.1 numerical configuration");
    return c;
}

std::vector<tensor_requirement> model_tensor_requirements(const model_config & c) {
    attention_layout layout(c.attention);
    std::vector<tensor_requirement> result;
    auto add = [&](std::string name, std::initializer_list<int64_t> shape, bool engram = false, bool optional = false) {
        result.push_back({std::move(name), shape, engram, optional});
    };
    const int64_t d = c.hidden, hd = c.attention.head_dim, h = c.heads, hc = c.streams;
    const int64_t mix = hc * (hc + 2), ff = c.expert_hidden, e = c.experts;
    add("token_embd", {d, c.vocabulary}); add("output_norm", {d}); add("output", {d, c.vocabulary});
    for (size_t i = 0; i < layout.layers().size(); ++i) {
        const std::string prefix = "blk." + std::to_string(i) + ".";
        auto layer = [&](const char * name, std::initializer_list<int64_t> shape, bool engram = false, bool optional = false) {
            add(prefix + name, shape, engram, optional);
        };
        layer("attn_norm", {d}); layer("attn_q_a", {d, c.query_rank}); layer("attn_q_a_norm", {c.query_rank});
        layer("attn_q_b", {c.query_rank, h * hd}); layer("attn_kv", {d, hd}); layer("attn_kv_a_norm", {hd});
        layer("attn_output_a", {h * hd / c.output_groups, int64_t(c.output_groups) * c.output_rank});
        layer("attn_output_b", {int64_t(c.output_groups) * c.output_rank, d}); layer("attn_sinks", {h});
        layer("ffn_norm", {d}); layer("ffn_gate_inp", {d, e}); layer("exp_probs_b", {e});
        layer("exp_probs_b_vl", {e}, false, true);
        layer("ffn_gate_exps", {d, ff, e}); layer("ffn_up_exps", {d, ff, e}); layer("ffn_down_exps", {ff, d, e});
        layer("ffn_gate_shexp", {d, ff * c.shared_experts}); layer("ffn_up_shexp", {d, ff * c.shared_experts});
        layer("ffn_down_shexp", {ff * c.shared_experts, d});
        for (const char * sublayer : {"attn", "ffn"}) {
            add(prefix + "hc_" + sublayer + "_fn", {d * hc, mix});
            add(prefix + "hc_" + sublayer + "_base", {mix});
            add(prefix + "hc_" + sublayer + "_scale", {3});
        }
        const auto & info = layout.layers()[i];
        if (info.kv_owner == int32_t(i)) {
            layer("attn_compressor_kv", {d, hd}); layer("attn_compressor_norm", {hd});
            if (info.compress_ratio > 1) layer("attn_compressor_gate", {d, hd});
            layer("indexer.attn_k", {hd, c.attention.index_head_dim}); layer("indexer.k_norm", {c.attention.index_head_dim});
        }
        if (info.topk_owner == int32_t(i)) {
            layer("indexer.attn_q_b", {c.query_rank, int64_t(c.index_heads) * c.attention.index_head_dim});
            layer("indexer.proj", {d, c.index_heads});
        }
        if (std::find(c.engram_layers.begin(), c.engram_layers.end(), i) != c.engram_layers.end()) {
            layer("engram_wkv", {int64_t(c.engram_heads) * (c.engram_max_ngram - 1) * c.engram_head_dim, d * (hc + 1)}, true);
            layer("engram_q_norm", {d, hc}, true); layer("engram_k_norm", {d, hc}, true);
        }
    }
    return result;
}

std::map<std::string, ggml_tensor *> bind_model_tensors(const model_config & c, const tensor_lookup & lookup, bool require_engram) {
    std::map<std::string, ggml_tensor *> result;
    for (const auto & requirement : model_tensor_requirements(c)) {
        ggml_tensor * tensor = nullptr;
        for (const char * suffix : {"", ".weight", ".bias"}) if (auto * found = lookup(requirement.name + suffix)) {
            if (tensor && tensor != found) throw std::invalid_argument("ambiguous V4.1 tensor: " + requirement.name);
            tensor = found;
        }
        if (!tensor) {
            if (requirement.optional || (requirement.engram && !require_engram)) continue;
            throw std::invalid_argument("missing V4.1 tensor: " + requirement.name);
        }
        if (tensor->type != GGML_TYPE_F32 && tensor->type != GGML_TYPE_F16 && tensor->type != GGML_TYPE_BF16 &&
            !ggml_is_quantized(tensor->type)) throw std::invalid_argument("invalid V4.1 weight type: " + requirement.name);
        for (int axis = 0; axis < GGML_MAX_DIMS; ++axis) {
            const int64_t expected = axis < int(requirement.shape.size()) ? requirement.shape[axis] : 1;
            if (tensor->ne[axis] != expected) throw std::invalid_argument("invalid V4.1 tensor shape: " + requirement.name);
        }
        result.emplace(requirement.name, tensor);
    }
    return result;
}
} // namespace deepseek_v41
