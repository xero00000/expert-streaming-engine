#pragma once
#include "llama-deepseek-v41-layout.h"
#include "ggml.h"
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace deepseek_v41 {

struct model_config {
    attention_layout_config attention;
    uint32_t vocabulary = 0, hidden = 0, heads = 0, rotary = 0;
    uint32_t query_rank = 0, output_rank = 0, output_groups = 0;
    uint32_t experts = 0, active_experts = 0, shared_experts = 0, expert_hidden = 0;
    uint32_t index_heads = 0, index_topk = 0, candidate_blocks = 0, candidate_block_size = 0;
    uint32_t streams = 0, sinkhorn_iterations = 0;
    uint32_t context = 0, original_context = 0;
    float norm_eps = 0, hc_eps = 0, expert_scale = 0, swiglu_limit = 0;
    float rope_theta = 0, compressed_rope_theta = 0, rope_factor = 0, beta_fast = 0, beta_slow = 0;
    std::vector<uint32_t> engram_layers;
    uint32_t engram_heads = 0, engram_head_dim = 0, engram_max_ngram = 0;
    static model_config from_json(const std::string & source);
};

struct tensor_requirement {
    std::string name; // normalized stem, without .weight or .bias
    std::vector<int64_t> shape; // GGML axis order
    bool engram = false;
    bool optional = false; // currently only image-routing biases in text-only runs
};

std::vector<tensor_requirement> model_tensor_requirements(const model_config & config);

// Lookup may return a GGUF metadata tensor or an allocated model tensor.
// A canonical stem, .weight, or .bias spelling is accepted, but ambiguity is
// rejected. Complete text-model binding requires the small Engram projection
// tensors too; the giant row tables live in separately validated row stores.
using tensor_lookup = std::function<ggml_tensor *(const std::string &)>;
std::map<std::string, ggml_tensor *> bind_model_tensors(const model_config & config,
    const tensor_lookup & lookup, bool require_engram = true);

} // namespace deepseek_v41
