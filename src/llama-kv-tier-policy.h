#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum class llama_kv_side : uint8_t {
    key,
    value,
};

struct llama_kv_tier_assignment {
    ggml_type key   = GGML_TYPE_F16;
    ggml_type value = GGML_TYPE_F16;
};

struct llama_kv_tier_override {
    llama_kv_side side = llama_kv_side::key;
    uint32_t begin = 0;
    uint32_t end   = 0; // exclusive
    ggml_type type = GGML_TYPE_F16;
};

struct llama_kv_layer_geometry {
    uint32_t key_head_dim   = 0;
    uint32_t value_head_dim = 0;
    uint32_t head_count     = 0;
};

struct llama_kv_tier_candidate {
    ggml_type type = GGML_TYPE_F16;
    double quality_cost = 0.0; // zero is the F16 reference
};

struct llama_kv_layer_side {
    uint32_t layer = 0;
    llama_kv_side side = llama_kv_side::key;
};

struct llama_kv_vbr_request {
    uint32_t context = 0;
    size_t budget_bytes = 0;
    double quality_floor = 0.0; // maximum sensitivity-weighted quality cost
    std::vector<llama_kv_tier_candidate> candidates;
    // Least-sensitive entries come first. Every layer-side must occur exactly once.
    std::vector<llama_kv_layer_side> sensitivity_order;
};

struct llama_kv_vbr_result {
    std::vector<llama_kv_tier_assignment> tiers;
    size_t bytes = 0;
    double quality_cost = 0.0;
    bool budget_satisfied = false;
    std::string report;
};

struct llama_kv_tier_transition {
    llama_kv_layer_side target;
    ggml_type from = GGML_TYPE_F16;
    ggml_type to   = GGML_TYPE_F16;
};

using llama_kv_prepare_transition = std::function<bool(const llama_kv_tier_transition &, std::string &)>;
using llama_kv_transition_action = std::function<void()>;

// Build a strict, explicit layer map. Overlapping assignments on the same side
// are rejected so configuration mistakes cannot silently change precedence.
bool llama_kv_build_static_tier_map(
        uint32_t n_layer,
        ggml_type base_key,
        ggml_type base_value,
        const std::vector<llama_kv_tier_override> & overrides,
        std::vector<llama_kv_tier_assignment> & out,
        std::string & error);

size_t llama_kv_tier_plan_bytes(
        const std::vector<llama_kv_tier_assignment> & tiers,
        const std::vector<llama_kv_layer_geometry> & geometry,
        uint32_t context);

// Deterministic fine-grained VBR solver. It starts from the highest-quality
// candidate and applies one layer-side transition at a time in model-specific
// sensitivity order until the budget is met or the quality floor is reached.
bool llama_kv_solve_vbr(
        const std::vector<llama_kv_layer_geometry> & geometry,
        const llama_kv_vbr_request & request,
        llama_kv_vbr_result & out,
        std::string & error);

std::vector<llama_kv_tier_transition> llama_kv_plan_tier_transitions(
        const std::vector<llama_kv_tier_assignment> & current,
        const std::vector<llama_kv_tier_assignment> & target);

// Prepare every allocation/conversion before publishing any tier change. The
// commit callback must be non-failing and only swaps already-prepared storage.
// On preparation failure rollback is invoked and current remains unchanged.
bool llama_kv_apply_tier_transitions_atomic(
        std::vector<llama_kv_tier_assignment> & current,
        const std::vector<llama_kv_tier_assignment> & target,
        const llama_kv_prepare_transition & prepare,
        const llama_kv_transition_action & commit,
        const llama_kv_transition_action & rollback,
        std::string & error);

std::string llama_kv_tier_plan_report(
        const std::vector<llama_kv_tier_assignment> & tiers,
        size_t bytes,
        size_t budget_bytes,
        double quality_cost);
