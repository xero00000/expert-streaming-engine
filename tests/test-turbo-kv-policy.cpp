#include "llama-kv-tier-policy.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

static void test_static_map() {
    std::vector<llama_kv_tier_assignment> tiers;
    std::string error;
    const std::vector<llama_kv_tier_override> overrides = {
        { llama_kv_side::key,   0, 2, GGML_TYPE_TURBO4_0 },
        { llama_kv_side::value, 1, 3, GGML_TYPE_TURBO2_0 },
    };
    assert(llama_kv_build_static_tier_map(4, GGML_TYPE_F16, GGML_TYPE_F16, overrides, tiers, error));
    assert(tiers.size() == 4);
    assert(tiers[0].key == GGML_TYPE_TURBO4_0 && tiers[0].value == GGML_TYPE_F16);
    assert(tiers[1].key == GGML_TYPE_TURBO4_0 && tiers[1].value == GGML_TYPE_TURBO2_0);
    assert(tiers[2].key == GGML_TYPE_F16 && tiers[2].value == GGML_TYPE_TURBO2_0);

    auto overlap = overrides;
    overlap.push_back({ llama_kv_side::key, 1, 4, GGML_TYPE_TURBO8_0 });
    assert(!llama_kv_build_static_tier_map(4, GGML_TYPE_F16, GGML_TYPE_F16, overlap, tiers, error));
    assert(error.find("overlap") != std::string::npos);
    assert(tiers.empty());

    const std::vector<llama_kv_tier_override> bad_range = {
        { llama_kv_side::key, 4, 5, GGML_TYPE_TURBO4_0 },
    };
    assert(!llama_kv_build_static_tier_map(4, GGML_TYPE_F16, GGML_TYPE_F16, bad_range, tiers, error));
}

static void test_accounting_and_solver() {
    const std::vector<llama_kv_layer_geometry> geometry(3, { 96, 128, 8 });
    std::vector<llama_kv_tier_assignment> f16(geometry.size(), { GGML_TYPE_F16, GGML_TYPE_F16 });
    const size_t f16_bytes = llama_kv_tier_plan_bytes(f16, geometry, 4096);
    assert(f16_bytes != std::numeric_limits<size_t>::max());

    llama_kv_vbr_request request;
    request.context = 4096;
    request.budget_bytes = f16_bytes / 2;
    request.quality_floor = 100.0;
    request.candidates = {
        { GGML_TYPE_F16,        0.00 },
        { GGML_TYPE_TURBO8_0,   0.01 },
        { GGML_TYPE_TURBO4_0,   0.05 },
        { GGML_TYPE_TURBO3_TCQ, 0.10 },
        { GGML_TYPE_TURBO2_TCQ, 0.20 },
        { GGML_TYPE_TURBO1_TCQ, 0.45 },
    };
    // A deliberately asymmetric model-specific order proves K and V can move independently.
    request.sensitivity_order = {
        { 2, llama_kv_side::value }, { 2, llama_kv_side::key },
        { 1, llama_kv_side::value }, { 1, llama_kv_side::key },
        { 0, llama_kv_side::value }, { 0, llama_kv_side::key },
    };

    llama_kv_vbr_result result;
    std::string error;
    assert(llama_kv_solve_vbr(geometry, request, result, error));
    assert(result.budget_satisfied);
    assert(result.bytes <= request.budget_bytes);
    assert(result.tiers[2].value != GGML_TYPE_F16);
    assert(result.report.find("layer 2: K=") != std::string::npos);

    const auto first = result;
    assert(llama_kv_solve_vbr(geometry, request, result, error));
    assert(result.bytes == first.bytes);
    assert(result.quality_cost == first.quality_cost);
    assert(result.report == first.report);

    request.quality_floor = 0.0;
    request.budget_bytes = 1;
    assert(llama_kv_solve_vbr(geometry, request, result, error));
    assert(!result.budget_satisfied);
    assert(result.bytes == f16_bytes);

    request.sensitivity_order.pop_back();
    assert(!llama_kv_solve_vbr(geometry, request, result, error));
}

static void test_failure_atomic_transition() {
    std::vector<llama_kv_tier_assignment> current(2, { GGML_TYPE_F16, GGML_TYPE_F16 });
    const std::vector<llama_kv_tier_assignment> original = current;
    const std::vector<llama_kv_tier_assignment> target = {
        { GGML_TYPE_TURBO4_0, GGML_TYPE_F16 },
        { GGML_TYPE_F16, GGML_TYPE_TURBO2_TCQ },
    };
    const auto transitions = llama_kv_plan_tier_transitions(current, target);
    assert(transitions.size() == 2);
    assert(transitions[0].target.layer == 0 && transitions[0].target.side == llama_kv_side::key);
    assert(transitions[1].target.layer == 1 && transitions[1].target.side == llama_kv_side::value);

    int prepared = 0;
    int committed = 0;
    int rolled_back = 0;
    std::string error;
    assert(!llama_kv_apply_tier_transitions_atomic(current, target,
                [&](const llama_kv_tier_transition &, std::string & detail) {
                    ++prepared;
                    if (prepared == 2) {
                        detail = "injected allocation failure";
                        return false;
                    }
                    return true;
                },
                [&] { ++committed; },
                [&] { ++rolled_back; }, error));
    assert(current[0].key == original[0].key && current[1].value == original[1].value);
    assert(committed == 0 && rolled_back == 1);
    assert(error == "injected allocation failure");

    prepared = 0;
    assert(llama_kv_apply_tier_transitions_atomic(current, target,
                [&](const llama_kv_tier_transition &, std::string &) { ++prepared; return true; },
                [&] { ++committed; },
                [&] { ++rolled_back; }, error));
    assert(prepared == 2 && committed == 1 && rolled_back == 1);
    assert(current[0].key == target[0].key && current[1].value == target[1].value);
}

int main() {
    test_static_map();
    test_accounting_and_solver();
    test_failure_atomic_transition();
    std::cout << "Turbo KV tier policy tests passed\n";
    return 0;
}
