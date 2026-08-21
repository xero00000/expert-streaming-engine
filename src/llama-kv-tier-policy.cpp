#include "llama-kv-tier-policy.h"

#include "llama-kv-padding.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <unordered_set>

namespace {

bool valid_type(ggml_type type) {
    return type >= 0 && type < GGML_TYPE_COUNT && ggml_type_name(type) != nullptr;
}

size_t side_bytes(ggml_type type, uint32_t head_dim, uint32_t head_count, uint32_t context) {
    if (head_dim == 0 || head_count == 0 || context == 0) {
        return 0;
    }
    const uint32_t padded = llama_kv_head_dim_for_type(type, head_dim);
    const size_t row = ggml_row_size(type, padded);
    if (row > std::numeric_limits<size_t>::max() / head_count ||
            row * head_count > std::numeric_limits<size_t>::max() / context) {
        return std::numeric_limits<size_t>::max();
    }
    return row * head_count * context;
}

uint64_t side_id(const llama_kv_layer_side & value) {
    return (uint64_t(value.layer) << 1) | (value.side == llama_kv_side::value ? 1u : 0u);
}

ggml_type & selected(std::vector<llama_kv_tier_assignment> & tiers, const llama_kv_layer_side & ref) {
    return ref.side == llama_kv_side::key ? tiers[ref.layer].key : tiers[ref.layer].value;
}

uint32_t side_dim(const llama_kv_layer_geometry & geometry, llama_kv_side side) {
    return side == llama_kv_side::key ? geometry.key_head_dim : geometry.value_head_dim;
}

} // namespace

bool llama_kv_build_static_tier_map(
        uint32_t n_layer,
        ggml_type base_key,
        ggml_type base_value,
        const std::vector<llama_kv_tier_override> & overrides,
        std::vector<llama_kv_tier_assignment> & out,
        std::string & error) {
    out.clear();
    error.clear();
    if (n_layer == 0) {
        error = "KV tier map requires at least one layer";
        return false;
    }
    if (!valid_type(base_key) || !valid_type(base_value)) {
        error = "KV tier map has an invalid base type";
        return false;
    }

    out.assign(n_layer, { base_key, base_value });
    std::vector<bool> key_seen(n_layer, false);
    std::vector<bool> value_seen(n_layer, false);
    for (const auto & item : overrides) {
        if (!valid_type(item.type)) {
            error = "KV tier override has an invalid type";
            out.clear();
            return false;
        }
        if (item.begin >= item.end || item.end > n_layer) {
            error = "KV tier override range is outside the model layer range";
            out.clear();
            return false;
        }
        auto & seen = item.side == llama_kv_side::key ? key_seen : value_seen;
        for (uint32_t layer = item.begin; layer < item.end; ++layer) {
            if (seen[layer]) {
                error = "KV tier overrides overlap on layer " + std::to_string(layer) +
                        (item.side == llama_kv_side::key ? " K" : " V");
                out.clear();
                return false;
            }
            seen[layer] = true;
            if (item.side == llama_kv_side::key) {
                out[layer].key = item.type;
            } else {
                out[layer].value = item.type;
            }
        }
    }
    return true;
}

size_t llama_kv_tier_plan_bytes(
        const std::vector<llama_kv_tier_assignment> & tiers,
        const std::vector<llama_kv_layer_geometry> & geometry,
        uint32_t context) {
    if (tiers.size() != geometry.size()) {
        return std::numeric_limits<size_t>::max();
    }
    size_t total = 0;
    for (size_t i = 0; i < tiers.size(); ++i) {
        const size_t key = side_bytes(tiers[i].key, geometry[i].key_head_dim, geometry[i].head_count, context);
        const size_t value = side_bytes(tiers[i].value, geometry[i].value_head_dim, geometry[i].head_count, context);
        if (key == std::numeric_limits<size_t>::max() || value == std::numeric_limits<size_t>::max() ||
                total > std::numeric_limits<size_t>::max() - key ||
                total + key > std::numeric_limits<size_t>::max() - value) {
            return std::numeric_limits<size_t>::max();
        }
        total += key + value;
    }
    return total;
}

bool llama_kv_solve_vbr(
        const std::vector<llama_kv_layer_geometry> & geometry,
        const llama_kv_vbr_request & request,
        llama_kv_vbr_result & out,
        std::string & error) {
    out = {};
    error.clear();
    if (geometry.empty() || request.context == 0 || request.candidates.empty()) {
        error = "VBR requires layers, a non-zero context, and tier candidates";
        return false;
    }
    if (!std::isfinite(request.quality_floor) || request.quality_floor < 0.0) {
        error = "VBR quality floor must be finite and non-negative";
        return false;
    }
    if (request.sensitivity_order.size() != geometry.size() * 2) {
        error = "VBR sensitivity order must contain every layer-side exactly once";
        return false;
    }
    for (size_t i = 0; i < request.candidates.size(); ++i) {
        const auto & candidate = request.candidates[i];
        if (!valid_type(candidate.type) || !std::isfinite(candidate.quality_cost) || candidate.quality_cost < 0.0 ||
                (i > 0 && candidate.quality_cost <= request.candidates[i - 1].quality_cost)) {
            error = "VBR candidates must have valid types and strictly increasing quality cost";
            return false;
        }
    }

    std::unordered_set<uint64_t> seen;
    for (const auto & ref : request.sensitivity_order) {
        if (ref.layer >= geometry.size() || !seen.insert(side_id(ref)).second) {
            error = "VBR sensitivity order contains an invalid or duplicate layer-side";
            return false;
        }
    }

    out.tiers.assign(geometry.size(), { request.candidates.front().type, request.candidates.front().type });
    std::vector<uint8_t> tier_indices(geometry.size() * 2, 0);
    out.bytes = llama_kv_tier_plan_bytes(out.tiers, geometry, request.context);
    if (out.bytes == std::numeric_limits<size_t>::max()) {
        error = "VBR byte accounting overflow";
        return false;
    }

    // One full pass advances each side by one tier. Repeating preserves the
    // supplied model ordering at every quality level and is deterministic.
    for (size_t candidate_index = 1;
            candidate_index < request.candidates.size() && out.bytes > request.budget_bytes;
            ++candidate_index) {
        for (size_t rank = 0; rank < request.sensitivity_order.size() && out.bytes > request.budget_bytes; ++rank) {
            const auto & ref = request.sensitivity_order[rank];
            const size_t index_slot = size_t(ref.layer) * 2 + (ref.side == llama_kv_side::value ? 1 : 0);
            if (tier_indices[index_slot] != candidate_index - 1) {
                continue;
            }
            const double sensitivity = 1.0 + double(rank) / double(request.sensitivity_order.size());
            const double delta = (request.candidates[candidate_index].quality_cost -
                    request.candidates[candidate_index - 1].quality_cost) * sensitivity;
            if (out.quality_cost + delta > request.quality_floor + 1e-12) {
                continue;
            }

            const auto & layer_geometry = geometry[ref.layer];
            ggml_type & current = selected(out.tiers, ref);
            const size_t before = side_bytes(current, side_dim(layer_geometry, ref.side),
                    layer_geometry.head_count, request.context);
            const size_t after = side_bytes(request.candidates[candidate_index].type,
                    side_dim(layer_geometry, ref.side), layer_geometry.head_count, request.context);
            if (after >= before) {
                continue;
            }
            current = request.candidates[candidate_index].type;
            tier_indices[index_slot] = candidate_index;
            out.bytes -= before - after;
            out.quality_cost += delta;
        }
    }

    out.budget_satisfied = out.bytes <= request.budget_bytes;
    out.report = llama_kv_tier_plan_report(out.tiers, out.bytes, request.budget_bytes, out.quality_cost);
    return true;
}

std::vector<llama_kv_tier_transition> llama_kv_plan_tier_transitions(
        const std::vector<llama_kv_tier_assignment> & current,
        const std::vector<llama_kv_tier_assignment> & target) {
    std::vector<llama_kv_tier_transition> result;
    if (current.size() != target.size()) {
        return result;
    }
    for (uint32_t layer = 0; layer < current.size(); ++layer) {
        if (current[layer].key != target[layer].key) {
            result.push_back({ { layer, llama_kv_side::key }, current[layer].key, target[layer].key });
        }
        if (current[layer].value != target[layer].value) {
            result.push_back({ { layer, llama_kv_side::value }, current[layer].value, target[layer].value });
        }
    }
    return result;
}

bool llama_kv_apply_tier_transitions_atomic(
        std::vector<llama_kv_tier_assignment> & current,
        const std::vector<llama_kv_tier_assignment> & target,
        const llama_kv_prepare_transition & prepare,
        const llama_kv_transition_action & commit,
        const llama_kv_transition_action & rollback,
        std::string & error) {
    error.clear();
    if (current.size() != target.size() || !prepare || !commit || !rollback) {
        error = "invalid failure-atomic KV transition request";
        return false;
    }
    const auto transitions = llama_kv_plan_tier_transitions(current, target);
    for (const auto & transition : transitions) {
        if (!prepare(transition, error)) {
            rollback();
            if (error.empty()) {
                error = "KV tier transition preparation failed";
            }
            return false;
        }
    }
    commit();
    current = target;
    return true;
}

std::string llama_kv_tier_plan_report(
        const std::vector<llama_kv_tier_assignment> & tiers,
        size_t bytes,
        size_t budget_bytes,
        double quality_cost) {
    std::ostringstream stream;
    stream << "KV VBR: " << bytes << "/" << budget_bytes << " bytes, quality cost "
           << std::fixed << std::setprecision(6) << quality_cost;
    for (size_t layer = 0; layer < tiers.size(); ++layer) {
        stream << "\n  layer " << layer << ": K=" << ggml_type_name(tiers[layer].key)
               << ", V=" << ggml_type_name(tiers[layer].value);
    }
    return stream.str();
}
