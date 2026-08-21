#include "resource-planner.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <exception>
#include <limits>
#include <sstream>

namespace {

bool add_checked(uint64_t & value, uint64_t add) {
    if (add > std::numeric_limits<uint64_t>::max() - value) {
        return false;
    }
    value += add;
    return true;
}

uint64_t mul_saturated(uint64_t a, uint64_t b) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) {
        return std::numeric_limits<uint64_t>::max();
    }
    return a*b;
}

common_memory_policy resolve_policy(const common_resource_plan_input & input) {
    if (input.requested_policy != COMMON_MEMORY_POLICY_AUTO) {
        return input.requested_policy;
    }
    if (!input.model_is_moe) {
        return COMMON_MEMORY_POLICY_RESIDENT;
    }
    // Cache mode does not need the stream-only I/O staging allocation. An
    // unset RAM ceiling is explicitly unlimited rather than a zero-byte cap.
    if (input.max_ram_bytes == 0 ||
            input.expert_source_bytes <= input.max_ram_bytes - std::min(input.max_ram_bytes, input.dense_ram_bytes)) {
        return COMMON_MEMORY_POLICY_CACHE;
    }
    return COMMON_MEMORY_POLICY_STREAM;
}

std::vector<common_kv_quality> qualities_descending(common_kv_quality floor) {
    std::vector<common_kv_quality> result;
    for (int quality = COMMON_KV_QUALITY_F16; quality >= (int) floor; --quality) {
        result.push_back((common_kv_quality) quality);
    }
    return result;
}

bool allocate_context(
        const common_resource_plan_input & input,
        common_resource_plan & plan,
        uint32_t context,
        common_kv_quality quality) {
    const uint64_t per_token = input.kv_bytes_per_token[quality];
    if (per_token == 0 || input.devices.empty()) {
        return false;
    }
    const uint64_t kv_total = mul_saturated(per_token, context);
    if (kv_total == std::numeric_limits<uint64_t>::max()) {
        return false;
    }

    uint64_t weight_sum = 0;
    std::vector<uint64_t> usable;
    usable.reserve(input.devices.size());
    const int transient_device = input.transient_device < 0
        ? input.devices.front().id : input.transient_device;
    const uint64_t transient_need = std::max(input.mtp_bytes, input.multimodal_bytes);
    for (const auto & device : input.devices) {
        uint64_t fixed = device.reserve_bytes;
        if (!add_checked(fixed, device.dense_bytes) || !add_checked(fixed, device.graph_bytes) ||
                (device.id == transient_device && !add_checked(fixed, transient_need)) ||
                fixed > device.free_bytes) {
            return false;
        }
        const uint64_t available = device.free_bytes - fixed;
        usable.push_back(available);
        if (!add_checked(weight_sum, available)) {
            return false;
        }
    }
    if (weight_sum == 0 || kv_total > weight_sum) {
        return false;
    }

    plan.devices.clear();
    uint64_t assigned = 0;
    for (size_t i = 0; i < input.devices.size(); ++i) {
        const auto & source = input.devices[i];
        const uint64_t share = i + 1 == input.devices.size()
            ? kv_total - assigned
            : (uint64_t) (((__uint128_t) kv_total * usable[i])/weight_sum);
        if (share > usable[i]) {
            return false;
        }
        assigned += share;
        common_resource_device_plan device;
        device.id = source.id;
        device.capacity_bytes = source.free_bytes;
        device.reserve_bytes = source.reserve_bytes;
        device.dense_bytes = source.dense_bytes;
        device.graph_bytes = source.graph_bytes;
        device.kv_bytes = share;
        device.planned_bytes = source.dense_bytes + source.graph_bytes + share;
        device.headroom_bytes = source.free_bytes - source.reserve_bytes - device.planned_bytes;
        plan.devices.push_back(device);
    }
    return true;
}

bool better_context_choice(
        common_resource_preference preference,
        uint32_t ceiling,
        uint32_t context,
        common_kv_quality quality,
        uint32_t best_context,
        common_kv_quality best_quality) {
    if (best_context == 0) {
        return true;
    }
    if (preference == COMMON_RESOURCE_PREFERENCE_LATENCY) {
        return quality > best_quality || (quality == best_quality && context > best_context);
    }
    // Balanced preserves the requested context when possible, then chooses
    // the highest quality. If the request cannot fit, it behaves like the
    // throughput policy and maximizes usable context before quality.
    if (preference == COMMON_RESOURCE_PREFERENCE_BALANCED && context == ceiling) {
        return best_context != ceiling || quality > best_quality;
    }
    if (preference == COMMON_RESOURCE_PREFERENCE_BALANCED && best_context == ceiling) {
        return false;
    }
    return context > best_context || (context == best_context && quality > best_quality);
}

void json_string(std::ostringstream & out, const std::string & value) {
    out << '"';
    for (char c : value) {
        if (c == '"' || c == '\\') out << '\\';
        if (c == '\n') out << "\\n";
        else out << c;
    }
    out << '"';
}

} // namespace

bool common_parse_byte_size(const std::string & text, uint64_t & bytes, std::string & error) {
    error.clear();
    if (text.empty() || text.front() == '-') {
        error = "memory size must be a non-negative number with an optional KiB/MiB/GiB suffix";
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str()) {
        error = "invalid memory size: " + text;
        return false;
    }
    std::string suffix(end);
    std::transform(suffix.begin(), suffix.end(), suffix.begin(),
        [](unsigned char c) { return (char) std::tolower(c); });
    uint64_t multiplier = 1;
    if (suffix.empty() || suffix == "b") multiplier = 1;
    else if (suffix == "kib") multiplier = 1024ULL;
    else if (suffix == "mib") multiplier = 1024ULL*1024ULL;
    else if (suffix == "gib") multiplier = 1024ULL*1024ULL*1024ULL;
    else {
        error = "unsupported memory-size suffix in: " + text;
        return false;
    }
    if (value > std::numeric_limits<uint64_t>::max()/multiplier) {
        error = "memory size overflows 64 bits: " + text;
        return false;
    }
    bytes = (uint64_t) value*multiplier;
    return true;
}

bool common_parse_token_count(const std::string & text, uint32_t & count, std::string & error) {
    error.clear();
    if (text.empty() || text.front() == '-') {
        error = "token count must be non-negative";
        return false;
    }
    char * end = nullptr;
    errno = 0;
    const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
    if (errno != 0 || end == text.c_str()) {
        error = "invalid token count: " + text;
        return false;
    }
    uint64_t multiplier = 1;
    if (*end != '\0') {
        if ((end[0] == 'k' || end[0] == 'K') && end[1] == '\0') multiplier = 1024;
        else if ((end[0] == 'm' || end[0] == 'M') && end[1] == '\0') multiplier = 1024*1024;
        else {
            error = "unsupported token-count suffix in: " + text;
            return false;
        }
    }
    if (value > std::numeric_limits<uint32_t>::max()/multiplier) {
        error = "token count exceeds uint32 range: " + text;
        return false;
    }
    count = (uint32_t) (value*multiplier);
    return true;
}

bool common_parse_memory_policy(const std::string & text, common_memory_policy & policy) {
    if (text == "auto") policy = COMMON_MEMORY_POLICY_AUTO;
    else if (text == "resident") policy = COMMON_MEMORY_POLICY_RESIDENT;
    else if (text == "cache" || text == "hybrid") policy = COMMON_MEMORY_POLICY_CACHE;
    else if (text == "stream") policy = COMMON_MEMORY_POLICY_STREAM;
    else return false;
    return true;
}

bool common_parse_resource_preference(const std::string & text, common_resource_preference & preference) {
    if (text == "balanced") preference = COMMON_RESOURCE_PREFERENCE_BALANCED;
    else if (text == "latency") preference = COMMON_RESOURCE_PREFERENCE_LATENCY;
    else if (text == "throughput") preference = COMMON_RESOURCE_PREFERENCE_THROUGHPUT;
    else return false;
    return true;
}

bool common_parse_kv_quality(const std::string & text, common_kv_quality & quality) {
    if (text == "turbo1") quality = COMMON_KV_QUALITY_TURBO1;
    else if (text == "turbo2") quality = COMMON_KV_QUALITY_TURBO2;
    else if (text == "turbo3") quality = COMMON_KV_QUALITY_TURBO3;
    else if (text == "turbo4") quality = COMMON_KV_QUALITY_TURBO4;
    else if (text == "turbo8") quality = COMMON_KV_QUALITY_TURBO8;
    else if (text == "q8" || text == "q8_0") quality = COMMON_KV_QUALITY_Q8;
    else if (text == "f16") quality = COMMON_KV_QUALITY_F16;
    else return false;
    return true;
}

bool common_parse_resource_backend(const std::string & text, common_resource_backend & backend) {
    if (text == "mmap") backend = COMMON_RESOURCE_BACKEND_MMAP;
    else if (text == "pread") backend = COMMON_RESOURCE_BACKEND_PREAD;
    else if (text == "io_uring") backend = COMMON_RESOURCE_BACKEND_IO_URING;
    else return false;
    return true;
}

std::string common_memory_policy_name(common_memory_policy policy) {
    switch (policy) {
        case COMMON_MEMORY_POLICY_AUTO: return "auto";
        case COMMON_MEMORY_POLICY_RESIDENT: return "resident";
        case COMMON_MEMORY_POLICY_CACHE: return "cache";
        case COMMON_MEMORY_POLICY_STREAM: return "stream";
    }
    return "unknown";
}

std::string common_kv_quality_name(common_kv_quality quality) {
    switch (quality) {
        case COMMON_KV_QUALITY_TURBO1: return "turbo1";
        case COMMON_KV_QUALITY_TURBO2: return "turbo2";
        case COMMON_KV_QUALITY_TURBO3: return "turbo3";
        case COMMON_KV_QUALITY_TURBO4: return "turbo4";
        case COMMON_KV_QUALITY_TURBO8: return "turbo8";
        case COMMON_KV_QUALITY_Q8: return "q8_0";
        case COMMON_KV_QUALITY_F16: return "f16";
    }
    return "unknown";
}

std::string common_resource_backend_name(common_resource_backend backend) {
    switch (backend) {
        case COMMON_RESOURCE_BACKEND_MMAP: return "mmap";
        case COMMON_RESOURCE_BACKEND_PREAD: return "pread";
        case COMMON_RESOURCE_BACKEND_IO_URING: return "io_uring";
    }
    return "unknown";
}

bool common_resource_plan_solve(
        const common_resource_plan_input & input,
        common_resource_plan & plan,
        std::string & error) {
    error.clear();
    common_resource_plan candidate;
    if (input.slots == 0 || input.devices.empty()) {
        error = "resource planning requires at least one slot and one device";
        return false;
    }
    if (!input.backend_available) {
        error = "requested storage backend is unavailable; fallback is forbidden";
        return false;
    }
    if (input.min_kv_quality < COMMON_KV_QUALITY_TURBO1 || input.min_kv_quality > COMMON_KV_QUALITY_F16) {
        error = "invalid minimum KV quality";
        return false;
    }

    candidate.policy = resolve_policy(input);
    candidate.backend = input.requested_backend;
    candidate.slots = input.slots;
    candidate.ram_capacity_bytes = input.max_ram_bytes;
    candidate.io_staging_bytes = candidate.policy == COMMON_MEMORY_POLICY_STREAM ? input.io_staging_bytes : 0;

    uint64_t ram_fixed = input.dense_ram_bytes;
    if (!add_checked(ram_fixed, candidate.io_staging_bytes) ||
            (input.max_ram_bytes != 0 && ram_fixed > input.max_ram_bytes)) {
        error = "fixed RAM allocations exceed --max-ram";
        return false;
    }
    const uint64_t ram_left = input.max_ram_bytes == 0
        ? std::numeric_limits<uint64_t>::max() - ram_fixed
        : input.max_ram_bytes - ram_fixed;
    candidate.expert_ram_bytes = candidate.policy == COMMON_MEMORY_POLICY_RESIDENT
        ? input.expert_source_bytes
        : std::min(input.requested_expert_ram_bytes, ram_left);
    if (candidate.expert_ram_bytes > ram_left) {
        error = "resident expert weights exceed --max-ram";
        return false;
    }
    if (candidate.policy != COMMON_MEMORY_POLICY_RESIDENT &&
            candidate.expert_ram_bytes != 0 &&
            candidate.expert_ram_bytes < input.min_expert_ram_bytes) {
        candidate.expert_ram_bytes = 0;
    }
    const uint64_t ram_after_experts = ram_left - candidate.expert_ram_bytes;
    candidate.aux_ram_bytes = std::min(input.requested_aux_ram_bytes, ram_after_experts);
    candidate.ram_planned_bytes = ram_fixed + candidate.expert_ram_bytes + candidate.aux_ram_bytes;

    const uint32_t ceiling = input.max_context == 0 ? input.requested_context
        : input.requested_context == 0 ? input.max_context
        : std::min(input.requested_context, input.max_context);
    const uint32_t alignment = std::max(1U, input.context_alignment);
    const uint32_t aligned_ceiling = ceiling/alignment*alignment;
    const uint32_t aligned_minimum = ((input.min_context + alignment - 1)/alignment)*alignment;
    if (aligned_ceiling < aligned_minimum) {
        error = "requested context is below the minimum context";
        return false;
    }

    bool context_found = false;
    common_resource_plan best_candidate;
    for (const auto quality : qualities_descending(input.min_kv_quality)) {
        if (input.kv_bytes_per_token[quality] == 0) {
            continue;
        }
        uint32_t low = aligned_minimum/alignment;
        uint32_t high = aligned_ceiling/alignment;
        uint32_t best = 0;
        while (low <= high) {
            const uint32_t middle_units = low + (high - low)/2;
            const uint32_t middle = middle_units*alignment;
            common_resource_plan attempt = candidate;
            if (allocate_context(input, attempt, middle, quality)) {
                best = middle;
                low = middle_units + 1;
            } else {
                if (middle_units == 0) break;
                high = middle_units - 1;
            }
        }
        if (best != 0) {
            common_resource_plan attempt = candidate;
            attempt.kv_quality = quality;
            attempt.context = best;
            if (!allocate_context(input, attempt, best, quality)) {
                error = "internal resource planning instability";
                return false;
            }
            if (!context_found || better_context_choice(
                    input.preference, aligned_ceiling, best, quality,
                    best_candidate.context, best_candidate.kv_quality)) {
                best_candidate = std::move(attempt);
                context_found = true;
            }
        }
    }
    if (!context_found) {
        error = "no context fits without crossing the declared KV quality floor and device reserves";
        return false;
    }
    candidate = std::move(best_candidate);

    candidate.batch = std::max(1U, std::min(input.requested_batch, candidate.context));
    candidate.ubatch = std::max(1U, std::min(input.requested_ubatch, candidate.batch));

    const uint64_t transient_need = std::max(input.mtp_bytes, input.multimodal_bytes);
    const int transient_device = input.transient_device < 0
        ? candidate.devices.front().id : input.transient_device;
    auto transient_it = std::find_if(candidate.devices.begin(), candidate.devices.end(),
        [transient_device](const common_resource_device_plan & device) {
            return device.id == transient_device;
        });
    if (transient_need != 0 && transient_it == candidate.devices.end()) {
        error = "transient device is not present in the resource plan";
        return false;
    }
    candidate.transient_capacity_bytes = transient_need == 0 ? 0
        : std::min(transient_need, transient_it->headroom_bytes);
    candidate.transient_swap = input.mtp_bytes != 0 && input.multimodal_bytes != 0 &&
        candidate.transient_capacity_bytes >= transient_need;
    candidate.draft_resident = input.mtp_bytes != 0 && candidate.transient_capacity_bytes >= input.mtp_bytes;
    if (input.require_draft && !candidate.draft_resident) {
        error = "required draft allocation does not fit while preserving device reserves";
        return false;
    }

    for (auto & device : candidate.devices) {
        device.transient_bytes = device.id == transient_device ? candidate.transient_capacity_bytes : 0;
        const uint64_t expert = candidate.policy == COMMON_MEMORY_POLICY_RESIDENT
            ? 0 : std::min(input.requested_expert_vram_bytes_per_device,
                device.headroom_bytes - std::min(device.headroom_bytes, device.transient_bytes));
        device.expert_cache_bytes = expert;
        device.planned_bytes += device.transient_bytes + expert;
        device.headroom_bytes = device.capacity_bytes - device.reserve_bytes - device.planned_bytes;
    }

    // A CPU-only plan uses the synthetic host device for KV/workspace while
    // the RAM fields cover weights and caches. Reconcile the two views so the
    // same physical bytes cannot be spent twice.
    const auto host = std::find_if(candidate.devices.begin(), candidate.devices.end(),
        [](const common_resource_device_plan & device) { return device.id < 0; });
    if (host != candidate.devices.end() && input.max_ram_bytes != 0) {
        uint64_t combined = candidate.ram_planned_bytes;
        const uint64_t host_non_weight = host->planned_bytes - std::min(host->planned_bytes, input.dense_ram_bytes);
        if (!add_checked(combined, host_non_weight)) {
            error = "host RAM plan overflows 64-bit accounting";
            return false;
        }
        uint64_t excess = combined > input.max_ram_bytes ? combined - input.max_ram_bytes : 0;
        const uint64_t aux_reduction = std::min(excess, candidate.aux_ram_bytes);
        candidate.aux_ram_bytes -= aux_reduction;
        candidate.ram_planned_bytes -= aux_reduction;
        excess -= aux_reduction;
        if (excess != 0 && candidate.policy != COMMON_MEMORY_POLICY_RESIDENT) {
            const uint64_t expert_reduction = std::min(excess, candidate.expert_ram_bytes);
            candidate.expert_ram_bytes -= expert_reduction;
            candidate.ram_planned_bytes -= expert_reduction;
            excess -= expert_reduction;
            if (candidate.expert_ram_bytes != 0 && candidate.expert_ram_bytes < input.min_expert_ram_bytes) {
                const uint64_t remainder = candidate.expert_ram_bytes;
                candidate.expert_ram_bytes = 0;
                candidate.ram_planned_bytes -= remainder;
            }
        }
        if (excess != 0) {
            error = "host weights, KV, and workspace exceed --max-ram";
            return false;
        }
    }

    candidate.reason = "deterministic " + common_memory_policy_name(candidate.policy) +
        " plan at " + common_kv_quality_name(candidate.kv_quality) +
        " KV quality with explicit per-device reserves";
    plan = std::move(candidate);
    return true;
}

std::string common_resource_plan_json(const common_resource_plan & plan) {
    std::ostringstream out;
    out << '{';
    out << "\"policy\":"; json_string(out, common_memory_policy_name(plan.policy));
    out << ",\"backend\":"; json_string(out, common_resource_backend_name(plan.backend));
    out << ",\"kv_quality\":"; json_string(out, common_kv_quality_name(plan.kv_quality));
    out << ",\"context\":" << plan.context << ",\"slots\":" << plan.slots;
    out << ",\"batch\":" << plan.batch << ",\"ubatch\":" << plan.ubatch;
    out << ",\"ram_capacity_bytes\":" << plan.ram_capacity_bytes;
    out << ",\"ram_planned_bytes\":" << plan.ram_planned_bytes;
    out << ",\"expert_ram_bytes\":" << plan.expert_ram_bytes;
    out << ",\"aux_ram_bytes\":" << plan.aux_ram_bytes;
    out << ",\"io_staging_bytes\":" << plan.io_staging_bytes;
    out << ",\"transient_capacity_bytes\":" << plan.transient_capacity_bytes;
    out << ",\"transient_swap\":" << (plan.transient_swap ? "true" : "false");
    out << ",\"draft_resident\":" << (plan.draft_resident ? "true" : "false");
    out << ",\"reason\":"; json_string(out, plan.reason);
    out << ",\"devices\":[";
    for (size_t i = 0; i < plan.devices.size(); ++i) {
        if (i != 0) out << ',';
        const auto & device = plan.devices[i];
        out << "{\"id\":" << device.id
            << ",\"capacity_bytes\":" << device.capacity_bytes
            << ",\"reserve_bytes\":" << device.reserve_bytes
            << ",\"dense_bytes\":" << device.dense_bytes
            << ",\"graph_bytes\":" << device.graph_bytes
            << ",\"kv_bytes\":" << device.kv_bytes
            << ",\"expert_cache_bytes\":" << device.expert_cache_bytes
            << ",\"transient_bytes\":" << device.transient_bytes
            << ",\"planned_bytes\":" << device.planned_bytes
            << ",\"headroom_bytes\":" << device.headroom_bytes << '}';
    }
    out << "]}";
    return out.str();
}

bool common_resource_apply_plan_atomic(
        common_resource_plan & current,
        const common_resource_plan & target,
        const std::function<bool(const common_resource_plan &, std::string &)> & prepare,
        const std::function<void()> & commit,
        const std::function<void()> & rollback,
        std::string & error,
        common_resource_transition_stats * stats) {
    const common_resource_plan original = current;
    bool prepared = false;
    error.clear();
    if (stats) ++stats->attempts;
    try {
        if (!prepare(target, error)) {
            if (stats) ++stats->prepare_failures;
            try {
                rollback();
            } catch (...) {
                // The logical plan is still restored below. A failed prepare
                // remains the primary error reported to the caller.
            }
            if (stats) ++stats->rollbacks;
            current = original;
            if (error.empty()) error = "resource plan preparation failed";
            return false;
        }
        prepared = true;
        current = target;
        commit();
        if (stats) ++stats->commits;
        return true;
    } catch (const std::exception & exception) {
        if (prepared) {
            if (stats) ++stats->commit_failures;
        } else {
            if (stats) ++stats->prepare_failures;
        }
        try {
            rollback();
        } catch (...) {
            // Preserve the logical plan even if a backend rollback hook itself
            // reports failure. The caller receives the original exception.
        }
        if (stats) ++stats->rollbacks;
        current = original;
        error = std::string("resource plan transition failed: ") + exception.what();
        return false;
    } catch (...) {
        if (stats) {
            if (prepared) ++stats->commit_failures;
            else ++stats->prepare_failures;
        }
        try { rollback(); } catch (...) {}
        if (stats) ++stats->rollbacks;
        current = original;
        error = "resource plan transition failed with an unknown exception";
        return false;
    }
}
