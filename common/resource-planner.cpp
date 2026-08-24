#include "resource-planner.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <limits>
#include <set>
#include <sstream>
#include <tuple>

#include <nlohmann/json.hpp>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace {

using calibration_format_key = std::tuple<int32_t, uint64_t, uint64_t, uint64_t>;
using calibration_device_key = std::tuple<std::string, int32_t, uint64_t, uint64_t, uint64_t>;

bool json_positive_u64(const nlohmann::json & value, uint64_t & result) {
    if (value.is_number_unsigned()) {
        result = value.get<uint64_t>();
    } else if (value.is_number_integer()) {
        const int64_t signed_value = value.get<int64_t>();
        if (signed_value <= 0) return false;
        result = uint64_t(signed_value);
    } else {
        return false;
    }
    return result > 0;
}

bool json_format_key(
        const nlohmann::json & value,
        calibration_format_key & key) {
    if (!value.is_object() || !value.contains("ggml_type_id")) return false;
    uint64_t type = 0;
    uint64_t input_width = 0;
    uint64_t expert_width = 0;
    uint64_t bytes = 0;
    if (!json_positive_u64(value["ggml_type_id"], type) ||
            type > uint64_t(std::numeric_limits<int32_t>::max()) ||
            !value.contains("input_width") ||
            !json_positive_u64(value["input_width"], input_width) ||
            !value.contains("expert_width") ||
            !json_positive_u64(value["expert_width"], expert_width) ||
            !value.contains("bytes_per_expert_component") ||
            !json_positive_u64(value["bytes_per_expert_component"], bytes)) return false;
    key = {int32_t(type), input_width, expert_width, bytes};
    return true;
}

bool json_finite_number(
        const nlohmann::json & value,
        double & result,
        double minimum,
        double maximum = std::numeric_limits<double>::infinity()) {
    if (!value.is_number()) return false;
    result = value.get<double>();
    return std::isfinite(result) && result >= minimum && result <= maximum;
}

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

uint64_t mul_div_floor(uint64_t a, uint64_t b, uint64_t divisor) {
#if defined(_MSC_VER) && defined(_M_X64)
    uint64_t high = 0;
    const uint64_t low = _umul128(a, b, &high);
    uint64_t remainder = 0;
    return _udiv128(high, low, divisor, &remainder);
#else
    return (uint64_t) (((__uint128_t) a*b)/divisor);
#endif
}

bool mul_div_ceil_checked(uint64_t a, uint32_t b, uint32_t divisor, uint64_t & result) {
    if (divisor == 0) return false;
    const uint64_t whole = a/divisor;
    const uint64_t remainder = a%divisor;
    if (b != 0 && whole > std::numeric_limits<uint64_t>::max()/b) return false;
    result = whole*b;
    const uint64_t remainder_product = remainder*uint64_t(b);
    const uint64_t rounded = remainder_product/divisor + (remainder_product%divisor != 0 ? 1 : 0);
    return add_checked(result, rounded);
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
    // Request-time owner swaps prepare the incoming module while the outgoing
    // module remains rollback-capable. Reserve their sum as a preparation peak
    // even though steady-state capacity is only the larger owner.
    uint64_t transient_preparation_peak = input.mtp_bytes;
    if (!add_checked(transient_preparation_peak, input.multimodal_bytes)) {
        return false;
    }
    for (const auto & device : input.devices) {
        const uint64_t prefill_staging = plan.expert_prefill_staging_enabled && device.id >= 0
            ? input.requested_expert_prefill_staging_bytes_per_device : 0;
        uint64_t fixed = device.reserve_bytes;
        if (!add_checked(fixed, device.dense_bytes) || !add_checked(fixed, device.graph_bytes) ||
                !add_checked(fixed, prefill_staging) ||
                (device.id == transient_device &&
                    !add_checked(fixed, transient_preparation_peak)) ||
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
            : mul_div_floor(kv_total, usable[i], weight_sum);
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
        device.expert_prefill_staging_bytes = plan.expert_prefill_staging_enabled && source.id >= 0
            ? input.requested_expert_prefill_staging_bytes_per_device : 0;
        device.planned_bytes = source.dense_bytes + source.graph_bytes +
            device.expert_prefill_staging_bytes + share;
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

bool common_parse_transient_policy(const std::string & text, common_transient_policy & policy) {
    if (text == "off") policy = COMMON_TRANSIENT_POLICY_OFF;
    else if (text == "shared") policy = COMMON_TRANSIENT_POLICY_SHARED;
    else if (text == "mtp-only") policy = COMMON_TRANSIENT_POLICY_MTP_ONLY;
    else if (text == "multimodal-only") policy = COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
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

std::string common_transient_policy_name(common_transient_policy policy) {
    switch (policy) {
        case COMMON_TRANSIENT_POLICY_OFF: return "off";
        case COMMON_TRANSIENT_POLICY_SHARED: return "shared";
        case COMMON_TRANSIENT_POLICY_MTP_ONLY: return "mtp-only";
        case COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY: return "multimodal-only";
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
    candidate.expert_prefill_staging_enabled = candidate.policy != COMMON_MEMORY_POLICY_RESIDENT &&
        input.requested_expert_prefill_staging_bytes_per_device != 0;
    if (input.require_expert_prefill_staging && !candidate.expert_prefill_staging_enabled) {
        error = "required expert prefill staging needs a non-resident MoE memory policy";
        return false;
    }

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

    auto find_context = [&](const common_resource_plan & base, common_resource_plan & best_candidate) {
        bool context_found = false;
        for (const auto quality : qualities_descending(input.min_kv_quality)) {
            if (input.kv_bytes_per_token[quality] == 0) continue;
            uint32_t low = aligned_minimum/alignment;
            uint32_t high = aligned_ceiling/alignment;
            uint32_t best = 0;
            while (low <= high) {
                const uint32_t middle_units = low + (high - low)/2;
                const uint32_t middle = middle_units*alignment;
                common_resource_plan attempt = base;
                if (allocate_context(input, attempt, middle, quality)) {
                    best = middle;
                    low = middle_units + 1;
                } else {
                    if (middle_units == 0) break;
                    high = middle_units - 1;
                }
            }
            if (best == 0) continue;
            common_resource_plan attempt = base;
            attempt.kv_quality = quality;
            attempt.context = best;
            if (!allocate_context(input, attempt, best, quality)) return false;
            if (!context_found || better_context_choice(
                    input.preference, aligned_ceiling, best, quality,
                    best_candidate.context, best_candidate.kv_quality)) {
                best_candidate = std::move(attempt);
                context_found = true;
            }
        }
        return context_found;
    };

    common_resource_plan best_candidate;
    bool optional_prefill_staging_dropped = false;
    bool context_found = find_context(candidate, best_candidate);
    if (!context_found && candidate.expert_prefill_staging_enabled &&
            !input.require_expert_prefill_staging) {
        candidate.expert_prefill_staging_enabled = false;
        optional_prefill_staging_dropped = true;
        context_found = find_context(candidate, best_candidate);
    }
    if (!context_found) {
        error = input.require_expert_prefill_staging && candidate.expert_prefill_staging_enabled
            ? "required expert prefill staging does not fit without crossing the context/KV floor or device reserves"
            : "no context fits without crossing the declared KV quality floor and device reserves";
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
    candidate.transient_device = transient_device;
    candidate.transient_mtp_bytes = input.mtp_bytes;
    candidate.transient_multimodal_bytes = input.multimodal_bytes;
    candidate.transient_capacity_bytes = 0;
    candidate.transient_policy = COMMON_TRANSIENT_POLICY_OFF;
    if (input.mtp_bytes != 0 && input.multimodal_bytes != 0) {
        if (transient_it->headroom_bytes < transient_need) {
            error = "configured shared transient modules do not fit while preserving the device reserve";
            return false;
        }
        candidate.transient_policy = COMMON_TRANSIENT_POLICY_SHARED;
        candidate.transient_capacity_bytes = transient_need;
    } else if (input.mtp_bytes != 0) {
        if (transient_it->headroom_bytes < input.mtp_bytes) {
            error = "configured MTP transient module does not fit while preserving the device reserve";
            return false;
        }
        candidate.transient_policy = COMMON_TRANSIENT_POLICY_MTP_ONLY;
        candidate.transient_capacity_bytes = input.mtp_bytes;
    } else if (input.multimodal_bytes != 0) {
        if (transient_it->headroom_bytes < input.multimodal_bytes) {
            error = "configured multimodal transient module does not fit while preserving the device reserve";
            return false;
        }
        candidate.transient_policy = COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
        candidate.transient_capacity_bytes = input.multimodal_bytes;
    }
    candidate.transient_swap = candidate.transient_policy == COMMON_TRANSIENT_POLICY_SHARED;
    candidate.draft_resident = candidate.transient_policy == COMMON_TRANSIENT_POLICY_SHARED ||
        candidate.transient_policy == COMMON_TRANSIENT_POLICY_MTP_ONLY;
    if (input.require_draft && !candidate.draft_resident) {
        error = "required draft allocation does not fit while preserving device reserves";
        return false;
    }

    for (auto & device : candidate.devices) {
        device.transient_bytes = device.id == transient_device ? candidate.transient_capacity_bytes : 0;
        uint64_t transient_preparation_reserve = device.transient_bytes;
        if (device.id == transient_device &&
                candidate.transient_policy == COMMON_TRANSIENT_POLICY_SHARED &&
                !add_checked(transient_preparation_reserve,
                    std::min(input.mtp_bytes, input.multimodal_bytes))) {
            error = "shared transient preparation reserve overflows 64-bit bytes";
            return false;
        }
        const uint64_t expert = candidate.policy == COMMON_MEMORY_POLICY_RESIDENT
            ? 0 : std::min(input.requested_expert_vram_bytes_per_device,
                device.headroom_bytes -
                    std::min(device.headroom_bytes, transient_preparation_reserve));
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
    if (optional_prefill_staging_dropped) {
        candidate.reason += "; optional expert prefill staging disabled to preserve the context/KV floor";
    }
    plan = std::move(candidate);
    return true;
}

bool common_expert_split_solve(
        const common_expert_split_input & input,
        common_expert_split_plan & plan,
        std::string & error) {
    error.clear();
    plan = {};
    if (!input.calibration_complete) {
        error = "hardware calibration is incomplete";
        return false;
    }
    if (input.misses == 0) return true;
    if (!std::isfinite(input.cpu_ns_per_expert) || !std::isfinite(input.upload_ns_per_expert) ||
            !(input.cpu_ns_per_expert > 0) || !(input.upload_ns_per_expert > 0)) {
        error = "expert split requires positive contended CPU and upload costs";
        return false;
    }
    if (!std::isfinite(input.minimum_confidence) || !std::isfinite(input.cpu_confidence) ||
            !std::isfinite(input.upload_confidence) ||
            input.minimum_confidence < 0 || input.minimum_confidence > 1 ||
            input.cpu_confidence < input.minimum_confidence ||
            input.upload_confidence < input.minimum_confidence) {
        error = "hardware calibration confidence is below the planner threshold";
        return false;
    }
    if (!std::isfinite(input.hysteresis_fraction) ||
            input.hysteresis_fraction < 0 || input.hysteresis_fraction >= 1) {
        error = "expert split hysteresis must be in [0, 1)";
        return false;
    }

    auto completion = [&](uint32_t uploads) {
        const double cpu_ns = double(input.misses - uploads)*input.cpu_ns_per_expert;
        const double upload_ns = double(uploads)*input.upload_ns_per_expert;
        return std::max(cpu_ns, upload_ns);
    };
    uint32_t best_uploads = 0;
    double best_ns = completion(0);
    for (uint32_t uploads = 1; uploads <= input.misses; ++uploads) {
        const double candidate = completion(uploads);
        if (candidate < best_ns) {
            best_ns = candidate;
            best_uploads = uploads;
        }
    }

    bool retained = false;
    if (input.previous_upload_experts >= 0 &&
            uint32_t(input.previous_upload_experts) <= input.misses) {
        const uint32_t previous = uint32_t(input.previous_upload_experts);
        const double previous_ns = completion(previous);
        const double required_improvement = previous_ns*input.hysteresis_fraction;
        if (previous_ns - best_ns <= required_improvement) {
            best_uploads = previous;
            best_ns = previous_ns;
            retained = true;
        }
    }
    plan.upload_experts = best_uploads;
    plan.cpu_experts = input.misses - best_uploads;
    plan.predicted_step_ns = best_ns;
    plan.retained_by_hysteresis = retained;
    return true;
}

bool common_expert_calibration_parse_json(
        const std::string & text,
        common_expert_calibration_profile & profile,
        std::string & error) {
    error.clear();
    profile = {};
    try {
        const nlohmann::json root = nlohmann::json::parse(text);
        if (!root.is_object()) {
            error = "hardware profile root must be an object";
            return false;
        }
        const auto & source = root.at("benchmark_source");
        if (!source.is_object() || source.value("planner_ready", false) != true ||
                source.value("calibration_level", std::string()) != "planner") {
            error = "hardware profile is not planner-ready";
            return false;
        }
        const auto & measurements = root.at("measurements");
        const auto & cpu = measurements.at("cpu_moe");
        const auto & cpu_profiles = cpu.at("model_profiles");
        if (cpu.value("status", std::string()) != "measured" ||
                !cpu_profiles.is_array() || cpu_profiles.empty()) {
            error = "hardware profile has no model-backed CPU calibration";
            return false;
        }
        std::set<calibration_format_key> formats;
        for (const auto & item : cpu_profiles) {
            calibration_format_key key;
            double nrmse = 0;
            double confidence = 0;
            double relative_standard_error = 0;
            uint64_t sample_count = 0;
            if (!json_format_key(item, key) || !formats.insert(key).second ||
                    item.value("correctness", std::string()) !=
                            "single-thread-and-dequantized-scalar-reference" ||
                    !item.contains("independent_reference_nrmse") ||
                    !json_finite_number(item["independent_reference_nrmse"], nrmse, 0, 0.08) ||
                    !item.contains("confidence") ||
                    !json_finite_number(item["confidence"], confidence, 0.80, 1.0) ||
                    !item.contains("relative_standard_error") ||
                    !json_finite_number(item["relative_standard_error"], relative_standard_error, 0) ||
                    !item.contains("sample_count") ||
                    !json_positive_u64(item["sample_count"], sample_count) || sample_count < 7) {
                error = "hardware profile CPU calibration is incomplete or duplicated";
                return false;
            }
        }

        const auto & contention = measurements.at("cpu_cache_contention");
        const auto & devices = contention.at("devices");
        if (contention.value("status", std::string()) != "measured" ||
                contention.value("upload_path", std::string()) !=
                        "pread_to_bounded_ram_lease_to_async_upload" ||
                contention.value("distribution", std::string()) != "warm-steady-state" ||
                !devices.is_array() || devices.empty()) {
            error = "hardware profile contention path is incomplete";
            return false;
        }

        common_expert_calibration_profile candidate;
        std::set<std::string> backends;
        std::set<calibration_device_key> contention_keys;
        for (const auto & device : devices) {
            const std::string backend = device.value("backend", std::string());
            const auto & entries = device.at("profiles");
            if (backend.empty() || !backends.insert(backend).second ||
                    !entries.is_array() || entries.size() != formats.size()) {
                error = "hardware profile contention devices are incomplete or duplicated";
                return false;
            }
            std::set<calibration_format_key> device_formats;
            for (const auto & item : entries) {
                calibration_format_key format;
                double cpu_ns = 0;
                double upload_ns = 0;
                double cpu_confidence = 0;
                double upload_confidence = 0;
                double cpu_rse = 0;
                double upload_rse = 0;
                uint64_t cpu_samples = 0;
                uint64_t upload_samples = 0;
                if (!json_format_key(item, format) || formats.count(format) != 1 ||
                        !device_formats.insert(format).second ||
                        !item.contains("cpu_ns_per_expert_component") ||
                        !json_finite_number(item["cpu_ns_per_expert_component"], cpu_ns, 0) || cpu_ns == 0 ||
                        !item.contains("upload_ns_per_expert_component") ||
                        !json_finite_number(item["upload_ns_per_expert_component"], upload_ns, 0) || upload_ns == 0 ||
                        !item.contains("cpu_confidence") ||
                        !json_finite_number(item["cpu_confidence"], cpu_confidence, 0.80, 1.0) ||
                        !item.contains("upload_confidence") ||
                        !json_finite_number(item["upload_confidence"], upload_confidence, 0.80, 1.0) ||
                        !item.contains("cpu_relative_standard_error") ||
                        !json_finite_number(item["cpu_relative_standard_error"], cpu_rse, 0) ||
                        !item.contains("upload_relative_standard_error") ||
                        !json_finite_number(item["upload_relative_standard_error"], upload_rse, 0) ||
                        !item.contains("cpu_sample_count") ||
                        !json_positive_u64(item["cpu_sample_count"], cpu_samples) || cpu_samples < 7 ||
                        !item.contains("upload_sample_count") ||
                        !json_positive_u64(item["upload_sample_count"], upload_samples) || upload_samples < 7) {
                    error = "hardware profile contention entry is invalid or below confidence";
                    return false;
                }
                const auto [type, input_width, expert_width, bytes] = format;
                calibration_device_key device_key = {
                    backend, type, input_width, expert_width, bytes
                };
                if (!contention_keys.insert(device_key).second) {
                    error = "hardware profile contention entry is duplicated";
                    return false;
                }
                candidate.entries.push_back({
                    {backend, type, input_width, expert_width, bytes},
                    cpu_ns, upload_ns, cpu_confidence, upload_confidence
                });
            }
            if (device_formats != formats) {
                error = "hardware profile contention format matrix is incomplete";
                return false;
            }
        }

        const auto & upload = measurements.at("expert_cache_upload");
        const auto & leases = upload.at("lease_upload_profiles");
        if (upload.value("status", std::string()) != "measured" ||
                !leases.is_array() || leases.size() != contention_keys.size()) {
            error = "hardware profile leased-upload matrix is incomplete";
            return false;
        }
        std::set<calibration_device_key> lease_keys;
        for (const auto & item : leases) {
            calibration_format_key format;
            double confidence = 0;
            double rse = 0;
            uint64_t samples = 0;
            const std::string backend = item.value("backend", std::string());
            if (backend.empty() || backends.count(backend) != 1 ||
                    item.value("storage_backend", std::string()) != "pread" ||
                    item.value("distribution", std::string()) != "warm-steady-state" ||
                    !json_format_key(item, format) ||
                    !item.contains("confidence") ||
                    !json_finite_number(item["confidence"], confidence, 0.80, 1.0) ||
                    !item.contains("relative_standard_error") ||
                    !json_finite_number(item["relative_standard_error"], rse, 0) ||
                    !item.contains("sample_count") ||
                    !json_positive_u64(item["sample_count"], samples) || samples < 7) {
                error = "hardware profile leased-upload entry is invalid or below confidence";
                return false;
            }
            const auto [type, input_width, expert_width, bytes] = format;
            if (!lease_keys.insert({backend, type, input_width, expert_width, bytes}).second) {
                error = "hardware profile leased-upload entry is duplicated";
                return false;
            }
        }
        if (lease_keys != contention_keys) {
            error = "hardware profile leased-upload matrix does not match contention calibration";
            return false;
        }
        profile = std::move(candidate);
        return true;
    } catch (const nlohmann::json::exception & exception) {
        error = std::string("invalid hardware profile JSON: ") + exception.what();
        return false;
    } catch (const std::exception & exception) {
        error = std::string("invalid hardware profile: ") + exception.what();
        return false;
    }
}

bool common_expert_calibration_lookup(
        const common_expert_calibration_profile & profile,
        const common_expert_calibration_key & key,
        common_expert_calibration_entry & entry) {
    const auto found = std::find_if(profile.entries.begin(), profile.entries.end(),
            [&](const common_expert_calibration_entry & candidate) {
                return candidate.key == key;
            });
    if (found == profile.entries.end()) return false;
    entry = *found;
    return true;
}

bool common_expert_split_solve_calibrated(
        const common_expert_calibration_profile & profile,
        const std::vector<common_expert_calibration_key> & components,
        uint32_t misses,
        int32_t previous_upload_experts,
        double hysteresis_fraction,
        common_expert_split_plan & plan,
        std::string & error) {
    error.clear();
    if (components.empty()) {
        error = "expert split requires calibrated component geometry";
        return false;
    }
    common_expert_split_input input;
    input.misses = misses;
    input.previous_upload_experts = previous_upload_experts;
    input.hysteresis_fraction = hysteresis_fraction;
    input.cpu_confidence = 1.0;
    input.upload_confidence = 1.0;
    for (const auto & key : components) {
        common_expert_calibration_entry entry;
        if (!common_expert_calibration_lookup(profile, key, entry)) {
            error = "hardware calibration has no entry for a required expert component";
            return false;
        }
        if (entry.cpu_ns_per_expert_component >
                    std::numeric_limits<double>::max() - input.cpu_ns_per_expert ||
                entry.upload_ns_per_expert_component >
                    std::numeric_limits<double>::max() - input.upload_ns_per_expert) {
            error = "calibrated expert component costs overflow";
            return false;
        }
        input.cpu_ns_per_expert += entry.cpu_ns_per_expert_component;
        input.upload_ns_per_expert += entry.upload_ns_per_expert_component;
        input.cpu_confidence = std::min(input.cpu_confidence, entry.cpu_confidence);
        input.upload_confidence = std::min(input.upload_confidence, entry.upload_confidence);
    }
    input.calibration_complete = true;
    return common_expert_split_solve(input, plan, error);
}

bool common_resource_rebalance_target(
        const common_resource_plan & current,
        const common_resource_rebalance_request & request,
        common_resource_plan & target,
        std::string & error) {
    error.clear();
    if (current.context == 0 || current.slots == 0 || current.devices.empty()) {
        error = "current resource plan is incomplete";
        return false;
    }
    const uint32_t context = request.context == 0 ? current.context : request.context;
    if (context < current.slots || context%current.slots != 0) {
        error = "target context must provide an equal whole-token capacity for every slot";
        return false;
    }
    if (request.set_expert_cache_bytes_per_device &&
            request.expert_cache_bytes_per_device > 0 &&
            current.policy == COMMON_MEMORY_POLICY_RESIDENT) {
        error = "resident policy has no mutable deferred-expert cache";
        return false;
    }

    target = current;
    target.context = context;
    if (request.set_transient_policy) {
        uint64_t transient_capacity = 0;
        switch (request.transient_policy) {
            case COMMON_TRANSIENT_POLICY_OFF:
                break;
            case COMMON_TRANSIENT_POLICY_SHARED:
                if (current.transient_mtp_bytes == 0 ||
                        current.transient_multimodal_bytes == 0) {
                    error = "shared transient policy requires configured MTP and multimodal modules";
                    return false;
                }
                transient_capacity = std::max(
                    current.transient_mtp_bytes,
                    current.transient_multimodal_bytes);
                break;
            case COMMON_TRANSIENT_POLICY_MTP_ONLY:
                if (current.transient_mtp_bytes == 0) {
                    error = "mtp-only transient policy requires a configured MTP module";
                    return false;
                }
                transient_capacity = current.transient_mtp_bytes;
                break;
            case COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY:
                if (current.transient_multimodal_bytes == 0) {
                    error = "multimodal-only transient policy requires a configured multimodal module";
                    return false;
                }
                transient_capacity = current.transient_multimodal_bytes;
                break;
            default:
                error = "unknown transient policy";
                return false;
        }
        target.transient_policy = request.transient_policy;
        target.transient_capacity_bytes = transient_capacity;
        target.transient_swap = request.transient_policy == COMMON_TRANSIENT_POLICY_SHARED;
        target.draft_resident = request.transient_policy == COMMON_TRANSIENT_POLICY_SHARED ||
            request.transient_policy == COMMON_TRANSIENT_POLICY_MTP_ONLY;
    }
    bool has_accelerator = false;
    bool has_transient_device = false;
    for (auto & device : target.devices) {
        if (device.capacity_bytes < device.reserve_bytes) {
            error = "device capacity is below its immutable reserve";
            return false;
        }
        uint64_t kv_bytes = 0;
        if (!mul_div_ceil_checked(device.kv_bytes, context, current.context, kv_bytes)) {
            error = "target KV allocation overflows 64-bit accounting";
            return false;
        }
        device.kv_bytes = kv_bytes;
        if (device.id >= 0) {
            has_accelerator = true;
            if (request.set_expert_cache_bytes_per_device) {
                device.expert_cache_bytes = request.expert_cache_bytes_per_device;
            }
        } else if (request.set_expert_cache_bytes_per_device) {
            device.expert_cache_bytes = 0;
        }
        if (device.id == target.transient_device) {
            has_transient_device = true;
            if (request.set_transient_policy) {
                device.transient_bytes = target.transient_capacity_bytes;
            }
        } else if (request.set_transient_policy) {
            device.transient_bytes = 0;
        }

        uint64_t planned = 0;
        if (!add_checked(planned, device.dense_bytes) ||
                !add_checked(planned, device.graph_bytes) ||
                !add_checked(planned, device.kv_bytes) ||
                !add_checked(planned, device.expert_cache_bytes) ||
                !add_checked(planned, device.expert_prefill_staging_bytes) ||
                !add_checked(planned, device.transient_bytes)) {
            error = "target device plan overflows 64-bit accounting";
            return false;
        }
        const uint64_t usable = device.capacity_bytes - device.reserve_bytes;
        if (planned > usable) {
            error = "target KV/expert/transient allocation crosses a device reserve";
            return false;
        }
        device.planned_bytes = planned;
        device.headroom_bytes = usable - planned;
        if (target.transient_policy == COMMON_TRANSIENT_POLICY_SHARED &&
                device.id == target.transient_device &&
                device.headroom_bytes < std::min(
                    target.transient_mtp_bytes,
                    target.transient_multimodal_bytes)) {
            error = "target allocation consumes the rollback reserve required for shared transient owner swaps";
            return false;
        }
    }
    if (request.set_expert_cache_bytes_per_device &&
            request.expert_cache_bytes_per_device > 0 && !has_accelerator) {
        error = "target expert cache requires an accelerator";
        return false;
    }
    if (request.set_transient_policy && target.transient_capacity_bytes != 0 &&
            !has_transient_device) {
        error = "serialized transient device is not present in the resource plan";
        return false;
    }
    target.reason = "validated runtime KV/expert/transient rebalance target; no resources mutated";
    return true;
}

bool common_resource_rebalance_preparation_peak(
        const common_resource_plan & current,
        const common_resource_plan & target,
        common_resource_preparation_peak & report,
        std::string & error,
        bool force_expert_cache_preparation,
        bool force_transient_preparation) {
    error.clear();
    if (current.devices.empty() || current.devices.size() != target.devices.size()) {
        error = "current and target resource plans have different device topologies";
        return false;
    }

    common_resource_preparation_peak candidate;
    candidate.prepares_kv = current.context != target.context;
    candidate.prepares_expert_cache = force_expert_cache_preparation;
    candidate.prepares_transient = force_transient_preparation ||
        current.transient_policy != target.transient_policy ||
        current.transient_capacity_bytes != target.transient_capacity_bytes;
    if (current.transient_device != target.transient_device ||
            current.transient_mtp_bytes != target.transient_mtp_bytes ||
            current.transient_multimodal_bytes != target.transient_multimodal_bytes) {
        error = "runtime rebalance cannot change transient topology or module bounds";
        return false;
    }
    for (size_t index = 0; index < current.devices.size(); ++index) {
        const auto & live = current.devices[index];
        const auto & next = target.devices[index];
        if (live.id != next.id || live.capacity_bytes != next.capacity_bytes ||
                live.reserve_bytes != next.reserve_bytes) {
            error = "runtime rebalance cannot change device topology, capacity, or reserve";
            return false;
        }
        if (live.dense_bytes != next.dense_bytes || live.graph_bytes != next.graph_bytes ||
                live.expert_prefill_staging_bytes != next.expert_prefill_staging_bytes) {
            error = "runtime KV/expert rebalance cannot change immutable device allocations";
            return false;
        }
        candidate.prepares_kv = candidate.prepares_kv || live.kv_bytes != next.kv_bytes;
        candidate.prepares_expert_cache = candidate.prepares_expert_cache ||
            live.expert_cache_bytes != next.expert_cache_bytes;
        candidate.prepares_transient = candidate.prepares_transient ||
            live.transient_bytes != next.transient_bytes;
    }

    uint64_t transient_additional_peak = 0;
    if (candidate.prepares_transient && current.transient_device >= 0) {
        enum transient_owner {
            TRANSIENT_OWNER_NONE,
            TRANSIENT_OWNER_MTP,
            TRANSIENT_OWNER_MULTIMODAL,
        };
        transient_owner possible[3] = {};
        size_t possible_count = 0;
        if (force_transient_preparation) {
            possible[possible_count++] = TRANSIENT_OWNER_NONE;
            possible[possible_count++] = TRANSIENT_OWNER_MTP;
            possible[possible_count++] = TRANSIENT_OWNER_MULTIMODAL;
        } else {
            switch (current.transient_policy) {
                case COMMON_TRANSIENT_POLICY_OFF:
                    possible[possible_count++] = TRANSIENT_OWNER_NONE;
                    break;
                case COMMON_TRANSIENT_POLICY_MTP_ONLY:
                    possible[possible_count++] = TRANSIENT_OWNER_MTP;
                    break;
                case COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY:
                    possible[possible_count++] = TRANSIENT_OWNER_MULTIMODAL;
                    break;
                case COMMON_TRANSIENT_POLICY_SHARED:
                    possible[possible_count++] = TRANSIENT_OWNER_MTP;
                    possible[possible_count++] = TRANSIENT_OWNER_MULTIMODAL;
                    break;
                default:
                    error = "current resource plan has an invalid transient policy";
                    return false;
            }
        }
        const auto owner_bytes = [&](transient_owner owner) {
            return owner == TRANSIENT_OWNER_MTP ? current.transient_mtp_bytes :
                owner == TRANSIENT_OWNER_MULTIMODAL
                    ? current.transient_multimodal_bytes : uint64_t(0);
        };
        uint64_t combined_owner_peak = 0;
        for (size_t i = 0; i < possible_count; ++i) {
            const transient_owner live_owner = possible[i];
            transient_owner target_owner = TRANSIENT_OWNER_NONE;
            switch (target.transient_policy) {
                case COMMON_TRANSIENT_POLICY_OFF:
                    break;
                case COMMON_TRANSIENT_POLICY_MTP_ONLY:
                    target_owner = TRANSIENT_OWNER_MTP;
                    break;
                case COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY:
                    target_owner = TRANSIENT_OWNER_MULTIMODAL;
                    break;
                case COMMON_TRANSIENT_POLICY_SHARED:
                    target_owner = live_owner == TRANSIENT_OWNER_NONE
                        ? TRANSIENT_OWNER_MTP : live_owner;
                    break;
                default:
                    error = "target resource plan has an invalid transient policy";
                    return false;
            }
            uint64_t owner_peak = owner_bytes(live_owner);
            if (target_owner != TRANSIENT_OWNER_NONE &&
                    (force_transient_preparation || target_owner != live_owner) &&
                    !add_checked(owner_peak, owner_bytes(target_owner))) {
                error = "transient owner preparation peak overflows 64-bit accounting";
                return false;
            }
            combined_owner_peak = std::max(combined_owner_peak, owner_peak);
        }

        const auto live_device = std::find_if(
            current.devices.begin(), current.devices.end(),
            [&](const common_resource_device_plan & device) {
                return device.id == current.transient_device;
            });
        if (live_device == current.devices.end()) {
            error = "current resource plan omitted its transient device";
            return false;
        }
        transient_additional_peak = combined_owner_peak > live_device->transient_bytes
            ? combined_owner_peak - live_device->transient_bytes : 0;
    }

    candidate.devices.reserve(current.devices.size());
    for (size_t index = 0; index < current.devices.size(); ++index) {
        const auto & live = current.devices[index];
        const auto & next = target.devices[index];
        if (live.capacity_bytes < live.reserve_bytes) {
            error = "device capacity is below its immutable reserve";
            return false;
        }

        uint64_t current_live = 0;
        if (!add_checked(current_live, live.dense_bytes) ||
                !add_checked(current_live, live.graph_bytes) ||
                !add_checked(current_live, live.kv_bytes) ||
                !add_checked(current_live, live.expert_cache_bytes) ||
                !add_checked(current_live, live.expert_prefill_staging_bytes) ||
                !add_checked(current_live, live.transient_bytes)) {
            error = "current device plan overflows 64-bit accounting";
            return false;
        }
        uint64_t target_live = 0;
        if (!add_checked(target_live, next.dense_bytes) ||
                !add_checked(target_live, next.graph_bytes) ||
                !add_checked(target_live, next.kv_bytes) ||
                !add_checked(target_live, next.expert_cache_bytes) ||
                !add_checked(target_live, next.expert_prefill_staging_bytes) ||
                !add_checked(target_live, next.transient_bytes)) {
            error = "target device plan overflows 64-bit accounting";
            return false;
        }
        if (current_live != live.planned_bytes || target_live != next.planned_bytes) {
            error = "runtime rebalance received stale device planned-byte accounting";
            return false;
        }

        const uint64_t usable = live.capacity_bytes - live.reserve_bytes;
        if (current_live > usable || target_live > usable) {
            error = "current or target device plan crosses a device reserve";
            return false;
        }

        common_resource_device_preparation_peak device;
        device.id = live.id;
        device.capacity_bytes = live.capacity_bytes;
        device.reserve_bytes = live.reserve_bytes;
        device.current_live_bytes = current_live;
        device.target_live_bytes = target_live;
        device.prepared_kv_bytes = candidate.prepares_kv ? next.kv_bytes : 0;
        device.prepared_expert_cache_bytes = candidate.prepares_expert_cache
            ? next.expert_cache_bytes : 0;
        device.prepared_transient_bytes = candidate.prepares_transient &&
                live.id == current.transient_device
            ? transient_additional_peak : 0;
        device.peak_bytes = current_live;
        if (!add_checked(device.peak_bytes, device.prepared_kv_bytes) ||
                !add_checked(device.peak_bytes, device.prepared_expert_cache_bytes) ||
                !add_checked(device.peak_bytes, device.prepared_transient_bytes)) {
            error = "resource preparation peak overflows 64-bit accounting on device " +
                std::to_string(device.id);
            return false;
        }
        if (device.peak_bytes > usable) {
            error = "resource preparation peak crosses the reserve/headroom on device " +
                std::to_string(device.id) + ": requires " +
                std::to_string(device.peak_bytes) + " bytes with " +
                std::to_string(usable) + " usable";
            return false;
        }
        device.peak_headroom_bytes = usable - device.peak_bytes;
        candidate.devices.push_back(device);
    }

    report = std::move(candidate);
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
    out << ",\"transient_device\":" << plan.transient_device;
    out << ",\"transient_policy\":"; json_string(out, common_transient_policy_name(plan.transient_policy));
    out << ",\"transient_mtp_bytes\":" << plan.transient_mtp_bytes;
    out << ",\"transient_multimodal_bytes\":" << plan.transient_multimodal_bytes;
    out << ",\"transient_capacity_bytes\":" << plan.transient_capacity_bytes;
    out << ",\"expert_prefill_staging_enabled\":"
        << (plan.expert_prefill_staging_enabled ? "true" : "false");
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
            << ",\"expert_prefill_staging_bytes\":" << device.expert_prefill_staging_bytes
            << ",\"transient_bytes\":" << device.transient_bytes
            << ",\"planned_bytes\":" << device.planned_bytes
            << ",\"headroom_bytes\":" << device.headroom_bytes << '}';
    }
    out << "]}";
    return out.str();
}

std::string common_resource_preparation_peak_json(const common_resource_preparation_peak & report) {
    std::ostringstream out;
    out << '{';
    out << "\"prepares_kv\":" << (report.prepares_kv ? "true" : "false");
    out << ",\"prepares_expert_cache\":" << (report.prepares_expert_cache ? "true" : "false");
    out << ",\"prepares_transient\":" << (report.prepares_transient ? "true" : "false");
    out << ",\"devices\":[";
    for (size_t i = 0; i < report.devices.size(); ++i) {
        if (i != 0) out << ',';
        const auto & device = report.devices[i];
        out << "{\"id\":" << device.id
            << ",\"capacity_bytes\":" << device.capacity_bytes
            << ",\"reserve_bytes\":" << device.reserve_bytes
            << ",\"current_live_bytes\":" << device.current_live_bytes
            << ",\"target_live_bytes\":" << device.target_live_bytes
            << ",\"prepared_kv_bytes\":" << device.prepared_kv_bytes
            << ",\"prepared_expert_cache_bytes\":" << device.prepared_expert_cache_bytes
            << ",\"prepared_transient_bytes\":" << device.prepared_transient_bytes
            << ",\"peak_bytes\":" << device.peak_bytes
            << ",\"peak_headroom_bytes\":" << device.peak_headroom_bytes << '}';
    }
    out << "]}";
    return out.str();
}

uint64_t common_resource_cache_limit_bytes(int64_t capacity_mib) {
    constexpr uint64_t mib = 1024ULL*1024ULL;
    if (capacity_mib <= 0) {
        return 0;
    }
    return mul_saturated((uint64_t) capacity_mib, mib);
}

bool common_resource_should_enable_fit(bool has_accelerator, int32_t n_cpu_moe) {
    return has_accelerator && n_cpu_moe == 0;
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
