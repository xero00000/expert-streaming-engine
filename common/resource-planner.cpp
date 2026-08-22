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
