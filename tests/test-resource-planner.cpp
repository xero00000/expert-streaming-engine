#include "resource-planner.h"

#include <cstdint>
#include <string>
#include <stdexcept>

namespace {

#define REQUIRE(condition) do { \
    if (!(condition)) throw std::runtime_error("requirement failed: " #condition); \
} while (0)

constexpr uint64_t MiB = 1024ULL*1024ULL;

common_resource_plan_input base_input() {
    common_resource_plan_input input;
    input.model_is_moe = true;
    input.max_ram_bytes = 40ULL*1024*MiB;
    input.dense_ram_bytes = 8ULL*1024*MiB;
    input.expert_source_bytes = 24ULL*1024*MiB;
    input.requested_expert_ram_bytes = 16ULL*1024*MiB;
    input.requested_aux_ram_bytes = 4ULL*1024*MiB;
    input.requested_expert_vram_bytes_per_device = 512*MiB;
    input.io_staging_bytes = 256*MiB;
    input.mtp_bytes = 1200*MiB;
    input.multimodal_bytes = 1500*MiB;
    input.requested_context = 128*1024;
    input.max_context = 128*1024;
    input.min_context = 4096;
    input.slots = 2;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_TURBO4] = 64*1024;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_TURBO8] = 96*1024;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_Q8] = 128*1024;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_F16] = 256*1024;
    input.devices = {
        { 0, 16ULL*1024*MiB, 1024*MiB, 4ULL*1024*MiB, 512*MiB },
        { 1,  8ULL*1024*MiB, 1024*MiB, 2ULL*1024*MiB, 256*MiB },
    };
    input.transient_device = 1;
    return input;
}

void test_deterministic_budget_and_json() {
    auto input = base_input();
    common_resource_plan first;
    common_resource_plan second;
    std::string error;
    REQUIRE(common_resource_plan_solve(input, first, error));
    REQUIRE(common_resource_plan_solve(input, second, error));
    REQUIRE(common_resource_plan_json(first) == common_resource_plan_json(second));
    REQUIRE(first.policy == COMMON_MEMORY_POLICY_CACHE);
    REQUIRE(first.expert_ram_bytes == input.requested_expert_ram_bytes);
    REQUIRE(first.aux_ram_bytes == 4ULL*1024*MiB);
    REQUIRE(first.context == input.requested_context);
    REQUIRE(first.kv_quality == COMMON_KV_QUALITY_TURBO8);
    REQUIRE(first.transient_swap);
    REQUIRE(first.devices[0].transient_bytes == 0);
    REQUIRE(first.devices[1].transient_bytes == input.multimodal_bytes);
    for (const auto & device : first.devices) {
        REQUIRE(device.expert_cache_bytes <= input.requested_expert_vram_bytes_per_device);
        REQUIRE(device.planned_bytes + device.reserve_bytes <= device.capacity_bytes);
    }
    const std::string json = common_resource_plan_json(first);
    REQUIRE(json.find("\"policy\":\"cache\"") != std::string::npos);
    REQUIRE(json.find("\"kv_quality\":\"turbo8\"") != std::string::npos);
}

void test_preference_tradeoff() {
    auto input = base_input();
    input.devices = {{ 0, 12ULL*1024*MiB, 1024*MiB, 1024*MiB, 0 }};
    input.transient_device = 0;
    input.mtp_bytes = 0;
    input.multimodal_bytes = 0;

    common_resource_plan throughput;
    common_resource_plan latency;
    std::string error;
    input.preference = COMMON_RESOURCE_PREFERENCE_THROUGHPUT;
    REQUIRE(common_resource_plan_solve(input, throughput, error));
    input.preference = COMMON_RESOURCE_PREFERENCE_LATENCY;
    REQUIRE(common_resource_plan_solve(input, latency, error));
    REQUIRE(throughput.context > latency.context);
    REQUIRE(throughput.kv_quality < latency.kv_quality);
    REQUIRE(latency.kv_quality == COMMON_KV_QUALITY_F16);
}

void test_no_silent_fallbacks() {
    auto input = base_input();
    common_resource_plan plan;
    std::string error;
    input.backend_available = false;
    REQUIRE(!common_resource_plan_solve(input, plan, error));
    REQUIRE(error.find("fallback is forbidden") != std::string::npos);

    input = base_input();
    input.min_kv_quality = COMMON_KV_QUALITY_F16;
    input.devices[0].free_bytes = input.devices[0].reserve_bytes + input.devices[0].dense_bytes + 1;
    input.devices[1].free_bytes = input.devices[1].reserve_bytes + input.devices[1].dense_bytes + 1;
    REQUIRE(!common_resource_plan_solve(input, plan, error));
    REQUIRE(error.find("quality floor") != std::string::npos);
}

void test_atomic_rollback() {
    common_resource_plan current;
    current.context = 4096;
    common_resource_plan target;
    target.context = 8192;
    std::string error;
    int rollbacks = 0;
    common_resource_transition_stats stats;
    const bool applied = common_resource_apply_plan_atomic(
        current, target,
        [](const common_resource_plan &, std::string & prepare_error) {
            prepare_error = "injected allocation failure";
            return false;
        },
        [] {},
        [&rollbacks] { ++rollbacks; },
        error, &stats);
    REQUIRE(!applied);
    REQUIRE(current.context == 4096);
    REQUIRE(rollbacks == 1);
    REQUIRE(error == "injected allocation failure");
    REQUIRE(stats.attempts == 1 && stats.prepare_failures == 1 && stats.rollbacks == 1);

    const bool commit_applied = common_resource_apply_plan_atomic(
        current, target,
        [](const common_resource_plan &, std::string &) { return true; },
        [] { throw std::runtime_error("injected commit failure"); },
        [&rollbacks] { ++rollbacks; },
        error, &stats);
    REQUIRE(!commit_applied);
    REQUIRE(current.context == 4096);
    REQUIRE(rollbacks == 2);
    REQUIRE(stats.attempts == 2 && stats.commit_failures == 1 && stats.rollbacks == 2);
    REQUIRE(error.find("injected commit failure") != std::string::npos);
}

void test_interface_parsers() {
    uint64_t bytes = 0;
    uint32_t tokens = 0;
    std::string error;
    REQUIRE(common_parse_byte_size("40GiB", bytes, error));
    REQUIRE(bytes == 40ULL*1024*MiB);
    REQUIRE(common_parse_byte_size("1GiB", bytes, error));
    REQUIRE(bytes == 1024*MiB);
    REQUIRE(!common_parse_byte_size("4GB", bytes, error));
    REQUIRE(common_parse_token_count("128K", tokens, error));
    REQUIRE(tokens == 128*1024);
    REQUIRE(!common_parse_token_count("128KiB", tokens, error));

    common_memory_policy policy;
    common_resource_preference preference;
    common_kv_quality quality;
    common_resource_backend backend;
    REQUIRE(common_parse_memory_policy("auto", policy));
    REQUIRE(common_parse_memory_policy("hybrid", policy));
    REQUIRE(policy == COMMON_MEMORY_POLICY_CACHE);
    REQUIRE(common_parse_resource_preference("throughput", preference));
    REQUIRE(common_parse_kv_quality("turbo4", quality));
    REQUIRE(common_parse_resource_backend("io_uring", backend));
    REQUIRE(!common_parse_kv_quality("q4_0", quality));
}

void test_compatibility_presets_and_host_budget() {
    auto input = base_input();
    common_resource_plan plan;
    std::string error;

    input.requested_policy = COMMON_MEMORY_POLICY_RESIDENT;
    if (!common_resource_plan_solve(input, plan, error)) {
        throw std::runtime_error("resident preset solve failed: " + error);
    }
    REQUIRE(plan.policy == COMMON_MEMORY_POLICY_RESIDENT);
    REQUIRE(plan.expert_ram_bytes == input.expert_source_bytes);
    REQUIRE(plan.io_staging_bytes == 0);

    input.requested_policy = COMMON_MEMORY_POLICY_STREAM;
    if (!common_resource_plan_solve(input, plan, error)) {
        throw std::runtime_error("stream preset solve failed: " + error);
    }
    REQUIRE(plan.policy == COMMON_MEMORY_POLICY_STREAM);
    REQUIRE(plan.io_staging_bytes == input.io_staging_bytes);

    input = base_input();
    input.max_ram_bytes = 1024*MiB;
    input.dense_ram_bytes = 128*MiB;
    input.expert_source_bytes = 512*MiB;
    input.requested_expert_ram_bytes = 512*MiB;
    input.requested_aux_ram_bytes = 512*MiB;
    input.requested_expert_vram_bytes_per_device = 0;
    input.mtp_bytes = 0;
    input.multimodal_bytes = 0;
    input.requested_context = 4096;
    input.max_context = 4096;
    input.devices = {{ -1, 1024*MiB, 0, 128*MiB, 128*MiB }};
    if (!common_resource_plan_solve(input, plan, error)) {
        throw std::runtime_error("host budget solve failed: " + error);
    }
    const uint64_t host_non_weight = plan.devices.front().planned_bytes - input.dense_ram_bytes;
    REQUIRE(plan.ram_planned_bytes + host_non_weight <= input.max_ram_bytes);
    REQUIRE(plan.aux_ram_bytes < input.requested_aux_ram_bytes);
}

void test_auto_unlimited_ram_and_minimum_expert_component() {
    auto input = base_input();
    common_resource_plan plan;
    std::string error;

    input.max_ram_bytes = 0;
    REQUIRE(common_resource_plan_solve(input, plan, error));
    REQUIRE(plan.policy == COMMON_MEMORY_POLICY_CACHE);

    input = base_input();
    input.requested_policy = COMMON_MEMORY_POLICY_CACHE;
    input.max_ram_bytes = input.dense_ram_bytes + 32*MiB;
    input.min_expert_ram_bytes = 64*MiB;
    REQUIRE(common_resource_plan_solve(input, plan, error));
    REQUIRE(plan.expert_ram_bytes == 0);
}

void test_rollback_hook_failure_is_single_shot() {
    common_resource_plan current;
    current.context = 4096;
    common_resource_plan target;
    target.context = 8192;
    std::string error;
    int rollbacks = 0;
    common_resource_transition_stats stats;
    REQUIRE(!common_resource_apply_plan_atomic(
        current, target,
        [](const common_resource_plan &, std::string & prepare_error) {
            prepare_error = "injected prepare failure";
            return false;
        },
        [] {},
        [&rollbacks] {
            ++rollbacks;
            throw std::runtime_error("injected rollback failure");
        },
        error, &stats));
    REQUIRE(current.context == 4096);
    REQUIRE(rollbacks == 1);
    REQUIRE(stats.prepare_failures == 1 && stats.rollbacks == 1);
    REQUIRE(error == "injected prepare failure");
}

void test_native_override_normalization() {
    REQUIRE(common_resource_cache_limit_bytes(0) == 0);
    REQUIRE(common_resource_cache_limit_bytes(-1) == 0);
    REQUIRE(common_resource_cache_limit_bytes(4096) == 4096ULL*MiB);
    REQUIRE(common_resource_should_enable_fit(true, 0));
    REQUIRE(!common_resource_should_enable_fit(true, 1));
    REQUIRE(!common_resource_should_enable_fit(true, 999));
    REQUIRE(!common_resource_should_enable_fit(false, 0));

    auto input = base_input();
    input.requested_expert_ram_bytes = common_resource_cache_limit_bytes(0);
    input.requested_expert_vram_bytes_per_device = common_resource_cache_limit_bytes(0);
    common_resource_plan plan;
    std::string error;
    REQUIRE(common_resource_plan_solve(input, plan, error));
    REQUIRE(plan.expert_ram_bytes == 0);
    for (const auto & device : plan.devices) {
        REQUIRE(device.expert_cache_bytes == 0);
    }
}

} // namespace

int main() {
    test_deterministic_budget_and_json();
    test_preference_tradeoff();
    test_no_silent_fallbacks();
    test_atomic_rollback();
    test_interface_parsers();
    test_compatibility_presets_and_host_budget();
    test_auto_unlimited_ram_and_minimum_expert_component();
    test_rollback_hook_failure_is_single_shot();
    test_native_override_normalization();
    return 0;
}
