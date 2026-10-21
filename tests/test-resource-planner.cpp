#include "resource-planner.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <stdexcept>

#include <nlohmann/json.hpp>

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
    input.qwen4exp_ple_ram_bytes = 256*MiB;
    input.requested_aux_ram_bytes = 4ULL*1024*MiB;
    input.requested_expert_vram_bytes_per_device = 512*MiB;
    input.requested_expert_prefill_staging_bytes_per_device = 256*MiB;
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

common_resource_plan runtime_plan(
        uint64_t capacity,
        uint64_t reserve,
        uint64_t fixed,
        uint64_t kv,
        uint64_t expert,
        uint32_t context = 4096) {
    common_resource_plan plan;
    plan.policy = COMMON_MEMORY_POLICY_CACHE;
    plan.context = context;
    plan.slots = 1;
    common_resource_device_plan device;
    device.id = 0;
    device.capacity_bytes = capacity;
    device.reserve_bytes = reserve;
    device.dense_bytes = fixed;
    device.kv_bytes = kv;
    device.expert_cache_bytes = expert;
    device.planned_bytes = fixed + kv + expert;
    device.headroom_bytes = capacity - reserve - device.planned_bytes;
    plan.devices.push_back(device);
    return plan;
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
    REQUIRE(first.qwen4exp_ple_ram_bytes == input.qwen4exp_ple_ram_bytes);
    REQUIRE(first.aux_ram_bytes == 4ULL*1024*MiB);
    REQUIRE(first.ram_planned_bytes == input.dense_ram_bytes + first.expert_ram_bytes +
        first.qwen4exp_ple_ram_bytes + first.aux_ram_bytes);
    REQUIRE(first.context == input.requested_context);
    REQUIRE(first.kv_quality == COMMON_KV_QUALITY_TURBO8);
    REQUIRE(first.transient_swap);
    REQUIRE(first.transient_device == input.transient_device);
    REQUIRE(first.transient_policy == COMMON_TRANSIENT_POLICY_SHARED);
    REQUIRE(first.transient_mtp_bytes == input.mtp_bytes);
    REQUIRE(first.transient_multimodal_bytes == input.multimodal_bytes);
    REQUIRE(first.devices[0].transient_bytes == 0);
    REQUIRE(first.devices[1].transient_bytes == input.multimodal_bytes);
    REQUIRE(first.devices[1].headroom_bytes >=
        std::min(input.mtp_bytes, input.multimodal_bytes));
    for (const auto & device : first.devices) {
        REQUIRE(device.expert_cache_bytes <= input.requested_expert_vram_bytes_per_device);
        REQUIRE(device.expert_prefill_staging_bytes ==
                input.requested_expert_prefill_staging_bytes_per_device);
        REQUIRE(device.planned_bytes + device.reserve_bytes <= device.capacity_bytes);
    }
    const std::string json = common_resource_plan_json(first);
    REQUIRE(json.find("\"policy\":\"cache\"") != std::string::npos);
    REQUIRE(json.find("\"kv_quality\":\"turbo8\"") != std::string::npos);
    REQUIRE(json.find("\"expert_prefill_staging_enabled\":true") != std::string::npos);
    REQUIRE(json.find("\"qwen4exp_ple_ram_bytes\":268435456") != std::string::npos);
    REQUIRE(json.find("\"transient_policy\":\"shared\"") != std::string::npos);
}

void test_optional_prefill_staging_falls_back_without_violating_context_floor() {
    auto input = base_input();
    input.devices = {{ 0, 3ULL*1024*MiB, 512*MiB, 512*MiB, 256*MiB }};
    input.transient_device = 0;
    input.mtp_bytes = 0;
    input.multimodal_bytes = 0;
    input.requested_context = 4096;
    input.max_context = 4096;
    input.min_context = 4096;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_TURBO4] = 128*1024;
    input.requested_expert_prefill_staging_bytes_per_device = 2ULL*1024*MiB;

    common_resource_plan plan;
    std::string error;
    REQUIRE(common_resource_plan_solve(input, plan, error));
    REQUIRE(!plan.expert_prefill_staging_enabled);
    REQUIRE(plan.devices.front().expert_prefill_staging_bytes == 0);
    REQUIRE(plan.reason.find("optional expert prefill staging disabled") != std::string::npos);

    input.require_expert_prefill_staging = true;
    REQUIRE(!common_resource_plan_solve(input, plan, error));
    REQUIRE(error.find("required expert prefill staging") != std::string::npos);

    auto resident = base_input();
    resident.requested_policy = COMMON_MEMORY_POLICY_RESIDENT;
    resident.require_expert_prefill_staging = true;
    REQUIRE(!common_resource_plan_solve(resident, plan, error));
    REQUIRE(error.find("non-resident MoE memory policy") != std::string::npos);
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

void test_runtime_rebalance_target_is_pure_and_reserve_bounded() {
    auto input = base_input();
    common_resource_plan current;
    std::string error;
    REQUIRE(common_resource_plan_solve(input, current, error));
    const std::string original = common_resource_plan_json(current);

    common_resource_rebalance_request request;
    request.context = current.context/2;
    request.set_expert_cache_bytes_per_device = true;
    request.expert_cache_bytes_per_device = 256*MiB;
    common_resource_plan target;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    REQUIRE(target.context == current.context/2);
    REQUIRE(common_resource_plan_json(current) == original);
    for (size_t index = 0; index < target.devices.size(); ++index) {
        REQUIRE(target.devices[index].expert_cache_bytes == 256*MiB);
        REQUIRE(target.devices[index].kv_bytes <= current.devices[index].kv_bytes);
        REQUIRE(target.devices[index].planned_bytes + target.devices[index].reserve_bytes <=
                target.devices[index].capacity_bytes);
    }

    request.context = current.slots + 1;
    REQUIRE(!common_resource_rebalance_target(current, request, target, error));
    REQUIRE(error.find("whole-token capacity") != std::string::npos);
    REQUIRE(common_resource_plan_json(current) == original);

    request.context = current.context;
    request.expert_cache_bytes_per_device = std::numeric_limits<uint64_t>::max();
    REQUIRE(!common_resource_rebalance_target(current, request, target, error));
    REQUIRE(error.find("overflows") != std::string::npos || error.find("reserve") != std::string::npos);
    REQUIRE(common_resource_plan_json(current) == original);

    request.expert_cache_bytes_per_device = 0;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    for (const auto & device : target.devices) REQUIRE(device.expert_cache_bytes == 0);
}

void test_runtime_transient_policy_targets_are_explicit_and_pure() {
    auto input = base_input();
    common_resource_plan current;
    common_resource_plan target;
    std::string error;
    REQUIRE(common_resource_plan_solve(input, current, error));
    const std::string original = common_resource_plan_json(current);

    common_resource_rebalance_request request;
    request.set_transient_policy = true;
    request.transient_policy = COMMON_TRANSIENT_POLICY_MTP_ONLY;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    REQUIRE(target.transient_policy == COMMON_TRANSIENT_POLICY_MTP_ONLY);
    REQUIRE(target.transient_capacity_bytes == input.mtp_bytes);
    REQUIRE(target.devices[1].transient_bytes == input.mtp_bytes);
    REQUIRE(!target.transient_swap);
    REQUIRE(target.draft_resident);
    REQUIRE(common_resource_plan_json(current) == original);

    request.transient_policy = COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    REQUIRE(target.transient_capacity_bytes == input.multimodal_bytes);
    REQUIRE(!target.draft_resident);

    request.transient_policy = COMMON_TRANSIENT_POLICY_OFF;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    REQUIRE(target.transient_capacity_bytes == 0);
    REQUIRE(target.devices[1].transient_bytes == 0);

    request.transient_policy = COMMON_TRANSIENT_POLICY_SHARED;
    auto missing_multimodal = current;
    missing_multimodal.transient_multimodal_bytes = 0;
    REQUIRE(!common_resource_rebalance_target(
        missing_multimodal, request, target, error));
    REQUIRE(error.find("requires configured MTP and multimodal") != std::string::npos);
    REQUIRE(common_resource_plan_json(current) == original);

    auto no_transient = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 10*MiB);
    request.transient_policy = COMMON_TRANSIENT_POLICY_OFF;
    REQUIRE(common_resource_rebalance_target(
        no_transient, request, target, error));
    REQUIRE(target.transient_policy == COMMON_TRANSIENT_POLICY_OFF);
}

void test_shared_transient_policy_preserves_request_swap_rollback_reserve() {
    auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 10*MiB);
    current.transient_device = 0;
    current.transient_policy = COMMON_TRANSIENT_POLICY_SHARED;
    current.transient_mtp_bytes = 15*MiB;
    current.transient_multimodal_bytes = 25*MiB;
    current.transient_capacity_bytes = 25*MiB;
    current.transient_swap = true;
    current.draft_resident = true;
    current.devices.front().transient_bytes = 25*MiB;
    current.devices.front().planned_bytes += 25*MiB;
    current.devices.front().headroom_bytes -= 25*MiB;
    REQUIRE(current.devices.front().headroom_bytes == 25*MiB);

    common_resource_rebalance_request request;
    request.context = current.context;
    request.set_expert_cache_bytes_per_device = true;
    request.expert_cache_bytes_per_device = 25*MiB;
    common_resource_plan target;
    std::string error;
    REQUIRE(!common_resource_rebalance_target(current, request, target, error));
    REQUIRE(error.find("rollback reserve") != std::string::npos);
}

void test_preparation_peak_rejects_final_fit_without_double_buffer_headroom() {
    const auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 40*MiB, 10*MiB);
    const auto target = runtime_plan(100*MiB, 10*MiB, 10*MiB, 60*MiB, 10*MiB, 8192);
    REQUIRE(target.devices.front().planned_bytes + target.devices.front().reserve_bytes <=
            target.devices.front().capacity_bytes);

    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(!common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(error.find("preparation peak crosses") != std::string::npos);
}

void test_preparation_peak_detects_u64_overflow_without_mutating_report() {
    const uint64_t maximum = std::numeric_limits<uint64_t>::max();
    const auto current = runtime_plan(maximum, 0, 0, maximum - 5, 0);
    const auto target = runtime_plan(maximum, 0, 0, maximum - 6, 0, 8192);
    common_resource_preparation_peak report;
    report.prepares_expert_cache = true;
    report.devices.push_back({});
    std::string error;
    REQUIRE(!common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(error.find("preparation peak overflows") != std::string::npos);
    REQUIRE(report.prepares_expert_cache);
    REQUIRE(report.devices.size() == 1);
}

void test_preparation_peak_accepts_exact_reserve_boundary() {
    const auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 10*MiB);
    const auto target = runtime_plan(100*MiB, 10*MiB, 10*MiB, 50*MiB, 10*MiB, 8192);
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(report.prepares_kv);
    REQUIRE(!report.prepares_expert_cache);
    REQUIRE(report.devices.front().current_live_bytes == 40*MiB);
    REQUIRE(report.devices.front().prepared_kv_bytes == 50*MiB);
    REQUIRE(report.devices.front().prepared_expert_cache_bytes == 0);
    REQUIRE(report.devices.front().peak_bytes == 90*MiB);
    REQUIRE(report.devices.front().peak_headroom_bytes == 0);
}

void test_preparation_peak_reports_single_pool_replacement() {
    const auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 10*MiB);
    const auto target = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 30*MiB);
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(!report.prepares_kv);
    REQUIRE(report.prepares_expert_cache);
    REQUIRE(report.devices.front().prepared_kv_bytes == 0);
    REQUIRE(report.devices.front().prepared_expert_cache_bytes == 30*MiB);
    REQUIRE(report.devices.front().peak_bytes == 70*MiB);
    REQUIRE(report.devices.front().peak_headroom_bytes == 20*MiB);

    const std::string json = common_resource_preparation_peak_json(report);
    REQUIRE(json.find("\"prepares_kv\":false") != std::string::npos);
    REQUIRE(json.find("\"prepares_expert_cache\":true") != std::string::npos);
    REQUIRE(json.find("\"peak_headroom_bytes\":20971520") != std::string::npos);
}

void test_preparation_peak_counts_transient_owner_replacement() {
    auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 10*MiB);
    current.transient_device = 0;
    current.transient_policy = COMMON_TRANSIENT_POLICY_MTP_ONLY;
    current.transient_mtp_bytes = 20*MiB;
    current.transient_multimodal_bytes = 30*MiB;
    current.transient_capacity_bytes = 20*MiB;
    current.devices.front().transient_bytes = 20*MiB;
    current.devices.front().planned_bytes += 20*MiB;
    current.devices.front().headroom_bytes -= 20*MiB;

    auto target = current;
    target.transient_policy = COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
    target.transient_capacity_bytes = 30*MiB;
    target.draft_resident = false;
    target.devices.front().transient_bytes = 30*MiB;
    target.devices.front().planned_bytes += 10*MiB;
    target.devices.front().headroom_bytes -= 10*MiB;

    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_preparation_peak(
        current, target, report, error));
    REQUIRE(report.prepares_transient);
    REQUIRE(report.devices.front().current_live_bytes == 60*MiB);
    REQUIRE(report.devices.front().target_live_bytes == 70*MiB);
    REQUIRE(report.devices.front().prepared_transient_bytes == 30*MiB);
    REQUIRE(report.devices.front().peak_bytes == 90*MiB);
    REQUIRE(report.devices.front().peak_headroom_bytes == 0);

    const std::string json = common_resource_preparation_peak_json(report);
    REQUIRE(json.find("\"prepares_transient\":true") != std::string::npos);
    REQUIRE(json.find("\"prepared_transient_bytes\":31457280") != std::string::npos);
}

void test_preparation_peak_couples_shared_transient_owner_bounds() {
    auto shared = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 10*MiB);
    shared.transient_device = 0;
    shared.transient_policy = COMMON_TRANSIENT_POLICY_SHARED;
    shared.transient_mtp_bytes = 20*MiB;
    shared.transient_multimodal_bytes = 30*MiB;
    shared.transient_capacity_bytes = 30*MiB;
    shared.transient_swap = true;
    shared.draft_resident = true;
    shared.devices.front().transient_bytes = 30*MiB;
    shared.devices.front().planned_bytes += 30*MiB;
    shared.devices.front().headroom_bytes -= 30*MiB;

    for (const auto policy : {
            COMMON_TRANSIENT_POLICY_MTP_ONLY,
            COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY}) {
        auto target = shared;
        target.transient_policy = policy;
        target.transient_swap = false;
        target.draft_resident = policy == COMMON_TRANSIENT_POLICY_MTP_ONLY;
        target.transient_capacity_bytes = policy == COMMON_TRANSIENT_POLICY_MTP_ONLY
            ? 20*MiB : 30*MiB;
        target.devices.front().planned_bytes -= target.devices.front().transient_bytes;
        target.devices.front().transient_bytes = target.transient_capacity_bytes;
        target.devices.front().planned_bytes += target.devices.front().transient_bytes;
        target.devices.front().headroom_bytes =
            target.devices.front().capacity_bytes - target.devices.front().reserve_bytes -
            target.devices.front().planned_bytes;

        common_resource_preparation_peak report;
        std::string error;
        REQUIRE(common_resource_rebalance_preparation_peak(
            shared, target, report, error));
        REQUIRE(report.prepares_transient);
        // Worst case is one live owner plus the other prepared owner: A+B.
        // The current shared capacity already contributes max(A,B), so only
        // min(A,B) is additional at the exact reserve boundary.
        REQUIRE(report.devices.front().prepared_transient_bytes == 20*MiB);
        REQUIRE(report.devices.front().peak_bytes == 90*MiB);
        REQUIRE(report.devices.front().peak_headroom_bytes == 0);
    }

    auto multimodal_only = shared;
    multimodal_only.transient_policy = COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
    multimodal_only.transient_swap = false;
    multimodal_only.draft_resident = false;
    common_resource_preparation_peak reverse;
    std::string error;
    REQUIRE(common_resource_rebalance_preparation_peak(
        multimodal_only, shared, reverse, error));
    REQUIRE(reverse.prepares_transient);
    // Enabling shared mode preserves the already-resident multimodal owner.
    REQUIRE(reverse.devices.front().prepared_transient_bytes == 0);
    REQUIRE(reverse.devices.front().peak_bytes == 70*MiB);
}

void test_preparation_peak_counts_same_target_expert_reconciliation() {
    const auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 25*MiB);
    const auto target = current;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_preparation_peak(
            current, target, report, error, true));
    REQUIRE(report.prepares_expert_cache);
    REQUIRE(report.devices.front().current_live_bytes == 55*MiB);
    REQUIRE(report.devices.front().prepared_expert_cache_bytes == 25*MiB);
    REQUIRE(report.devices.front().peak_bytes == 80*MiB);
    REQUIRE(report.devices.front().peak_headroom_bytes == 10*MiB);
}

void test_preparation_peak_rejects_same_target_expert_reconciliation_without_headroom() {
    const auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 20*MiB, 35*MiB);
    const auto target = current;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(!common_resource_rebalance_preparation_peak(
            current, target, report, error, true));
    REQUIRE(error.find("preparation peak crosses") != std::string::npos);
}

void test_preparation_peak_rejects_combined_cross_pool_trade() {
    const auto current = runtime_plan(100*MiB, 10*MiB, 10*MiB, 50*MiB, 10*MiB);
    const auto target = runtime_plan(100*MiB, 10*MiB, 10*MiB, 10*MiB, 50*MiB, 8192);
    REQUIRE(current.devices.front().planned_bytes == target.devices.front().planned_bytes);

    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(!common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(error.find("preparation peak crosses") != std::string::npos);
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
    common_transient_policy transient_policy;
    REQUIRE(common_parse_memory_policy("auto", policy));
    REQUIRE(common_parse_memory_policy("hybrid", policy));
    REQUIRE(policy == COMMON_MEMORY_POLICY_CACHE);
    REQUIRE(common_parse_resource_preference("throughput", preference));
    REQUIRE(common_parse_kv_quality("turbo4", quality));
    REQUIRE(common_parse_resource_backend("io_uring", backend));
    REQUIRE(common_parse_transient_policy("shared", transient_policy));
    REQUIRE(transient_policy == COMMON_TRANSIENT_POLICY_SHARED);
    REQUIRE(common_transient_policy_name(COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY) ==
            "multimodal-only");
    REQUIRE(!common_parse_transient_policy("auto", transient_policy));
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
    input.max_ram_bytes = input.dense_ram_bytes + input.qwen4exp_ple_ram_bytes + 32*MiB;
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

void test_calibrated_expert_split_solver() {
    common_expert_split_input input;
    input.calibration_complete = true;
    input.cpu_confidence = input.upload_confidence = 0.95;
    std::string error;
    common_expert_split_plan plan;

    input.misses = 1;
    input.cpu_ns_per_expert = 100;
    input.upload_ns_per_expert = 20;
    REQUIRE(common_expert_split_solve(input, plan, error));
    REQUIRE(plan.cpu_experts == 0 && plan.upload_experts == 1);

    input.cpu_ns_per_expert = 20;
    input.upload_ns_per_expert = 100;
    REQUIRE(common_expert_split_solve(input, plan, error));
    REQUIRE(plan.cpu_experts == 1 && plan.upload_experts == 0);

    input.misses = 8;
    input.cpu_ns_per_expert = input.upload_ns_per_expert = 100;
    REQUIRE(common_expert_split_solve(input, plan, error));
    REQUIRE(plan.cpu_experts == 4 && plan.upload_experts == 4);

    input.previous_upload_experts = 3;
    input.hysteresis_fraction = 0.30;
    REQUIRE(common_expert_split_solve(input, plan, error));
    REQUIRE(plan.upload_experts == 3 && plan.retained_by_hysteresis);

    input.calibration_complete = false;
    REQUIRE(!common_expert_split_solve(input, plan, error));
    REQUIRE(error.find("incomplete") != std::string::npos);
    input.calibration_complete = true;
    input.cpu_confidence = 0.5;
    REQUIRE(!common_expert_split_solve(input, plan, error));
    REQUIRE(error.find("confidence") != std::string::npos);
    input.cpu_confidence = 0.95;
    input.cpu_ns_per_expert = std::numeric_limits<double>::quiet_NaN();
    REQUIRE(!common_expert_split_solve(input, plan, error));
}

nlohmann::json calibrated_profile_json() {
    const nlohmann::json format = {
        {"ggml_type_id", 16},
        {"input_width", 4096},
        {"expert_width", 2048},
        {"bytes_per_expert_component", 2162688},
    };
    nlohmann::json cpu = format;
    cpu.update({
        {"correctness", "single-thread-and-dequantized-scalar-reference"},
        {"independent_reference_nrmse", 0.004},
        {"sample_count", 21},
        {"relative_standard_error", 0.02},
        {"confidence", 0.92},
    });
    nlohmann::json contention = format;
    contention.update({
        {"cpu_ns_per_expert_component", 120000.0},
        {"upload_ns_per_expert_component", 90000.0},
        {"cpu_sample_count", 21},
        {"cpu_relative_standard_error", 0.02},
        {"cpu_confidence", 0.92},
        {"upload_sample_count", 21},
        {"upload_relative_standard_error", 0.02},
        {"upload_confidence", 0.93},
    });
    nlohmann::json lease = format;
    lease.update({
        {"backend", "CUDA0"},
        {"storage_backend", "pread"},
        {"distribution", "warm-steady-state"},
        {"sample_count", 21},
        {"relative_standard_error", 0.02},
        {"confidence", 0.94},
    });
    return {
        {"benchmark_source", {
            {"planner_ready", true},
            {"calibration_level", "planner"},
        }},
        {"measurements", {
            {"cpu_moe", {
                {"status", "measured"},
                {"model_profiles", nlohmann::json::array({cpu})},
            }},
            {"cpu_cache_contention", {
                {"status", "measured"},
                {"upload_path", "pread_to_bounded_ram_lease_to_async_upload"},
                {"distribution", "warm-steady-state"},
                {"devices", nlohmann::json::array({{
                    {"backend", "CUDA0"},
                    {"profiles", nlohmann::json::array({contention})},
                }})},
            }},
            {"expert_cache_upload", {
                {"status", "measured"},
                {"lease_upload_profiles", nlohmann::json::array({lease})},
            }},
        }},
    };
}

void test_native_calibration_profile_gate() {
    common_expert_calibration_profile profile;
    std::string error;
    auto valid = calibrated_profile_json();
    REQUIRE(common_expert_calibration_parse_json(valid.dump(), profile, error));
    REQUIRE(profile.entries.size() == 1);
    common_expert_calibration_entry entry;
    common_expert_calibration_key key{"CUDA0", 16, 4096, 2048, 2162688};
    REQUIRE(common_expert_calibration_lookup(profile, key, entry));
    REQUIRE(entry.cpu_ns_per_expert_component == 120000.0);
    REQUIRE(entry.upload_ns_per_expert_component == 90000.0);
    common_expert_split_plan split;
    REQUIRE(common_expert_split_solve_calibrated(
            profile, {key, key}, 7, -1, 0.05, split, error));
    REQUIRE(split.cpu_experts == 3 && split.upload_experts == 4);
    common_expert_calibration_key missing = key;
    missing.ggml_type = 19;
    REQUIRE(!common_expert_split_solve_calibrated(
            profile, {key, missing}, 7, -1, 0.05, split, error));
    REQUIRE(error.find("required expert component") != std::string::npos);
    key.backend = "CUDA1";
    REQUIRE(!common_expert_calibration_lookup(profile, key, entry));

    auto low_confidence = calibrated_profile_json();
    low_confidence["measurements"]["cpu_cache_contention"]["devices"][0]["profiles"][0]["cpu_confidence"] = 0.79;
    REQUIRE(!common_expert_calibration_parse_json(low_confidence.dump(), profile, error));
    REQUIRE(error.find("confidence") != std::string::npos);

    auto missing_lease = calibrated_profile_json();
    missing_lease["measurements"]["expert_cache_upload"]["lease_upload_profiles"] = nlohmann::json::array();
    REQUIRE(!common_expert_calibration_parse_json(missing_lease.dump(), profile, error));
    REQUIRE(error.find("leased-upload") != std::string::npos);

    auto forged_ready = calibrated_profile_json();
    forged_ready["benchmark_source"]["planner_ready"] = false;
    REQUIRE(!common_expert_calibration_parse_json(forged_ready.dump(), profile, error));
    REQUIRE(error.find("planner-ready") != std::string::npos);
}

common_resource_plan_input forecast_input() {
    auto input = base_input();
    input.requested_policy = COMMON_MEMORY_POLICY_CACHE;
    input.max_ram_bytes = 1024*MiB;
    input.dense_ram_bytes = 64*MiB;
    input.qwen4exp_ple_ram_bytes = 0;
    input.requested_expert_ram_bytes = input.requested_aux_ram_bytes = 0;
    input.requested_expert_vram_bytes_per_device = 0;
    input.requested_expert_prefill_staging_bytes_per_device = 0;
    input.mtp_bytes = input.multimodal_bytes = 0;
    input.slots = 1;
    input.requested_context = input.max_context = 65536;
    input.min_context = 4096;
    input.context_alignment = 256;
    input.min_kv_quality = COMMON_KV_QUALITY_F16;
    for (auto & bytes : input.kv_bytes_per_token) bytes = 0;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_F16] = (64*MiB+4096*1024)/4096;
    input.kv_bytes_for_context = [](common_kv_quality quality, uint32_t context) {
        return quality == COMMON_KV_QUALITY_F16 ? 64*MiB+uint64_t(context)*1024 : 0;
    };
    input.devices = {{0,256*MiB,16*MiB,0,0}};
    input.transient_device = 0;
    return input;
}

void test_native_forecast_long_context() {
    auto input = forecast_input();
    common_resource_plan plan;
    std::string error;
    REQUIRE(common_resource_plan_solve(input,plan,error));
    REQUIRE(plan.context == 65536);
    REQUIRE(plan.devices.front().kv_bytes == 128*MiB);
    REQUIRE(plan.kv_native_sizing && bool(plan.kv_bytes_for_context));
    REQUIRE(plan.kv_bytes_for_context(512) == 64*MiB+512*1024);
}

void test_native_forecast_short_context() {
    auto input = forecast_input();
    input.requested_context = input.max_context = input.min_context = 512;
    input.devices.front().free_bytes = 32*MiB;
    input.devices.front().reserve_bytes = 0;
    common_resource_plan plan;
    std::string error;
    REQUIRE(!common_resource_plan_solve(input,plan,error));
}

void test_native_forecast_rebalance() {
    auto current = runtime_plan(1024*MiB,16*MiB,32*MiB,128*MiB,0,65536);
    current.kv_native_sizing = true;
    current.kv_bytes_for_context = [](uint32_t context) { return 64*MiB+uint64_t(context)*1024; };
    common_resource_rebalance_request request;
    request.context = 4096;
    common_resource_plan target;
    std::string error;
    REQUIRE(common_resource_rebalance_target(current,request,target,error));
    REQUIRE(target.devices.front().kv_bytes == 68*MiB);
    REQUIRE(current.devices.front().kv_bytes == 128*MiB);
    REQUIRE(target.kv_native_sizing && target.kv_bytes_for_context(65536) == 128*MiB);
}

void test_native_forecast_step_storage() {
    auto current = runtime_plan(1024*MiB,16*MiB,32*MiB,80*MiB,0,65536);
    current.kv_native_sizing = true;
    current.kv_bytes_for_context = [](uint32_t context) { return (16+uint64_t((context+1023)/1024))*MiB; };
    common_resource_rebalance_request request;
    request.context = 65024;
    common_resource_plan target;
    std::string error;
    REQUIRE(common_resource_rebalance_target(current,request,target,error));
    REQUIRE(target.devices.front().kv_bytes == 80*MiB);
}

void test_native_forecast_failure_and_rounding() {
    auto current=runtime_plan(1024*MiB,16*MiB,32*MiB,128*MiB,0,65536);
    current.kv_native_sizing=true;
    common_resource_rebalance_request request;request.context=4096;
    common_resource_plan target;
    std::string error;
    REQUIRE(!common_resource_rebalance_target(current,request,target,error));
    for (uint64_t unavailable:{uint64_t(0),uint64_t(UINT64_MAX)}) {
        current.kv_bytes_for_context=[unavailable](uint32_t){return unavailable;};
        REQUIRE(!common_resource_rebalance_target(current,request,target,error));
    }
    const uint64_t half=1ULL<<62;
    current.devices={runtime_plan(UINT64_MAX,0,0,half,0).devices[0],runtime_plan(UINT64_MAX,0,0,half,0).devices[0]};
    current.devices[1].id=1;
    current.kv_bytes_for_context=[half](uint32_t){return 2*(half-1);};
    REQUIRE(common_resource_rebalance_target(current,request,target,error));
    REQUIRE(target.devices[0].kv_bytes==half-1 && target.devices[1].kv_bytes==half-1);
    current.devices={runtime_plan(1024*MiB,0,0,43*MiB+1,0).devices[0],
                     runtime_plan(1024*MiB,0,0,20*MiB+2,0).devices[0],
                     runtime_plan(1024*MiB,0,0,37*MiB-3,0).devices[0]};
    current.devices[1].id=1;current.devices[2].id=2;
    current.kv_bytes_for_context=[](uint32_t){return 80*MiB;};
    REQUIRE(common_resource_rebalance_target(current,request,target,error));
    uint64_t total=0;for (const auto & device:target.devices)total+=device.kv_bytes;
    REQUIRE(total>=80*MiB && total<=80*MiB+2);
    REQUIRE(current.devices[0].kv_bytes==43*MiB+1);
    auto input=forecast_input();input.slots=8192;input.min_context=4096;
    REQUIRE(common_resource_plan_solve(input,target,error));
    REQUIRE(target.context>=input.slots);
    input.min_context=UINT32_MAX;input.context_alignment=256;
    REQUIRE(!common_resource_plan_solve(input,target,error));
}

void test_native_forecast_host_budget() {
    auto current=runtime_plan(1024*MiB,0,128*MiB,256*MiB,0,4096);
    auto & host=current.devices.front();host.id=-1;
    host.graph_bytes=64*MiB;host.planned_bytes+=64*MiB;host.headroom_bytes-=64*MiB;
    current.ram_capacity_bytes=1024*MiB;
    current.ram_planned_bytes=640*MiB;
    current.expert_ram_bytes=512*MiB;
    current.kv_native_sizing=true;
    current.kv_bytes_for_context=[](uint32_t context){return 128*MiB+uint64_t(context)*32768;};
    common_resource_rebalance_request request;request.context=8192;
    common_resource_plan target;
    std::string error;
    REQUIRE(!common_resource_rebalance_target(current,request,target,error));
    REQUIRE(current.devices.front().kv_bytes==256*MiB && current.expert_ram_bytes==512*MiB);
}

void test_host_cache_preparation_peak_rejects_combined_budget() {
    auto current=runtime_plan(1024*MiB,0,128*MiB,256*MiB,0,4096);
    auto & host=current.devices.front();host.id=-1;
    host.graph_bytes=64*MiB;host.planned_bytes+=64*MiB;host.headroom_bytes-=64*MiB;
    current.ram_capacity_bytes=1024*MiB;
    current.ram_planned_bytes=640*MiB;
    current.expert_ram_bytes=512*MiB;
    current.kv_native_sizing=true;
    current.kv_bytes_for_context=[](uint32_t context){return 128*MiB+uint64_t(context)*32768;};
    common_resource_rebalance_request request;request.context=2048;
    common_resource_plan target;
    std::string error;
    REQUIRE(common_resource_rebalance_target(current,request,target,error));
    REQUIRE(target.devices.front().kv_bytes==192*MiB);
    // Final usage is 896 MiB. Preparing the complete new KV cache while the
    // old one is rollback-capable needs 1152 MiB including the RAM cache.
    common_resource_preparation_peak report;report.prepares_transient=true;
    report.devices.push_back({});report.devices.front().peak_bytes=123;
    const auto before=common_resource_plan_json(current);
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(error.find("RAM preparation peak")!=std::string::npos);
    REQUIRE(report.prepares_transient && report.devices.size()==1 && report.devices.front().peak_bytes==123);
    REQUIRE(common_resource_plan_json(current)==before);
}

void test_host_transient_preparation_peak_rejects_combined_budget() {
    auto current=runtime_plan(128*MiB,0,16*MiB,16*MiB,0,4096);
    auto & host=current.devices.front();host.id=-1;
    host.graph_bytes=8*MiB;host.transient_bytes=24*MiB;
    host.planned_bytes+=32*MiB;host.headroom_bytes-=32*MiB;
    current.ram_capacity_bytes=128*MiB;current.ram_planned_bytes=64*MiB;
    current.expert_ram_bytes=48*MiB;
    current.transient_device=-1;current.transient_policy=COMMON_TRANSIENT_POLICY_MTP_ONLY;
    current.transient_mtp_bytes=24*MiB;current.transient_multimodal_bytes=32*MiB;
    current.transient_capacity_bytes=24*MiB;
    auto target=current;target.transient_policy=COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
    target.transient_capacity_bytes=32*MiB;target.devices.front().transient_bytes=32*MiB;
    target.devices.front().planned_bytes+=8*MiB;target.devices.front().headroom_bytes-=8*MiB;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(error.find("RAM preparation peak")!=std::string::npos);
    current.ram_capacity_bytes=target.ram_capacity_bytes=160*MiB;
    current.devices.front().capacity_bytes=target.devices.front().capacity_bytes=160*MiB;
    current.devices.front().headroom_bytes+=32*MiB;target.devices.front().headroom_bytes+=32*MiB;
    REQUIRE(common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(report.devices.front().prepared_transient_bytes==32*MiB);
    REQUIRE(report.ram_current_bytes==112*MiB && report.ram_target_bytes==120*MiB);
    REQUIRE(report.ram_peak_bytes==144*MiB && report.ram_peak_headroom_bytes==16*MiB);
}

void test_host_cache_preparation_peak_boundaries() {
    for(uint64_t cache:{uint64_t(0),384*MiB,512*MiB})for(uint64_t reserve:{uint64_t(0),16*MiB})for(int margin:{-1,0,1}) {
        const uint64_t expected_peak=640*MiB+cache;
        const uint64_t usable=margin<0?expected_peak-1:expected_peak+margin;
        auto current=runtime_plan(usable+reserve,reserve,128*MiB,256*MiB,0,4096);
        auto & host=current.devices.front();host.id=-1;
        host.graph_bytes=64*MiB;host.planned_bytes+=64*MiB;host.headroom_bytes-=64*MiB;
        current.ram_capacity_bytes=usable+reserve;current.ram_planned_bytes=128*MiB+cache;
        current.expert_ram_bytes=cache;current.kv_native_sizing=true;
        current.kv_bytes_for_context=[](uint32_t context){return 128*MiB+uint64_t(context)*32768;};
        common_resource_rebalance_request request;request.context=2048;
        common_resource_plan target;
        std::string error;
        REQUIRE(common_resource_rebalance_target(current,request,target,error));
        common_resource_preparation_peak report;report.ram_peak_bytes=123;
        if(margin<0) {
            REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
            REQUIRE(report.ram_peak_bytes==123);
        }else {
            REQUIRE(common_resource_rebalance_preparation_peak(current,target,report,error));
            REQUIRE(report.ram_capacity_bytes==usable+reserve && report.ram_usable_bytes==usable);
            REQUIRE(report.ram_current_bytes==448*MiB+cache && report.ram_target_bytes==384*MiB+cache);
            REQUIRE(report.ram_peak_bytes==expected_peak && report.ram_peak_headroom_bytes==uint64_t(margin));
            REQUIRE(report.devices.front().prepared_kv_bytes==192*MiB);
            const auto json=common_resource_preparation_peak_json(report);
            REQUIRE(json.find("\"ram_peak_bytes\":"+std::to_string(expected_peak))!=std::string::npos);
        }
    }
}

void test_host_cache_preparation_peak_overflow_and_immutable_budget() {
    auto current=runtime_plan(UINT64_MAX,0,0,32,0,4096);current.devices.front().id=-1;
    current.ram_capacity_bytes=UINT64_MAX;current.ram_planned_bytes=UINT64_MAX-50;
    auto target=current;target.context=8192;target.devices.front().kv_bytes=64;
    target.devices.front().planned_bytes=64;target.devices.front().headroom_bytes=UINT64_MAX-64;
    common_resource_preparation_peak report;report.ram_peak_bytes=123;
    std::string error;
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(error.find("RAM preparation peak overflows")!=std::string::npos && report.ram_peak_bytes==123);
    target=current;target.ram_capacity_bytes--;
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(error.find("immutable RAM")!=std::string::npos);
    target=current;target.ram_planned_bytes--;
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(error.find("immutable RAM")!=std::string::npos);
}

void test_tracked_device_memory_preparation() {
    // The old zero fixed forecast admits a 96 MiB double-buffer peak; actual
    // weights plus cached workspace raise it to 144 MiB and cross the reserve.
    auto current = runtime_plan(128*MiB, 8*MiB, 0, 64*MiB, 0, 4096);
    current.devices.front().id = 2;
    common_resource_rebalance_request request;
    request.context = 2048;
    common_resource_plan target;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(report.devices.front().peak_bytes == 96*MiB);
    const auto original_current = common_resource_plan_json(current);
    const auto original_target = common_resource_plan_json(target);
    for (const std::vector<common_resource_device_memory> & bad : {
            std::vector<common_resource_device_memory>{},
            {{0, 16*MiB, 32*MiB}}, {{2, 16*MiB, 32*MiB}, {2, 0, 0}},
            {{2, 16*MiB, 32*MiB}, {0, 0, 0}}, {{-1, 0, 0}},
            {{2, UINT64_MAX, 1}}}) {
        REQUIRE(!common_resource_reconcile_device_memory(current, target, bad, error));
        REQUIRE(common_resource_plan_json(current) == original_current);
        REQUIRE(common_resource_plan_json(target) == original_target);
    }
    REQUIRE(common_resource_reconcile_device_memory(current, target, {{2, 16*MiB, 32*MiB}}, error));
    REQUIRE(current.devices.front().dense_bytes == 16*MiB && current.devices.front().graph_bytes == 32*MiB);
    REQUIRE(target.devices.front().dense_bytes == 16*MiB && target.devices.front().graph_bytes == 32*MiB);
    REQUIRE(current.devices.front().planned_bytes == 112*MiB && target.devices.front().planned_bytes == 80*MiB);
    REQUIRE(!common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(report.devices.front().peak_bytes == 96*MiB); // failure leaves report intact
    current.devices.front().capacity_bytes = target.devices.front().capacity_bytes = 160*MiB;
    current.devices.front().headroom_bytes = 40*MiB;
    target.devices.front().headroom_bytes = 72*MiB;
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(report.devices.front().peak_bytes == 144*MiB);
    REQUIRE(report.devices.front().peak_headroom_bytes == 8*MiB);
    // A second inventory read must not add the same backing twice or lower a
    // larger forecast when observed allocations shrink.
    const auto reconciled = common_resource_plan_json(current);
    REQUIRE(common_resource_reconcile_device_memory(current, target, {{2, 8*MiB, 16*MiB}}, error));
    REQUIRE(common_resource_plan_json(current) == reconciled);
    auto stale = current;
    ++stale.devices.front().planned_bytes;
    const auto stale_before = common_resource_plan_json(stale);
    REQUIRE(!common_resource_reconcile_device_memory(stale, target, {{2, 0, 0}}, error));
    REQUIRE(common_resource_plan_json(stale) == stale_before);
    auto host = current;
    host.devices.front().id = -1;
    auto host_target = host;
    const auto host_before = common_resource_plan_json(host);
    REQUIRE(common_resource_reconcile_device_memory(host, host_target, {}, error));
    REQUIRE(common_resource_plan_json(host) == host_before);
    // Input order may differ from scheduler/plan order on noncontiguous GPUs.
    current.devices.push_back(current.devices.front());
    target.devices.push_back(target.devices.front());
    current.devices.back().id = target.devices.back().id = 0;
    REQUIRE(common_resource_reconcile_device_memory(current, target, {{0, 24*MiB, 0}, {2, 0, 0}}, error));
    REQUIRE(current.devices.front().dense_bytes == 16*MiB && current.devices.back().dense_bytes == 24*MiB);
}

void test_tracked_host_memory_preparation() {
    auto current = runtime_plan(256*MiB, 8*MiB, 16*MiB, 64*MiB, 0, 4096);
    auto & host = current.devices.front();
    host.id = -1;
    host.graph_bytes = 16*MiB;
    host.planned_bytes += 16*MiB;
    host.headroom_bytes -= 16*MiB;
    current.ram_capacity_bytes = 256*MiB;
    current.ram_planned_bytes = 96*MiB;
    current.expert_ram_bytes = 40*MiB;
    current.qwen4exp_ple_ram_bytes = 32*MiB;
    current.aux_ram_bytes = 8*MiB;
    common_resource_rebalance_request request;
    request.context = 2048;
    common_resource_plan target;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(report.ram_peak_bytes == 208*MiB && report.ram_tracked_extra_bytes == 0);
    const auto before = common_resource_preparation_peak_json(report);
    const common_resource_host_memory observed{24*MiB, 40*MiB, 48*MiB};
    REQUIRE(!common_resource_rebalance_preparation_peak(current, target, report, error, false, false, &observed));
    REQUIRE(error.find("RAM preparation peak") != std::string::npos);
    REQUIRE(common_resource_preparation_peak_json(report) == before);
    current.ram_capacity_bytes = target.ram_capacity_bytes = 272*MiB;
    current.devices.front().capacity_bytes = target.devices.front().capacity_bytes = 272*MiB;
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error, false, false, &observed));
    REQUIRE(report.ram_current_bytes == 224*MiB && report.ram_target_bytes == 192*MiB);
    REQUIRE(report.ram_peak_bytes == 256*MiB && report.ram_tracked_extra_bytes == 48*MiB);
    REQUIRE(report.ram_peak_headroom_bytes == 8*MiB);
    REQUIRE(report.ram_host_weights_limit_bytes == 24*MiB && report.ram_host_fixed_limit_bytes == 40*MiB &&
            report.ram_host_ple_cache_limit_bytes == 48*MiB);
    const auto once = common_resource_preparation_peak_json(report);
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error, false, false, &observed));
    REQUIRE(common_resource_preparation_peak_json(report) == once);
    // Existing reservations cover smaller actual backing without being added
    // again. A mapped span never enters the allocated-weight input.
    const common_resource_host_memory covered{8*MiB, 8*MiB, 16*MiB};
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error, false, false, &covered));
    REQUIRE(report.ram_tracked_extra_bytes == 0 && report.ram_peak_bytes == 208*MiB);
    REQUIRE(report.ram_host_weights_limit_bytes == 16*MiB && report.ram_host_fixed_limit_bytes == 16*MiB &&
            report.ram_host_ple_cache_limit_bytes == 32*MiB);
    const auto covered_report = common_resource_preparation_peak_json(report);
    const common_resource_host_memory overflowing{UINT64_MAX, 1, 0};
    REQUIRE(!common_resource_rebalance_preparation_peak(current, target, report, error, false, false, &overflowing));
    REQUIRE(common_resource_preparation_peak_json(report) == covered_report);
    auto wrong = target;
    ++wrong.expert_ram_bytes;
    REQUIRE(!common_resource_rebalance_preparation_peak(current, wrong, report, error, false, false, &covered));
    REQUIRE(common_resource_preparation_peak_json(report) == covered_report);
    wrong = current;
    wrong.io_staging_bytes = 32*MiB;
    REQUIRE(!common_resource_rebalance_preparation_peak(wrong, wrong, report, error, false, false, &covered));
    REQUIRE(common_resource_preparation_peak_json(report) == covered_report);
    // GPU-only plans still own host graph/output/metadata. Their RAM budget
    // must include these bytes even without a serialized host device row.
    current.devices.front().id = target.devices.front().id = 2;
    for (int margin : {-1, 0, 1}) {
        current.ram_capacity_bytes = target.ram_capacity_bytes = 160*MiB + margin;
        const bool accepted = common_resource_rebalance_preparation_peak(
            current, target, report, error, false, false, &observed);
        REQUIRE(accepted == (margin >= 0));
        if (accepted) {
            REQUIRE(report.ram_current_bytes == 160*MiB && report.ram_peak_bytes == 160*MiB);
            REQUIRE(report.ram_tracked_extra_bytes == 64*MiB && report.ram_peak_headroom_bytes == uint64_t(margin));
        }
    }
    current.ram_capacity_bytes = target.ram_capacity_bytes = 0;
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error, false, false, &observed));
    REQUIRE(report.ram_capacity_bytes == 0 && report.ram_usable_bytes == UINT64_MAX && report.ram_peak_bytes == 160*MiB);
    REQUIRE(common_resource_rebalance_preparation_peak(current, target, report, error));
    REQUIRE(report.ram_capacity_bytes == 0 && report.ram_peak_bytes == 0); // original overload retained
}

void test_native_kv_preparation_receipt() {
    auto current = runtime_plan(128*MiB, 0, 8*MiB, 20*MiB, 0, 4096);
    current.devices.front().id = 2;
    current.devices.front().graph_bytes = 4*MiB;
    current.devices.front().planned_bytes += 4*MiB;
    current.devices.front().headroom_bytes -= 4*MiB;
    current.ram_capacity_bytes = 48*MiB;
    current.ram_planned_bytes = 16*MiB;
    common_resource_rebalance_request request;
    request.context = 2048;
    common_resource_plan target;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_target(current, request, target, error));
    const common_resource_host_memory host{0, 8*MiB, 0};
    common_resource_kv_preparation kv;
    kv.current = {{2,20*MiB},{-1,4*MiB}};
    kv.target = {{2,16*MiB},{-1,8*MiB}};
    kv.host_metadata_bytes = 2*MiB;
    kv.host_staging_bytes = MiB;
    REQUIRE(common_resource_rebalance_preparation_peak(current,target,report,error,false,false,&host,&kv));
    REQUIRE(report.devices.front().current_live_bytes==32*MiB && report.devices.front().prepared_kv_bytes==16*MiB);
    REQUIRE(report.devices.front().peak_bytes==48*MiB);
    REQUIRE(report.ram_current_bytes==28*MiB && report.ram_target_bytes==32*MiB && report.ram_peak_bytes==39*MiB);
    REQUIRE(report.prepared_host_metadata_bytes==2*MiB && report.prepared_host_staging_bytes==MiB);
    for(int margin:{-1,0,1}) {
        current.ram_capacity_bytes=target.ram_capacity_bytes=39*MiB+margin;
        const auto before=common_resource_preparation_peak_json(report);
        const bool accepted=common_resource_rebalance_preparation_peak(current,target,report,error,false,false,&host,&kv);
        REQUIRE(accepted==(margin>=0));
        if(accepted) REQUIRE(report.ram_peak_headroom_bytes==uint64_t(margin));
        else REQUIRE(common_resource_preparation_peak_json(report)==before);
    }
    const auto good=common_resource_preparation_peak_json(report);
    for(const std::vector<llama_kv_preparation_device> & bad:{
            std::vector<llama_kv_preparation_device>{{2,16*MiB},{2,0}}, {{3,0}}, {{-2,0}}, {{2,UINT64_MAX}}}) {
        auto wrong=kv;wrong.target=bad;
        REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error,false,false,&host,&wrong));
        REQUIRE(common_resource_preparation_peak_json(report)==good);
    }
    auto overflow=kv;overflow.host_staging_bytes=UINT64_MAX;
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error,false,false,&host,&overflow));
    REQUIRE(common_resource_preparation_peak_json(report)==good);
    auto same=kv;same.target=same.current;same.host_metadata_bytes=same.host_staging_bytes=0;
    REQUIRE(common_resource_rebalance_preparation_peak(current,current,report,error,false,false,&host,&same));
    REQUIRE(!report.prepares_kv && report.ram_peak_bytes==28*MiB);
    // A planned host row already includes its KV; the receipt must not add it again.
    current.devices.front().id=target.devices.front().id=-1;
    kv.current={{-1,20*MiB}};kv.target={{-1,16*MiB}};
    current.ram_capacity_bytes=target.ram_capacity_bytes=128*MiB;
    REQUIRE(common_resource_rebalance_preparation_peak(current,target,report,error,false,false,&host,&kv));
    REQUIRE(report.ram_current_bytes==44*MiB && report.ram_target_bytes==40*MiB && report.ram_peak_bytes==63*MiB);
}

void test_gpu_cache_preparation_preserves_host_ram_budget() {
    auto current=runtime_plan(100*MiB,0,10*MiB,20*MiB,10*MiB,4096);
    current.ram_capacity_bytes=current.ram_planned_bytes=512*MiB;
    auto target=current;target.devices.front().expert_cache_bytes=20*MiB;
    target.devices.front().planned_bytes+=10*MiB;target.devices.front().headroom_bytes-=10*MiB;
    common_resource_preparation_peak report;
    std::string error;
    REQUIRE(common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(report.prepares_expert_cache && report.devices.front().prepared_expert_cache_bytes==20*MiB);
    REQUIRE(report.ram_current_bytes==512*MiB && report.ram_target_bytes==512*MiB);
    REQUIRE(report.ram_peak_bytes==512*MiB && report.ram_peak_headroom_bytes==0);
    current.ram_planned_bytes++;target=current;
    REQUIRE(!common_resource_rebalance_preparation_peak(current,target,report,error));
    REQUIRE(error.find("combined RAM")!=std::string::npos);
}

void test_native_host_context_reservation() {
    common_resource_plan original;
    original.ram_capacity_bytes = 16*MiB;
    original.ram_planned_bytes = 16*MiB;
    original.host_graph_metadata_bytes = 2*MiB;
    original.aux_ram_bytes = 10*MiB;
    original.devices.push_back({0});
    llama_context_host_requirements required = {4*MiB, 0, 3*MiB+123, 0};
    std::string error;
    auto plan = original;
    REQUIRE(common_resource_plan_reserve_host_context(plan, required, error));
    REQUIRE(plan.host_context_metadata_bytes == MiB+123);
    REQUIRE(plan.aux_ram_bytes == 8*MiB);
    REQUIRE(plan.ram_planned_bytes == 15*MiB+123);
    REQUIRE(nlohmann::json::parse(common_resource_plan_json(plan)).at("host_context_metadata_bytes") == MiB+123);
    // Admission must account for host KV even when the plan has only GPU rows.
    required.kv_bytes = MiB;
    REQUIRE(common_resource_plan_reserve_host_context(plan, required, error));
    REQUIRE(plan.aux_ram_bytes == 7*MiB);
    for (int delta : {-1, 0, 1}) {
        plan = original;
        plan.ram_capacity_bytes = 8*MiB+123+delta;
        const auto before = common_resource_plan_json(plan);
        const bool accepted = common_resource_plan_reserve_host_context(plan, required, error);
        REQUIRE(accepted == (delta >= 0));
        if (accepted) REQUIRE(plan.aux_ram_bytes == 0 && plan.ram_planned_bytes == 7*MiB+123);
        else REQUIRE(common_resource_plan_json(plan) == before);
    }
    plan = original;
    required.fixed_bytes = UINT64_MAX;
    REQUIRE(!common_resource_plan_reserve_host_context(plan, required, error));
    REQUIRE(common_resource_plan_json(plan) == common_resource_plan_json(original));
    std::puts("PASS: native context reservations, whole-MiB auxiliary capacity, host KV, exact boundaries and atomic rejection");
}

void test_host_graph_metadata_reservation() {
    common_resource_plan_input input;
    input.requested_policy = COMMON_MEMORY_POLICY_CACHE;
    input.dense_ram_bytes = 4*MiB;
    input.host_graph_metadata_bytes = 8*MiB;
    input.requested_context = input.max_context = input.min_context = 512;
    input.context_alignment = 32;
    input.slots = 1;
    input.requested_batch = input.requested_ubatch = 32;
    input.min_kv_quality = COMMON_KV_QUALITY_F16;
    input.kv_bytes_per_token[COMMON_KV_QUALITY_F16] = 32;
    const uint64_t exact = 12*MiB + 512*32;
    std::string error;
    common_resource_plan plan;
    for (int delta : {-1, 0, 1}) {
        input.max_ram_bytes = exact + delta;
        input.devices = {{-1, input.max_ram_bytes, 0, 4*MiB, 0}};
        const bool accepted = common_resource_plan_solve(input, plan, error);
        REQUIRE(accepted == (delta >= 0));
        if (accepted) {
            REQUIRE(plan.ram_planned_bytes == 12*MiB && plan.host_graph_metadata_bytes == 8*MiB);
            REQUIRE(nlohmann::json::parse(common_resource_plan_json(plan)).at("host_graph_metadata_bytes") == 8*MiB);
        }
    }
    input.max_ram_bytes = 16*MiB;
    input.devices.front().free_bytes = input.max_ram_bytes;
    REQUIRE(common_resource_plan_solve(input, plan, error));
    common_resource_host_memory host = {4*MiB, 8*MiB, 0};
    common_resource_preparation_peak report;
    REQUIRE(common_resource_rebalance_preparation_peak(plan, plan, report, error, false, false, &host));
    REQUIRE(report.ram_current_bytes == exact && report.ram_peak_bytes == exact && report.ram_tracked_extra_bytes == 0);
    REQUIRE(report.ram_host_weights_limit_bytes == 4*MiB && report.ram_host_fixed_limit_bytes == 8*MiB);
    host.fixed_bytes = 10*MiB;
    REQUIRE(common_resource_rebalance_preparation_peak(plan, plan, report, error, false, false, &host));
    REQUIRE(report.ram_current_bytes == exact + 2*MiB && report.ram_tracked_extra_bytes == 2*MiB);
    auto changed = plan;
    changed.host_graph_metadata_bytes++;
    const auto previous = common_resource_preparation_peak_json(report);
    REQUIRE(!common_resource_rebalance_preparation_peak(plan, changed, report, error, false, false, &host));
    REQUIRE(common_resource_preparation_peak_json(report) == previous);
    changed = plan;
    changed.host_context_metadata_bytes++;
    REQUIRE(!common_resource_rebalance_preparation_peak(plan, changed, report, error, false, false, &host));
    REQUIRE(common_resource_preparation_peak_json(report) == previous);
    input.host_graph_metadata_bytes = UINT64_MAX;
    REQUIRE(!common_resource_plan_solve(input, changed, error));
    std::puts("PASS: graph metadata reserved before startup solve, byte boundaries/overflow, tracked-host reuse and immutable reservation");
}

} // namespace

int main(int argc, char ** argv) {
    try {
    if (argc == 3 && std::string(argv[1]) == "--forecast-case") {
        const std::string name = argv[2];
        if (name == "long") test_native_forecast_long_context();
        else if (name == "short") test_native_forecast_short_context();
        else if (name == "rebalance") test_native_forecast_rebalance();
        else if (name == "step") test_native_forecast_step_storage();
        else if (name == "host") test_native_forecast_host_budget();
        else if (name == "host-peak") test_host_cache_preparation_peak_rejects_combined_budget();
        else if (name == "host-transient-peak") test_host_transient_preparation_peak_rejects_combined_budget();
        else throw std::runtime_error("unknown forecast case");
        std::puts("PASS: selected context forecast control");
        return 0;
    }
    REQUIRE(argc == 1);
    test_deterministic_budget_and_json();
    test_optional_prefill_staging_falls_back_without_violating_context_floor();
    test_preference_tradeoff();
    test_no_silent_fallbacks();
    test_atomic_rollback();
    test_runtime_rebalance_target_is_pure_and_reserve_bounded();
    test_runtime_transient_policy_targets_are_explicit_and_pure();
    test_shared_transient_policy_preserves_request_swap_rollback_reserve();
    test_preparation_peak_rejects_final_fit_without_double_buffer_headroom();
    test_preparation_peak_detects_u64_overflow_without_mutating_report();
    test_preparation_peak_accepts_exact_reserve_boundary();
    test_preparation_peak_reports_single_pool_replacement();
    test_preparation_peak_counts_transient_owner_replacement();
    test_preparation_peak_couples_shared_transient_owner_bounds();
    test_preparation_peak_counts_same_target_expert_reconciliation();
    test_preparation_peak_rejects_same_target_expert_reconciliation_without_headroom();
    test_preparation_peak_rejects_combined_cross_pool_trade();
    test_interface_parsers();
    test_compatibility_presets_and_host_budget();
    test_auto_unlimited_ram_and_minimum_expert_component();
    test_rollback_hook_failure_is_single_shot();
    test_native_override_normalization();
    test_calibrated_expert_split_solver();
    test_native_calibration_profile_gate();
    test_native_forecast_long_context();
    test_native_forecast_short_context();
    test_native_forecast_rebalance();
    test_native_forecast_step_storage();
    test_native_forecast_failure_and_rounding();
    test_native_forecast_host_budget();
    test_host_cache_preparation_peak_rejects_combined_budget();
    test_host_transient_preparation_peak_rejects_combined_budget();
    test_host_cache_preparation_peak_boundaries();
    test_host_cache_preparation_peak_overflow_and_immutable_budget();
    test_gpu_cache_preparation_preserves_host_ram_budget();
    test_tracked_device_memory_preparation();
    test_tracked_host_memory_preparation();
    test_native_kv_preparation_receipt();
    test_host_graph_metadata_reservation();
    test_native_host_context_reservation();
    std::puts("PASS: resource planning including fixed/stepped native cache forecasts");
    return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL: %s\n",error.what());
        return 1;
    }
}
