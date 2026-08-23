#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

enum common_memory_policy {
    COMMON_MEMORY_POLICY_AUTO,
    COMMON_MEMORY_POLICY_RESIDENT,
    COMMON_MEMORY_POLICY_CACHE,
    COMMON_MEMORY_POLICY_STREAM,
};

enum common_resource_preference {
    COMMON_RESOURCE_PREFERENCE_BALANCED,
    COMMON_RESOURCE_PREFERENCE_LATENCY,
    COMMON_RESOURCE_PREFERENCE_THROUGHPUT,
};

// Ordered from lowest to highest quality. A floor forbids every lower value.
enum common_kv_quality {
    COMMON_KV_QUALITY_TURBO1,
    COMMON_KV_QUALITY_TURBO2,
    COMMON_KV_QUALITY_TURBO3,
    COMMON_KV_QUALITY_TURBO4,
    COMMON_KV_QUALITY_TURBO8,
    COMMON_KV_QUALITY_Q8,
    COMMON_KV_QUALITY_F16,
};

enum common_resource_backend {
    COMMON_RESOURCE_BACKEND_MMAP,
    COMMON_RESOURCE_BACKEND_PREAD,
    COMMON_RESOURCE_BACKEND_IO_URING,
};

struct common_resource_device_input {
    int id = 0;
    uint64_t free_bytes = 0;
    uint64_t reserve_bytes = 0;
    uint64_t dense_bytes = 0;
    uint64_t graph_bytes = 0;
};

struct common_resource_plan_input {
    common_memory_policy requested_policy = COMMON_MEMORY_POLICY_AUTO;
    common_resource_preference preference = COMMON_RESOURCE_PREFERENCE_BALANCED;
    common_resource_backend requested_backend = COMMON_RESOURCE_BACKEND_PREAD;
    bool backend_available = true;
    bool model_is_moe = false;

    uint64_t max_ram_bytes = 0;
    uint64_t dense_ram_bytes = 0;
    uint64_t expert_source_bytes = 0;
    uint64_t requested_expert_ram_bytes = 0;
    uint64_t min_expert_ram_bytes = 0;
    uint64_t requested_aux_ram_bytes = 0;
    uint64_t requested_expert_vram_bytes_per_device = 0;
    // Total bytes for two independent whole-layer prefill staging lanes on
    // each accelerator. The planner may disable this optional optimization
    // when it would cross the context/KV quality floor.
    uint64_t requested_expert_prefill_staging_bytes_per_device = 0;
    bool require_expert_prefill_staging = false;
    uint64_t io_staging_bytes = 0;

    uint64_t mtp_bytes = 0;
    uint64_t multimodal_bytes = 0;
    // Device which hosts mutually-exclusive MTP/mmproj modules. -1 selects
    // the first device deterministically.
    int transient_device = -1;
    bool require_draft = false;

    uint32_t requested_context = 0;
    uint32_t max_context = 0;
    uint32_t min_context = 512;
    uint32_t context_alignment = 1;
    uint32_t slots = 1;
    uint32_t requested_batch = 2048;
    uint32_t requested_ubatch = 512;
    common_kv_quality min_kv_quality = COMMON_KV_QUALITY_TURBO4;

    // Complete KV allocation per token, across all layers, for each quality.
    // Missing or zero entries are unavailable.
    uint64_t kv_bytes_per_token[COMMON_KV_QUALITY_F16 + 1] = {};
    std::vector<common_resource_device_input> devices;
};

struct common_resource_device_plan {
    int id = 0;
    uint64_t capacity_bytes = 0;
    uint64_t reserve_bytes = 0;
    uint64_t dense_bytes = 0;
    uint64_t graph_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t expert_prefill_staging_bytes = 0;
    uint64_t transient_bytes = 0;
    uint64_t planned_bytes = 0;
    uint64_t headroom_bytes = 0;
};

struct common_resource_plan {
    common_memory_policy policy = COMMON_MEMORY_POLICY_RESIDENT;
    common_resource_backend backend = COMMON_RESOURCE_BACKEND_PREAD;
    common_kv_quality kv_quality = COMMON_KV_QUALITY_F16;
    uint32_t context = 0;
    uint32_t slots = 0;
    uint32_t batch = 0;
    uint32_t ubatch = 0;
    uint64_t ram_capacity_bytes = 0;
    uint64_t ram_planned_bytes = 0;
    uint64_t expert_ram_bytes = 0;
    uint64_t aux_ram_bytes = 0;
    uint64_t io_staging_bytes = 0;
    uint64_t transient_capacity_bytes = 0;
    bool expert_prefill_staging_enabled = false;
    bool transient_swap = false;
    bool draft_resident = false;
    std::string reason;
    std::vector<common_resource_device_plan> devices;
};

struct common_resource_transition_stats {
    uint64_t attempts = 0;
    uint64_t commits = 0;
    uint64_t rollbacks = 0;
    uint64_t prepare_failures = 0;
    uint64_t commit_failures = 0;
};

// A zero context leaves the current context unchanged. Expert-cache changes
// are explicit because zero is itself a valid request that disables the pool.
struct common_resource_rebalance_request {
    uint32_t context = 0;
    bool set_expert_cache_bytes_per_device = false;
    uint64_t expert_cache_bytes_per_device = 0;
};

// Peak device accounting while a runtime rebalance is still failure-atomic.
// The current live allocation cannot be released until every replacement pool
// has been prepared, so prepared bytes are additional to current_live_bytes.
struct common_resource_device_preparation_peak {
    int id = 0;
    uint64_t capacity_bytes = 0;
    uint64_t reserve_bytes = 0;
    uint64_t current_live_bytes = 0;
    uint64_t target_live_bytes = 0;
    uint64_t prepared_kv_bytes = 0;
    uint64_t prepared_expert_cache_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t peak_headroom_bytes = 0;
};

struct common_resource_preparation_peak {
    bool prepares_kv = false;
    bool prepares_expert_cache = false;
    std::vector<common_resource_device_preparation_peak> devices;
};

struct common_expert_split_input {
    uint32_t misses = 0;
    // Contended costs from a complete, topology-current hardware profile.
    double cpu_ns_per_expert = 0;
    double upload_ns_per_expert = 0;
    double cpu_confidence = 0;
    double upload_confidence = 0;
    double minimum_confidence = 0.80;
    double hysteresis_fraction = 0.05;
    int32_t previous_upload_experts = -1;
    bool calibration_complete = false;
};

struct common_expert_split_plan {
    uint32_t cpu_experts = 0;
    uint32_t upload_experts = 0;
    double predicted_step_ns = 0;
    bool retained_by_hysteresis = false;
};

struct common_expert_calibration_key {
    std::string backend;
    int32_t ggml_type = -1;
    uint64_t input_width = 0;
    uint64_t expert_width = 0;
    uint64_t bytes_per_expert_component = 0;

    bool operator==(const common_expert_calibration_key & other) const {
        return backend == other.backend && ggml_type == other.ggml_type &&
                input_width == other.input_width && expert_width == other.expert_width &&
                bytes_per_expert_component == other.bytes_per_expert_component;
    }
};

struct common_expert_calibration_entry {
    common_expert_calibration_key key;
    double cpu_ns_per_expert_component = 0;
    double upload_ns_per_expert_component = 0;
    double cpu_confidence = 0;
    double upload_confidence = 0;
};

struct common_expert_calibration_profile {
    std::vector<common_expert_calibration_entry> entries;
};

bool common_resource_plan_solve(
    const common_resource_plan_input & input,
    common_resource_plan & plan,
    std::string & error);

// Pure target derivation for a running plan. This performs no allocation and
// never mutates current; callers can expose it as a safe dry-run boundary.
bool common_resource_rebalance_target(
    const common_resource_plan & current,
    const common_resource_rebalance_request & request,
    common_resource_plan & target,
    std::string & error);

// Validate the double-buffer preparation peak for a derived target. A pool is
// treated as a whole-plan replacement when its context/allocation differs on
// any device; its complete target allocation is then counted on every device.
// force_expert_cache_preparation covers explicit same-target reconciliation,
// where the realized allocation must be rebuilt despite no serialized delta.
// report is updated only after all devices pass overflow and reserve checks.
bool common_resource_rebalance_preparation_peak(
    const common_resource_plan & current,
    const common_resource_plan & target,
    common_resource_preparation_peak & report,
    std::string & error,
    bool force_expert_cache_preparation = false);

bool common_expert_split_solve(
    const common_expert_split_input & input,
    common_expert_split_plan & plan,
    std::string & error);

// Parse only a complete planner-ready production calibration matrix. The
// launcher remains responsible for verifying the topology fingerprint before
// passing a profile to native policy.
bool common_expert_calibration_parse_json(
    const std::string & text,
    common_expert_calibration_profile & profile,
    std::string & error);

bool common_expert_calibration_lookup(
    const common_expert_calibration_profile & profile,
    const common_expert_calibration_key & key,
    common_expert_calibration_entry & entry);

// Sum every component required by one routed expert, then solve the integer
// CPU/upload split for this decode step. Missing calibration fails closed.
bool common_expert_split_solve_calibrated(
    const common_expert_calibration_profile & profile,
    const std::vector<common_expert_calibration_key> & components,
    uint32_t misses,
    int32_t previous_upload_experts,
    double hysteresis_fraction,
    common_expert_split_plan & plan,
    std::string & error);

std::string common_resource_plan_json(const common_resource_plan & plan);
std::string common_resource_preparation_peak_json(const common_resource_preparation_peak & report);
std::string common_memory_policy_name(common_memory_policy policy);
std::string common_kv_quality_name(common_kv_quality quality);
std::string common_resource_backend_name(common_resource_backend backend);

bool common_parse_byte_size(const std::string & text, uint64_t & bytes, std::string & error);
bool common_parse_token_count(const std::string & text, uint32_t & count, std::string & error);
bool common_parse_memory_policy(const std::string & text, common_memory_policy & policy);
bool common_parse_resource_preference(const std::string & text, common_resource_preference & preference);
bool common_parse_kv_quality(const std::string & text, common_kv_quality & quality);
bool common_parse_resource_backend(const std::string & text, common_resource_backend & backend);

// Normalize launcher/native-controller overrides before solving the plan.
// Zero is an explicit disabled cache, not a request for an automatic maximum.
uint64_t common_resource_cache_limit_bytes(int64_t capacity_mib);
bool common_resource_should_enable_fit(bool has_accelerator, int32_t n_cpu_moe);

bool common_resource_apply_plan_atomic(
    common_resource_plan & current,
    const common_resource_plan & target,
    const std::function<bool(const common_resource_plan &, std::string &)> & prepare,
    const std::function<void()> & commit,
    const std::function<void()> & rollback,
    std::string & error,
    common_resource_transition_stats * stats = nullptr);
