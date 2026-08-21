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

bool common_resource_plan_solve(
    const common_resource_plan_input & input,
    common_resource_plan & plan,
    std::string & error);

std::string common_resource_plan_json(const common_resource_plan & plan);
std::string common_memory_policy_name(common_memory_policy policy);
std::string common_kv_quality_name(common_kv_quality quality);
std::string common_resource_backend_name(common_resource_backend backend);

bool common_parse_byte_size(const std::string & text, uint64_t & bytes, std::string & error);
bool common_parse_token_count(const std::string & text, uint32_t & count, std::string & error);
bool common_parse_memory_policy(const std::string & text, common_memory_policy & policy);
bool common_parse_resource_preference(const std::string & text, common_resource_preference & preference);
bool common_parse_kv_quality(const std::string & text, common_kv_quality & quality);
bool common_parse_resource_backend(const std::string & text, common_resource_backend & backend);

bool common_resource_apply_plan_atomic(
    common_resource_plan & current,
    const common_resource_plan & target,
    const std::function<bool(const common_resource_plan &, std::string &)> & prepare,
    const std::function<void()> & commit,
    const std::function<void()> & rollback,
    std::string & error,
    common_resource_transition_stats * stats = nullptr);
