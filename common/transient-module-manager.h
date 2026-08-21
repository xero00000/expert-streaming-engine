#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

enum common_transient_module_kind {
    COMMON_TRANSIENT_MODULE_MTP,
    COMMON_TRANSIENT_MODULE_VISION,
    COMMON_TRANSIENT_MODULE_AUDIO,
    COMMON_TRANSIENT_MODULE_RERANKER,
    COMMON_TRANSIENT_MODULE_EMBEDDING,
    COMMON_TRANSIENT_MODULE_LORA,
};

struct common_transient_device_budget {
    size_t capacity_bytes = 0;
    size_t safety_margin_bytes = 0;
};

// Callbacks operate on the actual backend allocation. They must be idempotent,
// must not re-enter the manager, and return false without changing prior state.
struct common_transient_module_desc {
    std::string id;
    common_transient_module_kind kind = COMMON_TRANSIENT_MODULE_MTP;
    int device = 0;
    size_t bytes = 0;
    bool initially_resident = false;
    std::vector<std::string> streams;
    std::function<bool()> quiesce;
    std::function<bool()> activate;
    std::function<bool()> deactivate;
    std::function<void()> resume;
};

struct common_transient_module_status {
    std::string id;
    common_transient_module_kind kind = COMMON_TRANSIENT_MODULE_MTP;
    int device = 0;
    size_t bytes = 0;
    bool resident = false;
    uint64_t pins = 0;
};

struct common_transient_module_telemetry {
    uint64_t transactions = 0;
    uint64_t swaps = 0;
    uint64_t rollbacks = 0;
    uint64_t failures = 0;
    uint64_t oom_rejections = 0;
    uint64_t bytes_moved = 0;
    uint64_t swap_latency_us = 0;
    std::string last_failure;
    std::vector<common_transient_module_status> modules;
    std::map<int, size_t> resident_bytes;
};

class common_transient_module_manager {
public:
    explicit common_transient_module_manager(std::map<int, common_transient_device_budget> budgets);

    bool register_module(common_transient_module_desc desc, std::string * error = nullptr);

    // Pins the requested modules until release(). Returns zero on failure.
    // Overlapping leases can share already-resident modules; pinned modules are
    // never eviction candidates. Prior residency is restored after the last
    // overlapping pin is released.
    uint64_t acquire(
        const std::vector<std::string> & requested,
        bool restore_prior = true,
        std::string * error = nullptr);

    bool release(uint64_t lease_id, bool success = true, std::string * error = nullptr);

    // Capacity is reserved before callbacks run. restore_prior restores the exact
    // prior residency after body; a failed body always rolls back.
    bool run(
        const std::vector<std::string> & requested,
        const std::function<bool()> & body,
        bool restore_prior = true,
        std::string * error = nullptr);

    common_transient_module_telemetry snapshot() const;

private:
    struct module_entry {
        common_transient_module_desc desc;
        bool resident = false;
        uint64_t last_use = 0;
        uint64_t pins = 0;
    };

    struct lease_entry {
        std::vector<std::string> requested;
        std::vector<std::string> activated;
        std::vector<std::string> evicted;
        bool restore_prior = true;
    };

    struct restore_entry {
        std::vector<std::string> activated;
        std::vector<std::string> evicted;
        bool rollback = false;
    };

    bool restore_locked(
        const std::vector<std::string> & activated,
        const std::vector<std::string> & evicted,
        std::string & error);
    void resume_locked(const std::vector<std::string> & affected);
    void fail_locked(const std::string & message, std::string * error);
    bool apply_pending_restores_locked(std::string & error);
    size_t usable_bytes_locked(int device) const;
    std::map<int, size_t> resident_bytes_locked() const;

    mutable std::mutex mutex_;
    std::map<int, common_transient_device_budget> budgets_;
    std::map<std::string, module_entry> modules_;
    common_transient_module_telemetry telemetry_;
    uint64_t clock_ = 0;
    uint64_t next_lease_id_ = 1;
    std::map<uint64_t, lease_entry> leases_;
    std::vector<restore_entry> pending_restores_;
};
