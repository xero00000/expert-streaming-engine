#pragma once

#include "transient-policy.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
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

    // Prepared residency callbacks retain the old owner until finalize.
    // prepare_residency(false) must not free or detach the live owner;
    // prepare_residency(true) allocates a complete off-side owner. Publish and
    // rollback are allocation-free inverse swaps. Callbacks must not throw or
    // re-enter the manager. Once publish succeeds, rollback and finalize are
    // invariants: returning false from either is fatal because a partially
    // restored owner set cannot be exposed safely.
    std::function<void *(bool target_resident, std::string * error)> prepare_residency;
    std::function<bool(void * transaction, std::string * error)> publish_residency;
    std::function<bool(void * transaction, std::string * error)> rollback_residency;
    std::function<bool(void * transaction, std::string * error)> finalize_residency;
    std::function<void(void * transaction)> free_residency;
};

struct common_transient_module_status {
    std::string id;
    common_transient_module_kind kind = COMMON_TRANSIENT_MODULE_MTP;
    int device = 0;
    size_t bytes = 0;
    bool enabled = true;
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
    std::map<int, common_transient_device_budget> budgets;
    common_transient_policy policy = COMMON_TRANSIENT_POLICY_SHARED;
    uint64_t active_leases = 0;
    uint64_t pending_restores = 0;
    bool reconfiguration_open = false;
};

struct common_transient_policy_transaction;

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

    // A policy transaction derives capacity and enabled modules from the
    // registered MTP/multimodal bounds. Preparation owns every off-side module
    // candidate while leaving live residency untouched. Only one policy
    // transaction may be open and acquisitions are excluded through finalize;
    // free closes the finalized handle after the caller publishes logical state.
    common_transient_policy_transaction * prepare_policy(
        common_transient_policy target,
        std::string * error = nullptr);
    bool publish_policy(common_transient_policy_transaction * transaction, std::string * error = nullptr);
    bool rollback_policy(common_transient_policy_transaction * transaction, std::string * error = nullptr);
    bool finalize_policy(common_transient_policy_transaction * transaction, std::string * error = nullptr);
    bool free_policy(common_transient_policy_transaction * transaction, std::string * error = nullptr);

    // Owner-thread shutdown hook. HTTP admission must already be closed and
    // all request workers joined. Resolves every remaining request lease
    // without exposing a partially published exact-owner epoch.
    bool resolve_all_leases_for_shutdown(std::string * error = nullptr);

    bool module_enabled(const std::string & id) const;
    common_transient_module_telemetry snapshot() const;

private:
    struct module_entry {
        common_transient_module_desc desc;
        bool enabled = true;
        bool resident = false;
        uint64_t last_use = 0;
        uint64_t pins = 0;
    };

    struct residency_transaction {
        struct transition {
            std::string id;
            std::shared_ptr<void> handle;
            bool prior_resident = false;
            bool target_resident = false;
            bool published = false;
        };
        std::vector<transition> modules;
        bool published = false;
    };

    struct residency_epoch {
        std::vector<std::string> activated;
        std::vector<std::string> evicted;
        std::vector<std::string> affected;
        std::unique_ptr<residency_transaction> owners;
        uint64_t open_leases = 0;
        bool committed = false;
    };

    struct lease_entry {
        std::vector<std::string> requested;
        bool restore_prior = true;
        std::shared_ptr<residency_epoch> epoch;
    };

    std::unique_ptr<residency_transaction> prepare_residency_locked(
        const std::vector<std::string> & evicted,
        const std::vector<std::string> & activated,
        std::string & error);
    bool publish_residency_locked(
        residency_transaction & transaction,
        std::string & error);
    bool rollback_residency_locked(
        residency_transaction & transaction,
        std::string & error);
    bool finalize_residency_locked(
        residency_transaction & transaction,
        std::string & error);
    void resume_locked(const std::vector<std::string> & affected);
    void fail_locked(const std::string & message, std::string * error);
    size_t usable_bytes_locked(int device) const;
    std::map<int, size_t> resident_bytes_locked() const;

    mutable std::mutex mutex_;
    std::map<int, common_transient_device_budget> budgets_;
    std::map<std::string, module_entry> modules_;
    common_transient_module_telemetry telemetry_;
    uint64_t clock_ = 0;
    uint64_t next_lease_id_ = 1;
    std::map<uint64_t, lease_entry> leases_;
    std::shared_ptr<residency_epoch> active_residency_epoch_;
    common_transient_policy policy_ = COMMON_TRANSIENT_POLICY_SHARED;
    common_transient_policy_transaction * active_policy_transaction_ = nullptr;
    uint64_t policy_generation_ = 0;
};
