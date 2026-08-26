#include "transient-module-manager.h"

#include "ggml.h"

#include <algorithm>
#include <memory>
#include <set>

namespace {

struct transient_telemetry_latency_scope {
    uint64_t & total;
    int64_t started_us = ggml_time_us();

    ~transient_telemetry_latency_scope() noexcept {
        const int64_t finished_us = ggml_time_us();
        if (finished_us > started_us) {
            total += static_cast<uint64_t>(finished_us - started_us);
        }
    }
};

} // namespace

enum common_transient_policy_transaction_state {
    COMMON_TRANSIENT_POLICY_PREPARED,
    COMMON_TRANSIENT_POLICY_PUBLISHED,
    COMMON_TRANSIENT_POLICY_ROLLED_BACK,
    COMMON_TRANSIENT_POLICY_FINALIZED,
};

struct common_transient_policy_module_transition {
    std::string id;
    std::shared_ptr<void> handle;
    bool prior_enabled = true;
    bool target_enabled = true;
    bool prior_resident = false;
    bool target_resident = false;
    bool published = false;
};

struct common_transient_policy_transaction {
    common_transient_module_manager * owner = nullptr;
    common_transient_policy prior_policy = COMMON_TRANSIENT_POLICY_SHARED;
    common_transient_policy target_policy = COMMON_TRANSIENT_POLICY_SHARED;
    common_transient_policy_transaction_state state = COMMON_TRANSIENT_POLICY_PREPARED;
    uint64_t generation = 0;
    std::map<int, common_transient_device_budget> prior_budgets;
    std::map<int, common_transient_device_budget> target_budgets;
    std::vector<common_transient_policy_module_transition> modules;
    bool poisoned = false;
};

common_transient_module_manager::common_transient_module_manager(
        std::map<int, common_transient_device_budget> budgets)
    : budgets_(std::move(budgets)) {
}

size_t common_transient_module_manager::usable_bytes_locked(int device) const {
    const auto it = budgets_.find(device);
    if (it == budgets_.end() || it->second.safety_margin_bytes >= it->second.capacity_bytes) {
        return 0;
    }
    return it->second.capacity_bytes - it->second.safety_margin_bytes;
}

std::map<int, size_t> common_transient_module_manager::resident_bytes_locked() const {
    std::map<int, size_t> result;
    for (const auto & pair : modules_) {
        if (pair.second.resident) {
            result[pair.second.desc.device] += pair.second.desc.bytes;
        }
    }
    return result;
}

void common_transient_module_manager::fail_locked(const std::string & message, std::string * error) {
    telemetry_.failures++;
    telemetry_.last_failure = message;
    if (error != nullptr) {
        *error = message;
    }
}

bool common_transient_module_manager::register_module(
        common_transient_module_desc desc,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_policy_transaction_ != nullptr) {
        fail_locked("cannot register a transient module while policy reconfiguration is open", error);
        return false;
    }
    if (desc.id.empty()) {
        fail_locked("transient module id must not be empty", error);
        return false;
    }
    if (modules_.count(desc.id) != 0) {
        fail_locked("duplicate transient module: " + desc.id, error);
        return false;
    }
    if (budgets_.count(desc.device) == 0) {
        fail_locked("transient module references an unbudgeted device: " + desc.id, error);
        return false;
    }
    if (desc.bytes > usable_bytes_locked(desc.device)) {
        fail_locked("transient module exceeds the device budget: " + desc.id, error);
        return false;
    }

    module_entry entry;
    entry.resident = desc.initially_resident;
    entry.last_use = ++clock_;
    entry.desc = std::move(desc);
    const std::string id = entry.desc.id;
    const int device = entry.desc.device;
    modules_.emplace(id, std::move(entry));

    const auto resident = resident_bytes_locked();
    const auto used = resident.find(device);
    if (used != resident.end() && used->second > usable_bytes_locked(device)) {
        modules_.erase(id);
        fail_locked("initial transient residency exceeds the device budget: " + id, error);
        return false;
    }
    policy_generation_++;
    return true;
}

common_transient_policy_transaction * common_transient_module_manager::prepare_policy(
        common_transient_policy target,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    telemetry_.transactions++;
    transient_telemetry_latency_scope latency{telemetry_.swap_latency_us};
    if (active_policy_transaction_ != nullptr) {
        fail_locked("another transient policy transaction is already open", error);
        return nullptr;
    }
    if (!leases_.empty() || active_residency_epoch_ != nullptr) {
        fail_locked("transient policy reconfiguration requires no active leases or residency epoch", error);
        return nullptr;
    }
    if (target < COMMON_TRANSIENT_POLICY_OFF ||
            target > COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY) {
        fail_locked("unknown transient policy", error);
        return nullptr;
    }

    std::vector<std::string> quiesced;
    try {
        auto transaction = std::make_unique<common_transient_policy_transaction>();
        transaction->owner = this;
        transaction->prior_policy = policy_;
        transaction->target_policy = target;
        transaction->generation = policy_generation_;
        transaction->prior_budgets = budgets_;
        transaction->target_budgets = budgets_;

        std::string mtp_id;
        std::string multimodal_id;
        for (const auto & pair : modules_) {
            switch (pair.second.desc.kind) {
                case COMMON_TRANSIENT_MODULE_MTP:
                    if (!mtp_id.empty()) {
                        fail_locked("transient policy currently supports one aggregate MTP module", error);
                        return nullptr;
                    }
                    mtp_id = pair.first;
                    break;
                case COMMON_TRANSIENT_MODULE_VISION:
                    if (!multimodal_id.empty()) {
                        fail_locked("transient policy currently supports one aggregate multimodal module", error);
                        return nullptr;
                    }
                    multimodal_id = pair.first;
                    break;
                default:
                    fail_locked("transient policy does not yet cover the registered module kind", error);
                    return nullptr;
            }
        }
        const bool enable_mtp = target == COMMON_TRANSIENT_POLICY_SHARED ||
            target == COMMON_TRANSIENT_POLICY_MTP_ONLY;
        const bool enable_multimodal = target == COMMON_TRANSIENT_POLICY_SHARED ||
            target == COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY;
        if (enable_mtp && mtp_id.empty()) {
            fail_locked("requested transient policy requires a configured MTP module", error);
            return nullptr;
        }
        if (enable_multimodal && multimodal_id.empty()) {
            fail_locked("requested transient policy requires a configured multimodal module", error);
            return nullptr;
        }

        std::string preferred;
        if (target == COMMON_TRANSIENT_POLICY_MTP_ONLY) {
            preferred = mtp_id;
        } else if (target == COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY) {
            preferred = multimodal_id;
        } else if (target == COMMON_TRANSIENT_POLICY_SHARED) {
            for (const std::string * candidate : {&mtp_id, &multimodal_id}) {
                if (!candidate->empty() && modules_.at(*candidate).resident &&
                        (preferred.empty() || modules_.at(*candidate).last_use >
                            modules_.at(preferred).last_use)) {
                    preferred = *candidate;
                }
            }
            if (preferred.empty()) preferred = !mtp_id.empty() ? mtp_id : multimodal_id;
        }

        for (auto & per_device : transaction->target_budgets) {
            size_t required = 0;
            for (const auto & pair : modules_) {
                const bool enabled = pair.first == mtp_id ? enable_mtp : enable_multimodal;
                if (enabled && pair.second.desc.device == per_device.first) {
                    required = std::max(required, pair.second.desc.bytes);
                }
            }
            if (required > SIZE_MAX - per_device.second.safety_margin_bytes) {
                fail_locked("transient policy capacity overflows the safety margin", error);
                return nullptr;
            }
            per_device.second.capacity_bytes = required + per_device.second.safety_margin_bytes;
        }

        for (const auto & pair : modules_) {
            const bool target_enabled = pair.first == mtp_id ? enable_mtp : enable_multimodal;
            const bool target_resident = !preferred.empty() && pair.first == preferred;
            common_transient_policy_module_transition transition;
            transition.id = pair.first;
            transition.prior_enabled = pair.second.enabled;
            transition.target_enabled = target_enabled;
            transition.prior_resident = pair.second.resident;
            transition.target_resident = target_resident;
            if (transition.prior_resident != transition.target_resident) {
                const auto & desc = pair.second.desc;
                if (!desc.prepare_residency || !desc.publish_residency ||
                        !desc.rollback_residency || !desc.finalize_residency ||
                        !desc.free_residency) {
                    fail_locked("transient module lacks prepared residency callbacks: " + pair.first, error);
                    return nullptr;
                }
            }
            transaction->modules.push_back(std::move(transition));
        }
        std::stable_sort(
            transaction->modules.begin(), transaction->modules.end(),
            [](const auto & lhs, const auto & rhs) {
                return lhs.target_resident < rhs.target_resident;
            });

        for (const auto & transition : transaction->modules) {
            if (transition.prior_resident == transition.target_resident) continue;
            auto & desc = modules_.at(transition.id).desc;
            if (desc.quiesce && !desc.quiesce()) {
                resume_locked(quiesced);
                quiesced.clear();
                fail_locked("failed to quiesce transient module for policy preparation: " +
                    transition.id, error);
                return nullptr;
            }
            quiesced.push_back(transition.id);
        }
        for (auto & transition : transaction->modules) {
            if (transition.prior_resident == transition.target_resident) continue;
            const auto & desc = modules_.at(transition.id).desc;
            void * prepared = desc.prepare_residency(transition.target_resident, error);
            if (prepared == nullptr) {
                resume_locked(quiesced);
                quiesced.clear();
                fail_locked("failed to prepare transient module residency: " +
                    transition.id, error);
                return nullptr;
            }
            try {
                transition.handle = std::shared_ptr<void>(
                    prepared,
                    [release = desc.free_residency](void * handle) {
                        release(handle);
                    });
            } catch (...) {
                desc.free_residency(prepared);
                throw;
            }
        }
        resume_locked(quiesced);
        quiesced.clear();

        active_policy_transaction_ = transaction.get();
        return transaction.release();
    } catch (const std::exception & exception) {
        resume_locked(quiesced);
        fail_locked(std::string("failed to prepare transient policy: ") + exception.what(), error);
        return nullptr;
    } catch (...) {
        resume_locked(quiesced);
        fail_locked("unexpected failure while preparing transient policy", error);
        return nullptr;
    }
}

bool common_transient_module_manager::publish_policy(
        common_transient_policy_transaction * transaction,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    transient_telemetry_latency_scope latency{telemetry_.swap_latency_us};
    if (transaction == nullptr || active_policy_transaction_ != transaction ||
            transaction->owner != this || transaction->poisoned ||
            transaction->state != COMMON_TRANSIENT_POLICY_PREPARED ||
            transaction->generation != policy_generation_) {
        fail_locked("invalid prepared transient policy transaction", error);
        return false;
    }
    if (!leases_.empty() || active_residency_epoch_ != nullptr) {
        fail_locked("transient policy publication lost its exclusive safe point", error);
        return false;
    }

    size_t published = 0;
    for (; published < transaction->modules.size(); ++published) {
        auto & transition = transaction->modules[published];
        if (!transition.handle) continue;
        auto & desc = modules_.at(transition.id).desc;
        if (!desc.publish_residency(transition.handle.get(), error)) {
            bool restored = true;
            bool rolled_back_any = false;
            for (size_t reverse = published; reverse > 0; --reverse) {
                auto & prior = transaction->modules[reverse - 1];
                if (prior.published && prior.handle) {
                    auto & prior_desc = modules_.at(prior.id).desc;
                    const bool rolled_back = prior_desc.rollback_residency(
                        prior.handle.get(), error);
                    restored &= rolled_back;
                    prior.published = !rolled_back;
                    if (rolled_back) {
                        rolled_back_any = true;
                        telemetry_.swaps++;
                        telemetry_.bytes_moved += prior_desc.bytes;
                    }
                }
            }
            if (rolled_back_any) telemetry_.rollbacks++;
            transaction->poisoned = !restored;
            fail_locked(restored
                ? "failed to publish transient module residency: " + transition.id
                : "transient policy publication could not restore prior owners", error);
            return false;
        }
        transition.published = true;
        telemetry_.swaps++;
        telemetry_.bytes_moved += desc.bytes;
    }

    for (const auto & transition : transaction->modules) {
        auto & module = modules_.at(transition.id);
        module.enabled = transition.target_enabled;
        module.resident = transition.target_resident;
        if (module.resident) module.last_use = ++clock_;
    }
    for (const auto & target_budget : transaction->target_budgets) {
        budgets_.at(target_budget.first) = target_budget.second;
    }
    policy_ = transaction->target_policy;
    transaction->state = COMMON_TRANSIENT_POLICY_PUBLISHED;
    return true;
}

bool common_transient_module_manager::rollback_policy(
        common_transient_policy_transaction * transaction,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    transient_telemetry_latency_scope latency{telemetry_.swap_latency_us};
    if (transaction == nullptr || active_policy_transaction_ != transaction ||
            transaction->owner != this || transaction->poisoned ||
            transaction->state != COMMON_TRANSIENT_POLICY_PUBLISHED) {
        fail_locked("invalid published transient policy transaction", error);
        return false;
    }

    size_t rolled_back = 0;
    for (; rolled_back < transaction->modules.size(); ++rolled_back) {
        const size_t index = transaction->modules.size() - 1 - rolled_back;
        auto & transition = transaction->modules[index];
        if (!transition.published || !transition.handle) continue;
        auto & desc = modules_.at(transition.id).desc;
        if (!desc.rollback_residency(transition.handle.get(), error)) {
            bool republished = true;
            for (size_t redo = index + 1; redo < transaction->modules.size(); ++redo) {
                auto & prior = transaction->modules[redo];
                if (!prior.published && prior.handle) {
                    auto & prior_desc = modules_.at(prior.id).desc;
                    const bool redone = prior_desc.publish_residency(prior.handle.get(), error);
                    republished &= redone;
                    prior.published = redone;
                    if (redone) {
                        telemetry_.swaps++;
                        telemetry_.bytes_moved += prior_desc.bytes;
                    }
                }
            }
            transaction->poisoned = !republished;
            fail_locked(republished
                ? "failed to roll back transient module residency: " + transition.id
                : "transient policy rollback could not restore published owners", error);
            return false;
        }
        transition.published = false;
        telemetry_.swaps++;
        telemetry_.bytes_moved += desc.bytes;
    }

    for (const auto & transition : transaction->modules) {
        auto & module = modules_.at(transition.id);
        module.enabled = transition.prior_enabled;
        module.resident = transition.prior_resident;
        if (module.resident) module.last_use = ++clock_;
    }
    for (const auto & prior_budget : transaction->prior_budgets) {
        budgets_.at(prior_budget.first) = prior_budget.second;
    }
    policy_ = transaction->prior_policy;
    transaction->state = COMMON_TRANSIENT_POLICY_ROLLED_BACK;
    telemetry_.rollbacks++;
    return true;
}

bool common_transient_module_manager::finalize_policy(
        common_transient_policy_transaction * transaction,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    transient_telemetry_latency_scope latency{telemetry_.swap_latency_us};
    if (transaction == nullptr || active_policy_transaction_ != transaction ||
            transaction->owner != this || transaction->poisoned ||
            transaction->state != COMMON_TRANSIENT_POLICY_PUBLISHED) {
        fail_locked("invalid published transient policy transaction finalization", error);
        return false;
    }
    for (auto & transition : transaction->modules) {
        if (transition.handle &&
                !modules_.at(transition.id).desc.finalize_residency(
                    transition.handle.get(), error)) {
            transaction->poisoned = true;
            fail_locked("transient module owner finalization failed: " + transition.id, error);
            return false;
        }
    }
    transaction->state = COMMON_TRANSIENT_POLICY_FINALIZED;
    // Keep acquisitions excluded until free_policy() closes the finalized
    // handle. The server uses that interval to publish its logical plan and
    // capabilities under the same reader-visible commit boundary.
    return true;
}

bool common_transient_module_manager::free_policy(
        common_transient_policy_transaction * transaction,
        std::string * error) {
    if (transaction == nullptr) return true;
    if (transaction->owner != this || transaction->poisoned) {
        if (error != nullptr) *error = "cannot free a poisoned or foreign transient policy transaction";
        return false;
    }
    if (transaction->state == COMMON_TRANSIENT_POLICY_PUBLISHED &&
            !rollback_policy(transaction, error)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto & transition : transaction->modules) transition.handle.reset();
        if (active_policy_transaction_ == transaction) {
            active_policy_transaction_ = nullptr;
            if (transaction->state == COMMON_TRANSIENT_POLICY_FINALIZED) {
                policy_generation_++;
            }
        }
        transaction->owner = nullptr;
    }
    delete transaction;
    return true;
}

bool common_transient_module_manager::resolve_all_leases_for_shutdown(
        std::string * error) {
    while (true) {
        uint64_t lease_id = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_policy_transaction_ != nullptr) {
                fail_locked(
                    "cannot resolve request leases while policy reconfiguration is open",
                    error);
                return false;
            }
            if (leases_.empty()) {
                if (active_residency_epoch_ != nullptr) {
                    GGML_ABORT("transient residency epoch has no owning leases at shutdown\n");
                }
                return true;
            }
            lease_id = leases_.begin()->first;
        }
        if (!release(lease_id, false, error)) {
            return false;
        }
    }
}

void common_transient_module_manager::resume_locked(const std::vector<std::string> & affected) {
    for (auto it = affected.rbegin(); it != affected.rend(); ++it) {
        const auto module = modules_.find(*it);
        if (module != modules_.end() && module->second.desc.resume) {
            module->second.desc.resume();
        }
    }
}

std::unique_ptr<common_transient_module_manager::residency_transaction>
common_transient_module_manager::prepare_residency_locked(
        const std::vector<std::string> & evicted,
        const std::vector<std::string> & activated,
        std::string & error) {
    try {
        auto transaction = std::make_unique<residency_transaction>();
        transaction->modules.reserve(evicted.size() + activated.size());
        const auto prepare = [&](const std::string & id, bool target_resident) {
            auto & module = modules_.at(id);
            const auto & desc = module.desc;
            if (!desc.prepare_residency || !desc.publish_residency ||
                    !desc.rollback_residency || !desc.finalize_residency ||
                    !desc.free_residency) {
                error = "transient module lacks exact prepared residency callbacks: " + id;
                return false;
            }
            void * prepared = desc.prepare_residency(target_resident, &error);
            if (prepared == nullptr) {
                if (error.empty()) error = "failed to prepare transient module owner: " + id;
                return false;
            }
            residency_transaction::transition transition;
            transition.id = id;
            transition.prior_resident = module.resident;
            transition.target_resident = target_resident;
            try {
                transition.handle = std::shared_ptr<void>(
                    prepared,
                    [release = desc.free_residency](void * handle) {
                        release(handle);
                    });
            } catch (...) {
                desc.free_residency(prepared);
                throw;
            }
            transaction->modules.push_back(std::move(transition));
            return true;
        };
        for (const auto & id : evicted) {
            if (!prepare(id, false)) return nullptr;
        }
        for (const auto & id : activated) {
            if (!prepare(id, true)) return nullptr;
        }
        return transaction;
    } catch (const std::exception & exception) {
        error = std::string("failed to prepare exact transient owners: ") + exception.what();
        return nullptr;
    } catch (...) {
        error = "unexpected failure while preparing exact transient owners";
        return nullptr;
    }
}

bool common_transient_module_manager::publish_residency_locked(
        residency_transaction & transaction,
        std::string & error) {
    if (transaction.published) {
        error = "transient residency transaction is already published";
        return false;
    }
    size_t published = 0;
    for (; published < transaction.modules.size(); ++published) {
        auto & transition = transaction.modules[published];
        auto & desc = modules_.at(transition.id).desc;
        if (!desc.publish_residency(transition.handle.get(), &error)) {
            while (published > 0) {
                --published;
                auto & prior = transaction.modules[published];
                const bool rolled_back = modules_.at(prior.id).desc.rollback_residency(
                    prior.handle.get(), &error);
                if (!rolled_back) {
                    GGML_ABORT(
                        "partial transient publication could not restore exact prior owner: %s\n",
                        prior.id.c_str());
                }
                prior.published = false;
            }
            if (error.empty()) {
                error = "failed to publish transient module owner: " + transition.id;
            }
            return false;
        }
        transition.published = true;
    }
    for (const auto & transition : transaction.modules) {
        auto & module = modules_.at(transition.id);
        module.resident = transition.target_resident;
        if (module.resident) module.last_use = ++clock_;
        telemetry_.swaps++;
        telemetry_.bytes_moved += module.desc.bytes;
    }
    transaction.published = true;
    return true;
}

bool common_transient_module_manager::rollback_residency_locked(
        residency_transaction & transaction,
        std::string & error) {
    if (!transaction.published) {
        error = "transient residency transaction is not published";
        return false;
    }
    for (auto it = transaction.modules.rbegin(); it != transaction.modules.rend(); ++it) {
        if (!it->published) continue;
        if (!modules_.at(it->id).desc.rollback_residency(it->handle.get(), &error)) {
            GGML_ABORT(
                "published transient owner could not be restored exactly: %s\n",
                it->id.c_str());
        }
        it->published = false;
    }
    for (const auto & transition : transaction.modules) {
        auto & module = modules_.at(transition.id);
        module.resident = transition.prior_resident;
        if (module.resident) module.last_use = ++clock_;
        telemetry_.swaps++;
        telemetry_.bytes_moved += module.desc.bytes;
    }
    transaction.published = false;
    telemetry_.rollbacks++;
    return true;
}

bool common_transient_module_manager::finalize_residency_locked(
        residency_transaction & transaction,
        std::string & error) {
    if (!transaction.published) {
        error = "transient residency transaction is not published";
        return false;
    }
    for (auto & transition : transaction.modules) {
        if (!modules_.at(transition.id).desc.finalize_residency(
                transition.handle.get(), &error)) {
            if (error.empty()) error = "failed to finalize transient owner: " + transition.id;
            return false;
        }
        transition.published = false;
    }
    transaction.published = false;
    return true;
}

uint64_t common_transient_module_manager::acquire(
        const std::vector<std::string> & requested,
        bool restore_prior,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t started_us = ggml_time_us();
    telemetry_.transactions++;

    if (active_policy_transaction_ != nullptr) {
        fail_locked("transient module acquisition is blocked by policy reconfiguration", error);
        return 0;
    }

    std::set<std::string> requested_set;
    std::map<int, size_t> needed;
    for (const std::string & id : requested) {
        const auto it = modules_.find(id);
        if (it == modules_.end()) {
            fail_locked("unknown transient module: " + id, error);
            return 0;
        }
        if (!it->second.enabled) {
            fail_locked("transient module is disabled by the active policy: " + id, error);
            return 0;
        }
        if (requested_set.insert(id).second && !it->second.resident) {
            needed[it->second.desc.device] += it->second.desc.bytes;
        }
    }

    auto projected = resident_bytes_locked();
    std::vector<std::string> evicted;
    for (const auto & per_device : needed) {
        const int device = per_device.first;
        const size_t usable = usable_bytes_locked(device);
        if (per_device.second > usable) {
            telemetry_.oom_rejections++;
            fail_locked("requested transient set exceeds the device budget", error);
            return 0;
        }
        std::vector<std::string> candidates;
        for (const auto & pair : modules_) {
            if (pair.second.resident && pair.second.pins == 0 &&
                    pair.second.desc.device == device && requested_set.count(pair.first) == 0) {
                candidates.push_back(pair.first);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [&](const std::string & lhs, const std::string & rhs) {
            return modules_.at(lhs).last_use < modules_.at(rhs).last_use;
        });
        for (const std::string & id : candidates) {
            if (projected[device] + per_device.second <= usable) {
                break;
            }
            projected[device] -= modules_.at(id).desc.bytes;
            evicted.push_back(id);
        }
        if (projected[device] + per_device.second > usable) {
            telemetry_.oom_rejections++;
            fail_locked("insufficient evictable transient capacity on device " + std::to_string(device), error);
            return 0;
        }
        projected[device] += per_device.second;
    }

    std::vector<std::string> activated;
    std::vector<std::string> affected = evicted;
    for (const auto & id : requested_set) {
        if (!modules_.at(id).resident) {
            activated.push_back(id);
            affected.push_back(id);
        }
    }
    std::shared_ptr<residency_epoch> epoch = active_residency_epoch_;
    if (epoch != nullptr && !affected.empty()) {
        fail_locked(
            "transient owner swap is blocked until the active residency epoch resolves",
            error);
        telemetry_.swap_latency_us += ggml_time_us() - started_us;
        return 0;
    }
    if (epoch != nullptr) {
        const bool joins_epoch = std::any_of(
            requested_set.begin(), requested_set.end(),
            [&](const std::string & id) {
                return std::find(
                    epoch->activated.begin(), epoch->activated.end(), id) !=
                    epoch->activated.end();
            });
        if (!joins_epoch) epoch.reset();
    }

    std::vector<std::string> quiesced;
    try {
        quiesced.reserve(affected.size());
    } catch (const std::exception & exception) {
        fail_locked(
            std::string("failed to reserve transient quiesce bookkeeping: ") +
                exception.what(),
            error);
        telemetry_.swap_latency_us += ggml_time_us() - started_us;
        return 0;
    } catch (...) {
        fail_locked("failed to reserve transient quiesce bookkeeping", error);
        telemetry_.swap_latency_us += ggml_time_us() - started_us;
        return 0;
    }
    for (const std::string & id : affected) {
        auto & module = modules_.at(id);
        if (module.desc.quiesce && !module.desc.quiesce()) {
            resume_locked(quiesced);
            fail_locked("failed to quiesce transient module: " + id, error);
            telemetry_.swap_latency_us += ggml_time_us() - started_us;
            return 0;
        }
        quiesced.push_back(id);
    }

    std::string owner_error;
    if (!affected.empty()) {
        auto owners = prepare_residency_locked(evicted, activated, owner_error);
        if (owners == nullptr) {
            resume_locked(quiesced);
            fail_locked(owner_error.empty()
                ? "failed to prepare exact transient owners" : owner_error, error);
            telemetry_.swap_latency_us += ggml_time_us() - started_us;
            return 0;
        }
        try {
            epoch = std::make_shared<residency_epoch>();
            epoch->activated = std::move(activated);
            epoch->evicted = std::move(evicted);
            epoch->affected = std::move(affected);
            epoch->owners = std::move(owners);
        } catch (const std::exception & exception) {
            resume_locked(quiesced);
            fail_locked(
                std::string("failed to reserve transient residency epoch: ") +
                    exception.what(),
                error);
            telemetry_.swap_latency_us += ggml_time_us() - started_us;
            return 0;
        } catch (...) {
            resume_locked(quiesced);
            fail_locked("failed to reserve transient residency epoch", error);
            telemetry_.swap_latency_us += ggml_time_us() - started_us;
            return 0;
        }
    }

    uint64_t lease_id = 0;
    do {
        lease_id = next_lease_id_++;
    } while (lease_id == 0 || leases_.count(lease_id) != 0);

    // Complete every allocation that the lease needs before changing a live
    // owner. If insertion fails, destroying the prepared handles leaves the
    // physical and logical residency exactly as it was.
    std::map<uint64_t, lease_entry>::iterator inserted_lease;
    try {
        lease_entry lease;
        lease.requested.assign(requested_set.begin(), requested_set.end());
        lease.restore_prior = restore_prior;
        lease.epoch = epoch;
        const auto inserted = leases_.emplace(lease_id, std::move(lease));
        if (!inserted.second) {
            GGML_ABORT("transient lease id collision after uniqueness check\n");
        }
        inserted_lease = inserted.first;
    } catch (const std::exception & exception) {
        resume_locked(quiesced);
        fail_locked(
            std::string("failed to reserve transient lease bookkeeping: ") +
                exception.what(),
            error);
        telemetry_.swap_latency_us += ggml_time_us() - started_us;
        return 0;
    } catch (...) {
        resume_locked(quiesced);
        fail_locked("failed to reserve transient lease bookkeeping", error);
        telemetry_.swap_latency_us += ggml_time_us() - started_us;
        return 0;
    }

    const bool opens_epoch = epoch != nullptr && active_residency_epoch_ == nullptr;
    if (opens_epoch && !publish_residency_locked(*epoch->owners, owner_error)) {
        leases_.erase(inserted_lease);
        resume_locked(quiesced);
        fail_locked(owner_error.empty()
            ? "failed to publish exact transient owners" : owner_error, error);
        telemetry_.swap_latency_us += ggml_time_us() - started_us;
        return 0;
    }
    if (epoch != nullptr) {
        if (opens_epoch) active_residency_epoch_ = epoch;
        epoch->open_leases++;
    }
    for (const std::string & id : requested_set) {
        auto & module = modules_.at(id);
        module.last_use = ++clock_;
        module.pins++;
    }

    resume_locked(quiesced);
    telemetry_.swap_latency_us += ggml_time_us() - started_us;
    return lease_id;
}

bool common_transient_module_manager::release(uint64_t lease_id, bool success, std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t started_us = ggml_time_us();
    const auto lease_it = leases_.find(lease_id);
    if (lease_it == leases_.end()) {
        fail_locked("unknown transient module lease: " + std::to_string(lease_id), error);
        return false;
    }

    const auto & stored_lease = lease_it->second;
    for (const std::string & id : stored_lease.requested) {
        auto & module = modules_.at(id);
        if (module.pins == 0) {
            fail_locked("transient module pin underflow: " + id, error);
            return false;
        }
    }

    const auto epoch = stored_lease.epoch;
    if (epoch != nullptr &&
            (active_residency_epoch_ != epoch || epoch->open_leases == 0)) {
        GGML_ABORT("transient lease references an invalid residency epoch\n");
    }
    const bool resolves_epoch = epoch != nullptr && epoch->open_leases == 1;
    std::vector<std::string> quiesced;
    if (resolves_epoch) {
        try {
            quiesced.reserve(epoch->affected.size());
        } catch (const std::exception & exception) {
            fail_locked(
                std::string("failed to reserve transient release bookkeeping: ") +
                    exception.what(),
                error);
            return false;
        } catch (...) {
            fail_locked("failed to reserve transient release bookkeeping", error);
            return false;
        }
        for (const auto & id : epoch->affected) {
            auto & module = modules_.at(id);
            if (module.desc.quiesce && !module.desc.quiesce()) {
                resume_locked(quiesced);
                fail_locked(
                    "failed to quiesce transient module while resolving residency epoch: " + id,
                    error);
                return false;
            }
            quiesced.push_back(id);
        }
    }

    lease_entry lease = std::move(lease_it->second);
    leases_.erase(lease_it);
    for (const std::string & id : lease.requested) {
        auto & module = modules_.at(id);
        module.pins--;
    }

    if (epoch != nullptr) {
        epoch->committed = epoch->committed || (success && !lease.restore_prior);
        epoch->open_leases--;
        if (epoch->open_leases == 0) {
            std::string owner_error;
            if (epoch->committed) {
                if (!finalize_residency_locked(*epoch->owners, owner_error)) {
                    GGML_ABORT("transient owner finalization invariant failed: %s\n",
                        owner_error.c_str());
                }
            } else if (!rollback_residency_locked(*epoch->owners, owner_error)) {
                GGML_ABORT("transient owner rollback invariant failed: %s\n",
                    owner_error.c_str());
            }
            active_residency_epoch_.reset();
            resume_locked(quiesced);
        }
    }
    telemetry_.swap_latency_us += ggml_time_us() - started_us;
    if (!success) {
        fail_locked("transient module transaction body failed", error);
        // The body outcome selects rollback semantics; it is not a failure of
        // lease retirement itself. Callers can report the body error while a
        // true return confirms that pins and exact owners were resolved.
        return true;
    }
    return true;
}

bool common_transient_module_manager::run(
        const std::vector<std::string> & requested,
        const std::function<bool()> & body,
        bool restore_prior,
        std::string * error) {
    const uint64_t lease_id = acquire(requested, restore_prior, error);
    if (lease_id == 0) {
        return false;
    }
    bool body_ok = false;
    try {
        body_ok = !body || body();
    } catch (const std::exception & exception) {
        // Retire the lease before formatting or propagating anything from the
        // exception. In particular, std::bad_alloc must not strand pins or a
        // published off-side owner.
        std::string release_error;
        if (!release(lease_id, false, &release_error)) {
            GGML_ABORT("transient run could not retire its hidden lease: %s\n",
                release_error.c_str());
        }
        if (error != nullptr) {
            try {
                *error = std::string("transient module transaction body threw: ") +
                    exception.what();
            } catch (...) {
                error->clear();
            }
        }
        return false;
    } catch (...) {
        std::string release_error;
        if (!release(lease_id, false, &release_error)) {
            GGML_ABORT("transient run could not retire its hidden lease: %s\n",
                release_error.c_str());
        }
        if (error != nullptr) {
            try {
                *error = "transient module transaction body threw an unknown exception";
            } catch (...) {
                error->clear();
            }
        }
        return false;
    }
    std::string release_error;
    const bool release_ok = release(lease_id, body_ok, &release_error);
    if (!release_ok) {
        GGML_ABORT("transient run could not retire its hidden lease: %s\n",
            release_error.c_str());
    }
    return body_ok;
}

common_transient_module_telemetry common_transient_module_manager::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = telemetry_;
    result.modules.clear();
    result.resident_bytes = resident_bytes_locked();
    result.budgets = budgets_;
    result.policy = policy_;
    result.active_leases = leases_.size();
    result.pending_restores = active_residency_epoch_ != nullptr &&
            !active_residency_epoch_->committed
        ? 1 : 0;
    result.reconfiguration_open = active_policy_transaction_ != nullptr;
    for (const auto & pair : modules_) {
        result.modules.push_back({
            pair.first,
            pair.second.desc.kind,
            pair.second.desc.device,
            pair.second.desc.bytes,
            pair.second.enabled,
            pair.second.resident,
            pair.second.pins,
        });
    }
    return result;
}

bool common_transient_module_manager::module_enabled(const std::string & id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto module = modules_.find(id);
    if (module == modules_.end()) return false;
    if (active_policy_transaction_ != nullptr) {
        const auto transition = std::find_if(
            active_policy_transaction_->modules.begin(),
            active_policy_transaction_->modules.end(),
            [&id](const auto & item) { return item.id == id; });
        if (transition != active_policy_transaction_->modules.end()) {
            return transition->prior_enabled;
        }
    }
    return module->second.enabled;
}
