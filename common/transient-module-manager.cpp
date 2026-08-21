#include "transient-module-manager.h"

#include "ggml.h"

#include <algorithm>
#include <set>

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
    return true;
}

void common_transient_module_manager::resume_locked(const std::vector<std::string> & affected) {
    for (auto it = affected.rbegin(); it != affected.rend(); ++it) {
        const auto module = modules_.find(*it);
        if (module != modules_.end() && module->second.desc.resume) {
            module->second.desc.resume();
        }
    }
}

bool common_transient_module_manager::restore_locked(
        const std::vector<std::string> & activated,
        const std::vector<std::string> & evicted,
        std::string & error) {
    bool ok = true;
    for (auto it = activated.rbegin(); it != activated.rend(); ++it) {
        auto & module = modules_.at(*it);
        if (module.resident && module.desc.deactivate && module.desc.deactivate()) {
            module.resident = false;
            telemetry_.swaps++;
            telemetry_.bytes_moved += module.desc.bytes;
        } else if (module.resident) {
            ok = false;
            error += (error.empty() ? "" : "; ") + std::string("failed to deactivate ") + *it + " during rollback";
        }
    }
    for (auto it = evicted.rbegin(); it != evicted.rend(); ++it) {
        auto & module = modules_.at(*it);
        if (!module.resident && module.desc.activate && module.desc.activate()) {
            module.resident = true;
            module.last_use = ++clock_;
            telemetry_.swaps++;
            telemetry_.bytes_moved += module.desc.bytes;
        } else if (!module.resident) {
            ok = false;
            error += (error.empty() ? "" : "; ") + std::string("failed to restore ") + *it;
        }
    }
    telemetry_.rollbacks++;
    return ok;
}

uint64_t common_transient_module_manager::acquire(
        const std::vector<std::string> & requested,
        bool restore_prior,
        std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t started_us = ggml_time_us();
    telemetry_.transactions++;

    std::set<std::string> requested_set;
    std::map<int, size_t> needed;
    for (const std::string & id : requested) {
        const auto it = modules_.find(id);
        if (it == modules_.end()) {
            fail_locked("unknown transient module: " + id, error);
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
            affected.push_back(id);
        }
    }
    std::vector<std::string> quiesced;
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

    std::string rollback_error;
    for (const std::string & id : evicted) {
        auto & module = modules_.at(id);
        if (!module.desc.deactivate || !module.desc.deactivate()) {
            restore_locked(activated, evicted, rollback_error);
            resume_locked(quiesced);
            fail_locked("failed to deactivate transient module: " + id +
                (rollback_error.empty() ? "" : "; " + rollback_error), error);
            telemetry_.swap_latency_us += ggml_time_us() - started_us;
            return 0;
        }
        module.resident = false;
        telemetry_.swaps++;
        telemetry_.bytes_moved += module.desc.bytes;
    }
    for (const std::string & id : requested_set) {
        auto & module = modules_.at(id);
        if (!module.resident) {
            if (!module.desc.activate || !module.desc.activate()) {
                restore_locked(activated, evicted, rollback_error);
                resume_locked(quiesced);
                fail_locked("failed to activate transient module: " + id +
                    (rollback_error.empty() ? "" : "; " + rollback_error), error);
                telemetry_.swap_latency_us += ggml_time_us() - started_us;
                return 0;
            }
            module.resident = true;
            activated.push_back(id);
            telemetry_.swaps++;
            telemetry_.bytes_moved += module.desc.bytes;
        }
        module.last_use = ++clock_;
        module.pins++;
    }

    resume_locked(quiesced);
    telemetry_.swap_latency_us += ggml_time_us() - started_us;

    uint64_t lease_id = next_lease_id_++;
    if (lease_id == 0) {
        lease_id = next_lease_id_++;
    }
    leases_.emplace(lease_id, lease_entry{
        std::vector<std::string>(requested_set.begin(), requested_set.end()),
        std::move(activated),
        std::move(evicted),
        restore_prior,
    });
    return lease_id;
}

bool common_transient_module_manager::apply_pending_restores_locked(std::string & error) {
    bool all_ok = true;
    for (size_t reverse = 0; reverse < pending_restores_.size();) {
        const size_t index = pending_restores_.size() - 1 - reverse;
        auto & plan = pending_restores_[index];
        const bool pinned = std::any_of(plan.activated.begin(), plan.activated.end(), [&](const std::string & id) {
            return modules_.at(id).pins != 0;
        });
        if (pinned) {
            reverse++;
            continue;
        }

        std::vector<std::string> affected = plan.activated;
        affected.insert(affected.end(), plan.evicted.begin(), plan.evicted.end());
        std::vector<std::string> quiesced;
        bool quiesce_ok = true;
        for (const std::string & id : affected) {
            auto & module = modules_.at(id);
            if (module.desc.quiesce && !module.desc.quiesce()) {
                error += (error.empty() ? "" : "; ") + std::string("failed to quiesce ") + id + " during restore";
                quiesce_ok = false;
                break;
            }
            quiesced.push_back(id);
        }
        if (!quiesce_ok) {
            resume_locked(quiesced);
            all_ok = false;
            reverse++;
            continue;
        }

        std::vector<std::string> deactivated;
        std::vector<std::string> restored;
        bool restore_ok = true;
        for (auto it = plan.activated.rbegin(); it != plan.activated.rend(); ++it) {
            auto & module = modules_.at(*it);
            if (!module.resident) {
                continue;
            }
            if (!module.desc.deactivate || !module.desc.deactivate()) {
                error += (error.empty() ? "" : "; ") + std::string("failed to deactivate ") + *it + " during restore";
                restore_ok = false;
                break;
            }
            module.resident = false;
            deactivated.push_back(*it);
            telemetry_.swaps++;
            telemetry_.bytes_moved += module.desc.bytes;
        }
        if (restore_ok) {
            for (auto it = plan.evicted.rbegin(); it != plan.evicted.rend(); ++it) {
                auto & module = modules_.at(*it);
                if (module.resident) {
                    continue;
                }
                if (!module.desc.activate || !module.desc.activate()) {
                    error += (error.empty() ? "" : "; ") + std::string("failed to reactivate ") + *it + " during restore";
                    restore_ok = false;
                    break;
                }
                module.resident = true;
                module.last_use = ++clock_;
                restored.push_back(*it);
                telemetry_.swaps++;
                telemetry_.bytes_moved += module.desc.bytes;
            }
        }

        if (!restore_ok) {
            // Failure-atomic restore: return to the residency that existed before
            // this restore attempt, then leave the plan queued for a later retry.
            for (auto it = restored.rbegin(); it != restored.rend(); ++it) {
                auto & module = modules_.at(*it);
                if (module.resident && module.desc.deactivate && module.desc.deactivate()) {
                    module.resident = false;
                    telemetry_.swaps++;
                    telemetry_.bytes_moved += module.desc.bytes;
                }
            }
            for (auto it = deactivated.rbegin(); it != deactivated.rend(); ++it) {
                auto & module = modules_.at(*it);
                if (!module.resident && module.desc.activate && module.desc.activate()) {
                    module.resident = true;
                    module.last_use = ++clock_;
                    telemetry_.swaps++;
                    telemetry_.bytes_moved += module.desc.bytes;
                }
            }
            telemetry_.rollbacks++;
            all_ok = false;
            resume_locked(quiesced);
            reverse++;
            continue;
        }

        if (plan.rollback) {
            telemetry_.rollbacks++;
        }
        resume_locked(quiesced);
        pending_restores_.erase(pending_restores_.begin() + index);
        // Start again from the newest plan because this restoration can unblock it.
        reverse = 0;
    }
    return all_ok;
}

bool common_transient_module_manager::release(uint64_t lease_id, bool success, std::string * error) {
    std::lock_guard<std::mutex> lock(mutex_);
    const int64_t started_us = ggml_time_us();
    const auto lease_it = leases_.find(lease_id);
    if (lease_it == leases_.end()) {
        fail_locked("unknown transient module lease: " + std::to_string(lease_id), error);
        return false;
    }

    lease_entry lease = std::move(lease_it->second);
    leases_.erase(lease_it);
    for (const std::string & id : lease.requested) {
        auto & module = modules_.at(id);
        if (module.pins == 0) {
            fail_locked("transient module pin underflow: " + id, error);
            return false;
        }
        module.pins--;
    }
    if ((lease.restore_prior || !success) && (!lease.activated.empty() || !lease.evicted.empty())) {
        pending_restores_.push_back({
            std::move(lease.activated),
            std::move(lease.evicted),
            !success,
        });
    }

    std::string restore_error;
    const bool restored = apply_pending_restores_locked(restore_error);
    telemetry_.swap_latency_us += ggml_time_us() - started_us;
    if (!restored) {
        fail_locked("transient module restore failed" +
            (restore_error.empty() ? "" : "; " + restore_error), error);
        return false;
    }
    if (!success) {
        fail_locked("transient module transaction body failed", error);
        return false;
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
    const bool body_ok = !body || body();
    std::string release_error;
    const bool release_ok = release(lease_id, body_ok, &release_error);
    if (!release_ok && error != nullptr && error->empty()) {
        *error = body_ok ? release_error : "transient module transaction body failed";
    }
    return body_ok && release_ok;
}

common_transient_module_telemetry common_transient_module_manager::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto result = telemetry_;
    result.modules.clear();
    result.resident_bytes = resident_bytes_locked();
    for (const auto & pair : modules_) {
        result.modules.push_back({
            pair.first,
            pair.second.desc.kind,
            pair.second.desc.device,
            pair.second.desc.bytes,
            pair.second.resident,
            pair.second.pins,
        });
    }
    return result;
}
