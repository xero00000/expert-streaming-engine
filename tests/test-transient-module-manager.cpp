#include "transient-module-manager.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ \
                  << ": " #condition << '\n'; \
        std::abort(); \
    } \
} while (false)

struct fake_module {
    bool resident = false;
    bool fail_activate = false;
    bool fail_deactivate = false;
    int state_generation = 0;
    int stale_entries = 0;
    int quiesces = 0;
    int resumes = 0;
    bool fail_quiesce = false;
    bool fail_policy_publish = false;
    bool fail_policy_rollback = false;
    bool fail_policy_finalize = false;
};

struct fake_residency_transaction {
    fake_module * module = nullptr;
    bool offside_resident = false;
    int offside_generation = 0;
    int offside_stale_entries = 0;
    bool published = false;
};

static common_transient_module_desc describe(
        const std::string & id,
        common_transient_module_kind kind,
        int device,
        size_t bytes,
        fake_module & fake) {
    common_transient_module_desc desc;
    desc.id = id;
    desc.kind = kind;
    desc.device = device;
    desc.bytes = bytes;
    desc.initially_resident = fake.resident;
    desc.streams = {id + "-stream"};
    desc.quiesce = [&fake]() {
        if (fake.fail_quiesce) return false;
        fake.quiesces++;
        return true;
    };
    desc.resume = [&fake]() { fake.resumes++; };
    desc.activate = [&fake]() {
        if (fake.fail_activate) {
            return false;
        }
        CHECK(!fake.resident);
        fake.resident = true;
        fake.state_generation++;
        fake.stale_entries = 0;
        return true;
    };
    desc.deactivate = [&fake]() {
        if (fake.fail_deactivate) {
            return false;
        }
        CHECK(fake.resident);
        fake.resident = false;
        return true;
    };
    desc.prepare_residency = [&fake](bool target_resident, std::string *) -> void * {
        return new fake_residency_transaction{
            &fake,
            target_resident,
            target_resident ? fake.state_generation + 1 : 0,
            0,
            false,
        };
    };
    desc.publish_residency = [](void * opaque, std::string *) {
        auto * transaction = static_cast<fake_residency_transaction *>(opaque);
        if (transaction->module->fail_policy_publish) return false;
        std::swap(transaction->module->resident, transaction->offside_resident);
        std::swap(transaction->module->state_generation, transaction->offside_generation);
        std::swap(transaction->module->stale_entries, transaction->offside_stale_entries);
        transaction->published = true;
        return true;
    };
    desc.rollback_residency = [](void * opaque, std::string *) {
        auto * transaction = static_cast<fake_residency_transaction *>(opaque);
        if (transaction->module->fail_policy_rollback) return false;
        std::swap(transaction->module->resident, transaction->offside_resident);
        std::swap(transaction->module->state_generation, transaction->offside_generation);
        std::swap(transaction->module->stale_entries, transaction->offside_stale_entries);
        transaction->published = false;
        return true;
    };
    desc.finalize_residency = [](void * opaque, std::string *) {
        auto * transaction = static_cast<fake_residency_transaction *>(opaque);
        if (transaction->module->fail_policy_finalize) return false;
        transaction->published = false;
        transaction->offside_resident = false;
        transaction->offside_generation = 0;
        transaction->offside_stale_entries = 0;
        return true;
    };
    desc.free_residency = [](void * opaque) {
        delete static_cast<fake_residency_transaction *>(opaque);
    };
    return desc;
}

static bool is_resident(const common_transient_module_telemetry & telemetry, const std::string & id) {
    for (const auto & module : telemetry.modules) {
        if (module.id == id) {
            return module.resident;
        }
    }
    CHECK(false);
    return false;
}

static bool is_enabled(const common_transient_module_telemetry & telemetry, const std::string & id) {
    for (const auto & module : telemetry.modules) {
        if (module.id == id) {
            return module.enabled;
        }
    }
    CHECK(false);
    return false;
}

static size_t usable_budget(
        const common_transient_module_telemetry & telemetry,
        int device) {
    const auto & budget = telemetry.budgets.at(device);
    CHECK(budget.capacity_bytes >= budget.safety_margin_bytes);
    return budget.capacity_bytes - budget.safety_margin_bytes;
}

static void test_prepared_policy_transactions() {
    common_transient_module_manager manager({{0, {1000, 100}}});
    fake_module mtp{true};
    fake_module vision;
    CHECK(manager.register_module(describe(
        "mtp", COMMON_TRANSIENT_MODULE_MTP, 0, 600, mtp)));
    CHECK(manager.register_module(describe(
        "multimodal", COMMON_TRANSIENT_MODULE_VISION, 0, 700, vision)));

    std::string error;
    const auto telemetry_initial = manager.snapshot();
    auto * prepared = manager.prepare_policy(
        COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY, &error);
    CHECK(prepared != nullptr);
    auto before_publish = manager.snapshot();
    CHECK(before_publish.policy == COMMON_TRANSIENT_POLICY_SHARED);
    CHECK(before_publish.reconfiguration_open);
    CHECK(is_resident(before_publish, "mtp"));
    CHECK(!is_resident(before_publish, "multimodal"));
    CHECK(before_publish.transactions == telemetry_initial.transactions + 1);
    CHECK(before_publish.swaps == telemetry_initial.swaps);
    CHECK(before_publish.bytes_moved == telemetry_initial.bytes_moved);
    CHECK(manager.acquire({"mtp"}, false, &error) == 0);

    CHECK(manager.publish_policy(prepared, &error));
    auto published = manager.snapshot();
    CHECK(published.policy == COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY);
    CHECK(!is_enabled(published, "mtp"));
    CHECK(is_enabled(published, "multimodal"));
    CHECK(!is_resident(published, "mtp"));
    CHECK(is_resident(published, "multimodal"));
    CHECK(usable_budget(published, 0) == 700);
    CHECK(published.swaps == telemetry_initial.swaps + 2);
    CHECK(published.bytes_moved == telemetry_initial.bytes_moved + 1300);

    // Freeing a published but unfinalized handle restores exact owners,
    // feature availability, and capacity.
    CHECK(manager.free_policy(prepared, &error));
    auto restored = manager.snapshot();
    CHECK(restored.policy == COMMON_TRANSIENT_POLICY_SHARED);
    CHECK(!restored.reconfiguration_open);
    CHECK(is_enabled(restored, "mtp"));
    CHECK(is_enabled(restored, "multimodal"));
    CHECK(is_resident(restored, "mtp"));
    CHECK(!is_resident(restored, "multimodal"));
    CHECK(usable_budget(restored, 0) == 900);
    CHECK(restored.swaps == telemetry_initial.swaps + 4);
    CHECK(restored.bytes_moved == telemetry_initial.bytes_moved + 2600);
    CHECK(restored.rollbacks == telemetry_initial.rollbacks + 1);

    // A later module publication failure reverses every earlier owner swap and
    // leaves the same prepared handle retryable.
    auto * retryable = manager.prepare_policy(
        COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY, &error);
    CHECK(retryable != nullptr);
    vision.fail_policy_publish = true;
    CHECK(!manager.publish_policy(retryable, &error));
    CHECK(mtp.resident && !vision.resident);
    CHECK(manager.snapshot().policy == COMMON_TRANSIENT_POLICY_SHARED);
    vision.fail_policy_publish = false;
    CHECK(manager.publish_policy(retryable, &error));

    // Rollback failure is itself failure-atomic: the published owners remain
    // active, and rollback can retry after the backend recovers.
    vision.fail_policy_rollback = true;
    const auto before_injected_rollback = manager.snapshot();
    CHECK(!manager.rollback_policy(retryable, &error));
    CHECK(!mtp.resident && vision.resident);
    CHECK(manager.snapshot().policy == COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY);
    CHECK(manager.snapshot().swaps == before_injected_rollback.swaps);
    CHECK(manager.snapshot().bytes_moved == before_injected_rollback.bytes_moved);
    vision.fail_policy_rollback = false;
    CHECK(manager.rollback_policy(retryable, &error));
    const auto after_injected_rollback = manager.snapshot();
    CHECK(after_injected_rollback.swaps == before_injected_rollback.swaps + 2);
    CHECK(after_injected_rollback.bytes_moved ==
        before_injected_rollback.bytes_moved + 1300);
    CHECK(after_injected_rollback.rollbacks ==
        before_injected_rollback.rollbacks + 1);
    CHECK(manager.free_policy(retryable, &error));
    CHECK(mtp.resident && !vision.resident);

    auto * committed = manager.prepare_policy(
        COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY, &error);
    CHECK(committed != nullptr);
    CHECK(manager.publish_policy(committed, &error));
    CHECK(manager.finalize_policy(committed, &error));
    CHECK(manager.snapshot().reconfiguration_open);
    CHECK(manager.acquire({"multimodal"}, false, &error) == 0);
    CHECK(manager.free_policy(committed, &error));
    CHECK(manager.snapshot().policy == COMMON_TRANSIENT_POLICY_MULTIMODAL_ONLY);
    CHECK(manager.acquire({"mtp"}, false, &error) == 0);

    auto * disabled = manager.prepare_policy(COMMON_TRANSIENT_POLICY_OFF, &error);
    CHECK(disabled != nullptr);
    CHECK(manager.publish_policy(disabled, &error));
    CHECK(manager.finalize_policy(disabled, &error));
    CHECK(manager.free_policy(disabled, &error));
    auto off = manager.snapshot();
    CHECK(off.policy == COMMON_TRANSIENT_POLICY_OFF);
    CHECK(usable_budget(off, 0) == 0);
    CHECK(!is_resident(off, "mtp") && !is_resident(off, "multimodal"));

    auto * shared = manager.prepare_policy(COMMON_TRANSIENT_POLICY_SHARED, &error);
    CHECK(shared != nullptr);
    CHECK(manager.publish_policy(shared, &error));
    CHECK(manager.finalize_policy(shared, &error));
    CHECK(manager.free_policy(shared, &error));
    auto shared_state = manager.snapshot();
    CHECK(shared_state.policy == COMMON_TRANSIENT_POLICY_SHARED);
    CHECK(is_resident(shared_state, "mtp"));
    CHECK(!is_resident(shared_state, "multimodal"));
    CHECK(usable_budget(shared_state, 0) == 700);

    const uint64_t pinned = manager.acquire({"mtp"}, false, &error);
    CHECK(pinned != 0);
    CHECK(manager.prepare_policy(COMMON_TRANSIENT_POLICY_OFF, &error) == nullptr);
    CHECK(manager.release(pinned));
}

int main() {
    common_transient_module_manager manager({
        {0, {1000, 100}},
        {1, {700, 100}},
    });

    fake_module mtp{true};
    fake_module vision;
    fake_module audio;
    fake_module lora;

    CHECK(manager.register_module(describe("mtp", COMMON_TRANSIENT_MODULE_MTP, 0, 600, mtp)));
    CHECK(manager.register_module(describe("vision", COMMON_TRANSIENT_MODULE_VISION, 0, 700, vision)));
    CHECK(manager.register_module(describe("audio", COMMON_TRANSIENT_MODULE_AUDIO, 1, 400, audio)));
    CHECK(manager.register_module(describe("lora", COMMON_TRANSIENT_MODULE_LORA, 1, 300, lora)));

    bool saw_vision = false;
    CHECK(manager.run({"vision"}, [&]() {
        saw_vision = vision.resident && !mtp.resident;
        return true;
    }));
    CHECK(saw_vision);
    CHECK(mtp.resident && !vision.resident);
    CHECK(mtp.stale_entries == 0);

    // Model an image request leaving module-private cache entries, followed by
    // text/MTP. Exact rollback restores the original MTP generation and cache,
    // rather than reconstructing a merely equivalent owner.
    const uint64_t image_request = manager.acquire({"vision"});
    CHECK(image_request != 0);
    vision.stale_entries = 7;
    const int mtp_generation_before_restore = mtp.state_generation;
    CHECK(manager.release(image_request));
    CHECK(mtp.resident && !vision.resident);
    CHECK(mtp.state_generation == mtp_generation_before_restore);
    CHECK(mtp.stale_entries == 0);

    // Two asynchronous slots share one resident module. Releasing the first
    // slot must not restore/evict anything still pinned by the second.
    const uint64_t slot_a = manager.acquire({"vision"});
    const uint64_t slot_b = manager.acquire({"vision"});
    CHECK(slot_a != 0 && slot_b != 0);
    auto overlapping = manager.snapshot();
    CHECK(vision.resident && !mtp.resident);
    for (const auto & module : overlapping.modules) {
        if (module.id == "vision") CHECK(module.pins == 2);
    }
    CHECK(manager.release(slot_a));
    CHECK(vision.resident && !mtp.resident);
    CHECK(manager.release(slot_b));
    CHECK(mtp.resident && !vision.resident);

    // Overlapping consumers of one owner swap form one residency epoch. A
    // successful request commits that epoch even if the request which opened
    // it later fails, preventing restoration of an MTP owner whose cache no
    // longer matches the target context changed by the successful request.
    const uint64_t image_owner = manager.acquire({"vision"}, false);
    const uint64_t image_overlap = manager.acquire({"vision"}, false);
    CHECK(image_owner != 0 && image_overlap != 0);
    CHECK(manager.release(image_overlap, true));
    CHECK(vision.resident && !mtp.resident);
    CHECK(manager.release(image_owner, false));
    CHECK(vision.resident && !mtp.resident);
    auto committed_overlap = manager.snapshot();
    CHECK(committed_overlap.active_leases == 0);
    CHECK(committed_overlap.pending_restores == 0);

    // A later text request starts a new epoch and can safely construct/commit
    // MTP against the target state left by the image request.
    const uint64_t text_after_image = manager.acquire({"mtp"}, false);
    CHECK(text_after_image != 0);
    CHECK(manager.release(text_after_image, true));
    CHECK(mtp.resident && !vision.resident);

    // Shutdown may race an HTTP guard that owns a published request epoch but
    // never handed its lease into a queued task. The owner-thread drain must
    // resolve the raw manager lease and restore exact prior ownership.
    const uint64_t abandoned_image = manager.acquire({"vision"}, false);
    CHECK(abandoned_image != 0);
    CHECK(vision.resident && !mtp.resident);
    CHECK(manager.resolve_all_leases_for_shutdown());
    CHECK(mtp.resident && !vision.resident);
    auto drained = manager.snapshot();
    CHECK(drained.active_leases == 0);
    CHECK(drained.pending_restores == 0);

    bool saw_two_devices = false;
    CHECK(manager.run({"vision", "audio"}, [&]() {
        saw_two_devices = vision.resident && audio.resident && !mtp.resident;
        return true;
    }));
    CHECK(saw_two_devices);
    CHECK(mtp.resident && !vision.resident && !audio.resident);

    vision.fail_policy_publish = true;
    std::string error;
    CHECK(!manager.run({"vision"}, []() { return true; }, true, &error));
    CHECK(error.find("publish") != std::string::npos);
    CHECK(mtp.resident && !vision.resident);
    vision.fail_policy_publish = false;

    mtp.fail_policy_publish = true;
    error.clear();
    CHECK(!manager.run({"vision"}, []() { return true; }, true, &error));
    CHECK(error.find("publish") != std::string::npos);
    CHECK(mtp.resident && !vision.resident);
    mtp.fail_policy_publish = false;

    // A release that cannot reach its quiescent safe point leaves its lease,
    // pins, and owner epoch untouched so the same release can be retried. Once a
    // published inverse swap starts, its rollback callback is an invariant and
    // cannot return a recoverable failure without exposing mixed owners.
    const uint64_t restore_failure = manager.acquire({"vision"});
    CHECK(restore_failure != 0);
    vision.fail_quiesce = true;
    CHECK(!manager.release(restore_failure));
    CHECK(vision.resident && !mtp.resident);
    vision.fail_quiesce = false;
    CHECK(manager.release(restore_failure));
    CHECK(mtp.resident && !vision.resident);

    error.clear();
    CHECK(!manager.run({"audio", "lora"}, []() { return true; }, true, &error));
    CHECK(error.find("exceeds the device budget") != std::string::npos);
    CHECK(!audio.resident && !lora.resident);

    CHECK(!manager.run({"vision"}, []() { return false; }));
    CHECK(mtp.resident && !vision.resident);

    error.clear();
    CHECK(!manager.run({"vision"}, []() -> bool {
        throw std::runtime_error("injected body exception");
    }, true, &error));
    CHECK(error.find("body threw") != std::string::npos);
    CHECK(mtp.resident && !vision.resident);
    CHECK(manager.snapshot().active_leases == 0);

    const auto telemetry = manager.snapshot();
    CHECK(is_resident(telemetry, "mtp"));
    CHECK(!is_resident(telemetry, "vision"));
    CHECK(telemetry.resident_bytes.at(0) == 600);
    CHECK(telemetry.transactions == 15);
    CHECK(telemetry.rollbacks >= 3);
    CHECK(telemetry.failures >= 4);
    CHECK(telemetry.oom_rejections == 1);
    CHECK(telemetry.swaps > 0);
    CHECK(telemetry.bytes_moved > 0);
    CHECK(mtp.quiesces == mtp.resumes);
    CHECK(vision.quiesces == vision.resumes);

    test_prepared_policy_transactions();

    std::cout << "PASS: transient module and policy transactions are bounded and failure-atomic\n";
    return 0;
}
