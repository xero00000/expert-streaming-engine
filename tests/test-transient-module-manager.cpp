#include "transient-module-manager.h"

#include <cassert>
#include <iostream>
#include <string>

struct fake_module {
    bool resident = false;
    bool fail_activate = false;
    bool fail_deactivate = false;
    int state_generation = 0;
    int stale_entries = 0;
    int quiesces = 0;
    int resumes = 0;
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
    desc.quiesce = [&fake]() { fake.quiesces++; return true; };
    desc.resume = [&fake]() { fake.resumes++; };
    desc.activate = [&fake]() {
        if (fake.fail_activate) {
            return false;
        }
        assert(!fake.resident);
        fake.resident = true;
        fake.state_generation++;
        fake.stale_entries = 0;
        return true;
    };
    desc.deactivate = [&fake]() {
        if (fake.fail_deactivate) {
            return false;
        }
        assert(fake.resident);
        fake.resident = false;
        return true;
    };
    return desc;
}

static bool is_resident(const common_transient_module_telemetry & telemetry, const std::string & id) {
    for (const auto & module : telemetry.modules) {
        if (module.id == id) {
            return module.resident;
        }
    }
    assert(false);
    return false;
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

    assert(manager.register_module(describe("mtp", COMMON_TRANSIENT_MODULE_MTP, 0, 600, mtp)));
    assert(manager.register_module(describe("vision", COMMON_TRANSIENT_MODULE_VISION, 0, 700, vision)));
    assert(manager.register_module(describe("audio", COMMON_TRANSIENT_MODULE_AUDIO, 1, 400, audio)));
    assert(manager.register_module(describe("lora", COMMON_TRANSIENT_MODULE_LORA, 1, 300, lora)));

    bool saw_vision = false;
    assert(manager.run({"vision"}, [&]() {
        saw_vision = vision.resident && !mtp.resident;
        return true;
    }));
    assert(saw_vision);
    assert(mtp.resident && !vision.resident);
    assert(mtp.stale_entries == 0);

    // Model an image request leaving module-private cache entries, followed by
    // text/MTP. Reconstruction must start a new generation with no stale state.
    const uint64_t image_request = manager.acquire({"vision"});
    assert(image_request != 0);
    vision.stale_entries = 7;
    const int mtp_generation_before_restore = mtp.state_generation;
    assert(manager.release(image_request));
    assert(mtp.resident && !vision.resident);
    assert(mtp.state_generation == mtp_generation_before_restore + 1);
    assert(mtp.stale_entries == 0);

    // Two asynchronous slots share one resident module. Releasing the first
    // slot must not restore/evict anything still pinned by the second.
    const uint64_t slot_a = manager.acquire({"vision"});
    const uint64_t slot_b = manager.acquire({"vision"});
    assert(slot_a != 0 && slot_b != 0);
    auto overlapping = manager.snapshot();
    assert(vision.resident && !mtp.resident);
    for (const auto & module : overlapping.modules) {
        if (module.id == "vision") assert(module.pins == 2);
    }
    assert(manager.release(slot_a));
    assert(vision.resident && !mtp.resident);
    assert(manager.release(slot_b));
    assert(mtp.resident && !vision.resident);

    bool saw_two_devices = false;
    assert(manager.run({"vision", "audio"}, [&]() {
        saw_two_devices = vision.resident && audio.resident && !mtp.resident;
        return true;
    }));
    assert(saw_two_devices);
    assert(mtp.resident && !vision.resident && !audio.resident);

    vision.fail_activate = true;
    std::string error;
    assert(!manager.run({"vision"}, []() { return true; }, true, &error));
    assert(error.find("failed to activate") != std::string::npos);
    assert(mtp.resident && !vision.resident);
    vision.fail_activate = false;

    mtp.fail_deactivate = true;
    error.clear();
    assert(!manager.run({"vision"}, []() { return true; }, true, &error));
    assert(error.find("failed to deactivate") != std::string::npos);
    assert(mtp.resident && !vision.resident);
    mtp.fail_deactivate = false;

    // A failed restore is itself failure-atomic and remains queued. A later
    // release retries it after the backend failure clears.
    const uint64_t restore_failure = manager.acquire({"vision"});
    assert(restore_failure != 0);
    mtp.fail_activate = true;
    assert(!manager.release(restore_failure));
    assert(vision.resident && !mtp.resident);
    mtp.fail_activate = false;
    const uint64_t retry = manager.acquire({"vision"});
    assert(retry != 0);
    assert(manager.release(retry));
    assert(mtp.resident && !vision.resident);

    error.clear();
    assert(!manager.run({"audio", "lora"}, []() { return true; }, true, &error));
    assert(error.find("exceeds the device budget") != std::string::npos);
    assert(!audio.resident && !lora.resident);

    assert(!manager.run({"vision"}, []() { return false; }));
    assert(mtp.resident && !vision.resident);

    const auto telemetry = manager.snapshot();
    assert(is_resident(telemetry, "mtp"));
    assert(!is_resident(telemetry, "vision"));
    assert(telemetry.resident_bytes.at(0) == 600);
    assert(telemetry.transactions == 11);
    assert(telemetry.rollbacks >= 3);
    assert(telemetry.failures >= 4);
    assert(telemetry.oom_rejections == 1);
    assert(telemetry.swaps > 0);
    assert(telemetry.bytes_moved > 0);
    assert(mtp.quiesces == mtp.resumes);
    assert(vision.quiesces == vision.resumes);

    std::cout << "PASS: transient module transactions are bounded and failure-atomic\n";
    return 0;
}
