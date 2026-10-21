#include "llama-expert-cache.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ese.h"
#include "../ggml/src/ggml-expert-loader.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#include <fcntl.h>
#endif

using namespace llama_expert_cache;

namespace {

#define REQUIRE(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string("requirement failed: ") + #condition); \
} while (false)

template<class Exception, class Function>
void require_throws(Function && function) {
    bool threw = false;
    try {
        function();
    } catch (const Exception &) {
        threw = true;
    }
    REQUIRE(threw);
}

class memory_source final : public source {
public:
    memory_source(uint32_t id, source_identity identity, read_backend backend, std::vector<uint8_t> bytes) :
        id_(id), identity_(identity), backend_(backend), bytes_(std::move(bytes)) {
    }

    uint32_t id() const override { return id_; }
    source_identity identity() const override { return identity_; }
    uint64_t size() const override { return bytes_.size(); }
    read_backend backend() const override { return backend_; }

    void read(uint64_t offset, void * destination, size_t length) const override {
        if (offset > bytes_.size() || length > bytes_.size() - size_t(offset)) {
            throw std::out_of_range("test source read outside bounds");
        }
        std::memcpy(destination, bytes_.data() + offset, length);
        ++reads_;
    }

    uint64_t reads() const { return reads_.load(); }

private:
    uint32_t id_;
    source_identity identity_;
    read_backend backend_;
    std::vector<uint8_t> bytes_;
    mutable std::atomic<uint64_t> reads_{0};
};

// A source gate makes concurrency observable without relying on disk speed.
// Each 32-byte component can be held independently, including both extents.
class gated_source final : public source {
public:
    uint32_t id() const override { return 1; }
    source_identity identity() const override { return { 77, 88 }; }
    uint64_t size() const override { return 256; }
    read_backend backend() const override { return read_backend::pread; }
    void read(uint64_t offset, void * destination, size_t length) const override {
        if (offset + length > size()) throw std::out_of_range("gated source range");
        const size_t component = offset / 32;
        std::unique_lock<std::mutex> lock(mutex_);
        ++calls_[component];
        changed_.notify_all();
        changed_.wait(lock, [&] { return !held_[component]; });
        if (failed_[component]) throw std::runtime_error("injected storage failure");
        lock.unlock();
        auto * bytes = static_cast<uint8_t *>(destination);
        for (size_t i = 0; i < length; ++i) bytes[i] = uint8_t(offset + i);
    }
    void hold(size_t component) {
        std::lock_guard<std::mutex> lock(mutex_);
        held_[component] = true;
    }
    bool entered(size_t component) const {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, std::chrono::seconds(2), [&] { return calls_[component] != 0; });
    }
    void release_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        held_.fill(false);
        changed_.notify_all();
    }
    void fail(size_t component, bool enabled) {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_[component] = enabled;
    }
    uint64_t calls(size_t component) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return calls_[component];
    }
    bool advise_will_need(uint64_t offset, uint64_t length) const override {
        if (offset > size() || length > size() - offset) throw std::out_of_range("gated advice range");
        std::lock_guard<std::mutex> lock(mutex_);
        ++advice_calls_[offset / 32];
        advised_bytes_ += length;
        changed_.notify_all();
        return true;
    }
    bool advised(size_t component) const {
        std::unique_lock<std::mutex> lock(mutex_);
        return changed_.wait_for(lock, std::chrono::seconds(2), [&] { return advice_calls_[component] == 2; });
    }
    uint64_t advised_bytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return advised_bytes_;
    }
private:
    mutable std::mutex mutex_;
    mutable std::condition_variable changed_;
    std::array<bool, 8> held_{};
    std::array<bool, 8> failed_{};
    mutable std::array<uint64_t, 8> calls_{};
    mutable std::array<uint64_t, 8> advice_calls_{};
    mutable uint64_t advised_bytes_ = 0;
};

descriptor_spec make_spec(expert_key key, uint32_t source_id, source_identity source_id_value, uint64_t offset) {
    descriptor_spec spec;
    spec.key = key;
    spec.ggml_type = 0;
    spec.rank = 2;
    spec.quant_axis = 0;
    spec.block_elements = 1;
    spec.block_bytes = 1;
    spec.dimensions = {{ 8, 4, 0, 0 }};
    spec.strides = {{ 1, 8, 0, 0 }};
    spec.model_identity = { 0x1111222233334444ULL, 0x5555666677778888ULL };
    spec.extents = {
        { source_id, source_id_value, offset, 13 },
        { source_id, source_id_value, offset + 13, 19 },
    };
    return spec;
}

std::vector<uint8_t> expected_bytes(uint8_t first) {
    std::vector<uint8_t> result(32);
    for (size_t i = 0; i < result.size(); ++i) result[i] = uint8_t(first + i);
    return result;
}

source_identity file_identity(uint32_t source_id, const std::vector<uint8_t> & bytes) {
    auto hash = [](uint64_t value, const void * data, size_t size) {
        const auto * input = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            value ^= input[i];
            value *= 1099511628211ULL;
        }
        return value;
    };
    const uint64_t file_size = bytes.size();
    const size_t sample_size = std::min<size_t>(64*1024, bytes.size());
    uint64_t high = 1469598103934665603ULL;
    uint64_t low  = 7809847782465536322ULL;
    high = hash(high, &source_id, sizeof(source_id));
    high = hash(high, &file_size, sizeof(file_size));
    high = hash(high, bytes.data(), sample_size);
    low = hash(low, bytes.data() + bytes.size() - sample_size, sample_size);
    low = hash(low, &file_size, sizeof(file_size));
    low = hash(low, &source_id, sizeof(source_id));
    return { high, low };
}

void test_descriptor_validation() {
    const source_identity identity = { 1, 2 };
    const std::map<uint32_t, uint64_t> sizes = {{ 7, 256 }};
    auto valid = descriptor::make(make_spec({ 3, 4, component::up }, 7, identity, 16), sizes);
    REQUIRE(valid->bytes() == 32);
    REQUIRE(valid->key() == expert_key({ 3, 4, component::up }));

    auto bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.extents[1].offset = 20;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 240);
    require_throws<std::out_of_range>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.dimensions[1] = UINT64_MAX;
    require_throws<std::overflow_error>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.extents[1].length = 18;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 99, identity, 16);
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.source_alignment = 32;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.source_alignment = 3;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });
}

void test_exact_reads_hits_and_backend_telemetry() {
    const source_identity identities[] = {{ 10, 11 }, { 20, 21 }, { 30, 31 }};
    const read_backend backends[] = { read_backend::mmap, read_backend::pread, read_backend::io_uring };
    for (size_t backend = 0; backend < 3; ++backend) {
        std::vector<uint8_t> bytes(128, 0);
        const auto expected = expected_bytes(uint8_t(40 + backend*40));
        std::copy(expected.begin(), expected.end(), bytes.begin() + 32);
        auto storage = std::make_shared<memory_source>(uint32_t(backend + 1), identities[backend], backends[backend], bytes);
        auto desc = descriptor::make(
                make_spec({ 1, uint32_t(backend), component::down }, storage->id(), identities[backend], 32),
                {{ storage->id(), storage->size() }});
        ram_cache cache({ 64, 32 }, { storage });

        {
            auto lease = cache.acquire(desc);
            REQUIRE(lease.size() == expected.size());
            REQUIRE(reinterpret_cast<uintptr_t>(lease.data()) % 64 == 0);
            REQUIRE(std::equal(expected.begin(), expected.end(), lease.data()));
        }
        {
            auto lease = cache.acquire(desc);
            REQUIRE(std::equal(expected.begin(), expected.end(), lease.data()));
        }
        const auto stats = cache.stats();
        REQUIRE(stats.hits == 1);
        REQUIRE(stats.misses == 1);
        REQUIRE(stats.resident_bytes == 32);
        REQUIRE(stats.peak_resident_bytes == 32);
        REQUIRE(stats.bytes_read[backend] == 32);
        REQUIRE(storage->reads() == 2); // the descriptor has two shard extents
    }
}

void test_deterministic_eviction_and_lease_lifetime(bool frequency = false) {
    const source_identity identity = { 0xaaaa, 0xbbbb };
    std::vector<uint8_t> bytes(256);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i);
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::pread, bytes);
    const std::map<uint32_t, uint64_t> sizes = {{ 1, storage->size() }};
    auto a = descriptor::make(make_spec({ 0, 0, component::gate }, 1, identity, 0), sizes);
    auto b = descriptor::make(make_spec({ 0, 1, component::gate }, 1, identity, 32), sizes);
    auto c = descriptor::make(make_spec({ 0, 2, component::gate }, 1, identity, 64), sizes);
    auto d = descriptor::make(make_spec({ 0, 3, component::gate }, 1, identity, 96), sizes);
    // Two 32-byte payloads require 96 bytes in the fixed arena because every
    // independently leased component begins at a 64-byte boundary.
    ram_cache cache({ 96, 32, frequency }, { storage });

    auto lease_a = cache.acquire(a);
    {
        auto lease_b = cache.acquire(b);
        REQUIRE(cache.stats().resident_bytes == 64);
    }
    auto lease_c = cache.acquire(c); // B is the only unpinned candidate.
    REQUIRE(cache.stats().resident_bytes == 64);
    REQUIRE(cache.stats().evictions == 1);
    REQUIRE(lease_a.data()[0] == 0);
    REQUIRE(lease_c.data()[0] == 64);

    const auto before_failure = cache.stats();
    const auto reads_before_failure = storage->reads();
    require_throws<capacity_error>([&] { cache.acquire(d); });
    const auto after_failure = cache.stats();
    REQUIRE(after_failure.resident_bytes == before_failure.resident_bytes);
    REQUIRE(after_failure.evictions == before_failure.evictions);
    REQUIRE(after_failure.pinned_capacity_failures == before_failure.pinned_capacity_failures + 1);
    REQUIRE(storage->reads() == reads_before_failure); // reject before storage I/O
    REQUIRE(!cache.erase(a->key()));
    require_throws<capacity_error>([&] { cache.clear(); });

    lease_c = {};
    auto lease_d = cache.acquire(d);
    REQUIRE(lease_d.data()[0] == 96);
    REQUIRE(cache.stats().resident_bytes == 64);
    REQUIRE(cache.stats().evictions == 2);
}

void test_ram_frequency_retention() {
    const source_identity identity = { 41, 92 };
    std::vector<uint8_t> bytes(256);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i);
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::pread, bytes);
    std::vector<std::shared_ptr<const descriptor>> descriptors;
    for (int i = 0; i < 8; ++i) descriptors.push_back(descriptor::make(
        make_spec({0, uint32_t(i), component::gate}, 1, identity, i * 32), {{1, storage->size()}}));
    auto sweep = [&](bool frequency) {
        ram_cache cache({160, 32, frequency}, {storage});
        auto access = [&](int i) {
            auto value = cache.acquire(descriptors[i]);
            REQUIRE(value.size() == 32 && std::memcmp(value.data(), bytes.data() + i*32, 32) == 0);
        };
        for (int i = 0; i < 10; ++i) access(0);
        for (int pass = 0; pass < 40; ++pass) {
            access(0);
            for (int i = 1; i < 8; ++i) access(i);
        }
        const auto misses = cache.stats().misses;
        // Cross two aging epochs through ready hits. Every read remains exact,
        // and clear resets the retention history while preserving identities.
        for (int i = 0; i < 33000; ++i) access(0);
        cache.clear(); access(7); access(0);
        REQUIRE(cache.stats().resident_bytes <= 160);
        return misses;
    };
    const auto lru = sweep(false), frequency = sweep(true);
    REQUIRE(frequency < lru);
    std::cout << "RAM frequency retention sweep: " << lru << " -> " << frequency << " storage reads\n";
}

void test_capacity_and_identity_fail_closed() {
    const source_identity identity = { 9, 8 };
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::mmap, std::vector<uint8_t>(128, 7));
    auto desc = descriptor::make(make_spec({ 2, 2, component::up }, 1, identity, 0), {{ 1, 128 }});

    require_throws<std::invalid_argument>([&] { ram_cache({ 0, 0 }, { storage }); });
    require_throws<std::invalid_argument>([&] { ram_cache({ 64, 65 }, { storage }); });
    ram_cache too_small({ 16, 16 }, { storage });
    require_throws<capacity_error>([&] { too_small.acquire(desc); });
    REQUIRE(too_small.stats().resident_bytes == 0);

    auto wrong_storage = std::make_shared<memory_source>(1, source_identity{ 7, 7 }, read_backend::mmap, std::vector<uint8_t>(128, 7));
    ram_cache wrong_identity({ 64, 32 }, { wrong_storage });
    require_throws<std::invalid_argument>([&] { wrong_identity.acquire(desc); });
    REQUIRE(wrong_identity.stats().resident_bytes == 0);

    ram_cache canonical_descriptor({ 64, 32 }, { storage });
    auto original = canonical_descriptor.acquire(desc);
    REQUIRE(original.data()[0] == 7);
    original = {};
    canonical_descriptor.clear();
    auto alternate_spec = make_spec(desc->key(), 1, identity, 32);
    auto alternate = descriptor::make(std::move(alternate_spec), {{ 1, 128 }});
    require_throws<std::invalid_argument>([&] { canonical_descriptor.acquire(alternate); });
    REQUIRE(canonical_descriptor.stats().resident_bytes == 0);
}

void test_file_storage_backends() {
#if defined(__linux__)
    char path[] = "/tmp/ese-expert-cache-XXXXXX";
    const int fd = ::mkstemp(path);
    REQUIRE(fd >= 0);
    struct cleanup {
        const char * path;
        ~cleanup() { ::unlink(path); }
    } remove_file{path};

    std::vector<uint8_t> bytes(128);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(255 - i);
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
        REQUIRE(count > 0);
        written += size_t(count);
    }
    REQUIRE(::close(fd) == 0);
    require_throws<std::runtime_error>([&] {
        file_source changed(99, { 1, 2 }, path, read_backend::pread);
    });

    const read_backend backends[] = { read_backend::mmap, read_backend::pread, read_backend::io_uring };
    for (size_t i = 0; i < 3; ++i) {
        const uint32_t source_id = uint32_t(i + 1);
        REQUIRE(identify_file_source(path, source_id) == file_identity(source_id, bytes));
        std::shared_ptr<file_source> storage;
        try {
            storage = std::make_shared<file_source>(
                    source_id, file_identity(source_id, bytes), path, backends[i]);
        } catch (const std::runtime_error & error) {
            REQUIRE(backends[i] == read_backend::io_uring);
            REQUIRE(std::string(error.what()).find("io_uring is unavailable") != std::string::npos);
            std::cout << "SKIP: native io_uring unavailable: " << error.what() << '\n';
            continue;
        }
        {
            std::array<uint8_t, 32> output = {};
            storage->read(32, output.data(), output.size());
            REQUIRE(std::equal(output.begin(), output.end(), bytes.begin() + 32));
            std::array<uint8_t, 8> batch_a = {};
            std::array<uint8_t, 8> batch_b = {};
            storage->read_many({
                {8, batch_a.data(), batch_a.size()},
                {96, batch_b.data(), batch_b.size()},
            });
            REQUIRE(std::equal(batch_a.begin(), batch_a.end(), bytes.begin() + 8));
            REQUIRE(std::equal(batch_b.begin(), batch_b.end(), bytes.begin() + 96));
            // Cross the 64-entry ring boundary and exercise bit 63 in the
            // allocation-free completion-identity mask.
            std::array<uint8_t, 129> many_output{};
            std::vector<source::read_request> many_requests;
            for (size_t index = 0; index < many_output.size(); ++index)
                many_requests.push_back({index % bytes.size(), &many_output[index], 1});
            storage->read_many(many_requests);
            for (size_t index = 0; index < many_output.size(); ++index)
                REQUIRE(many_output[index] == bytes[index % bytes.size()]);
            storage->evict_range_from_os_cache(0, bytes.size());
            require_throws<std::out_of_range>([&] { storage->evict_range_from_os_cache(120, 16); });
            require_throws<std::out_of_range>([&] { storage->read(120, output.data(), output.size()); });
            require_throws<std::invalid_argument>([&] { storage->read(0, nullptr, 1); });
            if (backends[i] == read_backend::io_uring) {
                // A short completion must not poison the next read's CQ state.
                const int writer = ::open(path, O_WRONLY);
                REQUIRE(writer >= 0);
                REQUIRE(::ftruncate(writer, 16) == 0);
                require_throws<std::runtime_error>([&] { storage->read(32, output.data(), output.size()); });
                storage->read(0, batch_a.data(), batch_a.size());
                REQUIRE(std::equal(batch_a.begin(), batch_a.end(), bytes.begin()));
                require_throws<std::runtime_error>([&] {
                    storage->read_many({{0, batch_a.data(), batch_a.size()}, {96, batch_b.data(), batch_b.size()}});
                });
                storage->read_many({{0, batch_b.data(), batch_b.size()}});
                REQUIRE(std::equal(batch_b.begin(), batch_b.end(), bytes.begin()));
                REQUIRE(::pwrite(writer, bytes.data(), bytes.size(), 0) == ssize_t(bytes.size()));
                REQUIRE(::close(writer) == 0);
            }
        }
    }
#endif
}

void test_concurrent_single_fill() {
    const source_identity identity = { 77, 88 };
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::pread, std::vector<uint8_t>(128, 5));
    auto desc = descriptor::make(make_spec({ 4, 5, component::down }, 1, identity, 0), {{ 1, 128 }});
    ram_cache cache({ 64, 32 }, { storage });
    std::array<std::thread, 8> threads;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    for (auto & thread : threads) {
        thread = std::thread([&] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            auto lease = cache.acquire(desc);
            REQUIRE(lease.size() == 32 && lease.data()[0] == 5);
        });
    }
    while (ready.load() != int(threads.size())) std::this_thread::yield();
    go.store(true);
    for (auto & thread : threads) thread.join();
    const auto stats = cache.stats();
    REQUIRE(stats.misses == 1);
    REQUIRE(stats.hits == threads.size() - 1);
    REQUIRE(storage->reads() == 2);
    REQUIRE(stats.active_leases == 0);
}

void test_ready_hit_during_blocked_miss() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto a = make(0), b = make(1);
    ram_cache cache({128, 32}, {storage});
    auto resident = cache.acquire(a);
    storage->hold(1);
    auto miss = std::async(std::launch::async, [&] { return cache.acquire(b); });
    const bool entered = storage->entered(1);
    auto hit = std::async(std::launch::async, [&] { return cache.acquire(a); });
    const bool responsive = hit.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    // Always release the source before asserting, including on the old code.
    storage->release_all();
    auto hit_lease = hit.get();
    auto miss_lease = miss.get();
    REQUIRE(entered);
    REQUIRE(responsive);
    REQUIRE(hit_lease.data() == resident.data());
    REQUIRE(miss_lease.data()[0] == 32);
}

void test_independent_fills_share_bounded_staging() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto a = make(0), b = make(1);
    ram_cache cache({128, 64}, {storage});
    storage->hold(0);
    storage->hold(1);
    auto first = std::async(std::launch::async, [&] { return cache.acquire(a); });
    const bool first_entered = storage->entered(0);
    auto second = std::async(std::launch::async, [&] { return cache.acquire(b); });
    const bool second_entered = storage->entered(1);
    storage->release_all();
    auto x = first.get();
    auto y = second.get();
    REQUIRE(first_entered && second_entered);
    REQUIRE(x.data() != y.data());
    for (size_t i = 0; i < 32; ++i) {
        REQUIRE(x.data()[i] == i);
        REQUIRE(y.data()[i] == 32 + i);
    }
    const auto stats = cache.stats();
    REQUIRE(stats.peak_active_loads == 2);
    REQUIRE(stats.peak_staging_used_bytes == 64);
    REQUIRE(stats.staging_capacity_bytes == 64);
    REQUIRE(stats.active_loads == 0 && stats.staging_used_bytes == 0 && stats.reserved_bytes == 0);
}

template<class Predicate>
bool eventually(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return true;
}

void test_staging_backpressure_preserves_ready_hits() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto a = make(0), b = make(1), c = make(2);
    ram_cache cache({192, 32}, {storage});
    auto ready = cache.acquire(a);
    storage->hold(1);
    auto first = std::async(std::launch::async, [&] { return cache.acquire(b); });
    const bool entered = storage->entered(1);
    auto second = std::async(std::launch::async, [&] { return cache.acquire(c); });
    const bool waited = eventually([&] { return cache.stats().staging_waits != 0; });
    const bool no_early_read = storage->calls(2) == 0;
    auto hit = cache.acquire(a);
    storage->release_all();
    auto x = first.get();
    auto y = second.get();
    REQUIRE(entered && waited && no_early_read);
    REQUIRE(hit.data() == ready.data());
    REQUIRE(x.data()[0] == 32 && y.data()[0] == 64);
    REQUIRE(cache.stats().peak_staging_used_bytes == 32);
    REQUIRE(cache.stats().peak_active_loads == 1);
}

void test_failed_fill_preserves_victims_and_coalesces_error() {
    auto storage = std::make_shared<gated_source>();
    const auto a = descriptor::make(make_spec({0, 0, component::up}, 1, storage->identity(), 0), {{1, 256}});
    const auto b = descriptor::make(make_spec({0, 1, component::up}, 1, storage->identity(), 32), {{1, 256}});
    ram_cache cache({32, 32}, {storage});
    { auto ready = cache.acquire(a); }
    storage->hold(1);
    storage->fail(1, true);
    auto failing = [&] {
        try { cache.acquire(b); return false; }
        catch (const std::runtime_error & error) { return std::string(error.what()) == "injected storage failure"; }
    };
    auto first = std::async(std::launch::async, failing);
    const bool entered = storage->entered(1);
    auto second = std::async(std::launch::async, failing);
    const bool coalesced = eventually([&] { return cache.stats().coalesced_loads == 1; });
    const bool protected_from_erase = !cache.erase(a->key());
    bool clear_rejected = false;
    try { cache.clear(); } catch (const capacity_error &) { clear_rejected = true; }
    auto still_ready = cache.acquire(a); // reserved victims remain readable
    storage->release_all();
    const bool first_failed = first.get(), second_failed = second.get();
    REQUIRE(entered && coalesced && protected_from_erase && clear_rejected);
    REQUIRE(first_failed && second_failed);
    REQUIRE(still_ready.data()[0] == 0);
    REQUIRE(storage->calls(1) == 1);
    REQUIRE(cache.stats().evictions == 0 && cache.stats().resident_bytes == 32);
    REQUIRE(cache.stats().failed_loads == 1 && cache.stats().reserved_bytes == 0);
    REQUIRE(cache.stats().active_loads == 0 && cache.stats().staging_used_bytes == 0);
    still_ready = {};
    storage->fail(1, false);
    auto retry = cache.acquire(b);
    REQUIRE(retry.data()[0] == 32);
    REQUIRE(cache.stats().evictions == 1);
}

void test_reserved_victim_can_become_leased() {
    for (bool alternate_region : {false, true}) {
        auto storage = std::make_shared<gated_source>();
        auto make = [&](uint32_t i) {
            return descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
        };
        const auto a = make(0), b = make(1), c = make(2);
        ram_cache cache({alternate_region ? 96u : 32u, 32}, {storage});
        { auto ready = cache.acquire(a); }
        if (alternate_region) { auto ready = cache.acquire(c); }
        storage->hold(1);
        auto miss = std::async(std::launch::async, [&] { return cache.acquire(b); });
        const bool entered = storage->entered(1);
        auto protected_victim = cache.acquire(a);
        storage->release_all();
        bool rejected = false;
        ram_cache::lease result;
        try { result = miss.get(); } catch (const capacity_error &) { rejected = true; }
        REQUIRE(entered);
        REQUIRE(protected_victim.data()[0] == 0);
        REQUIRE(rejected == !alternate_region);
        if (alternate_region) REQUIRE(result.data()[0] == 32);
        REQUIRE(cache.stats().evictions == (alternate_region ? 1 : 0));
        REQUIRE(cache.stats().active_loads == 0 && cache.stats().reserved_bytes == 0);
    }
}

void test_concurrent_eviction_stress(bool frequency = false) {
    std::vector<uint8_t> bytes(256);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i);
    auto storage = std::make_shared<memory_source>(1, source_identity{77, 88}, read_backend::pread, bytes);
    std::array<std::shared_ptr<const descriptor>, 8> descriptors;
    for (uint32_t i = 0; i < descriptors.size(); ++i) {
        descriptors[i] = descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    }
    ram_cache cache({256, 96, frequency}, {storage});
    std::array<std::future<size_t>, 8> tasks;
    std::atomic<bool> start{false};
    for (size_t worker = 0; worker < tasks.size(); ++worker) {
        tasks[worker] = std::async(std::launch::async, [&, worker] {
            while (!start.load()) std::this_thread::yield();
            size_t completed = 0;
            for (size_t step = 0; step < (frequency ? 4096 : 256); ++step) {
                const size_t index = (step*5 + worker*3) % descriptors.size();
                try {
                    auto value = cache.acquire(descriptors[index]);
                    std::this_thread::yield(); // permit other fills/evictions while leased
                    for (size_t byte = 0; byte < value.size(); ++byte) REQUIRE(value.data()[byte] == index*32 + byte);
                    ++completed;
                } catch (const capacity_error &) {
                    // In-flight leases may occupy every admissible destination.
                }
            }
            return completed;
        });
    }
    start.store(true);
    size_t completed = 0;
    for (auto & task : tasks) completed += task.get();
    REQUIRE(completed > 0);
    const auto stats = cache.stats();
    REQUIRE(stats.peak_staging_used_bytes <= 96);
    REQUIRE(stats.peak_resident_bytes <= 256);
    REQUIRE(stats.active_loads == 0 && stats.active_leases == 0);
    REQUIRE(stats.staging_used_bytes == 0 && stats.reserved_bytes == 0);
    cache.clear();
    REQUIRE(cache.stats().resident_bytes == 0);
}

struct alignas(64) kernel_lease_fixture {
    std::array<std::array<float, 32*32>, 3> experts;
    std::atomic<uint64_t> acquires{0};
    std::atomic<uint64_t> releases{0};
};

bool kernel_lease_acquire(void * user_data, int layer, int expert, int component, ggml_expert_lease * lease) {
    auto * fixture = static_cast<kernel_lease_fixture *>(user_data);
    if (layer != 0 || component != 0 || expert < 0 || expert >= int(fixture->experts.size())) return false;
    auto & bytes = fixture->experts[size_t(expert)];
    lease->data = bytes.data();
    lease->size = bytes.size()*sizeof(float);
    lease->handle = new int(expert);
    ++fixture->acquires;
    return true;
}

void kernel_lease_release(void * user_data, void * handle) {
    auto * fixture = static_cast<kernel_lease_fixture *>(user_data);
    delete static_cast<int *>(handle);
    ++fixture->releases;
}

void test_load_queue_completion_order_and_bounds() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto a = make(0), b = make(1), c = make(2);
    ram_cache cache({256, 64}, {storage});
    load_queue queue(cache, 2, 2, 64);
    storage->hold(0);
    const bool first = queue.try_submit(10, a);
    const bool entered = storage->entered(0);
    const bool second = queue.try_submit(20, b);
    const bool full = !queue.try_submit(30, c);
    auto take = std::async(std::launch::async, [&] {
        load_queue::completion result;
        queue.next(result, true);
        return result;
    });
    const bool independent = take.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    storage->release_all();
    auto fast = take.get();
    REQUIRE(first && entered && second && full && independent);
    REQUIRE(fast.id == 20 && !fast.error && fast.payload.data()[0] == 32);
    load_queue::completion slow;
    REQUIRE(queue.next(slow, true));
    REQUIRE(slow.id == 10 && !slow.error && slow.payload.data()[0] == 0);
    REQUIRE(queue.stats().outstanding_bytes == 0);
    REQUIRE(queue.stats().peak_outstanding_bytes == 64);
    REQUIRE(cache.stats().peak_active_loads == 2);
    REQUIRE(!queue.next(slow, true));

    // Results remain lease protected after removal from the queue.
    REQUIRE(queue.try_submit(30, c));
    load_queue::completion third;
    REQUIRE(queue.next(third, true));
    REQUIRE(!third.error && third.payload.data()[0] == 64);
    REQUIRE(fast.payload.data()[0] == 32 && slow.payload.data()[0] == 0);
}

void test_load_queue_ready_hits_and_error_recovery() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({0, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto a = make(0), b = make(1), c = make(2), d = make(3);
    ram_cache cache({256, 96}, {storage});
    { auto ready = cache.acquire(c); }
    load_queue queue(cache, 2, 3, 96);
    storage->hold(0);
    storage->hold(1);
    const bool first = queue.try_submit(0, a), second = queue.try_submit(1, b);
    const bool both = storage->entered(0) && storage->entered(1);
    const bool hit = queue.try_submit(2, c);
    load_queue::completion result;
    const bool immediate = queue.next(result, false);
    const bool right_hit = immediate && result.id == 2 && result.payload && result.payload.data()[0] == 64;
    storage->release_all();
    REQUIRE(first && second && both && hit && right_hit);
    REQUIRE(queue.stats().ready_hits == 1);
    result.payload = {};
    REQUIRE(queue.next(result, true));
    result.payload = {};
    REQUIRE(queue.next(result, true));
    result.payload = {};

    storage->fail(3, true);
    REQUIRE(queue.try_submit(3, d));
    require_throws<std::invalid_argument>([&] { queue.try_submit(3, d); });
    REQUIRE(queue.next(result, true));
    REQUIRE(result.id == 3 && result.error && !result.payload);
    require_throws<std::runtime_error>([&] { std::rethrow_exception(result.error); });
    storage->fail(3, false);
    REQUIRE(queue.try_submit(3, d));
    REQUIRE(queue.next(result, true));
    REQUIRE(!result.error && result.payload.data()[0] == 96);
    result.payload = {};
    REQUIRE(queue.stats().outstanding_bytes == 0);
    REQUIRE(cache.stats().active_leases == 0);
    REQUIRE(cache.stats().reserved_bytes == 0);
    require_throws<std::invalid_argument>([&] { load_queue invalid(cache, 0, 1, 32); });
    require_throws<std::invalid_argument>([&] { load_queue invalid(cache, 2, 1, 32); });
    require_throws<std::invalid_argument>([&] { load_queue invalid(cache, 1, 1, 97); });
    load_queue small(cache, 1, 1, 16);
    REQUIRE(!small.try_submit(0, a));
}

void test_load_queue_shutdown_joins_reads() {
    auto storage = std::make_shared<gated_source>();
    auto a = descriptor::make(make_spec({0, 0, component::up}, 1, storage->identity(), 0), {{1, 256}});
    ram_cache cache({128, 32}, {storage});
    auto queue = std::make_unique<load_queue>(cache, 1, 1, 32);
    storage->hold(0);
    const bool submitted = queue->try_submit(0, a);
    const bool entered = storage->entered(0);
    auto close = std::async(std::launch::async, [owned = std::move(queue)]() mutable { owned.reset(); });
    const bool waits = close.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout;
    storage->release_all();
    close.get();
    REQUIRE(submitted && entered && waits);
    REQUIRE(cache.stats().active_loads == 0);
    REQUIRE(cache.stats().active_leases == 0);
    REQUIRE(cache.stats().staging_used_bytes == 0);
}

void test_read_ahead_identity_and_retention() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({i, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto a = make(0), b = make(1);
    ram_cache cache({128, 64, true}, {storage});
    REQUIRE(cache.advise_will_need(b) == 32);
    REQUIRE(storage->advised_bytes() == 32);
    REQUIRE(cache.stats().misses == 0 && cache.stats().hits == 0 && cache.stats().resident_bytes == 0);
    { auto lease = cache.acquire(a); }
    const auto before = cache.stats();
    REQUIRE(cache.advise_will_need(a) == 0);
    REQUIRE(cache.stats().hits == before.hits && cache.stats().misses == before.misses);
    REQUIRE(storage->advised_bytes() == 32);
    require_throws<std::invalid_argument>([&] { cache.advise_will_need(make(0)); });
    require_throws<std::invalid_argument>([&] { cache.advise_will_need(nullptr); });
    auto bad = make_spec({2, 2, component::up}, 1, storage->identity(), 64);
    ++bad.extents[1].identity.high;
    const auto bad_desc = descriptor::make(bad, {{1, 256}});
    require_throws<std::invalid_argument>([&] { cache.advise_will_need(bad_desc); });
    REQUIRE(storage->advised_bytes() == 32); // all extents validated before the first hint
}

void test_external_resident_arena_ownership() {
    auto storage = std::make_shared<gated_source>();
    const auto desc = descriptor::make(make_spec({0, 0, component::gate}, 1, storage->identity(), 0), {{1, 256}});
    auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 256);
    REQUIRE(buffer != nullptr);
    const auto address = reinterpret_cast<uintptr_t>(ggml_backend_buffer_get_base(buffer));
    auto * base = reinterpret_cast<uint8_t *>((address + 63) & ~uintptr_t(63));
    REQUIRE(uintptr_t(base) - address + 128 <= ggml_backend_buffer_get_size(buffer));
    int frees = 0;
    std::shared_ptr<uint8_t> arena(base, [&](uint8_t *) { ++frees; ggml_backend_buffer_free(buffer); });
    {
        std::shared_ptr<uint8_t> misaligned(arena, base + 1);
        require_throws<std::invalid_argument>([&] { ram_cache invalid({128, 32}, {storage}, misaligned); });
    }
    {
        ram_cache cache({128, 32}, {storage}, arena);
        arena.reset();
        auto lease = cache.acquire(desc);
        REQUIRE(lease.data() == base && lease.size() == 32);
        for (size_t i = 0; i < lease.size(); ++i) REQUIRE(lease.data()[i] == uint8_t(i));
        REQUIRE(frees == 0 && cache.stats().resident_bytes == 32);
    }
    REQUIRE(frees == 1);
}

void test_read_ahead_demand_priority_budget_and_expiration() {
    auto storage = std::make_shared<gated_source>();
    auto make = [&](uint32_t i) {
        return descriptor::make(make_spec({i, i, component::up}, 1, storage->identity(), i*32), {{1, 256}});
    };
    const auto demand_desc = make(0), a = make(1), b = make(2), c = make(3);
    ram_cache cache({256, 64}, {storage});
    load_queue demand(cache, 1, 1, 32);
    storage->hold(0);
    const bool submitted = demand.try_submit(0, demand_desc);
    const bool entered = storage->entered(0);
    read_ahead_queue future(cache, &demand, 64);
    future.advance(1, 1, 0);
    const bool accepted = future.submit(a) && future.submit(b);
    const bool bounded = !future.submit(c) && !future.submit(a);
    future.advance(1, 1, 1);
    const bool deadline = future.stats().expired == 1;
    future.advance(1, 2, 0);
    const bool new_pass = future.stats().expired == 2 && future.submit(c);
    const bool priority = storage->advised_bytes() == 0;
    const bool in_flight_skipped = cache.advise_will_need(demand_desc) == 0;
    storage->release_all(); // release before checking failures so teardown cannot block
    load_queue::completion completion;
    const bool taken = demand.next(completion, true);
    const bool advised = storage->advised(3);
    future.stop();
    REQUIRE(submitted && entered && accepted && bounded && deadline && new_pass && priority && in_flight_skipped);
    REQUIRE(taken && !completion.error && completion.payload.data()[0] == 0 && advised);
    REQUIRE(future.stats().advised_bytes == 32 && future.stats().outstanding_bytes == 0);
    REQUIRE(future.stats().peak_outstanding_bytes == 64 && future.stats().failed == 0);
    REQUIRE(cache.stats().misses == 1 && storage->calls(3) == 0);
    REQUIRE(!future.submit(a));
    require_throws<std::invalid_argument>([&] { read_ahead_queue invalid(cache, nullptr, 0); });
    require_throws<std::invalid_argument>([&] { read_ahead_queue invalid(cache, nullptr, (1ULL << 30) + 1); });
}

void test_cpu_router_softmax_replay() {
    // Match the model router's softmax -> reshape -> sort -> view -> gather.
    // Poison its output and change logits between replays: routing must never
    // consume a previous invocation's softmax storage. Check independent
    // probabilities as well as IDs, including scaled and masked softmax.
    for (int threads : {1, 4}) {
        for (bool masked : {false, true}) {
            ggml_init_params params = {1024*1024, nullptr, false};
            auto * ctx = ggml_init(params);
            REQUIRE(ctx != nullptr);
            struct cleanup { ggml_context * ctx; ~cleanup() { ggml_free(ctx); } } free_ctx{ctx};
            constexpr int experts = 8, rows = 3, selected = 2;
            auto * logits = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, rows);
            auto * mask = masked ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, experts, rows) : nullptr;
            const float scale = masked ? 0.5f : 1.0f;
            auto * probs = ggml_soft_max_ext(ctx, logits, mask, scale, 0.0f);
            auto * ids = ggml_top_k(ctx, probs, selected);
            auto * weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, experts, rows), ids);
            auto * graph = ggml_new_graph_custom(ctx, 32, false);
            ggml_build_forward_expand(graph, weights);
            auto plan = ggml_graph_plan(graph, threads);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.empty() ? nullptr : work.data();
            for (int trial = 0; trial < 3; ++trial) {
                std::fill_n(static_cast<float *>(probs->data), experts*rows,
                        std::numeric_limits<float>::quiet_NaN());
                for (int row = 0; row < rows; ++row) {
                    for (int expert = 0; expert < experts; ++expert) {
                        static_cast<float *>(logits->data)[row*experts + expert] =
                                float((expert + 3*row + 2*trial) % experts) - 4.0f;
                        if (mask) static_cast<float *>(mask->data)[row*experts + expert] =
                                expert == row ? -INFINITY : 0.0f;
                    }
                }
                REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
                for (int row = 0; row < rows; ++row) {
                    std::array<double, experts> expected = {};
                    std::array<int, experts> order = {};
                    double sum = 0;
                    for (int expert = 0; expert < experts; ++expert) {
                        const float value = static_cast<float *>(logits->data)[row*experts + expert];
                        expected[expert] = mask && expert == row ? 0 : std::exp(double(value)*scale);
                        sum += expected[expert];
                        order[expert] = expert;
                    }
                    std::sort(order.begin(), order.end(), [&](int a, int b) { return expected[a] > expected[b]; });
                    for (int expert = 0; expert < experts; ++expert) {
                        const float actual = static_cast<float *>(probs->data)[row*experts + expert];
                        REQUIRE(std::abs(actual - expected[expert]/sum) < 1e-6);
                    }
                    for (int slot = 0; slot < selected; ++slot) {
                        const auto actual_id = *reinterpret_cast<int32_t *>(
                                static_cast<char *>(ids->data) + row*ids->nb[1] + slot*ids->nb[0]);
                        REQUIRE(actual_id == order[slot]);
                        const float actual_weight = static_cast<float *>(weights->data)[row*selected + slot];
                        REQUIRE(std::abs(actual_weight - expected[order[slot]]/sum) < 1e-6);
                    }
                }
            }
        }
    }
}

void test_cpu_kernel_exact_parity_and_no_original_fallback() {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct cleanup { ggml_context * ctx; ~cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    constexpr int64_t k = 32;
    constexpr int64_t m = 32;
    constexpr int64_t n_experts = 3;
    constexpr int64_t n_used = 2;
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_experts);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight");
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);

    kernel_lease_fixture fixture;
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int64_t i = 0; i < k*m; ++i) {
            fixture.experts[size_t(expert)][size_t(i)] = float(1 + expert*17 + i);
        }
        std::memcpy((uint8_t *) weights->data + expert*weights->nb[2],
                fixture.experts[size_t(expert)].data(), fixture.experts[size_t(expert)].size()*sizeof(float));
    }
    std::array<float, k*n_used> input_values = {};
    for (size_t i = 0; i < input_values.size(); ++i) input_values[i] = float(int(i % 9) - 4)/3.0f;
    const std::array<int32_t, n_used> route = {{ 2, 0 }};
    std::memcpy(input->data, input_values.data(), sizeof(input_values));
    std::memcpy(ids->data, route.data(), sizeof(route));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    auto compute = [&](bool leases) {
        ggml_cplan plan = ggml_graph_plan(graph, 2);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
        if (leases) {
            plan.expert_lease_acquire = kernel_lease_acquire;
            plan.expert_lease_release = kernel_lease_release;
            plan.expert_lease_user_data = &fixture;
            plan.expert_lease_required = true;
        }
        REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
        std::vector<float> result(size_t(ggml_nelements(output)));
        std::memcpy(result.data(), output->data, result.size()*sizeof(float));
        return result;
    };

    const auto baseline = compute(false);
    std::memset((uint8_t *) weights->data + route[0]*weights->nb[2], 0, weights->nb[2]);
    std::memset((uint8_t *) weights->data + route[1]*weights->nb[2], 0, weights->nb[2]);
    const auto leased = compute(true);
    REQUIRE(leased == baseline);
    REQUIRE(fixture.acquires > 0 && fixture.acquires == fixture.releases);

    const auto corrupted = compute(false);
    REQUIRE(corrupted != baseline);
    REQUIRE(*(const int32_t *) ids->data == route[0]);
    REQUIRE(*((const int32_t *) ids->data + 1) == route[1]);
}

void test_cpu_lease_chunks_with_bounded_cache() {
#if !defined(_WIN32) && defined(ESE_TEST_EXPERT_LEASE_CHUNKING)
    const char * old = std::getenv("ESE_CPU_EXPERT_LEASE_CHUNKING");
    const bool had_old = old != nullptr;
    const std::string saved = old ? old : "";
    struct restore_setting {
        bool present; std::string value;
        ~restore_setting() {
            if (present) setenv("ESE_CPU_EXPERT_LEASE_CHUNKING", value.c_str(), 1);
            else unsetenv("ESE_CPU_EXPERT_LEASE_CHUNKING");
        }
    } restore{had_old, saved};
    struct fixture {
        std::unique_ptr<ram_cache> cache;
        std::vector<std::shared_ptr<const descriptor>> descriptors;
        std::atomic<uint64_t> acquired{0}, released{0}, active{0}, peak{0}, refused{0};
        static bool acquire(void * opaque, int layer, int expert, int component, ggml_expert_lease * out) {
            auto & f = *static_cast<fixture *>(opaque);
            if (layer != 0 || component != 2 || expert < 0 || size_t(expert) >= f.descriptors.size()) return false;
            try {
                auto lease = std::make_unique<ram_cache::lease>(f.cache->acquire(f.descriptors[size_t(expert)]));
                out->data = lease->data(); out->size = lease->size(); out->handle = lease.release();
                ++f.acquired;
                const uint64_t active = ++f.active;
                uint64_t peak = f.peak.load();
                while (peak < active && !f.peak.compare_exchange_weak(peak, active)) {}
                return true;
            } catch (const capacity_error &) { ++f.refused; return false; }
        }
        static void release(void * opaque, void * handle) {
            auto & f = *static_cast<fixture *>(opaque);
            delete static_cast<ram_cache::lease *>(handle);
            --f.active; ++f.released;
        }
    } f;
    constexpr int experts = 8, used = 8, tokens = 3;
    struct shape { ggml_type type; int64_t k, m; };
    // Production gate/up and down dimensions/types, plus a small float case
    // whose row count leaves some of eight threads without an initial chunk.
    for (const shape s : {shape{GGML_TYPE_F32, 256, 96}, shape{GGML_TYPE_IQ3_S, 2560, 640},
                         shape{GGML_TYPE_IQ4_NL, 640, 2560}, shape{GGML_TYPE_Q2_0, 640, 2560}}) {
        std::cout << "Checking leased CPU " << ggml_type_name(s.type) << std::endl;
        ggml_init_params params = {1024*1024, nullptr, true};
        auto * ctx = ggml_init(params);
        REQUIRE(ctx != nullptr);
        struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } cleanup{ctx};
        auto * w = ggml_new_tensor_3d(ctx, s.type, s.k, s.m, experts);
        ggml_set_name(w, "blk.0.ffn_down_exps.weight");
        auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, s.k, s.type == GGML_TYPE_IQ3_S ? 1 : used, tokens);
        auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
        auto * y = ggml_mul_mat_id(ctx, w, x, ids);
        auto * graph = ggml_new_graph_custom(ctx, 16, false);
        ggml_build_forward_expand(graph, y);
        auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_cpu_buffer_type());
        REQUIRE(buffer != nullptr);
        struct buffer_cleanup { ggml_backend_buffer_t buffer; ~buffer_cleanup() { ggml_backend_buffer_free(buffer); } } free_buffer{buffer};
        std::vector<float> values(size_t(s.k*s.m*experts));
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i*37 + i/127)%509) - 254)/512;
        std::vector<uint8_t> bytes(ggml_nbytes(w));
        if (s.type == GGML_TYPE_Q2_0) {
            // GSQ Q2_0 is decode-only; construct valid production-format blocks.
            const size_t block = ggml_type_size(s.type);
            for (size_t i = 0; i < bytes.size(); i += block) {
                const auto scale = ggml_fp32_to_fp16(float(1 + i/block%7)/64);
                std::memcpy(bytes.data()+i, &scale, sizeof(scale));
                for (size_t j = 2; j < block; ++j) bytes[i+j] = uint8_t((i/block*37+j*19)%256);
            }
        } else {
            std::vector<float> importance(size_t(s.k), 1.0f);
            REQUIRE(ggml_quantize_chunk(s.type, values.data(), bytes.data(), 0, s.m*experts, s.k,
                    importance.data(), nullptr) == bytes.size());
        }
        ggml_backend_tensor_set(w, bytes.data(), 0, bytes.size());
        auto storage = std::make_shared<memory_source>(1, source_identity{12, 34}, read_backend::pread, bytes);
        f.descriptors.clear();
        for (int e = 0; e < experts; ++e) {
            descriptor_spec spec;
            spec.key = {0, uint32_t(e), component::down}; spec.ggml_type = uint32_t(s.type);
            spec.rank = 2; spec.quant_axis = 0;
            spec.block_elements = uint32_t(ggml_blck_size(s.type)); spec.block_bytes = uint32_t(ggml_type_size(s.type));
            spec.dimensions = {{uint64_t(s.k), uint64_t(s.m), 0, 0}};
            spec.strides = {{ggml_type_size(s.type), w->nb[1], 0, 0}};
            spec.model_identity = {56, 78}; spec.extents = {{1, storage->identity(), uint64_t(e)*w->nb[2], w->nb[2]}};
            f.descriptors.push_back(descriptor::make(spec, {{1, storage->size()}}));
        }
        std::vector<float> input(size_t(ggml_nelements(x)));
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i%31) - 15)/17;
        ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
        std::array<int32_t, used*tokens> routes;
        for (size_t i = 0; i < routes.size(); ++i) routes[i] = i%used == 7 ? -1 : int32_t((i + i/used)%experts);
        ggml_backend_tensor_set(ids, routes.data(), 0, sizeof(routes));
        auto compute = [&](bool leased, bool chunked) {
            setenv("ESE_CPU_EXPERT_LEASE_CHUNKING", chunked ? "1" : "0", 1);
            auto plan = ggml_graph_plan(graph, 8);
            std::vector<uint8_t> work(plan.work_size);
            plan.work_data = work.data();
            if (leased) {
                plan.expert_lease_acquire = fixture::acquire; plan.expert_lease_release = fixture::release;
                plan.expert_lease_user_data = &f; plan.expert_lease_required = true;
            }
            REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
            std::vector<float> result(size_t(ggml_nelements(y)));
            std::memcpy(result.data(), y->data, ggml_nbytes(y));
            return result;
        };
        const auto uncached = compute(false, false);
        // Every successful cached calculation must use leases, not these zeros.
        std::memset(w->data, 0, ggml_nbytes(w));
        f.cache = std::make_unique<ram_cache>(cache_config{uint64_t(experts)*w->nb[2], w->nb[2]},
                std::vector<std::shared_ptr<const source>>{storage});
        const auto reference = compute(true, false);
        if (s.type != GGML_TYPE_F32) REQUIRE(reference == uncached);
        for (int slots : {1, 4}) {
            f.cache = std::make_unique<ram_cache>(cache_config{uint64_t(slots)*w->nb[2], w->nb[2]},
                    std::vector<std::shared_ptr<const source>>{storage});
            const uint64_t old_acquires = f.acquired.load();
            f.peak = 0; f.refused = 0;
            REQUIRE(compute(true, true) == reference);
            REQUIRE(f.active == 0 && f.acquired == f.released);
            REQUIRE(f.cache->stats().active_leases == 0 && f.cache->stats().reserved_bytes == 0);
            REQUIRE(f.cache->stats().resident_bytes <= uint64_t(slots)*w->nb[2]);
#if defined(ESE_TEST_EXPERT_LEASE_CHUNKING)
            REQUIRE(f.acquired - old_acquires == experts);
            REQUIRE(f.peak <= uint64_t(slots));
            if (slots == 1) REQUIRE(f.refused > 0);
#endif
            REQUIRE(f.cache->stats().evictions > 0);
            // Reuse the node after both grouped and tiny-cache forced evictions.
            REQUIRE(compute(true, true) == reference);
            REQUIRE(f.active == 0 && f.acquired == f.released);
            std::cout << "Leased CPU " << ggml_type_name(s.type) << ", slots=" << slots
                      << ": exact, acquisitions=" << f.acquired - old_acquires
                      << ", peak=" << f.peak << ", refused=" << f.refused << '\n';
        }
        f.cache.reset();
    }
#else
    std::cout << "SKIP: CPU lease chunks require IQK, expert chunking and POSIX environment controls\n";
#endif
}

void test_route_partition_is_complementary() {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct cleanup { ggml_context * ctx; ~cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    const std::array<int32_t, 8> route = {{ 7, 3, 5, 1, 2, -1, 6, 4 }};
    std::memcpy(ids->data, route.data(), sizeof(route));
    ggml_tensor * gpu_ids = ggml_ese_route_partition(ctx, ids, 0, 2);
    ggml_tensor * cpu_ids = ggml_ese_route_partition(ctx, ids, 2, 2);
    REQUIRE(ggml_ese_route_get_role(gpu_ids) == GGML_ESE_ROUTE_GPU);
    REQUIRE(ggml_ese_route_get_role(cpu_ids) == GGML_ESE_ROUTE_CPU);
    ggml_tensor * activations = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 4, 2);
    ggml_tensor * biases = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 8);
    std::fill_n(static_cast<float *>(activations->data), ggml_nelements(activations), 0.0f);
    for (int expert = 0; expert < 8; ++expert) {
        static_cast<float *>(biases->data)[2*expert + 0] = float(expert + 1);
        static_cast<float *>(biases->data)[2*expert + 1] = float(100 + expert);
    }
    ggml_tensor * biased_cpu = ggml_add_id(ctx, activations, biases, cpu_ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, gpu_ids);
    ggml_build_forward_expand(graph, cpu_ids);
    ggml_build_forward_expand(graph, biased_cpu);
    ggml_cplan plan = ggml_graph_plan(graph, 2);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const auto * gpu = static_cast<const int32_t *>(gpu_ids->data);
    const auto * cpu = static_cast<const int32_t *>(cpu_ids->data);
    for (size_t i = 0; i < route.size(); ++i) {
        const bool gpu_position = i%4 < 2;
        REQUIRE(gpu[i] == (gpu_position ? route[i] : -1));
        REQUIRE(cpu[i] == (gpu_position ? -1 : route[i]));
        if (route[i] >= 0) {
            REQUIRE((gpu[i] >= 0) != (cpu[i] >= 0));
        }
        const float * biased = static_cast<const float *>(biased_cpu->data) + 2*i;
        if (gpu_position || route[i] < 0) {
            REQUIRE(biased[0] == 0.0f && biased[1] == 0.0f);
        } else {
            REQUIRE(biased[0] == float(route[i] + 1));
            REQUIRE(biased[1] == float(100 + route[i]));
        }
    }
}

void test_masked_route_preserves_weighted_positions() {
    ggml_init_params params = {1024*1024, nullptr, false};
    auto * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct cleanup { ggml_context * ctx; ~cleanup() { ggml_free(ctx); } } free_ctx{ctx};
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    // Distinct weights, padding and a repeated expert make reordering only IDs,
    // dropping a route position, or evaluating a position twice observable.
    const std::array<int32_t, 8> route = {{7, 3, 5, 1, 2, -1, 6, 2}};
    std::memcpy(ids->data, route.data(), sizeof(route));
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    auto * matrix = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 8, 8);
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 1, 2);
    auto * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 4, 2);
    for (int i = 0; i < ggml_nelements(matrix); ++i)
        static_cast<float *>(matrix->data)[i] = float((i/256 + 1)*(i%3 - 1))/8.0f;
    for (int i = 0; i < ggml_nelements(input); ++i)
        static_cast<float *>(input->data)[i] = float(i%7 - 3)/16.0f;
    for (int i = 0; i < 8; ++i) static_cast<float *>(weights->data)[i] = float(i + 1)/16.0f;
    auto * gpu_ids = ggml_ese_route_partition_masked(ctx, ids, mask, GGML_ESE_ROUTE_GPU);
    auto * cpu_ids = ggml_ese_route_partition_masked(ctx, ids, mask, GGML_ESE_ROUTE_CPU);
    auto * full = ggml_mul(ctx, ggml_mul_mat_id(ctx, matrix, input, ids), weights);
    auto * split = ggml_add(ctx,
            ggml_mul(ctx, ggml_mul_mat_id(ctx, matrix, input, gpu_ids), weights),
            ggml_mul(ctx, ggml_mul_mat_id(ctx, matrix, input, cpu_ids), weights));
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, full);
    ggml_build_forward_expand(graph, split);
    auto plan = ggml_graph_plan(graph, 2);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    for (unsigned pattern : {0u, 255u, 170u, 85u, 132u, 57u, 16u, 64u}) {
        for (int i = 0; i < 8; ++i) static_cast<int32_t *>(mask->data)[i] = (pattern >> i)&1;
        REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
        REQUIRE(std::memcmp(full->data, split->data, ggml_nbytes(full)) == 0);
        const auto * gpu = static_cast<const int32_t *>(gpu_ids->data);
        const auto * cpu = static_cast<const int32_t *>(cpu_ids->data);
        for (size_t i = 0; i < route.size(); ++i) {
            if (route[i] >= 0) REQUIRE((gpu[i] == route[i]) != (cpu[i] == route[i]));
            else REQUIRE(gpu[i] == -1 && cpu[i] == -1);
        }
    }
}

std::vector<float> compute_cpu_fused_moe_layout(
        bool merged,
        const std::vector<ggml_fp16_t> & up_weights,
        const std::vector<ggml_fp16_t> & gate_weights,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 384;
    constexpr int64_t m = 128;
    constexpr int64_t n_experts = 2;
    constexpr int64_t n_used = 2;
    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    struct backend_cleanup { ggml_backend_t value; ~backend_cleanup() { ggml_backend_free(value); } } free_backend{backend};

    ggml_init_params params = {
        ggml_tensor_overhead()*20 + ggml_graph_overhead_custom(16, false), nullptr, true
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * value; ~context_cleanup() { ggml_free(value); } } free_ctx{ctx};

    ggml_tensor * up = ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, k, merged ? 2*m : m, n_experts);
    ggml_tensor * gate = merged ? nullptr : ggml_new_tensor_3d(
            ctx, GGML_TYPE_F16, k, m, n_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_moe_up_gate(ctx, up, gate, input, ids, GGML_UNARY_OP_SILU);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    struct buffer_cleanup {
        ggml_backend_buffer_t value;
        ~buffer_cleanup() { ggml_backend_buffer_free(value); }
    } free_buffer{buffer};

    REQUIRE(up_weights.size() == size_t(k*m*n_experts));
    REQUIRE(gate_weights.size() == up_weights.size());
    if (merged) {
        std::vector<ggml_fp16_t> merged_weights(2*up_weights.size());
        const size_t expert_elements = size_t(k*m);
        for (int expert = 0; expert < n_experts; ++expert) {
            auto * destination = merged_weights.data() + size_t(expert)*2*expert_elements;
            std::copy_n(gate_weights.data() + size_t(expert)*expert_elements, expert_elements, destination);
            std::copy_n(up_weights.data() + size_t(expert)*expert_elements, expert_elements,
                    destination + expert_elements);
        }
        ggml_backend_tensor_set(up, merged_weights.data(), 0, merged_weights.size()*sizeof(merged_weights[0]));
    } else {
        ggml_backend_tensor_set(up, up_weights.data(), 0, up_weights.size()*sizeof(up_weights[0]));
        ggml_backend_tensor_set(gate, gate_weights.data(), 0, gate_weights.size()*sizeof(gate_weights[0]));
    }
    REQUIRE(input_values.size() == size_t(k*n_used));
    const std::array<int32_t, n_used> route = {{1, 0}};
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(ids, route.data(), 0, sizeof(route));

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

void test_cpu_fused_moe_merged_workspace() {
    constexpr int64_t k = 384;
    constexpr int64_t m = 128;
    constexpr int64_t n_experts = 2;
    std::vector<ggml_fp16_t> up(size_t(k*m*n_experts));
    std::vector<ggml_fp16_t> gate(up.size());
    for (size_t i = 0; i < up.size(); ++i) {
        up[i] = ggml_fp32_to_fp16(float(int((i*17 + i/31) % 257) - 128)/64.0f);
        gate[i] = ggml_fp32_to_fp16(float(int((i*29 + i/19) % 251) - 125)/64.0f);
    }
    std::vector<float> input(size_t(k*2));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = float(int((i*13) % 97) - 48)/24.0f;
    }
    const auto separate = compute_cpu_fused_moe_layout(false, up, gate, input);
    const auto merged = compute_cpu_fused_moe_layout(true, up, gate, input);
    REQUIRE(merged == separate);
}

std::vector<float> compute_cuda_mul_mat_id(
        ggml_backend_t backend,
        const std::vector<ggml_fp16_t> & weights,
        int64_t stored_experts,
        int64_t logical_experts,
        const std::array<int32_t, 2> & route,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 768;
    constexpr int64_t m = 384;
    constexpr int64_t n_used = 2;

    ggml_init_params params = {
        ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * tensor_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_mul_mat_id(ctx, tensor_weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    struct buffer_cleanup {
        ggml_backend_buffer_t buffer;
        ~buffer_cleanup() { ggml_backend_buffer_free(buffer); }
    } free_buffer{buffer};

    REQUIRE(weights.size() == size_t(k*m*stored_experts));
    REQUIRE(input_values.size() == size_t(k*n_used));
    ggml_backend_tensor_set(tensor_weights, weights.data(), 0, weights.size()*sizeof(weights[0]));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(ids, route.data(), 0, sizeof(route));

    // Reproduce the scheduler's redirected tensor geometry.  A compact cache
    // can expose either the model's logical width or a larger capacity-sized
    // physical slot width, including its matching fourth-dimension stride.
    tensor_weights->ne[2] = logical_experts;
    tensor_weights->nb[3] = tensor_weights->nb[2]*size_t(logical_experts);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

std::vector<float> compute_hybrid_mul_mat_id(
        ggml_backend_t cuda_backend,
        const std::vector<ggml_fp16_t> & weights,
        int32_t gpu_expert,
        const std::array<int32_t, 2> & cpu_route,
        const std::array<int32_t, 2> & gpu_route,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 768;
    constexpr int64_t m = 384;
    constexpr int64_t n_experts = 8;
    constexpr int64_t n_used = 2;
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    REQUIRE(cpu_backend != nullptr);
    struct cpu_cleanup { ggml_backend_t value; ~cpu_cleanup() { ggml_backend_free(value); } } free_cpu{cpu_backend};

    ggml_init_params params = {
        ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(32, false), nullptr, true
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * value; ~context_cleanup() { ggml_free(value); } } free_ctx{ctx};

    ggml_tensor * cpu_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, n_experts);
    ggml_tensor * gpu_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * cpu_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * gpu_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_set_input(input);
    ggml_set_input(cpu_ids);
    ggml_set_input(gpu_ids);
    ggml_tensor * cpu_output = ggml_mul_mat_id(ctx, cpu_weights, input, cpu_ids);
    ggml_tensor * gpu_output = ggml_mul_mat_id(ctx, gpu_weights, input, gpu_ids);
    ggml_tensor * branches[] = {gpu_output, cpu_output};
    ggml_tensor * output = ggml_ese_reduce_to(ctx, branches, 2, GGML_OP_ADD, 0);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = {cuda_backend, cpu_backend};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, 64, true);
    REQUIRE(scheduler != nullptr);
    struct scheduler_cleanup {
        ggml_backend_sched_t value;
        ~scheduler_cleanup() { ggml_backend_sched_free(value); }
    } free_scheduler{scheduler};
    ggml_backend_sched_set_split_mode_graph(scheduler, true, true);
    ggml_backend_sched_set_tensor_backend(scheduler, cpu_weights, cpu_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, gpu_weights, cuda_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, cpu_output, cpu_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, gpu_output, cuda_backend);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));

    REQUIRE(weights.size() == size_t(k*m*n_experts));
    REQUIRE(gpu_expert >= 0 && gpu_expert < n_experts);
    REQUIRE(input_values.size() == size_t(k*n_used));
    ggml_backend_tensor_set(cpu_weights, weights.data(), 0, weights.size()*sizeof(weights[0]));
    ggml_backend_tensor_set(gpu_weights,
            weights.data() + size_t(gpu_expert)*size_t(k*m), 0,
            size_t(k*m)*sizeof(weights[0]));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(cpu_ids, cpu_route.data(), 0, sizeof(cpu_route));
    ggml_backend_tensor_set(gpu_ids, gpu_route.data(), 0, sizeof(gpu_route));
    REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(scheduler);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

std::vector<float> compute_cuda_fused_moe(
        ggml_backend_t backend,
        const std::vector<ggml_fp16_t> & up_weights,
        const std::vector<ggml_fp16_t> & gate_weights,
        int64_t stored_experts,
        int64_t logical_experts,
        const std::array<int32_t, 2> & route,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 384;
    constexpr int64_t m = 768;
    constexpr int64_t n_used = 2;

    ggml_init_params params = {
        ggml_tensor_overhead()*20 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_moe_up_gate(ctx, up, gate, input, ids, GGML_UNARY_OP_SILU);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    struct buffer_cleanup {
        ggml_backend_buffer_t buffer;
        ~buffer_cleanup() { ggml_backend_buffer_free(buffer); }
    } free_buffer{buffer};

    REQUIRE(up_weights.size() == size_t(k*m*stored_experts));
    REQUIRE(gate_weights.size() == up_weights.size());
    ggml_backend_tensor_set(up, up_weights.data(), 0, up_weights.size()*sizeof(up_weights[0]));
    ggml_backend_tensor_set(gate, gate_weights.data(), 0, gate_weights.size()*sizeof(gate_weights[0]));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(ids, route.data(), 0, sizeof(route));

    up->ne[2] = gate->ne[2] = logical_experts;
    up->nb[3] = up->nb[2]*size_t(logical_experts);
    gate->nb[3] = gate->nb[2]*size_t(logical_experts);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

// Verify the CUDA kernels read nb[2], including the quantized fused path.
// Poison between experts makes accidentally using a packed stride observable.
std::vector<float> compute_cuda_padded_experts(
        ggml_backend_t backend, ggml_type type, bool fused, size_t padding, int64_t tokens = 1,
        bool sparse = false, bool fuse_down = false, int64_t experts = 4, int32_t slot_shift = 0) {
    constexpr int64_t k = 768, m = 128;
    REQUIRE(experts >= 4 + slot_shift && slot_shift >= 0);
    ggml_init_params params = {ggml_tensor_overhead()*20 + ggml_graph_overhead_custom(16, false), nullptr, true};
    auto * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};
    auto * up = ggml_new_tensor_3d(ctx, type, k, m, experts);
    auto * gate = fused ? ggml_new_tensor_3d(ctx, type, k, m, experts) : nullptr;
    auto * down = fuse_down ? ggml_new_tensor_3d(ctx, GGML_TYPE_IQ4_NL, m, k, experts) : nullptr;
    up->nb[2] += padding;
    up->nb[3] = up->nb[2]*experts;
    if (gate) { gate->nb[2] = up->nb[2]; gate->nb[3] = up->nb[3]; }
    const int64_t used = sparse ? 10 : 2;
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, sparse && !fused ? used : 1, tokens);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, tokens);
    auto * output = fused ? ggml_moe_up_gate(ctx, up, gate, input, ids, GGML_UNARY_OP_SILU)
                         : ggml_mul_mat_id(ctx, up, input, ids);
    auto * hidden = output;
    if (down) output = ggml_mul_mat_id(ctx, down, hidden, ids);
    auto * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    struct buffer_cleanup { ggml_backend_buffer_t buffer; ~buffer_cleanup() { ggml_backend_buffer_free(buffer); } } free_buffer{buffer};
    ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_EXPERT_CACHE);
    std::vector<float> importance(k, 1.0f);
    struct retained_tail { ggml_tensor view; size_t offset; std::vector<uint8_t> bytes; };
    std::vector<retained_tail> tails;
    for (auto * tensor : {up, gate, down}) {
        if (!tensor) continue;
        const size_t tensor_payload = tensor->nb[2] - (tensor == down ? 0 : padding);
        std::vector<float> values(size_t(tensor->ne[0]*tensor->ne[1]*experts));
        for (size_t i = 0; i < values.size(); ++i) {
            const size_t shift = size_t(slot_shift*tensor->ne[0]*tensor->ne[1]);
            const size_t semantic_index = i >= shift ? i - shift : 0;
            values[i] = float(int((semantic_index*(tensor == up ? 7 : 11)) % 37) - 18)/128.0f;
        }
        std::vector<uint8_t> packed(tensor_payload*experts);
        REQUIRE(ggml_quantize_chunk(tensor->type, values.data(), packed.data(), 0, tensor->ne[1]*experts, tensor->ne[0],
                importance.data(), nullptr) == packed.size());
        std::vector<uint8_t> storage(ggml_nbytes(tensor), 0xff);
        for (size_t expert = 0; expert < experts; ++expert) {
            std::memcpy(storage.data() + expert*tensor->nb[2], packed.data() + expert*tensor_payload, tensor_payload);
        }
        ggml_backend_tensor_set(tensor, storage.data(), 0, storage.size());
        const size_t span = ggml_nbytes(tensor);
        const size_t allocation = ggml_backend_buffer_get_alloc_size(buffer, tensor);
        if (allocation > span) {
            // A shorter alias can end inside a larger retained payload. Use
            // finite quantization bytes there and verify neither alias init
            // nor indexed execution clears them. The cache owns this region.
            retained_tail tail{*tensor, span, std::vector<uint8_t>(allocation - span, 0x3f)};
            tail.view.type = GGML_TYPE_I8;
            tail.view.ne[0] = int64_t(allocation);
            tail.view.ne[1] = tail.view.ne[2] = tail.view.ne[3] = 1;
            tail.view.nb[0] = 1;
            tail.view.nb[1] = tail.view.nb[2] = tail.view.nb[3] = allocation;
            ggml_backend_tensor_set(&tail.view, tail.bytes.data(), span, tail.bytes.size());
            ggml_backend_buffer_init_tensor(buffer, tensor);
            tails.push_back(std::move(tail));
        }
    }
    std::vector<float> input_values(size_t(ggml_nelements(input)));
    for (size_t i = 0; i < input_values.size(); ++i) input_values[i] = float(int((i % k) % 13) - 6)/16.0f;
    std::vector<int32_t> route(size_t(used*tokens));
    for (size_t i = 0; i < route.size(); ++i)
        route[i] = sparse ? (i%used == 0 ? 3 : i%used == 8 ? 1 : -1) : i % 2 ? 1 : 3;
    for (auto & id : route) if (id >= 0) id += slot_shift;
    ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));
    ggml_backend_tensor_set(ids, route.data(), 0, route.size()*sizeof(int32_t));
    if (sparse) {
        for (auto * tensor : {hidden, output}) {
            std::vector<float> poison(size_t(ggml_nelements(tensor)), std::numeric_limits<float>::quiet_NaN());
            ggml_backend_tensor_set(tensor, poison.data(), 0, ggml_nbytes(tensor));
        }
    }
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, ggml_nbytes(output));
    for (auto & tail : tails) {
        std::vector<uint8_t> actual(tail.bytes.size());
        ggml_backend_tensor_get(&tail.view, actual.data(), tail.offset, actual.size());
        REQUIRE(actual == tail.bytes);
    }
    REQUIRE(std::all_of(result.begin(), result.end(), [](float x) { return std::isfinite(x); }));
    if (sparse) {
        for (size_t i = 0; i < route.size(); ++i) {
            if (route[i] < 0) {
                for (int64_t row = 0; row < output->ne[0]; ++row)
                    REQUIRE(result[i*size_t(output->ne[0]) + size_t(row)] == 0.0f);
            }
        }
    }
    return result;
}

void test_cuda_router_precedes_compact_staging() {
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return;
    constexpr int width = 256, rows = 8, experts = 16, used = 10;
    auto * cuda = ggml_backend_cuda_init(0, nullptr, nullptr);
    auto * cpu = ggml_backend_cpu_init();
    REQUIRE(cuda && cpu);
    struct backend_cleanup {
        ggml_backend_t cuda, cpu;
        ~backend_cleanup() { ggml_backend_free(cuda); ggml_backend_free(cpu); }
    } free_backends{cuda, cpu};
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    REQUIRE(ctx);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};
    auto * weight = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, rows, experts);
    ggml_set_name(weight, "blk.0.ffn_up_exps.weight");
    auto * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, 1);
    auto * router = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 512, 1);
    ggml_set_input(input); ggml_set_input(router);
    std::vector<ggml_backend_buffer_t> buffers;
    struct buffer_cleanup {
        std::vector<ggml_backend_buffer_t> & buffers;
        ~buffer_cleanup() { for (auto * buffer : buffers) ggml_backend_buffer_free(buffer); }
    } free_buffers{buffers};
    for (auto * tensor : {weight, input, router}) {
        auto buft = ggml_backend_get_default_buffer_type(tensor == weight ? cpu : cuda);
        auto buffer = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
        REQUIRE(buffer); buffers.push_back(buffer);
        ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
        if (tensor == weight) {
            ggml_backend_buffer_set_usage(buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(buffer, 0xff); // Leases must supply every actual weight byte.
        }
    }
    auto * ids = ggml_top_k(ctx, router, used);
    REQUIRE(ids->nb[1] == 512*sizeof(int32_t));
    auto * output = ggml_mul_mat_id(ctx, weight, input, ids);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);
    ggml_backend_t backends[] = {cuda, cpu};
    auto * scheduler = ggml_backend_sched_new(backends, nullptr, 2, 128, false);
    REQUIRE(scheduler);
    struct scheduler_cleanup {
        ggml_backend_sched_t scheduler;
        ~scheduler_cleanup() { if (scheduler) ggml_backend_sched_free(scheduler); }
    };
    struct provider {
        std::vector<ggml_fp16_t> values;
        ggml_tensor * ids;
        const int32_t * expected = nullptr;
        size_t acquires = 0, releases = 0, active = 0;
        struct lease { provider * owner; };
        static bool acquire(void * opaque, int layer, int expert, int component,
                ggml_backend_expert_lease * result) {
            auto & source = *static_cast<provider *>(opaque);
            REQUIRE(layer == 0 && component == 1 && expert >= 0 && expert < experts);
            // This callback runs during copy_inputs, before expert graph_compute.
            // Its IDs must already be the computed TOP_K, not graph-arena bytes.
            std::array<int32_t, used> actual;
            ggml_backend_tensor_get(source.ids, actual.data(), 0, sizeof(actual));
            REQUIRE(std::equal(actual.begin(), actual.end(), source.expected));
            auto * handle = new lease{&source};
            result->data = source.values.data() + size_t(expert)*width*rows;
            result->size = width*rows*sizeof(ggml_fp16_t);
            result->handle = handle;
            ++source.acquires; ++source.active;
            return true;
        }
        static void release(void * opaque, void * handle) {
            auto & source = *static_cast<provider *>(opaque);
            std::unique_ptr<lease> held(static_cast<lease *>(handle));
            REQUIRE(held->owner == &source && source.active > 0);
            ++source.releases; --source.active;
        }
    } source;
    scheduler_cleanup free_scheduler{scheduler};
    source.ids = ids;
    source.values.resize(width*rows*experts);
    for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < rows; ++row) {
        const auto value = ggml_fp32_to_fp16(float((expert + 1)*(row + 1))/256.f);
        std::fill_n(source.values.data() + (expert*rows + row)*width, width, value);
    }
    ggml_backend_sched_set_active_expert_decode(scheduler, true);
    ggml_backend_sched_set_only_active_experts(scheduler, true);
    ggml_backend_sched_set_expert_lease_callbacks(scheduler, provider::acquire, provider::release, &source, true);
    const uint64_t capacity = 64*256 + 13*weight->nb[2] + 256;
    ggml_backend_sched_set_expert_cache_capacity(scheduler, capacity, 0, 1);
    for (auto * tensor : {input, router, ids, output}) ggml_backend_sched_set_tensor_backend(scheduler, tensor, cuda);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));
    REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler, output) == cuda);
    REQUIRE(output->src[0]->ne[2] < experts); // Exercise a real bounded compact copy.
    const std::array<std::array<int32_t, used>, 4> routes = {{
        {{10,5,0,8,11,3,14,4,6,2}}, {{10,5,0,8,11,3,14,4,6,2}},
        {{15,12,13,7,1,10,5,0,8,11}}, {{10,5,0,8,11,3,14,4,6,2}},
    }};
    // Keep every partial sum representable even in the default F16 reduction.
    // A dense 256-term sum can round despite exact weights/products, which
    // would conflate arithmetic precision with router/cache ordering.
    std::vector<float> input_values(width, 0.f);
    std::fill_n(input_values.begin(), 16, 1.f/16);
    for (size_t pass = 0; pass < routes.size(); ++pass) {
        source.expected = routes[pass].data();
        const auto before = source.acquires;
        std::vector<float> logits(512, -1000.f);
        for (int position = 0; position < used; ++position) logits[routes[pass][position]] = 100.f - position;
        ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(float));
        ggml_backend_tensor_set(router, logits.data(), 0, logits.size()*sizeof(float));
        // The old scheduler reads these arbitrary negative F32 bit patterns
        // before its same-split ARGSORT producer has executed, then aborts.
        std::array<int32_t, used> poison;
        poison.fill(-1070692928);
        ggml_backend_tensor_set(ids, poison.data(), 0, sizeof(poison));
        REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(scheduler);
        std::array<int32_t, used> computed;
        ggml_backend_tensor_get(ids, computed.data(), 0, sizeof(computed));
        REQUIRE(computed == routes[pass]);
        std::vector<float> actual(rows*used);
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
        for (int position = 0; position < used; ++position) for (int row = 0; row < rows; ++row) {
            // Constant F16 rows dotted with 16 copies of 1/16 and zero padding.
            // This oracle needs neither production decoding nor another backend.
            const double expected = double((routes[pass][position] + 1)*(row + 1))/256;
            if (!std::isfinite(actual[position*rows + row]) ||
                    std::abs(actual[position*rows + row] - expected) > 2e-5*(1 + std::abs(expected))) {
                std::cerr << "router-cache mismatch: pass=" << pass << " position=" << position
                          << " expert=" << routes[pass][position] << " row=" << row
                          << " actual=" << actual[position*rows + row] << " expected=" << expected << '\n';
            }
            REQUIRE(std::isfinite(actual[position*rows + row]));
            REQUIRE(std::abs(actual[position*rows + row] - expected) <= 2e-5*(1 + std::abs(expected)));
        }
        if (pass == 1) REQUIRE(source.acquires == before);
        std::array<ggml_backend_sched_resource_device_stats, 2> stats;
        REQUIRE(ggml_backend_sched_get_resource_device_stats(scheduler, stats.data(), stats.size()));
        REQUIRE(stats[0].expert_cache_allocated_bytes <= capacity);
    }
    // Release the scheduler before checking its pending upload leases.
    ggml_backend_sched_free(scheduler); free_scheduler.scheduler = nullptr;
    REQUIRE(source.acquires > 0 && source.acquires == source.releases && source.active == 0);
#endif
}

void test_cuda_stable_expert_reduction() {
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return;
    auto * backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    REQUIRE(backend != nullptr);
    struct cleanup { ggml_backend_t backend; ~cleanup() { ggml_backend_free(backend); } } free_backend{backend};
    // More than eight route positions forces indexed MMQ. Poisoned negative
    // positions and slot shifts check both the virtual empty bounds and the
    // independence of reduction arithmetic from the expert table geometry.
    for (const auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_Q4_0, GGML_TYPE_IQ3_XXS}) {
        for (const int tokens : {1, 2}) {
            const auto reference = compute_cuda_padded_experts(backend, type, false, 0, tokens, true);
            for (const int count : {17, 64, 97}) {
                REQUIRE(compute_cuda_padded_experts(backend, type, false, 4096, tokens, true, false,
                        count, count - 4) == reference);
            }
        }
    }
#endif
}

void test_cuda_sparse_quantized_routes() {
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return;
    auto * backend = ggml_backend_cuda_init(0, nullptr, nullptr);
    REQUIRE(backend != nullptr);
    struct cleanup { ggml_backend_t backend; ~cleanup() { ggml_backend_free(backend); } } free_backend{backend};
    // Top-10 reaches indexed MMQ for down projections; fused up/gate reaches
    // MMVQ and its fused successor. Poison destination rows before each run.
    for (auto type : {GGML_TYPE_IQ4_NL, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ1_S}) {
        for (bool fused : {false, true}) {
            for (int64_t tokens : {1, 2}) {
                std::cout << "Sparse CUDA case: type=" << ggml_type_name(type)
                          << " fused=" << fused << " tokens=" << tokens << std::endl;
                const auto sparse = compute_cuda_padded_experts(backend, type, fused, 4096, tokens, true);
                const auto dense = compute_cuda_padded_experts(backend, type, fused, 0, tokens);
                float max_error = 0;
                for (int64_t token = 0; token < tokens; ++token) {
                    for (int expert = 0; expert < 2; ++expert) {
                        for (int row = 0; row < 128; ++row) {
                            const size_t sparse_offset = size_t((token*10 + (expert ? 8 : 0))*128 + row);
                            const size_t dense_offset = size_t((token*2 + expert)*128 + row);
                            max_error = std::max(max_error, std::abs(sparse[sparse_offset] - dense[dense_offset]));
                        }
                    }
                }
                // Different MMVQ/MMQ tile widths can change accumulation;
                // valid expert rows must still agree with the dense control.
                std::cout << "Sparse/dense valid-row error: " << max_error << std::endl;
                REQUIRE(max_error < 2e-3f);
            }
        }
    }
    compute_cuda_padded_experts(backend, GGML_TYPE_IQ3_XXS, true, 4096, 1, true, true);
#endif
}

void test_cuda_compact_remap_exact_parity() {
    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        if (std::strncmp(ggml_backend_reg_get_name(i), "CUDA", 4) == 0) {
            backend = ggml_backend_reg_init_backend(i, nullptr);
            if (backend) break;
        }
    }
    if (!backend) return;
    struct backend_cleanup { ggml_backend_t backend; ~backend_cleanup() { ggml_backend_free(backend); } } free_backend{backend};

    for (auto type : {GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q2_K}) {
        for (bool fused : {false, true}) {
            const auto packed = compute_cuda_padded_experts(backend, type, fused, 0);
            const auto padded = compute_cuda_padded_experts(backend, type, fused, 4096);
            REQUIRE(packed == padded);
        }
        // Two tokens exercise indexed MMQ instead of the one-token MMVQ
        // shortcut, including its temporary-compute padding-clear guard.
        REQUIRE(compute_cuda_padded_experts(backend, type, false, 0, 2) ==
                compute_cuda_padded_experts(backend, type, false, 4096, 2));
    }

    constexpr int64_t k = 768;
    constexpr int64_t m = 384;
    constexpr int64_t n_experts = 8;
    constexpr int64_t n_used = 2;
    std::vector<ggml_fp16_t> full_weights(size_t(k*m*n_experts));
    for (size_t i = 0; i < full_weights.size(); ++i) {
        full_weights[i] = ggml_fp32_to_fp16(float(int((i*17 + i/31) % 257) - 128)/64.0f);
    }
    std::vector<float> input_values(size_t(k*n_used));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = float(int((i*13) % 97) - 48)/24.0f;
    }

    const std::array<int32_t, n_used> route = {{ 6, 1 }};
    const std::array<int32_t, n_used> remapped = {{ 0, 1 }};
    std::vector<ggml_fp16_t> compact_weights(size_t(k*m*n_used));
    for (int slot = 0; slot < n_used; ++slot) {
        std::copy_n(full_weights.data() + size_t(route[slot])*size_t(k*m), size_t(k*m),
                compact_weights.data() + size_t(slot)*size_t(k*m));
    }

    const auto full = compute_cuda_mul_mat_id(
            backend, full_weights, n_experts, n_experts, route, input_values);
    const auto compact = compute_cuda_mul_mat_id(
            backend, compact_weights, n_used, n_experts, remapped, input_values);
    REQUIRE(compact == full);

    // Frontier MoE models can expose hundreds of logical experts while the
    // bounded cache keeps only the routed experts resident.  The redirected
    // tensor must accept that compact physical geometry after ids are remapped.
    constexpr int64_t large_logical_experts = 256;
    const auto compact_large_logical = compute_cuda_mul_mat_id(
            backend, compact_weights, n_used, large_logical_experts, remapped, input_values);
    REQUIRE(compact_large_logical == full);

    // A capacity-sized cache may retain entries from multiple logical layers,
    // so a remapped slot can be greater than the model's expert count.  The
    // redirected tensor must expose the physical slot geometry to the kernel.
    constexpr int64_t extended_slots = 10;
    const std::array<int32_t, n_used> extended_route = {{ 8, 9 }};
    std::vector<ggml_fp16_t> extended_weights(size_t(k*m*extended_slots), ggml_fp32_to_fp16(0.0f));
    for (int slot = 0; slot < n_used; ++slot) {
        std::copy_n(full_weights.data() + size_t(route[slot])*size_t(k*m), size_t(k*m),
                extended_weights.data() + size_t(extended_route[slot])*size_t(k*m));
    }
    const auto extended = compute_cuda_mul_mat_id(
            backend, extended_weights, extended_slots, extended_slots, extended_route, input_values);
    REQUIRE(extended == full);

    // Negative route sentinels turn unassigned route positions into exact
    // zeroes. This permits independent CPU and GPU branches to execute on the
    // same activation and sum back to the unsplit routed result.
    const auto hybrid = compute_hybrid_mul_mat_id(
            backend, full_weights, route[1], {{route[0], -1}}, {{-1, 0}}, input_values);
    double squared_hybrid_error = 0;
    double squared_full = 0;
    for (size_t i = 0; i < full.size(); ++i) {
        REQUIRE(std::isfinite(hybrid[i]));
        const double error = double(hybrid[i]) - full[i];
        squared_hybrid_error += error*error;
        squared_full += double(full[i])*full[i];
        if (i >= size_t(m)) {
            // The GPU-owned route position is the same CUDA kernel plus an
            // exact CPU zero, so a masked branch must not perturb it at all.
            REQUIRE(hybrid[i] == full[i]);
        }
    }
    const double hybrid_nrmse = std::sqrt(squared_hybrid_error/squared_full);
    REQUIRE(hybrid_nrmse <= 5.0e-3);

    constexpr int64_t fused_k = 384;
    constexpr int64_t fused_m = 768;
    std::vector<ggml_fp16_t> full_fused_up(size_t(fused_k*fused_m*n_experts));
    std::vector<ggml_fp16_t> full_fused_gate(size_t(fused_k*fused_m*n_experts));
    for (size_t i = 0; i < full_fused_up.size(); ++i) {
        full_fused_up[i] = ggml_fp32_to_fp16(float(int((i*17 + i/31) % 257) - 128)/64.0f);
        full_fused_gate[i] = ggml_fp32_to_fp16(float(int((i*29 + i/19) % 251) - 125)/64.0f);
    }
    std::vector<ggml_fp16_t> compact_fused_up(size_t(fused_k*fused_m*n_used));
    std::vector<ggml_fp16_t> compact_fused_gate(size_t(fused_k*fused_m*n_used));
    for (int slot = 0; slot < n_used; ++slot) {
        std::copy_n(full_fused_up.data() + size_t(route[slot])*size_t(fused_k*fused_m), size_t(fused_k*fused_m),
                compact_fused_up.data() + size_t(slot)*size_t(fused_k*fused_m));
        std::copy_n(full_fused_gate.data() + size_t(route[slot])*size_t(fused_k*fused_m), size_t(fused_k*fused_m),
                compact_fused_gate.data() + size_t(slot)*size_t(fused_k*fused_m));
    }
    std::vector<float> fused_input(size_t(fused_k*n_used));
    for (size_t i = 0; i < fused_input.size(); ++i) {
        fused_input[i] = float(int((i*13) % 97) - 48)/24.0f;
    }
    const auto full_fused = compute_cuda_fused_moe(
            backend, full_fused_up, full_fused_gate, n_experts, n_experts, route, fused_input);
    const auto compact_fused = compute_cuda_fused_moe(
            backend, compact_fused_up, compact_fused_gate, n_used, n_experts, remapped, fused_input);
    REQUIRE(compact_fused == full_fused);

    std::vector<ggml_fp16_t> extended_fused_up(
            size_t(fused_k*fused_m*extended_slots), ggml_fp32_to_fp16(0.0f));
    std::vector<ggml_fp16_t> extended_fused_gate(
            size_t(fused_k*fused_m*extended_slots), ggml_fp32_to_fp16(0.0f));
    for (int slot = 0; slot < n_used; ++slot) {
        std::copy_n(full_fused_up.data() + size_t(route[slot])*size_t(fused_k*fused_m), size_t(fused_k*fused_m),
                extended_fused_up.data() + size_t(extended_route[slot])*size_t(fused_k*fused_m));
        std::copy_n(full_fused_gate.data() + size_t(route[slot])*size_t(fused_k*fused_m), size_t(fused_k*fused_m),
                extended_fused_gate.data() + size_t(extended_route[slot])*size_t(fused_k*fused_m));
    }
    const auto extended_fused = compute_cuda_fused_moe(
            backend, extended_fused_up, extended_fused_gate,
            extended_slots, extended_slots, extended_route, fused_input);
    REQUIRE(extended_fused == full_fused);
}

void test_hybrid_runtime_guard_is_one_way_and_fail_closed() {
    ggml_backend_expert_hybrid_guard_window window = {};
    REQUIRE(ggml_backend_expert_hybrid_guard_evaluate(
            &window, 0, 2, 1000, 4000000, 8) ==
            GGML_BACKEND_EXPERT_HYBRID_GUARD_DISABLED);

    // Prefill can exercise the bounded upload path before any mixed decode.
    // It must not revoke a guard whose CPU branch has not executed yet.
    window = { 3, 0, 50000, 50000, 50000, 0, 0 };
    REQUIRE(ggml_backend_expert_hybrid_guard_evaluate(
            &window, 100, 2, 1000, 4000000, 8) ==
            GGML_BACKEND_EXPERT_HYBRID_GUARD_MONITORING);

    window = { 3, 0, 1000, 1000, 1000, 1600, 8 };
    REQUIRE(ggml_backend_expert_hybrid_guard_evaluate(
            &window, 100, 2, 1000, 4000000, 8) ==
            GGML_BACKEND_EXPERT_HYBRID_GUARD_MONITORING);

    window.forced_fallbacks = 1;
    REQUIRE(ggml_backend_expert_hybrid_guard_evaluate(
            &window, 100, 2, 1000, 4000000, 8) ==
            GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_FALLBACK);

    window.forced_fallbacks = 0;
    window.cpu_compute_ns = 6401;
    REQUIRE(ggml_backend_expert_hybrid_guard_evaluate(
            &window, 100, 2, 1000, 4000000, 8) ==
            GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_CPU_DRIFT);

    window.cpu_compute_ns = 1600;
    window.lease_acquire_ns = 12001;
    window.transfer_submit_ns = 0;
    window.transfer_wait_ns = 0;
    REQUIRE(ggml_backend_expert_hybrid_guard_evaluate(
            &window, 100, 2, 1000, 4000000, 8) ==
            GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_UPLOAD_DRIFT);
}

void test_exact_cuda_pinned_staging_allocation() {
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return;
    // Deliberately avoid a page/MiB boundary. The staging contract exposes the
    // caller's exact usable byte count rather than an allocator size class.
    constexpr size_t requested = 1000003;
    ggml_backend_buffer_t buffer = ggml_backend_cuda_host_buffer_alloc(requested);
    REQUIRE(buffer != nullptr);
    REQUIRE(ggml_backend_buffer_get_size(buffer) == requested);
    REQUIRE(std::strcmp(ggml_backend_buffer_name(buffer), "CUDA_Host") == 0);
    ggml_backend_buffer_free(buffer);
#endif
}

std::vector<float> compute_cuda_cache_preflight(bool reverse_completions, bool with_bias = false, bool cached = true, bool mixed_layout = false, ggml_type first_type = GGML_TYPE_F16,
        bool frequency_sweep = false, size_t * acquisition_count = nullptr, int token_batch = 1, bool cross_device = false,
        bool check_residency_choice = false, bool check_route_protection = false,
        bool check_packing_capacity = false, bool shared_layer_ids = false,
        bool check_prefill_reserve_recovery = false) {
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() < (cross_device ? 2 : 1)) return {};
    struct mock_source {
        ggml_tensor * up = nullptr;
        ggml_tensor * down = nullptr;
        size_t batches = 0;
        size_t acquires = 0;
        size_t outstanding = 0;
        struct batch { mock_source * source; std::vector<int> experts; int component; int layer; size_t remaining; };
        static bool acquire(void * user, int layer, int expert, int component, ggml_backend_expert_lease * result) {
            auto & source = *static_cast<mock_source *>(user);
            ++source.acquires;
            ++source.outstanding;
            auto * tensor = layer == 1 || component != 1 ? source.down : source.up;
            auto data = std::make_unique<std::vector<uint8_t>>(tensor->nb[2]);
            std::memcpy(data->data(), static_cast<uint8_t *>(tensor->data) + expert*tensor->nb[2], data->size());
            result->data = data->data(); result->size = data->size(); result->handle = data.release();
            return true;
        }
        static void release(void * user, void * handle) {
            auto & source = *static_cast<mock_source *>(user);
            REQUIRE(source.outstanding > 0);
            --source.outstanding;
            delete static_cast<std::vector<uint8_t> *>(handle);
        }
        static void * begin(void * user, int layer, const int * experts, size_t count, int component) {
            auto * source = static_cast<mock_source *>(user);
            ++source->batches;
            return new batch{source, std::vector<int>(experts, experts+count), component, layer, count};
        }
        static bool next(void * opaque, size_t * index, ggml_backend_expert_lease * result) {
            auto & b = *static_cast<batch *>(opaque);
            REQUIRE(b.remaining > 0);
            *index = --b.remaining;
            return acquire(b.source, b.layer, b.experts[*index], b.component, result);
        }
        static void end(void * opaque) { delete static_cast<batch *>(opaque); }
    } source;

    ggml_backend_t cuda = ggml_backend_cuda_init(0, nullptr, nullptr);
    ggml_backend_t activation = cross_device ? ggml_backend_cuda_init(1, nullptr, nullptr) : cuda;
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cuda != nullptr && cpu != nullptr);
    struct backend_cleanup {
        ggml_backend_t cuda;
        ggml_backend_t cpu;
        ggml_backend_t activation;
        ~backend_cleanup() { if (activation != cuda) ggml_backend_free(activation); ggml_backend_free(cuda); ggml_backend_free(cpu); }
    } free_backends{cuda, cpu, activation};

    ggml_init_params params = {
        ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(32, false), nullptr, true
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * value; ~context_cleanup() { ggml_free(value); } } free_ctx{ctx};

    constexpr int64_t width = 64;
    const int64_t experts = check_packing_capacity ? 128 : 8;
    constexpr int64_t used = 2;
    ggml_tensor * up = ggml_new_tensor_3d(ctx, first_type, width, width, experts);
    ggml_tensor * down = ggml_new_tensor_3d(ctx, mixed_layout ? GGML_TYPE_F32 : GGML_TYPE_F16, width, width, experts);
    ggml_set_name(up, "blk.0.ffn_up_exps.weight");
    ggml_set_name(down, mixed_layout ? "blk.1.ffn_up_exps.weight" : "blk.0.ffn_down_exps.weight");
    source.up = up; source.down = down;
    REQUIRE(token_batch >= 1 && token_batch <= 8 && (token_batch == 1 || !mixed_layout));
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, used, token_batch);
    const int64_t id_width = used + (token_batch > 1 ? 2 : 0);
    ggml_tensor * ids_storage = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, id_width, token_batch);
    ggml_tensor * ids = token_batch > 1 ? ggml_view_2d(ctx, ids_storage, used, token_batch, ids_storage->nb[1], 0) : ids_storage;
    ggml_tensor * second_ids = mixed_layout && !shared_layer_ids ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, 1) : ids;
    if (mixed_layout || token_batch == 1) ggml_set_input(second_ids);
    ggml_set_input(input);
    if (token_batch == 1) ggml_set_input(ids);
    ggml_set_input(ids_storage);
    ggml_tensor * up_bias = with_bias ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, experts) : nullptr;
    ggml_tensor * down_bias = with_bias ? ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, experts) : nullptr;
    if (with_bias) {
        ggml_set_name(up_bias, "blk.0.ffn_up_exps.bias");
        ggml_set_name(down_bias, mixed_layout ? "blk.1.ffn_up_exps.bias" : "blk.0.ffn_down_exps.bias");
    }
    ggml_tensor * hidden = ggml_mul_mat_id(ctx, up, input, ids);
    ggml_tensor * first_projection = hidden;
    if (with_bias) hidden = ggml_add_id(ctx, hidden, up_bias, ids);
    ggml_tensor * output = ggml_mul_mat_id(ctx, down, hidden, second_ids);
    ggml_tensor * second_projection = output;
    if (with_bias) output = ggml_add_id(ctx, output, down_bias, second_ids);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);

    auto cpu_buft = ggml_backend_get_default_buffer_type(cpu);
    const size_t alignment = ggml_backend_buft_get_alignment(cpu_buft);
    const size_t up_bytes = ggml_backend_buft_get_alloc_size(cpu_buft, up);
    const size_t down_offset = (up_bytes + alignment - 1)/alignment*alignment;
    const size_t down_bytes = ggml_backend_buft_get_alloc_size(cpu_buft, down);
    const size_t bias_offset = (down_offset + down_bytes + alignment - 1)/alignment*alignment;
    const size_t bias_stride = with_bias ?
            (ggml_backend_buft_get_alloc_size(cpu_buft, up_bias) + alignment - 1)/alignment*alignment : 0;
    ggml_backend_buffer_t weights = ggml_backend_buft_alloc_buffer(
            cpu_buft, bias_offset + 2*bias_stride);
    REQUIRE(weights != nullptr);
    struct buffer_cleanup {
        ggml_backend_buffer_t value;
        ~buffer_cleanup() { ggml_backend_buffer_free(value); }
    } free_weights{weights};
    ggml_backend_buffer_set_usage(weights, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    auto * base = static_cast<uint8_t *>(ggml_backend_buffer_get_base(weights));
    ggml_backend_tensor_alloc(weights, up, base);
    ggml_backend_tensor_alloc(weights, down, base + down_offset);
    if (with_bias) {
        ggml_backend_tensor_alloc(weights, up_bias, base + bias_offset);
        ggml_backend_tensor_alloc(weights, down_bias, base + bias_offset + bias_stride);
    }

    std::vector<ggml_backend_t> backends = {cuda};
    if (cross_device) backends.push_back(activation);
    backends.push_back(cpu);
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends.data(), nullptr, backends.size(), 128, false);
    REQUIRE(scheduler != nullptr);
    struct scheduler_cleanup {
        ggml_backend_sched_t value;
        ~scheduler_cleanup() { ggml_backend_sched_free(value); }
    } free_scheduler{scheduler};
    ggml_backend_sched_set_only_active_experts(scheduler, true);
    if (reverse_completions) {
        ggml_backend_sched_set_expert_lease_callbacks(scheduler, mock_source::acquire, mock_source::release, &source, true);
        ggml_backend_sched_set_expert_load_callbacks(scheduler, {mock_source::begin, mock_source::next, mock_source::end});
    }
    // Four compact slots plus allocator headroom: the biased regression must
    // exercise both ID remapping and eviction, not only a warm identity map.
    const uint64_t capacity = check_packing_capacity ? 64*256 + 192*width*width*3 : with_bias && token_batch == 1 ? 64*256 + 5*2*width*width*sizeof(ggml_fp16_t) : 16ULL*1024*1024;
    if (cached) ggml_backend_sched_set_expert_cache_capacity(scheduler, capacity, 0, 1);
    const uint64_t prefill_capacity = 2*(ggml_nbytes(up) + ggml_nbytes(down));
    if (check_prefill_reserve_recovery) {
        REQUIRE(reverse_completions && !cached && token_batch > 1);
        ggml_backend_sched_set_active_expert_decode(scheduler, false);
        ggml_backend_sched_set_expert_prefill_staging(scheduler, prefill_capacity, 0);
    }
    ggml_backend_sched_set_tensor_backend(scheduler, hidden, cuda);
    ggml_backend_sched_set_tensor_backend(scheduler, output, cuda);
    ggml_backend_sched_set_tensor_backend(scheduler, input, activation);
    ggml_backend_sched_set_tensor_backend(scheduler, ids, activation);
    ggml_backend_sched_set_tensor_backend(scheduler, ids_storage, activation);
    ggml_backend_sched_set_tensor_backend(scheduler, second_ids, activation);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));

    REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler, hidden) == cuda);
    REQUIRE(ggml_backend_sched_get_tensor_backend(scheduler, output) == cuda);
    std::vector<ggml_backend_sched_resource_device_stats> stats(cross_device ? 2 : 1);
    REQUIRE(ggml_backend_sched_get_resource_device_stats(
            scheduler, stats.data(), stats.size()));
    if (cached) {
        REQUIRE(stats[0].expert_cache_capacity_bytes > 0);
        REQUIRE(stats[0].expert_cache_allocated_bytes > 64*256);
        REQUIRE(stats[0].expert_cache_allocated_bytes <= capacity);
        if (check_packing_capacity) {
            // F16 and F32 layers get independent local ID spaces. At this
            // budget uniform F32 slots hold only 143 entries; packing must
            // provide at least 192 without exceeding the same charged bytes.
            REQUIRE(mixed_layout && first_type == GGML_TYPE_F16);
            REQUIRE(first_projection->src[0]->ne[2] + second_projection->src[0]->ne[2] >= 192);
        }
        // Expert data already belongs to the cache. Graph scratch must not
        // allocate a second slot-sized copy of each component.
        REQUIRE(ggml_backend_sched_get_buffer_size(scheduler, cuda) < 1024*1024);
    }

    std::vector<ggml_fp16_t> weight_values(size_t(width*width*experts));
    for (size_t index = 0; index < weight_values.size(); ++index) {
        // Exactly representable diagonal math isolates bias/slot ownership
        // from the different accumulation precision of resident/cache kernels.
        const bool diagonal = index % width == (index/width) % width;
        const float value = with_bias ? (diagonal ? float(index/(width*width) + 1)/8.0f : 0.0f)
                                      : float(int(index % 31) - 15)/32.0f;
        weight_values[index] = ggml_fp32_to_fp16(value);
    }
    std::vector<float> input_values(size_t(width*used*token_batch));
    for (size_t index = 0; index < input_values.size(); ++index) {
        input_values[index] = first_type != GGML_TYPE_F16 ? 0.0f :
                with_bias ? float(int(index % 3) - 1)/2.0f : float(int(index % 17) - 8)/16.0f;
    }
    if (with_bias) {
        std::vector<float> biases(size_t(width*experts));
        for (size_t i = 0; i < biases.size(); ++i) biases[i] = float(i/width + 1)/8.0f;
        ggml_backend_tensor_set(up_bias, biases.data(), 0, ggml_nbytes(up_bias));
        for (float & value : biases) value *= -0.5f;
        ggml_backend_tensor_set(down_bias, biases.data(), 0, ggml_nbytes(down_bias));
    }
    std::vector<std::array<int32_t, used>> routes = mixed_layout
            ? std::vector<std::array<int32_t, used>>{{1, 6}, {1, 6}, {3, 7}, {3, 7}, {6, 1}}
            : std::vector<std::array<int32_t, used>>{{1, 6}, {3, 7}, {0, 4}, {6, 1}, {3, 7}};
    if (frequency_sweep) {
        REQUIRE(with_bias && !mixed_layout);
        routes.clear();
        for (int i = 0; i < 36; ++i) routes.push_back({2*(i%3), 2*(i%3)+1});
    }
    if (check_route_protection) {
        REQUIRE(with_bias && !mixed_layout && token_batch == 1);
        // Fill all four slots, then put a miss before the oldest resident in
        // the next route. That resident must survive admission of expert 0.
        routes = {{1, 6}, {3, 7}, {0, 1}};
    }
    if (first_type == GGML_TYPE_F16) {
        ggml_backend_tensor_set(up, weight_values.data(), 0, ggml_nbytes(up));
    } else {
        std::vector<float> values(weight_values.size());
        std::transform(weight_values.begin(), weight_values.end(), values.begin(), ggml_fp16_to_fp32);
        std::vector<uint8_t> packed(ggml_nbytes(up));
        REQUIRE(ggml_quantize_chunk(first_type, values.data(), packed.data(), 0, width*experts, width,
                nullptr, nullptr) == packed.size());
        ggml_backend_tensor_set(up, packed.data(), 0, packed.size());
    }
    if (!with_bias) std::reverse(weight_values.begin(), weight_values.end());
    if (mixed_layout) {
        std::vector<float> second_weights(weight_values.size());
        std::transform(weight_values.begin(), weight_values.end(), second_weights.begin(), ggml_fp16_to_fp32);
        ggml_backend_tensor_set(down, second_weights.data(), 0, ggml_nbytes(down));
    } else {
        ggml_backend_tensor_set(down, weight_values.data(), 0, ggml_nbytes(down));
    }
    std::vector<float> result;
    for (size_t pass = 0; pass < (with_bias ? routes.size() : 1); ++pass) {
        const auto acquires_before = source.acquires;
        ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));
        std::vector<int32_t> batch_ids(size_t(id_width*token_batch), -777);
        for (int token = 0; token < token_batch; ++token)
            std::copy(routes[(pass + token)%routes.size()].begin(), routes[(pass + token)%routes.size()].end(),
                    batch_ids.begin() + token*id_width);
        ggml_backend_tensor_set(ids_storage, batch_ids.data(), 0, batch_ids.size()*sizeof(int32_t));
        if (mixed_layout) ggml_backend_tensor_set(second_ids, routes[pass].data(), 0, sizeof(routes[pass]));
        REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
        ggml_backend_sched_synchronize(scheduler);
        if (check_route_protection && cached && pass == 2) {
            // One cold expert, two weight components. Re-uploading expert 1
            // would add two more acquisitions without changing the result.
            REQUIRE(source.acquires - acquires_before == 2);
        }
        if (check_residency_choice && pass == 0) {
            // A complete warm expert lies behind a cold expert and a padding
            // sentinel. Readiness must include both projections on the actual
            // output device, rather than an OR of all devices' ready bits.
            ggml_init_params probe_params = {1024*1024, nullptr, false};
            auto * probe_ctx = ggml_init(probe_params);
            REQUIRE(probe_ctx != nullptr);
            struct probe_cleanup { ggml_context * ctx; ~probe_cleanup() { ggml_free(ctx); } } free_probe{probe_ctx};
            auto * probe_ids = ggml_new_tensor_2d(probe_ctx, GGML_TYPE_I32, 4, 1);
            const std::array<int32_t, 4> probe_route = {{ -1, 0, 6, 1 }};
            std::memcpy(probe_ids->data, probe_route.data(), sizeof(probe_route));
            const auto evaluate = [&](ggml_tensor * up_source, ggml_tensor * gate_source, ggml_tensor * anchor) {
                auto * mask = ggml_backend_sched_expert_route_choice(
                        probe_ctx, scheduler, probe_ids, 0, 1, up_source, gate_source, down);
                ggml_backend_sched_expert_route_bind_gpu_output(mask, anchor);
                auto * gpu = ggml_ese_route_partition_masked(probe_ctx, probe_ids, mask, GGML_ESE_ROUTE_GPU);
                auto * cpu_ids = ggml_ese_route_partition_masked(probe_ctx, probe_ids, mask, GGML_ESE_ROUTE_CPU);
                REQUIRE(ggml_ese_route_get_role(gpu) == GGML_ESE_ROUTE_GPU);
                REQUIRE(ggml_ese_route_get_role(cpu_ids) == GGML_ESE_ROUTE_CPU);
                auto * probe_graph = ggml_new_graph_custom(probe_ctx, 16, false);
                ggml_build_forward_expand(probe_graph, gpu);
                ggml_build_forward_expand(probe_graph, cpu_ids);
                REQUIRE(ggml_backend_graph_compute(cpu, probe_graph) == GGML_STATUS_SUCCESS);
                const auto * gpu_ids = static_cast<const int32_t *>(gpu->data);
                const auto * cpu_route = static_cast<const int32_t *>(cpu_ids->data);
                for (size_t i = 0; i < probe_route.size(); ++i) {
                    REQUIRE(gpu_ids[i] == -1 || gpu_ids[i] == probe_route[i]);
                    REQUIRE(cpu_route[i] == -1 || cpu_route[i] == probe_route[i]);
                    if (probe_route[i] >= 0) REQUIRE((gpu_ids[i] >= 0) != (cpu_route[i] >= 0));
                }
                std::array<int32_t, 4> result;
                std::memcpy(result.data(), mask->data, sizeof(result));
                return result;
            };
            const std::array<int32_t, 4> warm = {{0, 0, 1, 0}}, cold = {{0, 1, 0, 0}};
            REQUIRE(evaluate(up, nullptr, output) == (cached && !mixed_layout ? warm : cold));
            // A gate component that has never been uploaded makes an expert
            // incomplete, even though up/down exist. A wrong quantized layout
            // likewise must not count as a hit.
            auto * missing_gate = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F16, width, width, experts);
            ggml_set_name(missing_gate, "blk.0.ffn_gate_exps.weight");
            REQUIRE(evaluate(up, missing_gate, output) == cold);
            auto * wrong_layout = ggml_new_tensor_3d(probe_ctx, GGML_TYPE_F32, width, width, experts);
            ggml_set_name(wrong_layout, "blk.0.ffn_up_exps.weight");
            REQUIRE(evaluate(wrong_layout, nullptr, output) == cold);
            if (cross_device) REQUIRE(evaluate(up, nullptr, input) == cold);
        }
        if (mixed_layout && cached && (pass == 1 || pass == 3)) {
            // A different layer/layout ran since this payload was used. With
            // retained payloads, an identical route must not acquire/upload it.
            REQUIRE(source.acquires == acquires_before);
        }
        const size_t offset = result.size();
        result.resize(offset + size_t(ggml_nelements(output)));
        ggml_backend_tensor_get(output, result.data() + offset, 0, ggml_nbytes(output));
        if (check_prefill_reserve_recovery) {
            REQUIRE(source.outstanding == 0);
            REQUIRE(ggml_backend_sched_get_resource_device_stats(scheduler, stats.data(), stats.size()));
            const auto allocated = stats[0].expert_prefill_allocated_bytes;
            REQUIRE(allocated > 0 && allocated <= prefill_capacity);
            const auto acquisitions = source.acquires;
            // Raise only the required reserve, without exhausting the driver or
            // disturbing another process. The already admitted graph must fail
            // before output writes/leases and remain retryable with the same buffers.
            ggml_backend_sched_set_expert_prefill_staging(scheduler, prefill_capacity, UINT64_MAX);
            REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_ALLOC_FAILED);
            REQUIRE(source.acquires == acquisitions);
            REQUIRE(source.outstanding == 0);
            std::vector<float> observed(size_t(ggml_nelements(output)));
            ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));
            REQUIRE(observed == result);
            REQUIRE(ggml_backend_sched_get_resource_device_stats(scheduler, stats.data(), stats.size()));
            REQUIRE(stats[0].expert_prefill_allocated_bytes == allocated);
            ggml_backend_sched_set_expert_prefill_staging(scheduler, prefill_capacity, 0);
            // A normal invocation supplies its inputs again: graph scratch may
            // alias the previous invocation's activation input after execution.
            ggml_backend_tensor_set(input, input_values.data(), 0, ggml_nbytes(input));
            ggml_backend_tensor_set(ids_storage, batch_ids.data(), 0, batch_ids.size()*sizeof(int32_t));
            REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
            REQUIRE(source.outstanding == 0);
            ggml_backend_tensor_get(output, observed.data(), 0, ggml_nbytes(output));
            REQUIRE(observed == result);
            REQUIRE(ggml_backend_sched_get_resource_device_stats(scheduler, stats.data(), stats.size()));
            REQUIRE(stats[0].expert_prefill_allocated_bytes == allocated);
        }
        if (with_bias) {
            for (size_t i = 0; i < size_t(width*used*token_batch); ++i) {
                const size_t token = i/(width*used);
                const float gain = float(routes[(pass + token)%routes.size()][(i/width)%used] + 1)/8.0f;
                const float expected = (input_values[i]*gain + gain)*gain - gain/2;
                REQUIRE(result[offset + i] == expected);
            }
        }
        REQUIRE(std::all_of(result.begin(), result.end(), [](float value) { return std::isfinite(value); }));
        if (mixed_layout && cached && first_type != GGML_TYPE_F16 && (pass == 0 || pass == 2)) {
            // A graph rebuild initializes aliases of a shorter quantized
            // layout. Its CUDA tail clear must not overwrite the F32 payload
            // retained in the final slot of this shared cache allocation.
            ggml_tensor alias = *first_projection->src[0];
            REQUIRE(ggml_backend_buffer_get_usage(alias.buffer) == GGML_BACKEND_BUFFER_USAGE_EXPERT_CACHE);
            ggml_backend_buffer_init_tensor(alias.buffer, &alias);
        }
        if (mixed_layout && cached && (pass == 1 || pass == 3)) {
            // Reuse the graph after the old cache buffers are freed, and after
            // a publication rollback. Input aliases and the preflight stamp
            // must follow the currently published cache in both cases.
            const uint64_t replacement = pass == 1 ? capacity + 2*width*width*sizeof(float) : capacity;
            auto transaction = ggml_backend_sched_expert_cache_prepare(scheduler, replacement, 0, 1);
            REQUIRE(transaction != nullptr);
            REQUIRE(ggml_backend_sched_expert_cache_publish(transaction));
            if (pass == 3) REQUIRE(ggml_backend_sched_expert_cache_rollback(transaction));
            REQUIRE(ggml_backend_sched_expert_cache_finalize(transaction));
        }
    }
    // Prefill staging acquires individual leases; the compact decode cache
    // exercises the completion-order batch callbacks instead.
    if (reverse_completions && !check_prefill_reserve_recovery) REQUIRE(source.batches >= 2);
    if (check_prefill_reserve_recovery) REQUIRE(source.acquires > 0 && source.outstanding == 0);
    if (acquisition_count) *acquisition_count = source.acquires;
    return result;
#else
    (void) reverse_completions;
    (void) with_bias;
    (void) cached;
    (void) mixed_layout;
    (void) first_type;
    (void) frequency_sweep;
    (void) acquisition_count;
    (void) token_batch;
    (void) cross_device;
    (void) check_residency_choice;
    (void) check_route_protection;
    (void) check_packing_capacity;
    (void) shared_layer_ids;
    (void) check_prefill_reserve_recovery;
    return {};
#endif
}

void test_cuda_layout_pool_capacity_and_lifetimes() {
#if defined(GGML_USE_CUDA) && !defined(_WIN32)
    const char * previous = std::getenv("ESE_EXPERT_CACHE_PACKING");
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    setenv("ESE_EXPERT_CACHE_PACKING", "1", 1);
    struct restore_setting {
        bool present;
        std::string value;
        ~restore_setting() {
            if (present) setenv("ESE_EXPERT_CACHE_PACKING", value.c_str(), 1);
            else unsetenv("ESE_EXPERT_CACHE_PACKING");
        }
    } restore{had_previous, saved};
    // Exact diagonal/bias oracle, reversed completion, warm reuse, capacity
    // replacement, finalized old buffers, and publication rollback all run
    // through the existing helper. A second quantized case checks pool tails.
    const auto resident = compute_cuda_cache_preflight(false, true, false, true);
    REQUIRE(compute_cuda_cache_preflight(true, true, true, true, GGML_TYPE_F16,
            false, nullptr, 1, false, false, false, true) == resident);
    // Reusing logical routing IDs in another layer must create that layer's
    // own remap, rather than reuse physical slots from the first pool.
    REQUIRE(compute_cuda_cache_preflight(true, true, true, true, GGML_TYPE_F16,
            false, nullptr, 1, false, false, false, true, true) == resident);
    REQUIRE(compute_cuda_cache_preflight(true, true, true, true, GGML_TYPE_Q4_0) ==
            compute_cuda_cache_preflight(false, true, false, true, GGML_TYPE_Q4_0));
#endif
}

void test_cuda_route_residents_survive_admission() {
#if defined(GGML_USE_CUDA) && !defined(_WIN32)
    if (ggml_backend_cuda_get_device_count() == 0) return;
    const char * prior = getenv("ESE_EXPERT_CACHE_RETENTION");
    struct restore_env {
        bool present; std::string value;
        ~restore_env() {
            if (present) setenv("ESE_EXPERT_CACHE_RETENTION", value.c_str(), 1);
            else unsetenv("ESE_EXPERT_CACHE_RETENTION");
        }
    } restore{prior != nullptr, prior ? prior : ""};
    setenv("ESE_EXPERT_CACHE_RETENTION", "lru", 1);
    const auto resident = compute_cuda_cache_preflight(false, true, false, false,
            GGML_TYPE_F16, false, nullptr, 1, false, false, true);
    const auto cached = compute_cuda_cache_preflight(true, true, true, false,
            GGML_TYPE_F16, false, nullptr, 1, false, false, true);
    REQUIRE(cached == resident);
#endif
}

void test_cuda_frequency_retention() {
#if defined(GGML_USE_CUDA) && !defined(_WIN32)
    if (ggml_backend_cuda_get_device_count() == 0) return;
    const char * prior = getenv("ESE_EXPERT_CACHE_RETENTION");
    struct restore_env {
        bool present; std::string value;
        ~restore_env() {
            if (present) setenv("ESE_EXPERT_CACHE_RETENTION", value.c_str(), 1);
            else unsetenv("ESE_EXPERT_CACHE_RETENTION");
        }
    } restore{prior != nullptr, prior ? prior : ""};
    size_t lru_reads = 0, frequency_reads = 0;
    setenv("ESE_EXPERT_CACHE_RETENTION", "lru", 1);
    auto lru = compute_cuda_cache_preflight(true, true, true, false, GGML_TYPE_F16, true, &lru_reads);
    setenv("ESE_EXPERT_CACHE_RETENTION", "frequency", 1);
    auto frequency = compute_cuda_cache_preflight(true, true, true, false, GGML_TYPE_F16, true, &frequency_reads);
    REQUIRE(lru == frequency);
    REQUIRE(frequency_reads < lru_reads);
    std::cout << "Frequency retention sweep: " << lru_reads << " -> " << frequency_reads << " lease uploads\n";
#endif
}

void test_cuda_cache_preflight_realizes_all_components() {
    const auto sequential = compute_cuda_cache_preflight(false);
    const auto reversed = compute_cuda_cache_preflight(true);
    REQUIRE(sequential == reversed);
    const auto resident_bias = compute_cuda_cache_preflight(false, true, false);
    const auto cached_bias = compute_cuda_cache_preflight(true, true, true);
    REQUIRE(resident_bias.size() == cached_bias.size());
    REQUIRE(resident_bias == cached_bias);
    const auto resident_batch = compute_cuda_cache_preflight(false, true, false, false, GGML_TYPE_F16, false, nullptr, 4);
    const auto cached_batch = compute_cuda_cache_preflight(true, true, true, false, GGML_TYPE_F16, false, nullptr, 4);
    REQUIRE(resident_batch == cached_batch);
    const auto remote_batch = compute_cuda_cache_preflight(true, true, true, false, GGML_TYPE_F16, false, nullptr, 4, true);
    if (!remote_batch.empty()) REQUIRE(remote_batch == resident_batch);
    const auto resident_mixed = compute_cuda_cache_preflight(false, true, false, true);
    const auto cached_mixed = compute_cuda_cache_preflight(true, true, true, true);
    REQUIRE(resident_mixed == cached_mixed);
    const auto quantized_resident = compute_cuda_cache_preflight(false, true, false, true, GGML_TYPE_Q4_0);
    const auto quantized_cached = compute_cuda_cache_preflight(true, true, true, true, GGML_TYPE_Q4_0);
    REQUIRE(quantized_resident == quantized_cached);
}

void test_expert_cache_transaction_state_machine() {
    REQUIRE(ggml_backend_sched_expert_cache_prepare(nullptr, 0, 0, 1) == nullptr);
    REQUIRE(!ggml_backend_sched_expert_cache_publish(nullptr));
    REQUIRE(!ggml_backend_sched_expert_cache_rollback(nullptr));
    REQUIRE(!ggml_backend_sched_expert_cache_finalize(nullptr));

    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cpu != nullptr);
    struct cpu_cleanup {
        ggml_backend_t value;
        ~cpu_cleanup() { ggml_backend_free(value); }
    } free_cpu{cpu};

    ggml_backend_t backends[] = {cpu};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 1, 64, false);
    REQUIRE(scheduler != nullptr);
    struct scheduler_cleanup {
        ggml_backend_sched_t value;
        ~scheduler_cleanup() { ggml_backend_sched_free(value); }
    } free_scheduler{scheduler};

    auto * transaction = ggml_backend_sched_expert_cache_prepare(
            scheduler, 0, 0, 1);
    REQUIRE(transaction != nullptr);
    REQUIRE(ggml_backend_sched_expert_cache_prepare(scheduler, 0, 0, 1) == nullptr);
    REQUIRE(!ggml_backend_sched_expert_cache_rollback(transaction));
    REQUIRE(ggml_backend_sched_expert_cache_publish(transaction));
    REQUIRE(!ggml_backend_sched_expert_cache_publish(transaction));
    REQUIRE(ggml_backend_sched_expert_cache_rollback(transaction));
    REQUIRE(!ggml_backend_sched_expert_cache_rollback(transaction));
    REQUIRE(ggml_backend_sched_expert_cache_finalize(transaction));

    transaction = ggml_backend_sched_expert_cache_prepare(scheduler, 0, 0, 1);
    REQUIRE(transaction != nullptr);
    REQUIRE(ggml_backend_sched_expert_cache_finalize(transaction));

    transaction = ggml_backend_sched_expert_cache_prepare(scheduler, 0, 0, 1);
    REQUIRE(transaction != nullptr);
    REQUIRE(ggml_backend_sched_expert_cache_publish(transaction));
    REQUIRE(ggml_backend_sched_expert_cache_finalize(transaction));
    REQUIRE(ggml_backend_sched_replace_expert_cache(scheduler, 0, 0, 1));

    // A non-zero cache requires at least one device backend, and must fail
    // without leaving an outstanding scheduler transaction.
    REQUIRE(ggml_backend_sched_expert_cache_prepare(scheduler, 1, 0, 1) == nullptr);
    REQUIRE(ggml_backend_sched_replace_expert_cache(scheduler, 0, 0, 1));
}

void test_expert_cache_partial_publication_rollback() {
#ifdef GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() < 2) {
        std::cout << "SKIP: partial expert-cache publication requires two CUDA devices\n";
        return;
    }

    ggml_backend_t cuda0 = ggml_backend_cuda_init(0, nullptr, nullptr);
    ggml_backend_t cuda1 = ggml_backend_cuda_init(1, nullptr, nullptr);
    ggml_backend_t cpu = ggml_backend_cpu_init();
    REQUIRE(cuda0 != nullptr && cuda1 != nullptr && cpu != nullptr);
    struct backend_cleanup {
        ggml_backend_t cuda0;
        ggml_backend_t cuda1;
        ggml_backend_t cpu;
        ~backend_cleanup() {
            ggml_backend_free(cuda0);
            ggml_backend_free(cuda1);
            ggml_backend_free(cpu);
        }
    } free_backends{cuda0, cuda1, cpu};

    ggml_backend_t backends[] = {cuda0, cuda1, cpu};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 3, 64, false);
    REQUIRE(scheduler != nullptr);
    struct scheduler_cleanup {
        ggml_backend_sched_t value;
        ~scheduler_cleanup() { ggml_backend_sched_free(value); }
    } free_scheduler{scheduler};

    ggml_backend_sched_set_expert_cache_capacity(scheduler, 0, 17, 3);
    ggml_backend_sched_expert_cache_policy before_policy = {};
    REQUIRE(ggml_backend_sched_expert_cache_get_policy(scheduler, &before_policy));
    std::array<ggml_backend_sched_resource_device_stats, 2> before_devices = {};
    REQUIRE(ggml_backend_sched_get_resource_device_stats(
            scheduler, before_devices.data(), before_devices.size()));

    auto * transaction = ggml_backend_sched_expert_cache_prepare(
            scheduler, 0, 31, 5);
    REQUIRE(transaction != nullptr);
#if defined(_WIN32)
    _putenv_s("ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES", "1");
#else
    setenv("ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES", "1", 1);
#endif
    REQUIRE(!ggml_backend_sched_expert_cache_publish(transaction));
#if defined(_WIN32)
    _putenv_s("ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES", "");
#else
    unsetenv("ESE_EXPERT_CACHE_TRANSACTION_FAIL_AFTER_PUBLISHED_DEVICES");
#endif

    ggml_backend_sched_expert_cache_policy after_policy = {};
    REQUIRE(ggml_backend_sched_expert_cache_get_policy(scheduler, &after_policy));
    REQUIRE(after_policy.capacity_bytes_per_device == before_policy.capacity_bytes_per_device);
    REQUIRE(after_policy.reserve_bytes_per_device == before_policy.reserve_bytes_per_device);
    REQUIRE(after_policy.minimum_observations == before_policy.minimum_observations);
    REQUIRE(after_policy.slots == before_policy.slots);
    std::array<ggml_backend_sched_resource_device_stats, 2> after_devices = {};
    REQUIRE(ggml_backend_sched_get_resource_device_stats(
            scheduler, after_devices.data(), after_devices.size()));
    for (size_t index = 0; index < before_devices.size(); ++index) {
        REQUIRE(after_devices[index].backend_id == before_devices[index].backend_id);
        REQUIRE(after_devices[index].expert_cache_capacity_bytes ==
                before_devices[index].expert_cache_capacity_bytes);
        REQUIRE(after_devices[index].expert_cache_allocated_bytes ==
                before_devices[index].expert_cache_allocated_bytes);
        REQUIRE(after_devices[index].expert_cache_resident_bytes ==
                before_devices[index].expert_cache_resident_bytes);
    }

    // The failed publish remains a prepared owner: it is exclusive, can be
    // retried after the test fault is removed, and then finalizes normally.
    REQUIRE(ggml_backend_sched_expert_cache_prepare(scheduler, 0, 17, 3) == nullptr);
    REQUIRE(ggml_backend_sched_expert_cache_publish(transaction));
    REQUIRE(ggml_backend_sched_expert_cache_finalize(transaction));
    REQUIRE(ggml_backend_sched_expert_cache_get_policy(scheduler, &after_policy));
    REQUIRE(after_policy.capacity_bytes_per_device == 0);
    REQUIRE(after_policy.reserve_bytes_per_device == 31);
    REQUIRE(after_policy.minimum_observations == 5);
#endif
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::strcmp(argv[1], "--router-softmax-only") == 0) {
            test_cpu_router_softmax_replay();
            std::cout << "PASS: CPU router softmax, IDs and weights survive poisoned replay\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--router-cache-only") == 0) {
            test_cuda_router_precedes_compact_staging();
            std::cout << "PASS: computed GPU router precedes compact expert staging\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--sparse-cuda-only") == 0) {
            test_cuda_sparse_quantized_routes();
            std::cout << "PASS: sparse quantized CUDA route sentinels\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--packing-cache-only") == 0) {
            test_cuda_layout_pool_capacity_and_lifetimes();
            std::cout << "PASS: packed cache capacity, aliases and replacement\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--stable-mmq-only") == 0) {
            test_cuda_stable_expert_reduction();
            std::cout << "PASS: stable indexed MMQ across cache geometry\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--route-cache-only") == 0) {
            test_cuda_route_residents_survive_admission();
            std::cout << "PASS: cached route residents survive miss admission\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--cpu-lease-chunks-only") == 0) {
            test_cpu_lease_chunks_with_bounded_cache();
            std::cout << "PASS: exact CPU lease chunks under bounded cache and forced eviction\n";
            return 0;
        }
        if (argc == 2 && std::strcmp(argv[1], "--prefill-reserve-only") == 0) {
            compute_cuda_cache_preflight(true, false, false, false, GGML_TYPE_F16,
                    false, nullptr, 4, false, false, false, false, false, true);
            std::cout << "PASS: strict prefill reserve rejection and exact retry\n";
            return 0;
        }
        test_cpu_router_softmax_replay();
        test_descriptor_validation();
        test_exact_reads_hits_and_backend_telemetry();
        test_deterministic_eviction_and_lease_lifetime();
        test_deterministic_eviction_and_lease_lifetime(true);
        test_ram_frequency_retention();
        test_capacity_and_identity_fail_closed();
        test_file_storage_backends();
        test_concurrent_single_fill();
        test_ready_hit_during_blocked_miss();
        test_independent_fills_share_bounded_staging();
        test_staging_backpressure_preserves_ready_hits();
        test_failed_fill_preserves_victims_and_coalesces_error();
        test_reserved_victim_can_become_leased();
        test_concurrent_eviction_stress();
        test_concurrent_eviction_stress(true);
        test_load_queue_completion_order_and_bounds();
        test_load_queue_ready_hits_and_error_recovery();
        test_load_queue_shutdown_joins_reads();
        test_read_ahead_identity_and_retention();
        test_external_resident_arena_ownership();
        test_read_ahead_demand_priority_budget_and_expiration();
        test_cpu_kernel_exact_parity_and_no_original_fallback();
        test_cpu_lease_chunks_with_bounded_cache();
        test_route_partition_is_complementary();
        test_masked_route_preserves_weighted_positions();
        test_cpu_fused_moe_merged_workspace();
        test_cuda_compact_remap_exact_parity();
        test_cuda_router_precedes_compact_staging();
        test_cuda_sparse_quantized_routes();
        test_cuda_route_residents_survive_admission();
        test_cuda_frequency_retention();
        test_hybrid_runtime_guard_is_one_way_and_fail_closed();
        test_exact_cuda_pinned_staging_allocation();
        test_cuda_cache_preflight_realizes_all_components();
        compute_cuda_cache_preflight(true, false, false, false, GGML_TYPE_F16,
                false, nullptr, 4, false, false, false, false, false, true);
        test_cuda_layout_pool_capacity_and_lifetimes();
        REQUIRE(compute_cuda_cache_preflight(false, false, true, false, GGML_TYPE_F16,
                    false, nullptr, 1, false, true) ==
                compute_cuda_cache_preflight(false, false, false, false, GGML_TYPE_F16,
                    false, nullptr, 1, false, true));
        compute_cuda_cache_preflight(false, false, true, false, GGML_TYPE_F16,
                false, nullptr, 1, true, true);
        test_expert_cache_transaction_state_machine();
        test_expert_cache_partial_publication_rollback();
        std::cout << "PASS: checked bounded expert caches and exact full/compact CUDA parity\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
