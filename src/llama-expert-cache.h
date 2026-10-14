#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace llama_expert_cache {

enum class component : uint8_t {
    gate = 0,
    up   = 1,
    down = 2,
    gate_up = 3,
    normalization = 4,
    probability_bias = 5,
};

enum class read_backend : uint8_t {
    mmap = 0,
    pread = 1,
    io_uring = 2,
    count = 3,
};

struct source_identity {
    uint64_t high = 0;
    uint64_t low  = 0;

    bool operator==(const source_identity & other) const {
        return high == other.high && low == other.low;
    }
    bool operator!=(const source_identity & other) const { return !(*this == other); }
};

// Compute the same bounded file identity validated by file_source. This is
// exposed for calibration and descriptor construction without loading a GGUF.
source_identity identify_file_source(const std::string & path, uint32_t source_id);

struct expert_key {
    uint32_t layer  = 0;
    uint32_t expert = 0;
    component part  = component::gate;

    bool operator<(const expert_key & other) const {
        if (layer != other.layer) return layer < other.layer;
        if (expert != other.expert) return expert < other.expert;
        return uint8_t(part) < uint8_t(other.part);
    }
    bool operator==(const expert_key & other) const {
        return layer == other.layer && expert == other.expert && part == other.part;
    }
};

struct source_extent {
    uint32_t source = 0;
    source_identity identity;
    uint64_t offset = 0;
    uint64_t length = 0;
};

struct descriptor_spec {
    expert_key key;
    uint32_t ggml_type = 0;
    uint8_t rank = 0;
    uint8_t quant_axis = 0;
    uint32_t block_elements = 1;
    uint32_t block_bytes = 1;
    uint32_t source_alignment = 1;
    std::array<uint64_t, 4> dimensions = {{ 0, 0, 0, 0 }};
    std::array<uint64_t, 4> strides = {{ 0, 0, 0, 0 }};
    source_identity model_identity;
    std::vector<source_extent> extents;
};

// Immutable, fully checked description of one expert component. Construction
// is only possible through make(), which performs checked 64-bit geometry,
// bounds, identity, overlap, and extent validation.
class descriptor {
public:
    static std::shared_ptr<const descriptor> make(
            descriptor_spec spec,
            const std::map<uint32_t, uint64_t> & source_sizes);

    const expert_key & key() const { return key_; }
    uint32_t ggml_type() const { return ggml_type_; }
    uint8_t rank() const { return rank_; }
    uint8_t quant_axis() const { return quant_axis_; }
    uint32_t block_elements() const { return block_elements_; }
    uint32_t block_bytes() const { return block_bytes_; }
    uint32_t source_alignment() const { return source_alignment_; }
    const std::array<uint64_t, 4> & dimensions() const { return dimensions_; }
    const std::array<uint64_t, 4> & strides() const { return strides_; }
    const source_identity & model_identity() const { return model_identity_; }
    const std::vector<source_extent> & extents() const { return extents_; }
    uint64_t bytes() const { return bytes_; }

private:
    explicit descriptor(descriptor_spec spec, uint64_t bytes);

    const expert_key key_;
    const uint32_t ggml_type_;
    const uint8_t rank_;
    const uint8_t quant_axis_;
    const uint32_t block_elements_;
    const uint32_t block_bytes_;
    const uint32_t source_alignment_;
    const std::array<uint64_t, 4> dimensions_;
    const std::array<uint64_t, 4> strides_;
    const source_identity model_identity_;
    const std::vector<source_extent> extents_;
    const uint64_t bytes_;
};

class source {
public:
    struct read_request {
        uint64_t offset = 0;
        void * destination = nullptr;
        size_t length = 0;
    };

    virtual ~source() = default;
    virtual uint32_t id() const = 0;
    virtual source_identity identity() const = 0;
    virtual uint64_t size() const = 0;
    virtual read_backend backend() const = 0;
    virtual void read(uint64_t offset, void * destination, size_t length) const = 0;
    virtual void read_many(const std::vector<read_request> & requests) const;
    // Best-effort storage advice, with no payload allocation or cache admission.
    virtual bool advise_will_need(uint64_t, uint64_t) const { return false; }
};

// An exact storage backend is selected at construction. Unsupported io_uring
// is an error rather than a silent pread fallback, so telemetry and policy stay
// truthful.
class file_source final : public source {
public:
    file_source(uint32_t id, source_identity identity, std::string path, read_backend backend);
    ~file_source() override;
    file_source(const file_source &) = delete;
    file_source & operator=(const file_source &) = delete;

    uint32_t id() const override;
    source_identity identity() const override;
    uint64_t size() const override;
    read_backend backend() const override;
    void read(uint64_t offset, void * destination, size_t length) const override;
    void read_many(const std::vector<read_request> & requests) const override;
    bool advise_will_need(uint64_t offset, uint64_t length) const override;
    // Best-effort Linux page-cache control for reproducible cold benchmarks.
    // Throws if the requested range is invalid or the kernel rejects the hint.
    void evict_range_from_os_cache(uint64_t offset, uint64_t length) const;
    const std::string & path() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

struct cache_config {
    uint64_t capacity_bytes = 0;
    uint64_t staging_bytes = 0;
    bool frequency_retention = false;
};

struct cache_stats {
    uint64_t capacity_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t peak_resident_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t rejected_admissions = 0;
    uint64_t pinned_capacity_failures = 0;
    uint64_t active_leases = 0;
    uint64_t staging_waits = 0;
    uint64_t staging_wait_ns = 0;
    uint64_t storage_read_ns = 0;
    uint64_t lock_wait_ns = 0;
    uint64_t coalesced_loads = 0;
    uint64_t failed_loads = 0;
    uint64_t active_loads = 0;
    uint64_t peak_active_loads = 0;
    uint64_t staging_capacity_bytes = 0;
    uint64_t staging_used_bytes = 0;
    uint64_t peak_staging_used_bytes = 0;
    // Reservations overlap still-readable eviction candidates. They are not
    // additional physical allocations beyond the fixed resident arena.
    uint64_t reserved_bytes = 0;
    std::array<uint64_t, size_t(read_backend::count)> bytes_read = {{ 0, 0, 0 }};
};

class capacity_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ram_cache {
private:
    struct entry;
    struct pending_load;

public:
    class lease {
    public:
        lease() = default;
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;
        lease(lease && other) noexcept;
        lease & operator=(lease && other) noexcept;
        ~lease();

        const uint8_t * data() const;
        size_t size() const;
        explicit operator bool() const { return entry_ != nullptr; }

    private:
        friend class ram_cache;
        explicit lease(std::shared_ptr<entry> value);
        void release();
        std::shared_ptr<entry> entry_;
    };

    // Optional caller-allocated arena (at least capacity_bytes, aligned to 64
    // bytes). Shared ownership retains its allocator until the cache is freed.
    ram_cache(cache_config config, std::vector<std::shared_ptr<const source>> sources,
            std::shared_ptr<uint8_t> resident_arena = {});
    ram_cache(const ram_cache &) = delete;
    ram_cache & operator=(const ram_cache &) = delete;

    lease acquire(std::shared_ptr<const descriptor> value);
    // Return an empty lease on a miss without starting storage work.
    lease try_acquire(std::shared_ptr<const descriptor> value);
    // Advise absent exact ranges without changing retention, pins or demand
    // counters. Already resident/in-flight components need no extra advice.
    uint64_t advise_will_need(std::shared_ptr<const descriptor> value) const;
    bool erase(const expert_key & key);
    cache_stats stats() const;
    void clear();

private:
    struct entry {
        std::shared_ptr<const descriptor> desc;
        const uint8_t * data = nullptr;
        uint64_t arena_offset = 0;
        std::atomic<uint64_t> pins{0};
        uint64_t last_used = 0;
        uint64_t insertion_order = 0;
        uint32_t frequency = 0;
        bool ready = true;
        const pending_load * reserved_by = nullptr;
    };

    struct pending_load {
        enum class phase { loading, copying, ready, failed };
        phase state = phase::loading;
        std::shared_ptr<entry> result;
        std::vector<std::shared_ptr<entry>> victims;
        uint64_t staging_offset = 0;
        uint64_t arena_offset = 0;
        std::exception_ptr error;
        // Successful completion holds one pin until all coalesced callers
        // have obtained their own leases, even if the initiating caller exits.
        ~pending_load();
    };

    struct lru_less {
        bool operator()(const std::shared_ptr<entry> & a, const std::shared_ptr<entry> & b) const noexcept;
    };
    using offset_index = std::map<uint64_t, entry *>;
    using lru_index = std::set<std::shared_ptr<entry>, lru_less>;
    void touch(const std::shared_ptr<entry> & value, bool inserted = false);
    void observe(const expert_key & key);

    static bool checked_add(uint64_t a, uint64_t b, uint64_t & result);
    bool allocation_plan(
            uint64_t bytes,
            uint64_t & arena_offset,
            std::vector<std::shared_ptr<entry>> & victims,
            const pending_load * own_reservation = nullptr) const;
    bool staging_plan(uint64_t bytes, uint64_t & offset) const;
    void release_reservation(const std::shared_ptr<pending_load> & load);

    struct aligned_free {
        void operator()(uint8_t * pointer) const noexcept;
    };

    const cache_config config_;
    std::map<uint32_t, std::shared_ptr<const source>> sources_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    // Descriptor identity remains canonical even after its payload is evicted
    // or clear() drops all resident entries.
    std::map<expert_key, std::shared_ptr<const descriptor>> canonical_descriptors_;
    std::map<expert_key, uint32_t> observations_;
    std::map<expert_key, std::shared_ptr<entry>> entries_;
    offset_index by_offset_;
    lru_index by_recency_;
    std::map<expert_key, std::shared_ptr<pending_load>> pending_;
    // Concurrent fills own disjoint regions of this one bounded staging arena.
    // Storage reads and payload copies run without the metadata mutex.
    std::vector<uint8_t> staging_;
    // One fixed aligned arena makes the resident byte bound physical, not just
    // a telemetry counter. Entry pointers remain stable while leases are held.
    std::shared_ptr<uint8_t> resident_arena_;
    uint64_t clock_ = 0;
    uint64_t insertion_clock_ = 0;
    uint64_t frequency_clock_ = 0;
    cache_stats stats_;
};

// A small persistent worker pool with fixed request slots. Completion is
// consumed independently of submission order. Payloads still belong to the
// cache's fixed arenas; the queue never allocates another loading buffer.
// The cache must outlive the queue and all leases returned by it. Destruction
// discards queued requests, joins running reads, and releases untaken results.
// As with the cache, callers must stop using the object before destroying it.
class load_queue {
public:
    struct completion {
        uint64_t id = 0;
        ram_cache::lease payload;
        std::exception_ptr error;
    };
    struct queue_stats {
        uint64_t submitted = 0;
        uint64_t ready_hits = 0;
        uint64_t outstanding = 0;
        uint64_t outstanding_bytes = 0;
        uint64_t peak_outstanding_bytes = 0;
        uint64_t byte_limit = 0;
    };

    load_queue(ram_cache & cache, size_t workers, size_t requests, uint64_t byte_limit);
    ~load_queue();
    load_queue(const load_queue &) = delete;
    load_queue & operator=(const load_queue &) = delete;

    // Nonblocking admission. An ID must be unique until its result is taken.
    // False means request slots/bytes are unavailable (including oversized
    // requests). The caller can consume results or use the synchronous path.
    bool try_submit(uint64_t id, std::shared_ptr<const descriptor> value);
    // Wait only if requested and there are outstanding requests. A completed
    // storage error is returned in error, with an empty payload.
    bool next(completion & result, bool wait);
    queue_stats stats() const;

private:
    enum class phase { free, queued, running, ready };
    struct slot {
        phase state = phase::free;
        uint64_t id = 0;
        std::shared_ptr<const descriptor> desc;
        ram_cache::lease payload;
        std::exception_ptr error;
    };
    void worker();
    void stop();

    ram_cache & cache_;
    mutable std::mutex mutex_;
    std::condition_variable work_changed_;
    std::condition_variable completion_changed_;
    std::vector<slot> slots_;
    std::vector<std::thread> workers_;
    queue_stats stats_;
    bool stopping_ = false;
};

// Bounded speculative storage advice. New demand routes expire stale queued
// hints; the worker yields while the demand pool has outstanding work. Advice
// already accepted by the OS cannot be cancelled, so bytes are also bounded
// across each complete graph pass, not just across the software queue.
class read_ahead_queue {
public:
    struct queue_stats {
        uint64_t submitted = 0, advised_bytes = 0, skipped = 0, expired = 0, rejected = 0, failed = 0;
        uint64_t outstanding_bytes = 0, peak_outstanding_bytes = 0;
    };
    read_ahead_queue(ram_cache & cache, load_queue * demand, uint64_t pass_byte_limit);
    ~read_ahead_queue();
    read_ahead_queue(const read_ahead_queue &) = delete;
    read_ahead_queue & operator=(const read_ahead_queue &) = delete;
    void advance(uintptr_t owner, uint64_t pass, uint32_t layer);
    bool submit(std::shared_ptr<const descriptor> value);
    queue_stats stats() const;
    // Stop producers first; then cancel queued hints and join the advice call.
    void stop();
private:
    void worker();
    ram_cache & cache_;
    load_queue * demand_;
    const uint64_t pass_byte_limit_;
    mutable std::mutex mutex_;
    std::condition_variable changed_;
    std::deque<std::shared_ptr<const descriptor>> pending_;
    std::set<expert_key> seen_;
    uintptr_t owner_ = 0;
    uint64_t pass_ = 0, admitted_bytes_ = 0;
    uint32_t layer_ = 0;
    bool stopping_ = false;
    queue_stats stats_;
    std::thread worker_;
};

} // namespace llama_expert_cache
