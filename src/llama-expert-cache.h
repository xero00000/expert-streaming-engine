#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
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
    virtual ~source() = default;
    virtual uint32_t id() const = 0;
    virtual source_identity identity() const = 0;
    virtual uint64_t size() const = 0;
    virtual read_backend backend() const = 0;
    virtual void read(uint64_t offset, void * destination, size_t length) const = 0;
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
    const std::string & path() const;

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

struct cache_config {
    uint64_t capacity_bytes = 0;
    uint64_t staging_bytes = 0;
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
    std::array<uint64_t, size_t(read_backend::count)> bytes_read = {{ 0, 0, 0 }};
};

class capacity_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ram_cache {
private:
    struct entry;

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

    ram_cache(cache_config config, std::vector<std::shared_ptr<const source>> sources);
    ram_cache(const ram_cache &) = delete;
    ram_cache & operator=(const ram_cache &) = delete;

    lease acquire(std::shared_ptr<const descriptor> value);
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
    };

    static bool checked_add(uint64_t a, uint64_t b, uint64_t & result);
    bool allocation_plan(
            uint64_t bytes,
            uint64_t & arena_offset,
            std::vector<std::shared_ptr<entry>> & victims) const;

    struct aligned_free {
        void operator()(uint8_t * pointer) const noexcept;
    };

    const cache_config config_;
    std::map<uint32_t, std::shared_ptr<const source>> sources_;
    mutable std::mutex mutex_;
    // Descriptor identity remains canonical even after its payload is evicted
    // or clear() drops all resident entries.
    std::map<expert_key, std::shared_ptr<const descriptor>> canonical_descriptors_;
    std::map<expert_key, std::shared_ptr<entry>> entries_;
    // Single reusable fill arena. Misses are serialized by mutex_, so this
    // buffer never scales with request concurrency.
    std::vector<uint8_t> staging_;
    // One fixed aligned arena makes the resident byte bound physical, not just
    // a telemetry counter. Entry pointers remain stable while leases are held.
    std::unique_ptr<uint8_t, aligned_free> resident_arena_;
    uint64_t clock_ = 0;
    uint64_t insertion_clock_ = 0;
    cache_stats stats_;
};

} // namespace llama_expert_cache
