#pragma once

#include "llama-expert-cache.h"
#include "llama-qwen4exp-ple.h"
#include "llama-memory-accounting.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace qwen4exp_ple_storage {

namespace detail = llama_memory;

struct read_span {
    uint64_t offset = 0;
    uint64_t length = 0;

    bool operator<(const read_span & other) const {
        return offset < other.offset || (offset == other.offset && length < other.length);
    }
};

struct batch_result {
    detail::tracked_vector<std::shared_ptr<const std::vector<uint8_t>>> buffers;
};

struct async_read_stats {
    llama_expert_cache::read_backend backend = llama_expert_cache::read_backend::pread;
    uint64_t submitted_batches = 0;
    uint64_t logical_reads = 0;
    uint64_t logical_bytes = 0;
    uint64_t source_reads = 0;
    uint64_t source_bytes = 0;
    uint64_t source_batches = 0;
    uint64_t peak_source_batch_width = 0;
    uint64_t coalesced_reads = 0;
    uint64_t failed_reads = 0;
    uint64_t cancelled_batches = 0;
    uint64_t queued_reads = 0;
    uint64_t active_reads = 0;
    uint64_t peak_queue_depth = 0;
    uint64_t staging_capacity_bytes = 0;
    uint64_t staging_resident_bytes = 0;
    uint64_t staging_peak_resident_bytes = 0;
    uint64_t staging_rejected_batches = 0;
    uint64_t submit_to_ready_p50_ns = 0;
    uint64_t submit_to_ready_p95_ns = 0;
    uint64_t submit_to_ready_p99_ns = 0;
    // Live buffer capacities and shared-buffer control allocation requests.
    // staging_resident_bytes remains the logical reservation count.
    uint64_t staging_allocated_bytes = 0;
    uint64_t staging_peak_allocated_bytes = 0;
    uint64_t staging_control_allocated_bytes = 0;
    uint64_t staging_control_peak_allocated_bytes = 0;
    uint64_t staging_control_allocation_budget_bytes = UINT64_MAX;
    uint64_t metadata_allocated_bytes = 0;
    uint64_t metadata_peak_allocated_bytes = 0;
    uint64_t metadata_allocation_budget_bytes = UINT64_MAX;
    uint64_t metadata_allocation_failures = 0;
    uint64_t registered_owners = 0;
};

struct block_cache_stats {
    uint64_t capacity_bytes = 0;
    uint64_t resident_bytes = 0;
    uint64_t peak_resident_bytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t admissions = 0;
    uint64_t evictions = 0;
    uint64_t rejected_admissions = 0;
    uint64_t pinned_capacity_failures = 0;
    uint64_t active_leases = 0;
    // Backing allocation requests/capacities, separate from occupied payload.
    // Excludes allocator headers and the shared control object's storage.
    uint64_t arena_allocated_bytes = 0;
    uint64_t slot_allocated_bytes = 0;
    uint64_t index_allocated_bytes = 0;
    uint64_t index_peak_allocated_bytes = 0;
    uint64_t index_allocation_budget_bytes = UINT64_MAX;
};

class block_cache_capacity_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class staging_capacity_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Fixed-size slots make the row/page cache bound physical. A leased slot is
// never reused, so cancellation and eviction cannot invalidate consumer data.
class bounded_block_cache {
private:
    struct control;

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
        uint64_t key() const;
        explicit operator bool() const { return control_ != nullptr; }

    private:
        friend class bounded_block_cache;
        lease(std::shared_ptr<control> value, size_t slot, uint64_t key);
        void release();
        std::shared_ptr<control> control_;
        size_t slot_ = 0;
        uint64_t key_ = 0;
    };

    bounded_block_cache(uint64_t capacity_bytes, size_t block_size,
            uint64_t index_allocation_budget_bytes = UINT64_MAX);
    bounded_block_cache(const bounded_block_cache &) = delete;
    bounded_block_cache & operator=(const bounded_block_cache &) = delete;

    lease acquire(uint64_t key);
    lease admit(uint64_t key, const uint8_t * data, size_t size);
    block_cache_stats stats() const;

private:
    std::shared_ptr<control> control_;
};

// Provider-owned ticket/state controls, vectors, transient locator requests
// and owner registry allocation requests. Excludes allocator headers/impls.
struct provider_metadata_stats {
    uint64_t allocated_bytes = 0;
    uint64_t peak_allocated_bytes = 0;
    uint64_t allocation_budget_bytes = UINT64_MAX;
    uint64_t allocation_failures = 0;
    uint64_t registered_owners = 0;
};

struct row_store_stats {
    async_read_stats reads;
    block_cache_stats rows;
    provider_metadata_stats metadata;
};

struct page_store_stats {
    async_read_stats reads;
    block_cache_stats pages;
    provider_metadata_stats metadata;
};

struct two_tier_store_stats {
    async_read_stats reads;
    block_cache_stats pages;
    block_cache_stats rows;
    provider_metadata_stats metadata;
    provider_metadata_stats page_metadata;
};

class async_reader {
public:
    class ticket {
    public:
        ticket() = default;

        batch_result get() const;
        bool ready() const;
        void cancel() const;
        explicit operator bool() const { return impl_ != nullptr; }

    private:
        friend class async_reader;
        struct impl;
        explicit ticket(std::shared_ptr<impl> value);
        std::shared_ptr<impl> impl_;
    };

    async_reader(
        std::shared_ptr<const llama_expert_cache::source> source,
        size_t worker_count,
        uint64_t staging_capacity_bytes,
        size_t max_source_batch_width = 32,
        uint64_t control_allocation_budget_bytes = UINT64_MAX);
    ~async_reader();
    async_reader(const async_reader &) = delete;
    async_reader & operator=(const async_reader &) = delete;

    ticket submit(uint64_t owner, const std::vector<read_span> & spans);
    ticket submit(uint64_t owner, const read_span * spans, size_t count);
    size_t cancel_owner(uint64_t owner);
    async_read_stats stats() const;
    // Bounds tracked metadata allocation requests. Cannot lower below live use.
    bool set_metadata_allocation_budget(uint64_t bytes);

private:
    struct impl;
    std::unique_ptr<impl> impl_;
};

// Model-independent geometry for raw, fixed-size sidecar rows. Separate store
// instances own separate key spaces (e.g. one per DeepSeek Engram layer).
struct fixed_row_layout {
    uint64_t data_offset = 0;
    uint64_t row_count = 0;
    size_t row_bytes = 0;
};

class row_store {
private:
    struct impl;

public:
    class ticket;

    class staged_batch {
    public:
        size_t size() const { return rows_.size(); }
        const bounded_block_cache::lease & row(size_t index) const { return rows_.at(index); }
        void dequantize(float * output, size_t output_count) const;

    private:
        friend class ticket;
        detail::tracked_vector<bounded_block_cache::lease> rows_;
    };

    class ticket {
    public:
        ticket() = default;
        staged_batch get() const;
        void cancel() const;
        explicit operator bool() const { return state_ != nullptr; }

    private:
        friend class row_store;
        struct state;
        explicit ticket(std::shared_ptr<state> value);
        std::shared_ptr<state> state_;
    };

    row_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        qwen4exp_ple_extent_index extents,
        size_t worker_count,
        uint64_t row_cache_bytes,
        size_t max_source_batch_width = 32);

    // Raw-row mode reuses bounded leases, duplicate-read coalescing and owner
    // cancellation. Consume row() bytes with the model's own decoder; the
    // staged_batch::dequantize convenience method is Qwen IQ4_NL-only.
    row_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        fixed_row_layout layout,
        size_t worker_count,
        uint64_t row_cache_bytes,
        size_t max_source_batch_width = 32);

    ticket submit(uint64_t owner, const std::vector<uint64_t> & global_rows);
    ticket submit(uint64_t owner, const uint64_t * global_rows, size_t count);
    size_t cancel_owner(uint64_t owner);
    row_store_stats stats() const;
    // Bounds tracked ticket/control, vector and registry requests for this provider.
    bool set_metadata_allocation_budget(uint64_t bytes);

private:
    std::shared_ptr<impl> impl_;
};

// Controlled 4 KiB page-cache path. Page leases remain pinned until the
// selected rows have been consumed; rows crossing a page boundary are copied
// into a 90-byte local quantized buffer only at dequantization time.
class page_store {
private:
    struct impl;

public:
    class ticket;

    class staged_batch {
    public:
        size_t size() const { return rows_.size(); }
        void copy_quantized(size_t index, uint8_t * output, size_t output_size) const;
        void dequantize(float * output, size_t output_count) const;

    private:
        friend class ticket;
        struct row_view {
            size_t first_page = 0;
            size_t last_page = 0;
            size_t offset_in_first = 0;
            size_t length = 0;
        };
        detail::tracked_vector<bounded_block_cache::lease> pages_;
        detail::tracked_vector<row_view> rows_;
    };

    class ticket {
    public:
        ticket() = default;
        staged_batch get() const;
        void cancel() const;
        explicit operator bool() const { return state_ != nullptr; }

    private:
        friend class page_store;
        struct state;
        explicit ticket(std::shared_ptr<state> value);
        std::shared_ptr<state> state_;
    };

    page_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        qwen4exp_ple_extent_index extents,
        size_t worker_count,
        uint64_t page_cache_bytes,
        size_t max_source_batch_width = 32);

    ticket submit(uint64_t owner, const std::vector<uint64_t> & global_rows);
    ticket submit(uint64_t owner, const uint64_t * global_rows, size_t count);
    size_t cancel_owner(uint64_t owner);
    page_store_stats stats() const;
    // Bounds tracked ticket/control, vector and registry requests for this provider.
    bool set_metadata_allocation_budget(uint64_t bytes);

private:
    std::shared_ptr<impl> impl_;
};

class two_tier_store {
private:
    struct impl;

public:
    class ticket;

    class staged_batch {
    public:
        size_t size() const { return rows_.size(); }
        void dequantize(float * output, size_t output_count) const;

    private:
        friend class ticket;
        detail::tracked_vector<bounded_block_cache::lease> rows_;
    };

    class ticket {
    public:
        ticket() = default;
        staged_batch get() const;
        void cancel() const;
        explicit operator bool() const { return state_ != nullptr; }

    private:
        friend class two_tier_store;
        struct state;
        explicit ticket(std::shared_ptr<state> value);
        std::shared_ptr<state> state_;
    };

    two_tier_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        qwen4exp_ple_extent_index extents,
        size_t worker_count,
        uint64_t page_cache_bytes,
        uint64_t row_cache_bytes,
        size_t max_source_batch_width = 32);

    ticket submit(uint64_t owner, const std::vector<uint64_t> & global_rows);
    ticket submit(uint64_t owner, const uint64_t * global_rows, size_t count);
    size_t cancel_owner(uint64_t owner);
    two_tier_store_stats stats() const;
    // Bounds tracked ticket/control, vector and registry requests for this provider.
    bool set_metadata_allocation_budget(uint64_t bytes);

private:
    std::shared_ptr<impl> impl_;
};

} // namespace qwen4exp_ple_storage
