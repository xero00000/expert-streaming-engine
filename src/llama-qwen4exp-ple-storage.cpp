#include "llama-qwen4exp-ple-storage.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <future>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <utility>

namespace llama_memory {
struct allocation_counter {
    uint64_t allocated = 0;
    uint64_t peak = 0;
    uint64_t budget = UINT64_MAX;
};

struct allocation_state {
    mutable std::mutex mutex;
    allocation_counter memory;
    uint64_t failures = 0;
    uint64_t requests = 0;
};

void * allocate_metadata(const std::shared_ptr<allocation_state> & state,
        size_t count, size_t element_size, void * (*allocate)(size_t)) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (count > SIZE_MAX / element_size || count * element_size > state->memory.budget - state->memory.allocated) {
        ++state->failures;
        throw std::bad_alloc();
    }
    void * result;
    try { result = allocate(count); }
    catch (const std::bad_alloc &) { ++state->failures; throw; }
    ++state->requests;
    state->memory.allocated += count * element_size;
    state->memory.peak = std::max(state->memory.peak, state->memory.allocated);
    return result;
}

void deallocate_metadata(const std::shared_ptr<allocation_state> & state,
        void * value, size_t count, size_t element_size, void (*deallocate)(void *, size_t)) noexcept {
    std::lock_guard<std::mutex> lock(state->mutex);
    const uint64_t bytes = count * element_size;
    if (bytes > state->memory.allocated) std::terminate();
    state->memory.allocated -= bytes;
    deallocate(value, count);
}
std::shared_ptr<allocation_state> create_state() { return std::make_shared<allocation_state>(); }

allocation_stats snapshot(const std::shared_ptr<allocation_state> & state) {
    if (!state) return {};
    std::lock_guard<std::mutex> lock(state->mutex);
    return {state->memory.allocated, state->memory.peak, state->memory.budget, state->failures, state->requests};
}

bool set_budget(const std::shared_ptr<allocation_state> & state, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (bytes < state->memory.allocated) return false;
    state->memory.budget = bytes;
    return true;
}
} // namespace llama_memory

namespace qwen4exp_ple_storage {
namespace {
using clock_type = std::chrono::steady_clock;
using byte_buffer = std::vector<uint8_t>;
using shared_buffer = std::shared_ptr<const byte_buffer>;
using buffer_future = std::shared_future<shared_buffer>;
using allocation_counter = detail::allocation_counter;
using reader_metadata = detail::allocation_state;
template<class T> using reader_allocator = detail::metadata_allocator<T>;
template<class T> using reader_vector = detail::tracked_vector<T>;

// Non-owning inputs allow tracked transient vectors to pass through storage
// without copying them into another allocator's vector.
template<class T> struct input_view {
    const T * data;
    size_t count;
    bool empty() const { return count == 0; }
    size_t size() const { return count; }
    const T * begin() const { return data; }
    const T * end() const { return count ? data + count : data; }
    const T & operator[](size_t index) const { return data[index]; }
};

template<class K,class V> using reader_map=std::map<K,V,std::less<K>,reader_allocator<std::pair<const K,V>>>;

// std::map rebinds this allocator to its actual node type. Count those requests
// rather than estimating implementation-dependent node sizes from key/value.
template<class T> struct accounting_allocator {
    using value_type = T;
    allocation_counter * counter = nullptr;
    accounting_allocator() = default;
    explicit accounting_allocator(allocation_counter * value) : counter(value) {}
    template<class U> accounting_allocator(const accounting_allocator<U> & other) : counter(other.counter) {}
    T * allocate(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
        const uint64_t bytes = count * sizeof(T);
        if (counter && bytes > counter->budget - counter->allocated) throw std::bad_alloc();
        auto * result = std::allocator<T>{}.allocate(count);
        if (counter) {
            counter->allocated += bytes;
            counter->peak = std::max(counter->peak, counter->allocated);
        }
        return result;
    }
    void deallocate(T * value, size_t count) noexcept {
        if (counter) {
            const uint64_t bytes = count * sizeof(T);
            if (bytes > counter->allocated) std::terminate();
            counter->allocated -= bytes;
        }
        std::allocator<T>{}.deallocate(value, count);
    }
    template<class U> bool operator==(const accounting_allocator<U> & other) const { return counter == other.counter; }
    template<class U> bool operator!=(const accounting_allocator<U> & other) const { return !(*this == other); }
};

struct staging_budget {
    explicit staging_budget(uint64_t capacity, uint64_t control_budget)
        : capacity_bytes(capacity), controls{0,0,control_budget} {
        if (capacity == 0) throw std::invalid_argument("qwen4exp PLE staging capacity must be nonzero");
    }

    void reserve(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        if (bytes > capacity_bytes - resident_bytes) {
            ++rejected_batches;
            throw staging_capacity_error("qwen4exp PLE staging capacity is exhausted");
        }
        resident_bytes += bytes;
        peak_resident_bytes = std::max(peak_resident_bytes, resident_bytes);
    }

    void release(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        if (bytes > resident_bytes) std::terminate();
        resident_bytes -= bytes;
    }

    void allocated(uint64_t bytes) {
        std::lock_guard<std::mutex> lock(mutex);
        if (bytes > capacity_bytes - allocated_bytes)
            throw staging_capacity_error("qwen4exp PLE allocated staging capacity is exhausted");
        allocated_bytes += bytes;
        peak_allocated_bytes = std::max(peak_allocated_bytes, allocated_bytes);
    }

    void release_buffer(uint64_t bytes, uint64_t reserved) {
        std::lock_guard<std::mutex> lock(mutex);
        if (bytes > allocated_bytes || reserved > resident_bytes) std::terminate();
        allocated_bytes -= bytes;
        resident_bytes -= reserved;
    }

    void copy_stats(async_read_stats & result) const {
        std::lock_guard<std::mutex> lock(mutex);
        result.staging_capacity_bytes = capacity_bytes;
        result.staging_resident_bytes = resident_bytes;
        result.staging_peak_resident_bytes = peak_resident_bytes;
        result.staging_rejected_batches = rejected_batches;
        result.staging_allocated_bytes = allocated_bytes;
        result.staging_peak_allocated_bytes = peak_allocated_bytes;
        result.staging_control_allocated_bytes = controls.allocated;
        result.staging_control_peak_allocated_bytes = controls.peak;
        result.staging_control_allocation_budget_bytes = controls.budget;
    }

    mutable std::mutex mutex;
    const uint64_t capacity_bytes;
    uint64_t resident_bytes = 0;
    uint64_t peak_resident_bytes = 0;
    uint64_t rejected_batches = 0;
    uint64_t allocated_bytes = 0;
    uint64_t peak_allocated_bytes = 0;
    allocation_counter controls;
};

template<class T> struct staging_control_allocator {
    using value_type = T;
    std::shared_ptr<staging_budget> budget;
    explicit staging_control_allocator(std::shared_ptr<staging_budget> value) : budget(std::move(value)) {}
    template<class U> staging_control_allocator(const staging_control_allocator<U> & other) : budget(other.budget) {}
    T * allocate(size_t count) {
        if (count > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
        const uint64_t bytes = count * sizeof(T);
        std::lock_guard<std::mutex> lock(budget->mutex);
        if (bytes > budget->controls.budget - budget->controls.allocated) throw std::bad_alloc();
        auto * result = std::allocator<T>{}.allocate(count);
        budget->controls.allocated += bytes;
        budget->controls.peak = std::max(budget->controls.peak,budget->controls.allocated);
        return result;
    }
    void deallocate(T * value, size_t count) noexcept {
        std::lock_guard<std::mutex> lock(budget->mutex);
        const uint64_t bytes = count * sizeof(T);
        if (bytes > budget->controls.allocated) std::terminate();
        budget->controls.allocated -= bytes;
        std::allocator<T>{}.deallocate(value,count);
    }
    template<class U> bool operator==(const staging_control_allocator<U> & other) const { return budget==other.budget; }
    template<class U> bool operator!=(const staging_control_allocator<U> & other) const { return !(*this==other); }
};

struct read_operation {
    explicit read_operation(const std::shared_ptr<reader_metadata> & metadata)
        : promise(std::allocator_arg,reader_allocator<shared_buffer>(metadata)),future(promise.get_future().share()) {}
    read_span span;
    clock_type::time_point submitted;
    std::promise<shared_buffer> promise;
    buffer_future future;
    std::shared_ptr<staging_budget> budget;
    bool reservation_transferred = false;

    ~read_operation() {
        if (budget && !reservation_transferred) budget->release(span.length);
    }
};


} // namespace

struct bounded_block_cache::control {
    struct entry {
        uint64_t key = 0;
        uint64_t last_used = 0;
        uint64_t pins = 0;
        bool occupied = false;
    };

    control(uint64_t requested_capacity, size_t requested_block_size, uint64_t index_budget)
        : block_size(requested_block_size), index_memory{0, 0, index_budget},
          index(accounting_allocator<std::pair<const uint64_t, size_t>>(&index_memory)) {
        if (block_size == 0) throw std::invalid_argument("qwen4exp PLE cache block size must be nonzero");
        if (requested_capacity < block_size) {
            throw std::invalid_argument("qwen4exp PLE cache capacity is smaller than one block");
        }
        if (requested_capacity > std::numeric_limits<size_t>::max()) {
            throw std::overflow_error("qwen4exp PLE cache exceeds addressable memory");
        }
        const size_t slot_count = size_t(requested_capacity / block_size);
        if (slot_count > std::numeric_limits<size_t>::max() / block_size) {
            throw std::overflow_error("qwen4exp PLE cache arena overflows size_t");
        }
        arena.resize(slot_count * block_size);
        slots.resize(slot_count);
        snapshot.capacity_bytes = arena.size();
    }

    mutable std::mutex mutex;
    const size_t block_size;
    std::vector<uint8_t> arena;
    std::vector<entry> slots;
    allocation_counter index_memory;
    std::map<uint64_t, size_t, std::less<uint64_t>,
             accounting_allocator<std::pair<const uint64_t, size_t>>> index;
    uint64_t clock = 0;
    block_cache_stats snapshot;
};

bounded_block_cache::lease::lease(
        std::shared_ptr<bounded_block_cache::control> value,
        size_t slot,
        uint64_t key)
    : control_(std::move(value)), slot_(slot), key_(key) {}

bounded_block_cache::lease::lease(lease && other) noexcept
    : control_(std::move(other.control_)), slot_(other.slot_), key_(other.key_) {}

bounded_block_cache::lease & bounded_block_cache::lease::operator=(lease && other) noexcept {
    if (this != &other) {
        release();
        control_ = std::move(other.control_);
        slot_ = other.slot_;
        key_ = other.key_;
    }
    return *this;
}

bounded_block_cache::lease::~lease() { release(); }

void bounded_block_cache::lease::release() {
    if (!control_) return;
    {
        std::lock_guard<std::mutex> lock(control_->mutex);
        auto & entry = control_->slots.at(slot_);
        if (!entry.occupied || entry.key != key_ || entry.pins == 0) std::terminate();
        --entry.pins;
        --control_->snapshot.active_leases;
    }
    control_.reset();
}

const uint8_t * bounded_block_cache::lease::data() const {
    return control_ ? control_->arena.data() + slot_ * control_->block_size : nullptr;
}

size_t bounded_block_cache::lease::size() const {
    return control_ ? control_->block_size : 0;
}

uint64_t bounded_block_cache::lease::key() const { return key_; }

bounded_block_cache::bounded_block_cache(uint64_t capacity_bytes, size_t block_size, uint64_t index_budget)
    : control_(std::make_shared<control>(capacity_bytes, block_size, index_budget)) {}

bounded_block_cache::lease bounded_block_cache::acquire(uint64_t key) {
    std::lock_guard<std::mutex> lock(control_->mutex);
    const auto found = control_->index.find(key);
    if (found == control_->index.end()) {
        ++control_->snapshot.misses;
        return {};
    }
    auto & entry = control_->slots[found->second];
    ++control_->snapshot.hits;
    ++control_->snapshot.active_leases;
    ++entry.pins;
    entry.last_used = ++control_->clock;
    return lease(control_, found->second, key);
}

bounded_block_cache::lease bounded_block_cache::admit(
        uint64_t key,
        const uint8_t * data,
        size_t size) {
    if (data == nullptr || size != control_->block_size) {
        std::lock_guard<std::mutex> lock(control_->mutex);
        ++control_->snapshot.rejected_admissions;
        throw std::invalid_argument("qwen4exp PLE cache admission has invalid block geometry");
    }

    std::lock_guard<std::mutex> lock(control_->mutex);
    const auto existing = control_->index.find(key);
    if (existing != control_->index.end()) {
        auto & entry = control_->slots[existing->second];
        const uint8_t * resident = control_->arena.data() + existing->second * control_->block_size;
        if (std::memcmp(resident, data, size) != 0) {
            ++control_->snapshot.rejected_admissions;
            throw std::runtime_error("qwen4exp PLE cache key was admitted with different data");
        }
        ++entry.pins;
        entry.last_used = ++control_->clock;
        ++control_->snapshot.active_leases;
        return lease(control_, existing->second, key);
    }

    size_t target = control_->slots.size();
    for (size_t slot = 0; slot < control_->slots.size(); ++slot) {
        if (!control_->slots[slot].occupied) {
            target = slot;
            break;
        }
    }
    if (target == control_->slots.size()) {
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        for (size_t slot = 0; slot < control_->slots.size(); ++slot) {
            const auto & entry = control_->slots[slot];
            if (entry.pins == 0 && entry.last_used < oldest) {
                oldest = entry.last_used;
                target = slot;
            }
        }
    }
    if (target == control_->slots.size()) {
        ++control_->snapshot.rejected_admissions;
        ++control_->snapshot.pinned_capacity_failures;
        throw block_cache_capacity_error("qwen4exp PLE cache capacity is pinned");
    }

    // Prepare the only fallible allocation before changing slot bytes, old
    // index membership, pins or telemetry. A failed admission leaves cache
    // ownership intact, including an unpinned eviction candidate.
    control_->index.emplace(key, target);
    auto & entry = control_->slots[target];
    if (entry.occupied) {
        control_->index.erase(entry.key);
        ++control_->snapshot.evictions;
    }
    std::memcpy(control_->arena.data() + target * control_->block_size, data, size);
    entry = {key, ++control_->clock, 1, true};
    ++control_->snapshot.admissions;
    ++control_->snapshot.active_leases;
    control_->snapshot.resident_bytes = control_->index.size() * control_->block_size;
    control_->snapshot.peak_resident_bytes = std::max(
        control_->snapshot.peak_resident_bytes, control_->snapshot.resident_bytes);
    return lease(control_, target, key);
}

block_cache_stats bounded_block_cache::stats() const {
    std::lock_guard<std::mutex> lock(control_->mutex);
    auto result = control_->snapshot;
    result.arena_allocated_bytes = control_->arena.capacity();
    result.slot_allocated_bytes = control_->slots.capacity() * sizeof(control::entry);
    result.index_allocated_bytes = control_->index_memory.allocated;
    result.index_peak_allocated_bytes = control_->index_memory.peak;
    result.index_allocation_budget_bytes = control_->index_memory.budget;
    return result;
}

struct async_reader::ticket::impl {
    explicit impl(const std::shared_ptr<reader_metadata> & metadata) : futures(reader_allocator<buffer_future>(metadata)) {}
    uint64_t owner = 0;
    std::atomic<bool> cancelled{false};
    reader_vector<buffer_future> futures;
};

async_reader::ticket::ticket(std::shared_ptr<ticket::impl> value) : impl_(std::move(value)) {}

batch_result async_reader::ticket::get() const {
    if (!impl_) throw std::logic_error("qwen4exp PLE read ticket is empty");
    if (impl_->cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("qwen4exp PLE read ticket was cancelled");
    }
    batch_result result{reader_vector<shared_buffer>{impl_->futures.get_allocator()}};
    result.buffers.reserve(impl_->futures.size());
    for (const auto & future : impl_->futures) {
        shared_buffer buffer = future.get();
        if (impl_->cancelled.load(std::memory_order_acquire)) {
            throw std::runtime_error("qwen4exp PLE read ticket was cancelled");
        }
        result.buffers.push_back(std::move(buffer));
    }
    return result;
}

bool async_reader::ticket::ready() const {
    if (!impl_) return false;
    for (const auto & future : impl_->futures) {
        if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) return false;
    }
    return true;
}

void async_reader::ticket::cancel() const {
    if (impl_) impl_->cancelled.store(true, std::memory_order_release);
}

struct async_reader::impl {
    explicit impl(
            std::shared_ptr<const llama_expert_cache::source> value,
            size_t worker_count,
            uint64_t staging_capacity_bytes,
            size_t requested_max_source_batch_width,
            uint64_t control_budget)
        : metadata(std::make_shared<reader_metadata>()), source(std::move(value)),
          staging(std::make_shared<staging_budget>(staging_capacity_bytes,control_budget)),
          queue(reader_allocator<std::shared_ptr<read_operation>>(metadata)),
          inflight(reader_allocator<std::pair<const read_span,std::weak_ptr<read_operation>>>(metadata)),
          owners(reader_allocator<std::pair<const uint64_t,owner_vector>>(metadata)),
          workers(reader_allocator<std::thread>(metadata)),
          latency_samples(reader_allocator<uint64_t>(metadata)),
          max_source_batch_width(requested_max_source_batch_width) {
        if (!source) throw std::invalid_argument("qwen4exp PLE async reader requires a source");
        if (worker_count == 0) throw std::invalid_argument("qwen4exp PLE async reader requires workers");
        if (max_source_batch_width == 0 || max_source_batch_width > 64) {
            throw std::invalid_argument("qwen4exp PLE source batch width must be between 1 and 64");
        }
        const auto identity = source->identity();
        if (identity.high == 0 && identity.low == 0) {
            throw std::invalid_argument("qwen4exp PLE async reader source identity cannot be zero");
        }
        snapshot.backend = source->backend();
        latency_samples.resize(max_latency_samples);
        workers.reserve(worker_count);
        try {
            for (size_t index = 0; index < worker_count; ++index)
                workers.emplace_back([this] { worker_loop(); });
        } catch (...) {
            { std::lock_guard<std::mutex> lock(mutex);stopping=true; }
            ready.notify_all();for(auto & worker:workers)worker.join();throw;
        }
    }

    ~impl() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto & worker : workers) worker.join();
    }

    void worker_loop() {
        while (true) {
            reader_vector<std::shared_ptr<read_operation>> operations{reader_allocator<std::shared_ptr<read_operation>>(metadata)};
            {
                std::unique_lock<std::mutex> lock(mutex);
                ready.wait(lock, [&] { return stopping || !queue.empty(); });
                if (stopping && queue.empty()) return;
                const size_t count = std::min(queue.size(), max_source_batch_width);
                try { operations.reserve(count); }
                catch (...) {
                    const auto error=std::current_exception();
                    // No batch storage was published. Complete queued futures
                    // without allocating a replacement work list.
                    for(size_t i=0;i<count;++i) {
                        auto operation=std::move(queue.front());queue.pop_front();
                        --snapshot.queued_reads;++snapshot.failed_reads;
                        auto found=inflight.find(operation->span);
                        if(found!=inflight.end() && found->second.lock()==operation)inflight.erase(found);
                        operation->promise.set_exception(error);
                    }
                    continue;
                }
                for (size_t index = 0; index < count; ++index) {
                    operations.push_back(std::move(queue.front()));
                    queue.pop_front();
                }
                snapshot.queued_reads -= count;
                snapshot.active_reads += count;
            }

            reader_vector<std::shared_ptr<byte_buffer>> buffers{reader_allocator<std::shared_ptr<byte_buffer>>(metadata)};
            std::exception_ptr failure;
            try {
                buffers.reserve(operations.size());
                std::vector<llama_expert_cache::source::read_request> requests;
                requests.reserve(operations.size());
                for (const auto & operation : operations) {
                    if (operation->span.length > std::numeric_limits<size_t>::max()) {
                        throw std::overflow_error("qwen4exp PLE read exceeds addressable memory");
                    }
                    auto raw = std::make_unique<byte_buffer>(size_t(operation->span.length));
                    const uint64_t capacity = raw->capacity();
                    operation->budget->allocated(capacity);
                    // shared_ptr invokes the deleter if its control allocation
                    // fails. Transfer first, so the operation cannot release
                    // the same reservation again during stack unwinding.
                    operation->reservation_transferred = true;
                    auto buffer = std::shared_ptr<byte_buffer>(raw.release(),
                        [budget = operation->budget, capacity, bytes = operation->span.length](byte_buffer * value) {
                            delete value;
                            budget->release_buffer(capacity,bytes);
                        },staging_control_allocator<byte_buffer>(operation->budget));
                    requests.push_back({operation->span.offset, buffer->data(), buffer->size()});
                    buffers.push_back(std::move(buffer));
                }
                source->read_many(requests);
            } catch (...) {
                failure = std::current_exception();
            }

            const auto completed = clock_type::now();
            {
                std::lock_guard<std::mutex> lock(mutex);
                const size_t batch_size=operations.size();
                ++snapshot.source_batches;
                snapshot.peak_source_batch_width = std::max<uint64_t>(
                    snapshot.peak_source_batch_width, operations.size());
                if (!failure) {
                    snapshot.source_reads += operations.size();
                } else {
                    snapshot.failed_reads += operations.size();
                }
                for (size_t index = 0; index < operations.size(); ++index) {
                    const auto & operation = operations[index];
                    if (!failure) snapshot.source_bytes += operation->span.length;
                    const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        completed - operation->submitted).count();
                    latency_samples[latency_next]=uint64_t(std::max<int64_t>(0,elapsed));
                    latency_next=(latency_next+1)%max_latency_samples;
                    latency_size=std::min(latency_size+1,max_latency_samples);
                    auto found = inflight.find(operation->span);
                    if (found != inflight.end() && found->second.lock() == operation) {
                        inflight.erase(found);
                    }
                    if (failure) {
                        operation->promise.set_exception(failure);
                    } else {
                        operation->promise.set_value(shared_buffer(buffers[index]));
                    }
                }
                // Publish idle only after worker-owned metadata and operation
                // reservations are released. Future consumers retain their own
                // data/control storage independently.
                { reader_vector<std::shared_ptr<byte_buffer>> empty{reader_allocator<std::shared_ptr<byte_buffer>>(metadata)};buffers.swap(empty); }
                { reader_vector<std::shared_ptr<read_operation>> empty{reader_allocator<std::shared_ptr<read_operation>>(metadata)};operations.swap(empty); }
                snapshot.active_reads -= batch_size;
            }
        }
    }

    using owner_vector=reader_vector<std::weak_ptr<ticket::impl>>;
    std::shared_ptr<reader_metadata> metadata;
    std::shared_ptr<const llama_expert_cache::source> source;
    std::shared_ptr<staging_budget> staging;
    mutable std::mutex mutex;
    std::condition_variable ready;
    bool stopping = false;
    std::deque<std::shared_ptr<read_operation>,reader_allocator<std::shared_ptr<read_operation>>> queue;
    reader_map<read_span,std::weak_ptr<read_operation>> inflight;
    reader_map<uint64_t,owner_vector> owners;
    reader_vector<std::thread> workers;
    async_read_stats snapshot;
    reader_vector<uint64_t> latency_samples;
    size_t latency_size=0,latency_next=0;
    static constexpr size_t max_latency_samples = 4096;
    const size_t max_source_batch_width;
};

async_reader::async_reader(
        std::shared_ptr<const llama_expert_cache::source> source,
        size_t worker_count,
        uint64_t staging_capacity_bytes,
        size_t max_source_batch_width,
        uint64_t control_budget)
    : impl_(new impl(std::move(source), worker_count, staging_capacity_bytes, max_source_batch_width,control_budget)) {}

async_reader::~async_reader() = default;

async_reader::ticket async_reader::submit(uint64_t owner, const std::vector<read_span> & spans) {
    return submit(owner, spans.data(), spans.size());
}

async_reader::ticket async_reader::submit(uint64_t owner, const read_span * data, size_t count) {
    if (!data && count) throw std::invalid_argument("PLE read input is null");
    const input_view<read_span> spans{data, count};
    if (spans.empty()) throw std::invalid_argument("qwen4exp PLE read batch cannot be empty");
    uint64_t logical_bytes = 0;
    const uint64_t source_size = impl_->source->size();
    for (const read_span & span : spans) {
        if (span.length == 0 || span.length > std::numeric_limits<size_t>::max() ||
                span.offset > source_size || span.length > source_size - span.offset) {
            throw std::out_of_range("qwen4exp PLE read span is outside its source");
        }
        if (span.length > std::numeric_limits<uint64_t>::max() - logical_bytes) {
            throw std::overflow_error("qwen4exp PLE batch byte count overflows uint64");
        }
        logical_bytes += span.length;
    }
    auto state = std::allocate_shared<ticket::impl>(reader_allocator<ticket::impl>(impl_->metadata),impl_->metadata);
    state->owner = owner;
    state->futures.reserve(spans.size());

    {
        std::lock_guard<std::mutex> lock(impl_->mutex);
        if (impl_->stopping) throw std::runtime_error("qwen4exp PLE async reader is stopping");
        // Expired owner records must not grow with every accepted token.
        for(auto it=impl_->owners.begin();it!=impl_->owners.end();) {
            auto & tickets=it->second;
            tickets.erase(std::remove_if(tickets.begin(),tickets.end(),[](const auto & value){return value.expired();}),tickets.end());
            if(tickets.empty())it=impl_->owners.erase(it);else ++it;
        }

        reader_map<read_span,std::shared_ptr<read_operation>> pending{
            reader_allocator<std::pair<const read_span,std::shared_ptr<read_operation>>>(impl_->metadata)};
        uint64_t staging_bytes = 0;
        for (const read_span & span : spans) {
            auto found = impl_->inflight.find(span);
            const bool already_inflight = found != impl_->inflight.end() && !found->second.expired();
            if (already_inflight || pending.count(span) != 0) continue;
            if (span.length > std::numeric_limits<uint64_t>::max() - staging_bytes)
                throw std::overflow_error("qwen4exp PLE staging byte count overflows uint64");
            auto operation = std::allocate_shared<read_operation>(reader_allocator<read_operation>(impl_->metadata),impl_->metadata);
            operation->span = span;
            operation->submitted = clock_type::now();
            pending.emplace(span, std::move(operation));
            staging_bytes += span.length;
        }
        impl_->staging->reserve(staging_bytes);
        for (auto & entry : pending) entry.second->budget = impl_->staging;

        const size_t previous_queue=impl_->queue.size();
        const bool new_owner=impl_->owners.find(owner)==impl_->owners.end();
        try {
            for(const auto & entry:pending) {
                impl_->inflight.insert_or_assign(entry.first,std::weak_ptr<read_operation>(entry.second));
                impl_->queue.push_back(entry.second);
            }
            for(const auto & span:spans) {
                auto operation=impl_->inflight.at(span).lock();
                if(!operation)throw std::logic_error("PLE publication lost its operation");
                state->futures.push_back(operation->future);
            }
            auto owner_entry=impl_->owners.try_emplace(owner,impl::owner_vector{reader_allocator<std::weak_ptr<ticket::impl>>(impl_->metadata)});
            owner_entry.first->second.push_back(state);
        } catch (...) {
            while(impl_->queue.size()>previous_queue)impl_->queue.pop_back();
            for(const auto & entry:pending) {
                auto found=impl_->inflight.find(entry.first);
                if(found!=impl_->inflight.end() && found->second.lock()==entry.second)impl_->inflight.erase(found);
            }
            if(new_owner)impl_->owners.erase(owner);
            throw;
        }
        ++impl_->snapshot.submitted_batches;
        impl_->snapshot.logical_reads += spans.size();
        impl_->snapshot.logical_bytes += logical_bytes;
        impl_->snapshot.coalesced_reads += spans.size()-pending.size();
        impl_->snapshot.queued_reads += pending.size();
        impl_->snapshot.peak_queue_depth=std::max(impl_->snapshot.peak_queue_depth,
            impl_->snapshot.queued_reads+impl_->snapshot.active_reads);
    }
    impl_->ready.notify_all();
    return ticket(std::move(state));
}

size_t async_reader::cancel_owner(uint64_t owner) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    auto found = impl_->owners.find(owner);
    if (found == impl_->owners.end()) return 0;
    size_t cancelled = 0;
    for (const auto & weak : found->second) {
        if (auto state = weak.lock()) {
            bool expected = false;
            if (state->cancelled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                ++cancelled;
            }
        }
    }
    impl_->snapshot.cancelled_batches += cancelled;
    impl_->owners.erase(found);
    return cancelled;
}

async_read_stats async_reader::stats() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    async_read_stats result = impl_->snapshot;
    // Fixed scratch avoids allocations while publishing completion or querying
    // telemetry under a tight metadata budget. Sample order is immaterial.
    std::array<uint64_t,impl::max_latency_samples> samples;
    std::copy_n(impl_->latency_samples.begin(),impl_->latency_size,samples.begin());
    std::sort(samples.begin(),samples.begin()+impl_->latency_size);
    const auto quantile=[&](size_t numerator) -> uint64_t {
        if(!impl_->latency_size)return 0;
        const size_t rank=(impl_->latency_size*numerator+99)/100;
        return samples[std::max<size_t>(1,rank)-1];
    };
    result.submit_to_ready_p50_ns=quantile(50);
    result.submit_to_ready_p95_ns=quantile(95);
    result.submit_to_ready_p99_ns=quantile(99);
    result.registered_owners=impl_->owners.size();
    {
        std::lock_guard<std::mutex> lock(impl_->metadata->mutex);
        result.metadata_allocated_bytes=impl_->metadata->memory.allocated;
        result.metadata_peak_allocated_bytes=impl_->metadata->memory.peak;
        result.metadata_allocation_budget_bytes=impl_->metadata->memory.budget;
        result.metadata_allocation_failures=impl_->metadata->failures;
    }
    impl_->staging->copy_stats(result);
    return result;
}

bool async_reader::set_metadata_allocation_budget(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(impl_->metadata->mutex);
    if(bytes<impl_->metadata->memory.allocated)return false;
    impl_->metadata->memory.budget=bytes;return true;
}

namespace {
template<class Owners, class State, class Dispatch>
void publish_provider_ticket(std::mutex & mutex, Owners & owners, uint64_t owner,
        const std::shared_ptr<State> & state, Dispatch dispatch) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it = owners.begin(); it != owners.end();) {
        auto & tickets = it->second;
        tickets.erase(std::remove_if(tickets.begin(), tickets.end(),
                [](const auto & value) { return value.expired(); }), tickets.end());
        if (tickets.empty()) it = owners.erase(it); else ++it;
    }
    const bool new_owner = owners.find(owner) == owners.end();
    auto entry = owners.try_emplace(owner, typename Owners::mapped_type{
            typename Owners::mapped_type::allocator_type{owners.get_allocator().state}}).first;
    try {
        entry->second.push_back(state);
    } catch (...) {
        if (new_owner) owners.erase(entry);
        throw;
    }
    try {
        dispatch();
    } catch (...) {
        entry->second.pop_back();
        if (entry->second.empty()) owners.erase(entry);
        throw;
    }
}

template<class Owners>
provider_metadata_stats provider_metadata_snapshot(std::mutex & mutex, const Owners & owners,
        const std::shared_ptr<reader_metadata> & metadata) {
    std::lock_guard<std::mutex> owner_lock(mutex);
    std::lock_guard<std::mutex> memory_lock(metadata->mutex);
    return {metadata->memory.allocated, metadata->memory.peak, metadata->memory.budget,
            metadata->failures, owners.size()};
}

bool set_provider_metadata_budget(const std::shared_ptr<reader_metadata> & metadata, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(metadata->mutex);
    if (bytes < metadata->memory.allocated) return false;
    metadata->memory.budget = bytes;
    return true;
}
} // namespace

struct row_store::impl {
    using owner_vector = reader_vector<std::weak_ptr<ticket::state>>;
    impl(
            std::shared_ptr<const llama_expert_cache::source> source,
            std::function<read_span(uint64_t)> locator,
            size_t row_bytes,
            size_t worker_count,
            uint64_t row_cache_bytes,
            size_t max_source_batch_width)
        : metadata(std::make_shared<reader_metadata>()),
          reader(std::move(source), worker_count, row_cache_bytes, max_source_batch_width),
          cache(row_cache_bytes, row_bytes),
          locate(std::move(locator)),
          owners(reader_allocator<std::pair<const uint64_t,owner_vector>>(metadata)) {}

    std::shared_ptr<reader_metadata> metadata;
    async_reader reader;
    bounded_block_cache cache;
    std::function<read_span(uint64_t)> locate;
    mutable std::mutex owners_mutex;
    reader_map<uint64_t, owner_vector> owners;
};

struct row_store::ticket::state {
    explicit state(const std::shared_ptr<reader_metadata> & metadata)
        : cached(reader_allocator<bounded_block_cache::lease>(metadata)),
          keys(reader_allocator<uint64_t>(metadata)),
          miss_positions(reader_allocator<size_t>(metadata)) {}
    std::shared_ptr<row_store::impl> store;
    uint64_t owner = 0;
    std::atomic<bool> cancelled{false};
    mutable std::mutex consume_mutex;
    mutable bool consumed = false;
    reader_vector<bounded_block_cache::lease> cached;
    reader_vector<uint64_t> keys;
    reader_vector<size_t> miss_positions;
    async_reader::ticket reads;
};

row_store::ticket::ticket(std::shared_ptr<ticket::state> value) : state_(std::move(value)) {}

row_store::staged_batch row_store::ticket::get() const {
    if (!state_) throw std::logic_error("qwen4exp PLE row ticket is empty");
    std::lock_guard<std::mutex> consume_lock(state_->consume_mutex);
    if (state_->consumed) throw std::logic_error("qwen4exp PLE row ticket was already consumed");
    state_->consumed = true;
    if (state_->cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("qwen4exp PLE row ticket was cancelled");
    }

    staged_batch result;
    result.rows_ = std::move(state_->cached);
    if (state_->reads) {
        const batch_result loaded = state_->reads.get();
        if (loaded.buffers.size() != state_->miss_positions.size()) {
            throw std::runtime_error("qwen4exp PLE row batch completed with the wrong buffer count");
        }
        for (size_t index = 0; index < loaded.buffers.size(); ++index) {
            if (state_->cancelled.load(std::memory_order_acquire)) {
                throw std::runtime_error("qwen4exp PLE row ticket was cancelled");
            }
            const size_t position = state_->miss_positions[index];
            result.rows_[position] = state_->store->cache.admit(
                state_->keys[position], loaded.buffers[index]->data(), loaded.buffers[index]->size());
        }
    }
    return result;
}

void row_store::ticket::cancel() const {
    if (!state_) return;
    state_->cancelled.store(true, std::memory_order_release);
    state_->reads.cancel();
}

void row_store::staged_batch::dequantize(float * output, size_t output_count) const {
    constexpr size_t values_per_row = 160;
    if (output == nullptr || rows_.size() > std::numeric_limits<size_t>::max() / values_per_row ||
            output_count != rows_.size() * values_per_row) {
        throw std::invalid_argument("qwen4exp PLE staged output geometry is invalid");
    }
    for (size_t index = 0; index < rows_.size(); ++index) {
        qwen4exp_ple_dequantize_iq4_nl(
            rows_[index].data(), rows_[index].size(), output + index * values_per_row, values_per_row);
    }
}

row_store::row_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        qwen4exp_ple_extent_index extents,
        size_t worker_count,
        uint64_t row_cache_bytes,
        size_t max_source_batch_width)
    : impl_(std::make_shared<impl>(
        std::move(source), [extents](uint64_t row) {
            const auto extent = extents.locate_global(row);
            return read_span{extent.offset, extent.length};
        }, 90, worker_count, row_cache_bytes, max_source_batch_width)) {}

row_store::row_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        fixed_row_layout layout,
        size_t worker_count,
        uint64_t row_cache_bytes,
        size_t max_source_batch_width) {
    if (!source || layout.row_bytes == 0 || layout.data_offset > source->size() ||
            layout.row_count > (source->size() - layout.data_offset) / layout.row_bytes) {
        throw std::invalid_argument("fixed row layout exceeds source extent");
    }
    impl_ = std::make_shared<impl>(std::move(source), [layout](uint64_t row) {
        if (row >= layout.row_count) throw std::out_of_range("fixed row ID exceeds table");
        return read_span{layout.data_offset + row * layout.row_bytes, layout.row_bytes};
    }, layout.row_bytes, worker_count, row_cache_bytes, max_source_batch_width);
}

row_store::ticket row_store::submit(uint64_t owner, const std::vector<uint64_t> & global_rows) {
    return submit(owner, global_rows.data(), global_rows.size());
}

row_store::ticket row_store::submit(uint64_t owner, const uint64_t * data, size_t count) {
    if (!data && count) throw std::invalid_argument("PLE row input is null");
    const input_view<uint64_t> global_rows{data, count};
    if (global_rows.empty()) throw std::invalid_argument("qwen4exp PLE row batch cannot be empty");
    auto state = std::allocate_shared<ticket::state>(reader_allocator<ticket::state>(impl_->metadata), impl_->metadata);
    state->store = impl_;
    state->owner = owner;
    state->cached.resize(global_rows.size());
    state->keys.assign(global_rows.begin(), global_rows.end());

    reader_vector<read_span> spans{reader_allocator<read_span>(impl_->metadata)};
    spans.reserve(global_rows.size());
    state->miss_positions.reserve(global_rows.size());
    for (size_t index = 0; index < global_rows.size(); ++index) {
        const uint64_t row = global_rows[index];
        state->cached[index] = impl_->cache.acquire(row);
        if (!state->cached[index]) {
            const auto extent = impl_->locate(row);
            state->miss_positions.push_back(index);
            spans.push_back({extent.offset, extent.length});
        }
    }

    publish_provider_ticket(impl_->owners_mutex, impl_->owners, owner, state, [&] {
        if (!spans.empty()) state->reads = impl_->reader.submit(owner, spans.data(), spans.size());
    });
    return ticket(std::move(state));
}

size_t row_store::cancel_owner(uint64_t owner) {
    std::lock_guard<std::mutex> lock(impl_->owners_mutex);
    size_t cancelled = 0;
    auto found = impl_->owners.find(owner);
    if (found != impl_->owners.end()) {
        for (const auto & weak : found->second) {
            if (auto state = weak.lock()) {
                bool expected = false;
                if (state->cancelled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    ++cancelled;
                }
            }
        }
        impl_->owners.erase(found);
    }
    (void) impl_->reader.cancel_owner(owner);
    return cancelled;
}

bool row_store::set_metadata_allocation_budget(uint64_t bytes) {
    return set_provider_metadata_budget(impl_->metadata, bytes);
}

row_store_stats row_store::stats() const {
    return {impl_->reader.stats(), impl_->cache.stats(),
            provider_metadata_snapshot(impl_->owners_mutex, impl_->owners, impl_->metadata)};
}

namespace {
constexpr size_t ple_page_size = 4096;
constexpr size_t ple_quantized_row_size = 90;
constexpr size_t ple_values_per_row = 160;
}

struct page_store::impl {
    using owner_vector = reader_vector<std::weak_ptr<ticket::state>>;
    impl(
            std::shared_ptr<const llama_expert_cache::source> value,
            qwen4exp_ple_extent_index extent_index,
            size_t worker_count,
            uint64_t page_cache_bytes,
            size_t max_source_batch_width)
        : metadata(std::make_shared<reader_metadata>()),
          source(std::move(value)),
          reader(source, worker_count, page_cache_bytes, max_source_batch_width),
          cache(page_cache_bytes, ple_page_size),
          extents(std::move(extent_index)),
          owners(reader_allocator<std::pair<const uint64_t,owner_vector>>(metadata)) {}

    std::shared_ptr<reader_metadata> metadata;
    std::shared_ptr<const llama_expert_cache::source> source;
    async_reader reader;
    bounded_block_cache cache;
    qwen4exp_ple_extent_index extents;
    mutable std::mutex owners_mutex;
    reader_map<uint64_t, owner_vector> owners;
};

struct page_store::ticket::state {
    explicit state(const std::shared_ptr<reader_metadata> & metadata)
        : pages(reader_allocator<bounded_block_cache::lease>(metadata)),
          rows(reader_allocator<staged_batch::row_view>(metadata)),
          page_keys(reader_allocator<uint64_t>(metadata)),
          miss_positions(reader_allocator<size_t>(metadata)) {}
    std::shared_ptr<page_store::impl> store;
    uint64_t owner = 0;
    std::atomic<bool> cancelled{false};
    mutable std::mutex consume_mutex;
    mutable bool consumed = false;
    reader_vector<bounded_block_cache::lease> pages;
    reader_vector<staged_batch::row_view> rows;
    reader_vector<uint64_t> page_keys;
    reader_vector<size_t> miss_positions;
    async_reader::ticket reads;
};

page_store::ticket::ticket(std::shared_ptr<ticket::state> value) : state_(std::move(value)) {}

page_store::staged_batch page_store::ticket::get() const {
    if (!state_) throw std::logic_error("qwen4exp PLE page ticket is empty");
    std::lock_guard<std::mutex> consume_lock(state_->consume_mutex);
    if (state_->consumed) throw std::logic_error("qwen4exp PLE page ticket was already consumed");
    state_->consumed = true;
    if (state_->cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("qwen4exp PLE page ticket was cancelled");
    }

    staged_batch result;
    result.pages_ = std::move(state_->pages);
    result.rows_ = std::move(state_->rows);
    if (state_->reads) {
        const batch_result loaded = state_->reads.get();
        if (loaded.buffers.size() != state_->miss_positions.size()) {
            throw std::runtime_error("qwen4exp PLE page batch completed with the wrong buffer count");
        }
        std::array<uint8_t, ple_page_size> padded = {};
        for (size_t index = 0; index < loaded.buffers.size(); ++index) {
            if (state_->cancelled.load(std::memory_order_acquire)) {
                throw std::runtime_error("qwen4exp PLE page ticket was cancelled");
            }
            const auto & buffer = *loaded.buffers[index];
            if (buffer.empty() || buffer.size() > ple_page_size) {
                throw std::runtime_error("qwen4exp PLE page read returned invalid geometry");
            }
            const uint8_t * data = buffer.data();
            if (buffer.size() != ple_page_size) {
                padded.fill(0);
                std::memcpy(padded.data(), buffer.data(), buffer.size());
                data = padded.data();
            }
            const size_t position = state_->miss_positions[index];
            result.pages_[position] = state_->store->cache.admit(
                state_->page_keys[position], data, ple_page_size);
        }
    }
    return result;
}

void page_store::ticket::cancel() const {
    if (!state_) return;
    state_->cancelled.store(true, std::memory_order_release);
    state_->reads.cancel();
}

void page_store::staged_batch::copy_quantized(
        size_t index, uint8_t * output, size_t output_size) const {
    if (output == nullptr) throw std::invalid_argument("qwen4exp PLE page row output is null");
    const row_view & row = rows_.at(index);
    if (output_size != row.length || row.first_page >= pages_.size() || row.last_page >= pages_.size()) {
        throw std::invalid_argument("qwen4exp PLE page row output geometry is invalid");
    }
    const size_t first_bytes = std::min(row.length, ple_page_size - row.offset_in_first);
    std::memcpy(output, pages_[row.first_page].data() + row.offset_in_first, first_bytes);
    if (first_bytes != row.length) {
        if (row.last_page == row.first_page) {
            throw std::runtime_error("qwen4exp PLE page row boundary metadata is inconsistent");
        }
        std::memcpy(output + first_bytes, pages_[row.last_page].data(), row.length - first_bytes);
    }
}

void page_store::staged_batch::dequantize(float * output, size_t output_count) const {
    if (output == nullptr || rows_.size() > std::numeric_limits<size_t>::max() / ple_values_per_row ||
            output_count != rows_.size() * ple_values_per_row) {
        throw std::invalid_argument("qwen4exp PLE page staged output geometry is invalid");
    }
    std::array<uint8_t, ple_quantized_row_size> row = {};
    for (size_t index = 0; index < rows_.size(); ++index) {
        if (rows_[index].length != row.size()) {
            throw std::runtime_error("qwen4exp PLE page store requires 90-byte IQ4_NL rows");
        }
        copy_quantized(index, row.data(), row.size());
        qwen4exp_ple_dequantize_iq4_nl(
            row.data(), row.size(), output + index * ple_values_per_row, ple_values_per_row);
    }
}

page_store::page_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        qwen4exp_ple_extent_index extents,
        size_t worker_count,
        uint64_t page_cache_bytes,
        size_t max_source_batch_width)
    : impl_(std::make_shared<impl>(
        std::move(source), std::move(extents), worker_count, page_cache_bytes, max_source_batch_width)) {}

page_store::ticket page_store::submit(uint64_t owner, const std::vector<uint64_t> & global_rows) {
    return submit(owner, global_rows.data(), global_rows.size());
}

page_store::ticket page_store::submit(uint64_t owner, const uint64_t * data, size_t count) {
    if (!data && count) throw std::invalid_argument("PLE row input is null");
    const input_view<uint64_t> global_rows{data, count};
    if (global_rows.empty()) throw std::invalid_argument("qwen4exp PLE page batch cannot be empty");
    auto state = std::allocate_shared<ticket::state>(reader_allocator<ticket::state>(impl_->metadata), impl_->metadata);
    state->store = impl_;
    state->owner = owner;
    state->rows.reserve(global_rows.size());

    reader_map<uint64_t, size_t> page_positions{
            reader_allocator<std::pair<const uint64_t, size_t>>(impl_->metadata)};
    reader_vector<read_span> spans{reader_allocator<read_span>(impl_->metadata)};
    for (uint64_t global_row : global_rows) {
        const qwen4exp_ple_extent extent = impl_->extents.locate_global(global_row);
        if (extent.length != ple_quantized_row_size || extent.last_page < extent.first_page ||
                extent.last_page - extent.first_page > 1) {
            throw std::runtime_error("qwen4exp PLE row has unsupported page geometry");
        }
        size_t first_position = 0;
        size_t last_position = 0;
        for (uint64_t page = extent.first_page; page <= extent.last_page; ++page) {
            auto found = page_positions.find(page);
            size_t position;
            if (found != page_positions.end()) {
                position = found->second;
            } else {
                position = state->pages.size();
                page_positions.emplace(page, position);
                state->page_keys.push_back(page);
                state->pages.push_back(impl_->cache.acquire(page));
                if (!state->pages.back()) {
                    const uint64_t page_offset = page * uint64_t(ple_page_size);
                    if (page_offset >= impl_->source->size()) {
                        throw std::out_of_range("qwen4exp PLE page starts outside its source");
                    }
                    const uint64_t length = std::min<uint64_t>(
                        ple_page_size, impl_->source->size() - page_offset);
                    state->miss_positions.push_back(position);
                    spans.push_back({page_offset, length});
                }
            }
            if (page == extent.first_page) first_position = position;
            if (page == extent.last_page) last_position = position;
        }
        state->rows.push_back({
            first_position,
            last_position,
            size_t(extent.offset % ple_page_size),
            size_t(extent.length),
        });
    }

    publish_provider_ticket(impl_->owners_mutex, impl_->owners, owner, state, [&] {
        if (!spans.empty()) state->reads = impl_->reader.submit(owner, spans.data(), spans.size());
    });
    return ticket(std::move(state));
}

size_t page_store::cancel_owner(uint64_t owner) {
    std::lock_guard<std::mutex> lock(impl_->owners_mutex);
    size_t cancelled = 0;
    auto found = impl_->owners.find(owner);
    if (found != impl_->owners.end()) {
        for (const auto & weak : found->second) {
            if (auto state = weak.lock()) {
                bool expected = false;
                if (state->cancelled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    ++cancelled;
                }
            }
        }
        impl_->owners.erase(found);
    }
    (void) impl_->reader.cancel_owner(owner);
    return cancelled;
}

bool page_store::set_metadata_allocation_budget(uint64_t bytes) {
    return set_provider_metadata_budget(impl_->metadata, bytes);
}

page_store_stats page_store::stats() const {
    return {impl_->reader.stats(), impl_->cache.stats(),
            provider_metadata_snapshot(impl_->owners_mutex, impl_->owners, impl_->metadata)};
}

struct two_tier_store::impl {
    using owner_vector = reader_vector<std::weak_ptr<ticket::state>>;
    impl(
            std::shared_ptr<const llama_expert_cache::source> source,
            qwen4exp_ple_extent_index extents,
            size_t worker_count,
            uint64_t page_cache_bytes,
            uint64_t row_cache_bytes,
            size_t max_source_batch_width)
        : metadata(std::make_shared<reader_metadata>()),
          pages(std::move(source), std::move(extents), worker_count, page_cache_bytes, max_source_batch_width),
          rows(row_cache_bytes, ple_quantized_row_size),
          owners(reader_allocator<std::pair<const uint64_t,owner_vector>>(metadata)) {}

    std::shared_ptr<reader_metadata> metadata;
    page_store pages;
    bounded_block_cache rows;
    mutable std::mutex owners_mutex;
    reader_map<uint64_t, owner_vector> owners;
};

struct two_tier_store::ticket::state {
    explicit state(const std::shared_ptr<reader_metadata> & metadata)
        : cached(reader_allocator<bounded_block_cache::lease>(metadata)),
          keys(reader_allocator<uint64_t>(metadata)),
          miss_positions(reader_allocator<size_t>(metadata)) {}
    std::shared_ptr<two_tier_store::impl> store;
    uint64_t owner = 0;
    std::atomic<bool> cancelled{false};
    mutable std::mutex consume_mutex;
    mutable bool consumed = false;
    reader_vector<bounded_block_cache::lease> cached;
    reader_vector<uint64_t> keys;
    reader_vector<size_t> miss_positions;
    page_store::ticket pages;
};

two_tier_store::ticket::ticket(std::shared_ptr<ticket::state> value) : state_(std::move(value)) {}

two_tier_store::staged_batch two_tier_store::ticket::get() const {
    if (!state_) throw std::logic_error("qwen4exp PLE two-tier ticket is empty");
    std::lock_guard<std::mutex> consume_lock(state_->consume_mutex);
    if (state_->consumed) throw std::logic_error("qwen4exp PLE two-tier ticket was already consumed");
    state_->consumed = true;
    if (state_->cancelled.load(std::memory_order_acquire)) {
        throw std::runtime_error("qwen4exp PLE two-tier ticket was cancelled");
    }

    staged_batch result;
    result.rows_ = std::move(state_->cached);
    if (state_->pages) {
        page_store::staged_batch loaded = state_->pages.get();
        if (loaded.size() != state_->miss_positions.size()) {
            throw std::runtime_error("qwen4exp PLE two-tier page batch has the wrong row count");
        }
        std::array<uint8_t, ple_quantized_row_size> row = {};
        for (size_t index = 0; index < loaded.size(); ++index) {
            if (state_->cancelled.load(std::memory_order_acquire)) {
                throw std::runtime_error("qwen4exp PLE two-tier ticket was cancelled");
            }
            loaded.copy_quantized(index, row.data(), row.size());
            const size_t position = state_->miss_positions[index];
            result.rows_[position] = state_->store->rows.admit(
                state_->keys[position], row.data(), row.size());
        }
    }
    return result;
}

void two_tier_store::ticket::cancel() const {
    if (!state_) return;
    state_->cancelled.store(true, std::memory_order_release);
    state_->pages.cancel();
}

void two_tier_store::staged_batch::dequantize(float * output, size_t output_count) const {
    if (output == nullptr || rows_.size() > std::numeric_limits<size_t>::max() / ple_values_per_row ||
            output_count != rows_.size() * ple_values_per_row) {
        throw std::invalid_argument("qwen4exp PLE two-tier staged output geometry is invalid");
    }
    for (size_t index = 0; index < rows_.size(); ++index) {
        qwen4exp_ple_dequantize_iq4_nl(
            rows_[index].data(), rows_[index].size(),
            output + index * ple_values_per_row, ple_values_per_row);
    }
}

two_tier_store::two_tier_store(
        std::shared_ptr<const llama_expert_cache::source> source,
        qwen4exp_ple_extent_index extents,
        size_t worker_count,
        uint64_t page_cache_bytes,
        uint64_t row_cache_bytes,
        size_t max_source_batch_width)
    : impl_(std::make_shared<impl>(
        std::move(source), std::move(extents), worker_count, page_cache_bytes, row_cache_bytes,
        max_source_batch_width)) {}

two_tier_store::ticket two_tier_store::submit(uint64_t owner, const std::vector<uint64_t> & global_rows) {
    return submit(owner, global_rows.data(), global_rows.size());
}

two_tier_store::ticket two_tier_store::submit(uint64_t owner, const uint64_t * data, size_t count) {
    if (!data && count) throw std::invalid_argument("PLE row input is null");
    const input_view<uint64_t> global_rows{data, count};
    if (global_rows.empty()) throw std::invalid_argument("qwen4exp PLE two-tier batch cannot be empty");
    auto state = std::allocate_shared<ticket::state>(reader_allocator<ticket::state>(impl_->metadata), impl_->metadata);
    state->store = impl_;
    state->owner = owner;
    state->cached.resize(global_rows.size());
    state->keys.assign(global_rows.begin(), global_rows.end());

    reader_vector<uint64_t> page_rows{reader_allocator<uint64_t>(impl_->metadata)};
    page_rows.reserve(global_rows.size());
    state->miss_positions.reserve(global_rows.size());
    for (size_t index = 0; index < global_rows.size(); ++index) {
        state->cached[index] = impl_->rows.acquire(global_rows[index]);
        if (!state->cached[index]) {
            state->miss_positions.push_back(index);
            page_rows.push_back(global_rows[index]);
        }
    }
    publish_provider_ticket(impl_->owners_mutex, impl_->owners, owner, state, [&] {
        if (!page_rows.empty()) state->pages = impl_->pages.submit(owner, page_rows.data(), page_rows.size());
    });
    return ticket(std::move(state));
}

size_t two_tier_store::cancel_owner(uint64_t owner) {
    std::lock_guard<std::mutex> lock(impl_->owners_mutex);
    size_t cancelled = 0;
    auto found = impl_->owners.find(owner);
    if (found != impl_->owners.end()) {
        for (const auto & weak : found->second) {
            if (auto state = weak.lock()) {
                bool expected = false;
                if (state->cancelled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    ++cancelled;
                }
            }
        }
        impl_->owners.erase(found);
    }
    (void) impl_->pages.cancel_owner(owner);
    return cancelled;
}

bool two_tier_store::set_metadata_allocation_budget(uint64_t bytes) {
    return set_provider_metadata_budget(impl_->metadata, bytes);
}

two_tier_store_stats two_tier_store::stats() const {
    const page_store_stats page_stats = impl_->pages.stats();
    return {page_stats.reads, page_stats.pages, impl_->rows.stats(),
            provider_metadata_snapshot(impl_->owners_mutex, impl_->owners, impl_->metadata),
            page_stats.metadata};
}

} // namespace qwen4exp_ple_storage
