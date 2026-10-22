#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace llama_memory {
struct allocation_state;
struct allocation_stats {
    uint64_t allocated_bytes = 0;
    uint64_t peak_allocated_bytes = 0;
    uint64_t allocation_budget_bytes = UINT64_MAX;
    uint64_t allocation_failures = 0;
    uint64_t allocation_requests = 0;
};
std::shared_ptr<allocation_state> create_state();
allocation_stats snapshot(const std::shared_ptr<allocation_state> & state);
bool set_budget(const std::shared_ptr<allocation_state> & state, uint64_t bytes);

// Allocation callbacks keep the standard allocator's alignment behavior while
// the opaque ledger checks the budget and counts successful requests.
void * allocate_metadata(const std::shared_ptr<allocation_state> & state,
        size_t count, size_t element_size, void * (*allocate)(size_t));
void deallocate_metadata(const std::shared_ptr<allocation_state> & state,
        void * value, size_t count, size_t element_size, void (*deallocate)(void *, size_t)) noexcept;

template<class T> struct metadata_allocator {
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    std::shared_ptr<allocation_state> state;
    metadata_allocator() = default;
    explicit metadata_allocator(std::shared_ptr<allocation_state> value) noexcept : state(std::move(value)) {}
    template<class U> metadata_allocator(const metadata_allocator<U> & other) noexcept : state(other.state) {}
    T * allocate(size_t count) {
        if (!state) return std::allocator<T>{}.allocate(count);
        return static_cast<T *>(allocate_metadata(state, count, sizeof(T), [](size_t n) -> void * {
            return std::allocator<T>{}.allocate(n);
        }));
    }
    void deallocate(T * value, size_t count) noexcept {
        if (!state) { std::allocator<T>{}.deallocate(value, count); return; }
        deallocate_metadata(state, value, count, sizeof(T), [](void * data, size_t n) {
            std::allocator<T>{}.deallocate(static_cast<T *>(data), n);
        });
    }
    template<class U> bool operator==(const metadata_allocator<U> & other) const noexcept { return state == other.state; }
    template<class U> bool operator!=(const metadata_allocator<U> & other) const noexcept { return !(*this == other); }
};

template<class T> using tracked_vector = std::vector<T, metadata_allocator<T>>;
} // namespace llama_memory
