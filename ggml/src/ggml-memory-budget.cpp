#include "ggml-memory-budget.h"
#include <atomic>
#include <mutex>
#include <new>

struct ggml_memory_budget {
    std::atomic<uint64_t> references{1};
    std::mutex mutex;
    ggml_memory_budget_stats stats{};
};
ggml_memory_budget_t ggml_memory_budget_create(uint64_t bytes) {
    auto * value=new(std::nothrow) ggml_memory_budget;
    if(value) value->stats.budget_bytes=bytes;
    return value;
}
void ggml_memory_budget_retain(ggml_memory_budget_t budget) {
    if(budget) ++budget->references;
}
void ggml_memory_budget_free(ggml_memory_budget_t budget) {
    if(budget && --budget->references==0) {
        GGML_ASSERT(budget->stats.reserved_bytes==0);
        delete budget;
    }
}
bool ggml_memory_budget_reserve(ggml_memory_budget_t budget,uint64_t bytes) {
    if(!budget) return true;
    std::lock_guard<std::mutex> lock(budget->mutex);
    auto & s=budget->stats;
    if(s.reserved_bytes>s.budget_bytes || bytes>s.budget_bytes-s.reserved_bytes) {
        ++s.rejections;return false;
    }
    s.reserved_bytes+=bytes;
    if(s.reserved_bytes>s.peak_reserved_bytes) s.peak_reserved_bytes=s.reserved_bytes;
    return true;
}
void ggml_memory_budget_release(ggml_memory_budget_t budget,uint64_t bytes) {
    if(!budget) return;
    std::lock_guard<std::mutex> lock(budget->mutex);
    GGML_ASSERT(bytes<=budget->stats.reserved_bytes);
    budget->stats.reserved_bytes-=bytes;
}
bool ggml_memory_budget_set(ggml_memory_budget_t budget,uint64_t bytes) {
    if(!budget) return false;
    std::lock_guard<std::mutex> lock(budget->mutex);
    if(bytes<budget->stats.reserved_bytes) return false;
    budget->stats.budget_bytes=bytes;return true;
}
bool ggml_memory_budget_get(ggml_memory_budget_t budget,ggml_memory_budget_stats * stats) {
    if(!budget || !stats) return false;
    std::lock_guard<std::mutex> lock(budget->mutex);
    *stats=budget->stats;return true;
}
