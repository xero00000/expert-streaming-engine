#pragma once
#include "ggml.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ggml_memory_budget * ggml_memory_budget_t;
struct ggml_gallocr;
struct ggml_memory_budget_stats {
    uint64_t reserved_bytes;
    uint64_t peak_reserved_bytes;
    uint64_t budget_bytes;
    uint64_t rejections;
};
// Internal shared physical admission ledger. Reservations include pending
// driver allocations. Each owner retains the ledger until releasing its charge.
GGML_API ggml_memory_budget_t ggml_memory_budget_create(uint64_t bytes);
GGML_API void ggml_memory_budget_retain(ggml_memory_budget_t budget);
GGML_API void ggml_memory_budget_free(ggml_memory_budget_t budget);
GGML_API bool ggml_memory_budget_reserve(ggml_memory_budget_t budget, uint64_t bytes);
GGML_API void ggml_memory_budget_release(ggml_memory_budget_t budget, uint64_t bytes);
GGML_API bool ggml_memory_budget_set(ggml_memory_budget_t budget, uint64_t bytes);
GGML_API bool ggml_memory_budget_get(ggml_memory_budget_t budget, struct ggml_memory_budget_stats * stats);

// Attach a shared ledger to a canonical graph arena (aliases share one charge).
GGML_API bool ggml_gallocr_set_memory_budget(struct ggml_gallocr * alloc, int buffer_id, ggml_memory_budget_t budget);
#ifdef __cplusplus
}
#endif
