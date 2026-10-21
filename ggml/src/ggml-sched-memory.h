#pragma once

#include "ggml-backend.h"
#include "ggml-memory-budget.h"

#ifdef __cplusplus
extern "C" {
#endif

// Internal context-planning hooks. IDs follow scheduler backend order. Shared
// buffer types report one canonical arena and enforce the strictest alias quota.
GGML_API bool ggml_backend_sched_set_graph_buffer_budget(
        ggml_backend_sched_t sched, int backend_id, uint64_t bytes);
GGML_API bool ggml_backend_sched_get_graph_buffer_stats(
        ggml_backend_sched_t sched, int backend_id, struct ggml_gallocr_buffer_stats * stats);
GGML_API bool ggml_backend_sched_set_graph_memory_budget(
        ggml_backend_sched_t sched, int backend_id, ggml_memory_budget_t budget);
// Split and assign backends without allocating graph arenas, then commit that
// prepared reservation. Used to admit backend-private workspace before arenas.
GGML_API void ggml_backend_sched_prepare_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * graph);
GGML_API bool ggml_backend_sched_measure_prepared(ggml_backend_sched_t sched, uint64_t * bytes, size_t count);
GGML_API bool ggml_backend_sched_reserve_prepared(ggml_backend_sched_t sched);

#ifdef __cplusplus
}
#endif
