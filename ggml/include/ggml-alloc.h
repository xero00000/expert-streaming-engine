#pragma once

#include "ggml.h"

#ifdef  __cplusplus
extern "C" {
#endif

typedef struct ggml_backend_buffer_type * ggml_backend_buffer_type_t;
typedef struct ggml_backend_buffer * ggml_backend_buffer_t;
typedef struct ggml_backend * ggml_backend_t;

// Tensor allocator
struct ggml_tallocr {
    ggml_backend_buffer_t buffer;
    void * base;
    size_t alignment;
    size_t offset;
};

GGML_API struct ggml_tallocr ggml_tallocr_new(ggml_backend_buffer_t buffer);
GGML_API void                ggml_tallocr_alloc(struct ggml_tallocr * talloc, struct ggml_tensor * tensor);

// Graph allocator
/*
  Example usage:
    ggml_gallocr_t galloc = ggml_gallocr_new(ggml_bacckend_cpu_buffer_type());

    // optional: create a worst-case graph and reserve the buffers to avoid reallocations
    ggml_gallocr_reserve(galloc, build_graph(max_batch));

    // allocate the graph
    struct ggml_cgraph * graph = build_graph(batch);
    ggml_gallocr_alloc_graph(galloc, graph);

    printf("compute buffer size: %zu bytes\n", ggml_gallocr_get_buffer_size(galloc, 0));

    // evaluate the graph
    ggml_backend_graph_compute(backend, graph);
*/

// special tensor flags for use with the graph allocator:
//   ggml_set_input(): all input tensors are allocated at the beginning of the graph in non-overlapping addresses
//   ggml_set_output(): output tensors are never freed and never overwritten

typedef struct ggml_gallocr * ggml_gallocr_t;

GGML_API ggml_gallocr_t ggml_gallocr_new(ggml_backend_buffer_type_t buft);
GGML_API ggml_gallocr_t ggml_gallocr_new_n(ggml_backend_buffer_type_t * bufts, int n_bufs);
GGML_API void           ggml_gallocr_free(ggml_gallocr_t galloc);

// pre-allocate buffers from a measure graph - does not allocate or modify the graph
// call with a worst-case graph to avoid buffer reallocations
// not strictly required for single buffer usage: ggml_gallocr_alloc_graph will reallocate the buffers automatically if needed
// returns false if the buffer allocation failed
GGML_API bool ggml_gallocr_reserve(ggml_gallocr_t galloc, struct ggml_cgraph * graph);
GGML_API bool ggml_gallocr_reserve_n(
    ggml_gallocr_t galloc,
    struct ggml_cgraph * graph,
    const int * node_buffer_ids,
    const int * leaf_buffer_ids);

// automatic reallocation if the topology changes when using a single buffer
// returns false if using multiple buffers and a re-allocation is needed (call ggml_gallocr_reserve_n first to set the node buffers)
GGML_API bool ggml_gallocr_alloc_graph(ggml_gallocr_t galloc, struct ggml_cgraph * graph);

GGML_API size_t ggml_gallocr_get_buffer_size(ggml_gallocr_t galloc, int buffer_id);

// Measure the native arena layout without allocating buffers, charging quotas,
// or changing the committed layout/diagnostics. One byte count per buffer ID.
GGML_API bool ggml_gallocr_measure_n(ggml_gallocr_t galloc, struct ggml_cgraph * graph,
        const int * node_buffer_ids, const int * leaf_buffer_ids, uint64_t * bytes, size_t count);

struct ggml_gallocr_buffer_stats {
    uint64_t allocated_bytes;
    uint64_t peak_allocated_bytes;
    uint64_t allocation_budget_bytes;
    uint64_t required_bytes; // Last measured arena requirement, including rejection.
    uint64_t allocation_failures;
    uint64_t budget_rejections;
};
// Quota includes the live arena and its replacement during preparation.
// Aliased buffer types share one arena; the strictest alias quota applies.
GGML_API bool ggml_gallocr_set_buffer_budget(ggml_gallocr_t galloc, int buffer_id, uint64_t bytes);
GGML_API bool ggml_gallocr_get_buffer_stats(ggml_gallocr_t galloc, int buffer_id,
        struct ggml_gallocr_buffer_stats * stats);


// Utils
// Create a buffer and allocate all the tensors in a ggml_context
// Read-only sum of aligned backend allocation requests for unallocated owning
// tensors. No buffer allocation or tensor mutation; false leaves size untouched.
GGML_API bool ggml_backend_alloc_ctx_tensors_size(struct ggml_context * ctx,
        ggml_backend_buffer_type_t buft, size_t * size);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors_from_buft(struct ggml_context * ctx, ggml_backend_buffer_type_t buft);
GGML_API struct ggml_backend_buffer * ggml_backend_alloc_ctx_tensors(struct ggml_context * ctx, ggml_backend_t backend);

#ifdef  __cplusplus
}
#endif
