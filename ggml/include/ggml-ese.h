#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Shared production submission primitive for adaptive expert-cache fills and
// ESE hardware calibration. Completion is ordered by the caller with the cache
// transfer event or backend synchronization, respectively.
GGML_API void ggml_backend_expert_cache_upload_async(
        ggml_backend_t backend, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size);

// Build a reduction whose result storage is owned by one explicit branch.
// The scheduler copies heterogeneous sources to that backend after their
// event-fenced compute completes, then performs the local reduction there.
GGML_API struct ggml_tensor * ggml_ese_reduce_to(
        struct ggml_context * ctx,
        struct ggml_tensor ** tensors,
        int count,
        enum ggml_op op,
        int destination);

// Copy routed I32 expert IDs while retaining only the half-open route-position
// range [begin, begin + count). Every other position becomes the standard -1
// sentinel. Complementary partitions can run independently without evaluating
// an expert twice.
GGML_API struct ggml_tensor * ggml_ese_route_partition(
        struct ggml_context * ctx,
        struct ggml_tensor * ids,
        int begin,
        int count);

enum ggml_ese_route_role {
    GGML_ESE_ROUTE_NONE = 0,
    GGML_ESE_ROUTE_GPU  = 1,
    GGML_ESE_ROUTE_CPU  = 2,
};

// Identifies which heterogeneous branch consumes a partition tensor.
GGML_API enum ggml_ese_route_role ggml_ese_route_get_role(
        const struct ggml_tensor * ids);

// Pin the final tensor of a heterogeneous branch to the branch's backend.
GGML_API void ggml_ese_tensor_set_role(
        struct ggml_tensor * tensor,
        enum ggml_ese_route_role role);
GGML_API enum ggml_ese_route_role ggml_ese_tensor_get_role(
        const struct ggml_tensor * tensor);

#ifdef __cplusplus
}
#endif
