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

#ifdef __cplusplus
}
#endif
