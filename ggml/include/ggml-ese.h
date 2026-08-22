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

#ifdef __cplusplus
}
#endif
