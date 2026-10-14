#pragma once

#include "ggml-backend.h"
#include "ggml-memory-budget.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_cuda_pool_memory_stats {
    uint64_t physical_bytes;
    uint64_t active_bytes;
    uint64_t cached_bytes;
    uint64_t virtual_reserved_bytes;
    uint64_t peak_physical_bytes;
    uint64_t allocation_budget_bytes;
    uint64_t budget_rejections;
};

// Owner-thread snapshot after graph completion. Does not initialize a pool or
// synchronize a backend. device is a CUDA-visible physical device ID, including
// pools a split backend owns on devices other than its primary device.
// Invalid backend/device/output returns false without modifying the output.
GGML_API bool ggml_backend_cuda_get_pool_memory_stats(
        ggml_backend_t backend, int device, struct ggml_cuda_pool_memory_stats * stats);

// Owner-thread admission cap on physical backing, active plus cached. Creates
// the pool object without allocating GPU backing. Cannot lower below committed
// physical bytes. UINT64_MAX retains unlimited admission.
GGML_API bool ggml_backend_cuda_set_pool_memory_budget(
        ggml_backend_t backend, int device, uint64_t bytes);
GGML_API bool ggml_backend_cuda_set_pool_shared_memory_budget(
        ggml_backend_t backend, int device, ggml_memory_budget_t budget);

// Known single-device explicit-F32 dense projection workspace. Uses the same
// direct-kernel shape predicates as dispatch; false leaves bytes unchanged for
// unplanned operations/layouts. This is not an all-operator memory estimate.
GGML_API bool ggml_backend_cuda_explicit_matmul_workspace(
        ggml_backend_t backend, const struct ggml_tensor * tensor, uint64_t * bytes);
// Initialize the cuBLAS handle and retain reusable physical pool backing before
// graph reservation. Does not execute operators or read/write model tensors.
GGML_API bool ggml_backend_cuda_prepare_explicit_workspace(ggml_backend_t backend, uint64_t bytes);

#ifdef __cplusplus
}
#endif
