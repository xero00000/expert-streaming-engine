#pragma once

#include "ggml-backend.h"

// Internal llama/ggml integration. begin may decline by returning nullptr.
// next returns the submitted expert's index and transfers one lease handle;
// end drains outstanding work and releases results not transferred to ggml.
// The existing lease-release callback owns returned handles until H2D ends.
struct ggml_expert_load_callbacks {
    void * (*begin)(void * user, int layer, const int * experts, size_t count, int component);
    bool (*next)(void * batch, size_t * index, ggml_backend_expert_lease * lease);
    void (*end)(void * batch);
};

GGML_API void ggml_backend_sched_set_expert_load_callbacks(
        ggml_backend_sched_t sched, ggml_expert_load_callbacks callbacks);

// Single-token, bounded hybrid choice. Prefer complete, layout-compatible
// experts on the GPU branch's actual device, then fill the calibrated budget
// in router order. Compute once before either branch changes cache residency.
// All tensors and the scheduler must outlive the graph. Bind the final GPU
// output after constructing the branches, before graph allocation/evaluation.
GGML_API ggml_tensor * ggml_backend_sched_expert_route_choice(
        ggml_context * ctx, ggml_backend_sched_t sched, ggml_tensor * ids,
        int layer, int gpu_count, ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down);
GGML_API void ggml_backend_sched_expert_route_bind_gpu_output(
        ggml_tensor * choice, ggml_tensor * gpu_output);

// Read-only forecast support. Residency queries do not touch cache policy.
GGML_API uint32_t ggml_backend_sched_expert_ready_mask(ggml_backend_sched_t sched, int layer, int expert);
GGML_API uint64_t ggml_backend_sched_expert_pass(ggml_backend_sched_t sched);
// Smoothed precision of previous predictions that missed VRAM, or zero until
// at least sixteen such predictions have been checked against real routing.
GGML_API float ggml_backend_sched_expert_prediction_precision(ggml_backend_sched_t sched, int layer);
GGML_API void ggml_backend_sched_trace_expert_prediction(ggml_backend_sched_t sched,
        int layer, const int32_t * experts, const float * scores, size_t count);
