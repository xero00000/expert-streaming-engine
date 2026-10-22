#include "expert-f32.cuh"
#include "ggml-expert-f32.h"

#include <climits>

struct expert_f32_indexed_args {
    const char * up, * gate, * input, * ids;
    char * output;
    int columns, rows, experts, input_slots;
    size_t up_row, up_plane, gate_row, gate_plane;
    size_t input_slot, input_token, id_slot, id_token, output_slot, output_token;
};

template <ggml_type type, bool fused>
static __global__ void expert_f32_indexed(expert_f32_indexed_args args) {
    const int64_t row = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (row >= args.rows) return;
    const int slot = blockIdx.y, token = blockIdx.z;
    const int expert = *(const int32_t *)(args.ids + slot * args.id_slot + token * args.id_token);
    float * output = (float *)(args.output + slot * args.output_slot + token * args.output_token) + row;
    if (expert < 0 || expert >= args.experts) {
        if (lane == 0) *output = 0.0f;
        return;
    }
    const float * input = (const float *)(args.input +
            (slot % args.input_slots) * args.input_slot + token * args.input_token);
    const char * up = args.up + expert * args.up_plane + row * args.up_row;
    const char * gate = fused ? args.gate + expert * args.gate_plane + row * args.gate_row : nullptr;
    if constexpr (!fused) {
        const float result = expert_f32_checked_project<type>(up, input, args.columns, lane);
        if (lane == 0) *output = result;
        return;
    }
    double partial_up = 0, partial_gate = 0;
    for (int column = lane; column < args.columns; column += 32) {
        const double x = expert_f32_to_double(input[column]);
        partial_up += expert_f32_to_double(expert_f32_weight<type>(up, column)) * x;
        if constexpr (fused) partial_gate += expert_f32_to_double(expert_f32_weight<type>(gate, column)) * x;
    }
    const double sum_up = expert_f32_reduce(partial_up);
    double sum_gate = 0;
    if constexpr (fused) sum_gate = expert_f32_reduce(partial_gate);
    if (lane == 0) {
        const float u = expert_f64_to_float(sum_up);
        if constexpr (fused) {
            const double g = expert_f32_to_double(expert_f64_to_float(sum_gate));
            *output = expert_f64_to_float((g / (1.0 + exp(-g))) * expert_f32_to_double(u));
        } else {
            *output = u;
        }
    }
}

bool ggml_cuda_expert_f32_supported_layout(const ggml_tensor * op) {
    if (!ggml_expert_f32_supported(op)) return false;
    const bool fused = op->op == GGML_OP_MOE_FUSED_UP_GATE;
    const auto * w = op->src[0];
    const auto * x = op->src[fused ? 2 : 1];
    const auto * ids = op->src[fused ? 3 : 2];
    if (w->ne[0] > INT_MAX - 32 || op->ne[0] > INT_MAX - 4 || w->ne[2] > INT_MAX ||
        x->ne[1] > INT_MAX || ids->ne[0] > 65535 || ids->ne[1] > 65535) return false;
    if (x->view_offs % 4 || x->nb[1] % 4 || x->nb[2] % 4 ||
        ids->view_offs % 4 || ids->nb[0] % 4 || ids->nb[1] % 4 ||
        op->view_offs % 4 || op->nb[1] % 4 || op->nb[2] % 4) return false;
    return true;
}

template <bool fused>
static void launch(ggml_type type, dim3 grid, expert_f32_indexed_args args, cudaStream_t stream) {
    switch (type) {
#define CASE(T) case T: expert_f32_indexed<T, fused><<<grid, 128, 0, stream>>>(args); break;
        CASE(GGML_TYPE_IQ2_S)
        CASE(GGML_TYPE_IQ2_XXS)
        CASE(GGML_TYPE_IQ1_M)
        CASE(GGML_TYPE_IQ3_XXS)
        CASE(GGML_TYPE_IQ3_S)
        CASE(GGML_TYPE_IQ4_NL)
        CASE(GGML_TYPE_IQ4_XS)
        CASE(GGML_TYPE_Q2_0)
#undef CASE
        default: GGML_ABORT("unsupported explicit expert weight type");
    }
}

void ggml_cuda_expert_f32(ggml_tensor * op, cudaStream_t stream) {
    GGML_ASSERT(ggml_cuda_expert_f32_supported_layout(op));
    const bool fused = op->op == GGML_OP_MOE_FUSED_UP_GATE;
    const auto * w = op->src[0];
    const auto * gate = fused ? op->src[1] : nullptr;
    const auto * x = op->src[fused ? 2 : 1];
    const auto * ids = op->src[fused ? 3 : 2];
    expert_f32_indexed_args args = {};
    args.up = (const char *) w->data;
    args.gate = gate ? (const char *) gate->data : args.up;
    if (fused && !gate) args.up += op->ne[0] * w->nb[1];
    args.input = (const char *) x->data;
    args.ids = (const char *) ids->data;
    args.output = (char *) op->data;
    args.columns = int(w->ne[0]); args.rows = int(op->ne[0]);
    args.experts = int(w->ne[2]); args.input_slots = int(x->ne[1]);
    args.up_row = w->nb[1]; args.up_plane = w->nb[2];
    args.gate_row = gate ? gate->nb[1] : w->nb[1];
    args.gate_plane = gate ? gate->nb[2] : w->nb[2];
    args.input_slot = x->nb[1]; args.input_token = x->nb[2];
    args.id_slot = ids->nb[0]; args.id_token = ids->nb[1];
    args.output_slot = op->nb[1]; args.output_token = op->nb[2];
    const dim3 grid((args.rows + 3) / 4, unsigned(ids->ne[0]), unsigned(ids->ne[1]));
    if (fused) launch<true>(w->type, grid, args, stream);
    else launch<false>(w->type, grid, args, stream);
    CUDA_CHECK(cudaGetLastError());
}
