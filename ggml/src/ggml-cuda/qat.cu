#include "qat.cuh"
#include "../ggml-qat.h"
#include "../ggml-qat-impl.h"

// One warp owns each scaling group. The 16-element format leaves the upper
// lanes empty but participates in the same reductions on every lane.
static __global__ void qat_f32(const float * input, float * output, int64_t groups, int mode) {
    const int64_t group = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (group >= groups) return;
    const int lane = threadIdx.x % 32;
    const int width = mode == 2 ? 16 : 32;
    const float value = lane < width ? input[group * width + lane] : 0.0f;
    float maximum = fabsf(value);
    const unsigned invalid = __ballot_sync(0xffffffff, !isfinite(value));
    for (int offset = 16; offset; offset /= 2) {
        maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, offset));
    }
    const float scale = ggml_qat_group_scale(maximum, mode);
    if (lane < width) output[group * width + lane] = invalid ? NAN : ggml_qat_value(value, scale, mode);
}

void ggml_cuda_op_qat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_qat_mode(dst);
    GGML_ASSERT(mode >= 0 && dst->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(dst->src[0]));
    const int64_t groups = ggml_nelements(dst) / (mode == 2 ? 16 : 32);
    qat_f32<<<(groups + 7) / 8, 256, 0, ctx.stream()>>>((const float *) dst->src[0]->data, (float *) dst->data, groups, mode);
}
