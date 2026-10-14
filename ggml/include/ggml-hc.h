#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Recompute post/comb in FP64 from [S*(S+2), T] projected values and
// retain that precision through the residual sum. Output remains F32.
GGML_API struct ggml_tensor * ggml_hc_post_precise(
        struct ggml_context * ctx,
        struct ggml_tensor * x,
        struct ggml_tensor * res,
        struct ggml_tensor * projected,
        struct ggml_tensor * scale,
        struct ggml_tensor * bias,
        int n_iters,
        float eps);

#ifdef __cplusplus
}
#endif
