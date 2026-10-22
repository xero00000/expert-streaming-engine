#include "fattn-vec-f32.cuh"
#include "fattn-vec-f32-interface.cuh"

void ggml_cuda_flash_attn_ext_vec_f32_512(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_cuda_flash_attn_ext_vec_f32_case<512, 512, GGML_TYPE_F16, GGML_TYPE_F16>(ctx, dst);
}
