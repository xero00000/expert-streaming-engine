#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

#ifdef GGML_USE_CUDA
static void pointer_updates(ggml_backend_t backend) {
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx, "pointer-update context");
    auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 64, 32);
    auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 128);
    auto * input = ggml_view_1d(ctx, storage, 64, 0);
    auto * output = ggml_mul_mat(ctx, weights, input);
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "pointer-update buffer");
    std::vector<float> w(64 * 32, 1), values(128, .125f), actual(32);
    std::fill(values.begin() + 64, values.end(), .25f);
    ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(float));
    ggml_backend_tensor_set(storage, values.data(), 0, values.size() * sizeof(float));
    for (size_t offset : {size_t(0), size_t(64 * sizeof(float)), size_t(0)}) {
        input->view_offs = offset;
        input->data = static_cast<unsigned char *>(storage->data) + offset;
        std::memcpy(input->op_params, &offset, sizeof(offset));
        const float expected = offset ? 16 : 8;
        for (int replay = 0; replay < 3; ++replay) {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "kernel-only update/replay");
            ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
            for (const float value : actual) require(value == expected, "kernel-only pointer update changed arithmetic");
        }
    }
    std::puts("CUDA graph pointer updates: 9 analytical update/replay cases pass");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}
#endif

int main(int argc, char ** argv) {
    try {
#ifdef GGML_USE_CUDA
        const int device = argc > 1 ? std::atoi(argv[1]) : 0;
        require(cuInit(0) == CUDA_SUCCESS, "driver init");
        CUdevice dev; CUcontext primary;
        require(cuDeviceGet(&dev, device) == CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&primary, dev) == CUDA_SUCCESS &&
                cuCtxSetCurrent(primary) == CUDA_SUCCESS, "primary context");
        auto * backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        require(backend, "backend");
        auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
        require(ctx, "context");
        constexpr int maximum_width = 64, rows = 32;
        auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, maximum_width, rows);
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, maximum_width, 1);
        auto * output = ggml_mul_mat(ctx, weights, input);
        ggml_mul_mat_set_prec(output, GGML_PREC_F32);
        auto * graph = ggml_new_graph_custom(ctx, 64, false);
        ggml_build_forward_expand(graph, output);
        auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer, "allocation");
        std::vector<float> values(maximum_width, .125f), actual(rows);
        ggml_backend_tensor_set(input, values.data(), 0, values.size() * sizeof(float));
        // Reuse the exact graph, output and input addresses. Only leaf layout
        // changes, as happens when a metadata arena is reused for another graph.
        // Interleaved replays reset the update counter; disabling graph capture
        // after repeated updates must not conceal an invalid cache match.
        int cases = 0;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_F32}) {
            weights->type = type;
            weights->nb[0] = ggml_type_size(type);
            for (int width : {64, 32, 64}) {
                weights->ne[0] = maximum_width;
                input->ne[0] = width;
                std::vector<unsigned char> storage(maximum_width * rows * sizeof(float), 0);
                for (int r = 0; r < rows; ++r) for (int c = 0; c < maximum_width; ++c) {
                    const float value = 1.f + r / 32.f;
                    auto * dest = storage.data() + r * weights->nb[1] + c * weights->nb[0];
                    if (type == GGML_TYPE_F32) std::memcpy(dest, &value, sizeof(value));
                    else { const auto half = ggml_fp32_to_fp16(value); std::memcpy(dest, &half, sizeof(half)); }
                }
                // Upload all logical columns before shrinking the leaf view.
                // Excluded columns remain nonzero: a stale 64-column capture
                // must disagree with the 32-column analytical result.
                ggml_backend_tensor_set(weights, storage.data(), 0, ggml_nbytes(weights));
                weights->ne[0] = width;
                require(ggml_backend_supports_op(backend, output), "native matrix support");
                for (int replay = 0; replay < 3; ++replay) {
                    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute");
                    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                    for (int r = 0; r < rows; ++r) {
                        const float expected = width * (1.f + r / 32.f) * .125f;
                        if (!std::isfinite(actual[r]) || std::fabs(actual[r] - expected) > 1.e-5f) {
                            std::fprintf(stderr, "layout mismatch type=%s width=%d replay=%d row=%d got=%g expected=%g\n",
                                    ggml_type_name(type), width, replay, r, actual[r], expected);
                            throw std::runtime_error("captured graph used stale leaf layout");
                        }
                    }
                    ++cases;
                }
            }
        }
        std::printf("CUDA graph leaf layout: %d shape/type/strided-input/replay cases pass\n", cases);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
        pointer_updates(backend);
        ggml_backend_free(backend);
        require(cuDevicePrimaryCtxRelease(dev) == CUDA_SUCCESS, "primary release");
#else
        (void) argc; (void) argv; std::puts("SKIP: CUDA graph layout test");
#endif
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
