#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

int main() {
    try {
#ifdef GGML_USE_CUDA
#ifdef _WIN32
        _putenv_s("ESE_V41_DIRECT_BF16_HEAD", "1"); _putenv_s("ESE_CUDA_DENSE_BATCH_VECTOR", "1");
#else
        setenv("ESE_V41_DIRECT_BF16_HEAD", "1", 1); setenv("ESE_CUDA_DENSE_BATCH_VECTOR", "1", 1);
#endif
        for (int dev = 0; dev < ggml_backend_cuda_get_device_count(); ++dev) {
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu(ggml_backend_cuda_init(dev, nullptr, nullptr), ggml_backend_free);
            float maximum = 0;
            for (int width : {256, 5120}) for (int n : {1, 5, 8}) {
                const int rows = width == 256 ? 129280 : 1024;
                std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024*1024, nullptr, true}), ggml_free);
                auto weights = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, width, rows); ggml_set_name(weights, "output");
                auto input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, n);
                auto direct = ggml_mul_mat(ctx.get(), weights, input); ggml_mul_mat_set_prec(direct, GGML_PREC_F32);
                auto reference = ggml_mul_mat(ctx.get(), ggml_cast(ctx.get(), weights, GGML_TYPE_F32), input);
                ggml_mul_mat_set_prec(reference, GGML_PREC_F32);
                auto graph = ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph, reference); ggml_build_forward_expand(graph, direct);
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                    ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get()), ggml_backend_buffer_free);
                if (!buffer) throw std::runtime_error("head fixture allocation failed");
                std::vector<ggml_bf16_t> w(size_t(width) * rows);
                uint32_t seed = 41;
                auto random = [&]() { seed = seed * 1664525u + 1013904223u; return (float(seed >> 8) / 16777216.0f - 0.5f); };
                for (auto & x : w) x = ggml_fp32_to_bf16(0.2f * random());
                std::vector<float> x(width * n); for (auto & v : x) v = 4 * random();
                ggml_backend_tensor_set(weights, w.data(), 0, w.size() * sizeof(ggml_bf16_t));
                ggml_backend_tensor_set(input, x.data(), 0, x.size() * sizeof(float));
                if (ggml_backend_graph_compute(gpu.get(), graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("head fixture compute failed");
                std::vector<float> a(rows*n), b(rows*n);
                ggml_backend_tensor_get(direct, a.data(), 0, a.size()*sizeof(float));
                ggml_backend_tensor_get(reference, b.data(), 0, b.size()*sizeof(float));
                for (size_t i = 0; i < a.size(); ++i) {
                    const float error = std::abs(a[i]-b[i]); maximum = std::max(maximum, error);
                    if (!std::isfinite(a[i]) || error > 1e-4f) throw std::runtime_error("direct BF16 head exceeds F32 error tolerance");
                }
                for (int token = 0; token < n; ++token) for (int row : {0, rows/2, rows-1}) {
                    double expected = 0;
                    for (int col = 0; col < width; ++col) expected += double(ggml_bf16_to_fp32(w[size_t(row)*width+col])) * x[token*width+col];
                    if (std::abs(double(a[token*rows+row])-expected) > 1e-4) throw std::runtime_error("direct BF16 head differs from FP64 dot product");
                }
            }
            std::printf("BF16 head CUDA%d: B1/B5/B8, hidden/Markov widths, maximum F32-reference error %.8g\n", dev, maximum);
        }
#endif
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
