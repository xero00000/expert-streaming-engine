#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static std::vector<float> run(ggml_backend_t backend, int width, int batches, bool strided) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({1024*1024, nullptr, true}), ggml_free);
    const int pitch = width + (strided ? 5 : 0), ipitch = strided ? 7 : 4;
    auto storage = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_BF16, pitch, 9, batches);
    auto source = ggml_view_3d(ctx.get(), storage, width, 9, batches, storage->nb[1], storage->nb[2], 0);
    auto id_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, ipitch, batches);
    auto ids = ggml_view_2d(ctx.get(), id_storage, 4, batches, id_storage->nb[1], 0);
    auto output = ggml_get_rows(ctx.get(), source, ids);
    if (!ggml_backend_supports_op(backend, output)) throw std::runtime_error("BF16 GET_ROWS is unsupported");
    auto graph = ggml_new_graph(ctx.get()); ggml_build_forward_expand(graph, output);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend), ggml_backend_buffer_free);
    if (!buffer) throw std::runtime_error("allocation failed");
    std::vector<ggml_bf16_t> data(ggml_nelements(storage));
    for (size_t i = 0; i < data.size(); ++i) data[i] = ggml_fp32_to_bf16(std::sin(float(i)) * 100);
    std::vector<int32_t> indices(ipitch * batches, 0);
    for (int b = 0; b < batches; ++b) { indices[b*ipitch] = 8; indices[b*ipitch+1] = 0; indices[b*ipitch+2] = 3; indices[b*ipitch+3] = 8; }
    ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(ggml_bf16_t));
    ggml_backend_tensor_set(id_storage, indices.data(), 0, indices.size() * sizeof(int32_t));
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) throw std::runtime_error("GET_ROWS failed");
    std::vector<float> values(ggml_nelements(output)); ggml_backend_tensor_get(output, values.data(), 0, values.size() * sizeof(float));
    return values;
}
int main() {
    try {
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(), ggml_backend_free);
#ifdef GGML_USE_CUDA
        for (int dev = 0; dev < ggml_backend_cuda_get_device_count(); ++dev) {
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> gpu(ggml_backend_cuda_init(dev, nullptr, nullptr), ggml_backend_free);
            for (int width : {1, 257, 5120}) for (int batches : {1, 3}) for (bool strided : {false, true}) {
                auto reference = run(cpu.get(), width, batches, strided), actual = run(gpu.get(), width, batches, strided);
                if (reference.size() != actual.size() || std::memcmp(reference.data(), actual.data(), reference.size() * sizeof(float)))
                    throw std::runtime_error("BF16 gather differs from CPU");
            }
            std::printf("BF16 GET_ROWS CUDA%d: 12 contiguous/strided/repeated-index cases exact\n", dev);
        }
#else
        run(cpu.get(), 257, 3, true);
#endif
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
