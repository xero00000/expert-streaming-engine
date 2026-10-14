#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void run(int threads, int tokens, int channels, int planes, int batches,
        bool packed_rows, bool reshape) {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
        ggml_init({4 * 1024 * 1024, nullptr, true}), ggml_free);
    check(bool(ctx), "metadata allocation failed");
    auto * source = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, channels, tokens, planes, batches);
    auto * transposed = ggml_transpose(ctx.get(), source);
    const int width = reshape ? tokens * channels : tokens;
    const int rows = reshape ? 1 : channels;
    const int pitch = width + (packed_rows ? 0 : 13);
    const int plane_stride = pitch * rows + 17;
    const int batch_stride = plane_stride * planes + 19;
    const int offset = 11;
    const int storage_size = offset + batch_stride * batches + 23;
    auto * storage = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, storage_size);
    auto * view = ggml_view_4d(ctx.get(), storage, width, rows, planes, batches,
        pitch * sizeof(float), plane_stride * sizeof(float), batch_stride * sizeof(float),
        offset * sizeof(float));
    auto * copy = ggml_cpy(ctx.get(), transposed, view);
    auto * graph = ggml_new_graph_custom(ctx.get(), 1024, false);
    ggml_build_forward_expand(graph, copy);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    check(bool(backend), "CPU backend initialization failed");
    ggml_backend_cpu_set_n_threads(backend.get(), threads);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
        ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
    check(bool(buffer), "tensor allocation failed");

    std::vector<float> input(channels * tokens * planes * batches);
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(i + 1) * 0.25f;
    std::vector<float> expected(storage_size, -12345.5f);
    ggml_backend_tensor_set(source, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_tensor_set(storage, expected.data(), 0, expected.size() * sizeof(float));
    for (int b = 0; b < batches; ++b) for (int p = 0; p < planes; ++p)
        for (int c = 0; c < channels; ++c) for (int t = 0; t < tokens; ++t) {
            const int logical = c * tokens + t;
            const int destination = offset + b * batch_stride + p * plane_stride +
                (logical / width) * pitch + logical % width;
            expected[destination] = input[((b * planes + p) * tokens + t) * channels + c];
        }
    check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "copy execution failed");
    ggml_backend_synchronize(backend.get());
    std::vector<float> actual(storage_size);
    ggml_backend_tensor_get(storage, actual.data(), 0, actual.size() * sizeof(float));
    if (actual != expected) {
        size_t index = 0;
        while (index < actual.size() && actual[index] == expected[index]) ++index;
        std::fprintf(stderr, "copy mismatch: threads=%d tokens=%d channels=%d planes=%d batches=%d "
            "packed=%d reshape=%d index=%zu expected=%g actual=%g\n",
            threads, tokens, channels, planes, batches, packed_rows, reshape, index,
            expected[index], actual[index]);
        throw std::runtime_error("transpose copy corrupted destination values or padding");
    }
}

int main() {
    try {
        int cases = 0;
        // Strided destinations first give the old runtime a deterministic
        // negative control before testing reshaped destinations.
        for (bool reshape : {false, true}) for (bool packed : {false, true})
            for (int threads : {1, 2, 4, 8}) for (int tokens : {1, 4, 8, 12, 33})
                for (int channels : {8, 32}) for (int planes : {1, 4}) for (int batches : {1, 2}) {
                    run(threads, tokens, channels, planes, batches, packed, reshape);
                    ++cases;
                }
        std::printf("PASS: %d CPU transpose copies preserve values and all destination gaps\n", cases);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "COPY LAYOUT ERROR: %s\n", error.what());
        return 1;
    }
    return 0;
}
