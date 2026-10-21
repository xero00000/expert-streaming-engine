#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// IQ4_XS's unsigned lookup values reach 241. Adjacent large Q8 activations
// must not saturate a signed 16-bit intermediate in the AVX2 matrix kernel.
static void run(int width, int tokens, int pattern) {
    constexpr int rows = 16, experts = 2, used = 2;
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "CPU backend");
    auto * context = ggml_init({1024*1024, nullptr, true});
    require(context != nullptr, "context");
    auto * weights = ggml_new_tensor_3d(context, GGML_TYPE_IQ4_XS, width, rows, experts);
    auto * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, width, used, tokens);
    auto * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, used, tokens);
    auto * output = ggml_mul_mat_id(context, weights, input, ids);
    auto * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    require(buffer != nullptr, "tensor allocation");

    const size_t blocks_per_row = size_t(width)/QK_K;
    std::vector<block_iq4_xs> packed(blocks_per_row*rows*experts);
    const int varied_scales[8] = {-32, -17, -1, 1, 7, 15, 23, 31};
    for (size_t i = 0; i < packed.size(); ++i) {
        auto & block = packed[i];
        std::memset(&block, 0, sizeof(block));
        block.d = ggml_fp32_to_fp16(float(1+i%3)/32);
        for (int sub = 0; sub < 8; ++sub) {
            const unsigned scale = unsigned((pattern == 2 ? varied_scales[sub] : 1)+32);
            block.scales_l[sub/2] |= uint8_t((scale & 15) << (4*(sub%2)));
            block.scales_h |= uint16_t((scale >> 4) << (2*sub));
        }
        for (size_t j = 0; j < sizeof(block.qs); ++j) {
            block.qs[j] = pattern == 0 ? 0xff : pattern == 1 ? 0x00 : uint8_t((i+17*j)%256);
        }
    }
    std::vector<float> activations(size_t(width)*used*tokens);
    std::vector<int32_t> routes(used*tokens);
    for (int token = 0; token < tokens; ++token) for (int route = 0; route < used; ++route) {
        routes[token*used+route] = (token+route)%experts;
        for (int col = 0; col < width; ++col) {
            activations[(token*used+route)*width+col] =
                pattern < 2 ? (route == 0 ? 1.f : -1.f) : ((col/8+route+token)%2 ? -1.f : 1.f);
        }
    }
    ggml_backend_tensor_set(weights, packed.data(), 0, packed.size()*sizeof(packed[0]));
    ggml_backend_tensor_set(input, activations.data(), 0, activations.size()*sizeof(float));
    ggml_backend_tensor_set(ids, routes.data(), 0, routes.size()*sizeof(int32_t));
    std::vector<float> decoded(width), actual(size_t(rows)*used*tokens), previous;
    for (int threads : {1, 4}) {
        ggml_backend_cpu_set_n_threads(backend, threads);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
        ggml_backend_tensor_get(output, actual.data(), 0, actual.size()*sizeof(float));
        for (int token = 0; token < tokens; ++token) for (int route = 0; route < used; ++route) {
            const int expert = routes[token*used+route];
            for (int row = 0; row < rows; ++row) {
                dequantize_row_iq4_xs(packed.data()+(expert*rows+row)*blocks_per_row, decoded.data(), width);
                double expected = 0;
                for (int col = 0; col < width; ++col)
                    expected += double(decoded[col])*activations[(token*used+route)*width+col];
                const float got = actual[(token*used+route)*rows+row];
                if (!std::isfinite(got) || std::abs(got-expected) > 0.002+1.e-5*std::abs(expected)) {
                    std::fprintf(stderr, "IQ4_XS width=%d tokens=%d pattern=%d threads=%d row=%d: got %.9g expected %.9g\n",
                                 width, tokens, pattern, threads, row, got, expected);
                    throw std::runtime_error("IQ4_XS CPU differs from independent scalar dequantization");
                }
            }
        }
        if (!previous.empty()) require(previous == actual, "thread-count output mismatch");
        previous = actual;
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(context);
    ggml_backend_free(backend);
}

int main() {
    try {
        for (int width : {256, 512, 2048}) for (int tokens : {1, 2, 7})
            for (int pattern = 0; pattern < 3; ++pattern) run(width, tokens, pattern);
        std::puts("IQ4_XS CPU scalar and threading regressions passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
