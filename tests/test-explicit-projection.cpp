#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void check_cpu_packed_projection() {
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
        ggml_backend_cpu_init(), ggml_backend_free);
    if (!backend) throw std::runtime_error("CPU backend initialization failed");
    int cases = 0;
    for (auto type : {GGML_TYPE_IQ3_S, GGML_TYPE_Q2_0})
    for (int threads : {1, 4})
    for (int width : {256, 768})
    for (int rows : {7, 257})
    for (int cols : {1, 3})
    for (bool padded : {false, true}) {
        ggml_backend_cpu_set_n_threads(backend.get(), threads);
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
            ggml_init({1 << 20, nullptr, true}), ggml_free);
        auto parent = ggml_new_tensor_2d(ctx.get(), type, width + (padded ? 256 : 0), rows);
        auto w = ggml_view_2d(ctx.get(), parent, width, rows, parent->nb[1], 0);
        const int stride = width + (padded ? 17 : 0);
        ggml_tensor * storage[2], * x[2];
        for (int ptr = 0; ptr < 2; ++ptr) {
            storage[ptr] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, stride, cols);
            x[ptr] = ggml_view_2d(ctx.get(), storage[ptr], width, cols, storage[ptr]->nb[1], 0);
        }
        auto y = ggml_mul_mat(ctx.get(), w, x[0]);
        if (padded) {
            y->nb[1] = (rows + 13)*sizeof(float);
            y->nb[2] = y->nb[1]*cols;
            y->nb[3] = y->nb[2];
        }
        auto graph = ggml_new_graph(ctx.get());
        ggml_build_forward_expand(graph, y);
        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
        if (!buffer) throw std::runtime_error("packed projection allocation failed");
        std::vector<uint8_t> packed(ggml_nbytes(parent), 0);
        std::vector<float> decoded(size_t(rows)*width);
        // Independently construct stored codes and their known values. No production
        // quantizer/dequantizer supplies the oracle. IQ3_S grid indices 0/1 are
        // [1,1,1,1]/[3,1,1,1]; Q2_0 symbols are {-1,0,1,2}.
        const size_t block_bytes = type == GGML_TYPE_IQ3_S ? 110 : 18;
        const int block_width = type == GGML_TYPE_IQ3_S ? 256 : 64;
        for (int row = 0; row < rows; ++row) for (int block = 0; block < width/block_width; ++block) {
            auto bytes = packed.data() + row*parent->nb[1] + block*block_bytes;
            const ggml_fp16_t scale = ggml_fp32_to_fp16(0.125f);
            std::memcpy(bytes, &scale, sizeof(scale));
            for (int k = 0; k < block_width; ++k) {
                float value;
                if (type == GGML_TYPE_IQ3_S) {
                    const int grid = (k/4 + row + block) & 1;
                    const int sign = (k + row + block) % 3 == 0;
                    const int multiplier = (k/32 + row + block) & 15;
                    bytes[2 + k/4] = uint8_t(grid);
                    if (sign) bytes[74 + k/8] |= uint8_t(1 << (k%8));
                    bytes[106 + k/64] |= uint8_t(multiplier << (4*((k/32)%2)));
                    value = 0.125f*(1 + 2*multiplier)*(grid && k%4 == 0 ? 3 : 1)*(sign ? -1 : 1);
                } else {
                    const int code = (k + row + block) & 3;
                    bytes[2 + k/4] |= uint8_t(code << (2*(k%4)));
                    value = 0.125f*(code - 1);
                }
                decoded[size_t(row)*width + block*block_width + k] = value;
            }
        }
        ggml_backend_tensor_set(parent, packed.data(), 0, packed.size());
        std::vector<float> inputs[2];
        for (int ptr = 0; ptr < 2; ++ptr) {
            inputs[ptr].resize(size_t(stride)*cols, -12345.f);
            for (int col = 0; col < cols; ++col) for (int k = 0; k < width; ++k)
                inputs[ptr][size_t(col)*stride+k] = float(std::sin(k*.017 + col + ptr)*.3 + std::cos(k*.031)*.07);
            ggml_backend_tensor_set(storage[ptr], inputs[ptr].data(), 0, inputs[ptr].size()*sizeof(float));
        }
        std::vector<float> sentinel(ggml_nbytes(y)/sizeof(float), -6789.f);
        ggml_backend_tensor_set(y, sentinel.data(), 0, sentinel.size()*sizeof(float));
        std::vector<float> defaults[2];
        for (int mode : {0, 1, 0, 1}) for (int ptr = 0; ptr < 2; ++ptr) {
            y->src[1] = x[ptr];
            ggml_mul_mat_set_prec(y, mode ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
            std::vector<float> first;
            for (int repeat = 0; repeat < 2; ++repeat) {
                if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                    throw std::runtime_error("packed projection compute failed");
                std::vector<float> got(size_t(rows)*cols);
                for (int col = 0; col < cols; ++col)
                    ggml_backend_tensor_get(y, got.data()+size_t(col)*rows, col*y->nb[1], rows*sizeof(float));
                if (repeat && got != first) throw std::runtime_error("packed projection replay changed");
                first = got;
                if (mode) {
                    for (int col = 0; col < cols; ++col) for (int row = 0; row < rows; ++row) {
                        double sum = 0;
                        for (int k = 0; k < width; ++k)
                            sum += double(decoded[size_t(row)*width+k])*double(inputs[ptr][size_t(col)*stride+k]);
                        if (got[size_t(col)*rows+row] != float(sum))
                            throw std::runtime_error("CPU packed F32 projection differs from independent code oracle");
                    }
                } else {
                    if (defaults[ptr].empty()) defaults[ptr] = got;
                    else if (defaults[ptr] != got) throw std::runtime_error("default projection changed after F32 switch");
                }
                ++cases;
            }
        }
        for (int ptr = 0; ptr < 2; ++ptr) {
            std::vector<float> got(inputs[ptr].size());
            ggml_backend_tensor_get(storage[ptr], got.data(), 0, got.size()*sizeof(float));
            if (got != inputs[ptr]) throw std::runtime_error("packed projection modified input/padding");
        }
        if (padded) {
            std::vector<float> got(sentinel.size());
            ggml_backend_tensor_get(y, got.data(), 0, got.size()*sizeof(float));
            for (int col = 0; col < cols-1; ++col) for (int row = rows; row < rows+13; ++row)
                if (got[size_t(col)*(rows+13)+row] != -6789.f)
                    throw std::runtime_error("packed projection modified output padding");
        }
    }
    std::printf("CPU packed explicit F32 projection: %d independent-code cases passed\n", cases);
}

// CPU reproduction of the existing double, 32-lane shuffle reduction.
// Check exact arithmetic under calibration, graph replay and weight mutation.
int main() {
    try {
        check_cpu_packed_projection();
#ifdef GGML_USE_CUDA
        int count = 0;
        for (int device = 0; device < ggml_backend_cuda_get_device_count(); ++device) {
            const char * selected = std::getenv("ESE_PROJECTION_TEST_DEVICE");
            if (selected && device != std::atoi(selected)) continue;
            std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(
                ggml_backend_cuda_init(device, nullptr, nullptr), ggml_backend_free);
            if (!backend) throw std::runtime_error("CUDA backend initialization failed");
            for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
                for (int geometry = 0; geometry < 3; ++geometry) {
                    const int width = geometry == 0 ? 1024 : geometry == 1 ? 2560 : 10240;
                    const int rows = geometry == 0 ? 17 : geometry == 1 ? 256 : 4;
                    for (int cols : {1, 3}) {
                        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(
                            ggml_init({1 << 20, nullptr, true}), ggml_free);
                        auto w = ggml_new_tensor_2d(ctx.get(), type, width, rows);
                        auto x = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, cols);
                        auto y = ggml_mul_mat(ctx.get(), w, x);
                        ggml_mul_mat_set_prec(y, GGML_PREC_F32);
                        auto graph = ggml_new_graph(ctx.get());
                        ggml_build_forward_expand(graph, y);
                        std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                            ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
                        if (!buffer) throw std::runtime_error("projection fixture allocation failed");
                        std::vector<float> weights(size_t(width)*rows), input(size_t(width)*cols), got(size_t(rows)*cols);
                        std::vector<ggml_fp16_t> half(weights.size());
                        std::vector<ggml_bf16_t> bf16(weights.size());
                        for (int phase = 0; phase < 2; ++phase) {
                            for (size_t i = 0; i < weights.size(); ++i) {
                                weights[i] = float(std::sin(i*.013 + phase)*.05);
                                half[i] = ggml_fp32_to_fp16(weights[i]);
                                bf16[i] = ggml_fp32_to_bf16(weights[i]);
                                if (type == GGML_TYPE_F16) weights[i] = ggml_fp16_to_fp32(half[i]);
                                if (type == GGML_TYPE_BF16) weights[i] = ggml_bf16_to_fp32(bf16[i]);
                            }
                            const void * data = type == GGML_TYPE_F32 ? (void *) weights.data() :
                                type == GGML_TYPE_F16 ? (void *) half.data() : (void *) bf16.data();
                            ggml_backend_tensor_set(w, data, 0, ggml_nbytes(w));
                            for (int pattern = 0; pattern < 4; ++pattern) {
                                for (size_t i = 0; i < input.size(); ++i) {
                                    const float value = float(std::sin(i*.017) + std::cos(i*.031));
                                    input[i] = pattern == 0 ? value : pattern == 1 ? value*32768.f :
                                        pattern == 2 ? ((i & 1) ? -1024.f : 1024.f) : value*1e-7f;
                                }
                                ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
                                std::vector<float> expected(got.size());
                                for (int col = 0; col < cols; ++col) for (int row = 0; row < rows; ++row) {
                                    std::array<double, 32> sums = {};
                                    for (int lane = 0; lane < 32; ++lane)
                                        for (int k = lane; k < width; k += 32)
                                            sums[lane] += double(weights[size_t(row)*width + k])*double(input[size_t(col)*width + k]);
                                    for (int offset = 16; offset > 0; offset /= 2)
                                        for (int lane = 0; lane + offset < 32; ++lane) sums[lane] += sums[lane + offset];
                                    expected[size_t(col)*rows + row] = float(sums[0]);
                                }
                                for (int repeat = 0; repeat < 3; ++repeat) {
                                    if (ggml_backend_graph_compute(backend.get(), graph) != GGML_STATUS_SUCCESS)
                                        throw std::runtime_error("projection compute failed");
                                    ggml_backend_tensor_get(y, got.data(), 0, ggml_nbytes(y));
                                    if (memcmp(got.data(), expected.data(), ggml_nbytes(y)))
                                        throw std::runtime_error("projection differs from the reference lane arithmetic");
                                    ++count;
                                }
                            }
                        }
                        ggml_backend_synchronize(backend.get());
                    }
                }
            }
        }
        if (count == 0) throw std::runtime_error("no CUDA test device selected");
        std::printf("Explicit F32 projection: %d exact cases passed\n", count);
#else
        std::puts("Explicit projection: CUDA coverage unavailable in this build");
#endif
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
