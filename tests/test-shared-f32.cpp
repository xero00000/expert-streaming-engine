#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }
template<typename T> static std::vector<T> read(const std::string & file) {
    std::ifstream input(file, std::ios::binary | std::ios::ate);
    check(bool(input), "fixture open");
    const size_t size = size_t(input.tellg()); check(size % sizeof(T) == 0, "fixture alignment");
    std::vector<T> result(size / sizeof(T)); input.seekg(0);
    input.read(reinterpret_cast<char *>(result.data()), size); check(bool(input), "fixture read");
    return result;
}

int main(int argc, char ** argv) {
    try {
        check(argc == 9, "usage: fixture gate_type up_type down_type device threads precision output.bin");
        const std::string path = argv[1];
        const ggml_type types[] = {ggml_type(std::stoi(argv[2])), ggml_type(std::stoi(argv[3])), ggml_type(std::stoi(argv[4]))};
        const int device = std::stoi(argv[5]), threads = std::stoi(argv[6]), precision = std::stoi(argv[7]);
        check(precision == 0 || precision == 1, "precision mode");
        const char * pad_env = std::getenv("ESE_SHARED_TEST_PAD");
        const int padding = pad_env ? std::stoi(pad_env) : precision ? 17 : 0;
        check(padding >= 0 && padding <= 64, "input padding");
        std::ofstream artifact(argv[8], std::ios::binary); check(bool(artifact), "output artifact");
        ggml_backend_t raw_backend = nullptr;
#ifdef GGML_USE_CUDA
        CUdevice cuda_device = 0; CUcontext primary = nullptr;
        if (device >= 0) {
            check(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&cuda_device, device) == CUDA_SUCCESS &&
                  cuDevicePrimaryCtxRetain(&primary, cuda_device) == CUDA_SUCCESS && cuCtxSetCurrent(primary) == CUDA_SUCCESS,
                  "CUDA primary context");
            raw_backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        }
#else
        check(device < 0, "CUDA unavailable");
#endif
        if (device < 0) { raw_backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(raw_backend, threads); }
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw_backend, ggml_backend_free);
        check(bool(backend), "backend initialization");
        constexpr int hidden = 2560, inner = 640, max_tokens = 8;
        const auto input = read<float>(path + "/input.f32"); check(input.size() == hidden * max_tokens, "input dimensions");
        const char * names[] = {"gate", "up", "down"};
        std::vector<uint8_t> bytes[3]; std::vector<float> expected[4];
        for (int i=0; i<3; ++i) bytes[i] = read<uint8_t>(path + "/" + names[i] + ".bin");
        const char * outputs[] = {"gate", "up", "fused", "down"};
        for (int i=0; i<4; ++i) expected[i] = read<float>(path + "/" + outputs[i] + ".f32");
        size_t comparisons = 0; double maximum = 0;
        for (int tokens : {1, 2, 8}) {
            std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({4<<20, nullptr, true}), ggml_free);
            check(bool(ctx), "tensor context");
            auto * storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, hidden+padding, tokens);
            auto * x = ggml_view_2d(ctx.get(), storage, hidden, tokens, storage->nb[1], 0);
            auto * g = ggml_new_tensor_2d(ctx.get(), types[0], hidden, inner);
            auto * u = ggml_new_tensor_2d(ctx.get(), types[1], hidden, inner);
            auto * d = ggml_new_tensor_2d(ctx.get(), types[2], inner, hidden);
            ggml_tensor * weights[] = {g, u, d};
            auto * gate = ggml_mul_mat(ctx.get(), g, x);
            auto * up = ggml_mul_mat(ctx.get(), u, x);
            auto * fused = ggml_fused_mul_unary(ctx.get(), gate, up, GGML_UNARY_OP_SILU);
            auto * down = ggml_mul_mat(ctx.get(), d, fused);
            ggml_mul_mat_set_prec(gate, precision ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
            ggml_mul_mat_set_prec(up, precision ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
            ggml_mul_mat_set_prec(down, precision ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
            ggml_tensor * result[] = {gate, up, fused, down};
            auto * graph = ggml_new_graph_custom(ctx.get(), 1024, false);
            for (auto * tensor : result) {
                check(ggml_backend_supports_op(backend.get(), tensor), "required operator unsupported");
                ggml_set_output(tensor); ggml_build_forward_expand(graph, tensor);
            }
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()), ggml_backend_buffer_free);
            check(bool(buffer), "backend allocation");
            for (int i=0; i<3; ++i) {
                check(ggml_nbytes(weights[i]) == bytes[i].size(), "weight dimensions");
                ggml_backend_tensor_set(weights[i], bytes[i].data(), 0, bytes[i].size());
            }
            std::vector<float> padded((hidden+padding)*tokens, -12345.f);
            for (int t=0; t<tokens; ++t) std::copy_n(input.data()+t*hidden, hidden, padded.data()+t*(hidden+padding));
            ggml_backend_tensor_set(storage, padded.data(), 0, padded.size()*sizeof(float));
            std::vector<float> first[4];
            for (int replay=0; replay<3; ++replay) {
                check(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "graph compute");
                for (int i=0; i<4; ++i) {
                    const int rows = i == 3 ? hidden : inner;
                    std::vector<float> got(rows*tokens);
                    ggml_backend_tensor_get(result[i], got.data(), 0, got.size()*sizeof(float));
                    check(expected[i].size() == rows*max_tokens, "oracle dimensions");
                    if (!replay) {
                        first[i] = got;
                        artifact.write(reinterpret_cast<const char *>(got.data()), got.size()*sizeof(float));
                    } else check(got == first[i], "replay changed output");
                    for (size_t j=0; j<got.size(); ++j) {
                        check(std::isfinite(got[j]), "nonfinite output");
                        const double error = std::abs(double(got[j])-expected[i][j]);
                        maximum = std::max(maximum, error);
                        if (precision && error > 2e-5 * (1 + std::abs(double(expected[i][j])))) {
                            std::fprintf(stderr, "stage=%s tokens=%d index=%zu got=%.9g expected=%.9g error=%.9g\n",
                                         outputs[i], tokens, j, got[j], expected[i][j], error);
                            throw std::runtime_error("explicit projection disagrees with original-weight oracle");
                        }
                        ++comparisons;
                    }
                }
            }
            std::vector<float> after(padded.size()); ggml_backend_tensor_get(storage, after.data(), 0, after.size()*sizeof(float));
            check(after == padded, "input padding mutated");
        }
        artifact.close(); check(bool(artifact), "artifact write");
        ggml_backend_synchronize(backend.get()); backend.reset();
#ifdef GGML_USE_CUDA
        if (device >= 0) check(cuDevicePrimaryCtxRelease(cuda_device) == CUDA_SUCCESS, "CUDA context release");
#endif
        std::printf("PASS: shared projections device=%d threads=%d precision=%d comparisons=%zu max_abs=%.9g\n",
                    device, threads, precision, comparisons, maximum);
    } catch (const std::exception & error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
    return 0;
}
