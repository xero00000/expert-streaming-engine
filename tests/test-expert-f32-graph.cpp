#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-expert-f32.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<unsigned char> read(const std::string & path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(bool(stream), "payload open");
    std::vector<unsigned char> result(size_t(stream.tellg()));
    stream.seekg(0); stream.read(reinterpret_cast<char *>(result.data()), result.size());
    require(bool(stream), "payload read"); return result;
}

static void unsupported_queries(ggml_backend_t backend, ggml_tensor * h, ggml_tensor * y) {
    int cases = 0;
    auto reject = [&](ggml_tensor * original, const std::function<void(ggml_tensor &, ggml_tensor *)> & change) {
        auto op = *original;
        ggml_tensor sources[GGML_MAX_SRC];
        for (int i = 0; i < GGML_MAX_SRC; ++i) if (op.src[i]) { sources[i] = *op.src[i]; op.src[i] = &sources[i]; }
        change(op, sources);
        require(!ggml_backend_supports_op(backend, &op), "unsupported explicit layout accepted");
        ++cases;
    };
    reject(h, [](ggml_tensor & op, ggml_tensor *) { op.op_params[0] = GGML_UNARY_OP_GELU; });
    reject(h, [](ggml_tensor & op, ggml_tensor *) { const float limit = 1; std::memcpy(op.op_params + 1, &limit, sizeof(limit)); });
    reject(h, [](ggml_tensor & op, ggml_tensor * src) { op.src[4] = &src[0]; });
    reject(h, [](ggml_tensor &, ggml_tensor * src) { src[1].type = GGML_TYPE_F32; });
    reject(h, [](ggml_tensor &, ggml_tensor * src) { src[1].nb[2] = 1; });
    reject(h, [](ggml_tensor &, ggml_tensor * src) { src[2].view_offs = 1; });
    reject(h, [](ggml_tensor &, ggml_tensor * src) { src[3].view_offs = 1; });
    reject(h, [](ggml_tensor & op, ggml_tensor *) { op.nb[1] += 1; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { --src[0].ne[0]; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[0].view_offs = 1; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[0].nb[1] = 2; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[0].type = GGML_TYPE_F32; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[1].type = GGML_TYPE_F16; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[1].ne[3] = 2; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[2].type = GGML_TYPE_I64; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[2].ne[0] = 65536; });
    reject(y, [](ggml_tensor &, ggml_tensor * src) { src[2].ne[1] = 65536; });
    reject(y, [](ggml_tensor & op, ggml_tensor *) { op.ne[3] = 2; });
#ifdef GGML_USE_CUDA
    auto * split = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_split_buffer_type(nullptr), 128);
    require(split, "split parent metadata buffer");
    reject(y, [split](ggml_tensor &, ggml_tensor * src) { src[0].buffer = split; });
    ggml_backend_buffer_free(split);
#endif
    std::printf("expert_f32_support: rejected=%d passed=true\n", cases);
}

static void run(ggml_backend_t backend, const std::string & directory, ggml_type up_type, ggml_type down_type) {
    constexpr int k = 256, m = 256, n = 8, experts = 2, slots = 2;
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true}); require(ctx, "context");
    auto * up = ggml_new_tensor_3d(ctx, up_type, k, m, experts);
    auto * gate = ggml_new_tensor_3d(ctx, up_type, k, m, experts);
    auto * down = ggml_new_tensor_3d(ctx, down_type, m, n, experts);
    auto * storage = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2 * k);
    auto * x = ggml_view_3d(ctx, storage, k, 1, 1, k * sizeof(float), k * sizeof(float), 0);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, slots, 1);
    auto * h = ggml_moe_up_gate(ctx, up, gate, x, ids, GGML_UNARY_OP_SILU);
    auto * y = ggml_mul_mat_id(ctx, down, h, ids);
    auto * paired_default = ggml_mul_mat_id(ctx, up, x, ids);
    auto * paired_explicit = ggml_mul_mat_id(ctx, gate, x, ids);
    ggml_mul_mat_set_prec(paired_explicit, GGML_PREC_F32);
    ggml_mul_mat_set_prec(h, GGML_PREC_F32); ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    unsupported_queries(backend, h, y);
    auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "allocation");
    std::vector<unsigned char> bytes[3];
    const char * names[] = {"up.bin", "gate.bin", "down.bin"};
    ggml_tensor * tensors[] = {up, gate, down};
    for (int role = 0; role < 3; ++role) {
        auto original = read(directory + "/" + names[role]);
        auto * tensor = tensors[role];
        const int original_columns = role == 2 ? 640 : 2560;
        const size_t original_row = ggml_row_size(tensor->type, original_columns);
        bytes[role].resize(ggml_nbytes(tensor));
        for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < tensor->ne[1]; ++row)
            std::memcpy(bytes[role].data() + expert * tensor->nb[2] + row * tensor->nb[1],
                        original.data() + (row + expert * 13) * original_row, tensor->nb[1]);
        ggml_backend_tensor_set(tensor, bytes[role].data(), 0, bytes[role].size());
    }
    std::vector<float> inputs(2 * k);
    for (int i = 0; i < 2 * k; ++i) inputs[i] = float((i * 17) % 101 - 50) / 64;
    ggml_backend_tensor_set(storage, inputs.data(), 0, inputs.size() * sizeof(float));
    int cases = 0;
    std::vector<float> default_baseline;
    // Change precision on the exact graph, tensor and buffer addresses. Mixed
    // modes catch a legacy fusion consuming an explicit-precision next node.
    for (int mode : {0, 3, 1, 2, 3, 0}) {
        ggml_mul_mat_set_prec(h, mode & 2 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
        ggml_mul_mat_set_prec(y, mode & 1 ? GGML_PREC_F32 : GGML_PREC_DEFAULT);
        require(ggml_backend_supports_op(backend, h) && ggml_backend_supports_op(backend, y), "mode support");
        for (int offset : {0, k, 0}) {
            x->view_offs = offset * sizeof(float);
            x->data = static_cast<char *>(storage->data) + x->view_offs;
            std::memcpy(x->op_params, &x->view_offs, sizeof(x->view_offs));
            const int32_t routes[slots] = {offset ? 1 : 0, offset ? 0 : 1};
            ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
            for (int replay = 0; replay < 3; ++replay) {
                require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute/update/replay");
                std::vector<float> hidden(m * slots), actual(n * slots);
                ggml_backend_tensor_get(h, hidden.data(), 0, hidden.size() * sizeof(float));
                ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
                if (mode & 2) for (int slot = 0; slot < slots; ++slot) for (int row = 0; row < m; ++row) {
                    const int expert = routes[slot];
                    const float u = ggml_expert_f32_dot(up_type, bytes[0].data() + expert * up->nb[2] + row * up->nb[1], inputs.data() + offset, k);
                    const float g = ggml_expert_f32_dot(up_type, bytes[1].data() + expert * gate->nb[2] + row * gate->nb[1], inputs.data() + offset, k);
                    const float expected = ggml_expert_f32_silu_product(g, u);
                    require(std::memcmp(&expected, &hidden[slot * m + row], sizeof(float)) == 0, "fused explicit precision mismatch");
                }
                if (mode & 1) for (int slot = 0; slot < slots; ++slot) for (int row = 0; row < n; ++row) {
                    const float expected = ggml_expert_f32_dot(down_type, bytes[2].data() + routes[slot] * down->nb[2] + row * down->nb[1], hidden.data() + slot * m, m);
                    require(std::memcmp(&expected, &actual[slot * n + row], sizeof(float)) == 0, "down explicit precision bypassed");
                }
                if (mode == 0 && offset == 0) {
                    if (default_baseline.empty()) default_baseline = actual;
                    require(std::memcmp(default_baseline.data(), actual.data(), actual.size() * sizeof(float)) == 0, "default result changed after precision switches");
                }
                ++cases;
            }
        }
    }
    std::printf("expert_f32_graph: cases=%d precision_modes=4 pointer_updates=true default_replay_exact=true passed=true\n", cases);
    auto * pair_graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(pair_graph, paired_default);
    ggml_build_forward_expand(pair_graph, paired_explicit);
    const int32_t routes[slots] = {0, 1};
    ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
    for (int replay = 0; replay < 3; ++replay) {
        require(ggml_backend_graph_compute(backend, pair_graph) == GGML_STATUS_SUCCESS, "paired projection compute");
        std::vector<float> actual(m * slots);
        ggml_backend_tensor_get(paired_explicit, actual.data(), 0, actual.size() * sizeof(float));
        for (int slot = 0; slot < slots; ++slot) for (int row = 0; row < m; ++row) {
            const float expected = ggml_expert_f32_dot(up_type, bytes[1].data() + routes[slot] * gate->nb[2] + row * gate->nb[1], inputs.data(), k);
            require(std::memcmp(&expected, &actual[slot * m + row], sizeof(float)) == 0, "paired explicit precision bypassed");
        }
    }
    std::puts("expert_f32_pair: replays=3 explicit_next_exact=true passed=true");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 5, "usage: test-expert-f32-graph DATA UP_TYPE DOWN_TYPE DEVICE");
#ifdef GGML_USE_CUDA
        const int device = std::atoi(argv[4]);
        CUdevice raw; CUcontext primary;
        require(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&raw, device) == CUDA_SUCCESS &&
                cuDevicePrimaryCtxRetain(&primary, raw) == CUDA_SUCCESS && cuCtxSetCurrent(primary) == CUDA_SUCCESS, "primary context");
        auto * backend = ggml_backend_cuda_init(device, nullptr, nullptr); require(backend, "CUDA backend");
        run(backend, argv[1], ggml_type(std::atoi(argv[2])), ggml_type(std::atoi(argv[3])));
        ggml_backend_free(backend);
        require(cuDevicePrimaryCtxRelease(raw) == CUDA_SUCCESS, "primary release");
#else
        throw std::runtime_error("CUDA build required");
#endif
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
