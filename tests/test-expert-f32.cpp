#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-expert-f32.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const char * what) {
    if (!value) throw std::runtime_error(what);
}
template <typename T> static std::vector<T> read(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    require(bool(f), "input open");
    const size_t bytes = size_t(f.tellg());
    require(bytes % sizeof(T) == 0, "input size");
    std::vector<T> v(bytes / sizeof(T)); f.seekg(0);
    f.read(reinterpret_cast<char *>(v.data()), bytes); require(bool(f), "input read"); return v;
}

struct source {
    std::vector<uint8_t> data[4]; // gate, up, down, merged gate/up
    size_t span[4] = {}, budget = 0, active = 0, peak = 0, acquires = 0, releases = 0, rejected = 0, gate_rejected = 0;
    struct handle { source * owner; std::vector<uint8_t> bytes; };
    static bool acquire(void * user, int layer, int expert, int component, ggml_expert_lease * result) {
        auto & s = *static_cast<source *>(user);
        require(layer == 0 && expert >= 0 && expert < 4 && component >= 0 && component < 4, "lease identity");
        if (s.active + s.span[component] > s.budget) { ++s.rejected; if (component == 0) ++s.gate_rejected; return false; }
        auto h = std::make_unique<handle>(); h->owner = &s;
        const auto first = s.data[component].begin() + expert * s.span[component];
        h->bytes.assign(first, first + s.span[component]);
        s.active += h->bytes.size(); s.peak = std::max(s.peak, s.active); ++s.acquires;
        result->data = h->bytes.data(); result->size = h->bytes.size(); result->handle = h.release(); return true;
    }
    static void release(void * user, void * opaque) {
        std::unique_ptr<handle> h(static_cast<handle *>(opaque));
        auto & s = *static_cast<source *>(user); require(h->owner == &s, "lease owner");
        require(s.active >= h->bytes.size(), "lease accounting");
        s.active -= h->bytes.size(); ++s.releases;
        std::fill(h->bytes.begin(), h->bytes.end(), 0xff);
    }
};

static void run(const std::string & path, ggml_type up_type, ggml_type down_type, int threads, bool leases, int device = -1, bool merged = false) {
    constexpr int k = 2560, m = 640, n = 2560, n_as = 4, n_ids = 6;
    const int tokens = std::getenv("ESE_EXPERT_F32_TEST_TOKENS") ? std::atoi(std::getenv("ESE_EXPERT_F32_TEST_TOKENS")) : 2;
    const int input_slots = std::getenv("ESE_EXPERT_F32_TEST_INPUT_SLOTS") ? std::atoi(std::getenv("ESE_EXPERT_F32_TEST_INPUT_SLOTS")) : 2;
    require(tokens > 0 && tokens <= 128 && input_slots > 0 && n_ids % input_slots == 0, "test geometry");
    const auto input = read<float>(path + "/input.f32");
    const auto raw_up = read<uint8_t>(path + "/up.bin"), raw_gate = read<uint8_t>(path + "/gate.bin"), raw_down = read<uint8_t>(path + "/down.bin");
    require(input.size() == k, "activation width");
    const size_t logical_up = ggml_row_size(up_type, k), logical_down = ggml_row_size(down_type, m);
    require(raw_up.size() == logical_up * m && raw_gate.size() == raw_up.size() && raw_down.size() == logical_down * n, "weight geometry");
    source s;
    const size_t row_stride[3] = {ggml_row_size(up_type, k + ggml_blck_size(up_type)), logical_up,
                                 ggml_row_size(down_type, m + ggml_blck_size(down_type))};
    const int rows[3] = {m, m, n};
    const std::vector<uint8_t> * original[3] = {&raw_gate, &raw_up, &raw_down};
    for (int role = 0; role < 3; ++role) {
        s.span[role] = row_stride[role] * (rows[role] + 1);
        s.data[role].assign(s.span[role] * n_as, 0xff);
        const size_t width = role == 2 ? logical_down : logical_up;
        for (int expert = 0; expert < n_as; ++expert) for (int row = 0; row < rows[role]; ++row) {
            const int source_row = (row + expert * 13 + role * 3) % rows[role];
            std::memcpy(s.data[role].data() + expert * s.span[role] + row * row_stride[role], original[role]->data() + source_row * width, width);
        }
    }
    s.budget = std::max(s.span[0] + 2 * s.span[1], 2 * s.span[2]);
    auto ctx = ggml_init({ggml_tensor_overhead() * 32 + ggml_graph_overhead_custom(32, false), nullptr, true});
    require(ctx, "context");
    ggml_tensor * weights[3];
    for (int role = 0; role < 3; ++role) {
        const auto type = role == 2 ? down_type : up_type;
        const int cols = role == 2 ? m : k;
        auto base = ggml_new_tensor_1d(ctx, type, int64_t(s.data[role].size() / ggml_type_size(type)) * ggml_blck_size(type));
        weights[role] = ggml_view_3d(ctx, base, cols, rows[role], n_as, row_stride[role], s.span[role], 0);
        ggml_set_name(weights[role], role == 0 ? "blk.0.ffn_gate_exps.weight" : role == 1 ? "blk.0.ffn_up_exps.weight" : "blk.0.ffn_down_exps.weight");
    }
    ggml_tensor * merged_weights = nullptr;
    if (merged) {
        const size_t pitch = row_stride[0];
        s.span[3] = pitch * (2 * m + 1);
        s.data[3].assign(n_as * s.span[3], 0xff);
        for (int expert = 0; expert < n_as; ++expert) for (int row = 0; row < m; ++row) {
            std::memcpy(s.data[3].data() + expert * s.span[3] + row * pitch,
                        s.data[0].data() + expert * s.span[0] + row * row_stride[0], logical_up);
            std::memcpy(s.data[3].data() + expert * s.span[3] + (row + m) * pitch,
                        s.data[1].data() + expert * s.span[1] + row * row_stride[1], logical_up);
        }
        auto base = ggml_new_tensor_1d(ctx, up_type, int64_t(s.data[3].size() / ggml_type_size(up_type)) * ggml_blck_size(up_type));
        merged_weights = ggml_view_3d(ctx, base, k, 2 * m, n_as, pitch, s.span[3], 0);
        ggml_set_name(merged_weights, "blk.0.ffn_gate_up_exps.weight");
        s.budget = std::max(s.span[3], 2 * s.span[2]);
    }
    auto input_base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (k + 13) * (input_slots + 1) * tokens);
    auto x = ggml_view_3d(ctx, input_base, k, input_slots, tokens, (k + 13) * sizeof(float), (k + 13) * (input_slots + 1) * sizeof(float), 0);
    auto id_base = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (n_ids + 3) * tokens);
    auto ids = ggml_view_2d(ctx, id_base, n_ids, tokens, (n_ids + 3) * sizeof(int32_t), 0);
    auto h = ggml_moe_up_gate(ctx, merged ? merged_weights : weights[1], merged ? nullptr : weights[0], x, ids, GGML_UNARY_OP_SILU);
    h->nb[1] = (m + 3) * sizeof(float); h->nb[2] = h->nb[1] * (n_ids + 1); h->nb[3] = h->nb[2] * tokens;
    auto y = ggml_mul_mat_id(ctx, weights[2], h, ids);
    y->nb[1] = (n + 3) * sizeof(float); y->nb[2] = y->nb[1] * (n_ids + 1); y->nb[3] = y->nb[2] * tokens;
    ggml_mul_mat_set_prec(h, GGML_PREC_F32); ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    auto graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, y);
    ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
    if (device >= 0) backend = ggml_backend_cuda_init(device, nullptr, nullptr);
#endif
    if (device < 0) {
        backend = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(backend, threads);
    }
    require(backend, "backend");
    require(ggml_backend_supports_op(backend, h) && ggml_backend_supports_op(backend, y), "precision support query");
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "buffers");
    for (int role = 0; role < 3; ++role) {
        if (leases) { std::vector<uint8_t> poison(s.data[role].size(), 0xff); ggml_backend_tensor_set(weights[role]->view_src, poison.data(), 0, poison.size()); }
        else ggml_backend_tensor_set(weights[role]->view_src, s.data[role].data(), 0, s.data[role].size());
    }
    if (merged) {
        if (leases) { std::vector<uint8_t> poison(s.data[3].size(), 0xff); ggml_backend_tensor_set(merged_weights->view_src, poison.data(), 0, poison.size()); }
        else ggml_backend_tensor_set(merged_weights->view_src, s.data[3].data(), 0, s.data[3].size());
    }
    std::vector<float> xdata(ggml_nelements(input_base), NAN);
    for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < input_slots; ++slot)
        for (int col = 0; col < k; ++col) xdata[token * (k + 13) * (input_slots + 1) + slot * (k + 13) + col] = input[col] + float(slot + token) / 32;
    ggml_backend_tensor_set(input_base, xdata.data(), 0, xdata.size() * sizeof(float));
    const int32_t route[2][n_ids] = {{2, 0, -1, 2, 3, 99}, {1, 2, -5, 0, 1, 4}};
    std::vector<int32_t> iddata((n_ids + 3) * tokens, 99);
    for (int token = 0; token < tokens; ++token) std::copy(route[token % 2], route[token % 2] + n_ids, iddata.begin() + token * (n_ids + 3));
    ggml_backend_tensor_set(id_base, iddata.data(), 0, iddata.size() * sizeof(int32_t));
    if (leases) ggml_backend_cpu_set_expert_lease_callbacks(backend, source::acquire, source::release, &s, true);
    constexpr uint32_t padding_bits = 0x7fc12345;
    for (auto * tensor : {h, y}) {
        std::vector<uint32_t> poison(ggml_nbytes(tensor) / sizeof(float), padding_bits);
        ggml_backend_tensor_set(tensor, poison.data(), 0, poison.size() * sizeof(float));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "backend compute");
    std::vector<uint8_t> actual(ggml_nbytes(y)); ggml_backend_tensor_get(y, actual.data(), 0, actual.size());
    for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < n_ids; ++slot) {
        const int expert = route[token % 2][slot]; std::vector<float> expected(n, 0);
        if (expert >= 0 && expert < n_as) {
            const float * column = xdata.data() + token * (k + 13) * (input_slots + 1) + (slot % input_slots) * (k + 13);
            std::vector<float> hidden(m);
            for (int row = 0; row < m; ++row) {
                const float u = ggml_expert_f32_dot(up_type, s.data[1].data() + expert * s.span[1] + row * row_stride[1], column, k);
                const float g = ggml_expert_f32_dot(up_type, s.data[0].data() + expert * s.span[0] + row * row_stride[0], column, k);
                hidden[row] = ggml_expert_f32_silu_product(g, u);
            }
            for (int row = 0; row < n; ++row)
                expected[row] = ggml_expert_f32_dot(down_type, s.data[2].data() + expert * s.span[2] + row * row_stride[2], hidden.data(), m);
        }
        require(std::memcmp(actual.data() + token * y->nb[2] + slot * y->nb[1], expected.data(), n * sizeof(float)) == 0, "indexed output mismatch");
    }
    for (auto * tensor : {h, y}) {
        std::vector<uint32_t> bytes(ggml_nbytes(tensor) / sizeof(float));
        ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size() * sizeof(float));
        std::vector<bool> logical(bytes.size(), false);
        for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < n_ids; ++slot)
            for (int row = 0; row < tensor->ne[0]; ++row)
                logical[(token * tensor->nb[2] + slot * tensor->nb[1]) / sizeof(float) + row] = true;
        for (size_t i = 0; i < bytes.size(); ++i) if (!logical[i]) require(bytes[i] == padding_bits, "output padding changed");
    }
    require(s.active == 0 && s.acquires == s.releases && s.peak <= s.budget, "lease lifetime/budget");
    if (leases) require(s.acquires > 0 && s.rejected > 0 && (merged || s.gate_rejected > 0), "lease admission pressure exercised");
    std::printf("expert_f32_cpu: threads=%d leases=%d values=%lld acquires=%zu rejected=%zu peak=%zu budget=%zu device=%d merged=%d tokens=%d input_slots=%d padding_exact=true passed=true\n", threads, leases, (long long) ggml_nelements(y), s.acquires, s.rejected, s.peak, s.budget, device, merged, tokens, input_slots);
    ggml_backend_buffer_free(buffer); ggml_backend_free(backend); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        require(argc >= 4 && argc <= 6, "usage: test-expert-f32 DATA_DIRECTORY UP_TYPE DOWN_TYPE [CUDA_DEVICE [MERGED]]");
        const int device = argc >= 5 ? std::atoi(argv[4]) : -1;
        const bool merged = argc == 6 && std::atoi(argv[5]) != 0;
#ifdef GGML_USE_CUDA
        CUcontext primary = nullptr;
        CUdevice raw = 0;
        if (device >= 0) {
            require(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&raw, device) == CUDA_SUCCESS &&
                    cuDevicePrimaryCtxRetain(&primary, raw) == CUDA_SUCCESS && cuCtxSetCurrent(primary) == CUDA_SUCCESS,
                    "CUDA primary context");
        }
#else
        require(device < 0, "CUDA unavailable in this build");
#endif
        if (device < 0) {
            for (int threads : {1, 8}) for (bool leases : {false, true})
                run(argv[1], ggml_type(std::atoi(argv[2])), ggml_type(std::atoi(argv[3])), threads, leases, -1, merged);
        } else {
            run(argv[1], ggml_type(std::atoi(argv[2])), ggml_type(std::atoi(argv[3])), 1, false, device, merged);
        }
#ifdef GGML_USE_CUDA
        if (primary) require(cuDevicePrimaryCtxRelease(raw) == CUDA_SUCCESS, "CUDA primary context release");
#endif
        return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
