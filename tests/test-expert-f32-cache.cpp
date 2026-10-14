#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-expert-f32.h"
#include "ggml-expert-loader.h"
#include "ggml-cuda.h"
#include <cuda.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<class T> static std::vector<T> read(const std::string & path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate); require(bool(f), "payload open");
    const size_t bytes = size_t(f.tellg()); require(bytes % sizeof(T) == 0, "payload size");
    std::vector<T> result(bytes / sizeof(T)); f.seekg(0);
    f.read(reinterpret_cast<char *>(result.data()), bytes); require(bool(f), "payload read"); return result;
}
struct source {
    std::vector<uint8_t> data[3]; size_t span[3] = {};
    size_t acquires = 0, releases = 0, active = 0, peak = 0, batches = 0;
    struct lease { source * owner; std::vector<uint8_t> bytes; };
    struct batch { source * owner; std::vector<int> experts; int component; size_t remaining; };
    static bool acquire(void * opaque, int layer, int expert, int component, ggml_backend_expert_lease * result) {
        auto & s = *static_cast<source *>(opaque);
        require(layer == 0 && expert >= 0 && expert < 8 && component >= 0 && component < 3, "lease identity");
        auto handle = std::make_unique<lease>(); handle->owner = &s;
        const auto first = s.data[component].begin() + expert * s.span[component];
        handle->bytes.assign(first, first + s.span[component]);
        s.active += handle->bytes.size(); s.peak = std::max(s.peak, s.active); ++s.acquires;
        result->data = handle->bytes.data(); result->size = handle->bytes.size(); result->handle = handle.release();
        return true;
    }
    static void release(void * opaque, void * value) {
        auto & s = *static_cast<source *>(opaque); std::unique_ptr<lease> handle(static_cast<lease *>(value));
        require(handle->owner == &s && s.active >= handle->bytes.size(), "lease accounting");
        s.active -= handle->bytes.size(); ++s.releases;
        std::fill(handle->bytes.begin(), handle->bytes.end(), 0xff);
    }
    static void * begin(void * opaque, int layer, const int * experts, size_t count, int component) {
        require(layer == 0, "batch layer"); auto * s = static_cast<source *>(opaque); ++s->batches;
        return new batch{s, std::vector<int>(experts, experts + count), component, count};
    }
    static bool next(void * opaque, size_t * index, ggml_backend_expert_lease * result) {
        auto & b = *static_cast<batch *>(opaque); require(b.remaining > 0, "batch exhaustion");
        *index = --b.remaining;
        return acquire(b.owner, 0, b.experts[*index], b.component, result);
    }
    static void end(void * opaque) { delete static_cast<batch *>(opaque); }
};

static void run(const std::string & directory, ggml_type up_type, ggml_type down_type, int device) {
    constexpr int k = 2560, m = 640, n = 2560, experts = 8, slots = 2;
    auto * cpu = ggml_backend_cpu_init(); auto * gpu = ggml_backend_cuda_init(device, nullptr, nullptr);
    require(cpu && gpu, "backends");
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true}); require(ctx, "context");
    source s;
    auto * gate = ggml_new_tensor_3d(ctx, up_type, k, m, experts);
    auto * up = ggml_new_tensor_3d(ctx, up_type, k, m, experts);
    auto * down = ggml_new_tensor_3d(ctx, down_type, m, n, experts);
    ggml_tensor * weights[] = {gate, up, down};
    const char * names[] = {"gate.bin", "up.bin", "down.bin"};
    const char * tensor_names[] = {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight", "blk.0.ffn_down_exps.weight"};
    ggml_backend_buffer_t buffers[3];
    for (int role = 0; role < 3; ++role) {
        auto * tensor = weights[role]; ggml_set_name(tensor, tensor_names[role]);
        auto payload = read<uint8_t>(directory + "/" + names[role]);
        s.span[role] = tensor->nb[2]; require(payload.size() == s.span[role], "weight geometry");
        s.data[role].resize(ggml_nbytes(tensor));
        for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < tensor->ne[1]; ++row)
            std::memcpy(s.data[role].data() + expert * s.span[role] + row * tensor->nb[1],
                        payload.data() + ((row + expert * 13) % tensor->ne[1]) * tensor->nb[1], tensor->nb[1]);
        auto buft = ggml_backend_get_default_buffer_type(cpu);
        buffers[role] = ggml_backend_buft_alloc_buffer(buft, ggml_backend_buft_get_alloc_size(buft, tensor));
        require(buffers[role], "host weights allocation");
        ggml_backend_buffer_set_usage(buffers[role], GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        ggml_backend_tensor_alloc(buffers[role], tensor, ggml_backend_buffer_get_base(buffers[role]));
        ggml_backend_buffer_clear(buffers[role], 0xff); // mandatory leases must supply every actual payload
    }
    auto * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, 1); ggml_set_input(x);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, slots, 1); ggml_set_input(ids);
    auto * h = ggml_moe_up_gate(ctx, up, gate, x, ids, GGML_UNARY_OP_SILU);
    auto * y = ggml_mul_mat_id(ctx, down, h, ids); ggml_set_output(y);
    ggml_mul_mat_set_prec(h, GGML_PREC_F32); ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, y);
    ggml_backend_t backends[] = {gpu, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, 2, 128, false); require(sched, "scheduler");
    ggml_backend_sched_set_active_expert_decode(sched, true);
    ggml_backend_sched_set_only_active_experts(sched, true);
    ggml_backend_sched_set_expert_lease_callbacks(sched, source::acquire, source::release, &s, true);
    ggml_backend_sched_set_expert_load_callbacks(sched, {source::begin, source::next, source::end});
    const uint64_t capacity = 64 * 256 + 3 * (s.span[0] + s.span[1] + s.span[2]) + 3 * 256;
    ggml_backend_sched_set_expert_cache_capacity(sched, capacity, 0, 1);
    for (auto * tensor : {x, ids, h, y}) ggml_backend_sched_set_tensor_backend(sched, tensor, gpu);
    require(ggml_backend_sched_alloc_graph(sched, graph), "scheduler allocation");
    require(ggml_backend_sched_get_tensor_backend(sched, h) == gpu && ggml_backend_sched_get_tensor_backend(sched, y) == gpu,
            "hidden CPU fallback");
    require(h->src[0]->ne[2] < experts && h->src[1]->ne[2] < experts && y->src[0]->ne[2] < experts, "compact weights not redirected");
    auto input = read<float>(directory + "/input.f32"); require(input.size() == k, "input width");
    const int32_t routes[][slots] = {{6, 1}, {6, 1}, {3, 7}, {0, 4}, {1, 6}};
    for (int pass = 0; pass < 5; ++pass) {
        const size_t before = s.acquires;
        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
        ggml_backend_tensor_set(ids, routes[pass], 0, sizeof(routes[pass]));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "scheduler compute");
        ggml_backend_sched_synchronize(sched);
        std::vector<float> actual(n * slots); ggml_backend_tensor_get(y, actual.data(), 0, actual.size() * sizeof(float));
        for (int slot = 0; slot < slots; ++slot) {
            const int expert = routes[pass][slot]; std::vector<float> hidden(m);
            for (int row = 0; row < m; ++row) {
                const float g = ggml_expert_f32_dot(up_type, s.data[0].data() + expert * s.span[0] + row * gate->nb[1], input.data(), k);
                const float u = ggml_expert_f32_dot(up_type, s.data[1].data() + expert * s.span[1] + row * up->nb[1], input.data(), k);
                hidden[row] = ggml_expert_f32_silu_product(g, u);
            }
            for (int row = 0; row < n; ++row) {
                const float expected = ggml_expert_f32_dot(down_type, s.data[2].data() + expert * s.span[2] + row * down->nb[1], hidden.data(), m);
                require(std::memcmp(&expected, &actual[slot * n + row], sizeof(float)) == 0, "compact remap output mismatch");
            }
        }
        if (pass == 1) require(s.acquires == before, "warm route reacquired weights");
        ggml_backend_sched_resource_device_stats stats;
        require(ggml_backend_sched_get_resource_device_stats(sched, &stats, 1) && stats.expert_cache_allocated_bytes <= capacity,
                "cache memory cap");
    }
    ggml_backend_sched_free(sched);
    require(s.active == 0 && s.acquires == s.releases && s.batches > 0, "leases did not drain");
    std::printf("expert_f32_cache: routes=5 values=%d logical_experts=8 compact=true warm_reacquires=0 reversed_completions=true acquires=%zu releases=%zu peak_lease_bytes=%zu capacity=%llu device=%d passed=true\n",
                5 * n * slots, s.acquires, s.releases, s.peak, (unsigned long long)capacity, device);
    for (auto * buffer : buffers) ggml_backend_buffer_free(buffer);
    ggml_free(ctx); ggml_backend_free(gpu); ggml_backend_free(cpu);
}
int main(int argc, char ** argv) {
    try {
        require(argc == 5, "usage: test-expert-f32-cache DATA UP_TYPE DOWN_TYPE DEVICE");
        const int device = std::atoi(argv[4]); CUdevice raw; CUcontext primary;
        require(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&raw, device) == CUDA_SUCCESS &&
                cuDevicePrimaryCtxRetain(&primary, raw) == CUDA_SUCCESS && cuCtxSetCurrent(primary) == CUDA_SUCCESS, "primary context");
        run(argv[1], ggml_type(std::atoi(argv[2])), ggml_type(std::atoi(argv[3])), device);
        require(cuDevicePrimaryCtxRelease(raw) == CUDA_SUCCESS, "primary release"); return 0;
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
