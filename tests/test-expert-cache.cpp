#include "llama-expert-cache.h"
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-ese.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
#endif

using namespace llama_expert_cache;

namespace {

#define REQUIRE(condition) do { \
    if (!(condition)) throw std::runtime_error(std::string("requirement failed: ") + #condition); \
} while (false)

template<class Exception, class Function>
void require_throws(Function && function) {
    bool threw = false;
    try {
        function();
    } catch (const Exception &) {
        threw = true;
    }
    REQUIRE(threw);
}

class memory_source final : public source {
public:
    memory_source(uint32_t id, source_identity identity, read_backend backend, std::vector<uint8_t> bytes) :
        id_(id), identity_(identity), backend_(backend), bytes_(std::move(bytes)) {
    }

    uint32_t id() const override { return id_; }
    source_identity identity() const override { return identity_; }
    uint64_t size() const override { return bytes_.size(); }
    read_backend backend() const override { return backend_; }

    void read(uint64_t offset, void * destination, size_t length) const override {
        if (offset > bytes_.size() || length > bytes_.size() - size_t(offset)) {
            throw std::out_of_range("test source read outside bounds");
        }
        std::memcpy(destination, bytes_.data() + offset, length);
        ++reads_;
    }

    uint64_t reads() const { return reads_; }

private:
    uint32_t id_;
    source_identity identity_;
    read_backend backend_;
    std::vector<uint8_t> bytes_;
    mutable uint64_t reads_ = 0;
};

descriptor_spec make_spec(expert_key key, uint32_t source_id, source_identity source_id_value, uint64_t offset) {
    descriptor_spec spec;
    spec.key = key;
    spec.ggml_type = 0;
    spec.rank = 2;
    spec.quant_axis = 0;
    spec.block_elements = 1;
    spec.block_bytes = 1;
    spec.dimensions = {{ 8, 4, 0, 0 }};
    spec.strides = {{ 1, 8, 0, 0 }};
    spec.model_identity = { 0x1111222233334444ULL, 0x5555666677778888ULL };
    spec.extents = {
        { source_id, source_id_value, offset, 13 },
        { source_id, source_id_value, offset + 13, 19 },
    };
    return spec;
}

std::vector<uint8_t> expected_bytes(uint8_t first) {
    std::vector<uint8_t> result(32);
    for (size_t i = 0; i < result.size(); ++i) result[i] = uint8_t(first + i);
    return result;
}

source_identity file_identity(uint32_t source_id, const std::vector<uint8_t> & bytes) {
    auto hash = [](uint64_t value, const void * data, size_t size) {
        const auto * input = static_cast<const uint8_t *>(data);
        for (size_t i = 0; i < size; ++i) {
            value ^= input[i];
            value *= 1099511628211ULL;
        }
        return value;
    };
    const uint64_t file_size = bytes.size();
    const size_t sample_size = std::min<size_t>(64*1024, bytes.size());
    uint64_t high = 1469598103934665603ULL;
    uint64_t low  = 7809847782465536322ULL;
    high = hash(high, &source_id, sizeof(source_id));
    high = hash(high, &file_size, sizeof(file_size));
    high = hash(high, bytes.data(), sample_size);
    low = hash(low, bytes.data() + bytes.size() - sample_size, sample_size);
    low = hash(low, &file_size, sizeof(file_size));
    low = hash(low, &source_id, sizeof(source_id));
    return { high, low };
}

void test_descriptor_validation() {
    const source_identity identity = { 1, 2 };
    const std::map<uint32_t, uint64_t> sizes = {{ 7, 256 }};
    auto valid = descriptor::make(make_spec({ 3, 4, component::up }, 7, identity, 16), sizes);
    REQUIRE(valid->bytes() == 32);
    REQUIRE(valid->key() == expert_key({ 3, 4, component::up }));

    auto bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.extents[1].offset = 20;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 240);
    require_throws<std::out_of_range>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.dimensions[1] = UINT64_MAX;
    require_throws<std::overflow_error>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.extents[1].length = 18;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 99, identity, 16);
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.source_alignment = 32;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });

    bad = make_spec({ 0, 0, component::gate }, 7, identity, 16);
    bad.source_alignment = 3;
    require_throws<std::invalid_argument>([&] { descriptor::make(bad, sizes); });
}

void test_exact_reads_hits_and_backend_telemetry() {
    const source_identity identities[] = {{ 10, 11 }, { 20, 21 }, { 30, 31 }};
    const read_backend backends[] = { read_backend::mmap, read_backend::pread, read_backend::io_uring };
    for (size_t backend = 0; backend < 3; ++backend) {
        std::vector<uint8_t> bytes(128, 0);
        const auto expected = expected_bytes(uint8_t(40 + backend*40));
        std::copy(expected.begin(), expected.end(), bytes.begin() + 32);
        auto storage = std::make_shared<memory_source>(uint32_t(backend + 1), identities[backend], backends[backend], bytes);
        auto desc = descriptor::make(
                make_spec({ 1, uint32_t(backend), component::down }, storage->id(), identities[backend], 32),
                {{ storage->id(), storage->size() }});
        ram_cache cache({ 64, 32 }, { storage });

        {
            auto lease = cache.acquire(desc);
            REQUIRE(lease.size() == expected.size());
            REQUIRE(reinterpret_cast<uintptr_t>(lease.data()) % 64 == 0);
            REQUIRE(std::equal(expected.begin(), expected.end(), lease.data()));
        }
        {
            auto lease = cache.acquire(desc);
            REQUIRE(std::equal(expected.begin(), expected.end(), lease.data()));
        }
        const auto stats = cache.stats();
        REQUIRE(stats.hits == 1);
        REQUIRE(stats.misses == 1);
        REQUIRE(stats.resident_bytes == 32);
        REQUIRE(stats.peak_resident_bytes == 32);
        REQUIRE(stats.bytes_read[backend] == 32);
        REQUIRE(storage->reads() == 2); // the descriptor has two shard extents
    }
}

void test_deterministic_eviction_and_lease_lifetime() {
    const source_identity identity = { 0xaaaa, 0xbbbb };
    std::vector<uint8_t> bytes(256);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(i);
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::pread, bytes);
    const std::map<uint32_t, uint64_t> sizes = {{ 1, storage->size() }};
    auto a = descriptor::make(make_spec({ 0, 0, component::gate }, 1, identity, 0), sizes);
    auto b = descriptor::make(make_spec({ 0, 1, component::gate }, 1, identity, 32), sizes);
    auto c = descriptor::make(make_spec({ 0, 2, component::gate }, 1, identity, 64), sizes);
    auto d = descriptor::make(make_spec({ 0, 3, component::gate }, 1, identity, 96), sizes);
    // Two 32-byte payloads require 96 bytes in the fixed arena because every
    // independently leased component begins at a 64-byte boundary.
    ram_cache cache({ 96, 32 }, { storage });

    auto lease_a = cache.acquire(a);
    {
        auto lease_b = cache.acquire(b);
        REQUIRE(cache.stats().resident_bytes == 64);
    }
    auto lease_c = cache.acquire(c); // B is the only unpinned candidate.
    REQUIRE(cache.stats().resident_bytes == 64);
    REQUIRE(cache.stats().evictions == 1);
    REQUIRE(lease_a.data()[0] == 0);
    REQUIRE(lease_c.data()[0] == 64);

    const auto before_failure = cache.stats();
    require_throws<capacity_error>([&] { cache.acquire(d); });
    const auto after_failure = cache.stats();
    REQUIRE(after_failure.resident_bytes == before_failure.resident_bytes);
    REQUIRE(after_failure.evictions == before_failure.evictions);
    REQUIRE(after_failure.pinned_capacity_failures == before_failure.pinned_capacity_failures + 1);
    REQUIRE(!cache.erase(a->key()));
    require_throws<capacity_error>([&] { cache.clear(); });

    lease_c = {};
    auto lease_d = cache.acquire(d);
    REQUIRE(lease_d.data()[0] == 96);
    REQUIRE(cache.stats().resident_bytes == 64);
    REQUIRE(cache.stats().evictions == 2);
}

void test_capacity_and_identity_fail_closed() {
    const source_identity identity = { 9, 8 };
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::mmap, std::vector<uint8_t>(128, 7));
    auto desc = descriptor::make(make_spec({ 2, 2, component::up }, 1, identity, 0), {{ 1, 128 }});

    require_throws<std::invalid_argument>([&] { ram_cache({ 0, 0 }, { storage }); });
    require_throws<std::invalid_argument>([&] { ram_cache({ 64, 65 }, { storage }); });
    ram_cache too_small({ 16, 16 }, { storage });
    require_throws<capacity_error>([&] { too_small.acquire(desc); });
    REQUIRE(too_small.stats().resident_bytes == 0);

    auto wrong_storage = std::make_shared<memory_source>(1, source_identity{ 7, 7 }, read_backend::mmap, std::vector<uint8_t>(128, 7));
    ram_cache wrong_identity({ 64, 32 }, { wrong_storage });
    require_throws<std::invalid_argument>([&] { wrong_identity.acquire(desc); });
    REQUIRE(wrong_identity.stats().resident_bytes == 0);

    ram_cache canonical_descriptor({ 64, 32 }, { storage });
    auto original = canonical_descriptor.acquire(desc);
    REQUIRE(original.data()[0] == 7);
    original = {};
    canonical_descriptor.clear();
    auto alternate_spec = make_spec(desc->key(), 1, identity, 32);
    auto alternate = descriptor::make(std::move(alternate_spec), {{ 1, 128 }});
    require_throws<std::invalid_argument>([&] { canonical_descriptor.acquire(alternate); });
    REQUIRE(canonical_descriptor.stats().resident_bytes == 0);
}

void test_file_storage_backends() {
#if defined(__linux__)
    char path[] = "/tmp/ese-expert-cache-XXXXXX";
    const int fd = ::mkstemp(path);
    REQUIRE(fd >= 0);
    struct cleanup {
        const char * path;
        ~cleanup() { ::unlink(path); }
    } remove_file{path};

    std::vector<uint8_t> bytes(128);
    for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t(255 - i);
    size_t written = 0;
    while (written < bytes.size()) {
        const ssize_t count = ::write(fd, bytes.data() + written, bytes.size() - written);
        REQUIRE(count > 0);
        written += size_t(count);
    }
    REQUIRE(::close(fd) == 0);
    require_throws<std::runtime_error>([&] {
        file_source changed(99, { 1, 2 }, path, read_backend::pread);
    });

    const read_backend backends[] = { read_backend::mmap, read_backend::pread, read_backend::io_uring };
    for (size_t i = 0; i < 3; ++i) {
        try {
            const uint32_t source_id = uint32_t(i + 1);
            REQUIRE(identify_file_source(path, source_id) == file_identity(source_id, bytes));
            auto storage = std::make_shared<file_source>(
                    source_id, file_identity(source_id, bytes), path, backends[i]);
            std::array<uint8_t, 32> output = {};
            storage->read(32, output.data(), output.size());
            REQUIRE(std::equal(output.begin(), output.end(), bytes.begin() + 32));
            require_throws<std::out_of_range>([&] { storage->read(120, output.data(), output.size()); });
            require_throws<std::invalid_argument>([&] { storage->read(0, nullptr, 1); });
        } catch (const std::runtime_error & error) {
            // Sandboxes commonly deny io_uring_setup. The backend must report
            // that denial explicitly; it must never relabel a pread fallback.
            REQUIRE(backends[i] == read_backend::io_uring);
            REQUIRE(std::string(error.what()).find("io_uring") != std::string::npos);
            std::cout << "SKIP: native io_uring unavailable: " << error.what() << '\n';
        }
    }
#endif
}

void test_concurrent_single_fill() {
    const source_identity identity = { 77, 88 };
    auto storage = std::make_shared<memory_source>(1, identity, read_backend::pread, std::vector<uint8_t>(128, 5));
    auto desc = descriptor::make(make_spec({ 4, 5, component::down }, 1, identity, 0), {{ 1, 128 }});
    ram_cache cache({ 64, 32 }, { storage });
    std::array<std::thread, 8> threads;
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    for (auto & thread : threads) {
        thread = std::thread([&] {
            ready.fetch_add(1);
            while (!go.load()) std::this_thread::yield();
            auto lease = cache.acquire(desc);
            REQUIRE(lease.size() == 32 && lease.data()[0] == 5);
        });
    }
    while (ready.load() != int(threads.size())) std::this_thread::yield();
    go.store(true);
    for (auto & thread : threads) thread.join();
    const auto stats = cache.stats();
    REQUIRE(stats.misses == 1);
    REQUIRE(stats.hits == threads.size() - 1);
    REQUIRE(storage->reads() == 2);
    REQUIRE(stats.active_leases == 0);
}

struct alignas(64) kernel_lease_fixture {
    std::array<std::array<float, 32*32>, 3> experts;
    std::atomic<uint64_t> acquires{0};
    std::atomic<uint64_t> releases{0};
};

bool kernel_lease_acquire(void * user_data, int layer, int expert, int component, ggml_expert_lease * lease) {
    auto * fixture = static_cast<kernel_lease_fixture *>(user_data);
    if (layer != 0 || component != 0 || expert < 0 || expert >= int(fixture->experts.size())) return false;
    auto & bytes = fixture->experts[size_t(expert)];
    lease->data = bytes.data();
    lease->size = bytes.size()*sizeof(float);
    lease->handle = new int(expert);
    ++fixture->acquires;
    return true;
}

void kernel_lease_release(void * user_data, void * handle) {
    auto * fixture = static_cast<kernel_lease_fixture *>(user_data);
    delete static_cast<int *>(handle);
    ++fixture->releases;
}

void test_cpu_kernel_exact_parity_and_no_original_fallback() {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct cleanup { ggml_context * ctx; ~cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    constexpr int64_t k = 32;
    constexpr int64_t m = 32;
    constexpr int64_t n_experts = 3;
    constexpr int64_t n_used = 2;
    ggml_tensor * weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, m, n_experts);
    ggml_set_name(weights, "blk.0.ffn_gate_exps.weight");
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_mul_mat_id(ctx, weights, input, ids);

    kernel_lease_fixture fixture;
    for (int expert = 0; expert < n_experts; ++expert) {
        for (int64_t i = 0; i < k*m; ++i) {
            fixture.experts[size_t(expert)][size_t(i)] = float(1 + expert*17 + i);
        }
        std::memcpy((uint8_t *) weights->data + expert*weights->nb[2],
                fixture.experts[size_t(expert)].data(), fixture.experts[size_t(expert)].size()*sizeof(float));
    }
    std::array<float, k*n_used> input_values = {};
    for (size_t i = 0; i < input_values.size(); ++i) input_values[i] = float(int(i % 9) - 4)/3.0f;
    const std::array<int32_t, n_used> route = {{ 2, 0 }};
    std::memcpy(input->data, input_values.data(), sizeof(input_values));
    std::memcpy(ids->data, route.data(), sizeof(route));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);
    auto compute = [&](bool leases) {
        ggml_cplan plan = ggml_graph_plan(graph, 2);
        std::vector<uint8_t> work(plan.work_size);
        plan.work_data = work.empty() ? nullptr : work.data();
        if (leases) {
            plan.expert_lease_acquire = kernel_lease_acquire;
            plan.expert_lease_release = kernel_lease_release;
            plan.expert_lease_user_data = &fixture;
            plan.expert_lease_required = true;
        }
        REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);
        std::vector<float> result(size_t(ggml_nelements(output)));
        std::memcpy(result.data(), output->data, result.size()*sizeof(float));
        return result;
    };

    const auto baseline = compute(false);
    std::memset((uint8_t *) weights->data + route[0]*weights->nb[2], 0, weights->nb[2]);
    std::memset((uint8_t *) weights->data + route[1]*weights->nb[2], 0, weights->nb[2]);
    const auto leased = compute(true);
    REQUIRE(leased == baseline);
    REQUIRE(fixture.acquires > 0 && fixture.acquires == fixture.releases);

    const auto corrupted = compute(false);
    REQUIRE(corrupted != baseline);
    REQUIRE(*(const int32_t *) ids->data == route[0]);
    REQUIRE(*((const int32_t *) ids->data + 1) == route[1]);
}

void test_route_partition_is_complementary() {
    ggml_init_params params = { 1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct cleanup { ggml_context * ctx; ~cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 4, 2);
    const std::array<int32_t, 8> route = {{ 7, 3, 5, 1, 2, -1, 6, 4 }};
    std::memcpy(ids->data, route.data(), sizeof(route));
    ggml_tensor * gpu_ids = ggml_ese_route_partition(ctx, ids, 0, 2);
    ggml_tensor * cpu_ids = ggml_ese_route_partition(ctx, ids, 2, 2);
    REQUIRE(ggml_ese_route_get_role(gpu_ids) == GGML_ESE_ROUTE_GPU);
    REQUIRE(ggml_ese_route_get_role(cpu_ids) == GGML_ESE_ROUTE_CPU);
    ggml_tensor * activations = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 4, 2);
    ggml_tensor * biases = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2, 8);
    std::fill_n(static_cast<float *>(activations->data), ggml_nelements(activations), 0.0f);
    for (int expert = 0; expert < 8; ++expert) {
        static_cast<float *>(biases->data)[2*expert + 0] = float(expert + 1);
        static_cast<float *>(biases->data)[2*expert + 1] = float(100 + expert);
    }
    ggml_tensor * biased_cpu = ggml_add_id(ctx, activations, biases, cpu_ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 8, false);
    ggml_build_forward_expand(graph, gpu_ids);
    ggml_build_forward_expand(graph, cpu_ids);
    ggml_build_forward_expand(graph, biased_cpu);
    ggml_cplan plan = ggml_graph_plan(graph, 2);
    std::vector<uint8_t> work(plan.work_size);
    plan.work_data = work.empty() ? nullptr : work.data();
    REQUIRE(ggml_graph_compute(graph, &plan) == GGML_STATUS_SUCCESS);

    const auto * gpu = static_cast<const int32_t *>(gpu_ids->data);
    const auto * cpu = static_cast<const int32_t *>(cpu_ids->data);
    for (size_t i = 0; i < route.size(); ++i) {
        const bool gpu_position = i%4 < 2;
        REQUIRE(gpu[i] == (gpu_position ? route[i] : -1));
        REQUIRE(cpu[i] == (gpu_position ? -1 : route[i]));
        if (route[i] >= 0) {
            REQUIRE((gpu[i] >= 0) != (cpu[i] >= 0));
        }
        const float * biased = static_cast<const float *>(biased_cpu->data) + 2*i;
        if (gpu_position || route[i] < 0) {
            REQUIRE(biased[0] == 0.0f && biased[1] == 0.0f);
        } else {
            REQUIRE(biased[0] == float(route[i] + 1));
            REQUIRE(biased[1] == float(100 + route[i]));
        }
    }
}

std::vector<float> compute_cuda_mul_mat_id(
        ggml_backend_t backend,
        const std::vector<ggml_fp16_t> & weights,
        int64_t stored_experts,
        int64_t logical_experts,
        const std::array<int32_t, 2> & route,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 768;
    constexpr int64_t m = 384;
    constexpr int64_t n_used = 2;

    ggml_init_params params = {
        ggml_tensor_overhead()*16 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * tensor_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_mul_mat_id(ctx, tensor_weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    struct buffer_cleanup {
        ggml_backend_buffer_t buffer;
        ~buffer_cleanup() { ggml_backend_buffer_free(buffer); }
    } free_buffer{buffer};

    REQUIRE(weights.size() == size_t(k*m*stored_experts));
    REQUIRE(input_values.size() == size_t(k*n_used));
    ggml_backend_tensor_set(tensor_weights, weights.data(), 0, weights.size()*sizeof(weights[0]));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(ids, route.data(), 0, sizeof(route));

    // The scheduler keeps the model's logical expert count while redirecting
    // the tensor data pointer to a smaller slot buffer.  Reproduce that exact
    // layout, including the original fourth-dimension stride.
    tensor_weights->ne[2] = logical_experts;
    tensor_weights->nb[3] = tensor_weights->nb[2]*size_t(logical_experts);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

std::vector<float> compute_hybrid_mul_mat_id(
        ggml_backend_t cuda_backend,
        const std::vector<ggml_fp16_t> & weights,
        int32_t gpu_expert,
        const std::array<int32_t, 2> & cpu_route,
        const std::array<int32_t, 2> & gpu_route,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 768;
    constexpr int64_t m = 384;
    constexpr int64_t n_experts = 8;
    constexpr int64_t n_used = 2;
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    REQUIRE(cpu_backend != nullptr);
    struct cpu_cleanup { ggml_backend_t value; ~cpu_cleanup() { ggml_backend_free(value); } } free_cpu{cpu_backend};

    ggml_init_params params = {
        ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(32, false), nullptr, true
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * value; ~context_cleanup() { ggml_free(value); } } free_ctx{ctx};

    ggml_tensor * cpu_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, n_experts);
    ggml_tensor * gpu_weights = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, 1);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * cpu_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * gpu_ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_set_input(input);
    ggml_set_input(cpu_ids);
    ggml_set_input(gpu_ids);
    ggml_tensor * cpu_output = ggml_mul_mat_id(ctx, cpu_weights, input, cpu_ids);
    ggml_tensor * gpu_output = ggml_mul_mat_id(ctx, gpu_weights, input, gpu_ids);
    ggml_tensor * branches[] = {gpu_output, cpu_output};
    ggml_tensor * output = ggml_ese_reduce_to(ctx, branches, 2, GGML_OP_ADD, 0);
    ggml_set_output(output);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_t backends[] = {cuda_backend, cpu_backend};
    ggml_backend_sched_t scheduler = ggml_backend_sched_new(
            backends, nullptr, 2, 64, true);
    REQUIRE(scheduler != nullptr);
    struct scheduler_cleanup {
        ggml_backend_sched_t value;
        ~scheduler_cleanup() { ggml_backend_sched_free(value); }
    } free_scheduler{scheduler};
    ggml_backend_sched_set_split_mode_graph(scheduler, true, true);
    ggml_backend_sched_set_tensor_backend(scheduler, cpu_weights, cpu_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, gpu_weights, cuda_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, cpu_output, cpu_backend);
    ggml_backend_sched_set_tensor_backend(scheduler, gpu_output, cuda_backend);
    REQUIRE(ggml_backend_sched_alloc_graph(scheduler, graph));

    REQUIRE(weights.size() == size_t(k*m*n_experts));
    REQUIRE(gpu_expert >= 0 && gpu_expert < n_experts);
    REQUIRE(input_values.size() == size_t(k*n_used));
    ggml_backend_tensor_set(cpu_weights, weights.data(), 0, weights.size()*sizeof(weights[0]));
    ggml_backend_tensor_set(gpu_weights,
            weights.data() + size_t(gpu_expert)*size_t(k*m), 0,
            size_t(k*m)*sizeof(weights[0]));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(cpu_ids, cpu_route.data(), 0, sizeof(cpu_route));
    ggml_backend_tensor_set(gpu_ids, gpu_route.data(), 0, sizeof(gpu_route));
    REQUIRE(ggml_backend_sched_graph_compute(scheduler, graph) == GGML_STATUS_SUCCESS);
    ggml_backend_sched_synchronize(scheduler);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

std::vector<float> compute_cuda_fused_moe(
        ggml_backend_t backend,
        const std::vector<ggml_fp16_t> & up_weights,
        const std::vector<ggml_fp16_t> & gate_weights,
        int64_t stored_experts,
        int64_t logical_experts,
        const std::array<int32_t, 2> & route,
        const std::vector<float> & input_values) {
    constexpr int64_t k = 384;
    constexpr int64_t m = 768;
    constexpr int64_t n_used = 2;

    ggml_init_params params = {
        ggml_tensor_overhead()*20 + ggml_graph_overhead_custom(16, false),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    REQUIRE(ctx != nullptr);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * up = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * gate = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_moe_up_gate(ctx, up, gate, input, ids, GGML_UNARY_OP_SILU);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    REQUIRE(buffer != nullptr);
    struct buffer_cleanup {
        ggml_backend_buffer_t buffer;
        ~buffer_cleanup() { ggml_backend_buffer_free(buffer); }
    } free_buffer{buffer};

    REQUIRE(up_weights.size() == size_t(k*m*stored_experts));
    REQUIRE(gate_weights.size() == up_weights.size());
    ggml_backend_tensor_set(up, up_weights.data(), 0, up_weights.size()*sizeof(up_weights[0]));
    ggml_backend_tensor_set(gate, gate_weights.data(), 0, gate_weights.size()*sizeof(gate_weights[0]));
    ggml_backend_tensor_set(input, input_values.data(), 0, input_values.size()*sizeof(input_values[0]));
    ggml_backend_tensor_set(ids, route.data(), 0, sizeof(route));

    up->ne[2] = gate->ne[2] = logical_experts;
    up->nb[3] = up->nb[2]*size_t(logical_experts);
    gate->nb[3] = gate->nb[2]*size_t(logical_experts);

    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> result(size_t(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size()*sizeof(result[0]));
    return result;
}

void test_cuda_compact_remap_exact_parity() {
    ggml_backend_t backend = nullptr;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        if (std::strncmp(ggml_backend_reg_get_name(i), "CUDA", 4) == 0) {
            backend = ggml_backend_reg_init_backend(i, nullptr);
            if (backend) break;
        }
    }
    if (!backend) return;
    struct backend_cleanup { ggml_backend_t backend; ~backend_cleanup() { ggml_backend_free(backend); } } free_backend{backend};

    constexpr int64_t k = 768;
    constexpr int64_t m = 384;
    constexpr int64_t n_experts = 8;
    constexpr int64_t n_used = 2;
    std::vector<ggml_fp16_t> full_weights(size_t(k*m*n_experts));
    for (size_t i = 0; i < full_weights.size(); ++i) {
        full_weights[i] = ggml_fp32_to_fp16(float(int((i*17 + i/31) % 257) - 128)/64.0f);
    }
    std::vector<float> input_values(size_t(k*n_used));
    for (size_t i = 0; i < input_values.size(); ++i) {
        input_values[i] = float(int((i*13) % 97) - 48)/24.0f;
    }

    const std::array<int32_t, n_used> route = {{ 6, 1 }};
    const std::array<int32_t, n_used> remapped = {{ 0, 1 }};
    std::vector<ggml_fp16_t> compact_weights(size_t(k*m*n_used));
    for (int slot = 0; slot < n_used; ++slot) {
        std::copy_n(full_weights.data() + size_t(route[slot])*size_t(k*m), size_t(k*m),
                compact_weights.data() + size_t(slot)*size_t(k*m));
    }

    const auto full = compute_cuda_mul_mat_id(
            backend, full_weights, n_experts, n_experts, route, input_values);
    const auto compact = compute_cuda_mul_mat_id(
            backend, compact_weights, n_used, n_experts, remapped, input_values);
    REQUIRE(compact == full);

    // Negative route sentinels turn unassigned route positions into exact
    // zeroes. This permits independent CPU and GPU branches to execute on the
    // same activation and sum back to the unsplit routed result.
    const auto hybrid = compute_hybrid_mul_mat_id(
            backend, full_weights, route[1], {{route[0], -1}}, {{-1, 0}}, input_values);
    double squared_hybrid_error = 0;
    double squared_full = 0;
    for (size_t i = 0; i < full.size(); ++i) {
        REQUIRE(std::isfinite(hybrid[i]));
        const double error = double(hybrid[i]) - full[i];
        squared_hybrid_error += error*error;
        squared_full += double(full[i])*full[i];
        if (i >= size_t(m)) {
            // The GPU-owned route position is the same CUDA kernel plus an
            // exact CPU zero, so a masked branch must not perturb it at all.
            REQUIRE(hybrid[i] == full[i]);
        }
    }
    const double hybrid_nrmse = std::sqrt(squared_hybrid_error/squared_full);
    REQUIRE(hybrid_nrmse <= 5.0e-3);

    constexpr int64_t fused_k = 384;
    constexpr int64_t fused_m = 768;
    std::vector<ggml_fp16_t> full_fused_up(size_t(fused_k*fused_m*n_experts));
    std::vector<ggml_fp16_t> full_fused_gate(size_t(fused_k*fused_m*n_experts));
    for (size_t i = 0; i < full_fused_up.size(); ++i) {
        full_fused_up[i] = ggml_fp32_to_fp16(float(int((i*17 + i/31) % 257) - 128)/64.0f);
        full_fused_gate[i] = ggml_fp32_to_fp16(float(int((i*29 + i/19) % 251) - 125)/64.0f);
    }
    std::vector<ggml_fp16_t> compact_fused_up(size_t(fused_k*fused_m*n_used));
    std::vector<ggml_fp16_t> compact_fused_gate(size_t(fused_k*fused_m*n_used));
    for (int slot = 0; slot < n_used; ++slot) {
        std::copy_n(full_fused_up.data() + size_t(route[slot])*size_t(fused_k*fused_m), size_t(fused_k*fused_m),
                compact_fused_up.data() + size_t(slot)*size_t(fused_k*fused_m));
        std::copy_n(full_fused_gate.data() + size_t(route[slot])*size_t(fused_k*fused_m), size_t(fused_k*fused_m),
                compact_fused_gate.data() + size_t(slot)*size_t(fused_k*fused_m));
    }
    std::vector<float> fused_input(size_t(fused_k*n_used));
    for (size_t i = 0; i < fused_input.size(); ++i) {
        fused_input[i] = float(int((i*13) % 97) - 48)/24.0f;
    }
    const auto full_fused = compute_cuda_fused_moe(
            backend, full_fused_up, full_fused_gate, n_experts, n_experts, route, fused_input);
    const auto compact_fused = compute_cuda_fused_moe(
            backend, compact_fused_up, compact_fused_gate, n_used, n_experts, remapped, fused_input);
    REQUIRE(compact_fused == full_fused);
}

} // namespace

int main() {
    try {
        test_descriptor_validation();
        test_exact_reads_hits_and_backend_telemetry();
        test_deterministic_eviction_and_lease_lifetime();
        test_capacity_and_identity_fail_closed();
        test_file_storage_backends();
        test_concurrent_single_fill();
        test_cpu_kernel_exact_parity_and_no_original_fallback();
        test_route_partition_is_complementary();
        test_cuda_compact_remap_exact_parity();
        std::cout << "PASS: checked bounded expert caches and exact full/compact CUDA parity\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
