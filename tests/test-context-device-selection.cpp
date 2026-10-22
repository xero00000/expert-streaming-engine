#include "llama.h"
#include <random>
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "ggml-cuda.h"
#include "ggml-backend.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool result, const char * message);

static void check_inventory(llama_context * ctx, bool mapped) {
    uint32_t count = 0;
    check(llama_resource_get_memory_pools(ctx, nullptr, 0, &count), "inventory count");
    std::vector<llama_resource_memory_pool> pools(count);
    check(llama_resource_get_memory_pools(ctx, pools.data(), count, &count), "inventory rows");
    uint64_t weights = 0, kv = 0, graph = 0, output = 0, mapped_bytes = 0;
    for (const auto & pool : pools) {
        check(pool.backend_id == -1 || (pool.backend_id >= 0 &&
              pool.backend_id < ggml_backend_sched_get_n_backends(ctx->sched)), "inventory backend location");
        int32_t device_id = -99;
        check(llama_resource_backend_device_id(ctx, pool.backend_id, &device_id), "inventory device mapping");
        const int32_t expected_device = pool.backend_id == -1 ||
            ggml_backend_is_cpu(ggml_backend_sched_get_backend(ctx->sched, pool.backend_id)) ? -1 :
            ggml_backend_cuda_get_device(ggml_backend_sched_get_backend(ctx->sched, pool.backend_id));
        check(device_id == expected_device, "inventory physical CUDA identity");
        if (pool.category == LLAMA_RESOURCE_MEMORY_WEIGHTS) weights += pool.bytes;
        if (pool.category == LLAMA_RESOURCE_MEMORY_MAPPED_WEIGHTS) mapped_bytes += pool.bytes;
        if (pool.category == LLAMA_RESOURCE_MEMORY_KV) kv += pool.bytes;
        if (pool.category == LLAMA_RESOURCE_MEMORY_GRAPH) graph += pool.bytes;
        if (pool.category == LLAMA_RESOURCE_MEMORY_OUTPUT) output += pool.bytes;
    }
    uint64_t expected_weights = 0, expected_graph = 0;
    for (auto * buffer : ctx->model.bufs) expected_weights += ggml_backend_buffer_get_size(buffer);
    for (auto * backend : ctx->backends)
        expected_graph += ggml_backend_sched_get_buffer_size(ctx->sched, backend);
    check(weights + mapped_bytes == expected_weights, "inventory weights differ from live buffers");
    check(mapped ? mapped_bytes > 0 : mapped_bytes == 0, "mapped span classification");
    check(kv == ctx->kv_self.total_size(), "inventory KV differs from live buffers");
    check(graph == expected_graph, "inventory scheduler arena capacity");
    check(output == ggml_backend_buffer_get_size(ctx->buf_output), "inventory output capacity");
    std::vector<llama_resource_memory_pool> second(count);
    check(llama_resource_get_memory_pools(ctx, second.data(), count, &count), "inventory reread");
    check(pools.size() == second.size(), "inventory reread count changed");
    for (size_t i = 0; i < pools.size(); ++i)
        check(pools[i].backend_id == second[i].backend_id && pools[i].category == second[i].category &&
              pools[i].bytes == second[i].bytes, "read-only inventory changed allocations");
    std::set<int32_t> expected_device_ids;
    for (auto * backend : ctx->backends)
        if (ggml_backend_is_cuda(backend)) expected_device_ids.insert(ggml_backend_cuda_get_device(backend));
    const uint32_t device_count = llama_resource_device_count(ctx);
    check(device_count == expected_device_ids.size(), "resource snapshot accelerator count");
    llama_resource_snapshot snapshot = {};
    std::vector<llama_resource_device_snapshot> devices(device_count);
    check(llama_resource_get_snapshot(ctx, &snapshot, devices.data(), device_count), "resource snapshot");
    std::set<int32_t> actual_device_ids;
    for (const auto & device : devices) actual_device_ids.insert(device.device_id);
    check(actual_device_ids == expected_device_ids && actual_device_ids.size() == devices.size(),
          "resource snapshot device IDs differ from selected CUDA devices");
    int32_t sentinel = 123;
    check(!llama_resource_backend_device_id(ctx, -2, &sentinel) && sentinel == 123, "unknown inventory mapping");
    check(!llama_resource_backend_device_id(ctx, ggml_backend_sched_get_n_backends(ctx->sched), &sentinel) &&
          sentinel == 123, "out-of-range inventory mapping");
    check(!llama_resource_backend_device_id(ctx, -1, nullptr), "null inventory mapping output");
    std::printf("memory_inventory: pools=%u weights=%llu mapped_span=%llu kv=%llu graph=%llu output=%llu\n",
        count, (unsigned long long) weights, (unsigned long long) mapped_bytes,
        (unsigned long long) kv, (unsigned long long) graph, (unsigned long long) output);
}

static void check(bool result, const char * message) {
    if (!result) throw std::runtime_error(message);
}

static void fixture(const std::filesystem::path & path) {
    auto * tensors = ggml_init({64 * ggml_tensor_overhead(), nullptr, true});
    check(tensors != nullptr, "fixture tensors");
    auto * meta = gguf_init_empty();
    gguf_set_val_str(meta, "general.architecture", "llama");
    gguf_set_val_str(meta, "tokenizer.ggml.model", "none");
    for (const auto & item : std::vector<std::pair<const char *, uint32_t>>{
        {"llama.block_count", 1}, {"llama.vocab_size", 32}, {"llama.context_length", 32},
        {"llama.embedding_length", 32}, {"llama.feed_forward_length", 64},
        {"llama.attention.head_count", 4}, {"llama.attention.head_count_kv", 4},
        {"llama.rope.dimension_count", 8}}) gguf_set_val_u32(meta, item.first, item.second);
    gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-6f);
    std::vector<std::vector<float>> payloads;
    payloads.reserve(16);
    auto tensor = [&](const char * name, int64_t width, int64_t rows = 1) {
        auto * t = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, width, rows);
        ggml_set_name(t, name);
        payloads.emplace_back(width * rows, 0.f);
        t->data = payloads.back().data();
        gguf_add_tensor(meta, t);
    };
    tensor("token_embd.weight", 32, 32); tensor("output.weight", 32, 32);
    tensor("output_norm.weight", 32); tensor("blk.0.attn_norm.weight", 32);
    tensor("blk.0.attn_q.weight", 32, 32); tensor("blk.0.attn_k.weight", 32, 32);
    tensor("blk.0.attn_v.weight", 32, 32); tensor("blk.0.attn_output.weight", 32, 32);
    tensor("blk.0.ffn_norm.weight", 32); tensor("blk.0.ffn_gate.weight", 32, 64);
    tensor("blk.0.ffn_up.weight", 32, 64); tensor("blk.0.ffn_down.weight", 64, 32);
    gguf_write_to_file(meta, path.string().c_str(), false);
    gguf_free(meta); ggml_free(tensors);
}

void check_native_device_plans(const std::filesystem::path & path, int count);

void check_native_primary_contexts(const std::filesystem::path & path, int count);

int main(int argc, char ** argv) {
    const auto path = std::filesystem::temp_directory_path() /
        ("ese-device-selection-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    try {
        const int count = ggml_backend_cuda_get_device_count();
        check(count > 0, "CUDA device required");
        fixture(path);
        if (argc > 1 && std::string(argv[1]) == "--primary-contexts") {
            check_native_primary_contexts(path, count);
            std::filesystem::remove(path);
            return 0;
        }
        const std::string first = std::string(GGML_CUDA_NAME) + "0";
        const std::string last = std::string(GGML_CUDA_NAME) + std::to_string(count - 1);
        for (int mode = 0; mode < 5; ++mode) {
            const bool cpu = mode == 2 || mode == 4;
            const std::string devices = cpu ? "none" : mode == 3 ? last :
                                        mode == 1 && count > 1 ? first + "," + last : first;
            auto mp = llama_model_default_params();
            mp.devices = devices.c_str(); mp.n_gpu_layers = cpu ? 0 : 99;
            mp.split_mode = mode == 0 || mode == 3 ? LLAMA_SPLIT_MODE_NONE : LLAMA_SPLIT_MODE_LAYER;
            mp.use_mmap = mode == 4;
            auto * model = llama_model_load_from_file(path.string().c_str(), mp);
            check(model != nullptr, "fixture model load");
            auto cp = llama_context_default_params();
            cp.n_ctx = 512; cp.n_batch = 4; cp.n_ubatch = 4; cp.n_seq_max = 1;
            cp.flash_attn = true;
            cp.n_threads = 1; cp.n_threads_batch = 1;
            auto * ctx = llama_init_from_model(model, cp);
            if (!ctx) { llama_free_model(model); check(false, "selected device context initialization"); }
            uint64_t graph_metadata_bytes = 0;
            check(llama_model_graph_metadata_bytes(model, cp, &graph_metadata_bytes) &&
                    graph_metadata_bytes == ctx->buf_compute_meta.capacity(), "native graph metadata forecast differs from allocation");
            std::set<std::string> actual;
            for (auto * backend : ctx->backends) {
                if (ggml_backend_is_cuda(backend)) actual.insert(ggml_backend_name(backend));
            }
            std::set<std::string> expected;
            if (!cpu) { expected.insert(mode == 3 ? last : first); if (mode == 1 && count > 1) expected.insert(last); }
            const bool exact = actual == expected;
            std::set<std::string> budget_devices;
            for(size_t i=0;i<ctx->compute_budgets.size();++i) if(ctx->compute_budgets[i])
                budget_devices.insert(std::string(GGML_CUDA_NAME)+std::to_string(i));
            check(budget_devices==expected,"shared budgets initialized an unselected device");
            check_inventory(ctx, mode == 4);
            llama_token token = 1;
            auto batch = llama_batch_get_one(&token, 1, 0, 0);
            check(llama_decode(ctx, batch) == 0, "fixture decode");
            check_inventory(ctx, mode == 4);
            llama_kv_preparation_memory expected_memory = {};
            check(llama_kv_cache_resize_memory(ctx, 256, &expected_memory, nullptr, 0), "resize sizing query");
            std::vector<llama_kv_preparation_device> projected(expected_memory.device_count);
            check(llama_kv_cache_resize_memory(ctx, 256, &expected_memory, projected.data(), projected.size()), "resize sizing rows");
            auto * transaction = llama_kv_cache_prepare_resize(ctx, 256);
            check(transaction != nullptr, "selected-device resize preparation");
            llama_kv_preparation_memory actual_memory = {};
            check(llama_kv_cache_transaction_memory(transaction, &actual_memory, nullptr, 0), "prepared memory query");
            std::vector<llama_kv_preparation_device> actual_devices(actual_memory.device_count);
            check(llama_kv_cache_transaction_memory(transaction, &actual_memory, actual_devices.data(), actual_devices.size()),
                  "prepared memory rows");
            check(expected_memory.host_metadata_bytes == actual_memory.host_metadata_bytes &&
                  expected_memory.host_staging_bytes == actual_memory.host_staging_bytes && projected.size() == actual_devices.size(),
                  "resize memory forecast differs from actual");
            for (size_t i = 0; i < projected.size(); ++i)
                check(projected[i].device_id == actual_devices[i].device_id && projected[i].kv_bytes == actual_devices[i].kv_bytes,
                      "resize backend placement forecast differs from actual");
            llama_kv_cache_transaction_free(transaction);
            check_inventory(ctx, mode == 4);
            std::printf("resize_memory: mode=%d devices=%u metadata=%llu staging=%llu exact\n", mode,
                expected_memory.device_count, (unsigned long long)expected_memory.host_metadata_bytes,
                (unsigned long long)expected_memory.host_staging_bytes);
            llama_free(ctx); llama_free_model(model);
            check(exact, "context contains an unselected CUDA backend");
            std::printf("selected_context: mode=%d devices=%s backends=%zu\n", mode, devices.c_str(), actual.size());
        }
        check_native_device_plans(path, count);
        std::filesystem::remove(path);
        std::puts("PASS: single GPU, noncontiguous selection, CPU-only context, main-GPU ID translation");
        return 0;
    } catch (const std::exception & error) {
        std::error_code ignored; std::filesystem::remove(path, ignored);
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
