#include "server-context.h"
#include "llama.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

bool server_log_json = false;
bool server_verbose = false;

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void run_result_defaults() {
    // Poison the storage so omitted scalar initialization is reproducible even
    // when an ordinary stack allocation happens to contain zero bytes.
    alignas(server_task_result) std::array<unsigned char, sizeof(server_task_result)> storage;
    storage.fill(0xa5);
    auto * original = new (storage.data()) server_task_result;
    const auto defaults = [](const server_task_result & result) {
        check(!result.stop && !result.error && !result.stream && !result.include_usage &&
              !result.truncated && !result.has_new_line, "result booleans are uninitialized");
        check(result.n_decoded == 0 && result.n_prompt_tokens == 0 &&
              result.n_prompt_tokens_cache == 0 && result.n_tokens_cached == 0,
              "result token counters are uninitialized");
        check(result.timings.prompt_n == -1 && result.timings.predicted_n == -1 &&
              result.timings.prompt_ms == 0 && result.timings.prompt_per_token_ms == 0 &&
              result.timings.prompt_per_second == 0 && result.timings.predicted_ms == 0 &&
              result.timings.predicted_per_token_ms == 0 && result.timings.predicted_per_second == 0,
              "result timing defaults are uninitialized");
    };
    try {
        original->id = 1701;
        server_task_result copied(*original);
        defaults(copied);
        server_response responses;
        responses.add_waiting_task_id(copied.id);
        responses.send(copied);
        check(responses.queue_results_legacy.size() == 1, "default result queue copy missing");
        defaults(responses.queue_results_legacy.front());
    } catch (...) {
        original->~server_task_result();
        throw;
    }
    original->~server_task_result();
    std::puts("server result defaults: poisoned storage, direct copy and queued copy exact");
}

static void run_slot_action_errors() {
    server_context server;
    int id = 1702;
    for (auto action : {SERVER_TASK_TYPE_SLOT_SAVE, SERVER_TASK_TYPE_SLOT_RESTORE, SERVER_TASK_TYPE_SLOT_ERASE}) {
        for (auto type : {ERROR_TYPE_INVALID_REQUEST, ERROR_TYPE_NOT_SUPPORTED, ERROR_TYPE_SERVER}) {
            server_task task;
            task.id = id++;
            task.type = action;
            server.queue_results.add_waiting_task_id(task.id);
            server.send_error(task, "slot action failed", type);
            check(server.queue_results.queue_results_legacy.size() == 1 && server.queue_results.queue_results.empty(),
                  "slot action error entered the wrong queue or was duplicated");
            const auto result = server.queue_results.recv(task.id);
            check(result.error && result.id == task.id && result.data == format_error_response("slot action failed", type),
                  "slot action error lost its identity or HTTP error payload");
            server.queue_results.remove_waiting_task_id(task.id);
            check(server.queue_results.queue_results_legacy.empty() && server.queue_results.queue_results.empty(),
                  "slot action error left an unused response");
        }
    }
    server_task completion;
    completion.id = id;
    completion.type = SERVER_TASK_TYPE_COMPLETION;
    server.queue_results.add_waiting_task_id(id);
    server.send_error(completion, "completion failed", ERROR_TYPE_SERVER);
    check(server.queue_results.queue_results_legacy.empty() && server.queue_results.queue_results.size() == 1,
          "completion error changed response queues");
    auto result = server.queue_results.recv_with_timeout({id}, 1);
    check(result && result->is_error() && result->to_json() == format_error_response("completion failed", ERROR_TYPE_SERVER),
          "typed completion error lost its type or payload");
    server.queue_results.remove_waiting_task_id(id);
    check(server.queue_results.queue_results.empty(), "typed error left an unused response");
    std::puts("slot action errors: all three actions and error classes delivered once; typed completion unchanged");
}

// A generated model makes the recovery test independent of downloads and of
// production architecture opt-ins. Three layers allow an abort after KV writes.
struct fixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ese-server-recovery-" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");

    fixture(bool moe = false) {
        auto * tensors = ggml_init({128 * ggml_tensor_overhead(), nullptr, true});
        auto * meta = gguf_init_empty();
        check(tensors && meta, "fixture allocation");
        gguf_set_val_str(meta, "general.architecture", "llama");
        gguf_set_val_str(meta, "tokenizer.ggml.model", "llama");
        for (auto item : std::vector<std::pair<const char *, uint32_t>> {
            {"llama.block_count", 3}, {"llama.vocab_size", 32},
            {"llama.context_length", 128}, {"llama.embedding_length", 32},
            {"llama.feed_forward_length", 64}, {"llama.attention.head_count", 4},
            {"llama.attention.head_count_kv", 4}, {"llama.rope.dimension_count", 8},
            {"tokenizer.ggml.unknown_token_id", 0}, {"tokenizer.ggml.bos_token_id", 1},
            {"tokenizer.ggml.eos_token_id", 2},
        }) gguf_set_val_u32(meta, item.first, item.second);
        if (moe) {
            gguf_set_val_u32(meta, "llama.expert_count", 4);
            gguf_set_val_u32(meta, "llama.expert_used_count", 2);
        }
        gguf_set_val_bool(meta, "tokenizer.ggml.add_bos_token", false);
        gguf_set_val_f32(meta, "llama.attention.layer_norm_rms_epsilon", 1e-6f);
        std::vector<std::string> pieces = {"<unk>", "<s>", "</s>", "▁", "t", "3", "8"};
        for (int i = 7; i < 32; ++i) pieces.push_back("▁t" + std::to_string(i));
        std::vector<const char *> names;
        for (const auto & piece : pieces) names.push_back(piece.c_str());
        gguf_set_arr_str(meta, "tokenizer.ggml.tokens", names.data(), names.size());
        std::vector<int32_t> types(32, 1);
        types[0] = 2;
        types[1] = types[2] = 3;
        gguf_set_arr_data(meta, "tokenizer.ggml.token_type", GGUF_TYPE_INT32, types.data(), types.size());
        std::vector<float> scores(32, 0);
        gguf_set_arr_data(meta, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32, scores.data(), scores.size());
        std::vector<std::vector<float>> payloads;
        payloads.reserve(40);
        auto tensor = [&](const std::string & name, int64_t width, int64_t rows = 1, int64_t experts = 1) {
            auto * value = experts == 1 ? ggml_new_tensor_2d(tensors, GGML_TYPE_F32, width, rows) :
                ggml_new_tensor_3d(tensors, GGML_TYPE_F32, width, rows, experts);
            ggml_set_name(value, name.c_str());
            payloads.emplace_back(width * rows * experts);
            auto & data = payloads.back();
            for (size_t i = 0; i < data.size(); ++i) data[i] =
                name.find("norm") != std::string::npos ? 1.f :
                float(int((i * 17 + payloads.size() * 11) % 101) - 50) * 0.008f;
            value->data = data.data();
            gguf_add_tensor(meta, value);
        };
        tensor("token_embd.weight", 32, 32);
        tensor("output.weight", 32, 32);
        tensor("output_norm.weight", 32);
        for (int i = 0; i < 3; ++i) {
            const std::string p = "blk." + std::to_string(i) + ".";
            tensor(p + "attn_norm.weight", 32);
            tensor(p + "attn_q.weight", 32, 32);
            tensor(p + "attn_k.weight", 32, 32);
            tensor(p + "attn_v.weight", 32, 32);
            tensor(p + "attn_output.weight", 32, 32);
            tensor(p + "ffn_norm.weight", 32);
            if (moe) {
                tensor(p + "ffn_gate_inp.weight", 32, 4);
                tensor(p + "ffn_gate_exps.weight", 32, 64, 4);
                tensor(p + "ffn_up_exps.weight", 32, 64, 4);
                tensor(p + "ffn_down_exps.weight", 64, 32, 4);
            } else {
                tensor(p + "ffn_gate.weight", 32, 64);
                tensor(p + "ffn_up.weight", 32, 64);
                tensor(p + "ffn_down.weight", 64, 32);
            }
        }
        gguf_write_to_file(meta, path.string().c_str(), false);
        gguf_free(meta);
        ggml_free(tensors);
    }

    ~fixture() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};

static std::vector<uint8_t> save(llama_context * ctx) {
    std::vector<uint8_t> bytes(llama_state_get_size(ctx));
    const size_t count = llama_state_get_data(ctx, bytes.data(), bytes.size());
    check(count > 0 && count <= bytes.size(), "snapshot failed");
    bytes.resize(count);
    return bytes;
}

static std::vector<float> decode(llama_context * ctx, llama_tokens tokens, int pos, int seq) {
    check(llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size(), pos, seq)) == 0,
        "teacher/recovery decode failed");
    llama_synchronize(ctx);
    const float * logits = llama_get_logits_ith(ctx, -1);
    check(logits != nullptr, "missing logits");
    std::vector<float> result(logits, logits + 32);
    for (float value : result) check(std::isfinite(value), "nonfinite logits");
    return result;
}

struct abort_control { int polls = 0; int threshold; };

struct route_width_control { int expected = 2; int observed = 0; bool invalid = false; };
static int capture_route_width(ggml_tensor * tensor, bool ask, void * opaque) {
    if (std::strncmp(tensor->name, "ffn_moe_topk-", 13) != 0) return 0;
    if (!ask) {
        auto & control = *static_cast<route_width_control *>(opaque);
        ++control.observed;
        control.invalid |= tensor->ne[0] != control.expected;
    }
    return 1;
}

static void run_moe_first_request() {
    fixture file(true);
    auto mp = llama_model_default_params();
    mp.devices = "";
    mp.n_gpu_layers = 0;
    std::unique_ptr<llama_model, decltype(&llama_free_model)> model(
            llama_model_load_from_file(file.path.string().c_str(), mp), llama_free_model);
    check(bool(model), "MoE fixture load failed");
    for (bool reuse : {false, true}) {
        route_width_control routes;
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = 8;
        cp.n_ubatch = 1;
        cp.n_threads = cp.n_threads_batch = 1;
        cp.graph_reuse = reuse;
        cp.cb_eval = capture_route_width;
        cp.cb_eval_user_data = &routes;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                llama_init_from_model(model.get(), cp), llama_free);
        check(bool(ctx), "MoE fixture context failed");
        const llama_tokens prompt = {1, 3, 8, 14};
        const auto teacher = decode(ctx.get(), prompt, 0, 0);
        check(routes.observed > 0 && !routes.invalid, "first BOS microbatch used warmup expert width");
        for (int trial = 0; trial < 3; ++trial) {
            llama_kv_cache_clear(ctx.get());
            llama_reset_timings(ctx.get());
            check(decode(ctx.get(), prompt, 0, 0) == teacher, "MoE cold replay changed logits");
            check(!routes.invalid, "timing reset enabled warmup expert width");
        }
        llama_kv_cache_clear(ctx.get());
        routes.expected = 4;
        routes.observed = 0;
        llama_set_warmup(ctx.get(), true);
        decode(ctx.get(), {1}, 0, 0);
        check(routes.observed > 0 && !routes.invalid, "explicit warmup did not execute every expert");
        llama_set_warmup(ctx.get(), false);
        llama_kv_cache_clear(ctx.get());
        llama_reset_timings(ctx.get());
        routes.expected = 2;
        check(decode(ctx.get(), prompt, 0, 0) == teacher && !routes.invalid,
                "warmup graph or timing reset contaminated real inference");
    }
    std::puts("MoE first request: BOS microbatches, graph reuse and timing resets preserve top-k and logits");
}

static bool abort_after(void * opaque) {
    auto & control = *static_cast<abort_control *>(opaque);
    return ++control.polls >= control.threshold;
}

static void run(const fixture & file, int threshold, bool system, bool fail_system = false) {
    server_context server;
    auto mp = llama_model_default_params();
    mp.devices = "";
    mp.n_gpu_layers = 0;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.use_mmap = false;
    server.model = llama_model_load_from_file(file.path.string().c_str(), mp);
    check(server.model != nullptr, "model load failed");
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 16;
    cp.n_seq_max = 3;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.flash_attn = false;
    cp.worst_case_tokens = 16;
    server.ctx = llama_init_from_model(server.model, cp);
    check(server.ctx != nullptr, "server context failed");
    std::unique_ptr<llama_context, decltype(&llama_free)> independent(
        llama_init_from_model(server.model, cp), llama_free);
    check(bool(independent), "independent context failed");
    server.params_base.n_parallel = 2;
    server.batch = llama_batch_init(16, 0, 3);
    server.slots.resize(3);
    for (int i = 0; i < 3; ++i) {
        auto & slot = server.slots[i];
        slot.id = i;
        slot.id_task = 100 + i;
        slot.ctx = i == 2 ? independent.get() : server.ctx;
        slot.t_start_generation = ggml_time_us();
    }
    server.slots[0].state = SLOT_STATE_PROCESSING;
    server.slots[2].state = SLOT_STATE_PROCESSING;
    if (system) {
        server.system_prompt = "t3 t8";
        server.system_need_update = true;
        server.system_prompt_update();
        check(!server.system_need_update && !server.system_tokens.empty(), "system prefix decode failed");
    }
    const int prefix = server.system_tokens.size();
    const llama_tokens prompt = {3, 8, 14, 5, 23, 11, 4, 19, 7, 16, 28, 9};
    const auto expected = decode(server.ctx, prompt, prefix, 0);
    // Capacity retries can split a prompt into an output-free first chunk and
    // a final output chunk. Both must preserve the same value-cache contents.
    check(llama_kv_cache_seq_rm(server.ctx, 0, prefix, -1), "teacher prefix removal failed");
    auto first_chunk = llama_batch_get_one(const_cast<llama_token *>(prompt.data()), 8, prefix, 0);
    std::vector<int8_t> no_outputs(8, 0);
    first_chunk.logits = no_outputs.data();
    check(llama_decode(server.ctx, first_chunk) == 0, "output-free prompt chunk failed");
    check(decode(server.ctx, llama_tokens(prompt.begin() + 8, prompt.end()), prefix + 8, 0) == expected,
        "split prompt differs from single-batch teacher");
    decode(server.ctx, {20, 15, 6, 27, 12, 4, 21, 10}, prefix, 1);
    decode(independent.get(), {7, 12, 18, 5}, 0, 2);
    const auto other_before = save(independent.get());
    llama_kv_cache_seq_rm(server.ctx, 0, prefix, -1);
    for (auto & slot : server.slots) {
        slot.cache_tokens.push_back(3);
        slot.n_past = 4;
        slot.sampled = 4;
        slot.drafted = {5, 6};
        slot.i_batch_dft = {0, 1, 2};
        slot.checkpoint_pos = 3;
        slot.do_checkpoint = true;
        slot.server_cached_prompt.tokens.push_back(3);
        slot.server_cached_prompt.checkpoints.emplace_back();
        slot.token_buffer.emplace_back();
        slot.n_buffer = 1;
        slot.rewind_status = true;
    }
    server.prompt_cache = std::make_unique<server_prompt_cache>(server.ctx, 1, 0);
    server.prompt_cache->states.emplace_back();
    server.queue_results.add_waiting_task_id(100);
    server.queue_results.add_waiting_task_id(101);
    server.queue_results.add_waiting_task_id(102);
    common_batch_clear(server.batch);
    for (int i = 0; i < (int) prompt.size(); ++i)
        common_batch_add(server.batch, prompt[i], prefix + i, {0}, i + 1 == (int) prompt.size());
    abort_control control{0, threshold};
    llama_set_abort_callback(server.ctx, abort_after, &control);
    int n_batch = 16;
    if (fail_system) {
        server.system_need_update = true;
        server.update_slots();
    } else {
        server.process_batch_tokens(n_batch);
    }
    llama_set_abort_callback(server.ctx, nullptr, nullptr);
    check(control.polls >= threshold, "actual graph abort callback not reached");
    check(n_batch == 16, "fatal abort retried as KV capacity exhaustion");
    check(server.slots[0].state == SLOT_STATE_IDLE, "failed request remained active");
    check(server.slots[1].state == SLOT_STATE_IDLE && server.slots[1].command == SLOT_COMMAND_NONE,
        "idle slot received a manufactured release");
    check(server.slots[2].state == SLOT_STATE_PROCESSING && server.slots[2].cache_tokens.size() == 1 &&
        server.slots[2].drafted.size() == 2 && save(independent.get()) == other_before,
        "independent context or slot changed");
    check(llama_get_kv_cache_used_cells(server.ctx) == 0, "failed context retained KV cells");
    for (int i = 0; i < 2; ++i) {
        const auto & slot = server.slots[i];
        check(slot.cache_tokens.empty() && slot.n_past == 0 && slot.drafted.empty() &&
            slot.i_batch_dft.empty() && slot.server_cached_prompt.tokens.empty() &&
            slot.server_cached_prompt.checkpoints.empty() && slot.token_buffer.empty() &&
            slot.n_buffer == 0 && !slot.rewind_status && slot.checkpoint_pos == -1 && !slot.do_checkpoint,
            "failed context retained cache/draft/rewind state");
    }
    check(server.prompt_cache->states.empty(), "failed context retained saved prompts");
    check(server.queue_results.queue_results.size() == 1, "wrong requests received errors");
    const auto * error = dynamic_cast<server_task_result_error *>(server.queue_results.queue_results[0].get());
    check(error && error->id == 100 && error->err_msg.find(fail_system ? "system prompt" : "graph was aborted")
        != std::string::npos, "graph abort reported as prompt capacity error");
    check(server.system_need_update == system, "system prefix replay flag lost");
    if (system) {
        server.system_prompt_update();
        check(!server.system_need_update && (int) server.system_tokens.size() == prefix,
            "system prefix was not rebuilt");
    }
    check(decode(server.ctx, prompt, prefix, 0) == expected, "recomputed logits differ from clean teacher");
    std::printf("server recovery: threshold=%d system=%d fail_system=%d exact\n", threshold, system, fail_system);
}

static void run_memory_recheck(const fixture & file) {
    server_context server;
    auto mp = llama_model_default_params();
    std::string devices;
    int32_t device_id = -1;
#ifdef GGML_USE_CUDA
    const int cuda_count = ggml_backend_cuda_get_device_count();
    if (cuda_count > 0) {
        device_id = cuda_count - 1;
        devices = std::string(GGML_CUDA_NAME) + std::to_string(device_id);
    }
#endif
    mp.devices = devices.c_str();
    mp.n_gpu_layers = device_id >= 0 ? 99 : 0;
    mp.split_mode = LLAMA_SPLIT_MODE_NONE;
    mp.use_mmap = false;
    server.model = llama_model_load_from_file(file.path.string().c_str(), mp);
    check(server.model != nullptr, "memory recheck model load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = cp.n_ubatch = 16;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1;
    cp.flash_attn = true;
    server.ctx = llama_init_from_model(server.model, cp);
    check(server.ctx != nullptr, "memory recheck context creation");
    server.batch = llama_batch_init(16, 0, 1);
    server.slots.resize(1);
    server.slots[0].ctx = server.ctx;
    const llama_tokens prompt = {3, 8, 14, 5, 23, 11, 4, 19, 7, 16, 28, 9};
    const auto expected = decode(server.ctx, prompt, 0, 0);
    const auto state = save(server.ctx);
    uint32_t count = 0;
    check(llama_resource_get_memory_pools(server.ctx, nullptr, 0, &count), "memory recheck inventory query");
    std::vector<llama_resource_memory_pool> before(count);
    check(llama_resource_get_memory_pools(server.ctx, before.data(), count, &count), "memory recheck inventory read");
    uint64_t weights = 0;
    for (const auto & pool : before) {
        int32_t mapped = -99;
        check(llama_resource_backend_device_id(server.ctx, pool.backend_id, &mapped), "memory recheck inventory mapping");
        if (pool.category == LLAMA_RESOURCE_MEMORY_WEIGHTS && mapped >= 0) weights += pool.bytes;
        if (device_id < 0) check(mapped == -1, "CPU inventory unexpectedly maps to accelerator");
    }
    check(device_id < 0 || weights > 0, "memory recheck needs nonzero accelerator weights");
    for (bool host_test : device_id >= 0 ? std::vector<bool>{false, true} : std::vector<bool>{true}) {
        server_task task;
        task.id = host_test ? 901 : 900;
        task.type = SERVER_TASK_TYPE_RESOURCE_REBALANCE;
        json device_limits = json::array();
        if (device_id >= 0) device_limits.push_back({
            {"id", device_id}, {"weights_bytes", host_test ? UINT64_MAX : uint64_t(0)},
            {"fixed_bytes", host_test ? UINT64_MAX : uint64_t(0)},
        });
        task.data = {{"expected_previous_plan", ""}, {"device_memory_limits", std::move(device_limits)},
            {"host_memory_limits", {{"weights_bytes", UINT64_MAX},
                {"fixed_bytes", host_test ? uint64_t(0) : UINT64_MAX}, {"ple_cache_bytes", UINT64_MAX}}}};
        server.queue_results.add_waiting_task_id(task.id);
        server.process_single_task(std::move(task));
        check(server.queue_results.queue_results_legacy.size() == 1, "memory recheck error response missing");
        const auto & result = server.queue_results.queue_results_legacy.front();
        check(result.error && result.data.dump().find("tracked resource memory changed before preparation") != std::string::npos,
              "stale tracked memory was not rejected before preparation");
        check(save(server.ctx) == state, "memory recheck rejection changed sequence state");
        std::vector<llama_resource_memory_pool> after(count);
        check(llama_resource_get_memory_pools(server.ctx, after.data(), count, &count), "memory recheck post-rejection inventory");
        check(before.size() == after.size(), "memory recheck changed inventory count");
        for (size_t i = 0; i < before.size(); ++i)
            check(before[i].backend_id == after[i].backend_id && before[i].category == after[i].category &&
                  before[i].bytes == after[i].bytes, "memory recheck changed physical allocation");
        server.queue_results.queue_results_legacy.clear();
        std::printf("server memory recheck: host=%d stale-limit rejection exact\n", host_test);
    }
    std::map<int32_t, uint64_t> kv_bytes;
    for (const auto & pool : before) if (pool.category == LLAMA_RESOURCE_MEMORY_KV) {
        int32_t mapped = -99;
        check(llama_resource_backend_device_id(server.ctx, pool.backend_id, &mapped), "KV recheck mapping");
        kv_bytes[mapped] += pool.bytes;
    }
    json current_kv = json::array();
    for (const auto & item : kv_bytes) current_kv.push_back({{"id", item.first}, {"kv_bytes", item.second}});
    llama_kv_preparation_memory memory = {};
    check(llama_kv_cache_resize_memory(server.ctx, 256, &memory, nullptr, 0), "KV recheck sizing count");
    std::vector<llama_kv_preparation_device> preparation_devices(memory.device_count);
    check(llama_kv_cache_resize_memory(server.ctx, 256, &memory, preparation_devices.data(), preparation_devices.size()), "KV recheck sizing rows");
    json receipt_devices = json::array();
    for (const auto & item : preparation_devices) receipt_devices.push_back({{"id", item.device_id}, {"kv_bytes", item.kv_bytes}});
    json receipt = {{"context", 256}, {"host_metadata_bytes", memory.host_metadata_bytes + 1},
        {"host_staging_bytes", memory.host_staging_bytes}, {"devices", std::move(receipt_devices)}};
    for (bool sizing_test : {false, true}) {
        server_task task;
        task.id = sizing_test ? 911 : 910;
        task.type = SERVER_TASK_TYPE_RESOURCE_REBALANCE;
        json device_limits = json::array();
        if (device_id >= 0) device_limits.push_back({{"id", device_id}, {"weights_bytes", UINT64_MAX}, {"fixed_bytes", UINT64_MAX}});
        task.data = {{"expected_previous_plan", ""}, {"device_memory_limits", std::move(device_limits)},
            {"host_memory_limits", {{"weights_bytes", UINT64_MAX}, {"fixed_bytes", UINT64_MAX}, {"ple_cache_bytes", UINT64_MAX}}},
            {"current_kv_storage", sizing_test ? current_kv : json::array()},
            {"scope", "kv-only"}, {"replace_kv", true}, {"replace_expert", false}, {"replace_transient", false},
            {"target_context", 256}, {"kv_preparation", receipt}};
        server.queue_results.add_waiting_task_id(task.id);
        server.process_single_task(std::move(task));
        check(server.queue_results.queue_results_legacy.size() == 1, "KV recheck error response missing");
        const auto & result = server.queue_results.queue_results_legacy.front();
        const char * message = sizing_test ? "KV preparation sizing changed before allocation" : "live KV storage changed before preparation";
        check(result.error && result.data.dump().find(message) != std::string::npos, "stale KV receipt was not rejected before allocation");
        check(save(server.ctx) == state, "KV recheck rejection changed sequence state");
        std::vector<llama_resource_memory_pool> after(count);
        check(llama_resource_get_memory_pools(server.ctx, after.data(), count, &count), "KV recheck post-rejection inventory");
        for (size_t i = 0; i < before.size(); ++i)
            check(before[i].backend_id == after[i].backend_id && before[i].category == after[i].category &&
                    before[i].bytes == after[i].bytes, "KV recheck changed physical allocation");
        server.queue_results.queue_results_legacy.clear();
        std::printf("server KV recheck: sizing=%d preallocation rejection exact\n", sizing_test);
    }
    llama_kv_cache_clear(server.ctx);
    check(decode(server.ctx, prompt, 0, 0) == expected, "memory recheck rejection changed replay output");
    std::printf("server memory recheck: device=%d weights=%llu state/allocations/output exact\n",
        device_id, (unsigned long long) weights);
}

static uint64_t tracked_host(llama_context * ctx) {
    uint32_t count = 0;
    check(llama_resource_get_memory_pools(ctx, nullptr, 0, &count), "host quota inventory count");
    std::vector<llama_resource_memory_pool> pools(count);
    check(llama_resource_get_memory_pools(ctx, pools.data(), count, &count), "host quota inventory rows");
    uint64_t bytes = 0;
    for (const auto & pool : pools) if (pool.backend_id == -1 && pool.category != LLAMA_RESOURCE_MEMORY_MAPPED_WEIGHTS) bytes += pool.bytes;
    return bytes;
}
static void run_host_admission(const fixture & file) {
    auto mp = llama_model_default_params();
    mp.devices = ""; mp.n_gpu_layers = 0; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.use_mmap = false;
    std::unique_ptr<llama_model, decltype(&llama_free_model)> model(llama_model_load_from_file(file.path.string().c_str(), mp), llama_free_model);
    check(bool(model), "host quota model load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 16; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1; cp.flash_attn = true; cp.type_k = cp.type_v = GGML_TYPE_F16;
    llama_context_host_limits limits = {1ULL<<30, 0, 0, 0, 0};
    std::unique_ptr<llama_context, decltype(&llama_free)> baseline(llama_init_from_model_with_host_limits(model.get(), cp, &limits), llama_free);
    check(bool(baseline), "host quota generous context");
    const auto steady = tracked_host(baseline.get());
    const auto teacher = decode(baseline.get(), {3}, 0, 0);
    check(tracked_host(baseline.get()) == steady, "host quota teacher changed retained capacity");
    baseline.reset();
    for (int delta : {-1, 0, 1}) {
        limits.capacity_bytes = steady + delta;
        std::unique_ptr<llama_context, decltype(&llama_free)> bounded(llama_init_from_model_with_host_limits(model.get(), cp, &limits), llama_free);
        check(bool(bounded), "host quota rejected a fitting startup/backoff");
        if (bounded) {
            check(tracked_host(bounded.get()) <= limits.capacity_bytes, "host quota exceeded startup capacity");
            if (delta < 0) check(llama_n_ubatch(bounded.get()) < cp.n_ubatch, "host quota failed to back off the oversized graph");
            else check(tracked_host(bounded.get()) == steady, "host quota changed fitting startup capacity");
            check(decode(bounded.get(), {3}, 0, 0) == teacher, "host quota changed native output");
            if (delta < 0) continue;
            const auto before = tracked_host(bounded.get());
            const auto used = llama_get_kv_cache_used_cells(bounded.get());
            const float * logits = llama_get_logits_ith(bounded.get(), -1);
            std::vector<float> saved(logits, logits+32);
            auto batch = llama_batch_init(2, 0, 1);
            for (int i = 0; i < 2; ++i) {
                batch.token[i] = 8+i; batch.pos[i] = 1+i; batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0; batch.logits[i] = 1;
            }
            batch.n_tokens = 2;
            const int result = llama_decode(bounded.get(), batch);
            llama_batch_free(batch);
            check(result < 0, "host quota accepted oversized output replacement");
            check(tracked_host(bounded.get()) == before && llama_get_kv_cache_used_cells(bounded.get()) == used,
                    "host quota output rejection changed allocations or KV occupancy");
            check(std::equal(saved.begin(), saved.end(), logits), "host quota output rejection changed retained logits");
        }
    }
    limits = {1ULL<<20, 0, 0, 0, 0};
    check(!llama_init_from_model_with_host_limits(model.get(), cp, &limits), "host quota accepted oversized native context preparation");
    limits = {UINT64_MAX, UINT64_MAX, 1, 0, 0};
    check(!llama_init_from_model_with_host_limits(model.get(), cp, &limits), "host quota accepted reservation overflow");
    limits = {100, 101, 0, 0, 0};
    check(!llama_init_from_model_with_host_limits(model.get(), cp, &limits), "host quota accepted excessive reservation");
    limits = {1ULL<<30, 0, 0, 0, 0};
    baseline.reset(llama_init_from_model_with_host_limits(model.get(), cp, &limits));
    check(bool(baseline), "host quota generous replay context");
    auto batch = llama_batch_init(2, 0, 1);
    for (int i = 0; i < 2; ++i) {
        batch.token[i] = 3+i; batch.pos[i] = i; batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    batch.n_tokens = 2;
    check(llama_decode(baseline.get(), batch) == 0, "host quota rejected fitting output growth");
    llama_batch_free(batch);
    std::printf("host quota: steady=%llu byte boundaries, reservation faults, retained output rejection and fitting growth pass\n", (unsigned long long)steady);
}

static void run_host_planner(const fixture & file) {
    auto mp = llama_model_default_params();
    mp.devices = ""; mp.n_gpu_layers = 0; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.use_mmap = false;
    std::unique_ptr<llama_model, decltype(&llama_free_model)> model(llama_model_load_from_file(file.path.string().c_str(), mp), llama_free_model);
    check(bool(model), "host planner model load");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 16; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1; cp.flash_attn = true; cp.type_k = cp.type_v = GGML_TYPE_F16;
    struct state { int calls = 0; int mode = 0; llama_context_host_requirements required = {}; } data;
    const auto planner = [](const llama_context_host_requirements * required, llama_context_host_limits * limits, void * opaque) {
        auto & data = *static_cast<state *>(opaque);
        ++data.calls; data.required = *required;
        // Leave the ordinary admission path to reject a one-byte overcommit.
        limits->weights_reservation_bytes = required->weights_bytes;
        limits->fixed_reservation_bytes = required->fixed_bytes;
        limits->other_reservation_bytes = limits->capacity_bytes-required->weights_bytes-limits->fixed_reservation_bytes-required->kv_bytes-(8ULL<<20);
        if (data.mode == 1) limits->other_reservation_bytes += (8ULL<<20)+1;
        if (data.mode == 2) ++limits->capacity_bytes;
        return data.mode != 3;
    };
    const llama_context_host_limits limits = {1ULL<<30, 0, 0, (1ULL<<30)-1, 0};
    for (int mode : {0, 1, 2, 3}) {
        data = {}; data.mode = mode;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model_with_host_planner(model.get(), cp, &limits, planner, &data), llama_free);
        check(data.calls >= 1 && (mode == 0 || data.calls == 1), "host planner callback phase count changed");
        check(data.required.fixed_bytes > 0 && data.required.weights_bytes > 0 && data.required.kv_bytes > 0,
            "host planner omitted native requirements");
        check(bool(ctx) == (mode == 0), "host planner admission/capacity/rejection boundary changed");
        if (ctx) check(!decode(ctx.get(), {3}, 0, 0).empty(), "host planner fitting inference failed");
    }
    std::puts("host planner: native receipt, fitting inference, one-byte excess, immutable capacity and explicit rejection pass");
}

static void run_host_kv_admission(const fixture & file) {
    auto mp = llama_model_default_params();
    mp.devices = ""; mp.n_gpu_layers = 0; mp.split_mode = LLAMA_SPLIT_MODE_NONE; mp.use_mmap = false;
    std::unique_ptr<llama_model, decltype(&llama_free_model)> model(llama_model_load_from_file(file.path.string().c_str(), mp), llama_free_model);
    check(bool(model), "host KV quota model");
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = cp.n_ubatch = 16; cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 1; cp.flash_attn = true; cp.type_k = cp.type_v = GGML_TYPE_F16;
    std::vector<ggml_type> full(3, GGML_TYPE_F32);
    for (int operation : {0, 1, 2}) {
        llama_context_host_limits limits = {1ULL<<30, 0, 0, 0, 0};
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model_with_host_limits(model.get(), cp, &limits), llama_free);
        check(bool(ctx), "host KV quota generous context");
        const auto teacher = decode(ctx.get(), {3}, 0, 0);
        const auto steady = tracked_host(ctx.get());
        const uint32_t target = operation == 0 ? 256 : 512;
        const auto forecast = [&](llama_kv_preparation_memory & memory, llama_kv_preparation_device * devices, uint32_t count) {
            return operation == 1 ? llama_kv_cache_retier_memory(ctx.get(), full.data(), full.data(), full.size(), &memory, devices, count) :
                llama_kv_cache_resize_memory(ctx.get(), target, &memory, devices, count);
        };
        llama_kv_preparation_memory memory = {};
        check(forecast(memory, nullptr, 0), "host KV quota forecast count");
        std::vector<llama_kv_preparation_device> devices(memory.device_count);
        check(forecast(memory, devices.data(), devices.size()), "host KV quota forecast rows");
        uint64_t incoming = memory.host_metadata_bytes+memory.host_staging_bytes;
        for (const auto & device : devices) if (device.device_id == -1) incoming += device.kv_bytes;
        check(incoming > 0, "host KV quota omitted transaction storage");
        ctx.reset();
        for (int delta : {-1, 0, 1}) {
            limits.capacity_bytes = steady+incoming+delta;
            ctx.reset(llama_init_from_model_with_host_limits(model.get(), cp, &limits));
            check(bool(ctx), "host KV quota bounded context");
            check(decode(ctx.get(), {3}, 0, 0) == teacher, "host KV quota changed teacher");
            const auto before = tracked_host(ctx.get());
            const auto state = save(ctx.get());
            auto * transaction = operation == 1 ? llama_kv_cache_prepare_retier(ctx.get(), full.data(), full.data(), full.size()) :
                llama_kv_cache_prepare_resize(ctx.get(), target);
            if (delta < 0) {
                if (transaction) llama_kv_cache_transaction_free(transaction);
                check(!transaction, "host KV quota accepted preparation one byte over its limit");
            } else {
                check(transaction != nullptr, "host KV quota rejected fitting preparation");
                check(tracked_host(ctx.get()) <= limits.capacity_bytes, "host KV quota exceeded retained storage capacity");
                llama_kv_preparation_memory actual = {};
                check(llama_kv_cache_transaction_memory(transaction, &actual, nullptr, 0), "host KV quota actual receipt");
                check(actual.host_metadata_bytes == memory.host_metadata_bytes && actual.host_staging_bytes == memory.host_staging_bytes,
                    "host KV quota receipt differs from native forecast");
                check(llama_kv_cache_transaction_publish(transaction), "host KV quota publish");
                check(tracked_host(ctx.get()) <= limits.capacity_bytes, "host KV quota publication exceeded capacity");
                check(llama_kv_cache_transaction_rollback(transaction), "host KV quota rollback");
                llama_kv_cache_transaction_free(transaction);
            }
            check(tracked_host(ctx.get()) == before && save(ctx.get()) == state, "host KV quota changed retained allocation/state");
            check(std::equal(teacher.begin(), teacher.end(), llama_get_logits_ith(ctx.get(), -1)), "host KV quota changed retained logits");
            ctx.reset();
        }
        std::printf("host KV quota: operation=%d incoming=%llu exact boundary and rollback pass\n", operation, (unsigned long long)incoming);
    }
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::strcmp(argv[1], "--slot-action-errors-only") == 0) {
        try { run_slot_action_errors(); return 0; }
        catch (const std::exception & error) { std::fprintf(stderr, "SERVER ACTION ERROR: %s\n", error.what()); return 1; }
    }
    llama_backend_init();
    try {
        run_result_defaults();
        run_slot_action_errors();
        run_moe_first_request();
        fixture file;
        for (int threshold : {1, 20}) for (bool system : {false, true}) run(file, threshold, system);
        run(file, 1, true, true);
        run_memory_recheck(file);
        run_host_admission(file);
        run_host_planner(file);
        run_host_kv_admission(file);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "SERVER RECOVERY ERROR: %s\n", error.what());
        llama_backend_free();
        return 1;
    }
    llama_backend_free();
    return 0;
}
