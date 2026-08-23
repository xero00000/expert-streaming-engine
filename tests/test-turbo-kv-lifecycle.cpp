#include "llama.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// CMake's default Release build defines NDEBUG; lifecycle gates must still run.
#undef assert
#define assert(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
    std::abort(); \
} } while (0)

static std::vector<llama_token> tokenize(llama_model * model, const char * text) {
    const int32_t length = int32_t(std::strlen(text));
    int32_t count = llama_tokenize(model, text, length, nullptr, 0, true, false);
    assert(count < 0);
    std::vector<llama_token> result(size_t(-count));
    count = llama_tokenize(model, text, length, result.data(), int32_t(result.size()), true, false);
    assert(count > 0);
    result.resize(size_t(count));
    return result;
}

static std::vector<uint8_t> save_sequence(llama_context * ctx, llama_seq_id sequence) {
    std::vector<uint8_t> result(llama_state_seq_get_size(ctx, sequence, 0));
    assert(!result.empty());
    assert(llama_state_seq_get_data(ctx, result.data(), result.size(), sequence, 0) == result.size());
    return result;
}

static void require_all_types(llama_context * ctx, ggml_type expected) {
    const uint32_t layers = llama_kv_cache_layer_count(ctx);
    std::vector<llama_kv_cache_layer_types> types(layers);
    assert(llama_kv_cache_get_layer_types(ctx, types.data(), layers));
    for (const auto & layer : types) {
        if (layer.type_k != GGML_TYPE_COUNT) assert(layer.type_k == expected);
        if (layer.type_v != GGML_TYPE_COUNT) assert(layer.type_v == expected);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    assert(model != nullptr);

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 512;
    context_params.n_batch = 64;
    context_params.n_ubatch = 64;
    context_params.n_seq_max = 2;
    context_params.flash_attn = true;
    context_params.offload_kqv = false;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    llama_context * ctx = llama_init_from_model(model, context_params);
    assert(ctx != nullptr);

    auto prompt = tokenize(model, "Failure-atomic KV retiering preserves every active sequence.");
    assert(prompt.size() < context_params.n_ctx / 2);
    assert(llama_decode(ctx, llama_batch_get_one(prompt.data(), int32_t(prompt.size()), 0, 0)) == 0);
    llama_synchronize(ctx);
    llama_kv_cache_seq_cp(ctx, 0, 1, 0, -1);
    const llama_pos saved_position = llama_kv_cache_seq_pos_max(ctx, 0);
    assert(saved_position >= 0 && llama_kv_cache_seq_pos_max(ctx, 1) == saved_position);

    const auto before_failure = save_sequence(ctx, 0);
    const uint32_t layers = llama_kv_cache_layer_count(ctx);
    std::vector<ggml_type> q8(layers, GGML_TYPE_Q8_0);

#if defined(_WIN32)
    _putenv_s("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", "1");
#else
    setenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", "1", 1);
#endif
    assert(!llama_kv_cache_retier(ctx, q8.data(), q8.data(), layers));
    require_all_types(ctx, GGML_TYPE_F16);
    assert(save_sequence(ctx, 0) == before_failure);
    assert(llama_kv_cache_seq_pos_max(ctx, 1) == saved_position);

#if defined(_WIN32)
    _putenv_s("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", "");
#else
    unsetenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS");
#endif
    assert(llama_kv_cache_retier(ctx, q8.data(), q8.data(), layers));
    require_all_types(ctx, GGML_TYPE_Q8_0);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == saved_position);
    assert(llama_kv_cache_seq_pos_max(ctx, 1) == saved_position);
    // Representation metadata is part of the checkpoint contract. An older
    // F16 checkpoint must be rejected without changing either active slot.
    assert(llama_state_seq_set_data(ctx, before_failure.data(), before_failure.size(), 0, 0) != before_failure.size());
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == saved_position);
    assert(llama_kv_cache_seq_pos_max(ctx, 1) == saved_position);

    llama_batch batch = llama_batch_init(2, 0, 1);
    batch.n_tokens = 2;
    for (int32_t i = 0; i < 2; ++i) {
        batch.token[i] = prompt.back();
        batch.pos[i] = saved_position + 1;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = i;
        batch.logits[i] = 1;
    }
    assert(llama_decode(ctx, batch) == 0);
    llama_synchronize(ctx);
    llama_batch_free(batch);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == saved_position + 1);
    assert(llama_kv_cache_seq_pos_max(ctx, 1) == saved_position + 1);

    llama_kv_cache_seq_add(ctx, 0, 0, -1, 1);
    assert(llama_kv_cache_update(ctx) == 0);
    llama_synchronize(ctx);
    const llama_pos shifted_position = saved_position + 2;
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == shifted_position);
    assert(llama_kv_cache_seq_pos_max(ctx, 1) == saved_position + 1);

    assert(llama_kv_cache_seq_rm(ctx, 1, -1, -1));
    assert(llama_kv_cache_seq_pos_min(ctx, 1) < 0);
    llama_kv_cache_defrag(ctx);
    assert(llama_kv_cache_update(ctx) == 0);
    llama_synchronize(ctx);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == shifted_position);

#if defined(_WIN32)
    _putenv_s("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", "1");
#else
    setenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", "1", 1);
#endif
    assert(llama_kv_cache_size(ctx) == 512);
    assert(!llama_kv_cache_resize(ctx, 1024));
    assert(!llama_kv_cache_resize(ctx, 300));
    assert(llama_kv_cache_size(ctx) == 512);
    assert(!llama_kv_cache_resize(ctx, 256));
    assert(llama_kv_cache_size(ctx) == 512);
    assert(llama_n_ctx(ctx) == 512);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == shifted_position);
#if defined(_WIN32)
    _putenv_s("ESE_TURBO_RETIER_FAIL_AFTER_ROWS", "");
#else
    unsetenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS");
#endif

    // Preparation succeeded and the candidate was made live, but a combined
    // transaction may still need to roll it back if a later publication fails.
    // This gate exercises that reversible boundary through the compatibility
    // API and proves that geometry, representation, and sequence state survive.
    const auto before_publish_rollback = save_sequence(ctx, 0);
#if defined(_WIN32)
    _putenv_s("ESE_KV_TRANSACTION_FAIL_AFTER_PUBLISH", "1");
#else
    setenv("ESE_KV_TRANSACTION_FAIL_AFTER_PUBLISH", "1", 1);
#endif
    assert(!llama_kv_cache_resize(ctx, 256));
    assert(llama_kv_cache_size(ctx) == 512);
    assert(llama_n_ctx(ctx) == 512);
    require_all_types(ctx, GGML_TYPE_Q8_0);
    assert(save_sequence(ctx, 0) == before_publish_rollback);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == shifted_position);
#if defined(_WIN32)
    _putenv_s("ESE_KV_TRANSACTION_FAIL_AFTER_PUBLISH", "");
#else
    unsetenv("ESE_KV_TRANSACTION_FAIL_AFTER_PUBLISH");
#endif

    assert(llama_kv_cache_resize(ctx, 256));
    assert(llama_kv_cache_size(ctx) == 256);
    assert(llama_n_ctx(ctx) == 256);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == shifted_position);

    llama_batch resized_batch = llama_batch_init(1, 0, 1);
    resized_batch.n_tokens = 1;
    resized_batch.token[0] = prompt.back();
    resized_batch.pos[0] = shifted_position + 1;
    resized_batch.n_seq_id[0] = 1;
    resized_batch.seq_id[0][0] = 0;
    resized_batch.logits[0] = 1;
    assert(llama_decode(ctx, resized_batch) == 0);
    llama_synchronize(ctx);
    llama_batch_free(resized_batch);
    const llama_pos resized_position = shifted_position + 1;
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == resized_position);

    assert(llama_kv_cache_resize(ctx, 512));
    assert(llama_kv_cache_size(ctx) == 512);
    assert(llama_n_ctx(ctx) == 512);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == resized_position);

    const auto q8_checkpoint = save_sequence(ctx, 0);
    assert(llama_kv_cache_seq_rm(ctx, 0, -1, -1));
    assert(llama_state_seq_set_data(ctx, q8_checkpoint.data(), q8_checkpoint.size(), 0, 0) == q8_checkpoint.size());
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == resized_position);
    assert(llama_kv_cache_seq_pos_min(ctx, 1) < 0);

    std::vector<ggml_type> f16(layers, GGML_TYPE_F16);
    assert(llama_kv_cache_retier(ctx, f16.data(), f16.data(), layers));
    require_all_types(ctx, GGML_TYPE_F16);
    assert(llama_kv_cache_seq_pos_max(ctx, 0) == resized_position);
    assert(llama_kv_cache_seq_pos_min(ctx, 1) < 0);

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::puts("PASS: failure-atomic KV retiering/resize, multi-slot decode, and checkpoint restore");
    return 0;
}
