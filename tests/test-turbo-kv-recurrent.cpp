#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAILED: %s\n", message);
        std::abort();
    }
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s RECURRENT_OR_HYBRID_MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    require(model != nullptr, "load recurrent/hybrid model");

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = 128;
    context_params.n_batch = 32;
    context_params.n_ubatch = 32;
    context_params.type_k = GGML_TYPE_F16;
    context_params.type_v = GGML_TYPE_F16;
    context_params.offload_kqv = false;
    llama_context * ctx = llama_init_from_model(model, context_params);
    require(ctx != nullptr, "initialize recurrent/hybrid context");

    const char * text = "Recurrent cache state remains valid.";
    int32_t token_count = llama_tokenize(model, text, int32_t(std::strlen(text)), nullptr, 0, true, false);
    require(token_count < 0, "size recurrent prompt tokenization");
    std::vector<llama_token> tokens(size_t(-token_count));
    token_count = llama_tokenize(model, text, int32_t(std::strlen(text)),
            tokens.data(), int32_t(tokens.size()), true, false);
    require(token_count > 0, "tokenize recurrent prompt");
    tokens.resize(size_t(token_count));
    require(llama_decode(ctx, llama_batch_get_one(tokens.data(), token_count, 0, 0)) == 0,
            "decode recurrent prompt before rejection test");
    llama_synchronize(ctx);

    const uint32_t layers = llama_kv_cache_layer_count(ctx);
    require(layers > 0, "recurrent/hybrid cache exposes layers");
    std::vector<llama_kv_cache_layer_types> before(layers);
    std::vector<llama_kv_cache_layer_types> after(layers);
    std::vector<ggml_type> q8(layers, GGML_TYPE_Q8_0);
    require(llama_kv_cache_get_layer_types(ctx, before.data(), layers), "read initial cache types");
    require(!llama_kv_cache_retier(ctx, q8.data(), q8.data(), layers),
            "retiering recurrent/hybrid cache fails closed");
    require(llama_kv_cache_get_layer_types(ctx, after.data(), layers), "read retained cache types");
    for (uint32_t layer = 0; layer < layers; ++layer) {
        require(before[layer].type_k == after[layer].type_k && before[layer].type_v == after[layer].type_v,
                "failed recurrent retier leaves representation unchanged");
    }

    llama_free(ctx);
    llama_free_model(model);
    llama_backend_free();
    std::puts("PASS: recurrent/hybrid KV retiering fails closed without mutation");
    return 0;
}
