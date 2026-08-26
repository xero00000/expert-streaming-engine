#include "llama.h"

#include <cstdio>
#include <cstdlib>

// CMake's default Release build defines NDEBUG; lifecycle gates must still run.
#undef assert
#define assert(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
    std::abort(); \
} } while (0)

static llama_context * make_cpu_context(llama_model * model) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.offload_kqv = false;
    return llama_init_from_model(model, params);
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

    llama_context * ctx = make_cpu_context(model);
    assert(ctx != nullptr);

    // A CPU-only scheduler rejects a non-zero cache without retaining an owner.
    assert(llama_expert_cache_prepare_resize(ctx, 1) == nullptr);

    // Prepared ownership is exclusive and can be discarded without publishing.
    llama_expert_cache_transaction_t transaction =
            llama_expert_cache_prepare_resize(ctx, 0);
    assert(transaction != nullptr);
    assert(llama_expert_cache_prepare_resize(ctx, 0) == nullptr);
    assert(!llama_expert_cache_transaction_rollback(transaction));
    assert(!llama_expert_cache_transaction_finalize(transaction));
    llama_expert_cache_transaction_free(transaction);

    // Freeing a published transaction must roll both scheduler and policy back.
    transaction = llama_expert_cache_prepare_resize(ctx, 0);
    assert(transaction != nullptr);
    assert(llama_expert_cache_transaction_publish(transaction));
    assert(llama_expert_cache_prepare_resize(ctx, 0) == nullptr);
    llama_expert_cache_transaction_free(transaction);

    // Explicit rollback leaves an owned shell that is released by free().
    transaction = llama_expert_cache_prepare_resize(ctx, 0);
    assert(transaction != nullptr);
    assert(llama_expert_cache_transaction_publish(transaction));
    assert(llama_expert_cache_transaction_rollback(transaction));
    assert(!llama_expert_cache_transaction_publish(transaction));
    assert(!llama_expert_cache_transaction_finalize(transaction));
    llama_expert_cache_transaction_free(transaction);

    // Finalize commits irreversibly but retains the shell until free().
    transaction = llama_expert_cache_prepare_resize(ctx, 0);
    assert(transaction != nullptr);
    assert(llama_expert_cache_transaction_publish(transaction));
    assert(llama_expert_cache_transaction_finalize(transaction));
    assert(!llama_expert_cache_transaction_rollback(transaction));
    assert(llama_expert_cache_prepare_resize(ctx, 0) == nullptr);
    llama_expert_cache_transaction_free(transaction);
    assert(llama_expert_cache_resize(ctx, 0));
    llama_free(ctx);

    // Teardown must safely dispose both prepared and published transactions
    // before releasing the context scheduler/backends.
    ctx = make_cpu_context(model);
    assert(ctx != nullptr);
    transaction = llama_expert_cache_prepare_resize(ctx, 0);
    assert(transaction != nullptr);
    llama_free(ctx);

    ctx = make_cpu_context(model);
    assert(ctx != nullptr);
    transaction = llama_expert_cache_prepare_resize(ctx, 0);
    assert(transaction != nullptr);
    assert(llama_expert_cache_transaction_publish(transaction));
    llama_free(ctx);

    llama_free_model(model);
    llama_backend_free();
    std::puts("PASS: expert-cache transaction ownership, rollback, finalize, and teardown");
    return 0;
}
