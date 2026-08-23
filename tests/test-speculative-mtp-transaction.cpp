#include "speculative.h"

#include <cstdio>
#include <type_traits>

static bool run_model_lifecycle(const char * model_path) {
    llama_backend_init();

    gpt_params params;
    params.model = model_path;
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_gpu_layers = 0;
    params.no_kv_offload = true;
    params.flash_attn = false;
    params.embedding = true;
    params.speculative.type = COMMON_SPECULATIVE_TYPE_MTP;
    params.speculative.n_max = 1;
    common_speculative_prepare_startup(params);

    llama_init_result initialized = llama_init_from_gpt_params(params);
    if (initialized.model == nullptr || initialized.context == nullptr) {
        std::fprintf(stderr, "failed to load MTP lifecycle model: %s\n", model_path);
        llama_backend_free();
        return false;
    }

    bool ok = common_speculative_finalize_startup(params, initialized.model);
    common_speculative * spec = nullptr;
    if (ok) {
        llama_set_embeddings(initialized.context, true);
        ok = common_speculative_try_init(params.speculative, initialized.context, &spec) ==
                COMMON_SPECULATIVE_INIT_READY;
    }

    const auto check = [&ok](bool condition, const char * message) {
        if (!condition) {
            std::fprintf(stderr, "MTP transaction check failed: %s\n", message);
            ok = false;
        }
        return condition;
    };

    if (ok) {
        llama_context * original = common_speculative_get_companion_ctx(spec);
        check(original != nullptr, "initial owner is resident");

        auto * suspend = common_speculative_mtp_prepare_residency(spec, false);
        check(suspend != nullptr, "prepare resident -> suspended");
        check(common_speculative_mtp_prepare_residency(spec, false) == nullptr,
                "reject a second open transaction");
        check(common_speculative_get_companion_ctx(spec) == original,
                "prepare leaves the live owner unchanged");
        check(!common_speculative_mtp_transaction_rollback(suspend),
                "reject rollback before publication");
        check(!common_speculative_mtp_transaction_finalize(suspend),
                "reject finalize before publication");
        check(common_speculative_mtp_transaction_publish(suspend), "publish suspension");
        check(!common_speculative_mtp_transaction_publish(suspend),
                "reject repeated publication");
        check(!common_speculative_mtp_resident(spec), "published suspension is visible");
        check(common_speculative_mtp_transaction_rollback(suspend), "rollback suspension");
        check(!common_speculative_mtp_transaction_rollback(suspend),
                "reject repeated rollback");
        check(common_speculative_get_companion_ctx(spec) == original,
                "rollback restores the exact context owner");
        common_speculative_mtp_transaction_free(suspend);

        suspend = common_speculative_mtp_prepare_residency(spec, false);
        check(suspend != nullptr, "prepare suspension for discard");
        common_speculative_mtp_transaction_free(suspend);
        check(common_speculative_get_companion_ctx(spec) == original,
                "free discards an unpublished candidate");

        suspend = common_speculative_mtp_prepare_residency(spec, false);
        check(suspend != nullptr && common_speculative_mtp_transaction_publish(suspend),
                "publish suspension for free rollback");
        common_speculative_mtp_transaction_free(suspend);
        check(common_speculative_get_companion_ctx(spec) == original,
                "free auto-rolls back the exact context owner");

        suspend = common_speculative_mtp_prepare_residency(spec, false);
        check(suspend != nullptr && common_speculative_mtp_transaction_publish(suspend),
                "publish committed suspension");
        check(common_speculative_mtp_transaction_finalize(suspend), "finalize suspension");
        common_speculative_mtp_transaction_free(suspend);
        check(!common_speculative_mtp_resident(spec), "finalize retires the old resident owner");

        auto * resume = common_speculative_mtp_prepare_residency(spec, true);
        check(resume != nullptr, "prepare suspended -> resident");
        check(!common_speculative_mtp_resident(spec), "resume prepare stays off-side");
        check(common_speculative_mtp_transaction_publish(resume), "publish resume");
        llama_context * resumed = common_speculative_get_companion_ctx(spec);
        check(resumed != nullptr, "published resume is visible");
        check(common_speculative_mtp_transaction_rollback(resume), "rollback resume");
        check(!common_speculative_mtp_resident(spec), "rollback resume restores suspension");
        common_speculative_mtp_transaction_free(resume);

        resume = common_speculative_mtp_prepare_residency(spec, true);
        check(resume != nullptr && common_speculative_mtp_transaction_publish(resume),
                "publish resume for free rollback");
        common_speculative_mtp_transaction_free(resume);
        check(!common_speculative_mtp_resident(spec), "free auto-rolls back resume");

        resume = common_speculative_mtp_prepare_residency(spec, true);
        check(resume != nullptr && common_speculative_mtp_transaction_publish(resume),
                "publish committed resume");
        check(common_speculative_mtp_transaction_finalize(resume), "finalize resume");
        common_speculative_mtp_transaction_free(resume);
        resumed = common_speculative_get_companion_ctx(spec);
        check(resumed != nullptr, "finalize retains the prepared resident owner");

        auto * no_op = common_speculative_mtp_prepare_residency(spec, true);
        check(no_op != nullptr && common_speculative_mtp_transaction_publish(no_op),
                "publish same-residency no-op");
        check(common_speculative_mtp_transaction_rollback(no_op), "rollback same-residency no-op");
        check(common_speculative_get_companion_ctx(spec) == resumed,
                "no-op rollback preserves the exact owner");
        common_speculative_mtp_transaction_free(no_op);

        no_op = common_speculative_mtp_prepare_residency(spec, true);
        check(no_op != nullptr && common_speculative_mtp_transaction_publish(no_op),
                "publish same-residency no-op for commit");
        check(common_speculative_mtp_transaction_finalize(no_op),
                "finalize same-residency no-op");
        common_speculative_mtp_transaction_free(no_op);
        check(common_speculative_get_companion_ctx(spec) == resumed,
                "no-op finalize preserves the exact owner");
    }

    common_speculative_free(spec);
    params.speculative.clear_dft();
    llama_free(initialized.context);
    llama_free_model(initialized.model);
    llama_backend_free();
    return ok;
}

int main(int argc, char ** argv) {
    static_assert(std::is_pointer_v<common_speculative_mtp_transaction_t>);
    static_assert(std::is_same_v<
            std::remove_pointer_t<common_speculative_mtp_transaction_t>,
            common_speculative_mtp_transaction>);

    // The opaque API's invalid-handle contract is deliberately total so owner
    // transaction cleanup can be unconditional on every preparation path.
    if (common_speculative_mtp_prepare_residency(nullptr, false) != nullptr ||
            common_speculative_mtp_prepare_residency(nullptr, true) != nullptr ||
            common_speculative_mtp_transaction_publish(nullptr) ||
            common_speculative_mtp_transaction_rollback(nullptr) ||
            common_speculative_mtp_transaction_finalize(nullptr)) {
        std::fputs("null MTP transaction contract failed\n", stderr);
        return 1;
    }
    common_speculative_mtp_transaction_free(nullptr);

    // Preserve the legacy no-MTP/null-spec convenience behavior.
    if (!common_speculative_suspend_mtp(nullptr) ||
            !common_speculative_resume_mtp(nullptr) ||
            !common_speculative_mtp_resident(nullptr)) {
        std::fputs("null speculative-state convenience contract failed\n", stderr);
        return 1;
    }

    if (argc > 1 && !run_model_lifecycle(argv[1])) {
        return 1;
    }

    std::puts(argc > 1
            ? "PASS: opaque MTP residency transaction lifecycle"
            : "PASS: opaque MTP residency transaction API is linked and null-safe");
    return 0;
}
