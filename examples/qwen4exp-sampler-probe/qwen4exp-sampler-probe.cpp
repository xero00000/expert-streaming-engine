// Artifact-dependent target sampler/rejection diagnostic. The loader gate stays closed.
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "llama-sampling.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
using probe_clock = std::chrono::steady_clock;
static int64_t elapsed_us(probe_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(probe_clock::now() - start).count();
}
using sampler_ptr = std::unique_ptr<common_sampler, decltype(&common_sampler_free)>;
static sampler_ptr sampler(llama_model * model, const common_params_sampling & params) {
    sampler_ptr result(common_sampler_init(model, params), common_sampler_free);
    require(bool(result), "sampler initialization");
    return result;
}
static void decode(llama_context * ctx, const std::vector<llama_token> & ids, int pos, bool all = true) {
    auto batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(batch, ids[i], pos + i, {0}, all || i + 1 == ids.size());
    }
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    require(ret == 0, "decode");
    llama_synchronize(ctx);
}
static std::vector<uint8_t> sequence(llama_context * ctx) {
    std::vector<uint8_t> result(llama_state_seq_get_size(ctx, 0, 0));
    require(llama_state_seq_get_data(ctx, result.data(), result.size(), 0, 0) == result.size(), "sequence save");
    return result;
}
static bool adaptive_equal(const common_sampler * a, const common_sampler * b) {
    if (!a->adapt_p_ctx || !b->adapt_p_ctx) return a->adapt_p_ctx == b->adapt_p_ctx;
    return a->adapt_p_ctx->rng == b->adapt_p_ctx->rng &&
           a->adapt_p_ctx->history == b->adapt_p_ctx->history &&
           a->adapt_p_ctx->orig_prob == b->adapt_p_ctx->orig_prob &&
           a->adapt_p_ctx->cum_probs == b->adapt_p_ctx->cum_probs &&
           a->adapt_p_ctx->cum_orig_prob == b->adapt_p_ctx->cum_orig_prob &&
           a->adapt_p_ctx->cum_cur_p == b->adapt_p_ctx->cum_cur_p &&
           a->adapt_p_ctx->max_xform_logit == b->adapt_p_ctx->max_xform_logit;
}

int main(int argc, char ** argv) {
    llama_context * ctx = nullptr;
    llama_model * model = nullptr;
    try {
        gpt_params params;
        if (!gpt_params_parse(argc, argv, params)) return 2;
        const int requested_mode = std::getenv("ESE_SAMPLER_MODE") ? std::atoi(std::getenv("ESE_SAMPLER_MODE")) : LLAMA_SPEC_CKPT_CPU;
        params.n_parallel = requested_mode == LLAMA_SPEC_CKPT_PER_STEP ? 1 : 2;
        params.warmup = false;
        llama_backend_init();
        llama_numa_init(params.numa);
        auto init = llama_init_from_gpt_params(params);
        model = init.model; ctx = init.context;
        require(ctx && model, "model initialization");
        const int prefix = std::getenv("ESE_SAMPLER_PREFIX") ? std::atoi(std::getenv("ESE_SAMPLER_PREFIX")) : 8;
        const int mode = requested_mode;
        const bool grammar_first = std::getenv("ESE_SAMPLER_GRAMMAR_FIRST") && std::atoi(std::getenv("ESE_SAMPLER_GRAMMAR_FIRST"));
        const bool use_commit = std::getenv("ESE_SAMPLER_USE_COMMIT") && std::atoi(std::getenv("ESE_SAMPLER_USE_COMMIT"));
        const bool timing = std::getenv("ESE_SAMPLER_TIMING") && std::atoi(std::getenv("ESE_SAMPLER_TIMING"));
        require(prefix >= 8 && prefix <= 256, "prefix range");
        require(mode == LLAMA_SPEC_CKPT_PER_STEP || mode == LLAMA_SPEC_CKPT_CPU || mode == LLAMA_SPEC_CKPT_GPU_FALLBACK, "checkpoint mode");
        if (mode == LLAMA_SPEC_CKPT_PER_STEP) require(llama_spec_ckpt_init(ctx, mode, 8) == mode, "complete-prefix startup capacity");
        auto source = common_tokenize(model, "The engine loads weights and computes a response. What is two plus two?\n", false, true);
        require(!source.empty(), "tokenization");
        std::vector<llama_token> prompt(prefix);
        for (int i = 0; i < prefix; ++i) { prompt[i] = source[i % source.size()]; decode(ctx, {prompt[i]}, i); }
        std::vector<uint8_t> base(llama_state_get_size(ctx));
        require(llama_state_get_data(ctx, base.data(), base.size()) == base.size(), "base save");
        auto restore = [&] {
            require(llama_state_set_data(ctx, base.data(), base.size()) == base.size(), "base restore");
            llama_set_rng_seed(ctx, 3419); // Mirostat uses the context's sampling RNG.
        };
        common_params_speculative spec_params;
        spec_params.type = COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE;
        // Verification includes its leading input; seven drafts fit eight rows.
        spec_params.n_max = mode == LLAMA_SPEC_CKPT_PER_STEP ? 7 : 8;
        spec_params.spec_ckpt_mode = mode;
        std::unique_ptr<common_speculative, decltype(&common_speculative_free)> spec(
            use_commit ? common_speculative_init(spec_params, ctx) : nullptr, common_speculative_free);
        require(!use_commit || bool(spec), "common speculative owner initialization");
        const llama_token seed = source[prefix % source.size()];
        bool exact = true;
        int cases = 0;
        for (int policy = 0; policy < 5; ++policy) {
            if (const char * selected = std::getenv("ESE_SAMPLER_POLICY")) {
                if (policy != std::atoi(selected)) continue;
            }
            common_params_sampling sp;
            sp.seed = 1741; sp.temp = policy == 0 ? 0.0f : 1.0f;
            sp.top_k = 40; sp.top_p = 0.95f; sp.min_p = 0.0f;
            if (policy == 1) sp.top_p = 1.0f; // retain candidates even on a sharply peaked row
            sp.penalty_repeat = 1.1f;
            if (policy == 2) {
                sp.adaptive_target = 0.1f; sp.adaptive_decay = 0.8f;
            }
            if (policy == 3) {
                sp.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [a-zA-Z0-9 .,!?\\n]+"};
            }
            if (policy == 4) sp.mirostat = 2;
            auto initial = sampler(model, sp);
            for (auto token : prompt) common_sampler_accept(initial.get(), ctx, token, false);
            common_sampler_accept(initial.get(), ctx, seed, true);
            for (int depth : {2, 4, 8}) {
                if (const char * selected = std::getenv("ESE_SAMPLER_DEPTH")) {
                    if (depth != std::atoi(selected)) continue;
                }
                for (int accepted = 1; accepted <= (use_commit ? depth : depth - 1); ++accepted) {
                    if (const char * selected = std::getenv("ESE_SAMPLER_ACCEPTED")) {
                        if (accepted != std::atoi(selected)) continue;
                    }
                    restore();
                    auto reference = sampler(model, sp);
                    common_sampler_clone(initial.get(), reference.get());
                    std::vector<llama_token> ids;
                    const auto scalar_start = probe_clock::now();
                    for (int i = 0; i < accepted; ++i) {
                        decode(ctx, {i ? ids.back() : seed}, prefix + i);
                        const auto id = common_sampler_sample(reference.get(), ctx, -1, grammar_first);
                        common_sampler_accept(reference.get(), ctx, id, true);
                        ids.push_back(id);
                    }
                    const auto scalar_us = elapsed_us(scalar_start);
                    const auto expected_state = sequence(ctx);
                    std::vector<float> expected_logits(llama_get_logits_ith(ctx, -1), llama_get_logits_ith(ctx, -1) + llama_n_vocab(model));
                    restore();
                    auto target = sampler(model, sp);
                    common_sampler_clone(initial.get(), target.get());
                    common_speculative_checkpoint checkpoint;
                    const auto checkpoint_start = probe_clock::now();
                    if (use_commit) {
                        require(common_speculative_before_draft(spec.get(), model, ctx, target.get(), sp, 0, prefix, seed, depth, mode), "common pre-draft checkpoint");
                        require(common_speculative_get_checkpoint(spec.get())->sampler == nullptr, "unnecessary sampler snapshot allocated");
                    } else {
                        require(llama_spec_ckpt_init(ctx, mode, depth) == mode, "checkpoint initialization");
                        require(llama_spec_ckpt_save(ctx, 0), "checkpoint save");
                        checkpoint.valid = true; checkpoint.mode = mode; checkpoint.seq_id = 0;
                        checkpoint.n_past = prefix; checkpoint.sampled = seed;
                        checkpoint.sampler = common_sampler_init(model, sp);
                        require(checkpoint.sampler != nullptr, "checkpoint sampler");
                        common_sampler_clone(initial.get(), checkpoint.sampler);
                    }
                    const auto checkpoint_us = elapsed_us(checkpoint_start);
                    // Drafts follow the scalar target until a forced rejection;
                    // the remaining future inputs include EOS to exercise history rollback.
                    std::vector<llama_token> draft(depth - 1, llama_token_eos(model));
                    std::copy(ids.begin(), ids.end() - 1, draft.begin());
                    if (accepted < depth) draft[accepted - 1] = (ids.back() + 1) % llama_n_vocab(model);
                    std::vector<llama_token> inputs = {seed};
                    inputs.insert(inputs.end(), draft.begin(), draft.end());
                    const auto decode_start = probe_clock::now();
                    decode(ctx, inputs, prefix);
                    const auto decode_us = elapsed_us(decode_start);
                    std::vector<int> indices(depth);
                    for (int i = 0; i < depth; ++i) indices[i] = i;
                    const auto sample_start = probe_clock::now();
                    const auto verified = common_sampler_sample_and_accept_n(target.get(), ctx, indices, draft, grammar_first);
                    const auto sample_us = elapsed_us(sample_start);
                    const bool verified_exact = verified == ids;
                    const bool verification_sampler_exact = target->rng == reference->rng &&
                        target->mirostat_mu == reference->mirostat_mu && adaptive_equal(target.get(), reference.get()) &&
                        target->prev == reference->prev;
                    auto post_verification = sampler(model, sp);
                    common_sampler_clone(target.get(), post_verification.get());
                    // Serving's pos_base includes the leading input seed.
                    const auto commit_start = probe_clock::now();
                    const bool restored = use_commit
                        ? common_speculative_commit(spec.get(), ctx, target.get(), 0, seed, verified, depth - 1, prefix + 1, {})
                        : common_speculative_checkpoint_restore(checkpoint, nullptr, ctx, target.get(), 0,
                            COMMON_SPECULATIVE_TYPE_NONE, seed, verified, depth - 1, {}, prefix);
                    llama_synchronize(ctx);
                    const auto commit_us = elapsed_us(commit_start);
                    require(!(use_commit ? common_speculative_get_checkpoint(spec.get())->valid : checkpoint.valid), "committed checkpoint retained");
                    const bool state_exact = restored && sequence(ctx) == expected_state;
                    const bool logits_exact = restored && std::memcmp(llama_get_logits_ith(ctx, -1), expected_logits.data(), expected_logits.size() * sizeof(float)) == 0;
                    const bool rng_exact = target->rng == post_verification->rng;
                    const bool mu_exact = target->mirostat_mu == post_verification->mirostat_mu;
                    const bool adaptive_exact = adaptive_equal(target.get(), post_verification.get());
                    const bool history_exact = target->prev == post_verification->prev && target->prev == reference->prev;
                    // Compare future draws against an untouched clone of the
                    // verified sampler on exactly the same target logits.
                    // Mirostat draws through a shared context RNG, checked by mu above.
                    int changed_draws = 0;
                    if (policy != 4 && restored) for (int i = 0; i < 32; ++i) {
                        changed_draws += common_sampler_sample(target.get(), ctx, -1, grammar_first) !=
                                         common_sampler_sample(post_verification.get(), ctx, -1, grammar_first);
                    }
                    const bool passed = verified_exact && verification_sampler_exact && state_exact && logits_exact && rng_exact && mu_exact && adaptive_exact && history_exact && changed_draws == 0;
                    exact &= passed; ++cases;
                    std::printf("sampler_recovery: {\"mode\":%d,\"prefix\":%d,\"grammar_first\":%s,\"use_commit\":%s,\"policy\":%d,\"depth\":%d,\"accepted\":%d,\"verified_exact\":%s,\"verification_sampler_exact\":%s,\"state_exact\":%s,\"logits_exact\":%s,\"rng_exact\":%s,\"mu_exact\":%s,\"adaptive_exact\":%s,\"history_exact\":%s,\"changed_draws\":%d}\n",
                        mode, prefix, grammar_first ? "true":"false", use_commit ? "true":"false", policy, depth, accepted, verified_exact ? "true":"false", verification_sampler_exact ? "true":"false", state_exact ? "true":"false", logits_exact ? "true":"false",
                        rng_exact ? "true":"false", mu_exact ? "true":"false", adaptive_exact ? "true":"false", history_exact ? "true":"false", changed_draws);
                    if (timing) {
                        // Forced drafts and the preceding scalar oracle bias cache state.
                        // These phase costs exclude draft generation and validation copies.
                        std::printf("sampler_timing: {\"mode\":%d,\"prefix\":%d,\"policy\":%d,\"depth\":%d,\"output_tokens\":%zu,\"qualified\":%s,\"scalar_us\":%lld,\"checkpoint_us\":%lld,\"decode_us\":%lld,\"sample_us\":%lld,\"commit_us\":%lld,\"cycle_us\":%lld}\n",
                            mode, prefix, policy, depth, verified.size(), passed ? "true":"false",
                            (long long)scalar_us, (long long)checkpoint_us, (long long)decode_us,
                            (long long)sample_us, (long long)commit_us,
                            (long long)(checkpoint_us + decode_us + sample_us + commit_us));
                    }
                    std::fflush(stdout);
                }
            }
        }
        spec.reset();
        llama_free(ctx); ctx = nullptr; llama_free_model(model); model = nullptr; llama_backend_free();
        std::printf("sampler_recovery_result: %s cases=%d\n", exact ? "qualified":"unqualified", cases);
        return exact && cases > 0 ? 0 : 3;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "sampler probe: %s\n", e.what());
        if (ctx) llama_free(ctx);
        if (model) llama_free_model(model);
        llama_backend_free();
        return 1;
    }
}
