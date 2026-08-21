#include "llama.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#undef assert
#define assert(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAILED: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
    std::abort(); \
} } while (0)

struct run_result {
    std::string name;
    double prompt_ms = 0;
    double decode_ms = 0;
    double nll = 0;
    std::vector<std::vector<float>> logits;
};

static std::vector<llama_token> tokenize(llama_model * model, const std::string & text) {
    int32_t count = llama_tokenize(model, text.c_str(), int32_t(text.size()), nullptr, 0, true, false);
    assert(count < 0);
    std::vector<llama_token> result(size_t(-count));
    count = llama_tokenize(model, text.c_str(), int32_t(text.size()), result.data(), int32_t(result.size()), true, false);
    assert(count > 0);
    result.resize(size_t(count));
    return result;
}

static double token_nll(const float * logits, int32_t vocab, llama_token target) {
    const float maximum = *std::max_element(logits, logits + vocab);
    double sum = 0;
    for (int32_t token = 0; token < vocab; ++token) {
        sum += std::exp(double(logits[token] - maximum));
    }
    return std::log(sum) + maximum - logits[target];
}

static run_result run_tier(
        llama_model * model,
        const std::vector<llama_token> & tokens,
        ggml_type type,
        const char * name,
        const std::vector<int32_t> & sample_positions) {
    const uint32_t layers = uint32_t(llama_n_layer(model));
    std::vector<ggml_type> map(layers, type);
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 128;
    params.n_batch = 64;
    params.n_ubatch = 64;
    params.n_seq_max = 1;
    params.flash_attn = true;
    params.offload_kqv = true;
    // Keep the legacy base types block-aligned; the exact experimental tier
    // enters through the per-layer map, whose allocator pads each head.
    params.type_k = GGML_TYPE_F16;
    params.type_v = GGML_TYPE_F16;
    params.type_k_layers = map.data();
    params.type_v_layers = map.data();
    params.n_type_k_layers = layers;
    params.n_type_v_layers = layers;

    llama_context * ctx = llama_init_from_model(model, params);
    assert(ctx != nullptr);
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    constexpr int32_t prompt_tokens = 32;
    constexpr int32_t eval_tokens = 64;
    assert(tokens.size() > size_t(prompt_tokens + eval_tokens));

    run_result result;
    result.name = name;
    const auto prompt_start = std::chrono::steady_clock::now();
    // The Phase 1 fused Turbo attention contract supports batches through 8;
    // keep prompt-processing measurement inside that native path.
    for (int32_t offset = 0; offset < prompt_tokens; offset += 8) {
        assert(llama_decode(ctx, llama_batch_get_one(
                        const_cast<llama_token *>(tokens.data()) + offset, 8, offset, 0)) == 0);
    }
    llama_synchronize(ctx);
    const auto prompt_end = std::chrono::steady_clock::now();
    result.prompt_ms = std::chrono::duration<double, std::milli>(prompt_end - prompt_start).count();

    auto collect = [&](int32_t position) {
        if (std::find(sample_positions.begin(), sample_positions.end(), position) == sample_positions.end()) return;
        const float * logits = llama_get_logits_ith(ctx, -1);
        assert(logits != nullptr);
        result.logits.emplace_back(logits, logits + vocab);
    };
    collect(prompt_tokens - 1);

    const auto decode_start = std::chrono::steady_clock::now();
    for (int32_t position = prompt_tokens; position < prompt_tokens + eval_tokens; ++position) {
        const float * prior = llama_get_logits_ith(ctx, -1);
        assert(prior != nullptr);
        result.nll += token_nll(prior, vocab, tokens[size_t(position)]);
        llama_token token = tokens[size_t(position)];
        assert(llama_decode(ctx, llama_batch_get_one(&token, 1, position, 0)) == 0);
        llama_synchronize(ctx);
        collect(position);
    }
    const auto decode_end = std::chrono::steady_clock::now();
    result.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
    result.nll /= eval_tokens;
    assert(result.logits.size() == sample_positions.size());
    llama_free(ctx);
    return result;
}

static double kld(const std::vector<float> & reference, const std::vector<float> & candidate) {
    assert(reference.size() == candidate.size());
    const float ref_max = *std::max_element(reference.begin(), reference.end());
    const float can_max = *std::max_element(candidate.begin(), candidate.end());
    double ref_sum = 0;
    double can_sum = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        ref_sum += std::exp(double(reference[i] - ref_max));
        can_sum += std::exp(double(candidate[i] - can_max));
    }
    const double ref_log_z = std::log(ref_sum) + ref_max;
    const double can_log_z = std::log(can_sum) + can_max;
    double value = 0;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double log_p = reference[i] - ref_log_z;
        value += std::exp(log_p) * (log_p - (candidate[i] - can_log_z));
    }
    return value;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], model_params);
    assert(model != nullptr);

    const std::string paragraph =
        "A careful engineer validates each cache transition before publishing it. "
        "The old representation remains active until every row has been converted, "
        "checked, and committed. Multiple requests continue independently while the "
        "controller balances memory, context depth, and measured output quality. ";
    std::string corpus;
    for (int i = 0; i < 12; ++i) corpus += paragraph;
    const auto tokens = tokenize(model, corpus);
    const std::vector<int32_t> depths = {31, 47, 63, 79, 95};

    struct tier { ggml_type type; const char * name; };
    const tier tiers[] = {
        { GGML_TYPE_F16,        "f16"        },
        { GGML_TYPE_Q8_0,       "q8_0"       },
        { GGML_TYPE_TURBO8_0,   "turbo8"     },
        { GGML_TYPE_TURBO4_0,   "turbo4"     },
        { GGML_TYPE_TURBO3_0,   "turbo3"     },
        { GGML_TYPE_TURBO2_0,   "turbo2"     },
        { GGML_TYPE_TURBO3_TCQ, "turbo3_tcq" },
        { GGML_TYPE_TURBO2_TCQ, "turbo2_tcq" },
        { GGML_TYPE_TURBO1_TCQ, "turbo1_tcq" },
    };

    const run_result reference = run_tier(model, tokens, tiers[0].type, tiers[0].name, depths);
    const double reference_ppl = std::exp(reference.nll);
    std::printf("tier,ppl,ppl_ratio,kld_mean,kld_p50,kld_p95,kld_max,prompt_tok_s,decode_tok_s\n");
    std::printf("f16,%.8f,1,0,0,0,0,%.3f,%.3f\n", reference_ppl,
            32000.0/reference.prompt_ms, 64000.0/reference.decode_ms);

    for (size_t tier_index = 1; tier_index < sizeof(tiers)/sizeof(tiers[0]); ++tier_index) {
        const run_result candidate = run_tier(model, tokens, tiers[tier_index].type, tiers[tier_index].name, depths);
        std::vector<double> distribution;
        for (size_t sample = 0; sample < reference.logits.size(); ++sample) {
            distribution.push_back(kld(reference.logits[sample], candidate.logits[sample]));
        }
        assert(std::all_of(distribution.begin(), distribution.end(), [](double value) {
            return std::isfinite(value) && value >= -1e-8;
        }));
        std::sort(distribution.begin(), distribution.end());
        const double mean = std::accumulate(distribution.begin(), distribution.end(), 0.0)/distribution.size();
        const double ppl = std::exp(candidate.nll);
        std::printf("%s,%.8f,%.8f,%.9f,%.9f,%.9f,%.9f,%.3f,%.3f\n",
                candidate.name.c_str(), ppl, ppl/reference_ppl, mean,
                distribution[distribution.size()/2], distribution.back(), distribution.back(),
                32000.0/candidate.prompt_ms, 64000.0/candidate.decode_ms);
    }

    llama_free_model(model);
    llama_backend_free();
    return 0;
}
