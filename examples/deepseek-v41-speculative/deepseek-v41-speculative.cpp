#include "common.h"
#include "llama.h"
#include "llama-deepseek-v41-dspark.h"
#include "llama-spec-features-dflash.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace {
using clock_type = std::chrono::steady_clock;
double seconds(clock_type::time_point start) { return std::chrono::duration<double>(clock_type::now() - start).count(); }
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
llama_token greedy(llama_context * ctx, int row, int vocab) {
    const float * logits = llama_get_logits_ith(ctx, row); check(logits, "missing target logits");
    for (int i = 0; i < vocab; ++i) check(std::isfinite(logits[i]), "nonfinite target logits");
    return std::max_element(logits, logits + vocab) - logits;
}
std::vector<float> features(llama_context * ctx, int count) {
    std::vector<int32_t> rows(count); std::iota(rows.begin(), rows.end(), 0);
    std::vector<float> output;
    check(llama_spec_copy_dflash_rows_from_output_indices(ctx, rows, output), "target feature capture failed"); return output;
}
}
int main(int argc, char ** argv) {
    try {
        std::string draft_path, stats_path;
        int max_draft = 5;
        std::vector<char *> args{argv[0]};
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--dspark-model" || arg == "--spec-stats" || arg == "--dspark-proposals") {
                check(i + 1 < argc, "missing speculative argument value");
                if (arg == "--dspark-model") draft_path = argv[++i];
                else if (arg == "--spec-stats") stats_path = argv[++i];
                else max_draft = std::stoi(argv[++i]);
            } else args.push_back(argv[i]);
        }
        gpt_params params;
        check(gpt_params_parse(args.size(), args.data(), params), "invalid arguments");
        check(!params.prompt.empty() && params.n_predict > 0 && params.sparams.temp == 0 && max_draft >= 1 && max_draft <= 5,
              "provide a prompt, positive -n, --temp 0, and 1..5 DSpark proposals");
        llama_backend_init(); llama_numa_init(params.numa);
        auto initialized = llama_init_from_gpt_params(params);
        std::unique_ptr<llama_model, decltype(&llama_free_model)> model(initialized.model, llama_free_model);
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(initialized.context, llama_free);
        check(model && ctx, "target initialization failed");
        check(llama_n_ubatch(ctx.get()) >= 6 && llama_n_batch(ctx.get()) >= 6, "use -ub 8 and -b >= 8");
        std::unique_ptr<deepseek_v41::dspark> draft;
        if (!draft_path.empty()) {
            draft.reset(new deepseek_v41::dspark(*model, draft_path));
            const auto & layers = draft->target_layers();
            check(llama_set_dflash_capture_layers(ctx.get(), layers.data(), layers.size()), "cannot enable DSpark feature capture");
            check(llama_spec_ckpt_init(ctx.get(), LLAMA_SPEC_CKPT_PER_STEP, 6) == LLAMA_SPEC_CKPT_PER_STEP,
                  "target lacks per-step speculative checkpoints");
        }
        auto prompt = common_tokenize(model.get(), params.prompt, false, true);
        check(!prompt.empty() && prompt.size() < llama_n_ctx(ctx.get()), "prompt exceeds target context");
        const int vocab = llama_n_vocab(model.get());
        auto prefill_start = clock_type::now();
        int position = 0;
        while (position < int(prompt.size())) {
            const int n = std::min(8, int(prompt.size()) - position);
            auto batch = llama_batch_get_one(prompt.data() + position, n, position, 0);
            check(llama_decode(ctx.get(), batch) == 0, "target prefill failed");
            if (draft) draft->ingest(features(ctx.get(), n), position, n);
            position += n;
        }
        llama_token known = greedy(ctx.get(), -1, vocab);
        const double prefill_s = seconds(prefill_start);
        auto start = clock_type::now();
        double draft_s = 0, verify_s = 0, ingest_s = 0;
        int proposed = 0, accepted_draft = 0, cycles = 0;
        std::vector<llama_token> generated;
        nlohmann::json steps = nlohmann::json::array();
        bool stop = false;
        while (!stop && int(generated.size()) < params.n_predict && position < int(llama_n_ctx(ctx.get()))) {
            if (llama_token_is_eog(model.get(), known)) { generated.push_back(known); break; }
            std::vector<llama_token> verify{known}, candidates;
            const int available = std::min(params.n_predict - int(generated.size()), int(llama_n_ctx(ctx.get())) - position);
            const auto draft_start = clock_type::now();
            if (draft && available > 1 && position + 5 <= int(llama_n_ctx(ctx.get()))) {
                candidates = draft->propose(known, position);
                candidates.resize(std::min({int(candidates.size()), max_draft, available - 1}));
                verify.insert(verify.end(), candidates.begin(), candidates.end());
            }
            draft_s += seconds(draft_start); proposed += candidates.size();
            if (draft) check(llama_spec_ckpt_save(ctx.get(), 0), "target checkpoint failed");
            const auto verify_start = clock_type::now();
            auto batch = llama_batch_get_one(verify.data(), verify.size(), position, 0);
            std::vector<int8_t> logits(verify.size(), 1); batch.logits = logits.data();
            check(llama_decode(ctx.get(), batch) == 0, "target speculative verification failed");
            int accepted = 1;
            while (accepted < int(verify.size()) && verify[accepted] == greedy(ctx.get(), accepted - 1, vocab)) ++accepted;
            known = greedy(ctx.get(), accepted - 1, vocab);
            for (int i = 0; i < accepted; ++i) if (llama_token_is_eog(model.get(), verify[i])) {
                accepted = i + 1; stop = true; break;
            }
            std::vector<float> retained;
            if (draft) {
                retained = features(ctx.get(), accepted);
                check(llama_spec_ckpt_restore(ctx.get(), 0, position, accepted - 1), "target accepted-prefix restore failed");
            }
            verify_s += seconds(verify_start);
            const auto ingest_start = clock_type::now();
            if (draft) draft->ingest(retained, position, accepted);
            ingest_s += seconds(ingest_start);
            for (int i = 0; i < accepted; ++i) {
                generated.push_back(verify[i]);
                if (!llama_token_is_eog(model.get(), verify[i])) std::printf("%s", common_token_to_piece(ctx.get(), verify[i]).c_str());
            }
            std::fflush(stdout);
            steps.push_back({{"position", position}, {"proposed", candidates}, {"accepted", accepted}});
            position += accepted; accepted_draft += accepted - 1; ++cycles;
            std::fprintf(stderr, "DSpark cycle %d: accepted %d/%zu draft tokens\n", cycles, accepted - 1, candidates.size());
        }
        const double decode_s = seconds(start);
        int visible = generated.size();
        if (!generated.empty() && llama_token_is_eog(model.get(), generated.back())) --visible;
        nlohmann::json stats{{"mode", draft ? "dspark-greedy" : "target-greedy"}, {"prompt_tokens", prompt},
            {"generated_tokens", generated}, {"visible_tokens", visible}, {"prefill_seconds", prefill_s}, {"decode_seconds", decode_s},
            {"visible_tokens_per_second", visible / decode_s}, {"draft_seconds", draft_s}, {"verify_seconds", verify_s},
            {"ingest_seconds", ingest_s}, {"proposed_tokens", proposed}, {"accepted_draft_tokens", accepted_draft}, {"cycles", cycles}, {"steps", steps}};
        if (!stats_path.empty()) { std::ofstream out(stats_path); out << stats.dump(2) << '\n'; check(bool(out), "cannot write speculative statistics"); }
        std::fprintf(stderr, "\nSpeculative summary: %s\n", stats.dump().c_str());
        llama_print_timings(ctx.get());
        if (draft) llama_spec_ckpt_discard(ctx.get());
    } catch (const std::exception & e) { std::fprintf(stderr, "DSpark error: %s\n", e.what()); return 1; }
    return 0;
}
