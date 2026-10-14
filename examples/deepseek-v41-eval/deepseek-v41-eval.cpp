#include "common.h"
#include "llama.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

// Teacher-force the same token sequence through two configurations so their
// complete logit distributions can be compared independently of sampling.
int main(int argc, char ** argv) {
    try {
        gpt_params params;
        if (!gpt_params_parse(argc, argv, params)) return 1;
        if (params.prompt.empty() || params.logits_file.empty() || params.n_predict < 1)
            throw std::invalid_argument("provide -f CORPUS --save-all-logits OUTPUT -n MAX_POSITIONS");
        llama_backend_init();
        llama_numa_init(params.numa);
        auto initialized = llama_init_from_gpt_params(params);
        std::unique_ptr<llama_model, decltype(&llama_free_model)> model(initialized.model, llama_free_model);
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(initialized.context, llama_free);
        if (!model || !ctx) throw std::runtime_error("model/context initialization failed");
        auto tokens = common_tokenize(model.get(), params.prompt, false, true);
        if (tokens.size() < 2) throw std::invalid_argument("corpus needs at least two tokens");
        const size_t count = std::min(tokens.size() - 1, size_t(params.n_predict));
        if (count > llama_n_ctx(ctx.get())) throw std::invalid_argument("corpus exceeds context capacity");
        const int vocabulary = llama_n_vocab(model.get());
        const std::filesystem::path final(params.logits_file), partial(params.logits_file + ".partial");
        std::ofstream output(partial, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot open logits output");
        nlohmann::json positions = nlohmann::json::array();
        const bool final_only = std::getenv("ESE_EVAL_FINAL_LOGIT_ONLY") != nullptr;
        for (size_t base = 0; base < count;) {
            const size_t batch_size = std::min(count - base, size_t(llama_n_ubatch(ctx.get())));
            const auto start = std::chrono::steady_clock::now();
            auto batch = llama_batch_get_one(tokens.data() + base, batch_size, int(base), 0);
            std::vector<int8_t> all_logits(batch_size, 1);
            if (final_only) {
                std::fill(all_logits.begin(), all_logits.end(), 0);
                if (base + batch_size == count) all_logits.back() = 1;
            }
            batch.logits = all_logits.data();
            if (llama_decode(ctx.get(), batch) != 0)
                throw std::runtime_error("teacher-forced decode failed");
            const double batch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            for (size_t offset = 0; offset < batch_size; ++offset) {
                if (final_only && !all_logits[offset]) continue;
                const size_t i = base + offset;
                auto * logits = llama_get_logits_ith(ctx.get(), final_only ? -1 : int(offset));
                if (!logits) throw std::runtime_error("missing logits");
                double maximum = -INFINITY;
                int argmax = 0;
                for (int j = 0; j < vocabulary; ++j) {
                    if (!std::isfinite(logits[j])) throw std::runtime_error("nonfinite logit");
                    if (logits[j] > maximum) { maximum = logits[j]; argmax = j; }
                }
                double sum = 0;
                for (int j = 0; j < vocabulary; ++j) sum += std::exp(double(logits[j]) - maximum);
                const double nll = maximum + std::log(sum) - logits[tokens[i + 1]];
                output.write(reinterpret_cast<const char *>(logits), vocabulary * sizeof(float));
                if (!output) throw std::runtime_error("logits write failed");
                const double ms = batch_ms / batch_size;
                positions.push_back({{"position", i}, {"input_token", tokens[i]}, {"next_token", tokens[i+1]},
                                     {"argmax", argmax}, {"nll", nll}, {"milliseconds", ms}});
                std::fprintf(stderr, "teacher-force %zu/%zu: %.2f ms, nll %.5f\n", i+1, count, ms, nll);
            }
            base += batch_size;
        }
        output.close();
        if (!output) throw std::runtime_error("logits close failed");
        std::filesystem::rename(partial, final);
        std::ofstream meta(params.logits_file + ".json");
        meta << nlohmann::json({{"vocabulary", vocabulary}, {"rows", positions.size()},
                {"dtype", "float32-native"}, {"tokens", tokens}, {"positions", positions}}).dump(2) << '\n';
        if (!meta) throw std::runtime_error("metadata write failed");
        llama_print_timings(ctx.get());
        std::printf("Saved %zu teacher-forced logit rows\n", positions.size());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
    return 0;
}
