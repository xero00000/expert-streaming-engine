#include "llama.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <memory>
#include <stdexcept>
int main(int argc, char ** argv) {
    try {
        if (argc != 3) throw std::runtime_error("usage: test-deepseek-v41-tokenizer VOCAB.gguf REFERENCE.json");
        std::ifstream input(argv[2]); auto fixture = nlohmann::json::parse(input);
        llama_backend_init();
        auto params = llama_model_default_params(); params.vocab_only = true;
        std::unique_ptr<llama_model, decltype(&llama_free_model)> model(llama_model_load_from_file(argv[1], params), llama_free_model);
        if (!model) throw std::runtime_error("vocabulary load failed");
        size_t count = 0;
        for (auto item : fixture.at("cases")) {
            auto text = item.at("text").get<std::string>();
            auto expected = item.at("ids").get<std::vector<llama_token>>();
            std::vector<llama_token> actual(text.size() * 4 + 16);
            auto n = llama_tokenize(model.get(), text.data(), text.size(), actual.data(), actual.size(), false, true);
            if (n < 0) throw std::runtime_error("token buffer too small");
            actual.resize(n);
            if (actual != expected) {
                std::fprintf(stderr, "Tokenizer mismatch case %zu: %s\nexpected %s\nactual %s\n", count,
                    text.c_str(), nlohmann::json(expected).dump().c_str(), nlohmann::json(actual).dump().c_str());
                return 1;
            }
            ++count;
        }
        std::printf("V4.1 tokenizer: %zu cases match released tokenizer IDs\n", count);
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
