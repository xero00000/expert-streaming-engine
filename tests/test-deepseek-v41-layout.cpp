#include "llama-deepseek-v41-layout.h"
#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <stdexcept>

using namespace deepseek_v41;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename F> static void rejects(F function) {
    bool threw = false;
    try { function(); } catch (const std::exception &) { threw = true; }
    check(threw, "invalid attention configuration was accepted");
}

int main(int argc, char ** argv) {
    try {
        check(argc == 2, "usage: test-deepseek-v41-layout OFFICIAL_CONFIG.json");
        std::ifstream input(argv[1]);
        const auto config = nlohmann::json::parse(input).at("text_config");
        attention_layout_config spec;
        spec.compress_ratios = config.at("compress_ratios").get<std::vector<uint32_t>>();
        spec.compress_ratios.resize(config.at("num_hidden_layers").get<size_t>());
        spec.kv_source_layers = config.at("kv_source_layer_ids").get<std::vector<uint32_t>>();
        spec.index_source_layers = config.at("index_source_layer_ids").get<std::vector<uint32_t>>();
        spec.candidate_source_layer = config.at("candidate_source_layer_id");
        spec.head_dim = config.at("head_dim");
        spec.index_head_dim = config.at("index_head_dim");
        spec.sliding_window = config.at("sliding_window");
        attention_layout layout(spec);
        check(layout.layers().size() == 40, "DSpark stages became main layers");
        int full = 0, reindex = 0, reuse = 0, window = 0;
        for (const auto & layer : layout.layers()) {
            full += layer.mode == attention_mode::full;
            reindex += layer.mode == attention_mode::reindex;
            reuse += layer.mode == attention_mode::reuse;
            window += layer.mode == attention_mode::window_only;
        }
        check(full == 4 && reindex == 4 && reuse == 30 && window == 2, "incorrect V4.1 attention modes");
        check(layout.layers()[14].index_key_owner == 14 && layout.layers()[19].kv_owner == 14,
              "encoder key ownership incorrect");
        check(layout.layers()[24].kv_owner == 20 && layout.layers()[24].index_key_owner == 20 &&
              layout.layers()[24].topk_owner == 24 && layout.layers()[27].topk_owner == 24 &&
              layout.layers()[39].candidate_owner == 20, "decoder reindex ownership incorrect");
        const auto million = layout.payload_budget(1000000, kv_encoding::fp4_compressed);
        check(million.main_kv_bytes == 720000000 && million.index_key_bytes == 170000000 &&
              million.global_bytes == 890000000, "890 MB global payload derivation failed");
        check(million.sliding_window_bytes == 2703360, "FP8 window payload derivation failed");
        check(layout.payload_budget(1048576, kv_encoding::fp4_compressed).global_bytes == 933232640,
              "configured max context payload incorrect");
        check(layout.payload_budget(1, kv_encoding::fp4_compressed).global_bytes == 356 &&
              layout.payload_budget(3, kv_encoding::fp4_compressed).global_bytes == 2136,
              "incomplete compression groups overallocated");
        const auto reference = layout.payload_budget(2048, kv_encoding::fp32_reference);
        check(reference.global_bytes == 13107200 && reference.sliding_window_bytes == 10485760,
              "reference FP32 payload incorrect");
        rejects([&] { (void) layout.payload_budget(UINT64_MAX, kv_encoding::fp4_compressed); });
        rejects([&] { (void) layout.payload_budget(0, kv_encoding::fp32_reference); });
        auto bad = spec;
        bad.index_source_layers.erase(bad.index_source_layers.begin());
        rejects([&] { attention_layout invalid(bad); });
        bad = spec;
        bad.compress_ratios[3] = 1;
        rejects([&] { attention_layout invalid(bad); });
        bad = spec;
        bad.kv_source_layers.push_back(20);
        rejects([&] { attention_layout invalid(bad); });
        bad = spec;
        bad.candidate_source_layer = 21;
        rejects([&] { attention_layout invalid(bad); });
        std::puts("V4.1 ownership, sharing modes, exact payload budgets and invalid configurations pass");
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
