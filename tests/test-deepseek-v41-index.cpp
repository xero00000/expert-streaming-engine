#include "llama-deepseek-v41-index.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <cstdio>
#include <limits>
#include <stdexcept>

using namespace deepseek_v41;
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char ** argv) {
    try {
        check(argc == 2, "usage: test-deepseek-v41-index REFERENCE.json");
        std::ifstream input(argv[1]);
        const auto fixture = nlohmann::json::parse(input);
        for (const auto & item : fixture.at("selections")) {
            const auto scores = item.at("scores").get<std::vector<float>>();
            const auto keep = select_candidate_blocks(scores, item.at("visible"), item.at("block_size"), item.at("top_blocks"));
            check(keep == item.at("candidates").get<std::vector<uint8_t>>(), "official candidate mask mismatch");
            check(select_index_positions(scores, item.at("visible"), item.at("top_k"), item.at("offset"), keep) ==
                  item.at("positions").get<std::vector<int32_t>>(), "top-k positions mismatch");
        }
        check(select_index_positions({1, 1, 1, 1}, 4, 2, 5) == std::vector<int32_t>({5, 6}), "nondeterministic ties");
        check(select_index_positions({1, 2}, 0, 2, 0) == std::vector<int32_t>({-1, -1}), "future row became visible");
        check(select_candidate_blocks({100, 99, 98, 97, -100, -101}, 5, 4, 1) ==
              std::vector<uint8_t>({0, 0, 0, 0, 1, 1}), "newest partial block was not pinned");
        bool rejected = false;
        try { select_index_positions({std::numeric_limits<float>::quiet_NaN()}, 1, 1, 0); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "NaN score accepted");
        std::printf("V4.1 candidate/top-k selection: %zu official cases pass\n", fixture.at("selections").size());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
