#include "server-common.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

bool server_log_json = false;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::cerr << "CHECK failed at " << __FILE__ << ':' << __LINE__ \
                  << ": " #condition << '\n'; \
        std::abort(); \
    } \
} while (false)

static mtmd_input_chunk * make_image_chunk() {
    json chunk_json = {
        {"type", MTMD_INPUT_CHUNK_TYPE_IMAGE},
        {"tokens_image", {
            {"nx", 2},
            {"ny", 3},
            {"use_mrope_pos", false},
            {"id", "media-prefix-fixture"},
        }},
    };
    return mtmd_input_chunk_from_json(chunk_json);
}

int main() {
    server_tokens tokens(llama_tokens {}, true);
    tokens.push_back(11);
    tokens.push_back(12);

    mtmd_input_chunk * image = make_image_chunk();
    CHECK(image != nullptr);
    CHECK(mtmd_input_chunk_get_n_tokens(image) == 6);
    tokens.push_back(image);
    mtmd_input_chunk_free(image);
    tokens.push_back(13);

    CHECK(tokens.size() == 9);
    CHECK(tokens.has_mtmd_data());
    CHECK(tokens.media_safe_prefix_size(0) == 0);
    CHECK(tokens.media_safe_prefix_size(2) == 2);
    CHECK(tokens.media_safe_prefix_size(3) == 2);
    CHECK(tokens.media_safe_prefix_size(7) == 2);
    CHECK(tokens.media_safe_prefix_size(8) == 8);
    CHECK(tokens.media_safe_prefix_size(9) == 9);
    CHECK(tokens.media_safe_prefix_size(99) == 9);

    bool threw = false;
    try {
        tokens.resize(4);
    } catch (const std::runtime_error &) {
        threw = true;
    }
    CHECK(threw);
    CHECK(tokens.size() == 9);
    CHECK(tokens.has_mtmd_data());

    tokens.shrink_to_media_safe_prefix(8);
    CHECK(tokens.size() == 8);
    CHECK(tokens.has_mtmd_data());
    CHECK(tokens.find_chunk(2) != nullptr);

    tokens.shrink_to_media_safe_prefix(2);
    CHECK(tokens.size() == 2);
    CHECK(!tokens.has_mtmd_data());

    image = make_image_chunk();
    CHECK(image != nullptr);
    tokens.push_back(image);
    mtmd_input_chunk_free(image);
    CHECK(tokens.has_mtmd_data());
    tokens.clear();
    CHECK(tokens.empty());
    CHECK(!tokens.has_mtmd_data());

    std::cout << "server token media-prefix tests: PASS\n";
    return 0;
}
