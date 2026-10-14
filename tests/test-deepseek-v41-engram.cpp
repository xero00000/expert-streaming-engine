#include "llama-deepseek-v41-engram.h"
#include "llama-qwen4exp-ple-storage.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

using nlohmann::json;
using namespace deepseek_v41;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename F> static void rejected(F function) {
    bool threw = false;
    try { function(); } catch (const std::exception &) { threw = true; }
    check(threw, "invalid operation was not rejected");
}

static json read_json(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open fixture: " + path);
    return json::parse(input);
}

static std::vector<uint8_t> unhex(const std::string & text) {
    check(text.size() % 2 == 0, "invalid hex length");
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < text.size(); i += 2) {
        bytes.push_back(uint8_t(std::stoul(text.substr(i, 2), nullptr, 16)));
    }
    return bytes;
}

static engram_hash_config load_config(const json & trace, const std::string & directory) {
    const auto & cfg = trace.at("config");
    engram_hash_config config;
    config.compressed_vocab_size = cfg.at("compressed_vocab_size");
    config.pad_compressed_id = cfg.at("pad_compressed_id");
    size_t count = cfg.at("token_map_count");
    check(count <= 1000000, "unsafe token-map size");
    std::ifstream input(directory + "/" + cfg.at("token_map_file").get<std::string>(), std::ios::binary);
    if (!input) throw std::runtime_error("missing token map");
    for (size_t i = 0; i < count; ++i) {
        std::array<uint8_t, 4> bytes{};
        input.read(reinterpret_cast<char *>(bytes.data()), 4);
        check(bool(input), "truncated token map");
        config.token_map.push_back(uint32_t(bytes[0]) | uint32_t(bytes[1]) << 8 |
                                  uint32_t(bytes[2]) << 16 | uint32_t(bytes[3]) << 24);
    }
    check(input.peek() == std::char_traits<char>::eof(), "unexpected extra token-map data");
    for (size_t l = 0; l < 2; ++l) {
        config.layers[l].layer_id = cfg.at("layer_ids").at(l);
        config.layers[l].rows = cfg.at("num_embeddings").at(l);
        config.layers[l].multipliers = cfg.at("multipliers").at(l).get<std::array<uint64_t, 4>>();
        config.layers[l].primes = cfg.at("primes").at(l).get<std::array<uint64_t, 24>>();
    }
    return config;
}

static void test_hashes(const engram_hash_config & config, const json & trace) {
    engram_sequence_state state(config);
    size_t tokens_tested = 0;
    for (const auto & fixture : trace.at("cases")) {
        const auto ids = fixture.at("token_ids").get<std::vector<uint32_t>>();
        const auto mask = fixture.at("text_mask").get<std::vector<bool>>();
        auto expected = [&](size_t i) { return fixture.at("row_ids").at(i).get<engram_rows>(); };
        state.clear();
        for (size_t i = 0; i < ids.size(); ++i) {
            check(state.accept(10, i, ids[i], mask[i]) == expected(i), "official hash parity failed");
            // Interleave another sequence to expose shared-history contamination.
            check(state.accept(11, i, ids[i], mask[i]) == expected(i), "sequence isolation failed");
            ++tokens_tested;
        }
        for (size_t split = 0; split < ids.size(); ++split) {
            state.rewind(10, split);
            auto saved = state.snapshot(10);
            state.copy(10, 12);
            state.reset(10);
            state.restore(10, saved);
            for (size_t i = split; i < ids.size(); ++i) {
                check(state.accept(10, i, ids[i], mask[i]) == expected(i), "rollback/restore parity failed");
                check(state.accept(12, i, ids[i], mask[i]) == expected(i), "fork parity failed");
            }
        }
        const auto saved = state.snapshot(10);
        rejected([&] { state.accept(10, ids.size() + 1, ids[0]); });
        rejected([&] { state.accept(10, ids.size(), uint32_t(config.token_map.size())); });
        rejected([&] { state.rewind(10, ids.size() + 1); });
        rejected([&] { state.restore(10, {-2}); });
        rejected([&] { state.restore(10, {int32_t(config.compressed_vocab_size)}); });
        check(state.snapshot(10) == saved, "rejected operation changed history");
        state.reset(10);
        check(state.position(10) == 0, "reset did not clear history");
        check(state.position(11) == ids.size(), "reset cleared unrelated sequence");
    }
    auto bad = config;
    bad.layers[0].multipliers[0] = std::numeric_limits<uint64_t>::max();
    rejected([&] { engram_sequence_state invalid(bad); });
    bad = config;
    ++bad.layers[0].rows;
    rejected([&] { engram_sequence_state invalid(bad); });
    bad = config;
    bad.token_map[0] = config.compressed_vocab_size;
    rejected([&] { engram_sequence_state invalid(bad); });
    std::printf("Engram: %zu reference tokens, every suffix rollback, interleaving and restore passed\n", tokens_tested);
}

static void test_rows(const std::string & directory) {
    for (int layer : {1, 14}) {
        auto fixture = read_json(directory + "/engram-layer" + std::to_string(layer) + "-row12345.json");
        auto bytes = unhex(fixture.at("weight_hex"));
        auto scales = unhex(fixture.at("scale_hex"));
        bytes.insert(bytes.end(), scales.begin(), scales.end());
        std::array<float, 256> output{};
        engram_decode_fp8(bytes.data(), bytes.size(), output.data(), output.size());
        check(output == fixture.at("decoded_before_bf16_cast").get<std::array<float, 256>>(),
              "production FP8 row decode parity failed");
        rejected([&] { engram_decode_fp8(bytes.data(), 263, output.data(), output.size()); });
        rejected([&] { engram_decode_fp8(nullptr, 264, output.data(), output.size()); });
    }
    std::printf("Engram: production FP8 rows match independent Torch fixtures\n");
    for (int layer : {1, 14}) {
        auto fixture = read_json(directory + "/engram-affine2-layer" + std::to_string(layer) + "-row12345.json");
        auto bytes = unhex(fixture.at("weight_hex"));
        for (const char * name : {"scale_hex", "bias_hex"}) {
            auto part = unhex(fixture.at(name));
            bytes.insert(bytes.end(), part.begin(), part.end());
        }
        std::array<float, 256> output{};
        engram_decode_affine2(bytes.data(), bytes.size(), output.data(), output.size());
        check(output == fixture.at("decoded_before_bf16_cast").get<std::array<float, 256>>(),
              "production affine-2 row decode parity failed");
        rejected([&] { engram_decode_affine2(bytes.data(), 79, output.data(), output.size()); });
    }
    std::printf("Engram: production affine-2 rows match FP32 fixture values\n");
}

class memory_source final : public llama_expert_cache::source {
public:
    explicit memory_source(std::vector<uint8_t> data) : data_(std::move(data)) {}
    uint32_t id() const override { return 41; }
    llama_expert_cache::source_identity identity() const override { return {41, 1}; }
    uint64_t size() const override { return data_.size(); }
    llama_expert_cache::read_backend backend() const override { return llama_expert_cache::read_backend::pread; }
    void read(uint64_t offset, void * destination, size_t count) const override {
        if (offset > data_.size() || count > data_.size() - offset) throw std::out_of_range("source read");
        std::memcpy(destination, data_.data() + offset, count);
    }
private:
    std::vector<uint8_t> data_;
};

static void test_raw_store() {
    using namespace qwen4exp_ple_storage;
    for (size_t row_bytes : {size_t(80), size_t(264)}) {
        constexpr size_t prefix = 4090; // first row crosses a 4 KiB page boundary
        std::vector<uint8_t> data(prefix + 3 * row_bytes);
        for (size_t i = prefix; i < data.size(); ++i) data[i] = uint8_t(i * 41);
        auto source = std::make_shared<memory_source>(data);
        fixed_row_layout layout{prefix, 3, row_bytes};
        row_store store(source, layout, 2, row_bytes * 2);
        auto first = store.submit(1, {0, 1, 0}).get();
        check(first.size() == 3, "raw row count");
        check(std::memcmp(first.row(0).data(), data.data() + prefix, row_bytes) == 0, "raw row bytes");
        check(first.row(0).data() == first.row(2).data(), "duplicate row lease did not coalesce");
        auto warm = store.submit(2, {1}).get();
        check(store.stats().reads.source_reads == 2, "warm raw row caused source I/O");
        rejected([&] { (void) store.submit(3, {2}).get(); });
        check(std::memcmp(first.row(0).data(), data.data() + prefix, row_bytes) == 0,
              "pinned-capacity rejection corrupted a row");
        first = row_store::staged_batch{};
        warm = row_store::staged_batch{};
        auto replacement = store.submit(4, {2}).get();
        check(std::memcmp(replacement.row(0).data(), data.data() + prefix + 2 * row_bytes, row_bytes) == 0,
              "raw row eviction/reload corrupted bytes");
        auto cancelled = store.submit(5, {2});
        auto survivor = store.submit(6, {2});
        check(store.cancel_owner(5) == 1, "raw row cancellation owner");
        rejected([&] { (void) cancelled.get(); });
        check(survivor.get().size() == 1, "raw row cancellation affected another owner");
        rejected([&] { (void) store.submit(7, {3}); });
        rejected([&] { row_store invalid(source, fixed_row_layout{prefix, 4, row_bytes}, 1, row_bytes); });
        rejected([&] { row_store invalid(source, fixed_row_layout{UINT64_MAX, 3, row_bytes}, 1, row_bytes); });
        rejected([&] { row_store invalid(source, fixed_row_layout{prefix, 3, 0}, 1, row_bytes); });
        check(store.stats().rows.resident_bytes <= row_bytes * 2, "raw row cache exceeded capacity");
    }
    std::printf("Engram: bounded 80/264-byte row I/O, coalescing, leases, cancellation and eviction passed\n");
}

int main(int argc, char ** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: test-deepseek-v41-engram FIXTURE_DIRECTORY");
        const std::string directory = argv[1];
        const auto trace = read_json(directory + "/engram-hash-reference.json");
        test_hashes(load_config(trace, directory), trace);
        test_rows(directory);
        test_raw_store();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
