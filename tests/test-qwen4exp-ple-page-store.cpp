#include "llama-qwen4exp-ple-storage.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;
using namespace qwen4exp_ple_storage;

static uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') return uint8_t(value - '0');
    if (value >= 'a' && value <= 'f') return uint8_t(value - 'a' + 10);
    throw std::invalid_argument("invalid fixture hex digit");
}

static std::vector<uint8_t> decode_hex(const std::string & encoded) {
    if (encoded.size() % 2 != 0) throw std::invalid_argument("invalid fixture hex length");
    std::vector<uint8_t> result(encoded.size() / 2);
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = uint8_t(hex_nibble(encoded[2 * index]) << 4) |
            hex_nibble(encoded[2 * index + 1]);
    }
    return result;
}

class memory_source final : public llama_expert_cache::source {
public:
    explicit memory_source(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

    uint32_t id() const override { return 10; }
    llama_expert_cache::source_identity identity() const override { return {0x1111, 0x2222}; }
    uint64_t size() const override { return bytes_.size(); }
    llama_expert_cache::read_backend backend() const override {
        return llama_expert_cache::read_backend::pread;
    }
    void read(uint64_t offset, void * destination, size_t length) const override {
        if (offset > bytes_.size() || length > bytes_.size() - offset) {
            throw std::out_of_range("page test source read is out of range");
        }
        std::memcpy(destination, bytes_.data() + offset, length);
    }

private:
    std::vector<uint8_t> bytes_;
};

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s PLE_ROW_FIXTURE.json\n", argv[0]);
        return 1;
    }
    std::ifstream input(argv[1]);
    const json fixture = json::parse(input);
    const auto quantized = decode_hex(fixture.at("quantized_hex").get<std::string>());
    if (quantized.size() != 90) return 1;

    // Row zero starts six bytes before a page boundary. The source also ends
    // on a partial page, exercising straddle reconstruction and padded fill.
    constexpr uint64_t data_offset = 4090;
    std::vector<uint8_t> source_bytes(data_offset, 0);
    source_bytes.insert(source_bytes.end(), quantized.begin(), quantized.end());
    source_bytes.insert(source_bytes.end(), quantized.begin(), quantized.end());
    auto source = std::make_shared<memory_source>(std::move(source_bytes));
    qwen4exp_ple_hash_config hash_config;
    qwen4exp_ple_extent_index extents(hash_config, 0, data_offset, 90, 2);
    page_store store(source, extents, 2, 8192);

    llama_backend_init();
    auto first = store.submit(1, {0, 1, 0}).get();
    std::vector<uint8_t> copied(90);
    first.copy_quantized(0, copied.data(), copied.size());
    if (first.size() != 3 || copied != quantized) {
        std::fprintf(stderr, "PLE page store failed to reconstruct a straddling row\n");
        return 1;
    }

    std::vector<float> values(first.size() * 160);
    first.dequantize(values.data(), values.size());
    const json & expected = fixture.at("dequantized_values");
    for (size_t row = 0; row < first.size(); ++row) {
        for (size_t value = 0; value < 160; ++value) {
            if (values[row * 160 + value] != expected.at(value).get<float>()) {
                std::fprintf(stderr, "PLE page store dequantization mismatch\n");
                return 1;
            }
        }
    }

    auto second = store.submit(2, {0, 1}).get();
    const page_store_stats warm = store.stats();
    if (warm.reads.submitted_batches != 1 || warm.reads.logical_reads != 2 ||
            warm.reads.source_reads != 2 || warm.reads.source_bytes != source->size() ||
            warm.reads.failed_reads != 0 || warm.reads.staging_resident_bytes != 0 ||
            warm.pages.capacity_bytes != 8192 || warm.pages.resident_bytes != 8192 ||
            warm.pages.admissions != 2 || warm.pages.misses != 2 || warm.pages.hits != 2 ||
            warm.pages.active_leases != 4) {
        std::fprintf(stderr, "incorrect PLE page store cold/warm telemetry\n");
        return 1;
    }

    auto cancelled = store.submit(3, {0});
    if (store.cancel_owner(3) != 1) {
        std::fprintf(stderr, "PLE page store did not cancel its owner\n");
        return 1;
    }
    bool rejected = false;
    try {
        (void) cancelled.get();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "cancelled PLE page batch produced staged rows\n");
        return 1;
    }

    two_tier_store tiered(source, extents, 2, 8192, 180);
    auto tiered_first = tiered.submit(11, {0, 1, 0}).get();
    std::vector<float> tiered_values(tiered_first.size() * 160);
    tiered_first.dequantize(tiered_values.data(), tiered_values.size());
    for (size_t row = 0; row < tiered_first.size(); ++row) {
        for (size_t value = 0; value < 160; ++value) {
            if (tiered_values[row * 160 + value] != expected.at(value).get<float>()) {
                std::fprintf(stderr, "PLE two-tier dequantization mismatch\n");
                return 1;
            }
        }
    }

    auto tiered_second = tiered.submit(12, {0, 1}).get();
    const two_tier_store_stats tiered_warm = tiered.stats();
    if (tiered_warm.reads.logical_reads != 2 || tiered_warm.reads.source_reads != 2 ||
            tiered_warm.reads.failed_reads != 0 || tiered_warm.pages.misses != 2 ||
            tiered_warm.pages.hits != 0 || tiered_warm.pages.admissions != 2 ||
            tiered_warm.pages.resident_bytes != 8192 || tiered_warm.rows.misses != 3 ||
            tiered_warm.rows.hits != 2 || tiered_warm.rows.admissions != 2 ||
            tiered_warm.rows.resident_bytes != 180 || tiered_warm.rows.active_leases != 5) {
        std::fprintf(stderr, "incorrect PLE two-tier cold/warm telemetry\n");
        return 1;
    }

    auto tiered_cancelled = tiered.submit(13, {0});
    if (tiered.cancel_owner(13) != 1) {
        std::fprintf(stderr, "PLE two-tier store did not cancel its owner\n");
        return 1;
    }
    rejected = false;
    try {
        (void) tiered_cancelled.get();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "cancelled PLE two-tier batch produced staged rows\n");
        return 1;
    }

    llama_backend_free();
    return 0;
}
