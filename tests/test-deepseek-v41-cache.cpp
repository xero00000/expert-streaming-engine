#include "llama-deepseek-v41-cache.h"
#include "../ggml/src/ggml-qat.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <stdexcept>
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

using namespace deepseek_v41;
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template <typename F> static void rejects(F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected, "invalid cache operation accepted");
}
static nlohmann::json read(const char * path) {
    std::ifstream file(path); return nlohmann::json::parse(file);
}
int main(int argc, char ** argv) {
    try {
        check(argc == 3 || argc == 4, "usage: test-deepseek-v41-cache CONFIG.json QAT_REFERENCE.json [CUDA_DEVICE]");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(nullptr, ggml_backend_free);
        ggml_backend_buffer_type_t cache_type = nullptr;
        if (argc == 4) {
#ifdef GGML_USE_CUDA
            backend.reset(ggml_backend_cuda_init(std::stoi(argv[3]), nullptr, nullptr));
            check(bool(backend), "CUDA backend unavailable");
            cache_type = ggml_backend_get_default_buffer_type(backend.get());
#else
            throw std::invalid_argument("CUDA support is not compiled");
#endif
        }
        const auto fixture = read(argv[2]);
        for (const auto & item : fixture.at("cases")) {
            const auto format = ggml_qat_format(item.at("mode").get<int>());
            const auto source = item.at("inputs").at("x").at("values").get<std::vector<float>>();
            const auto expected = item.at("expected").at("output").at("values").get<std::vector<float>>();
            const size_t width = item.at("inputs").at("x").at("shape").at(0);
            std::vector<uint8_t> bytes(ggml_qat_row_size(width, format));
            std::vector<float> decoded(width);
            for (size_t start = 0; start < source.size(); start += width) {
                check(ggml_qat_pack_row(source.data() + start, bytes.data(), width, format), "row packing failed");
                check(ggml_qat_unpack_row(bytes.data(), decoded.data(), width, format), "row unpacking failed");
                for (size_t i = 0; i < width; ++i) check(decoded[i] == expected[start + i], "packed QAT reference mismatch");
            }
        }
        // Independent wire-format check, not just pack/unpack round trips.
        std::vector<float> levels = {0, .5f, 1, 1.5f, 2, 3, 4, 6, -0.0f, -.5f, -1, -1.5f, -2, -3, -4, -6};
        std::vector<uint8_t> bytes(9);
        check(ggml_qat_pack_row(levels.data(), bytes.data(), 16, GGML_QAT_FP4_E4M3), "FP4 code packing failed");
        check(bytes == std::vector<uint8_t>({0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x38}), "FP4 wire format mismatch");
        check(ggml_qat_row_size(512, GGML_QAT_FP4_E4M3) == 288 &&
              ggml_qat_row_size(128, GGML_QAT_FP4_E8M0) == 68 &&
              ggml_qat_row_size(512, GGML_QAT_FP8_E8M0) == 528, "production packed row sizes incorrect");
        bytes[8] = 0;
        check(!ggml_qat_unpack_row(bytes.data(), levels.data(), 16, GGML_QAT_FP4_E4M3), "zero scale accepted");

        const auto c = read(argv[1]).at("text_config");
        attention_layout_config config;
        config.compress_ratios = c.at("compress_ratios").get<std::vector<uint32_t>>();
        config.compress_ratios.resize(c.at("num_hidden_layers").get<size_t>());
        config.kv_source_layers = c.at("kv_source_layer_ids").get<std::vector<uint32_t>>();
        config.index_source_layers = c.at("index_source_layer_ids").get<std::vector<uint32_t>>();
        config.candidate_source_layer = c.at("candidate_source_layer_id");
        config.head_dim = c.at("head_dim"); config.index_head_dim = c.at("index_head_dim");
        config.sliding_window = 4; // wrap repeatedly while retaining production row geometry/ownership
        attention_layout layout(config);
        attention_cache cache(config, 16, cache_type);
        const auto budget = layout.payload_budget(16, kv_encoding::fp4_compressed);
        check(cache.reserved_payload_bytes() == budget.global_bytes + budget.sliding_window_bytes, "reserved cache payload mismatch");
        std::vector<float> kv(config.head_dim), key(config.index_head_dim);
        auto append_to = [&](attention_cache & target, uint32_t position) {
            for (uint32_t layer = 0; layer < config.compress_ratios.size(); ++layer) {
                std::fill(kv.begin(), kv.end(), float(position + 1));
                target.append_window(layer, position, kv);
                if (layout.layers()[layer].kv_owner == int32_t(layer) && (position + 1) % config.compress_ratios[layer] == 0) {
                    std::fill(kv.begin(), kv.end(), float(layer + 1));
                    std::fill(key.begin(), key.end(), float(layer + 1));
                    target.append_global(layer, position / config.compress_ratios[layer], kv, key);
                }
            }
        };
        auto append = [&](uint32_t position) { append_to(cache, position); };
        cache.begin();
        for (uint32_t position = 0; position < 6; ++position) append(position);
        cache.commit();
        const auto old_window = cache.read_window(0);
        check(old_window.front() == 3 && old_window.back() == 6, "sliding window chronology incorrect");
        const auto owner2 = cache.read_global(2, true, {0});
        const auto owner8 = cache.read_global(8, true, {0});
        check(owner2 != owner8 && cache.read_global(3, true, {0}) == owner2, "global index key owners aliased");
        cache.begin(); append(6); // odd half-rate step must keep the same owner despite no publication
        check(cache.read_global(3, true, {0}) == owner2, "incomplete group changed index-key owner");
        append(7); cache.abort();
        check(cache.window_end(0) == 6 && cache.read_window(0) == old_window && cache.global_rows(2) == 3,
              "cache abort did not restore ring/global state");
        cache.begin();
        cache.append_window(0, 6, kv);
        rejects([&] { cache.commit(); });
        cache.abort();
        cache.begin();
        rejects([&] { cache.append_global(3, 3, kv, key); });
        auto invalid = key; invalid[0] = std::numeric_limits<float>::quiet_NaN();
        rejects([&] { cache.append_global(2, 3, kv, invalid); });
        check(cache.global_rows(2) == 3, "failed paired publication changed main cache");
        cache.abort();
        for (uint32_t accepted = 0; accepted <= 8; ++accepted) {
            cache.clear(); cache.begin();
            for (uint32_t position = 0; position < 6; ++position) append(position);
            cache.commit(); cache.save_checkpoint(8);
            // Abort must rewind both the live cache and its checkpoint journal.
            cache.begin();
            for (uint32_t position = 6; position < 9; ++position) append(position);
            cache.abort();
            cache.begin();
            rejects([&] { cache.save_checkpoint(); });
            for (uint32_t position = 6; position < 14; ++position) append(position);
            cache.commit();
            rejects([&] { cache.restore_checkpoint(15); });
            cache.restore_checkpoint(6 + accepted);
            check(!cache.has_checkpoint(), "restored checkpoint was not consumed");
            attention_cache expected(config, 16);
            expected.begin();
            for (uint32_t position = 0; position < 6 + accepted; ++position) append_to(expected, position);
            expected.commit();
            for (uint32_t layer = 0; layer < config.compress_ratios.size(); ++layer) {
                check(cache.window_end(layer) == expected.window_end(layer) &&
                      cache.read_window(layer) == expected.read_window(layer), "checkpoint window prefix mismatch");
                if (layout.layers()[layer].kv_owner >= 0) {
                    check(cache.global_rows(layer) == expected.global_rows(layer), "checkpoint compressed row count mismatch");
                    std::vector<uint32_t> rows(cache.global_rows(layer));
                    for (uint32_t i = 0; i < rows.size(); ++i) rows[i] = i;
                    for (bool index_keys : {false, true})
                        check(cache.read_global(layer, index_keys, rows) == expected.read_global(layer, index_keys, rows),
                              "checkpoint compressed prefix mismatch");
                }
            }
        }
        cache.clear(); cache.begin();
        for (uint32_t position = 0; position < 6; ++position) append(position);
        cache.commit();
        cache.begin();
        for (uint32_t position = 6; position < 16; ++position) append(position);
        cache.commit();
        check(cache.packed_payload_bytes() == budget.global_bytes + budget.sliding_window_bytes, "actual packed payload differs from planner");
        if (cache_type) {
            auto full_config = config;
            full_config.sliding_window = c.at("sliding_window");
            const uint32_t maximum = c.at("max_position_embeddings");
            attention_layout full_layout(full_config);
            const auto full_budget = full_layout.payload_budget(maximum, kv_encoding::fp4_compressed);
            const size_t full_bytes = full_budget.global_bytes + full_budget.sliding_window_bytes;
            attention_cache full_cache(full_config, maximum, cache_type);
            check(full_cache.reserved_payload_bytes() == full_bytes, "maximum-context GPU allocation differs from planner");
            // Exercise every byte of a device allocation of the same size,
            // including its final byte, without pretending to evaluate 1M tokens.
            packed_cache_storage storage(cache_type, full_bytes);
            storage.resize(full_bytes);
            std::vector<uint8_t> input(4 * 1024 * 1024), output(input.size());
            for (size_t offset = 0; offset < full_bytes; offset += input.size()) {
                const size_t count = std::min(input.size(), full_bytes - offset);
                for (size_t i = 0; i < count; ++i) input[i] = uint8_t((i * 17 + offset / 4096) % 251);
                storage.write(input.data(), offset, count);
                storage.read(output.data(), offset, count);
                check(std::equal(input.begin(), input.begin() + count, output.begin()), "full-capacity device storage mismatch");
            }
            rejects([&] { storage.resize(full_bytes + 1); });
            rejects([&] { storage.read(output.data(), full_bytes, 1); });
            rejects([&] { storage.write(input.data(), full_bytes, 1); });
            std::printf("Maximum context %u: %zu packed bytes allocated on %s; complete capacity byte I/O passed (allocation/storage test, not a full-context inference test)\n",
                    maximum, full_bytes, ggml_backend_buft_name(cache_type));
        }
        std::printf("V4.1 packed cache: exact QAT parity, ring wrap, ownership, atomic publication, abort and accepted-prefix checkpoints pass; payload=%llu bytes\n",
                    (unsigned long long) cache.packed_payload_bytes());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what()); return 1;
    }
}
