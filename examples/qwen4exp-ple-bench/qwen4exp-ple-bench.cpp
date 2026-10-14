#include "llama-expert-cache.h"
#include "llama-qwen4exp-ple-storage.h"
#include "llama-qwen4exp-ple.h"
#include "llama.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using namespace qwen4exp_ple_storage;

struct options {
    std::string shard;
    std::string manifest;
    std::string trace;
    std::string backend = "pread";
    std::string storage = "row";
    std::string cache_state = "warm";
    uint64_t cache_mib = 1;
    size_t workers = 4;
    size_t iterations = 2;
    size_t sessions = 1;
    uint64_t overlap_us = 0;
    size_t pressure_tokens = 0;
    uint64_t pressure_seed = 0x4a9d7c31ULL;
};

static options parse_options(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto value = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument("missing value for " + arg);
            return argv[index];
        };
        if (arg == "--shard") result.shard = value();
        else if (arg == "--manifest") result.manifest = value();
        else if (arg == "--trace") result.trace = value();
        else if (arg == "--backend") result.backend = value();
        else if (arg == "--storage") result.storage = value();
        else if (arg == "--cache-state") result.cache_state = value();
        else if (arg == "--cache-mib") result.cache_mib = std::stoull(value());
        else if (arg == "--workers") result.workers = std::stoull(value());
        else if (arg == "--iterations") result.iterations = std::stoull(value());
        else if (arg == "--sessions") result.sessions = std::stoull(value());
        else if (arg == "--overlap-us") result.overlap_us = std::stoull(value());
        else if (arg == "--pressure-tokens") result.pressure_tokens = std::stoull(value());
        else if (arg == "--pressure-seed") result.pressure_seed = std::stoull(value());
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (result.shard.empty() || result.manifest.empty() || result.trace.empty()) {
        throw std::invalid_argument("--shard, --manifest, and --trace are required");
    }
    if (result.cache_mib == 0 || result.workers == 0 || result.iterations == 0 || result.sessions == 0) {
        throw std::invalid_argument("cache, workers, iterations, and sessions must be nonzero");
    }
    if (result.storage != "row" && result.storage != "page" && result.storage != "two-tier") {
        throw std::invalid_argument("storage must be row, page, or two-tier");
    }
    if (result.cache_state != "warm" && result.cache_state != "cold") {
        throw std::invalid_argument("cache-state must be warm or cold");
    }
    return result;
}

static json read_json(const std::string & path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open benchmark metadata");
    return json::parse(input);
}

static uint64_t percentile(std::vector<uint64_t> samples, size_t numerator) {
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    const size_t rank = (samples.size() * numerator + 99) / 100;
    return samples[std::min(samples.size() - 1, std::max<size_t>(1, rank) - 1)];
}

static llama_expert_cache::read_backend parse_backend(const std::string & value) {
    if (value == "mmap") return llama_expert_cache::read_backend::mmap;
    if (value == "pread") return llama_expert_cache::read_backend::pread;
    if (value == "io_uring") return llama_expert_cache::read_backend::io_uring;
    throw std::invalid_argument("backend must be mmap, pread, or io_uring");
}

static uint64_t splitmix64(uint64_t & state) {
    uint64_t value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        const json manifest = read_json(opts.manifest);
        const json trace = read_json(opts.trace);
        const json & ple_manifest = manifest.at("ple");
        const json & ple_trace = trace.at("ple");

        if (std::filesystem::path(opts.shard).filename() != ple_manifest.at("shard").get<std::string>()) {
            throw std::runtime_error("selected shard does not contain the manifest PLE tensor");
        }
        const auto file_size = std::filesystem::file_size(opts.shard);
        const auto & files = manifest.at("files");
        const auto file_entry = std::find_if(files.begin(), files.end(), [&](const json & value) {
            return value.at("name") == ple_manifest.at("shard");
        });
        if (file_entry == files.end() || file_size != file_entry->at("size_bytes").get<uint64_t>()) {
            throw std::runtime_error("selected shard size does not match the production manifest");
        }

        qwen4exp_ple_hash_config hash_config;
        hash_config.eos_token_id = ple_trace.at("eos_token_id").get<uint32_t>();
        for (size_t index = 0; index < hash_config.layer_multipliers.size(); ++index) {
            hash_config.layer_multipliers[index] =
                ple_trace.at("layer_multipliers").at(index).get<uint64_t>();
        }
        for (size_t head = 0; head < hash_config.head_offsets.size(); ++head) {
            hash_config.head_offsets[head] = ple_trace.at("head_offsets").at(head).get<uint64_t>();
            hash_config.head_vocab_sizes[head] =
                ple_trace.at("head_vocab_sizes").at(head).get<uint64_t>();
        }

        std::vector<std::array<uint64_t, 16>> reference_rows;
        for (const json & sequence : trace.at("sequences")) {
            qwen4exp_ple_hasher hasher(hash_config);
            const json & tokens = sequence.at("token_ids");
            const json & expected = sequence.at("row_ids");
            for (size_t token = 0; token < tokens.size(); ++token) {
                const auto rows = hasher.accept(tokens.at(token).get<uint32_t>());
                for (size_t head = 0; head < rows.size(); ++head) {
                    if (rows[head] != expected.at(token).at(head).get<uint64_t>()) {
                        throw std::runtime_error("PLE hash replay diverged before benchmark");
                    }
                }
                reference_rows.push_back(rows);
            }
        }

        std::vector<std::vector<std::array<uint64_t, 16>>> session_rows(opts.sessions);
        if (opts.pressure_tokens == 0) {
            for (auto & rows : session_rows) rows = reference_rows;
        } else {
            // Independent, reproducible session histories defeat the exact-read
            // coalescing pattern used by the short reference replay. Periodic
            // PLE-EOS tokens also exercise segment resets in long prompts.
            for (size_t session = 0; session < opts.sessions; ++session) {
                qwen4exp_ple_hasher hasher(hash_config);
                uint64_t state = opts.pressure_seed ^ (0xd1b54a32d192ed03ULL * (session + 1));
                auto & rows = session_rows[session];
                rows.reserve(opts.pressure_tokens);
                for (size_t token = 0; token < opts.pressure_tokens; ++token) {
                    const uint32_t token_id = token != 0 && token % 257 == 0
                        ? hash_config.eos_token_id
                        : uint32_t(splitmix64(state) % 248320ULL);
                    rows.push_back(hasher.accept(token_id));
                }
            }
        }
        const size_t tokens_per_session = session_rows.front().size();

        const auto backend = parse_backend(opts.backend);
        const auto identity = llama_expert_cache::identify_file_source(opts.shard, 1);
        auto source = std::make_shared<llama_expert_cache::file_source>(
            1, identity, opts.shard, backend);
        if (opts.cache_state == "cold") {
            source->evict_range_from_os_cache(
                ple_manifest.at("data_offset").get<uint64_t>(),
                ple_manifest.at("size_bytes").get<uint64_t>());
        }
        qwen4exp_ple_extent_index extents(
            hash_config,
            file_entry->at("split.no").get<uint16_t>(),
            ple_manifest.at("data_offset").get<uint64_t>(),
            ple_manifest.at("row_size_bytes").get<uint64_t>(),
            ple_manifest.at("row_count").get<uint64_t>());
        llama_backend_init();
        std::vector<uint64_t> token_latency_ns;
        token_latency_ns.reserve(tokens_per_session * opts.iterations * opts.sessions);
        std::vector<uint64_t> ple_wait_ns;
        ple_wait_ns.reserve(tokens_per_session * opts.iterations * opts.sessions);
        std::vector<float> dequantized(16 * 160);
        auto replay = [&](auto & store) {
            for (size_t iteration = 0; iteration < opts.iterations; ++iteration) {
                for (size_t token = 0; token < tokens_per_session; ++token) {
                    const auto token_started = std::chrono::steady_clock::now();
                    using ticket_type = decltype(store.submit(uint64_t(0), std::vector<uint64_t>{}));
                    std::vector<ticket_type> tickets;
                    tickets.reserve(opts.sessions);
                    for (size_t session = 0; session < opts.sessions; ++session) {
                        const auto & rows = session_rows[session][token];
                        const std::vector<uint64_t> requested(rows.begin(), rows.end());
                        tickets.push_back(store.submit(session + 1, requested));
                    }
                    if (opts.overlap_us != 0) {
                        std::this_thread::sleep_for(std::chrono::microseconds(opts.overlap_us));
                    }
                    for (const auto & ticket : tickets) {
                        const auto wait_started = std::chrono::steady_clock::now();
                        auto staged = ticket.get();
                        ple_wait_ns.push_back(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - wait_started).count()));
                        staged.dequantize(dequantized.data(), dequantized.size());
                        token_latency_ns.push_back(uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - token_started).count()));
                    }
                }
            }
        };

        async_read_stats read_stats;
        block_cache_stats cache_stats;
        block_cache_stats secondary_cache_stats;
        std::string cache_name;
        std::string secondary_cache_name;
        uint64_t elapsed_ns = 0;
        if (opts.storage == "row") {
            row_store store(source, extents, opts.workers, opts.cache_mib * 1024 * 1024);
            const auto started = std::chrono::steady_clock::now();
            replay(store);
            elapsed_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
            const row_store_stats measured = store.stats();
            read_stats = measured.reads;
            cache_stats = measured.rows;
            cache_name = "row_cache";
        } else if (opts.storage == "page") {
            page_store store(source, extents, opts.workers, opts.cache_mib * 1024 * 1024);
            const auto started = std::chrono::steady_clock::now();
            replay(store);
            elapsed_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
            const page_store_stats measured = store.stats();
            read_stats = measured.reads;
            cache_stats = measured.pages;
            cache_name = "page_cache";
        } else {
            const uint64_t total_cache_bytes = opts.cache_mib * 1024 * 1024;
            const uint64_t page_cache_bytes = total_cache_bytes / 2;
            const uint64_t row_cache_bytes = total_cache_bytes - page_cache_bytes;
            two_tier_store store(
                source, extents, opts.workers, page_cache_bytes, row_cache_bytes);
            const auto started = std::chrono::steady_clock::now();
            replay(store);
            elapsed_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count());
            const two_tier_store_stats measured = store.stats();
            read_stats = measured.reads;
            cache_stats = measured.pages;
            secondary_cache_stats = measured.rows;
            cache_name = "page_cache";
            secondary_cache_name = "row_cache";
        }
        llama_backend_free();

        const uint64_t token_count = tokens_per_session * opts.iterations * opts.sessions;
        json output = {
            {"schema_version", 1},
            {"architecture", "qwen4exp"},
            {"backend", opts.backend},
            {"storage_mode", opts.storage},
            {"cache_state", opts.cache_state},
            {"os_cache_control", opts.cache_state == "cold"
                ? (opts.backend == "mmap" ? "madvise+posix_fadvise_dontneed" : "posix_fadvise_dontneed")
                : "none"},
            {"cache_mib", opts.cache_mib},
            {"workers", opts.workers},
            {"iterations", opts.iterations},
            {"sessions", opts.sessions},
            {"synthetic_overlap_us", opts.overlap_us},
            {"workload", opts.pressure_tokens == 0 ? "reference-trace" : "deterministic-pressure"},
            {"pressure_tokens_per_session", opts.pressure_tokens},
            {"pressure_seed", opts.pressure_seed},
            {"tokens", token_count},
            {"logical_quantized_bytes", token_count * 16 * 90},
            {"elapsed_ns", elapsed_ns},
            {"tokens_per_second", elapsed_ns == 0 ? 0.0 : 1.0e9 * double(token_count) / double(elapsed_ns)},
            {"token_latency_ns", {
                {"p50", percentile(token_latency_ns, 50)},
                {"p95", percentile(token_latency_ns, 95)},
                {"p99", percentile(token_latency_ns, 99)}
            }},
            {"ple_wait_ns", {
                {"p50", percentile(ple_wait_ns, 50)},
                {"p95", percentile(ple_wait_ns, 95)},
                {"p99", percentile(ple_wait_ns, 99)}
            }},
            {"reads", {
                {"logical_reads", read_stats.logical_reads},
                {"logical_bytes", read_stats.logical_bytes},
                {"source_reads", read_stats.source_reads},
                {"source_bytes", read_stats.source_bytes},
                {"source_batches", read_stats.source_batches},
                {"peak_source_batch_width", read_stats.peak_source_batch_width},
                {"coalesced_reads", read_stats.coalesced_reads},
                {"failed_reads", read_stats.failed_reads},
                {"peak_queue_depth", read_stats.peak_queue_depth},
                {"staging_capacity_bytes", read_stats.staging_capacity_bytes},
                {"staging_resident_bytes", read_stats.staging_resident_bytes},
                {"staging_peak_resident_bytes", read_stats.staging_peak_resident_bytes},
                {"staging_rejected_batches", read_stats.staging_rejected_batches},
                {"submit_to_ready_p50_ns", read_stats.submit_to_ready_p50_ns},
                {"submit_to_ready_p95_ns", read_stats.submit_to_ready_p95_ns},
                {"submit_to_ready_p99_ns", read_stats.submit_to_ready_p99_ns}
            }}
        };
        output[cache_name] = {
            {"capacity_bytes", cache_stats.capacity_bytes},
            {"resident_bytes", cache_stats.resident_bytes},
            {"peak_resident_bytes", cache_stats.peak_resident_bytes},
            {"hits", cache_stats.hits},
            {"misses", cache_stats.misses},
            {"admissions", cache_stats.admissions},
            {"evictions", cache_stats.evictions},
            {"pinned_capacity_failures", cache_stats.pinned_capacity_failures}
        };
        if (!secondary_cache_name.empty()) {
            output[secondary_cache_name] = {
                {"capacity_bytes", secondary_cache_stats.capacity_bytes},
                {"resident_bytes", secondary_cache_stats.resident_bytes},
                {"peak_resident_bytes", secondary_cache_stats.peak_resident_bytes},
                {"hits", secondary_cache_stats.hits},
                {"misses", secondary_cache_stats.misses},
                {"admissions", secondary_cache_stats.admissions},
                {"evictions", secondary_cache_stats.evictions},
                {"pinned_capacity_failures", secondary_cache_stats.pinned_capacity_failures}
            };
        }
        std::cout << output.dump(2) << '\n';
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "qwen4exp-ple-bench: " << error.what() << '\n';
        return 1;
    }
}
