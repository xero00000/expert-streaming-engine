#include "llama-qwen4exp-ple.h"
#include "llama.h"

#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <nlohmann/json.hpp>

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') return uint8_t(value - '0');
    if (value >= 'a' && value <= 'f') return uint8_t(value - 'a' + 10);
    if (value >= 'A' && value <= 'F') return uint8_t(value - 'A' + 10);
    throw std::invalid_argument("invalid hex digit in PLE row fixture");
}

static std::vector<uint8_t> decode_hex(const std::string & encoded) {
    if (encoded.size() % 2 != 0) {
        throw std::invalid_argument("odd-length PLE row fixture hex string");
    }
    std::vector<uint8_t> decoded(encoded.size() / 2);
    for (size_t index = 0; index < decoded.size(); ++index) {
        decoded[index] = uint8_t(hex_nibble(encoded[2 * index]) << 4) |
            hex_nibble(encoded[2 * index + 1]);
    }
    return decoded;
}

static bool history_memory_cases(const qwen4exp_ple_hash_config & config) {
    qwen4exp_ple_sequence_state history(config);
    if(history.memory_stats().allocated_bytes)return false;
    (void)history.accept(0,0,101);auto first=history.memory_stats();
    if(!first.allocated_bytes || history.set_memory_budget(first.allocated_bytes-1) ||
            !history.set_memory_budget(first.allocated_bytes))return false;
    const auto before=history.all_tokens();size_t failures=0;
    for(int sequence:{0,9}) {
        bool failed=false;try{(void)history.accept(sequence,1,102);}catch(const std::bad_alloc &){failed=true;}
        if(!failed || history.all_tokens()!=before)return false;
        ++failures;
    }
    history.set_memory_budget(UINT64_MAX);
    qwen4exp_ple_hasher oracle(config);(void)oracle.accept(101);
    if(history.accept(0,1,config.eos_token_id)!=oracle.accept(config.eos_token_id) ||
            history.accept(0,2,777)!=oracle.accept(777))return false;
    const auto original_bytes=history.memory_stats().allocated_bytes;
    auto independent=history;
    if(history.memory_stats().allocated_bytes!=original_bytes || independent.all_tokens()!=history.all_tokens()) {
        std::fprintf(stderr,"history copy shares accounting: original=%llu before=%llu copy=%llu\n",
                (unsigned long long)history.memory_stats().allocated_bytes,(unsigned long long)original_bytes,
                (unsigned long long)independent.memory_stats().allocated_bytes);return false;
    }
    independent.set_memory_budget(UINT64_MAX);(void)independent.accept(0,3,778);
    if(history.memory_stats().allocated_bytes!=original_bytes || history.tokens(0).size()!=3)return false;
    qwen4exp_ple_sequence_state destination(config);(void)destination.accept(1,0,50);
    const auto destination_before=destination.all_tokens();auto bounded=destination.memory_stats();
    destination.set_memory_budget(bounded.allocated_bytes);
    bool rejected=false;try{destination=independent;}catch(const std::bad_alloc &){rejected=true;}
    if(!rejected || destination.all_tokens()!=destination_before ||
            destination.memory_stats().allocated_bytes!=bounded.allocated_bytes)return false;
    destination.set_memory_budget(UINT64_MAX);destination=independent;
    if(destination.all_tokens()!=independent.all_tokens() || destination.memory_stats().allocated_bytes==0)return false;
    // Keep the selected sequence without a replacement node allocation.
    (void)destination.accept(2,0,88);auto keep_before=destination.memory_stats();
    destination.set_memory_budget(keep_before.allocated_bytes);auto keep_tokens=destination.tokens(0);
    destination.keep(0);auto kept=destination.memory_stats();
    if(destination.tokens(0)!=keep_tokens || !destination.tokens(2).empty() ||
            kept.allocation_requests!=keep_before.allocation_requests) {
        std::fprintf(stderr,"history keep reallocated: requests=%llu/%llu\n",
                (unsigned long long)kept.allocation_requests,(unsigned long long)keep_before.allocation_requests);return false;
    }
    destination.set_memory_budget(UINT64_MAX);
    // Sweep preparation failures in copy/restore, preserving complete histories.
    size_t rejected_copy=0,rejected_restore=0,admitted=0;
    for(uint64_t slack=0;slack<=2048;slack+=16) {
        qwen4exp_ple_sequence_state candidate(config);
        for(int i=0;i<4;++i)(void)candidate.accept(0,i,uint32_t(100+i));
        (void)candidate.accept(1,10,200);
        auto initial=candidate.all_tokens();auto base=candidate.memory_stats();candidate.set_memory_budget(base.allocated_bytes+slack);
        bool failed=false;try{candidate.copy(0,1,0,-1);}catch(const std::bad_alloc &){failed=true;}

        if(failed){++rejected_copy;if(candidate.all_tokens()!=initial)return false;}else ++admitted;
        candidate.set_memory_budget(UINT64_MAX);
        auto snapshot=candidate.history(0);initial=candidate.all_tokens();base=candidate.memory_stats();candidate.set_memory_budget(base.allocated_bytes+slack);
        failed=false;try{candidate.restore_history(1,snapshot);}catch(const std::bad_alloc &){failed=true;}
        if(failed){++rejected_restore;if(candidate.all_tokens()!=initial)return false;}else ++admitted;
    }
    if(!rejected_copy || !rejected_restore || !admitted)return false;
    std::printf("PLE history allocations: initial=%llu bytes; failed accepts=%zu; copy failures=%zu restore failures=%zu admissions=%zu; independent copy/assignment and nonallocating keep pass\n",
            (unsigned long long)first.allocated_bytes,failures,rejected_copy,rejected_restore,admitted);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s PLE_REFERENCE_TRACE.json PLE_ROW_FIXTURE.json\n", argv[0]);
        return 1;
    }

    std::ifstream input(argv[1]);
    if (!input) {
        std::fprintf(stderr, "failed to open PLE reference trace: %s\n", argv[1]);
        return 1;
    }
    const json trace = json::parse(input);
    const json & ple = trace.at("ple");

    qwen4exp_ple_hash_config config;
    config.eos_token_id = ple.at("eos_token_id").get<uint32_t>();
    for (size_t index = 0; index < config.layer_multipliers.size(); ++index) {
        config.layer_multipliers[index] = ple.at("layer_multipliers").at(index).get<uint64_t>();
    }
    for (size_t head = 0; head < config.head_vocab_sizes.size(); ++head) {
        config.head_vocab_sizes[head] = ple.at("head_vocab_sizes").at(head).get<uint64_t>();
        config.head_offsets[head] = ple.at("head_offsets").at(head).get<uint64_t>();
    }

    if(!history_memory_cases(config)) { std::fprintf(stderr,"PLE history allocation/budget cases failed\n");return 1; }

    qwen4exp_ple_hasher hasher(config);
    for (const json & sequence : trace.at("sequences")) {
        hasher.reset();
        const json & tokens = sequence.at("token_ids");
        const json & expected = sequence.at("row_ids");
        if (tokens.size() != expected.size()) {
            std::fprintf(stderr, "invalid PLE trace sequence length\n");
            return 1;
        }
        for (size_t token_index = 0; token_index < tokens.size(); ++token_index) {
            const auto actual = hasher.accept(tokens.at(token_index).get<uint32_t>());
            for (size_t head = 0; head < actual.size(); ++head) {
                if (actual[head] != expected.at(token_index).at(head).get<uint64_t>()) {
                    const std::string name = sequence.at("name").get<std::string>();
                    std::fprintf(stderr, "PLE row mismatch in %s at token %zu head %zu\n",
                        name.c_str(), token_index, head);
                    return 1;
                }
            }
        }
    }

    // PLE host history must survive the same rollback/copy/serialize lifecycle
    // as the recurrent convolution state. Validate reconstruction after a
    // suffix removal, where retaining only the latest two IDs would be wrong.
    qwen4exp_ple_sequence_state sequence_state(config);
    (void) sequence_state.accept(7, 0, 101);
    (void) sequence_state.accept(7, 1, 102);
    (void) sequence_state.accept(7, 2, 103);
    sequence_state.remove(7, 2, -1);
    const auto rollback_rows = sequence_state.accept(7, 2, 777);
    qwen4exp_ple_hasher rollback_reference(config);
    (void) rollback_reference.accept(101);
    (void) rollback_reference.accept(102);
    if (rollback_rows != rollback_reference.accept(777)) {
        std::fprintf(stderr, "PLE sequence rollback did not reconstruct hash history\n");
        return 1;
    }
    sequence_state.copy(7, 8, -1, -1);
    if (sequence_state.tokens(8) != sequence_state.tokens(7)) {
        std::fprintf(stderr, "PLE sequence copy lost token history\n");
        return 1;
    }
    const auto serialized_history = sequence_state.tokens(7);
    sequence_state.reset(7);
    sequence_state.restore(7, serialized_history);
    if (sequence_state.tokens(7) != serialized_history) {
        std::fprintf(stderr, "PLE sequence restore lost token history\n");
        return 1;
    }
    (void) sequence_state.accept(9, 0, 999);
    sequence_state.keep(7);
    if (!sequence_state.tokens(9).empty() || sequence_state.tokens(7) != serialized_history) {
        std::fprintf(stderr, "PLE sequence keep retained the wrong histories\n");
        return 1;
    }

    // A conflict after an earlier missing position must leave the entire
    // destination unchanged, including its rolling hash state.
    qwen4exp_ple_sequence_state atomic_state(config);
    atomic_state.restore(1, {{0, 101}, {1, 102}, {2, 103}});
    atomic_state.restore(2, {{0, 101}, {2, 999}});
    auto unchanged = atomic_state.all_tokens();
    bool conflict = false;
    try { atomic_state.copy(1, 2, 0, -1); }
    catch (const std::invalid_argument &) { conflict = true; }
    if (!conflict || atomic_state.all_tokens() != unchanged) {
        std::fprintf(stderr, "PLE conflicting copy changed live history\n"); return 1;
    }
    qwen4exp_ple_hasher atomic_reference(config);
    (void) atomic_reference.accept(101);
    (void) atomic_reference.accept(999);
    if (atomic_state.accept(2, 3, 777) != atomic_reference.accept(777)) {
        std::fprintf(stderr, "PLE failed copy changed rolling hash state\n"); return 1;
    }
    unchanged = atomic_state.all_tokens();
    for (const auto & invalid : std::vector<std::vector<qwen4exp_ple_sequence_state::token_entry>>{
            {{0, 101}, {1, 102}, {1, 103}}, {{-1, 101}, {0, 102}}}) {
        conflict = false;
        try { atomic_state.restore(2, invalid); }
        catch (const std::invalid_argument &) { conflict = true; }
        if (!conflict || atomic_state.all_tokens() != unchanged) {
            std::fprintf(stderr, "PLE malformed restore changed live history\n"); return 1;
        }
    }
    atomic_state.remove(2, 5, 2);
    atomic_state.copy(1, 3, 5, 2);
    if (atomic_state.all_tokens() != unchanged) {
        std::fprintf(stderr, "PLE empty/reversed range changed live history\n"); return 1;
    }
    atomic_state.remove(-1, 1, 3);
    if (atomic_state.tokens(1) != std::vector<qwen4exp_ple_sequence_state::token_entry>{{0, 101}} ||
        atomic_state.tokens(2) != std::vector<qwen4exp_ple_sequence_state::token_entry>{{0, 101}, {3, 777}}) {
        std::fprintf(stderr, "PLE wildcard removal did not retain out-of-range tokens\n"); return 1;
    }
    qwen4exp_ple_hasher retained_reference(config);
    (void) retained_reference.accept(101); (void) retained_reference.accept(777);
    if (atomic_state.accept(2, 4, 888) != retained_reference.accept(888)) {
        std::fprintf(stderr, "PLE wildcard removal did not rebuild rolling hash\n"); return 1;
    }

    qwen4exp_ple_sequence_state positioned(config);
    positioned.restore(0, {{0, 101}, {1, 102}, {2, 103}, {3, 104}});
    positioned.divide(0, 0, -1, 2);
    if (positioned.tokens(0) != std::vector<qwen4exp_ple_sequence_state::token_entry>{
            {0, 101}, {0, 102}, {1, 103}, {1, 104}}) {
        std::fprintf(stderr, "PLE compression lost coincident tokens\n"); return 1;
    }
    positioned.add(0, 0, 1, 10);
    if (positioned.tokens(0) != std::vector<qwen4exp_ple_sequence_state::token_entry>{
            {10, 101}, {10, 102}, {1, 103}, {1, 104}}) {
        std::fprintf(stderr, "PLE shift reordered token chronology\n"); return 1;
    }
    const auto positioned_history = positioned.history(0);
    positioned.restore_history(1, positioned_history);
    positioned.copy(0, 2, 0, -1);
    positioned.copy(0, 2, 0, -1); // Compressed prefix copies are idempotent.
    qwen4exp_ple_hasher position_reference(config);
    for (uint32_t token : {101u, 102u, 103u, 104u}) (void) position_reference.accept(token);
    const auto position_expected = position_reference.accept(777);
    for (int seq : {0, 1, 2}) {
        if (positioned.accept(seq, 11, 777) != position_expected) {
            std::fprintf(stderr, "PLE transformed/restored/copied history changed hash\n"); return 1;
        }
    }
    const auto position_before = positioned.all_tokens();
    conflict = false;
    try { positioned.divide(0, 0, -1, 0); }
    catch (const std::invalid_argument &) { conflict = true; }
    if (!conflict || positioned.all_tokens() != position_before) {
        std::fprintf(stderr, "PLE invalid divisor changed history\n"); return 1;
    }
    auto invalid_history = positioned.history(0);
    invalid_history[1].order = invalid_history[0].order;
    conflict = false;
    try { positioned.restore_history(0, invalid_history); }
    catch (const std::invalid_argument &) { conflict = true; }
    if (!conflict || positioned.all_tokens() != position_before) {
        std::fprintf(stderr, "PLE invalid chronological restore changed history\n"); return 1;
    }
    positioned.add(0, 10, -1, -12);
    if (positioned.tokens(0) != std::vector<qwen4exp_ple_sequence_state::token_entry>{{1, 103}, {1, 104}}) {
        std::fprintf(stderr, "PLE negative shift retained removed tokens\n"); return 1;
    }
    qwen4exp_ple_hasher removed_reference(config);
    (void) removed_reference.accept(103); (void) removed_reference.accept(104);
    if (positioned.accept(0, 2, 888) != removed_reference.accept(888)) {
        std::fprintf(stderr, "PLE negative shift did not reconstruct hash\n"); return 1;
    }

    qwen4exp_ple_extent_index extents(config, 1, 364622656, 90, 320001536);
    const auto first = extents.locate(0, 0);
    if (first.shard != 1 || first.offset != 364622656 || first.length != 90 ||
        first.first_page != first.last_page) {
        std::fprintf(stderr, "invalid first PLE row extent\n");
        return 1;
    }
    const auto straddled = extents.locate(0, 36);
    if (straddled.offset != 364625896 || straddled.first_page != 89019 ||
        straddled.last_page != 89020) {
        std::fprintf(stderr, "PLE page-straddling row was not indexed correctly\n");
        return 1;
    }
    const auto final = extents.locate(15, config.head_vocab_sizes[15] - 1);
    if (final.global_row != 320001445 || final.offset + final.length != 29164752796) {
        std::fprintf(stderr, "invalid final populated PLE row extent\n");
        return 1;
    }

    bool rejected = false;
    try {
        (void) extents.locate(16, 0);
    } catch (const std::out_of_range &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "PLE extent index accepted an invalid head\n");
        return 1;
    }

    std::ifstream fixture_input(argv[2]);
    if (!fixture_input) {
        std::fprintf(stderr, "failed to open PLE row fixture: %s\n", argv[2]);
        return 1;
    }
    const json fixture = json::parse(fixture_input);
    const auto quantized = decode_hex(fixture.at("quantized_hex").get<std::string>());
    const json & expected_values = fixture.at("dequantized_values");
    if (quantized.size() != 90 || expected_values.size() != 160) {
        std::fprintf(stderr, "invalid PLE IQ4_NL row fixture geometry\n");
        return 1;
    }
    std::array<float, 160> values = {};
    llama_backend_init();
    qwen4exp_ple_dequantize_iq4_nl(
        quantized.data(), quantized.size(), values.data(), values.size());
    for (size_t index = 0; index < values.size(); ++index) {
        if (values[index] != expected_values.at(index).get<float>()) {
            std::fprintf(stderr, "PLE IQ4_NL dequantization mismatch at value %zu: actual %.12g expected %.12g\n",
                index, double(values[index]), double(expected_values.at(index).get<float>()));
            return 1;
        }
    }

    std::vector<uint8_t> quantized_batch;
    quantized_batch.reserve(16 * quantized.size());
    for (size_t row = 0; row < 16; ++row) {
        quantized_batch.insert(quantized_batch.end(), quantized.begin(), quantized.end());
    }
    std::vector<float> backend_values(16 * values.size());
    auto check_backend = [&](ggml_backend_t backend, const char * name) {
        qwen4exp_ple_dequantize_iq4_nl_backend(
            backend, quantized_batch.data(), 16, backend_values.data(), backend_values.size());
        for (size_t row = 0; row < 16; ++row) {
            for (size_t index = 0; index < values.size(); ++index) {
                if (backend_values[row * values.size() + index] != values[index]) {
                    std::fprintf(stderr, "PLE %s backend mismatch at row %zu value %zu\n",
                        name, row, index);
                    return false;
                }
            }
        }
        return true;
    };
    ggml_backend_t cpu_backend = ggml_backend_cpu_init();
    if (cpu_backend == nullptr || !check_backend(cpu_backend, "CPU")) return 1;
    ggml_backend_free(cpu_backend);

#ifdef GGML_USE_CUDA
    const int cuda_device_count = ggml_backend_cuda_get_device_count();
    if (cuda_device_count <= 0) {
        std::fprintf(stderr, "CUDA build has no device for PLE reference parity\n");
        return 1;
    }
    for (int device = 0; device < cuda_device_count; ++device) {
        ggml_backend_t cuda_backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        const std::string name = "CUDA:" + std::to_string(device);
        if (cuda_backend == nullptr || !check_backend(cuda_backend, name.c_str())) return 1;
        ggml_backend_free(cuda_backend);
    }
#endif

    rejected = false;
    try {
        qwen4exp_ple_dequantize_iq4_nl(
            quantized.data(), quantized.size() - 1, values.data(), values.size());
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "PLE dequantizer accepted invalid row geometry\n");
        return 1;
    }
    llama_backend_free();
    return 0;
}
