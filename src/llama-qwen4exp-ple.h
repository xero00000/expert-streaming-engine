#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "llama-memory-accounting.h"

struct qwen4exp_ple_hash_config {
    uint32_t eos_token_id = 0;
    std::array<uint64_t, 3> layer_multipliers = {};
    std::array<uint64_t, 16> head_vocab_sizes = {};
    std::array<uint64_t, 16> head_offsets = {};
};

void qwen4exp_ple_dequantize_iq4_nl(
    const uint8_t * row,
    size_t row_size,
    float * output,
    size_t output_count);

// Reference accelerator path. The caller owns the backend; this routine
// stages one or more contiguous 90-byte rows, executes the backend's IQ4_NL
// conversion graph, and returns rows concatenated as F32 values.
void qwen4exp_ple_dequantize_iq4_nl_backend(
    ggml_backend_t backend,
    const uint8_t * rows,
    size_t row_count,
    float * output,
    size_t output_count);

struct qwen4exp_ple_extent {
    uint16_t shard = 0;
    uint64_t global_row = 0;
    uint64_t offset = 0;
    uint64_t length = 0;
    uint64_t first_page = 0;
    uint64_t last_page = 0;
};

class qwen4exp_ple_extent_index {
public:
    qwen4exp_ple_extent_index(
        const qwen4exp_ple_hash_config & config,
        uint16_t shard,
        uint64_t tensor_data_offset,
        uint64_t row_size,
        uint64_t row_count,
        uint64_t page_size = 4096);

    qwen4exp_ple_extent locate(uint32_t head, uint64_t row) const;
    qwen4exp_ple_extent locate_global(uint64_t global_row) const;

private:
    qwen4exp_ple_hash_config config_;
    uint16_t shard_;
    uint64_t tensor_data_offset_;
    uint64_t row_size_;
    uint64_t row_count_;
    uint64_t page_size_;
};

// Stateful reference implementation of the Qwen4-Exp bigram/trigram row lookup.
// It intentionally uses integer operations only so traces are reproducible across
// CPU and CUDA implementations.
class qwen4exp_ple_hasher {
public:
    explicit qwen4exp_ple_hasher(qwen4exp_ple_hash_config config);

    std::array<uint64_t, 16> accept(uint32_t token_id);
    void reset();

private:
    static uint64_t signed_remainder(uint64_t value_bits, uint64_t divisor);

    qwen4exp_ple_hash_config config_;
    std::array<uint64_t, 2> history_ = {};
    size_t history_size_ = 0;
};

// Position-tagged token history for PLE sequence lifecycle operations. Keeping
// the complete lightweight token stream allows suffix removal to reconstruct
// the rolling bigram/trigram state exactly; two cached predecessor IDs alone
// are insufficient after speculative rollback.
class qwen4exp_ple_sequence_state {
public:
    using token_entry = std::pair<int64_t, uint32_t>;
    struct history_entry {
        int64_t order;
        int64_t position;
        uint32_t token;
    };

    explicit qwen4exp_ple_sequence_state(qwen4exp_ple_hash_config config);
    qwen4exp_ple_sequence_state(const qwen4exp_ple_sequence_state & other);
    qwen4exp_ple_sequence_state & operator=(const qwen4exp_ple_sequence_state & other);
    llama_memory::allocation_stats memory_stats() const;
    // Budget includes token/sequence nodes, hash buckets and tracked preparation.
    bool set_memory_budget(uint64_t bytes);


    std::array<uint64_t, 16> accept(int32_t seq_id, int64_t position, uint32_t token_id);
    void remove(int32_t seq_id, int64_t p0, int64_t p1);
    void copy(int32_t src_seq_id, int32_t dst_seq_id, int64_t p0, int64_t p1);
    void add(int32_t seq_id, int64_t p0, int64_t p1, int64_t delta);
    void divide(int32_t seq_id, int64_t p0, int64_t p1, int divisor);
    std::vector<history_entry> history(int32_t seq_id) const;
    void restore_history(int32_t seq_id, const std::vector<history_entry> & history);
    void keep(int32_t seq_id);
    void reset(int32_t seq_id);
    void clear();

    std::vector<token_entry> tokens(int32_t seq_id) const;
    std::vector<std::pair<int32_t, std::vector<token_entry>>> all_tokens() const;
    void restore(int32_t seq_id, const std::vector<token_entry> & tokens);

private:
    struct sequence {
        explicit sequence(const qwen4exp_ple_hash_config & config,
                const std::shared_ptr<llama_memory::allocation_state> & memory)
            : tokens(std::less<int64_t>{}, llama_memory::metadata_allocator<std::pair<const int64_t, token_entry>>(memory)),
              hasher(config) {}
        // Key is chronological order, independent of the rotary position.
        std::map<int64_t, token_entry, std::less<int64_t>,
                llama_memory::metadata_allocator<std::pair<const int64_t, token_entry>>> tokens;
        qwen4exp_ple_hasher hasher;
        int64_t max_position = -1;
    };

    qwen4exp_ple_sequence_state(const qwen4exp_ple_sequence_state & other, uint64_t budget);
    void rebuild(sequence & state);
    void swap(qwen4exp_ple_sequence_state & other) noexcept;

    qwen4exp_ple_hash_config config_;
    std::shared_ptr<llama_memory::allocation_state> memory_;
    std::unordered_map<int32_t, sequence, std::hash<int32_t>, std::equal_to<int32_t>,
            llama_memory::metadata_allocator<std::pair<const int32_t, sequence>>> sequences_;
};
