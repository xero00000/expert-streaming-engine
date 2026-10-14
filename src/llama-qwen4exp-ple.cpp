#include "llama-qwen4exp-ple.h"

#include "ggml-quants.h"
#include "ggml.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_set>

void qwen4exp_ple_dequantize_iq4_nl(
        const uint8_t * row,
        size_t row_size,
        float * output,
        size_t output_count) {
    if (row == nullptr || output == nullptr) {
        throw std::invalid_argument("qwen4exp PLE dequantization buffers must be non-null");
    }
    if (output_count == 0 || output_count % 32 != 0 ||
            ggml_row_size(GGML_TYPE_IQ4_NL, output_count) != row_size) {
        throw std::invalid_argument("qwen4exp PLE IQ4_NL row geometry is invalid");
    }
    dequantize_row_iq4_nl(
        reinterpret_cast<const block_iq4_nl *>(row), output, int64_t(output_count));
}

void qwen4exp_ple_dequantize_iq4_nl_backend(
        ggml_backend_t backend,
        const uint8_t * rows,
        size_t row_count,
        float * output,
        size_t output_count) {
    constexpr size_t values_per_row = 160;
    constexpr size_t bytes_per_row = 90;
    if (backend == nullptr || rows == nullptr || output == nullptr || row_count == 0 ||
            row_count > std::numeric_limits<size_t>::max() / values_per_row ||
            output_count != row_count * values_per_row ||
            row_count > size_t(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("qwen4exp PLE backend dequantization geometry is invalid");
    }

    const size_t context_size = ggml_tensor_overhead() * 4 + ggml_graph_overhead_custom(8, false);
    ggml_init_params params = {context_size, nullptr, true};
    ggml_context * context = ggml_init(params);
    if (context == nullptr) throw std::runtime_error("qwen4exp PLE backend context allocation failed");
    struct context_cleanup {
        ggml_context * value;
        ~context_cleanup() { ggml_free(value); }
    } free_context{context};

    ggml_tensor * quantized = ggml_new_tensor_2d(
        context, GGML_TYPE_IQ4_NL, values_per_row, int64_t(row_count));
    ggml_tensor * dequantized = ggml_cast(context, quantized, GGML_TYPE_F32);
    if (!ggml_backend_supports_op(backend, dequantized)) {
        throw std::runtime_error("qwen4exp PLE backend does not support IQ4_NL to F32 conversion");
    }
    ggml_cgraph * graph = ggml_new_graph_custom(context, 8, false);
    ggml_build_forward_expand(graph, dequantized);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (buffer == nullptr) throw std::runtime_error("qwen4exp PLE backend buffer allocation failed");
    struct buffer_cleanup {
        ggml_backend_buffer_t value;
        ~buffer_cleanup() { ggml_backend_buffer_free(value); }
    } free_buffer{buffer};

    ggml_backend_tensor_set(quantized, rows, 0, row_count * bytes_per_row);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("qwen4exp PLE backend dequantization graph failed");
    }
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(dequantized, output, 0, output_count * sizeof(float));
}

qwen4exp_ple_extent_index::qwen4exp_ple_extent_index(
        const qwen4exp_ple_hash_config & config,
        uint16_t shard,
        uint64_t tensor_data_offset,
        uint64_t row_size,
        uint64_t row_count,
        uint64_t page_size)
    : config_(config),
      shard_(shard),
      tensor_data_offset_(tensor_data_offset),
      row_size_(row_size),
      row_count_(row_count),
      page_size_(page_size) {
    if (row_size_ == 0 || page_size_ == 0) {
        throw std::invalid_argument("qwen4exp PLE row and page sizes must be nonzero");
    }
    if (row_count_ > (std::numeric_limits<uint64_t>::max() - tensor_data_offset_) / row_size_) {
        throw std::overflow_error("qwen4exp PLE tensor extent overflows uint64");
    }
    for (size_t head = 0; head < config_.head_offsets.size(); ++head) {
        const uint64_t offset = config_.head_offsets[head];
        const uint64_t size = config_.head_vocab_sizes[head];
        if (offset > row_count_ || size > row_count_ - offset) {
            throw std::out_of_range("qwen4exp PLE head range exceeds tensor rows");
        }
    }
}

qwen4exp_ple_extent qwen4exp_ple_extent_index::locate(uint32_t head, uint64_t row) const {
    if (head >= config_.head_offsets.size()) {
        throw std::out_of_range("qwen4exp PLE head is out of range");
    }
    if (row >= config_.head_vocab_sizes[head]) {
        throw std::out_of_range("qwen4exp PLE row is out of range for its head");
    }
    return locate_global(config_.head_offsets[head] + row);
}

qwen4exp_ple_extent qwen4exp_ple_extent_index::locate_global(uint64_t global_row) const {
    if (global_row >= row_count_) {
        throw std::out_of_range("qwen4exp PLE global row is out of range");
    }
    const uint64_t offset = tensor_data_offset_ + global_row * row_size_;
    const uint64_t last_byte = offset + row_size_ - 1;
    return {
        shard_,
        global_row,
        offset,
        row_size_,
        offset / page_size_,
        last_byte / page_size_,
    };
}

qwen4exp_ple_hasher::qwen4exp_ple_hasher(qwen4exp_ple_hash_config config)
    : config_(config) {
    for (uint64_t size : config_.head_vocab_sizes) {
        if (size == 0) {
            throw std::invalid_argument("qwen4exp PLE head vocabulary size must be nonzero");
        }
    }
}

uint64_t qwen4exp_ple_hasher::signed_remainder(uint64_t value_bits, uint64_t divisor) {
    const uint64_t remainder = value_bits % divisor;
    if (value_bits <= uint64_t(std::numeric_limits<int64_t>::max())) {
        return remainder;
    }

    // PyTorch performs the hash in signed int64 tensors and uses floor-style
    // remainder for a positive divisor. Reconstruct (value_bits - 2^64) mod
    // divisor without implementation-defined unsigned-to-signed conversion.
    const uint64_t two_to_64_mod = (std::numeric_limits<uint64_t>::max() % divisor + 1) % divisor;
    return remainder >= two_to_64_mod
        ? remainder - two_to_64_mod
        : remainder + (divisor - two_to_64_mod);
}

std::array<uint64_t, 16> qwen4exp_ple_hasher::accept(uint32_t token_id) {
    const uint64_t eos = config_.eos_token_id;
    const uint64_t current = token_id;
    const uint64_t previous = history_size_ == 0 ? eos : history_[history_size_ - 1];
    const uint64_t previous_2 = history_size_ < 2 ? eos : history_[0];

    const uint64_t bigram = current * config_.layer_multipliers[0]
        ^ previous * config_.layer_multipliers[1];
    const uint64_t trigram = bigram ^ previous_2 * config_.layer_multipliers[2];

    std::array<uint64_t, 16> rows = {};
    for (size_t head = 0; head < rows.size(); ++head) {
        const uint64_t mixed = head < 8 ? bigram : trigram;
        rows[head] = config_.head_offsets[head]
            + signed_remainder(mixed, config_.head_vocab_sizes[head]);
    }

    if (token_id == config_.eos_token_id) {
        history_size_ = 0;
    } else {
        if (history_size_ == 2) {
            history_[0] = history_[1];
            history_size_ = 1;
        }
        history_[history_size_++] = token_id;
    }
    return rows;
}

void qwen4exp_ple_hasher::reset() {
    history_size_ = 0;
}

qwen4exp_ple_sequence_state::qwen4exp_ple_sequence_state(qwen4exp_ple_hash_config config)
    : config_(std::move(config)), memory_(llama_memory::create_state()),
      sequences_(0, std::hash<int32_t>{}, std::equal_to<int32_t>{},
            llama_memory::metadata_allocator<std::pair<const int32_t, sequence>>(memory_)) {}

qwen4exp_ple_sequence_state::qwen4exp_ple_sequence_state(const qwen4exp_ple_sequence_state & other)
    : qwen4exp_ple_sequence_state(other, other.memory_stats().allocation_budget_bytes) {}

qwen4exp_ple_sequence_state::qwen4exp_ple_sequence_state(
        const qwen4exp_ple_sequence_state & other, uint64_t budget)
    : qwen4exp_ple_sequence_state(other.config_) {
    (void) set_memory_budget(budget);
    for (const auto & entry : other.sequences_) {
        auto & target = sequences_.try_emplace(entry.first, config_, memory_).first->second;
        target.tokens.insert(entry.second.tokens.begin(), entry.second.tokens.end());
        target.hasher = entry.second.hasher;
        target.max_position = entry.second.max_position;
    }
}

qwen4exp_ple_sequence_state & qwen4exp_ple_sequence_state::operator=(const qwen4exp_ple_sequence_state & other) {
    if (this == &other) return *this;
    qwen4exp_ple_sequence_state prepared(other, memory_stats().allocation_budget_bytes);
    swap(prepared);
    return *this;
}

void qwen4exp_ple_sequence_state::swap(qwen4exp_ple_sequence_state & other) noexcept {
    std::swap(config_, other.config_);
    memory_.swap(other.memory_);
    sequences_.swap(other.sequences_);
}

llama_memory::allocation_stats qwen4exp_ple_sequence_state::memory_stats() const {
    return llama_memory::snapshot(memory_);
}

bool qwen4exp_ple_sequence_state::set_memory_budget(uint64_t bytes) {
    return llama_memory::set_budget(memory_, bytes);
}

std::array<uint64_t, 16> qwen4exp_ple_sequence_state::accept(
        int32_t seq_id, int64_t position, uint32_t token_id) {
    if (position < 0 || position == std::numeric_limits<int64_t>::max())
        throw std::invalid_argument("qwen4exp PLE token position is invalid");
    auto [it, inserted] = sequences_.try_emplace(seq_id, config_, memory_);
    sequence & state = it->second;
    if (position <= state.max_position)
        throw std::invalid_argument("qwen4exp PLE new token position must follow retained positions");
    if (!state.tokens.empty() && state.tokens.rbegin()->first >= std::numeric_limits<int64_t>::max() - 1)
        throw std::invalid_argument("qwen4exp PLE chronological order overflows");
    const int64_t order = state.tokens.empty() ? position : state.tokens.rbegin()->first + 1;
    try { state.tokens.emplace(order, token_entry{position, token_id}); }
    catch (...) {
        if (inserted) sequences_.erase(it);
        throw;
    }
    state.max_position = position;
    return state.hasher.accept(token_id);
}

void qwen4exp_ple_sequence_state::rebuild(sequence & state) {
    state.hasher.reset();
    state.max_position = -1;
    for (const auto & entry : state.tokens) {
        (void) state.hasher.accept(entry.second.second);
        state.max_position = std::max(state.max_position, entry.second.first);
    }
}

void qwen4exp_ple_sequence_state::remove(int32_t seq_id, int64_t p0, int64_t p1) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<int64_t>::max();
    if (p0 >= p1) return;
    for (auto it = sequences_.begin(); it != sequences_.end();) {
        if (seq_id >= 0 && it->first != seq_id) { ++it; continue; }
        auto & state = it->second;
        bool changed = false;
        for (auto token = state.tokens.begin(); token != state.tokens.end();) {
            const auto position = token->second.first;
            if (p0 <= position && position < p1) { token = state.tokens.erase(token); changed = true; }
            else ++token;
        }
        if (state.tokens.empty()) it = sequences_.erase(it);
        else { if (changed) rebuild(state); ++it; }
    }
}

void qwen4exp_ple_sequence_state::copy(
        int32_t src_seq_id, int32_t dst_seq_id, int64_t p0, int64_t p1) {
    if (src_seq_id == dst_seq_id) return;
    const auto source = sequences_.find(src_seq_id);
    if (source == sequences_.end()) return;
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<int64_t>::max();
    if (p0 >= p1) return;
    sequence prepared(config_, memory_);
    const auto destination = sequences_.find(dst_seq_id);
    if (destination != sequences_.end()) prepared = destination->second;
    std::unordered_set<int64_t, std::hash<int64_t>, std::equal_to<int64_t>,
            llama_memory::metadata_allocator<int64_t>> copied_positions(0, std::hash<int64_t>{},
                    std::equal_to<int64_t>{}, llama_memory::metadata_allocator<int64_t>(memory_));
    for (const auto & entry : source->second.tokens)
        if (p0 <= entry.second.first && entry.second.first < p1) copied_positions.insert(entry.second.first);
    for (const auto & other : prepared.tokens) {
        if (!copied_positions.count(other.second.first)) continue;
        const auto peer = source->second.tokens.find(other.first);
        if (peer == source->second.tokens.end() || peer->second != other.second)
            throw std::invalid_argument("qwen4exp PLE sequence copy has conflicting token positions");
    }
    bool copied = false;
    for (const auto & entry : source->second.tokens) {
        const auto position = entry.second.first;
        if (position < p0 || position >= p1) continue;
        // Copies share chronology, including multiple distinct tokens whose
        // compressed rotary positions coincide.
        const auto [existing, added] = prepared.tokens.emplace(entry);
        if (!added && existing->second != entry.second)
            throw std::invalid_argument("qwen4exp PLE sequence copy has conflicting token order");
        copied = true;
    }
    if (!copied) return;
    rebuild(prepared);
    sequences_.insert_or_assign(dst_seq_id, std::move(prepared));
}

void qwen4exp_ple_sequence_state::add(int32_t seq_id, int64_t p0, int64_t p1, int64_t delta) {
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<int64_t>::max();
    if (!delta || p0 >= p1) return;
    // Validate the whole operation before touching any sequence.
    for (const auto & seq : sequences_) {
        if (seq_id >= 0 && seq.first != seq_id) continue;
        for (const auto & entry : seq.second.tokens) {
            const auto pos = entry.second.first;
            if (p0 <= pos && pos < p1 && (delta > 0 && pos > std::numeric_limits<int64_t>::max() - delta))
                throw std::invalid_argument("qwen4exp PLE position shift overflows");
        }
    }
    for (auto seq = sequences_.begin(); seq != sequences_.end();) {
        if (seq_id >= 0 && seq->first != seq_id) { ++seq; continue; }
        auto & state = seq->second;
        bool removed = false;
        state.max_position = -1;
        for (auto entry = state.tokens.begin(); entry != state.tokens.end();) {
            auto & pos = entry->second.first;
            if (p0 <= pos && pos < p1) pos += delta;
            if (pos < 0) { entry = state.tokens.erase(entry); removed = true; }
            else { state.max_position = std::max(state.max_position, pos); ++entry; }
        }
        if (state.tokens.empty()) seq = sequences_.erase(seq);
        else { if (removed) rebuild(state); ++seq; }
    }
}

void qwen4exp_ple_sequence_state::divide(int32_t seq_id, int64_t p0, int64_t p1, int divisor) {
    if (divisor <= 0) throw std::invalid_argument("qwen4exp PLE position divisor must be positive");
    if (p0 < 0) p0 = 0;
    if (p1 < 0) p1 = std::numeric_limits<int64_t>::max();
    if (p0 >= p1 || divisor == 1) return;
    for (auto & seq : sequences_) {
        if (seq_id >= 0 && seq.first != seq_id) continue;
        seq.second.max_position = -1;
        for (auto & entry : seq.second.tokens) {
            auto & pos = entry.second.first;
            if (p0 <= pos && pos < p1) pos /= divisor;
            seq.second.max_position = std::max(seq.second.max_position, pos);
        }
    }
}

std::vector<qwen4exp_ple_sequence_state::history_entry>
qwen4exp_ple_sequence_state::history(int32_t seq_id) const {
    std::vector<history_entry> result;
    const auto found = sequences_.find(seq_id);
    if (found != sequences_.end()) {
        result.reserve(found->second.tokens.size());
        for (const auto & entry : found->second.tokens)
            result.push_back({entry.first, entry.second.first, entry.second.second});
    }
    return result;
}

void qwen4exp_ple_sequence_state::restore_history(int32_t seq_id, const std::vector<history_entry> & history) {
    if (history.empty()) { reset(seq_id); return; }
    sequence prepared(config_, memory_);
    int64_t previous = -1;
    for (const auto & entry : history) {
        if (entry.order <= previous || entry.order == std::numeric_limits<int64_t>::max() || entry.position < 0)
            throw std::invalid_argument("qwen4exp PLE restore has invalid chronological order or position");
        prepared.tokens.emplace(entry.order, token_entry{entry.position, entry.token});
        previous = entry.order;
    }
    rebuild(prepared);
    sequences_.insert_or_assign(seq_id, std::move(prepared));
}

void qwen4exp_ple_sequence_state::keep(int32_t seq_id) {
    auto found = sequences_.find(seq_id);
    if (found == sequences_.end()) {
        sequences_.clear();
        return;
    }
    // Erase peers in place: keep() must not allocate after deleting the retained
    // sequence. This also avoids copying or rebuilding its hasher/token order.
    for (auto it = sequences_.begin(); it != sequences_.end();) {
        if (it->first != seq_id) it = sequences_.erase(it); else ++it;
    }
}

void qwen4exp_ple_sequence_state::reset(int32_t seq_id) {
    sequences_.erase(seq_id);
}

void qwen4exp_ple_sequence_state::clear() {
    sequences_.clear();
}

std::vector<qwen4exp_ple_sequence_state::token_entry>
qwen4exp_ple_sequence_state::tokens(int32_t seq_id) const {
    std::vector<token_entry> result;
    const auto found = sequences_.find(seq_id);
    if (found != sequences_.end())
        for (const auto & entry : found->second.tokens) result.push_back(entry.second);
    return result;
}

std::vector<std::pair<int32_t, std::vector<qwen4exp_ple_sequence_state::token_entry>>>
qwen4exp_ple_sequence_state::all_tokens() const {
    std::vector<std::pair<int32_t, std::vector<token_entry>>> result;
    for (const auto & seq : sequences_) result.emplace_back(seq.first, tokens(seq.first));
    std::sort(result.begin(), result.end(), [](const auto & a, const auto & b) { return a.first < b.first; });
    return result;
}

void qwen4exp_ple_sequence_state::restore(int32_t seq_id, const std::vector<token_entry> & tokens) {
    // Legacy helper input uses unique original positions as chronology. Engine
    // snapshots use restore_history and preserve ordinals after compression.
    std::map<int64_t, uint32_t> sorted;
    for (const auto & token : tokens) {
        if (token.first < 0 || !sorted.emplace(token).second)
            throw std::invalid_argument("qwen4exp PLE restore contains invalid or duplicate token positions");
    }
    std::vector<history_entry> entries;
    for (const auto & token : sorted) entries.push_back({token.first, token.first, token.second});
    restore_history(seq_id, entries);
}
