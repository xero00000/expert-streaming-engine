#include "llama-deepseek-v41-engram.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace deepseek_v41 {

engram_sequence_state::engram_sequence_state(engram_hash_config config) : config_(std::move(config)) {
    if (config_.compressed_vocab_size < 2 ||
            config_.compressed_vocab_size > uint32_t(std::numeric_limits<int32_t>::max()) ||
            config_.pad_compressed_id >= config_.compressed_vocab_size || config_.token_map.empty()) {
        throw std::invalid_argument("invalid Engram compressed vocabulary");
    }
    for (uint32_t value : config_.token_map) {
        if (value >= config_.compressed_vocab_size) {
            throw std::invalid_argument("Engram token map exceeds compressed vocabulary");
        }
    }
    if (config_.layers[0].layer_id == config_.layers[1].layer_id) {
        throw std::invalid_argument("duplicate Engram layer IDs");
    }
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto & spec = config_.layers[layer];
        for (uint64_t multiplier : spec.multipliers) {
            if ((multiplier & 1) == 0 || multiplier >
                    uint64_t(std::numeric_limits<int64_t>::max()) / (config_.compressed_vocab_size - 1)) {
                throw std::invalid_argument("unsafe Engram hash multiplier");
            }
        }
        uint64_t offset = 0;
        for (size_t bucket = 0; bucket < 24; ++bucket) {
            uint64_t size = spec.primes[bucket];
            if (size == 0 || offset > spec.rows || size > spec.rows - offset) {
                throw std::invalid_argument("Engram bucket exceeds table extent");
            }
            offsets_[layer][bucket] = offset;
            offset += size;
        }
        if (offset != spec.rows) {
            throw std::invalid_argument("Engram bucket sizes disagree with table rows");
        }
    }
}

engram_rows engram_sequence_state::accept(int32_t sequence_id, size_t pos, uint32_t token_id, bool is_text) {
    if (token_id >= config_.token_map.size()) {
        throw std::out_of_range("Engram token ID out of range");
    }
    if (pos != position(sequence_id)) {
        throw std::out_of_range("Engram accept requires contiguous positions; rewind before replacing a suffix");
    }
    auto & history = sequences_[sequence_id];
    int32_t compressed = is_text ? int32_t(config_.token_map[token_id]) : -1;
    std::array<uint64_t, 4> tokens{};
    bool blocked = false;
    for (size_t shift = 0; shift < 4; ++shift) {
        int32_t source = shift == 0 ? compressed : (pos >= shift ? history[pos - shift] : -1);
        blocked = blocked || source == -1;
        tokens[shift] = blocked ? config_.pad_compressed_id : uint32_t(source);
    }
    engram_rows rows{};
    for (size_t layer = 0; layer < 2; ++layer) {
        const auto & spec = config_.layers[layer];
        uint64_t rolling = tokens[0] * spec.multipliers[0];
        for (size_t shift = 1; shift < 4; ++shift) {
            rolling ^= tokens[shift] * spec.multipliers[shift];
            for (size_t head = 0; head < 8; ++head) {
                size_t bucket = (shift - 1) * 8 + head;
                rows[layer][bucket] = rolling % spec.primes[bucket] + offsets_[layer][bucket];
            }
        }
    }
    // Publish history only once output calculation succeeds.
    history.push_back(compressed);
    return rows;
}

size_t engram_sequence_state::position(int32_t sequence_id) const {
    auto found = sequences_.find(sequence_id);
    return found == sequences_.end() ? 0 : found->second.size();
}

void engram_sequence_state::rewind(int32_t sequence_id, size_t pos) {
    auto found = sequences_.find(sequence_id);
    if (pos > position(sequence_id)) {
        throw std::out_of_range("cannot rewind Engram beyond current sequence");
    }
    if (found != sequences_.end()) {
        found->second.resize(pos);
    }
}

void engram_sequence_state::reset(int32_t sequence_id) { sequences_.erase(sequence_id); }
void engram_sequence_state::clear() { sequences_.clear(); }

std::vector<int32_t> engram_sequence_state::snapshot(int32_t sequence_id) const {
    auto found = sequences_.find(sequence_id);
    return found == sequences_.end() ? std::vector<int32_t>{} : found->second;
}

void engram_sequence_state::restore(int32_t sequence_id, const std::vector<int32_t> & tokens) {
    for (int32_t value : tokens) {
        if (value < -1 || (value >= 0 && uint32_t(value) >= config_.compressed_vocab_size)) {
            throw std::out_of_range("invalid compressed Engram token in snapshot");
        }
    }
    auto copy = tokens;
    sequences_[sequence_id].swap(copy);
}

void engram_sequence_state::copy(int32_t source, int32_t destination) {
    restore(destination, snapshot(source));
}

void engram_decode_fp8(const uint8_t * row, size_t bytes, float * output, size_t count) {
    if (row == nullptr || output == nullptr || bytes != 264 || count != 256) {
        throw std::invalid_argument("Engram FP8 row requires 264 bytes and 256 output values");
    }
    for (size_t i = 0; i < 256; ++i) {
        uint8_t value = row[i];
        int exponent = (value >> 3) & 15;
        int mantissa = value & 7;
        float decoded = exponent == 15 && mantissa == 7 ? std::numeric_limits<float>::quiet_NaN() :
            exponent == 0 ? std::ldexp(float(mantissa), -9) : std::ldexp(1.0f + mantissa / 8.0f, exponent - 7);
        if (value & 128) decoded = -decoded;
        uint8_t scale_byte = row[256 + i / 32];
        float scale = scale_byte == 255 ? std::numeric_limits<float>::quiet_NaN() : std::ldexp(1.0f, int(scale_byte) - 127);
        output[i] = decoded * scale;
    }
}

static float read_bf16_le(const uint8_t * data) {
    uint32_t bits = (uint32_t(data[0]) | uint32_t(data[1]) << 8) << 16;
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void engram_decode_affine2(const uint8_t * row, size_t bytes, float * output, size_t count) {
    if (row == nullptr || output == nullptr || bytes != 80 || count != 256) {
        throw std::invalid_argument("Engram affine-2 row requires 80 bytes and 256 output values");
    }
    for (size_t i = 0; i < 256; ++i) {
        const unsigned q = (row[i / 4] >> ((i % 4) * 2)) & 3;
        const float scale = read_bf16_le(row + 64 + (i / 64) * 2);
        const float bias = read_bf16_le(row + 72 + (i / 64) * 2);
        output[i] = float(q) * scale + bias;
    }
}

} // namespace deepseek_v41
