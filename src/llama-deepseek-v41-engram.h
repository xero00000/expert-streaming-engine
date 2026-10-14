#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace deepseek_v41 {

// Constants exported from the official tokenizer normalization and NumPy RNG.
// Table row IDs are local to each layer's separate Engram tensor.
struct engram_layer_config {
    uint32_t layer_id = 0;
    uint64_t rows = 0;
    std::array<uint64_t, 4> multipliers{};
    std::array<uint64_t, 24> primes{};
};

struct engram_hash_config {
    uint32_t compressed_vocab_size = 0;
    uint32_t pad_compressed_id = 0;
    std::vector<uint32_t> token_map;
    std::array<engram_layer_config, 2> layers{};
};

using engram_rows = std::array<std::array<uint64_t, 24>, 2>;

// Keeps compressed token history (four bytes/position) for exact suffix rollback.
// Instances are owned by one inference context; calls require external serialization.
// Image spans are DEAD history entries, not ordinary pad tokens. EOS is ordinary
// history too: sequence lifecycle, not a guessed token delimiter, resets the hash.
class engram_sequence_state {
public:
    explicit engram_sequence_state(engram_hash_config config);

    engram_rows accept(int32_t sequence_id, size_t position, uint32_t token_id, bool is_text = true);
    void rewind(int32_t sequence_id, size_t position);
    void reset(int32_t sequence_id);
    void clear();
    void copy(int32_t source_sequence_id, int32_t destination_sequence_id);

    // Snapshots contain compressed IDs, or -1 for an image/dead token. Restore
    // only under the same model/token-map identity as the exporting context.
    std::vector<int32_t> snapshot(int32_t sequence_id) const;
    void restore(int32_t sequence_id, const std::vector<int32_t> & compressed_tokens);
    size_t position(int32_t sequence_id) const;

private:
    engram_hash_config config_;
    std::array<std::array<uint64_t, 24>, 2> offsets_{};
    std::unordered_map<int32_t, std::vector<int32_t>> sequences_;
};

// Portable CPU row decoders, yielding FP32 before the model's BF16 cast.
// FP8 sidecar: 256 E4M3FN bytes then 8 E8M0 scale bytes (groups of 32).
void engram_decode_fp8(const uint8_t * row, size_t bytes, float * output, size_t count);

// MLX affine-2 sidecar: 64 little-endian packed weight bytes, four BF16 scales
// (8 bytes), then four BF16 biases (8 bytes), groups of 64. This preserves the
// existing quantization; it does not requantize or expand the full table.
void engram_decode_affine2(const uint8_t * row, size_t bytes, float * output, size_t count);

} // namespace deepseek_v41
