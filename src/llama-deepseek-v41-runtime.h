#pragma once
#include "llama-deepseek-v41-forward.h"
#include "llama-deepseek-v41-engram.h"
#include "llama-qwen4exp-ple-storage.h"
#include <memory>

namespace deepseek_v41 {

enum class engram_row_format { fp8, affine2 };
struct engram_asset {
    std::shared_ptr<const llama_expert_cache::source> source;
    qwen4exp_ple_storage::fixed_row_layout layout;
    engram_row_format format = engram_row_format::affine2;
};

struct runtime_assets {
    engram_hash_config hash;
    std::array<engram_asset, 2> rows;
};
// Embedded GGUF JSON metadata: version=1, hash (including token_map), and
// two raw row-file descriptors. Files are relative to the GGUF directory.
runtime_assets load_runtime_assets(const model_config & model, const std::string & metadata,
                                  const std::string & model_directory);

// One append-only text sequence. Caller serializes use and synchronizes graph
// execution before commit/abort/clear. All row I/O completes before begin returns.
class runtime_state {
public:
    runtime_state(model_config model, uint32_t capacity, engram_hash_config hash,
        std::array<engram_asset, 2> assets, uint64_t row_cache_bytes = 1 << 20,
        ggml_backend_buffer_type_t cache_type = nullptr);
    runtime_state(const runtime_state &) = delete;
    runtime_state & operator=(const runtime_state &) = delete;
    void begin(int32_t sequence, uint32_t position, uint32_t token);
    void begin_batch(int32_t sequence, uint32_t position, const std::vector<uint32_t> & tokens);
    void commit();
    void abort();
    void save_checkpoint(int32_t sequence, uint32_t max_tokens = attention_forward_state::max_batch);
    void restore_checkpoint(int32_t sequence, uint32_t position);
    void discard_checkpoint();
    void clear();
    uint32_t position() const;
    bool has_sequence(int32_t sequence) const { return bound_ && sequence == sequence_; }
    // Full deletion and empty ranges are supported. Historical suffix removal
    // requires replay because overwritten window rows are no longer retained.
    bool remove(int32_t sequence, int32_t first, int32_t end);
    bool active() const { return active_; }
    const std::map<uint32_t, std::vector<float>> & decoded_rows() const;
    attention_forward_state & forward() { return forward_; }
    qwen4exp_ple_storage::row_store_stats row_stats(size_t layer) const;
private:
    uint32_t capacity_;
    attention_forward_state forward_;
    engram_sequence_state hash_;
    std::array<uint32_t, 2> layers_;
    std::array<engram_row_format, 2> formats_;
    std::array<std::unique_ptr<qwen4exp_ple_storage::row_store>, 2> stores_;
    std::map<uint32_t, std::vector<float>> decoded_;
    int32_t sequence_ = 0;
    uint32_t begin_position_ = 0;
    uint64_t owner_ = 0;
    bool bound_ = false, active_ = false;
    int32_t checkpoint_sequence_ = 0;
    bool checkpoint_bound_ = false;
};
} // namespace deepseek_v41
