#pragma once

#include <cstdint>
#include <vector>

namespace deepseek_v41 {

enum class attention_mode { window_only, full, reindex, reuse };

struct attention_layout_config {
    std::vector<uint32_t> compress_ratios; // main layers only; DSpark is a separate graph
    std::vector<uint32_t> kv_source_layers;
    std::vector<uint32_t> index_source_layers;
    int32_t candidate_source_layer = -1;
    uint32_t head_dim = 0;
    uint32_t index_head_dim = 0;
    uint32_t sliding_window = 0;
};

struct attention_layer_layout {
    attention_mode mode = attention_mode::window_only;
    uint32_t compress_ratio = 0;
    int32_t kv_owner = -1;
    int32_t index_key_owner = -1;
    int32_t topk_owner = -1;
    int32_t candidate_owner = -1;
};

enum class kv_encoding { fp32_reference, fp4_compressed };

struct attention_payload_budget {
    uint64_t main_kv_bytes = 0;
    uint64_t index_key_bytes = 0;
    uint64_t sliding_window_bytes = 0;
    uint64_t global_bytes = 0;
};

// Explicit ownership prevents a reindexer or an incomplete compressor group
// from accidentally consuming whichever key tensor another layer last wrote.
class attention_layout {
public:
    explicit attention_layout(attention_layout_config config);
    const std::vector<attention_layer_layout> & layers() const { return layers_; }

    // Logical payload only: excludes allocator alignment, scratch, candidate
    // workspaces, weights and model state. FP4 encoding is a future graph path,
    // not a declaration that these buffers are already allocated/compressed.
    attention_payload_budget payload_budget(uint64_t context, kv_encoding encoding) const;

private:
    attention_layout_config config_;
    std::vector<attention_layer_layout> layers_;
};

} // namespace deepseek_v41
