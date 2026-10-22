#pragma once
#include <cstdint>
#include <vector>

namespace deepseek_v41 {

// Host selection for one query. Scores are after the index-head weighted sum.
// Only [0, visible) is causal; NaNs and positive infinities are rejected.
// Ties use the lower position, making selection deterministic across backends.
std::vector<uint8_t> select_candidate_blocks(const std::vector<float> & scores,
    uint32_t visible, uint32_t block_size, uint32_t top_blocks);

// Return min(top_k, scores.size()) entries in position order, offset into the
// attention row union; missing reachable picks are trailing -1 sentinels.
// An empty candidate vector means no candidate filtering.
std::vector<int32_t> select_index_positions(const std::vector<float> & scores,
    uint32_t visible, uint32_t top_k, uint32_t offset,
    const std::vector<uint8_t> & candidates = {});

} // namespace deepseek_v41
