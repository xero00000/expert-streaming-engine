#include "llama-deepseek-v41-index.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace deepseek_v41 {
static void validate(const std::vector<float> & scores, uint32_t visible) {
    if (visible > scores.size() || scores.size() > uint64_t(INT32_MAX))
        throw std::invalid_argument("invalid V4.1 index extent");
    for (float score : scores) if (std::isnan(score) || score == INFINITY)
        throw std::invalid_argument("nonfinite V4.1 index score");
}

static void best_first(std::vector<uint32_t> & positions, const std::vector<float> & scores, size_t count) {
    auto better = [&](uint32_t a, uint32_t b) { return scores[a] > scores[b] || (scores[a] == scores[b] && a < b); };
    std::partial_sort(positions.begin(), positions.begin() + count, positions.end(), better);
    positions.resize(count);
}

std::vector<uint8_t> select_candidate_blocks(const std::vector<float> & scores,
    uint32_t visible, uint32_t block_size, uint32_t top_blocks) {
    validate(scores, visible);
    if (!block_size || !top_blocks) throw std::invalid_argument("invalid V4.1 candidate parameters");
    std::vector<uint8_t> keep(scores.size(), 0);
    if (!visible) return keep;
    const size_t blocks = (uint64_t(visible) + block_size - 1) / block_size;
    std::vector<float> maxima(blocks, -INFINITY);
    for (uint32_t i = 0; i < visible; ++i) maxima[i / block_size] = std::max(maxima[i / block_size], scores[i]);
    maxima[(visible - 1) / block_size] = INFINITY; // pin newest partly filled block
    std::vector<uint32_t> order(blocks);
    std::iota(order.begin(), order.end(), 0);
    best_first(order, maxima, std::min<size_t>(top_blocks, blocks));
    for (uint32_t block : order) if (maxima[block] > -INFINITY) {
        // Candidate masks represent blocks, including their currently unseen
        // suffix. The consuming indexer must apply its own causal visibility.
        const size_t end = std::min<uint64_t>(scores.size(), uint64_t(block + 1) * block_size);
        for (size_t i = uint64_t(block) * block_size; i < end; ++i) keep[i] = 1;
    }
    return keep;
}

std::vector<int32_t> select_index_positions(const std::vector<float> & scores,
    uint32_t visible, uint32_t top_k, uint32_t offset, const std::vector<uint8_t> & candidates) {
    validate(scores, visible);
    if ((!candidates.empty() && candidates.size() != scores.size()) ||
        uint64_t(offset) + scores.size() > uint64_t(INT32_MAX))
        throw std::invalid_argument("invalid V4.1 index offset/candidate mask");
    const size_t count = std::min<size_t>(top_k, scores.size());
    std::vector<uint32_t> positions;
    for (uint32_t i = 0; i < visible; ++i)
        if ((candidates.empty() || candidates[i]) && std::isfinite(scores[i])) positions.push_back(i);
    best_first(positions, scores, std::min(count, positions.size()));
    std::sort(positions.begin(), positions.end());
    std::vector<int32_t> result(count, -1);
    for (size_t i = 0; i < positions.size(); ++i) result[i] = int32_t(positions[i] + offset);
    return result;
}
} // namespace deepseek_v41
