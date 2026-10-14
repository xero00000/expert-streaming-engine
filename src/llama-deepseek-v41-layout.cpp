#include "llama-deepseek-v41-layout.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace deepseek_v41 {

static uint64_t multiply(uint64_t a, uint64_t b) {
    if (b && a > std::numeric_limits<uint64_t>::max() / b) throw std::overflow_error("V4.1 cache size overflow");
    return a * b;
}

static uint64_t add(uint64_t a, uint64_t b) {
    if (a > std::numeric_limits<uint64_t>::max() - b) throw std::overflow_error("V4.1 cache size overflow");
    return a + b;
}

attention_layout::attention_layout(attention_layout_config config) : config_(std::move(config)) {
    const size_t count = config_.compress_ratios.size();
    if (count == 0 || count > size_t(std::numeric_limits<int32_t>::max()) ||
            config_.head_dim == 0 || config_.index_head_dim == 0 || config_.sliding_window == 0) {
        throw std::invalid_argument("invalid V4.1 attention geometry");
    }
    for (const auto * sources : {&config_.kv_source_layers, &config_.index_source_layers}) {
        if (!std::is_sorted(sources->begin(), sources->end()) ||
                std::adjacent_find(sources->begin(), sources->end()) != sources->end()) {
            throw std::invalid_argument("V4.1 cache source IDs must be sorted and unique");
        }
        for (uint32_t source : *sources) {
            if (source >= count || config_.compress_ratios[source] == 0) {
                throw std::invalid_argument("V4.1 cache source must be a compressed main layer");
            }
        }
    }
    if (config_.candidate_source_layer < -1 ||
            (config_.candidate_source_layer >= 0 && !std::binary_search(config_.index_source_layers.begin(),
                config_.index_source_layers.end(), uint32_t(config_.candidate_source_layer)))) {
        throw std::invalid_argument("V4.1 candidate source must own an indexer");
    }
    int32_t kv_owner = -1, topk_owner = -1;
    for (size_t layer = 0; layer < count; ++layer) {
        const uint32_t ratio = config_.compress_ratios[layer];
        attention_layer_layout item;
        item.compress_ratio = ratio;
        if (ratio == 0) {
            layers_.push_back(item);
            continue;
        }
        const bool owns_kv = std::binary_search(config_.kv_source_layers.begin(), config_.kv_source_layers.end(), uint32_t(layer));
        const bool owns_indexer = std::binary_search(config_.index_source_layers.begin(), config_.index_source_layers.end(), uint32_t(layer));
        if (owns_kv) {
            if (!owns_indexer) throw std::invalid_argument("V4.1 KV source must publish index keys and top-k indices");
            kv_owner = int32_t(layer);
        }
        if (kv_owner < 0 || config_.compress_ratios[kv_owner] != ratio) {
            throw std::invalid_argument("V4.1 layer has no compatible shared KV source");
        }
        if (owns_indexer) topk_owner = int32_t(layer);
        if (topk_owner < kv_owner) throw std::invalid_argument("V4.1 layer reuses indices from another KV bank");
        item.mode = owns_kv ? attention_mode::full : owns_indexer ? attention_mode::reindex : attention_mode::reuse;
        item.kv_owner = kv_owner;
        item.index_key_owner = kv_owner;
        item.topk_owner = topk_owner;
        if (config_.candidate_source_layer >= 0 && int32_t(layer) >= config_.candidate_source_layer) {
            const int32_t candidate_kv = int32_t(layer) == config_.candidate_source_layer ? kv_owner :
                layers_[config_.candidate_source_layer].kv_owner;
            if (candidate_kv != kv_owner) throw std::invalid_argument("V4.1 candidate pool refers to another KV bank");
            item.candidate_owner = config_.candidate_source_layer;
        }
        layers_.push_back(item);
    }
}

attention_payload_budget attention_layout::payload_budget(uint64_t context, kv_encoding encoding) const {
    if (context == 0) throw std::invalid_argument("V4.1 context must be positive");
    uint64_t main_row = 0, index_row = 0, window_row = 0;
    if (encoding == kv_encoding::fp32_reference) {
        main_row = multiply(config_.head_dim, 4);
        index_row = multiply(config_.index_head_dim, 4);
        window_row = main_row;
    } else if (encoding == kv_encoding::fp4_compressed) {
        if (config_.head_dim % 32 || config_.index_head_dim % 32) {
            throw std::invalid_argument("V4.1 quantized cache dimensions must be multiples of 32");
        }
        // Main KV: FP4 + one E4M3 scale/16. Index keys: FP4 + one E8M0 scale/32.
        main_row = uint64_t(config_.head_dim) / 2 + config_.head_dim / 16;
        index_row = uint64_t(config_.index_head_dim) / 2 + config_.index_head_dim / 32;
        // SWA uses FP8 + one E8M0 scale/32, independently in every layer.
        window_row = uint64_t(config_.head_dim) + config_.head_dim / 32;
    } else {
        throw std::invalid_argument("unknown V4.1 KV encoding");
    }
    attention_payload_budget budget;
    for (uint32_t owner : config_.kv_source_layers) {
        const uint64_t rows = context / config_.compress_ratios[owner];
        budget.main_kv_bytes = add(budget.main_kv_bytes, multiply(rows, main_row));
        budget.index_key_bytes = add(budget.index_key_bytes, multiply(rows, index_row));
    }
    budget.global_bytes = add(budget.main_kv_bytes, budget.index_key_bytes);
    budget.sliding_window_bytes = multiply(multiply(window_row, config_.sliding_window), layers_.size());
    return budget;
}

} // namespace deepseek_v41
