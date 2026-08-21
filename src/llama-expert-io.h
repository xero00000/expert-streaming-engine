#pragma once

#include "llama-expert-cache.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_file_range {
    size_t first = 0;
    size_t last  = 0;

    bool empty() const {
        return first >= last;
    }
};

struct llama_expert_tensor_index {
    size_t deferred_bytes = 0;
    size_t dense_bytes = 0;

    std::vector<std::vector<llama_file_range>> file_ranges;

    struct source_info {
        uint32_t id = 0;
        llama_expert_cache::source_identity identity;
        uint64_t size = 0;
        std::string path;
    };

    llama_expert_cache::source_identity model_identity;
    std::vector<source_info> sources;
    std::map<llama_expert_cache::expert_key,
            std::shared_ptr<const llama_expert_cache::descriptor>> descriptors;

    bool empty() const {
        return deferred_bytes == 0;
    }

    const llama_expert_cache::descriptor * find(
            const llama_expert_cache::expert_key & key) const {
        const auto found = descriptors.find(key);
        return found == descriptors.end() ? nullptr : found->second.get();
    }
};
