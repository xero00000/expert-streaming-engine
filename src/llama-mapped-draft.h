#pragma once

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

#include <string>

struct llama_mapped_draft_result {
    ggml_tensor * logits = nullptr;
    ggml_tensor * candidate_ids = nullptr;
    ggml_tensor * centroid_logits = nullptr;
};

bool llama_mapped_draft_head_valid(
    int64_t n_vocab,
    int64_t n_centroids,
    int64_t top_k,
    const ggml_tensor * output,
    const ggml_tensor * centroids,
    const ggml_tensor * token_ordering,
    const ggml_tensor * hidden);

// Reads the publisher-supplied map after model tensors are loaded. A valid map
// is an exact permutation of [0, n_vocab), so every gather/scatter index is
// bounded and every target token has unambiguous provenance.
bool llama_mapped_draft_token_ordering_valid(
    const ggml_tensor * token_ordering,
    int64_t n_vocab,
    std::string * error = nullptr);

llama_mapped_draft_result llama_build_mapped_draft_logits(
    ggml_context * ctx,
    ggml_tensor * output,
    ggml_tensor * centroids,
    ggml_tensor * token_ordering,
    ggml_tensor * hidden,
    int64_t n_vocab,
    int64_t n_centroids,
    int64_t top_k);
