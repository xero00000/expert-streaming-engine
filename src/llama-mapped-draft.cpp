#include "llama-mapped-draft.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>

bool llama_mapped_draft_token_ordering_valid(
        const ggml_tensor * token_ordering,
        int64_t n_vocab,
        std::string * error) {
    if (token_ordering == nullptr || token_ordering->type != GGML_TYPE_I32 ||
            n_vocab <= 0 || ggml_nelements(token_ordering) != n_vocab ||
            token_ordering->buffer == nullptr) {
        if (error != nullptr) {
            *error = "mapped draft token ordering is missing, unloaded, or has the wrong shape/type";
        }
        return false;
    }

    std::vector<int32_t> ordering((size_t) n_vocab);
    ggml_backend_tensor_get(token_ordering, ordering.data(), 0, ordering.size()*sizeof(ordering[0]));
    std::vector<uint8_t> seen((size_t) n_vocab, 0);
    for (int64_t draft_id = 0; draft_id < n_vocab; ++draft_id) {
        const int32_t target_id = ordering[(size_t) draft_id];
        if (target_id < 0 || target_id >= n_vocab) {
            if (error != nullptr) {
                *error = "mapped draft token ordering contains out-of-range target id " +
                    std::to_string(target_id) + " at draft id " + std::to_string(draft_id);
            }
            return false;
        }
        if (seen[(size_t) target_id] != 0) {
            if (error != nullptr) {
                *error = "mapped draft token ordering contains duplicate target id " +
                    std::to_string(target_id);
            }
            return false;
        }
        seen[(size_t) target_id] = 1;
    }
    return true;
}

bool llama_mapped_draft_head_valid(
        int64_t n_vocab,
        int64_t n_centroids,
        int64_t top_k,
        const ggml_tensor * output,
        const ggml_tensor * centroids,
        const ggml_tensor * token_ordering,
        const ggml_tensor * hidden) {
    return n_vocab > 0 && n_centroids > 0 && top_k > 0 && top_k <= n_centroids &&
        n_vocab % n_centroids == 0 && output != nullptr && centroids != nullptr &&
        token_ordering != nullptr && hidden != nullptr && hidden->ne[1] == 1 &&
        output->ne[0] == hidden->ne[0] && output->ne[1] == n_vocab &&
        centroids->ne[0] == hidden->ne[0] && centroids->ne[1] == n_centroids &&
        token_ordering->type == GGML_TYPE_I32 && ggml_nelements(token_ordering) == n_vocab;
}

llama_mapped_draft_result llama_build_mapped_draft_logits(
        ggml_context * ctx,
        ggml_tensor * output,
        ggml_tensor * centroids,
        ggml_tensor * token_ordering,
        ggml_tensor * hidden,
        int64_t n_vocab,
        int64_t n_centroids,
        int64_t top_k) {
    GGML_ASSERT(llama_mapped_draft_head_valid(
        n_vocab, n_centroids, top_k, output, centroids, token_ordering, hidden));

    const int64_t vocab_per_centroid = n_vocab / n_centroids;
    const int64_t n_candidates = vocab_per_centroid * top_k;
    llama_mapped_draft_result result;
    result.centroid_logits = ggml_mul_mat(ctx, centroids, hidden);
    ggml_tensor * centroid_ids = ggml_cont(ctx, ggml_top_k(ctx, result.centroid_logits, top_k));
    ggml_tensor * ordering = ggml_reshape_2d(ctx, token_ordering, vocab_per_centroid, n_centroids);
    result.candidate_ids = ggml_cont(ctx, ggml_get_rows(ctx, ordering, centroid_ids));
    result.candidate_ids = ggml_reshape_1d(ctx, result.candidate_ids, n_candidates);

    ggml_tensor * candidate_head = ggml_get_rows(ctx, output, result.candidate_ids);
    ggml_tensor * candidate_logits = ggml_mul_mat(ctx, candidate_head, hidden);
    ggml_tensor * full = ggml_fill(
        ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n_vocab), -1e30f);
    ggml_tensor * sparse_rows = ggml_reshape_2d(
        ctx, ggml_cont(ctx, candidate_logits), 1, n_candidates);
    result.logits = ggml_set_rows(ctx, full, sparse_rows, result.candidate_ids);
    result.logits = ggml_reshape_2d(ctx, ggml_cont(ctx, result.logits), n_vocab, 1);
    return result;
}
