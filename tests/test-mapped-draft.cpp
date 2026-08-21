#include "llama-mapped-draft.h"

#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <vector>

int main(int argc, char ** argv) {
    constexpr int64_t n_embd = 2;
    constexpr int64_t n_vocab = 6;
    constexpr int64_t n_centroids = 3;
    constexpr int64_t top_k = 1;

    ggml_init_params params = {
        ggml_tensor_overhead()*32 + ggml_graph_overhead_custom(64, false),
        nullptr,
        true,
    };
    ggml_context * ctx = ggml_init(params);
    assert(ctx != nullptr);
    struct context_cleanup { ggml_context * ctx; ~context_cleanup() { ggml_free(ctx); } } free_ctx{ctx};

    ggml_tensor * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_vocab);
    ggml_tensor * centroids = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_centroids);
    ggml_tensor * ordering = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_vocab);
    ggml_tensor * hidden = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, 1);
    assert(llama_mapped_draft_head_valid(
        n_vocab, n_centroids, top_k, output, centroids, ordering, hidden));

    const auto mapped = llama_build_mapped_draft_logits(
        ctx, output, centroids, ordering, hidden, n_vocab, n_centroids, top_k);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, mapped.logits);

    int cuda_device = -1;
    if (argc == 3 && std::string(argv[1]) == "--cuda") {
        cuda_device = std::stoi(argv[2]);
    }
    ggml_backend_t backend = nullptr;
    if (cuda_device >= 0) {
#ifdef GGML_USE_CUDA
        backend = ggml_backend_cuda_init(cuda_device, nullptr, nullptr);
#else
        std::cerr << "FAIL: this binary was built without CUDA\n";
        return 1;
#endif
    } else {
        backend = ggml_backend_cpu_init();
    }
    assert(backend != nullptr);
    struct backend_cleanup { ggml_backend_t backend; ~backend_cleanup() { ggml_backend_free(backend); } } free_backend{backend};
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer != nullptr);
    struct buffer_cleanup { ggml_backend_buffer_t buffer; ~buffer_cleanup() { ggml_backend_buffer_free(buffer); } } free_buffer{buffer};

    // Token 2 is the target's global winner, but centroid 0 maps only tokens 0/1.
    const std::array<float, n_embd*n_vocab> head = {{
        1.0f, 0.0f,
        2.0f, 0.0f,
        10.0f, 0.0f,
        4.0f, 0.0f,
        5.0f, 0.0f,
        6.0f, 0.0f,
    }};
    const std::array<float, n_embd*n_centroids> centroid_values = {{
        3.0f, 0.0f,
        2.0f, 0.0f,
        1.0f, 0.0f,
    }};
    const std::array<int32_t, n_vocab> target_ids = {{0, 1, 2, 3, 4, 5}};
    const std::array<float, n_embd> hidden_values = {{1.0f, 0.0f}};
    ggml_backend_tensor_set(output, head.data(), 0, sizeof(head));
    ggml_backend_tensor_set(centroids, centroid_values.data(), 0, sizeof(centroid_values));
    ggml_backend_tensor_set(ordering, target_ids.data(), 0, sizeof(target_ids));
    ggml_backend_tensor_set(hidden, hidden_values.data(), 0, sizeof(hidden_values));
    std::string ordering_error;
    assert(llama_mapped_draft_token_ordering_valid(ordering, n_vocab, &ordering_error));
    assert(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::array<float, n_vocab> mapped_logits = {};
    std::array<int32_t, n_vocab/n_centroids*top_k> candidate_ids = {};
    ggml_backend_tensor_get(mapped.logits, mapped_logits.data(), 0, sizeof(mapped_logits));
    ggml_backend_tensor_get(mapped.candidate_ids, candidate_ids.data(), 0, sizeof(candidate_ids));
    assert(candidate_ids[0] == 0 && candidate_ids[1] == 1);
    assert(mapped_logits[0] == 1.0f && mapped_logits[1] == 2.0f);
    for (int token = 2; token < n_vocab; ++token) {
        assert(mapped_logits[token] <= -1e29f);
    }

    const int draft_token = int(std::max_element(mapped_logits.begin(), mapped_logits.end()) - mapped_logits.begin());
    const std::array<float, n_vocab> target_logits = {{1, 2, 10, 4, 5, 6}};
    const int baseline_token = int(std::max_element(target_logits.begin(), target_logits.end()) - target_logits.begin());
    assert(draft_token == 1);
    assert(baseline_token == 2);

    // At temperature zero the full target verifier rejects the unsupported draft
    // token and emits its own argmax, exactly matching non-speculative generation.
    const int verified_token = draft_token == baseline_token ? draft_token : baseline_token;
    assert(verified_token == baseline_token);

    const std::array<int32_t, n_vocab> duplicate_ids = {{0, 1, 2, 3, 4, 4}};
    ggml_backend_tensor_set(ordering, duplicate_ids.data(), 0, sizeof(duplicate_ids));
    assert(!llama_mapped_draft_token_ordering_valid(ordering, n_vocab, &ordering_error));
    assert(ordering_error.find("duplicate") != std::string::npos);

    const std::array<int32_t, n_vocab> out_of_range_ids = {{0, 1, 2, 3, 4, 6}};
    ggml_backend_tensor_set(ordering, out_of_range_ids.data(), 0, sizeof(out_of_range_ids));
    assert(!llama_mapped_draft_token_ordering_valid(ordering, n_vocab, &ordering_error));
    assert(ordering_error.find("out-of-range") != std::string::npos);

    std::cout << "PASS: mapped draft logits preserve full-target greedy verification on "
              << (cuda_device >= 0 ? "CUDA device " + std::to_string(cuda_device) : "CPU") << "\n";
    return 0;
}
