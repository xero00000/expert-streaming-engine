#include "llama-qwen4exp-ops.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static void check_integer_repeat(ggml_backend_t backend) {
    auto * ctx = ggml_init({1 << 20, nullptr, true});
    require(ctx, "integer-repeat context");
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_I32, 19, 5, 2);
    auto * input = ggml_view_3d(ctx, storage, 17, 3, 2, storage->nb[1], storage->nb[2], 0);
    auto * output = ggml_repeat_4d(ctx, input, 34, 6, 4, 1);
    require(ggml_backend_supports_op(backend, output), "integer-repeat support query");
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "integer-repeat buffers");
    const uint32_t patterns[] = {0x7c01, 0xfc01, 0x7fff, 0xffff, 0x80000000,
        0xffffffff, 0x7c017e00, 0xfc01fe00, 0, 1, 1048575};
    std::vector<uint32_t> values(19 * 5 * 2), actual(34 * 6 * 4);
    for (size_t i = 0; i < values.size(); ++i) values[i] = patterns[i % (sizeof(patterns) / sizeof(patterns[0]))];
    ggml_backend_tensor_set(storage, values.data(), 0, values.size() * sizeof(uint32_t));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "integer-repeat compute");
    ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(uint32_t));
    for (int z = 0; z < 4; ++z) for (int y = 0; y < 6; ++y) for (int x = 0; x < 34; ++x)
        require(actual[size_t((z * 6 + y) * 34 + x)] == values[size_t((z % 2 * 5 + y % 3) * 19 + x % 17)],
                "integer-repeat bit preservation");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static ggml_tensor * attention_output(ggml_context * ctx, ggml_tensor * q,
        ggml_tensor * k, ggml_tensor * v, ggml_tensor * mask, bool flash) {
    if (flash) {
        auto * output = ggml_flash_attn_ext(ctx, q, k, v, mask, 0.0625f, 0, 0);
        ggml_flash_attn_ext_set_prec(output, GGML_PREC_F32);
        return output;
    }
    auto * scores = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    auto * probabilities = ggml_soft_max_ext(ctx, scores, mask, 0.0625f, 0);
    auto * output = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, v)), probabilities);
    ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    return output;
}

static void check(ggml_backend_t backend, int64_t keys, ggml_type cache_type, bool attention) {
    constexpr int64_t heads = 2, queries = 3, kept = 2304, width = 256, query_heads = 24;
    auto * ctx = ggml_init({4 << 20, nullptr, true});
    require(ctx, "context");
    auto * rows = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, keys, heads);
    auto * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, keys, queries);
    auto * selected = qwen4exp_build_compact_attention_indices(ctx, mask, kept);
    auto * graph = ggml_new_graph_custom(ctx, 512, false);
    qwen4exp_compact_attention_rows compact[queries];
    ggml_tensor * dense_output[queries] = {}, * compact_output[queries] = {};
    ggml_tensor * k = nullptr, * v = nullptr, * q = nullptr;
    if (attention) {
        k = ggml_new_tensor_3d(ctx, cache_type, width, keys, heads);
        v = ggml_new_tensor_3d(ctx, cache_type, width, keys, heads);
        q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, query_heads);
    }
    for (int64_t query = 0; query < queries; ++query) {
        compact[query] = qwen4exp_build_compact_attention_rows(ctx, rows, mask, selected, query);
        ggml_build_forward_expand(graph, compact[query].rows);
        ggml_build_forward_expand(graph, compact[query].mask);
        if (!attention) continue;
        auto * dense_k = qwen4exp_build_cache_rows(ctx, k, rows);
        auto * dense_v = qwen4exp_build_cache_rows(ctx, v, rows);
        auto * dense_mask = ggml_view_2d(ctx, mask, keys, 1, mask->nb[1], query * mask->nb[1]);
        dense_mask = ggml_cast(ctx, ggml_pad(ctx, dense_mask, 0, GGML_KQ_MASK_PAD - 1, 0, 0), GGML_TYPE_F16);
        auto * compact_k = qwen4exp_build_cache_rows(ctx, k, compact[query].rows);
        auto * compact_v = qwen4exp_build_cache_rows(ctx, v, compact[query].rows);
        dense_output[query] = attention_output(ctx, q, dense_k, dense_v, dense_mask, cache_type == GGML_TYPE_F16);
        compact_output[query] = attention_output(ctx, q, compact_k, compact_v, compact[query].mask, cache_type == GGML_TYPE_F16);
        ggml_build_forward_expand(graph, dense_output[query]);
        ggml_build_forward_expand(graph, compact_output[query]);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "buffers");
    std::vector<int32_t> row_data(size_t(keys * heads));
    for (int64_t head = 0; head < heads; ++head) for (int64_t i = 0; i < keys; ++i)
        row_data[size_t(head * keys + i)] = int32_t((i * 73 + head * 17) % keys);
    std::vector<float> mask_data(size_t(keys * queries), -std::numeric_limits<float>::infinity());
    std::vector<std::vector<int32_t>> active(queries);
    for (int query = 0; query < queries; ++query) {
        const int count = query == 0 ? 17 : query == 1 ? 2079 : 1;
        for (int i = 0; i < count; ++i) {
            const int32_t index = int32_t((int64_t(i) * 97 + query * 31) % keys);
            mask_data[size_t(query * keys + index)] = 0;
            active[query].push_back(index);
        }
        std::sort(active[query].begin(), active[query].end());
    }
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
    ggml_backend_tensor_set(mask, mask_data.data(), 0, mask_data.size() * sizeof(float));
    if (attention) {
        std::vector<float> values(size_t(width * keys * heads));
        for (int role = 0; role < 2; ++role) {
            auto * tensor = role ? v : k;
            for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i * 7 + role * 19) % 127) - 63) / 127;
            if (cache_type == GGML_TYPE_F32) {
                ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
            } else if (cache_type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> packed(values.size());
                for (size_t i = 0; i < values.size(); ++i) packed[i] = ggml_fp32_to_fp16(values[i]);
                ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size() * sizeof(ggml_fp16_t));
            } else {
                std::vector<uint8_t> packed(ggml_nbytes(tensor));
                require(ggml_quantize_chunk(cache_type, values.data(), packed.data(), 0, keys * heads, width,
                        nullptr, nullptr) == packed.size(), "quantize");
                ggml_backend_tensor_set(tensor, packed.data(), 0, packed.size());
            }
        }
        values.resize(width * query_heads);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(int((i * 11) % 97) - 48) / 97;
        ggml_backend_tensor_set(q, values.data(), 0, values.size() * sizeof(float));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    std::vector<int32_t> logical(size_t(kept * queries));
    ggml_backend_tensor_get(selected, logical.data(), 0, logical.size() * sizeof(int32_t));
    for (int query = 0; query < queries; ++query) {
        std::vector<int32_t> physical(size_t(kept * heads));
        std::vector<ggml_fp16_t> compact_mask(size_t(kept * GGML_KQ_MASK_PAD));
        ggml_backend_tensor_get(compact[query].rows, physical.data(), 0, physical.size() * sizeof(int32_t));
        ggml_backend_tensor_get(compact[query].mask, compact_mask.data(), 0, compact_mask.size() * sizeof(ggml_fp16_t));
        std::vector<int32_t> unique;
        for (int64_t i = 0; i < kept; ++i) {
            const int32_t index = logical[size_t(query * kept + i)];
            require(index >= 0 && index < keys, "logical range");
            unique.push_back(index);
            if (i < int64_t(active[query].size())) require(index == active[query][size_t(i)], "visible order");
            const float expected = mask_data[size_t(query * keys + index)];
            require(ggml_fp16_to_fp32(compact_mask[size_t(i)]) == expected, "mask preserved");
            for (int64_t head = 0; head < heads; ++head) {
                if (physical[size_t(head * kept + i)] != row_data[size_t(head * keys + index)]) {
                    std::fprintf(stderr, "row mismatch: backend=%s keys=%lld query=%d slot=%lld head=%lld logical=%d actual=%d expected=%d\n",
                            ggml_backend_name(backend), (long long) keys, query, (long long) i, (long long) head,
                            index, physical[size_t(head * kept + i)], row_data[size_t(head * keys + index)]);
                    require(false, "physical head mapping");
                }
            }
        }
        std::sort(unique.begin(), unique.end());
        require(std::adjacent_find(unique.begin(), unique.end()) == unique.end(), "no duplicated logical rows");
        if (!attention) continue;
        std::vector<float> reference(width * query_heads), actual(reference.size());
        ggml_backend_tensor_get(dense_output[query], reference.data(), 0, reference.size() * sizeof(float));
        ggml_backend_tensor_get(compact_output[query], actual.data(), 0, actual.size() * sizeof(float));
        for (size_t i = 0; i < actual.size(); ++i) {
            if (!std::isfinite(actual[i]) || std::abs(actual[i] - reference[i]) > 2e-5f * (1 + std::abs(reference[i]))) {
                std::fprintf(stderr, "attention mismatch: backend=%s cache=%s query=%d index=%zu actual=%.9g dense=%.9g\n",
                        ggml_backend_name(backend), ggml_type_name(cache_type), query, i, actual[i], reference[i]);
                require(false, "dense/compact attention bound");
            }
        }
    }
    std::printf("compact QSA: backend=%s keys=%lld cache=%s attention=%d passed\n",
            ggml_backend_name(backend), (long long) keys, ggml_type_name(cache_type), attention);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main() {
    try {
        auto run = [](ggml_backend_t backend, bool attention) {
            require(backend, "backend");
            check_integer_repeat(backend);
            check(backend, 4096, GGML_TYPE_F32, attention);
            if (attention) {
                check(backend, 4096, GGML_TYPE_Q4_0, true);
                check(backend, 4096, GGML_TYPE_F16, true);
            }
            check(backend, 1048576, GGML_TYPE_F32, false);
            ggml_backend_free(backend);
        };
        auto * cpu = ggml_backend_cpu_init();
        ggml_backend_cpu_set_n_threads(cpu, 4);
        run(cpu, false);
#ifdef GGML_USE_CUDA
        if (ggml_backend_cuda_get_device_count()) run(ggml_backend_cuda_init(0, nullptr, nullptr), true);
#endif
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
