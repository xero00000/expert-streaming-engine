#include "llama-context.h"
#include "llama-model.h"
#include "llama-qwen4exp-ops.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
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

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static void rotate(std::vector<double> & row, int rotary, int position) {
    for (int d = 0; d < rotary / 2; ++d) {
        const double angle = position * std::pow(10000., -2. * d / rotary);
        const double a = row[d], b = row[d + rotary / 2];
        row[d] = a * std::cos(angle) - b * std::sin(angle);
        row[d + rotary / 2] = a * std::sin(angle) + b * std::cos(angle);
    }
}

static int position(int row, int shift) { return shift + row + (shift && row >= 17 ? 5 : 0); }

static void check_log(ggml_backend_t compute, ggml_backend_t cpu) {
    auto * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx, "log context");
    auto * x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 12);
    auto * y = ggml_log(ctx, x);
    ggml_set_input(x); ggml_set_output(y);
    require(ggml_backend_supports_op(compute, y), "contiguous F32 log supported");
    if (compute != cpu) {
        auto * strided = ggml_view_2d(ctx, x, 2, 2, 4 * sizeof(float), 0);
        require(!ggml_backend_supports_op(compute, ggml_log(ctx, strided)), "strided log rejected");
        auto * half = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, 8);
        require(!ggml_backend_supports_op(compute, ggml_log(ctx, half)), "F16 log rejected");
    }
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, compute);
    require(buffer, "log allocation");
    const float values[2][12] = {
        {0.f, -0.f, 1.f, .25f, 4.f, INFINITY, -1.f, NAN,
         std::numeric_limits<float>::denorm_min(), std::numeric_limits<float>::min(),
         std::numeric_limits<float>::max(), std::nextafter(1.f, 2.f)},
        {1.f, 0.f, 4.f, .5f, 16.f, -INFINITY, NAN, .125f,
         1.e-30f, 1.e30f, -std::numeric_limits<float>::denorm_min(), std::nextafter(1.f, 0.f)}};
    for (const auto & input : values) {
        ggml_backend_tensor_set(x, input, 0, sizeof(input));
        require(ggml_backend_graph_compute(compute, graph) == GGML_STATUS_SUCCESS, "log compute");
        float output[12]; ggml_backend_tensor_get(y, output, 0, sizeof(output));
        for (int i = 0; i < 12; ++i) {
            uint32_t bits; std::memcpy(&bits, &input[i], sizeof(bits));
            double value = input[i];
            if ((bits & 0x7f800000u) == 0 && (bits & 0x007fffffu)) {
                value = std::ldexp(double(bits & 0x007fffffu), -149);
                if (bits & 0x80000000u) value = -value;
            }
            const float expected = float(std::log(value));
            if (std::isnan(expected)) require(std::isnan(output[i]), "log NaN");
            else if (std::isinf(expected)) require(output[i] == expected, "log signed infinity");
            else {
                const float tolerance = i < 8 ? 1.e-6f :
                        2 * std::numeric_limits<float>::epsilon() * (1 + std::fabs(expected));
                if (!std::isfinite(output[i]) || std::fabs(output[i] - expected) >= tolerance)
                    std::fprintf(stderr, "LOG input=%a output=%a expected=%a tolerance=%g\n",
                            input[i], output[i], expected, tolerance);
                require(std::isfinite(output[i]) && std::fabs(output[i] - expected) < tolerance, "log finite value");
            }
        }
    }
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    std::puts("LOG: finite/domain/replay checks pass");
}

static void run(ggml_backend_t compute, ggml_backend_t cpu, int dims, int heads, int owned, int queries, int shift) {
    constexpr int physical = 2048, ratio = 4, selected_count = 2;
    constexpr float eps = 1.e-6f;
    const int rotary = dims / 2;
    llama_model model{};
    model.arch = LLM_ARCH_LLAMA; model.hparams.n_layer = 0; model.hparams.n_embd = 4; model.hparams.n_vocab = 4;
    llama_context lctx(model); lctx.cparams = {}; lctx.cparams.flash_attn = true;
    lctx.kv_self.size = physical; lctx.kv_self.n = physical; lctx.kv_self.cells.resize(physical);
    llama_batch batch{}; batch.n_tokens = queries;
    std::vector<llama_pos> positions(queries);
    std::vector<int32_t> sequence_counts(queries, 1);
    llama_seq_id sequence = 0;
    std::vector<llama_seq_id *> sequences(queries, &sequence);
    for (int q = 0; q < queries; ++q) positions[q] = position(owned - queries + q, shift);
    batch.pos = positions.data(); batch.n_seq_id = sequence_counts.data(); batch.seq_id = sequences.data();
    auto populate_cells = [&](bool shuffled) {
        lctx.kv_self.cells.assign(physical, {});
        for (int r = 0; r < owned; ++r) {
            const int slot = shuffled ? (r * 73 + 91) % physical : r;
            lctx.kv_self.cells[slot].pos = position(r, shift); lctx.kv_self.cells[slot].seq_id.insert(0);
        }
        for (int r = 0; r < 256; ++r) {
            const int slot = shuffled ? ((r + 1024) * 73 + 91) % physical : r + 1024;
            lctx.kv_self.cells[slot].pos = r; lctx.kv_self.cells[slot].seq_id.insert(1);
        }
    };
    populate_cells(false);
    auto input = llama_qwen4exp_attention_layout(lctx, batch).at(0);
    const int keys = input.key_count, blocks = keys / ratio;
    auto * in = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    require(in && ctx, "contexts");
    input.rows = ggml_new_tensor_2d(in, GGML_TYPE_I32, keys, 2);
    input.causal_mask = ggml_new_tensor_2d(in, GGML_TYPE_F32, keys, 16);
    input.block_pos = ggml_new_tensor_1d(in, GGML_TYPE_I32, blocks);
    input.block_end = ggml_new_tensor_1d(in, GGML_TYPE_I32, blocks);
    input.block_map = ggml_new_tensor_2d(in, GGML_TYPE_I32, ratio, blocks);
    input.tail = ggml_new_tensor_2d(in, GGML_TYPE_I32, ratio - 1, queries);
    input.tail_enabled = ggml_new_tensor_2d(in, GGML_TYPE_F32, 1, queries);
    auto * raw = ggml_new_tensor_2d(in, GGML_TYPE_F16, dims, physical);
    auto * gamma = ggml_new_tensor_1d(in, GGML_TYPE_F32, dims);
    auto * query = ggml_new_tensor_3d(in, GGML_TYPE_F32, dims, heads, queries);
    auto * query_pos = ggml_new_tensor_1d(in, GGML_TYPE_I32, queries);
    for (auto * t : {input.rows, input.causal_mask, input.block_pos, input.block_end, input.block_map,
            input.tail, input.tail_enabled, raw, gamma, query, query_pos}) ggml_set_input(t);
    auto * leaf_buffer = ggml_backend_alloc_ctx_tensors(in, cpu); require(leaf_buffer, "host inputs");
    lctx.qwen4exp_attention_inputs = {input};

    auto * ids = ggml_view_2d(ctx, input.rows, keys, 1, input.rows->nb[1], 0);
    auto * ordered = qwen4exp_build_cache_rows(ctx, raw, ids);
    auto * pooled = qwen4exp_build_qsa_pooled_keys(ctx, ordered, gamma, ratio, blocks, eps);
    auto * block_keys = qwen4exp_build_qsa_rope(ctx, ggml_reshape_3d(ctx, pooled, dims, 1, blocks),
            input.block_pos, rotary, 4096, 10000, 1, 0, 1, 32, 1);
    block_keys = ggml_reshape_2d(ctx, block_keys, dims, blocks);
    auto * norm_query = ggml_rms_norm(ctx, query, eps);
    auto * rotated_query = qwen4exp_build_qsa_rope(ctx, norm_query, query_pos,
            rotary, 4096, 10000, 1, 0, 1, 32, 1);
    auto * transposed = ggml_cont(ctx, ggml_transpose(ctx, input.causal_mask));
    auto * end_masks = ggml_cont(ctx, ggml_transpose(ctx, ggml_get_rows(ctx, transposed, input.block_end)));
    auto * block_mask = ggml_view_2d(ctx, end_masks, blocks, queries, end_masks->nb[1], 0);
    const auto selection = qwen4exp_build_qsa_top_blocks(ctx, rotated_query, block_keys, block_mask, selected_count);
    auto * tokens = qwen4exp_build_qsa_block_token_indices(ctx, selection.top_blocks, input.block_map);
    auto * selected = qwen4exp_build_qsa_sparse_mask(ctx, input.causal_mask, tokens);
    auto * tail = qwen4exp_build_qsa_sparse_mask(ctx, input.causal_mask, input.tail);
    selected = ggml_view_2d(ctx, selected, keys, queries, selected->nb[1], 0);
    tail = ggml_view_2d(ctx, tail, keys, queries, tail->nb[1], 0);
    auto * mask = qwen4exp_build_qsa_mask_union(ctx, selected, tail, input.tail_enabled);
    // These intermediates are inspected after the complete graph finishes.
    // Keep their storage alive rather than reading a later allocator alias.
    ggml_set_output(selection.scores); ggml_set_output(selection.top_blocks); ggml_set_output(mask);
    auto * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, selection.scores); ggml_build_forward_expand(graph, selection.top_blocks);
    ggml_build_forward_expand(graph, mask);
    ggml_backend_t backends[] = {compute, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, compute == cpu ? 1 : 2, 256, false);
    require(sched, "scheduler");
    int integer_fallbacks = 0;
    for (int i = 0; i < graph->n_nodes; ++i) {
        auto * node = graph->nodes[i];
        if (ggml_is_noop(node)) continue;
        if (ggml_backend_supports_op(compute, node)) ggml_backend_sched_set_tensor_backend(sched, node, compute);
        else {
            if (node->op != GGML_OP_REPEAT || node->type != GGML_TYPE_I32)
                std::fprintf(stderr, "unsupported arithmetic op=%s type=%s ne=[%lld,%lld,%lld,%lld] params0=%d\n",
                        ggml_op_name(node->op), ggml_type_name(node->type), (long long) node->ne[0], (long long) node->ne[1],
                        (long long) node->ne[2], (long long) node->ne[3], node->op_params[0]);
            require(node->op == GGML_OP_REPEAT && node->type == GGML_TYPE_I32, "unexpected numerical CPU fallback");
            ggml_backend_sched_set_tensor_backend(sched, node, cpu); ++integer_fallbacks;
        }
    }
    require(ggml_backend_sched_alloc_graph(sched, graph), "graph allocation");
    require(ggml_backend_sched_get_tensor_backend(sched, selection.scores) == compute &&
            ggml_backend_sched_get_tensor_backend(sched, pooled) == compute &&
            ggml_backend_sched_get_tensor_backend(sched, mask) == compute, "arithmetic backend assignment");

    std::vector<ggml_fp16_t> logical(dims * owned), storage(dims * physical);
    std::vector<float> norm(dims), qvalues(dims * heads * queries);
    for (int r = 0; r < owned; ++r) for (int d = 0; d < dims; ++d) {
        const float x = d < rotary ? .35f * std::sin(r * .137f + d * .23f) :
                .9f + .13f * std::sin(r * .071f + d * .17f);
        logical[r * dims + d] = ggml_fp32_to_fp16(x);
    }
    for (int d = 0; d < dims; ++d) norm[d] = 1.f + .05f * std::sin(d * .11f);
    for (int q = 0; q < queries; ++q) for (int h = 0; h < heads; ++h) for (int d = 0; d < dims; ++d)
        qvalues[(q * heads + h) * dims + d] = d < rotary ?
                .35f * std::cos(d * .27f + q * .13f + h * .51f) : .9f + .12f * std::cos(d * .19f + q * .17f + h * .53f);
    ggml_backend_tensor_set(gamma, norm.data(), 0, norm.size() * sizeof(float));
    ggml_backend_tensor_set(query, qvalues.data(), 0, qvalues.size() * sizeof(float));
    ggml_backend_tensor_set(query_pos, positions.data(), 0, positions.size() * sizeof(int32_t));

    std::vector<float> first_scores, first_mask;
    std::vector<int32_t> first_top;
    double maximum_error = 0;
    for (int layout = 0; layout < 2; ++layout) {
        populate_cells(layout != 0);
        llama_qwen4exp_set_attention_inputs(lctx, batch);
        std::fill(storage.begin(), storage.end(), ggml_fp32_to_fp16(200.f));
        for (int r = 0; r < owned; ++r) {
            const int slot = layout ? (r * 73 + 91) % physical : r;
            std::copy_n(logical.data() + r * dims, dims, storage.data() + slot * dims);
        }
        ggml_backend_tensor_set(raw, storage.data(), 0, storage.size() * sizeof(ggml_fp16_t));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "QSA compute");
        std::vector<float> scores(blocks * queries), actual_mask(keys * queries);
        std::vector<int32_t> top(selected_count * queries);
        ggml_backend_tensor_get(selection.scores, scores.data(), 0, scores.size() * sizeof(float));
        ggml_backend_tensor_get(selection.top_blocks, top.data(), 0, top.size() * sizeof(int32_t));
        ggml_backend_tensor_get(mask, actual_mask.data(), 0, actual_mask.size() * sizeof(float));
        for (int q = 0; q < queries; ++q) {
            const int visible = owned - queries + q + 1;
            std::vector<std::vector<double>> qrows(heads, std::vector<double>(dims));
            for (int h = 0; h < heads; ++h) {
                double squares = 0;
                for (int d = 0; d < dims; ++d) { const auto x = qvalues[(q * heads + h) * dims + d]; squares += double(x) * x; }
                for (int d = 0; d < dims; ++d) qrows[h][d] = qvalues[(q * heads + h) * dims + d] / std::sqrt(squares / dims + eps);
                rotate(qrows[h], rotary, positions[q]);
            }
            std::vector<double> expected(blocks, -INFINITY);
            for (int b = 0; b < visible / ratio; ++b) {
                std::vector<double> key(dims, 0); double squares = 0;
                for (int d = 0; d < dims; ++d) {
                    for (int r = 0; r < ratio; ++r) key[d] += ggml_fp16_to_fp32(logical[(b * ratio + r) * dims + d]);
                    key[d] /= ratio; squares += key[d] * key[d];
                }
                for (int d = 0; d < dims; ++d) key[d] = key[d] / std::sqrt(squares / dims + eps) * norm[d];
                rotate(key, rotary, position(b * ratio, shift));
                double score = 0;
                for (int h = 0; h < heads; ++h) {
                    double dot = 0; for (int d = 0; d < dims; ++d) dot += qrows[h][d] * key[d];
                    score += std::max(0., dot);
                }
                expected[b] = score / std::sqrt(double(dims));
            }
            for (int b = 0; b < blocks; ++b) {
                const auto got = scores[q * blocks + b];
                if (std::isfinite(expected[b])) {
                    const double error = std::fabs(got - expected[b]); maximum_error = std::max(maximum_error, error);
                    if (!std::isfinite(got) || error > 2.e-4 * (1 + std::fabs(expected[b])))
                        std::fprintf(stderr, "score mismatch dims=%d owned=%d queries=%d shift=%d layout=%d q=%d block=%d got=%.9g expected=%.12g\n",
                                dims, owned, queries, shift, layout, q, b, got, expected[b]);
                    require(std::isfinite(got) && error <= 2.e-4 * (1 + std::fabs(expected[b])), "QSA scores disagree with F64 oracle");
                } else require(std::isinf(got) && got < 0, "incomplete/future block score must be negative infinity");
            }
            std::vector<int> ranked(blocks); std::iota(ranked.begin(), ranked.end(), 0);
            std::stable_sort(ranked.begin(), ranked.end(), [&](int a, int b) { return expected[a] > expected[b]; });
            require(expected[ranked[1]] - expected[ranked[2]] > 1.e-7, "oracle fixture has tied selection boundary");
            for (int rank = 0; rank < selected_count; ++rank)
                require(top[q * selected_count + rank] == ranked[rank], "QSA selected block differs from independent ranking");
            for (int r = 0; r < keys; ++r) {
                const bool allowed = r < visible &&
                        (r >= visible - visible % ratio || r / ratio == ranked[0] || r / ratio == ranked[1]);
                const float got = actual_mask[q * keys + r];
                require(allowed ? got == 0 : std::isinf(got) && got < 0, "QSA sparse/tail mask disagrees with independent token set");
            }
        }
        if (!layout) { first_scores = scores; first_top = top; first_mask = actual_mask; }
        else require(first_top == top && std::memcmp(first_scores.data(), scores.data(), scores.size() * sizeof(float)) == 0 &&
                std::memcmp(first_mask.data(), actual_mask.data(), actual_mask.size() * sizeof(float)) == 0,
                "QSA depends on physical placement or unrelated sequence");
    }
    std::printf("qsa_selection: dims=%d heads=%d owned=%d padded=%d queries=%d shift=%d top=2 physical_exact=true oracle_max=%.9g integer_fallbacks=%d\n",
            dims, heads, owned, keys, queries, shift, maximum_error, integer_fallbacks);
    ggml_backend_sched_free(sched); ggml_backend_buffer_free(leaf_buffer); ggml_free(ctx); ggml_free(in);
    lctx.qwen4exp_attention_inputs.clear();
}

int main(int argc, char ** argv) {
    try {
        auto * cpu = ggml_backend_cpu_init(); require(cpu, "CPU backend");
        ggml_backend_t compute = cpu;
#ifdef GGML_USE_CUDA
        CUdevice dev = 0; bool retained = false;
        if (argc > 1 && std::strcmp(argv[1], "cpu") != 0) {
            const int device = std::atoi(argv[1]); CUcontext primary;
            require(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&dev, device) == CUDA_SUCCESS &&
                    cuDevicePrimaryCtxRetain(&primary, dev) == CUDA_SUCCESS && cuCtxSetCurrent(primary) == CUDA_SUCCESS, "CUDA primary");
            retained = true; compute = ggml_backend_cuda_init(device, nullptr, nullptr); require(compute, "CUDA backend");
        }
#else
        (void) argc; (void) argv;
#endif
        check_log(compute, cpu);
        int cases = 0;
        for (int dims : {8, 128}) for (int heads : {2, 4}) for (int owned : {31, 32, 33, 255, 256, 257, 511, 512, 513})
            for (int queries : {1, 4}) for (int shift : {0, 11}) {
                run(compute, cpu, dims, heads, owned, queries, shift); ++cases;
            }
        std::printf("QSA selection-limited pipeline: %d cases x two physical layouts pass\n", cases);
        if (compute != cpu) ggml_backend_free(compute);
        ggml_backend_free(cpu);
#ifdef GGML_USE_CUDA
        if (retained) require(cuDevicePrimaryCtxRelease(dev) == CUDA_SUCCESS, "primary release");
#endif
        return 0;
    } catch (const std::exception & error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
