#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"
#include "../llama-qwen4exp-ops.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

ggml_cgraph * llm_build_context::build_qwen4exp() {
    ggml_cgraph * gf = new_graph_custom();
    delta_net delta(lctx, batch);

    const int64_t hidden_size = hparams.n_embd;
    const int64_t stream_count = hparams.qwen4exp_hc_count;
    const int64_t stream_width = hidden_size * stream_count;

    ggml_tensor * token_embeddings = llm_build_inp_embd(
            ctx0, lctx, hparams, batch, model.tok_embd, cb);
    cb(token_embeddings, "qwen4exp_token_embeddings", -1);
    ggml_tensor * hidden = qwen4exp_build_initial_streams(
            ctx0, token_embeddings, stream_count);

    ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * KQ_mask = nullptr;
    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    bool has_ple = false;
    for (int il = 0; il < n_layer; ++il) has_ple |= hparams.qwen4exp_ple_layers[il];
    if (has_ple) {
        lctx.inp_qwen4exp_ple = ggml_new_tensor_2d(
                ctx0, GGML_TYPE_F32, hidden_size, n_tokens);
        cb(lctx.inp_qwen4exp_ple, "inp_qwen4exp_ple", -1);
        ggml_set_input(lctx.inp_qwen4exp_ple);
    }
    int64_t qsa_ratio = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (!hparams.is_recurrent(il)) {
            qsa_ratio = hparams.qwen4exp_compress_ratios[il];
            break;
        }
    }
    if (qsa_ratio > 0) {
        lctx.qwen4exp_attention_inputs = llama_qwen4exp_attention_layout(
                lctx, batch, n_kv > int32_t(kv_self.n) ? n_kv : 0);
        int64_t cache_heads = 1;
        for (int il = 0; il < n_layer; ++il)
            if (!hparams.is_recurrent(il)) cache_heads = std::max(cache_heads, int64_t(hparams.n_head_kv(il)));
        for (auto & input : lctx.qwen4exp_attention_inputs) {
            const int64_t keys = input.key_count;
            const int64_t queries = input.query_count;
            const int64_t blocks = keys / qsa_ratio;
            GGML_ASSERT(blocks > 0);
            input.rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, keys, cache_heads);
            input.causal_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, keys, GGML_PAD(queries, GGML_KQ_MASK_PAD));
            input.block_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, blocks);
            input.block_end = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, blocks);
            input.block_map = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, qsa_ratio, blocks);
            input.tail = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, qsa_ratio - 1, queries);
            input.tail_enabled = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, queries);
            input.layer_masks.resize(n_layer, nullptr);
            for (auto * tensor : {input.rows, input.causal_mask, input.block_pos, input.block_end,
                    input.block_map, input.tail, input.tail_enabled}) ggml_set_input(tensor);
        }
    }

    const llama_seq_id prefix_seq = batch.n_seq_id && batch.seq_id ? batch.seq_id[0][0] : batch.all_seq_id;
    const bool save_prefixes = kv_self.ckpt.qwen4exp_prefix_offset(prefix_seq) >= 0;
    if (save_prefixes && has_ple) {
        const int64_t length = (hparams.qwen4exp_ple_conv_kernel - 1) * hparams.qwen4exp_ple_ngram_size;
        lctx.inp_qwen4exp_prefix_columns = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, length * n_tokens);
        ggml_set_input(lctx.inp_qwen4exp_prefix_columns);
        cb(lctx.inp_qwen4exp_prefix_columns, "inp_qwen4exp_prefix_columns", -1);
    }

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * ple_prefix_states = nullptr;

        if (hparams.qwen4exp_ple_layers[il]) {
            // The production scheduler will split mixed-sequence batches before
            // this graph. A recurrent row and its rolling n-gram history must
            // never be shared by unrelated sequences.
            GGML_ASSERT(batch.n_tokens > 0);
            const bool has_sequence_metadata = batch.n_seq_id && batch.seq_id;
            const llama_seq_id seq_id = has_sequence_metadata ? batch.seq_id[0][0] : 0;
            for (int64_t i = 1; has_sequence_metadata && i < batch.n_tokens; ++i) {
                GGML_ASSERT(batch.n_seq_id[i] > 0 && batch.seq_id[i][0] == seq_id);
            }
            GGML_ASSERT(seq_id >= 0 && (uint64_t) seq_id < kv_self.s_l[il]->ne[1]);

            const int64_t state_length =
                (hparams.qwen4exp_ple_conv_kernel - 1) * hparams.qwen4exp_ple_ngram_size;
            const int64_t base_state = hparams.n_embd_v_s();
            const bool reset_state = has_sequence_metadata && batch.pos && batch.pos[0] == 0;
            const auto state = qwen4exp_build_ple_state(ctx0, kv_self.s_l[il],
                    base_state, seq_id, stream_width, state_length, reset_state);
            qwen4exp_ple_gate_trace gate_trace;
            const auto gate = qwen4exp_build_ple_gate(ctx0, hidden, lctx.inp_qwen4exp_ple,
                    layer.qwen4exp_ple_key, layer.qwen4exp_ple_value,
                    layer.qwen4exp_ple_norm_key, layer.qwen4exp_ple_norm_query,
                    layer.qwen4exp_ple_norm_conv, hidden_size, stream_count,
                    hparams.f_norm_rms_eps, &gate_trace);
            const char * explicit_env = std::getenv("ESE_QWEN4EXP_EXPERT_F32");
            if (explicit_env && std::strcmp(explicit_env, "1") == 0) {
                // Preserve the F32 embedding when projecting BF16 weights.
                // Rounding it/output to BF16 is amplified by key RMSNorm.
                ggml_mul_mat_set_prec(gate_trace.key_projected, GGML_PREC_F32);
                ggml_mul_mat_set_prec(gate_trace.value_projected, GGML_PREC_F32);
            }
            cb(gate_trace.key_projected, "qwen4exp_ple_key_projected", il);
            cb(gate_trace.key_normed, "qwen4exp_ple_key_normed", il);
            cb(gate_trace.query_normed, "qwen4exp_ple_query_normed", il);
            cb(gate_trace.gate_dot, "qwen4exp_ple_gate_dot", il);
            cb(gate_trace.gate_sigmoid, "qwen4exp_ple_gate_sigmoid", il);
            cb(gate_trace.value_projected, "qwen4exp_ple_value_projected", il);
            cb(gate.gated_value, "qwen4exp_ple_gated_value", il);
            cb(gate.gated_value_normed, "qwen4exp_ple_gated_value_normed", il);
            const auto conv = qwen4exp_build_ple_dilated_conv(ctx0,
                    gate.gated_value_normed, layer.qwen4exp_ple_conv,
                    state.prior_state, hparams.qwen4exp_ple_conv_kernel,
                    hparams.qwen4exp_ple_ngram_size,
                    save_prefixes ? &ple_prefix_states : nullptr,
                    save_prefixes ? lctx.inp_qwen4exp_prefix_columns : nullptr);
            cb(conv.output, "qwen4exp_ple_conv_output", il);
            hidden = ggml_add(ctx0, hidden,
                    qwen4exp_build_ple_output(ctx0, gate.gated_value, conv.output));
            cb(hidden, "qwen4exp_ple_residual", il);
            ggml_build_forward_expand(gf,
                    qwen4exp_build_ple_state_write(ctx0, conv.next_state, state.destination));
        }

        const auto attn_gr = qwen4exp_build_gated_residual(ctx0, hidden,
                layer.qwen4exp_hc_attn_norm, layer.qwen4exp_hc_attn_down,
                layer.qwen4exp_hc_attn_up, layer.qwen4exp_hc_attn_inject,
                hidden_size, stream_count, hparams.f_norm_rms_eps);
        cb(attn_gr.mixed_input, "qwen4exp_attn_input", il);

        ggml_tensor * block_output = nullptr;
        if (hparams.is_recurrent(il)) {
            block_output = delta.build_qwen4exp_attn_linear(
                    ctx0, gf, attn_gr.mixed_input, nullptr, il, cb, ple_prefix_states);
        } else {
            const int64_t ratio = hparams.qwen4exp_compress_ratios[il];
            GGML_ASSERT(ratio == qsa_ratio && ratio > 0);
            GGML_ASSERT((size_t) il < kv_self.kr_l.size() && kv_self.kr_l[il]);
            const char * arithmetic_policy = std::getenv("ESE_QWEN4EXP_EXPERT_F32");
            const auto indexer_precision = arithmetic_policy && std::strcmp(arithmetic_policy, "1") == 0
                ? GGML_PREC_F32 : GGML_PREC_DEFAULT;
            ggml_tensor * raw_key = llm_build_lora_mm(
                    lctx, ctx0, layer.qwen4exp_indexer_k_proj, attn_gr.mixed_input, indexer_precision);
            cb(raw_key, "qwen4exp_indexer_key", il);
            const auto cached = qwen4exp_build_qsa_cached_keys(
                    ctx0, raw_key, kv_self.kr_l[il], kv_head, n_kv);
            lctx.dsa_cache_copies[il].cpy = cached.write;
            lctx.dsa_cache_copies[il].step = kv_self.kr_l[il]->nb[1];
            ggml_build_forward_expand(gf, cached.write);

            for (auto & input : lctx.qwen4exp_attention_inputs) {
                const int64_t sequence_keys = input.key_count;
                const int64_t sequence_queries = input.query_count;
                ggml_tensor * KQ_mask = input.causal_mask;
                ggml_tensor * sparse_mask = KQ_mask;
                auto * ids = ggml_view_2d(ctx0, input.rows, sequence_keys, 1, input.rows->nb[1], 0);
                auto * ordered_keys = qwen4exp_build_cache_rows(ctx0, cached.cached_keys, ids);
                if (ggml_backend_buffer_is_host(kv_self.kr_l[il]->buffer))
                    ggml_backend_sched_set_tensor_backend(lctx.sched, ordered_keys, lctx.backend_cpu);
                const int64_t complete_blocks = sequence_keys / ratio;
                if (complete_blocks > 0) {
                    GGML_ASSERT(input.block_pos &&
                            input.block_end && input.block_map);
                    ggml_tensor * block_pos = input.block_pos;
                    ggml_tensor * block_keys = qwen4exp_build_qsa_pooled_keys(ctx0,
                            ordered_keys, layer.qwen4exp_indexer_k_norm,
                            ratio, complete_blocks, hparams.f_norm_rms_eps);
                    block_keys = ggml_reshape_3d(ctx0, block_keys,
                            hparams.indexer_head_size, 1, complete_blocks);
                    block_keys = qwen4exp_build_qsa_rope(ctx0, block_keys, block_pos, n_rot,
                            n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);
                    block_keys = ggml_reshape_2d(ctx0, block_keys,
                            hparams.indexer_head_size, complete_blocks);

                    ggml_tensor * index_query = llm_build_lora_mm(
                            lctx, ctx0, layer.qwen4exp_indexer_q_proj, ggml_view_2d(ctx0, attn_gr.mixed_input,
                                attn_gr.mixed_input->ne[0], sequence_queries, attn_gr.mixed_input->nb[1],
                                input.first_query * attn_gr.mixed_input->nb[1]), indexer_precision);
                    cb(index_query, "qwen4exp_indexer_query", il);
                    index_query = ggml_reshape_3d(ctx0, index_query,
                            hparams.indexer_head_size, hparams.indexer_n_head, sequence_queries);
                    index_query = llm_build_norm(ctx0, index_query, hparams,
                            layer.qwen4exp_indexer_q_norm, nullptr, LLM_NORM_RMS, cb, il);
                    ggml_tensor * text_pos = ggml_view_1d(ctx0, inp_pos, sequence_queries, input.first_query * inp_pos->nb[0]);
                    index_query = qwen4exp_build_qsa_rope(ctx0, index_query, text_pos, n_rot,
                            n_ctx_orig, freq_base, freq_scale,
                            ext_factor, attn_factor, beta_fast, beta_slow);

                    ggml_tensor * block_ends = input.block_end;
                    ggml_tensor * mask_transposed = ggml_cont(ctx0, ggml_transpose(ctx0, KQ_mask));
                    ggml_tensor * block_mask = ggml_get_rows(ctx0, mask_transposed, block_ends);
                    block_mask = ggml_cont(ctx0, ggml_transpose(ctx0, block_mask));
                    block_mask = ggml_view_2d(ctx0, block_mask, complete_blocks, sequence_queries,
                            block_mask->nb[1], 0);
                    if (block_mask->type != GGML_TYPE_F32) {
                        block_mask = ggml_cast(ctx0, block_mask, GGML_TYPE_F32);
                    }

                    const int64_t selected_blocks = std::min<int64_t>(
                            hparams.indexer_top_k / ratio, complete_blocks);
                    const auto selection = qwen4exp_build_qsa_top_blocks(ctx0,
                            index_query, block_keys, block_mask, selected_blocks);
                    ggml_tensor * selected_tokens = qwen4exp_build_qsa_block_token_indices(
                            ctx0, selection.top_blocks, input.block_map);
                    ggml_tensor * selected_mask = qwen4exp_build_qsa_sparse_mask(
                            ctx0, KQ_mask, selected_tokens);
                    ggml_tensor * tail_mask = qwen4exp_build_qsa_sparse_mask(
                            ctx0, KQ_mask, input.tail);
                    selected_mask = ggml_view_2d(ctx0, selected_mask,
                            sequence_keys, sequence_queries, selected_mask->nb[1], 0);
                    tail_mask = ggml_view_2d(ctx0, tail_mask,
                            sequence_keys, sequence_queries, tail_mask->nb[1], 0);
                    sparse_mask = qwen4exp_build_qsa_mask_union(ctx0,
                            selected_mask, tail_mask, input.tail_enabled);
                    if (flash_attn) {
                        sparse_mask = build_deepseek2_dsa_fa_mask(sparse_mask,
                                ggml_cast(ctx0, KQ_mask, GGML_TYPE_F16));
                    }
                    cb(selection.top_blocks, "qwen4exp_qsa_top_blocks", il);
                    cb(sparse_mask, "qwen4exp_qsa_sparse_mask", il);
                }

                if (flash_attn && sparse_mask->type != GGML_TYPE_F16) sparse_mask = ggml_cast(ctx0, sparse_mask, GGML_TYPE_F16);
                input.layer_masks[il] = sparse_mask;
            }

            const float kq_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(hparams.n_embd_head_k(il)))
                : hparams.f_attention_scale;
            ggml_tensor * text_pos = ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
            block_output = build_std_attention(gf, nullptr, attn_gr.mixed_input,
                    text_pos, nullptr, nullptr, KQ_mask, nullptr, nullptr,
                    kq_scale, 0.0f, 0, il, true, false, false, false, false);
        }
        cb(block_output, "qwen4exp_attn_output", il);
        hidden = qwen4exp_build_stream_injection(ctx0, attn_gr.hyper_input,
                block_output, attn_gr.injection_weights, hidden_size, stream_count);
        cb(hidden, "qwen4exp_attn_residual", il);

        const auto ffn_gr = qwen4exp_build_gated_residual(ctx0, hidden,
                layer.qwen4exp_hc_ffn_norm, layer.qwen4exp_hc_ffn_down,
                layer.qwen4exp_hc_ffn_up, layer.qwen4exp_hc_ffn_inject,
                hidden_size, stream_count, hparams.f_norm_rms_eps);
        cb(ffn_gr.mixed_input, "qwen4exp_ffn_input", il);
        block_output = llm_build_std_moe_ffn(ctx0, lctx, nullptr, ffn_gr.mixed_input,
                layer.ffn_gate_inp, nullptr,
                layer.ffn_up_exps, nullptr, layer.ffn_gate_exps, nullptr,
                layer.ffn_down_exps, nullptr, nullptr,
                layer.ffn_up_shexp, nullptr, layer.ffn_gate_shexp, nullptr,
                layer.ffn_down_shexp, nullptr, n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX, LLM_FFN_SILU,
                cb, il, gf, false, nullptr, nullptr, layer.ffn_gate_inp_shexp);
        cb(block_output, "qwen4exp_ffn_output", il);
        hidden = qwen4exp_build_stream_injection(ctx0, ffn_gr.hyper_input,
                block_output, ffn_gr.injection_weights, hidden_size, stream_count);
        cb(hidden, "qwen4exp_layer_output", il);
    }

    const auto output_gr = qwen4exp_build_gated_residual(ctx0, hidden,
            model.qwen4exp_hc_output_norm, model.qwen4exp_hc_output_down,
            model.qwen4exp_hc_output_up, nullptr,
            hidden_size, stream_count, hparams.f_norm_rms_eps);
    ggml_tensor * output = output_gr.mixed_input;
    cb(output, "qwen4exp_output_input", -1);
    if (inp_out_ids) output = ggml_get_rows(ctx0, output, inp_out_ids);
    output = build_output(lctx, ctx0, output, model.output, nullptr, cb);
    cb(output, "result_output", -1);
    ggml_build_forward_expand(gf, output);
    return gf;
}
