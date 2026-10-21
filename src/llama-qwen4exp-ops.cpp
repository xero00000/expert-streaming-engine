#include "llama-qwen4exp-ops.h"

#include <cmath>

bool qwen4exp_cache_type_supported(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: case GGML_TYPE_F32: case GGML_TYPE_BF16:
        case GGML_TYPE_Q4_0: case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1: case GGML_TYPE_Q8_0:
            return true;
        default: return false;
    }
}

bool qwen4exp_validate_cache_type(ggml_type type, int64_t head_width,
        bool value_cache, bool flash_attn, std::string & error) {
    error.clear();
    if (!qwen4exp_cache_type_supported(type)) {
        error = "unsupported Qwen4Exp cache type";
        return false;
    }
    if (head_width <= 0 || head_width % ggml_blck_size(type) != 0) {
        error = "Qwen4Exp cache head width must be positive and block-aligned";
        return false;
    }
    if (value_cache && ggml_is_quantized(type) && !flash_attn) {
        error = "quantized Qwen4Exp V cache requires Flash Attention";
        return false;
    }
    return true;
}

ggml_tensor * qwen4exp_build_cache_rows(
        ggml_context * ctx, ggml_tensor * cache, ggml_tensor * row_ids) {
    GGML_ASSERT(ctx && cache && row_ids && cache->ne[3] == 1);
    GGML_ASSERT(qwen4exp_cache_type_supported(cache->type));
    GGML_ASSERT(cache->nb[0] == ggml_type_size(cache->type));
    GGML_ASSERT(row_ids->type == GGML_TYPE_I32 && row_ids->ne[1] == cache->ne[2]);
    GGML_ASSERT(row_ids->ne[2] == 1 && row_ids->ne[3] == 1);
    GGML_ASSERT(row_ids->nb[0] == sizeof(int32_t));
    // GET_ROWS accepts independent row/head strides. Its same-type path avoids a conversion and
    // an additional full-size F32 intermediate for the usual F16 KV cache.
    // BF16 and packed rows use the native F32 gather. Only selected rows
    // become F32; the persistent cache remains in its original representation.
    const bool same_type = cache->type == GGML_TYPE_F16 || cache->type == GGML_TYPE_F32;
    return ggml_get_rows_ext(ctx, cache, row_ids, same_type, false);
}

ggml_tensor * qwen4exp_build_compact_attention_indices(
        ggml_context * ctx, ggml_tensor * sparse_mask, int64_t capacity) {
    GGML_ASSERT(ctx && sparse_mask && sparse_mask->type == GGML_TYPE_F32);
    GGML_ASSERT(capacity > 0 && capacity <= sparse_mask->ne[0]);
    GGML_ASSERT(sparse_mask->ne[0] <= (1 << 24));
    // Masks contain only zero and -infinity. Distinct negative integer ranks
    // break visible-row ties without changing the mask used by attention.
    // This preserves key order even on backends with an unstable argsort.
    auto * rank = ggml_arange(ctx, 0, float(sparse_mask->ne[0]), 1);
    auto * scores = ggml_add(ctx, sparse_mask, ggml_scale(ctx, rank, -1));
    return ggml_cont(ctx, ggml_top_k(ctx, scores, capacity));
}

qwen4exp_compact_attention_rows qwen4exp_build_compact_attention_rows(
        ggml_context * ctx, ggml_tensor * physical_rows, ggml_tensor * sparse_mask,
        ggml_tensor * selected_rows, int64_t query) {
    GGML_ASSERT(ctx && physical_rows && sparse_mask && selected_rows);
    GGML_ASSERT(physical_rows->type == GGML_TYPE_I32 && selected_rows->type == GGML_TYPE_I32);
    GGML_ASSERT(sparse_mask->type == GGML_TYPE_F32);
    GGML_ASSERT(physical_rows->ne[0] == sparse_mask->ne[0]);
    GGML_ASSERT(query >= 0 && query < selected_rows->ne[1] && query < sparse_mask->ne[1]);
    const int64_t kept = selected_rows->ne[0];
    auto * logical = ggml_view_2d(ctx, selected_rows, kept, 1,
            selected_rows->nb[1], query * selected_rows->nb[1]);
    auto * per_head = ggml_repeat_4d(ctx, logical, kept, physical_rows->ne[1], 1, 1);
    auto * rows = ggml_get_rows_ext(ctx, physical_rows, per_head, true, true);
    auto * query_mask = ggml_view_2d(ctx, sparse_mask, sparse_mask->ne[0], 1,
            sparse_mask->nb[1], query * sparse_mask->nb[1]);
    auto * mask_rows = ggml_cont(ctx, ggml_transpose(ctx, query_mask));
    auto * mask = ggml_reshape_2d(ctx, ggml_get_rows(ctx, mask_rows, logical), kept, 1);
    mask = ggml_pad(ctx, mask, 0, GGML_KQ_MASK_PAD - 1, 0, 0);
    mask = ggml_cast(ctx, mask, GGML_TYPE_F16);
    return {rows, mask};
}

ggml_tensor * qwen4exp_build_recurrent_output_gate(
        ggml_context * ctx, ggml_tensor * normalized, ggml_tensor * gate) {
    GGML_ASSERT(ctx && normalized && gate && ggml_are_same_shape(normalized, gate));
    return ggml_mul(ctx, normalized, ggml_sigmoid(ctx, gate));
}

ggml_tensor * qwen4exp_build_qsa_rope(ggml_context * ctx, ggml_tensor * input,
        ggml_tensor * positions, int rotary_dims, int original_context,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    GGML_ASSERT(ctx && input && positions);
    GGML_ASSERT(rotary_dims > 0 && rotary_dims % 2 == 0 && rotary_dims <= input->ne[0]);
    return ggml_rope_ext(ctx, input, positions, nullptr, rotary_dims, GGML_ROPE_TYPE_NEOX,
            original_context, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
}

ggml_tensor * qwen4exp_build_l2_norm(ggml_context * ctx, ggml_tensor * input, float eps) {
    GGML_ASSERT(ctx && input && eps >= 0.0f);
    // RMSNorm divides the squared sum by row width. Rescale both epsilon and
    // output to recover the pinned sum-based L2 formula using native kernels.
    const float width = float(input->ne[0]);
    return ggml_scale(ctx, ggml_rms_norm(ctx, input, eps / width), 1.0f / std::sqrt(width));
}

ggml_tensor * qwen4exp_build_initial_streams(
        ggml_context * ctx,
        ggml_tensor  * token_embeddings,
        int64_t        stream_count) {
    GGML_ASSERT(ctx && token_embeddings && stream_count > 0);
    return ggml_repeat_4d(ctx, token_embeddings,
            token_embeddings->ne[0] * stream_count,
            token_embeddings->ne[1], token_embeddings->ne[2], token_embeddings->ne[3]);
}

static ggml_tensor * qwen4exp_grouped_rms_norm(
        ggml_context * ctx,
        ggml_tensor  * input,
        ggml_tensor  * weight,
        int64_t        hidden_size,
        int64_t        stream_count,
        float          rms_eps) {
    const int64_t n_tokens = input->ne[1];
    GGML_ASSERT(input->ne[0] == hidden_size * stream_count);
    GGML_ASSERT(ggml_nelements(weight) == hidden_size * stream_count);
    ggml_tensor * grouped = ggml_reshape_3d(ctx, input, hidden_size, stream_count, n_tokens);
    grouped = ggml_rms_norm(ctx, grouped, rms_eps);
    ggml_tensor * flattened = ggml_reshape_2d(
            ctx, grouped, hidden_size * stream_count, n_tokens);
    return ggml_mul(ctx, flattened, weight);
}

qwen4exp_gated_residual_result qwen4exp_build_gated_residual(
        ggml_context * ctx,
        ggml_tensor  * hyper_input,
        ggml_tensor  * norm_weight,
        ggml_tensor  * mix_down,
        ggml_tensor  * mix_up,
        ggml_tensor  * inject,
        int64_t        hidden_size,
        int64_t        stream_count,
        float          rms_eps) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(hyper_input != nullptr);
    GGML_ASSERT(norm_weight != nullptr);
    GGML_ASSERT(mix_down != nullptr);
    GGML_ASSERT(mix_up != nullptr);
    GGML_ASSERT(hidden_size > 0);
    GGML_ASSERT(stream_count > 0);

    const int64_t hyper_size = hidden_size * stream_count;
    const int64_t n_tokens = hyper_input->ne[1];
    GGML_ASSERT(hyper_input->ne[0] == hyper_size);
    GGML_ASSERT(norm_weight->ne[0] == hyper_size && ggml_nelements(norm_weight) == hyper_size);
    GGML_ASSERT(mix_down->ne[0] == hyper_size);
    GGML_ASSERT(mix_up->ne[0] == mix_down->ne[1] && mix_up->ne[1] == hyper_size);
    GGML_ASSERT(inject == nullptr || (inject->ne[0] == hyper_size && inject->ne[1] == stream_count));

    // Transformers (fc5c5bde) evaluates (1 + weight), but Qwen4Exp GGUFs
    // store that effective gamma directly (the production tensors are
    // centered near 1, like ordinary llama.cpp RMSNorm weights).
    ggml_tensor * normed = qwen4exp_grouped_rms_norm(
            ctx, hyper_input, norm_weight, hidden_size, stream_count, rms_eps);
    ggml_tensor * normed_3d = ggml_reshape_3d(ctx, normed, hidden_size, stream_count, n_tokens);

    // silu(down(normed) / stream_count) -> sigmoid(up(...)).
    ggml_tensor * mix = ggml_mul_mat(ctx, mix_down, normed);
    ggml_mul_mat_set_prec(mix, GGML_PREC_F32);
    mix = ggml_silu(ctx, ggml_scale(ctx, mix, 1.0f / float(stream_count)));
    mix = ggml_mul_mat(ctx, mix_up, mix);
    ggml_mul_mat_set_prec(mix, GGML_PREC_F32);
    mix = ggml_sigmoid(ctx, mix);
    mix = ggml_reshape_3d(ctx, mix, hidden_size, stream_count, n_tokens);

    ggml_tensor * weighted = ggml_mul(ctx, mix, normed_3d);
    ggml_tensor * mixed_input = ggml_sum_rows_ext(ctx, weighted, 1);
    mixed_input = ggml_scale(ctx, ggml_reshape_2d(ctx, mixed_input, hidden_size, n_tokens),
            1.0f / float(stream_count));

    ggml_tensor * injection_weights = nullptr;
    if (inject != nullptr) {
        injection_weights = ggml_mul_mat(ctx, inject, normed);
        ggml_mul_mat_set_prec(injection_weights, GGML_PREC_F32);
        injection_weights = ggml_scale(ctx, injection_weights, 1.0f / float(stream_count));
        injection_weights = ggml_scale(ctx, ggml_sigmoid(ctx, injection_weights), 2.0f);
    }

    return { mixed_input, hyper_input, injection_weights };
}

ggml_tensor * qwen4exp_build_stream_injection(
        ggml_context * ctx,
        ggml_tensor  * hyper_input,
        ggml_tensor  * block_output,
        ggml_tensor  * injection_weights,
        int64_t        hidden_size,
        int64_t        stream_count) {
    GGML_ASSERT(ctx != nullptr);
    GGML_ASSERT(hyper_input != nullptr);
    GGML_ASSERT(block_output != nullptr);
    GGML_ASSERT(injection_weights != nullptr);
    const int64_t n_tokens = hyper_input->ne[1];
    GGML_ASSERT(hyper_input->ne[0] == hidden_size * stream_count);
    GGML_ASSERT(block_output->ne[0] == hidden_size && block_output->ne[1] == n_tokens);
    GGML_ASSERT(injection_weights->ne[0] == stream_count && injection_weights->ne[1] == n_tokens);

    ggml_tensor * output_3d = ggml_reshape_3d(ctx, block_output, hidden_size, 1, n_tokens);
    ggml_tensor * weights_3d = ggml_reshape_3d(ctx, injection_weights, 1, stream_count, n_tokens);
    output_3d = ggml_repeat_4d(ctx, output_3d, hidden_size, stream_count, n_tokens, 1);
    weights_3d = ggml_repeat_4d(ctx, weights_3d, hidden_size, stream_count, n_tokens, 1);
    ggml_tensor * injected = ggml_mul(ctx, output_3d, weights_3d);
    injected = ggml_reshape_2d(ctx, ggml_cont(ctx, injected), hidden_size * stream_count, n_tokens);
    return ggml_add(ctx, hyper_input, injected);
}

qwen4exp_ple_gate_result qwen4exp_build_ple_gate(
        ggml_context * ctx,
        ggml_tensor  * hidden_states,
        ggml_tensor  * embeddings,
        ggml_tensor  * key_projection,
        ggml_tensor  * value_projection,
        ggml_tensor  * key_norm_weight,
        ggml_tensor  * query_norm_weight,
        ggml_tensor  * conv_norm_weight,
        int64_t        hidden_size,
        int64_t        stream_count,
        float          rms_eps) {
    return qwen4exp_build_ple_gate(ctx, hidden_states, embeddings, key_projection,
            value_projection, key_norm_weight, query_norm_weight, conv_norm_weight,
            hidden_size, stream_count, rms_eps, nullptr);
}

qwen4exp_ple_gate_result qwen4exp_build_ple_gate(
        ggml_context * ctx,
        ggml_tensor  * hidden_states,
        ggml_tensor  * embeddings,
        ggml_tensor  * key_projection,
        ggml_tensor  * value_projection,
        ggml_tensor  * key_norm_weight,
        ggml_tensor  * query_norm_weight,
        ggml_tensor  * conv_norm_weight,
        int64_t        hidden_size,
        int64_t        stream_count,
        float          rms_eps,
        qwen4exp_ple_gate_trace * trace) {
    GGML_ASSERT(ctx && hidden_states && embeddings && key_projection && value_projection);
    GGML_ASSERT(key_norm_weight && query_norm_weight && conv_norm_weight);
    const int64_t hyper_size = hidden_size * stream_count;
    const int64_t n_tokens = hidden_states->ne[1];
    GGML_ASSERT(hidden_states->ne[0] == hyper_size);
    GGML_ASSERT(embeddings->ne[1] == n_tokens);
    GGML_ASSERT(key_projection->ne[0] == embeddings->ne[0] && key_projection->ne[1] == hyper_size);
    GGML_ASSERT(value_projection->ne[0] == embeddings->ne[0] && value_projection->ne[1] == hidden_size);

    ggml_tensor * key_projected = ggml_mul_mat(ctx, key_projection, embeddings);
    ggml_tensor * key = key_projected;
    key = qwen4exp_grouped_rms_norm(ctx, key, key_norm_weight,
            hidden_size, stream_count, rms_eps);
    ggml_tensor * query = qwen4exp_grouped_rms_norm(ctx, hidden_states, query_norm_weight,
            hidden_size, stream_count, rms_eps);
    key = ggml_reshape_3d(ctx, key, hidden_size, stream_count, n_tokens);
    query = ggml_reshape_3d(ctx, query, hidden_size, stream_count, n_tokens);

    ggml_tensor * gate = ggml_sum_rows(ctx, ggml_mul(ctx, key, query));
    gate = ggml_scale(ctx, gate, 1.0f / std::sqrt(float(hidden_size)));
    ggml_tensor * gate_dot = gate;
    ggml_tensor * gate_sign = ggml_sgn(ctx, gate);
    ggml_tensor * gate_magnitude = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, gate), 1.0e-6f, INFINITY));
    gate = ggml_sigmoid(ctx, ggml_mul(ctx, gate_magnitude, gate_sign));
    ggml_tensor * gate_sigmoid = gate;

    ggml_tensor * value_projected = ggml_mul_mat(ctx, value_projection, embeddings);
    ggml_tensor * value = value_projected;
    value = ggml_reshape_3d(ctx, value, hidden_size, 1, n_tokens);
    value = ggml_repeat_4d(ctx, value, hidden_size, stream_count, n_tokens, 1);
    gate = ggml_repeat_4d(ctx, gate, hidden_size, stream_count, n_tokens, 1);
    ggml_tensor * gated = ggml_mul(ctx, gate, value);
    gated = ggml_reshape_2d(ctx, ggml_cont(ctx, gated), hyper_size, n_tokens);
    ggml_tensor * normalized = qwen4exp_grouped_rms_norm(ctx, gated, conv_norm_weight,
            hidden_size, stream_count, rms_eps);
    if (trace) {
        *trace = { key_projected, key, query, gate_dot, gate_sigmoid, value_projected };
    }
    return { gated, normalized };
}

qwen4exp_ple_state_result qwen4exp_build_ple_state(
        ggml_context * ctx,
        ggml_tensor  * state_storage,
        int64_t        base_state_elements,
        int64_t        sequence_row,
        int64_t        channels,
        int64_t        state_length,
        bool           reset_state) {
    GGML_ASSERT(ctx && state_storage);
    GGML_ASSERT(state_storage->type == GGML_TYPE_F32);
    GGML_ASSERT(base_state_elements >= 0 && sequence_row >= 0);
    GGML_ASSERT(channels > 0 && state_length > 0);
    GGML_ASSERT(base_state_elements + channels * state_length <= state_storage->ne[0]);
    GGML_ASSERT(sequence_row < state_storage->ne[1]);

    const size_t offset = size_t(sequence_row) * state_storage->nb[1]
        + size_t(base_state_elements) * sizeof(float);
    ggml_tensor * destination = ggml_view_2d(ctx, state_storage, channels, state_length,
            size_t(channels) * sizeof(float), offset);
    ggml_tensor * prior = destination;
    if (reset_state) {
        prior = ggml_scale(ctx, prior, 0.0f);
    }
    return { prior, destination };
}

ggml_tensor * qwen4exp_build_ple_state_write(
        ggml_context * ctx,
        ggml_tensor  * next_state,
        ggml_tensor  * destination) {
    GGML_ASSERT(ctx && next_state && destination);
    GGML_ASSERT(ggml_are_same_shape(next_state, destination));
    return ggml_cpy(ctx, next_state, destination);
}

qwen4exp_ple_conv_result qwen4exp_build_ple_dilated_conv(
        ggml_context * ctx,
        ggml_tensor  * normalized_input,
        ggml_tensor  * conv_weight,
        ggml_tensor  * prior_state,
        int64_t        kernel_size,
        int64_t        dilation) {
    return qwen4exp_build_ple_dilated_conv(ctx, normalized_input, conv_weight,
            prior_state, kernel_size, dilation, nullptr, nullptr);
}

qwen4exp_ple_conv_result qwen4exp_build_ple_dilated_conv(
        ggml_context * ctx,
        ggml_tensor  * normalized_input,
        ggml_tensor  * conv_weight,
        ggml_tensor  * prior_state,
        int64_t        kernel_size,
        int64_t        dilation,
        ggml_tensor ** prefix_states,
        ggml_tensor  * prefix_columns) {
    GGML_ASSERT(ctx && normalized_input && conv_weight);
    GGML_ASSERT(kernel_size > 0 && dilation > 0);
    const int64_t channels = normalized_input->ne[0];
    const int64_t n_tokens = normalized_input->ne[1];
    const int64_t state_length = (kernel_size - 1) * dilation;
    GGML_ASSERT(conv_weight->ne[0] == kernel_size && conv_weight->ne[1] == channels);
    // GGUFs ship this depthwise weight as F32 or F16/BF16 (the Strata Coder
    // artifact stores it as F16); the elementwise tap MULs below require F32
    // operands, so promote once here. F32 weights keep the original path.
    if (conv_weight->type != GGML_TYPE_F32) {
        conv_weight = ggml_cast(ctx, conv_weight, GGML_TYPE_F32);
    }

    if (prior_state == nullptr) {
        GGML_ASSERT(n_tokens > 0);
        ggml_tensor * zero_column = ggml_view_2d(ctx, normalized_input, channels, 1,
                normalized_input->nb[1], 0);
        zero_column = ggml_scale(ctx, zero_column, 0.0f);
        prior_state = ggml_repeat_4d(ctx, zero_column, channels, state_length, 1, 1);
    }
    GGML_ASSERT(prior_state->ne[0] == channels && prior_state->ne[1] == state_length);

    ggml_tensor * history = ggml_concat(ctx, prior_state, normalized_input, 1);
    ggml_tensor * convolution = nullptr;
    for (int64_t tap = 0; tap < kernel_size; ++tap) {
        ggml_tensor * source = ggml_view_2d(ctx, history, channels, n_tokens,
                history->nb[1], size_t(tap * dilation) * history->nb[1]);
        ggml_tensor * weight = ggml_view_2d(ctx, conv_weight, 1, channels,
                conv_weight->nb[1], size_t(tap) * conv_weight->nb[0]);
        weight = ggml_reshape_2d(ctx, ggml_cont(ctx, weight), channels, 1);
        ggml_tensor * product = ggml_mul(ctx, source, weight);
        convolution = convolution == nullptr ? product : ggml_add(ctx, convolution, product);
    }
    convolution = ggml_silu(ctx, convolution);
    ggml_tensor * next_state = ggml_view_2d(ctx, history, channels, state_length,
            history->nb[1], size_t(n_tokens) * history->nb[1]);
    next_state = ggml_cont(ctx, next_state);
    if (prefix_states) {
        GGML_ASSERT(state_length > 0 && n_tokens > 0);
        // Use startup-initialized integer indices for all rolling windows. GGML
        // views validate their packed extent before accepting strides, so an
        // overlapping 3D view cannot represent this journal safely. One batched
        // gather copies state bytes without changing convolution arithmetic.
        GGML_ASSERT(prefix_columns && prefix_columns->type == GGML_TYPE_I32 &&
                ggml_is_contiguous(prefix_columns) && prefix_columns->ne[0] == state_length * n_tokens &&
                prefix_columns->ne[1] == 1 && prefix_columns->ne[2] == 1 && prefix_columns->ne[3] == 1);
        *prefix_states = ggml_reshape_3d(ctx, ggml_get_rows(ctx, history, prefix_columns),
                channels, state_length, n_tokens);
    }
    return { convolution, next_state };
}

ggml_tensor * qwen4exp_build_ple_output(
        ggml_context * ctx,
        ggml_tensor  * gated_value,
        ggml_tensor  * convolution_output) {
    GGML_ASSERT(ctx && gated_value && convolution_output);
    GGML_ASSERT(ggml_are_same_shape(gated_value, convolution_output));
    return ggml_add(ctx, gated_value, convolution_output);
}

qwen4exp_qsa_cache_result qwen4exp_build_qsa_cached_keys(
        ggml_context * ctx,
        ggml_tensor  * current_raw_keys,
        ggml_tensor  * key_cache,
        int64_t        kv_head,
        int64_t        n_kv) {
    GGML_ASSERT(ctx && current_raw_keys && key_cache);
    GGML_ASSERT(current_raw_keys->ne[0] == key_cache->ne[0]);
    GGML_ASSERT(kv_head >= 0 && n_kv > 0);
    GGML_ASSERT(kv_head + current_raw_keys->ne[1] <= key_cache->ne[1]);
    GGML_ASSERT(n_kv <= key_cache->ne[1]);
    GGML_ASSERT(kv_head + current_raw_keys->ne[1] <= n_kv);

    const int64_t head_dim = key_cache->ne[0];
    ggml_tensor * destination = ggml_view_2d(ctx, key_cache,
            head_dim, current_raw_keys->ne[1], key_cache->nb[1],
            size_t(kv_head) * key_cache->nb[1]);
    ggml_tensor * write = ggml_cpy(ctx, current_raw_keys, destination);
    ggml_tensor * cached_keys = ggml_view_2d(ctx, key_cache,
            head_dim, n_kv, key_cache->nb[1], 0);
    return { write, cached_keys };
}

ggml_tensor * qwen4exp_build_qsa_pooled_keys(
        ggml_context * ctx,
        ggml_tensor  * raw_keys,
        ggml_tensor  * norm_weight,
        int64_t        compress_ratio,
        int64_t        complete_blocks,
        float          rms_eps) {
    GGML_ASSERT(ctx && raw_keys && norm_weight);
    GGML_ASSERT(compress_ratio > 0 && complete_blocks > 0);
    const int64_t head_dim = raw_keys->ne[0];
    GGML_ASSERT(raw_keys->ne[1] >= compress_ratio * complete_blocks);
    GGML_ASSERT(ggml_nelements(norm_weight) == head_dim);

    // The persistent indexer cache follows the configured KV cache type and
    // is F16 by default. CPU sum_rows is F32-only; pooling in F32 also matches
    // the reference accumulation semantics before effective-gamma RMSNorm.
    if (raw_keys->type != GGML_TYPE_F32) {
        raw_keys = ggml_cast(ctx, raw_keys, GGML_TYPE_F32);
    }

    ggml_tensor * complete = ggml_view_2d(ctx, raw_keys, head_dim,
            compress_ratio * complete_blocks, raw_keys->nb[1], 0);
    complete = ggml_reshape_3d(ctx, complete, head_dim, compress_ratio, complete_blocks);
    ggml_tensor * pooled = ggml_sum_rows_ext(ctx, complete, 1);
    pooled = ggml_scale(ctx, pooled, 1.0f / float(compress_ratio));
    pooled = ggml_reshape_2d(ctx, pooled, head_dim, complete_blocks);
    pooled = ggml_rms_norm(ctx, pooled, rms_eps);
    return ggml_mul(ctx, pooled, norm_weight);
}

qwen4exp_qsa_selection_result qwen4exp_build_qsa_top_blocks(
        ggml_context * ctx,
        ggml_tensor  * query,
        ggml_tensor  * block_keys,
        ggml_tensor  * block_mask,
        int64_t        selected_blocks) {
    GGML_ASSERT(ctx && query && block_keys && block_mask);
    const int64_t head_dim = query->ne[0];
    const int64_t n_heads = query->ne[1];
    const int64_t n_queries = query->ne[2];
    const int64_t n_blocks = block_keys->ne[1];
    GGML_ASSERT(block_keys->ne[0] == head_dim);
    GGML_ASSERT(block_mask->ne[0] == n_blocks && block_mask->ne[1] == n_queries);
    GGML_ASSERT(selected_blocks > 0 && selected_blocks <= n_blocks);

    ggml_tensor * query_2d = ggml_reshape_2d(ctx, query, head_dim, n_heads * n_queries);
    ggml_tensor * scores = ggml_mul_mat(ctx, block_keys, query_2d);
    scores = ggml_relu(ctx, scores);
    scores = ggml_reshape_3d(ctx, scores, n_blocks, n_heads, n_queries);
    scores = ggml_cont(ctx, ggml_permute(ctx, scores, 1, 0, 2, 3));
    scores = ggml_sum_rows(ctx, scores);
    scores = ggml_reshape_2d(ctx, scores, n_blocks, n_queries);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(float(head_dim)));
    scores = ggml_add(ctx, scores, block_mask);
    ggml_tensor * top_blocks = ggml_cont(ctx, ggml_top_k(ctx, scores, selected_blocks));
    return { scores, top_blocks };
}

ggml_tensor * qwen4exp_build_qsa_token_indices(
        ggml_context * ctx,
        ggml_tensor  * top_blocks,
        ggml_tensor  * block_token_map,
        ggml_tensor  * tail_tokens) {
    GGML_ASSERT(ctx && top_blocks && block_token_map && tail_tokens);
    GGML_ASSERT(top_blocks->type == GGML_TYPE_I32);
    GGML_ASSERT(block_token_map->type == GGML_TYPE_I32);
    GGML_ASSERT(tail_tokens->type == GGML_TYPE_I32);
    const int64_t n_queries = top_blocks->ne[1];
    GGML_ASSERT(tail_tokens->ne[1] == n_queries);

    ggml_tensor * selected = qwen4exp_build_qsa_block_token_indices(
            ctx, top_blocks, block_token_map);
    return ggml_concat(ctx, selected, tail_tokens, 0);
}

ggml_tensor * qwen4exp_build_qsa_block_token_indices(
        ggml_context * ctx,
        ggml_tensor  * top_blocks,
        ggml_tensor  * block_token_map) {
    GGML_ASSERT(ctx && top_blocks && block_token_map);
    GGML_ASSERT(top_blocks->type == GGML_TYPE_I32);
    GGML_ASSERT(block_token_map->type == GGML_TYPE_I32);
    const int64_t selected_blocks = top_blocks->ne[0];
    const int64_t n_queries = top_blocks->ne[1];
    const int64_t compress_ratio = block_token_map->ne[0];

    ggml_tensor * batched_map = ggml_repeat_4d(ctx, block_token_map,
            compress_ratio, block_token_map->ne[1], n_queries, 1);
    ggml_tensor * selected = ggml_get_rows(ctx, batched_map, top_blocks);
    return ggml_reshape_2d(ctx, selected, compress_ratio * selected_blocks, n_queries);
}

ggml_tensor * qwen4exp_build_qsa_sparse_mask(
        ggml_context * ctx,
        ggml_tensor  * causal_mask,
        ggml_tensor  * selected_token_indices) {
    GGML_ASSERT(ctx && causal_mask && selected_token_indices);
    GGML_ASSERT(causal_mask->type == GGML_TYPE_F32 || causal_mask->type == GGML_TYPE_F16);
    GGML_ASSERT(selected_token_indices->type == GGML_TYPE_I32);
    GGML_ASSERT(selected_token_indices->ne[1] <= causal_mask->ne[1]);
    return ggml_indexer_mask(ctx, causal_mask, selected_token_indices);
}

ggml_tensor * qwen4exp_build_qsa_mask_union(
        ggml_context * ctx,
        ggml_tensor  * block_mask,
        ggml_tensor  * tail_mask,
        ggml_tensor  * tail_enabled) {
    GGML_ASSERT(ctx && block_mask && tail_mask && tail_enabled);
    GGML_ASSERT(ggml_are_same_shape(block_mask, tail_mask));
    GGML_ASSERT(tail_enabled->type == GGML_TYPE_F32);
    GGML_ASSERT(tail_enabled->ne[0] == 1 && tail_enabled->ne[1] == block_mask->ne[1]);

    if (block_mask->type != GGML_TYPE_F32) block_mask = ggml_cast(ctx, block_mask, GGML_TYPE_F32);
    if (tail_mask->type  != GGML_TYPE_F32) tail_mask  = ggml_cast(ctx, tail_mask,  GGML_TYPE_F32);
    ggml_tensor * block_selected = ggml_exp(ctx, block_mask);
    ggml_tensor * tail_selected  = ggml_exp(ctx, tail_mask);
    ggml_tensor * enabled = ggml_repeat_4d(ctx, tail_enabled,
            tail_mask->ne[0], tail_mask->ne[1], tail_mask->ne[2], tail_mask->ne[3]);
    tail_selected = ggml_mul(ctx, tail_selected, enabled);
    ggml_tensor * selected = ggml_clamp(ctx,
            ggml_add(ctx, block_selected, tail_selected), 0.0f, 1.0f);
    return ggml_log(ctx, selected);
}

ggml_tensor * qwen4exp_build_recurrent_prefix_states(
        ggml_context * ctx,
        ggml_tensor  * conv_result,
        ggml_tensor  * delta_result,
        ggml_tensor  * ple_prefix_states) {
    GGML_ASSERT(ctx && conv_result && delta_result);
    GGML_ASSERT(conv_result->op == GGML_OP_SSM_CONV && delta_result->op == GGML_OP_DELTA_NET);
    // The ordinary fused state-copy optimization can redirect the final SSM
    // tail into its persistent cache slot. A journal also reads that tail, so
    // retain the result and decline that optimization for this producer.
    ggml_set_output(delta_result);
    auto * conv_prior = conv_result->src[0];
    auto * conv_input = conv_result->src[1];
    auto * delta_prior = delta_result->src[5];
    auto * delta_value = delta_result->src[2];
    const int64_t tokens = conv_input->ne[1];
    const int64_t conv_length = conv_prior->ne[0];
    const int64_t conv_channels = conv_input->ne[0];
    const int64_t conv_width = conv_length * conv_channels;
    const int64_t ssm_width = ggml_nelements(delta_prior);
    GGML_ASSERT(tokens > 0 && conv_length > 0 && conv_channels > 0);
    GGML_ASSERT(conv_prior->ne[2] == 1 && conv_prior->ne[3] == 1);
    GGML_ASSERT(delta_prior->ne[3] == 1 && delta_value->ne[1] == tokens && delta_value->ne[3] == 1);
    GGML_ASSERT(conv_result->type == GGML_TYPE_F32 && delta_result->type == GGML_TYPE_F32);

    // SSM_CONV result appends [kernel,channels] with one padding element
    // before each channel's retained history, as consumed by build_qkv.
    auto * conv_final = ggml_cont(ctx, ggml_view_2d(ctx, conv_result,
            conv_length, conv_channels, (conv_length + 1) * sizeof(float),
            (1 + conv_channels * tokens) * sizeof(float)));
    conv_final = ggml_reshape_2d(ctx, conv_final, conv_width, 1);
    const int64_t output_width = delta_value->ne[0] * delta_value->ne[2];
    auto * ssm_final = ggml_view_2d(ctx, delta_result, ssm_width, 1,
            ssm_width * sizeof(float), tokens * output_width * sizeof(float));
    auto * conv_states = conv_final;
    auto * ssm_states = ssm_final;
    if (tokens > 1) {
        auto * conv_steps = conv_result->src[4];
        auto * ssm_steps = delta_result->src[6];
        GGML_ASSERT(conv_steps && ssm_steps);
        GGML_ASSERT(conv_steps->type == GGML_TYPE_F32 && ssm_steps->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(conv_steps) && ggml_is_contiguous(ssm_steps));
        GGML_ASSERT(ggml_nelements(conv_steps) >= conv_width * tokens);
        GGML_ASSERT(ggml_nelements(ssm_steps) >= ssm_width * (tokens - 1));
        auto * conv_early = ggml_view_2d(ctx, conv_steps, conv_width, tokens - 1,
                conv_width * sizeof(float), 0);
        auto * ssm_early = ggml_view_2d(ctx, ssm_steps, ssm_width, tokens - 1,
                ssm_width * sizeof(float), 0);
        conv_states = ggml_concat(ctx, conv_early, conv_final, 1);
        ssm_states = ggml_concat(ctx, ssm_early, ssm_final, 1);
    }
    auto * states = ggml_concat(ctx, conv_states, ssm_states, 0);
    if (ple_prefix_states) {
        GGML_ASSERT(ple_prefix_states->type == GGML_TYPE_F32 && ggml_is_contiguous(ple_prefix_states));
        GGML_ASSERT(ple_prefix_states->ne[2] == tokens && ple_prefix_states->ne[3] == 1);
        auto * ple_states = ggml_reshape_2d(ctx, ple_prefix_states,
                ple_prefix_states->ne[0] * ple_prefix_states->ne[1], tokens);
        states = ggml_concat(ctx, states, ple_states, 0);
    }
    GGML_ASSERT(ggml_is_contiguous(states));
    return states;
}
