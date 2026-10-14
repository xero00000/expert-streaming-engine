#pragma once

#include "ggml.h"

#include <cstdint>
#include <string>

// Native CPU/CUDA cache-row gather and write formats. Validate before cache
// allocation or live migration; packed rows require block-aligned head widths.
bool qwen4exp_cache_type_supported(ggml_type type);
bool qwen4exp_validate_cache_type(ggml_type type, int64_t head_width,
        bool value_cache, bool flash_attn, std::string & error);

// Gather [channels, physical rows, heads] into an explicitly ordered cache.
// row_ids is I32 [ordered rows, heads]. IDs must be valid physical rows;
// callers supply sequence order and mask any repeated padding rows. Preserve
// F16/F32 bits; BF16 and packed formats decode only selected rows into F32.
ggml_tensor * qwen4exp_build_cache_rows(
        ggml_context * ctx, ggml_tensor * cache, ggml_tensor * row_ids);

// Reference L2 normalization is x / sqrt(sum(x*x) + eps), not GGML's
// clamp-based L2 normalization. Rows run along the first tensor dimension.
ggml_tensor * qwen4exp_build_l2_norm(ggml_context * ctx, ggml_tensor * input, float eps);

// Qwen3.8-Flash-Next's configured recurrent output gate is sigmoid. Its
// convolution and expert activation remain SiLU.
ggml_tensor * qwen4exp_build_recurrent_output_gate(
        ggml_context * ctx, ggml_tensor * normalized, ggml_tensor * gate);

// The indexer shares the text model's partial rotary dimension. Its head
// width can be larger; features beyond rotary_dims remain unrotated.
ggml_tensor * qwen4exp_build_qsa_rope(ggml_context * ctx, ggml_tensor * input,
        ggml_tensor * positions, int rotary_dims, int original_context,
        float freq_base, float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow);

// Expands token embeddings into the initial hyper-connection layout. Each
// stream receives an identical hidden vector, matching Tensor.repeat(...,
// hc_count) in the pinned reference.
ggml_tensor * qwen4exp_build_initial_streams(
        ggml_context * ctx,
        ggml_tensor  * token_embeddings,
        int64_t        stream_count);

// Qwen4Exp's gated residual (GR) reads all hyper-connection streams and
// returns one mixed hidden vector plus optional per-stream injection weights.
// The input and norm tensors use GGML's [hidden * streams, tokens] layout.
struct qwen4exp_gated_residual_result {
    ggml_tensor * mixed_input       = nullptr; // [hidden, tokens]
    ggml_tensor * hyper_input       = nullptr; // original [hidden * streams, tokens]
    ggml_tensor * injection_weights = nullptr; // [streams, tokens], absent for the output mixer
};

qwen4exp_gated_residual_result qwen4exp_build_gated_residual(
        ggml_context * ctx,
        ggml_tensor  * hyper_input,
        ggml_tensor  * norm_weight,
        ggml_tensor  * mix_down,
        ggml_tensor  * mix_up,
        ggml_tensor  * inject,
        int64_t        hidden_size,
        int64_t        stream_count,
        float          rms_eps);

// Broadcast one block output back into every stream using the GR injection
// weights: hyper_input + flatten(block_output[..., None, :] * weights).
ggml_tensor * qwen4exp_build_stream_injection(
        ggml_context * ctx,
        ggml_tensor  * hyper_input,
        ggml_tensor  * block_output,
        ggml_tensor  * injection_weights,
        int64_t        hidden_size,
        int64_t        stream_count);

struct qwen4exp_ple_gate_result {
    ggml_tensor * gated_value        = nullptr; // [hidden * streams, tokens]
    ggml_tensor * gated_value_normed = nullptr; // [hidden * streams, tokens]
};

// Optional intermediates for the graph's bounded evaluation callback. Keep
// the original two-pointer result and function available to existing callers.
struct qwen4exp_ple_gate_trace {
    ggml_tensor * key_projected      = nullptr;
    ggml_tensor * key_normed         = nullptr;
    ggml_tensor * query_normed       = nullptr;
    ggml_tensor * gate_dot           = nullptr;
    ggml_tensor * gate_sigmoid       = nullptr;
    ggml_tensor * value_projected    = nullptr;
};

// PLE projection and stream gating before the stateful dilated convolution.
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
        float          rms_eps);

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
        qwen4exp_ple_gate_trace * trace);

struct qwen4exp_ple_conv_result {
    ggml_tensor * output     = nullptr; // SiLU depthwise-convolution result [channels, tokens]
    ggml_tensor * next_state = nullptr; // most recent dilation * (kernel - 1) inputs
};

struct qwen4exp_ple_state_result {
    ggml_tensor * prior_state = nullptr; // [channels, state_length]
    ggml_tensor * destination = nullptr; // raw writable storage view
};

// Maps PLE's flattened convolution tail inside one recurrent sequence row and
// creates the writeback node. The caller expands write after all consumers of
// prior_state. reset_state supplies a graph-local zero prior without relying
// on stale storage when a sequence restarts at position zero.
qwen4exp_ple_state_result qwen4exp_build_ple_state(
        ggml_context * ctx,
        ggml_tensor  * state_storage,
        int64_t        base_state_elements,
        int64_t        sequence_row,
        int64_t        channels,
        int64_t        state_length,
        bool           reset_state);

ggml_tensor * qwen4exp_build_ple_state_write(
        ggml_context * ctx,
        ggml_tensor  * next_state,
        ggml_tensor  * destination);

qwen4exp_ple_conv_result qwen4exp_build_ple_dilated_conv(
        ggml_context * ctx,
        ggml_tensor  * normalized_input,
        ggml_tensor  * conv_weight,
        ggml_tensor  * prior_state,
        int64_t        kernel_size,
        int64_t        dilation);

// Optional contiguous [channels, state_length, tokens] rolling-state journal.
// Entry t is the exact convolution input history after token t. Callers must
// budget its storage and keep recurrent state/token history journals separately.
// prefix_columns is a contiguous I32 input of state_length * tokens entries,
// initialized to t + h + 1 at t * state_length + h before graph execution.
// The ordinary overload creates no journal and retains its existing ABI.
qwen4exp_ple_conv_result qwen4exp_build_ple_dilated_conv(
        ggml_context * ctx,
        ggml_tensor  * normalized_input,
        ggml_tensor  * conv_weight,
        ggml_tensor  * prior_state,
        int64_t        kernel_size,
        int64_t        dilation,
        ggml_tensor ** prefix_states,
        ggml_tensor  * prefix_columns);

// PLE injects both the directly gated value and the nonlinear dilated-conv
// branch before the attention hyper-connection mixer.
ggml_tensor * qwen4exp_build_ple_output(
        ggml_context * ctx,
        ggml_tensor  * gated_value,
        ggml_tensor  * convolution_output);

struct qwen4exp_qsa_selection_result {
    ggml_tensor * scores     = nullptr; // [complete_blocks, queries]
    ggml_tensor * top_blocks = nullptr; // I32 [selected_blocks, queries]
};

struct qwen4exp_qsa_cache_result {
    ggml_tensor * write       = nullptr; // CPY node for the current raw-key batch
    ggml_tensor * cached_keys = nullptr; // [head_dim, n_kv] raw keys after the write
};

// Writes raw (pre-pool, pre-norm, pre-RoPE) QSA keys at kv_head and exposes
// the complete cache prefix used for block compression. The caller must add
// write to the graph before consumers of cached_keys. For graph reuse, write
// is also the node registered in llama_context::dsa_cache_copies with
// step = key_cache->nb[1].
qwen4exp_qsa_cache_result qwen4exp_build_qsa_cached_keys(
        ggml_context * ctx,
        ggml_tensor  * current_raw_keys,
        ggml_tensor  * key_cache,
        int64_t        kv_head,
        int64_t        n_kv);

// Mean-pools each complete contiguous key block and applies the GGUF effective
// RMSNorm gamma. RoPE is applied by the caller at each block's start position.
ggml_tensor * qwen4exp_build_qsa_pooled_keys(
        ggml_context * ctx,
        ggml_tensor  * raw_keys,
        ggml_tensor  * norm_weight,
        int64_t        compress_ratio,
        int64_t        complete_blocks,
        float          rms_eps);

// Scores already-pooled, normalized and RoPE-transformed block keys using
// sum_h(ReLU(q_h dot k_block)) / sqrt(head_dim), then applies the causal block
// mask and selects the highest-scoring complete blocks. Tail tokens are not
// complete blocks and remain a separate, always-included attention segment.
qwen4exp_qsa_selection_result qwen4exp_build_qsa_top_blocks(
        ggml_context * ctx,
        ggml_tensor  * query,
        ggml_tensor  * block_keys,
        ggml_tensor  * block_mask,
        int64_t        selected_blocks);

// Expands each selected block through a static [compress_ratio, block_count]
// token map and appends the per-query causal tail. Padding slots must repeat a
// valid selected token: ggml_indexer_mask does not accept the reference's -1
// dummy index.
ggml_tensor * qwen4exp_build_qsa_token_indices(
        ggml_context * ctx,
        ggml_tensor  * top_blocks,
        ggml_tensor  * block_token_map,
        ggml_tensor  * tail_tokens);

ggml_tensor * qwen4exp_build_qsa_block_token_indices(
        ggml_context * ctx,
        ggml_tensor  * top_blocks,
        ggml_tensor  * block_token_map);

ggml_tensor * qwen4exp_build_qsa_sparse_mask(
        ggml_context * ctx,
        ggml_tensor  * causal_mask,
        ggml_tensor  * selected_token_indices);

// Unions complete-block and variable-length tail masks exactly. Inputs use
// additive attention-mask values (0 or -inf); tail_enabled is F32 [1, queries]
// and suppresses padding-only tail rows without introducing sentinel indices.
ggml_tensor * qwen4exp_build_qsa_mask_union(
        ggml_context * ctx,
        ggml_tensor  * block_mask,
        ggml_tensor  * tail_mask,
        ggml_tensor  * tail_enabled);

// Assemble a complete recurrent cache row after each input, without state
// arithmetic: [conv history | SSM state | optional PLE history] x tokens.
// Both operators must have one sequence and per-step storage for tokens > 1.
// Final columns come from operator results, making their side-effect journals
// depend on their producers in the graph. Optional PLE history is [C,L,tokens].
ggml_tensor * qwen4exp_build_recurrent_prefix_states(
        ggml_context * ctx,
        ggml_tensor  * conv_result,
        ggml_tensor  * delta_result,
        ggml_tensor  * ple_prefix_states = nullptr);
