#pragma once

#include "llama-build-context.h"

#include <utility>

struct delta_net {
    delta_net(llama_context & lctx, const llama_batch & batch);
    ~delta_net();

    // Qwen4Exp GGUF conversion stores value heads in cyclic key-head order.
    // Verified against original layer-0 alpha/beta/conv/decay tensors: source
    // repeat_interleave heads are transposed [key_head,repeat] -> [repeat,key_head].
    static constexpr int qwen4exp_repeat_type = 1;

    // Used for speculative decoding to enable per-step state checkpoint restoration.
    bool save_per_step_states = false;

    static std::pair<ggml_tensor *, ggml_tensor *> build_fused_delta_net(ggml_context * ctx0,
                      ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                      ggml_tensor * g, ggml_tensor * beta, ggml_tensor * state,
                      int il, const llm_build_cb & cb, int repeat_type,
                      ggml_tensor * per_step_ckpt = nullptr);

    ggml_tensor * build_layer_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
            uint32_t state_seq_id_local, bool reset_state_local, int il, const llm_build_cb & cb) const;

    ggml_tensor * build_layer_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const;

    // Qwen4Exp applies its four-stream GR before the recurrent block and its
    // stream injection after it, so this variant intentionally omits the
    // ordinary pre-norm and residual add performed by build_layer_attn_linear.
    ggml_tensor * build_qwen4exp_attn_linear(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * mixed_input, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb, ggml_tensor * ple_prefix_states = nullptr) const;

    ggml_tensor * build_layer_kda(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_out_ids, int il, const llm_build_cb & cb) const;

private:

    llama_context     & lctx;
    const llama_batch & batch;
    std::vector<llama_seq_id> token_seq_ids;
    bool all_same_seq;
    bool has_unique_seq_ids;

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * input, int il, const llm_build_cb & cb,
            ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input, int il,
            const llm_build_cb & cb);

    static std::pair<ggml_tensor *, ggml_tensor *> build_qkvz(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * wqkv, ggml_tensor * wqkv_gate, ggml_tensor * ssm_in,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, ggml_tensor * input,
            int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static std::pair<ggml_tensor *, ggml_tensor *> build_beta_gate(llama_context & lctx, ggml_context * ctx0,
            ggml_tensor * ssm_beta_alpha, ggml_tensor * ssm_beta, ggml_tensor * ssm_alpha,
            ggml_tensor * ssm_dt, ggml_tensor * ssm_a, int64_t num_k_heads, int64_t num_v_heads, int64_t n_seqs,
            ggml_tensor * cur, int il, const llm_build_cb & cb, ggml_cgraph * gf);

    static ggml_tensor * build_qkv(ggml_context * ctx0, ggml_tensor * state_storage, ggml_tensor * ssm_conv1d,
            ggml_tensor * qkv_mixed, ggml_tensor * inp_s_seq_qnext, ggml_tensor * beta, ggml_tensor * gate,
            int64_t head_k_dim, int64_t num_k_heads, int64_t head_v_dim, int64_t num_v_heads, int64_t ssm_d_conv,
            int64_t state_seq_id_local, uint32_t qnext_state_slots, bool reset_state_local,
            float eps_norm, int repeat_type, int il, const llm_build_cb & cb, ggml_cgraph * gf,
            ggml_tensor * per_step_ssm = nullptr, ggml_tensor * per_step_conv = nullptr,
            bool qwen4exp_l2 = false, ggml_tensor * prefix_destination = nullptr,
            ggml_tensor * ple_prefix_states = nullptr);

    static ggml_tensor * build_gated_output(llama_context & lctx, ggml_context * ctx0, ggml_tensor * ssm_norm, ggml_tensor * ssm_out,
            ggml_tensor * output, ggml_tensor * z, int64_t head_v_dim, int64_t num_v_heads, int64_t n_tok, int il, const llm_build_cb & cb);

    ggml_tensor * build_layer_kda_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * cur, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
            uint32_t state_seq_id, bool reset_state, int il, const llm_build_cb & cb) const;

    ggml_tensor * build_qwen4exp_attn_linear_core(ggml_context * ctx0, ggml_cgraph * gf,
            ggml_tensor * mixed_input, ggml_tensor * inp_s_seq_qnext, ggml_tensor * inp_out_ids,
            uint32_t state_seq_id, bool reset_state, int il, const llm_build_cb & cb, ggml_tensor * ple_prefix_states = nullptr) const;
};
