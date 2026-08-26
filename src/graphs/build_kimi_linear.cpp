#include "../llama-build-context.h"
#include "../llama-context.h"
#include "../llama-delta-net.h"
#include "../llama-model.h"

ggml_cgraph * llm_build_context::build_kimi_linear() {
    ggml_cgraph * gf = new_graph_custom();

    int32_t n_tokens_cur = n_tokens;
    auto inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);
    auto KQ_mask = build_inp_KQ_mask();

    lctx.inp_s_seq_qnext = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 1, n_tokens);
    cb(lctx.inp_s_seq_qnext, "inp_s_seq_qnext", -1);
    ggml_set_input(lctx.inp_s_seq_qnext);

    delta_net delta(lctx, batch);
    const float kq_scale = 1.0f / sqrtf(float(hparams.n_embd_head_k(0)));
    constexpr float attn_factor_scaled = 1.0f;
#ifdef GGML_USE_VULKAN
    constexpr bool use_f32_attn_precision = true;
#else
    const bool use_f32_attn_precision = lctx.cparams.graph_attn_precision == GGML_TYPE_F32;
#endif
    const bool pp_opt = n_tokens >= 128 && lctx.cparams.mla_attn > 1;

    ggml_tensor * cur = nullptr;
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * ffn_inp = nullptr;

        if (hparams.is_recurrent(il)) {
            cur = delta.build_layer_kda(ctx0, gf, inpL, nullptr, il, cb);
            ffn_inp = cur;
        } else {
            cur = build_deepseek2_layer_attention(gf, il, inpL, KQ_mask, nullptr, nullptr,
                    kq_scale, attn_factor_scaled, use_f32_attn_precision,
                    /*is_lite=*/true, pp_opt);
            ffn_inp = ggml_add(ctx0, cur, inpL);
        }

        if (il == n_layer - 1) {
            auto inp_out_ids = build_inp_out_ids();
            n_tokens_cur = n_outputs;
            ffn_inp = ggml_get_rows(ctx0, ffn_inp, inp_out_ids);
        }

        cur = llm_build_norm(ctx0, ffn_inp, hparams, model.layers[il].ffn_norm,
                nullptr, LLM_NORM_RMS, cb, il);

        if (il < static_cast<int>(hparams.n_layer_dense_lead)) {
            cur = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up,   nullptr, nullptr,
                    model.layers[il].ffn_gate, nullptr, nullptr,
                    model.layers[il].ffn_down, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
        } else {
            auto moe = llm_build_moe_ffn(ctx0, lctx, cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    true, hparams.expert_weights_scale,
                    static_cast<llm_expert_gating_func_type>(hparams.expert_gating_func),
                    cb, il, gf, false, model.layers[il].ffn_up_gate_exps);
            auto shared = llm_build_ffn(ctx0, lctx, nullptr, cur,
                    model.layers[il].ffn_up_shexp,   nullptr, nullptr,
                    model.layers[il].ffn_gate_shexp, nullptr, nullptr,
                    model.layers[il].ffn_down_shexp, nullptr, nullptr,
                    nullptr, LLM_FFN_SILU, LLM_FFN_PAR, cb, il, gf);
            cur = ggml_add(ctx0, moe, shared);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);
        inpL = cur;
    }

    GGML_UNUSED(n_tokens_cur);
    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);
    ggml_build_forward_expand(gf, cur);
    return gf;
}
