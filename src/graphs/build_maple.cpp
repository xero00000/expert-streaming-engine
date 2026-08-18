#include "../llama-build-context.h"
#include "../llama-model.h"
#include "../llama-context.h"

// DeepGrove Maple-Preview (20B-A1B ternary MoE):
//   - 3:1 SWA-512 : full-attention (GA) hybrid
//   - Flash-head QK norms (attn_q_norm / attn_k_norm)
//   - Partial RoPE on SWA (n_rot_swa=64); no RoPE on GA (n_rot=0)
//   - 256 experts, top-8, Softmax + renorm routing, SiLU MoE FFN
//
// Ported from stamsam/llama.cpp (prism branch) maple graph into the ik-style
// build_std_attention / llm_build_std_moe_ffn helpers.

ggml_cgraph * llm_build_context::build_maple() {
    ggml_cgraph * gf = new_graph_custom();

    struct ggml_tensor * cur;
    struct ggml_tensor * inpL;

    inpL = llm_build_inp_embd(ctx0, lctx, hparams, batch, model.tok_embd, cb);

    struct ggml_tensor * inp_pos     = build_inp_pos();
    struct ggml_tensor * inp_out_ids = n_tokens > 1 ? build_inp_out_ids() : nullptr;
    struct ggml_tensor * KQ_mask     = build_inp_KQ_mask();
    struct ggml_tensor * KQ_mask_swa = hparams.n_swa > 0 ? build_inp_KQ_mask_swa() : nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const bool is_swa = hparams.swa_layers[il];
        const int64_t n_embd_head = hparams.n_embd_head_v(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_k(il));

        auto KQ_mask_l = is_swa ? KQ_mask_swa : KQ_mask;
        GGML_ASSERT(KQ_mask_l != nullptr);

        // build_std_attention applies q/k RMSNorm when present, then rope_n_rot(il)
        // (0 on full-attention layers, 64 on SWA for Maple GGUFs).
        cur = build_std_attention(gf, model.layers[il].attn_norm, inpL,
                inp_pos, il == n_layer - 1 ? inp_out_ids : nullptr, nullptr, KQ_mask_l,
                nullptr, nullptr, 1.0f/sqrtf(float(n_embd_head)), 0.0f, is_swa ? hparams.n_swa : 0, il, true, false, true);

        cur = llm_build_std_moe_ffn(ctx0, lctx, model.layers[il].ffn_norm, cur,
                model.layers[il].ffn_gate_inp,  nullptr,
                model.layers[il].ffn_up_exps,   nullptr,
                model.layers[il].ffn_gate_exps, nullptr,
                model.layers[il].ffn_down_exps, nullptr,
                nullptr,
                nullptr, nullptr,
                nullptr, nullptr,
                nullptr, nullptr,
                n_expert, n_expert_used,
                LLM_FFN_SILU, true, false, 0.0f,
                LLM_EXPERT_GATING_FUNC_SOFTMAX,
                LLM_FFN_SILU, cb, il, gf, true,
                model.layers[il].ffn_up_gate_exps);

        cur = lctx.cvec.apply_to(ctx0, cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_output(lctx, ctx0, inpL, model.output, model.output_norm, cb);
    cb(cur, "result_output", -1);

    ggml_build_forward_expand(gf, cur);

    return gf;
}
