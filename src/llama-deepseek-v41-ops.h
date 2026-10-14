#pragma once

#include "ggml.h"

namespace deepseek_v41 {

struct hc_mixes {
    ggml_tensor * pre = nullptr;  // [streams, tokens], consumed by the NEXT sublayer
    ggml_tensor * post = nullptr; // [streams, tokens]
    ggml_tensor * comb = nullptr; // [destination stream, source stream, tokens]
};

// x is [hidden, streams, tokens]. The projection uses one normalization
// statistic over all hidden*streams values, matching the released V4.1 method.
hc_mixes build_hc_mixes(ggml_context * ctx, ggml_tensor * x,
                        ggml_tensor * projection, ggml_tensor * scale, ggml_tensor * base,
                        float norm_eps, int sinkhorn_iterations, float hc_eps);

// V4.1 attention consumes the previous block's FFN pre-mix. Its FFN consumes
// this block's attention pre-mix. Do not feed a sublayer its own new pre-mix.
ggml_tensor * build_hc_collapse(ggml_context * ctx, ggml_tensor * x, ggml_tensor * incoming_pre);

ggml_tensor * build_hc_expand(ggml_context * ctx, ggml_tensor * sublayer_output,
                             ggml_tensor * residual, const hc_mixes & mixes);

struct engram_injection {
    ggml_tensor * output = nullptr; // [hidden, streams, tokens]
    ggml_tensor * gate = nullptr;   // [1, streams, tokens]
};

// Embeddings are already gathered/dequantized and flattened per token.
// key_value_projection maps [lookup_width] to [hidden*(streams+1)].
// q/k weights are [hidden, streams], text_mask is [1, 1, tokens] with 0/1 values.
// Production precision rounds decoded embeddings, applies FP8 QAT, and rounds
// projection and final residual outputs to BF16 in F32 containers.
engram_injection build_engram_injection(
    ggml_context * ctx, ggml_tensor * x, ggml_tensor * embeddings,
    ggml_tensor * key_value_projection, ggml_tensor * q_weight, ggml_tensor * k_weight,
    ggml_tensor * text_mask, float norm_eps, bool production_precision = false);

// Shared SwiGLU expert. Input/output use F32 containers for BF16 values.
// The returned output is rounded before the caller adds the routed FP32 sum.
ggml_tensor * build_shared_expert(ggml_context * ctx, ggml_tensor * x,
    ggml_tensor * gate_weight, ggml_tensor * up_weight, ggml_tensor * down_weight);

struct compression_result {
    ggml_tensor * latent = nullptr; // complete groups, normalized; null when no group completed
    ggml_tensor * tail_values = nullptr;
    ggml_tensor * tail_scores = nullptr;
};

// Projected values/scores already concatenate any prior incomplete group with
// the current chunk, in chronological order. Outputs retain the new tail for
// the next call; no compression means no latent publication, not an empty cache.
compression_result build_compression(
    ggml_context * ctx, ggml_tensor * values, ggml_tensor * scores,
    ggml_tensor * norm_weight, int ratio, float norm_eps, bool bf16_pool = false);

struct rope_config {
    int rotary_dimensions = 0;
    int original_context = 0;
    float theta = 10000.0f;
    float factor = 1.0f;
    float beta_fast = 32.0f;
    float beta_slow = 1.0f;
};

// Rotates only the tail, with adjacent real/imaginary pairs. Positions are I32
// absolute token positions (compressed row j uses j*ratio). Attention output
// uses inverse=true to undo the query rotation before its grouped projection.
ggml_tensor * build_rope_tail(ggml_context * ctx, ggml_tensor * x,
                              ggml_tensor * positions, const rope_config & config, bool inverse = false);

// q=[D,H,T], kv=[D,N], additive mask=[N,pad(T,16)], sinks=[H].
// The caller supplies the union of visible window and selected compressed rows.
// K and V are the same latent, including its rotary tail. Sink logits enter
// the softmax denominator only. Output is [D,H,T].
ggml_tensor * build_attention_core(ggml_context * ctx, ggml_tensor * q,
    ggml_tensor * kv, ggml_tensor * mask, ggml_tensor * sinks);

// Input has already undergone inverse RoPE. wo_a is [D*H/G,R,G]; wo_b
// is [R*G,hidden]. Each group projects only its own heads.
ggml_tensor * build_attention_output(ggml_context * ctx, ggml_tensor * attended,
    ggml_tensor * wo_a, ggml_tensor * wo_b);

// q=[D,H,T], k=[D,N], weights=[H,T] (already scaled by 1/sqrt(D*H)).
// Q and K have already undergone RoPE and FP4 activation quantization.
// Output [N,T] is the weighted sum over rectified head scores, before masking.
ggml_tensor * build_indexer_scores(ggml_context * ctx, ggml_tensor * q,
    ggml_tensor * k, ggml_tensor * weights, bool bf16_intermediates = false);

} // namespace deepseek_v41
