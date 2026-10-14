#include "llama-deepseek-v41-ops.h"
#include "ggml-hc.h"
#include "../ggml/src/ggml-qat.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace deepseek_v41 {

static void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

static void check_streams(ggml_tensor * x) {
    require(x && x->type == GGML_TYPE_F32 && x->ne[0] > 0 && x->ne[1] > 0 && x->ne[1] <= 8 &&
            x->ne[2] > 0 && x->ne[3] == 1, "V4.1 streams must be FP32 [hidden, streams, tokens]");
}

hc_mixes build_hc_mixes(ggml_context * ctx, ggml_tensor * x,
                       ggml_tensor * projection, ggml_tensor * scale, ggml_tensor * base,
                       float norm_eps, int sinkhorn_iterations, float hc_eps) {
    check_streams(x);
    const int64_t hidden = x->ne[0], streams = x->ne[1], tokens = x->ne[2];
    const int64_t width = streams * (streams + 2);
    require(projection && projection->ne[0] == hidden * streams && projection->ne[1] == width &&
            ggml_n_dims(projection) <= 2, "invalid V4.1 HC projection");
    require(scale && base && ggml_nelements(scale) == 3 && ggml_nelements(base) == width,
            "invalid V4.1 HC coefficients");
    require(norm_eps > 0 && std::isfinite(norm_eps) && hc_eps > 0 && std::isfinite(hc_eps) && sinkhorn_iterations > 1,
            "invalid V4.1 HC numerical parameters");
    auto * flat = ggml_reshape_2d(ctx, ggml_cont(ctx, x), hidden * streams, tokens);
    // Match projection(x) / rms(x), keeping the normalization reduction FP32.
    auto * deviation = ggml_sqrt(ctx, ggml_scale_bias(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, flat)),
                                                    1.0f / float(hidden * streams), norm_eps));
    auto * raw_projection = ggml_mul_mat(ctx, projection, flat);
    ggml_mul_mat_set_prec(raw_projection, GGML_PREC_F32);
    auto * projected = ggml_div(ctx, raw_projection, deviation);
    auto * coefficients = ggml_hc_pre(ctx, projected, scale, base, int(streams), sinkhorn_iterations, hc_eps);
    hc_mixes result;
    result.pre = ggml_view_2d(ctx, coefficients, streams, tokens, streams * sizeof(float), 0);
    result.post = ggml_view_2d(ctx, coefficients, streams, tokens, streams * sizeof(float), streams * tokens * sizeof(float));
    result.comb = ggml_view_3d(ctx, coefficients, streams, streams, tokens,
                             streams * sizeof(float), streams * streams * sizeof(float), 2 * streams * tokens * sizeof(float));
    return result;
}

ggml_tensor * build_hc_collapse(ggml_context * ctx, ggml_tensor * x, ggml_tensor * incoming_pre) {
    check_streams(x);
    require(incoming_pre && incoming_pre->type == GGML_TYPE_F32 && incoming_pre->ne[0] == x->ne[1] &&
            incoming_pre->ne[1] == x->ne[2] && ggml_n_dims(incoming_pre) <= 2, "invalid V4.1 incoming pre-mix");
    auto * weights = ggml_reshape_4d(ctx, ggml_cont(ctx, incoming_pre), 1, x->ne[1], x->ne[2], 1);
    return ggml_mul_multi_add(ctx, x, weights);
}

ggml_tensor * build_hc_expand(ggml_context * ctx, ggml_tensor * sublayer_output,
                            ggml_tensor * residual, const hc_mixes & mixes) {
    check_streams(residual);
    require(sublayer_output && sublayer_output->type == GGML_TYPE_F32 && sublayer_output->ne[0] == residual->ne[0] &&
            sublayer_output->ne[1] == residual->ne[2] && ggml_n_dims(sublayer_output) <= 2 &&
            mixes.post && mixes.comb, "invalid V4.1 HC expansion");
    // Preserve the original projection as an explicit graph dependency.
    // FP32 pre-mixes keep their existing HC_PRE semantics; expansion retains
    // FP64 post/comb coefficients until its final F32 result to avoid BF16 ties.
    auto * coefficients = mixes.comb->view_src;
    require(coefficients && coefficients->op == GGML_OP_HC_PRE && mixes.post->view_src == coefficients,
            "V4.1 HC expansion requires matching HC_PRE views");
    float eps;
    std::memcpy(&eps, &coefficients->op_params[2], sizeof(eps));
    return ggml_hc_post_precise(ctx, sublayer_output, residual, coefficients->src[0],
            coefficients->src[1], coefficients->src[2], coefficients->op_params[1], eps);
}

engram_injection build_engram_injection(
        ggml_context * ctx, ggml_tensor * x, ggml_tensor * embeddings,
        ggml_tensor * key_value_projection, ggml_tensor * q_weight, ggml_tensor * k_weight,
        ggml_tensor * text_mask, float norm_eps, bool production_precision) {
    check_streams(x);
    const int64_t hidden = x->ne[0], streams = x->ne[1], tokens = x->ne[2];
    require(embeddings && embeddings->ne[1] == tokens && ggml_n_dims(embeddings) <= 2 &&
            key_value_projection && key_value_projection->ne[0] == embeddings->ne[0] &&
            key_value_projection->ne[1] == hidden * (streams + 1), "invalid V4.1 Engram projection");
    require(q_weight && k_weight && q_weight->ne[0] == hidden && q_weight->ne[1] == streams &&
            ggml_are_same_shape(q_weight, k_weight), "invalid V4.1 Engram gate weights");
    require(text_mask && text_mask->type == GGML_TYPE_F32 && text_mask->ne[0] == 1 &&
            text_mask->ne[1] == 1 && text_mask->ne[2] == tokens && text_mask->ne[3] == 1,
            "invalid V4.1 Engram text mask");
    require(norm_eps > 0 && std::isfinite(norm_eps), "invalid V4.1 Engram epsilon");
    const auto bf16 = [&](ggml_tensor * t) {
        return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
    };
    if (production_precision) embeddings = ggml_qat(ctx, ggml_cont(ctx, bf16(embeddings)), GGML_QAT_FP8_E8M0);
    auto * projection = ggml_mul_mat(ctx, key_value_projection, embeddings);
    ggml_mul_mat_set_prec(projection, GGML_PREC_F32);
    if (production_precision) projection = bf16(projection);
    auto * key = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, projection, hidden * streams, tokens,
                                                      projection->nb[1], 0)), hidden, streams, tokens);
    auto * value = ggml_reshape_3d(ctx, ggml_cont(ctx, ggml_view_2d(ctx, projection, hidden, tokens,
            projection->nb[1], hidden * streams * sizeof(float))), hidden, 1, tokens);
    auto * weight = ggml_mul(ctx, ggml_cast(ctx, q_weight, GGML_TYPE_F32), ggml_cast(ctx, k_weight, GGML_TYPE_F32));
    // Normalize each HC copy independently. Normalizing operands first avoids
    // underflow of a product of two tiny variances at norm_eps=1e-20.
    auto * product = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, x, norm_eps), weight), ggml_rms_norm(ctx, key, norm_eps));
    auto * dot = ggml_scale(ctx, ggml_sum_rows(ctx, product), 1.0f / std::sqrt(float(hidden)));
    auto * magnitude = ggml_sqrt(ctx, ggml_clamp(ctx, ggml_abs(ctx, dot), 1e-6f, std::numeric_limits<float>::infinity()));
    // SUM_ROWS starts from +0. For a zero dot, copysign(sqrt(1e-6), +0) is
    // positive, not zero as sign(dot)*sqrt(...) would be.
    auto * sign = ggml_scale_bias(ctx, ggml_step(ctx, ggml_neg(ctx, dot)), -2.0f, 1.0f);
    auto * gate = ggml_mul(ctx, ggml_sigmoid(ctx, ggml_mul(ctx, magnitude, sign)), text_mask);
    auto * injection = ggml_mul(ctx, ggml_repeat(ctx, value, x), gate);
    auto output = ggml_add(ctx, x, injection);
    return {production_precision ? bf16(output) : output, gate};
}

ggml_tensor * build_shared_expert(ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * gate_weight, ggml_tensor * up_weight, ggml_tensor * down_weight) {
    require(x && x->type == GGML_TYPE_F32 && ggml_n_dims(x) <= 2 &&
            gate_weight && up_weight && down_weight &&
            ggml_n_dims(gate_weight) == 2 && ggml_are_same_shape(gate_weight, up_weight) &&
            gate_weight->ne[0] == x->ne[0] && ggml_n_dims(down_weight) == 2 &&
            down_weight->ne[0] == gate_weight->ne[1] && down_weight->ne[1] == x->ne[0],
            "invalid V4.1 shared expert geometry");
    const auto bf16 = [&](ggml_tensor * t) {
        return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
    };
    const auto project = [&](ggml_tensor * weight, ggml_tensor * input) {
        auto result = ggml_mul_mat(ctx, weight, input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32);
        return bf16(result);
    };
    auto input = ggml_qat(ctx, ggml_cont(ctx, x), GGML_QAT_FP8_E8M0);
    auto gate = ggml_clamp(ctx, project(gate_weight, input), -INFINITY, 10.0f);
    auto up = ggml_clamp(ctx, project(up_weight, input), -10.0f, 10.0f);
    auto hidden = bf16(ggml_mul(ctx, ggml_silu(ctx, gate), up));
    hidden = ggml_qat(ctx, ggml_cont(ctx, hidden), GGML_QAT_FP8_E8M0);
    return project(down_weight, hidden);
}

compression_result build_compression(
        ggml_context * ctx, ggml_tensor * values, ggml_tensor * scores,
    ggml_tensor * norm_weight, int ratio, float norm_eps, bool bf16_pool) {
    require(values && values->type == GGML_TYPE_F32 && ggml_n_dims(values) <= 2 &&
            norm_weight && norm_weight->ne[0] == values->ne[0] && ggml_n_dims(norm_weight) == 1,
            "invalid V4.1 compressor geometry");
    require(ratio > 0 && norm_eps > 0 && std::isfinite(norm_eps), "invalid V4.1 compressor parameters");
    require(ratio == 1 || (scores && scores->type == GGML_TYPE_F32 && ggml_are_same_shape(values, scores)),
            "V4.1 pooled compression requires a score for each value");
    const int64_t dim = values->ne[0], tokens = values->ne[1];
    const int64_t groups = tokens / ratio, cutoff = groups * ratio, remainder = tokens - cutoff;
    compression_result result;
    if (remainder) {
        result.tail_values = ggml_view_2d(ctx, values, dim, remainder, values->nb[1], cutoff * values->nb[1]);
        result.tail_scores = ggml_view_2d(ctx, scores, dim, remainder, scores->nb[1], cutoff * scores->nb[1]);
    }
    if (groups == 0) return result;
    ggml_tensor * pooled = values;
    if (ratio > 1) {
        auto pack = [&](ggml_tensor * tensor) {
            auto * complete = ggml_cont(ctx, ggml_view_2d(ctx, tensor, dim, cutoff, tensor->nb[1], 0));
            auto * grouped = ggml_reshape_3d(ctx, complete, dim, ratio, groups);
            return ggml_cont(ctx, ggml_permute(ctx, grouped, 1, 0, 2, 3));
        };
        auto * gates = ggml_soft_max(ctx, pack(scores));
        pooled = ggml_reshape_2d(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, pack(values), gates)), dim, groups);
    }
    if (bf16_pool) pooled = ggml_cast(ctx, ggml_cast(ctx, pooled, GGML_TYPE_BF16), GGML_TYPE_F32);
    result.latent = ggml_mul(ctx, ggml_rms_norm(ctx, pooled, norm_eps), ggml_cast(ctx, norm_weight, GGML_TYPE_F32));
    return result;
}

ggml_tensor * build_rope_tail(ggml_context * ctx, ggml_tensor * x,
                             ggml_tensor * positions, const rope_config & config, bool inverse) {
    require(x && x->type == GGML_TYPE_F32 && x->ne[3] == 1 && positions &&
            positions->type == GGML_TYPE_I32 && ggml_nelements(positions) == x->ne[2], "invalid V4.1 RoPE inputs");
    const int64_t dim = x->ne[0], heads = x->ne[1], tokens = x->ne[2];
    const int rd = config.rotary_dimensions;
    require(rd > 0 && rd <= dim && rd % 2 == 0 && config.theta > 0 && std::isfinite(config.theta) &&
            config.original_context >= 0 && config.factor >= 1 && std::isfinite(config.factor) &&
            config.beta_fast > 0 && config.beta_slow > 0, "invalid V4.1 RoPE parameters");
    const int64_t plain = dim - rd;
    auto * tail = ggml_cont(ctx, ggml_view_3d(ctx, x, rd, heads, tokens, x->nb[1], x->nb[2], plain * sizeof(float)));
    const float scale = config.original_context ? 1.0f / config.factor : 1.0f;
    const float ext = config.original_context ? 1.0f : 0.0f;
    // GGML's YaRN adds a magnitude multiplier; the DeepSeek reference changes
    // frequencies only, so cancel that multiplier explicitly.
    const float magnitude = config.original_context ? 1.0f / (1.0f + 0.1f * std::log(config.factor)) : 1.0f;
    constexpr int adjacent_pair_mode = 0;
    auto * rotated = inverse ? ggml_rope_back(ctx, tail, positions, nullptr, rd, adjacent_pair_mode,
        config.original_context, config.theta, scale, ext, magnitude, config.beta_fast, config.beta_slow) :
        ggml_rope_ext(ctx, tail, positions, nullptr, rd, adjacent_pair_mode,
        config.original_context, config.theta, scale, ext, magnitude, config.beta_fast, config.beta_slow);
    if (!plain) return rotated;
    auto * prefix = ggml_cont(ctx, ggml_view_3d(ctx, x, plain, heads, tokens, x->nb[1], x->nb[2], 0));
    return ggml_concat(ctx, prefix, rotated, 0);
}

ggml_tensor * build_attention_core(ggml_context * ctx, ggml_tensor * q,
    ggml_tensor * kv, ggml_tensor * mask, ggml_tensor * sinks) {
    require(q && kv && mask && sinks && q->type == GGML_TYPE_F32 &&
            q->ne[3] == 1 && ggml_n_dims(kv) <= 2 && kv->ne[0] == q->ne[0] &&
            mask->ne[0] == kv->ne[1] && mask->ne[1] >= q->ne[2] &&
            mask->ne[1] % GGML_KQ_MASK_PAD == 0 && ggml_n_dims(mask) <= 2 &&
            ggml_nelements(sinks) == q->ne[1], "invalid V4.1 attention geometry");
    auto * query = ggml_permute(ctx, q, 0, 2, 1, 3);
    // Use the existing accelerator cache format for this initial integration.
    // The explicit F16 cast is included in the reference; packed QAT cache
    // decoding will replace this temporary representation.
    const int64_t padding = (256 - kv->ne[1] % 256) % 256;
    if (padding) {
        kv = ggml_pad(ctx, kv, 0, int(padding), 0, 0);
        auto * extra = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, padding, mask->ne[1]);
        mask = ggml_concat(ctx, mask, ggml_fill(ctx, extra, -INFINITY), 0);
    }
    auto * cache = ggml_cast(ctx, kv, GGML_TYPE_F16);
    auto * result = ggml_flash_attn_ext(ctx, query, cache, cache,
        ggml_cast(ctx, mask, GGML_TYPE_F16), 1.0f / std::sqrt(float(q->ne[0])), 0.0f, 0.0f);
    ggml_flash_attn_ext_set_prec(result, GGML_PREC_F32);
    ggml_flash_attn_ext_add_sinks(result, sinks);
    result->op_params[4] = GGML_FLASH_ATTN_EXT_IQK_DISABLED;
    return result;
}

ggml_tensor * build_attention_output(ggml_context * ctx, ggml_tensor * attended,
    ggml_tensor * wo_a, ggml_tensor * wo_b) {
    require(attended && wo_a && wo_b && attended->ne[3] == 1 &&
            wo_a->ne[3] == 1 && ggml_n_dims(wo_b) <= 2, "invalid V4.1 output projection rank");
    const int64_t width = attended->ne[0] * attended->ne[1];
    const int64_t tokens = attended->ne[2], groups = wo_a->ne[2], rank = wo_a->ne[1];
    require(groups > 0 && attended->ne[1] % groups == 0 && wo_a->ne[0] == width / groups &&
            wo_b->ne[0] == rank * groups, "invalid V4.1 grouped output geometry");
    auto * grouped = ggml_reshape_3d(ctx, ggml_cont(ctx, attended), width / groups, groups, tokens);
    grouped = ggml_cont(ctx, ggml_permute(ctx, grouped, 0, 2, 1, 3));
    auto * projected = ggml_mul_mat(ctx, wo_a, grouped);
    ggml_mul_mat_set_prec(projected, GGML_PREC_F32);
    projected = ggml_cont(ctx, ggml_permute(ctx, projected, 0, 2, 1, 3));
    auto * result = ggml_mul_mat(ctx, wo_b, ggml_reshape_2d(ctx, projected, rank * groups, tokens));
    ggml_mul_mat_set_prec(result, GGML_PREC_F32);
    return result;
}

ggml_tensor * build_indexer_scores(ggml_context * ctx, ggml_tensor * q,
    ggml_tensor * k, ggml_tensor * weights, bool bf16_intermediates) {
    require(q && k && weights && q->ne[3] == 1 && ggml_n_dims(k) <= 2 &&
            q->ne[0] == k->ne[0] && weights->ne[0] == q->ne[1] &&
            weights->ne[1] == q->ne[2] && ggml_n_dims(weights) <= 2,
            "invalid V4.1 indexer score geometry");
    auto * dots = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(dots, GGML_PREC_F32);
    auto round = [&](ggml_tensor * value) {
        return bf16_intermediates ? ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_BF16), GGML_TYPE_F32) : value;
    };
    dots = round(dots);
    auto * weighted = ggml_mul(ctx, ggml_relu(ctx, dots),
        ggml_reshape_3d(ctx, weights, 1, q->ne[1], q->ne[2]));
    auto * heads_first = ggml_cont(ctx, ggml_permute(ctx, round(weighted), 1, 0, 2, 3));
    return round(ggml_reshape_2d(ctx, ggml_sum_rows(ctx, heads_first), k->ne[1], q->ne[2]));
}

} // namespace deepseek_v41
