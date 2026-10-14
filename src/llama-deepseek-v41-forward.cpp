#include "llama-deepseek-v41-forward.h"
#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-model.h"
#include "llama-deepseek-v41-index.h"
#include "llama-deepseek-v41-ops.h"
#include "../ggml/src/ggml-qat.h"
#include "../ggml/src/ggml-expert-loader.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

namespace deepseek_v41 {
static void require(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

attention_forward_state::attention_forward_state(model_config config, uint32_t capacity,
                                                 ggml_backend_buffer_type_t cache_type)
    : config_(std::move(config)), layout_(config_.attention), cache_(config_.attention, capacity, cache_type), capacity_(capacity) {
    full_indexer_ = std::getenv("ESE_V41_FULL_INDEXER") != nullptr;
    if (const char * mode = std::getenv("ESE_V41_EARLY_ROUTER")) {
        require(std::strcmp(mode, "off") == 0 || std::strcmp(mode, "probe") == 0 || std::strcmp(mode, "advice") == 0,
                "invalid V4.1 early router mode");
        early_router_ = std::strcmp(mode, "off") != 0;
    }
    if (const char * confidence = std::getenv("ESE_V41_PREFETCH_CONFIDENCE")) {
        char * end = nullptr;
        prefetch_confidence_ = std::strtof(confidence, &end);
        require(end != confidence && !*end && std::isfinite(prefetch_confidence_) &&
                prefetch_confidence_ >= 0.0f && prefetch_confidence_ <= 1.0f, "invalid V4.1 prefetch confidence");
    }
    const size_t n = layout_.layers().size();
    values_.resize(n); scores_.resize(n); selected_.resize(n * max_batch); callbacks_.resize(n * max_batch);
    candidates_.resize(max_batch);
    for (uint32_t layer = 0; layer < n; ++layer)
        for (uint32_t token = 0; token < max_batch; ++token)
            for (size_t op = 0; op < callbacks_[layer * max_batch + token].size(); ++op)
                callbacks_[layer * max_batch + token][op] = {this, layer, operation(op), token};
}
bool attention_forward_state::all_index_rows_visible(uint32_t layer, uint32_t position) const {
    const auto & info = layout_.layers()[layer];
    if (full_indexer_ || !info.compress_ratio) return false;
    const uint64_t count = (uint64_t(position) + 1) / info.compress_ratio;
    // Selection is returned in chronological order. When both limits cover
    // the whole visible prefix, scores cannot remove or reorder a valid row.
    // Candidate consumers share their owner's KV bank (checked by the layout).
    return count <= config_.index_topk && (info.candidate_owner < 0 ||
        count <= uint64_t(config_.candidate_blocks) * config_.candidate_block_size);
}
void attention_forward_state::clear() {
    require(!active_, "cannot clear active V4.1 forward state");
    cache_.clear();
    checkpoint_values_.clear(); checkpoint_scores_.clear();
    checkpoint_value_steps_.clear(); checkpoint_score_steps_.clear();
    for (auto * groups : {&values_, &scores_, &saved_values_, &saved_scores_}) for (auto & group : *groups) group.clear();
    for (auto & group : selected_) group.clear();
    for (auto & candidate : candidates_) candidate.clear();
    error_.clear(); position_ = 0; batch_tokens_ = 1; build_token_index_ = 0;
}

void attention_forward_state::save_checkpoint(uint32_t max_tokens) {
    require(!active_ && max_tokens > 0 && max_tokens <= max_batch, "invalid V4.1 forward checkpoint");
    auto values = values_, scores = scores_;
    std::vector<std::vector<std::vector<float>>> value_steps(values_.size()), score_steps(scores_.size());
    for (size_t layer = 0; layer < values_.size(); ++layer) {
        value_steps[layer].resize(max_tokens); score_steps[layer].resize(max_tokens);
        if (layout_.layers()[layer].kv_owner == int32_t(layer) && layout_.layers()[layer].compress_ratio == 2)
            for (uint32_t token = 0; token < max_tokens; ++token) {
                value_steps[layer][token].reserve(config_.attention.head_dim);
                score_steps[layer][token].reserve(config_.attention.head_dim);
            }
    }
    cache_.save_checkpoint(max_tokens);
    checkpoint_values_ = std::move(values); checkpoint_scores_ = std::move(scores);
    checkpoint_value_steps_ = std::move(value_steps); checkpoint_score_steps_ = std::move(score_steps);
}
void attention_forward_state::discard_checkpoint() {
    require(!active_, "cannot discard active V4.1 forward checkpoint");
    cache_.discard_checkpoint();
    checkpoint_values_.clear(); checkpoint_scores_.clear();
    checkpoint_value_steps_.clear(); checkpoint_score_steps_.clear();
}
void attention_forward_state::restore_checkpoint(uint32_t position) {
    require(!active_ && cache_.has_checkpoint() && position >= cache_.checkpoint_position() &&
            position <= cache_.window_end(0), "invalid V4.1 forward checkpoint restore");
    const uint32_t accepted = position - cache_.checkpoint_position();
    require(!checkpoint_value_steps_.empty() && accepted <= checkpoint_value_steps_[0].size(), "V4.1 checkpoint prefix exceeds journal");
    auto values = checkpoint_values_, scores = checkpoint_scores_;
    if (accepted) for (size_t layer = 0; layer < values.size(); ++layer)
        if (layout_.layers()[layer].kv_owner == int32_t(layer) && layout_.layers()[layer].compress_ratio == 2) {
            values[layer] = checkpoint_value_steps_[layer][accepted - 1];
            scores[layer] = checkpoint_score_steps_[layer][accepted - 1];
        }
    cache_.restore_checkpoint(position);
    values_.swap(values); scores_.swap(scores);
    checkpoint_values_.clear(); checkpoint_scores_.clear();
    checkpoint_value_steps_.clear(); checkpoint_score_steps_.clear();
    for (auto & selected : selected_) selected.clear();
    for (auto & candidate : candidates_) candidate.clear();
    error_.clear(); position_ = position; batch_tokens_ = 1;
}

void attention_forward_state::begin(uint32_t position, uint32_t tokens) {
    require(!active_ && position == cache_.window_end(0), "nonsequential V4.1 attention forward");
    require(tokens > 0 && tokens <= max_batch && position <= capacity_ && tokens <= capacity_ - position,
            "invalid V4.1 forward batch extent");
    auto values = values_, scores = scores_;
    cache_.begin(); saved_values_ = std::move(values); saved_scores_ = std::move(scores);
    for (auto & item : selected_) item.clear();
    for (auto & candidate : candidates_) candidate.clear();
    error_.clear(); position_ = position; batch_tokens_ = tokens; active_ = true;
}
void attention_forward_state::commit() {
    require(active_, "no V4.1 forward transaction");
    if (!error_.empty()) throw std::runtime_error(error_);
    require(cache_.window_end(0) == position_ + batch_tokens_, "V4.1 graph did not publish its batch extent");
    cache_.commit(); saved_values_.clear(); saved_scores_.clear(); active_ = false;
}
void attention_forward_state::abort() {
    require(active_, "no V4.1 forward transaction");
    cache_.abort(); values_.swap(saved_values_); scores_.swap(saved_scores_);
    saved_values_.clear(); saved_scores_.clear(); error_.clear(); active_ = false;
}

ggml_tensor * attention_forward_state::node(ggml_context * ctx, uint32_t layer, operation op,
    std::initializer_list<int64_t> shape, ggml_tensor * a, ggml_tensor * b, ggml_tensor * c, ggml_type type) {
    if (!b) b = a;
    if (!c) c = a;
    auto * pattern = ggml_map_custom3(ctx, a, b, c, callback, 1, &callbacks_.at(layer * max_batch + build_token_index_).at(size_t(op)));
    auto * result = ggml_new_tensor(ctx, type, shape.size(), shape.begin());
    result->op = pattern->op;
    std::memcpy(result->op_params, pattern->op_params, sizeof(result->op_params));
    result->src[0] = a; result->src[1] = b; result->src[2] = c;
    ggml_set_name(result, ("v41_state_" + std::to_string(layer) + "_" + std::to_string(size_t(op)) + "_" + std::to_string(build_token_index_)).c_str());
    return result;
}
void attention_forward_state::callback(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
    const ggml_tensor * c, int ith, int nth, void * userdata) {
    (void) nth;
    if (ith) return;
    auto & data = *static_cast<callback_data *>(userdata);
    try {
        require(data.state->active_, "V4.1 state node executed outside a transaction");
        if (!data.state->error_.empty()) throw std::runtime_error(data.state->error_);
        data.state->execute(data, dst, a, b, c);
    } catch (const std::exception & error) {
        // Never throw across the GGML worker boundary. The caller checks the
        // recorded failure before committing and aborts the whole forward.
        data.state->error_ = error.what();
        if (dst->type == GGML_TYPE_I32) std::fill_n(static_cast<int32_t *>(dst->data), ggml_nelements(dst), 0);
        else std::fill_n(static_cast<float *>(dst->data), ggml_nelements(dst), 0.0f);
    }
}
void attention_forward_state::execute(const callback_data & data, ggml_tensor * dst,
    const ggml_tensor * a, const ggml_tensor * b, const ggml_tensor * c) {
    const uint32_t layer = data.layer, dim = config_.attention.head_dim;
    const auto & info = layout_.layers()[layer];
    require(data.token < batch_tokens_, "V4.1 state node is outside the active batch");
    const uint32_t position = position_ + data.token;
    const size_t selection_slot = size_t(layer) * max_batch + data.token;
    auto vector = [](const ggml_tensor * tensor, size_t count) {
        require(tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && ggml_nelements(tensor) >= int64_t(count),
                "invalid V4.1 state input");
        const auto * values = static_cast<const float *>(tensor->data);
        return std::vector<float>(values, values + count);
    };
    auto * output = static_cast<float *>(dst->data);
    if (data.op == operation::position || data.op == operation::compressed_position) {
        require(a->type == GGML_TYPE_I32 && ggml_nelements(a) == 1 &&
                *static_cast<const int32_t *>(a->data) == int32_t(position), "V4.1 rotary/cache positions disagree");
        *static_cast<int32_t *>(dst->data) = data.op == operation::position ? int32_t(position) : int32_t(position + 1 - info.compress_ratio);
        return;
    }
    std::fill_n(output, ggml_nelements(dst), 0.0f);
    switch (data.op) {
        case operation::position: break; // handled above with its integer output type
        case operation::compressed_position: break;
        case operation::prefetch: {
            // A forecast is only I/O advice; failed or non-finite forecasts
            // never replace the actual router or change its expert weights.
            if (!advice_context_ || !advice_context_->sched || layer == 0) break;
            auto scores = vector(a, config_.experts);
            if (std::any_of(scores.begin(), scores.end(), [](float value) { return !std::isfinite(value); })) break;
            std::vector<int32_t> ids(config_.experts);
            std::iota(ids.begin(), ids.end(), 0);
            std::partial_sort(ids.begin(), ids.begin() + config_.active_experts, ids.end(), [&](int32_t left, int32_t right) {
                return scores[left] != scores[right] ? scores[left] > scores[right] : left < right;
            });
            ids.resize(config_.active_experts);
            std::vector<float> selected;
            for (int32_t expert : ids) selected.push_back(scores[expert]);
            auto & context = *advice_context_;
            ggml_backend_sched_trace_expert_prediction(context.sched, int(layer), ids.data(), selected.data(), ids.size());
            auto & queue = context.model.expert_read_ahead;
            if (queue) {
                queue->advance(reinterpret_cast<uintptr_t>(context.sched),
                        ggml_backend_sched_expert_pass(context.sched), layer - 1);
                if (ggml_backend_sched_expert_prediction_precision(context.sched, int(layer)) < prefetch_confidence_) break;
                for (int32_t expert : ids) {
                    const uint32_t ready = ggml_backend_sched_expert_ready_mask(context.sched, int(layer), expert);
                    for (uint32_t part = 0; part < 3; ++part) {
                        if (ready & (1u << part)) continue;
                        const auto found = context.model.expert_index.descriptors.find({layer, uint32_t(expert),
                                llama_expert_cache::component(part)});
                        if (found != context.model.expert_index.descriptors.end()) queue->submit(found->second);
                    }
                }
            }
        } break;
        case operation::pool: {
            auto values = vector(a, dim), scores = vector(b, dim);
            if (position % 2 == 0) {
                values_[layer] = std::move(values); scores_[layer] = std::move(scores);
            } else {
                require(values_[layer].size() == dim && scores_[layer].size() == dim, "missing V4.1 partial compression state");
                std::copy(values_[layer].begin(), values_[layer].end(), output);
                std::copy(values.begin(), values.end(), output + dim);
                std::copy(scores_[layer].begin(), scores_[layer].end(), output + 2 * dim);
                std::copy(scores.begin(), scores.end(), output + 3 * dim);
                values_[layer].clear(); scores_[layer].clear();
            }
            if (cache_.has_checkpoint()) {
                const size_t step = position - cache_.checkpoint_position();
                checkpoint_value_steps_.at(layer).at(step) = values_[layer];
                checkpoint_score_steps_.at(layer).at(step) = scores_[layer];
            }
        } break;
        case operation::publish: {
            cache_.append_window(layer, position, vector(a, dim));
            if (info.kv_owner == int32_t(layer) && (position + 1) % info.compress_ratio == 0)
                cache_.append_global(layer, position / info.compress_ratio, vector(b, dim), vector(c, config_.attention.index_head_dim));
        } break;
        case operation::keys: {
            const uint32_t count = (position + 1) / info.compress_ratio;
            require(cache_.global_rows(layer) >= count, "missing V4.1 shared cache prefix");
            std::vector<uint32_t> rows(count);
            std::iota(rows.begin(), rows.end(), 0);
            auto keys = cache_.read_global(layer, true, rows);
            require(keys.size() == size_t(ggml_nelements(dst)), "V4.1 index cache extent changed during execution");
            std::copy(keys.begin(), keys.end(), output);
        } break;
        case operation::select: {
            const uint32_t count = (position + 1) / info.compress_ratio;
            require(cache_.global_rows(layer) >= count, "missing V4.1 selection cache prefix");
            auto scores = vector(a, count);
            if (info.candidate_owner == int32_t(layer))
                candidates_[data.token] = select_candidate_blocks(scores, count, config_.candidate_block_size, config_.candidate_blocks);
            const bool filtered = info.candidate_owner >= 0 && info.candidate_owner < int32_t(layer);
            if (filtered) require(candidates_[data.token].size() == count, "missing V4.1 candidate publication");
            selected_[selection_slot] = select_index_positions(scores, count, config_.index_topk, 0,
                                                       filtered ? candidates_[data.token] : std::vector<uint8_t>{});
        } break;
        case operation::gather: {
            const size_t rows = ggml_nelements(dst) / (dim + 1);
            float * mask = output + rows * dim;
            std::fill_n(mask, rows, -INFINITY);
            require(cache_.window_end(layer) == position + 1, "V4.1 window publication crossed a causal boundary");
            auto window = cache_.read_window(layer);
            std::copy(window.begin(), window.end(), output);
            size_t count = window.size() / dim;
            if (info.topk_owner >= 0) {
                std::vector<uint32_t> selected;
                if (all_index_rows_visible(layer, position)) {
                    selected.resize((position + 1) / info.compress_ratio);
                    std::iota(selected.begin(), selected.end(), 0);
                } else {
                    for (auto row : selected_.at(size_t(info.topk_owner) * max_batch + data.token)) if (row >= 0) {
                        require(uint32_t(row) < (position + 1) / info.compress_ratio, "V4.1 selection contains a future row");
                        selected.push_back(uint32_t(row));
                    }
                }
                auto global = cache_.read_global(layer, false, selected);
                require(window.size() + global.size() <= rows * dim, "V4.1 gather output too small");
                std::copy(global.begin(), global.end(), output + window.size()); count += selected.size();
            }
            std::fill_n(mask, count, 0.0f);
        } break;
    }
}

ggml_tensor * attention_forward_state::build(ggml_context * ctx, const std::map<std::string, ggml_tensor *> & weights,
    uint32_t layer, ggml_tensor * x, ggml_tensor * positions, bool production_precision, ggml_tensor * after) {
    require(layer < layout_.layers().size() && x->ne[0] == config_.hidden && x->ne[1] == 1 &&
            ggml_n_dims(x) <= 2 && ggml_nelements(positions) == 1, "V4.1 attention forward expects one token");
    const auto & info = layout_.layers()[layer];
    const int64_t dim = config_.attention.head_dim, heads = config_.heads;
    positions = node(ctx, layer, operation::position, {1}, positions, after, nullptr, GGML_TYPE_I32);
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    auto weight = [&](const char * name) { return weights.at(prefix + name); };
    auto activation = [&](ggml_tensor * value) {
        return production_precision ? ggml_cast(ctx, ggml_cast(ctx, value, GGML_TYPE_BF16), GGML_TYPE_F32) : value;
    };
    auto linear = [&](ggml_tensor * input, const char * name, bool original_quantized = true, bool round_output = true) {
        if (production_precision && original_quantized) input = ggml_qat(ctx, ggml_cont(ctx, input), GGML_QAT_FP8_E8M0);
        auto * result = ggml_mul_mat(ctx, weight(name), input);
        ggml_mul_mat_set_prec(result, GGML_PREC_F32); return round_output ? activation(result) : result;
    };
    auto norm = [&](ggml_tensor * input, const char * name) {
        return activation(ggml_mul(ctx, ggml_rms_norm(ctx, input, config_.norm_eps), ggml_cast(ctx, weight(name), GGML_TYPE_F32)));
    };
    rope_config rope{int(config_.rotary), info.compress_ratio ? int(config_.original_context) : 0,
        info.compress_ratio ? config_.compressed_rope_theta : config_.rope_theta,
        config_.rope_factor, config_.beta_fast, config_.beta_slow};
    auto rotate = [&](ggml_tensor * value, ggml_tensor * pos, bool inverse = false) {
        return activation(build_rope_tail(ctx, value, pos, rope, inverse));
    };
    auto * qr = norm(linear(x, "attn_q_a"), "attn_q_a_norm");
    auto * q = rotate(ggml_reshape_3d(ctx, linear(qr, "attn_q_b"), dim, heads, 1), positions);
    auto * window = rotate(ggml_reshape_3d(ctx, norm(linear(x, "attn_kv"), "attn_kv_a_norm"), dim, 1, 1), positions);
    ggml_tensor * main = window, * index = window;
    if (info.kv_owner == int32_t(layer)) {
        auto * values = linear(x, "attn_compressor_kv", false, info.compress_ratio == 1);
        const bool complete = (position_ + 1) % info.compress_ratio == 0;
        if (info.compress_ratio == 2) {
            auto * packed = node(ctx, layer, operation::pool, {dim, 2, 2}, values, linear(x, "attn_compressor_gate", false, false), positions);
            main = index = packed; // preserve incomplete-group state as an execution dependency
            if (complete) {
                auto * v = ggml_view_2d(ctx, packed, dim, 2, dim * sizeof(float), 0);
                auto * s = ggml_view_2d(ctx, packed, dim, 2, dim * sizeof(float), 2 * dim * sizeof(float));
                values = activation(build_compression(ctx, v, s, weight("attn_compressor_norm"), 2, config_.norm_eps, production_precision).latent);
            }
        } else values = norm(values, "attn_compressor_norm");
        if (complete) {
            // A compressed group takes its first token's rotary position.
            auto * compressed_pos = node(ctx, layer, operation::compressed_position, {1}, positions, nullptr, nullptr, GGML_TYPE_I32);
            index = norm(linear(values, "indexer.attn_k", false), "indexer.k_norm");
            index = rotate(ggml_reshape_3d(ctx, index, config_.attention.index_head_dim, 1, 1), compressed_pos);
            main = rotate(ggml_reshape_3d(ctx, values, dim, 1, 1), compressed_pos);
        }
    }
    auto * published = node(ctx, layer, operation::publish, {1}, ggml_cont(ctx, window), ggml_cont(ctx, main), ggml_cont(ctx, index));
    ggml_tensor * selection = published;
    const uint32_t global_count = info.compress_ratio ? (position_ + 1) / info.compress_ratio : 0;
    if (info.topk_owner == int32_t(layer) && global_count && !all_index_rows_visible(layer, position_)) {
        auto * keys = node(ctx, layer, operation::keys, {config_.attention.index_head_dim, global_count}, published);
        auto * index_q = rotate(ggml_reshape_3d(ctx, linear(qr, "indexer.attn_q_b"), config_.attention.index_head_dim,
                                              config_.index_heads, 1), positions);
        index_q = ggml_qat(ctx, ggml_cont(ctx, index_q), GGML_QAT_FP4_E8M0);
        auto * head_weights = activation(ggml_scale(ctx, linear(x, "indexer.proj", false),
            1.0f / std::sqrt(float(config_.attention.index_head_dim) * config_.index_heads)));
        selection = node(ctx, layer, operation::select, {1}, build_indexer_scores(ctx, index_q, keys, head_weights, production_precision), published);
    }
    const int64_t rows = std::min(position_ + 1, config_.attention.sliding_window) + std::min(global_count, config_.index_topk);
    auto * gathered = node(ctx, layer, operation::gather, {(dim + 1) * rows}, published, selection);
    auto * kv = ggml_view_2d(ctx, gathered, dim, rows, dim * sizeof(float), 0);
    auto * mask = ggml_view_2d(ctx, gathered, rows, 1, rows * sizeof(float), dim * rows * sizeof(float));
    mask = ggml_pad(ctx, mask, 0, GGML_KQ_MASK_PAD - 1, 0, 0);
    auto * attended = activation(build_attention_core(ctx, q, kv, mask, weight("attn_sinks")));
    attended = rotate(attended, positions, true);
    auto * wo_a = ggml_reshape_3d(ctx, weight("attn_output_a"), dim * heads / config_.output_groups,
                                 config_.output_rank, config_.output_groups);
    // Separate the two output GEMMs to preserve the BF16 and FP8-input boundary.
    auto * grouped = ggml_reshape_3d(ctx, ggml_cont(ctx, attended), dim * heads / config_.output_groups, config_.output_groups, 1);
    grouped = ggml_cont(ctx, ggml_permute(ctx, grouped, 0, 2, 1, 3));
    auto * low_rank = ggml_mul_mat(ctx, wo_a, grouped); ggml_mul_mat_set_prec(low_rank, GGML_PREC_F32);
    low_rank = activation(ggml_cont(ctx, ggml_permute(ctx, low_rank, 0, 2, 1, 3)));
    return linear(ggml_reshape_2d(ctx, low_rank, config_.output_rank * config_.output_groups, 1), "attn_output_b");
}
block_output attention_forward_state::build_block(ggml_context * ctx, llama_context & lctx,
        const std::map<std::string, ggml_tensor *> & weights, uint32_t layer,
        ggml_tensor * x, ggml_tensor * incoming_pre, ggml_tensor * positions,
        ggml_cgraph * graph, ggml_tensor * engram_rows, ggml_tensor * text_mask) {
    require(config_.shared_experts == 1 && config_.swiglu_limit == 10.0f,
            "unsupported V4.1 expert configuration");
    require(lctx.model.arch == LLM_ARCH_DEEPSEEK41 && layer < layout_.layers().size() &&
            x && x->type == GGML_TYPE_F32 && x->ne[0] == config_.hidden &&
            x->ne[1] == config_.streams && x->ne[2] >= 1 && x->ne[2] <= max_batch && x->ne[3] == 1 &&
            ggml_nelements(positions) == x->ne[2], "V4.1 block expects a bounded residual-stream batch");
    const std::string prefix = "blk." + std::to_string(layer) + ".";
    const auto weight = [&](const std::string & name) { return weights.at(prefix + name); };
    const auto bf16 = [&](ggml_tensor * t) {
        return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
    };
    const auto norm = [&](ggml_tensor * t, const char * name) {
        return bf16(ggml_mul(ctx, ggml_rms_norm(ctx, t, config_.norm_eps), ggml_cast(ctx, weight(name), GGML_TYPE_F32)));
    };
    const auto mixes = [&](ggml_tensor * t, const std::string & sublayer) {
        return build_hc_mixes(ctx, t, weight("hc_" + sublayer + "_fn"),
                weight("hc_" + sublayer + "_scale"), weight("hc_" + sublayer + "_base"),
                config_.norm_eps, config_.sinkhorn_iterations, config_.hc_eps);
    };
    const bool has_engram = std::find(config_.engram_layers.begin(), config_.engram_layers.end(), layer) != config_.engram_layers.end();
    require(has_engram == bool(engram_rows), "V4.1 block Engram rows do not match configured layer");
    if (has_engram) {
        require(text_mask, "V4.1 Engram requires a text mask");
        x = build_engram_injection(ctx, x, engram_rows, weight("engram_wkv"),
                weight("engram_q_norm"), weight("engram_k_norm"), text_mask, config_.norm_eps, true).output;
    }
    auto attn_mix = mixes(x, "attn");
    auto attn_input = norm(bf16(build_hc_collapse(ctx, x, incoming_pre)), "attn_norm");
    auto router_weight = ggml_cast(ctx, weight("ffn_gate_inp"), GGML_TYPE_F32);
    const auto pin_router = [&](ggml_tensor * tensor) {
        if (!lctx.sched || !weight("ffn_gate_inp")->buffer) return;
        auto * buft = ggml_backend_buffer_get_type(weight("ffn_gate_inp")->buffer);
        for (auto * backend : lctx.backends) {
            if ((ggml_backend_get_default_buffer_type(backend) == buft ||
                 (ggml_backend_buffer_is_host(weight("ffn_gate_inp")->buffer) && backend == lctx.backend_cpu)) &&
                    ggml_backend_supports_op(backend, tensor)) {
                ggml_backend_sched_set_tensor_backend(lctx.sched, tensor, backend);
                break;
            }
        }
    };
    if (early_router_ && layer > 0 && x->ne[2] == 1) {
        advice_context_ = &lctx;
        // Keep the current hyper-connection residual mixing and omit only the
        // not-yet-computed attention contribution. This predicts this layer's
        // FFN route before attention executes, without using future tokens.
        auto zero = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config_.hidden, 1), 0.0f);
        auto residual = bf16(build_hc_expand(ctx, zero, x, attn_mix));
        auto approximate = norm(bf16(build_hc_collapse(ctx, residual, attn_mix.pre)), "ffn_norm");
        auto logits = ggml_mul_mat(ctx, router_weight, approximate);
        ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
        pin_router(router_weight); pin_router(logits);
        auto scores = ggml_add(ctx, ggml_sqrt_softplus(ctx, logits), weight("exp_probs_b"));
        ggml_build_forward_expand(graph, node(ctx, layer, operation::prefetch, {1}, ggml_cont(ctx, scores)));
    }
    ggml_tensor * attended = nullptr;
    {
        struct restore_build {
            attention_forward_state & state; uint32_t position, token;
            ~restore_build() { state.position_ = position; state.build_token_index_ = token; }
        } restore{*this, position_, build_token_index_};
        ggml_tensor * preceding = nullptr;
        for (uint32_t token = 0; token < uint32_t(x->ne[2]); ++token) {
            position_ = restore.position + token; build_token_index_ = token;
            auto input = ggml_view_2d(ctx, attn_input, config_.hidden, 1, attn_input->nb[1], token * attn_input->nb[1]);
            auto pos = ggml_view_1d(ctx, positions, 1, token * sizeof(int32_t));
            auto output = build(ctx, weights, layer, input, pos, true, preceding);
            attended = attended ? ggml_concat(ctx, attended, output, 1) : output;
            preceding = output;
        }
    }
    x = bf16(build_hc_expand(ctx, attended, x, attn_mix));
    auto ffn_mix = mixes(x, "ffn");
    auto ffn_input = norm(bf16(build_hc_collapse(ctx, x, attn_mix.pre)), "ffn_norm");
    ggml_backend_t expert_backend = nullptr;
    if (const char * requested = std::getenv("ESE_V41_EXPERT_DEVICE")) {
        uint32_t count = layout_.layers().size();
        if (const char * limit = std::getenv("ESE_V41_EXPERT_LAYER_COUNT")) {
            char * end = nullptr;
            const long parsed = std::strtol(limit, &end, 10);
            require(end && !*end && parsed >= 0 && parsed <= long(count), "invalid V4.1 expert layer count");
            count = uint32_t(parsed);
        }
        if (lctx.sched && layer < count) {
            for (auto * backend : lctx.backends)
                if (std::strcmp(ggml_backend_name(backend), requested) == 0) expert_backend = backend;
            require(expert_backend, "requested V4.1 expert backend is unavailable");
        }
    }
    llm_build_cb cb = [&](ggml_tensor * t, const char * name, int il) {
        ggml_format_name(t, "%s-%d", name, il);
        if (lctx.sched && std::strcmp(name, "ffn_moe_logits") == 0 && weight("ffn_gate_inp")->buffer) {
            // The F32 cast of a BF16 router is a compute tensor. Preserve the
            // router weight's device when surrounding activations live on a
            // different GPU; otherwise scheduling can move this sensitive dot
            // product and change BF16/QAT routing through round-off.
            pin_router(t);
        }
        if (expert_backend && t->op == GGML_OP_MUL_MAT_ID) {
            require(ggml_backend_supports_op(expert_backend, t), "requested V4.1 expert backend cannot execute routing");
            ggml_backend_sched_set_tensor_backend(lctx.sched, t, expert_backend);
        }
    };
    auto routed = llm_build_context::llm_build_moe_ffn(ctx, lctx, ffn_input,
            router_weight, weight("ffn_up_exps"),
            weight("ffn_gate_exps"), weight("ffn_down_exps"), weight("exp_probs_b"),
            config_.experts, config_.active_experts, LLM_FFN_SILU, config_.active_experts > 1,
            true, config_.expert_scale, LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS, cb, layer, graph);
    auto shared = build_shared_expert(ctx, ffn_input, weight("ffn_gate_shexp"),
            weight("ffn_up_shexp"), weight("ffn_down_shexp"));
    auto ffn_output = bf16(ggml_add(ctx, routed, shared));
    return {bf16(build_hc_expand(ctx, ffn_output, x, ffn_mix)), ffn_mix.pre};
}

ggml_tensor * attention_forward_state::build_chunk(ggml_context * ctx, llama_context & lctx,
        const std::map<std::string, ggml_tensor *> & weights, ggml_tensor * token,
        ggml_tensor * position, const std::map<uint32_t, ggml_tensor *> & engram_rows,
        ggml_cgraph * graph, uint32_t reserve_position, ggml_tensor * output_ids) {
    struct restore_position { uint32_t & value; uint32_t previous; ~restore_position() { value = previous; } } restore{position_, position_};
    if (reserve_position != UINT32_MAX) {
        require(!active_ && reserve_position < config_.context, "invalid V4.1 reserve graph position");
        position_ = reserve_position;
    }
    require(token && token->type == GGML_TYPE_I32 && ggml_nelements(token) >= 1 && ggml_nelements(token) <= max_batch &&
            ggml_nelements(position) == ggml_nelements(token) && graph &&
            engram_rows.size() == config_.engram_layers.size(), "invalid V4.1 token graph inputs");
    const int64_t count = ggml_nelements(token);
    require(position_ < capacity_ && count <= capacity_ - position_, "V4.1 graph batch exceeds capacity");
    const auto bf16 = [&](ggml_tensor * t) {
        return ggml_cast(ctx, ggml_cast(ctx, t, GGML_TYPE_BF16), GGML_TYPE_F32);
    };
    auto embedding = bf16(ggml_get_rows(ctx, weights.at("token_embd"), token));
    auto x = ggml_repeat(ctx, ggml_reshape_3d(ctx, embedding, config_.hidden, 1, count),
            ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config_.hidden, config_.streams, count));
    auto pre = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, count), 1.0f);
    if (config_.streams > 1) pre = ggml_concat(ctx, pre,
            ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config_.streams - 1, count), 0.0f), 0);
    auto mask = ggml_fill(ctx, ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 1, count), 1.0f);
    for (uint32_t layer = 0; layer < layout_.layers().size(); ++layer) {
        if (lctx.dflash.capture && std::find(lctx.dflash.capture->layer_ids.begin(),
                lctx.dflash.capture->layer_ids.end(), int32_t(layer)) != lctx.dflash.capture->layer_ids.end()) {
            // DSpark consumes the mean of the residual streams at the input
            // of its target layers. Its released targets have no Engram.
            require(std::find(config_.engram_layers.begin(), config_.engram_layers.end(), layer) == config_.engram_layers.end(),
                    "DSpark feature capture on an Engram layer is unsupported");
            auto streams_first = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));
            auto mean = ggml_scale(ctx, ggml_sum_rows(ctx, streams_first), 1.0f / config_.streams);
            mean = bf16(ggml_reshape_2d(ctx, mean, config_.hidden, count));
            ggml_format_name(mean, "dspark_input-%u", layer);
            ggml_set_output(mean);
            ggml_build_forward_expand(graph, mean);
        }
        auto rows = engram_rows.find(layer);
        auto block = build_block(ctx, lctx, weights, layer, x, pre, position, graph,
                rows == engram_rows.end() ? nullptr : rows->second, mask);
        x = block.streams; pre = block.next_pre;
    }
    auto hidden = bf16(build_hc_collapse(ctx, x, pre));
    hidden = bf16(ggml_mul(ctx, ggml_rms_norm(ctx, hidden, config_.norm_eps),
            ggml_cast(ctx, weights.at("output_norm"), GGML_TYPE_F32)));
    if (output_ids) hidden = ggml_get_rows(ctx, hidden, output_ids);
    auto * output_weight = weights.at("output");
    auto * output_f32 = std::getenv("ESE_V41_DIRECT_BF16_HEAD") && output_weight->type == GGML_TYPE_BF16 ?
        output_weight : ggml_cast(ctx, output_weight, GGML_TYPE_F32);
    auto logits = ggml_mul_mat(ctx, output_f32, hidden);
    ggml_mul_mat_set_prec(logits, GGML_PREC_F32);
    // The cast is no longer a weight tensor from the scheduler's perspective.
    // Preserve the head's placement explicitly: otherwise the large F32 copy
    // can migrate to the last transformer device (or off CPU), defeating an
    // output override and requiring several extra GiB on that device.
    if (lctx.sched && output_weight->buffer) {
        auto * buft = ggml_backend_buffer_get_type(output_weight->buffer);
        for (auto * backend : lctx.backends) {
            const bool holds_weight = ggml_backend_get_default_buffer_type(backend) == buft ||
                    (ggml_backend_buffer_is_host(output_weight->buffer) && backend == lctx.backend_cpu);
            if (holds_weight &&
                    ggml_backend_supports_op(backend, output_f32) && ggml_backend_supports_op(backend, logits)) {
                ggml_backend_sched_set_tensor_backend(lctx.sched, output_f32, backend);
                ggml_backend_sched_set_tensor_backend(lctx.sched, logits, backend);
                break;
            }
        }
    }
    ggml_set_name(logits, "result_output");
    ggml_build_forward_expand(graph, logits);
    return logits;
}

ggml_tensor * attention_forward_state::build_token(ggml_context * ctx, llama_context & lctx,
        const std::map<std::string, ggml_tensor *> & weights, ggml_tensor * token,
        ggml_tensor * position, const std::map<uint32_t, ggml_tensor *> & engram_rows,
        ggml_cgraph * graph, uint32_t reserve_position) {
    require(token && ggml_nelements(token) == 1, "V4.1 single-token graph received a batch");
    return build_chunk(ctx, lctx, weights, token, position, engram_rows, graph, reserve_position);
}

} // namespace deepseek_v41
