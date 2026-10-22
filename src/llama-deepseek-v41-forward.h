#pragma once
#include "llama-deepseek-v41-cache.h"
#include "llama-deepseek-v41-model.h"
#include <array>

struct llama_context;

namespace deepseek_v41 {

struct block_output { ggml_tensor * streams; ggml_tensor * next_pre; };

class attention_forward_state {
public:
    static constexpr uint32_t max_batch = 8;
    attention_forward_state(model_config config, uint32_t capacity,
                            ggml_backend_buffer_type_t cache_type = nullptr);
    attention_forward_state(const attention_forward_state &) = delete;
    attention_forward_state & operator=(const attention_forward_state &) = delete;
    void clear();
    void begin(uint32_t position, uint32_t tokens = 1);
    void commit();
    void abort();
    void save_checkpoint(uint32_t max_tokens = max_batch);
    void restore_checkpoint(uint32_t position);
    void discard_checkpoint();
    uint32_t position() const { return position_; }
    const attention_cache & cache() const { return cache_; }
    const std::string & error() const { return error_; }

    // One token, [hidden,1], already attention-normalized. Graph construction
    // is read-only; only execution of the CPU state nodes changes the cache.
    // Production uses BF16 boundaries and FP8 activation quantization for the
    // projections stored as FP8 in the original checkpoint. false is an
    // initialized-FP32 reference mode, not a production accuracy claim.
    ggml_tensor * build(ggml_context * ctx, const std::map<std::string, ggml_tensor *> & weights,
        uint32_t layer, ggml_tensor * x, ggml_tensor * positions, bool production_precision = true,
        ggml_tensor * after = nullptr);

    // Full text transformer block, including its optional Engram injection.
    // Decoded rows are supplied by the runtime; x is [hidden,streams,1].
    block_output build_block(ggml_context * ctx, llama_context & lctx,
        const std::map<std::string, ggml_tensor *> & weights, uint32_t layer,
        ggml_tensor * x, ggml_tensor * incoming_pre, ggml_tensor * positions,
        ggml_cgraph * graph, ggml_tensor * engram_rows = nullptr, ggml_tensor * text_mask = nullptr);

    // A complete one-token text graph. Row inputs must cover exactly the
    // configured Engram layers. Caller begins/commits/aborts the state.
    ggml_tensor * build_token(ggml_context * ctx, llama_context & lctx,
        const std::map<std::string, ggml_tensor *> & weights, ggml_tensor * token,
        ggml_tensor * position, const std::map<uint32_t, ggml_tensor *> & engram_rows,
        ggml_cgraph * graph, uint32_t reserve_position = UINT32_MAX);
    // Layer-wise chunk: attention remains causally ordered within each layer;
    // HC, Engram and expert GEMMs share the token dimension.
    ggml_tensor * build_chunk(ggml_context * ctx, llama_context & lctx,
        const std::map<std::string, ggml_tensor *> & weights, ggml_tensor * tokens,
        ggml_tensor * positions, const std::map<uint32_t, ggml_tensor *> & engram_rows,
        ggml_cgraph * graph, uint32_t reserve_position = UINT32_MAX, ggml_tensor * output_ids = nullptr);

private:
    enum class operation { pool, publish, keys, select, gather, position, compressed_position, prefetch };
    struct callback_data { attention_forward_state * state; uint32_t layer; operation op; uint32_t token; };
    model_config config_;
    attention_layout layout_;
    attention_cache cache_;
    uint32_t capacity_;
    uint32_t position_ = 0;
    uint32_t batch_tokens_ = 1, build_token_index_ = 0;
    bool active_ = false;
    bool full_indexer_ = false;
    bool early_router_ = false;
    float prefetch_confidence_ = 0.85f;
    llama_context * advice_context_ = nullptr;
    std::vector<std::vector<float>> values_, scores_, saved_values_, saved_scores_;
    std::vector<std::vector<float>> checkpoint_values_, checkpoint_scores_;
    std::vector<std::vector<std::vector<float>>> checkpoint_value_steps_, checkpoint_score_steps_;
    std::vector<std::vector<int32_t>> selected_;
    std::vector<std::vector<uint8_t>> candidates_;
    std::vector<std::array<callback_data, 8>> callbacks_;
    std::string error_;
    bool all_index_rows_visible(uint32_t layer, uint32_t position) const;

    static void callback(ggml_tensor * dst, const ggml_tensor * a, const ggml_tensor * b,
                         const ggml_tensor * c, int ith, int nth, void * userdata);
    void execute(const callback_data & data, ggml_tensor * dst, const ggml_tensor * a,
                 const ggml_tensor * b, const ggml_tensor * c);
    ggml_tensor * node(ggml_context * ctx, uint32_t layer, operation op, std::initializer_list<int64_t> shape,
                       ggml_tensor * a, ggml_tensor * b = nullptr, ggml_tensor * c = nullptr, ggml_type type = GGML_TYPE_F32);
};

} // namespace deepseek_v41
