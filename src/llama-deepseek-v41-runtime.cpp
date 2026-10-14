#include "llama-deepseek-v41-runtime.h"
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include "../vendor/nlohmann/json.hpp"

namespace deepseek_v41 {
static void check(bool ok, const char * message) { if (!ok) throw std::invalid_argument(message); }

runtime_assets load_runtime_assets(const model_config & model, const std::string & metadata,
                                  const std::string & model_directory) {
    const auto document = nlohmann::json::parse(metadata);
    check(document.at("version").get<int>() == 1 && document.at("rows").size() == 2,
            "unsupported V4.1 asset metadata");
    runtime_assets result;
    const auto & hash = document.at("hash");
    check(hash.at("token_map").size() == model.vocabulary && hash.at("layers").size() == 2,
            "V4.1 asset hash geometry mismatch");
    result.hash.compressed_vocab_size = hash.at("compressed_vocab_size");
    result.hash.pad_compressed_id = hash.at("pad_compressed_id");
    result.hash.token_map = hash.at("token_map").get<std::vector<uint32_t>>();
    for (size_t i = 0; i < 2; ++i) {
        auto & spec = result.hash.layers[i];
        const auto & h = hash.at("layers").at(i);
        spec.layer_id = h.at("layer_id"); spec.rows = h.at("rows");
        spec.multipliers = h.at("multipliers").get<std::array<uint64_t, 4>>();
        spec.primes = h.at("primes").get<std::array<uint64_t, 24>>();
        const auto & row = document.at("rows").at(i);
        check(i < model.engram_layers.size() && spec.layer_id == model.engram_layers[i] &&
                row.at("layer_id").get<uint32_t>() == spec.layer_id && row.at("row_count").get<uint64_t>() == spec.rows,
                "V4.1 asset layer/row identity mismatch");
        const std::string encoding = row.at("encoding");
        check(encoding == "affine2-g64-bf16" || encoding == "fp8-e4m3-e8m0-g32", "unsupported V4.1 Engram encoding");
        auto & asset = result.rows[i];
        asset.format = encoding == "affine2-g64-bf16" ? engram_row_format::affine2 : engram_row_format::fp8;
        asset.layout = {row.at("data_offset").get<uint64_t>(), spec.rows, asset.format == engram_row_format::affine2 ? size_t(80) : size_t(264)};
        const std::filesystem::path relative(row.at("file").get<std::string>());
        check(!relative.empty() && !relative.is_absolute(), "V4.1 asset path must be relative to its model");
        for (const auto & part : relative) check(part != "..", "V4.1 asset path escapes model directory");
        const auto path = (std::filesystem::path(model_directory) / relative).string();
        auto identity = llama_expert_cache::identify_file_source(path, uint32_t(i));
        asset.source = std::make_shared<llama_expert_cache::file_source>(uint32_t(i), identity, path, llama_expert_cache::read_backend::pread);
        const auto bytes = asset.source->size();
        check(bytes == row.at("file_bytes").get<uint64_t>() && asset.layout.data_offset <= bytes &&
                spec.rows <= (bytes - asset.layout.data_offset) / asset.layout.row_bytes,
                "V4.1 Engram file is truncated or has mismatched geometry");
    }
    engram_sequence_state validate(result.hash);
    return result;
}

runtime_state::runtime_state(model_config model, uint32_t capacity, engram_hash_config hash,
        std::array<engram_asset, 2> assets, uint64_t row_cache_bytes, ggml_backend_buffer_type_t cache_type)
    : capacity_(capacity), forward_(model, capacity, cache_type), hash_(hash) {
    check(capacity <= model.context && hash.token_map.size() == model.vocabulary &&
            model.engram_layers.size() == 2 && model.engram_head_dim == 256 &&
            model.engram_heads == 8 && model.engram_max_ngram == 4,
            "V4.1 runtime/hash model geometry mismatch");
    for (size_t i = 0; i < 2; ++i) {
        check(hash.layers[i].layer_id == model.engram_layers[i] && assets[i].layout.row_count == hash.layers[i].rows,
                "V4.1 Engram table/hash geometry mismatch");
        const size_t bytes = assets[i].format == engram_row_format::fp8 ? 264 : 80;
        check((assets[i].format == engram_row_format::fp8 || assets[i].format == engram_row_format::affine2) &&
                assets[i].layout.row_bytes == bytes && row_cache_bytes >= bytes * 24,
                "V4.1 Engram row format/cache geometry mismatch");
        layers_[i] = hash.layers[i].layer_id; formats_[i] = assets[i].format;
        stores_[i] = std::make_unique<qwen4exp_ple_storage::row_store>(assets[i].source, assets[i].layout, 2, row_cache_bytes, 24);
        decoded_[layers_[i]].reserve(24 * 256);
    }
}

uint32_t runtime_state::position() const { return active_ ? begin_position_ : forward_.cache().window_end(0); }
bool runtime_state::remove(int32_t sequence, int32_t first, int32_t end) {
    check(!active_, "cannot remove an active V4.1 sequence");
    if (!bound_ || (sequence >= 0 && sequence != sequence_)) return true;
    const uint32_t start = first < 0 ? 0 : uint32_t(first);
    const uint32_t stop = end < 0 ? position() : uint32_t(end);
    if (start >= position() || stop <= start) return true;
    if (start == 0 && stop >= position()) { clear(); return true; }
    return false;
}
void runtime_state::begin(int32_t sequence, uint32_t pos, uint32_t token) {
    begin_batch(sequence, pos, {token});
}
void runtime_state::begin_batch(int32_t sequence, uint32_t pos, const std::vector<uint32_t> & tokens) {
    check(!active_ && (!bound_ || sequence == sequence_) && pos == position() && pos < capacity_ &&
            !tokens.empty() && tokens.size() <= attention_forward_state::max_batch && tokens.size() <= capacity_ - pos,
            "V4.1 runtime requires a bounded contiguous batch in its bound sequence");
    check(hash_.position(sequence) == pos, "V4.1 hash/cache positions disagree");
    forward_.begin(pos, tokens.size());
    sequence_ = sequence; begin_position_ = pos; active_ = true;
    try {
        ++owner_;
        for (auto & item : decoded_) item.second.resize(tokens.size() * 24 * 256);
        for (size_t token = 0; token < tokens.size(); ++token) {
            const auto rows = hash_.accept(sequence, pos + token, tokens[token]);
            std::array<qwen4exp_ple_storage::row_store::ticket, 2> tickets;
            for (size_t i = 0; i < 2; ++i) tickets[i] = stores_[i]->submit(owner_, {rows[i].begin(), rows[i].end()});
            for (size_t i = 0; i < 2; ++i) {
                auto batch = tickets[i].get();
                check(batch.size() == 24, "V4.1 Engram row fetch count mismatch");
                auto & output = decoded_.at(layers_[i]);
                for (size_t row = 0; row < 24; ++row) {
                    const auto & bytes = batch.row(row);
                    auto decode = formats_[i] == engram_row_format::fp8 ? engram_decode_fp8 : engram_decode_affine2;
                    decode(bytes.data(), bytes.size(), output.data() + (token * 24 + row) * 256, 256);
                }
            }
        }
        bound_ = true;
    } catch (...) {
        abort();
        throw;
    }
}
void runtime_state::commit() {
    check(active_, "no active V4.1 runtime token");
    // Forward commit validates every layer and may throw. Keep hash rollback
    // available until that validation has succeeded.
    forward_.commit();
    active_ = false;
    for (auto & item : decoded_) item.second.clear();
}
void runtime_state::abort() {
    check(active_, "no active V4.1 runtime token");
    for (auto & store : stores_) store->cancel_owner(owner_);
    forward_.abort();
    hash_.rewind(sequence_, begin_position_);
    for (auto & item : decoded_) item.second.clear();
    active_ = false;
    if (begin_position_ == 0) bound_ = false;
}
void runtime_state::save_checkpoint(int32_t sequence, uint32_t max_tokens) {
    check(!active_ && sequence >= 0 && (!bound_ || sequence_ == sequence) && hash_.position(sequence) == position(),
            "invalid V4.1 runtime checkpoint sequence");
    forward_.save_checkpoint(max_tokens);
    checkpoint_sequence_ = sequence; checkpoint_bound_ = bound_;
}
void runtime_state::restore_checkpoint(int32_t sequence, uint32_t pos) {
    check(!active_ && forward_.cache().has_checkpoint() && sequence == checkpoint_sequence_ &&
            (!bound_ || sequence == sequence_) && hash_.position(sequence) == position(),
            "invalid V4.1 runtime checkpoint restore sequence");
    forward_.restore_checkpoint(pos);
    hash_.rewind(sequence, pos);
    sequence_ = sequence; bound_ = pos > 0 || checkpoint_bound_;
    for (auto & item : decoded_) item.second.clear();
}
void runtime_state::discard_checkpoint() {
    check(!active_, "cannot discard active V4.1 runtime checkpoint");
    forward_.discard_checkpoint();
}
void runtime_state::clear() {
    if (active_) abort();
    forward_.clear(); hash_.clear(); bound_ = false;
    for (auto & item : decoded_) item.second.clear();
}
const std::map<uint32_t, std::vector<float>> & runtime_state::decoded_rows() const {
    check(active_, "V4.1 Engram inputs are only available during a token transaction");
    return decoded_;
}
qwen4exp_ple_storage::row_store_stats runtime_state::row_stats(size_t layer) const { return stores_.at(layer)->stats(); }
} // namespace deepseek_v41
