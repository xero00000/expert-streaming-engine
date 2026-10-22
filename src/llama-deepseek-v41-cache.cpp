#include "llama-deepseek-v41-cache.h"
#include "../ggml/src/ggml-qat.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace deepseek_v41 {
static void check(bool condition, const char * message) {
    if (!condition) throw std::invalid_argument(message);
}

struct packed_cache_storage::device_allocation {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor tensor = {};
    ~device_allocation() { ggml_backend_buffer_free(buffer); }
};
packed_cache_storage::packed_cache_storage() = default;
packed_cache_storage::~packed_cache_storage() = default;
packed_cache_storage::packed_cache_storage(packed_cache_storage &&) noexcept = default;
packed_cache_storage & packed_cache_storage::operator=(packed_cache_storage &&) noexcept = default;
packed_cache_storage::packed_cache_storage(ggml_backend_buffer_type_t type, size_t capacity)
    : type_(type), capacity_(capacity) {
    if (!type) { host_.reserve(capacity); return; }
    if (!capacity) return;
    check(capacity <= size_t(INT64_MAX), "V4.1 device cache extent is too large");
    device_ = std::make_unique<device_allocation>();
    // Only metadata is copied out of this temporary no-allocation context.
    // Keeping one context per bank would exhaust ggml's context table when
    // taking snapshots of all 40 windows.
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context(
            ggml_init({ggml_tensor_overhead() + 1024, nullptr, true}), ggml_free);
    check(context != nullptr, "V4.1 device cache tensor allocation failed");
    device_->tensor = *ggml_new_tensor_1d(context.get(), GGML_TYPE_I8, int64_t(capacity));
    context.reset();
    const size_t allocation = ggml_backend_buft_get_alloc_size(type, &device_->tensor);
    device_->buffer = ggml_backend_buft_alloc_buffer(type, allocation);
    check(device_->buffer != nullptr, "V4.1 device cache allocation failed");
    ggml_backend_tensor_alloc(device_->buffer, &device_->tensor, ggml_backend_buffer_get_base(device_->buffer));
}
packed_cache_storage::packed_cache_storage(const packed_cache_storage & other)
    : packed_cache_storage(other.type_, other.capacity_) { *this = other; }
packed_cache_storage & packed_cache_storage::operator=(const packed_cache_storage & other) {
    if (this == &other) return *this;
    if (type_ != other.type_ || capacity_ != other.capacity_) {
        packed_cache_storage replacement(other);
        *this = std::move(replacement);
        return *this;
    }
    resize(other.size_);
    if (device_ && size_) ggml_backend_tensor_copy(&other.device_->tensor, &device_->tensor);
    else if (!type_) host_ = other.host_;
    return *this;
}
void packed_cache_storage::resize(size_t size) {
    check(size <= capacity_, "V4.1 cache storage capacity exceeded");
    if (!type_) host_.resize(size);
    size_ = size;
}
void packed_cache_storage::read(void * destination, size_t offset, size_t count) const {
    check(offset <= size_ && count <= size_ - offset, "V4.1 cache read out of bounds");
    if (!count) return;
    if (device_) ggml_backend_tensor_get(&device_->tensor, destination, offset, count);
    else std::memcpy(destination, host_.data() + offset, count);
}
void packed_cache_storage::write(const void * source, size_t offset, size_t count) {
    check(offset <= size_ && count <= size_ - offset, "V4.1 cache write out of bounds");
    if (!count) return;
    if (device_) ggml_backend_tensor_set(&device_->tensor, source, offset, count);
    else std::memcpy(host_.data() + offset, source, count);
}
void packed_cache_storage::zero() {
    if (device_) ggml_backend_buffer_clear(device_->buffer, 0);
    else std::fill(host_.begin(), host_.end(), 0);
}

attention_cache::attention_cache(attention_layout_config config, uint32_t capacity_tokens,
                                 ggml_backend_buffer_type_t storage_type)
    : config_(std::move(config)), layout_(config_), capacity_(capacity_tokens) {
    check(capacity_ > 0 && config_.sliding_window > 0, "invalid V4.1 cache capacity");
    const size_t layers = layout_.layers().size();
    windows_.resize(layers); global_.resize(layers); index_.resize(layers);
    auto init = [storage_type](bank & b, uint32_t width, ggml_qat_format format, uint32_t capacity) {
        b.width = width; b.format = format; b.row_bytes = ggml_qat_row_size(width, format);
        check(b.row_bytes && capacity <= SIZE_MAX / b.row_bytes, "invalid packed V4.1 cache geometry");
        b.bytes = packed_cache_storage(storage_type, size_t(capacity) * b.row_bytes);
    };
    for (uint32_t layer = 0; layer < layers; ++layer) {
        init(windows_[layer].storage, config_.head_dim, GGML_QAT_FP8_E8M0, config_.sliding_window);
        windows_[layer].storage.bytes.resize(size_t(config_.sliding_window) * windows_[layer].storage.row_bytes);
        windows_[layer].storage.bytes.zero();
        if (layout_.layers()[layer].kv_owner == int32_t(layer)) {
            const uint32_t rows = capacity_ / layout_.layers()[layer].compress_ratio;
            init(global_[layer], config_.head_dim, GGML_QAT_FP4_E4M3, rows);
            init(index_[layer], config_.index_head_dim, GGML_QAT_FP4_E8M0, rows);
        }
    }
}

void attention_cache::clear() {
    check(!active_, "cannot clear an active V4.1 cache transaction");
    for (auto & w : windows_) { w.end = 0; w.storage.bytes.zero(); }
    for (auto * banks : {&global_, &index_}) for (auto & b : *banks) { b.rows = 0; b.bytes.resize(0); }
    saved_windows_.clear(); saved_rows_.clear();
    discard_checkpoint();
}

void attention_cache::save_checkpoint(uint32_t max_tokens) {
    check(!active_ && max_tokens > 0 && max_tokens <= 8, "invalid V4.1 checkpoint extent");
    const uint32_t start = windows_.front().end;
    check(start <= capacity_ && max_tokens <= capacity_ - start, "V4.1 checkpoint exceeds capacity");
    auto windows = windows_;
    std::vector<std::vector<uint8_t>> updates(windows_.size());
    for (size_t i = 0; i < windows_.size(); ++i) {
        check(windows_[i].end == start, "V4.1 checkpoint requires complete layer state");
        updates[i].reserve(size_t(max_tokens) * windows_[i].storage.row_bytes);
    }
    checkpoint_windows_ = std::move(windows);
    checkpoint_updates_ = std::move(updates);
    checkpoint_position_ = start; checkpoint_limit_ = start + max_tokens; checkpoint_ = true;
}
void attention_cache::discard_checkpoint() {
    check(!active_, "cannot discard an active V4.1 checkpoint");
    checkpoint_ = false; checkpoint_windows_.clear(); checkpoint_updates_.clear(); saved_update_sizes_.clear();
}
void attention_cache::restore_checkpoint(uint32_t position) {
    check(!active_ && checkpoint_ && position >= checkpoint_position_ && position <= checkpoint_limit_,
            "invalid V4.1 checkpoint restore position");
    auto restored = checkpoint_windows_;
    for (size_t layer = 0; layer < windows_.size(); ++layer) {
        const auto bytes = windows_[layer].storage.row_bytes;
        check(position <= windows_[layer].end && checkpoint_updates_[layer].size() >= size_t(position - checkpoint_position_) * bytes,
                "V4.1 checkpoint prefix was not evaluated");
        for (uint32_t token = checkpoint_position_; token < position; ++token)
            restored[layer].storage.bytes.write(
                    checkpoint_updates_[layer].data() + size_t(token - checkpoint_position_) * bytes,
                    (token % config_.sliding_window) * bytes, bytes);
        restored[layer].end = position;
        if (global_[layer].row_bytes)
            check(global_[layer].rows >= position / layout_.layers()[layer].compress_ratio,
                    "V4.1 checkpoint global prefix is missing");
    }
    windows_ = std::move(restored);
    for (size_t layer = 0; layer < global_.size(); ++layer) if (global_[layer].row_bytes) {
        const auto rows = position / layout_.layers()[layer].compress_ratio;
        global_[layer].rows = index_[layer].rows = rows;
        global_[layer].bytes.resize(size_t(rows) * global_[layer].row_bytes);
        index_[layer].bytes.resize(size_t(rows) * index_[layer].row_bytes);
    }
    discard_checkpoint();
}

void attention_cache::begin() {
    check(!active_, "V4.1 cache transaction already active");
    // Only bounded sliding windows need copying. Global banks are append-only.
    // Reuse the bounded window snapshots so GPU storage is not allocated on
    // every token. The copy assignment overwrites each existing allocation.
    saved_windows_ = windows_;
    std::vector<size_t> update_sizes;
    if (checkpoint_) for (const auto & updates : checkpoint_updates_) update_sizes.push_back(updates.size());
    std::vector<uint32_t> rows;
    for (const auto & b : global_) rows.push_back(b.rows);
    saved_rows_ = std::move(rows);
    saved_update_sizes_ = std::move(update_sizes); active_ = true;
}
void attention_cache::commit() {
    check(active_, "no V4.1 cache transaction");
    const uint32_t end = windows_.front().end;
    for (const auto & w : windows_) check(w.end == end, "incomplete V4.1 layer window publication");
    for (uint32_t layer = 0; layer < global_.size(); ++layer) if (global_[layer].row_bytes) {
        check(global_[layer].rows == end / layout_.layers()[layer].compress_ratio,
              "incomplete V4.1 global cache publication");
    }
    saved_rows_.clear(); active_ = false;
}
void attention_cache::abort() {
    check(active_, "no V4.1 cache transaction");
    windows_.swap(saved_windows_);
    for (size_t i = 0; i < global_.size(); ++i) {
        global_[i].rows = index_[i].rows = saved_rows_[i];
        global_[i].bytes.resize(size_t(saved_rows_[i]) * global_[i].row_bytes);
        index_[i].bytes.resize(size_t(saved_rows_[i]) * index_[i].row_bytes);
    }
    if (checkpoint_) for (size_t i = 0; i < checkpoint_updates_.size(); ++i)
        checkpoint_updates_[i].resize(saved_update_sizes_[i]);
    saved_update_sizes_.clear();
    saved_windows_.clear(); saved_rows_.clear(); active_ = false;
}

uint32_t attention_cache::owner(uint32_t layer) const {
    check(layer < layout_.layers().size() && layout_.layers()[layer].kv_owner >= 0, "layer has no V4.1 global cache");
    return uint32_t(layout_.layers()[layer].kv_owner);
}
std::vector<uint8_t> attention_cache::pack(const bank & b, const std::vector<float> & row) {
    check(row.size() == b.width, "invalid V4.1 cache row width");
    std::vector<uint8_t> packed(b.row_bytes);
    check(ggml_qat_pack_row(row.data(), packed.data(), b.width, ggml_qat_format(b.format)), "unrepresentable V4.1 cache row");
    return packed;
}
void attention_cache::decode(const bank & b, uint32_t row, float * dst) {
    check(size_t(row + 1) * b.row_bytes <= b.bytes.size(), "V4.1 cache row out of bounds");
    std::vector<uint8_t> packed(b.row_bytes);
    b.bytes.read(packed.data(), size_t(row) * b.row_bytes, packed.size());
    check(ggml_qat_unpack_row(packed.data(), dst, b.width, ggml_qat_format(b.format)),
          "invalid packed V4.1 cache row");
}
void attention_cache::append_window(uint32_t layer, uint32_t position, const std::vector<float> & row) {
    check(active_ && layer < windows_.size() && position < capacity_, "invalid V4.1 window write");
    auto & w = windows_[layer];
    check(position == w.end, "nonsequential V4.1 window write");
    auto packed = pack(w.storage, row);
    if (checkpoint_) {
        check(position >= checkpoint_position_ && position < checkpoint_limit_, "V4.1 checkpoint journal is full");
        auto & updates = checkpoint_updates_[layer];
        check(updates.size() == size_t(position - checkpoint_position_) * w.storage.row_bytes,
                "nonsequential V4.1 checkpoint journal");
        updates.insert(updates.end(), packed.begin(), packed.end());
    }
    w.storage.bytes.write(packed.data(), (position % config_.sliding_window) * w.storage.row_bytes, packed.size());
    ++w.end;
}
void attention_cache::append_global(uint32_t layer, uint32_t row, const std::vector<float> & kv,
                                     const std::vector<float> & index_k) {
    check(active_ && owner(layer) == layer, "V4.1 global write by non-owner");
    auto & b = global_[layer]; auto & k = index_[layer];
    check(row == b.rows && row < capacity_ / layout_.layers()[layer].compress_ratio, "nonsequential V4.1 global write");
    auto packed_kv = pack(b, kv), packed_index = pack(k, index_k);
    // Capacities were reserved at construction; no allocation can split publication.
    const size_t kv_offset = b.bytes.size(), index_offset = k.bytes.size();
    b.bytes.resize(kv_offset + packed_kv.size());
    k.bytes.resize(index_offset + packed_index.size());
    b.bytes.write(packed_kv.data(), kv_offset, packed_kv.size());
    k.bytes.write(packed_index.data(), index_offset, packed_index.size());
    ++b.rows; ++k.rows;
}
std::vector<float> attention_cache::read_window(uint32_t layer) const {
    check(layer < windows_.size(), "invalid V4.1 window layer");
    const auto & w = windows_[layer];
    const uint32_t count = std::min(w.end, config_.sliding_window), start = w.end - count;
    std::vector<float> result(size_t(count) * w.storage.width);
    std::vector<uint8_t> packed(w.storage.bytes.size());
    w.storage.bytes.read(packed.data(), 0, packed.size());
    for (uint32_t i = 0; i < count; ++i)
        check(ggml_qat_unpack_row(packed.data() + ((start + i) % config_.sliding_window) * w.storage.row_bytes,
                result.data() + size_t(i) * w.storage.width, w.storage.width, ggml_qat_format(w.storage.format)),
                "invalid packed V4.1 window row");
    return result;
}
std::vector<float> attention_cache::read_global(uint32_t layer, bool index_keys, const std::vector<uint32_t> & rows) const {
    const auto & b = index_keys ? index_[owner(layer)] : global_[owner(layer)];
    std::vector<float> result(rows.size() * b.width);
    if (rows.empty()) return result;
    const auto bounds = std::minmax_element(rows.begin(), rows.end());
    check(*bounds.second < b.rows, "unpublished V4.1 global cache row");
    const size_t span = size_t(*bounds.second - *bounds.first + 1);
    // Read dense prefixes in one transfer. Sparse gathers retain bounded
    // row staging instead of making a permanent host copy of the KV bank.
    std::vector<uint8_t> packed;
    if (span <= rows.size() * 32 || span * b.row_bytes <= 8 * 1024 * 1024) {
        packed.resize(span * b.row_bytes);
        b.bytes.read(packed.data(), size_t(*bounds.first) * b.row_bytes, packed.size());
    }
    for (size_t i = 0; i < rows.size(); ++i) {
        check(rows[i] < b.rows, "unpublished V4.1 global cache row");
        if (packed.empty()) decode(b, rows[i], result.data() + i * b.width);
        else check(ggml_qat_unpack_row(packed.data() + size_t(rows[i] - *bounds.first) * b.row_bytes,
                    result.data() + i * b.width, b.width, ggml_qat_format(b.format)), "invalid packed V4.1 global row");
    }
    return result;
}
uint32_t attention_cache::global_rows(uint32_t layer) const { return global_[owner(layer)].rows; }
uint32_t attention_cache::window_end(uint32_t layer) const {
    check(layer < windows_.size(), "invalid V4.1 window layer"); return windows_[layer].end;
}
uint64_t attention_cache::packed_payload_bytes() const {
    uint64_t bytes = 0;
    for (const auto & b : global_) bytes += b.bytes.size();
    for (const auto & b : index_) bytes += b.bytes.size();
    for (const auto & w : windows_) bytes += w.storage.bytes.size();
    return bytes;
}
uint64_t attention_cache::reserved_payload_bytes() const {
    uint64_t bytes = 0;
    for (const auto & b : global_) bytes += b.bytes.capacity();
    for (const auto & b : index_) bytes += b.bytes.capacity();
    for (const auto & w : windows_) bytes += w.storage.bytes.capacity();
    for (const auto & w : saved_windows_) bytes += w.storage.bytes.capacity();
    return bytes;
}
} // namespace deepseek_v41
