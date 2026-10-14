#pragma once
#include "llama-deepseek-v41-layout.h"
#include <cstdint>
#include <cstddef>
#include <vector>
#include <memory>
#include "ggml-backend.h"

namespace deepseek_v41 {

// Packed bytes have one authoritative allocation, either host or backend.
// Copies are deep snapshots; reads use temporary host staging only.
class packed_cache_storage {
public:
    packed_cache_storage();
    packed_cache_storage(ggml_backend_buffer_type_t type, size_t capacity);
    ~packed_cache_storage();
    packed_cache_storage(const packed_cache_storage & other);
    packed_cache_storage & operator=(const packed_cache_storage & other);
    packed_cache_storage(packed_cache_storage &&) noexcept;
    packed_cache_storage & operator=(packed_cache_storage &&) noexcept;
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    void resize(size_t size);
    void read(void * destination, size_t offset, size_t count) const;
    void write(const void * source, size_t offset, size_t count);
    void zero();
private:
    struct device_allocation;
    ggml_backend_buffer_type_t type_ = nullptr;
    size_t size_ = 0, capacity_ = 0;
    std::vector<uint8_t> host_;
    std::unique_ptr<device_allocation> device_;
};

// One sequence's packed KV state. A transaction covers one complete forward
// chunk; partial layer execution can be aborted without publishing its state.
// Arbitrary historical rewind is not provided by this class.
class attention_cache {
public:
    attention_cache(attention_layout_config config, uint32_t capacity_tokens,
                    ggml_backend_buffer_type_t storage_type = nullptr);
    void clear();
    void begin();
    void commit();
    void abort();
    bool in_transaction() const { return active_; }
    // One bounded speculative checkpoint, consumed by restore. Global rows
    // remain append-only; only windows and the short overwrite journal copy.
    void save_checkpoint(uint32_t max_tokens = 8);
    void restore_checkpoint(uint32_t position);
    void discard_checkpoint();
    bool has_checkpoint() const { return checkpoint_; }
    uint32_t checkpoint_position() const { return checkpoint_position_; }

    // Position must be the next position for that layer. The row is post-RoPE.
    void append_window(uint32_t layer, uint32_t position, const std::vector<float> & row);
    // Only the owning layer can append, and K and index K publish atomically.
    // Both inputs are post-RoPE. No call is made for an incomplete group.
    void append_global(uint32_t owner, uint32_t row, const std::vector<float> & kv,
                       const std::vector<float> & index_k);

    // Chronological window ending at the layer's most recently appended row.
    std::vector<float> read_window(uint32_t layer) const;
    std::vector<float> read_global(uint32_t layer, bool index_keys, const std::vector<uint32_t> & rows) const;
    uint32_t global_rows(uint32_t layer) const;
    uint32_t window_end(uint32_t layer) const;
    uint64_t packed_payload_bytes() const;
    uint64_t reserved_payload_bytes() const;

private:
    struct bank {
        uint32_t width = 0;
        int format = 0;
        size_t row_bytes = 0;
        uint32_t rows = 0;
        packed_cache_storage bytes;
    };
    struct window {
        uint32_t end = 0;
        bank storage;
    };
    attention_layout_config config_;
    attention_layout layout_;
    uint32_t capacity_;
    std::vector<window> windows_;
    std::vector<bank> global_, index_;
    std::vector<window> saved_windows_;
    std::vector<uint32_t> saved_rows_;
    bool active_ = false;
    bool checkpoint_ = false;
    uint32_t checkpoint_position_ = 0, checkpoint_limit_ = 0;
    std::vector<window> checkpoint_windows_;
    std::vector<std::vector<uint8_t>> checkpoint_updates_;
    std::vector<size_t> saved_update_sizes_;
    uint32_t owner(uint32_t layer) const;
    static std::vector<uint8_t> pack(const bank & b, const std::vector<float> & row);
    static void decode(const bank & b, uint32_t row, float * dst);
};

} // namespace deepseek_v41
