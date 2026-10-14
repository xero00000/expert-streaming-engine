#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <vector>

static bool check_type(const std::filesystem::path & path, uint32_t type, bool supported,
                       uint64_t row = 64, uint64_t rows = 1, bool create_tensors = false) {
    // Construct the wire header directly: the writer cannot create a tensor
    // with an unimplemented type. All integers in this fixture are little-endian.
    std::vector<unsigned char> bytes = {'G', 'G', 'U', 'F'};
    const auto append = [&](uint64_t value, unsigned width) {
        for (unsigned i = 0; i < width; ++i) bytes.push_back((value >> (8 * i)) & 0xff);
    };
    append(3, 4);  // version
    append(1, 8);  // tensor count
    append(0, 8);  // metadata count
    append(6, 8);
    for (char c : "weight") if (c) bytes.push_back(c);
    append(2, 4);  // dimensions
    append(row, 8);
    append(rows, 8);
    append(type, 4);
    append(0, 8);  // tensor offset
    while (bytes.size() % 32) bytes.push_back(0);
    bytes.resize(bytes.size() + 256, 0);

    FILE * file = ggml_fopen(path.string().c_str(), "wb");
    if (!file) return false;
    const bool written = std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    const int closed = std::fclose(file);
    if (!written || closed != 0) return false;

    ggml_context * tensors = nullptr;
    gguf_init_params params = {};
    params.no_alloc = true;
    params.ctx = create_tensors ? &tensors : nullptr;
    gguf_context * context = gguf_init_from_file(path.string().c_str(), params);
    const bool passed = (context != nullptr) == supported;
    if (context) gguf_free(context);
    if (tensors) ggml_free(tensors);
    if (!passed) std::fprintf(stderr, "unexpected GGUF parse result for type %u\n", type);
    return passed;
}

static bool check_alignment(const std::filesystem::path & path, uint32_t alignment, bool supported, bool wrong_type = false) {
    gguf_context * fixture = gguf_init_empty();
    if (wrong_type) gguf_set_val_str(fixture, "general.alignment", "32");
    else gguf_set_val_u32(fixture, "general.alignment", alignment);
    // Write metadata only: the writer itself cannot align to zero.
    const size_t size = gguf_get_meta_size(fixture);
    std::vector<unsigned char> bytes(size);
    gguf_get_meta_data(fixture, bytes.data());
    gguf_free(fixture);
    FILE * file = ggml_fopen(path.string().c_str(), "wb");
    if (!file) return false;
    const bool written = std::fwrite(bytes.data(), 1, size, file) == size;
    const int closed = std::fclose(file);
    if (!written || closed != 0) return false;
    gguf_init_params params = {};
    params.no_alloc = true;
    gguf_context * context = gguf_init_from_file(path.string().c_str(), params);
    const bool passed = (context != nullptr) == supported;
    if (context) gguf_free(context);
    return passed;
}

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("ese-gguf-type-" + std::to_string(nonce) + ".gguf");
    bool passed = check_type(path, GGML_TYPE_F32, true);
    // Removed legacy formats and an unassigned modern format ID.
    passed = check_type(path, GGML_TYPE_Q2_0, true) && passed;
    for (uint32_t type : {4u, 5u, 40u}) passed = check_type(path, type, false) && passed;
    // Total elements fit Q4_0 blocks, but each 48-element row does not. Reject
    // consistently before ggml_new_tensor/ggml_row_size can assert.
    for (bool create_tensors : {false, true}) {
        passed = check_type(path, GGML_TYPE_Q4_0, false, 48, 2, create_tensors) && passed;
        passed = check_type(path, GGML_TYPE_Q4_0, true, 32, 2, create_tensors) && passed;
        passed = check_type(path, GGML_TYPE_F32, true, 48, 2, create_tensors) && passed;
    }
    passed = check_alignment(path, 32, true) && passed;
    passed = check_alignment(path, 0, false) && passed;
    passed = check_alignment(path, 3, false) && passed;
    passed = check_alignment(path, 32, false, true) && passed;
    std::error_code error;
    std::filesystem::remove(path, error);
    return passed && !error ? 0 : 1;
}
