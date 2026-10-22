// Read-only metadata probe for the Qwen3.8-Flash-Next GSQ-RCO IQ1_M GGUF
// split. This is not a CTest fixture: it requires caller-supplied shard paths
// and exercises the current source tree's GGUF reader (including GGML type id
// 42, Q2_0) without allocating tensor data and without touching the
// production model-load architecture gate. Usage:
//
//   test-qwen4exp-artifact SHARD1.gguf SHARD2.gguf
//
// Shard 1 expectations (maintainer's Strata Coder artifact):
//   - general.architecture == qwen4exp, 48 layers, 256 experts, top-10;
//   - exactly 1,223 tensor records, exactly nine Q2_0 tensors;
//   - no PLE tensor (the per-layer embedding lives in shard 2).
// Shard 2 expectations:
//   - exactly one tensor: per_layer_token_embd.weight, IQ4_NL,
//     shape 160 x 320,001,536, 28,800,138,240 bytes of payload.

#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <map>
#include <string>

namespace {

void require(bool value, const std::string & message) {
    if (!value) {
        std::fprintf(stderr, "FAIL: %s\n", message.c_str());
        std::exit(1);
    }
}

uint32_t require_u32(gguf_context * gguf, const char * key, uint32_t expected) {
    const int key_id = gguf_find_key(gguf, key);
    require(key_id >= 0, std::string("missing metadata key ") + key);
    require(gguf_get_kv_type(gguf, key_id) == GGUF_TYPE_UINT32,
            std::string(key) + " is not UINT32");
    const uint32_t value = gguf_get_val_u32(gguf, key_id);
    require(value == expected, std::string(key) + ": expected " +
            std::to_string(expected) + ", got " + std::to_string(value));
    return value;
}

void print_kv(const gguf_context * gguf, const char * key) {
    const int key_id = gguf_find_key(gguf, key);
    if (key_id < 0) return;
    switch (gguf_get_kv_type(gguf, key_id)) {
        case GGUF_TYPE_UINT16:
            std::printf("  %s = %u\n", key, (unsigned) gguf_get_val_u16(gguf, key_id));
            break;
        case GGUF_TYPE_UINT32:
            std::printf("  %s = %u\n", key, (unsigned) gguf_get_val_u32(gguf, key_id));
            break;
        case GGUF_TYPE_INT32:
            std::printf("  %s = %d\n", key, gguf_get_val_i32(gguf, key_id));
            break;
        case GGUF_TYPE_UINT64:
            std::printf("  %s = %llu\n", key,
                    (unsigned long long) gguf_get_val_u64(gguf, key_id));
            break;
        case GGUF_TYPE_STRING:
            std::printf("  %s = %s\n", key, gguf_get_val_str(gguf, key_id));
            break;
        default:
            break;
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: %s SHARD1.gguf SHARD2.gguf\n", argv[0]);
        return 1;
    }
    const std::filesystem::path shard1_path(argv[1]);
    const std::filesystem::path shard2_path(argv[2]);
    require(std::filesystem::exists(shard1_path), "shard 1 path does not exist");
    require(std::filesystem::exists(shard2_path), "shard 2 path does not exist");

    // ---- Shard 1: model tensors and architecture metadata ----
    ggml_context * meta1 = nullptr;
    const gguf_init_params params1{ /*no_alloc=*/true, &meta1 };
    gguf_context * shard1 = gguf_init_from_file(shard1_path.string().c_str(), params1);
    require(shard1 != nullptr, "shard 1 did not parse with the current-source GGUF reader");
    require(meta1 != nullptr, "shard 1 tensor metadata context");

    const int arch_id = gguf_find_key(shard1, "general.architecture");
    require(arch_id >= 0 && std::string(gguf_get_val_str(shard1, arch_id)) == "qwen4exp",
            "shard 1 architecture is not qwen4exp");
    require_u32(shard1, "qwen4exp.block_count", 48);
    require_u32(shard1, "qwen4exp.expert_count", 256);
    require_u32(shard1, "qwen4exp.expert_used_count", 10);

    const int tensor_count = gguf_get_n_tensors(shard1);
    require(tensor_count == 1223, "shard 1 expected 1,223 tensor records, got " +
            std::to_string(tensor_count));

    int q2_0_count = 0;
    bool ple_in_shard1 = false;
    std::map<std::string, int> type_histogram;
    for (int i = 0; i < tensor_count; ++i) {
        const ggml_type type = gguf_get_tensor_type(shard1, i);
        if (type == GGML_TYPE_Q2_0) ++q2_0_count;
        ++type_histogram[ggml_type_name(type)];
        const char * name = gguf_get_tensor_name(shard1, i);
        if (std::string(name) == "per_layer_token_embd.weight") ple_in_shard1 = true;
    }
    require(q2_0_count == 9, "shard 1 expected nine Q2_0 (type id 42) tensors, got " +
            std::to_string(q2_0_count));
    require(!ple_in_shard1, "PLE tensor must not live in shard 1");

    std::printf("shard 1: %s (%zd bytes)\n", shard1_path.filename().c_str(),
            (std::ptrdiff_t) std::filesystem::file_size(shard1_path));
    std::printf("  %d tensors, %d Q2_0, arch/layer/expert/used metadata exact\n",
            tensor_count, q2_0_count);
    for (const auto & entry : type_histogram) {
        std::printf("  type %-14s %d\n", entry.first.c_str(), entry.second);
    }
    print_kv(shard1, "split.no");
    print_kv(shard1, "split.count");
    print_kv(shard1, "split.tensors.count");
    print_kv(shard1, "general.file_type");

    gguf_free(shard1);
    ggml_free(meta1);

    // ---- Shard 2: the per-layer embedding payload ----
    ggml_context * meta2 = nullptr;
    const gguf_init_params params2{ /*no_alloc=*/true, &meta2 };
    gguf_context * shard2 = gguf_init_from_file(shard2_path.string().c_str(), params2);
    require(shard2 != nullptr, "shard 2 did not parse with the current-source GGUF reader");
    require(meta2 != nullptr, "shard 2 tensor metadata context");

    const int ple_count = gguf_get_n_tensors(shard2);
    require(ple_count == 1, "shard 2 expected exactly one tensor, got " +
            std::to_string(ple_count));

    ggml_tensor * ple = ggml_get_first_tensor(meta2);
    require(ple != nullptr, "shard 2 exposes no tensor metadata");
    require(std::string(ggml_get_name(ple)) == "per_layer_token_embd.weight",
            std::string("shard 2 tensor is ") + ggml_get_name(ple) +
            ", expected per_layer_token_embd.weight");
    require(ple->type == GGML_TYPE_IQ4_NL, "PLE tensor is not IQ4_NL");
    require(ple->ne[0] == 160 && ple->ne[1] == 320001536 && ple->ne[2] == 1 && ple->ne[3] == 1,
            "PLE tensor shape is not 160 x 320,001,536");
    require(ggml_nbytes(ple) == 28800138240ull,
            "PLE tensor payload is not 28,800,138,240 bytes");
    const unsigned long long shard2_bytes =
            (unsigned long long) std::filesystem::file_size(shard2_path);
    require(shard2_bytes >= ggml_nbytes(ple) &&
            shard2_bytes - ggml_nbytes(ple) < (1ull << 20),
            "shard 2 file size does not bracket the PLE payload");
    require(ple->data == nullptr, "no_alloc probe must not hold PLE tensor data");

    std::printf("shard 2: %s (%llu bytes)\n", shard2_path.filename().c_str(), shard2_bytes);
    std::printf("  %s: %s, %lld x %lld, %lld bytes payload\n", ggml_get_name(ple),
            ggml_type_name(ple->type), (long long) ple->ne[0], (long long) ple->ne[1],
            (long long) ggml_nbytes(ple));

    gguf_free(shard2);
    ggml_free(meta2);

    std::printf("PASS: current-source GGUF reader parses both qwen4exp shards; "
            "architecture gate untouched\n");
    return 0;
}
