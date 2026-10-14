#include "llama-model-loader.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool result, const std::string & message) {
    if (!result) throw std::runtime_error(message);
}

struct scratch_directory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("ese-loader-splits-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    scratch_directory() { std::filesystem::create_directories(path); }
    ~scratch_directory() { std::error_code error; std::filesystem::remove_all(path, error); }
};

struct options {
    uint16_t index = 1, count = 2;
    int32_t total = 2;
    bool missing_index = false, missing_count = false, missing_total = false;
    bool wrong_index_type = false, wrong_arch_type = false, duplicate = false;
    const char * architecture = nullptr; // Upstream split writer stores architecture only in shard zero.
};

static void write_shard(const std::filesystem::path & path, bool first, options opt, bool empty_first = false) {
    auto * tensors = ggml_init({4096, nullptr, true});
    check(tensors != nullptr, "fixture tensor metadata");
    auto * meta = gguf_init_empty();
    check(meta != nullptr, "fixture GGUF metadata");
    if (first) gguf_set_val_str(meta, "general.architecture", opt.architecture ? opt.architecture : "qwen4exp");
    else if (opt.wrong_arch_type) gguf_set_val_u32(meta, "general.architecture", 1);
    else if (opt.architecture) gguf_set_val_str(meta, "general.architecture", opt.architecture);
    if (!opt.missing_index) {
        if (opt.wrong_index_type) gguf_set_val_u32(meta, "split.no", opt.index);
        else gguf_set_val_u16(meta, "split.no", opt.index);
    }
    if (!opt.missing_count) gguf_set_val_u16(meta, "split.count", opt.count);
    if (!opt.missing_total) gguf_set_val_i32(meta, "split.tensors.count", opt.total);
    std::vector<std::vector<float>> payloads;
    const unsigned n = first ? (empty_first ? 0 : 1) : (empty_first ? 2 : 1);
    payloads.reserve(n);
    for (unsigned i = 0; i < n; ++i) {
        auto * tensor = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 4, 3);
        const auto name = first || opt.duplicate ? "token_embd.weight" :
                          (i ? "blk.0.ffn_up_exps.weight" : "blk.0.ffn_gate_exps.weight");
        ggml_set_name(tensor, name);
        payloads.emplace_back(12, float(first ? 1 : 2 + i));
        tensor->data = payloads.back().data();
        gguf_add_tensor(meta, tensor);
    }
    gguf_write_to_file(meta, path.string().c_str(), false);
    gguf_free(meta);
    ggml_free(tensors);
}

static void load(const std::filesystem::path & path, bool expect_success, const std::string & label) {
    try {
        llama_model_loader loader(path.string(), 0, false, false, false, false, false, false, false, nullptr, nullptr);
        check(expect_success, "malformed shard accepted: " + label);
        check(loader.n_tensors == 2 && loader.files.size() == 2, "valid split inventory");
        check(loader.check_tensor_dims("blk.0.ffn_gate_exps.weight", {4, 3}, true) != nullptr,
              "secondary tensor dimensions");
        bool rejected_shape = false;
        try { loader.check_tensor_dims("blk.0.ffn_gate_exps.weight", {4, 4}, true); }
        catch (const std::exception &) { rejected_shape = true; }
        check(rejected_shape, "wrong tensor shape accepted");
    } catch (const std::exception & error) {
        if (expect_success || std::string(error.what()).find("malformed shard accepted:") == 0) throw;
        std::printf("rejected %s: %s\n", label.c_str(), error.what());
    }
}

int main(int argc, char ** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--inventory") {
            llama_model_loader loader(argv[2], 0, false, false, false, false, false, false, false, nullptr, nullptr);
            std::printf("split_inventory: architecture=%s files=%zu tensors=%d\n",
                        loader.arch_name.c_str(), loader.files.size(), loader.n_tensors);
            return 0;
        }
        const bool failure_only = argc > 1 && std::string(argv[1]) == "--failure-cleanup";
        scratch_directory dir;
        const auto first = dir.path / "fixture-00001-of-00002.gguf";
        const auto second = dir.path / "fixture-00002-of-00002.gguf";
        options main; main.index = 0;
        auto pair = [&](options secondary = {}, bool empty_first = false) {
            write_shard(first, true, main, empty_first);
            write_shard(second, false, secondary, empty_first);
        };
        pair(); load(first, true, "valid metadata-only secondary");
        pair({}, true); load(first, true, "valid empty first shard");
        options matching; matching.architecture = "qwen4exp";
        pair(matching); load(first, true, "valid repeated architecture");
        main.architecture = "llama";
        pair(); load(first, true, "valid conventional architecture split");
        main.architecture = nullptr;
        main.count = 0; // The bundled merge tool leaves split.count=0.
        write_shard(first, true, main);
        {
            llama_model_loader loader(first.string(), 0, false, false, false, false, false, false, false, nullptr, nullptr);
            check(loader.files.size() == 1 && loader.n_tensors == 1, "merged/unsplit inventory");
        }
        main.count = 2;
        if (!failure_only) {
            std::vector<std::pair<std::string, options>> bad;
            options opt;
            opt.index = 0; bad.emplace_back("wrong secondary index", opt); opt = {};
            opt.count = 3; bad.emplace_back("wrong secondary split count", opt); opt = {};
            opt.total = 3; bad.emplace_back("wrong secondary total", opt); opt = {};
            opt.missing_index = true; bad.emplace_back("missing secondary index", opt); opt = {};
            opt.missing_count = true; bad.emplace_back("missing secondary split count", opt); opt = {};
            opt.missing_total = true; bad.emplace_back("missing secondary total", opt); opt = {};
            opt.wrong_index_type = true; bad.emplace_back("wrong secondary index type", opt); opt = {};
            opt.architecture = "llama"; bad.emplace_back("mixed architecture", opt); opt = {};
            opt.wrong_arch_type = true; bad.emplace_back("wrong architecture type", opt); opt = {};
            opt.duplicate = true; bad.emplace_back("duplicate tensor name", opt);
            for (const auto & item : bad) { pair(item.second); load(first, false, item.first); }
            pair(); load(second, false, "not the first shard");
            main.total = 3; pair(); load(first, false, "wrong primary total"); main.total = 2;
            main.missing_total = true; pair(); load(first, false, "missing primary total"); main.missing_total = false;
        }
        // Repeat existing failures to expose constructor leaks; every iteration
        // also proves a valid model remains loadable afterward.
        for (int i = 0; i < 8; ++i) {
            pair(); std::filesystem::remove(second); load(first, false, "missing secondary file");
            pair(); std::filesystem::resize_file(second, 16); load(first, false, "truncated secondary header");
            pair(); std::filesystem::resize_file(second, std::filesystem::file_size(second) - 20);
            load(first, false, "truncated secondary tensor payload");
            pair(); load(first, true, "valid split after failed construction");
        }
        std::puts("PASS: split inventory, tensor shape, malformed shards and failure cleanup");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
