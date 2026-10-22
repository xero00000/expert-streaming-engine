#include "ggml.h"
#include "llama.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>

static void capture_log(enum ggml_log_level, const char * text, void * user_data) {
    static_cast<std::string *>(user_data)->append(text);
}

int main() {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
        ("ese-qwen4exp-probe-" + std::to_string(nonce) + ".gguf");

    gguf_context * fixture = gguf_init_empty();
    gguf_set_val_str(fixture, "general.architecture", "qwen4exp");
    gguf_write_to_file(fixture, path.string().c_str(), false);
    gguf_free(fixture);

    std::string logs;
    llama_log_set(capture_log, &logs);
    llama_model * model = llama_model_load_from_file(path.string().c_str(), llama_model_default_params());
    llama_log_set(nullptr, nullptr);
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);

    if (model != nullptr) {
        std::fprintf(stderr, "qwen4exp fixture unexpectedly loaded\n");
        llama_free_model(model);
        return 1;
    }
    if (logs.find("unsupported architecture: qwen4exp") == std::string::npos) {
        std::fprintf(stderr, "missing qwen4exp support diagnostic; logs:\n%s", logs.c_str());
        return 1;
    }
    if (logs.find("unknown model architecture") != std::string::npos) {
        std::fprintf(stderr, "qwen4exp was reported as unknown; logs:\n%s", logs.c_str());
        return 1;
    }

    return 0;
}
