#include "common.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

struct route_capture {
    std::map<std::string, std::vector<int32_t>> tensors;
    std::map<std::string, std::vector<float>> float_tensors;
    std::vector<uint8_t> staging;
    bool invalid_type = false;
};

bool has_prefix(const char * name, const char * prefix) {
    return std::strncmp(name, prefix, std::strlen(prefix)) == 0;
}

bool is_layer_tensor(const char * name, const char * prefix) {
    const size_t length = std::strlen(prefix);
    if (std::strncmp(name, prefix, length) != 0 || name[length] != '-') {
        return false;
    }
    const char * suffix = name + length + 1;
    if (*suffix == '\0') return false;
    for (; *suffix != '\0'; ++suffix) {
        if (*suffix < '0' || *suffix > '9') return false;
    }
    return true;
}

bool filter_contains(const char * filter, const std::string & name) {
    if (filter == nullptr) {
        return false;
    }
    if (std::strcmp(filter, "1") == 0) {
        return true;
    }
    const std::string list(filter);
    size_t begin = 0;
    while (begin <= list.size()) {
        const size_t end = list.find(',', begin);
        if (list.substr(begin, end - begin) == name) {
            return true;
        }
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return false;
}

int capture_routes(ggml_tensor * tensor, bool ask, void * user_data) {
    // Hybrid graphs also expose ffn_moe_topk_gpu/cpu partition tensors.  The
    // canonical route probe compares the model's pre-partition top-k decision,
    // so accept only the exact layer-suffixed base names.
    const bool wanted_route = is_layer_tensor(tensor->name, "ffn_moe_topk") ||
        is_layer_tensor(tensor->name, "qwen4exp_qsa_top_blocks");
    const bool wanted_float = filter_contains(std::getenv("QWEN4EXP_FLOAT_TENSORS"), tensor->name);
    const bool wanted = wanted_route || wanted_float;
    if (ask || !wanted) {
        return wanted ? 1 : 0;
    }

    auto & capture = *static_cast<route_capture *>(user_data);
    const ggml_type expected_type = wanted_float ? GGML_TYPE_F32 : GGML_TYPE_I32;
    if (tensor->type != expected_type) {
        capture.invalid_type = true;
        return 0;
    }

    const uint8_t * data = static_cast<const uint8_t *>(tensor->data);
    if (!ggml_backend_buffer_is_host(tensor->buffer)) {
        capture.staging.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor, capture.staging.data(), 0, capture.staging.size());
        data = capture.staging.data();
    }

    std::vector<int32_t> * route_values = nullptr;
    std::vector<float> * float_values = nullptr;
    if (wanted_float) {
        float_values = &capture.float_tensors[tensor->name];
        float_values->reserve(ggml_nelements(tensor));
    } else {
        route_values = &capture.tensors[tensor->name];
        route_values->reserve(ggml_nelements(tensor));
    }
    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; ++i0) {
                    const size_t offset = size_t(i3) * tensor->nb[3] + size_t(i2) * tensor->nb[2] +
                        size_t(i1) * tensor->nb[1] + size_t(i0) * tensor->nb[0];
                    if (wanted_float) {
                        float value;
                        std::memcpy(&value, data + offset, sizeof(value));
                        float_values->push_back(value);
                    } else {
                        int32_t value;
                        std::memcpy(&value, data + offset, sizeof(value));
                        route_values->push_back(value);
                    }
                }
            }
        }
    }
    return 1;
}

void hash_byte(uint64_t & hash, uint8_t byte) {
    hash ^= byte;
    hash *= UINT64_C(1099511628211);
}

void hash_string(uint64_t & hash, const std::string & value) {
    for (unsigned char byte : value) {
        hash_byte(hash, byte);
    }
    hash_byte(hash, 0);
}

void hash_i32(uint64_t & hash, int32_t value) {
    const uint32_t bits = uint32_t(value);
    for (unsigned shift = 0; shift < 32; shift += 8) {
        hash_byte(hash, uint8_t(bits >> shift));
    }
}

} // namespace

int main(int argc, char ** argv) {
    gpt_params params;
    if (!gpt_params_parse(argc, argv, params)) {
        gpt_params_print_usage(argc, argv, params);
        return 1;
    }

    route_capture capture;
    params.cb_eval = capture_routes;
    params.cb_eval_user_data = &capture;
    params.warmup = false;

    llama_backend_init();
    llama_numa_init(params.numa);
    llama_init_result initialized = llama_init_from_gpt_params(params);
    if (initialized.model == nullptr || initialized.context == nullptr) {
        std::fprintf(stderr, "qwen4exp-route-probe: failed to initialize model\n");
        return 1;
    }

    const bool add_bos = llama_should_add_bos_token(initialized.model);
    std::vector<llama_token> tokens = common_tokenize(initialized.context, params.prompt, add_bos);
    if (const char * raw_tokens = std::getenv("QWEN4EXP_TOKEN_IDS")) {
        tokens.clear();
        const char * cursor = raw_tokens;
        do {
            char * end = nullptr;
            const long token = std::strtol(cursor, &end, 10);
            if (end == cursor || (*end != '\0' && *end != ',') || token < 0 ||
                    token >= llama_n_vocab(initialized.model) || tokens.size() >= 4096) {
                std::fprintf(stderr, "qwen4exp-route-probe: invalid QWEN4EXP_TOKEN_IDS\n");
                return 1;
            }
            tokens.push_back(llama_token(token));
            if (*end == '\0') break;
            cursor = end + 1;
        } while (true);
    }
    std::printf("qwen4exp_input_tokens: [");
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::printf("%s%d", i == 0 ? "" : ",", tokens[i]);
    }
    std::printf("]\n");
    if (tokens.empty() || llama_decode(initialized.context,
            llama_batch_get_one(tokens.data(), int32_t(tokens.size()), 0, 0)) != 0) {
        std::fprintf(stderr, "qwen4exp-route-probe: evaluation failed\n");
        return 1;
    }
    llama_synchronize(initialized.context);

    int repeat_count = 1;
    if (const char * repeat_env = std::getenv("QWEN4EXP_DECODE_REPEATS")) {
        char * end = nullptr;
        const long parsed = std::strtol(repeat_env, &end, 10);
        if (end == repeat_env || *end != '\0' || parsed < 1 || parsed > 1024) {
            std::fprintf(stderr, "qwen4exp-route-probe: invalid QWEN4EXP_DECODE_REPEATS\n");
            return 1;
        }
        repeat_count = int(parsed);
    }
    llama_token repeated_token = tokens.back();
    for (int repeat = 1; repeat < repeat_count; ++repeat) {
        if (llama_decode(initialized.context,
                llama_batch_get_one(&repeated_token, 1, int32_t(tokens.size()) + repeat - 1, 0)) != 0) {
            std::fprintf(stderr, "qwen4exp-route-probe: repeated evaluation failed at step %d\n", repeat);
            return 1;
        }
        llama_synchronize(initialized.context);
    }

    uint64_t digest = UINT64_C(14695981039346656037);
    uint64_t moe_digest = UINT64_C(14695981039346656037);
    uint64_t qsa_digest = UINT64_C(14695981039346656037);
    size_t value_count = 0;
    size_t moe_tensor_count = 0;
    size_t qsa_tensor_count = 0;
    const char * print_values = std::getenv("QWEN4EXP_ROUTE_VALUES");
    for (const auto & [name, values] : capture.tensors) {
        const bool is_moe = has_prefix(name.c_str(), "ffn_moe_topk");
        uint64_t & category_digest = is_moe ? moe_digest : qsa_digest;
        uint64_t tensor_digest = UINT64_C(14695981039346656037);
        hash_string(digest, name);
        hash_string(category_digest, name);
        hash_string(tensor_digest, name);
        for (int32_t value : values) {
            hash_i32(digest, value);
            hash_i32(category_digest, value);
            hash_i32(tensor_digest, value);
        }
        std::printf("qwen4exp_route_tensor: {\"name\":\"%s\",\"fnv1a64\":\"%016" PRIx64
                "\",\"values\":%zu}\n", name.c_str(), tensor_digest, values.size());
        if (filter_contains(print_values, name)) {
            std::printf("qwen4exp_route_values: {\"name\":\"%s\",\"values\":[", name.c_str());
            for (size_t i = 0; i < values.size(); ++i) {
                std::printf("%s%d", i == 0 ? "" : ",", values[i]);
            }
            std::printf("]}\n");
        }
        std::fflush(stdout);
        value_count += values.size();
        moe_tensor_count += is_moe;
        qsa_tensor_count += !is_moe;
    }

    std::printf(
        "qwen4exp_route_digest: {\"fnv1a64\":\"%016" PRIx64
        "\",\"moe_fnv1a64\":\"%016" PRIx64 "\",\"qsa_fnv1a64\":\"%016" PRIx64
        "\",\"tensors\":%zu,\"moe_tensors\":%zu,\"qsa_tensors\":%zu,\"values\":%zu}\n",
        digest, moe_digest, qsa_digest, capture.tensors.size(), moe_tensor_count, qsa_tensor_count, value_count);

    const char * print_float_values = std::getenv("QWEN4EXP_FLOAT_VALUES");
    for (const auto & [name, values] : capture.float_tensors) {
        uint64_t tensor_digest = UINT64_C(14695981039346656037);
        double sum = 0.0;
        double sum_sq = 0.0;
        float min_value = INFINITY;
        float max_value = -INFINITY;
        hash_string(tensor_digest, name);
        for (float value : values) {
            int32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            hash_i32(tensor_digest, bits);
            sum += value;
            sum_sq += double(value)*double(value);
            min_value = std::min(min_value, value);
            max_value = std::max(max_value, value);
        }
        const double mean = values.empty() ? 0.0 : sum/values.size();
        const double rms = values.empty() ? 0.0 : std::sqrt(sum_sq/values.size());
        std::printf("qwen4exp_float_tensor: {\"name\":\"%s\",\"fnv1a64\":\"%016" PRIx64
                "\",\"values\":%zu,\"mean\":%.9g,\"rms\":%.9g,\"min\":%.9g,\"max\":%.9g}\n",
                name.c_str(), tensor_digest, values.size(), mean, rms, min_value, max_value);
        if (filter_contains(print_float_values, name)) {
            std::printf("qwen4exp_float_values: {\"name\":\"%s\",\"values\":[", name.c_str());
            for (size_t i = 0; i < values.size(); ++i) {
                std::printf("%s%.9g", i == 0 ? "" : ",", values[i]);
            }
            std::printf("]}\n");
        }
    }

    llama_free(initialized.context);
    llama_free_model(initialized.model);
    llama_backend_free();

    if (capture.invalid_type || moe_tensor_count == 0 || value_count == 0) {
        std::fprintf(stderr, "qwen4exp-route-probe: route capture was incomplete\n");
        return 1;
    }
    return 0;
}
