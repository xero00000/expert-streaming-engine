// Optional integration test: requires the bounded production expert fixtures.
// Tests actual quantized MUL_MAT_ID kernels, not full-model quality or speed.
#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using nlohmann::json;

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

int main(int argc, char ** argv) {
    try {
        require(argc == 3, "usage: test-deepseek-v41-expert FIXTURE_DIRECTORY cpu|CUDA_DEVICE_NUMBER");
        std::string directory = argv[1];
        std::ifstream manifest(directory + "/fixture.json");
        require(bool(manifest), "missing expert manifest");
        const json fixture = json::parse(manifest);
        ggml_backend_t raw_backend = nullptr;
        if (std::string(argv[2]) == "cpu") {
            raw_backend = ggml_backend_cpu_init();
            if (raw_backend) ggml_backend_cpu_set_n_threads(raw_backend, 4);
        } else {
#ifdef GGML_USE_CUDA
            int device = std::stoi(argv[2]);
            require(device >= 0 && device < ggml_backend_cuda_get_device_count(), "invalid CUDA device");
            raw_backend = ggml_backend_cuda_init(device, nullptr, nullptr);
#else
            throw std::runtime_error("CUDA backend not compiled; refusing CPU fallback");
#endif
        }
        require(raw_backend != nullptr, "backend initialization failed");
        std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> backend(raw_backend, ggml_backend_free);
        json results = json::array();
        for (const auto & record : fixture.at("records")) {
            const int64_t k = record.at("shape").at(0), m = record.at("shape").at(1);
            const ggml_type type = ggml_type(record.at("type_id").get<int>());
            require((type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_Q2_K) && k > 0 && k <= 8192 && m > 0 && m <= 8192,
                    "unexpected expert fixture geometry");
            const size_t row_bytes = ggml_row_size(type, k);
            std::vector<uint8_t> weights(row_bytes * m);
            std::ifstream input(directory + "/" + record.at("file").get<std::string>(), std::ios::binary);
            input.read(reinterpret_cast<char *>(weights.data()), weights.size());
            require(bool(input) && input.peek() == std::char_traits<char>::eof(), "expert fixture length mismatch");
            auto to_float = ggml_internal_get_type_traits(type).to_float;
            require(to_float != nullptr, "missing reference dequantizer");
            for (int64_t n : {int64_t(1), int64_t(8)}) {
                ggml_init_params params{ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(32, false), nullptr, true};
                std::unique_ptr<ggml_context, decltype(&ggml_free)> context(ggml_init(params), ggml_free);
                require(bool(context), "graph context allocation failed");
                auto * weight = ggml_new_tensor_3d(context.get(), type, k, m, 1);
                auto * activation = ggml_new_tensor_3d(context.get(), GGML_TYPE_F32, k, 1, n);
                auto * ids = ggml_new_tensor_2d(context.get(), GGML_TYPE_I32, 1, n);
                auto * output = ggml_mul_mat_id(context.get(), weight, activation, ids);
                require(ggml_backend_supports_op(backend.get(), output), "backend cannot execute quantized expert operation");
                auto * graph = ggml_new_graph_custom(context.get(), 32, false);
                ggml_build_forward_expand(graph, output);
                auto * buffer = ggml_backend_alloc_ctx_tensors(context.get(), backend.get());
                require(buffer != nullptr, "backend allocation failed");
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> cleanup(buffer, ggml_backend_buffer_free);
                std::vector<float> x(k * n), expected(m * n), actual(m * n), decoded(k);
                std::vector<int32_t> routes(n, 0);
                for (size_t i = 0; i < x.size(); ++i) x[i] = std::sin(float(i + 1) * .731f) + .5f * std::cos(float(i + 17) * .137f);
                for (int64_t row = 0; row < m; ++row) {
                    to_float(weights.data() + row * row_bytes, decoded.data(), k);
                    for (int64_t token = 0; token < n; ++token) {
                        double sum = 0;
                        for (int64_t col = 0; col < k; ++col) sum += double(decoded[col]) * x[token * k + col];
                        expected[token * m + row] = float(sum);
                    }
                }
                ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
                ggml_backend_tensor_set(activation, x.data(), 0, x.size() * sizeof(float));
                ggml_backend_tensor_set(ids, routes.data(), 0, routes.size() * sizeof(int32_t));
                require(ggml_backend_graph_compute(backend.get(), graph) == GGML_STATUS_SUCCESS, "expert execution failed");
                ggml_backend_synchronize(backend.get());
                ggml_backend_tensor_get(output, actual.data(), 0, actual.size() * sizeof(float));
                double squared_error = 0, reference_norm = 0, actual_norm = 0, dot = 0;
                for (size_t i = 0; i < actual.size(); ++i) {
                    require(std::isfinite(actual[i]) && std::isfinite(expected[i]), "nonfinite expert output");
                    double error = double(actual[i]) - expected[i];
                    squared_error += error * error;
                    reference_norm += double(expected[i]) * expected[i];
                    actual_norm += double(actual[i]) * actual[i];
                    dot += double(actual[i]) * expected[i];
                }
                require(reference_norm > 0 && actual_norm > 0, "degenerate expert output");
                double rel_rms = std::sqrt(squared_error / reference_norm);
                double cosine = dot / std::sqrt(reference_norm * actual_norm);
                results.push_back({{"type", record.at("type")}, {"tokens", n}, {"relative_rms", rel_rms}, {"cosine", cosine}});
                require(rel_rms <= .01 && cosine >= .999, "expert kernel failed predeclared 1% RMS / .999 cosine gate");
            }
        }
        std::printf("%s\n", json{{"backend", ggml_backend_name(backend.get())}, {"scope", "real expert MUL_MAT_ID parity only"},
                                 {"results", results}}.dump(2).c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
