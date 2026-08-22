#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-ese.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct options {
    size_t bytes = 256ULL * 1024 * 1024;
    int iterations = 7;
    int threads = 0;
    bool json = false;
    std::string model;
};

size_t parse_size(const std::string & value) {
    size_t used = 0;
    const unsigned long long result = std::stoull(value, &used);
    if (used != value.size() || result == 0) {
        throw std::runtime_error("--bytes must be a positive integer");
    }
    return static_cast<size_t>(result);
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--json") {
            result.json = true;
        } else if ((arg == "--bytes" || arg == "--iterations" || arg == "--threads" || arg == "--model") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "--bytes") result.bytes = parse_size(value);
            if (arg == "--iterations") result.iterations = std::stoi(value);
            if (arg == "--threads") result.threads = std::stoi(value);
            if (arg == "--model") result.model = value;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (!result.json) throw std::runtime_error("only --json output is currently supported");
    if (result.iterations < 3) throw std::runtime_error("--iterations must be at least 3");
    if (result.threads < 0) throw std::runtime_error("--threads cannot be negative");
    return result;
}

double seconds_since(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2.0;
}

double gbps(size_t bytes, double seconds) {
    return static_cast<double>(bytes) / seconds / 1.0e9;
}

std::string json_escape(const std::string & value) {
    std::string escaped;
    for (const unsigned char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default:
                if (ch < 0x20) {
                    const char digits[] = "0123456789abcdef";
                    escaped += "\\u00";
                    escaped += digits[ch >> 4];
                    escaped += digits[ch & 0x0f];
                } else {
                    escaped += static_cast<char>(ch);
                }
        }
    }
    return escaped;
}

ggml_backend_t find_cuda_backend(std::string & name) {
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        const char * candidate = ggml_backend_reg_get_name(i);
        if (candidate && std::string(candidate).rfind("CUDA", 0) == 0) {
            ggml_backend_t backend = ggml_backend_reg_init_backend(i, nullptr);
            if (backend) {
                name = candidate;
                return backend;
            }
        }
    }
    return nullptr;
}

struct transfer_probe {
    ggml_backend_t backend = nullptr;
    ggml_context * context = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;

    transfer_probe() = default;
    transfer_probe(const transfer_probe &) = delete;
    transfer_probe & operator=(const transfer_probe &) = delete;
    transfer_probe(transfer_probe && other) noexcept
        : backend(other.backend), context(other.context), buffer(other.buffer), tensor(other.tensor) {
        other.backend = nullptr;
        other.context = nullptr;
        other.buffer = nullptr;
        other.tensor = nullptr;
    }

    ~transfer_probe() {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (context) ggml_free(context);
        if (backend) ggml_backend_free(backend);
    }
};

transfer_probe make_transfer_probe(size_t bytes, std::string & backend_name) {
    transfer_probe probe;
    probe.backend = find_cuda_backend(backend_name);
    if (!probe.backend) return probe;
    ggml_init_params params = { ggml_tensor_overhead() * 2, nullptr, true };
    probe.context = ggml_init(params);
    if (!probe.context) throw std::runtime_error("could not create GGML transfer context");
    probe.tensor = ggml_new_tensor_1d(probe.context, GGML_TYPE_I8, static_cast<int64_t>(bytes));
    probe.buffer = ggml_backend_alloc_ctx_tensors(probe.context, probe.backend);
    if (!probe.buffer) throw std::runtime_error("could not allocate CUDA transfer buffer");
    return probe;
}

struct cpu_moe_result {
    int64_t input_width = 0;
    int64_t expert_width = 0;
    int64_t routed_experts = 0;
    int threads = 0;
    double latency_ms_median = 0;
    double effective_gbps_median = 0;
    double checksum = 0;
};

cpu_moe_result measure_cpu_moe(int iterations, int requested_threads) {
    constexpr int64_t k = 512;
    constexpr int64_t m = 1024;
    constexpr int64_t n_used = 2;
    constexpr int64_t stored_experts = 2;

    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) throw std::runtime_error("could not initialize the GGML CPU backend");
    struct backend_cleanup {
        ggml_backend_t value;
        ~backend_cleanup() { ggml_backend_free(value); }
    } free_backend{backend};
    // A memory-bound expert kernel should use physical cores, not SMT siblings.
    // The portable fallback cannot query topology, so use half of the logical
    // concurrency here; the persisted identity retains the exact CPU topology.
    const int threads = requested_threads > 0
            ? requested_threads
            : std::max(1u, std::thread::hardware_concurrency() / 2);

    ggml_init_params params = {
        ggml_tensor_overhead() * 20 + ggml_graph_overhead_custom(16, false), nullptr, true
    };
    ggml_context * context = ggml_init(params);
    if (!context) throw std::runtime_error("could not create CPU MoE benchmark context");
    struct context_cleanup {
        ggml_context * value;
        ~context_cleanup() { ggml_free(value); }
    } free_context{context};

    ggml_tensor * weights = ggml_new_tensor_3d(context, GGML_TYPE_F16, k, m, stored_experts);
    ggml_tensor * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, k, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_mul_mat_id(context, weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) throw std::runtime_error("could not allocate CPU MoE benchmark tensors");
    struct buffer_cleanup {
        ggml_backend_buffer_t value;
        ~buffer_cleanup() { ggml_backend_buffer_free(value); }
    } free_buffer{buffer};

    std::vector<ggml_fp16_t> expert_weights(static_cast<size_t>(k * m * stored_experts));
    std::vector<float> inputs(static_cast<size_t>(k * n_used));
    for (size_t i = 0; i < expert_weights.size(); ++i) {
        expert_weights[i] = ggml_fp32_to_fp16(float(int((i * 17 + i / 31) % 257) - 128) / 128.0f);
    }
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = float(int((i * 13) % 97) - 48) / 48.0f;
    }
    const int32_t route[n_used] = {0, 1};
    ggml_backend_tensor_set(weights, expert_weights.data(), 0, expert_weights.size() * sizeof(expert_weights[0]));
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(inputs[0]));
    ggml_backend_tensor_set(ids, route, 0, sizeof(route));

    ggml_backend_cpu_set_n_threads(backend, 1);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML single-thread CPU MoE reference failed");
    }
    std::vector<float> reference(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, reference.data(), 0, reference.size() * sizeof(reference[0]));

    ggml_backend_cpu_set_n_threads(backend, threads);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("GGML CPU MoE warmup failed");
    }
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        const auto start = clock_type::now();
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GGML CPU MoE benchmark failed");
        }
        ggml_backend_synchronize(backend);
        samples.push_back(seconds_since(start));
    }
    std::vector<float> result(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(result[0]));
    double checksum = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        const float value = result[i];
        if (!std::isfinite(value)) throw std::runtime_error("GGML CPU MoE produced a non-finite value");
        if (std::abs(value - reference[i]) > 1.0e-5f) {
            throw std::runtime_error("GGML CPU MoE failed single-thread output parity");
        }
        checksum += value;
    }
    if (!std::isfinite(checksum) || checksum == 0) {
        throw std::runtime_error("GGML CPU MoE produced an invalid checksum");
    }
    const double elapsed = median(samples);
    const size_t weight_bytes = expert_weights.size() * sizeof(ggml_fp16_t);
    return {k, m, n_used, threads, elapsed * 1000.0, gbps(weight_bytes, elapsed), checksum};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        llama_backend_init();
        struct backend_runtime_cleanup {
            ~backend_runtime_cleanup() { llama_backend_free(); }
        } backend_runtime;

        std::vector<uint8_t> source(opts.bytes, 0x5a);
        std::vector<uint8_t> destination(opts.bytes, 0);
        std::vector<double> host_samples;
        for (int i = 0; i < opts.iterations; ++i) {
            const auto start = clock_type::now();
            std::memcpy(destination.data(), source.data(), opts.bytes);
            host_samples.push_back(gbps(opts.bytes, seconds_since(start)));
            source[static_cast<size_t>(i) % source.size()] ^= destination[static_cast<size_t>(i) % destination.size()];
        }

        std::string backend_name;
        transfer_probe transfer = make_transfer_probe(opts.bytes, backend_name);
        std::vector<double> h2d_samples;
        std::vector<double> d2h_samples;
        std::vector<double> contention_h2d_samples;
        std::vector<double> expert_cache_upload_samples;
        if (transfer.backend) {
            ggml_backend_tensor_set(transfer.tensor, source.data(), 0, opts.bytes);
            ggml_backend_synchronize(transfer.backend);
            for (int i = 0; i < opts.iterations; ++i) {
                auto start = clock_type::now();
                ggml_backend_tensor_set(transfer.tensor, source.data(), 0, opts.bytes);
                ggml_backend_synchronize(transfer.backend);
                h2d_samples.push_back(gbps(opts.bytes, seconds_since(start)));

                start = clock_type::now();
                ggml_backend_tensor_get(transfer.tensor, destination.data(), 0, opts.bytes);
                ggml_backend_synchronize(transfer.backend);
                d2h_samples.push_back(gbps(opts.bytes, seconds_since(start)));

                std::thread host_copy([&]() {
                    std::memcpy(destination.data(), source.data(), opts.bytes);
                });
                start = clock_type::now();
                ggml_backend_tensor_set(transfer.tensor, source.data(), 0, opts.bytes);
                ggml_backend_synchronize(transfer.backend);
                contention_h2d_samples.push_back(gbps(opts.bytes, seconds_since(start)));
                host_copy.join();

                start = clock_type::now();
                ggml_backend_expert_cache_upload_async(
                        transfer.backend, transfer.tensor, source.data(), 0, opts.bytes);
                ggml_backend_synchronize(transfer.backend);
                expert_cache_upload_samples.push_back(gbps(opts.bytes, seconds_since(start)));
            }
        }
        const cpu_moe_result cpu_moe = measure_cpu_moe(opts.iterations, opts.threads);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n  \"benchmark_source\": {\n"
                  << "    \"tool\": \"ese-hardware-bench\",\n"
                  << "    \"calibration_level\": \"baseline\",\n"
                  << "    \"planner_ready\": false,\n"
                  << "    \"transfer_path\": \"ggml_backend_tensor_set/get\",\n"
                  << "    \"model_requested\": \"" << json_escape(opts.model) << "\",\n"
                  << "    \"model_used\": false\n"
                  << "  },\n  \"measurements\": {\n"
                  << "    \"host_memory\": {\"copy_gbps_median\": " << median(host_samples)
                  << ", \"bytes\": " << opts.bytes << ", \"iterations\": " << opts.iterations << "},\n";
        if (transfer.backend) {
            std::cout << "    \"gpu_transfer\": {\"status\": \"measured\", \"backend\": \""
                      << json_escape(backend_name) << "\", \"h2d_gbps_median\": " << median(h2d_samples)
                      << ", \"d2h_gbps_median\": " << median(d2h_samples)
                      << ", \"contended_h2d_gbps_median\": " << median(contention_h2d_samples) << "},\n";
        } else {
            std::cout << "    \"gpu_transfer\": {\"status\": \"unavailable\", \"reason\": \"no CUDA backend\"},\n";
        }
        std::cout << "    \"cpu_moe\": {\"status\": \"measured\", \"operation\": \"ggml_mul_mat_id\", "
                  << "\"ggml_type\": \"f16\", \"input_width\": " << cpu_moe.input_width
                  << ", \"expert_width\": " << cpu_moe.expert_width
                  << ", \"routed_experts\": " << cpu_moe.routed_experts
                  << ", \"threads\": " << cpu_moe.threads
                  << ", \"latency_ms_median\": " << cpu_moe.latency_ms_median
                  << ", \"effective_gbps_median\": " << cpu_moe.effective_gbps_median
                  << ", \"correctness\": \"single-thread-parity\", \"checksum\": " << cpu_moe.checksum << "},\n"
                  << "    \"expert_cache_upload\": {";
        if (transfer.backend) {
            std::cout << "\"status\": \"measured\", \"path\": \"ggml_backend_expert_cache_upload_async\", "
                      << "\"gbps_median\": " << median(expert_cache_upload_samples)
                      << ", \"bytes_per_expert_component\": " << opts.bytes;
        } else {
            std::cout << "\"status\": \"unavailable\", \"reason\": \"no CUDA backend\"";
        }
        std::cout << "}\n"
                  << "  }\n}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
