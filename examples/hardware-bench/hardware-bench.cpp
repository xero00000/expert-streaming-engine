#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
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
        } else if ((arg == "--bytes" || arg == "--iterations" || arg == "--model") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "--bytes") result.bytes = parse_size(value);
            if (arg == "--iterations") result.iterations = std::stoi(value);
            if (arg == "--model") result.model = value;
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (!result.json) throw std::runtime_error("only --json output is currently supported");
    if (result.iterations < 3) throw std::runtime_error("--iterations must be at least 3");
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
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n  \"benchmark_source\": {\n"
                  << "    \"tool\": \"ese-hardware-bench\",\n"
                  << "    \"transfer_path\": \"ggml_backend_tensor_set/get\",\n"
                  << "    \"model\": \"" << json_escape(opts.model) << "\"\n"
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
        std::cout << "    \"cpu_moe\": {\"status\": \"unavailable\", \"reason\": \"production model probe not implemented\"},\n"
                  << "    \"expert_cache_upload\": {\"status\": \"unavailable\", \"reason\": \"production cache probe not implemented\"}\n"
                  << "  }\n}\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
