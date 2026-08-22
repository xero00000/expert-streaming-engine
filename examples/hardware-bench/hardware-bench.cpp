#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-ese.h"
#include "llama.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

struct model_expert_spec {
    int type = -1;
    int64_t input_width = 0;
    int64_t expert_width = 0;
    int64_t expert_count = 0;
    size_t bytes = 0;
    uint64_t data_offset = 0;
    std::string source;
};

struct options {
    size_t bytes = 256ULL * 1024 * 1024;
    int iterations = 7;
    int threads = 0;
    bool json = false;
    std::string model;
    std::vector<model_expert_spec> model_experts;
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
        } else if ((arg == "--bytes" || arg == "--iterations" || arg == "--threads" ||
                    arg == "--model" || arg == "--model-expert-spec") && i + 1 < argc) {
            const std::string value = argv[++i];
            if (arg == "--bytes") result.bytes = parse_size(value);
            if (arg == "--iterations") result.iterations = std::stoi(value);
            if (arg == "--threads") result.threads = std::stoi(value);
            if (arg == "--model") result.model = value;
            if (arg == "--model-expert-spec") {
                model_expert_spec spec;
                size_t start = 0;
                std::vector<std::string> fields;
                for (size_t end = value.find('|');; end = value.find('|', start)) {
                    if (end == std::string::npos) { fields.push_back(value.substr(start)); break; }
                    fields.push_back(value.substr(start, end - start));
                    start = end + 1;
                }
                if (fields.size() != 7) throw std::runtime_error("invalid --model-expert-spec");
                spec.type = std::stoi(fields[0]);
                spec.input_width = std::stoll(fields[1]);
                spec.expert_width = std::stoll(fields[2]);
                spec.expert_count = std::stoll(fields[3]);
                spec.bytes = parse_size(fields[4]);
                spec.data_offset = std::stoull(fields[5]);
                spec.source = fields[6];
                result.model_experts.push_back(spec);
            }
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (!result.json) throw std::runtime_error("only --json output is currently supported");
    if (result.iterations < 3) throw std::runtime_error("--iterations must be at least 3");
    if (result.threads < 0) throw std::runtime_error("--threads cannot be negative");
    if (!result.model.empty() && result.model_experts.empty()) {
        throw std::runtime_error("model-backed calibration requires complete expert geometry");
    }
    for (const auto & spec : result.model_experts) {
        if (spec.type < 0 || spec.input_width <= 0 || spec.expert_width <= 0 ||
                spec.expert_count <= 0 || spec.bytes == 0 || spec.source.empty()) {
            throw std::runtime_error("model-backed calibration contains invalid expert geometry");
        }
    }
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

ggml_backend_t find_cuda_backend(std::string & name, size_t ordinal = 0) {
    size_t found = 0;
    for (size_t i = 0; i < ggml_backend_reg_get_count(); ++i) {
        const char * candidate = ggml_backend_reg_get_name(i);
        if (candidate && std::string(candidate).rfind("CUDA", 0) == 0) {
            if (found++ != ordinal) continue;
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

transfer_probe make_transfer_probe(size_t bytes, std::string & backend_name, size_t ordinal = 0) {
    transfer_probe probe;
    probe.backend = find_cuda_backend(backend_name, ordinal);
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

struct device_transfer_result {
    std::string backend;
    double h2d_gbps = 0;
    double d2h_gbps = 0;
    double contended_h2d_gbps = 0;
    std::vector<double> expert_upload_gbps;
};

struct device_contention_result {
    std::string backend;
    std::vector<cpu_moe_result> cpu;
    std::vector<double> upload_gbps;
};

cpu_moe_result measure_cpu_moe(
        int iterations, int requested_threads,
        const std::function<double()> & concurrent_work = {},
        std::vector<double> * concurrent_samples = nullptr) {
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
        std::atomic<bool> worker_ready{false};
        std::atomic<bool> start_work{false};
        double concurrent_seconds = 0;
        std::thread worker;
        if (concurrent_work) {
            worker = std::thread([&]() {
                worker_ready.store(true, std::memory_order_release);
                while (!start_work.load(std::memory_order_acquire)) std::this_thread::yield();
                concurrent_seconds = concurrent_work();
            });
            while (!worker_ready.load(std::memory_order_acquire)) std::this_thread::yield();
        }
        const auto start = clock_type::now();
        start_work.store(true, std::memory_order_release);
        const ggml_status compute_status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        samples.push_back(seconds_since(start));
        if (worker.joinable()) {
            worker.join();
            if (concurrent_samples) concurrent_samples->push_back(concurrent_seconds);
        }
        if (compute_status != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("GGML CPU MoE benchmark failed");
        }
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

cpu_moe_result measure_model_cpu_moe(
        const model_expert_spec & spec, int iterations, int requested_threads,
        const std::function<double()> & concurrent_work = {},
        std::vector<double> * concurrent_samples = nullptr) {
    constexpr int64_t n_used = 2;
    constexpr int64_t stored_experts = 2;
    const ggml_type type = static_cast<ggml_type>(spec.type);
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) throw std::runtime_error("could not initialize model CPU MoE backend");
    struct backend_cleanup { ggml_backend_t value; ~backend_cleanup() { ggml_backend_free(value); } } free_backend{backend};
    const int threads = requested_threads > 0
            ? requested_threads : std::max(1u, std::thread::hardware_concurrency() / 2);

    ggml_init_params params = {
        ggml_tensor_overhead() * 20 + ggml_graph_overhead_custom(16, false), nullptr, true
    };
    ggml_context * context = ggml_init(params);
    if (!context) throw std::runtime_error("could not create model CPU MoE context");
    struct context_cleanup { ggml_context * value; ~context_cleanup() { ggml_free(value); } } free_context{context};

    ggml_tensor * weights = ggml_new_tensor_3d(
            context, type, spec.input_width, spec.expert_width, stored_experts);
    ggml_tensor * input = ggml_new_tensor_3d(context, GGML_TYPE_F32, spec.input_width, n_used, 1);
    ggml_tensor * ids = ggml_new_tensor_2d(context, GGML_TYPE_I32, n_used, 1);
    ggml_tensor * output = ggml_mul_mat_id(context, weights, input, ids);
    ggml_cgraph * graph = ggml_new_graph_custom(context, 16, false);
    ggml_build_forward_expand(graph, output);
    const size_t expected_bytes = ggml_nbytes(weights);
    if (expected_bytes != spec.bytes * stored_experts) {
        std::ostringstream message;
        message << "GGUF expert byte geometry mismatch for " << spec.source
                << ": metadata span implies " << spec.bytes * stored_experts
                << " bytes but GGML expects " << expected_bytes;
        throw std::runtime_error(message.str());
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(context, backend);
    if (!buffer) throw std::runtime_error("could not allocate model CPU MoE tensors");
    struct buffer_cleanup { ggml_backend_buffer_t value; ~buffer_cleanup() { ggml_backend_buffer_free(value); } } free_buffer{buffer};

    std::vector<uint8_t> expert_weights(expected_bytes);
    std::ifstream source(spec.source, std::ios::binary);
    if (!source) throw std::runtime_error("could not open GGUF expert source: " + spec.source);
    source.seekg(static_cast<std::streamoff>(spec.data_offset));
    source.read(reinterpret_cast<char *>(expert_weights.data()), static_cast<std::streamsize>(expert_weights.size()));
    if (!source || static_cast<size_t>(source.gcount()) != expert_weights.size()) {
        throw std::runtime_error("could not read complete GGUF expert payload: " + spec.source);
    }
    std::vector<float> inputs(static_cast<size_t>(spec.input_width * n_used));
    for (size_t i = 0; i < inputs.size(); ++i) {
        inputs[i] = float(int((i * 13) % 97) - 48) / 48.0f;
    }
    const int32_t route[n_used] = {0, 1};
    ggml_backend_tensor_set(weights, expert_weights.data(), 0, expert_weights.size());
    ggml_backend_tensor_set(input, inputs.data(), 0, inputs.size() * sizeof(inputs[0]));
    ggml_backend_tensor_set(ids, route, 0, sizeof(route));

    ggml_backend_cpu_set_n_threads(backend, 1);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("single-thread model CPU MoE reference failed");
    }
    std::vector<float> reference(static_cast<size_t>(ggml_nelements(output)));
    ggml_backend_tensor_get(output, reference.data(), 0, reference.size() * sizeof(reference[0]));
    ggml_backend_cpu_set_n_threads(backend, threads);
    if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("model CPU MoE warmup failed");
    }
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        std::atomic<bool> worker_ready{false};
        std::atomic<bool> start_work{false};
        double concurrent_seconds = 0;
        std::thread worker;
        if (concurrent_work) {
            worker = std::thread([&]() {
                worker_ready.store(true, std::memory_order_release);
                while (!start_work.load(std::memory_order_acquire)) std::this_thread::yield();
                concurrent_seconds = concurrent_work();
            });
            while (!worker_ready.load(std::memory_order_acquire)) std::this_thread::yield();
        }
        const auto start = clock_type::now();
        start_work.store(true, std::memory_order_release);
        const ggml_status compute_status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_synchronize(backend);
        samples.push_back(seconds_since(start));
        if (worker.joinable()) {
            worker.join();
            if (concurrent_samples) concurrent_samples->push_back(concurrent_seconds);
        }
        if (compute_status != GGML_STATUS_SUCCESS) throw std::runtime_error("model CPU MoE benchmark failed");
    }
    std::vector<float> result(reference.size());
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(result[0]));
    double checksum = 0;
    for (size_t i = 0; i < result.size(); ++i) {
        if (!std::isfinite(result[i]) || std::abs(result[i] - reference[i]) > 1.0e-4f) {
            throw std::runtime_error("model CPU MoE failed single-thread output parity");
        }
        checksum += result[i];
    }
    if (!std::isfinite(checksum)) throw std::runtime_error("model CPU MoE produced invalid checksum");
    const double elapsed = median(samples);
    return {spec.input_width, spec.expert_width, n_used, threads,
            elapsed * 1000.0, gbps(expected_bytes, elapsed), checksum};
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const options opts = parse_options(argc, argv);
        llama_backend_init();
        struct backend_runtime_cleanup {
            ~backend_runtime_cleanup() { llama_backend_free(); }
        } backend_runtime;

        size_t expert_upload_bytes = opts.bytes;
        size_t representative_index = 0;
        for (size_t i = 0; i < opts.model_experts.size(); ++i) {
            if (opts.model_experts[i].bytes > expert_upload_bytes || i == 0) {
                expert_upload_bytes = opts.model_experts[i].bytes;
                representative_index = i;
            }
        }
        const size_t allocation_bytes = std::max(opts.bytes, expert_upload_bytes);
        std::vector<uint8_t> source(allocation_bytes, 0x5a);
        std::vector<uint8_t> destination(allocation_bytes, 0);
        std::vector<double> host_samples;
        for (int i = 0; i < opts.iterations; ++i) {
            const auto start = clock_type::now();
            std::memcpy(destination.data(), source.data(), opts.bytes);
            host_samples.push_back(gbps(opts.bytes, seconds_since(start)));
            source[static_cast<size_t>(i) % source.size()] ^= destination[static_cast<size_t>(i) % destination.size()];
        }

        std::string backend_name;
        transfer_probe transfer = make_transfer_probe(allocation_bytes, backend_name);
        std::vector<double> h2d_samples;
        std::vector<double> d2h_samples;
        std::vector<double> contention_h2d_samples;
        std::vector<double> expert_cache_upload_samples;
        std::vector<std::vector<double>> model_expert_upload_samples(opts.model_experts.size());
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

                if (opts.model_experts.empty()) {
                    start = clock_type::now();
                    ggml_backend_expert_cache_upload_async(
                            transfer.backend, transfer.tensor, source.data(), 0, expert_upload_bytes);
                    ggml_backend_synchronize(transfer.backend);
                    expert_cache_upload_samples.push_back(gbps(expert_upload_bytes, seconds_since(start)));
                } else {
                    for (size_t spec_index = 0; spec_index < opts.model_experts.size(); ++spec_index) {
                        const auto & spec = opts.model_experts[spec_index];
                        start = clock_type::now();
                        ggml_backend_expert_cache_upload_async(
                                transfer.backend, transfer.tensor, source.data(), 0, spec.bytes);
                        ggml_backend_synchronize(transfer.backend);
                        model_expert_upload_samples[spec_index].push_back(gbps(spec.bytes, seconds_since(start)));
                    }
                }
            }
        }
        std::vector<device_transfer_result> device_transfers;
        if (transfer.backend) {
            device_transfer_result primary;
            primary.backend = backend_name;
            primary.h2d_gbps = median(h2d_samples);
            primary.d2h_gbps = median(d2h_samples);
            primary.contended_h2d_gbps = median(contention_h2d_samples);
            if (opts.model_experts.empty()) {
                primary.expert_upload_gbps.push_back(median(expert_cache_upload_samples));
            } else {
                for (const auto & samples : model_expert_upload_samples) {
                    primary.expert_upload_gbps.push_back(median(samples));
                }
            }
            device_transfers.push_back(std::move(primary));

            for (size_t ordinal = 1;; ++ordinal) {
                std::string extra_name;
                transfer_probe extra = make_transfer_probe(allocation_bytes, extra_name, ordinal);
                if (!extra.backend) break;
                std::vector<double> extra_h2d;
                std::vector<double> extra_d2h;
                std::vector<double> extra_contended;
                std::vector<std::vector<double>> extra_uploads(
                        std::max<size_t>(1, opts.model_experts.size()));
                ggml_backend_tensor_set(extra.tensor, source.data(), 0, opts.bytes);
                ggml_backend_synchronize(extra.backend);
                for (int i = 0; i < opts.iterations; ++i) {
                    auto start = clock_type::now();
                    ggml_backend_tensor_set(extra.tensor, source.data(), 0, opts.bytes);
                    ggml_backend_synchronize(extra.backend);
                    extra_h2d.push_back(gbps(opts.bytes, seconds_since(start)));
                    start = clock_type::now();
                    ggml_backend_tensor_get(extra.tensor, destination.data(), 0, opts.bytes);
                    ggml_backend_synchronize(extra.backend);
                    extra_d2h.push_back(gbps(opts.bytes, seconds_since(start)));
                    std::thread host_copy([&]() { std::memcpy(destination.data(), source.data(), opts.bytes); });
                    start = clock_type::now();
                    ggml_backend_tensor_set(extra.tensor, source.data(), 0, opts.bytes);
                    ggml_backend_synchronize(extra.backend);
                    extra_contended.push_back(gbps(opts.bytes, seconds_since(start)));
                    host_copy.join();
                    if (opts.model_experts.empty()) {
                        start = clock_type::now();
                        ggml_backend_expert_cache_upload_async(
                                extra.backend, extra.tensor, source.data(), 0, expert_upload_bytes);
                        ggml_backend_synchronize(extra.backend);
                        extra_uploads[0].push_back(gbps(expert_upload_bytes, seconds_since(start)));
                    } else {
                        for (size_t j = 0; j < opts.model_experts.size(); ++j) {
                            start = clock_type::now();
                            ggml_backend_expert_cache_upload_async(
                                    extra.backend, extra.tensor, source.data(), 0, opts.model_experts[j].bytes);
                            ggml_backend_synchronize(extra.backend);
                            extra_uploads[j].push_back(gbps(opts.model_experts[j].bytes, seconds_since(start)));
                        }
                    }
                }
                device_transfer_result measured;
                measured.backend = extra_name;
                measured.h2d_gbps = median(extra_h2d);
                measured.d2h_gbps = median(extra_d2h);
                measured.contended_h2d_gbps = median(extra_contended);
                for (const auto & samples : extra_uploads) measured.expert_upload_gbps.push_back(median(samples));
                device_transfers.push_back(std::move(measured));
            }
        }
        const cpu_moe_result cpu_moe = measure_cpu_moe(opts.iterations, opts.threads);
        std::vector<cpu_moe_result> model_cpu_moe;
        model_cpu_moe.reserve(opts.model_experts.size());
        for (const auto & spec : opts.model_experts) {
            model_cpu_moe.push_back(measure_model_cpu_moe(spec, opts.iterations, opts.threads));
        }
        cpu_moe_result contended_cpu_moe;
        std::vector<double> contended_expert_upload_seconds;
        std::vector<cpu_moe_result> contended_model_cpu_moe;
        std::vector<std::vector<double>> contended_model_upload_seconds(opts.model_experts.size());
        if (transfer.backend) {
            contended_cpu_moe = measure_cpu_moe(
                opts.iterations, opts.threads,
                [&]() {
                    const auto start = clock_type::now();
                    ggml_backend_expert_cache_upload_async(
                            transfer.backend, transfer.tensor, source.data(), 0, expert_upload_bytes);
                    ggml_backend_synchronize(transfer.backend);
                    return seconds_since(start);
                },
                &contended_expert_upload_seconds);
            for (size_t i = 0; i < opts.model_experts.size(); ++i) {
                const auto & spec = opts.model_experts[i];
                contended_model_cpu_moe.push_back(measure_model_cpu_moe(
                    spec, opts.iterations, opts.threads,
                    [&]() {
                        const auto start = clock_type::now();
                        ggml_backend_expert_cache_upload_async(
                                transfer.backend, transfer.tensor, source.data(), 0, spec.bytes);
                        ggml_backend_synchronize(transfer.backend);
                        return seconds_since(start);
                    },
                    &contended_model_upload_seconds[i]));
            }
        }
        std::vector<device_contention_result> device_contentions;
        if (transfer.backend) {
            device_contention_result primary;
            primary.backend = backend_name;
            if (opts.model_experts.empty()) {
                primary.cpu.push_back(contended_cpu_moe);
                primary.upload_gbps.push_back(
                        gbps(expert_upload_bytes, median(contended_expert_upload_seconds)));
            } else {
                primary.cpu = contended_model_cpu_moe;
                for (size_t i = 0; i < opts.model_experts.size(); ++i) {
                    primary.upload_gbps.push_back(
                            gbps(opts.model_experts[i].bytes, median(contended_model_upload_seconds[i])));
                }
            }
            device_contentions.push_back(std::move(primary));
            for (size_t ordinal = 1;; ++ordinal) {
                std::string extra_name;
                transfer_probe extra = make_transfer_probe(allocation_bytes, extra_name, ordinal);
                if (!extra.backend) break;
                device_contention_result measured;
                measured.backend = extra_name;
                const size_t count = std::max<size_t>(1, opts.model_experts.size());
                for (size_t i = 0; i < count; ++i) {
                    const size_t bytes = opts.model_experts.empty() ? expert_upload_bytes : opts.model_experts[i].bytes;
                    std::vector<double> upload_seconds;
                    auto concurrent = [&]() {
                        const auto start = clock_type::now();
                        ggml_backend_expert_cache_upload_async(
                                extra.backend, extra.tensor, source.data(), 0, bytes);
                        ggml_backend_synchronize(extra.backend);
                        return seconds_since(start);
                    };
                    measured.cpu.push_back(opts.model_experts.empty()
                            ? measure_cpu_moe(opts.iterations, opts.threads, concurrent, &upload_seconds)
                            : measure_model_cpu_moe(
                                    opts.model_experts[i], opts.iterations, opts.threads,
                                    concurrent, &upload_seconds));
                    measured.upload_gbps.push_back(gbps(bytes, median(upload_seconds)));
                }
                device_contentions.push_back(std::move(measured));
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n  \"benchmark_source\": {\n"
                  << "    \"tool\": \"ese-hardware-bench\",\n"
                  << "    \"calibration_level\": \"baseline\",\n"
                  << "    \"planner_ready\": false,\n"
                  << "    \"transfer_path\": \"ggml_backend_tensor_set/get\",\n"
                  << "    \"model_requested\": \"" << json_escape(opts.model) << "\",\n"
                  << "    \"model_used\": " << (!opts.model_experts.empty() ? "true" : "false") << ",\n"
                  << "    \"model_usage\": "
                  << (!opts.model_experts.empty() ? "[\"expert_cache_upload_geometry\"]" : "[]") << "\n"
                  << "  },\n  \"measurements\": {\n"
                  << "    \"host_memory\": {\"copy_gbps_median\": " << median(host_samples)
                  << ", \"bytes\": " << opts.bytes << ", \"iterations\": " << opts.iterations << "},\n";
        if (transfer.backend) {
            std::cout << "    \"gpu_transfer\": {\"status\": \"measured\", \"backend\": \""
                      << json_escape(backend_name) << "\", \"h2d_gbps_median\": " << median(h2d_samples)
                      << ", \"d2h_gbps_median\": " << median(d2h_samples)
                      << ", \"contended_h2d_gbps_median\": " << median(contention_h2d_samples)
                      << ", \"devices\": [";
            for (size_t i = 0; i < device_transfers.size(); ++i) {
                if (i) std::cout << ", ";
                const auto & device = device_transfers[i];
                std::cout << "{\"backend\": \"" << json_escape(device.backend)
                          << "\", \"h2d_gbps_median\": " << device.h2d_gbps
                          << ", \"d2h_gbps_median\": " << device.d2h_gbps
                          << ", \"contended_h2d_gbps_median\": " << device.contended_h2d_gbps
                          << ", \"expert_upload_profiles\": [";
                for (size_t j = 0; j < device.expert_upload_gbps.size(); ++j) {
                    if (j) std::cout << ", ";
                    std::cout << "{";
                    if (!opts.model_experts.empty()) {
                        const auto & spec = opts.model_experts[j];
                        std::cout << "\"ggml_type_id\": " << spec.type
                                  << ", \"ggml_type\": \"" << ggml_type_name(static_cast<ggml_type>(spec.type))
                                  << "\", \"input_width\": " << spec.input_width
                                  << ", \"expert_width\": " << spec.expert_width
                                  << ", \"bytes_per_expert_component\": " << spec.bytes << ", ";
                    } else {
                        std::cout << "\"bytes\": " << expert_upload_bytes << ", ";
                    }
                    std::cout << "\"gbps_median\": " << device.expert_upload_gbps[j] << "}";
                }
                std::cout << "]}";
            }
            std::cout << "]},\n";
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
                  << ", \"correctness\": \"single-thread-parity\", \"checksum\": " << cpu_moe.checksum;
        if (!model_cpu_moe.empty()) {
            std::cout << ", \"model_profiles\": [";
            for (size_t i = 0; i < model_cpu_moe.size(); ++i) {
                if (i) std::cout << ", ";
                const auto & spec = opts.model_experts[i];
                const auto & result = model_cpu_moe[i];
                std::cout << "{\"ggml_type_id\": " << spec.type
                          << ", \"ggml_type\": \"" << ggml_type_name(static_cast<ggml_type>(spec.type))
                          << "\", \"input_width\": " << spec.input_width
                          << ", \"expert_width\": " << spec.expert_width
                          << ", \"expert_count\": " << spec.expert_count
                          << ", \"bytes_read\": " << spec.bytes * 2
                          << ", \"latency_ms_median\": " << result.latency_ms_median
                          << ", \"effective_gbps_median\": " << result.effective_gbps_median
                          << ", \"correctness\": \"single-thread-parity\", \"checksum\": "
                          << result.checksum << "}";
            }
            std::cout << "]";
        }
        std::cout << "},\n"
                  << "    \"cpu_cache_contention\": {";
        if (transfer.backend) {
            std::cout << "\"status\": \"measured\", \"cpu_operation\": \"ggml_mul_mat_id\", "
                      << "\"cpu_latency_ms_median\": " << contended_cpu_moe.latency_ms_median
                      << ", \"cpu_effective_gbps_median\": " << contended_cpu_moe.effective_gbps_median
                      << ", \"upload_gbps_median\": "
                      << gbps(expert_upload_bytes, median(contended_expert_upload_seconds))
                      << ", \"bytes_per_expert_component\": " << expert_upload_bytes;
            if (!contended_model_cpu_moe.empty()) {
                std::cout << ", \"model_profiles\": [";
                for (size_t i = 0; i < contended_model_cpu_moe.size(); ++i) {
                    if (i) std::cout << ", ";
                    const auto & spec = opts.model_experts[i];
                    const auto & result = contended_model_cpu_moe[i];
                    std::cout << "{\"ggml_type_id\": " << spec.type
                              << ", \"ggml_type\": \"" << ggml_type_name(static_cast<ggml_type>(spec.type))
                              << "\", \"input_width\": " << spec.input_width
                              << ", \"expert_width\": " << spec.expert_width
                              << ", \"expert_count\": " << spec.expert_count
                              << ", \"bytes_per_expert_component\": " << spec.bytes
                              << ", \"cpu_latency_ms_median\": " << result.latency_ms_median
                              << ", \"cpu_effective_gbps_median\": " << result.effective_gbps_median
                              << ", \"upload_gbps_median\": "
                              << gbps(spec.bytes, median(contended_model_upload_seconds[i])) << "}";
                }
                std::cout << "]";
            }
            std::cout << ", \"devices\": [";
            for (size_t i = 0; i < device_contentions.size(); ++i) {
                if (i) std::cout << ", ";
                const auto & device = device_contentions[i];
                std::cout << "{\"backend\": \"" << json_escape(device.backend) << "\", \"profiles\": [";
                for (size_t j = 0; j < device.cpu.size(); ++j) {
                    if (j) std::cout << ", ";
                    const auto & result = device.cpu[j];
                    std::cout << "{";
                    if (!opts.model_experts.empty()) {
                        const auto & spec = opts.model_experts[j];
                        std::cout << "\"ggml_type_id\": " << spec.type
                                  << ", \"ggml_type\": \"" << ggml_type_name(static_cast<ggml_type>(spec.type))
                                  << "\", \"input_width\": " << spec.input_width
                                  << ", \"expert_width\": " << spec.expert_width
                                  << ", \"bytes_per_expert_component\": " << spec.bytes << ", ";
                    } else {
                        std::cout << "\"bytes_per_expert_component\": " << expert_upload_bytes << ", ";
                    }
                    std::cout << "\"cpu_latency_ms_median\": " << result.latency_ms_median
                              << ", \"cpu_effective_gbps_median\": " << result.effective_gbps_median
                              << ", \"upload_gbps_median\": " << device.upload_gbps[j] << "}";
                }
                std::cout << "]}";
            }
            std::cout << "]";
            std::cout << "},\n";
        } else {
            std::cout << "\"status\": \"unavailable\", \"reason\": \"no CUDA backend\"},\n";
        }
        std::cout
                  << "    \"expert_cache_upload\": {";
        if (transfer.backend) {
            std::cout << "\"status\": \"measured\", \"path\": \"ggml_backend_expert_cache_upload_async\", "
                      << "\"gbps_median\": " << (opts.model_experts.empty()
                              ? median(expert_cache_upload_samples)
                              : median(model_expert_upload_samples[representative_index]))
                      << ", \"bytes_per_expert_component\": " << expert_upload_bytes;
            if (!opts.model_experts.empty()) {
                std::cout << ", \"model_profiles\": [";
                for (size_t i = 0; i < opts.model_experts.size(); ++i) {
                    const auto & spec = opts.model_experts[i];
                    if (i) std::cout << ", ";
                    std::cout << "{\"ggml_type_id\": " << spec.type
                              << ", \"ggml_type\": \"" << ggml_type_name(static_cast<ggml_type>(spec.type))
                              << "\", \"input_width\": " << spec.input_width
                              << ", \"expert_width\": " << spec.expert_width
                              << ", \"expert_count\": " << spec.expert_count
                              << ", \"bytes_per_expert_component\": " << spec.bytes
                              << ", \"gbps_median\": " << median(model_expert_upload_samples[i]) << "}";
                }
                std::cout << "]";
            }
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
