#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-ese.h"
#include "llama.h"
#include "llama-expert-cache.h"

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
#include <numeric>
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
    int iterations = 21;
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

struct sample_summary {
    size_t count = 0;
    double median = 0;
    double coefficient_variation = 0;
    double relative_standard_error = 0;
    double confidence = 0;
};

sample_summary summarize(const std::vector<double> & values) {
    if (values.empty()) throw std::runtime_error("cannot summarize empty measurements");
    double mean = 0;
    for (double value : values) mean += value;
    mean /= values.size();
    double variance = 0;
    for (double value : values) variance += (value - mean) * (value - mean);
    variance /= values.size();
    const double cv = mean != 0 ? std::sqrt(variance) / std::abs(mean) : 1.0;
    const double median_value = median(values);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(values.size());
    for (double value : values) absolute_deviations.push_back(std::abs(value - median_value));
    // The policy consumes a median, so confidence must describe uncertainty
    // in that robust central estimate rather than raw run-to-run variation.
    // MAD resists isolated scheduler/interrupt outliers while still rejecting
    // a genuinely broad or bimodal series. 1.4826 makes it sigma-consistent
    // for a normal distribution; division by sqrt(n) estimates uncertainty.
    const double robust_sigma = 1.4826 * median(std::move(absolute_deviations));
    const double relative_standard_error = robust_sigma /
            std::max(std::abs(median_value), std::numeric_limits<double>::epsilon()) /
            std::sqrt(double(values.size()));
    const double sample_factor = std::min(1.0, double(values.size()) / 7.0);
    const double confidence = sample_factor / (1.0 + 4.0 * relative_standard_error);
    return {values.size(), median_value, cv, relative_standard_error, confidence};
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
    size_t sample_count = 0;
    double coefficient_variation = 0;
    double relative_standard_error = 0;
    double confidence = 0;
    double independent_reference_nrmse = 0;
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
    std::vector<sample_summary> upload_statistics;
};

struct leased_upload_result {
    std::string backend;
    size_t spec_index = 0;
    sample_summary statistics;
};

class leased_expert_upload_probe {
public:
    leased_expert_upload_probe(
            const model_expert_spec & spec,
            ggml_backend_t backend,
            ggml_tensor * destination) :
        bytes_(spec.bytes), backend_(backend), destination_(destination) {
        using namespace llama_expert_cache;
        constexpr uint32_t source_id = 0;
        const source_identity identity = identify_file_source(spec.source, source_id);
        file_ = std::make_shared<file_source>(
                source_id, identity, spec.source, read_backend::pread);
        descriptor_spec description;
        description.key = {0, 0, component::gate};
        description.ggml_type = uint32_t(spec.type);
        description.rank = 2;
        description.quant_axis = 0;
        description.block_elements = uint32_t(ggml_blck_size(static_cast<ggml_type>(spec.type)));
        description.block_bytes = uint32_t(ggml_type_size(static_cast<ggml_type>(spec.type)));
        description.source_alignment = uint32_t(std::gcd<uint64_t>(32, spec.bytes));
        description.dimensions = {{uint64_t(spec.input_width), uint64_t(spec.expert_width), 0, 0}};
        description.strides[0] = description.block_bytes;
        description.strides[1] = uint64_t(ggml_row_size(
                static_cast<ggml_type>(spec.type), spec.input_width));
        description.model_identity = identity;
        description.extents.push_back({source_id, identity, spec.data_offset, spec.bytes});
        expert_ = descriptor::make(std::move(description), {{source_id, file_->size()}});
        cache_ = std::make_unique<ram_cache>(
                cache_config{spec.bytes, spec.bytes},
                std::vector<std::shared_ptr<const source>>{file_});
    }

    double measure_once() {
        cache_->clear();
        const auto start = clock_type::now();
        {
            auto lease = cache_->acquire(expert_);
            if (!lease || lease.size() != bytes_) {
                throw std::runtime_error("production expert lease returned the wrong payload size");
            }
            ggml_backend_expert_cache_upload_async(
                    backend_, destination_, lease.data(), 0, lease.size());
            ggml_backend_synchronize(backend_);
        }
        return seconds_since(start);
    }

    void validate(size_t expected_reads) const {
        using namespace llama_expert_cache;
        const cache_stats stats = cache_->stats();
        if (stats.active_leases != 0 ||
                stats.bytes_read[size_t(read_backend::pread)] < bytes_ * expected_reads) {
            throw std::runtime_error(
                    "production expert lease telemetry did not account for calibration reads");
        }
    }

private:
    size_t bytes_;
    ggml_backend_t backend_;
    ggml_tensor * destination_;
    std::shared_ptr<llama_expert_cache::file_source> file_;
    std::shared_ptr<const llama_expert_cache::descriptor> expert_;
    std::unique_ptr<llama_expert_cache::ram_cache> cache_;
};

sample_summary measure_leased_expert_upload(
        const model_expert_spec & spec,
        ggml_backend_t backend,
        ggml_tensor * destination,
        int iterations) {
    leased_expert_upload_probe probe(spec, backend, destination);
    // Planner calibration models recurring decode misses. Keep the cold-start
    // distribution separate instead of mixing first-use initialization into
    // a steady-state series and inflating its dispersion.
    probe.measure_once();
    std::vector<double> samples;
    for (int i = 0; i < iterations; ++i) {
        samples.push_back(probe.measure_once());
    }
    probe.validate(size_t(iterations) + 1);
    return summarize(samples);
}

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
    const sample_summary statistics = summarize(samples);
    const double elapsed = statistics.median;
    const size_t weight_bytes = expert_weights.size() * sizeof(ggml_fp16_t);
    return {k, m, n_used, threads, elapsed * 1000.0, gbps(weight_bytes, elapsed), checksum,
            statistics.count, statistics.coefficient_variation,
            statistics.relative_standard_error, statistics.confidence};
}

cpu_moe_result measure_model_cpu_moe(
        const model_expert_spec & spec, int iterations, int requested_threads,
        const std::function<double()> & concurrent_work = {},
        std::vector<double> * concurrent_samples = nullptr,
        bool validate_independent_reference = false) {
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
    double independent_reference_nrmse = 0;
    if (validate_independent_reference) {
        const ggml_type_traits_t traits = ggml_internal_get_type_traits(type);
        if (!traits.to_float) {
            throw std::runtime_error("model expert type has no independent dequantization path");
        }
        const size_t row_bytes = ggml_row_size(type, spec.input_width);
        if (row_bytes * size_t(spec.expert_width) != spec.bytes) {
            throw std::runtime_error("model expert row geometry does not match its GGUF payload span");
        }
        std::vector<float> row(size_t(spec.input_width));
        long double squared_error = 0;
        long double squared_reference = 0;
        for (int64_t routed = 0; routed < n_used; ++routed) {
            const uint8_t * expert = expert_weights.data() + size_t(routed) * spec.bytes;
            const float * activation = inputs.data() + size_t(routed) * spec.input_width;
            for (int64_t output_row = 0; output_row < spec.expert_width; ++output_row) {
                traits.to_float(expert + size_t(output_row) * row_bytes, row.data(), spec.input_width);
                long double expected = 0;
                for (int64_t column = 0; column < spec.input_width; ++column) {
                    expected += static_cast<long double>(row[size_t(column)]) * activation[size_t(column)];
                }
                const long double observed = result[
                        size_t(routed) * spec.expert_width + size_t(output_row)];
                const long double error = observed - expected;
                squared_error += error * error;
                squared_reference += expected * expected;
            }
        }
        independent_reference_nrmse = std::sqrt(double(
                squared_error / std::max<long double>(squared_reference, 1.0e-30L)));
        if (!std::isfinite(independent_reference_nrmse) || independent_reference_nrmse > 0.08) {
            std::ostringstream message;
            message << "model CPU MoE failed independent dequantized scalar reference: NRMSE="
                    << independent_reference_nrmse;
            throw std::runtime_error(message.str());
        }
    }
    const sample_summary statistics = summarize(samples);
    const double elapsed = statistics.median;
    return {spec.input_width, spec.expert_width, n_used, threads,
            elapsed * 1000.0, gbps(expected_bytes, elapsed), checksum,
            statistics.count, statistics.coefficient_variation,
            statistics.relative_standard_error, statistics.confidence,
            independent_reference_nrmse};
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
            model_cpu_moe.push_back(measure_model_cpu_moe(
                    spec, opts.iterations, opts.threads, {}, nullptr, true));
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
                leased_expert_upload_probe lease_probe(
                        spec, transfer.backend, transfer.tensor);
                lease_probe.measure_once();
                contended_model_cpu_moe.push_back(measure_model_cpu_moe(
                    spec, opts.iterations, opts.threads,
                    [&]() {
                        return lease_probe.measure_once();
                    },
                    &contended_model_upload_seconds[i]));
                lease_probe.validate(size_t(opts.iterations) + 1);
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
                primary.upload_statistics.push_back(summarize(contended_expert_upload_seconds));
            } else {
                primary.cpu = contended_model_cpu_moe;
                for (size_t i = 0; i < opts.model_experts.size(); ++i) {
                    primary.upload_gbps.push_back(
                            gbps(opts.model_experts[i].bytes, median(contended_model_upload_seconds[i])));
                    primary.upload_statistics.push_back(summarize(contended_model_upload_seconds[i]));
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
                    if (opts.model_experts.empty()) {
                        auto concurrent = [&]() {
                            const auto start = clock_type::now();
                            ggml_backend_expert_cache_upload_async(
                                    extra.backend, extra.tensor, source.data(), 0, bytes);
                            ggml_backend_synchronize(extra.backend);
                            return seconds_since(start);
                        };
                        measured.cpu.push_back(measure_cpu_moe(
                                opts.iterations, opts.threads, concurrent, &upload_seconds));
                    } else {
                        leased_expert_upload_probe lease_probe(
                                opts.model_experts[i], extra.backend, extra.tensor);
                        lease_probe.measure_once();
                        measured.cpu.push_back(measure_model_cpu_moe(
                                opts.model_experts[i], opts.iterations, opts.threads,
                                [&]() { return lease_probe.measure_once(); }, &upload_seconds));
                        lease_probe.validate(size_t(opts.iterations) + 1);
                    }
                    measured.upload_gbps.push_back(gbps(bytes, median(upload_seconds)));
                    measured.upload_statistics.push_back(summarize(upload_seconds));
                }
                device_contentions.push_back(std::move(measured));
            }
        }
        std::vector<leased_upload_result> leased_uploads;
        if (transfer.backend && !opts.model_experts.empty()) {
            for (size_t ordinal = 0;; ++ordinal) {
                std::string lease_backend_name;
                transfer_probe lease_probe = make_transfer_probe(
                        allocation_bytes, lease_backend_name, ordinal);
                if (!lease_probe.backend) break;
                for (size_t spec_index = 0; spec_index < opts.model_experts.size(); ++spec_index) {
                    leased_uploads.push_back({
                            lease_backend_name,
                            spec_index,
                            measure_leased_expert_upload(
                                    opts.model_experts[spec_index], lease_probe.backend,
                                    lease_probe.tensor, opts.iterations)});
                }
            }
        }

        const auto confident = [](double confidence, double relative_standard_error) {
            return std::isfinite(confidence) && confidence >= 0.80 &&
                    std::isfinite(relative_standard_error) && relative_standard_error >= 0;
        };
        bool planner_ready = opts.iterations >= 7 && !opts.model_experts.empty() &&
                !device_transfers.empty() && model_cpu_moe.size() == opts.model_experts.size() &&
                device_contentions.size() == device_transfers.size() &&
                leased_uploads.size() == device_transfers.size() * opts.model_experts.size();
        for (const auto & result : model_cpu_moe) {
            planner_ready = planner_ready &&
                    confident(result.confidence, result.relative_standard_error) &&
                    std::isfinite(result.independent_reference_nrmse) &&
                    result.independent_reference_nrmse <= 0.08;
        }
        for (const auto & result : leased_uploads) {
            planner_ready = planner_ready && confident(
                    result.statistics.confidence,
                    result.statistics.relative_standard_error);
        }
        for (const auto & device : device_contentions) {
            planner_ready = planner_ready &&
                    device.cpu.size() == opts.model_experts.size() &&
                    device.upload_statistics.size() == opts.model_experts.size();
            for (size_t i = 0; i < device.cpu.size() && i < device.upload_statistics.size(); ++i) {
                planner_ready = planner_ready &&
                        confident(device.cpu[i].confidence, device.cpu[i].relative_standard_error) &&
                        confident(device.upload_statistics[i].confidence,
                                device.upload_statistics[i].relative_standard_error);
            }
        }

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "{\n  \"benchmark_source\": {\n"
                  << "    \"tool\": \"ese-hardware-bench\",\n"
                  << "    \"calibration_level\": \""
                  << (planner_ready ? "planner" : "baseline") << "\",\n"
                  << "    \"planner_ready\": " << (planner_ready ? "true" : "false") << ",\n"
                  << "    \"transfer_path\": \"ggml_backend_tensor_set/get\",\n"
                  << "    \"model_requested\": \"" << json_escape(opts.model) << "\",\n"
                  << "    \"model_used\": " << (!opts.model_experts.empty() ? "true" : "false") << ",\n"
                  << "    \"model_usage\": "
                  << (!opts.model_experts.empty()
                          ? "[\"expert_payload_cpu_moe\",\"expert_cache_upload_geometry\",\"bounded_ram_lease_upload\",\"bounded_ram_lease_contention\",\"per_device_contention\"]"
                          : "[]") << "\n"
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
                  << ", \"sample_count\": " << cpu_moe.sample_count
                  << ", \"coefficient_variation\": " << cpu_moe.coefficient_variation
                  << ", \"relative_standard_error\": " << cpu_moe.relative_standard_error
                  << ", \"confidence\": " << cpu_moe.confidence
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
                          << ", \"bytes_per_expert_component\": " << spec.bytes
                          << ", \"bytes_read\": " << spec.bytes * 2
                          << ", \"latency_ms_median\": " << result.latency_ms_median
                          << ", \"effective_gbps_median\": " << result.effective_gbps_median
                          << ", \"sample_count\": " << result.sample_count
                          << ", \"coefficient_variation\": " << result.coefficient_variation
                          << ", \"relative_standard_error\": " << result.relative_standard_error
                          << ", \"confidence\": " << result.confidence
                          << ", \"correctness\": \"single-thread-and-dequantized-scalar-reference\""
                          << ", \"independent_reference_nrmse\": " << result.independent_reference_nrmse
                          << ", \"checksum\": "
                          << result.checksum << "}";
            }
            std::cout << "]";
        }
        std::cout << "},\n"
                  << "    \"cpu_cache_contention\": {";
        if (transfer.backend) {
            std::cout << "\"status\": \"measured\", \"cpu_operation\": \"ggml_mul_mat_id\", "
                      << "\"distribution\": \"warm-steady-state\", "
                      << "\"upload_path\": \""
                      << (opts.model_experts.empty()
                              ? "ggml_backend_expert_cache_upload_async"
                              : "pread_to_bounded_ram_lease_to_async_upload") << "\", "
                      << "\"cpu_latency_ms_median\": " << contended_cpu_moe.latency_ms_median
                      << ", \"cpu_effective_gbps_median\": " << contended_cpu_moe.effective_gbps_median
                      << ", \"cpu_sample_count\": " << contended_cpu_moe.sample_count
                      << ", \"cpu_coefficient_variation\": " << contended_cpu_moe.coefficient_variation
                      << ", \"cpu_relative_standard_error\": " << contended_cpu_moe.relative_standard_error
                      << ", \"cpu_confidence\": " << contended_cpu_moe.confidence
                      << ", \"upload_gbps_median\": "
                      << gbps(expert_upload_bytes, median(contended_expert_upload_seconds))
                      << ", \"upload_sample_count\": " << summarize(contended_expert_upload_seconds).count
                      << ", \"upload_coefficient_variation\": "
                      << summarize(contended_expert_upload_seconds).coefficient_variation
                      << ", \"upload_relative_standard_error\": "
                      << summarize(contended_expert_upload_seconds).relative_standard_error
                      << ", \"upload_confidence\": "
                      << summarize(contended_expert_upload_seconds).confidence
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
                              << ", \"cpu_sample_count\": " << result.sample_count
                              << ", \"cpu_coefficient_variation\": " << result.coefficient_variation
                              << ", \"cpu_relative_standard_error\": " << result.relative_standard_error
                              << ", \"cpu_confidence\": " << result.confidence
                              << ", \"upload_gbps_median\": "
                              << gbps(spec.bytes, median(contended_model_upload_seconds[i]))
                              << ", \"upload_sample_count\": " << summarize(contended_model_upload_seconds[i]).count
                              << ", \"upload_coefficient_variation\": "
                              << summarize(contended_model_upload_seconds[i]).coefficient_variation
                              << ", \"upload_relative_standard_error\": "
                              << summarize(contended_model_upload_seconds[i]).relative_standard_error
                              << ", \"upload_confidence\": "
                              << summarize(contended_model_upload_seconds[i]).confidence << "}";
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
                              << ", \"cpu_ns_per_expert_component\": "
                              << result.latency_ms_median * 1.0e6 / double(result.routed_experts)
                              << ", \"cpu_sample_count\": " << result.sample_count
                              << ", \"cpu_coefficient_variation\": " << result.coefficient_variation
                              << ", \"cpu_relative_standard_error\": " << result.relative_standard_error
                              << ", \"cpu_confidence\": " << result.confidence
                              << ", \"upload_gbps_median\": " << device.upload_gbps[j]
                              << ", \"upload_ns_per_expert_component\": "
                              << device.upload_statistics[j].median * 1.0e9
                              << ", \"upload_sample_count\": " << device.upload_statistics[j].count
                              << ", \"upload_coefficient_variation\": "
                              << device.upload_statistics[j].coefficient_variation
                              << ", \"upload_relative_standard_error\": "
                              << device.upload_statistics[j].relative_standard_error
                              << ", \"upload_confidence\": " << device.upload_statistics[j].confidence << "}";
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
            if (!leased_uploads.empty()) {
                std::cout << ", \"lease_upload_profiles\": [";
                for (size_t i = 0; i < leased_uploads.size(); ++i) {
                    if (i) std::cout << ", ";
                    const auto & result = leased_uploads[i];
                    const auto & spec = opts.model_experts[result.spec_index];
                    const auto & stats = result.statistics;
                    std::cout << "{\"backend\": \"" << json_escape(result.backend)
                              << "\", \"storage_backend\": \"pread\", \"distribution\": \"warm-steady-state\""
                              << ", \"ggml_type_id\": " << spec.type
                              << ", \"ggml_type\": \"" << ggml_type_name(static_cast<ggml_type>(spec.type))
                              << "\", \"input_width\": " << spec.input_width
                              << ", \"expert_width\": " << spec.expert_width
                              << ", \"expert_count\": " << spec.expert_count
                              << ", \"bytes_per_expert_component\": " << spec.bytes
                              << ", \"end_to_end_ms_median\": " << stats.median * 1000.0
                              << ", \"end_to_end_gbps_median\": " << gbps(spec.bytes, stats.median)
                              << ", \"sample_count\": " << stats.count
                              << ", \"coefficient_variation\": " << stats.coefficient_variation
                              << ", \"relative_standard_error\": " << stats.relative_standard_error
                              << ", \"confidence\": " << stats.confidence << "}";
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
