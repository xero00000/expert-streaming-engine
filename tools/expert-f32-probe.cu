// Original-payload expert arithmetic experiment. No model/dispatch changes.
#include "expert-f32.cuh"
#include "ggml-expert-f32.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static void check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(result));
        std::exit(1);
    }
}
static void check_driver(CUresult result) {
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "CUDA driver: %d\n", int(result));
        std::exit(1);
    }
}
template <typename T>
static std::vector<T> read(const std::string & path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() <= 0 || size_t(file.tellg()) % sizeof(T)) {
        std::fprintf(stderr, "invalid input: %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<T> data(size_t(file.tellg()) / sizeof(T));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(data.data()), data.size() * sizeof(T));
    if (!file) {
        std::exit(1);
    }
    return data;
}
static void write(const std::string & path, const std::vector<float> & data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(float));
    if (!file) {
        std::exit(1);
    }
}
static bool compare(const char * label, const std::vector<float> & actual,
                    const std::vector<float> & expected, bool exact) {
    if (actual.size() != expected.size()) {
        return false;
    }
    double max_error = 0;
    int unequal = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) || !std::isfinite(expected[i])) {
            return false;
        }
        max_error = std::max(max_error, std::abs(double(actual[i]) - expected[i]));
        if (std::memcmp(&actual[i], &expected[i], sizeof(float))) {
            ++unequal;
        }
    }
    std::printf("compare: label=%s values=%zu unequal_bits=%d max_abs=%.12g exact_required=%d\n",
                label, actual.size(), unequal, max_error, exact);
    return exact ? unequal == 0 : max_error < 2e-5;
}

static std::vector<float> project(ggml_type type, const std::vector<uint8_t> & weights,
                                 const std::vector<float> & input, int rows) {
    const size_t stride = ggml_row_size(type, input.size());
    if (stride * rows != weights.size()) {
        std::abort();
    }
    std::vector<float> output(rows);
    std::vector<double> prepared(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        prepared[i] = ggml_expert_f32_to_double(input[i]);
    }
    for (int row = 0; row < rows; ++row) {
        output[row] = ggml_expert_f32_dot_prepared(type, weights.data() + row * stride,
                                                 prepared.data(), input.size());
    }
    return output;
}

// Independent full-row dequantization and long-double sequential accumulation.
static std::vector<float> oracle(ggml_type type, const std::vector<uint8_t> & weights,
                                const std::vector<float> & input, int rows) {
    const auto decode = ggml_internal_get_type_traits(type).to_float;
    const size_t stride = ggml_row_size(type, input.size());
    std::vector<float> output(rows), decoded(input.size());
    for (int row = 0; row < rows; ++row) {
        decode(weights.data() + row * stride, decoded.data(), input.size());
        long double sum = 0;
        for (size_t col = 0; col < input.size(); ++col) {
            sum += static_cast<long double>(decoded[col]) * static_cast<long double>(input[col]);
        }
        output[row] = float(sum);
    }
    return output;
}

template <ggml_type type>
__global__ static void decode_weights(const void * weights, float * output,
                                     int columns, int rows, size_t stride) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element < rows * columns) {
        output[element] = expert_f32_weight<type>((const char *) weights +
                          (element / columns) * stride, element % columns);
    }
}

template <ggml_type type>
static bool check_decoder(const char * label, const std::vector<uint8_t> & weights,
                          void * device_weights, int columns, int rows) {
    const size_t stride = ggml_row_size(type, columns);
    std::vector<float> expected(rows * columns), actual(rows * columns);
    const auto decode = ggml_internal_get_type_traits(type).to_float;
    for (int row = 0; row < rows; ++row) {
        decode(weights.data() + row * stride, expected.data() + row * columns, columns);
    }
    float * device_decoded;
    check(cudaMalloc(&device_decoded, actual.size() * sizeof(float)));
    decode_weights<type><<<(actual.size() + 127) / 128, 128>>>(device_weights, device_decoded, columns, rows, stride);
    check(cudaGetLastError());
    check(cudaMemcpy(actual.data(), device_decoded, actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
    check(cudaFree(device_decoded));
    return compare(label, actual, expected, true);
}

#define EXPERT_F32_TYPES(X) X(GGML_TYPE_IQ2_S) X(GGML_TYPE_IQ3_XXS) X(GGML_TYPE_IQ3_S) \
                           X(GGML_TYPE_IQ4_NL) X(GGML_TYPE_IQ4_XS) X(GGML_TYPE_Q2_0)

static bool check_decoder_type(ggml_type type, const char * label, const std::vector<uint8_t> & weights,
                               void * device_weights, int columns, int rows) {
    switch (type) {
#define CASE(T) case T: return check_decoder<T>(label, weights, device_weights, columns, rows);
        EXPERT_F32_TYPES(CASE)
#undef CASE
        default: std::abort();
    }
}

static void launch_up_gate(ggml_type type, void * up, void * gate, float * input, float * hidden,
                           float * u, float * g, int columns, int rows, size_t stride, cudaStream_t stream = nullptr) {
    switch (type) {
#define CASE(T) case T: expert_f32_up_gate<T><<<(rows + 3) / 4, 128, 0, stream>>>(up, gate, input, hidden, u, g, columns, rows, stride); break;
        EXPERT_F32_TYPES(CASE)
#undef CASE
        default: std::abort();
    }
}

static void launch_down(ggml_type type, void * weights, float * input, float * output,
                        int columns, int rows, size_t stride, cudaStream_t stream = nullptr) {
    switch (type) {
#define CASE(T) case T: expert_f32_project<T><<<(rows + 3) / 4, 128, 0, stream>>>(weights, input, output, columns, rows, stride); break;
        EXPERT_F32_TYPES(CASE)
#undef CASE
        default: std::abort();
    }
}
#undef EXPERT_F32_TYPES

int main(int argc, char ** argv) {
    if (argc != 3 && argc != 4 && argc != 6) {
        std::fprintf(stderr, "usage: expert-f32-probe DATA_DIRECTORY DEVICE [BENCH_ITERATIONS [UP_TYPE DOWN_TYPE]]\n");
        return 2;
    }
    const std::string path = argv[1];
    const int device = std::atoi(argv[2]);
    const int iterations = argc >= 4 ? std::atoi(argv[3]) : 0;
    const auto up_type = argc == 6 ? ggml_type(std::atoi(argv[4])) : GGML_TYPE_IQ3_XXS;
    const auto down_type = argc == 6 ? ggml_type(std::atoi(argv[5])) : GGML_TYPE_Q2_0;
    if (iterations < 0 || iterations > 10000) {
        return 2;
    }
    auto context = ggml_init({1024 * 1024, nullptr, false});
    if (!context) {
        return 1;
    }
    ggml_free(context);
    check_driver(cuInit(0));
    CUdevice raw;
    CUcontext primary;
    check_driver(cuDeviceGet(&raw, device));
    check_driver(cuDevicePrimaryCtxRetain(&primary, raw));
    check_driver(cuCtxSetCurrent(primary));
    check(cudaSetDevice(device));

    const auto input = read<float>(path + "/input.f32");
    const auto up = read<uint8_t>(path + "/up.bin");
    const auto gate = read<uint8_t>(path + "/gate.bin");
    const auto down = read<uint8_t>(path + "/down.bin");
    constexpr int columns = 2560, hidden_rows = 640, output_rows = 2560;
    const size_t up_stride = ggml_row_size(up_type, columns);
    const size_t down_stride = ggml_row_size(down_type, hidden_rows);
    if (input.size() != columns || up.size() != up_stride * hidden_rows ||
        gate.size() != up.size() || down.size() != down_stride * output_rows) {
        return 2;
    }
    void * device_up, * device_gate, * device_down;
    float * device_input, * device_hidden, * device_output, * device_u, * device_g;
    check(cudaMalloc(&device_up, up.size()));
    check(cudaMalloc(&device_gate, gate.size()));
    check(cudaMalloc(&device_down, down.size()));
    check(cudaMalloc(&device_input, input.size() * sizeof(float)));
    check(cudaMalloc(&device_hidden, hidden_rows * sizeof(float)));
    check(cudaMalloc(&device_output, output_rows * sizeof(float)));
    check(cudaMalloc(&device_u, hidden_rows * sizeof(float)));
    check(cudaMalloc(&device_g, hidden_rows * sizeof(float)));
    check(cudaMemcpy(device_up, up.data(), up.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_gate, gate.data(), gate.size(), cudaMemcpyHostToDevice));
    check(cudaMemcpy(device_down, down.data(), down.size(), cudaMemcpyHostToDevice));
    bool passed = check_decoder_type(up_type, "decode_up", up, device_up, columns, hidden_rows);
    passed = check_decoder_type(up_type, "decode_gate", gate, device_gate, columns, hidden_rows) && passed;
    passed = check_decoder_type(down_type, "decode_down", down, device_down, hidden_rows, output_rows) && passed;

    for (const std::string label : {"captured", "alternating", "zero", "subnormal"}) {
        auto x = input;
        if (label != "captured") {
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] = label == "zero" ? 0.0f :
                       label == "subnormal" ? std::ldexp(float(int(i % 7) - 3), -145) :
                       float(int(i % 13) - 6) / 16.0f;
            }
        }
        const auto start = std::chrono::steady_clock::now();
        auto cpu_u = project(up_type, up, x, hidden_rows);
        auto cpu_g = project(up_type, gate, x, hidden_rows);
        std::vector<float> cpu_h(hidden_rows);
        for (int i = 0; i < hidden_rows; ++i) {
            cpu_h[i] = ggml_expert_f32_silu_product(cpu_g[i], cpu_u[i]);
        }
        auto cpu_o = project(down_type, down, cpu_h, output_rows);
        const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        const auto oracle_u = oracle(up_type, up, x, hidden_rows);
        const auto oracle_g = oracle(up_type, gate, x, hidden_rows);
        std::vector<float> oracle_h(hidden_rows);
        for (int row = 0; row < hidden_rows; ++row) {
            const long double g = oracle_g[row];
            oracle_h[row] = float((g * static_cast<long double>(oracle_u[row])) / (1.0L + std::exp(-g)));
        }
        passed = compare("oracle_up", cpu_u, oracle_u, false) && passed;
        passed = compare("oracle_gate", cpu_g, oracle_g, false) && passed;
        passed = compare("oracle_hidden", cpu_h, oracle_h, false) && passed;
        passed = compare("oracle_down", cpu_o, oracle(down_type, down, oracle_h, output_rows), false) && passed;
        check(cudaMemcpy(device_input, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice));
        launch_up_gate(up_type, device_up, device_gate, device_input,
                       device_hidden, device_u, device_g, columns, hidden_rows, up_stride);
        check(cudaGetLastError());
        launch_down(down_type, device_down, device_hidden, device_output, hidden_rows, output_rows, down_stride);
        check(cudaGetLastError());
        std::vector<float> gpu_u(hidden_rows), gpu_g(hidden_rows), gpu_h(hidden_rows), gpu_o(output_rows);
        check(cudaMemcpy(gpu_u.data(), device_u, gpu_u.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(gpu_g.data(), device_g, gpu_g.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(gpu_h.data(), device_hidden, gpu_h.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check(cudaMemcpy(gpu_o.data(), device_output, gpu_o.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::printf("case: device=%d input=%s cpu_reference_ms=%.6f\n", device, label.c_str(), cpu_ms);
        passed = compare("cpu_gpu_up", gpu_u, cpu_u, true) && passed;
        passed = compare("cpu_gpu_gate", gpu_g, cpu_g, true) && passed;
        passed = compare("cpu_gpu_hidden", gpu_h, cpu_h, true) && passed;
        passed = compare("cpu_gpu_down", gpu_o, cpu_o, true) && passed;
        for (const auto & item : std::vector<std::pair<std::string, std::vector<float>>>{
                {"cpu-up", cpu_u}, {"cpu-gate", cpu_g}, {"cpu-hidden", cpu_h}, {"cpu-down", cpu_o},
                {"gpu-up", gpu_u}, {"gpu-gate", gpu_g}, {"gpu-hidden", gpu_h}, {"gpu-down", gpu_o}}) {
            write(path + "/" + label + "-" + item.first + "-" + std::to_string(device) + ".f32", item.second);
        }
    }
    if (iterations) {
        check(cudaMemcpy(device_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
        cudaStream_t stream;
        cudaGraph_t graph;
        cudaGraphExec_t executable;
        check(cudaStreamCreate(&stream));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        launch_up_gate(up_type, device_up, device_gate, device_input,
                       device_hidden, nullptr, nullptr, columns, hidden_rows, up_stride, stream);
        launch_down(down_type, device_down, device_hidden, device_output, hidden_rows, output_rows, down_stride, stream);
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiate(&executable, graph, 0));
        for (int i = 0; i < 10; ++i) {
            check(cudaGraphLaunch(executable, stream));
        }
        check(cudaStreamSynchronize(stream));
        cudaEvent_t start, end;
        check(cudaEventCreate(&start));
        check(cudaEventCreate(&end));
        check(cudaEventRecord(start, stream));
        for (int i = 0; i < iterations; ++i) {
            check(cudaGraphLaunch(executable, stream));
        }
        check(cudaEventRecord(end, stream));
        check(cudaEventSynchronize(end));
        float ms;
        check(cudaEventElapsedTime(&ms, start, end));
        std::printf("prototype_timing: device=%d iterations=%d resident_pipeline_ms=%.9g weight_bytes=%zu\n",
                    device, iterations, ms / iterations, up.size() + gate.size() + down.size());
        check(cudaEventDestroy(end));
        check(cudaEventDestroy(start));
        check(cudaGraphExecDestroy(executable));
        check(cudaGraphDestroy(graph));
        check(cudaStreamDestroy(stream));
    }
    for (void * pointer : {device_up, device_gate, device_down, (void *) device_input,
                          (void *) device_hidden, (void *) device_output, (void *) device_u, (void *) device_g}) {
        check(cudaFree(pointer));
    }
    check_driver(cuDevicePrimaryCtxRelease(raw));
    std::printf("prototype_result: device=%d passed=%d\n", device, passed);
    return passed ? 0 : 1;
}
