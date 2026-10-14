// Independent dequantization oracle for the production IQ3_XXS/Q8_1 vector dot.
#include "vecdotq.cuh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

static void check_driver(CUresult status) {
    if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "CUDA driver: %d\n", int(status));
        std::exit(1);
    }
}

__global__ static void dots(const block_iq3_xxs * weights, const block_q8_1 * inputs,
                           float * outputs, int count) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= count) {
        return;
    }
    for (int segment = 0; segment < 8; ++segment) {
        outputs[row * 8 + segment] = vec_dot_iq3_xxs_q8_1(weights, inputs + row * 8, row, segment * 2);
    }
}

static uint32_t random_word(uint32_t & state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

static bool run(int device) {
    // Establish the primary context explicitly so all-API memcheck sees no
    // speculative invalid-context probes during runtime initialization.
    CUdevice raw;
    CUcontext primary;
    check_driver(cuDeviceGet(&raw, device));
    check_driver(cuDevicePrimaryCtxRetain(&primary, raw));
    check_driver(cuCtxSetCurrent(primary));
    check(cudaSetDevice(device));

    constexpr int batch = 4096;
    constexpr int exhaustive = 256 * 16 * 128;
    constexpr int dense = 8192;
    std::vector<block_iq3_xxs> weights(batch);
    std::vector<block_q8_1> inputs(batch * 8);
    std::vector<float> output(batch * 8), expected(batch * 8);
    block_iq3_xxs * device_weights;
    block_q8_1 * device_inputs;
    float * device_outputs;
    check(cudaMalloc(&device_weights, weights.size() * sizeof(weights[0])));
    check(cudaMalloc(&device_inputs, inputs.size() * sizeof(inputs[0])));
    check(cudaMalloc(&device_outputs, output.size() * sizeof(output[0])));
    const auto dequantize = ggml_internal_get_type_traits(GGML_TYPE_IQ3_XXS).to_float;
    if (!dequantize) {
        std::abort();
    }
    double max_error = 0;
    int mismatch_count = 0;
    for (int offset = 0; offset < exhaustive + dense; offset += batch) {
        const int count = std::min(batch, exhaustive + dense - offset);
        for (int row = 0; row < count; ++row) {
            const int id = offset + row;
            uint32_t rng = uint32_t(id) + 1;
            auto & weight = weights[row];
            weight.d = __float2half(std::ldexp(1.0f, (id % 7) - 3));
            for (int segment = 0; segment < 8; ++segment) {
                uint32_t aux = 0;
                if (id < exhaustive) {
                    const int grid = id % 256;
                    const int scale = (id / 256) % 16;
                    const int signs = id / (256 * 16);
                    std::memset(weight.qs + segment * 8, grid, 8);
                    aux = uint32_t(scale) << 28;
                    for (int group = 0; group < 4; ++group) {
                        aux |= uint32_t(signs) << (7 * group);
                    }
                } else {
                    for (int grid = 0; grid < 8; ++grid) {
                        weight.qs[segment * 8 + grid] = uint8_t(random_word(rng));
                    }
                    aux = random_word(rng);
                }
                std::memcpy(weight.qs + 64 + 4 * segment, &aux, sizeof(aux));
                auto & input = inputs[row * 8 + segment];
                input.data.d = __float2half(std::ldexp(1.0f, ((id + segment) % 5) - 2));
                input.data.s = __float2half(0.0f); // unused by this dot product
                std::memset(input.qs, 0, sizeof(input.qs));
                if (id < exhaustive) {
                    // Every segment is exercised. Both activation signs and
                    // all within-segment positions vary independently of scale.
                    const int position = (id / 128 + segment * 7) % 32;
                    input.qs[position] = ((id + segment) & 1) ? -127 : 127;
                } else {
                    for (auto & value : input.qs) {
                        value = int(random_word(rng) % 255) - 127;
                    }
                }
            }
            float decoded[QK_K];
            dequantize(&weight, decoded, QK_K);
            for (int segment = 0; segment < 8; ++segment) {
                const auto & input = inputs[row * 8 + segment];
                const double scale = __half2float(input.data.d);
                double sum = 0;
                for (int column = 0; column < 32; ++column) {
                    sum += double(decoded[segment * 32 + column]) * scale * input.qs[column];
                }
                expected[row * 8 + segment] = float(sum);
            }
        }
        check(cudaMemcpy(device_weights, weights.data(), count * sizeof(weights[0]), cudaMemcpyHostToDevice));
        check(cudaMemcpy(device_inputs, inputs.data(), count * 8 * sizeof(inputs[0]), cudaMemcpyHostToDevice));
        dots<<<(count + 127) / 128, 128>>>(device_weights, device_inputs, device_outputs, count);
        check(cudaGetLastError());
        check(cudaMemcpy(output.data(), device_outputs, count * 8 * sizeof(output[0]), cudaMemcpyDeviceToHost));
        for (int i = 0; i < count * 8; ++i) {
            const double error = std::abs(double(output[i]) - expected[i]);
            max_error = std::max(max_error, error);
            // All scales are powers of two. The integer dot and fractional
            // weight scale fit exactly in F32, so no tolerance is necessary.
            if (!std::isfinite(output[i]) || output[i] != expected[i]) {
                if (mismatch_count < 4) {
                    std::fprintf(stderr, "device=%d case=%d segment=%d actual=%.9g expected=%.9g\n",
                                 device, offset + i / 8, i % 8, output[i], expected[i]);
                }
                ++mismatch_count;
            }
        }
    }
    check(cudaFree(device_outputs));
    check(cudaFree(device_inputs));
    check(cudaFree(device_weights));
    check_driver(cuDevicePrimaryCtxRelease(raw));
    std::printf("iq3_dot: device=%d cases=%d partial_dots=%d max_error=%.9g mismatches=%d\n",
                device, exhaustive + dense, (exhaustive + dense) * 8, max_error, mismatch_count);
    return mismatch_count == 0;
}

int main(int argc, char ** argv) {
    // The public type conversion callbacks require ggml's FP16 tables.
    auto context = ggml_init({1024 * 1024, nullptr, false});
    if (!context) {
        return 1;
    }
    ggml_free(context);
    const auto initialized = cuInit(0);
    if (initialized == CUDA_ERROR_NO_DEVICE) {
        return 77;
    }
    check_driver(initialized);
    int devices = 0;
    check(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        return 77;
    }
    const int first = argc == 2 ? std::atoi(argv[1]) : 0;
    const int last = argc == 2 ? first + 1 : devices;
    if (first < 0 || last > devices || argc > 2) {
        return 1;
    }
    bool passed = true;
    for (int device = first; device < last; ++device) {
        passed = run(device) && passed;
    }
    return passed ? 0 : 1;
}
