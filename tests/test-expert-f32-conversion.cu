#include "ggml-cuda/expert-f32.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

static __global__ void check_conversion(unsigned long long * errors, unsigned * first) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned which = 0; which < 2; ++which) {
        // Every positive and negative subnormal, including signed zero, plus
        // 16,777,216 spread normal/special patterns. The integer oracle never
        // converts a float, so compiler FTZ settings cannot affect it.
        const unsigned bits = which ? i * 2654435761u : ((i >> 1) | ((i & 1) << 31));
        const unsigned exponent = (bits >> 23) & 255, mantissa = bits & 0x7fffff;
        if (exponent == 255 && mantissa) continue; // NaN payloads are unspecified.
        auto expected = static_cast<unsigned long long>(bits >> 31) << 63;
        if (exponent == 255) {
            expected |= 0x7ff0000000000000ull;
        } else if (exponent) {
            expected |= (static_cast<unsigned long long>(exponent + 896) << 52) |
                        (static_cast<unsigned long long>(mantissa) << 29);
        } else if (mantissa) {
            const int leading = 31 - __clz(mantissa);
            expected |= (static_cast<unsigned long long>(leading + 874) << 52) |
                        (static_cast<unsigned long long>(mantissa ^ (1u << leading)) << (52 - leading));
        }
        const double actual = expert_f32_to_double(__uint_as_float(bits));
        if (static_cast<unsigned long long>(__double_as_longlong(actual)) != expected) {
            atomicAdd(errors, 1ull);
            atomicMin(first, bits);
        }
    }
}

#define CHECK_CUDA(call) do { \
    const auto error = (call); \
    if (error != cudaSuccess) { \
        std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(error)); \
        return 2; \
    } \
} while (0)

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    unsigned long long * errors = nullptr, count = 0;
    unsigned * first = nullptr, first_bits = 0;
    CHECK_CUDA(cudaMalloc(&errors, sizeof(count)));
    CHECK_CUDA(cudaMalloc(&first, sizeof(first_bits)));
    CHECK_CUDA(cudaMemset(errors, 0, sizeof(count)));
    CHECK_CUDA(cudaMemset(first, 255, sizeof(first_bits)));
    check_conversion<<<65536, 256>>>(errors, first);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&first_bits, first, sizeof(first_bits), cudaMemcpyDeviceToHost));
    std::printf("16,777,216 signed zero/subnormal inputs plus 16,777,216 spread patterns: "
                "errors=%llu first=0x%08x\n", count, first_bits);
    CHECK_CUDA(cudaFree(errors));
    CHECK_CUDA(cudaFree(first));
    return count ? 1 : 0;
}
