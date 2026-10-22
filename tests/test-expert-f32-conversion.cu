#include "ggml-cuda/expert-f32.cuh"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <initializer_list>

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

static __device__ unsigned dot_hash(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; return x ^ (x >> 16);
}
static __device__ void dot_values(unsigned row, int column, float & a, float & b) {
    const unsigned mode = row % 12;
    unsigned x = dot_hash(row * 8191u + column), y = dot_hash(x);
    a = __uint_as_float((x & 0x807fffffu) | (126u << 23));
    b = __uint_as_float((y & 0x807fffffu) | (126u << 23));
    if (mode == 1) {
        a = __uint_as_float((x & 0x807fffffu) | ((1 + (x % 40)) << 23));
        b = __uint_as_float((y & 0x807fffffu) | ((1 + (y % 40)) << 23));
    } else if (mode == 2) {
        a = __uint_as_float((x & 0x807fffffu) | (250u << 23));
        b = __uint_as_float((y & 0x807fffffu) | (250u << 23));
    } else if (mode == 3 || mode == 10) {
        x = dot_hash(row * 8191u + (column & ~32)); y = dot_hash(x);
        a = __uint_as_float((x & 0x807fffffu) | (126u << 23));
        if (column & 32) a = -a;
        b = __uint_as_float((y & 0x807fffffu) | (126u << 23));
    } else if (mode == 4) {
        a = column == 0 ? 1.0f : column == 32 ? 0x1p-24f : column == 64 ? ((row & 16) ? 0x1p-60f : -0x1p-60f) : 0;
        b = 1;
    } else if (mode == 5) {
        a = __uint_as_float((x & 0x807fffffu) | ((17 + x % 220) << 23));
        b = __uint_as_float((y & 0x807fffffu) | ((17 + y % 220) << 23));
    } else if (mode == 6) { a = fabsf(a); b = fabsf(b); }
    else if (mode == 7) { a = __uint_as_float(x & 0x80000000u); }
    else if (mode == 8) { a *= 0x1p-31f; b *= 0x1p-30f; }
    else if (mode == 9) { a *= 0x1p31f; b *= 0x1p31f; }
    else if (mode == 11) {
        // Spread finite/special bit patterns, including subnormal operands.
        a = __uint_as_float(x); b = __uint_as_float(y);
    }
}
static __global__ void check_dots(int columns, unsigned long long * errors,
        unsigned long long * accepted, unsigned long long * fallback) {
    const unsigned row = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    expert_checked_dot dot;
    double partial = 0;
    for (int column = lane; column < columns; column += 32) {
        float a, b; dot_values(row, column, a, b);
        dot.add(a, b);
        partial += expert_f32_to_double(a) * expert_f32_to_double(b);
    }
    const float reference = expert_f64_to_float(expert_f32_reduce(partial));
    float result = 0;
    const bool used = dot.finish(columns, result);
    if (!used) result = reference;
    if (lane == 0) {
        atomicAdd(used ? accepted : fallback, 1ull);
        if (__float_as_uint(reference) != __float_as_uint(result) && !(isnan(reference) && isnan(result))) atomicAdd(errors, 1ull);
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
    unsigned long long * accepted = nullptr, * fallback = nullptr, naccepted = 0, nfallback = 0;
    CHECK_CUDA(cudaMalloc(&accepted, sizeof(count)));
    CHECK_CUDA(cudaMalloc(&fallback, sizeof(count)));
    CHECK_CUDA(cudaMemset(accepted, 0, sizeof(count)));
    CHECK_CUDA(cudaMemset(fallback, 0, sizeof(count)));
    for (int columns : {32, 96, 640, 2560, 8192, 65536, 65568}) {
        check_dots<<<256, 128>>>(columns, errors, accepted, fallback);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    CHECK_CUDA(cudaMemcpy(&count, errors, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&naccepted, accepted, sizeof(count), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&nfallback, fallback, sizeof(count), cudaMemcpyDeviceToHost));
    std::printf("checked dot rounding cells: accepted=%llu fallback=%llu bitwise_errors=%llu\n", naccepted, nfallback, count);
    if (!naccepted || !nfallback) count++;
    CHECK_CUDA(cudaFree(accepted));
    CHECK_CUDA(cudaFree(fallback));
    CHECK_CUDA(cudaFree(errors));
    CHECK_CUDA(cudaFree(first));
    return count ? 1 : 0;
}
