#pragma once

#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_HC_INLINE static __host__ __device__ inline
#else
#define GGML_HC_INLINE static inline
#endif

// Keep FP64 coefficients through the residual expression. Rounding them to
// F32 first can move the final result across a BF16 rounding midpoint.
GGML_HC_INLINE double ggml_hc_mul(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dmul_rn(a, b);
#else
    volatile double result = a * b;
    return result;
#endif
}

GGML_HC_INLINE double ggml_hc_add(double a, double b) {
#ifdef __CUDA_ARCH__
    return __dadd_rn(a, b);
#else
    volatile double result = a + b;
    return result;
#endif
}

GGML_HC_INLINE void ggml_hc_precise_coefficients(
        const float * projected, const float * scale, const float * bias,
        int streams, int iterations, float epsilon, double * post, double * comb) {
    const int S = streams;
    const double eps = epsilon;
    for (int d = 0; d < S; ++d) {
        const double v = ggml_hc_add(ggml_hc_mul(projected[S+d], scale[1]), bias[S+d]);
        post[d] = 2.0 / (1.0 + exp(-v));
    }
    for (int i = 0; i < S*S; ++i) {
        comb[i] = ggml_hc_add(ggml_hc_mul(projected[2*S+i], scale[2]), bias[2*S+i]);
    }
    for (int r = 0; r < S; ++r) {
        double maximum = comb[r*S];
        for (int c = 1; c < S; ++c) if (comb[r*S+c] > maximum) maximum = comb[r*S+c];
        double sum = 0.0;
        for (int c = 0; c < S; ++c) {
            comb[r*S+c] = exp(comb[r*S+c] - maximum);
            sum = ggml_hc_add(sum, comb[r*S+c]);
        }
        for (int c = 0; c < S; ++c) comb[r*S+c] = ggml_hc_add(comb[r*S+c] / sum, eps);
    }
    // Initial column normalization, followed by row/column pairs.
    for (int pass = 0; pass < 2*iterations-1; ++pass) {
        const int columns = pass % 2 == 0;
        for (int outer = 0; outer < S; ++outer) {
            double sum = 0.0;
            for (int inner = 0; inner < S; ++inner) {
                sum = ggml_hc_add(sum, comb[columns ? inner*S+outer : outer*S+inner]);
            }
            sum = ggml_hc_add(sum, eps);
            for (int inner = 0; inner < S; ++inner) comb[columns ? inner*S+outer : outer*S+inner] /= sum;
        }
    }
}

#undef GGML_HC_INLINE
