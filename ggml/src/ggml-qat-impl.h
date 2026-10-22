#pragma once

#include <math.h>

#ifdef __CUDACC__
#define GGML_QAT_HD __host__ __device__
#else
#define GGML_QAT_HD
#endif

static GGML_QAT_HD inline float ggml_qat_rne_positive(float value) {
    const float lower = floorf(value);
    const float remainder = value - lower;
    return lower + (remainder > 0.5f || (remainder == 0.5f && fmodf(lower, 2.0f) != 0.0f) ? 1.0f : 0.0f);
}

static GGML_QAT_HD inline float ggml_qat_e4m3(float value) {
    const float magnitude = fabsf(value);
    if (!isfinite(magnitude)) return NAN;
    int exponent = 0;
    frexpf(magnitude, &exponent);
    const float step = magnitude < 0.015625f ? 0.001953125f : ldexpf(1.0f, exponent - 4);
    const float quantized = ggml_qat_rne_positive(magnitude / step) * step;
    return quantized > 448.0f ? NAN : copysignf(quantized, value);
}

static GGML_QAT_HD inline float ggml_qat_power2_scale(float value) {
    int exponent = 0;
    const float fraction = frexpf(value, &exponent);
    return ldexpf(1.0f, fraction == 0.5f ? exponent - 1 : exponent);
}

static GGML_QAT_HD inline float ggml_qat_e2m1(float value) {
    const float levels[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = fminf(fabsf(value), 6.0f);
    int best = 0;
    float distance = magnitude;
    for (int i = 1; i < 8; ++i) {
        const float next = fabsf(magnitude - levels[i]);
        if (next < distance || (next == distance && i % 2 == 0)) {
            best = i;
            distance = next;
        }
    }
    return copysignf(levels[best], value);
}

static GGML_QAT_HD inline float ggml_qat_group_scale(float maximum, int mode) {
    if (mode == 0) return ggml_qat_power2_scale(fmaxf(maximum, 1e-4f) * (1.0f / 448.0f));
    if (mode == 1) return ggml_qat_power2_scale(fmaxf(maximum, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    return ggml_qat_e4m3(fmaxf(maximum, 6.0f * 0x1p-9f) / 6.0f);
}

static GGML_QAT_HD inline float ggml_qat_value(float value, float scale, int mode) {
    if (mode == 0) return ggml_qat_e4m3(fminf(448.0f, fmaxf(-448.0f, value / scale))) * scale;
    return ggml_qat_e2m1(value / scale) * scale;
}

#undef GGML_QAT_HD
