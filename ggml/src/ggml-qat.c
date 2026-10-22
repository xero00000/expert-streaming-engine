#include "ggml-qat.h"
#include "ggml-qat-impl.h"

#include <string.h>

// MAP_CUSTOM1 uses the start of op_params for its callback/task/userdata record.
// The final slot is reserved here for the quantization format; no userdata
// allocation or lifetime crosses graph construction/execution.
#define QAT_FORMAT_SLOT (GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1)

static void ggml_qat_forward(struct ggml_tensor * dst, const struct ggml_tensor * input,
                             int ith, int nth, void * userdata) {
    (void) userdata;
    const int mode = dst->op_params[QAT_FORMAT_SLOT];
    const int width = mode == GGML_QAT_FP4_E4M3 ? 16 : 32;
    GGML_ASSERT(mode >= 0 && mode <= 2 && input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(input->ne[0] % width == 0 && ggml_are_same_shape(input, dst));
    const float * source = (const float *) input->data;
    float * output = (float *) dst->data;
    const int64_t groups = ggml_nelements(input) / width;
    for (int64_t group = ith; group < groups; group += nth) {
        float maximum = 0.0f;
        int finite = 1;
        for (int i = 0; i < width; ++i) {
            const float value = source[group * width + i];
            finite = finite && isfinite(value);
            maximum = fmaxf(maximum, fabsf(value));
        }
        const float scale = ggml_qat_group_scale(maximum, mode);
        for (int i = 0; i < width; ++i) {
            output[group * width + i] = finite ? ggml_qat_value(source[group * width + i], scale, mode) : NAN;
        }
    }
}

struct ggml_tensor * ggml_qat(struct ggml_context * ctx, struct ggml_tensor * input, enum ggml_qat_format format) {
    GGML_ASSERT(format >= GGML_QAT_FP8_E8M0 && format <= GGML_QAT_FP4_E4M3);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && ggml_is_contiguous(input));
    GGML_ASSERT(input->ne[0] % (format == GGML_QAT_FP4_E4M3 ? 16 : 32) == 0);
    struct ggml_tensor * output = ggml_map_custom1(ctx, input, ggml_qat_forward, GGML_N_TASKS_MAX, NULL);
    output->op_params[QAT_FORMAT_SLOT] = format;
    return output;
}

int ggml_qat_mode(const struct ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_MAP_CUSTOM1) return -1;
    // The callback is the first member of GGML's custom1 parameter record.
    ggml_custom1_op_t callback;
    memcpy(&callback, tensor->op_params, sizeof(callback));
    if (callback != ggml_qat_forward) return -1;
    const int mode = tensor->op_params[QAT_FORMAT_SLOT];
    return mode >= 0 && mode <= 2 ? mode : -1;
}

size_t ggml_qat_row_size(int64_t width, enum ggml_qat_format format) {
    if (format < 0 || format > 2 || width <= 0 || width % (format == 2 ? 16 : 32)) return 0;
    if ((uint64_t) width > SIZE_MAX / 2) return 0;
    return (size_t) width / (format == 0 ? 1 : 2) + (size_t) width / (format == 2 ? 16 : 32);
}

static uint8_t qat_e4m3_code(float value) {
    const uint8_t sign = signbit(value) ? 128 : 0;
    const float magnitude = fabsf(value);
    if (magnitude < 0.015625f) return sign | (uint8_t) (magnitude * 512.0f);
    int exponent;
    const float fraction = frexpf(magnitude, &exponent);
    return sign | (uint8_t) (((exponent + 6) << 3) + (int) (fraction * 16.0f - 8.0f));
}

static float qat_e4m3_decode(uint8_t code) {
    const int exponent = (code >> 3) & 15, mantissa = code & 7;
    if (exponent == 15 && mantissa == 7) return NAN;
    const float magnitude = exponent ? ldexpf(1.0f + mantissa * 0.125f, exponent - 7) : mantissa * 0.001953125f;
    return code & 128 ? -magnitude : magnitude;
}

bool ggml_qat_pack_row(const float * input, void * packed, int64_t width, enum ggml_qat_format format) {
    const size_t bytes = ggml_qat_row_size(width, format);
    if (!bytes || !input || !packed) return false;
    uint8_t * values = (uint8_t *) packed;
    uint8_t * scales = values + width / (format == 0 ? 1 : 2);
    const int group_width = format == 2 ? 16 : 32;
    const float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    memset(packed, 0, bytes);
    for (int64_t group = 0; group < width / group_width; ++group) {
        float maximum = 0;
        for (int i = 0; i < group_width; ++i) {
            const float value = input[group * group_width + i];
            if (!isfinite(value)) return false;
            maximum = fmaxf(maximum, fabsf(value));
        }
        const float scale = ggml_qat_group_scale(maximum, format);
        if (!isfinite(scale) || scale <= 0) return false;
        if (format == 2) scales[group] = qat_e4m3_code(scale);
        else {
            int exponent;
            frexpf(scale, &exponent);
            const int code = exponent + 126;
            if (code < 0 || code >= 255) return false;
            scales[group] = (uint8_t) code;
        }
        for (int i = 0; i < group_width; ++i) {
            const int64_t pos = group * group_width + i;
            const float normalized = input[pos] / scale;
            if (format == 0) {
                values[pos] = qat_e4m3_code(ggml_qat_e4m3(fminf(448, fmaxf(-448, normalized))));
            } else {
                const float rounded = ggml_qat_e2m1(normalized);
                uint8_t code = signbit(rounded) ? 8 : 0;
                for (int j = 0; j < 8; ++j) if (fabsf(rounded) == levels[j]) { code |= j; break; }
                values[pos / 2] |= code << ((pos % 2) * 4);
            }
        }
    }
    return true;
}

bool ggml_qat_unpack_row(const void * packed, float * output, int64_t width, enum ggml_qat_format format) {
    if (!ggml_qat_row_size(width, format) || !packed || !output) return false;
    const uint8_t * values = (const uint8_t *) packed;
    const uint8_t * scales = values + width / (format == 0 ? 1 : 2);
    const int group_width = format == 2 ? 16 : 32;
    const float levels[] = {0, .5f, 1, 1.5f, 2, 3, 4, 6};
    for (int64_t pos = 0; pos < width; ++pos) {
        const uint8_t scale_code = scales[pos / group_width];
        const float scale = format == 2 ? qat_e4m3_decode(scale_code) :
            (scale_code == 255 ? NAN : ldexpf(1.0f, (int) scale_code - 127));
        if (!isfinite(scale) || scale <= 0) return false;
        float value;
        if (format == 0) value = qat_e4m3_decode(values[pos]);
        else {
            const uint8_t code = (values[pos / 2] >> ((pos % 2) * 4)) & 15;
            value = code & 8 ? -levels[code & 7] : levels[code & 7];
        }
        if (!isfinite(value)) return false;
        output[pos] = value * scale;
    }
    return true;
}
