#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-quants.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

// Populate valid stored codes directly; quantizer choices cannot hide decoder errors.
static void block(ggml_type type, unsigned char * target, unsigned seed) {
    std::vector<unsigned char> bytes(ggml_type_size(type));
    for (auto & byte : bytes) {
        seed = seed * 1664525u + 1013904223u;
        byte = static_cast<unsigned char>(seed >> 24);
    }
    if (type == GGML_TYPE_IQ1_M) {
        block_iq1_m value;
        std::memcpy(&value, bytes.data(), sizeof(value));
        const uint16_t scale = ggml_fp32_to_fp16(0.03125f);
        uint16_t scales[4]; std::memcpy(scales, value.scales, sizeof(scales));
        for (int i = 0; i < 4; ++i)
            scales[i] = (scales[i] & 0x0fff) | (((scale >> (4 * i)) & 15) << 12);
        std::memcpy(value.scales, scales, sizeof(scales));
        std::memcpy(target, &value, sizeof(value));
    } else if (type == GGML_TYPE_IQ2_XXS) {
        block_iq2_xxs value;
        std::memcpy(&value, bytes.data(), sizeof(value));
        value.d = ggml_fp32_to_fp16(0.03125f);
        std::memcpy(target, &value, sizeof(value));
    } else {
        check(type == GGML_TYPE_IQ4_XS, "fixture quantization");
        block_iq4_xs value;
        std::memcpy(&value, bytes.data(), sizeof(value));
        value.d = ggml_fp32_to_fp16(0.03125f);
        std::memcpy(target, &value, sizeof(value));
    }
}

static float reference_dot(ggml_type type, const unsigned char * weights,
                           const float * input) {
    float decoded[256];
    ggml_internal_get_type_traits(type).to_float(weights, decoded, 256);
    double sum = 0;
    for (int i = 0; i < 256; ++i) {
        check(std::isfinite(decoded[i]), "nonfinite fixture weight");
        sum += double(decoded[i]) * input[i];
    }
    return float(sum);
}

static void run(ggml_backend_t backend, ggml_type type, bool merged, bool padded) {
    constexpr int width = 256, hidden = 256, rows = 19, experts = 3, slots = 4, tokens = 2;
    auto * ctx = ggml_init({2 << 20, nullptr, true}); check(ctx, "context");
    ggml_tensor * weights[3], * parents[3];
    const ggml_type types[] = {type, type, GGML_TYPE_IQ4_XS};
    std::vector<unsigned char> payloads[3];
    for (int role = 0; role < 3; ++role) {
        const int count = role == 2 ? rows : hidden;
        parents[role] = ggml_new_tensor_3d(ctx, types[role], width + (padded ? 256 : 0), count, experts);
        weights[role] = ggml_view_3d(ctx, parents[role], width, count, experts,
                                   parents[role]->nb[1], parents[role]->nb[2], 0);
        payloads[role].assign(ggml_nbytes(parents[role]), 0xff);
        for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < count; ++row)
            block(types[role], payloads[role].data() + expert * weights[role]->nb[2] + row * weights[role]->nb[1],
                  unsigned(17 + role * 239 + expert * 977 + row * 31));
    }
    ggml_tensor * paired = nullptr;
    std::vector<unsigned char> paired_bytes;
    if (merged) {
        paired = ggml_new_tensor_3d(ctx, type, width, 2 * hidden, experts);
        paired_bytes.resize(ggml_nbytes(paired));
        for (int expert = 0; expert < experts; ++expert) for (int row = 0; row < hidden; ++row)
            for (int role = 0; role < 2; ++role)
                std::memcpy(paired_bytes.data() + expert * paired->nb[2] + (row + (1 - role) * hidden) * paired->nb[1],
                            payloads[role].data() + expert * weights[role]->nb[2] + row * weights[role]->nb[1], paired->nb[1]);
    }
    const int pitch = width + (padded ? 7 : 0);
    auto * input_base = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, pitch * tokens);
    auto * input = ggml_view_3d(ctx, input_base, width, 1, tokens, pitch * sizeof(float), pitch * sizeof(float), 0);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, slots, tokens);
    auto * h = ggml_moe_up_gate(ctx, merged ? paired : weights[0], merged ? nullptr : weights[1], input, ids, GGML_UNARY_OP_SILU);
    auto * output = ggml_mul_mat_id(ctx, weights[2], h, ids);
    ggml_mul_mat_set_prec(h, GGML_PREC_F32); ggml_mul_mat_set_prec(output, GGML_PREC_F32);
    if (padded) {
        for (auto * tensor : {h, output}) {
            tensor->nb[1] = (tensor->ne[0] + 3) * sizeof(float);
            tensor->nb[2] = tensor->nb[1] * (slots + 1);
            tensor->nb[3] = tensor->nb[2] * tokens;
        }
    }
    auto * graph = ggml_new_graph_custom(ctx, 32, false); ggml_build_forward_expand(graph, output);
    check(ggml_backend_supports_op(backend, h) && ggml_backend_supports_op(backend, output), "explicit packed expert support");
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); check(buffer, "allocation");
    for (int role = 0; role < 3; ++role)
        ggml_backend_tensor_set(parents[role], payloads[role].data(), 0, payloads[role].size());
    if (merged) ggml_backend_tensor_set(paired, paired_bytes.data(), 0, paired_bytes.size());
    const int32_t routes[][slots] = {{2, 0, -1, 2}, {1, 2, 7, 0}};
    ggml_backend_tensor_set(ids, routes, 0, sizeof(routes));
    std::vector<float> values(pitch * tokens, -12345.f), previous;
    constexpr uint32_t padding = 0x7fc12345;
    for (int pass = 0; pass < 3; ++pass) {
        for (int token = 0; token < tokens; ++token) for (int i = 0; i < width; ++i)
            values[token * pitch + i] = float((i * 19 + token * 7 + (pass == 2 ? 5 : 0)) % 31 - 15) / 128;
        ggml_backend_tensor_set(input_base, values.data(), 0, values.size() * sizeof(float));
        for (auto * tensor : {h, output}) {
            std::vector<uint32_t> poison(ggml_nbytes(tensor) / sizeof(float), padding);
            ggml_backend_tensor_set(tensor, poison.data(), 0, poison.size() * sizeof(float));
        }
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
        std::vector<float> actual(rows * slots * tokens);
        for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < slots; ++slot) {
            std::vector<float> expected(rows, 0.f), activations(hidden);
            const int expert = routes[token][slot];
            if (expert >= 0 && expert < experts) {
                for (int row = 0; row < hidden; ++row) {
                    const float u = reference_dot(type, payloads[0].data() + expert * weights[0]->nb[2] + row * weights[0]->nb[1], values.data() + token * pitch);
                    const float g = reference_dot(type, payloads[1].data() + expert * weights[1]->nb[2] + row * weights[1]->nb[1], values.data() + token * pitch);
                    activations[row] = float((double(g) / (1 + std::exp(-double(g)))) * double(u));
                }
                for (int row = 0; row < rows; ++row)
                    expected[row] = reference_dot(GGML_TYPE_IQ4_XS, payloads[2].data() + expert * weights[2]->nb[2] + row * weights[2]->nb[1], activations.data());
            }
            float * got = actual.data() + (token * slots + slot) * rows;
            ggml_backend_tensor_get(output, got, token * output->nb[2] + slot * output->nb[1], rows * sizeof(float));
            check(std::memcmp(got, expected.data(), rows * sizeof(float)) == 0, "stored-code indexed expert oracle mismatch");
        }
        if (pass == 1) check(actual == previous, "replay differs");
        if (pass == 0) previous = actual;
        std::vector<float> preserved(values.size());
        ggml_backend_tensor_get(input_base, preserved.data(), 0, preserved.size() * sizeof(float));
        check(preserved == values, "activation or input padding changed");
        for (auto * tensor : {h, output}) {
            std::vector<uint32_t> bytes(ggml_nbytes(tensor) / sizeof(float));
            ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size() * sizeof(float));
            std::vector<bool> logical(bytes.size(), false);
            for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < slots; ++slot)
                for (int row = 0; row < tensor->ne[0]; ++row)
                    logical[(token * tensor->nb[2] + slot * tensor->nb[1]) / sizeof(float) + row] = true;
            for (size_t i = 0; i < bytes.size(); ++i)
                if (!logical[i]) check(bytes[i] == padding, "output padding changed");
        }
    }
    std::printf("expert_iq_formats: type=%s merged=%d padded=%d passed=true\n", ggml_type_name(type), merged, padded);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        check(argc <= 2, "usage: test-expert-iq-formats [CUDA_DEVICE]");
        const int device = argc == 2 ? std::atoi(argv[1]) : -1;
        for (int threads : {1, 8}) {
            ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
            if (device >= 0) backend = ggml_backend_cuda_init(device, nullptr, nullptr);
#endif
            if (device < 0) { backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend, threads); }
            check(backend, "backend");
            for (auto type : {GGML_TYPE_IQ1_M, GGML_TYPE_IQ2_XXS})
                for (bool merged : {false, true}) for (bool padded : {false, true}) run(backend, type, merged, padded);
            ggml_backend_free(backend);
            if (device >= 0) break;
        }
    } catch (const std::exception & error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
    return 0;
}
