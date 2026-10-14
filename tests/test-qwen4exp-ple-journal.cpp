#include "ggml.h"
#include "ggml-backend.h"
#include "llama-qwen4exp-ops.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static std::vector<float> read(ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(float));
    return values;
}
static bool same(const std::vector<float> & a, const std::vector<float> & b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}
struct outputs {
    std::vector<float> journal, convolution;
};
// Diagnostic control for reuse of freed graph metadata addresses. Default
// coverage frees each arena and exercises the backend's ordinary graph cache.
static std::vector<ggml_context *> retained_metadata;
static outputs evaluate(ggml_backend_t backend, int channels, int kernel, int dilation,
                        const std::vector<float> & inputs, const std::vector<float> * prior, ggml_type weight_type) {
    const int tokens = inputs.size() / channels;
    const int length = (kernel - 1) * dilation;
    auto * ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    check(ctx, "context");
    // Padded input/state columns catch journals that assume packed source rows.
    auto * raw_input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels + 2, tokens);
    auto * input = ggml_view_2d(ctx, raw_input, channels, tokens, raw_input->nb[1], sizeof(float));
    auto * raw_state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, channels + 2, length);
    auto * state = ggml_view_2d(ctx, raw_state, channels, length, raw_state->nb[1], sizeof(float));
    auto * weights = ggml_new_tensor_2d(ctx, weight_type, kernel, channels);
    auto * columns = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, length * tokens);
    ggml_tensor * journal = nullptr;
    const auto candidate = qwen4exp_build_ple_dilated_conv(ctx, input, weights,
            prior ? state : nullptr, kernel, dilation, &journal, columns);
    const auto ordinary = qwen4exp_build_ple_dilated_conv(ctx, input, weights,
            prior ? state : nullptr, kernel, dilation);
    check(journal && ggml_is_contiguous(journal) && journal->ne[0] == channels &&
          journal->ne[1] == length && journal->ne[2] == tokens, "journal layout");
    auto * graph = ggml_new_graph_custom(ctx, 512, false);
    for (auto * value : {candidate.output, candidate.next_state, ordinary.output, ordinary.next_state, journal}) {
        ggml_build_forward_expand(graph, value);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    check(buffer, "allocation");
    std::vector<float> padded_input(size_t(channels + 2) * tokens, -777.f);
    std::vector<float> padded_state(size_t(channels + 2) * length, 555.f);
    for (int t = 0; t < tokens; ++t) for (int c = 0; c < channels; ++c) {
        padded_input[size_t(t) * (channels + 2) + c + 1] = inputs[size_t(t) * channels + c];
    }
    if (prior) for (int t = 0; t < length; ++t) for (int c = 0; c < channels; ++c) {
        padded_state[size_t(t) * (channels + 2) + c + 1] = prior->at(size_t(t) * channels + c);
    }
    std::vector<float> weight_values(size_t(kernel) * channels);
    for (size_t i = 0; i < weight_values.size(); ++i) weight_values[i] = .07f * std::cos(float(i) * .19f);
    ggml_backend_tensor_set(raw_input, padded_input.data(), 0, padded_input.size() * sizeof(float));
    ggml_backend_tensor_set(raw_state, padded_state.data(), 0, padded_state.size() * sizeof(float));
    if (weight_type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> half(weight_values.size());
        ggml_fp32_to_fp16_row(weight_values.data(), half.data(), half.size());
        ggml_backend_tensor_set(weights, half.data(), 0, half.size() * sizeof(ggml_fp16_t));
    } else {
        ggml_backend_tensor_set(weights, weight_values.data(), 0, weight_values.size() * sizeof(float));
    }
    std::vector<int32_t> column_ids(size_t(length) * tokens);
    for (int t = 0; t < tokens; ++t) for (int h = 0; h < length; ++h) column_ids[size_t(t) * length + h] = t + h + 1;
    ggml_backend_tensor_set(columns, column_ids.data(), 0, column_ids.size() * sizeof(int32_t));
    if (std::getenv("ESE_PLE_JOURNAL_TRACE")) {
        std::fprintf(stderr, "ple_journal_compute: channels=%d kernel=%d dilation=%d tokens=%d prior=%d\n",
                     channels, kernel, dilation, tokens, int(prior != nullptr));
        std::fflush(stderr);
    }
    for (int replay = 0; replay < 2; ++replay) {
        check(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "compute");
    }
    outputs result{read(journal), read(candidate.output)};
    check(same(result.convolution, read(ordinary.output)), "journal changes convolution bits");
    check(same(read(candidate.next_state), read(ordinary.next_state)), "journal changes final state bits");
    const auto final = read(candidate.next_state);
    check(std::memcmp(final.data(), result.journal.data() + size_t(tokens - 1) * length * channels,
                      final.size() * sizeof(float)) == 0, "last journal differs from final state");
    // Independent FIFO oracle: retain only the latest history columns at each input.
    std::deque<std::vector<float>> history;
    for (int t = 0; t < length; ++t) {
        std::vector<float> column(channels, 0.f);
        if (prior) std::memcpy(column.data(), prior->data() + size_t(t) * channels, channels * sizeof(float));
        history.push_back(column);
    }
    for (int t = 0; t < tokens; ++t) {
        history.pop_front();
        history.emplace_back(inputs.begin() + size_t(t) * channels, inputs.begin() + size_t(t + 1) * channels);
        for (int h = 0; h < length; ++h) {
            check(std::memcmp(history[h].data(), result.journal.data() +
                    (size_t(t) * length + h) * channels, channels * sizeof(float)) == 0, "FIFO prefix oracle");
        }
    }
    ggml_backend_buffer_free(buffer);
    if (std::getenv("ESE_PLE_JOURNAL_RETAIN_METADATA")) retained_metadata.push_back(ctx);
    else ggml_free(ctx);
    return result;
}
static int run(ggml_backend_t backend, int & cuts) {
    int cases = 0;
    const char * limit = std::getenv("ESE_PLE_JOURNAL_MAX_CASES");
    const int max_cases = limit ? std::stoi(limit) :
        (std::getenv("ESE_PLE_JOURNAL_RETAIN_METADATA") ? 8 : 1080);
    check(max_cases > 0, "case limit");
    for (int channels : {1, 3, 32, 160, 2560, 10240}) for (int kernel : {2, 3, 4}) for (int dilation : {1, 2, 4}) {
        for (int tokens : {1, 2, 3, 4, 8}) for (bool zero : {false, true}) for (ggml_type weight_type : {GGML_TYPE_F32, GGML_TYPE_F16}) {
            const int length = (kernel - 1) * dilation;
            std::vector<float> input(size_t(channels) * tokens), prior(size_t(channels) * length);
            for (size_t i = 0; i < input.size(); ++i) {
                // Positive first column gives a canonical +0 null prior. Later
                // signed zeros must survive copying without arithmetic changes.
                input[i] = i < size_t(channels) ? .1f + float(i) * .001f : .3f * std::sin(float(i) * .17f);
                if (i >= size_t(channels) && i % 11 == 0) input[i] = -0.f;
            }
            for (size_t i = 0; i < prior.size(); ++i) prior[i] = i % 13 == 0 ? -0.f : -.2f * std::cos(float(i) * .11f);
            const auto original = evaluate(backend, channels, kernel, dilation, input, zero ? nullptr : &prior, weight_type);
            for (int kept = 0; kept <= tokens; ++kept) {
            auto poisoned = input;
            for (size_t i = size_t(kept) * channels; i < poisoned.size(); ++i) poisoned[i] = (i % 2 ? 128.f : -256.f);
            const auto future = evaluate(backend, channels, kernel, dilation, poisoned, zero ? nullptr : &prior, weight_type);
            check(std::memcmp(original.journal.data(), future.journal.data(), size_t(kept) * length * channels * sizeof(float)) == 0,
                  "future suffix changes retained-prefix state");
            std::vector<float> retained(size_t(length) * channels, 0.f);
            if (kept > 0) {
                std::memcpy(retained.data(), original.journal.data() + size_t(kept - 1) * length * channels,
                            retained.size() * sizeof(float));
            } else if (!zero) {
                retained = prior;
            }
            std::vector<float> next(channels, .125f);
            const auto resumed = evaluate(backend, channels, kernel, dilation, next, &retained, weight_type);
            // Construct the scalar teacher's rolling history independently.
            // Comparing to a larger batch would also test pre-existing CPU
            // scalar/batch arithmetic differences, beyond journal correctness.
            std::deque<std::vector<float>> history;
            for (int h = 0; h < length; ++h) {
                std::vector<float> column(channels, 0.f);
                if (!zero) std::memcpy(column.data(), prior.data() + size_t(h) * channels, channels * sizeof(float));
                history.push_back(column);
            }
            for (int t = 0; t < kept; ++t) {
                history.pop_front();
                history.emplace_back(input.begin() + size_t(t) * channels, input.begin() + size_t(t + 1) * channels);
            }
            std::vector<float> teacher_history;
            for (const auto & column : history) teacher_history.insert(teacher_history.end(), column.begin(), column.end());
            const auto reference = evaluate(backend, channels, kernel, dilation, next, &teacher_history, weight_type);
            if (std::memcmp(resumed.convolution.data(), reference.convolution.data(),
                            channels * sizeof(float)) != 0) {
                std::fprintf(stderr, "continuation mismatch: channels=%d kernel=%d dilation=%d tokens=%d zero=%d kept=%d\n",
                             channels, kernel, dilation, tokens, int(zero), kept);
                for (int c = 0; c < channels; ++c) {
                    if (std::memcmp(&resumed.convolution[c], &reference.convolution[c], sizeof(float))) {
                        std::fprintf(stderr, "channel=%d resumed=%a reference=%a\n", c, resumed.convolution[c], reference.convolution[c]);
                    }
                }
            }
            check(std::memcmp(resumed.convolution.data(), reference.convolution.data(),
                              channels * sizeof(float)) == 0, "restored-prefix continuation bits");
            check(std::memcmp(resumed.journal.data(), reference.journal.data(),
                              size_t(length) * channels * sizeof(float)) == 0, "restored-prefix next history");
            ++cuts;
            }
            ++cases;
            if (cases == max_cases) return cases;
        }
    }
    return cases;
}
int main(int argc, char ** argv) {
    ggml_backend_t backend = nullptr;
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : -1;
#ifdef GGML_USE_CUDA
        CUdevice cuda_device; CUcontext primary = nullptr;
        if (device >= 0) {
            check(cuInit(0) == CUDA_SUCCESS && cuDeviceGet(&cuda_device, device) == CUDA_SUCCESS &&
                  cuDevicePrimaryCtxRetain(&primary, cuda_device) == CUDA_SUCCESS &&
                  cuCtxSetCurrent(primary) == CUDA_SUCCESS, "CUDA primary context");
            backend = ggml_backend_cuda_init(device, nullptr, nullptr);
        } else
#endif
        {
            check(device == -1, "CUDA unavailable");
            backend = ggml_backend_cpu_init();
            ggml_backend_cpu_set_n_threads(backend, 2);
        }
        check(backend, "backend");
        int cuts = 0;
        const int cases = run(backend, cuts);
        ggml_backend_free(backend); backend = nullptr;
        for (auto * ctx : retained_metadata) ggml_free(ctx);
        retained_metadata.clear();
#ifdef GGML_USE_CUDA
        if (device >= 0) check(cuDevicePrimaryCtxRelease(cuda_device) == CUDA_SUCCESS, "primary release");
#endif
        std::printf("PASS: %d PLE prefix journal cases, %d accepted cuts; F32/F16 weights, FIFO bits, poisoned suffix, continuation\n", cases, cuts);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        if (backend) ggml_backend_free(backend);
        for (auto * ctx : retained_metadata) ggml_free(ctx);
        retained_metadata.clear();
        return 1;
    }
}
