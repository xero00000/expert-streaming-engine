#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

static std::vector<float> get(ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
    return values;
}

static void exact(const std::vector<float> & actual, const std::vector<float> & expected,
        const std::string & label) {
    require(actual.size() == expected.size(), label+": size");
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::memcmp(&actual[i], &expected[i], sizeof(float)) == 0,
                label+": mismatch at "+std::to_string(i));
    }
}

static void close(const std::vector<float> & actual, const std::vector<double> & expected,
        const std::string & label) {
    require(actual.size() == expected.size(), label+": size");
    for (size_t i = 0; i < actual.size(); ++i) {
        if (!std::isfinite(actual[i]) ||
                std::abs(double(actual[i])-expected[i]) > 2e-5*(1+std::abs(expected[i]))) {
            char detail[160];
            std::snprintf(detail, sizeof(detail), " actual=%.9g expected=%.17g", actual[i], expected[i]);
            throw std::runtime_error(label+": mismatch at "+std::to_string(i)+detail);
        }
    }
}

static void run_case(ggml_backend_t backend, const std::string & mode, int nr, int nt, int nc) {
    ggml_context * ctx = ggml_init({1<<20, nullptr, true});
    require(ctx != nullptr, "metadata context");
    ggml_backend_buffer_t buffer = nullptr;
    try {
        auto * state = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nc-1, nr, 1);
        auto * input = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nr, nt);
        auto * weights = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
        auto * seq = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 1, nt);
        auto * conv = ggml_ssm_conv(ctx, state, input, weights, seq, nullptr);
        auto * view = ggml_view_2d(ctx, conv, nr, nt, nr*sizeof(float), 0);
        auto * silu = ggml_silu(ctx, view);
        auto * graph = ggml_new_graph_custom(ctx, 32, false);
        const bool terminal = mode == "terminal";
        const bool terminal_view = mode == "terminal-view";
        const bool prefix_conv = mode == "prefix-conv";
        const bool prefix_view = mode == "prefix-view";
        require(terminal || terminal_view || prefix_conv || prefix_view || mode == "full", "graph mode");
        ggml_build_forward_expand(graph, terminal ? conv : terminal_view ? view : silu);
        require(graph->n_nodes == (terminal ? 1 : terminal_view ? 2 : 3), "graph topology");
        require(graph->nodes[0] == conv, "convolution node");
        buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer != nullptr, "tensor buffer");

        std::vector<float> s(ggml_nelements(state)), x(ggml_nelements(input)), c(ggml_nelements(weights));
        for (size_t i = 0; i < s.size(); ++i) s[i] = 0.031f*float(int(i%19)-9);
        for (size_t i = 0; i < x.size(); ++i) x[i] = 0.047f*float(int(i%29)-14);
        for (size_t i = 0; i < c.size(); ++i) c[i] = 0.023f*float(int(i%13)-6);
        std::vector<int32_t> sequences(nt, 0);
        ggml_backend_tensor_set(state, s.data(), 0, s.size()*sizeof(float));
        ggml_backend_tensor_set(input, x.data(), 0, x.size()*sizeof(float));
        ggml_backend_tensor_set(weights, c.data(), 0, c.size()*sizeof(float));
        ggml_backend_tensor_set(seq, sequences.data(), 0, sequences.size()*sizeof(int32_t));
        std::vector<float> conv_canary(ggml_nelements(conv), -9876.0f);
        std::vector<float> silu_canary(ggml_nelements(silu), -12345.0f);
        ggml_backend_tensor_set(conv, conv_canary.data(), 0, conv_canary.size()*sizeof(float));
        ggml_backend_tensor_set(silu, silu_canary.data(), 0, silu_canary.size()*sizeof(float));

        // Independent causal convolution over the supplied history and input.
        auto sample = [&](int channel, int time) {
            return time < 0 ? s[channel*(nc-1)+time+(nc-1)] : x[time*nr+channel];
        };
        std::vector<double> reference(nr*nt), reference_silu(nr*nt);
        for (int t = 0; t < nt; ++t) for (int r = 0; r < nr; ++r) {
            double sum = 0;
            for (int k = 0; k < nc; ++k) sum += double(sample(r,t-(nc-1)+k))*double(c[r*nc+k]);
            reference[t*nr+r] = sum;
            reference_silu[t*nr+r] = sum/(1+std::exp(-sum));
        }
        std::vector<float> reference_state(nc*nr);
        for (int r = 0; r < nr; ++r) for (int k = 0; k < nc; ++k)
            reference_state[r*nc+k] = sample(r,nt-nc+k);

        const std::string label = mode+" nr="+std::to_string(nr)+" nt="+std::to_string(nt)+" nc="+std::to_string(nc);
        auto check_state = [&]() {
            auto values = get(conv);
            exact(std::vector<float>(values.begin()+nr*nt, values.end()), reference_state, label+" updated history");
        };
        if (prefix_conv || prefix_view) {
            const int boundary = prefix_conv ? 1 : 2;
            auto first = ggml_graph_view(graph, 0, boundary);
            require(ggml_backend_graph_compute(backend, &first) == GGML_STATUS_SUCCESS, label+" prefix compute");
            ggml_backend_synchronize(backend);
            close(get(view), reference, label+" prefix raw output");
            check_state();
            exact(get(silu), silu_canary, label+" future SiLU must remain untouched");
            auto rest = ggml_graph_view(graph, boundary, graph->n_nodes);
            require(ggml_backend_graph_compute(backend, &rest) == GGML_STATUS_SUCCESS, label+" remainder compute");
            ggml_backend_synchronize(backend);
            close(get(silu), reference_silu, label+" staged SiLU");
        } else {
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, label+" compute");
            ggml_backend_synchronize(backend);
            if (terminal || terminal_view) {
                close(get(view), reference, label+" raw output");
                exact(get(silu), silu_canary, label+" unused SiLU must remain untouched");
            } else {
                // The full fused graph need only materialize its final output and history.
                close(get(silu), reference_silu, label+" full SiLU");
            }
        }
        check_state();
        exact(get(state), s, label+" input history");
        exact(get(input), x, label+" input");
        exact(get(weights), c, label+" weights");
        std::vector<int32_t> actual_seq(nt);
        ggml_backend_tensor_get(seq, actual_seq.data(), 0, actual_seq.size()*sizeof(int32_t));
        require(actual_seq == sequences, label+" sequence IDs unchanged");
        ggml_backend_buffer_free(buffer);
        buffer = nullptr;
        ggml_free(ctx);
    } catch (...) {
        if (buffer) ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        throw;
    }
}

int main(int argc, char ** argv) {
    auto backend = ggml_backend_cpu_init();
    if (!backend) return 1;
    try {
        if (argc == 7 && std::string(argv[1]) == "--case") {
            ggml_backend_cpu_set_n_threads(backend, std::stoi(argv[6]));
            run_case(backend, argv[2], std::stoi(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]));
            std::puts("PASS: selected convolution graph-boundary control");
        } else {
            require(argc == 1, "usage: test-ssm-conv-boundary [--case MODE CHANNELS TOKENS KERNEL THREADS]");
            size_t cases = 0;
            for (int threads : {1,2,8}) {
                ggml_backend_cpu_set_n_threads(backend, threads);
                for (int nr : {16,17,32}) for (int nt : {1,4,33,64}) for (int nc : {3,4})
                    for (const char * mode : {"terminal","terminal-view","full","prefix-conv","prefix-view"}) {
                        run_case(backend, mode, nr, nt, nc);
                        ++cases;
                    }
            }
            std::printf("PASS: %zu convolution cases; graph boundaries, scalar outputs, exact histories and inputs\n", cases);
        }
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        ggml_backend_free(backend);
        return 1;
    }
}
