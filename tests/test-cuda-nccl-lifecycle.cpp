#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

struct reduction {
    int model_tag = 0;
    int count;
    ggml_type type;
    ggml_context * metadata = nullptr;
    std::vector<ggml_backend_t> backends;
    std::vector<ggml_backend_buffer_t> buffers;
    std::vector<ggml_tensor *> sources;
    ggml_cgraph * graph = nullptr;
    static constexpr int elements = 256;
    explicit reduction(int count, ggml_type type) : count(count), type(type) {
        metadata = ggml_init({1024*1024, nullptr, true});
        check(metadata != nullptr, "metadata allocation");
        for (int i = 0; i < count; ++i) {
            auto * backend = ggml_backend_cuda_init(i, nullptr, &model_tag);
            check(backend != nullptr, "backend initialization");
            backends.push_back(backend);
            auto * tensor = ggml_new_tensor_1d(metadata, type, elements);
            const auto buft = ggml_backend_get_default_buffer_type(backend);
            auto * buffer = ggml_backend_buft_alloc_buffer(buft,
                    ggml_backend_buft_get_alloc_size(buft, tensor));
            check(buffer != nullptr, "source allocation");
            ggml_backend_tensor_alloc(buffer, tensor, ggml_backend_buffer_get_base(buffer));
            check(tensor->data != nullptr, "source tensor allocation");
            buffers.push_back(buffer);sources.push_back(tensor);
        }
        auto * sum = ggml_reduce(metadata, sources.data(), count, GGML_OP_ADD);
        ggml_set_output(sum);
        graph = ggml_new_graph_custom(metadata, 8, false);
        ggml_build_forward_expand(graph, sum);
        check(graph->n_nodes == 1, "unexpected reduction graph");
    }
    ~reduction() {
        for (auto backend : backends) ggml_backend_synchronize(backend);
        for (auto buffer : buffers) ggml_backend_buffer_free(buffer);
        for (auto backend : backends) ggml_backend_free(backend);
        ggml_free(metadata);
    }
    std::vector<uint8_t> encode(const std::vector<float> & values) const {
        std::vector<uint8_t> result(values.size()*ggml_type_size(type));
        for (size_t i = 0; i < values.size(); ++i) {
            auto * destination = result.data() + i*ggml_type_size(type);
            if (type == GGML_TYPE_F32) {
                std::memcpy(destination, &values[i], sizeof(float));
            } else if (type == GGML_TYPE_F16) {
                const auto value = ggml_fp32_to_fp16(values[i]);
                std::memcpy(destination, &value, sizeof(value));
            } else {
                const auto value = ggml_fp32_to_bf16(values[i]);
                std::memcpy(destination, &value, sizeof(value));
            }
        }
        return result;
    }
    void run(int repeats) {
        for (int iteration = 0; iteration < repeats; ++iteration) {
            std::vector<float> values(elements), expected(elements);
            std::vector<uint8_t> actual(elements*ggml_type_size(type));
            for (int device = 0; device < count; ++device) {
                for (int i = 0; i < elements; ++i) {
                    values[i] = float(1 + device%4 + i%4 + iteration%4);
                    expected[i] += values[i];
                }
                const auto encoded = encode(values);
                ggml_backend_tensor_set(sources[device], encoded.data(), 0, encoded.size());
            }
            check(ggml_backend_graph_compute(backends.back(), graph) == GGML_STATUS_SUCCESS,
                    "reduction graph compute");
            const auto encoded_expected = encode(expected);
            for (int device = 0; device < count; ++device) {
                ggml_backend_synchronize(backends[device]);
                ggml_backend_tensor_get(sources[device], actual.data(), 0, actual.size());
                check(actual == encoded_expected,
                        "reduction output differs from exact integer sum");
            }
        }
    }
};

int main(int argc, char ** argv) {
    const int visible = ggml_backend_cuda_get_device_count();
    if (visible < 2) { std::puts("SKIP: reduction needs two visible GPUs"); return 77; }
    try {
        const bool partial = argc > 1 && std::strcmp(argv[1], "partial") == 0;
        const int count = partial ? 2 : visible;
        for (auto type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16}) {
        reduction a(count, type);
        reduction b(count, type);
        // Independent context maps are constructed serially. The first NCCL
        // use races from two threads; groups on the shared communicator must
        // remain owned by a single thread at a time.
        std::atomic<int> ready{0};std::atomic<bool> go{false};
        std::exception_ptr failures[2];
        auto work = [&](int index, reduction & value) {
            ++ready;while (!go.load()) std::this_thread::yield();
            try { value.run(8); } catch (...) { failures[index] = std::current_exception(); }
        };
        std::thread one(work, 0, std::ref(a));
        std::thread two(work, 1, std::ref(b));
        while (ready.load() != 2) std::this_thread::yield();
        go = true;one.join();two.join();
        for (auto & failure : failures) if (failure) std::rethrow_exception(failure);
        std::printf("cuda_collective: visible=%d participants=%d type=%s contexts=2 groups=16 exact=true\n",
                visible,count,ggml_type_name(type));
        }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());return 1;
    }
}
