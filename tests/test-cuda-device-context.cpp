#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <cuda.h>
#include <cstring>
#include <cuda_runtime.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

extern void ggml_cuda_set_device(int device);

static void runtime_check(cudaError_t result) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "runtime failure: %s\n", cudaGetErrorString(result));
        std::abort();
    }
}
static void driver_check(CUresult result) {
    if (result != CUDA_SUCCESS) {
        std::fprintf(stderr, "driver failure: %d\n", int(result));
        std::abort();
    }
}

int main() {
    int devices = 0;
    runtime_check(cudaGetDeviceCount(&devices));
    if (devices < 1) return 2;
    std::atomic<int> missing_context{0};
    std::atomic<int> wrong_device{0};
    auto select = [](int device) { ggml_cuda_set_device(device); };
    auto check = [&](int device) {
        CUcontext context = nullptr;
        driver_check(cuCtxGetCurrent(&context));
        if (!context) { ++missing_context; return; }
        CUdevice actual = -1;
        driver_check(cuCtxGetDevice(&actual));
        if (actual != device) ++wrong_device;
    };
    // Existing backend streams retain their creating device. Exercise both
    // one backend and two backends on the same GPU while another GPU is current.
    int copy_failures = 0;
    for (int owner = 0; owner < devices; ++owner) {
        auto * a = ggml_backend_cuda_init(owner, nullptr, nullptr);
        auto * b = ggml_backend_cuda_init(owner, nullptr, nullptr);
        auto * meta = ggml_init({1024 * 1024, nullptr, true});
        if (!a || !b || !meta) return 4;
        auto * source = ggml_new_tensor_1d(meta, GGML_TYPE_F32, 256);
        auto * destination = ggml_new_tensor_1d(meta, GGML_TYPE_F32, 256);
        auto * buffer = ggml_backend_alloc_ctx_tensors(meta, a);
        if (!buffer) return 4;
        std::vector<float> expected(256), actual(256);
        for (int i = 0; i < 256; ++i) expected[i] = float(i) / 16 - 7;
        ggml_backend_tensor_set(source, expected.data(), 0, expected.size() * sizeof(float));
        ggml_backend_tensor_copy_async(a, a, source, destination); // Prime the source stream.
        ggml_backend_synchronize(a);
        for (int repeat = 0; repeat < 8; ++repeat) {
            runtime_check(cudaSetDevice(devices > 1 ? (owner + 1) % devices : owner));
            if (devices == 1) driver_check(cuCtxSetCurrent(nullptr));
            ggml_backend_tensor_copy_async(a, repeat % 2 ? b : a, source, destination);
            ggml_backend_synchronize(a); ggml_backend_synchronize(b);
            ggml_backend_tensor_get(destination, actual.data(), 0, actual.size() * sizeof(float));
            if (std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(float))) ++copy_failures;
        }
        ggml_backend_buffer_free(buffer); ggml_free(meta);
        ggml_backend_free(b); ggml_backend_free(a);
    }
    std::printf("async_copy_device_context: devices=%d cases=%d exact=%s\n", devices, devices * 8, copy_failures ? "false" : "true");
    if (copy_failures) return 5;
    // Model loading starts one main thread and eight fresh upload workers.
    select(0);
    check(0);
    std::vector<std::thread> workers;
    for (int t = 0; t < 8; ++t) {
        workers.emplace_back([&] {
            select(0);
            check(0);
            for (int d = 0; d < devices; ++d) {
                select(d);
                check(d);
                runtime_check(cudaFree(nullptr));
                // Driver clients can detach a context between backend calls.
                driver_check(cuCtxSetCurrent(nullptr));
                select(d);
                check(d);
                select(0);
                check(0);
            }
        });
    }
    for (auto & worker : workers) worker.join();
    runtime_check(cudaSetDevice(0));
    for (int i = 0; i < 1000; ++i) { select(0); check(0); }
    std::printf("device_context: {\"devices\":%d,\"missing_context\":%d,\"wrong_device\":%d}\n",
        devices, missing_context.load(), wrong_device.load());
    return missing_context || wrong_device ? 3 : 0;
}
