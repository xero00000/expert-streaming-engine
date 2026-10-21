#include "../ggml/src/ggml-cuda/convert.cuh"
#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}
static __global__ void scalar_reference(const nv_bfloat16 * x, float * y, int64_t n) {
    const int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if (i < n) y[i] = __bfloat162float(x[i]);
}

static void run_case(int rows, int columns, int source_offset, int destination_offset) {
    const int64_t n = int64_t(rows)*columns;
    nv_bfloat16 * source = nullptr;
    float * output = nullptr, * reference = nullptr;
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    check(cudaMalloc(&source,(n+16)*sizeof(nv_bfloat16)));
    check(cudaMalloc(&output,(n+32)*sizeof(float)));
    check(cudaMalloc(&reference,n*sizeof(float)));
    struct cleanup {
        nv_bfloat16 * source; float * output; float * reference; cudaStream_t stream;
        ~cleanup() { cudaFree(source); cudaFree(output); cudaFree(reference); cudaStreamDestroy(stream); }
    } free_buffers{source,output,reference,stream};
    const auto convert = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);
    require(convert != nullptr,"BF16 converter missing");
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    convert(source+source_offset,output+destination_offset,rows,columns,stream);
    check(cudaStreamEndCapture(stream,&graph));
    check(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    struct graph_cleanup {
        cudaGraph_t graph; cudaGraphExec_t executable;
        ~graph_cleanup() { cudaGraphExecDestroy(executable); cudaGraphDestroy(graph); }
    } free_graph{graph,executable};
    std::vector<uint16_t> input(n+16);
    std::vector<uint32_t> actual(n+32), expected(n);
    for (int replay = 0; replay < 3; ++replay) {
        // Every BF16 bit pattern, including signed zeros, subnormals,
        // infinities and signaling/quiet NaNs, occurs in the 65536-row case.
        for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i*40503+replay*12345);
        check(cudaMemcpyAsync(source,input.data(),input.size()*sizeof(uint16_t),cudaMemcpyHostToDevice,stream));
        check(cudaMemsetAsync(output,0xa5,(n+32)*sizeof(float),stream));
        check(cudaGraphLaunch(executable,stream));
        scalar_reference<<<(n+255)/256,256,0,stream>>>(source+source_offset,reference,n);
        check(cudaGetLastError());
        check(cudaMemcpyAsync(actual.data(),output,actual.size()*sizeof(uint32_t),cudaMemcpyDeviceToHost,stream));
        check(cudaMemcpyAsync(expected.data(),reference,n*sizeof(float),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));
        require(std::memcmp(actual.data()+destination_offset,expected.data(),n*sizeof(float)) == 0,
                "BF16 expansion differs from scalar CUDA conversion");
        for (int64_t i = 0; i < n; ++i) {
            const auto bits = input[i+source_offset];
            if ((bits&0x7f80) != 0x7f80 || (bits&0x7f) == 0)
                require(expected[i] == uint32_t(bits)<<16,"BF16 expansion differs from independent bit oracle");
        }
        for (int64_t i = 0; i < int64_t(actual.size()); ++i)
            if (i < destination_offset || i >= n+destination_offset)
                require(actual[i] == 0xa5a5a5a5,"BF16 converter overwrote destination guard");
    }
    if (n == 3276800 && source_offset == 0 && destination_offset == 0) {
        cudaEvent_t begin,end;
        check(cudaEventCreate(&begin)); check(cudaEventCreate(&end));
        float elapsed[2];
        for (int variant = 0; variant < 2; ++variant) {
            check(cudaEventRecord(begin,stream));
            for (int repeat = 0; repeat < 50; ++repeat) {
                if (variant == 0) scalar_reference<<<(n+255)/256,256,0,stream>>>(source,output,n);
                else convert(source,output,rows,columns,stream);
            }
            check(cudaEventRecord(end,stream)); check(cudaEventSynchronize(end));
            check(cudaEventElapsedTime(&elapsed[variant],begin,end));
        }
        check(cudaEventDestroy(begin)); check(cudaEventDestroy(end));
        std::printf("BF16 component timing: scalar_us=%.3f selected_us=%.3f (diagnostic only)\n",
                elapsed[0]*20,elapsed[1]*20);
    }
}

int main() {
    try {
        int devices = 0; check(cudaGetDeviceCount(&devices));
        if (devices == 0) return 77;
        for (int device = 0; device < devices; ++device) {
            check(cudaSetDevice(device));
            const std::array<std::array<int,2>,11> shapes = {{{1,1},{1,3},{1,4},{3,5},{1,255},
                    {1,256},{1,1023},{1,1024},{1,1025},{256,256},{10240,320}}};
            for (const auto shape : shapes)
                for (int src : {0,1}) for (int dst : {0,1}) run_case(shape[0],shape[1],src,dst);
            std::printf("PASS: BF16 converter device=%d shapes/tails/alignments/all-bit-patterns/guards/capture/replay exact\n",device);
        }
    } catch (const std::exception & error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
