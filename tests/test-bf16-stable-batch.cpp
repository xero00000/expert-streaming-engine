#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
#ifdef GGML_USE_CUDA
static int run(int device, int rows, const char * fixtures) {
    const int width = 2560;
    auto * backend = ggml_backend_cuda_init(device, nullptr, nullptr); check(backend, "backend");
    auto * wc = ggml_init({1024*1024, nullptr, true}); check(wc, "weight context");
    auto * weights = ggml_new_tensor_2d(wc, GGML_TYPE_BF16, width, rows);
    auto * wb = ggml_backend_alloc_ctx_tensors(wc, backend); check(wb, "weight allocation");
    ggml_backend_buffer_set_usage(wb, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<ggml_bf16_t> data(size_t(width)*rows);
    if (fixtures) {
        const std::string name = std::string(fixtures) + (rows == 2560 ? "/blk.1.ple_value.weight.bf16" : "/blk.1.ple_key.weight.bf16");
        std::ifstream file(name, std::ios::binary | std::ios::ate); check(bool(file), "fixture open");
        check(size_t(file.tellg()) == data.size()*2, "fixture geometry"); file.seekg(0);
        file.read(reinterpret_cast<char *>(data.data()), data.size()*2); check(bool(file), "fixture read");
    } else {
        std::vector<ggml_bf16_t> table(4096);
        for (size_t i=0; i<table.size(); ++i) table[i]=ggml_fp32_to_bf16(.03f*std::sin(float(i)*.017f));
        for (int r=0;r<rows;++r) for (int k=0;k<width;++k) data[size_t(r)*width+k]=table[(r*37+k*13)%4096];
    }
    ggml_backend_tensor_set(weights, data.data(), 0, data.size()*2);
    int cases = 0;
    for (int tokens=1; tokens<=8; ++tokens) {
        auto * ctx=ggml_init({8*1024*1024,nullptr,true}); check(ctx,"context");
        auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,tokens);
        auto * batch=ggml_mul_mat(ctx,weights,x);
        auto * graph=ggml_new_graph_custom(ctx,1024,false); ggml_build_forward_expand(graph,batch);
        std::vector<ggml_tensor *> one;
        for (int t=0;t<tokens;++t) {
            auto * column=ggml_view_2d(ctx,x,width,1,x->nb[1],t*x->nb[1]);
            one.push_back(ggml_mul_mat(ctx,weights,column)); ggml_build_forward_expand(graph,one.back());
        }
        auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); check(buffer,"compute allocation");
        std::vector<float> input(size_t(width)*tokens), result(size_t(rows)*tokens), reference(rows);
        for (int pattern=0;pattern<4;++pattern) {
            for (int t=0;t<tokens;++t) for (int k=0;k<width;++k) {
                float value=.3f*std::cos(float(k)*.071f+float(t)*.31f);
                if(pattern==1)value*=1.e-7f;
                if(pattern==2)value=k==(t*17+1)%width?1.f:0.f;
                if(pattern==3)value*=32768.f;
                input[size_t(t)*width+k]=value;
            }
            ggml_backend_tensor_set(x,input.data(),0,input.size()*4);
            // Repeated graph execution covers capture/replay and changing inputs.
            for(int replay=0;replay<3;++replay) check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"graph compute");
            ggml_backend_tensor_get(batch,result.data(),0,result.size()*4);
            for(int t=0;t<tokens;++t) {
                ggml_backend_tensor_get(one[t],reference.data(),0,reference.size()*4);
                check(std::memcmp(result.data()+size_t(t)*rows,reference.data(),rows*4)==0,"batch differs from scalar bits");
                for(int row:{0,rows/2,rows-1}) {
                    double expected=0,absolute=0;
                    for(int k=0;k<width;++k) {
                        const double w=ggml_bf16_to_fp32(data[size_t(row)*width+k]);
                        const double v=ggml_bf16_to_fp32(ggml_fp32_to_bf16(input[size_t(t)*width+k]));
                        expected+=w*v;absolute+=std::fabs(w*v);
                    }
                    // Bound FP32 accumulation and BF16 intermediate/final rounding.
                    check(std::isfinite(result[size_t(t)*rows+row]) && std::fabs(result[size_t(t)*rows+row]-expected)<=.01*absolute+1.e-6,"quantization-aware double oracle");
                }
            }
            ++cases;
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_buffer_free(wb); ggml_free(wc); ggml_backend_free(backend);
    return cases;
}
#endif
int main(int argc,char ** argv) {
    try {
#ifdef GGML_USE_CUDA
        const int device=argc>1?std::stoi(argv[1]):0;
        const char * flag=std::getenv("ESE_CUDA_BF16_STABLE_BATCH");
        if(!flag || std::strcmp(flag,"1")) { std::puts("SKIP: set ESE_CUDA_BF16_STABLE_BATCH=1"); return 0; }
        check(cuInit(0)==CUDA_SUCCESS,"driver"); CUdevice dev; CUcontext primary;
        check(cuDeviceGet(&dev,device)==CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&primary,dev)==CUDA_SUCCESS && cuCtxSetCurrent(primary)==CUDA_SUCCESS,"primary context");
        const char * fixtures=argc>2?argv[2]:nullptr;
        const int cases=run(device,2560,fixtures)+run(device,10240,fixtures);
        check(cuDevicePrimaryCtxRelease(dev)==CUDA_SUCCESS,"primary release");
        std::printf("PASS: %d backend BF16 batch cases; scalar bits, double oracle, graph replay\n",cases);
#else
        (void)argc;(void)argv; std::puts("SKIP: CUDA BF16 batch test");
#endif
        return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
