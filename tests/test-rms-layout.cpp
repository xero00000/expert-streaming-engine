#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

#ifdef GGML_USE_CUDA
static void run(int device, int width, int tokens, int pattern) {
    constexpr int heads=16;
    auto * backend=ggml_backend_cuda_init(device,nullptr,nullptr); require(backend,"CUDA backend");
    auto * ctx=ggml_init({8*1024*1024,nullptr,true}); require(ctx,"tensor context");
    // Convolved Q/K: consecutive heads inside each padded token column.
    auto * storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width*heads+17,tokens);
    auto * input=ggml_view_3d(ctx,storage,width,tokens,heads,storage->nb[1],width*sizeof(float),0);
    auto * batch=ggml_rms_norm(ctx,input,1.e-6f/width);
    auto * graph=ggml_new_graph_custom(ctx,1024,false); ggml_build_forward_expand(graph,batch);
    std::vector<ggml_tensor *> refs;
    for(int t=0;t<tokens;++t) {
        auto * column=ggml_view_2d(ctx,storage,width,heads,width*sizeof(float),t*storage->nb[1]);
        require(ggml_is_contiguous(column),"contiguous reference rows");
        refs.push_back(ggml_rms_norm(ctx,column,1.e-6f/width));ggml_build_forward_expand(graph,refs.back());
    }
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend); require(buffer,"tensor allocation");
    std::vector<float> values(size_t(width*heads+17)*tokens,-12345.f);
    for(int t=0;t<tokens;++t) for(int head=0;head<heads;++head) for(int k=0;k<width;++k) {
        float x=std::sin(float(k)*.17f+float(head)*.031f+float(t))*.037f;
        if(pattern==1) x*=32768.f;
        if(pattern==2) x*=1.e-7f;
        if(pattern==3) x=0.f;
        values[size_t(t)*(width*heads+17)+head*width+k]=x;
    }
    ggml_backend_tensor_set(storage,values.data(),0,values.size()*sizeof(float));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"RMS compute");
    std::vector<float> got(size_t(width)*heads*tokens),single(size_t(width)*heads);
    ggml_backend_tensor_get(batch,got.data(),0,got.size()*sizeof(float)); size_t changed=0;bool formula=true;
    for(int t=0;t<tokens;++t) {
        ggml_backend_tensor_get(refs[t],single.data(),0,single.size()*sizeof(float));
        for(int head=0;head<heads;++head) {
            double squares=0;
            for(int k=0;k<width;++k) {double x=values[size_t(t)*(width*heads+17)+head*width+k];squares+=x*x;}
            const double scale=1.0/std::sqrt(squares/width+double(1.e-6f/width));
            for(int k=0;k<width;++k) {
                const float value=got[(size_t(head)*tokens+t)*width+k],ref=single[size_t(head)*width+k];
                changed+=std::memcmp(&value,&ref,sizeof(float))!=0;
                const double expected=values[size_t(t)*(width*heads+17)+head*width+k]*scale;
                formula &= std::isfinite(value) && std::fabs(value-expected)<=3.e-6*(1.0+std::fabs(expected));
            }
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    require(formula,"RMS differs from double CPU formula");
    if(changed) std::fprintf(stderr,"RMS layout mismatch: device=%d width=%d tokens=%d pattern=%d changed=%zu\n",device,width,tokens,pattern,changed);
    require(changed==0,"strided RMS changed contiguous-row arithmetic");
}
#endif

int main(int argc,char ** argv) {
    try {
#ifdef GGML_USE_CUDA
        const int device=argc>1?std::stoi(argv[1]):0;
        require(cuInit(0)==CUDA_SUCCESS,"driver init");CUdevice dev;CUcontext primary;
        require(cuDeviceGet(&dev,device)==CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&primary,dev)==CUDA_SUCCESS &&
                cuCtxSetCurrent(primary)==CUDA_SUCCESS,"primary context");int count=0;
        for(int width:{32,64,128,255,256,511,1023,1024,2560}) for(int tokens:{1,2,4,8}) for(int pattern=0;pattern<4;++pattern) {
            run(device,width,tokens,pattern);++count;
        }
        require(cuDevicePrimaryCtxRelease(dev)==CUDA_SUCCESS,"release primary context");
        std::printf("PASS: %d strided RMS cases match contiguous rows exactly and double CPU formula\n",count);
#else
        (void)argc;(void)argv;std::puts("SKIP: CUDA RMS layout test");
#endif
        return 0;
    } catch(const std::exception & error) {std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
