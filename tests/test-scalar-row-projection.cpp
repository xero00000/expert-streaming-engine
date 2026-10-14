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
#include <vector>

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

#ifdef GGML_USE_CUDA
static void run(int device, enum ggml_type type, int width, int tokens) {
    auto * backend=ggml_backend_cuda_init(device,nullptr,nullptr);
    check(backend,"CUDA backend");
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});check(ctx,"tensor context");
    auto * w=ggml_new_tensor_2d(ctx,type,width,1);
    auto * storage=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width+17,tokens);
    auto * x=ggml_view_2d(ctx,storage,width,tokens,storage->nb[1],0);
    auto * batch=ggml_mul_mat(ctx,w,x);ggml_mul_mat_set_prec(batch,GGML_PREC_F32);
    auto * graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,batch);
    std::vector<ggml_tensor *> reference;
    for(int i=0;i<tokens;i++) {
        auto * column=ggml_view_2d(ctx,x,width,1,x->nb[1],i*x->nb[1]);
        auto * single=ggml_mul_mat(ctx,w,column); // established single-row DEFAULT arithmetic
        reference.push_back(single);ggml_build_forward_expand(graph,single);
    }
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);check(buffer,"tensor allocation");
    std::vector<float> weights(width),values((width+17)*tokens,-12345.f);
    for(int i=0;i<width;i++) weights[i]=std::sin(float(i)*.17f)*.037f;
    for(int t=0;t<tokens;t++) for(int i=0;i<width;i++) values[t*(width+17)+i]=std::cos(float(i)*.071f+float(t))*.91f;
    if(type==GGML_TYPE_F32) ggml_backend_tensor_set(w,weights.data(),0,width*sizeof(float));
    else if(type==GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> bytes(width);for(int i=0;i<width;i++) bytes[i]=ggml_fp32_to_fp16(weights[i]);
        ggml_backend_tensor_set(w,bytes.data(),0,width*sizeof(ggml_fp16_t));
    } else {
        std::vector<ggml_bf16_t> bytes(width);for(int i=0;i<width;i++) bytes[i]=ggml_fp32_to_bf16(weights[i]);
        ggml_backend_tensor_set(w,bytes.data(),0,width*sizeof(ggml_bf16_t));
    }
    ggml_backend_tensor_set(storage,values.data(),0,values.size()*sizeof(float));
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"projection computation");
    std::vector<float> result(tokens);ggml_backend_tensor_get(batch,result.data(),0,result.size()*sizeof(float));
    bool exact=true;
    for(int t=0;t<tokens;t++) {
        float single;ggml_backend_tensor_get(reference[t],&single,0,sizeof(single));
        if(std::memcmp(&single,&result[t],sizeof(float))) {
            std::fprintf(stderr,"mismatch device=%d type=%d width=%d tokens=%d token=%d single=%.9g batch=%.9g\n",
                    device,int(type),width,tokens,t,single,result[t]);exact=false;
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    check(exact,"batched scalar projection changed single-token arithmetic");
}
#endif

int main(int argc,char ** argv) {
    try {
#ifdef GGML_USE_CUDA
        const int device=argc>1?std::stoi(argv[1]):0;
        check(cuInit(0)==CUDA_SUCCESS,"CUDA driver initialization");CUdevice dev;CUcontext primary;
        check(cuDeviceGet(&dev,device)==CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&primary,dev)==CUDA_SUCCESS &&
                cuCtxSetCurrent(primary)==CUDA_SUCCESS,"CUDA primary context activation");
        for(auto type:{GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_BF16}) for(int width:{2559,2560}) for(int n:{1,2,4,8}) run(device,type,width,n);
        check(cuDevicePrimaryCtxRelease(dev)==CUDA_SUCCESS,"CUDA primary context release");
        std::puts("PASS: scalar-row FP32 batches match single-token math for F32/F16/BF16, odd widths and padded inputs");
#else
        (void)argc;(void)argv;std::puts("SKIP: CUDA scalar-row projection test");
#endif
        return 0;
    } catch(const std::exception & error) {std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
