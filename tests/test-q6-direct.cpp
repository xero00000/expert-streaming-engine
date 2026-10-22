#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
template<typename T> static std::vector<T> read(const std::string & path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);check(bool(f),"fixture open");
    const size_t bytes=size_t(f.tellg());check(bytes%sizeof(T)==0,"fixture size");
    std::vector<T> result(bytes/sizeof(T));f.seekg(0);f.read((char *)result.data(),bytes);check(bool(f),"fixture read");return result;
}

int main(int argc,char ** argv) {
    try {
        check(argc==11,"usage: model offset width rows input.f32 gold.f32 device precision outputs.bin threads");
        const int width=std::stoi(argv[3]), master_rows=std::stoi(argv[4]), device=std::stoi(argv[7]);
        const int only_precision=std::stoi(argv[8]);
        check(width>0&&width%256==0&&master_rows>=257&&(only_precision==0||only_precision==1),"fixture geometry");
        const auto input=read<float>(argv[5]), gold=read<float>(argv[6]);
        check(input.size()==size_t(width)*8&&gold.size()==size_t(master_rows)*8,"oracle shape");
        const size_t row_bytes=ggml_row_size(GGML_TYPE_Q6_K,width);
        std::ifstream model(argv[1],std::ios::binary);check(bool(model),"model read-only open");
        model.seekg(std::stoull(argv[2]));std::vector<unsigned char> packed(row_bytes*master_rows);
        model.read((char *)packed.data(),packed.size());check(bool(model),"original head payload read");
        ggml_backend_t raw=nullptr;
#ifdef GGML_USE_CUDA
        CUdevice cu_device=0;CUcontext primary=nullptr;
        if(device>=0) {
            check(cuInit(0)==CUDA_SUCCESS&&cuDeviceGet(&cu_device,device)==CUDA_SUCCESS&&
                  cuDevicePrimaryCtxRetain(&primary,cu_device)==CUDA_SUCCESS&&cuCtxSetCurrent(primary)==CUDA_SUCCESS,"primary context");
            raw=ggml_backend_cuda_init(device,nullptr,nullptr);
        }
#else
        check(device<0,"CUDA unavailable");
#endif
        if(device<0) {raw=ggml_backend_cpu_init();ggml_backend_cpu_set_n_threads(raw,std::stoi(argv[10]));}
        std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(raw,ggml_backend_free);check(bool(backend),"backend");
        std::ofstream artifact(argv[9],std::ios::binary);check(bool(artifact),"artifact open");
        size_t values=0,cases=0;double maximum=0;
        for(int rows:{7,257,master_rows}) for(int tokens:{1,2,8})
            for(int layout=0;layout<(rows==master_rows||!only_precision?1:2);++layout) {
            const bool padded=layout==1;
            const int padding=padded?17:0;
            std::unique_ptr<ggml_context,decltype(&ggml_free)> ctx(ggml_init({8<<20,nullptr,true}),ggml_free);check(bool(ctx),"context");
            auto * parent_w=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_Q6_K,width+(padded?256:0),rows);
            auto * w=ggml_view_2d(ctx.get(),parent_w,width,rows,parent_w->nb[1],0);
            ggml_tensor * storage[2], * x[2];
            for(int i=0;i<2;++i) {
                storage[i]=ggml_new_tensor_2d(ctx.get(),GGML_TYPE_F32,width+padding,tokens);
                x[i]=ggml_view_2d(ctx.get(),storage[i],width,tokens,storage[i]->nb[1],0);
            }
            auto * y=ggml_mul_mat(ctx.get(),w,x[0]);
            if(padded) {y->nb[1]=(rows+padding)*sizeof(float);y->nb[2]=y->nb[1]*tokens;y->nb[3]=y->nb[2];}
            auto * graph=ggml_new_graph_custom(ctx.get(),128,false);ggml_build_forward_expand(graph,y);
            check(ggml_backend_supports_op(backend.get(),y),"operator unsupported");
            std::unique_ptr<ggml_backend_buffer,decltype(&ggml_backend_buffer_free)> buffer(
                ggml_backend_alloc_ctx_tensors(ctx.get(),backend.get()),ggml_backend_buffer_free);check(bool(buffer),"allocation");
            std::vector<unsigned char> weights(ggml_nbytes(parent_w),0xff);
            for(int row=0;row<rows;++row)std::memcpy(weights.data()+row*parent_w->nb[1],packed.data()+row*row_bytes,row_bytes);
            ggml_backend_tensor_set(parent_w,weights.data(),0,weights.size());
            for(int ptr=0;ptr<2;++ptr) {
                std::vector<float> data(size_t(width+padding)*tokens,-12345.f);
                for(int token=0;token<tokens;++token)std::copy_n(input.data()+((token+ptr)%8)*width,width,data.data()+token*(width+padding));
                ggml_backend_tensor_set(storage[ptr],data.data(),0,data.size()*sizeof(float));
            }
            std::vector<float> initial(ggml_nbytes(y)/sizeof(float),-6789.f);
            ggml_backend_tensor_set(y,initial.data(),0,initial.size()*sizeof(float));
            std::vector<int> modes=only_precision?std::vector<int>{1}:std::vector<int>{0};
            // Tail/padded cases exercise transitions from the flat reference
            // topology to the direct kernel; the full head tests each policy.
            if(rows!=master_rows&&!padded&&only_precision)modes={0,1,0,1};
            std::vector<float> previous[2];
            for(int mode:modes) for(int ptr=0;ptr<2;++ptr) {
                ggml_mul_mat_set_prec(y,mode?GGML_PREC_F32:GGML_PREC_DEFAULT);y->src[1]=x[ptr];
                std::vector<float> first;
                for(int replay=0;replay<3;++replay) {
                    check(ggml_backend_graph_compute(backend.get(),graph)==GGML_STATUS_SUCCESS,"compute");
                    std::vector<float> got(size_t(rows)*tokens);
                    for(int token=0;token<tokens;++token)ggml_backend_tensor_get(y,got.data()+size_t(token)*rows,token*y->nb[1],rows*sizeof(float));
                    if(replay)check(std::memcmp(first.data(),got.data(),got.size()*sizeof(float))==0,"replay changed bits");
                    else {first=got;artifact.write((const char *)got.data(),got.size()*sizeof(float));}
                    for(int token=0;token<tokens;++token)for(int row=0;row<rows;++row) {
                        const float actual=got[size_t(token)*rows+row];check(std::isfinite(actual),"nonfinite result");
                        const float expected=gold[size_t((token+ptr)%8)*master_rows+row];
                        const double error=std::abs(double(actual)-expected);if(mode)maximum=std::max(maximum,error);
                        if(mode&&error>2e-5*(1+std::abs(double(expected)))) {
                            std::fprintf(stderr,"row=%d token=%d actual=%.9g expected=%.9g error=%.9g\n",row,token,actual,expected,error);
                            throw std::runtime_error("original-weight oracle mismatch");
                        }
                        ++values;
                    }
                    if(mode==0) {
                        if(previous[ptr].empty())previous[ptr]=got;
                        else check(std::memcmp(previous[ptr].data(),got.data(),got.size()*sizeof(float))==0,"default bits changed after precision switch");
                    }
                    ++cases;
                }
            }
            for(int ptr=0;ptr<2;++ptr) {
                std::vector<float> data(size_t(width+padding)*tokens);
                ggml_backend_tensor_get(storage[ptr],data.data(),0,data.size()*sizeof(float));
                for(int token=0;token<tokens;++token)for(int i=width;i<width+padding;++i)check(data[token*(width+padding)+i]==-12345.f,"input padding overwritten");
            }
            if(padded) {
                std::vector<float> data(initial.size());ggml_backend_tensor_get(y,data.data(),0,data.size()*sizeof(float));
                for(int token=0;token<tokens-1;++token)for(int row=rows;row<rows+padding;++row)check(data[token*(rows+padding)+row]==-6789.f,"output padding overwritten");
            }
        }
        artifact.close();check(bool(artifact),"artifact write");ggml_backend_synchronize(backend.get());backend.reset();
#ifdef GGML_USE_CUDA
        if(device>=0)check(cuDevicePrimaryCtxRelease(cu_device)==CUDA_SUCCESS,"primary release");
#endif
        std::printf("PASS: q6_direct device=%d precision=%d cases=%zu values=%zu max_abs=%.9g\n",device,only_precision,cases,values,maximum);
    } catch(const std::exception & error) {std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
    return 0;
}
