#include "ggml.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool ok,const char * what) {if(!ok)throw std::runtime_error(what);}
#ifdef GGML_USE_CUDA
static double activation(double g,ggml_unary_op op) {
    if(op==GGML_UNARY_OP_RELU)return std::max(g,0.0);
    if(op==GGML_UNARY_OP_GELU)return .5*g*(1+std::tanh(.7978845608028654*g*(1+.044715*g*g)));
    return g>=0?g/(1+std::exp(-g)):g*std::exp(g)/(1+std::exp(g));
}
static void run(int device,ggml_type type,int width,int rows,int tokens,ggml_unary_op op) {
    auto * backend=ggml_backend_cuda_init(device,nullptr,nullptr);require(backend,"backend");
    auto * wc=ggml_init({1024*1024,nullptr,true});require(wc,"weight context");
    auto * up=ggml_new_tensor_2d(wc,type,width,rows),*gate=ggml_new_tensor_2d(wc,type,width,rows);
    auto * wb=ggml_backend_alloc_ctx_tensors(wc,backend);require(wb,"weights");
    ggml_backend_buffer_set_usage(wb,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    std::vector<float> values(size_t(width)*rows),importance(width,1.f),u(values.size()),g(values.size());
    const auto traits=ggml_internal_get_type_traits(type);require(traits.to_float,"dequantizer");
    for(int side=0;side<2;++side) {
        for(int row=0;row<rows;++row)for(int col=0;col<width;++col)
            values[size_t(row)*width+col]=side?.04f*std::sin(col*.027f+row*.17f):.02f*std::cos(col*.033f+row*.23f);
        std::vector<uint8_t> bytes(ggml_nbytes(up));
        require(ggml_quantize_chunk(type,values.data(),bytes.data(),0,rows,width,importance.data(),nullptr)==bytes.size(),"quantization size");
        ggml_backend_tensor_set(side?gate:up,bytes.data(),0,bytes.size());
        traits.to_float(bytes.data(),side?g.data():u.data(),values.size());
    }
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});require(ctx,"context");
    auto * x=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,tokens);
    auto * batch=ggml_fused_up_gate(ctx,up,gate,x,op);
    require(ggml_backend_supports_op(backend,batch),"supported fused operator");
    auto * graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,batch);
    std::vector<ggml_tensor *> single;
    for(int t=0;t<tokens;++t) {
        auto * one=ggml_view_2d(ctx,x,width,1,x->nb[1],t*x->nb[1]);
        auto * y=ggml_fused_up_gate(ctx,up,gate,one,op);single.push_back(y);ggml_build_forward_expand(graph,y);
    }
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"allocation");
    std::vector<float> inputs(size_t(width)*tokens),got(size_t(rows)*tokens),serial(rows);
    for(int t=0;t<tokens;++t)for(int col=0;col<width;++col) {
        const float scale=t%4==1?0.f:t%4==2?1.e-4f:t%4==3?12.f:.7f;
        inputs[size_t(t)*width+col]=scale*std::cos(col*.071f+t*.37f);
    }
    ggml_backend_tensor_set(x,inputs.data(),0,inputs.size()*4);
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"compute");
    ggml_backend_tensor_get(batch,got.data(),0,got.size()*4);
    bool exact=true,oracle=true;
    for(int t=0;t<tokens;++t) {
        ggml_backend_tensor_get(single[t],serial.data(),0,serial.size()*4);
        for(int row=0;row<rows;++row) {
            const float value=got[size_t(t)*rows+row];require(std::isfinite(value)&&std::isfinite(serial[row]),"finite result");
            exact &= std::memcmp(&value,&serial[row],4)==0;
        }
        // Independent dequantized double oracle on edge/middle rows. Its bound
        // includes activation quantization; bit equality above covers every row.
        for(int row:{0,rows/2,rows-1}) {
            double du=0,dg=0,eu=0,eg=0;
            for(int first=0;first<width;first+=32) {
                double maximum=0;
                for(int col=first;col<first+32;++col)maximum=std::max(maximum,std::fabs(double(inputs[size_t(t)*width+col])));
                const double input_error=maximum*(1./254+1./2048);
                for(int col=first;col<first+32;++col) {
                    const double a=u[size_t(row)*width+col],b=g[size_t(row)*width+col],v=inputs[size_t(t)*width+col];
                    du+=a*v;dg+=b*v;eu+=4*std::fabs(a)*input_error;eg+=4*std::fabs(b)*input_error;
                }
            }
            const double ag=activation(dg,op),expected=du*ag;
            const double bound=std::fabs(ag)*eu+1.15*std::fabs(du)*eg+1.15*eu*eg+5.e-4*(1+std::fabs(expected));
            oracle &= std::fabs(double(got[size_t(t)*rows+row])-expected)<=bound;
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_buffer_free(wb);ggml_free(wc);ggml_backend_free(backend);
    if(!exact||!oracle)std::fprintf(stderr,"fused failure device=%d type=%d width=%d rows=%d tokens=%d op=%d exact=%d oracle=%d\n",device,int(type),width,rows,tokens,int(op),exact,oracle);
    require(exact&&oracle,"parallel fused batch disagrees with scalar arithmetic or quantization-aware oracle");
}
#endif
int main(int argc,char ** argv) {
 try {
#ifdef GGML_USE_CUDA
    const int device=argc>1?std::stoi(argv[1]):0;
    const char * enabled=std::getenv("ESE_CUDA_FUSED_BATCH_VECTOR");require(enabled&&std::strcmp(enabled,"1")==0,"set ESE_CUDA_FUSED_BATCH_VECTOR=1");
    require(cuInit(0)==CUDA_SUCCESS,"driver init");CUdevice dev;CUcontext primary;
    require(cuDeviceGet(&dev,device)==CUDA_SUCCESS&&cuDevicePrimaryCtxRetain(&primary,dev)==CUDA_SUCCESS&&cuCtxSetCurrent(primary)==CUDA_SUCCESS,"primary context");
    int cases=0;
    for(auto type:{GGML_TYPE_Q4_0,GGML_TYPE_Q4_K,GGML_TYPE_Q6_K,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_NL,GGML_TYPE_IQ4_XS})
        for(auto shape:{std::pair{512,31},std::pair{2560,640}})for(int tokens:{1,2,3,4,5,8})
            for(auto op:{GGML_UNARY_OP_SILU,GGML_UNARY_OP_RELU,GGML_UNARY_OP_GELU}) {run(device,type,shape.first,shape.second,tokens,op);++cases;}
    require(cuDevicePrimaryCtxRelease(dev)==CUDA_SUCCESS,"primary release");
    std::printf("PASS: %d fused dense batch cases match scalar bits and quantization-aware double oracle\n",cases);
#else
    (void)argc;(void)argv;std::puts("SKIP: CUDA fused dense batch test");
#endif
    return 0;
 }catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
