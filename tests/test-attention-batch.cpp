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
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok,const char * what) {if(!ok) throw std::runtime_error(what);}

#ifdef GGML_USE_CUDA
static bool run(int device,int dk,int dv,int keys,int tokens,float softcap,int requested_heads=0) {
    constexpr int kvheads=2;
    const int heads=requested_heads?requested_heads:(dk==256?24:4);
    const int krow=dk+8, vrow=dv+16;
    auto * backend=ggml_backend_cuda_init(device,nullptr,nullptr);require(backend,"CUDA backend");
    auto * ctx=ggml_init({8*1024*1024,nullptr,true});require(ctx,"tensor context");
    auto * qs=ggml_new_tensor_3d(ctx,GGML_TYPE_F32,dk,heads,tokens);
    auto * q=ggml_permute(ctx,qs,0,2,1,3);
    auto * ks=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,krow,keys+17,kvheads);
    auto * vs=ggml_new_tensor_3d(ctx,GGML_TYPE_F16,vrow,keys+13,kvheads);
    auto * k=ggml_view_3d(ctx,ks,dk,keys,kvheads,ks->nb[1],ks->nb[2],0);
    auto * v=ggml_view_3d(ctx,vs,dv,keys,kvheads,vs->nb[1],vs->nb[2],0);
    const int padding=GGML_PAD(tokens,GGML_KQ_MASK_PAD);
    auto * ms=ggml_new_tensor_2d(ctx,GGML_TYPE_F16,keys,padding+tokens);
    auto * mask=ggml_view_2d(ctx,ms,keys,padding,ms->nb[1],0);
    const float scale=1.f/std::sqrt(float(dk));
    auto * batch=ggml_flash_attn_ext(ctx,q,k,v,mask,scale,0.f,softcap);
    const ggml_prec precision=GGML_PREC_DEFAULT;
    ggml_flash_attn_ext_set_prec(batch,precision);
    require(ggml_backend_supports_op(backend,batch),"supported DEFAULT attention must be advertised");
    auto * graph=ggml_new_graph_custom(ctx,1024,false);ggml_build_forward_expand(graph,batch);
    std::vector<ggml_tensor *> single;
    for(int t=0;t<tokens;++t) {
        auto * qt=ggml_view_3d(ctx,q,dk,1,heads,q->nb[1],q->nb[2],t*q->nb[1]);
        auto * mt=ggml_view_2d(ctx,ms,keys,16,ms->nb[1],t*ms->nb[1]);
        auto * output=ggml_flash_attn_ext(ctx,qt,k,v,mt,scale,0.f,softcap);
        ggml_flash_attn_ext_set_prec(output,precision);single.push_back(output);ggml_build_forward_expand(graph,output);
    }
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"tensor allocation");
    std::vector<float> qvalues(size_t(dk)*heads*tokens),kvalues(size_t(dk)*keys*kvheads),vvalues(size_t(dv)*keys*kvheads);
    std::vector<ggml_fp16_t> kstorage(size_t(krow)*(keys+17)*kvheads),vstorage(size_t(vrow)*(keys+13)*kvheads);
    std::vector<ggml_fp16_t> masks(size_t(keys)*(padding+tokens),ggml_fp32_to_fp16(-INFINITY));
    for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h) for(int col=0;col<dk;++col)
        qvalues[(size_t(t)*heads+h)*dk+col]=3.1f*std::sin(col*.031f+h*.29f+t*.13f);
    for(int h=0;h<kvheads;++h) for(int pos=0;pos<keys;++pos) {
        for(int col=0;col<dk;++col) {
            const auto half=ggml_fp32_to_fp16(.7f*std::cos(col*.027f+pos*.013f+h*.31f));
            kstorage[(size_t(h)*(keys+17)+pos)*krow+col]=half;kvalues[(size_t(h)*keys+pos)*dk+col]=ggml_fp16_to_fp32(half);
        }
        for(int col=0;col<dv;++col) {
            const auto half=ggml_fp32_to_fp16(.3f*std::sin(col*.047f+pos*.071f+h*.21f));
            vstorage[(size_t(h)*(keys+13)+pos)*vrow+col]=half;vvalues[(size_t(h)*keys+pos)*dv+col]=ggml_fp16_to_fp32(half);
        }
    }
    for(int t=0;t<tokens;++t) for(int pos=0;pos<=keys/2+t;++pos)
        if(pos%7!=3) masks[size_t(t)*keys+pos]=ggml_fp32_to_fp16(0.f);
    ggml_backend_tensor_set(qs,qvalues.data(),0,qvalues.size()*4);
    ggml_backend_tensor_set(ks,kstorage.data(),0,kstorage.size()*2);
    ggml_backend_tensor_set(vs,vstorage.data(),0,vstorage.size()*2);
    ggml_backend_tensor_set(ms,masks.data(),0,masks.size()*2);
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"attention compute");
    std::vector<float> got(size_t(dv)*heads*tokens),ref(size_t(dv)*heads);
    ggml_backend_tensor_get(batch,got.data(),0,got.size()*4);
    double maximum_error=0;bool finite=true,oracle=true,consistency=true;
    for(int t=0;t<tokens;++t) {
        ggml_backend_tensor_get(single[t],ref.data(),0,ref.size()*4);
        for(int h=0;h<heads;++h) {
            const int kh=h/(heads/kvheads);std::vector<double> scores(keys,-INFINITY);double maximum=-INFINITY;
            for(int pos=0;pos<keys;++pos) {
                if(!std::isfinite(ggml_fp16_to_fp32(masks[size_t(t)*keys+pos]))) continue;
                double dot=0;
                for(int col=0;col<dk;++col) dot+=double(qvalues[(size_t(t)*heads+h)*dk+col])*kvalues[(size_t(kh)*keys+pos)*dk+col];
                const double score=softcap==0.f?dot*scale:softcap*std::tanh(dot*scale/softcap);
                scores[pos]=score;maximum=std::max(maximum,score);
            }
            double sum=0;for(auto & score:scores) {score=std::exp(score-maximum);sum+=score;}
            for(int col=0;col<dv;++col) {
                double expected=0;
                for(int pos=0;pos<keys;++pos) expected+=scores[pos]*vvalues[(size_t(kh)*keys+pos)*dv+col];
                expected/=sum;const float value=got[(size_t(t)*heads+h)*dv+col],serial=ref[size_t(h)*dv+col];
                const double tolerance=2.e-3*(1.+std::fabs(expected));
                finite &= std::isfinite(value) && std::isfinite(serial);
                oracle &= std::fabs(value-expected)<=tolerance && std::fabs(serial-expected)<=tolerance;
                consistency &= std::memcmp(&value,&serial,sizeof(float)) == 0;
                maximum_error=std::max(maximum_error,std::fabs(value-expected));
            }
        }
    }
    ggml_backend_buffer_free(buffer);ggml_free(ctx);ggml_backend_free(backend);
    if(!finite || !oracle || !consistency)
        std::fprintf(stderr,"attention precision failure device=%d dk=%d dv=%d keys=%d tokens=%d softcap=%.1f max_cpu_error=%.9g\n",device,dk,dv,keys,tokens,softcap,maximum_error);
    require(finite && oracle,"DEFAULT attention disagrees with CPU oracle");
    std::printf("attention_batch: {\"device\":%d,\"keys\":%d,\"tokens\":%d,\"exact\":%s,\"cpu_max_abs\":%.9g}\n",device,keys,tokens,consistency?"true":"false",maximum_error);
    return consistency;
}

#endif

int main(int argc,char ** argv) {
    try {
#ifdef GGML_USE_CUDA
        const int device=argc>1?std::stoi(argv[1]):0;
        require(cuInit(0)==CUDA_SUCCESS,"driver init");CUdevice dev;CUcontext primary;
        require(cuDeviceGet(&dev,device)==CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&primary,dev)==CUDA_SUCCESS &&
                cuCtxSetCurrent(primary)==CUDA_SUCCESS,"primary context");int count=0;
        const char * control=std::getenv("ESE_CUDA_ATTENTION_STABLE_BATCH");
        require(control && std::strcmp(control,"1")==0,"set ESE_CUDA_ATTENTION_STABLE_BATCH=1 for canonical MMA qualification");
        bool exact=true;
        for(int width:{64,80,96,112,128,192,256}) for(int keys:{256,1024}) for(int tokens:{4,17}) {
            exact &= run(device,width,width,keys,tokens,0.f);++count;
        }
        for(int keys:{256,512,1024,4096}) for(int tokens:{1,2,4,8,17,31}) {
            exact &= run(device,256,256,keys,tokens,0.f);++count;
        }
        for(int heads:{8,48}) {exact &= run(device,256,256,1024,17,0.f,heads);++count;}
        require(cuDevicePrimaryCtxRelease(dev)==CUDA_SUCCESS,"primary release");
        std::printf("DEFAULT MMA scalar/batch: %d cases exact=%s\n",count,exact?"true":"false");
        if(!exact) return 3;
#else
        (void)argc;(void)argv;std::puts("SKIP: CUDA attention batch test");
#endif
        return 0;
    } catch(const std::exception & e) {std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
