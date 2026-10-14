#include "llama.h"
#include "ggml-alloc.h"
#include <random>
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../ggml/src/ggml-cuda-pool-memory.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool ok, const char * why) { if (!ok) throw std::runtime_error(why); }

#ifdef GGML_USE_CUDA
static ggml_cuda_pool_memory_stats snapshot(ggml_backend_t backend, int device) {
    ggml_cuda_pool_memory_stats stats{};
    check(ggml_backend_cuda_get_pool_memory_stats(backend,device,&stats),"CUDA pool snapshot");
    check(stats.active_bytes+stats.cached_bytes==stats.physical_bytes,"physical partition");
    check(stats.peak_physical_bytes>=stats.physical_bytes,"physical high water mark");
    return stats;
}
static uint64_t inventory(llama_context & ctx) {
    uint32_t count=0;
    check(llama_resource_get_memory_pools(&ctx,nullptr,0,&count),"inventory size");
    std::vector<llama_resource_memory_pool> rows(count);
    check(llama_resource_get_memory_pools(&ctx,rows.data(),count,&count),"inventory read");
    uint64_t sum=0;
    for (const auto & row:rows) if (row.category==LLAMA_RESOURCE_MEMORY_CUDA_POOL) {
        check(row.backend_id==0,"device remapped to scheduler backend");sum+=row.bytes;
    }
    return sum;
}
// Real cuBLAS FP16 projection allocates converted input/output workspaces.
// Constant powers of two give an exact independently known output.
static void projection(ggml_backend_t backend, int width, int rows, int tokens, bool rejected=false) {
    auto * meta=ggml_init({1<<20,nullptr,true});check(meta,"graph metadata");
    auto * weights=ggml_new_tensor_2d(meta,GGML_TYPE_F16,width,rows);
    auto * input=ggml_new_tensor_2d(meta,GGML_TYPE_F32,width,tokens);
    auto * output=ggml_mul_mat(meta,weights,input);
    auto * graph=ggml_new_graph_custom(meta,32,false);ggml_build_forward_expand(graph,output);
    auto * buffer=ggml_backend_alloc_ctx_tensors(meta,backend);check(buffer,"tensor allocation");
    std::vector<ggml_fp16_t> w(size_t(width)*rows,ggml_fp32_to_fp16(0.125f));
    std::vector<float> x(size_t(width)*tokens,0.25f),y(size_t(rows)*tokens);
    ggml_backend_tensor_set(weights,w.data(),0,w.size()*sizeof(w[0]));
    ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(x[0]));
    std::fill(y.begin(),y.end(),-17.0f);
    ggml_backend_tensor_set(output,y.data(),0,y.size()*sizeof(y[0]));
    const auto status=ggml_backend_graph_compute(backend,graph);
    ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(output,y.data(),0,y.size()*sizeof(y[0]));
    bool exact=true;for (float value:y) exact&=value==(rejected?-17.0f:float(width)/32);
    ggml_backend_buffer_free(buffer);ggml_free(meta);
    check(status==(rejected?GGML_STATUS_ALLOC_FAILED:GGML_STATUS_SUCCESS),"projection allocation status");
    check(exact,"independent exact projection output");
}
static void shared_graph_pool(ggml_backend_t backend,int device) {
    ggml_cuda_pool_memory_stats pool=snapshot(backend,device);
    auto ledger=ggml_memory_budget_create(UINT64_MAX);check(ledger,"shared ledger");
    auto alloc=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));check(alloc,"shared graph allocator");
    check(ggml_backend_cuda_set_pool_shared_memory_budget(backend,device,ledger)&&
          ggml_gallocr_set_memory_budget(alloc,0,ledger),"graph and pool share ledger");
    auto * small=ggml_init({1<<20,nullptr,true});auto * large=ggml_init({1<<20,nullptr,true});check(small&&large,"shared graph metadata");
    auto * input=ggml_new_tensor_1d(small,GGML_TYPE_F32,256);auto * output=ggml_scale(small,input,2);
    auto * graph=ggml_new_graph_custom(small,32,false);ggml_build_forward_expand(graph,output);
    auto * large_input=ggml_new_tensor_1d(large,GGML_TYPE_F32,262144);auto * large_output=ggml_scale(large,large_input,2);
    auto * grown=ggml_new_graph_custom(large,32,false);ggml_build_forward_expand(grown,large_output);
    check(ggml_gallocr_reserve(alloc,graph)&&ggml_gallocr_alloc_graph(alloc,graph),"small shared graph");
    uint64_t old=ggml_gallocr_get_buffer_size(alloc,0);ggml_memory_budget_stats s={};
    check(ggml_memory_budget_get(ledger,&s)&&s.reserved_bytes==pool.physical_bytes+old,"graph plus cached pool charged once");
    check(ggml_memory_budget_set(ledger,s.reserved_bytes),"no shared headroom");void * pointer=input->data;
    check(!ggml_gallocr_reserve(alloc,grown)&&ggml_memory_budget_get(ledger,&s)&&
          s.reserved_bytes==pool.physical_bytes+old&&input->data==pointer,"shared cap retains graph/pool backing");
    std::vector<float> x(256,0.25f),y(256);ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"old graph recovery");
    ggml_backend_tensor_get(output,y.data(),0,y.size()*sizeof(float));for(float v:y)check(v==0.5f,"old graph exact");
    projection(backend,1024,1024,4096,true);
    check(snapshot(backend,device).physical_bytes==pool.physical_bytes,"graph charge blocks pool growth");
    auto measure=ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));check(ggml_gallocr_reserve(measure,grown),"measure large graph");
    uint64_t needed=ggml_gallocr_get_buffer_size(measure,0);ggml_gallocr_free(measure);
    check(ggml_memory_budget_set(ledger,pool.physical_bytes+old+needed)&&ggml_gallocr_reserve(alloc,grown),"admit transient graph plus pool");
    check(ggml_memory_budget_get(ledger,&s)&&s.reserved_bytes==pool.physical_bytes+needed,"retired old graph releases charge");
    ggml_gallocr_free(alloc);check(ggml_memory_budget_get(ledger,&s)&&s.reserved_bytes==pool.physical_bytes,"cached pool remains charged after graph free");
    check(ggml_backend_cuda_set_pool_shared_memory_budget(backend,device,nullptr)&&ggml_memory_budget_get(ledger,&s)&&s.reserved_bytes==0,"detach transfers ownership without phantom charge");
    ggml_memory_budget_free(ledger);ggml_free(small);ggml_free(large);
}
static void workspace_plan_case(int device) {
    int owner=0;auto backend=ggml_backend_cuda_init(device,nullptr,&owner);check(backend,"workspace backend");
    auto * meta=ggml_init({1<<20,nullptr,true});check(meta,"workspace metadata");
    auto * w=ggml_new_tensor_2d(meta,GGML_TYPE_F16,512,512);
    auto * x=ggml_new_tensor_2d(meta,GGML_TYPE_F32,512,17);
    auto * y=ggml_mul_mat(meta,w,x);ggml_mul_mat_set_prec(y,GGML_PREC_F32);
    auto * graph=ggml_new_graph_custom(meta,32,false);ggml_build_forward_expand(graph,y);
    auto buffer=ggml_backend_alloc_ctx_tensors(meta,backend);check(buffer,"workspace tensors");
    uint64_t bytes=123;
    check(ggml_backend_cuda_explicit_matmul_workspace(backend,y,&bytes)&&bytes==512ull*512*4,"F16 explicit materialization predicted");
    check(!ggml_backend_cuda_explicit_matmul_workspace(nullptr,y,&bytes)&&bytes==512ull*512*4,"invalid query unchanged");
    check(ggml_backend_cuda_set_pool_memory_budget(backend,device,0)&&
          !ggml_backend_cuda_prepare_explicit_workspace(backend,bytes)&&snapshot(backend,device).physical_bytes==0,"prepare rejects before backing");
    check(ggml_backend_cuda_set_pool_memory_budget(backend,device,UINT64_MAX)&&
          ggml_backend_cuda_prepare_explicit_workspace(backend,bytes),"prepare workspace");
    auto before=snapshot(backend,device);check(before.physical_bytes>=bytes && before.active_bytes==0,"prepared cached backing");
    check(ggml_backend_cuda_prepare_explicit_workspace(backend,bytes)&&snapshot(backend,device).physical_bytes==before.physical_bytes,"repeat preparation reuses backing");
    std::vector<ggml_fp16_t> weights(512*512,ggml_fp32_to_fp16(0.125f));std::vector<float> input(512*17,0.25f),output(512*17);
    ggml_backend_tensor_set(w,weights.data(),0,weights.size()*sizeof(weights[0]));ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
    check(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"prepared projection");ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(y,output.data(),0,output.size()*sizeof(float));for(float v:output)check(v==16.f,"prepared exact projection");
    auto after=snapshot(backend,device);check(after.physical_bytes==before.physical_bytes && after.active_bytes==0,"execution uses prepared backing");
    auto * head_meta=ggml_init({1<<20,nullptr,true});check(head_meta,"head metadata");
    auto * head_w=ggml_new_tensor_2d(head_meta,GGML_TYPE_BF16,128,1024);ggml_set_name(head_w,"output");
    auto * head_x=ggml_new_tensor_2d(head_meta,GGML_TYPE_F32,128,2);
    auto * head_y=ggml_mul_mat(head_meta,head_w,head_x);ggml_mul_mat_set_prec(head_y,GGML_PREC_F32);
    auto * head_graph=ggml_new_graph_custom(head_meta,32,false);ggml_build_forward_expand(head_graph,head_y);
    auto head_buffer=ggml_backend_alloc_ctx_tensors(head_meta,backend);check(head_buffer,"head tensors");
    const bool direct=std::getenv("ESE_V41_DIRECT_BF16_HEAD")!=nullptr;
    bytes=999;check(ggml_backend_cuda_explicit_matmul_workspace(backend,head_y,&bytes)&&bytes==(direct?0:128ull*1024*4),"direct head does not reserve materialization");
    ggml_mul_mat_set_prec(head_y,GGML_PREC_DEFAULT);bytes=999;
    check(!ggml_backend_cuda_explicit_matmul_workspace(backend,head_y,&bytes)&&bytes==999,"unplanned precision leaves output unchanged");ggml_mul_mat_set_prec(head_y,GGML_PREC_F32);
    std::vector<ggml_bf16_t> hw(128*1024,ggml_fp32_to_bf16(0.125f));std::vector<float> hx(128*2,0.25f),hy(1024*2);
    ggml_backend_tensor_set(head_w,hw.data(),0,hw.size()*sizeof(hw[0]));ggml_backend_tensor_set(head_x,hx.data(),0,hx.size()*sizeof(float));
    check(ggml_backend_graph_compute(backend,head_graph)==GGML_STATUS_SUCCESS,"head execution");ggml_backend_synchronize(backend);
    ggml_backend_tensor_get(head_y,hy.data(),0,hy.size()*sizeof(float));for(float v:hy)check(v==4.f,"head output exact");
    ggml_backend_buffer_free(head_buffer);ggml_free(head_meta);
    ggml_backend_buffer_free(buffer);ggml_free(meta);ggml_backend_free(backend);
    std::printf("CUDA workspace device%d: prediction/admission/reuse/exact execution passed\n",device);
}
static void real_cases() {
    const int count=ggml_backend_cuda_get_device_count();check(count>0,"CUDA device required");
    for (int device=0;device<count;++device) {
        llama_model model{};model.arch=LLM_ARCH_LLAMA;llama_context ctx(model);
        ctx.backends={ggml_backend_cuda_init(device,nullptr,nullptr),ggml_backend_cpu_init()};
        check(ctx.backends[0] && ctx.backends[1],"CUDA/CPU backends");
        ctx.sched=ggml_backend_sched_new(ctx.backends.data(),nullptr,2,32,false);check(ctx.sched,"scheduler");
        auto empty=snapshot(ctx.backends[0],device);
        check(!empty.physical_bytes && !empty.virtual_reserved_bytes && inventory(ctx)==0,"query does not create pool");
        check(!ggml_backend_cuda_set_pool_memory_budget(ctx.backends[1],device,0) &&
              !ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],-1,0) &&
              !ggml_backend_cuda_set_pool_memory_budget(nullptr,device,0),"invalid budget rejected");
        check(ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],device,0),"cold zero cap");
        projection(ctx.backends[0],256,256,128,true);
        auto cold_rejected=snapshot(ctx.backends[0],device);
        check(cold_rejected.physical_bytes==0 && cold_rejected.active_bytes==0 &&
              cold_rejected.budget_rejections==1 && inventory(ctx)==0,"cold admission before allocation");
        check(ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],device,UINT64_MAX),"remove cold cap");
        for (int other=0;other<count;++other) if (other!=device)
            check(snapshot(ctx.backends[0],other).physical_bytes==0,"uncreated remote pool");
        ggml_cuda_pool_memory_stats untouched{1,2,3,4,5},sentinel=untouched;
        check(!ggml_backend_cuda_get_pool_memory_stats(ctx.backends[0],-1,&untouched) &&
              !ggml_backend_cuda_get_pool_memory_stats(ctx.backends[0],count,&untouched) &&
              !ggml_backend_cuda_get_pool_memory_stats(ctx.backends[1],device,&untouched) &&
              !ggml_backend_cuda_get_pool_memory_stats(nullptr,device,&untouched) &&
              !ggml_backend_cuda_get_pool_memory_stats(ctx.backends[0],device,nullptr) &&
              std::memcmp(&untouched,&sentinel,sizeof(sentinel))==0,"invalid snapshots preserve output");
        projection(ctx.backends[0],256,256,128);
        auto warm=snapshot(ctx.backends[0],device);
        check(warm.physical_bytes>0 && warm.active_bytes==0 && inventory(ctx)==warm.physical_bytes,"cached physical inventory");
        check(!ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],device,warm.physical_bytes-1) &&
              snapshot(ctx.backends[0],device).allocation_budget_bytes==UINT64_MAX,"cannot lower cap below backing");
        check(ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],device,warm.physical_bytes),"hold warm backing cap");
        projection(ctx.backends[0],1024,1024,2048,true);
        auto blocked=snapshot(ctx.backends[0],device);
        check(blocked.physical_bytes==warm.physical_bytes && blocked.active_bytes==0 &&
              blocked.budget_rejections==warm.budget_rejections+1 && inventory(ctx)==warm.physical_bytes,"growth rejected before backing mutation");
        projection(ctx.backends[0],256,256,128);
        check(ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],device,6ull<<20),"admit first temporary allocation");
        projection(ctx.backends[0],1024,1024,2048,true);
        auto partial=snapshot(ctx.backends[0],device);
        check(partial.physical_bytes>warm.physical_bytes && partial.physical_bytes<=(6ull<<20) &&
              partial.active_bytes==0 && partial.budget_rejections==blocked.budget_rejections+1 &&
              inventory(ctx)==partial.physical_bytes,"later rejection unwinds leases and respects cap");
        projection(ctx.backends[0],256,256,128);
        check(ggml_backend_cuda_set_pool_memory_budget(ctx.backends[0],device,UINT64_MAX),"remove growth cap");
        projection(ctx.backends[0],1024,1024,2048);
        auto grown=snapshot(ctx.backends[0],device);
        check(grown.physical_bytes>warm.physical_bytes && grown.active_bytes==0 &&
              inventory(ctx)==grown.physical_bytes,"workspace growth inventory");
        projection(ctx.backends[0],256,256,128);
        auto reused=snapshot(ctx.backends[0],device);
        check(reused.physical_bytes==grown.physical_bytes && reused.peak_physical_bytes==grown.peak_physical_bytes &&
              inventory(ctx)==reused.physical_bytes,"cached reuse does not double charge");
        if (reused.virtual_reserved_bytes)
            check(reused.virtual_reserved_bytes==(1ull<<35) && reused.physical_bytes<reused.virtual_reserved_bytes,"virtual reservation excluded");
        std::printf("CUDA pool device%d: cold/quota/partial/retry/growth/reuse exact; physical=%llu cached=%llu virtual=%llu bytes\n",
                    device,(unsigned long long)reused.physical_bytes,(unsigned long long)reused.cached_bytes,
                    (unsigned long long)reused.virtual_reserved_bytes);
        shared_graph_pool(ctx.backends[0],device);
        std::printf("CUDA shared graph/pool device%d: mutual admission/replacement/retry/cleanup passed\n",device);
        workspace_plan_case(device);
    }
}
#endif

int main(int argc,char ** argv) {
    try {
        if (argc==2 && std::strcmp(argv[1],"--cuda")==0) {
#ifdef GGML_USE_CUDA
            real_cases();return 0;
#else
            throw std::runtime_error("CUDA build required");
#endif
        }
        check(argc==1,"usage: test-cuda-pool-memory [--cuda]");
#ifdef GGML_USE_CUDA
        ggml_cuda_pool_memory_stats output{1,2,3,4,5},original=output;
        check(!ggml_backend_cuda_get_pool_memory_stats(nullptr,0,&output) &&
              std::memcmp(&output,&original,sizeof(output))==0,"null query leaves output unchanged");
#endif
        std::puts("CUDA pool query default: no GPU initialization");return 0;
    } catch (const std::exception & e) {std::fprintf(stderr,"CUDA pool memory: %s\n",e.what());return 1;}
}
