#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "../ggml/src/ggml-backend-impl.h"
#include "../ggml/src/ggml-sched-memory.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

struct storage {
    ggml_backend_buffer_type type{};
    bool fail_next=false;
    size_t calls=0,live=0,failures=0;
    std::vector<size_t> requests;
    storage() {
        type.context=this;
        type.iface.get_name=[](auto){return "graph-budget-test";};
        type.iface.get_alignment=[](auto){return size_t(256);};
        type.iface.is_host=[](auto){return true;};
        type.iface.alloc_buffer=[](auto type,size_t bytes)->ggml_backend_buffer_t {
            auto & owner=*static_cast<storage *>(type->context);++owner.calls;
            owner.requests.push_back(bytes);
            if(owner.fail_next){owner.fail_next=false;++owner.failures;return nullptr;}
            void * data=nullptr;if(posix_memalign(&data,256,std::max<size_t>(bytes,256)))return nullptr;
            struct allocation {storage * owner;void * data;size_t bytes;};
            auto * value=new allocation{&owner,data,bytes};owner.live+=bytes;
            ggml_backend_buffer_i iface{};
            iface.get_name=[](auto){return "graph-budget-test";};
            iface.get_base=[](auto b){return static_cast<allocation *>(b->context)->data;};
            iface.free_buffer=[](auto b){auto * p=static_cast<allocation *>(b->context);p->owner->live-=p->bytes;std::free(p->data);delete p;};
            iface.set_tensor=[](auto,auto * t,const void * p,size_t offset,size_t n){std::memcpy(static_cast<char *>(t->data)+offset,p,n);};
            iface.get_tensor=[](auto,auto * t,void * p,size_t offset,size_t n){std::memcpy(p,static_cast<char *>(t->data)+offset,n);};
            return ggml_backend_buffer_init(type,iface,value,bytes);
        };
    }
};
static void check(bool ok,const char * message){if(!ok)throw std::runtime_error(message);}
struct fixture {
    ggml_context * ctx=nullptr;ggml_cgraph * graph=nullptr;ggml_tensor * input=nullptr,*output=nullptr;
    fixture(int n) {
        ctx=ggml_init({1<<20,nullptr,true});check(ctx,"metadata");input=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,n);
        ggml_set_input(input);ggml_set_output(input); // Input bytes must survive replay.
        auto * middle=ggml_scale(ctx,input,2);output=ggml_scale(ctx,middle,3);ggml_set_output(output);
        graph=ggml_new_graph_custom(ctx,128,false);ggml_build_forward_expand(graph,output);
    }
    ~fixture(){ggml_free(ctx);}
    void initialize() {
        std::vector<float> data(ggml_nelements(input));for(size_t i=0;i<data.size();++i)data[i]=float(i)*0.25f;
        ggml_backend_tensor_set(input,data.data(),0,data.size()*sizeof(float));
    }
    bool exact(ggml_backend_t backend) {
        if(ggml_backend_graph_compute(backend,graph)!=GGML_STATUS_SUCCESS)return false;
        std::vector<float> data(ggml_nelements(output));ggml_backend_tensor_get(output,data.data(),0,data.size()*sizeof(float));
        for(size_t i=0;i<data.size();++i)if(data[i]!=float(i)*1.5f)return false;
        return true;
    }
};
static ggml_gallocr_buffer_stats stats(ggml_gallocr_t alloc,int id) {
    ggml_gallocr_buffer_stats value{};check(ggml_gallocr_get_buffer_stats(alloc,id,&value),"stats");return value;
}
static void independent() {
    storage first,second;ggml_backend_buffer_type_t types[]={&first.type,&second.type};
    auto * alloc=ggml_gallocr_new_n(types,2);auto * cpu=ggml_backend_cpu_init();check(cpu,"CPU");
    {
    fixture old(16),large(128);const int nodes[]={1,0},leaves[]={0};
    check(ggml_gallocr_reserve_n(alloc,old.graph,nodes,leaves) && ggml_gallocr_alloc_graph(alloc,old.graph),"initial reserve");
    old.initialize();check(old.exact(cpu),"initial exact");auto * old_buffer=old.input->buffer;auto * old_output=old.output->data;
    const auto a=stats(alloc,0),b=stats(alloc,1);const auto first_live=first.live,second_live=second.live;
    const auto before_calls0=first.calls,before_calls1=second.calls;
    uint64_t sizing[2]={0,0};
    check(ggml_gallocr_measure_n(alloc,large.graph,nodes,leaves,sizing,2),"native arena measurement");
    check(sizing[0]>a.allocated_bytes && sizing[1]>b.allocated_bytes,"native arena requirements missing");
    check(first.calls==before_calls0 && second.calls==before_calls1 && first.live==first_live && second.live==second_live,
          "measurement allocated buffer storage");
    check(stats(alloc,0).required_bytes==a.required_bytes && stats(alloc,1).required_bytes==b.required_bytes,
          "measurement changed committed diagnostics");
    check(ggml_gallocr_alloc_graph(alloc,old.graph) && old.exact(cpu),"measurement changed retained layout/output");
    check(!ggml_gallocr_measure_n(alloc,large.graph,nodes,leaves,sizing,1),"measurement accepted incomplete output array");
    second.fail_next=true;
    check(!ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves),"allocation failure ignored");
    check(second.failures==1 && first.live==first_live && second.live==second_live,"failed candidate/old arena ownership");
    check(old.input->buffer==old_buffer && old.output->data==old_output && old.exact(cpu),"failure changed old graph bytes/pointers");
    check(ggml_gallocr_alloc_graph(alloc,old.graph) && old.exact(cpu),"failed reservation replaced committed layout");
    check(!ggml_gallocr_alloc_graph(alloc,large.graph),"failed graph published oversized layout");
    const auto measured=stats(alloc,0);check(measured.required_bytes>a.allocated_bytes,"failed measurement missing");
    check(measured.required_bytes==sizing[0] && stats(alloc,1).required_bytes==sizing[1],"measurement differs from actual reserve layout");
    check(!ggml_gallocr_set_buffer_budget(alloc,0,a.allocated_bytes-1) &&
          ggml_gallocr_set_buffer_budget(alloc,0,a.allocated_bytes+measured.required_bytes-1),"quota live bound");
    const auto calls0=first.calls,calls1=second.calls;
    check(!ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves) && first.calls==calls0 && second.calls==calls1,"quota admission allocated a device buffer");
    check(old.exact(cpu),"quota changed old graph");
    check(ggml_gallocr_set_buffer_budget(alloc,0,a.allocated_bytes+measured.required_bytes),"growth quota");
    check(ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves) && ggml_gallocr_alloc_graph(alloc,large.graph),"growth retry");
    large.initialize();check(large.exact(cpu),"grown exact");
    check(stats(alloc,0).peak_allocated_bytes==a.allocated_bytes+stats(alloc,0).allocated_bytes &&
          stats(alloc,1).peak_allocated_bytes==b.allocated_bytes+stats(alloc,1).allocated_bytes,"old-plus-new peak");
    }
    ggml_gallocr_free(alloc);ggml_backend_free(cpu);check(first.live==0 && second.live==0,"teardown leaked arenas");
}
static void aliases() {
    storage shared;ggml_backend_buffer_type_t types[]={&shared.type,&shared.type};auto * alloc=ggml_gallocr_new_n(types,2);
    auto * cpu=ggml_backend_cpu_init();check(cpu,"CPU");
    {
    fixture old(16),large(128);const int nodes[]={1,0},leaves[]={0};
    check(ggml_gallocr_reserve_n(alloc,old.graph,nodes,leaves) && ggml_gallocr_alloc_graph(alloc,old.graph),"alias initial");
    old.initialize();check(old.exact(cpu),"alias initial exact");const auto before=stats(alloc,0);const auto live=shared.live;
    check(ggml_gallocr_get_buffer_size(alloc,1)==0 && stats(alloc,1).allocated_bytes==before.allocated_bytes,"shared type counted twice");
    shared.fail_next=true;
    check(!ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves) && shared.live==live && old.exact(cpu),"alias failure freed shared arena");
    const auto need=stats(alloc,0).required_bytes;
    check(ggml_gallocr_set_buffer_budget(alloc,1,before.allocated_bytes+need-1),"alias quota");auto calls=shared.calls;
    check(!ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves) && shared.calls==calls && old.exact(cpu),"secondary alias quota ignored");
    check(stats(alloc,0).allocation_budget_bytes==stats(alloc,1).allocation_budget_bytes,"alias effective budget mismatch");
    check(ggml_gallocr_set_buffer_budget(alloc,1,before.allocated_bytes+need) &&
          ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves) && ggml_gallocr_alloc_graph(alloc,large.graph),"alias growth retry");
    large.initialize();check(large.exact(cpu),"alias grown exact");
    }
    ggml_gallocr_free(alloc);ggml_backend_free(cpu);check(!shared.live,"shared arena double-free/leak");
}
#ifdef GGML_USE_CUDA
static void cuda_arenas() {
    const int count=ggml_backend_cuda_get_device_count();check(count>0,"no CUDA devices");
    for(int device=0;device<count;++device) {
        auto * backend=ggml_backend_cuda_init(device,nullptr,nullptr);check(backend,"CUDA initialization");
        for(bool shared:{false,true}) {
            auto type=ggml_backend_get_default_buffer_type(backend);
            ggml_backend_buffer_type_t types[]={type,type};
            auto * alloc=ggml_gallocr_new_n(types,shared?2:1);
            {
                fixture old(16),large(128);const int nodes[]={shared?1:0,0},leaves[]={0};
                check(ggml_gallocr_reserve_n(alloc,old.graph,nodes,leaves) &&
                      ggml_gallocr_alloc_graph(alloc,old.graph),"CUDA initial reservation");
                old.initialize();check(old.exact(backend),"CUDA initial exact");
                auto * old_buffer=old.input->buffer;auto * old_data=old.output->data;
                const auto before=stats(alloc,0);const int quota_id=shared?1:0;
                check(ggml_gallocr_set_buffer_budget(alloc,quota_id,before.allocated_bytes),"CUDA live quota");
                check(!ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves),"CUDA quota ignored");
                const auto rejected=stats(alloc,0);
                check(rejected.required_bytes>before.allocated_bytes && rejected.budget_rejections==1 &&
                      rejected.allocated_bytes==before.allocated_bytes &&
                      rejected.peak_allocated_bytes==before.peak_allocated_bytes,"CUDA rejection accounting");
                check(old.input->buffer==old_buffer && old.output->data==old_data &&
                      ggml_gallocr_alloc_graph(alloc,old.graph) && old.exact(backend),"CUDA rejection changed graph");
                check(ggml_gallocr_set_buffer_budget(alloc,quota_id,before.allocated_bytes+rejected.required_bytes) &&
                      ggml_gallocr_reserve_n(alloc,large.graph,nodes,leaves) &&
                      ggml_gallocr_alloc_graph(alloc,large.graph),"CUDA growth retry");
                large.initialize();check(large.exact(backend),"CUDA grown exact");
                const auto grown=stats(alloc,0);
                check(grown.peak_allocated_bytes==before.allocated_bytes+grown.allocated_bytes,"CUDA growth peak");
                if(shared)check(ggml_gallocr_get_buffer_size(alloc,1)==0 &&
                               stats(alloc,1).allocated_bytes==grown.allocated_bytes,"CUDA alias inventory");
                std::printf("CUDA device%d aliases%d: live%llu peak%llu rejection%llu exact\n",device,shared,
                            (unsigned long long)grown.allocated_bytes,(unsigned long long)grown.peak_allocated_bytes,
                            (unsigned long long)grown.budget_rejections);
            }
            ggml_gallocr_free(alloc);
        }
        ggml_backend_free(backend);
    }
}
#endif
static void test_buffer_allocation_sizing() {
    storage owner;
    owner.type.iface.get_max_size = [](auto) { return size_t(512); };
    auto * ctx = ggml_init({1<<20, nullptr, true});
    check(ctx, "allocation sizing metadata");
    auto * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 33);
    ggml_view_1d(ctx, a, 16, 0);
    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 65);
    size_t bytes = 77;
    check(ggml_backend_alloc_ctx_tensors_size(ctx, &owner.type, &bytes) && bytes == 768,
          "aligned allocation sizing includes owning tensors only");
    check(owner.requests.empty() && owner.live == 0 && a->data == nullptr,
          "allocation sizing allocated or initialized storage");
    owner.type.iface.get_max_size = [](auto) { return size_t(256); };
    bytes = 99;
    check(!ggml_backend_alloc_ctx_tensors_size(ctx, &owner.type, &bytes) && bytes == 99,
          "oversized tensor sizing published partial bytes");
    owner.type.iface.get_max_size = [](auto) { return SIZE_MAX; };
    owner.type.iface.get_alloc_size = [](auto, auto *) { return SIZE_MAX; };
    check(!ggml_backend_alloc_ctx_tensors_size(ctx, &owner.type, &bytes) && bytes == 99,
          "aligned size overflow was accepted");
    owner.type.iface.get_alloc_size = nullptr;
    owner.type.iface.get_max_size = [](auto) { return size_t(512); };
    auto * buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, &owner.type);
    check(buffer && ggml_backend_buffer_get_size(buffer) == 768 && owner.live == 768,
          "sizing differs from split buffer allocation");
    check(owner.requests == std::vector<size_t>({256,512}), "split allocation requests differ from sizing");
    check(ggml_backend_alloc_ctx_tensors_size(ctx, &owner.type, &bytes) && bytes == 0,
          "already allocated tensors counted twice");
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    check(owner.live == 0, "allocation sizing fixture retained storage");
}

int main(int argc,char ** argv){try{
    test_buffer_allocation_sizing();
    if(argc==2 && std::strcmp(argv[1],"--cuda")==0) {
#ifdef GGML_USE_CUDA
        cuda_arenas();
#else
        throw std::runtime_error("CUDA support not built");
#endif
    } else {check(argc==1,"usage: test-graph-arena-budget [--cuda]");independent();aliases();}
    std::puts("PASS: arena/layout failure atomicity, shared aliases, budget before allocation, exact old/new replay and teardown");return 0;
}catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
