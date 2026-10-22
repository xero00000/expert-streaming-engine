#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-memory-budget.h"
#include <atomic>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

static void check(bool ok,const char * why){if(!ok)throw std::runtime_error(why);}
static ggml_memory_budget_stats stats(ggml_memory_budget_t b){ggml_memory_budget_stats s{};check(ggml_memory_budget_get(b,&s),"ledger query");return s;}
struct graph {
    ggml_context * meta;ggml_tensor * input,*output;ggml_cgraph * value;
    graph(int n){meta=ggml_init({1<<20,nullptr,true});check(meta,"metadata");input=ggml_new_tensor_1d(meta,GGML_TYPE_F32,n);ggml_set_input(input);ggml_set_output(input);auto * mid=ggml_scale(meta,input,2);output=ggml_scale(meta,mid,3);ggml_set_output(output);value=ggml_new_graph_custom(meta,32,false);ggml_build_forward_expand(value,output);}
    ~graph(){ggml_free(meta);}
    void initialize(){std::vector<float> x(ggml_nelements(input));for(size_t i=0;i<x.size();++i)x[i]=float(i)*0.25f;ggml_backend_tensor_set(input,x.data(),0,x.size()*sizeof(float));}
    bool exact(ggml_backend_t cpu){if(ggml_backend_graph_compute(cpu,value)!=GGML_STATUS_SUCCESS)return false;std::vector<float> y(ggml_nelements(output));ggml_backend_tensor_get(output,y.data(),0,y.size()*sizeof(float));for(size_t i=0;i<y.size();++i)if(y[i]!=float(i)*1.5f)return false;return true;}
};
static void graph_cases(){
    auto * cpu=ggml_backend_cpu_init();auto type=ggml_backend_cpu_buffer_type();
    auto a=ggml_gallocr_new(type),b=ggml_gallocr_new(type);auto ledger=ggml_memory_budget_create(1<<20);check(cpu&&a&&b&&ledger,"owners");
    check(ggml_gallocr_set_memory_budget(a,0,ledger)&&ggml_gallocr_set_memory_budget(b,0,ledger),"attach arenas");
    {
        graph first(16),second(16),grown(128);
        check(ggml_gallocr_reserve(a,first.value)&&ggml_gallocr_alloc_graph(a,first.value),"first graph");first.initialize();
        check(ggml_gallocr_reserve(b,second.value)&&ggml_gallocr_alloc_graph(b,second.value),"second graph");second.initialize();
        const uint64_t old=ggml_gallocr_get_buffer_size(a,0),other=ggml_gallocr_get_buffer_size(b,0);
        check(stats(ledger).reserved_bytes==old+other,"combined arenas");
        check(ggml_memory_budget_set(ledger,old+other),"no remaining headroom");void * before=first.input->data;
        check(!ggml_gallocr_reserve(a,grown.value)&&stats(ledger).reserved_bytes==old+other,"shared cap rejects replacement");
        check(first.input->data==before&&first.exact(cpu)&&second.exact(cpu),"both old graphs remain exact");
        auto rejected=ggml_memory_budget_create(old-1);check(!ggml_gallocr_set_memory_budget(a,0,rejected)&&stats(rejected).reserved_bytes==0,"failed reattachment rolls back");ggml_memory_budget_free(rejected);
        auto measure=ggml_gallocr_new(type);graph measured(128);check(ggml_gallocr_reserve(measure,measured.value),"measure replacement");uint64_t needed=ggml_gallocr_get_buffer_size(measure,0);ggml_gallocr_free(measure);
        check(ggml_memory_budget_set(ledger,old+other+needed)&&ggml_gallocr_reserve(a,grown.value)&&ggml_gallocr_alloc_graph(a,grown.value),"admit old plus new");grown.initialize();
        check(stats(ledger).reserved_bytes==needed+other&&stats(ledger).peak_reserved_bytes==old+other+needed&&grown.exact(cpu)&&second.exact(cpu),"retire old charge after replacement");
        check(ggml_gallocr_reserve(a,grown.value)&&stats(ledger).reserved_bytes==needed+other,"reused arena single charge");
    }
    ggml_memory_budget_retain(ledger);ggml_memory_budget_free(ledger); // Creator releases; consumers and observer keep it alive.
    ggml_gallocr_free(a);ggml_gallocr_free(b);check(stats(ledger).reserved_bytes==0,"all graph owners released");ggml_memory_budget_free(ledger);ggml_backend_free(cpu);
    ggml_backend_buffer_type_t types[]={type,type};auto alias=ggml_gallocr_new_n(types,2);auto shared=ggml_memory_budget_create(1<<20);
    {graph f(16);check(ggml_gallocr_reserve(alias,f.value),"aliased arena");uint64_t live=ggml_gallocr_get_buffer_size(alias,0);check(ggml_gallocr_set_memory_budget(alias,1,shared)&&ggml_gallocr_set_memory_budget(alias,0,shared)&&stats(shared).reserved_bytes==live,"aliases charge once");}
    ggml_gallocr_free(alias);check(stats(shared).reserved_bytes==0,"aliased cleanup");ggml_memory_budget_free(shared);
}
static void concurrency(){
    auto ledger=ggml_memory_budget_create(64);std::atomic<int> ready{0},done{0},accepted{0};std::vector<std::thread> workers;
    for(int i=0;i<8;++i)workers.emplace_back([&]{++ready;while(ready.load()<8)std::this_thread::yield();bool ok=ggml_memory_budget_reserve(ledger,64);if(ok)++accepted;++done;while(done.load()<8)std::this_thread::yield();if(ok)ggml_memory_budget_release(ledger,64);});
    for(auto & w:workers)w.join();auto s=stats(ledger);check(accepted==1&&s.reserved_bytes==0&&s.peak_reserved_bytes==64&&s.rejections==7,"concurrent owners cannot spend same headroom");
    check(!ggml_memory_budget_reserve(ledger,UINT64_MAX)&&stats(ledger).reserved_bytes==0,"overflow rejects without wrap");ggml_memory_budget_free(ledger);
}
int main(){try{graph_cases();concurrency();std::puts("Shared memory budget: graph replacement/alias/lifetime/concurrency passed");return 0;}catch(const std::exception & e){std::fprintf(stderr,"Shared budget: %s\n",e.what());return 1;}}
