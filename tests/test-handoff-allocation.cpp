#include "llama.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <random>
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../ggml/src/ggml-backend-impl.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <vector>

// A host-backed device adapter executes real CPU graphs but advertises a
// distinct, non-host buffer type. This exercises scheduler transfers and
// deterministic allocator failures without depending on an actual GPU OOM.
struct device_state {
    ggml_backend_t cpu = nullptr;
    ggml_backend_buffer_type type{};
    bool fail_next = false;
    size_t allocations = 0, failures = 0, live = 0;
};
struct buffer_state { device_state * owner; void * data; size_t size; };
static ggml_guid fake_guid = {0x73,0x16,0x25,0x47};
static void require(bool ok,const char * message) { if(!ok)throw std::runtime_error(message); }
static ggml_backend_buffer_t allocate(ggml_backend_buffer_type_t type,size_t size) {
    auto & owner=*static_cast<device_state *>(type->context);++owner.allocations;
    if(owner.fail_next){owner.fail_next=false;++owner.failures;return nullptr;}
    void * data=nullptr;if(posix_memalign(&data,256,std::max<size_t>(size,256)))return nullptr;
    auto * state=new buffer_state{&owner,data,size};owner.live+=size;
    ggml_backend_buffer_i iface{};
    iface.get_name=[](auto){return "handoff-test-buffer";};
    iface.get_base=[](auto buffer){return static_cast<buffer_state *>(buffer->context)->data;};
    iface.free_buffer=[](auto buffer){auto * value=static_cast<buffer_state *>(buffer->context);value->owner->live-=value->size;std::free(value->data);delete value;};
    iface.set_tensor=[](auto,auto * tensor,const void * data,size_t offset,size_t count){std::memcpy(static_cast<char *>(tensor->data)+offset,data,count);};
    iface.get_tensor=[](auto,auto * tensor,void * data,size_t offset,size_t count){std::memcpy(data,static_cast<const char *>(tensor->data)+offset,count);};
    iface.cpy_tensor=[](auto,const auto * src,auto * dst){std::memcpy(dst->data,src->data,ggml_nbytes(src));return true;};
    iface.clear=[](auto buffer,uint8_t value){auto * state=static_cast<buffer_state *>(buffer->context);std::memset(state->data,value,state->size);};
    iface.memset_tensor=[](auto,auto * tensor,uint8_t value,size_t offset,size_t count){std::memset(static_cast<char *>(tensor->data)+offset,value,count);};
    return ggml_backend_buffer_init(type,iface,state,size);
}
static ggml_backend_t device(device_state & state) {
    state.cpu=ggml_backend_cpu_init();require(state.cpu,"CPU delegate");
    state.type.context=&state;
    state.type.iface.get_name=[](auto){return "handoff-test-device";};
    state.type.iface.alloc_buffer=allocate;
    state.type.iface.get_alignment=[](auto){return size_t(256);};
    state.type.iface.is_host=[](auto){return false;};
    auto * backend=new ggml_backend{};backend->guid=&fake_guid;backend->context=&state;
    backend->iface.get_name=[](auto){return "handoff-test-device";};
    backend->iface.free=[](auto value){auto * state=static_cast<device_state *>(value->context);ggml_backend_free(state->cpu);state->cpu=nullptr;delete value;};
    backend->iface.get_default_buffer_type=[](auto value){return &static_cast<device_state *>(value->context)->type;};
    backend->iface.supports_buft=[](auto value,auto type){return type==&static_cast<device_state *>(value->context)->type;};
    backend->iface.supports_op=[](auto value,const auto * op){return ggml_backend_supports_op(static_cast<device_state *>(value->context)->cpu,op);};
    backend->iface.graph_compute=[](auto value,auto * graph){return ggml_backend_graph_compute(static_cast<device_state *>(value->context)->cpu,graph);};
    return backend;
}
struct graph {
    ggml_context * meta=nullptr;ggml_backend_buffer_t input_buffer=nullptr;
    ggml_tensor * input=nullptr,*output=nullptr;ggml_cgraph * value=nullptr;
    int first_index=0;float bias=0;
    graph(llama_context & ctx,int count,bool view=false) {
        ggml_backend_sched_reset(ctx.sched);
        meta=ggml_init({1<<20,nullptr,true});require(meta,"graph metadata");
        input=ggml_new_tensor_1d(meta,GGML_TYPE_F32,count);ggml_set_input(input);
        auto * selected=input;
        if(view){first_index=2;selected=ggml_view_1d(meta,input,count-2,2*sizeof(float));}
        auto * first=ggml_scale(meta,selected,2);output=ggml_scale(meta,first,3);ggml_set_output(output);
        value=ggml_new_graph_custom(meta,128,false);ggml_build_forward_expand(value,output);
        input_buffer=ggml_backend_alloc_buffer(ctx.backends[0],count*sizeof(float));require(input_buffer,"input buffer");
        ggml_backend_tensor_alloc(input_buffer,input,ggml_backend_buffer_get_base(input_buffer));
        std::vector<float> values(count);for(int i=0;i<count;++i)values[i]=float(i)*0.25f;
        ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
        ggml_backend_sched_set_tensor_backend(ctx.sched,input,ctx.backends[0]);
        if(view)ggml_backend_sched_set_tensor_backend(ctx.sched,selected,ctx.backends[0]);
        ggml_backend_sched_set_tensor_backend(ctx.sched,first,ctx.backends[1]);
        ggml_backend_sched_set_tensor_backend(ctx.sched,output,ctx.backends[0]);
    }
    ~graph(){ggml_backend_buffer_free(input_buffer);ggml_free(meta);}
    void change_input() {
        bias+=1;std::vector<float> values(ggml_nelements(input));
        for(size_t i=0;i<values.size();++i)values[i]=float(i)*0.25f+bias;
        ggml_backend_tensor_set(input,values.data(),0,values.size()*sizeof(float));
    }
    bool exact()const {
        std::vector<float> values(ggml_nelements(output));ggml_backend_tensor_get(output,values.data(),0,values.size()*sizeof(float));
        for(size_t i=0;i<values.size();++i)if(values[i]!=float(i+first_index)*1.5f+6*bias)return false;
        return true;
    }
};
static ggml_status compute(llama_context & ctx,ggml_cgraph * value) {
    // Explicit node assignments must be allocated before graph_compute's
    // automatic reset path; this is also the native llama execution contract.
    if(!ggml_backend_sched_alloc_graph(ctx.sched,value))return GGML_STATUS_ALLOC_FAILED;
    return ggml_backend_sched_graph_compute(ctx.sched,value);
}
static ggml_backend_sched_handoff_stats stats(llama_context & ctx,int id) {
    ggml_backend_sched_handoff_stats value{};require(ggml_backend_sched_get_handoff_stats(ctx.sched,id,&value),"handoff stats");return value;
}
static uint64_t inventory(llama_context & ctx,int backend) {
    uint32_t count=0;require(llama_resource_get_memory_pools(&ctx,nullptr,0,&count),"inventory count");
    std::vector<llama_resource_memory_pool> rows(count);require(llama_resource_get_memory_pools(&ctx,rows.data(),count,&count),"inventory rows");
    uint64_t bytes=0;for(const auto & row:rows)if(row.backend_id==backend && row.category==LLAMA_RESOURCE_MEMORY_HANDOFF)bytes+=row.bytes;
    return bytes;
}
#ifdef GGML_USE_CUDA
static int real_cases() {
    try {
        const int count=ggml_backend_cuda_get_device_count();require(count>=2,"two CUDA devices required");
        for(int a=0;a<count;++a)for(int b=a+1;b<count;++b) {
            llama_model model{};model.arch=LLM_ARCH_LLAMA;llama_context ctx(model);
            ctx.backends={ggml_backend_cuda_init(a,nullptr,nullptr),ggml_backend_cuda_init(b,nullptr,nullptr),ggml_backend_cpu_init()};
            for(auto * backend:ctx.backends)require(backend,"CUDA/CPU backend");
            ctx.sched=ggml_backend_sched_new(ctx.backends.data(),nullptr,3,128,false);require(ctx.sched,"CUDA scheduler");
            ggml_backend_sched_set_split_mode_graph(ctx.sched,true,false);ggml_backend_sched_set_max_extra_alloc(ctx.sched,1);
            {graph reserve(ctx,1024);require(ggml_backend_sched_reserve(ctx.sched,reserve.value),"CUDA graph reserve");}
            {graph warm(ctx,16);require(compute(ctx,warm.value)==GGML_STATUS_SUCCESS && warm.exact(),"CUDA warm exact");}
            for(int i=0;i<2;++i)require(stats(ctx,i).allocated_bytes==256 && inventory(ctx,i)==256,"CUDA warm inventory");
            for(int i=0;i<2;++i)require(ggml_backend_sched_set_handoff_budget(ctx.sched,i,512),"CUDA budget");
            {graph blocked(ctx,128);require(compute(ctx,blocked.value)==GGML_STATUS_ALLOC_FAILED,"CUDA quota rejection");}
            {graph small(ctx,16);require(compute(ctx,small.value)==GGML_STATUS_SUCCESS && small.exact(),"CUDA small recovery");}
            for(int i=0;i<2;++i)require(ggml_backend_sched_set_handoff_budget(ctx.sched,i,768),"CUDA growth budget");
            {graph grown(ctx,128);require(compute(ctx,grown.value)==GGML_STATUS_SUCCESS && grown.exact(),"CUDA growth exact");
             grown.change_input();require(ggml_backend_sched_graph_compute(ctx.sched,grown.value)==GGML_STATUS_SUCCESS && grown.exact(),"CUDA reused graph exact");}
            {graph viewed(ctx,130,true);require(compute(ctx,viewed.value)==GGML_STATUS_SUCCESS && viewed.exact(),"CUDA offset view exact");}
            for(int i=0;i<2;++i)require(stats(ctx,i).allocated_bytes==512 && stats(ctx,i).peak_allocated_bytes==768 && inventory(ctx,i)==512,"CUDA grown inventory");
            std::printf("CUDA handoff pair %d/%d: cold/warm/growth/quota/retry/reuse/view exact; live512 peak768 bytes\n",a,b);
        }
        return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"CUDA FAIL: %s\n",e.what());return 1;}
}
#endif

int main(int argc,char ** argv) {
#ifdef GGML_USE_CUDA
    if(argc==2 && std::strcmp(argv[1],"--cuda")==0)return real_cases();
#else
    (void)argc;(void)argv;
#endif
    std::array<device_state,2> states;
    try {
        {
        llama_model model{};model.arch=LLM_ARCH_LLAMA;llama_context ctx(model);
        ctx.backends={device(states[0]),device(states[1]),ggml_backend_cpu_init()};
        ctx.sched=ggml_backend_sched_new(ctx.backends.data(),nullptr,3,128,false);require(ctx.sched,"scheduler");
        ggml_backend_sched_set_split_mode_graph(ctx.sched,true,false);ggml_backend_sched_set_max_extra_alloc(ctx.sched,1);
        {graph large(ctx,1024);require(ggml_backend_sched_reserve(ctx.sched,large.value),"graph arena reservation");}
        // First allocation failure must return ALLOC_FAILED with no publication.
        {graph initial(ctx,16);states[1].fail_next=true;
         require(compute(ctx,initial.value)==GGML_STATUS_ALLOC_FAILED,"initial handoff failure status");
         require(states[1].failures==1 && stats(ctx,1).allocation_failures==1 && stats(ctx,0).allocated_bytes==0 && stats(ctx,1).allocated_bytes==0,"initial failure published candidate");}
        {graph warm(ctx,16);require(compute(ctx,warm.value)==GGML_STATUS_SUCCESS && warm.exact(),"warm exact retry");}
        std::array<ggml_backend_buffer_t,2> original;
        for(int i=0;i<2;++i){original[i]=ggml_backend_sched_get_handoff_buffer(ctx.sched,i);require(original[i] && stats(ctx,i).allocated_bytes==256 && inventory(ctx,i)==256,"warm handoff inventory");}
        // On growth, the first backend's successful candidate must be released
        // if the second backend fails. Existing arenas remain identical/live.
        {graph growing(ctx,128);states[1].fail_next=true;auto live=states[0].live;
         require(compute(ctx,growing.value)==GGML_STATUS_ALLOC_FAILED,"growth failure status");
         require(states[1].failures==2 && stats(ctx,1).allocation_failures==2 && states[0].live==live,"failed growth retained candidate allocation");
         for(int i=0;i<2;++i)require(ggml_backend_sched_get_handoff_buffer(ctx.sched,i)==original[i] && inventory(ctx,i)==256,"failed growth replaced old arena");}
        for(int i=0;i<2;++i)require(!ggml_backend_sched_set_handoff_budget(ctx.sched,i,255) && ggml_backend_sched_set_handoff_budget(ctx.sched,i,512),"budget lowering/live count");
        {graph bounded(ctx,128);auto calls=states[0].allocations;
         require(compute(ctx,bounded.value)==GGML_STATUS_ALLOC_FAILED,"live-plus-candidate quota ignored");
         require(states[0].allocations==calls && stats(ctx,0).budget_rejections==1,"budget rejection allocated buffer");}
        {graph small(ctx,16);require(compute(ctx,small.value)==GGML_STATUS_SUCCESS && small.exact(),"small retry after budget rejection");}
        for(int i=0;i<2;++i)require(ggml_backend_sched_set_handoff_budget(ctx.sched,i,768),"growth budget");
        {graph grown(ctx,128);require(compute(ctx,grown.value)==GGML_STATUS_SUCCESS && grown.exact(),"grown exact retry");
         grown.change_input();require(ggml_backend_sched_graph_compute(ctx.sched,grown.value)==GGML_STATUS_SUCCESS && grown.exact(),"reused graph exact");}
        {graph viewed(ctx,130,true);require(compute(ctx,viewed.value)==GGML_STATUS_SUCCESS && viewed.exact(),"offset view exact");}
        for(int i=0;i<2;++i)require(stats(ctx,i).allocated_bytes==512 && stats(ctx,i).peak_allocated_bytes==768 && inventory(ctx,i)==512,"grown live/peak/native inventory");
        require(!ggml_backend_sched_get_handoff_stats(ctx.sched,3,nullptr),"invalid stats query");
        }
        for(const auto & state:states)require(state.live==0 && !state.cpu,"teardown leaked device buffers");
        std::puts("PASS: handoff allocation failure, two-device atomic growth, live-plus-candidate budget, exact retry, inventory and teardown");return 0;
    }catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
