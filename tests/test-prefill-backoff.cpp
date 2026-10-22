#include "llama.h"
#include <random>
#include "../src/llama-context.h"
#include "../ggml/src/ggml-sched-memory.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

static void check(bool result,const char * message) { if(!result)throw std::runtime_error(message); }
using model_ptr=std::unique_ptr<llama_model,decltype(&llama_free_model)>;
using context_ptr=std::unique_ptr<llama_context,decltype(&llama_free)>;
struct fixture {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("ese-prefill-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".gguf");
    fixture() {
        auto * tensors=ggml_init({64*ggml_tensor_overhead(),nullptr,true});check(tensors,"fixture tensors");
        auto * meta=gguf_init_empty();check(meta,"fixture metadata");
        gguf_set_val_str(meta,"general.architecture","llama");gguf_set_val_str(meta,"tokenizer.ggml.model","none");
        for(auto item:std::vector<std::pair<const char *,uint32_t>>{
            {"llama.block_count",1},{"llama.vocab_size",32},{"llama.context_length",64},
            {"llama.embedding_length",32},{"llama.feed_forward_length",64},
            {"llama.attention.head_count",4},{"llama.attention.head_count_kv",4},
            {"llama.rope.dimension_count",8}})gguf_set_val_u32(meta,item.first,item.second);
        gguf_set_val_f32(meta,"llama.attention.layer_norm_rms_epsilon",1e-6f);
        std::vector<std::vector<float>> payloads;payloads.reserve(16);
        auto tensor=[&](const char * name,int64_t width,int64_t rows=1) {
            auto * value=ggml_new_tensor_2d(tensors,GGML_TYPE_F32,width,rows);ggml_set_name(value,name);
            payloads.emplace_back(width*rows);auto & data=payloads.back();
            for(size_t i=0;i<data.size();++i)data[i]=std::strstr(name,"norm")?1.f:
                float(int((i*17+payloads.size()*11)%101)-50)*0.001f;
            value->data=data.data();gguf_add_tensor(meta,value);
        };
        tensor("token_embd.weight",32,32);tensor("output.weight",32,32);tensor("output_norm.weight",32);
        tensor("blk.0.attn_norm.weight",32);tensor("blk.0.attn_q.weight",32,32);
        tensor("blk.0.attn_k.weight",32,32);tensor("blk.0.attn_v.weight",32,32);tensor("blk.0.attn_output.weight",32,32);
        tensor("blk.0.ffn_norm.weight",32);tensor("blk.0.ffn_gate.weight",32,64);
        tensor("blk.0.ffn_up.weight",32,64);tensor("blk.0.ffn_down.weight",64,32);
        gguf_write_to_file(meta,path.string().c_str(),false);gguf_free(meta);ggml_free(tensors);
    }
    ~fixture(){std::error_code ignored;std::filesystem::remove(path,ignored);}
};
static ggml_gallocr_buffer_stats stats(llama_context * ctx,int id) {
    ggml_gallocr_buffer_stats value{};
    check(ggml_backend_sched_get_graph_buffer_stats(ctx->sched,id,&value),"graph stats");return value;
}
static std::vector<uint8_t> snapshot(llama_context * ctx) {
    std::vector<uint8_t> result(llama_state_get_size(ctx));
    const auto n=llama_state_get_data(ctx,result.data(),result.size());check(n>0 && n<=result.size(),"snapshot");
    result.resize(n);return result;
}
static std::vector<float> decode(llama_context * ctx,int count=16) {
    std::vector<llama_token> tokens(count);for(int i=0;i<count;++i)tokens[i]=1+i%29;
    auto batch=llama_batch_get_one(tokens.data(),count,0,0);check(llama_decode(ctx,batch)==0,"chunked decode");
    std::vector<float> result;
    for(int i=0;i<count;++i) {
        const auto * values=llama_get_logits_ith(ctx,i);check(values,"token logits");
        result.insert(result.end(),values,values+32);
    }
    bool nonzero=false;for(float v:result){check(std::isfinite(v),"nonfinite fixture logit");nonzero|=v!=0;}
    check(nonzero,"vacuous zero-weight oracle");return result;
}
static void cases(const fixture & file,const char * devices,bool gpu) {
    auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=gpu?99:0;mp.use_mmap=false;mp.split_mode=LLAMA_SPLIT_MODE_NONE;
    model_ptr model(llama_model_load_from_file(file.path.string().c_str(),mp),llama_free_model);check(bool(model),"model load");
    auto cp=llama_context_default_params();cp.n_ctx=64;cp.n_batch=16;cp.n_seq_max=1;
    cp.n_threads=1;cp.n_threads_batch=1;cp.logits_all=true;cp.worst_case_tokens=16;
    cp.n_ubatch=1;context_ptr smallest(llama_init_from_model(model.get(),cp),llama_free);check(bool(smallest),"one-token context");
    auto largest_arena=[](llama_context * ctx) {
        uint64_t largest=0;
        for(int i=0;i<ggml_backend_sched_get_n_backends(ctx->sched);++i)
            largest=std::max(largest,stats(ctx,i).allocated_bytes);
        return largest;
    };
    const auto minimum=largest_arena(smallest.get());smallest.reset();
    cp.n_ubatch=16;context_ptr full(llama_init_from_model(model.get(),cp),llama_free);check(bool(full),"full context");
    const auto maximum=largest_arena(full.get());check(maximum>minimum,"fixture cannot exercise budget backoff");
    const auto reference=decode(full.get());const auto preserved=snapshot(full.get());
    cp.compute_buffer_budget_bytes=minimum+(maximum-minimum)/3;
    const auto admitted_cap=cp.compute_buffer_budget_bytes;
    context_ptr bounded(llama_init_from_model(model.get(),cp),llama_free);check(bool(bounded),"bounded context");
    check(llama_n_ubatch(bounded.get())>=1 && llama_n_ubatch(bounded.get())<16,"effective microbatch not reduced");
    check(bounded->cparams.worst_graph_tokens<=int(llama_n_ubatch(bounded.get())),"stale output reservation bound");
    uint64_t rejected_count=0,peak=0;
    for(int id=0;id<ggml_backend_sched_get_n_backends(bounded->sched);++id) {
        const auto admitted=stats(bounded.get(),id);
        rejected_count+=admitted.budget_rejections;peak=std::max(peak,admitted.peak_allocated_bytes);
        check(admitted.allocated_bytes<=cp.compute_buffer_budget_bytes &&
              admitted.peak_allocated_bytes<=cp.compute_buffer_budget_bytes,"graph cap not enforced");
    }
    check(rejected_count>0,"no budget rejection exercised");
    const auto actual=decode(bounded.get());check(actual.size()==reference.size(),"logit size");
    float worst=0;for(size_t i=0;i<actual.size();++i) {
        worst=std::max(worst,std::abs(actual[i]-reference[i]));
        check(std::abs(actual[i]-reference[i])<=2e-5f*(1+std::abs(reference[i])),"chunked output disagrees");
    }
    cp.attention_type=LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    context_ptr noncausal(llama_init_from_model(model.get(),cp),llama_free);
    check(!noncausal,"non-causal input silently chunked");
    cp.attention_type=LLAMA_ATTENTION_TYPE_CAUSAL;cp.compute_buffer_budget_bytes=1;
    context_ptr rejected(llama_init_from_model(model.get(),cp),llama_free);check(!rejected,"impossible graph budget admitted");
    check(snapshot(full.get())==preserved,"failed new context changed existing session");
    if(gpu) {
        cp.compute_buffer_budget_bytes=0;cp.expert_vram_reserve_bytes=UINT64_MAX;
        context_ptr reserve_rejected(llama_init_from_model(model.get(),cp),llama_free);
        check(!reserve_rejected,"automatic CUDA budget ignored reserve");
        check(snapshot(full.get())==preserved,"automatic reserve rejection changed existing session");
        cp.expert_vram_reserve_bytes=0;
    }
    check(!ggml_backend_sched_get_graph_buffer_stats(nullptr,0,nullptr) &&
          !ggml_backend_sched_set_graph_buffer_budget(full->sched,-1,0),"invalid scheduler budget request accepted");

    // Exercise the original requested microbatch size as an actual context
    // reservation, while keeping the synthetic model small and deterministic.
    cp.n_ctx=1024;cp.n_batch=1024;cp.n_ubatch=1;cp.worst_case_tokens=1024;cp.compute_buffer_budget_bytes=0;
    context_ptr large_min(llama_init_from_model(model.get(),cp),llama_free);check(bool(large_min),"large-context minimum");
    const auto large_minimum=largest_arena(large_min.get());large_min.reset();
    cp.n_ubatch=1024;
    context_ptr large_full(llama_init_from_model(model.get(),cp),llama_free);check(bool(large_full),"1024-token reservation");
    const auto large_maximum=largest_arena(large_full.get());check(large_maximum>large_minimum,"large reservation fixture");
    cp.compute_buffer_budget_bytes=large_minimum+(large_maximum-large_minimum)/16;
    context_ptr large_bounded(llama_init_from_model(model.get(),cp),llama_free);check(bool(large_bounded),"1024 budget recovery");
    check(llama_n_ubatch(large_bounded.get())<1024,"1024 microbatch not reduced");
    const int prompt_tokens=2*int(llama_n_ubatch(large_bounded.get()));
    const auto large_reference=decode(large_full.get(),prompt_tokens);
    const auto large_actual=decode(large_bounded.get(),prompt_tokens);
    for(size_t i=0;i<large_actual.size();++i)
        check(std::abs(large_actual[i]-large_reference[i])<=2e-5f*(1+std::abs(large_reference[i])),"1024 recovery output disagrees");
    uint64_t large_peak=0;
    for(int id=0;id<ggml_backend_sched_get_n_backends(large_bounded->sched);++id) {
        const auto value=stats(large_bounded.get(),id);large_peak=std::max(large_peak,value.peak_allocated_bytes);
        check(value.peak_allocated_bytes<=cp.compute_buffer_budget_bytes,"1024 graph peak exceeds cap");
    }
    check(snapshot(full.get())==preserved,"large reservation changed existing session");
    std::printf("prefill %s: requested1024 effective%u cap%llu peak%llu prompt%d; nonzero chunked prompt output parity passed\n",
        devices,llama_n_ubatch(large_bounded.get()),(unsigned long long)cp.compute_buffer_budget_bytes,
        (unsigned long long)large_peak,prompt_tokens);
    std::printf("prefill %s: requested16 effective%u min%llu full%llu cap%llu peak%llu rejected%llu logit_error%.9g; noncausal/impossible rejected; old session exact\n",
        devices,llama_n_ubatch(bounded.get()),(unsigned long long)minimum,(unsigned long long)maximum,
        (unsigned long long)admitted_cap,(unsigned long long)peak,
        (unsigned long long)rejected_count,worst);
}
int main(int argc,char ** argv) {
    try {
        fixture file;
        if(argc==2 && std::strcmp(argv[1],"--cuda")==0) {
#ifdef GGML_USE_CUDA
            const int count=ggml_backend_cuda_get_device_count();check(count>0,"CUDA device required");
            for(int i=0;i<count;++i){const std::string device=std::string(GGML_CUDA_NAME)+std::to_string(i);cases(file,device.c_str(),true);}
#else
            throw std::runtime_error("CUDA support not built");
#endif
        } else {check(argc==1,"usage: test-prefill-backoff [--cuda]");cases(file,"none",false);}
        std::puts("PASS: budget-backed causal prefill recovery, output parity, persistent chunk bound, noncausal failure and session preservation");return 0;
    } catch(const std::exception & e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
}
