#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

static void check(bool ok,const char * message) { if(!ok) throw std::runtime_error(message); }
using model_ptr=std::unique_ptr<llama_model,decltype(&llama_free_model)>;
using context_ptr=std::unique_ptr<llama_context,decltype(&llama_free)>;
struct fixture {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("ese-decode-status-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".gguf");
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

static int run(llama_context * ctx,int position,int count,std::vector<float> * output=nullptr) {
    std::vector<llama_token> tokens(count);
    for(int i=0;i<count;++i) tokens[i]=1+(position+i)%29;
    const int status=llama_decode(ctx,llama_batch_get_one(tokens.data(),count,position,0));
    llama_synchronize(ctx);
    if(status==0 && output) {
        const auto * values=llama_get_logits_ith(ctx,-1);check(values,"logits missing");
        output->assign(values,values+32);
        for(float value:*output) check(std::isfinite(value),"nonfinite logits");
    }
    return status;
}
static std::vector<uint8_t> save(llama_context * ctx) {
    std::vector<uint8_t> bytes(llama_state_get_size(ctx));
    const auto n=llama_state_get_data(ctx,bytes.data(),bytes.size());
    check(n>0 && n<=bytes.size(),"snapshot failed");bytes.resize(n);return bytes;
}
struct abort_control { int polls=0; int threshold=1; };
static bool abort_after(void * data) {
    auto & control=*static_cast<abort_control *>(data);
    return ++control.polls>=control.threshold;
}
int main() {
    llama_backend_init();
    try {
        fixture file;
        auto mp=llama_model_default_params();mp.devices="";mp.n_gpu_layers=0;
        mp.split_mode=LLAMA_SPLIT_MODE_NONE;mp.use_mmap=false;
        model_ptr model(llama_model_load_from_file(file.path.string().c_str(),mp),llama_free_model);
        check(bool(model),"model load failed");
        auto cp=llama_context_default_params();cp.n_ctx=32;cp.n_batch=8;cp.n_ubatch=8;
        cp.n_seq_max=1;cp.n_threads=1;cp.n_threads_batch=1;cp.worst_case_tokens=8;
        context_ptr ctx(llama_init_from_model(model.get(),cp),llama_free);check(bool(ctx),"context failed");
        check(run(ctx.get(),0,4)==0,"prefix decode failed");
        const auto before=save(ctx.get());
        std::vector<float> expected;check(run(ctx.get(),4,1,&expected)==0,"teacher continuation failed");
        for(int threshold:{1,12}) {
            check(llama_state_set_data(ctx.get(),before.data(),before.size())==before.size(),"base restore failed");
            abort_control control{0,threshold};llama_set_abort_callback(ctx.get(),abort_after,&control);
            const int status=run(ctx.get(),4,4);
            llama_set_abort_callback(ctx.get(),nullptr,nullptr);
            check(control.polls>=threshold,"abort callback not reached");
            check(status==LLAMA_DECODE_ERROR_ABORTED && status<0,"graph abort aliases KV capacity warning");
            check(llama_state_set_data(ctx.get(),before.data(),before.size())==before.size(),"abort recovery failed");
            check(save(ctx.get())==before,"restored session differs");
            std::vector<float> actual;check(run(ctx.get(),4,1,&actual)==0 && actual==expected,"abort continuation differs");
        }
        context_ptr full(llama_init_from_model(model.get(),cp),llama_free);check(bool(full),"capacity context failed");
        const int capacity=llama_n_ctx(full.get());
        for(int position=0;position<capacity;position+=8)
            check(run(full.get(),position,std::min(8,capacity-position))==0,"capacity fill failed");
        check(llama_get_kv_cache_used_cells(full.get())==capacity,"capacity fixture not full");
        check(run(full.get(),capacity,1)==1,"KV capacity warning changed");
        std::puts("Decode status: early/later graph aborts are negative; checkpoint recovery exact; KV exhaustion remains1");
    } catch(const std::exception & error) {
        std::fprintf(stderr,"DECODE STATUS ERROR: %s\n",error.what());llama_backend_free();return 1;
    }
    llama_backend_free();return 0;
}
