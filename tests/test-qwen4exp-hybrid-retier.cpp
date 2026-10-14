#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-quants.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool value,const char * message) { if(!value)throw std::runtime_error(message); }
static std::vector<uint8_t> bytes(ggml_tensor * tensor) {
    std::vector<uint8_t> out(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor,out.data(),0,out.size());return out;
}
static std::vector<llama_resource_memory_pool> inventory(llama_context & ctx) {
    uint32_t count=0;require(llama_resource_get_memory_pools(&ctx,nullptr,0,&count),"memory inventory query");
    std::vector<llama_resource_memory_pool> result(count);
    require(llama_resource_get_memory_pools(&ctx,result.data(),count,&count),"memory inventory read");return result;
}
static uint64_t pool_bytes(llama_context & ctx,llama_resource_memory_category category) {
    uint64_t total=0;for(const auto & item:inventory(ctx))if(item.category==category)total+=item.bytes;return total;
}

int main() {
    try {
        llama_model model{};model.arch=LLM_ARCH_LLAMA;
        auto & hp=model.hparams;hp.n_layer=2;hp.n_embd=32;hp.n_vocab=4;hp.n_ctx_train=64;
        hp.n_head_arr.fill(1);hp.n_head_kv_arr.fill(1);hp.n_embd_head_k_full=hp.n_embd_head_v_full=32;
        hp.recurrent_layer_arr.fill(false);hp.recurrent_layer_arr[0]=true;
        hp.ssm_d_state=128;hp.ssm_d_inner=2048;hp.ssm_dt_rank=8;hp.ssm_n_group=2;hp.ssm_d_conv=2;
        hp.indexer_head_size=32;
        model.layers.resize(2);model.buft_layer.assign(2,llama_model::layer_buft(ggml_backend_cpu_buffer_type()));
        llama_context ctx(model);model.arch=LLM_ARCH_QWEN4EXP;
        ctx.cparams={};ctx.cparams.n_ctx=32;ctx.n_ctx_max=64;ctx.cparams.n_seq_max=2;
        ctx.cparams.n_batch=ctx.cparams.n_ubatch=2;
        ctx.cparams.idx_type_k=GGML_TYPE_F32;
        auto * backend=ggml_backend_cpu_init();require(backend,"CPU backend");ctx.backends.push_back(backend);
        ctx.sched=ggml_backend_sched_new(ctx.backends.data(),nullptr,1,256,false);require(ctx.sched,"scheduler");
        auto & kv=ctx.kv_self;kv.size=kv.n=32;kv.used=4;kv.head=4;kv.hybrid=true;
        kv.qwen4exp_ordered=true;kv.v_trans=false;kv.type_k=kv.type_v=GGML_TYPE_F32;
        kv.cells.resize(32);for(int i=0;i<32;++i)kv.cells[i].src=i;
        for(int i=0;i<4;++i) {kv.cells[i].pos=i/2;kv.cells[i].qwen4exp_order=i/2;kv.cells[i].seq_id.insert(i%2);}
        auto * tensors=ggml_init({1024*1024,nullptr,true});require(tensors,"tensor metadata");kv.ctxs.push_back(tensors);
        kv.k_l={nullptr,ggml_new_tensor_2d(tensors,GGML_TYPE_F32,32,32)};
        kv.v_l={nullptr,ggml_new_tensor_1d(tensors,GGML_TYPE_F32,32*32)};
        kv.s_l={ggml_new_tensor_2d(tensors,GGML_TYPE_F32,hp.n_embd_v_s(),2),nullptr};
        kv.kr_l={nullptr,ggml_new_tensor_2d(tensors,GGML_TYPE_F32,32,32)};
        auto * weights=ggml_init({4096,nullptr,false});require(weights,"dummy model metadata");model.ctxs.push_back(weights);
        model.layers[1].qwen4exp_indexer_k_proj=ggml_new_tensor_1d(weights,GGML_TYPE_F32,1);
#ifdef GGML_USE_CUDA
        auto * buffer=ggml_backend_alloc_ctx_tensors_from_buft(tensors,ggml_backend_cuda_host_buffer_type());
#else
        auto * buffer=ggml_backend_alloc_ctx_tensors(tensors,backend);
#endif
        require(buffer,"tensor allocation");kv.bufs.push_back(buffer);
        ggml_backend_sched_set_tensor_backend(ctx.sched,kv.s_l[0],backend);
        ggml_backend_buffer_clear(buffer,0);
        for(auto * tensor:{kv.k_l[1],kv.v_l[1],kv.s_l[0],kv.kr_l[1]}) {
            std::vector<float> input(ggml_nelements(tensor));
            for(size_t i=0;i<input.size();++i)input[i]=i%11==0?-0.f:std::sin(float(i)*.031f);
            ggml_backend_tensor_set(tensor,input.data(),0,input.size()*sizeof(float));
        }
        require(kv.s_l[0]->nb[1]>1024*1024,"fixture must exceed the staging chunk");
        require(llama_spec_ckpt_init(&ctx,LLAMA_SPEC_CKPT_GPU_FALLBACK,2)==LLAMA_SPEC_CKPT_GPU_FALLBACK,"checkpoint mode");
        require(llama_spec_ckpt_save(&ctx,0) && llama_spec_ckpt_save(&ctx,1),"initial checkpoints");
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV)==kv.total_size(),"KV inventory bytes");
        uint64_t checkpoint_bytes=0;for(auto * value:kv.ckpt.shadow_bufs)checkpoint_bytes+=ggml_backend_buffer_get_size(value);
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_CHECKPOINT)==checkpoint_bytes,"checkpoint inventory bytes");
        auto pools=inventory(ctx);auto sentinel=pools;std::memset(sentinel.data(),0xa5,sentinel.size()*sizeof(sentinel[0]));
        const auto sentinel_before=sentinel;uint32_t required=0;
        require(!llama_resource_get_memory_pools(&ctx,sentinel.data(),pools.size()-1,&required) && required==pools.size() &&
                std::memcmp(sentinel.data(),sentinel_before.data(),sentinel.size()*sizeof(sentinel[0]))==0,"short inventory published partial rows");
        uint64_t original_total=0;for(const auto & item:pools){require(item.backend_id==-1,"host inventory location");original_total+=item.bytes;}
        const auto model_buffers_capacity=model.bufs.capacity();
        model.bufs.push_back(kv.bufs[0]);uint64_t alias_total=0;for(const auto & item:inventory(ctx))alias_total+=item.bytes;
        const uint64_t model_array_growth=(model.bufs.capacity()-model_buffers_capacity)*sizeof(ggml_backend_buffer_t);
        model.bufs.pop_back();
        require(alias_total==original_total+model_array_growth,"inventory counted a shared buffer twice or omitted its owning array");
        const std::array<std::vector<uint8_t>,4> original={bytes(kv.k_l[1]),bytes(kv.v_l[1]),bytes(kv.s_l[0]),bytes(kv.kr_l[1])};
        const std::array<ggml_type,2> half={GGML_TYPE_F16,GGML_TYPE_F16};
        for (auto invalid : {GGML_TYPE_I32, GGML_TYPE_Q4_K, GGML_TYPE_COUNT}) {
            auto params=llama_context_default_params();params.n_ctx=32;
            params.type_k=invalid;
            require(!llama_init_from_model(&model,params),"context accepted unsupported global key format");
            params.type_k=GGML_TYPE_F16;params.type_v=invalid;
            require(!llama_init_from_model(&model,params),"context accepted unsupported global value format");
            params.type_v=GGML_TYPE_F16;params.idx_type_k=invalid;
            require(!llama_init_from_model(&model,params),"context accepted unsupported global indexer format");
            const std::array<ggml_type,2> bad={invalid,invalid};
            llama_kv_preparation_memory rejected_memory={17,19,23};
            llama_kv_preparation_device rejected_device={29,31};
            require(!llama_kv_cache_retier_memory(&ctx,bad.data(),half.data(),2,&rejected_memory,&rejected_device,1),
                    "sizing accepted unsupported key type");
            require(!llama_kv_cache_retier_memory(&ctx,half.data(),bad.data(),2,&rejected_memory,&rejected_device,1),
                    "sizing accepted unsupported value type");
            require(rejected_memory.host_metadata_bytes==17 && rejected_memory.host_staging_bytes==19 &&
                    rejected_memory.device_count==23 && rejected_device.device_id==29 && rejected_device.kv_bytes==31,
                    "invalid sizing published partial outputs");
            require(!llama_kv_cache_prepare_retier(&ctx,bad.data(),half.data(),2),"unsupported key type accepted");
            require(!llama_kv_cache_prepare_retier(&ctx,half.data(),bad.data(),2),"unsupported value type accepted");
            require(!ctx.kv_cache_transaction && bytes(kv.k_l[1])==original[0] && bytes(kv.v_l[1])==original[1] &&
                    bytes(kv.s_l[0])==original[2] && bytes(kv.kr_l[1])==original[3] &&
                    kv.ckpt.sequences[0].saved && kv.ckpt.sequences[1].saved,"type preflight mutated live storage/owners");
        }
        for(const char * row:{"0","3","11"}) {
            setenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS",row,1);
            require(!llama_kv_cache_retier(&ctx,half.data(),half.data(),2),"injected failure was ignored");
            unsetenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS");
            require(bytes(kv.k_l[1])==original[0] && bytes(kv.v_l[1])==original[1] &&
                    bytes(kv.s_l[0])==original[2] && bytes(kv.kr_l[1])==original[3],"failure changed live bytes");
            require(kv.ckpt.sequences[0].saved && kv.ckpt.sequences[1].saved,"failure discarded checkpoints");
        }
        kv.save_per_step_ssm=true;
        require(!llama_kv_cache_prepare_retier(&ctx,half.data(),half.data(),2),"active journal accepted");kv.save_per_step_ssm=false;
        const auto host_metadata_before=pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_HOST_METADATA);
        const auto inventory_before_forecast=inventory(ctx);
        llama_kv_preparation_memory forecast={};
        require(llama_kv_cache_retier_memory(&ctx,half.data(),half.data(),2,&forecast,nullptr,0),"retier memory sizing query");
        require(forecast.device_count==1 && forecast.host_staging_bytes>1024*1024,"retier sizing omitted retained migration arrays");
        std::vector<llama_kv_preparation_device> projected(forecast.device_count);
        require(llama_kv_cache_retier_memory(&ctx,half.data(),half.data(),2,&forecast,projected.data(),projected.size()),"retier memory sizing rows");
        auto insufficient=projected;std::memset(insufficient.data(),0xa5,insufficient.size()*sizeof(insufficient[0]));
        const auto unmodified=insufficient;
        llama_kv_preparation_memory short_memory{17,18,19};
        require(!llama_kv_cache_retier_memory(&ctx,half.data(),half.data(),2,&short_memory,insufficient.data(),0) &&
                short_memory.host_metadata_bytes==17 && short_memory.host_staging_bytes==18 && short_memory.device_count==1 &&
                std::memcmp(insufficient.data(),unmodified.data(),sizeof(insufficient[0]))==0,"sizing short array mutated outputs");
        const auto inventory_after_forecast=inventory(ctx);
        require(inventory_before_forecast.size()==inventory_after_forecast.size(),"sizing changed inventory count");
        for(size_t i=0;i<inventory_before_forecast.size();++i)
            require(inventory_before_forecast[i].backend_id==inventory_after_forecast[i].backend_id &&
                    inventory_before_forecast[i].category==inventory_after_forecast[i].category &&
                    inventory_before_forecast[i].bytes==inventory_after_forecast[i].bytes,"sizing changed live capacity");
        auto * transaction=llama_kv_cache_prepare_retier(&ctx,half.data(),half.data(),2);require(transaction,"prepare");
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_HOST_METADATA)>host_metadata_before,
                "prepared host transaction/alternate vectors missing from inventory");
        const uint64_t original_kv_bytes=pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV);
        const uint64_t candidate_bytes=pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV_TRANSACTION);
        require(projected[0].device_id==-1 && projected[0].kv_bytes==candidate_bytes,"forecast KV allocation differs from actual");
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_HOST_METADATA)-host_metadata_before==forecast.host_metadata_bytes,
                "forecast metadata differs from actual tracked capacity");
        llama_kv_preparation_memory actual={};
        require(llama_kv_cache_transaction_memory(transaction,&actual,nullptr,0) &&
                actual.host_metadata_bytes==forecast.host_metadata_bytes && actual.host_staging_bytes==forecast.host_staging_bytes,
                "actual transaction memory differs from forecast");
        std::printf("retier memory: buffer=%llu metadata=%llu staging=%llu exact\n",
            (unsigned long long)candidate_bytes,(unsigned long long)actual.host_metadata_bytes,
            (unsigned long long)actual.host_staging_bytes);
        require(candidate_bytes>0,"prepared candidate missing from inventory");
        require(llama_kv_cache_transaction_publish(transaction),"publish");
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV)==candidate_bytes &&
                pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV_TRANSACTION)==original_kv_bytes,"published/rollback storage inventory");
        require(bytes(kv.s_l[0])==original[2],"publication changed recurrent slots");
        require(llama_kv_cache_transaction_rollback(transaction),"rollback");llama_kv_cache_transaction_free(transaction);
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV_TRANSACTION)==0 &&
                pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_KV)==original_kv_bytes,"rollback inventory retained candidate");
        require(pool_bytes(ctx,LLAMA_RESOURCE_MEMORY_HOST_METADATA)==host_metadata_before,
                "freed transaction retained host metadata allocations");
        require(bytes(kv.k_l[1])==original[0] && bytes(kv.v_l[1])==original[1] &&
                bytes(kv.s_l[0])==original[2] && bytes(kv.kr_l[1])==original[3],"rollback changed bytes");
        require(llama_kv_cache_retier(&ctx,half.data(),half.data(),2),"commit");
        require(kv.ckpt.allocated && kv.ckpt.selected_spec_mode==LLAMA_SPEC_CKPT_GPU_FALLBACK &&
                !kv.ckpt.sequences[0].saved && !kv.ckpt.sequences[1].saved,"commit checkpoint contract");
        require(bytes(kv.s_l[0])==original[2],"commit changed recurrent slots");
        const auto indexer=bytes(kv.kr_l[1]);require(std::memcmp(indexer.data(),original[3].data(),4*32*sizeof(float))==0,"commit changed occupied indexer rows");
        for(size_t family=0;family<2;++family) {
            const auto actual=bytes(family?kv.v_l[1]:kv.k_l[1]);
            std::array<ggml_fp16_t,4*32> expected;
            for(size_t i=0;i<expected.size();++i) {float value;std::memcpy(&value,original[family].data()+i*sizeof(float),sizeof(float));expected[i]=ggml_fp32_to_fp16(value);}
            require(std::memcmp(actual.data(),expected.data(),sizeof(expected))==0,"half migration differs from scalar codec");
        }
        require(llama_spec_ckpt_save(&ctx,0) && llama_spec_ckpt_save(&ctx,1),"banks unusable after commit");
        require(llama_spec_ckpt_restore_ex(&ctx,0,2,-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED &&
                llama_spec_ckpt_restore_ex(&ctx,1,2,-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,"new checkpoints unusable");
        const std::array<ggml_type,2> q8={GGML_TYPE_Q8_0,GGML_TYPE_Q8_0},full={GGML_TYPE_F32,GGML_TYPE_F32};
        require(llama_kv_cache_retier(&ctx,q8.data(),half.data(),2) && kv.k_l[1]->type==GGML_TYPE_Q8_0,
                "half to packed key migration");
        require(bytes(kv.s_l[0])==original[2],"packed migration changed recurrent slots");
        require(llama_kv_cache_retier(&ctx,full.data(),full.data(),2) && kv.k_l[1]->type==GGML_TYPE_F32 &&
                kv.v_l[1]->type==GGML_TYPE_F32,"packed/half to full migration");
        require(bytes(kv.s_l[0])==original[2],"full migration changed recurrent slots");
        for (auto type : {GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_BF16}) {
            auto before=bytes(kv.k_l[1]);std::array<float,4*32> input;
            std::memcpy(input.data(),before.data(),sizeof(input));
            std::vector<uint8_t> encoded(ggml_row_size(type,32)*4);
            std::array<float,4*32> decoded;
            switch(type) {
                case GGML_TYPE_Q4_0:
                    // This fork intentionally selects IQK's optimized Q4_0
                    // runtime encoder, which differs from the scalar reference.
                    quantize_row_q4_0(input.data(),encoded.data(),input.size());
                    dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(encoded.data()),decoded.data(),input.size());break;
                case GGML_TYPE_Q4_1:
                    quantize_row_q4_1_ref(input.data(),reinterpret_cast<block_q4_1 *>(encoded.data()),input.size());
                    dequantize_row_q4_1(reinterpret_cast<const block_q4_1 *>(encoded.data()),decoded.data(),input.size());break;
                case GGML_TYPE_Q5_0:
                    quantize_row_q5_0_ref(input.data(),reinterpret_cast<block_q5_0 *>(encoded.data()),input.size());
                    dequantize_row_q5_0(reinterpret_cast<const block_q5_0 *>(encoded.data()),decoded.data(),input.size());break;
                case GGML_TYPE_Q5_1:
                    quantize_row_q5_1_ref(input.data(),reinterpret_cast<block_q5_1 *>(encoded.data()),input.size());
                    dequantize_row_q5_1(reinterpret_cast<const block_q5_1 *>(encoded.data()),decoded.data(),input.size());break;
                case GGML_TYPE_BF16:
                    for(size_t i=0;i<input.size();++i) {
                        uint32_t bits;std::memcpy(&bits,&input[i],4);
                        bits+=0x7fff+((bits>>16)&1);uint16_t stored=bits>>16;
                        std::memcpy(encoded.data()+2*i,&stored,2);bits=uint32_t(stored)<<16;
                        std::memcpy(&decoded[i],&bits,4);
                    }
                    break;
                default:throw std::runtime_error("unexpected migration test format");
            }
            const std::array<ggml_type,2> packed={type,type};
            require(llama_kv_cache_retier(&ctx,packed.data(),full.data(),2),"additional format migration");
            auto actual=bytes(kv.k_l[1]);
            require(std::memcmp(actual.data(),encoded.data(),encoded.size())==0,"migration differs from direct typed codec/IEEE BF16");
            require(llama_kv_cache_retier(&ctx,full.data(),full.data(),2),"additional format reconstruction");
            actual=bytes(kv.k_l[1]);
            require(std::memcmp(actual.data(),decoded.data(),sizeof(decoded))==0,"migration reconstructed different F32 rows");
            require(bytes(kv.s_l[0])==original[2],"additional format migration changed recurrent slots");
        }
        std::puts("PASS: hybrid retier chunked state/indexer migration, three failure phases, rollback, half codec, retained checkpoint banks");
        return 0;
    }catch(const std::exception & error) {std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
