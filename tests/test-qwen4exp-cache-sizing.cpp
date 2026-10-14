#include "llama.h"
#include "llama-model.h"
#include "llama-model-loader.h"
#include "resource-planner.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

static void require(bool value, const std::string & message) {
    if (!value) throw std::runtime_error(message);
}

static ggml_type type(const std::string & name) {
    if (name == "f32") return GGML_TYPE_F32;
    if (name == "f16") return GGML_TYPE_F16;
    if (name == "q8_0") return GGML_TYPE_Q8_0;
    throw std::runtime_error("unsupported test type");
}

// Compare the predictor with the payload sizes of the tensors required by
// the cache layout. Metadata construction allocates no tensor data buffers.
static void check_case(const llama_model & model, ggml_type kv_type, ggml_type indexer_type,
        uint32_t context, uint32_t slots, bool flash) {
    auto params = llama_context_default_params();
    params.n_ctx = context; params.n_seq_max = slots;
    params.n_batch = 64; params.n_ubatch = 32; params.flash_attn = flash;
    uint64_t graph_metadata = 0;
    require(llama_model_graph_metadata_bytes(&model, params, &graph_metadata), "native graph metadata query");
    require(graph_metadata == uint64_t(65536)*ggml_tensor_overhead()+ggml_graph_overhead_custom(65536, false),
            "graph metadata does not match native node arena");
    uint64_t sentinel = 17;
    require(!llama_model_graph_metadata_bytes(nullptr, params, &sentinel) && sentinel == 17, "null graph model changed output");
    params.n_ctx = UINT32_MAX;
    require(!llama_model_graph_metadata_bytes(&model, params, &sentinel) && sentinel == 17, "invalid graph context changed output");
    auto * ctx = ggml_init({1<<20, nullptr, true});
    require(ctx != nullptr, "tensor metadata");
    try {
        uint64_t total = 0;
        uint64_t indexer_total = 0;
        const auto & hp = model.hparams;
        for (uint32_t il = 0; il < hp.n_layer; ++il) {
            uint64_t expected = 0;
            auto add = [&](ggml_tensor * tensor) { expected += ggml_nbytes(tensor); };
            if (hp.recurrent_layer_arr[il]) {
                const uint32_t state_slots = std::min(std::max(1U,slots),context);
                const uint64_t width = uint64_t(hp.n_embd_v_s())+hp.qwen4exp_ple_conv_state_size(il);
                add(ggml_new_tensor_2d(ctx,GGML_TYPE_F32,width,state_slots));
            } else {
                add(ggml_new_tensor_2d(ctx,kv_type,hp.n_embd_head_k(il),uint64_t(hp.n_head_kv(il))*context));
                add(ggml_new_tensor_1d(ctx,kv_type,uint64_t(hp.n_embd_head_v(il))*hp.n_head_kv(il)*context));
                if (model.arch == LLM_ARCH_QWEN4EXP) {
                    auto * indexer = ggml_new_tensor_2d(ctx,indexer_type,hp.indexer_head_size,context);
                    add(indexer);
                    indexer_total += ggml_nbytes(indexer);
                }
            }
            const uint64_t predicted = model.cache_size(il,kv_type,kv_type,indexer_type,context,0,slots,flash);
            if (predicted != expected) {
                throw std::runtime_error("layer="+std::to_string(il)+" context="+std::to_string(context)+
                        " slots="+std::to_string(slots)+" kv="+ggml_type_name(kv_type)+
                        " indexer="+ggml_type_name(indexer_type)+" flash="+std::to_string(flash)+
                        " predicted="+std::to_string(predicted)+" tensor_payload="+std::to_string(expected));
            }
            total += expected;
        }
        require(llama_model_kv_bytes(&model,kv_type,kv_type,indexer_type,context,0,slots,flash)==total,
                "public total differs from cache tensor payload");
        std::unique_ptr<llama_kv_size_profile,decltype(&llama_kv_size_profile_free)> profile(
                llama_model_kv_size_profile_create(&model),llama_kv_size_profile_free);
        require(bool(profile),"cache profile allocation");
        require(llama_kv_size_profile_bytes(profile.get(),kv_type,kv_type,indexer_type,context,0,slots,flash)==total,
                "retained profile differs from cache tensor payload");
        require(model.bufs.empty(), "weight buffers must remain absent");
        std::printf("cache_sizing: context=%u slots=%u kv=%s indexer=%s flash=%d total=%llu indexer_bytes=%llu weight_buffers=0\n",
                context,slots,ggml_type_name(kv_type),ggml_type_name(indexer_type),flash,
                (unsigned long long)total,(unsigned long long)indexer_total);
        ggml_free(ctx);
    } catch (...) {
        ggml_free(ctx);
        throw;
    }
}

static void synthetic(llama_model & model, bool qwen) {
    model.arch = qwen ? LLM_ARCH_QWEN4EXP : LLM_ARCH_LLAMA;
    auto & hp = model.hparams;
    hp.n_layer = 4;
    hp.n_embd = 128;
    hp.n_head_arr.fill(4);
    hp.n_head_kv_arr.fill(2);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
    hp.recurrent_layer_arr.fill(false);
    hp.indexer_head_size = 32;
    if (qwen) {
        hp.ssm_n_group = 2;
        hp.ssm_d_inner = 64;
        hp.ssm_d_state = 8;
        hp.ssm_dt_rank = 4;
        hp.ssm_d_conv = 4;
        hp.qwen4exp_hc_count = 2;
        hp.qwen4exp_ple_conv_kernel = 4;
        hp.qwen4exp_ple_ngram_size = 3;
        hp.qwen4exp_ple_layers[0] = hp.qwen4exp_ple_layers[2] = true;
        for (int il = 0; il < 3; ++il) hp.recurrent_layer_arr[il] = true;
        hp.qwen4exp_compress_ratios[3] = 4;
    }
}

static void profile_lifetime() {
    auto model=std::make_unique<llama_model>();
    synthetic(*model,true);
    auto query=common_native_cache_query(model.get(),GGML_TYPE_F16,0,3,false);
    require(bool(query),"native cache query allocation");
    const uint32_t contexts[]={32,256,4096,32768,65536};
    uint64_t expected[5];
    for (size_t i=0;i<5;++i) expected[i]=llama_model_kv_bytes(model.get(),GGML_TYPE_F16,GGML_TYPE_F16,GGML_TYPE_F16,contexts[i],0,3,false);
    // The retained query owns immutable geometry, including the architecture.
    model->arch=LLM_ARCH_LLAMA;
    model->hparams.indexer_head_size=64;
    model->hparams.ssm_d_inner=128;
    model.reset();
    for (size_t i=0;i<5;++i) require(query(GGML_TYPE_F16,contexts[i])==expected[i],"query depends on freed/mutated model");
    auto copied=query;
    query={};
    for (size_t i=0;i<5;++i) require(copied(GGML_TYPE_F16,contexts[i])==expected[i],"copied query lifetime");
    require(!llama_model_kv_size_profile_create(nullptr),"null profile source");
    llama_kv_size_profile_free(nullptr);
    require(llama_kv_size_profile_bytes(nullptr,GGML_TYPE_F16,GGML_TYPE_F16,GGML_TYPE_F16,32,0,1,false)==0,"null profile query");
    require(copied(GGML_TYPE_COUNT,32)==0 && copied(GGML_TYPE_F16,0)==0,"invalid profile query");
    std::puts("PASS: native query owns immutable metadata across model mutation/free and query copies");
}

static void native_planning(const llama_model & model) {
    auto query=common_native_cache_query(&model,GGML_TYPE_F16,0,1,false);
    require(bool(query),"planning metadata snapshot");
    common_resource_plan_input input;
    input.requested_policy=COMMON_MEMORY_POLICY_CACHE;
    input.max_ram_bytes=1ULL<<30;
    input.slots=1;
    input.requested_context=input.max_context=65536;
    input.min_context=512;
    input.context_alignment=256;
    input.min_kv_quality=COMMON_KV_QUALITY_F16;
    input.devices={{0,2ULL<<30,64ULL<<20,0,0}};
    input.kv_bytes_for_context=[query](common_kv_quality quality,uint32_t context) {
        return quality==COMMON_KV_QUALITY_F16 ? query(GGML_TYPE_F16,context) : 0;
    };
    common_resource_plan plan,target;
    std::string error;
    require(common_resource_plan_solve(input,plan,error),"native plan solve: "+error);
    const uint64_t long_bytes=llama_model_kv_bytes(&model,GGML_TYPE_F16,GGML_TYPE_F16,GGML_TYPE_F16,65536,0,1,false);
    require(plan.context==65536 && plan.devices.front().kv_bytes==long_bytes,"native long-context sizing");
    const uint64_t short_bytes=query(GGML_TYPE_F16,4096);
    common_resource_rebalance_request request;request.context=4096;
    require(common_resource_rebalance_target(plan,request,target,error),"native resize target: "+error);
    require(target.devices.front().kv_bytes==short_bytes,"native resize scales fixed state");
    const auto roundtrip=target;
    request.context=65536;
    require(common_resource_rebalance_target(roundtrip,request,target,error),"native regrowth target: "+error);
    require(target.devices.front().kv_bytes==long_bytes,"native resize roundtrip");
    if (model.hparams.n_layer==48) {
        input.requested_context=input.max_context=input.min_context=512;
        input.devices.front().free_bytes=128ULL<<20;
        input.devices.front().reserve_bytes=16ULL<<20;
        require(!common_resource_plan_solve(input,target,error),"native short-context fixed state omitted");
        common_resource_preparation_peak peak;
        request.context=4096;
        require(common_resource_rebalance_target(plan,request,target,error),"native shrink target");
        require(!common_resource_rebalance_preparation_peak(plan,target,peak,error),"replacement admits old plus full new cache without headroom");
    }
    std::printf("native_plan: context65536=%llu context4096=%llu; solve, shrink and regrowth exact\n",
            (unsigned long long)long_bytes,(unsigned long long)short_bytes);
}

int main(int argc, char ** argv) {
    try {
        profile_lifetime();
        if (argc == 8 && std::string(argv[1]) == "--case") {
            llama_model model{};
            require(std::string(argv[2])=="qwen" || std::string(argv[2])=="llama","test architecture");
            synthetic(model,std::string(argv[2])=="qwen");
            check_case(model,type(argv[3]),type(argv[4]),std::stoul(argv[5]),std::stoul(argv[6]),std::stoi(argv[7])!=0);
        } else if (argc == 3 && std::string(argv[1]) == "--artifact") {
            llama_model model{};
            llama_model_loader loader(argv[2],0,false,false,false,false,false,false,false,nullptr,nullptr);
            model.arch = LLM_ARCH_QWEN4EXP;
            llm_load_hparams(loader,model,true);
            size_t cases = 0;
            for (auto kv : {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
                for (auto indexer : {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
                    for (uint32_t context : {256U,4096U,32768U,65536U})
                        for (uint32_t slots : {1U,3U}) for (bool flash : {false,true}) {
                            check_case(model,kv,indexer,context,slots,flash);
                            ++cases;
                        }
            std::printf("PASS: %zu original-metadata cache-sizing cases; no weight buffers or context execution\n",cases);
            native_planning(model);
        } else {
            require(argc==1,"usage: test-qwen4exp-cache-sizing [--artifact SHARD | --case ARCH KV INDEXER CONTEXT SLOTS FLASH]");
            size_t cases = 0;
            for (bool qwen : {false,true}) {
                llama_model model{};synthetic(model,qwen);
                for (auto kv : {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
                    for (auto indexer : {GGML_TYPE_F32,GGML_TYPE_F16,GGML_TYPE_Q8_0})
                        for (uint32_t context : {32U,256U,4096U,32768U,65536U})
                            for (uint32_t slots : {1U,3U,64U}) for (bool flash : {false,true}) {
                                check_case(model,kv,indexer,context,slots,flash);
                                ++cases;
                            }
            }
            std::printf("PASS: %zu dense/hybrid cache-sizing cases; exact layer/total tensor payload sizes\n",cases);
            llama_model model{};synthetic(model,true);native_planning(model);
        }
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr,"FAIL: %s\n",error.what());
        return 1;
    }
}
