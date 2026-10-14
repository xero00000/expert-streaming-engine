// Artifact-dependent Qwen4Exp lifecycle diagnostic. Does not open the loader gate.
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../../src/llama-context.h"
#include "../../ggml/src/ggml-sched-memory.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include "../../ggml/src/ggml-cuda-pool-memory.h"
#endif

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
struct float_capture {
    std::array<int64_t,4> shape;
    std::vector<float> values;
};
struct attention_operand {
    int type = -1;
    std::array<int64_t,4> shape = {};
    std::vector<uint8_t> bytes;
};
struct attention_capture {
    int precision = -1;
    std::array<attention_operand,4> operands;
};
struct route_capture {
    std::map<std::string, std::vector<int32_t>> tensors;
    std::vector<uint8_t> staging;
    std::map<std::string,float_capture> floats;
    std::vector<std::string> float_order;
    std::map<std::string,attention_capture> attention;
    bool invalid_type = false;
    bool invalid_expert_precision = false;
    size_t expert_gate_ops = 0, expert_down_ops = 0;
    size_t expert_gate_gpu_ops = 0, expert_down_gpu_ops = 0;
    bool invalid_shared_precision = false;
    std::array<size_t,3> shared_ops = {}, shared_gpu_ops = {};
    bool invalid_head_precision = false;
    size_t head_ops = 0, head_gpu_ops = 0;
    size_t decode_calls = 0;
    std::atomic<bool> abort_armed{false};
    int arm_after_routes = -1;
    int observed_routes = 0;
    int abort_polls = 0;
};

static attention_operand capture_attention_operand(const ggml_tensor * tensor) {
    attention_operand result;
    if(!tensor) return result;
    result.type=tensor->type;std::copy(tensor->ne,tensor->ne+4,result.shape.begin());
    if((tensor->type!=GGML_TYPE_F32 && tensor->type!=GGML_TYPE_F16 && tensor->type!=GGML_TYPE_I32) || !tensor->buffer) return result;
    const size_t element=ggml_type_size(tensor->type);
    const size_t bytes=ggml_nelements(tensor)*element;
    if(bytes>1024*1024 || tensor->nb[0]!=element) return result;
    result.bytes.resize(bytes);size_t destination=0;
    // KV views can span gaps between heads in the full context allocation.
    // Read only actual rows instead of allocating their entire physical span.
    for(int64_t i3=0;i3<tensor->ne[3];++i3) for(int64_t i2=0;i2<tensor->ne[2];++i2)
        for(int64_t i1=0;i1<tensor->ne[1];++i1) {
            const size_t offset=i3*tensor->nb[3]+i2*tensor->nb[2]+i1*tensor->nb[1];
            const size_t row=tensor->ne[0]*element;
            ggml_backend_tensor_get(tensor,result.bytes.data()+destination,offset,row);destination+=row;
        }
    return result;
}

bool is_layer_tensor(const char * name, const char * prefix) {
    const size_t n = std::strlen(prefix);
    if(std::strncmp(name,prefix,n)!=0 || name[n]!='-' || name[n+1]=='\0') return false;
    for(const char * p=name+n+1;*p;++p) if(*p<'0' || *p>'9') return false;
    return true;
}

int capture_routes(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & capture = *static_cast<route_capture *>(user_data);
    const bool wanted_head = std::getenv("ESE_LIFECYCLE_ASSERT_HEAD_F32") && std::strcmp(tensor->name, "result_output") == 0;
    if (wanted_head) {
        capture.invalid_head_precision |= tensor->op != GGML_OP_MUL_MAT || tensor->op_params[0] != GGML_PREC_F32 ||
                !tensor->src[0] || tensor->src[0]->type != GGML_TYPE_Q6_K;
        if (!ask) {
            ++capture.head_ops;
            if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) ++capture.head_gpu_ops;
        }
    }
    const int shared_index = is_layer_tensor(tensor->name, "ffn_gate") ? 0 :
            is_layer_tensor(tensor->name, "ffn_up") ? 1 :
            is_layer_tensor(tensor->name, "ffn_shexp_out") ? 2 : -1;
    const bool wanted_shared = std::getenv("ESE_LIFECYCLE_ASSERT_SHARED_F32") && shared_index >= 0;
    if (wanted_shared) {
        capture.invalid_shared_precision |= tensor->op != GGML_OP_MUL_MAT || tensor->op_params[0] != GGML_PREC_F32;
        if (!ask) {
            ++capture.shared_ops[shared_index];
            if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) ++capture.shared_gpu_ops[shared_index];
        }
    }
    const bool expert_gate = is_layer_tensor(tensor->name, "ffn_moe_gate_par");
    const bool expert_down = is_layer_tensor(tensor->name, "ffn_moe_down");
    const bool wanted_expert = std::getenv("ESE_LIFECYCLE_ASSERT_EXPERT_F32") && (expert_gate || expert_down);
    if (wanted_expert) {
        const bool precise = expert_gate ? tensor->op == GGML_OP_MOE_FUSED_UP_GATE && tensor->op_params[2] == GGML_PREC_F32 :
                tensor->op == GGML_OP_MUL_MAT_ID && tensor->op_params[0] == GGML_PREC_F32;
        capture.invalid_expert_precision |= !precise;
        if (!ask) {
            if (expert_gate) ++capture.expert_gate_ops;
            else ++capture.expert_down_ops;
            if (tensor->buffer && !ggml_backend_buffer_is_host(tensor->buffer)) {
                if (expert_gate) ++capture.expert_gate_gpu_ops;
                else ++capture.expert_down_gpu_ops;
            }
        }
    }
    const bool wanted_route = is_layer_tensor(tensor->name,"ffn_moe_topk") ||
                        is_layer_tensor(tensor->name,"qwen4exp_qsa_top_blocks");
    const char * suffix=std::strrchr(tensor->name,'-');char * end=nullptr;
    const long layer=suffix && suffix[1] ? std::strtol(suffix+1,&end,10) : -1;
    const int max_layer=std::getenv("ESE_LIFECYCLE_TRACE_MAX_LAYER")
        ? std::atoi(std::getenv("ESE_LIFECYCLE_TRACE_MAX_LAYER")) : 0;
    const bool trace_type = tensor->type==GGML_TYPE_F32 ||
            (tensor->type==GGML_TYPE_F16 && is_layer_tensor(tensor->name,"qwen4exp_qsa_sparse_mask"));
    const bool wanted_float = std::getenv("ESE_LIFECYCLE_TRACE_FLOATS") && trace_type &&
            ggml_nelements(tensor)<=65536 && ggml_nbytes(tensor)<=262144 &&
            ((end && *end=='\0' && layer>=0 && layer<=std::min(max_layer,47)) ||
             std::strcmp(tensor->name,"inp_qwen4exp_ple")==0 ||
             std::strcmp(tensor->name,"qwen4exp_token_embeddings")==0 ||
             std::strcmp(tensor->name,"qwen4exp_output_input")==0);
    const bool wanted=wanted_route || wanted_float || wanted_expert || wanted_shared || wanted_head;
    if(ask || !wanted) return wanted;
    if ((wanted_expert || wanted_shared || wanted_head) && !wanted_route && !wanted_float) return true;
    if (wanted_route && capture.arm_after_routes >= 0 && ++capture.observed_routes >= capture.arm_after_routes)
        capture.abort_armed.store(true, std::memory_order_relaxed);
    if(wanted_route && tensor->type != GGML_TYPE_I32) {capture.invalid_type=true;return 0;}
    const uint8_t * data=static_cast<const uint8_t *>(tensor->data);
    if(!ggml_backend_buffer_is_host(tensor->buffer)) {
        capture.staging.resize(ggml_nbytes(tensor));
        ggml_backend_tensor_get(tensor,capture.staging.data(),0,capture.staging.size());
        data=capture.staging.data();
    }
    if(wanted_float) {
        const std::string sample_name = wanted_expert ? std::string(tensor->name) +
                (ggml_backend_buffer_is_host(tensor->buffer) ? ".cpu" : ".gpu") : tensor->name;
        if (wanted_expert && std::getenv("ESE_LIFECYCLE_TRACE_EXPERT_INPUTS")) {
            auto & expert = capture.attention[sample_name];
            const bool fused = tensor->op == GGML_OP_MOE_FUSED_UP_GATE;
            expert.precision = tensor->op_params[fused ? 2 : 0];
            expert.operands[0] = capture_attention_operand(tensor->src[fused ? 2 : 1]);
            expert.operands[1] = capture_attention_operand(tensor->src[fused ? 3 : 2]);
            expert.operands[2] = capture_attention_operand(tensor->src[0]);
            expert.operands[3] = capture_attention_operand(fused ? tensor->src[1] : nullptr);
        }
        if(tensor->op==GGML_OP_FLASH_ATTN_EXT && std::getenv("ESE_LIFECYCLE_TRACE_ATTENTION_INPUTS")) {
            auto & attention=capture.attention[tensor->name];attention.precision=tensor->op_params[3];
            for(int i=0;i<4;++i) attention.operands[i]=capture_attention_operand(tensor->src[i]);
        }
        capture.float_order.emplace_back(sample_name);
        auto & sample=capture.floats[sample_name];
        std::copy(tensor->ne,tensor->ne+4,sample.shape.begin());sample.values.clear();
        for(int64_t i3=0;i3<tensor->ne[3];++i3) for(int64_t i2=0;i2<tensor->ne[2];++i2)
            for(int64_t i1=0;i1<tensor->ne[1];++i1) for(int64_t i0=0;i0<tensor->ne[0];++i0) {
                const size_t offset=i3*tensor->nb[3]+i2*tensor->nb[2]+i1*tensor->nb[1]+i0*tensor->nb[0];
                float value;
                if(tensor->type==GGML_TYPE_F16) {
                    ggml_fp16_t half;std::memcpy(&half,data+offset,sizeof(half));value=ggml_fp16_to_fp32(half);
                } else std::memcpy(&value,data+offset,sizeof(value));
                sample.values.push_back(value);
            }
        // Inputs are graph leaves and need not receive an execution callback.
        // Read the embedding actually consumed by this completed projection.
        if(is_layer_tensor(tensor->name,"qwen4exp_ple_key_projected") && tensor->src[1]) {
            auto input=*tensor->src[1];
            std::strcpy(input.name,"inp_qwen4exp_ple");
            capture_routes(&input,false,user_data);
        }
        return 1;
    }
    auto & values=capture.tensors[tensor->name];
    for(int64_t i3=0;i3<tensor->ne[3];++i3) for(int64_t i2=0;i2<tensor->ne[2];++i2)
        for(int64_t i1=0;i1<tensor->ne[1];++i1) for(int64_t i0=0;i0<tensor->ne[0];++i0) {
            const size_t offset=i3*tensor->nb[3]+i2*tensor->nb[2]+i1*tensor->nb[1]+i0*tensor->nb[0];
            int32_t value;std::memcpy(&value,data+offset,sizeof(value));values.push_back(value);
        }
    return 1;
}
static void dump_floats(const route_capture & capture, const std::string & phase) {
    const char * root=std::getenv("ESE_LIFECYCLE_TRACE_DIRECTORY");if(!root) return;
    auto directory=std::filesystem::path(root)/phase;std::filesystem::create_directories(directory);
    std::ofstream metadata(directory/"shapes.json");metadata<<"{";bool first=true;
    for(const auto & item:capture.floats) {
        if(!first) metadata<<",";first=false;metadata<<"\""<<item.first<<"\":[";
        for(int i=0;i<4;i++) {if(i) metadata<<",";metadata<<item.second.shape[i];}metadata<<"]";
        std::ofstream data(directory/(item.first+".f32"),std::ios::binary);
        data.write(reinterpret_cast<const char *>(item.second.values.data()),item.second.values.size()*sizeof(float));
        if(!data) throw std::runtime_error("float trace write");
    }
    std::ofstream order(directory/"order.txt");for(const auto & name:capture.float_order) order<<name<<"\n";
    metadata<<"}";if(!metadata) throw std::runtime_error("float trace metadata write");
    for(const auto & item:capture.attention) {
        std::ofstream info(directory/(item.first+".attention.json"));
        info<<"{\"precision\":"<<item.second.precision<<",\"operands\":[";
        for(int i=0;i<4;++i) {
            if(i) info<<",";
            const auto & operand=item.second.operands[i];
            const auto name=item.first+".src"+std::to_string(i)+".raw";
            info<<"{\"type\":"<<operand.type<<",\"shape\":[";
            for(int axis=0;axis<4;++axis) {if(axis) info<<",";info<<operand.shape[axis];}
            info<<"],\"captured\":"<<(operand.bytes.empty()?"false":"true")<<",\"file\":\""<<name<<"\"}";
            if(!operand.bytes.empty()) {
                std::ofstream data(directory/name,std::ios::binary);
                data.write(reinterpret_cast<const char *>(operand.bytes.data()),operand.bytes.size());
                if(!data) throw std::runtime_error("attention input write");
            }
        }
        info<<"]}";if(!info) throw std::runtime_error("attention metadata write");
    }
}
} // namespace

static void require(bool ok, const char * reason) { if (!ok) throw std::runtime_error(reason); }
static std::vector<uint8_t> save_full(llama_context * ctx) {
    std::vector<uint8_t> data(llama_state_get_size(ctx));
    require(!data.empty() && llama_state_get_data(ctx, data.data(), data.size()) == data.size(), "full snapshot");
    return data;
}
static std::vector<uint8_t> save_seq(llama_context * ctx, int seq) {
    std::vector<uint8_t> data(llama_state_seq_get_size(ctx, seq, 0));
    require(!data.empty() && llama_state_seq_get_data(ctx, data.data(), data.size(), seq, 0) == data.size(), "sequence snapshot");
    return data;
}
struct observation { std::vector<float> logits; std::map<std::string, std::vector<int32_t>> routes; };
static observation decode(llama_context * ctx, llama_model * model, route_capture & capture, int token, int pos, int seq) {
    capture.tensors.clear(); capture.floats.clear(); capture.float_order.clear(); capture.attention.clear(); capture.invalid_type=false;
    capture.invalid_expert_precision=false; capture.expert_gate_ops=0; capture.expert_down_ops=0;
    capture.expert_gate_gpu_ops=0; capture.expert_down_gpu_ops=0;
    capture.invalid_shared_precision=false; capture.shared_ops={}; capture.shared_gpu_ops={};
    capture.invalid_head_precision=false; capture.head_ops=0; capture.head_gpu_ops=0;
    auto batch = llama_batch_init(1, 0, 1);
    batch.n_tokens=1; batch.token[0]=token; batch.pos[0]=pos; batch.n_seq_id[0]=1; batch.seq_id[0][0]=seq; batch.logits[0]=true;
    int status=llama_decode(ctx, batch); llama_batch_free(batch);
    require(status==0, "single-token decode"); llama_synchronize(ctx);
    observation result;
    const float * values=llama_get_logits_ith(ctx, -1);
    require(values && !capture.invalid_type && capture.tensors.size()==60, "outputs/routes missing");
    if (std::getenv("ESE_LIFECYCLE_ASSERT_EXPERT_F32")) {
        require(!capture.invalid_expert_precision && capture.expert_gate_ops >= 48 && capture.expert_down_ops >= 48,
                "explicit expert precision missing/bypassed");
        std::printf("expert_precision: gate_ops=%zu down_ops=%zu gate_gpu_ops=%zu down_gpu_ops=%zu explicit=true passed=true\n",
                capture.expert_gate_ops, capture.expert_down_ops, capture.expert_gate_gpu_ops, capture.expert_down_gpu_ops);
    }
    if (std::getenv("ESE_LIFECYCLE_ASSERT_SHARED_F32")) {
        require(!capture.invalid_shared_precision &&
                std::all_of(capture.shared_ops.begin(), capture.shared_ops.end(), [](size_t count) { return count >= 48; }),
                "explicit shared precision missing/bypassed");
        if (std::getenv("ESE_LIFECYCLE_ASSERT_SHARED_GPU"))
            require(capture.shared_ops == capture.shared_gpu_ops, "shared projection silently executed on host");
        std::printf("shared_precision: gate_ops=%zu up_ops=%zu down_ops=%zu gate_gpu_ops=%zu up_gpu_ops=%zu down_gpu_ops=%zu explicit=true passed=true\n",
                capture.shared_ops[0], capture.shared_ops[1], capture.shared_ops[2],
                capture.shared_gpu_ops[0], capture.shared_gpu_ops[1], capture.shared_gpu_ops[2]);
    }
    result.logits.assign(values, values+llama_n_vocab(model)); result.routes=capture.tensors;
    if (std::getenv("ESE_LIFECYCLE_ASSERT_HEAD_F32")) {
        require(!capture.invalid_head_precision && capture.head_ops >= 1, "explicit head precision missing/bypassed");
        if (std::getenv("ESE_LIFECYCLE_ASSERT_HEAD_GPU")) require(capture.head_ops == capture.head_gpu_ops, "head silently executed on host");
        std::printf("head_precision: ops=%zu gpu_ops=%zu explicit=true passed=true\n",capture.head_ops,capture.head_gpu_ops);
    }
    for (float v:result.logits) require(std::isfinite(v), "nonfinite logits");
    if (const char * path = std::getenv("ESE_LIFECYCLE_TOKEN_ROUTES")) {
        std::ofstream output(path, std::ios::app);
        output << "{\"decode\":" << capture.decode_calls++ << ",\"seq\":" << seq << ",\"pos\":" << pos << ",\"routes\":{";
        bool first = true;
        for (const auto & item : result.routes) {
            if (!first) output << ','; first = false;
            output << '"' << item.first << "\":[";
            for (size_t i = 0; i < item.second.size(); ++i) { if (i) output << ','; output << item.second[i]; }
            output << ']';
        }
        output << "}}\n"; require(bool(output), "per-token routes write");
    }
    if (std::getenv("ESE_LIFECYCLE_TRACE_EACH_DECODE") &&
            (!std::getenv("ESE_LIFECYCLE_TRACE_EACH_DECODE_MAX_POS") ||
             pos <= std::atoi(std::getenv("ESE_LIFECYCLE_TRACE_EACH_DECODE_MAX_POS"))))
        dump_floats(capture, "seq" + std::to_string(seq) + "-pos" + std::to_string(pos));
    return result;
}
static void compare(const char * name, const observation & a, const observation & b, bool compare_physical_qsa = true) {
    require(a.logits.size()==b.logits.size(), "logit width");
    double maximum=0; size_t changed=0;
    for(size_t i=0;i<a.logits.size();i++) {
        maximum=std::max(maximum,double(std::fabs(a.logits[i]-b.logits[i])));
        changed+=std::memcmp(&a.logits[i], &b.logits[i], sizeof(float))!=0;
    }
    bool routes=true;
    for (const auto & item:a.routes) {
        if(!compare_physical_qsa && is_layer_tensor(item.first.c_str(), "qwen4exp_qsa_top_blocks")) continue;
        const auto found=b.routes.find(item.first); routes &= found!=b.routes.end() && item.second==found->second;
    }
    if(!compare_physical_qsa) std::puts("lifecycle_route_scope: logits and 48 MoE routes; QSA physical block IDs omitted for independently placed sequence");
    std::printf("lifecycle: {\"test\":\"%s\",\"logits\":%zu,\"changed\":%zu,\"max_abs\":%.9g,\"routes_equal\":%s}\n",name,a.logits.size(),changed,maximum,routes?"true":"false");
    std::fflush(stdout); require(changed==0 && routes, name);
}
static bool abort_now(void * count) { ++*static_cast<int *>(count); return true; }
static bool abort_when_armed(void * data) {
    auto & capture = *static_cast<route_capture *>(data);
    ++capture.abort_polls;
    return capture.abort_armed.load(std::memory_order_relaxed);
}

// Persistent-context resource diagnostic. Replays a bounded short prompt;
// this does not qualify a filled long context or concurrent HTTP requests.
static void decode_soak(llama_context * ctx, llama_model * model,
        route_capture & capture, int seconds, int token_count) {
    const char * root = std::getenv("ESE_LIFECYCLE_DIRECTORY");
    require(root, "soak artifact directory");
    const auto directory = std::filesystem::path(root);
    std::filesystem::create_directories(directory);
    const char * prompt = std::getenv("ESE_LIFECYCLE_PROMPT");
    if (!prompt) prompt = "[INFO] token cache requested layer 7 expert 12. The engine loads weights and computes a response. What is two plus two?\n";
    const auto source = common_tokenize(model, prompt, false, true);
    require(source.size() >= size_t(token_count) && llama_n_ctx(ctx) >= uint32_t(token_count), "soak prompt/context");
    const std::vector<llama_token> tokens(source.begin(), source.begin()+token_count);
    std::ofstream ids(directory / "input-ids.json");
    ids << '[';
    for (size_t i = 0; i < tokens.size(); ++i) { if (i) ids << ','; ids << tokens[i]; }
    ids << "]\n";
    require(bool(ids), "soak input write");
    std::vector<observation> baseline;
    uint64_t decode_ns = 0;
    size_t rounds = 0, decoded_tokens = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    };
    std::vector<uint64_t> steady_compute;
    const auto checkpoint = [&] {
        std::vector<uint64_t> compute;
        for (int backend = 0; backend < ggml_backend_sched_get_n_backends(ctx->sched); ++backend) {
            ggml_gallocr_buffer_stats arena = {};
            require(ggml_backend_sched_get_graph_buffer_stats(ctx->sched, backend, &arena), "soak arena inventory");
            compute.push_back(arena.allocated_bytes);
#ifdef GGML_USE_CUDA
            if (ggml_backend_is_cuda(ctx->backends[backend])) {
                const int device = ggml_backend_cuda_get_device(ctx->backends[backend]);
                ggml_cuda_pool_memory_stats pool = {};
                require(ggml_backend_cuda_get_pool_memory_stats(ctx->backends[backend], device, &pool), "soak pool inventory");
                require(pool.active_bytes == 0, "soak pool leases did not drain");
                compute.push_back(pool.physical_bytes);
                std::printf("soak_pool: {\"round\":%zu,\"backend\":%d,\"physical_bytes\":%llu,\"active_bytes\":%llu}\n",
                        rounds, backend, (unsigned long long)pool.physical_bytes, (unsigned long long)pool.active_bytes);
            }
#endif
        }
        // Both reset and normal-position graphs have run after two rounds.
        if (rounds == 2) steady_compute = compute;
        if (rounds > 2) require(compute == steady_compute, "soak compute backing grew after warmup");
        for (size_t device = 0; device < ctx->compute_budgets.size(); ++device) if (ctx->compute_budgets[device]) {
            ggml_memory_budget_stats budget = {};
            require(ggml_memory_budget_get(ctx->compute_budgets[device], &budget), "soak shared budget inventory");
            require(budget.reserved_bytes <= budget.budget_bytes, "soak shared budget exceeded");
            std::printf("soak_shared: {\"round\":%zu,\"device\":%zu,\"reserved_bytes\":%llu,\"budget_bytes\":%llu,\"rejections\":%llu}\n",
                    rounds, device, (unsigned long long)budget.reserved_bytes,
                    (unsigned long long)budget.budget_bytes, (unsigned long long)budget.rejections);
        }
        const uint32_t device_count = llama_resource_device_count(ctx);
        std::vector<llama_resource_device_snapshot> devices(device_count);
        llama_resource_snapshot resources = {};
        require(llama_resource_get_snapshot(ctx, &resources, devices.data(), device_count), "soak resource snapshot");
        require(resources.expert_ram_resident_bytes <= resources.expert_ram_capacity_bytes &&
                resources.expert_ram_active_leases == 0, "soak expert cap/lease drain");
        std::printf("soak_expert_cache: {\"round\":%zu,\"resident_bytes\":%llu,\"capacity_bytes\":%llu,\"active_leases\":%llu}\n",
                rounds, (unsigned long long)resources.expert_ram_resident_bytes,
                (unsigned long long)resources.expert_ram_capacity_bytes,
                (unsigned long long)resources.expert_ram_active_leases);
        uint32_t count = 0;
        require(llama_resource_get_memory_pools(ctx, nullptr, 0, &count), "soak native inventory count");
        std::vector<llama_resource_memory_pool> pools(count);
        require(llama_resource_get_memory_pools(ctx, pools.data(), count, &count), "soak native inventory");
        for (const auto & pool : pools) std::printf("soak_inventory: {\"round\":%zu,\"backend\":%d,\"category\":%d,\"bytes\":%llu}\n",
                rounds, pool.backend_id, int(pool.category), (unsigned long long)pool.bytes);
        std::printf("soak_checkpoint: {\"rounds\":%zu,\"decoded_tokens\":%zu,\"elapsed_seconds\":%.6f,\"decode_ns\":%llu}\n",
                rounds, decoded_tokens, elapsed(), (unsigned long long)decode_ns);
        std::fflush(stdout);
    };
    do {
        llama_kv_cache_clear(ctx);
        require(llama_get_kv_cache_used_cells(ctx) == 0 &&
                std::all_of(ctx->qwen4exp_last_logit_pos.begin(), ctx->qwen4exp_last_logit_pos.end(),
                            [](llama_pos position) { return position == -1; }), "soak reset retained token/output ownership");
        for (size_t position = 0; position < tokens.size(); ++position) {
            const auto before = std::chrono::steady_clock::now();
            auto result = decode(ctx, model, capture, tokens[position], int(position), 0);
            decode_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-before).count();
            ++decoded_tokens;
            if (rounds == 0) baseline.push_back(std::move(result));
            else compare("soak-reset-replay", baseline[position], result);
        }
        require(llama_kv_cache_seq_pos_max(ctx, 0) == token_count - 1, "soak final position");
        ++rounds;
        checkpoint();
        if (rounds == 1) {
            std::ofstream logits(directory / "baseline-logits.bin", std::ios::binary);
            std::ofstream routes(directory / "baseline-routes.txt");
            for (size_t token = 0; token < baseline.size(); ++token) {
                logits.write(reinterpret_cast<const char *>(baseline[token].logits.data()), baseline[token].logits.size()*sizeof(float));
                for (const auto & item : baseline[token].routes) {
                    routes << token << ' ' << item.first;
                    for (int32_t id : item.second) routes << ' ' << id;
                    routes << '\n';
                }
            }
            require(bool(logits) && bool(routes), "soak baseline write");
        }
    } while (rounds < 3 || elapsed() < seconds);
    std::printf("soak_result: {\"passed\":true,\"requested_seconds\":%d,\"elapsed_seconds\":%.6f,\"rounds\":%zu,\"decoded_tokens\":%zu,\"decode_ns\":%llu,\"prompt_tokens\":%zu,\"filled_long_context_qualified\":false,\"concurrent_sessions_qualified\":false}\n",
            seconds, elapsed(), rounds, decoded_tokens, (unsigned long long)decode_ns, tokens.size());
    std::fflush(stdout);
}
int main(int argc,char ** argv) {
    llama_context * ctx=nullptr; llama_model * model=nullptr;
    try {
        gpt_params params; if(!gpt_params_parse(argc,argv,params)) return 2;
        const char * init_flag=std::getenv("ESE_LIFECYCLE_INIT_ONLY");
        const char * prefill_flag=std::getenv("ESE_LIFECYCLE_PREFILL_TOKENS");
        const char * soak_flag=std::getenv("ESE_LIFECYCLE_SOAK_SECONDS");
        const char * soak_tokens_flag=std::getenv("ESE_LIFECYCLE_SOAK_TOKENS");
        int soak_seconds=0;
        int soak_tokens=9;
        if(soak_tokens_flag) {
            size_t used=0;soak_tokens=std::stoi(soak_tokens_flag,&used);
            require(soak_flag && used==std::strlen(soak_tokens_flag) && soak_tokens>=1,
                    "soak token count must be positive and requires soak mode");
        }
        if(soak_flag) {
            size_t used=0;soak_seconds=std::stoi(soak_flag,&used);
            require(used==std::strlen(soak_flag) && soak_seconds>=1 && soak_seconds<=86400 && !init_flag && !prefill_flag,
                    "soak duration must be 1..86400 and cannot combine with init/prefill");
            require(!std::getenv("ESE_LIFECYCLE_TRACE_FLOATS") && !std::getenv("ESE_LIFECYCLE_TOKEN_ROUTES"),
                    "soak cannot accumulate full per-token trace artifacts");
        }
        int prefill_tokens=0;
        if(prefill_flag) {
            size_t used=0;prefill_tokens=std::stoi(prefill_flag,&used);
            require(used==std::strlen(prefill_flag) && prefill_tokens>=1 && prefill_tokens<=1024 && !init_flag,
                    "prefill count must be 1..1024 and cannot combine with init-only");
        }
        require(!init_flag || std::strcmp(init_flag,"1")==0,"init-only flag must be 1");
        const int requested_batch=params.n_batch,requested_ubatch=params.n_ubatch;
        if(std::getenv("ESE_LIFECYCLE_NO_GRAPH_REUSE")) params.graph_reuse=false;
        const bool direct_recovery = !std::getenv("ESE_LIFECYCLE_PREFIX_BANKS") &&
            std::getenv("ESE_LIFECYCLE_SPEC_MODE") &&
            std::stoi(std::getenv("ESE_LIFECYCLE_SPEC_MODE")) == LLAMA_SPEC_CKPT_PER_STEP &&
            (std::getenv("ESE_LIFECYCLE_ABORT_CHECKPOINT") ||
             (std::getenv("ESE_LIFECYCLE_BATCH_DIAGNOSTIC") && std::getenv("ESE_LIFECYCLE_BATCH_RECOVERY")));
        params.n_parallel=(direct_recovery||prefill_flag||soak_flag)?1:2; params.warmup=false;
        if(prefill_flag) params.logits_all=std::getenv("ESE_LIFECYCLE_PREFILL_ALL_LOGITS")!=nullptr;
        route_capture capture; params.cb_eval=capture_routes; params.cb_eval_user_data=&capture;
        llama_backend_init(); llama_numa_init(params.numa);
        auto init=llama_init_from_gpt_params(params); ctx=init.context; model=init.model;
        require(ctx && model, "model/context init");
        if(soak_flag) {
            decode_soak(ctx,model,capture,soak_seconds,soak_tokens);
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::puts("soak_teardown: completed");
            return 0;
        }
        if(prefill_flag) {
            const bool all_logits=std::getenv("ESE_LIFECYCLE_PREFILL_ALL_LOGITS")!=nullptr;
            const bool sequential=std::getenv("ESE_LIFECYCLE_PREFILL_SEQUENTIAL")!=nullptr;
            const char * root=std::getenv("ESE_LIFECYCLE_DIRECTORY");require(root,"prefill artifact directory");
            const auto directory=std::filesystem::path(root);
            std::filesystem::create_directories(directory);
            const char * prompt=std::getenv("ESE_LIFECYCLE_PROMPT");
            if(!prompt) prompt="[INFO] token cache requested layer 7 expert 12. The engine loads weights and computes a response. What is two plus two?\n";
            const auto source=common_tokenize(model,prompt,false,true);require(!source.empty(),"prefill tokenization");
            std::vector<llama_token> input(prefill_tokens);
            std::ofstream ids(directory/"input-ids.json");ids<<'[';
            for(int i=0;i<prefill_tokens;++i){input[i]=source[i%source.size()];if(i)ids<<',';ids<<input[i];}ids<<"]\n";require(bool(ids),"prefill input write");
            const auto inventory=[&](const char * phase) {
                std::printf("prefill_checkpoint: phase=%s\n",phase);
                for(size_t device=0;device<ctx->compute_budgets.size();++device) if(ctx->compute_budgets[device]) {
                    ggml_memory_budget_stats s={};require(ggml_memory_budget_get(ctx->compute_budgets[device],&s),"prefill shared budget");
                    std::printf("prefill_shared: {\"device\":%zu,\"reserved_bytes\":%llu,\"peak_reserved_bytes\":%llu,\"budget_bytes\":%llu,\"rejections\":%llu}\n",device,
                        (unsigned long long)s.reserved_bytes,(unsigned long long)s.peak_reserved_bytes,(unsigned long long)s.budget_bytes,(unsigned long long)s.rejections);
                }
                for(int i=0;i<ggml_backend_sched_get_n_backends(ctx->sched);++i) {
                    ggml_gallocr_buffer_stats s={};require(ggml_backend_sched_get_graph_buffer_stats(ctx->sched,i,&s),"prefill arena stats");
                    std::printf("prefill_graph: {\"backend_id\":%d,\"allocated_bytes\":%llu,\"peak_allocated_bytes\":%llu,\"budget_rejections\":%llu}\n",i,
                        (unsigned long long)s.allocated_bytes,(unsigned long long)s.peak_allocated_bytes,(unsigned long long)s.budget_rejections);
#ifdef GGML_USE_CUDA
                    if(ggml_backend_is_cuda(ctx->backends[i])) {
                        const int d=ggml_backend_cuda_get_device(ctx->backends[i]);ggml_cuda_pool_memory_stats p={};
                        require(ggml_backend_cuda_get_pool_memory_stats(ctx->backends[i],d,&p),"prefill pool stats");
                        std::printf("prefill_pool: {\"backend_id\":%d,\"device\":%d,\"physical_bytes\":%llu,\"active_bytes\":%llu,\"peak_physical_bytes\":%llu}\n",i,d,
                            (unsigned long long)p.physical_bytes,(unsigned long long)p.active_bytes,(unsigned long long)p.peak_physical_bytes);
                    }
#endif
                }
                std::fflush(stdout);
            };
            std::printf("prefill_config: {\"tokens\":%d,\"requested_batch\":%d,\"requested_ubatch\":%d,\"effective_ubatch\":%u,\"context_tokens\":%u,\"sequential\":%s,\"all_logits\":%s}\n",
                        prefill_tokens,requested_batch,requested_ubatch,llama_n_ubatch(ctx),llama_n_ctx(ctx),sequential?"true":"false",all_logits?"true":"false");
            inventory("initialized");
            std::ofstream logits(directory/"logits.bin",std::ios::binary),routes(directory/"routes.txt");
            const int vocab=llama_n_vocab(model);const auto start=std::chrono::steady_clock::now();
            if(sequential) {
                for(int i=0;i<prefill_tokens;++i) {
                    auto result=decode(ctx,model,capture,input[i],i,0);
                    if(all_logits||i==prefill_tokens-1)logits.write(reinterpret_cast<const char *>(result.logits.data()),result.logits.size()*sizeof(float));
                    for(const auto & item:result.routes)if(is_layer_tensor(item.first.c_str(),"ffn_moe_topk")) {
                        routes<<i<<' '<<item.first;for(int value:item.second)routes<<' '<<value;routes<<'\n';
                    }
                }
            } else {
                auto batch=llama_batch_init(prefill_tokens,0,1);batch.n_tokens=prefill_tokens;
                for(int i=0;i<prefill_tokens;++i){batch.token[i]=input[i];batch.pos[i]=i;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=all_logits||i==prefill_tokens-1;}
                const int status=llama_decode(ctx,batch);llama_batch_free(batch);llama_synchronize(ctx);
                std::printf("prefill_decode_status: %d\n",status);std::fflush(stdout);
                require(status==0 && !capture.invalid_type,"prefill batch execution");
                if(std::getenv("ESE_LIFECYCLE_ASSERT_EXPERT_F32"))
                    require(!capture.invalid_expert_precision && capture.expert_gate_ops>=48 && capture.expert_down_ops>=48,
                            "prefill explicit expert precision missing/bypassed");
                for(int i=0;i<prefill_tokens;++i)if(all_logits||i==prefill_tokens-1) {
                    const float * row=llama_get_logits_ith(ctx,i);require(row,"prefill logits");
                    for(int j=0;j<vocab;++j)require(std::isfinite(row[j]),"non-finite prefill logit");
                    logits.write(reinterpret_cast<const char *>(row),size_t(vocab)*sizeof(float));
                }
                size_t layers=0;
                for(const auto & item:capture.tensors)if(is_layer_tensor(item.first.c_str(),"ffn_moe_topk")) {
                    require(item.second.size()==size_t(prefill_tokens)*10,"prefill route geometry");++layers;
                    for(int i=0;i<prefill_tokens;++i){routes<<i<<' '<<item.first;for(int j=0;j<10;++j)routes<<' '<<item.second[i*10+j];routes<<'\n';}
                }
                require(layers==48,"prefill missing expert routes");
            }
            const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
            require(bool(logits)&&bool(routes),"prefill artifacts write");logits.close();routes.close();
            require(llama_kv_cache_seq_pos_max(ctx,0)==prefill_tokens-1,"prefill final position");
            inventory("executed");
            std::printf("prefill_execution: {\"tokens\":%d,\"decode_and_capture_ns\":%lld,\"position\":%d}\n",prefill_tokens,(long long)ns,llama_kv_cache_seq_pos_max(ctx,0));std::fflush(stdout);
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::puts("prefill_execution_result: passed; resources only, parity and throughput require separate qualification");return 0;
        }
        if(std::getenv("ESE_LIFECYCLE_CACHE_SIZING")) {
            uint64_t payload=0,indexer=0;
            std::set<const ggml_tensor *> counted;
            for(const auto * family:{&ctx->kv_self.k_l,&ctx->kv_self.v_l,&ctx->kv_self.s_l,&ctx->kv_self.kr_l})
                for(const auto * tensor:*family) if(tensor && counted.insert(tensor).second) {
                    payload+=ggml_nbytes(tensor);
                    if(family==&ctx->kv_self.kr_l) indexer+=ggml_nbytes(tensor);
                }
            const uint64_t predicted=llama_model_kv_bytes(model,ctx->kv_self.type_k,ctx->kv_self.type_v,
                    ctx->cparams.idx_type_k,ctx->kv_self.size,ctx->cparams.mla_attn,
                    ctx->cparams.n_seq_max,ctx->cparams.flash_attn);
            std::printf("cache_sizing_live: {\"context\":%u,\"slots\":%u,\"predicted_bytes\":%llu,\"tensor_payload_bytes\":%llu,\"indexer_bytes\":%llu}\n",
                    ctx->kv_self.size,ctx->cparams.n_seq_max,(unsigned long long)predicted,
                    (unsigned long long)payload,(unsigned long long)indexer);
            std::fflush(stdout);
            require(predicted==payload,"cache prediction differs from live tensor payloads");
        }
        if(std::getenv("ESE_LIFECYCLE_MEMORY_POOLS")) {
            uint32_t count=0;require(llama_resource_get_memory_pools(ctx,nullptr,0,&count),"memory inventory query");
            std::vector<llama_resource_memory_pool> pools(count);
            require(llama_resource_get_memory_pools(ctx,pools.data(),pools.size(),&count),"memory inventory read");
            for(const auto & pool:pools)
                std::printf("memory_pool: {\"backend_id\":%d,\"category\":%d,\"bytes\":%llu}\n",
                            pool.backend_id,int(pool.category),(unsigned long long)pool.bytes);
            std::fflush(stdout);
        }
        if(init_flag) {
            std::printf("prefill_init: {\"requested_batch\":%d,\"requested_ubatch\":%d,\"effective_ubatch\":%u,\"context_tokens\":%u,\"sequences\":%u}\n",
                        requested_batch,requested_ubatch,llama_n_ubatch(ctx),llama_n_ctx(ctx),ctx->cparams.n_seq_max);
            for(int i=0;i<ggml_backend_sched_get_n_backends(ctx->sched);++i) {
                ggml_gallocr_buffer_stats stats={};
                require(ggml_backend_sched_get_graph_buffer_stats(ctx->sched,i,&stats),"init graph statistics");
                std::printf("prefill_graph: {\"backend_id\":%d,\"allocated_bytes\":%llu,\"peak_allocated_bytes\":%llu,\"required_bytes\":%llu,\"budget_bytes\":%llu,\"allocation_failures\":%llu,\"budget_rejections\":%llu}\n",
                    i,(unsigned long long)stats.allocated_bytes,(unsigned long long)stats.peak_allocated_bytes,
                    (unsigned long long)stats.required_bytes,(unsigned long long)stats.allocation_budget_bytes,
                    (unsigned long long)stats.allocation_failures,(unsigned long long)stats.budget_rejections);
#ifdef GGML_USE_CUDA
                if(ggml_backend_is_cuda(ctx->backends[i])) {
                    const int device=ggml_backend_cuda_get_device(ctx->backends[i]);
                    size_t free_bytes=0,total_bytes=0;
                    ggml_backend_cuda_get_device_memory(device,&free_bytes,&total_bytes);
                    std::printf("prefill_device: {\"backend_id\":%d,\"visible_device_id\":%d,\"free_bytes\":%llu,\"total_bytes\":%llu}\n",
                        i,device,(unsigned long long)free_bytes,(unsigned long long)total_bytes);
                }
#endif
            }
            std::fflush(stdout);
            // Hold the initialized buffers for five 50ms external samples.
            // This diagnostic pause is excluded from performance measurements.
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::puts("prefill_init_result: passed; initialization only, no output parity or throughput qualification");
            return 0;
        }
        const int prefix=std::getenv("ESE_LIFECYCLE_PREFIX")?std::stoi(std::getenv("ESE_LIFECYCLE_PREFIX")):32;
        require(prefix>=8 && prefix<=256, "prefix range");
        const char * prompt=std::getenv("ESE_LIFECYCLE_PROMPT");
        if (!prompt) prompt="[INFO] token cache requested layer 7 expert 12. The engine loads weights and computes a response. What is two plus two?\n";
        auto source=common_tokenize(model, prompt, false, true);
        require(!source.empty(), "tokenization");
        std::vector<llama_token> tokens(prefix+9);
        for(size_t i=0;i<tokens.size();i++) tokens[i]=source[i%source.size()];
        if (const char * path=std::getenv("ESE_LIFECYCLE_INPUT_IDS")) {
            std::ofstream ids(path);
            ids << "[";
            for (int i=0;i<=prefix;i++) { if(i) ids << ","; ids << tokens[i]; }
            ids << "]\n";
            require(bool(ids), "input token artifact write");
        }
        std::ofstream teacher;
        if (const char * path=std::getenv("ESE_LIFECYCLE_TEACHER")) teacher.open(path,std::ios::binary);
        for(int i=0;i<prefix;i++) {
            auto result=decode(ctx,model,capture,tokens[i],i,0);
            if(teacher.is_open()) teacher.write(reinterpret_cast<const char *>(result.logits.data()),result.logits.size()*sizeof(float));
        }
        auto full=save_full(ctx); auto sequence=save_seq(ctx,0);
        const auto save_state_artifact = [&](const char * name, const std::vector<uint8_t> & data) {
            if (const char * path = std::getenv("ESE_LIFECYCLE_STATE_DIRECTORY")) {
                std::filesystem::create_directories(path);
                std::ofstream output(std::filesystem::path(path) / name, std::ios::binary);
                output.write(reinterpret_cast<const char *>(data.data()), data.size());
                require(bool(output), "sequence state artifact write");
            }
        };
        save_state_artifact("prefix-sequence.bin", sequence);
        std::printf("lifecycle_state: {\"prefix\":%d,\"full_bytes\":%zu,\"sequence_bytes\":%zu}\n",prefix,full.size(),sequence.size()); std::fflush(stdout);
        auto restore_full=[&] {require(llama_state_set_data(ctx,full.data(),full.size())==full.size(), "full restore");};
        auto restore_seq=[&](int seq) {llama_kv_cache_clear(ctx); require(llama_state_seq_set_data(ctx,sequence.data(),sequence.size(),seq,0)==sequence.size(), "seq restore");};
        auto continuation=[&](int seq) {return decode(ctx,model,capture,tokens[prefix],prefix,seq);};
        auto replay=[&](int seq) {observation r; for(int i=0;i<=prefix;i++) r=decode(ctx,model,capture,tokens[i],i,seq); return r;};
        if (std::getenv("ESE_LIFECYCLE_POSITION_DIAGNOSTIC")) {
            for (const std::string mode : {"shift", "divide", "partial_shift"}) {
                restore_full();
                if (mode == "shift") llama_kv_cache_seq_add(ctx, 0, 0, -1, 3);
                else if (mode == "divide") llama_kv_cache_seq_div(ctx, 0, 0, -1, 2);
                else llama_kv_cache_seq_add(ctx, 0, 0, prefix / 2, prefix);
                // Save without an explicit cache update: serialization must
                // flush pending key rotations and preserve token chronology.
                const auto shifted_full = save_full(ctx);
                const auto shifted_seq = save_seq(ctx, 0);
                save_state_artifact(("position-" + mode + "-sequence.bin").c_str(), shifted_seq);
                const llama_pos next = llama_kv_cache_seq_pos_max(ctx, 0) + 1;
                const auto control = decode(ctx, model, capture, tokens[prefix], next, 0);
                require(llama_state_set_data(ctx, shifted_full.data(), shifted_full.size()) == shifted_full.size(),
                        "position full-state restore");
                compare((mode + "_full_roundtrip").c_str(), control,
                        decode(ctx, model, capture, tokens[prefix], next, 0));
                llama_kv_cache_clear(ctx);
                require(llama_state_seq_set_data(ctx, shifted_seq.data(), shifted_seq.size(), 1, 0) == shifted_seq.size(),
                        "position sequence-state migration");
                compare((mode + "_sequence_migration").c_str(), control,
                        decode(ctx, model, capture, tokens[prefix], next, 1), false);
                std::printf("position_lifecycle: mode=%s next_position=%d full_bytes=%zu sequence_bytes=%zu passed=true\n",
                        mode.c_str(), next, shifted_full.size(), shifted_seq.size());
                std::fflush(stdout);
            }
            if (std::getenv("ESE_LIFECYCLE_SHARED_POSITION_DIAGNOSTIC")) {
                restore_full();
                const auto other_control = continuation(0);
                const auto transform = [&](const std::string & mode, int seq) {
                    if (mode == "shift") llama_kv_cache_seq_add(ctx, seq, 0, -1, 3);
                    else if (mode == "divide") llama_kv_cache_seq_div(ctx, seq, 0, -1, 2);
                    else llama_kv_cache_seq_add(ctx, seq, 0, prefix / 2, prefix);
                };
                for (const std::string mode : {"shift", "divide", "partial_shift"}) {
                    restore_full(); transform(mode, 0);
                    save_state_artifact(("shared-" + mode + "-private-sequence.bin").c_str(), save_seq(ctx, 0));
                    const auto next = llama_kv_cache_seq_pos_max(ctx, 0) + 1;
                    const auto private_control = decode(ctx, model, capture, tokens[prefix], next, 0);
                    restore_full(); llama_kv_cache_seq_cp(ctx, 0, 1, 0, -1);
                    const auto other_before = save_seq(ctx, 0);
                    transform(mode, 1);
                    const auto target = save_seq(ctx, 1);
                    const auto other_after = save_seq(ctx, 0);
                    require(other_after == other_before, "shared transform changed other sequence snapshot");
                    save_state_artifact(("shared-" + mode + "-target-sequence.bin").c_str(), target);
                    save_state_artifact(("shared-" + mode + "-other-sequence.bin").c_str(), other_after);
                    compare((mode + "_shared_target").c_str(), private_control,
                            decode(ctx, model, capture, tokens[prefix], next, 1), false);
                    compare((mode + "_shared_other").c_str(), other_control, continuation(0), false);
                    std::printf("shared_position_lifecycle: mode=%s other_snapshot_exact=true next_position=%d passed=true\n", mode.c_str(), next);
                    std::fflush(stdout);
                }
                restore_full();
                require(llama_state_seq_set_data(ctx, sequence.data(), sequence.size(), 1, 0) == sequence.size(),
                        "independently restored overlap prefix");
                const auto overlap_source = save_seq(ctx, 0);
                llama_kv_cache_seq_cp(ctx, 0, 1, 0, -1);
                require(save_seq(ctx, 0) == overlap_source, "overlapping copy changed source snapshot");
                require(llama_get_kv_cache_used_cells(ctx) == prefix, "overlapping copy retained redundant token rows");
                compare("overlap_copy_target", other_control, continuation(1), false);
                compare("overlap_copy_source", other_control, continuation(0), false);
            }
            restore_full();
            if (std::getenv("ESE_LIFECYCLE_POSITION_ONLY")) {
                llama_free(ctx); ctx = nullptr; llama_free_model(model); model = nullptr; llama_backend_free();
                std::puts("position_lifecycle_result: passed"); return 0;
            }
        }
        auto expected=continuation(0);
        if (std::getenv("ESE_LIFECYCLE_STATE_DIRECTORY"))
            save_state_artifact("continuation-sequence.bin", save_seq(ctx, 0));
        if(teacher.is_open()) {teacher.write(reinterpret_cast<const char *>(expected.logits.data()),expected.logits.size()*sizeof(float)); teacher.close();}
        if (const char * output=std::getenv("ESE_LIFECYCLE_BASELINE")) {
            std::ofstream logits(std::string(output)+".logits.bin",std::ios::binary);
            logits.write(reinterpret_cast<const char *>(expected.logits.data()), expected.logits.size()*sizeof(float));
            require(bool(logits), "baseline logits write");
            std::ofstream routes(std::string(output)+".routes.txt");
            for(const auto & item:expected.routes) {routes<<item.first; for(int32_t v:item.second) routes<<' '<<v; routes<<'\n';}
            require(bool(routes), "baseline route write");
        }
        if(std::getenv("ESE_LIFECYCLE_BASELINE_ONLY")) {
            llama_free(ctx); ctx=nullptr; llama_free_model(model); model=nullptr; llama_backend_free();
            std::puts("lifecycle_baseline: passed"); return 0;
        }
        if(std::getenv("ESE_LIFECYCLE_RETIER")) {
            restore_full();
            for (int i=0; i<prefix+1; ++i) decode(ctx,model,capture,source[(i+7)%source.size()],i,1);
            for (int seq : {0,1}) llama_kv_cache_seq_div(ctx,seq,0,-1,2);
            const auto base=save_full(ctx);
            const llama_pos starts[2]={llama_kv_cache_seq_pos_max(ctx,0)+1,llama_kv_cache_seq_pos_max(ctx,1)+1};
            const llama_token next[2]={tokens[prefix],source[(prefix+8)%source.size()]};
            const auto original0=decode(ctx,model,capture,next[0],starts[0],0);
            const auto original1=decode(ctx,model,capture,next[1],starts[1],1);
            require(llama_state_set_data(ctx,base.data(),base.size())==base.size(),"retier reference reset");
            const int requested=std::getenv("ESE_LIFECYCLE_RETIER_MODE")
                ? std::stoi(std::getenv("ESE_LIFECYCLE_RETIER_MODE")) : LLAMA_SPEC_CKPT_GPU_FALLBACK;
            require(requested==LLAMA_SPEC_CKPT_PER_STEP || requested==LLAMA_SPEC_CKPT_GPU_FALLBACK,"retier checkpoint mode");
            require(llama_spec_ckpt_init(ctx,requested,2)==requested,"retier checkpoint startup");
            const uint32_t layers=llama_kv_cache_layer_count(ctx);
            const auto select_type=[](const char * variable,ggml_type fallback) {
                const char * requested=std::getenv(variable);
                if(!requested)return fallback;
                for(auto type:{GGML_TYPE_F16,GGML_TYPE_F32,GGML_TYPE_BF16,GGML_TYPE_Q4_0,
                        GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0})
                    if(std::strcmp(requested,ggml_type_name(type))==0)return type;
                throw std::runtime_error("invalid retier probe format");
            };
            const ggml_type key=select_type("ESE_LIFECYCLE_RETIER_KEY",
                    std::getenv("ESE_LIFECYCLE_RETIER_Q8") ? GGML_TYPE_Q8_0 : GGML_TYPE_F16);
            const ggml_type value=select_type("ESE_LIFECYCLE_RETIER_VALUE",GGML_TYPE_F16);
            std::vector<ggml_type> keys(layers,key),values(layers,value),invalid_values(layers,GGML_TYPE_Q8_0);
            if (requested==LLAMA_SPEC_CKPT_PER_STEP) {
                require(llama_spec_ckpt_save(ctx,0),"retier active window save");
                const auto active=save_full(ctx);
                require(!llama_kv_cache_prepare_retier(ctx,keys.data(),values.data(),layers),"retier accepted active window");
                require(save_full(ctx)==active && ctx->kv_self.ckpt.qwen4exp_prefix_active[0],"active retier rejection changed state");
                llama_spec_ckpt_discard_seq(ctx,0);
            } else {
                require(llama_spec_ckpt_save(ctx,0) && llama_spec_ckpt_save(ctx,1),"retier fallback checkpoints");
            }
            const auto before=save_full(ctx);
            save_state_artifact("retier-before-0.bin",save_seq(ctx,0));
            save_state_artifact("retier-before-1.bin",save_seq(ctx,1));
            require(!llama_kv_cache_prepare_retier(ctx,keys.data(),invalid_values.data(),layers),"retier accepted non-Flash quantized values");
            require(save_full(ctx)==before,"invalid value tier changed state");
            int64_t state_rows=0,primary_rows=0;
            for(uint32_t layer=0;layer<layers;++layer) {
                if(ctx->kv_self.s_l[layer]) state_rows+=ctx->kv_self.s_l[layer]->ne[1];
                if(const auto * cache_key=ctx->kv_self.k_l[layer]) {
                    require(cache_key->ne[2]==1 && cache_key->ne[3]==1 && cache_key->ne[1]%ctx->kv_self.size==0,
                            "retier failure-count key geometry");
                    const auto heads=cache_key->ne[1]/ctx->kv_self.size;
                    for(const auto & cell:ctx->kv_self.cells) if(cell.pos>=0)
                        primary_rows+=heads*(ctx->kv_self.v_l[layer]?2:1);
                }
            }
            for(const int64_t failure_row:{int64_t(0),state_rows+1,state_rows+primary_rows+1}) {
                const auto injected=std::to_string(failure_row);
                setenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS",injected.c_str(),1);
                const bool failed=llama_kv_cache_retier(ctx,keys.data(),values.data(),layers);
                unsetenv("ESE_TURBO_RETIER_FAIL_AFTER_ROWS");
                require(!failed && save_full(ctx)==before,"failed hybrid migration changed state");
                std::printf("retier_failure: row=%lld full_state_exact=true\n",(long long)failure_row);std::fflush(stdout);
            }
            auto * transaction=llama_kv_cache_prepare_retier(ctx,keys.data(),values.data(),layers);
            require(transaction,"hybrid retier prepare");
            require(llama_kv_cache_transaction_publish(transaction),"hybrid retier publish");
            save_state_artifact("retier-published-0.bin",save_seq(ctx,0));
            save_state_artifact("retier-published-1.bin",save_seq(ctx,1));
            require(llama_kv_cache_transaction_rollback(transaction),"hybrid retier rollback");
            llama_kv_cache_transaction_free(transaction);
            require(save_full(ctx)==before,"hybrid rollback changed full state");
            if(requested==LLAMA_SPEC_CKPT_GPU_FALLBACK)
                require(ctx->kv_self.ckpt.sequences[0].saved && ctx->kv_self.ckpt.sequences[1].saved,
                        "hybrid rollback discarded checkpoints");
            if(requested==LLAMA_SPEC_CKPT_GPU_FALLBACK) {
                decode(ctx,model,capture,next[0],starts[0],0);decode(ctx,model,capture,next[1],starts[1],1);
                require(llama_spec_ckpt_restore_ex(ctx,0,starts[0],-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED &&
                        llama_spec_ckpt_restore_ex(ctx,1,starts[1],-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                        "hybrid rollback retained checkpoint payloads");
            }
            compare("retier_rollback_target",original0,decode(ctx,model,capture,next[0],starts[0],0));
            compare("retier_rollback_other",original1,decode(ctx,model,capture,next[1],starts[1],1));
            require(llama_state_set_data(ctx,base.data(),base.size())==base.size(),"retier commit reset");
            if(requested==LLAMA_SPEC_CKPT_GPU_FALLBACK)
                require(llama_spec_ckpt_save(ctx,0) && llama_spec_ckpt_save(ctx,1),"retier commit old checkpoints");
            require(llama_kv_cache_retier(ctx,keys.data(),values.data(),layers),"hybrid retier commit");
            require(ctx->kv_self.ckpt.selected_spec_mode==requested && ctx->kv_self.ckpt.allocated &&
                    !ctx->kv_self.ckpt.sequences[0].saved && !ctx->kv_self.ckpt.sequences[1].saved,
                    "hybrid commit did not retain usable banks and invalidate records");
            std::vector<llama_kv_cache_layer_types> types(layers);
            require(llama_kv_cache_get_layer_types(ctx,types.data(),layers),"retier types");
            for(const auto & type:types) {
                require(type.type_k==GGML_TYPE_COUNT || type.type_k==key,"retier key type");
                require(type.type_v==GGML_TYPE_COUNT || type.type_v==value,"retier value type");
            }
            save_state_artifact("retier-committed-0.bin",save_seq(ctx,0));
            save_state_artifact("retier-committed-1.bin",save_seq(ctx,1));
            const auto converted=save_full(ctx);
            const auto expected0=decode(ctx,model,capture,next[0],starts[0],0);
            const auto expected1=decode(ctx,model,capture,next[1],starts[1],1);
            require(llama_state_set_data(ctx,converted.data(),converted.size())==converted.size(),"retier converted reset");
            require(llama_spec_ckpt_save(ctx,0) && llama_spec_ckpt_save(ctx,1),"retier next windows reuse startup banks");
            decode(ctx,model,capture,next[0],starts[0],0);decode(ctx,model,capture,next[1],starts[1],1);
            const auto restore_kind=requested==LLAMA_SPEC_CKPT_PER_STEP ? LLAMA_SPEC_CKPT_RESTORE_DIRECT
                : LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED;
            require(llama_spec_ckpt_restore_ex(ctx,0,starts[0],-1)==restore_kind &&
                    llama_spec_ckpt_restore_ex(ctx,1,starts[1],-1)==restore_kind,"retier new checkpoints restore");
            compare("retier_committed_target_checkpoint",expected0,decode(ctx,model,capture,next[0],starts[0],0));
            compare("retier_committed_other_checkpoint",expected1,decode(ctx,model,capture,next[1],starts[1],1));
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::printf("hybrid_retier_result: passed key=%s value=%s checkpoint_mode=%d\n",
                    ggml_type_name(key),ggml_type_name(value),requested);return 0;
        }
        if(std::getenv("ESE_LIFECYCLE_DEFRAG")) {
            restore_full();
            for (int i=0; i<prefix+1; ++i)
                decode(ctx,model,capture,source[(i+7)%source.size()],i,1);
            const std::string mode = std::getenv("ESE_LIFECYCLE_DEFRAG_TRANSFORM")
                ? std::getenv("ESE_LIFECYCLE_DEFRAG_TRANSFORM") : "divide";
            require(mode=="none" || mode=="divide" || mode=="partial_shift" || mode=="shift", "defrag transform mode");
            for (int seq : {0,1}) {
                if (mode=="divide") llama_kv_cache_seq_div(ctx,seq,0,-1,2);
                else if (mode=="shift") llama_kv_cache_seq_add(ctx,seq,0,-1,3);
                else if (mode=="partial_shift") llama_kv_cache_seq_add(ctx,seq,0,prefix/2,prefix);
            }
            const llama_pos first = mode=="shift" ? 3 : mode=="partial_shift" ? prefix : 0;
            const auto base = save_full(ctx);
            require(llama_kv_cache_seq_rm(ctx,0,first,first+1), "defrag create holes");
            const llama_pos starts[2] = {llama_kv_cache_seq_pos_max(ctx,0)+1,llama_kv_cache_seq_pos_max(ctx,1)+1};
            const llama_token next[2] = {tokens[prefix],source[(prefix+8)%source.size()]};
            const auto state0 = save_seq(ctx,0), state1 = save_seq(ctx,1);
            save_state_artifact("defrag-before-0.bin",state0);
            save_state_artifact("defrag-before-1.bin",state1);
            const auto expected0 = decode(ctx,model,capture,next[0],starts[0],0);
            const auto expected1 = decode(ctx,model,capture,next[1],starts[1],1);
            require(llama_state_set_data(ctx,base.data(),base.size())==base.size(), "defrag base restore");
            require(llama_kv_cache_seq_rm(ctx,0,first,first+1), "defrag recreate holes after restore");
            require(llama_spec_ckpt_init(ctx,LLAMA_SPEC_CKPT_GPU_FALLBACK,4)==LLAMA_SPEC_CKPT_GPU_FALLBACK,
                    "defrag checkpoint mode");
            require(llama_spec_ckpt_save(ctx,0) && llama_spec_ckpt_save(ctx,1), "defrag checkpoint save");
            const int old_src[2] = {ctx->kv_self.cells[0].src,ctx->kv_self.cells[1].src};
            require(old_src[0]==0 && old_src[1]==1,"defrag initial recurrent owners");
            const auto before_cells = ctx->kv_self.cells;
            llama_kv_cache_defrag(ctx);
            require(llama_kv_cache_update(ctx)==0,"defrag cache update");
            llama_synchronize(ctx);
            bool moved=false;
            for (size_t i=0; i<before_cells.size(); ++i)
                moved |= before_cells[i].pos!=ctx->kv_self.cells[i].pos ||
                         before_cells[i].qwen4exp_order!=ctx->kv_self.cells[i].qwen4exp_order ||
                         before_cells[i].seq_id!=ctx->kv_self.cells[i].seq_id;
            std::printf("defrag_owners: moved=%s before=[%d,%d] after=[%d,%d]\n",moved?"true":"false",old_src[0],old_src[1],
                        ctx->kv_self.cells[0].src,ctx->kv_self.cells[1].src);std::fflush(stdout);
            require(moved,"defrag did not move attention rows");
            require(ctx->kv_self.cells[0].src==old_src[0] && ctx->kv_self.cells[1].src==old_src[1],
                    "defrag changed recurrent sequence owners");
            save_state_artifact("defrag-after-0.bin",save_seq(ctx,0));
            save_state_artifact("defrag-after-1.bin",save_seq(ctx,1));
            compare("defrag_target_continuation",expected0,decode(ctx,model,capture,next[0],starts[0],0));
            compare("defrag_other_continuation",expected1,decode(ctx,model,capture,next[1],starts[1],1));
            require(llama_spec_ckpt_restore_ex(ctx,0,starts[0],-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                    "defrag target checkpoint recovery");
            require(llama_spec_ckpt_restore_ex(ctx,1,starts[1],-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                    "defrag other checkpoint recovery");
            save_state_artifact("defrag-checkpoint-0.bin",save_seq(ctx,0));
            save_state_artifact("defrag-checkpoint-1.bin",save_seq(ctx,1));
            compare("defrag_target_checkpoint_continuation",expected0,decode(ctx,model,capture,next[0],starts[0],0));
            compare("defrag_other_checkpoint_continuation",expected1,decode(ctx,model,capture,next[1],starts[1],1));
            require(llama_spec_ckpt_restore_ex(ctx,0,starts[0],-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                    "defrag copy source recovery");
            require(llama_kv_cache_seq_rm(ctx,1,0,-1),"defrag copy destination clear");
            llama_kv_cache_seq_cp(ctx,0,1,0,-1);
            compare("defrag_copy_continuation",expected0,decode(ctx,model,capture,next[0],starts[0],1));
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::printf("defrag_lifecycle_result: passed mode=%s\n",mode.c_str());return 0;
        }
        if(std::getenv("ESE_LIFECYCLE_PREFIX_BANKS")) {
            require(params.n_parallel==2,"bank probe requires two sequences");
            restore_full();
            std::vector<float> base_last[2];
            const auto * initial_last=llama_get_logits_ith(ctx,-1);
            base_last[0].assign(initial_last,initial_last+llama_n_vocab(model));
            std::vector<llama_token> stream1(prefix+10);
            for(size_t i=0;i<stream1.size();++i) stream1[i]=source[(i+7)%source.size()];
            for(int i=0;i<prefix+1;++i) decode(ctx,model,capture,stream1[i],i,1);
            const auto * other_last=llama_get_logits_ith(ctx,-1);
            base_last[1].assign(other_last,other_last+llama_n_vocab(model));
            require(base_last[0]!=base_last[1],"bank oracle streams are degenerate");
            const char * bank_transform = std::getenv("ESE_LIFECYCLE_BANK_TRANSFORM");
            if (bank_transform) {
                const std::string mode(bank_transform);
                require(mode == "shift" || mode == "divide" || mode == "partial_shift", "bank transform mode");
                for (int seq : {0, 1}) {
                    if (mode == "shift") llama_kv_cache_seq_add(ctx, seq, 0, -1, 3);
                    else if (mode == "divide") llama_kv_cache_seq_div(ctx, seq, 0, -1, 2);
                    else llama_kv_cache_seq_add(ctx, seq, 0, prefix / 2, prefix);
                }
            }
            const llama_pos bank_start[2] = {llama_kv_cache_seq_pos_max(ctx, 0) + 1,
                                            llama_kv_cache_seq_pos_max(ctx, 1) + 1};
            require(llama_spec_ckpt_init(ctx,LLAMA_SPEC_CKPT_PER_STEP,8)==LLAMA_SPEC_CKPT_PER_STEP,"bank startup capacity");
            const auto bank_base=save_full(ctx);
            auto reset_bank=[&] {require(llama_state_set_data(ctx,bank_base.data(),bank_base.size())==bank_base.size(),"bank full-session restore");};
            auto input=[&](int seq,int index){return seq?stream1[prefix+1+index]:tokens[prefix+index];};
            auto position=[&](int seq,int index){return bank_start[seq]+index;};
            auto verify_rows=[&](int seq,int start,int count){
                auto batch=llama_batch_init(count,0,1);batch.n_tokens=count;
                for(int i=0;i<count;++i) {
                    batch.token[i]=input(seq,start+i);batch.pos[i]=position(seq,start+i);
                    batch.n_seq_id[i]=1;batch.seq_id[i][0]=seq;batch.logits[i]=true;
                }
                const int result=llama_decode(ctx,batch);llama_batch_free(batch);llama_synchronize(ctx);
                require(result==0,"bank verification call");
            };
            auto same_moe=[&](const observation & a,const observation & b){
                for(const auto & item:a.routes) if(is_layer_tensor(item.first.c_str(),"ffn_moe_topk")) {
                    const auto found=b.routes.find(item.first);
                    if(found==b.routes.end() || found->second!=item.second)return false;
                }
                return true;
            };
            int cases=0;
            for(int target:{0,1}) for(int kept=0;kept<=4;++kept) {
                if(const char * value=std::getenv("ESE_LIFECYCLE_BANK_TARGET")) if(target!=std::stoi(value))continue;
                if(const char * value=std::getenv("ESE_LIFECYCLE_BANK_ACCEPTED")) if(kept!=std::stoi(value))continue;
                const int other=1-target;
                reset_bank();
                for(int i=0;i<kept;++i) decode(ctx,model,capture,input(target,i),position(target,i),target);
                const auto target_state=save_seq(ctx,target);
                std::vector<float> target_last=base_last[target];
                if(kept) {const auto * last=llama_get_logits_ith(ctx,-1);target_last.assign(last,last+llama_n_vocab(model));}
                const auto target_next=decode(ctx,model,capture,input(target,kept),position(target,kept),target);
                reset_bank();
                for(int i=0;i<2;++i) decode(ctx,model,capture,input(other,i),position(other,i),other);
                const auto other_state=save_seq(ctx,other);
                const auto * last=llama_get_logits_ith(ctx,-1);
                const std::vector<float> other_last_expected(last,last+llama_n_vocab(model));
                const auto other_next=decode(ctx,model,capture,input(other,2),position(other,2),other);
                if(std::getenv("ESE_LIFECYCLE_BANK_TRACE"))
                    dump_floats(capture,"other-reference");
                reset_bank();
                // Save sequence0 while sequence1 owns the global last output.
                require(llama_spec_ckpt_save(ctx,0),"bank sequence0 save uses owned output");
                require(llama_spec_ckpt_save(ctx,1),"bank sequence1 save uses owned output");
                verify_rows(target,0,2);
                require(llama_spec_ckpt_init(ctx,LLAMA_SPEC_CKPT_PER_STEP,8)==LLAMA_SPEC_CKPT_PER_STEP,"live bank mode reselection");
                verify_rows(other,0,2);verify_rows(target,2,2);verify_rows(other,2,1);
                const auto other_before=save_seq(ctx,other);
                require(llama_spec_ckpt_restore_ex(ctx,target,bank_start[target],kept-1)==LLAMA_SPEC_CKPT_RESTORE_DIRECT,"bank target direct restore");
                require(save_seq(ctx,target)==target_state && save_seq(ctx,other)==other_before,"bank target restore changed another sequence");
                require(std::memcmp(llama_get_logits_ith(ctx,-1),target_last.data(),target_last.size()*sizeof(float))==0,"bank target output ownership");
                require(llama_spec_ckpt_save(ctx,target),"restored target can start next window");
                require(llama_spec_ckpt_restore_ex(ctx,target,position(target,kept),-1)==LLAMA_SPEC_CKPT_RESTORE_DIRECT,
                        "restored target empty next window");
                require(save_seq(ctx,target)==target_state && save_seq(ctx,other)==other_before,
                        "empty next window changed sequence state");
                require(std::memcmp(llama_get_logits_ith(ctx,-1),target_last.data(),target_last.size()*sizeof(float))==0,
                        "empty next window changed restored output");
                llama_spec_ckpt_discard_seq(ctx,target);
                require(llama_spec_ckpt_restore_ex(ctx,target,bank_start[target],-1)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"independent target discard");
                require(llama_spec_ckpt_restore_ex(ctx,other,bank_start[other],1)==LLAMA_SPEC_CKPT_RESTORE_DIRECT,"bank other window survived discard");
                const auto restored_target_state = save_seq(ctx,target);
                const auto restored_other_state = save_seq(ctx,other);
                if (restored_target_state != target_state || restored_other_state != other_state) {
                    save_state_artifact("bank-target-expected.bin", target_state);
                    save_state_artifact("bank-target-restored.bin", restored_target_state);
                    save_state_artifact("bank-other-expected.bin", other_state);
                    save_state_artifact("bank-other-restored.bin", restored_other_state);
                    std::printf("bank_restore_state: target=%d accepted=%d target_equal=%s other_equal=%s\n", target, kept,
                                restored_target_state==target_state?"true":"false", restored_other_state==other_state?"true":"false");
                    std::fflush(stdout);
                }
                require(restored_target_state==target_state,"bank other restore changed target");
                require(restored_other_state==other_state,"bank other accepted-prefix state mismatch");
                require(std::memcmp(llama_get_logits_ith(ctx,-1),other_last_expected.data(),other_last_expected.size()*sizeof(float))==0,"bank other output ownership");
                const auto actual_target_next=decode(ctx,model,capture,input(target,kept),position(target,kept),target);
                const auto other_after_target=save_seq(ctx,other);
                size_t first_other_change=0;
                while(first_other_change<std::min(other_state.size(),other_after_target.size()) &&
                      other_state[first_other_change]==other_after_target[first_other_change])++first_other_change;
                std::printf("bank_continuation_state: {\"target\":%d,\"accepted\":%d,\"other_state_unchanged\":%s,\"first_other_change\":%zu}\n",
                            target,kept,other_state==other_after_target?"true":"false",first_other_change);std::fflush(stdout);
                const auto actual_other_next=decode(ctx,model,capture,input(other,2),position(other,2),other);
                if(std::getenv("ESE_LIFECYCLE_BANK_TRACE"))
                    dump_floats(capture,"other-actual");
                if(std::getenv("ESE_LIFECYCLE_BANK_TARGET")) {
                    for(const auto & item:std::vector<std::pair<const char *,const observation *>>{
                        {"target-reference",&target_next},{"target-actual",&actual_target_next},
                        {"other-reference",&other_next},{"other-actual",&actual_other_next}}) {
                        const char * directory=std::getenv("ESE_LIFECYCLE_DIRECTORY");
                        require(directory,"focused bank artifact directory");
                        std::ofstream output(std::string(directory)+"/"+item.first+".logits.bin",std::ios::binary);
                        output.write(reinterpret_cast<const char *>(item.second->logits.data()),item.second->logits.size()*sizeof(float));
                        require(bool(output),"focused bank logits write");
                    }
                    compare("bank_target_continuation",actual_target_next,target_next,false);
                    compare("bank_other_continuation",actual_other_next,other_next,false);
                }
                require(actual_target_next.logits==target_next.logits && same_moe(actual_target_next,target_next),"bank target continuation");
                require(actual_other_next.logits==other_next.logits && same_moe(actual_other_next,other_next),"bank other continuation");
                const bool logical_qsa = std::getenv("ESE_LIFECYCLE_BANK_LOGICAL_QSA") != nullptr;
                if (logical_qsa) {
                    require(actual_target_next.routes==target_next.routes,"bank target logical QSA continuation");
                    require(actual_other_next.routes==other_next.routes,"bank other logical QSA continuation");
                }
                llama_spec_ckpt_discard_seq(ctx,other);++cases;
                std::printf("prefix_bank: {\"target\":%d,\"accepted\":%d,\"state_exact\":true,\"logits_exact\":true,\"continuation_exact\":true,\"independent_discard\":true,\"logical_qsa_checked\":%s}\n",target,kept,logical_qsa?"true":"false");std::fflush(stdout);
            }
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::printf("prefix_bank_result: qualified cases=%d\n",cases);return 0;
        }
        if(std::getenv("ESE_LIFECYCLE_ABORT_CHECKPOINT")) {
            require(direct_recovery && params.n_ubatch>0 && params.n_ubatch<=4,"abort checkpoint geometry");
            const int completed=params.n_ubatch;
            for(int kept=0;kept<=completed;++kept) {
                restore_full();
                for(int i=0;i<kept;++i) decode(ctx,model,capture,tokens[prefix+i],prefix+i,0);
                const auto expected_prefix=save_seq(ctx,0);
                const auto * last=llama_get_logits_ith(ctx,-1);require(last,"abort teacher last output");
                const std::vector<float> expected_last(last,last+llama_n_vocab(model));
                const auto expected_next=decode(ctx,model,capture,tokens[prefix+kept],prefix+kept,0);
                restore_full();
                require(llama_spec_ckpt_init(ctx,LLAMA_SPEC_CKPT_PER_STEP,8)==LLAMA_SPEC_CKPT_PER_STEP,"abort checkpoint capacity");
                require(llama_spec_ckpt_save(ctx,0),"abort checkpoint save");
                auto proposed=llama_batch_init(8,0,1);proposed.n_tokens=8;
                for(int i=0;i<8;++i) {
                    proposed.token[i]=tokens[prefix+i];proposed.pos[i]=prefix+i;
                    proposed.n_seq_id[i]=1;proposed.seq_id[i][0]=0;proposed.logits[i]=true;
                }
                capture.observed_routes=0;capture.arm_after_routes=65;capture.abort_polls=0;
                capture.abort_armed.store(false,std::memory_order_relaxed);
                llama_set_abort_callback(ctx,abort_when_armed,&capture);
                const int status=llama_decode(ctx,proposed);llama_batch_free(proposed);
                llama_set_abort_callback(ctx,nullptr,nullptr);llama_synchronize(ctx);
                capture.arm_after_routes=-1;
                require(status!=0 && capture.abort_armed.load() && capture.abort_polls>0,"later-chunk abort not exercised");
                const auto interrupted=save_seq(ctx,0);
                require(llama_spec_ckpt_restore_ex(ctx,0,prefix,completed)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"interrupted row became available");
                require(interrupted==save_seq(ctx,0),"failed interrupted-row restore mutated sequence");
                require(llama_spec_ckpt_restore_ex(ctx,0,prefix,kept-1)==LLAMA_SPEC_CKPT_RESTORE_DIRECT,"completed-prefix abort restore");
                const bool state_exact=save_seq(ctx,0)==expected_prefix;
                const bool last_exact=std::memcmp(llama_get_logits_ith(ctx,-1),expected_last.data(),expected_last.size()*sizeof(float))==0;
                const auto next=decode(ctx,model,capture,tokens[prefix+kept],prefix+kept,0);
                const bool continuation_exact=next.logits==expected_next.logits && next.routes==expected_next.routes;
                require(state_exact && last_exact && continuation_exact,"abort recovery differs from teacher");
                std::printf("prefix_abort: {\"microbatch\":%d,\"completed_inputs\":%d,\"accepted\":%d,\"status\":%d,\"route_callbacks\":%d,\"abort_polls\":%d,\"state_exact\":true,\"logits_exact\":true,\"continuation_exact\":true}\n",
                        params.n_ubatch,completed,kept,status,capture.observed_routes,capture.abort_polls);
                std::fflush(stdout);llama_spec_ckpt_discard_seq(ctx,0);
            }
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::puts("prefix_abort_result: qualified");return 0;
        }
        if(std::getenv("ESE_LIFECYCLE_BATCH_DIAGNOSTIC")) {
            // Diagnostic only: compare true verifier batches to the one-token
            // target path. A disagreement closes qualification, not generation.
            bool exact=true;
            for(int depth : {1,2,4,8}) {
                const char * selected_depth=std::getenv("ESE_LIFECYCLE_BATCH_DEPTH");
                if(selected_depth && depth!=1 && depth!=std::stoi(selected_depth)) continue;
                restore_full();
                std::vector<observation> reference;
                for(int i=0;i<depth;i++) {
                    reference.push_back(decode(ctx,model,capture,tokens[prefix+i],prefix+i,0));
                    dump_floats(capture,"ref-d"+std::to_string(depth)+"-t"+std::to_string(i));
                }
                auto reference_state=save_seq(ctx,0);
                auto reference_next=decode(ctx,model,capture,tokens[prefix+depth],prefix+depth,0);
                restore_full();capture.tensors.clear();capture.floats.clear(); capture.float_order.clear();capture.attention.clear();capture.invalid_type=false;
                auto verify=llama_batch_init(depth,0,1);verify.n_tokens=depth;
                for(int i=0;i<depth;i++) {
                    verify.token[i]=tokens[prefix+i];verify.pos[i]=prefix+i;
                    verify.n_seq_id[i]=1;verify.seq_id[i][0]=0;verify.logits[i]=true;
                }
                int result=llama_decode(ctx,verify);llama_batch_free(verify);llama_synchronize(ctx);
                require(result==0 && !capture.invalid_type && capture.tensors.size()==60,"batched verifier outputs/routes");
                dump_floats(capture,"batch-d"+std::to_string(depth));
                for(int i=0;i<depth;i++) {
                    const float * logits=llama_get_logits_ith(ctx,i);require(logits,"batched verifier logits");
                    size_t changed=0,route_changed=0;double maximum=0, squared=0, norm=0;
                    const auto & ref=reference[i];
                    if(const char * root=std::getenv("ESE_LIFECYCLE_BATCH_LOGITS_DIRECTORY")) {
                        auto directory=std::filesystem::path(root)/("depth-"+std::to_string(depth));
                        std::filesystem::create_directories(directory);
                        for(bool batched:{false,true}) {
                            auto name=std::string(batched?"batch-":"reference-")+std::to_string(i)+".bin";
                            std::ofstream data(directory/name,std::ios::binary);
                            data.write(reinterpret_cast<const char *>(batched?logits:ref.logits.data()),ref.logits.size()*sizeof(float));
                            require(bool(data),"batch logit dump write");
                        }
                    }
                    for(size_t j=0;j<ref.logits.size();j++) {
                        require(std::isfinite(logits[j]),"nonfinite batched logits");
                        const double delta=double(logits[j])-ref.logits[j];
                        changed+=std::memcmp(logits+j,ref.logits.data()+j,sizeof(float))!=0;
                        maximum=std::max(maximum,std::fabs(delta));squared+=delta*delta;norm+=double(ref.logits[j])*ref.logits[j];
                    }
                    for(const auto & item:ref.routes) {
                        if(!is_layer_tensor(item.first.c_str(),"ffn_moe_topk")) continue;
                        auto found=capture.tensors.find(item.first);
                        require(found!=capture.tensors.end() && found->second.size()==item.second.size()*depth,"batched MoE route geometry");
                        for(size_t j=0;j<item.second.size();j++) route_changed+=item.second[j]!=found->second[i*item.second.size()+j];
                    }
                    const bool top1=std::max_element(logits,logits+ref.logits.size())-logits ==
                        std::max_element(ref.logits.begin(),ref.logits.end())-ref.logits.begin();
                    std::printf("batch_verifier: {\"depth\":%d,\"token\":%d,\"changed_logits\":%zu,\"max_abs\":%.9g,\"nrmse\":%.9g,\"changed_moe_positions\":%zu,\"top1_equal\":%s}\n",
                            depth,i,changed,maximum,std::sqrt(squared/std::max(norm,1e-30)),route_changed,top1?"true":"false");
                    exact &= changed==0 && route_changed==0;std::fflush(stdout);
                }
                auto batch_state=save_seq(ctx,0);
                auto batch_next=decode(ctx,model,capture,tokens[prefix+depth],prefix+depth,0);
                const bool state_exact=reference_state==batch_state;
                const bool continuation_exact=reference_next.logits==batch_next.logits && reference_next.routes==batch_next.routes;
                std::printf("batch_state: {\"depth\":%d,\"reference_bytes\":%zu,\"batch_bytes\":%zu,\"state_exact\":%s,\"continuation_exact\":%s}\n",
                        depth,reference_state.size(),batch_state.size(),state_exact?"true":"false",continuation_exact?"true":"false");
                exact &= state_exact && continuation_exact;std::fflush(stdout);
                if(std::getenv("ESE_LIFECYCLE_BATCH_RECOVERY")) {
                    require(state_exact && continuation_exact,"recovery requires a qualified batch");
                    const int requested=std::getenv("ESE_LIFECYCLE_SPEC_MODE")
                        ? std::stoi(std::getenv("ESE_LIFECYCLE_SPEC_MODE")) : LLAMA_SPEC_CKPT_CPU;
                    require(requested==LLAMA_SPEC_CKPT_PER_STEP || requested==LLAMA_SPEC_CKPT_CPU || requested==LLAMA_SPEC_CKPT_GPU_FALLBACK,
                            "batch recovery checkpoint mode");
                    require(llama_spec_ckpt_init(ctx,requested,direct_recovery?8:depth)==requested,"batch recovery mode selection");
                    // Bound test-owned snapshots: release final-state comparisons
                    // before constructing one accepted-prefix reference at a time.
                    std::vector<uint8_t>().swap(reference_state);
                    std::vector<uint8_t>().swap(batch_state);
                    for(int accepted=0;accepted<=depth;++accepted) {
                        restore_full();
                        for(int i=0;i<accepted;++i) decode(ctx,model,capture,tokens[prefix+i],prefix+i,0);
                        const auto expected_prefix=save_seq(ctx,0);
                        const auto & expected_next=accepted<depth?reference[accepted]:reference_next;
                        restore_full();
                        if (direct_recovery) require(llama_spec_ckpt_init(ctx,requested,8)==requested,"direct window mode reselection");
                        require(llama_spec_ckpt_save(ctx,0),"true batch checkpoint save");
                        auto proposed=llama_batch_init(depth,0,1);proposed.n_tokens=depth;
                        for(int i=0;i<depth;++i) {
                            llama_token token=tokens[prefix+i];
                            if(i>=accepted) {
                                token=source[(prefix+i+17+accepted)%source.size()];
                                if(token==tokens[prefix+i]) token=(token+1)%llama_n_vocab(model);
                                if(i==depth-1 && llama_token_eos(model)>=0) token=llama_token_eos(model);
                            }
                            proposed.token[i]=token;proposed.pos[i]=prefix+i;
                            proposed.n_seq_id[i]=1;proposed.seq_id[i][0]=0;proposed.logits[i]=true;
                        }
                        const int call_chunk=std::getenv("ESE_LIFECYCLE_VERIFY_CALL_CHUNK")
                            ? std::stoi(std::getenv("ESE_LIFECYCLE_VERIFY_CALL_CHUNK")) : depth;
                        require(call_chunk>0 && (!std::getenv("ESE_LIFECYCLE_VERIFY_CALL_CHUNK") || direct_recovery),
                                "verification call chunk requires direct recovery");
                        size_t accepted_changed=0,accepted_routes_changed=0;
                        int verify_calls=0;
                        const auto verify_start=std::chrono::steady_clock::now();
                        for(int offset=0;offset<depth;offset+=call_chunk) {
                            auto chunk=proposed;chunk.n_tokens=std::min(call_chunk,depth-offset);
                            chunk.token+=offset;chunk.pos+=offset;chunk.n_seq_id+=offset;chunk.seq_id+=offset;chunk.logits+=offset;
                            capture.tensors.clear();capture.floats.clear();capture.float_order.clear();
                            capture.attention.clear();capture.invalid_type=false;
                            const int verified=llama_decode(ctx,chunk);llama_synchronize(ctx);++verify_calls;
                            require(verified==0 && !capture.invalid_type && capture.tensors.size()==60,"poisoned verification chunk");
                            // Compare each call's outputs before the next call overwrites them.
                            for(int i=0;i<chunk.n_tokens && offset+i<accepted;++i) {
                                const float * logits=llama_get_logits_ith(ctx,i);require(logits,"accepted chunk logits");
                                for(size_t j=0;j<reference[offset+i].logits.size();++j) {
                                    require(std::isfinite(logits[j]),"finite accepted logit");
                                    accepted_changed+=std::memcmp(logits+j,reference[offset+i].logits.data()+j,4)!=0;
                                }
                                for(const auto & item:reference[offset+i].routes) {
                                    if(!is_layer_tensor(item.first.c_str(),"ffn_moe_topk"))continue;
                                    auto found=capture.tensors.find(item.first);
                                    require(found!=capture.tensors.end() && found->second.size()==item.second.size()*chunk.n_tokens,"accepted chunk route geometry");
                                    for(size_t j=0;j<item.second.size();++j)
                                        accepted_routes_changed+=item.second[j]!=found->second[i*item.second.size()+j];
                                }
                            }
                        }
                        llama_batch_free(proposed);
                        const auto verify_end=std::chrono::steady_clock::now();
                        if (direct_recovery) {
                            const auto before_invalid=save_seq(ctx,0);
                            require(llama_spec_ckpt_restore_ex(ctx,1,prefix,accepted-1)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"wrong journal sequence");
                            require(llama_spec_ckpt_restore_ex(ctx,0,prefix+1,accepted-1)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"wrong journal base");
                            require(llama_spec_ckpt_restore_ex(ctx,0,prefix,depth)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"unwritten journal row");
                            require(llama_spec_ckpt_restore_ex(ctx,0,prefix,-2)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"invalid negative journal row");
                            require(before_invalid==save_seq(ctx,0),"invalid journal restore mutated sequence state");
                        }
                        const auto restore_start=std::chrono::steady_clock::now();
                        const auto restored=llama_spec_ckpt_restore_ex(ctx,0,prefix,accepted-1);
                        const auto restore_end=std::chrono::steady_clock::now();
                        require(restored==(direct_recovery?LLAMA_SPEC_CKPT_RESTORE_DIRECT:LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED),"true batch restore protocol");
                        if (direct_recovery && accepted>0) {
                            const auto * last=llama_get_logits_ith(ctx,-1);require(last,"directly retained accepted output");
                            require(std::memcmp(last,reference[accepted-1].logits.data(),reference[accepted-1].logits.size()*sizeof(float))==0,
                                    "directly retained accepted output differs");
                        }
                        const int replay_policy=std::getenv("ESE_LIFECYCLE_BATCH_REPLAY")
                            ? std::stoi(std::getenv("ESE_LIFECYCLE_BATCH_REPLAY")) : 0;
                        require(replay_policy>=0 && replay_policy<=2,"accepted replay policy");
                        if(!direct_recovery && replay_policy>0 && accepted>0) {
                            auto replay=llama_batch_init(accepted,0,1);replay.n_tokens=accepted;
                            for(int i=0;i<accepted;++i) {
                                replay.token[i]=tokens[prefix+i];replay.pos[i]=prefix+i;
                                replay.n_seq_id[i]=1;replay.seq_id[i][0]=0;
                                replay.logits[i]=replay_policy==1 || i==accepted-1;
                            }
                            capture.tensors.clear();capture.floats.clear();capture.float_order.clear();
                            capture.attention.clear();capture.invalid_type=false;
                            const int replayed=llama_decode(ctx,replay);
                            llama_batch_free(replay);llama_synchronize(ctx);
                            require(replayed==0 && !capture.invalid_type && capture.tensors.size()==60,"accepted batched replay");
                            const float * last=llama_get_logits_ith(ctx,-1);require(last,"accepted replay last output");
                            size_t changed=0;
                            for(size_t j=0;j<reference[accepted-1].logits.size();++j)
                                changed+=std::memcmp(last+j,reference[accepted-1].logits.data()+j,4)!=0;
                            exact &= changed==0;
                            std::printf("batch_replay_output: {\"mode\":%d,\"depth\":%d,\"accepted\":%d,\"policy\":%d,\"changed_logits\":%zu}\n",requested,depth,accepted,replay_policy,changed);
                            std::fflush(stdout);
                        } else if (!direct_recovery) {
                            for(int i=0;i<accepted;++i) decode(ctx,model,capture,tokens[prefix+i],prefix+i,0);
                        }
                        const auto replay_end=std::chrono::steady_clock::now();
                        const auto recovered_prefix=save_seq(ctx,0);
                        const bool recovered_exact=recovered_prefix==expected_prefix;
                        const auto recovered_next=decode(ctx,model,capture,tokens[prefix+accepted],prefix+accepted,0);
                        const bool recovered_continuation=expected_next.logits==recovered_next.logits &&
                                expected_next.routes==recovered_next.routes;
                        auto ns=[](auto a,auto b){return std::chrono::duration_cast<std::chrono::nanoseconds>(b-a).count();};
                        std::printf("batch_recovery: {\"mode\":%d,\"depth\":%d,\"accepted\":%d,\"poisoned_suffix\":%d,\"replay_policy\":%d,\"verify_calls\":%d,\"accepted_changed_logits\":%zu,\"accepted_changed_moe_positions\":%zu,\"state_exact\":%s,\"continuation_exact\":%s,\"reference_bytes\":%zu,\"recovered_bytes\":%zu,\"snapshot_bytes\":%zu,\"verify_ns\":%lld,\"restore_ns\":%lld,\"replay_ns\":%lld}\n",
                            requested,depth,accepted,depth-accepted,direct_recovery?-1:replay_policy,verify_calls,accepted_changed,accepted_routes_changed,
                            recovered_exact?"true":"false",recovered_continuation?"true":"false",
                            expected_prefix.size(),recovered_prefix.size(),expected_prefix.size()+recovered_prefix.size(),
                            (long long)ns(verify_start,verify_end),(long long)ns(restore_start,restore_end),(long long)ns(restore_end,replay_end));
                        std::fflush(stdout);
                        exact &= accepted_changed==0 && accepted_routes_changed==0 && recovered_exact && recovered_continuation;
                        llama_spec_ckpt_discard_seq(ctx,0);
                        require(llama_spec_ckpt_restore_ex(ctx,0,prefix,-1)==LLAMA_SPEC_CKPT_RESTORE_FAILED,"discarded true batch window restore");
                    }
                }
            }
            llama_free(ctx);ctx=nullptr;llama_free_model(model);model=nullptr;llama_backend_free();
            std::printf("batch_verifier_result: %s\n",exact?"qualified":"unqualified");return exact?0:3;
        }
        restore_full(); compare("full_restore",expected,continuation(0));
        restore_seq(0); compare("sequence_restore",expected,continuation(0));
        restore_seq(1); compare("sequence_migration",expected,continuation(1));
        restore_seq(0); llama_kv_cache_seq_cp(ctx,0,1,0,-1); compare("sequence_copy",expected,continuation(1));
        restore_seq(0); llama_kv_cache_seq_cp(ctx,0,1,0,-1); require(llama_kv_cache_seq_rm(ctx,0,0,-1), "remove source"); compare("copy_remove_source",expected,continuation(1));
        restore_seq(0); llama_kv_cache_seq_cp(ctx,0,1,0,-1); llama_kv_cache_seq_keep(ctx,1); compare("copy_keep_destination",expected,continuation(1));
        llama_kv_cache_clear(ctx); compare("clear_and_replay",expected,replay(0));
        require(llama_kv_cache_seq_rm(ctx,0,0,-1), "sequence remove"); compare("remove_and_replay",expected,replay(0));
        restore_seq(0);
        require(llama_state_seq_set_data(ctx,sequence.data(),sequence.size()-1,0,0)==0, "truncated restore accepted");
        compare("truncated_restore_rollback",expected,continuation(0));
        restore_full(); int abort_count=0; llama_set_abort_callback(ctx,abort_now,&abort_count);
        auto batch=llama_batch_init(1,0,1); batch.n_tokens=1; batch.token[0]=tokens[prefix]; batch.pos[0]=prefix; batch.n_seq_id[0]=1; batch.seq_id[0][0]=0; batch.logits[0]=true;
        int status=llama_decode(ctx,batch); llama_batch_free(batch); llama_set_abort_callback(ctx,nullptr,nullptr); llama_synchronize(ctx);
        std::printf("lifecycle_abort: {\"status\":%d,\"callbacks\":%d}\n",status,abort_count); std::fflush(stdout);
        require(status!=0 && abort_count>0, "abort not exercised"); restore_full(); compare("abort_restore_replay",expected,continuation(0));
        restore_full();
        const int requested_mode=std::getenv("ESE_LIFECYCLE_SPEC_MODE")
            ? std::stoi(std::getenv("ESE_LIFECYCLE_SPEC_MODE")) : LLAMA_SPEC_CKPT_CPU;
        require(requested_mode==LLAMA_SPEC_CKPT_CPU || requested_mode==LLAMA_SPEC_CKPT_GPU_FALLBACK,
                "checkpoint mode must be CPU (3) or GPU fallback (2)");
        const int mode=llama_spec_ckpt_init(ctx,requested_mode,4);
        require(mode==requested_mode,"checkpoint mode selection");
        if (std::getenv("ESE_LIFECYCLE_RESTORE_OWNERSHIP")) {
            for (const std::string kind : {"buffer", "file"}) {
                restore_full();
                for (int i = 0; i < prefix; ++i) decode(ctx, model, capture, tokens[(i + 3) % tokens.size()], i, 1);
                const auto pair_base = save_full(ctx);
                const auto other_control = decode(ctx, model, capture, tokens[4], prefix, 1);
                require(llama_state_set_data(ctx, pair_base.data(), pair_base.size()) == pair_base.size(), "ownership pair reset");
                require(llama_spec_ckpt_save(ctx, 0) && llama_spec_ckpt_save(ctx, 1), "ownership checkpoint save");
                const auto target_base = save_seq(ctx, 0), other_base = save_seq(ctx, 1);
                const auto outputs_and_cache = save_full(ctx);
                save_state_artifact(("ownership-" + kind + "-target-base.bin").c_str(), target_base);
                save_state_artifact(("ownership-" + kind + "-candidate.bin").c_str(), other_base);
                if (kind == "buffer") {
                    require(llama_state_seq_set_data(ctx, other_base.data(), other_base.size() - 1, 0, 0) == 0,
                            "ownership different-prompt truncated buffer accepted");
                } else {
                    const char * root = std::getenv("ESE_LIFECYCLE_DIRECTORY"); require(root, "ownership file directory");
                    const auto good = std::filesystem::path(root) / "ownership-good.session";
                    const auto bad = std::filesystem::path(root) / "ownership-truncated.session";
                    std::vector<llama_token> other_prompt(prefix);
                    for (int i = 0; i < prefix; ++i) other_prompt[i] = tokens[(i + 3) % tokens.size()];
                    require(llama_state_seq_save_file(ctx, good.c_str(), 1, other_prompt.data(), prefix) > 0, "ownership file save");
                    std::filesystem::copy_file(good, bad);
                    std::filesystem::resize_file(bad, std::filesystem::file_size(bad) - 1);
                    std::vector<llama_token> prompt(prefix, -9999); size_t prompt_count = size_t(-1);
                    require(llama_state_seq_load_file(ctx, bad.c_str(), 0, prompt.data(), prompt.size(), &prompt_count) == 0,
                            "ownership truncated file accepted");
                    require(prompt_count == size_t(-1) && std::all_of(prompt.begin(), prompt.end(), [](llama_token token) { return token == -9999; }),
                            "failed file restore published prompt tokens");
                }
                require(save_full(ctx) == outputs_and_cache, "failed restore changed output ownership or full cache");
                const auto recovered_target = save_seq(ctx, 0);
                require(recovered_target == target_base && save_seq(ctx, 1) == other_base, "failed restore changed sequence snapshots");
                save_state_artifact(("ownership-" + kind + "-recovered.bin").c_str(), recovered_target);
                for (int i = 0; i < 2; ++i) {
                    decode(ctx, model, capture, tokens[prefix + i], prefix + i, 0);
                    decode(ctx, model, capture, tokens[3 + i], prefix + i, 1);
                }
                const auto other_advanced = save_seq(ctx, 1);
                require(llama_spec_ckpt_restore_ex(ctx, 0, prefix, -1) == LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                        "failed restore lost target checkpoint");
                require(save_seq(ctx, 0) == target_base && save_seq(ctx, 1) == other_advanced,
                        "target checkpoint recovery changed other sequence");
                require(llama_spec_ckpt_restore_ex(ctx, 1, prefix, -1) == LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                        "failed restore lost other checkpoint");
                require(save_seq(ctx, 1) == other_base, "other checkpoint recovery bytes differ");
                compare(("failed_restore_" + kind + "_target").c_str(), expected, continuation(0), false);
                compare(("failed_restore_" + kind + "_other").c_str(), other_control,
                        decode(ctx, model, capture, tokens[4], prefix, 1), false);
                std::printf("restore_ownership: kind=%s full_state_exact=true both_checkpoints_recovered=true passed=true\n", kind.c_str());
                std::fflush(stdout);
                llama_spec_ckpt_discard(ctx);
                require(llama_spec_ckpt_init(ctx, mode, 4) == mode, "ownership checkpoint mode reselection");
            }
            restore_full();
            if (std::getenv("ESE_LIFECYCLE_RESTORE_OWNERSHIP_ONLY")) {
                llama_free(ctx); ctx = nullptr; llama_free_model(model); model = nullptr; llama_backend_free();
                std::puts("restore_ownership_result: passed"); return 0;
            }
        }
        std::vector<observation> expected_spec;
        for(int i=prefix;i<prefix+5;i++) expected_spec.push_back(decode(ctx,model,capture,tokens[i],i,0));
        for(int accepted=0;accepted<=4;accepted++) {
            restore_full();
            require(llama_spec_ckpt_save(ctx,0),"checkpoint save");
            for(int i=prefix;i<prefix+4;i++) decode(ctx,model,capture,tokens[i],i,0);
            auto restored=llama_spec_ckpt_restore_ex(ctx,0,prefix,accepted-1);
            require(restored==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,"checkpoint restore");
            for(int i=prefix;i<prefix+accepted;i++) decode(ctx,model,capture,tokens[i],i,0);
            compare(("speculation_accepted_"+std::to_string(accepted)).c_str(),expected_spec[accepted],
                    decode(ctx,model,capture,tokens[prefix+accepted],prefix+accepted,0));
            llama_spec_ckpt_discard(ctx);
            require(llama_spec_ckpt_init(ctx,mode,4)==mode,"checkpoint mode reselection");
        }
        restore_full();
        for(int i=0;i<prefix;i++) decode(ctx,model,capture,tokens[(i+3)%tokens.size()],i,1);
        auto target_before=save_seq(ctx,0);
        require(llama_spec_ckpt_save(ctx,0),"isolated checkpoint save");
        for(int i=prefix;i<prefix+4;i++) decode(ctx,model,capture,tokens[i],i,0);
        decode(ctx,model,capture,tokens[3],prefix,1);
        auto other_before=save_seq(ctx,1);
        require(llama_spec_ckpt_restore_ex(ctx,0,prefix,-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                "isolated checkpoint restore");
        require(save_seq(ctx,0)==target_before,"restored target sequence bytes differ");
        require(save_seq(ctx,1)==other_before,"independently advanced sequence was changed");
        std::printf("spec_isolation: target_full_bytes=%zu other_full_bytes=%zu both_exact=true\n",
                target_before.size(),other_before.size());std::fflush(stdout);
        llama_spec_ckpt_discard(ctx);
        require(llama_spec_ckpt_init(ctx,mode,4)==mode,"bank checkpoint mode selection");
        restore_full();
        for(int i=0;i<prefix;i++) decode(ctx,model,capture,tokens[(i+3)%tokens.size()],i,1);
        target_before=save_seq(ctx,0);
        require(llama_spec_ckpt_save(ctx,0),"first live window save");
        decode(ctx,model,capture,tokens[3],prefix,1);
        auto bank_other_base=save_seq(ctx,1);
        require(llama_spec_ckpt_init(ctx,mode,4)==mode,"second live window mode selection");
        require(llama_spec_ckpt_save(ctx,1),"second live window save");
        for(int i=prefix;i<prefix+4;i++) decode(ctx,model,capture,tokens[i],i,0);
        decode(ctx,model,capture,tokens[4],prefix+1,1);
        auto bank_other_advanced=save_seq(ctx,1);
        require(llama_spec_ckpt_restore_ex(ctx,0,prefix,-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                "first live window restore");
        require(save_seq(ctx,0)==target_before,"first bank window bytes differ");
        require(save_seq(ctx,1)==bank_other_advanced,"first bank restore changed the second sequence");
        llama_spec_ckpt_discard_seq(ctx,0);
        require(llama_spec_ckpt_restore_ex(ctx,0,prefix,-1)==LLAMA_SPEC_CKPT_RESTORE_FAILED,
                "discarded bank window was admitted");
        require(llama_spec_ckpt_restore_ex(ctx,1,prefix+1,-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                "second bank window was lost on discard");
        require(save_seq(ctx,1)==bank_other_base,"second bank window bytes differ");
        require(save_seq(ctx,0)==target_before,"second bank restore changed the first sequence");
        std::printf("spec_bank: first_full_bytes=%zu second_full_bytes=%zu both_exact=true independent_discard=true\n",
                target_before.size(),bank_other_base.size());std::fflush(stdout);
        llama_spec_ckpt_discard(ctx);
        require(llama_spec_ckpt_init(ctx,mode,4)==mode,"copy checkpoint mode selection");
        restore_full();
        llama_kv_cache_seq_cp(ctx,0,1,0,-1);
        auto copy_before=save_seq(ctx,1);
        auto source_before=save_seq(ctx,0);
        require(llama_spec_ckpt_save(ctx,1),"pending sequence-copy checkpoint save");
        for(int i=prefix;i<prefix+4;i++) decode(ctx,model,capture,tokens[i],i,1);
        require(llama_spec_ckpt_restore_ex(ctx,1,prefix,-1)==LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                "copied sequence checkpoint restore");
        require(save_seq(ctx,1)==copy_before,"copied target sequence bytes differ");
        require(save_seq(ctx,0)==source_before,"copied checkpoint changed its source sequence");
        std::printf("spec_pending_copy: sequence_bytes=%zu source_preserved=true target_exact=true\n",copy_before.size());
        std::fflush(stdout);
        llama_spec_ckpt_discard(ctx);
        restore_seq(0);
        const char * output=std::getenv("ESE_LIFECYCLE_DIRECTORY"); require(output, "missing lifecycle output directory");
        std::string whole_path=std::string(output)+"/whole.session", seq_path=std::string(output)+"/sequence.session";
        require(llama_state_save_file(ctx,whole_path.c_str(),tokens.data(),prefix), "whole file save");
        require(llama_state_seq_save_file(ctx,seq_path.c_str(),0,tokens.data(),prefix)>0, "sequence file save");
        std::vector<llama_token> loaded(prefix); size_t loaded_count=0;
        llama_kv_cache_clear(ctx); require(llama_state_load_file(ctx,whole_path.c_str(),loaded.data(),loaded.size(),&loaded_count), "whole file load");
        require(loaded_count==size_t(prefix) && std::equal(loaded.begin(),loaded.end(),tokens.begin()), "whole file tokens");
        compare("whole_session_file",expected,continuation(0));
        llama_kv_cache_clear(ctx); require(llama_state_seq_load_file(ctx,seq_path.c_str(),1,loaded.data(),loaded.size(),&loaded_count)>0, "sequence file load");
        require(loaded_count==size_t(prefix) && std::equal(loaded.begin(),loaded.end(),tokens.begin()), "sequence file tokens");
        compare("sequence_session_migration",expected,continuation(1));
        std::remove(whole_path.c_str()); std::remove(seq_path.c_str());
        restore_full();
        compare("independent_sequence",expected,replay(1),false);
        llama_free(ctx); ctx=nullptr; llama_free_model(model); model=nullptr; llama_backend_free();
        std::puts("lifecycle_result: passed"); return 0;
    } catch(const std::exception & e) {
        std::fprintf(stderr,"LIFECYCLE ERROR: %s\n",e.what());
        if(ctx) llama_free(ctx); if(model) llama_free_model(model); llama_backend_free(); return 1;
    }
}
