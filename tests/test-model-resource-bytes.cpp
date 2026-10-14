#include "llama.h"
#include "llama-model.h"
#include "resource-planner.h"

#include <cstdio>
#include <stdexcept>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void check(const llama_model * model, uint64_t dense, uint64_t expert) {
    uint64_t actual_dense = 1, actual_expert = 1;
    llama_model_resource_bytes(model, &actual_dense, &actual_expert);
    require(actual_dense == dense, "resident dense bytes differ");
    require(actual_expert == expert, "expert source bytes differ");
    llama_model_resource_bytes(model, &actual_dense, nullptr);
    require(actual_dense == dense, "dense-only query differs");
    llama_model_resource_bytes(model, nullptr, &actual_expert);
    require(actual_expert == expert, "expert-only query differs");
    llama_model_resource_bytes(model, nullptr, nullptr);
}

int main() {
    try {
        check(nullptr, 0, 0);
        llama_model model{};
        model.arch = LLM_ARCH_QWEN4EXP;
        // One checked descriptor makes the metadata index nonempty without
        // allocating a payload or opening a model file.
        llama_expert_cache::descriptor_spec spec;
        spec.rank = 1;
        spec.ggml_type = GGML_TYPE_F32;
        spec.block_bytes = 4;
        spec.dimensions[0] = 1;
        spec.strides[0] = 4;
        spec.model_identity = {1, 2};
        spec.extents = {{0, {3, 4}, 0, 4}};
        auto descriptor = llama_expert_cache::descriptor::make(spec, {{0, 4}});
        model.expert_index.descriptors.emplace(descriptor->key(), descriptor);
        model.expert_index.dense_bytes = 33251397120ULL;
        model.expert_index.deferred_bytes = 25157057808ULL;
        check(&model, model.expert_index.dense_bytes, model.expert_index.deferred_bytes);
        model.qwen4exp_ple_source.valid = true;
        model.qwen4exp_ple_source.tensor_bytes = 28800138240ULL;
        check(&model, 4451258880ULL, model.expert_index.deferred_bytes);

        common_resource_plan_input input;
        llama_model_resource_bytes(&model, &input.dense_ram_bytes, &input.expert_source_bytes);
        input.model_is_moe = true;
        input.requested_policy = COMMON_MEMORY_POLICY_CACHE;
        input.max_ram_bytes = 12ULL << 30;
        input.requested_expert_ram_bytes = 1ULL << 30;
        input.min_expert_ram_bytes = 4;
        input.qwen4exp_ple_ram_bytes = 256ULL << 20;
        input.io_staging_bytes = 4ULL << 20;
        input.requested_context = input.max_context = 4096;
        input.min_kv_quality = COMMON_KV_QUALITY_F16;
        input.kv_bytes_per_token[COMMON_KV_QUALITY_F16] = 1024;
        input.devices = {{-1, 12ULL << 30, 0, input.dense_ram_bytes, 0}};
        common_resource_plan plan;
        std::string error;
        require(common_resource_plan_solve(input, plan, error), "12 GiB cache plan rejected");
        require(plan.expert_ram_bytes == input.requested_expert_ram_bytes, "requested expert cache changed");
        require(plan.qwen4exp_ple_ram_bytes == input.qwen4exp_ple_ram_bytes, "PLE cache reservation lost");
        input.dense_ram_bytes = model.expert_index.dense_bytes;
        require(!common_resource_plan_solve(input, plan, error), "old source-byte forecast admits 12 GiB");
        std::puts("PASS: 12 GiB resident forecast admits requested cache; old source forecast rejects");

        model.arch = LLM_ARCH_LLAMA;
        check(&model, model.expert_index.dense_bytes, model.expert_index.deferred_bytes);
        model.arch = LLM_ARCH_QWEN4EXP;
        model.qwen4exp_ple_source.tensor_bytes = 0;
        check(&model, model.expert_index.dense_bytes, model.expert_index.deferred_bytes);
        model.qwen4exp_ple_source.tensor_bytes = model.expert_index.dense_bytes;
        check(&model, 0, model.expert_index.deferred_bytes);
        ++model.qwen4exp_ple_source.tensor_bytes;
        check(&model, UINT64_MAX, model.expert_index.deferred_bytes);

        // Without a deferred index, the existing materialized-tensor sum
        // already excludes skipped tensors. Never subtract the source twice.
        model.expert_index = {};
        auto * ctx = ggml_init({4096, nullptr, true});
        require(ctx != nullptr, "tensor metadata allocation");
        model.ctxs.push_back(ctx);
        model.tensors_by_name.emplace_back("dense.weight", ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 8));
        check(&model, 32, 0);
        require(model.bufs.empty(), "test allocated weight payloads");
        std::puts("PASS: null, indexed, non-Qwen, invalid PLE, zero, overflow and materialized fallback controls");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
