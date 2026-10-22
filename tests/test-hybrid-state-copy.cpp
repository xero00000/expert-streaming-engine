#include "../src/llama-build-context.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "ggml-backend.h"

#include <array>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool result, const char * message) {
    if (!result) throw std::runtime_error(message);
}

int main() {
    try {
        // Drive the real copy graph with different stored row widths: a base
        // recurrent row, an extended PLE row, and a shorter auxiliary row.
        llama_model model{};
        model.arch = LLM_ARCH_LLAMA; // no model files or experimental loader opt-in
        auto & hp = model.hparams;
        hp.n_layer = 3;
        hp.n_embd = 4;
        hp.n_head_arr.fill(1);
        hp.n_head_kv_arr.fill(1);
        hp.n_embd_head_k_full = 4;
        hp.n_embd_head_v_full = 4;
        hp.ssm_d_state = 2;
        hp.ssm_d_inner = 4;

        llama_context ctx(model);
        ctx.cparams.n_seq_max = 3;
        ctx.n_ctx_max = 16;
        ctx.buf_compute_meta.resize(16 * 1024 * 1024);
        ctx.kv_self.size = 8;
        ctx.kv_self.cells.resize(8);
        ctx.kv_self.s_l.resize(3);
        ctx.kv_self.k_l.resize(3, nullptr);
        ctx.kv_self.v_l.resize(3, nullptr);
        const std::array<int64_t, 3> widths = {8, 21, 5};
        auto * storage = ggml_init({4096 * 1024, nullptr, true});
        check(storage != nullptr, "state metadata allocation");
        ctx.kv_self.ctxs.push_back(storage);
        for (size_t il = 0; il < widths.size(); ++il) {
            ctx.kv_self.s_l[il] = ggml_new_tensor_2d(storage, GGML_TYPE_F32, widths[il], 3);
        }
        auto * backend = ggml_backend_cpu_init();
        check(backend != nullptr, "CPU backend initialization");
        ctx.backends.push_back(backend);
        auto * buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
        check(buffer != nullptr, "state tensor allocation");
        ctx.kv_self.bufs.push_back(buffer);
        ggml_backend_buffer_clear(buffer, 0);
        ctx.sched = ggml_backend_sched_new(ctx.backends.data(), nullptr, 1, 65536, false);
        check(ctx.sched != nullptr, "scheduler initialization");

        for (uint32_t i = 0; i < ctx.kv_self.size; ++i) {
            ctx.kv_self.cells[i].pos = i;
            ctx.kv_self.cells[i].src = i;
            ctx.kv_self.cells[i].seq_id.insert(0);
        }
        llama_kv_cache_seq_cp(&ctx, 0, 1, 0, -1);
        for (uint32_t i = 0; i < ctx.kv_self.size; ++i) {
            check(ctx.kv_self.cells[i].pos == int32_t(i), "sequence copy changed an attention-token position");
            check(ctx.kv_self.cells[i].has_seq_id(1), "sequence copy omitted shared attention tokens");
        }

        // Empty rollback destinations are legitimate snapshots, including their
        // recurrent row. Restore must not index a zero-token metadata batch.
        const size_t empty_size = llama_state_seq_get_size(&ctx, 2, 0);
        check(empty_size > 0, "empty sequence size");
        std::vector<uint8_t> empty(empty_size);
        check(llama_state_seq_get_data(&ctx, empty.data(), empty.size(), 2, 0) == empty.size(), "empty sequence save");
        check(llama_state_seq_set_data(&ctx, empty.data(), empty.size(), 2, 0) == empty.size(), "empty sequence restore");
        for (uint32_t i = 0; i < ctx.kv_self.size; ++i) {
            check(ctx.kv_self.cells[i].pos == int32_t(i), "empty restore changed another sequence");
        }

        const std::array<std::array<int32_t, 3>, 3> maps = {{{2, 0, 1}, {0, 0, 0}, {0, 1, 2}}};
        for (const auto & map : maps) {
            ctx.reset_scheduler();
            std::array<std::vector<float>, 3> input;
            for (size_t il = 0; il < widths.size(); ++il) {
                input[il].resize(widths[il] * 3);
                for (size_t i = 0; i < input[il].size(); ++i) input[il][i] = float(1000 * il + i + 1);
                ggml_backend_tensor_set(ctx.kv_self.s_l[il], input[il].data(), 0, input[il].size() * sizeof(float));
            }
            auto * graph = llm_build_context::llama_build_graph_s_copy(ctx);
            check(ggml_backend_sched_alloc_graph(ctx.sched, graph), "copy graph allocation");
            std::array<int32_t, 8> indices = {map[0], map[1], map[2], 3, 4, 5, 6, 7};
            ggml_backend_tensor_set(ctx.inp_s_copy, indices.data(), 0, sizeof(indices));
            check(ggml_backend_sched_graph_compute(ctx.sched, graph) == GGML_STATUS_SUCCESS, "copy graph execution");
            for (size_t il = 0; il < widths.size(); ++il) {
                std::vector<float> output(input[il].size());
                ggml_backend_tensor_get(ctx.kv_self.s_l[il], output.data(), 0, output.size() * sizeof(float));
                for (int slot = 0; slot < 3; ++slot) {
                    check(std::memcmp(output.data() + slot * widths[il],
                                input[il].data() + map[slot] * widths[il], widths[il] * sizeof(float)) == 0,
                            "copy graph failed to preserve a complete state row");
                }
            }
        }
        // Execute actual defrag graphs with overlapping source/destination
        // ranges, including packed rows and the auxiliary indexer cache.
        hp.n_embd = hp.n_embd_head_k_full = hp.n_embd_head_v_full = 32;
        hp.indexer_head_size = 32;
        ctx.cparams.flash_attn = false;
        ctx.kv_self.v_trans = false; // hybrid token-major values with Flash Attention off
        ctx.kv_self.kr_l.resize(3);
        auto * cache_storage = ggml_init({4096 * 1024, nullptr, true});
        check(cache_storage, "defrag cache metadata allocation");
        ctx.kv_self.ctxs.push_back(cache_storage);
        const std::array<ggml_type, 3> formats = {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0};
        for (size_t il=0; il<formats.size(); ++il) {
            ctx.kv_self.k_l[il] = ggml_new_tensor_3d(cache_storage,formats[il],32,1,8);
            ctx.kv_self.v_l[il] = ggml_new_tensor_1d(cache_storage,formats[il],32*8);
            ctx.kv_self.kr_l[il] = ggml_new_tensor_2d(cache_storage,formats[il],32,8);
        }
        auto * cache_buffer = ggml_backend_alloc_ctx_tensors(cache_storage,backend);
        check(cache_buffer, "defrag cache allocation");
        ctx.kv_self.bufs.push_back(cache_buffer);
        const std::array<std::array<uint32_t,8>,2> moves = {{{8,8,0,1,2,3,4,5},{0,8,8,1,2,3,4,5}}};
        int defrag_cases=0;
        for (const auto & map : moves) {
            ctx.reset_scheduler();
            std::array<std::vector<uint8_t>,3> state_before;
            for (size_t il=0; il<state_before.size(); ++il) {
                state_before[il].resize(ggml_nbytes(ctx.kv_self.s_l[il]));
                ggml_backend_tensor_get(ctx.kv_self.s_l[il],state_before[il].data(),0,state_before[il].size());
            }
            std::array<std::array<std::vector<uint8_t>,3>,3> original;
            for (size_t il=0; il<formats.size(); ++il) {
                const std::array<ggml_tensor *,3> tensors = {ctx.kv_self.k_l[il],ctx.kv_self.v_l[il],ctx.kv_self.kr_l[il]};
                for (size_t family=0; family<tensors.size(); ++family) {
                    auto & bytes = original[il][family]; bytes.resize(ggml_nbytes(tensors[family]));
                    for (size_t i=0; i<bytes.size(); ++i) bytes[i] = uint8_t(i*13+il*71+family*43);
                    ggml_backend_tensor_set(tensors[family],bytes.data(),0,bytes.size());
                }
            }
            auto * graph = llm_build_context::llama_build_graph_defrag(ctx,{map.begin(),map.end()});
            check(ggml_backend_sched_alloc_graph(ctx.sched,graph), "defrag graph allocation");
            check(ggml_backend_sched_graph_compute(ctx.sched,graph)==GGML_STATUS_SUCCESS,"defrag graph execution");
            for (size_t il=0; il<state_before.size(); ++il) {
                std::vector<uint8_t> actual(state_before[il].size());
                ggml_backend_tensor_get(ctx.kv_self.s_l[il],actual.data(),0,actual.size());
                check(actual==state_before[il],"defrag graph changed recurrent state bytes");
            }
            for (size_t il=0; il<formats.size(); ++il) {
                const std::array<ggml_tensor *,3> tensors = {ctx.kv_self.k_l[il],ctx.kv_self.v_l[il],ctx.kv_self.kr_l[il]};
                const size_t row = ggml_row_size(formats[il],32);
                for (size_t family=0; family<tensors.size(); ++family) {
                    std::vector<uint8_t> actual(ggml_nbytes(tensors[family]));
                    ggml_backend_tensor_get(tensors[family],actual.data(),0,actual.size());
                    for (uint32_t from=0; from<map.size(); ++from) if (map[from]!=map.size())
                        check(std::memcmp(actual.data()+map[from]*row,original[il][family].data()+from*row,row)==0,
                              "overlapping defrag changed cache row bytes");
                    ++defrag_cases;
                }
            }
        }
        std::printf("defrag_copy: cases=%d formats=F32,F16,Q8_0 families=K,V,indexer exact=true\n",defrag_cases);
        // Verify the architecture-specific checkpoint selection on the same
        // fake storage without loading model weights or enabling the loader.
        model.arch = LLM_ARCH_QWEN4EXP;
        check(llama_spec_ckpt_init(&ctx, LLAMA_SPEC_CKPT_PER_STEP, 4) == LLAMA_SPEC_CKPT_NONE,
                "incomplete Qwen4Exp per-step journal was admitted");
        check(llama_spec_ckpt_init(&ctx, LLAMA_SPEC_CKPT_AUTO, 4) == LLAMA_SPEC_CKPT_GPU_FALLBACK,
                "automatic checkpoint did not choose complete-row shadows");
        check(!ctx.kv_self.save_per_step_ssm, "per-step capture remained enabled");
        std::puts("PASS: variable-width rows, shared token positions, empty restore, complete-row checkpoint selection");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
