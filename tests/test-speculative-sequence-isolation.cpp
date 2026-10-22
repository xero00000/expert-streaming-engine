#include "../src/llama-build-context.h"
#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-qwen4exp-ple.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif

#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

// Keep public common headers separate from private architecture enums.
bool test_common_checkpoint_discard(llama_context * ctx, llama_seq_id seq_id, int mode);

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

#ifdef GGML_USE_CUDA
// Provide a primary context before the runtime's lazy cudaGetDevice probe.
// This lets --report-api-errors all check the actual row-copy operations
// without an expected no-current-context driver probe during test startup.
struct primary_context {
    CUdevice device = 0;
    CUcontext context = nullptr;
    explicit primary_context(int ordinal) {
        if (ordinal < 0) return;
        check(cuInit(0) == CUDA_SUCCESS, "CUDA driver initialization");
        check(cuDeviceGet(&device, ordinal) == CUDA_SUCCESS, "CUDA device selection");
        check(cuDevicePrimaryCtxRetain(&context, device) == CUDA_SUCCESS, "CUDA primary context retention");
        check(cuCtxSetCurrent(context) == CUDA_SUCCESS, "CUDA primary context activation");
    }
    ~primary_context() {
        if (context) cuDevicePrimaryCtxRelease(device);
    }
};
#endif

static void run(int mode, int target, int device, bool corrupt, bool bank = false) {
    llama_model model{};
    model.arch = LLM_ARCH_LLAMA; // construct a context without external model files
    auto & hp = model.hparams;
    hp.n_layer = 2;
    hp.n_embd = 4;
    hp.n_head_arr.fill(1);
    hp.n_head_kv_arr.fill(1);
    hp.n_embd_head_k_full = hp.n_embd_head_v_full = 4;
    llama_context ctx(model);
    ctx.cparams.n_seq_max = 3;
    ctx.n_ctx_max = 16;
    ctx.buf_compute_meta.resize(16 * 1024 * 1024);
    auto & kv = ctx.kv_self;
    kv.size = kv.n = 8;
    kv.cells.resize(8);
    kv.s_l.resize(2);
    kv.k_l.resize(2, nullptr);
    kv.v_l.resize(2, nullptr);
    kv.head = kv.used = 4;
    for (int i = 0; i < 8; ++i) {
        kv.cells[i].src = i;
        kv.cells[i].pos = i < 4 ? i % 2 : -1;
        if (i < 4) kv.cells[i].seq_id.insert(i / 2);
    }
    auto * storage = ggml_init({4096 * 1024, nullptr, true});
    check(storage != nullptr, "state metadata allocation");
    kv.ctxs.push_back(storage);
    const std::array<int, 2> widths = {8, 16};
    for (int il = 0; il < 2; ++il) kv.s_l[il] = ggml_new_tensor_2d(storage, GGML_TYPE_F32, widths[il], 3);
    ggml_backend_t backend = nullptr;
    if (device < 0) backend = ggml_backend_cpu_init();
#ifdef GGML_USE_CUDA
    else backend = ggml_backend_cuda_init(device, nullptr, nullptr);
#endif
    check(backend != nullptr, "backend initialization");
    ctx.backends.push_back(backend);
    if (device >= 0) {
        auto * cpu = ggml_backend_cpu_init();
        check(cpu != nullptr, "scheduler CPU backend initialization");
        ctx.backends.push_back(cpu);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(storage, backend);
    check(buffer != nullptr, "state tensor allocation");
    kv.bufs.push_back(buffer);
    ctx.sched = ggml_backend_sched_new(ctx.backends.data(), nullptr, ctx.backends.size(), 65536, false);
    check(ctx.sched != nullptr, "scheduler initialization");
    auto * graph = llm_build_context::llama_build_graph_s_copy(ctx);
    check(ggml_backend_sched_alloc_graph(ctx.sched, graph), "state graph allocation");
    std::array<int32_t, 8> indices = {0, 1, 2, 3, 4, 5, 6, 7};
    ggml_backend_tensor_set(ctx.inp_s_copy, indices.data(), 0, sizeof(indices));
    ggml_backend_buffer_clear(buffer, 0);
    check(ggml_backend_sched_graph_compute(ctx.sched, graph) == GGML_STATUS_SUCCESS, "state graph execution");
    std::array<std::vector<float>, 2> base, advanced;
    for (int il = 0; il < 2; ++il) {
        base[il].resize(widths[il] * 3);
        for (size_t i = 0; i < base[il].size(); ++i) base[il][i] = float(1000 * il + i + 1);
        ggml_backend_tensor_set(kv.s_l[il], base[il].data(), 0, base[il].size() * sizeof(float));
        advanced[il] = base[il];
        for (int i = 0; i < 2 * widths[il]; ++i) advanced[il][i] += 10000;
    }
    // Exercise real PLE history serialization/removal with synthetic tables.
    model.arch = LLM_ARCH_QWEN4EXP;
    qwen4exp_ple_hash_config config;
    config.head_vocab_sizes.fill(7);
    config.layer_multipliers = {3, 5, 7};
    for (int i = 0; i < 16; ++i) config.head_offsets[i] = 7 * i;
    ctx.qwen4exp_ple_sequences = std::make_unique<qwen4exp_ple_sequence_state>(config);
    for (int seq = 0; seq < 2; ++seq) for (int pos = 0; pos < 2; ++pos) {
        ctx.qwen4exp_ple_sequences->accept(seq, pos, 1 + pos + seq);
    }
    check(llama_spec_ckpt_init(&ctx, mode, 4) == mode, "checkpoint mode selection");
    check(llama_spec_ckpt_restore_ex(&ctx, target, 2, -1) == LLAMA_SPEC_CKPT_RESTORE_FAILED,
            "restore before save was admitted");
    check(!llama_spec_ckpt_save(&ctx, 9), "out-of-range checkpoint sequence was admitted");
    check(llama_spec_ckpt_save(&ctx, target), "checkpoint save");
    if (bank) check(llama_spec_ckpt_save(&ctx, 1 - target), "second simultaneous checkpoint save");
    if (mode == LLAMA_SPEC_CKPT_GPU_FALLBACK) {
        for (int il = 0; il < 2; ++il) {
            check(ggml_nbytes(kv.ckpt.s_l_shadow[il]) == size_t(widths[il]) * sizeof(float) * ctx.cparams.n_seq_max,
                    "shadow allocation differs from configured sequence capacity");
        }
    }
    for (int il = 0; il < 2; ++il) {
        ggml_backend_tensor_set(kv.s_l[il], advanced[il].data(), 0, advanced[il].size() * sizeof(float));
    }
    for (int seq = 0; seq < 2; ++seq) {
        kv.cells[4 + seq].pos = 2;
        kv.cells[4 + seq].seq_id.insert(seq);
        ctx.qwen4exp_ple_sequences->accept(seq, 2, 5 + seq);
    }
    kv.used = kv.head = 6;
    const int other = 1 - target;
    auto other_history = ctx.qwen4exp_ple_sequences->tokens(other);
    auto target_history = ctx.qwen4exp_ple_sequences->tokens(target);
    if (!bank) check(llama_spec_ckpt_restore_ex(&ctx, other, 2, -1) == LLAMA_SPEC_CKPT_RESTORE_FAILED,
            "wrong-sequence checkpoint restore was admitted");
    check(llama_spec_ckpt_restore_ex(&ctx, target, 1, -1) == LLAMA_SPEC_CKPT_RESTORE_FAILED,
            "wrong checkpoint position was admitted");
    if (corrupt) kv.ckpt.sequences[target].cpu_data.pop_back();
    auto result = llama_spec_ckpt_restore_ex(&ctx, target, 2, -1);
    check(result == (corrupt ? LLAMA_SPEC_CKPT_RESTORE_FAILED : LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED),
            "checkpoint restore result");
    check(kv.cells[4 + other].pos == 2 && kv.cells[4 + other].has_seq_id(other),
            "restore discarded another sequence's KV tail");
    check(ctx.qwen4exp_ple_sequences->tokens(other) == other_history, "restore changed another PLE history");
    if (corrupt) {
        check(kv.cells[4 + target].pos == 2 && kv.cells[4 + target].has_seq_id(target),
                "failed partial restore discarded the original KV tail");
        check(ctx.qwen4exp_ple_sequences->tokens(target) == target_history,
                "failed partial restore changed the original PLE history");
    } else {
        check(!kv.cells[4 + target].has_seq_id(target), "restore retained rejected KV tail");
        check(ctx.qwen4exp_ple_sequences->tokens(target).size() == 2, "restore retained rejected PLE history");
    }
    for (int il = 0; il < 2; ++il) {
        std::vector<float> output(base[il].size());
        ggml_backend_tensor_get(kv.s_l[il], output.data(), 0, output.size() * sizeof(float));
        for (int slot = 0; slot < 3; ++slot) {
            const auto & expected = slot == target && !corrupt ? base[il] : advanced[il];
            check(std::memcmp(output.data() + slot * widths[il], expected.data() + slot * widths[il],
                        widths[il] * sizeof(float)) == 0, "checkpoint changed the wrong recurrent row");
        }
    }
    if (bank) {
        const auto & other_slot = kv.ckpt.sequences[other];
        const void * allocation = mode == LLAMA_SPEC_CKPT_CPU ? (const void *) other_slot.cpu_data.data() :
            kv.ckpt.s_l_shadow[0]->data;
        check(llama_spec_ckpt_init(&ctx, mode, 4) == mode, "reinitializing a live bank");
        check(!llama_spec_ckpt_save(&ctx, 9), "invalid save modified a live bank");
        if (mode == LLAMA_SPEC_CKPT_CPU) {
            const size_t limit = kv.ckpt.sequences[target].cpu_limit;
            kv.ckpt.sequences[target].cpu_limit = 0;
            check(!llama_spec_ckpt_save(&ctx, target), "save exceeded the CPU payload cap");
            kv.ckpt.sequences[target].cpu_limit = limit;
        }
        check(test_common_checkpoint_discard(&ctx, target, mode), "common checkpoint retained its owner");
        check(!kv.ckpt.sequences[target].saved && kv.ckpt.sequences[other].saved,
                "sequence discard invalidated a different window");
        check(llama_spec_ckpt_restore_ex(&ctx, target, 2, -1) == LLAMA_SPEC_CKPT_RESTORE_FAILED,
                "discarded window remained restorable");
        check(llama_spec_ckpt_restore_ex(&ctx, other, 2, -1) == LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                "second saved window was lost");
        check(ctx.qwen4exp_ple_sequences->tokens(other).size() == 2, "second window retained PLE tail");
        check(!kv.cells[4 + other].has_seq_id(other), "second window retained KV tail");
        check(allocation == (mode == LLAMA_SPEC_CKPT_CPU ? (const void *) other_slot.cpu_data.data() :
                    kv.ckpt.s_l_shadow[0]->data), "live checkpoint bank reallocated");
        for (int il = 0; il < 2; ++il) {
            std::vector<float> output(base[il].size());
            ggml_backend_tensor_get(kv.s_l[il], output.data(), 0, output.size() * sizeof(float));
            for (int slot = 0; slot < 3; ++slot) {
                const auto & expected = slot == target && corrupt ? advanced[il] : base[il];
                check(std::memcmp(output.data() + slot * widths[il], expected.data() + slot * widths[il],
                            widths[il] * sizeof(float)) == 0, "second restore changed a different row");
            }
        }
    }
    if (bank) {
        check(llama_spec_ckpt_save(&ctx, target) && llama_spec_ckpt_save(&ctx, other),
                "saving refreshed windows");
        check(llama_kv_cache_seq_rm(&ctx, target, 0, -1), "remove checkpoint base");
        check(!kv.ckpt.sequences[target].saved && kv.ckpt.sequences[other].saved,
                "removing a prefix retained its checkpoint or erased another window");
        check(llama_spec_ckpt_restore_ex(&ctx, other, 2, -1) == LLAMA_SPEC_CKPT_RESTORE_BASE_REPLAY_REQUIRED,
                "unrelated window was lost after prefix removal");
        llama_kv_cache_clear(&ctx);
        for (const auto & snapshot : kv.ckpt.sequences) check(!snapshot.saved, "cache clear retained a window");
    }
    llama_spec_ckpt_discard(&ctx);
    for (const auto & snapshot : kv.ckpt.sequences) {
        check(!snapshot.saved && snapshot.cpu_data.empty(), "discard retained checkpoint ownership");
        check(snapshot.cpu_data.capacity() <= snapshot.cpu_limit, "CPU bank exceeded its startup bound");
    }
}

int main(int argc, char ** argv) {
    try {
        const int device = argc > 1 ? std::stoi(argv[1]) : -1;
#ifdef GGML_USE_CUDA
        primary_context primary(device);
#endif
        for (int target = 0; target < 2; ++target) {
            run(LLAMA_SPEC_CKPT_GPU_FALLBACK, target, device, false);
            run(LLAMA_SPEC_CKPT_CPU, target, device, false);
            run(LLAMA_SPEC_CKPT_CPU, target, device, true);
        }
        for (int target = 0; target < 2; ++target) {
            run(LLAMA_SPEC_CKPT_GPU_FALLBACK, target, device, false, true);
            run(LLAMA_SPEC_CKPT_CPU, target, device, false, true);
            run(LLAMA_SPEC_CKPT_CPU, target, device, true, true);
        }
        std::puts("PASS: sequence-local shadow/serialized checkpoints, PLE/KV tails, invalid ownership, atomic partial failure, bounded simultaneous windows and common-layer discard");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
