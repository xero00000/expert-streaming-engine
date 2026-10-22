#include <random>
#include "llama-deepseek-v41-dspark.h"
#include "llama-deepseek-v41-model.h"
#include "llama-deepseek-v41-ops.h"
#include "llama-build-context.h"
#include "llama-context.h"
#include "llama-model.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include "../ggml/src/ggml-qat.h"
#include "../vendor/nlohmann/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace deepseek_v41 {
namespace {
void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }
using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
}
struct dspark::impl {
    llama_model & target;
    llama_context build_context;
    model_config c;
    std::vector<int32_t> layers;
    int block = 0, noise = 0, end = 0;
    std::map<std::string, ggml_tensor *> weights;
    std::vector<backend_ptr> backends;
    std::vector<context_ptr> weight_contexts;
    std::vector<buffer_ptr> buffers;
    std::unique_ptr<ggml_backend_sched, decltype(&ggml_backend_sched_free)> sched{nullptr, ggml_backend_sched_free};
    std::array<std::vector<float>, 3> windows;

    impl(llama_model & model, const std::string & path, int first, int last) : target(model), build_context(model) {
        build_context.cparams = {};
        build_context.inp_tokens = nullptr;
        build_context.inp_embd = nullptr;
        check(model.arch == LLM_ARCH_DEEPSEEK41, "DSpark requires a V4.1 target");
        ggml_context * metadata = nullptr;
        std::unique_ptr<gguf_context, decltype(&gguf_free)> file(
            gguf_init_from_file(path.c_str(), {true, &metadata}), gguf_free);
        context_ptr meta(metadata, ggml_free);
        check(file && meta, "cannot read DSpark GGUF");
        auto key = [&](const char * name) { int i = gguf_find_key(file.get(), name); check(i >= 0, "missing DSpark metadata"); return i; };
        check(std::string(gguf_get_val_str(file.get(), key("general.architecture"))) == "deepseek41_dspark" &&
            gguf_get_val_u32(file.get(), key("deepseek41_dspark.version")) == 1, "unsupported DSpark package");
        const std::string config = gguf_get_val_str(file.get(), key("deepseek41_dspark.config"));
        const auto j = nlohmann::json::parse(config);
        c = model_config::from_json(config);
        c.experts = j.at("dspark_n_routed_experts"); c.active_experts = j.at("dspark_num_experts_per_tok");
        layers = j.at("dspark_target_layer_ids").get<std::vector<int32_t>>();
        block = j.at("dspark_block_size"); noise = j.at("dspark_noise_token_id");
        check(j.at("num_nextn_predict_layers") == 3 && block == 5 && layers == std::vector<int32_t>({37,38,39}) &&
              c.hidden == model.dsv41_config.hidden && c.vocabulary == model.dsv41_config.vocabulary &&
              c.experts == 128 && c.active_experts == 3, "unsupported DSpark geometry or target");
#ifdef GGML_USE_CUDA
        check(first != last && first >= 0 && last >= 0 && first < ggml_backend_cuda_get_device_count() &&
              last < ggml_backend_cuda_get_device_count(), "DSpark requires two distinct CUDA devices");
        for (int dev : {first, last}) backends.emplace_back(ggml_backend_cuda_init(dev, nullptr, nullptr), ggml_backend_free);
        // Include the target embedding's device so GET_ROWS reads it in place.
        for (int dev = 0; dev < ggml_backend_cuda_get_device_count(); ++dev)
            if (dev != first && dev != last) backends.emplace_back(ggml_backend_cuda_init(dev, nullptr, nullptr), ggml_backend_free);
#else
        (void)first; (void)last;
        throw std::runtime_error("DSpark executor currently requires CUDA");
#endif
        backends.emplace_back(ggml_backend_cpu_init(), ggml_backend_free);
        for (auto & backend : backends) check(bool(backend), "DSpark backend initialization failed");
        ggml_backend_cpu_set_n_threads(backends.back().get(), 8);
        for (int device = 0; device < 2; ++device) {
            weight_contexts.emplace_back(ggml_init({ggml_tensor_overhead() * 128, nullptr, true}), ggml_free);
            check(bool(weight_contexts.back()), "DSpark weight metadata allocation failed");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
            const std::string name = gguf_get_tensor_name(file.get(), i);
            check(name.rfind("mtp.0.", 0) == 0 || name.rfind("mtp.1.", 0) == 0 || name.rfind("mtp.2.", 0) == 0,
                  "unexpected DSpark tensor");
            const int device = name.rfind("mtp.2.", 0) == 0 ? 1 : 0;
            auto * tensor = ggml_dup_tensor(weight_contexts[device].get(), ggml_get_tensor(meta.get(), name.c_str()));
            ggml_set_name(tensor, name.c_str()); weights.emplace(name, tensor);
        }
        check(weights.size() == 78, "incomplete DSpark package");
        for (int device = 0; device < 2; ++device) {
            buffers.emplace_back(ggml_backend_alloc_ctx_tensors(weight_contexts[device].get(), backends[device].get()), ggml_backend_buffer_free);
            check(bool(buffers.back()), "DSpark weights do not fit the selected GPUs");
            ggml_backend_buffer_set_usage(buffers.back().get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            std::fprintf(stderr, "DSpark %s weights: %.1f MiB\n", ggml_backend_name(backends[device].get()),
                ggml_backend_buffer_get_size(buffers.back().get()) / 1048576.0);
        }
        std::ifstream input(path, std::ios::binary);
        std::vector<char> chunk(16 * 1024 * 1024);
        for (int64_t i = 0; i < gguf_get_n_tensors(file.get()); ++i) {
            auto * tensor = weights.at(gguf_get_tensor_name(file.get(), i));
            input.seekg(gguf_get_data_offset(file.get()) + gguf_get_tensor_offset(file.get(), i));
            for (size_t offset = 0; offset < ggml_nbytes(tensor);) {
                const size_t size = std::min(chunk.size(), ggml_nbytes(tensor) - offset);
                input.read(chunk.data(), size); check(bool(input), "truncated DSpark tensor data");
                ggml_backend_tensor_set(tensor, chunk.data(), offset, size); offset += size;
            }
        }
        std::vector<ggml_backend_t> raw;
        for (auto & backend : backends) raw.push_back(backend.get());
        sched.reset(ggml_backend_sched_new(raw.data(), nullptr, raw.size(), 16384, false));
        check(bool(sched), "DSpark scheduler initialization failed");
    }
    ggml_tensor * w(int stage, const std::string & name) { return weights.at("mtp." + std::to_string(stage) + "." + name); }
    struct graph {
        impl & self;
        context_ptr ctx{nullptr, ggml_free};
        ggml_cgraph * gf = nullptr;
        explicit graph(impl & s) : self(s) {
            ggml_backend_sched_reset(self.sched.get());
            ctx.reset(ggml_init({ggml_tensor_overhead() * 16384 + ggml_graph_overhead_custom(16384, false), nullptr, true}));
            check(bool(ctx), "DSpark graph metadata allocation failed");
            gf = ggml_new_graph_custom(ctx.get(), 16384, false);
        }
        ~graph() { ggml_backend_sched_reset(self.sched.get()); }
        ggml_tensor * input(ggml_type type, int64_t width, int64_t rows, const char * name) {
            auto t = ggml_new_tensor_2d(ctx.get(), type, width, rows); ggml_set_name(t, name); ggml_set_input(t); return t;
        }
        ggml_tensor * round(ggml_tensor * t) { return ggml_cast(ctx.get(), ggml_cast(ctx.get(), t, GGML_TYPE_BF16), GGML_TYPE_F32); }
        ggml_tensor * linear(ggml_tensor * t, ggml_tensor * weight, bool qat = true, bool rounded = true) {
            if (qat) t = ggml_qat(ctx.get(), ggml_cont(ctx.get(), t), GGML_QAT_FP8_E8M0);
            auto out = ggml_mul_mat(ctx.get(), weight, t); ggml_mul_mat_set_prec(out, GGML_PREC_F32);
            return rounded ? round(out) : out;
        }
        ggml_tensor * norm(ggml_tensor * t, ggml_tensor * weight) {
            return round(ggml_mul(ctx.get(), ggml_rms_norm(ctx.get(), t, self.c.norm_eps), ggml_cast(ctx.get(), weight, GGML_TYPE_F32)));
        }
        ggml_tensor * rotate(ggml_tensor * t, ggml_tensor * pos, bool inverse = false) {
            const auto & c = self.c;
            return round(build_rope_tail(ctx.get(), t, pos, {int(c.rotary), 0, c.rope_theta, c.rope_factor, c.beta_fast, c.beta_slow}, inverse));
        }
        ggml_tensor * kv(ggml_tensor * t, ggml_tensor * pos, int stage, int n) {
            auto out = norm(linear(t, self.w(stage, "attn_kv")), self.w(stage, "attn_kv_a_norm"));
            out = rotate(ggml_reshape_3d(ctx.get(), out, self.c.attention.head_dim, 1, n), pos);
            return ggml_reshape_2d(ctx.get(), ggml_qat(ctx.get(), ggml_cont(ctx.get(), out), GGML_QAT_FP8_E8M0), self.c.attention.head_dim, n);
        }
        void output(ggml_tensor * t) { ggml_set_output(t); ggml_build_forward_expand(gf, t); }
        void allocate() { check(ggml_backend_sched_alloc_graph(self.sched.get(), gf), "DSpark graph allocation failed"); }
        void compute() { check(ggml_backend_sched_graph_compute(self.sched.get(), gf) == GGML_STATUS_SUCCESS, "DSpark graph execution failed"); }
        std::vector<float> read(ggml_tensor * t) {
            check(t->type == GGML_TYPE_F32, "DSpark output must be float32");
            std::vector<float> out(ggml_nelements(t)); ggml_backend_tensor_get(t, out.data(), 0, out.size() * sizeof(float));
            for (float value : out) check(std::isfinite(value), "nonfinite DSpark output");
            return out;
        }
    };
    void ingest(const std::vector<float> & features, int position, int count) {
        check(position == end && count > 0 && count <= 8 && position <= int(c.context) - count &&
              features.size() == size_t(count) * c.hidden * layers.size(), "invalid DSpark accepted feature batch");
        graph g(*this);
        auto input = g.input(GGML_TYPE_F32, c.hidden * layers.size(), count, "dspark_features");
        auto positions = g.input(GGML_TYPE_I32, count, 1, "dspark_main_positions");
        auto main = g.norm(g.linear(input, w(0, "main_proj")), w(0, "main_norm"));
        std::array<ggml_tensor *, 3> outs;
        for (int s = 0; s < 3; ++s) { outs[s] = g.kv(main, positions, s, count); g.output(outs[s]); }
        g.allocate();
        std::vector<int32_t> pos(count); std::iota(pos.begin(), pos.end(), position);
        ggml_backend_tensor_set(input, features.data(), 0, features.size() * sizeof(float));
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t)); g.compute();
        // Publish all stages together after execution and finite checks succeed.
        std::array<std::vector<float>, 3> appended;
        for (int s = 0; s < 3; ++s) appended[s] = g.read(outs[s]);
        for (int s = 0; s < 3; ++s) {
            auto & v = windows[s]; v.insert(v.end(), appended[s].begin(), appended[s].end());
            const size_t limit = size_t(c.attention.sliding_window) * c.attention.head_dim;
            if (v.size() > limit) v.erase(v.begin(), v.end() - limit);
        }
        end += count;
    }
    std::vector<llama_token> propose(llama_token known, int position) {
        check(position == end && position > 0 && position <= int(c.context) - block && known >= 0 && known < int(c.vocabulary),
              "invalid DSpark proposal position");
        std::vector<float> base_logits;
        {
            graph g(*this); auto ctx = g.ctx.get(); const int n = block, d = c.hidden, hd = c.attention.head_dim;
            auto tokens = g.input(GGML_TYPE_I32, n, 1, "dspark_tokens");
            auto positions = g.input(GGML_TYPE_I32, n, 1, "dspark_positions");
            auto embedding = g.round(ggml_get_rows(ctx, target.dsv41_tensors.at("token_embd"), tokens));
            auto x = ggml_repeat(ctx, ggml_reshape_3d(ctx, embedding, d, 1, n), ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, c.streams, n));
            auto pre = ggml_concat(ctx, ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, n), 1),
                ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, c.streams - 1, n), 0), 0);
            std::array<ggml_tensor *, 3> cache_inputs;
            for (int s = 0; s < 3; ++s) {
                auto mixes = [&](ggml_tensor * t, const std::string & sub) {
                    return build_hc_mixes(ctx, t, w(s,"hc_"+sub+"_fn"), w(s,"hc_"+sub+"_scale"), w(s,"hc_"+sub+"_base"),
                        c.norm_eps, c.sinkhorn_iterations, c.hc_eps);
                };
                auto am = mixes(x, "attn");
                auto a = g.norm(g.round(build_hc_collapse(ctx, x, pre)), w(s,"attn_norm"));
                auto qr = g.norm(g.linear(a, w(s,"attn_q_a")), w(s,"attn_q_a_norm"));
                auto q = g.rotate(ggml_reshape_3d(ctx, g.linear(qr, w(s,"attn_q_b")), hd, c.heads, n), positions);
                auto draft_kv = g.kv(a, positions, s, n);
                const int cached = windows[s].size() / hd;
                cache_inputs[s] = g.input(GGML_TYPE_F32, hd, cached, "dspark_window");
                auto kv = ggml_concat(ctx, cache_inputs[s], draft_kv, 1);
                auto mask = ggml_fill(ctx, ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cached + n, GGML_KQ_MASK_PAD), 0);
                auto attended = g.rotate(g.round(build_attention_core(ctx, q, kv, mask, w(s,"attn_sink"))), positions, true);
                auto grouped = ggml_reshape_3d(ctx, ggml_cont(ctx, attended), hd * c.heads / c.output_groups, c.output_groups, n);
                grouped = ggml_cont(ctx, ggml_permute(ctx, grouped, 0, 2, 1, 3));
                auto low = g.linear(grouped, w(s,"attn_output_a"), false, false);
                low = g.round(ggml_cont(ctx, ggml_permute(ctx, low, 0, 2, 1, 3)));
                auto attn_out = g.linear(ggml_reshape_2d(ctx, low, c.output_rank * c.output_groups, n), w(s,"attn_output_b"));
                x = g.round(build_hc_expand(ctx, attn_out, x, am));
                auto fm = mixes(x, "ffn");
                auto fi = g.norm(g.round(build_hc_collapse(ctx, x, am.pre)), w(s,"ffn_norm"));
                llm_build_cb cb = [](ggml_tensor * t, const char * name, int layer) { ggml_format_name(t, "dspark_%s-%d", name, layer); };
                auto routed = llm_build_context::llm_build_moe_ffn(ctx, build_context, fi,
                    ggml_cast(ctx, w(s,"ffn_gate_inp"), GGML_TYPE_F32), w(s,"ffn_up_exps"), w(s,"ffn_gate_exps"),
                    w(s,"ffn_down_exps"), w(s,"exp_probs_b"), c.experts, c.active_experts, LLM_FFN_SILU, true, true,
                    c.expert_scale, LLM_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS, cb, s, g.gf);
                auto shared = build_shared_expert(ctx, fi, w(s,"ffn_gate_shexp"), w(s,"ffn_up_shexp"), w(s,"ffn_down_shexp"));
                x = g.round(build_hc_expand(ctx, g.round(ggml_add(ctx, routed, shared)), x, fm)); pre = fm.pre;
            }
            auto hidden = g.norm(g.round(build_hc_collapse(ctx, x, pre)), w(2,"output_norm"));
            // Draft-only approximation: BF16 weights with F32 accumulation avoid
            // a second 2.5 GiB F32 target-head copy. Verification is unchanged.
            auto logits = g.linear(hidden, target.dsv41_tensors.at("output"), false, false);
            g.output(logits); g.allocate();
            std::vector<int32_t> ids(n, noise), pos(n); ids[0] = known; std::iota(pos.begin(), pos.end(), position);
            ggml_backend_tensor_set(tokens, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_tensor_set(positions, pos.data(), 0, pos.size() * sizeof(int32_t));
            for (int s = 0; s < 3; ++s) ggml_backend_tensor_set(cache_inputs[s], windows[s].data(), 0, windows[s].size() * sizeof(float));
            g.compute(); base_logits = g.read(logits);
        }
        std::vector<llama_token> result;
        // Each Markov bias depends on the previous sampled token.
        graph g(*this);
        auto tid = g.input(GGML_TYPE_I32, 1, 1, "dspark_markov_token");
        auto embed = ggml_get_rows(g.ctx.get(), w(2,"markov_embd"), tid);
        auto bias = g.linear(ggml_cast(g.ctx.get(), embed, GGML_TYPE_F32), w(2,"markov_head"), false, false);
        g.output(bias); g.allocate();
        for (int i = 0; i < block; ++i) {
            ggml_backend_tensor_set(tid, &known, 0, sizeof(known)); g.compute(); auto values = g.read(bias);
            for (size_t j = 0; j < values.size(); ++j) values[j] += base_logits[size_t(i) * c.vocabulary + j];
            known = std::max_element(values.begin(), values.end()) - values.begin(); result.push_back(known);
        }
        return result;
    }
};
dspark::dspark(llama_model & model, const std::string & path, int first, int last) : p(new impl(model, path, first, last)) {}
dspark::~dspark() = default;
const std::vector<int32_t> & dspark::target_layers() const { return p->layers; }
void dspark::ingest(const std::vector<float> & features, int position, int count) { p->ingest(features, position, count); }
std::vector<llama_token> dspark::propose(llama_token known, int position) { return p->propose(known, position); }
}
