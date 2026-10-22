#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-quants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void run(bool merged, int slots) {
    constexpr int width = 256, rows = 64, experts = 3, routes = 10, tokens = 3;
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init({2<<20,nullptr,true}),ggml_free);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(ggml_backend_cpu_init(),ggml_backend_free);
    auto * up = ggml_new_tensor_3d(ctx.get(),GGML_TYPE_IQ4_XS,width,merged ? 2*rows : rows,experts);
    auto * gate = merged ? nullptr : ggml_new_tensor_3d(ctx.get(),GGML_TYPE_IQ4_XS,width,rows,experts);
    auto * x = ggml_new_tensor_3d(ctx.get(),GGML_TYPE_F32,width,slots,tokens);
    auto * ids = ggml_new_tensor_2d(ctx.get(),GGML_TYPE_I32,routes,tokens);
    auto * y = ggml_moe_up_gate(ctx.get(),up,gate,x,ids,GGML_UNARY_OP_SILU);
    auto * graph = ggml_new_graph_custom(ctx.get(),16,false); ggml_build_forward_expand(graph,y);
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(
            ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()),ggml_backend_buffer_free);
    require(bool(buffer),"fused route allocation");
    const auto upload = [&](ggml_tensor * tensor, bool gate_only) {
        std::vector<block_iq4_xs> weights(ggml_nbytes(tensor)/sizeof(block_iq4_xs));
        for (int e = 0; e < experts; ++e) for (int row = 0; row < tensor->ne[1]; ++row) {
            auto & block = weights[e*tensor->ne[1]+row];
            const bool is_gate = gate_only || (merged && row < rows);
            block.d = ggml_fp32_to_fp16(std::ldexp(1.f/1024,e+row%rows%3+(is_gate ? 1 : 0)));
            block.scales_h = 0xaaaa;
            std::fill(std::begin(block.scales_l),std::end(block.scales_l),0x11);
            std::fill(std::begin(block.qs),std::end(block.qs),0x88);
            // IQ4_XS code 8 is 1, scale code 33 is 1: the whole row
            // equals d, giving an independent constant-weight oracle.
        }
        ggml_backend_tensor_set(tensor,weights.data(),0,ggml_nbytes(tensor));
    };
    upload(up,false); if (gate) upload(gate,true);
    const std::array<int32_t,routes> pattern = {2,-1,0,2,1,0,0,2,1,2};
    std::vector<float> input(width*slots*tokens), expected(rows*routes*tokens), actual(expected.size()), single;
    std::vector<int32_t> selected(routes*tokens);
    for (int replay = 0; replay < 3; ++replay) {
        for (int column = 0; column < slots*tokens; ++column)
            std::fill_n(input.begin()+column*width,width,(column+replay)%2 ? -127.f/128 : 127.f/128);
        for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < routes; ++slot)
            selected[token*routes+slot] = pattern[(slot+token+replay)%routes];
        selected.back() = experts; // positive out-of-range route is masked too
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        ggml_backend_tensor_set(ids,selected.data(),0,selected.size()*sizeof(int32_t));
        std::fill(expected.begin(),expected.end(),0.f);
        for (int token = 0; token < tokens; ++token) for (int slot = 0; slot < routes; ++slot) {
            const int e = selected[token*routes+slot];
            if (e < 0 || e >= experts) continue;
            for (int row = 0; row < rows; ++row) {
                const float u = width*std::ldexp(1.f/1024,e+row%3)*input[(token*slots+slot%slots)*width];
                const float g = 2*u;
                expected[(token*routes+slot)*rows+row] = u*(g/(1+std::exp(-g)));
            }
        }
        for (int threads : {1,4}) {
            ggml_backend_cpu_set_n_threads(cpu.get(),threads);
            std::vector<float> poison(actual.size(),123456.f);
            ggml_backend_tensor_set(y,poison.data(),0,poison.size()*sizeof(float));
            require(ggml_backend_graph_compute(cpu.get(),graph) == GGML_STATUS_SUCCESS,"fused route compute");
            ggml_backend_tensor_get(y,actual.data(),0,actual.size()*sizeof(float));
            for (size_t i = 0; i < actual.size(); ++i)
                require(std::isfinite(actual[i]) && std::fabs(actual[i]-expected[i]) <= 2e-5f*(1+std::fabs(expected[i])),
                        "fused duplicate/masked/per-token route differs from analytical oracle");
            if (threads == 1) single = actual;
            else require(std::memcmp(single.data(),actual.data(),actual.size()*sizeof(float)) == 0,"fused route thread parity");
        }
    }
    std::printf("PASS: fused native routes merged=%d input_slots=%d duplicate/masked/batched/replay/analytical/thread-parity\n",int(merged),slots);
}

int main() {
    try {
        auto * init = ggml_init({1<<20,nullptr,false}); require(init,"quantizer initialization");
        for (bool merged : {false,true}) for (int slots : {1,10}) run(merged,slots);
        ggml_free(init);
    } catch (const std::exception & error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
