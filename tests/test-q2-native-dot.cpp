#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "../ggml/src/ggml-quants.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static void packed_integer_reference() {
    constexpr std::array<int, 7> extrema = {-128,-127,-1,0,1,126,127};
    size_t cases = 0;
    for (int width : {64,640,2560}) for (int pattern = 0; pattern < 256; ++pattern)
    for (int shift = 0; shift < int(extrema.size()); ++shift) {
        std::vector<block_q2_0> w(width/QK2_0);
        std::vector<block_q8_0> x(width/QK8_0);
        for (size_t i = 0; i < w.size(); ++i) {
            w[i].d = ggml_fp32_to_fp16(i%2 ? -.03125f : .0625f);
            for (size_t j = 0; j < sizeof(w[i].qs); ++j) w[i].qs[j] = uint8_t(pattern+j*17+i*29);
        }
        for (size_t i = 0; i < x.size(); ++i) {
            x[i].d = ggml_fp32_to_fp16(i%3 ? .015625f : -.0078125f);
            for (int j = 0; j < QK8_0; ++j) x[i].qs[j] = int8_t(extrema[(j+shift+i)%extrema.size()]);
        }
        double expected = 0;
        for (int b = 0; b < width/QK2_0; ++b) for (int half = 0; half < 2; ++half) {
            int dot = 0;
            for (int j = 0; j < QK8_0; ++j)
                dot += (((w[b].qs[(half*QK8_0+j)/4] >> (2*(j%4))) & 3)-1)*int(x[2*b+half].qs[j]);
            expected += double(ggml_fp16_to_fp32(w[b].d))*ggml_fp16_to_fp32(x[2*b+half].d)*dot;
        }
        float actual = NAN;
        ggml_vec_dot_q2_0_q8_0(width, &actual, 0, w.data(), 0, x.data(), 0, 1);
        require(actual == float(expected), "Q2 integer dot changed packed codes or saturated signed extrema");
        ++cases;
    }
    std::printf("Q2/Q8 integer reference: %zu code/extreme/width cases exact\n", cases);
}

struct lease_source {
    const std::vector<block_q2_0> & weights;
    size_t span;
    size_t active = 0, acquired = 0, released = 0, rejected = 0;
    struct handle { std::vector<uint8_t> data; };
    static bool acquire(void * user, int layer, int expert, int component, ggml_expert_lease * out) {
        auto & s = *static_cast<lease_source *>(user);
        require(layer == 0 && component == 2 && expert >= 0 && expert < 12, "Q2 lease identity");
        if (s.active == 2) { ++s.rejected; return false; }
        auto h = std::make_unique<handle>();
        const auto * first = reinterpret_cast<const uint8_t *>(s.weights.data())+expert*s.span;
        h->data.assign(first,first+s.span);
        ++s.active; ++s.acquired;
        out->data = h->data.data(); out->size = h->data.size(); out->handle = h.release(); return true;
    }
    static void release(void * user, void * opaque) {
        auto & s = *static_cast<lease_source *>(user);
        std::unique_ptr<handle> h(static_cast<handle *>(opaque));
        require(s.active > 0, "Q2 lease accounting"); --s.active; ++s.released;
        std::fill(h->data.begin(),h->data.end(),0xff);
    }
};

static void indexed_matrix(bool native, int width, int rows, int input_slots, bool leased) {
    using context_ptr = std::unique_ptr<ggml_context, decltype(&ggml_free)>;
    using backend_ptr = std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)>;
    using buffer_ptr = std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>;
    context_ptr ctx(ggml_init({2<<20,nullptr,true}), ggml_free);
    backend_ptr cpu(ggml_backend_cpu_init(), ggml_backend_free);
    auto * w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q2_0, width, rows, 12);
    ggml_set_name(w,"blk.0.ffn_down_exps.weight");
    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, input_slots, 1);
    auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 10, 1);
    auto * y = ggml_mul_mat_id(ctx.get(), w, x, ids);
    auto * graph = ggml_new_graph_custom(ctx.get(), 16, false); ggml_build_forward_expand(graph, y);
    buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(),cpu.get()), ggml_backend_buffer_free);
    require(bool(buffer), "Q2 matrix allocation");
    std::vector<block_q2_0> weights(ggml_nbytes(w)/sizeof(block_q2_0));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i].d = ggml_fp32_to_fp16(.015625f*float(int(i%7)-3));
        for (size_t j = 0; j < sizeof(weights[i].qs); ++j) weights[i].qs[j] = uint8_t(i*53+j*29);
    }
    ggml_backend_tensor_set(w,weights.data(),0,ggml_nbytes(w));
    lease_source source{weights,w->nb[2]};
    if (leased) {
        ggml_backend_cpu_set_expert_lease_callbacks(cpu.get(),lease_source::acquire,lease_source::release,&source,true);
        std::vector<uint8_t> poison(ggml_nbytes(w),0xff);
        ggml_backend_tensor_set(w,poison.data(),0,poison.size());
    }
    std::vector<float> input(width*input_slots), actual(rows*10), reference(actual.size()), one_thread;
    const auto q8 = ggml_internal_get_type_traits(GGML_TYPE_Q8_0);
    std::vector<uint8_t> quant(ggml_row_size(GGML_TYPE_Q8_0,width)*input_slots);
    // A model top-k list contains each expert at most once per token.
    // CPU route grouping reserves one mapping per expert/token pair.
    const int32_t route[10] = {11,-1,0,9,1,-1,6,4,8,2};
    for (int round = 0; round < 2; ++round) {
        for (size_t i = 0; i < input.size(); ++i) input[i] = std::cos(float(i)*.031f+round)*1.7f;
        ggml_backend_tensor_set(x,input.data(),0,input.size()*sizeof(float));
        int32_t selected[10]; for (int i = 0; i < 10; ++i) selected[i] = route[(i+round)%10];
        ggml_backend_tensor_set(ids,selected,0,sizeof(selected));
        if (native) for (int i = 0; i < input_slots; ++i)
            q8.from_float(input.data()+i*width,quant.data()+i*ggml_row_size(GGML_TYPE_Q8_0,width),width);
        std::fill(reference.begin(),reference.end(),0.f);
        for (int slot = 0; slot < 10; ++slot) if (selected[slot] >= 0) for (int row = 0; row < rows; ++row) {
            const void * weight = reinterpret_cast<const uint8_t *>(weights.data())+selected[slot]*w->nb[2]+row*w->nb[1];
            if (native) ggml_vec_dot_q2_0_q8_0(width,&reference[slot*rows+row],0,weight,0,
                    quant.data()+(slot%input_slots)*ggml_row_size(GGML_TYPE_Q8_0,width),0,1);
            else ggml_vec_dot_q2_0_f32(width,&reference[slot*rows+row],0,weight,0,input.data()+(slot%input_slots)*width,0,1);
        }
        for (int threads : {1,4}) {
            ggml_backend_cpu_set_n_threads(cpu.get(),threads);
            std::vector<float> poison(actual.size(),123456.f); ggml_backend_tensor_set(y,poison.data(),0,poison.size()*sizeof(float));
            require(ggml_backend_graph_compute(cpu.get(),graph) == GGML_STATUS_SUCCESS,"Q2 matrix compute");
            require(source.active == 0 && source.acquired == source.released,"Q2 dangling expert lease");
            ggml_backend_tensor_get(y,actual.data(),0,actual.size()*sizeof(float));
            require(std::memcmp(actual.data(),reference.data(),actual.size()*sizeof(float)) == 0,
                    "Q2 activation workspace, masked routes or row dot differs");
            if (threads == 1) one_thread = actual;
            else require(std::memcmp(actual.data(),one_thread.data(),actual.size()*sizeof(float)) == 0,"Q2 thread parity");
        }
    }
    if (leased) require(source.acquired > 0 && source.rejected > 0,"Q2 bounded lease path not exercised");
    std::printf("Q2 indexed matrix: native=%d width=%d rows=%d input_slots=%d leased=%d masked/replayed/one-four-thread exact\n",
            int(native),width,rows,input_slots,int(leased));
}

int main(int argc,char ** argv) {
    try {
        const bool native = argc < 2 || std::strcmp(argv[1],"legacy") != 0;
#ifdef _WIN32
        _putenv_s("ESE_CPU_Q2_0_Q8_DOT",native ? "1":"0");
#else
        setenv("ESE_CPU_Q2_0_Q8_DOT",native ? "1":"0",1);
#endif
#ifdef _WIN32
        _putenv_s("ESE_CPU_EXPERT_LEASE_CHUNKING","1");
#else
        setenv("ESE_CPU_EXPERT_LEASE_CHUNKING","1",1);
#endif
        auto * init = ggml_init({1<<20,nullptr,false}); require(init,"Q2 initialization");
        const auto traits = ggml_internal_get_type_traits(GGML_TYPE_Q2_0);
        require(traits.vec_dot_type == (native ? GGML_TYPE_Q8_0:GGML_TYPE_F32),"Q2 opt-in activation policy");
        packed_integer_reference();
        for (bool leased : {false,true}) for (int slots : {1,10}) {
            indexed_matrix(native,64,9,slots,leased);
            indexed_matrix(native,640,2560,slots,leased);
        }
        ggml_free(init);
    } catch (const std::exception & error) { std::fprintf(stderr,"FAIL: %s\n",error.what()); return 1; }
}
