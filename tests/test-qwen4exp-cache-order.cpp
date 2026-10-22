#include "llama-qwen4exp-ops.h"
#include "ggml-backend.h"
#include "../ggml/src/ggml-quants.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#include <cuda.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

template<class T> static std::vector<T> read(const std::string & path, size_t count) {
    std::ifstream file(path, std::ios::binary);
    require(bool(file), "open captured operand");
    std::vector<T> data(count);
    file.read(reinterpret_cast<char *>(data.data()), count * sizeof(T));
    require(size_t(file.gcount()) == count * sizeof(T) && file.peek() == EOF, "captured operand size");
    return data;
}

static void packed_host_gather_case(ggml_backend_t compute, ggml_backend_t cpu, int width, int heads, int rows) {
    auto * leaves = ggml_init({2*1024*1024, nullptr, true});
    auto * ctx = ggml_init({2*1024*1024, nullptr, true});
    require(leaves && ctx, "packed host contexts");
    const int physical = rows*2+5;
    auto * storage = ggml_new_tensor_3d(leaves, GGML_TYPE_Q4_0, width+32, heads+1, physical+2);
    const size_t offset = storage->nb[2] + ggml_type_size(GGML_TYPE_Q4_0);
    auto * cache = ggml_view_3d(ctx, storage, width, physical, heads, storage->nb[2], storage->nb[1], offset);
    auto * id_storage = ggml_new_tensor_2d(leaves, GGML_TYPE_I32, rows+7, heads);
    auto * ids = ggml_view_2d(ctx, id_storage, rows, heads, id_storage->nb[1], 0);
    auto * packed = ggml_get_rows_ext(ctx, cache, ids, true, false);
    auto * reference = qwen4exp_build_cache_rows(ctx, cache, ids);
    auto * expanded = ggml_cast(ctx, packed, GGML_TYPE_F32);
    ggml_set_output(packed); ggml_set_output(reference); ggml_set_output(expanded);
    require(ggml_backend_supports_op(cpu, packed) && ggml_backend_supports_op(compute, expanded), "packed gather support");
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, reference); ggml_build_forward_expand(graph, expanded);
    auto * input_buffer = ggml_backend_alloc_ctx_tensors(leaves, cpu);
    require(input_buffer, "packed host inputs");
    ggml_backend_t backends[] = {compute, cpu};
    auto * sched = ggml_backend_sched_new(backends, nullptr, compute == cpu ? 1 : 2, 64, false);
    require(sched, "packed scheduler");
    ggml_backend_sched_set_tensor_backend(sched, packed, cpu);
    ggml_backend_sched_set_tensor_backend(sched, reference, cpu);
    ggml_backend_sched_set_tensor_backend(sched, expanded, compute);
    require(ggml_backend_sched_alloc_graph(sched, graph), "packed graph allocation");
    require(ggml_backend_sched_get_tensor_backend(sched, packed) == cpu &&
            ggml_backend_sched_get_tensor_backend(sched, expanded) == compute, "packed transfer placement");
    std::vector<float> values(ggml_nelements(storage)), actual(ggml_nelements(expanded)), expected(actual.size());
    std::vector<uint8_t> payload(ggml_nbytes(storage)), raw(ggml_nbytes(packed));
    std::vector<int32_t> order(ggml_nelements(id_storage));
    for (int round = 0; round < 3; ++round) {
        for (size_t i = 0; i < values.size(); ++i) values[i] = i % 97 < 32 ? 0.f : std::sin(float(i)*.031f + round)*3.f;
        ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), payload.data(), 0,
                storage->ne[1]*storage->ne[2], storage->ne[0], nullptr, nullptr);
        for (int head = 0; head < heads; ++head) for (int row = 0; row < rows+7; ++row)
            order[head*(rows+7)+row] = (physical-1-row*2+head+round*7+physical*2) % physical;
        if (rows > 1) for (int head = 0; head < heads; ++head) order[head*(rows+7)+rows-1] = order[head*(rows+7)];
        ggml_backend_tensor_set(storage, payload.data(), 0, payload.size());
        ggml_backend_tensor_set(id_storage, order.data(), 0, order.size()*sizeof(int32_t));
        require(ggml_backend_sched_graph_compute(sched, graph) == GGML_STATUS_SUCCESS, "packed replay compute");
        ggml_backend_tensor_get(reference, expected.data(), 0, expected.size()*sizeof(float));
        ggml_backend_tensor_get(expanded, actual.data(), 0, actual.size()*sizeof(float));
        require(std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)) == 0, "packed handoff changed dequantized row bits");
        ggml_backend_tensor_get(packed, raw.data(), 0, raw.size());
        const size_t row_bytes = ggml_row_size(GGML_TYPE_Q4_0, width);
        for (int head = 0; head < heads; ++head) for (int row = 0; row < rows; ++row) {
            const size_t source_offset = offset + order[head*(rows+7)+row]*cache->nb[1]+head*cache->nb[2];
            require(std::memcmp(raw.data()+(head*rows+row)*row_bytes, payload.data()+source_offset, row_bytes) == 0,
                    "packed handoff changed stored Q4 bytes");
        }
    }
    std::printf("packed_host_gather: width=%d heads=%d rows=%d three replays bitwise exact packed=%zu expanded=%zu\n",
            width, heads, rows, raw.size(), actual.size()*sizeof(float));
    ggml_backend_sched_free(sched); ggml_backend_buffer_free(input_buffer); ggml_free(ctx); ggml_free(leaves);
}

static void q4_copy_destination_replay(ggml_backend_t backend, int width, int heads, int rows) {
    auto * ctx = ggml_init({2*1024*1024, nullptr, true}); require(ctx, "Q4 copy context");
    auto * storage = ggml_new_tensor_3d(ctx, GGML_TYPE_Q4_0, width+32, heads+1, rows+2);
    const size_t source_offset = storage->nb[2] + ggml_type_size(GGML_TYPE_Q4_0);
    auto * source = ggml_view_3d(ctx, storage, width, rows, heads, storage->nb[2], storage->nb[1], source_offset);
    const size_t pitch = width+11, head_pitch = pitch*rows+13;
    auto * destination = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, head_pitch*heads+128);
    auto * view = ggml_view_3d(ctx, destination, width, rows, heads, pitch*sizeof(float), head_pitch*sizeof(float), 32*sizeof(float));
    auto * copy = ggml_cpy(ctx, source, view);
    auto * graph = ggml_new_graph_custom(ctx, 64, false); ggml_build_forward_expand(graph, copy);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend); require(buffer, "Q4 copy allocation");
    std::vector<float> values(ggml_nelements(storage)), actual(ggml_nelements(destination)), expected(actual.size()), decoded(width);
    std::vector<uint8_t> payload(ggml_nbytes(storage));
    for (int round = 0; round < 3; ++round) {
        for (size_t i = 0; i < values.size(); ++i) values[i] = std::sin(float(i)*.019f+round)*7.f;
        ggml_quantize_chunk(GGML_TYPE_Q4_0, values.data(), payload.data(), 0,
                storage->ne[1]*storage->ne[2], storage->ne[0], nullptr, nullptr);
        ggml_backend_tensor_set(storage, payload.data(), 0, payload.size());
        std::fill(expected.begin(), expected.end(), -9999.25f);
        ggml_backend_tensor_set(destination, expected.data(), 0, expected.size()*sizeof(float));
        const size_t offset = 32+round*5;
        view->data = static_cast<float *>(destination->data)+offset;
        copy->data = view->data;
        for (int head = 0; head < heads; ++head) for (int row = 0; row < rows; ++row) {
            dequantize_row_q4_0(reinterpret_cast<const block_q4_0 *>(
                    payload.data()+source_offset+row*source->nb[1]+head*source->nb[2]), decoded.data(), width);
            std::copy(decoded.begin(), decoded.end(), expected.begin()+offset+head*head_pitch+row*pitch);
        }
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Q4 copy replay compute");
        ggml_backend_tensor_get(destination, actual.data(), 0, actual.size()*sizeof(float));
        require(std::memcmp(actual.data(), expected.data(), actual.size()*sizeof(float)) == 0,
                "Q4 copy changed row bits, guard bytes or captured destination");
    }
    std::printf("q4_copy_replay: width=%d heads=%d rows=%d strided rows/heads and changed destinations exact\n", width, heads, rows);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

static void gather_case(ggml_backend_t backend, ggml_type type, int width, int heads, int rows) {
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx, "context");
    // Padded rows and heads, plus a nonzero view offset. This covers actual
    // token-interleaved cache strides rather than assuming head-contiguous data.
    const int physical = 2 * rows + 5;
    const size_t size = ggml_type_size(type);
    auto * storage = ggml_new_tensor_3d(ctx, type, width + 8, heads + 1, physical + 2);
    const size_t view_offset = storage->nb[2] + (rows == 17 ? size : 0);
    auto * cache = ggml_view_3d(ctx, storage, width, physical, heads,
            storage->nb[2], storage->nb[1], view_offset);
    auto * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, rows, heads);
    auto * packed = qwen4exp_build_cache_rows(ctx, cache, ids);
    require(packed->type == type && packed->ne[0] == width && packed->ne[1] == rows &&
            packed->ne[2] == heads, "gather geometry and type");
    auto * graph = ggml_new_graph_custom(ctx, 64, false);
    ggml_build_forward_expand(graph, packed);
    require(ggml_backend_supports_op(backend, packed), "native gather support");
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "buffer");
    std::vector<int32_t> order(rows);
    for (int r = 0; r < rows; ++r) order[r] = 2 * (rows - 1 - r) + 1;
    // Include an exact repeated row, as needed for finite masked padding.
    if (rows > 2) order.back() = order.front();
    std::vector<float> floats(ggml_nelements(storage));
    for (size_t i = 0; i < floats.size(); ++i) floats[i] = std::sin(float(i) * .031f);
    for (size_t i = 0; i < floats.size(); i += 11) floats[i] = -0.0f;
    std::vector<ggml_fp16_t> halves(floats.size());
    for (size_t i = 0; i < floats.size(); ++i) halves[i] = ggml_fp32_to_fp16(floats[i]);
    const auto * source = type == GGML_TYPE_F16 ? reinterpret_cast<const unsigned char *>(halves.data()) :
                                                reinterpret_cast<const unsigned char *>(floats.data());
    ggml_backend_tensor_set(storage, source, 0, ggml_nbytes(storage));
    for (int h = 0; h < heads; ++h)
        ggml_backend_tensor_set(ids, order.data(), h * order.size() * sizeof(int32_t), order.size() * sizeof(int32_t));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "gather compute");
    std::vector<unsigned char> got(ggml_nbytes(packed));
    ggml_backend_tensor_get(packed, got.data(), 0, got.size());
    for (int h = 0; h < heads; ++h) for (int r = 0; r < rows; ++r) {
        const size_t offset = view_offset + order[r] * cache->nb[1] + h * cache->nb[2];
        require(std::memcmp(got.data() + (size_t(h) * rows + r) * width * size,
                source + offset, width * size) == 0, "stored row bits changed");
    }
    std::printf("cache_order: type=%s width=%d heads=%d rows=%d exact=true bytes=%zu\n",
            ggml_type_name(type), width, heads, rows, got.size());
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void encode(ggml_type type, const float * input, void * output, int64_t count) {
    switch (type) {
        case GGML_TYPE_BF16: ggml_fp32_to_bf16_row_ref(input, static_cast<ggml_bf16_t *>(output), count); break;
        case GGML_TYPE_Q4_0: quantize_row_q4_0_ref(input, static_cast<block_q4_0 *>(output), count); break;
        case GGML_TYPE_Q4_1: quantize_row_q4_1_ref(input, static_cast<block_q4_1 *>(output), count); break;
        case GGML_TYPE_Q5_0: quantize_row_q5_0_ref(input, static_cast<block_q5_0 *>(output), count); break;
        case GGML_TYPE_Q5_1: quantize_row_q5_1_ref(input, static_cast<block_q5_1 *>(output), count); break;
        case GGML_TYPE_Q8_0: quantize_row_q8_0_ref(input, static_cast<block_q8_0 *>(output), count); break;
        default: throw std::runtime_error("unexpected test codec");
    }
}

static void decode(ggml_type type, const void * input, float * output, int64_t count) {
    switch (type) {
        case GGML_TYPE_BF16:
            for (int64_t i = 0; i < count; ++i) {
                // Independent representation decode: BF16 is the high 16 bits of F32.
                uint16_t stored; std::memcpy(&stored, static_cast<const uint8_t *>(input) + 2*i, 2);
                uint32_t bits = uint32_t(stored) << 16; std::memcpy(output+i, &bits, 4);
            }
            break;
        case GGML_TYPE_Q4_0: dequantize_row_q4_0(static_cast<const block_q4_0 *>(input), output, count); break;
        case GGML_TYPE_Q4_1: dequantize_row_q4_1(static_cast<const block_q4_1 *>(input), output, count); break;
        case GGML_TYPE_Q5_0: dequantize_row_q5_0(static_cast<const block_q5_0 *>(input), output, count); break;
        case GGML_TYPE_Q5_1: dequantize_row_q5_1(static_cast<const block_q5_1 *>(input), output, count); break;
        case GGML_TYPE_Q8_0: dequantize_row_q8_0(static_cast<const block_q8_0 *>(input), output, count); break;
        default: throw std::runtime_error("unexpected test decoder");
    }
}

static void validate_formats() {
    const std::vector<ggml_type> supported = {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_BF16,
        GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0};
    std::string error;
    for (int index = 0; index <= GGML_TYPE_COUNT; ++index) {
        auto type = static_cast<ggml_type>(index);
        const bool listed = std::find(supported.begin(), supported.end(), type) != supported.end();
        require(qwen4exp_cache_type_supported(type) == listed, "cache format contract");
        for (int width : {-1, 0, 1, 31, 32, 63, 64, 128, 256}) for (bool value : {false, true})
            for (bool flash : {false, true}) {
                const bool packed = listed && index != GGML_TYPE_F16 && index != GGML_TYPE_F32 && index != GGML_TYPE_BF16;
                const bool expected = listed && width > 0 && (!packed || width % 32 == 0) && (!value || !packed || flash);
                require(qwen4exp_validate_cache_type(type, width, value, flash, error) == expected,
                    "cache format/width/attention preflight");
                require(expected ? error.empty() : !error.empty(), "cache preflight diagnostic");
            }
    }
    std::puts("cache_format_preflight: all enum formats, width alignment and quantized value policy checked");
}

static void gather_converted_case(ggml_backend_t backend, ggml_type type, int width, int heads, int rows) {
    auto * ctx=ggml_init({4*1024*1024,nullptr,true});require(ctx,"Q8 context");
    const int physical=2*rows+5;
    auto * storage=ggml_new_tensor_3d(ctx,type,width+64,heads+1,physical+2);
    const size_t offset=storage->nb[2]+(rows==17?ggml_type_size(type):0);
    auto * cache=ggml_view_3d(ctx,storage,width,physical,heads,storage->nb[2],storage->nb[1],offset);
    auto * ids=ggml_new_tensor_2d(ctx,GGML_TYPE_I32,rows,heads);
    auto * gathered=qwen4exp_build_cache_rows(ctx,cache,ids);
    require(gathered->type==GGML_TYPE_F32 && gathered->ne[0]==width && gathered->ne[1]==rows &&
            gathered->ne[2]==heads,"Q8 gathered geometry/type");
    auto * graph=ggml_new_graph_custom(ctx,64,false);ggml_build_forward_expand(graph,gathered);
    require(ggml_backend_supports_op(backend,gathered),"native Q8 gather support");
    auto * buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer,"Q8 buffer");
    std::vector<float> source(ggml_nelements(storage));
    for(size_t i=0;i<source.size();++i)source[i]=i%11==0?-0.f:std::sin(float(i)*.031f);
    std::vector<uint8_t> encoded(ggml_nbytes(storage));
    encode(type,source.data(),encoded.data(),source.size());
    ggml_backend_tensor_set(storage,encoded.data(),0,encoded.size());
    std::vector<int32_t> order(rows);
    for(int r=0;r<rows;++r)order[r]=2*(rows-1-r)+1;
    if(rows>2)order.back()=order.front();
    for(int h=0;h<heads;++h)ggml_backend_tensor_set(ids,order.data(),h*order.size()*sizeof(int32_t),order.size()*sizeof(int32_t));
    require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS,"Q8 gather compute");
    std::vector<float> actual(ggml_nelements(gathered)),expected(width);
    ggml_backend_tensor_get(gathered,actual.data(),0,actual.size()*sizeof(float));
    for(int h=0;h<heads;++h)for(int r=0;r<rows;++r) {
        const size_t start=offset+size_t(order[r])*cache->nb[1]+size_t(h)*cache->nb[2];
        decode(type,encoded.data()+start,expected.data(),width);
        require(std::memcmp(actual.data()+(size_t(h)*rows+r)*width,expected.data(),width*sizeof(float))==0,
                "Q8 gather differs from independent packed-row decoder");
    }
    std::vector<uint8_t> retained(encoded.size());ggml_backend_tensor_get(storage,retained.data(),0,retained.size());
    require(retained==encoded,"Q8 gather modified persistent bytes");
    std::printf("cache_order_converted: type=%s width=%d heads=%d rows=%d oracle_exact=true source_exact=true\n",
        ggml_type_name(type),width,heads,rows);
    ggml_backend_buffer_free(buffer);ggml_free(ctx);
}

struct captured {
    std::vector<float> q;
    std::vector<ggml_fp16_t> k, v, mask;
    std::vector<int32_t> rows;
};

static captured capture(const std::string & directory) {
    captured result;
    result.q = read<float>(directory + "/fa-3.src0.raw", 256 * 24);
    result.k = read<ggml_fp16_t>(directory + "/fa-3.src1.raw", 256 * 256 * 2);
    result.v = read<ggml_fp16_t>(directory + "/fa-3.src2.raw", 256 * 256 * 2);
    result.mask = read<ggml_fp16_t>(directory + "/fa-3.src3.raw", 256 * 16);
    for (int row = 0; row < 256; ++row) {
        if (std::isfinite(ggml_fp16_to_fp32(result.mask[row]))) result.rows.push_back(row);
    }
    require(!result.rows.empty(), "visible captured rows");
    return result;
}

static void captured_attention(ggml_backend_t backend, const std::string & reference, const std::string & actual) {
    constexpr int width = 256, keys = 256, heads = 24, kvheads = 2;
    auto a = capture(reference), b = capture(actual);
    require(a.q == b.q && a.rows.size() == b.rows.size(), "logical queries and key count");
    auto * ctx = ggml_init({8 * 1024 * 1024, nullptr, true});
    require(ctx, "attention context");
    auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, width, 1, heads);
    struct tensors { ggml_tensor * k, * v, * mask, * ids, * packed_k, * packed_v, * direct, * ordered; } inputs[2];
    auto * canonical_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, 16);
    auto * graph = ggml_new_graph_custom(ctx, 128, false);
    for (auto & t : inputs) {
        t.k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, keys, kvheads);
        t.v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, width, keys, kvheads);
        t.mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, keys, 16);
        t.ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, keys, kvheads);
        t.packed_k = qwen4exp_build_cache_rows(ctx, t.k, t.ids);
        t.packed_v = qwen4exp_build_cache_rows(ctx, t.v, t.ids);
        t.direct = ggml_flash_attn_ext(ctx, q, t.k, t.v, t.mask, .0625f, 0, 0);
        t.ordered = ggml_flash_attn_ext(ctx, q, t.packed_k, t.packed_v, canonical_mask, .0625f, 0, 0);
        require(ggml_backend_supports_op(backend, t.direct) && ggml_backend_supports_op(backend, t.ordered),
                "native attention support");
        ggml_build_forward_expand(graph, t.direct);
        ggml_build_forward_expand(graph, t.ordered);
    }
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer, "attention allocation");
    ggml_backend_tensor_set(q, a.q.data(), 0, a.q.size() * sizeof(float));
    std::vector<ggml_fp16_t> mask(keys * 16, ggml_fp32_to_fp16(-INFINITY));
    std::fill_n(mask.begin(), a.rows.size(), ggml_fp32_to_fp16(0));
    ggml_backend_tensor_set(canonical_mask, mask.data(), 0, mask.size() * sizeof(ggml_fp16_t));
    captured * captures[] = {&a, &b};
    for (int i = 0; i < 2; ++i) {
        auto & data = *captures[i]; auto & t = inputs[i];
        ggml_backend_tensor_set(t.k, data.k.data(), 0, data.k.size() * 2);
        ggml_backend_tensor_set(t.v, data.v.data(), 0, data.v.size() * 2);
        ggml_backend_tensor_set(t.mask, data.mask.data(), 0, data.mask.size() * 2);
        std::vector<int32_t> ids(keys, data.rows.front());
        std::copy(data.rows.begin(), data.rows.end(), ids.begin());
        for (int h = 0; h < kvheads; ++h)
            ggml_backend_tensor_set(t.ids, ids.data(), h * ids.size() * sizeof(int32_t), ids.size() * sizeof(int32_t));
    }
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "attention compute");
    std::vector<float> direct[2], ordered[2];
    std::vector<ggml_fp16_t> pk[2], pv[2];
    for (int i = 0; i < 2; ++i) {
        direct[i].resize(width * heads); ordered[i].resize(width * heads);
        pk[i].resize(width * keys * kvheads); pv[i].resize(pk[i].size());
        ggml_backend_tensor_get(inputs[i].direct, direct[i].data(), 0, direct[i].size() * 4);
        ggml_backend_tensor_get(inputs[i].ordered, ordered[i].data(), 0, ordered[i].size() * 4);
        ggml_backend_tensor_get(inputs[i].packed_k, pk[i].data(), 0, pk[i].size() * 2);
        ggml_backend_tensor_get(inputs[i].packed_v, pv[i].data(), 0, pv[i].size() * 2);
    }
    require(pk[0] == pk[1] && pv[0] == pv[1], "canonical K/V stored bits");
    require(std::memcmp(ordered[0].data(), ordered[1].data(), ordered[0].size() * 4) == 0,
            "canonical attention depends on original placement");
    size_t changed = 0; double maximum = 0, oracle_error = 0;
    for (size_t i = 0; i < direct[0].size(); ++i) {
        changed += std::memcmp(&direct[0][i], &direct[1][i], 4) != 0;
        maximum = std::max(maximum, double(std::fabs(direct[0][i] - direct[1][i])));
    }
    // Independent dense F64 oracle on the original visible half-precision rows.
    for (int h = 0; h < heads; ++h) {
        std::vector<double> scores(a.rows.size()); double top = -INFINITY, total = 0;
        const int kh = h / (heads / kvheads);
        for (size_t r = 0; r < a.rows.size(); ++r) {
            double dot = 0;
            for (int c = 0; c < width; ++c)
                dot += double(a.q[h * width + c]) * ggml_fp16_to_fp32(a.k[(kh * keys + a.rows[r]) * width + c]);
            scores[r] = dot * .0625; top = std::max(top, scores[r]);
        }
        for (auto & score : scores) { score = std::exp(score - top); total += score; }
        for (int c = 0; c < width; ++c) {
            double expected = 0;
            for (size_t r = 0; r < a.rows.size(); ++r)
                expected += scores[r] * ggml_fp16_to_fp32(a.v[(kh * keys + a.rows[r]) * width + c]);
            expected /= total;
            const double error = std::fabs(ordered[0][h * width + c] - expected);
            require(std::isfinite(ordered[0][h * width + c]) && error <= .002 * (1 + std::fabs(expected)),
                    "canonical attention disagrees with independent oracle");
            oracle_error = std::max(oracle_error, error);
        }
    }
    std::printf("captured_attention: visible=%zu physical_changed=%zu physical_max=%.12g ordered_exact=true oracle_max=%.12g\n",
            a.rows.size(), changed, maximum, oracle_error);
    require(changed > 0, "captured physical-placement negative control did not reproduce");
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
}

int main(int argc, char ** argv) {
    try {
        ggml_backend_t backend = nullptr;
#ifdef GGML_USE_CUDA
        if (argc > 1 && std::string(argv[1]) != "cpu") {
            const int device = std::stoi(argv[1]);
            require(cuInit(0) == CUDA_SUCCESS, "driver init");
            CUdevice dev; CUcontext context;
            require(cuDeviceGet(&dev, device) == CUDA_SUCCESS && cuDevicePrimaryCtxRetain(&context, dev) == CUDA_SUCCESS &&
                    cuCtxSetCurrent(context) == CUDA_SUCCESS, "primary context");
            backend = ggml_backend_cuda_init(device, nullptr, nullptr);
            require(backend, "CUDA backend requested");
        }
#endif
        if (!backend) backend = ggml_backend_cpu_init();
        require(backend, "backend");
        auto * host_backend = ggml_backend_is_cpu(backend) ? backend : ggml_backend_cpu_init();
        require(host_backend, "packed gather CPU backend");
        for (int width : {32, 256}) for (int heads : {1, 2}) for (int rows : {1, 17, 256, 2304})
            packed_host_gather_case(backend, host_backend, width, heads, rows);
        if (host_backend != backend) ggml_backend_free(host_backend);
        q4_copy_destination_replay(backend, 32, 1, 1);
        q4_copy_destination_replay(backend, 256, 2, 17);
        q4_copy_destination_replay(backend, 256, 2, 2304);
        validate_formats();
        for (auto type : {GGML_TYPE_F16, GGML_TYPE_F32}) for (int width : {63, 64, 256})
            for (int heads : {1, 2, 4}) for (int rows : {1, 17, 256})
                gather_case(backend, type, width, heads, rows);
        for(auto type:{GGML_TYPE_BF16,GGML_TYPE_Q4_0,GGML_TYPE_Q4_1,GGML_TYPE_Q5_0,GGML_TYPE_Q5_1,GGML_TYPE_Q8_0})
            for(int width:{32,64,256})for(int heads:{1,2,4})for(int rows:{1,17,256})
                gather_converted_case(backend,type,width,heads,rows);
        gather_converted_case(backend,GGML_TYPE_BF16,63,2,17);
        if (argc == 4) captured_attention(backend, argv[2], argv[3]);
        else require(argc <= 2, "usage: test-qwen4exp-cache-order [cpu|device] [reference-directory actual-directory]");
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
