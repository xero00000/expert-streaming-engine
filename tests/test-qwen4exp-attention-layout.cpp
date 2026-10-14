#include "llama-context.h"
#include "llama-model.h"

#include <cstdio>
#include <cmath>
#include <stdexcept>

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

int main() {
    try {
        llama_model model{};
        model.arch = LLM_ARCH_LLAMA;
        model.hparams.n_layer = 0;
        model.hparams.n_embd = 4;
        model.hparams.n_vocab = 4;
        llama_context ctx(model);
        ctx.cparams = {};
        ctx.cparams.flash_attn = true;
        ctx.kv_self.size = 1024;
        ctx.kv_self.n = 1024;
        ctx.kv_self.cells.resize(1024);
        auto & cells = ctx.kv_self.cells;
        for (int i = 0; i < 12; ++i) {
            cells[100 + i * 3].pos = 19 - i;
            cells[100 + i * 3].seq_id.insert(0);
        }
        for (int i = 0; i < 257; ++i) {
            cells[400 + i].pos = i;
            cells[400 + i].seq_id.insert(1);
        }
        llama_batch batch{};
        batch.n_tokens = 3;
        llama_seq_id s0 = 0, s1 = 1;
        llama_seq_id * sequences[] = {&s0, &s0, &s1};
        int32_t counts[] = {1, 1, 1};
        llama_pos positions[] = {18, 19, 256};
        batch.seq_id = sequences; batch.n_seq_id = counts; batch.pos = positions;
        auto layout = llama_qwen4exp_attention_layout(ctx, batch);
        require(layout.size() == 2 && layout[0].first_query == 0 && layout[0].query_count == 2 &&
                layout[1].first_query == 2 && layout[1].query_count == 1, "query grouping");
        require(layout[0].key_count == 256 && layout[1].key_count == 512, "sequence-owned padded lengths");
        for (int i = 0; i < 12; ++i)
            require(cells[layout[0].physical_rows[i]].pos == 8 + i, "logical position order");
        // Physical extent and unrelated sequence growth cannot resize seq0.
        cells[1000].pos = 257; cells[1000].seq_id.insert(1);
        auto changed = llama_qwen4exp_attention_layout(ctx, batch);
        require(changed[0].physical_rows == layout[0].physical_rows && changed[0].key_count == layout[0].key_count,
                "other sequence changed the attention window");
        // Shared copied cells participate in each owner's own ordered window.
        for (int i = 0; i < 12; ++i) cells[100 + i * 3].seq_id.insert(2);
        batch.n_tokens = 1; batch.seq_id = nullptr; batch.n_seq_id = nullptr; batch.all_seq_id = 2;
        changed = llama_qwen4exp_attention_layout(ctx, batch);
        require(changed.size() == 1 && changed[0].sequence == 2 && changed[0].physical_rows == layout[0].physical_rows,
                "shared cells and implicit sequence ID");
        cells[100 + 5 * 3].seq_id.erase(2);
        changed = llama_qwen4exp_attention_layout(ctx, batch);
        require(changed[0].physical_rows.size() == 11 && changed[0].key_count == 256, "removed retained position");
        ctx.cparams.flash_attn = false;
        changed = llama_qwen4exp_attention_layout(ctx, batch);
        require(changed[0].key_count == 32, "non-FA padding");
        auto reserved = llama_qwen4exp_attention_layout(ctx, batch, 1024);
        require(reserved[0].key_count == 1024, "worst-case reservation shape");
        auto * tensors = ggml_init({1024 * 1024, nullptr, true});
        auto * backend = ggml_backend_cpu_init();
        require(tensors && backend, "host input fixture");
        auto input = changed[0];
        input.rows = ggml_new_tensor_2d(tensors, GGML_TYPE_I32, 32, 2);
        input.causal_mask = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 32, 16);
        input.block_pos = ggml_new_tensor_1d(tensors, GGML_TYPE_I32, 8);
        input.block_end = ggml_new_tensor_1d(tensors, GGML_TYPE_I32, 8);
        input.block_map = ggml_new_tensor_2d(tensors, GGML_TYPE_I32, 4, 8);
        input.tail = ggml_new_tensor_2d(tensors, GGML_TYPE_I32, 3, 1);
        input.tail_enabled = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 1, 1);
        auto * buffer = ggml_backend_alloc_ctx_tensors(tensors, backend);
        require(buffer, "host inputs allocation");
        ctx.qwen4exp_attention_inputs = {input};
        llama_qwen4exp_set_attention_inputs(ctx, batch);
        auto & filled = ctx.qwen4exp_attention_inputs[0];
        const auto * row_ids = static_cast<const int32_t *>(filled.rows->data);
        const auto * mask = static_cast<const float *>(filled.causal_mask->data);
        for (int h = 0; h < 2; ++h) for (int r = 0; r < 32; ++r)
            require(row_ids[h * 32 + r] == (r < 11 ? changed[0].physical_rows[r] : changed[0].physical_rows.front()),
                    "finite padding uses a valid sequence-owned row");
        for (int r = 0; r < 32 * 16; ++r)
            require(r < 10 ? mask[r] == 0 : std::isinf(mask[r]) && mask[r] < 0,
                    "causal mask excludes future and padded rows/queries");
        const auto * tail = static_cast<const int32_t *>(filled.tail->data);
        require(tail[0] == 8 && tail[1] == 9 && tail[2] == 8 &&
                *static_cast<const float *>(filled.tail_enabled->data) == 1,
                "partial tail padding must repeat a tail row, not select logical key zero");
        const auto * starts = static_cast<const int32_t *>(filled.block_pos->data);
        require(starts[0] == 8 && starts[1] == 12 && starts[2] == 17 && starts[3] == 0,
                "QSA RoPE uses retained logical token positions");
        for (int i = 0; i < 32; ++i)
            require(static_cast<const int32_t *>(filled.block_map->data)[i] == i, "logical block mapping");
        // Compression and physical reuse must preserve chronological block
        // membership. Equal rotary positions cannot expose future batch tokens.
        for (size_t i = 0; i < changed[0].physical_rows.size(); ++i) {
            auto & cell = cells[changed[0].physical_rows[i]];
            cell.qwen4exp_order = int64_t(i);
            cell.pos = int32_t(i / 2);
        }
        ctx.kv_self.head = changed[0].physical_rows[5];
        positions[0] = 2;
        llama_qwen4exp_set_attention_inputs(ctx, batch);
        auto compressed_layout = llama_qwen4exp_attention_layout(ctx, batch);
        require(compressed_layout[0].physical_rows == changed[0].physical_rows, "compressed chronology changed row order");
        for (int r = 0; r < 32; ++r)
            require(r < 6 ? mask[r] == 0 : std::isinf(mask[r]) && mask[r] < 0,
                    "compressed causal mask exposed later chronological tokens");
        require(starts[0] == 0 && starts[1] == 2 && starts[2] == 4,
                "compressed QSA block rotary positions");
        for (size_t i = 0; i < changed[0].physical_rows.size(); ++i)
            cells[changed[0].physical_rows[i]].qwen4exp_order = -1;
        ggml_backend_buffer_free(buffer); ggml_free(tensors); ggml_backend_free(backend);
        ctx.qwen4exp_attention_inputs.clear();
        cells[0].pos = cells[changed[0].physical_rows.front()].pos; cells[0].seq_id.insert(2);
        bool rejected = false;
        try { llama_qwen4exp_attention_layout(ctx, batch); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "duplicate logical positions must be rejected");
        std::puts("Qwen attention layout: ownership/order/copy/removal/masks/block positions/tails/padding/reservation/duplicate checks pass");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
