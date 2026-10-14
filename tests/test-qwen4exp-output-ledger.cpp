#include "../src/llama-context.h"
#include "../src/llama-model.h"
#include "../src/llama-qwen4exp-ple.h"
#include "ggml-backend.h"
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <stdexcept>
#include <vector>

static void check(bool result, const char * message) {
    if (!result) throw std::runtime_error(message);
}

static void initialize(llama_context & ctx, unsigned slots) {
    ctx.cparams = {};
    ctx.cparams.n_seq_max = slots;
    ctx.n_ctx_max = 32;
    ctx.cparams.n_batch = 4;
    auto * backend = ggml_backend_cpu_init();
    check(backend != nullptr, "CPU backend");
    ctx.backends.push_back(backend);
    ctx.sched = ggml_backend_sched_new(ctx.backends.data(), nullptr, 1, 64, false);
    check(ctx.sched != nullptr, "scheduler");
    ctx.buf_output = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), slots * 4 * sizeof(float));
    check(ctx.buf_output != nullptr, "output allocation");
    ctx.logits = static_cast<float *>(ggml_backend_buffer_get_base(ctx.buf_output));
    ctx.logits_size = slots * 4;
    ctx.output_size = slots;
    ctx.output_ids = {0, -1, -1, -1};
    ctx.n_outputs = 1;
    std::fill_n(ctx.logits, ctx.logits_size, 99.f);
    qwen4exp_ple_hash_config config;
    config.head_vocab_sizes.fill(17);
    for (size_t i = 0; i < config.head_offsets.size(); ++i) config.head_offsets[i] = i * 17;
    config.layer_multipliers = {3, 5, 7};
    ctx.qwen4exp_ple_sequences = std::make_unique<qwen4exp_ple_sequence_state>(config);
}

static std::vector<uint8_t> save(llama_context & ctx) {
    std::vector<uint8_t> bytes(llama_state_get_size(&ctx));
    check(!bytes.empty(), "full-session size");
    check(llama_state_get_data(&ctx, bytes.data(), bytes.size()) == bytes.size(), "full-session save");
    return bytes;
}

static void replace_u32(std::vector<uint8_t> & bytes, size_t offset, uint32_t value) {
    check(offset + sizeof(value) <= bytes.size(), "fixture patch bounds");
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

int main() {
    try {
        llama_model model{};
        model.arch = LLM_ARCH_LLAMA;
        model.hparams.n_vocab = 4;
        model.hparams.n_embd = 4;
        model.hparams.n_layer = 0; // Test public session I/O without weights or loader opt-in.
        llama_context source(model);
        initialize(source, 3);
        model.arch = LLM_ARCH_QWEN4EXP;
        source.qwen4exp_last_logit_pos = {7, 11, -1};
        source.qwen4exp_last_logits = {1.f, -2.f, 3.f, -0.f, 11.f, 12.f, -13.f, 14.f,
                                      77.f, 77.f, 77.f, 77.f};
        const auto bytes = save(source);
        const uint32_t magic = 0x514a4f31;
        size_t marker = bytes.size();
        unsigned markers = 0;
        for (size_t i = 0; i + sizeof(magic) <= bytes.size(); ++i) {
            uint32_t value;
            std::memcpy(&value, bytes.data() + i, sizeof(value));
            if (value == magic) { marker = i; ++markers; }
        }
        check(markers == 1, "unique Qwen session version marker");
        for (unsigned slots : {3u, 4u}) {
            model.arch = LLM_ARCH_LLAMA;
            llama_context target(model);
            initialize(target, slots);
            model.arch = LLM_ARCH_QWEN4EXP;
            target.qwen4exp_last_logit_pos.assign(slots, 100);
            target.qwen4exp_last_logits.assign(slots * 4, -99.f);
            check(llama_state_set_data(&target, bytes.data(), bytes.size()) == bytes.size(), "full-session restore");
            check(target.qwen4exp_last_logit_pos[0] == 7 && target.qwen4exp_last_logit_pos[1] == 11,
                  "sequence-owned output positions");
            check(std::memcmp(target.qwen4exp_last_logits.data(), source.qwen4exp_last_logits.data(),
                              8 * sizeof(float)) == 0, "sequence-owned logits, including signed zero");
            check(std::all_of(target.qwen4exp_last_logit_pos.begin() + 2,
                              target.qwen4exp_last_logit_pos.end(), [](llama_pos pos) { return pos == -1; }),
                  "inactive and additional sequence positions");
            check(std::all_of(target.qwen4exp_last_logits.begin() + 8,
                              target.qwen4exp_last_logits.end(), [](float value) { return value == 0.f; }),
                  "inactive sequence outputs were not restored from unrelated bytes");
            check(target.n_outputs == 1 && target.logits[0] == 99.f, "global output remains independent");
            if (slots == 3) check(save(target) == bytes, "byte-exact session roundtrip");
        }
        // Reject malformed ownership records, including the previous session layout.
        std::vector<std::vector<uint8_t>> invalid;
        for (auto patch : std::array<std::pair<size_t, uint32_t>, 4>{{
                 {marker, 0}, {marker + 4, 4}, {marker + 8, 5}, {marker + 12, uint32_t(-2)}}}) {
            auto altered = bytes;
            replace_u32(altered, patch.first, patch.second);
            invalid.push_back(std::move(altered));
        }
        invalid.emplace_back(bytes.begin(), bytes.begin() + marker + 15);
        auto legacy = bytes;
        const size_t ownership_bytes = 12 + 3 * sizeof(llama_pos) + 8 * sizeof(float);
        legacy.erase(legacy.begin() + marker, legacy.begin() + marker + ownership_bytes);
        invalid.push_back(std::move(legacy));
        for (const auto & altered : invalid) {
            model.arch = LLM_ARCH_LLAMA;
            llama_context target(model);
            initialize(target, 3);
            model.arch = LLM_ARCH_QWEN4EXP;
            check(llama_state_set_data(&target, altered.data(), altered.size()) == 0,
                  "malformed or old Qwen output ownership accepted");
        }
        {
            model.arch = LLM_ARCH_LLAMA;
            llama_context too_small(model);
            initialize(too_small, 2);
            model.arch = LLM_ARCH_QWEN4EXP;
            check(llama_state_set_data(&too_small, bytes.data(), bytes.size()) == 0, "sequence capacity mismatch accepted");
        }
        // Base-only journal restore must retain the latest chronological
        // position even when an older token has a larger rotary position.
        for (const auto positions : std::array<std::array<llama_pos, 8>, 3>{{
                {{8, 9, 10, 11, 4, 5, 6, 7}},
                {{0, 0, 1, 1, 2, 2, 3, 3}},
                {{3, 4, 5, 6, 7, 8, 9, 10}}}}) {
            model.arch = LLM_ARCH_LLAMA;
            llama_context journal(model);
            initialize(journal, 3);
            model.arch = LLM_ARCH_QWEN4EXP;
            auto & kv = journal.kv_self;
            kv.size = kv.n = 16; kv.cells.resize(16); kv.used = 8;
            for (int i = 0; i < 8; ++i) {
                // Physical order differs from chronological order.
                auto & cell = kv.cells[7-i];
                cell.pos = positions[i]; cell.qwen4exp_order = i; cell.seq_id.insert(0);
            }
            auto * tensors = ggml_init({4096, nullptr, true});
            check(tensors, "journal tensor context"); kv.ctxs.push_back(tensors);
            auto * state = ggml_new_tensor_2d(tensors, GGML_TYPE_F32, 2, 3);
            auto * buffer = ggml_backend_alloc_ctx_tensors(tensors, journal.backends[0]);
            check(buffer, "journal state allocation"); kv.bufs.push_back(buffer); kv.s_l = {state};
            ggml_backend_sched_set_tensor_backend(journal.sched, state, journal.backends[0]);
            const std::array<float, 6> base_state{{1.f, -0.f, 3.f, 4.f, 5.f, 6.f}};
            ggml_backend_tensor_set(state, base_state.data(), 0, sizeof(base_state));
            check(kv.checkpoint_alloc_shadows(false, 3), "journal shadow allocation");
            auto & ckpt = kv.ckpt;
            ckpt.selected_spec_mode = LLAMA_SPEC_CKPT_PER_STEP;
            ckpt.per_step_max_allocated = 2;
            ckpt.qwen4exp_prefix_state = {state}; // Base-only restore uses the separate shadow.
            ckpt.qwen4exp_prefix_base_logits.resize(12);
            ckpt.qwen4exp_prefix_logits.resize(24);
            ckpt.qwen4exp_prefix_tokens.assign(3, 0);
            ckpt.qwen4exp_prefix_active.assign(3, 0);
            journal.qwen4exp_last_logit_pos = {positions.back(), -1, -1};
            journal.qwen4exp_last_logits = {1, -2, 3, -0.f, 5, 6, 7, 8, 9, 10, 11, 12};
            const auto owned_logits = journal.qwen4exp_last_logits;
            check(llama_spec_ckpt_save(&journal, 0), "transformed journal base save");
            const llama_pos start = *std::max_element(positions.begin(), positions.end()) + 1;
            check(ckpt.sequences[0].n_past == start, "transformed journal future start");
            std::array<float, 2> poison{{-100.f, 200.f}};
            ggml_backend_tensor_set(state, poison.data(), 0, sizeof(poison));
            journal.qwen4exp_last_logit_pos[0] = start;
            std::fill_n(journal.qwen4exp_last_logits.data(), 4, -99.f);
            check(llama_spec_ckpt_restore_ex(&journal, 0, start, -1) == LLAMA_SPEC_CKPT_RESTORE_DIRECT,
                  "transformed journal base restore");
            std::array<float, 6> recovered;
            ggml_backend_tensor_get(state, recovered.data(), 0, sizeof(recovered));
            check(std::memcmp(base_state.data(), recovered.data(), sizeof(recovered)) == 0,
                  "base restore changed recurrent state or another sequence");
            check(journal.qwen4exp_last_logit_pos[0] == positions.back() &&
                  std::memcmp(journal.qwen4exp_last_logits.data(), owned_logits.data(), 12 * sizeof(float)) == 0 &&
                  std::memcmp(journal.logits, owned_logits.data(), 4 * sizeof(float)) == 0,
                  "base restore changed chronological output ownership");
            check(llama_spec_ckpt_save(&journal, 0), "restored transformed journal resave");
            // A stale output must be rejected before replacing a live checkpoint.
            journal.qwen4exp_last_logit_pos[0] = start;
            const auto saved_position = ckpt.sequences[0].qwen4exp_logit_pos;
            check(!llama_spec_ckpt_save(&journal, 0) && ckpt.sequences[0].saved &&
                  ckpt.sequences[0].qwen4exp_logit_pos == saved_position,
                  "stale output replaced live transformed journal");
        }
        // Public copy must validate PLE conflicts before altering KV ownership,
        // recurrent copy metadata, or the destination's saved output.
        {
            model.arch = LLM_ARCH_LLAMA;
            llama_context lifecycle(model);
            initialize(lifecycle, 3);
            model.arch = LLM_ARCH_QWEN4EXP;
            auto & kv = lifecycle.kv_self;
            kv.size = kv.n = 8; kv.cells.resize(8); kv.used = 5; kv.head = 6;
            for (int i = 0; i < 3; ++i) {
                kv.cells[i].pos = i; kv.cells[i].seq_id.insert(0);
            }
            kv.cells[3].pos = 0; kv.cells[3].seq_id.insert(1);
            kv.cells[4].pos = 2; kv.cells[4].seq_id.insert(1);
            lifecycle.qwen4exp_ple_sequences->restore(0, {{0, 101}, {1, 102}, {2, 103}});
            lifecycle.qwen4exp_ple_sequences->restore(1, {{0, 101}, {2, 999}});
            lifecycle.qwen4exp_last_logit_pos = {2, 2, -1};
            lifecycle.qwen4exp_last_logits = {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0};
            kv.ckpt.sequences.resize(3);
            kv.ckpt.sequences[1].saved = true;
            kv.ckpt.sequences[1].n_past = 3;
            kv.ckpt.sequences[1].cpu_data = {1, 2, 3};
            kv.ckpt.qwen4exp_prefix_active = {0, 1, 0};
            kv.ckpt.qwen4exp_prefix_tokens = {0, 3, 0};
            const auto before_cells = kv.cells;
            const auto before_history = lifecycle.qwen4exp_ple_sequences->all_tokens();
            const auto before_logits = lifecycle.qwen4exp_last_logits;
            const auto before_positions = lifecycle.qwen4exp_last_logit_pos;
            bool rejected = false;
            try { llama_kv_cache_seq_cp(&lifecycle, 0, 1, 0, -1); }
            catch (const std::invalid_argument &) { rejected = true; }
            check(rejected, "public copy accepted conflicting PLE history");
            check(kv.head == 6 && kv.used == 5 && !kv.do_copy, "failed copy changed cache metadata");
            check(kv.ckpt.sequences[1].saved && kv.ckpt.sequences[1].n_past == 3 &&
                  kv.ckpt.sequences[1].cpu_data == std::vector<uint8_t>{1, 2, 3} &&
                  kv.ckpt.qwen4exp_prefix_active[1] == 1 && kv.ckpt.qwen4exp_prefix_tokens[1] == 3,
                  "failed copy destroyed destination checkpoint");
            for (size_t i = 0; i < kv.cells.size(); ++i)
                check(kv.cells[i].pos == before_cells[i].pos && kv.cells[i].src == before_cells[i].src &&
                      kv.cells[i].seq_id == before_cells[i].seq_id, "failed copy changed KV ownership");
            check(lifecycle.qwen4exp_ple_sequences->all_tokens() == before_history &&
                  lifecycle.qwen4exp_last_logits == before_logits &&
                  lifecycle.qwen4exp_last_logit_pos == before_positions, "failed copy changed side state");
            for (const auto range : std::array<std::pair<int,int>, 3>{{{1,1},{3,2},{10,12}}}) {
                llama_kv_cache_seq_cp(&lifecycle, 0, 1, range.first, range.second);
                check(kv.ckpt.sequences[1].saved && lifecycle.qwen4exp_last_logit_pos == before_positions &&
                      lifecycle.qwen4exp_last_logits == before_logits &&
                      lifecycle.qwen4exp_ple_sequences->all_tokens() == before_history && kv.head == 6 && kv.used == 5,
                      "empty/out-of-range copy changed destination");
            }
            rejected = false;
            try { llama_kv_cache_seq_cp(&lifecycle, 0, 3, 0, -1); }
            catch (const std::invalid_argument &) { rejected = true; }
            check(rejected && kv.ckpt.sequences[1].saved && lifecycle.qwen4exp_ple_sequences->all_tokens() == before_history,
                  "invalid destination ownership changed state");
            check(llama_kv_cache_seq_rm(&lifecycle, -1, 1, 2), "wildcard range removal");
            check(lifecycle.qwen4exp_ple_sequences->tokens(0) ==
                      std::vector<qwen4exp_ple_sequence_state::token_entry>{{0, 101}, {2, 103}} &&
                  lifecycle.qwen4exp_ple_sequences->tokens(1) ==
                      std::vector<qwen4exp_ple_sequence_state::token_entry>{{0, 101}, {2, 999}},
                  "wildcard removal erased unrelated PLE history");
            check(lifecycle.qwen4exp_last_logit_pos == before_positions, "wildcard removal invalidated unrelated outputs");
            lifecycle.qwen4exp_ple_sequences->reset(1);
            check(llama_kv_cache_seq_rm(&lifecycle, 1, -1, -1), "destination reset");
            llama_kv_cache_seq_cp(&lifecycle, 0, 1, -1, -1);
            check(lifecycle.qwen4exp_ple_sequences->tokens(0) == lifecycle.qwen4exp_ple_sequences->tokens(1),
                  "successful public copy lost history");
            check(lifecycle.qwen4exp_last_logit_pos[1] == 2 &&
                  std::equal(before_logits.begin(), before_logits.begin() + 4,
                             lifecycle.qwen4exp_last_logits.begin() + 4), "successful copy lost output");
            check(kv.cells[0].has_seq_id(1) && kv.cells[2].has_seq_id(1) && !kv.cells[1].has_seq_id(1),
                  "successful copy lost KV ownership");
        }
        {
            model.arch = LLM_ARCH_LLAMA;
            llama_context transformed(model);
            initialize(transformed, 3);
            model.arch = LLM_ARCH_QWEN4EXP;
            auto & kv = transformed.kv_self;
            kv.qwen4exp_ordered = true;
            kv.size = kv.n = 8; kv.cells.resize(8); kv.used = 4;
            for (int i = 0; i < 4; ++i) {
                kv.cells[i].pos = i; kv.cells[i].qwen4exp_order = i;
                kv.cells[i].seq_id.insert(0);
            }
            transformed.qwen4exp_ple_sequences->restore(0, {{0, 101}, {1, 102}, {2, 103}, {3, 104}});
            transformed.qwen4exp_last_logit_pos = {3, 7, -1};
            transformed.qwen4exp_last_logits = {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0};
            const auto logits_before = transformed.qwen4exp_last_logits;
            llama_kv_cache_seq_div(&transformed, 0, 0, -1, 2);
            check(transformed.qwen4exp_last_logit_pos == std::vector<llama_pos>{1, 7, -1}, "division output positions");
            check(transformed.qwen4exp_ple_sequences->tokens(0) ==
                  std::vector<qwen4exp_ple_sequence_state::token_entry>{{0, 101}, {0, 102}, {1, 103}, {1, 104}},
                  "division token history");
            for (int i = 0; i < 4; ++i)
                check(kv.cells[i].pos == i / 2 && kv.cells[i].qwen4exp_order == i &&
                      kv.cells[i].delta == i / 2 - i, "division cache positions/order/pending rotary shift");
            const auto compressed = save(transformed);
            // Both full and per-sequence serialization preserve duplicate rotary
            // positions and the chronological order, without requiring weights.
            model.arch = LLM_ARCH_LLAMA;
            llama_context restored(model);
            initialize(restored, 3);
            restored.kv_self.size = restored.kv_self.n = 8;
            restored.kv_self.cells.resize(8); restored.kv_self.qwen4exp_ordered = true;
            model.arch = LLM_ARCH_QWEN4EXP;
            check(llama_state_set_data(&restored, compressed.data(), compressed.size()) == compressed.size(),
                  "compressed full-state restore");
            check(restored.qwen4exp_ple_sequences->tokens(0) == transformed.qwen4exp_ple_sequences->tokens(0),
                  "compressed full-state history");
            for (int i = 0; i < 4; ++i)
                check(restored.kv_self.cells[i].pos == i / 2 && restored.kv_self.cells[i].qwen4exp_order == i,
                      "compressed restored cell order");
            // This metadata fixture has no key tensors. The real-model test
            // verifies that saving also applies pending rotary shifts.
            const size_t seq_size = llama_state_seq_get_size(&transformed, 0, 0);
            std::vector<uint8_t> seq_bytes(seq_size);
            check(llama_state_seq_get_data(&transformed, seq_bytes.data(), seq_size, 0, 0) == seq_size, "compressed seq save");
            check(llama_state_seq_set_data(&restored, seq_bytes.data(), seq_size, 1, 0) == seq_size, "compressed seq restore");
            check(restored.qwen4exp_ple_sequences->tokens(1) == transformed.qwen4exp_ple_sequences->tokens(0),
                  "compressed migrated history");
            llama_kv_cache_seq_cp(&restored, 0, 1, 0, -1);
            llama_batch overlap_query{};
            overlap_query.n_tokens = 1; overlap_query.all_seq_id = 1;
            const auto overlap_layout = llama_qwen4exp_attention_layout(restored, overlap_query);
            check(overlap_layout.size() == 1 && overlap_layout[0].physical_rows.size() == 4,
                  "overlapping identical copy retained duplicate token rows");
            check(restored.kv_self.used == 4, "overlapping copy did not release redundant private rows");
            const size_t migrated_size = llama_state_seq_get_size(&restored, 1, 0);
            std::vector<uint8_t> migrated(migrated_size);
            check(llama_state_seq_get_data(&restored, migrated.data(), migrated.size(), 1, 0) == migrated.size(),
                  "migrated fixture before malformed restore");
            for (bool bad_order : {false, true}) {
                auto malformed = seq_bytes;
                if (bad_order) {
                    const int64_t invalid_order = -1;
                    std::memcpy(malformed.data() + 16, &invalid_order, sizeof(invalid_order));
                } else replace_u32(malformed, 4, 0); // Wrong chronological metadata version.
                restored.qwen4exp_last_logit_pos[1] = 1;
                restored.kv_self.ckpt.sequences.resize(3);
                auto & previous_checkpoint = restored.kv_self.ckpt.sequences[1];
                previous_checkpoint.saved = true; previous_checkpoint.n_past = 2;
                previous_checkpoint.cpu_data = {7, 8, 9};
                restored.kv_self.ckpt.qwen4exp_prefix_active = {0, 1, 0};
                restored.kv_self.ckpt.qwen4exp_prefix_tokens = {0, 2, 0};
                restored.kv_self.save_per_step_ssm = true;
                const auto previous_outputs = restored.qwen4exp_last_logits;
                check(llama_state_seq_set_data(&restored, malformed.data(), malformed.size(), 1, 0) == 0,
                      "malformed chronological snapshot accepted");
                check(restored.qwen4exp_last_logit_pos[1] == 1 && previous_checkpoint.saved &&
                      previous_checkpoint.n_past == 2 && previous_checkpoint.cpu_data == std::vector<uint8_t>{7, 8, 9} &&
                      restored.qwen4exp_last_logits == previous_outputs && restored.kv_self.save_per_step_ssm &&
                      restored.kv_self.ckpt.qwen4exp_prefix_active[1] == 1 &&
                      restored.kv_self.ckpt.qwen4exp_prefix_tokens[1] == 2,
                      "malformed restore destroyed previous output/checkpoint ownership");
                const size_t recovered_size = llama_state_seq_get_size(&restored, 1, 0);
                std::vector<uint8_t> recovered(recovered_size);
                check(llama_state_seq_get_data(&restored, recovered.data(), recovered.size(), 1, 0) == recovered.size() &&
                      recovered == migrated, "malformed chronological restore lost KV/history");
                const auto root = std::getenv("QWEN4EXP_OWNERSHIP_TEST_DIRECTORY")
                    ? std::filesystem::path(std::getenv("QWEN4EXP_OWNERSHIP_TEST_DIRECTORY")) : std::filesystem::current_path();
                std::filesystem::create_directories(root);
                const auto good = (root / (std::string("ownership-good-") + std::to_string(bad_order) + ".session")).string();
                const auto bad = (root / (std::string("ownership-bad-") + std::to_string(bad_order) + ".session")).string();
                const std::array<llama_token, 4> saved_prompt = {101, 102, 103, 104};
                check(llama_state_seq_save_file(&restored, good.c_str(), 1, saved_prompt.data(), saved_prompt.size()) > 0,
                      "ownership file fixture save");
                std::filesystem::copy_file(good, bad, std::filesystem::copy_options::overwrite_existing);
                std::filesystem::resize_file(bad, std::filesystem::file_size(bad) - 1);
                std::array<llama_token, 4> published = {-17, -17, -17, -17}; size_t published_count = 99;
                check(llama_state_seq_load_file(&restored, bad.c_str(), 1, published.data(), published.size(), &published_count) == 0,
                      "truncated ownership file accepted");
                check(published == std::array<llama_token, 4>{-17, -17, -17, -17} && published_count == 99 &&
                      restored.qwen4exp_last_logit_pos[1] == 1 && previous_checkpoint.saved &&
                      previous_checkpoint.cpu_data == std::vector<uint8_t>{7, 8, 9} &&
                      restored.kv_self.ckpt.qwen4exp_prefix_active[1] == 1 && restored.kv_self.save_per_step_ssm,
                      "failed file restore published prompt/output/checkpoint state");
                check(llama_state_seq_load_file(&restored, good.c_str(), 1, published.data(), published.size() - 1, &published_count) == 0 &&
                      published_count == 99 && published[0] == -17 && previous_checkpoint.saved,
                      "prompt capacity failure changed ownership");
                check(llama_state_seq_load_file(&restored, good.c_str(), 1, published.data(), published.size(), &published_count) > 0 &&
                      published == saved_prompt && published_count == saved_prompt.size() &&
                      restored.qwen4exp_last_logit_pos[1] == -1 && !previous_checkpoint.saved &&
                      restored.kv_self.ckpt.qwen4exp_prefix_active[1] == 0,
                      "successful file restore did not commit prompt/ownership");
            }
            llama_kv_cache_seq_add(&transformed, 0, 0, 1, 5);
            check(kv.cells[0].pos == 5 && kv.cells[1].pos == 5 && kv.cells[2].pos == 1 &&
                  transformed.qwen4exp_last_logit_pos[0] == 1, "partial shift consistency");
            check(transformed.qwen4exp_ple_sequences->tokens(0) ==
                  std::vector<qwen4exp_ple_sequence_state::token_entry>{{5, 101}, {5, 102}, {1, 103}, {1, 104}},
                  "partial shift chronological history");
            auto before_invalid = transformed.qwen4exp_ple_sequences->all_tokens();
            for (int divisor : {0, -1}) {
                bool rejected = false;
                try { llama_kv_cache_seq_div(&transformed, 0, 0, -1, divisor); }
                catch (const std::invalid_argument &) { rejected = true; }
                check(rejected && transformed.qwen4exp_ple_sequences->all_tokens() == before_invalid,
                      "invalid divisor changed live state");
            }
            kv.cells[0].seq_id.insert(1);
            bool rejected = false;
            try { llama_kv_cache_seq_add(&transformed, 0, 5, 6, 1); }
            catch (const std::invalid_argument &) { rejected = true; }
            check(rejected && kv.cells[0].pos == 5 && transformed.qwen4exp_ple_sequences->all_tokens() == before_invalid,
                  "private shift changed a shared cache row");
            kv.cells[0].seq_id.erase(1);
            llama_kv_cache_seq_add(&transformed, 0, 5, 6, -6);
            check(kv.used == 2 && kv.cells[0].is_empty() && kv.cells[1].is_empty(), "negative shift cell removal");
            check(transformed.qwen4exp_ple_sequences->tokens(0) ==
                  std::vector<qwen4exp_ple_sequence_state::token_entry>{{1, 103}, {1, 104}}, "negative shift history");
            check(transformed.qwen4exp_last_logits == logits_before, "position transforms changed historical logits");
        }
        // Shared-prefix transforms copy exact cache bytes into free rows,
        // preserving every unselected owner's metadata and payloads.
        for (bool packed : {false, true}) {
            model.arch = LLM_ARCH_LLAMA;
            llama_context shared(model);
            initialize(shared, 3);
            model.arch = LLM_ARCH_QWEN4EXP;
            auto & kv = shared.kv_self;
            kv.size = kv.n = 16; kv.cells.resize(16); kv.used = 3;
            kv.qwen4exp_ordered = true; kv.v_trans = false;
            for (uint32_t row = 0; row < kv.size; ++row) kv.cells[row].src = 100 + row;
            for (int row = 0; row < 3; ++row) {
                kv.cells[row].pos = row; kv.cells[row].qwen4exp_order = row;
                kv.cells[row].seq_id = {0, 1};
            }
            auto * tensors = ggml_init({1024 * 1024, nullptr, true});
            check(tensors, "shared cache tensor context"); kv.ctxs.push_back(tensors);
            kv.k_l = {ggml_new_tensor_2d(tensors, packed ? GGML_TYPE_Q8_0 : GGML_TYPE_F32, packed ? 32 : 4, 16)};
            kv.v_l = {ggml_new_tensor_1d(tensors, packed ? GGML_TYPE_F16 : GGML_TYPE_F32, 4 * 16)};
            kv.kr_l = {ggml_new_tensor_2d(tensors, packed ? GGML_TYPE_F16 : GGML_TYPE_F32, 2, 16)};
            auto * copy_backend = shared.backends.front();
            if (const char * device = std::getenv("QWEN4EXP_COPY_TEST_CUDA_DEVICE")) {
#ifdef GGML_USE_CUDA
                copy_backend = ggml_backend_cuda_init(std::atoi(device), nullptr, nullptr);
                check(copy_backend, "CUDA copy backend"); shared.backends.push_back(copy_backend);
#else
                (void) device; throw std::runtime_error("CUDA copy test requested in CPU build");
#endif
            }
            auto * buffer = ggml_backend_alloc_ctx_tensors(tensors, copy_backend);
            check(buffer, "shared cache allocation"); kv.bufs.push_back(buffer);
            std::vector<std::vector<uint8_t>> original;
            for (auto * tensor : {kv.k_l[0], kv.v_l[0], kv.kr_l[0]}) {
                std::vector<uint8_t> bytes(ggml_nbytes(tensor));
                for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t((i * 37 + 19) % 251);
                ggml_backend_tensor_set(tensor, bytes.data(), 0, bytes.size()); original.push_back(bytes);
            }
            shared.qwen4exp_ple_sequences->restore(0, {{0, 101}, {1, 102}, {2, 103}});
            shared.qwen4exp_ple_sequences->copy(0, 1, 0, -1);
            shared.qwen4exp_last_logit_pos = {2, 2, -1};
            shared.qwen4exp_last_logits.assign(12, 7);
            kv.ckpt.sequences.resize(3); kv.ckpt.sequences[1].saved = true; kv.ckpt.sequences[1].n_past = 3;
            llama_kv_cache_seq_add(&shared, 0, 0, -1, 4);
            check(kv.used == 6 && shared.qwen4exp_last_logit_pos == std::vector<llama_pos>{6, 2, -1},
                  "shared shift usage/output ownership");
            for (int row = 0; row < 3; ++row) {
                check(kv.cells[row].seq_id == std::set<llama_seq_id>{1} && kv.cells[row].pos == row &&
                      kv.cells[row].qwen4exp_order == row && kv.cells[row].src == 100 + row,
                      "shared transform changed unselected metadata");
                check(kv.cells[3 + row].seq_id == std::set<llama_seq_id>{0} && kv.cells[3 + row].pos == row + 4 &&
                      kv.cells[3 + row].qwen4exp_order == row && kv.cells[3 + row].delta == 4 &&
                      kv.cells[3 + row].src == 103 + row, "shared transform did not preserve clone/recurrent slot metadata");
            }
            check(kv.ckpt.sequences[1].saved && kv.ckpt.sequences[1].n_past == 3, "shared transform invalidated other checkpoint");
            size_t family = 0;
            for (auto * tensor : {kv.k_l[0], kv.v_l[0], kv.kr_l[0]}) {
                std::vector<uint8_t> bytes(ggml_nbytes(tensor));
                ggml_backend_tensor_get(tensor, bytes.data(), 0, bytes.size());
                const size_t row_bytes = bytes.size() / 16;
                for (size_t row = 0; row < 16; ++row) {
                    const size_t expected = row >= 3 && row < 6 ? row - 3 : row;
                    check(std::equal(bytes.begin() + row * row_bytes, bytes.begin() + (row + 1) * row_bytes,
                                     original[family].begin() + expected * row_bytes), "shared row copy changed exact cache bytes");
                }
                ++family;
            }
            check(shared.qwen4exp_ple_sequences->tokens(1) ==
                  std::vector<qwen4exp_ple_sequence_state::token_entry>{{0, 101}, {1, 102}, {2, 103}}, "shared transform changed other PLE history");
            // Removing negative shifted tokens requires no clone even at full
            // capacity, and leaves the remaining owner's bytes and rows intact.
            llama_kv_cache_seq_add(&shared, 1, 0, 1, -1);
            check(kv.used == 5 && kv.cells[0].is_empty() && kv.cells[1].has_seq_id(1), "negative owner shift removal");
        }
        {
            model.arch = LLM_ARCH_LLAMA;
            llama_context full(model); initialize(full, 3);
            model.arch = LLM_ARCH_QWEN4EXP;
            auto & kv = full.kv_self;
            kv.size = kv.n = kv.used = 2; kv.cells.resize(2); kv.qwen4exp_ordered = true; kv.v_trans = false;
            for (int i = 0; i < 2; ++i) { kv.cells[i].pos = i; kv.cells[i].qwen4exp_order = i; kv.cells[i].seq_id = {0, 1}; }
            full.qwen4exp_ple_sequences->restore(0, {{0, 101}, {1, 102}});
            full.qwen4exp_ple_sequences->copy(0, 1, 0, -1);
            full.qwen4exp_last_logit_pos = {1, 1, -1}; full.qwen4exp_last_logits.assign(12, 7);
            const auto history_before = full.qwen4exp_ple_sequences->all_tokens();
            bool failed = false;
            try { llama_kv_cache_seq_add(&full, 0, 0, -1, 3); }
            catch (const std::runtime_error &) { failed = true; }
            check(failed && kv.used == 2 && kv.cells[0].pos == 0 && kv.cells[1].pos == 1 &&
                  kv.cells[0].seq_id == std::set<llama_seq_id>{0, 1} && !kv.has_shift &&
                  full.qwen4exp_ple_sequences->all_tokens() == history_before &&
                  full.qwen4exp_last_logit_pos == std::vector<llama_pos>{1, 1, -1}, "capacity failure changed a live shared prefix");
            llama_kv_cache_seq_add(&full, 0, 0, -1, -2);
            check(kv.used == 2 && kv.cells[0].seq_id == std::set<llama_seq_id>{1} &&
                  kv.cells[1].seq_id == std::set<llama_seq_id>{1} && full.qwen4exp_ple_sequences->tokens(0).empty() &&
                  full.qwen4exp_last_logit_pos == std::vector<llama_pos>{-1, 1, -1}, "negative shared shift damaged other owner");
        }
        // Conventional architectures retain their ordinary session format.
        model.arch = LLM_ARCH_LLAMA;
        const auto conventional = save(source);
        llama_context target(model);
        initialize(target, 3);
        check(llama_state_set_data(&target, conventional.data(), conventional.size()) == conventional.size(),
              "conventional session restore");
        check(save(target) == conventional, "conventional session roundtrip");
        check(target.qwen4exp_last_logit_pos.empty(), "conventional session created Qwen output ownership");
        target.kv_self.ckpt.sequences.resize(3);
        target.kv_self.ckpt.sequences[1].saved = true;
        target.kv_self.ckpt.sequences[1].cpu_data = {11, 12};
        const auto conventional_root = std::getenv("QWEN4EXP_OWNERSHIP_TEST_DIRECTORY")
            ? std::filesystem::path(std::getenv("QWEN4EXP_OWNERSHIP_TEST_DIRECTORY")) : std::filesystem::current_path();
        std::filesystem::create_directories(conventional_root);
        const auto normal_good = (conventional_root / "ownership-conventional-good.session").string();
        const auto normal_bad = (conventional_root / "ownership-conventional-bad.session").string();
        const std::array<llama_token, 2> normal_prompt = {1, 2};
        check(llama_state_seq_save_file(&target, normal_good.c_str(), 1, normal_prompt.data(), normal_prompt.size()) > 0,
              "conventional ownership file save");
        std::filesystem::copy_file(normal_good, normal_bad, std::filesystem::copy_options::overwrite_existing);
        std::filesystem::resize_file(normal_bad, std::filesystem::file_size(normal_bad) - 1);
        std::array<llama_token, 2> normal_published = {-17, -17}; size_t normal_count = 99;
        check(llama_state_seq_load_file(&target, normal_bad.c_str(), 1, normal_published.data(), normal_published.size(), &normal_count) == 0 &&
              normal_published == std::array<llama_token, 2>{-17, -17} && normal_count == 99 &&
              target.kv_self.ckpt.sequences[1].saved && target.kv_self.ckpt.sequences[1].cpu_data == std::vector<uint8_t>{11, 12},
              "conventional failed file restore changed prompt/checkpoint");
        check(llama_state_seq_load_file(&target, normal_good.c_str(), 1, normal_published.data(), normal_published.size(), &normal_count) > 0 &&
              normal_published == normal_prompt && normal_count == 2 && !target.kv_self.ckpt.sequences[1].saved,
              "conventional successful file restore did not commit prompt/checkpoint");
        std::puts("PASS: independent outputs, session roundtrip, capacity/version/truncation rejection, conventional session");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
