#include "llama-qwen4exp-ple-storage.h"
#include "llama.h"
#include <random>
#include "../src/llama-context.h"
#include "../src/llama-model.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using json = nlohmann::json;
using namespace qwen4exp_ple_storage;

static uint8_t hex_nibble(char value) {
    if (value >= '0' && value <= '9') return uint8_t(value - '0');
    if (value >= 'a' && value <= 'f') return uint8_t(value - 'a' + 10);
    throw std::invalid_argument("invalid fixture hex digit");
}

static std::vector<uint8_t> decode_hex(const std::string & encoded) {
    std::vector<uint8_t> result(encoded.size() / 2);
    for (size_t index = 0; index < result.size(); ++index) {
        result[index] = uint8_t(hex_nibble(encoded[2 * index]) << 4) |
            hex_nibble(encoded[2 * index + 1]);
    }
    return result;
}

class memory_source final : public llama_expert_cache::source {
public:
    explicit memory_source(std::vector<uint8_t> bytes) : bytes_(std::move(bytes)) {}

    uint32_t id() const override { return 9; }
    llama_expert_cache::source_identity identity() const override { return {0x9876, 0x5432}; }
    uint64_t size() const override { return bytes_.size(); }
    llama_expert_cache::read_backend backend() const override {
        return llama_expert_cache::read_backend::pread;
    }
    void read(uint64_t offset, void * destination, size_t length) const override {
        if (offset > bytes_.size() || length > bytes_.size() - offset) {
            throw std::out_of_range("memory source read is out of range");
        }
        std::memcpy(destination, bytes_.data() + offset, length);
    }

private:
    std::vector<uint8_t> bytes_;
};

static uint64_t pool_bytes(llama_context & ctx, llama_resource_memory_category category) {
    uint32_t count=0;
    if(!llama_resource_get_memory_pools(&ctx,nullptr,0,&count))throw std::runtime_error("PLE inventory count query");
    std::vector<llama_resource_memory_pool> pools(count);
    if(!llama_resource_get_memory_pools(&ctx,pools.data(),count,&count))throw std::runtime_error("PLE inventory rows");
    uint64_t result=0;
    for(const auto & pool:pools)if(pool.category==category) {
        if(pool.backend_id!=-1)throw std::runtime_error("PLE storage reported on a device");
        result+=pool.bytes;
    }
    return result;
}

template<class Store> static bool wait_for_staging_release(Store & store) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    do {
        const auto stats=store.stats().reads;
        if(stats.active_reads==0 && stats.queued_reads==0 && stats.staging_allocated_bytes==0 &&
                stats.staging_control_allocated_bytes==0 && stats.staging_resident_bytes==0)return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }while(std::chrono::steady_clock::now()<deadline);
    return false;
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s PLE_ROW_FIXTURE.json\n", argv[0]);
        return 1;
    }
    std::ifstream input(argv[1]);
    const json fixture = json::parse(input);
    const auto row = decode_hex(fixture.at("quantized_hex").get<std::string>());
    if (row.size() != 90) return 1;

    std::vector<uint8_t> source_bytes;
    for (size_t index = 0; index < 4; ++index) {
        source_bytes.insert(source_bytes.end(), row.begin(), row.end());
    }
    auto source = std::make_shared<memory_source>(std::move(source_bytes));
    qwen4exp_ple_hash_config hash_config;
    qwen4exp_ple_extent_index extents(hash_config, 0, 0, 90, 4);
    llama_backend_init();
    {
    llama_model model{};model.arch=LLM_ARCH_LLAMA;
    llama_context context(model);
    auto * backend=ggml_backend_cpu_init();if(!backend)return 1;
    context.backends.push_back(backend);
    context.sched=ggml_backend_sched_new(context.backends.data(),nullptr,1,256,false);
    if(!context.sched)return 1;
    context.qwen4exp_ple_row_store=std::make_unique<row_store>(source,extents,2,180);
    auto & store=*context.qwen4exp_ple_row_store;
    const auto cold_stats=store.stats();
    const auto cold=cold_stats.rows;
    const auto cold_metadata=pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_METADATA);
    if(cold.resident_bytes!=0 || pool_bytes(context,LLAMA_RESOURCE_MEMORY_PLE_CACHE)!=cold.arena_allocated_bytes ||
            cold_metadata<cold.slot_allocated_bytes || cold.index_allocated_bytes!=0)return 1;
    auto first_ticket=store.submit(1,{0,1,0});
    auto first=first_ticket.get();
    const auto held=store.stats();
    if(held.reads.staging_allocated_bytes!=180 || held.reads.staging_control_allocated_bytes==0 ||
            pool_bytes(context,LLAMA_RESOURCE_MEMORY_PLE_STAGING)!=held.reads.staging_allocated_bytes ||
            pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_METADATA)!=cold_metadata+
                held.rows.index_allocated_bytes+held.reads.staging_control_allocated_bytes+
                held.reads.metadata_allocated_bytes-cold_stats.reads.metadata_allocated_bytes+
                held.metadata.allocated_bytes-cold_stats.metadata.allocated_bytes) {
        std::fprintf(stderr,"PLE inventory missed live staged data/control storage\n");return 1;
    }
    first_ticket={};
    if(!wait_for_staging_release(store) || pool_bytes(context,LLAMA_RESOURCE_MEMORY_PLE_STAGING)!=0)return 1;
    if (first.size() != 3 || first.row(0).key() != 0 || first.row(1).key() != 1 ||
            first.row(0).data()[0] != row[0] || first.row(2).data()[89] != row[89]) {
        std::fprintf(stderr, "PLE row store returned incorrect staged rows\n");
        return 1;
    }

    std::vector<float> values(first.size() * 160);
    first.dequantize(values.data(), values.size());
    const json & expected = fixture.at("dequantized_values");
    for (size_t row_index = 0; row_index < first.size(); ++row_index) {
        for (size_t value = 0; value < 160; ++value) {
            if (values[row_index * 160 + value] != expected.at(value).get<float>()) {
                std::fprintf(stderr, "PLE row store dequantization mismatch\n");
                return 1;
            }
        }
    }

    auto second = store.submit(2, {0, 1}).get();
    const row_store_stats warm_stats = store.stats();
    if(pool_bytes(context,LLAMA_RESOURCE_MEMORY_PLE_CACHE)!=cold.arena_allocated_bytes ||
            pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_METADATA)-cold_metadata!=warm_stats.rows.index_allocated_bytes+
                warm_stats.reads.metadata_allocated_bytes-cold_stats.reads.metadata_allocated_bytes+
                warm_stats.metadata.allocated_bytes-cold_stats.metadata.allocated_bytes) {
        std::fprintf(stderr,"PLE native inventory missed allocated index nodes\n");return 1;
    }
    if (warm_stats.reads.submitted_batches != 1 || warm_stats.reads.logical_reads != 3 ||
            warm_stats.reads.source_reads != 2 || warm_stats.reads.coalesced_reads != 1 ||
            warm_stats.rows.misses != 3 || warm_stats.rows.hits != 2 ||
            warm_stats.rows.admissions != 2 || warm_stats.rows.resident_bytes != 180 ||
            warm_stats.rows.active_leases != 5) {
        std::fprintf(stderr, "incorrect PLE row store cold/warm telemetry\n");
        return 1;
    }

    auto cancelled = store.submit(3, {2});
    if (store.cancel_owner(3) != 1) {
        std::fprintf(stderr, "PLE row store did not cancel its owner\n");
        return 1;
    }
    bool rejected = false;
    try {
        (void) cancelled.get();
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "cancelled PLE row batch produced staged rows\n");
        return 1;
    }
    cancelled={};if(!wait_for_staging_release(store))return 1;

    // Both provider shapes expose their eagerly allocated payload arenas and
    // metadata even before any rows/pages are occupied.
    context.qwen4exp_ple_store=std::make_unique<two_tier_store>(source,extents,2,4096,180);
    auto tiers=context.qwen4exp_ple_store->stats();
    const uint64_t expected_arena=warm_stats.rows.arena_allocated_bytes+tiers.pages.arena_allocated_bytes+tiers.rows.arena_allocated_bytes;
    if(pool_bytes(context,LLAMA_RESOURCE_MEMORY_PLE_CACHE)!=expected_arena ||
            tiers.pages.resident_bytes!=0 || tiers.rows.resident_bytes!=0)return 1;
    auto staged=context.qwen4exp_ple_store->submit(4,{0}).get();
    if(!wait_for_staging_release(*context.qwen4exp_ple_store))return 1;
    tiers=context.qwen4exp_ple_store->stats();
    if(staged.size()!=1 || pool_bytes(context,LLAMA_RESOURCE_MEMORY_PLE_CACHE)!=expected_arena ||
            pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_METADATA)!=cold_metadata+warm_stats.rows.index_allocated_bytes+
                store.stats().reads.metadata_allocated_bytes-cold_stats.reads.metadata_allocated_bytes+
                tiers.reads.metadata_allocated_bytes+
                tiers.pages.slot_allocated_bytes+tiers.rows.slot_allocated_bytes+
                tiers.pages.index_allocated_bytes+tiers.rows.index_allocated_bytes+
                store.stats().metadata.allocated_bytes-cold_stats.metadata.allocated_bytes+
                tiers.metadata.allocated_bytes+tiers.page_metadata.allocated_bytes) {
        std::fprintf(stderr,"PLE two-tier inventory did not sum separate allocations\n");return 1;
    }
    // Published and pending histories have independent allocators; sum both
    // roots once, then release the pending root without changing the current one.
    qwen4exp_ple_hash_config history_config;history_config.head_vocab_sizes.fill(257);
    history_config.layer_multipliers={1,3,5};
    const auto journal_before=pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_JOURNAL);
    context.qwen4exp_ple_sequences=std::make_unique<qwen4exp_ple_sequence_state>(history_config);
    (void)context.qwen4exp_ple_sequences->accept(0,0,101);
    const auto current_bytes=context.qwen4exp_ple_sequences->memory_stats().allocated_bytes;
    context.qwen4exp_ple_sequences_pending=std::make_unique<qwen4exp_ple_sequence_state>(*context.qwen4exp_ple_sequences);
    if(context.qwen4exp_ple_sequences->memory_stats().allocated_bytes!=current_bytes)return 1;
    (void)context.qwen4exp_ple_sequences_pending->accept(0,1,102);
    const auto pending_bytes=context.qwen4exp_ple_sequences_pending->memory_stats().allocated_bytes;
    if(pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_JOURNAL)!=journal_before+current_bytes+pending_bytes)return 1;
    context.qwen4exp_ple_sequences_pending.reset();
    if(pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_JOURNAL)!=journal_before+current_bytes)return 1;
    context.qwen4exp_ple_sequences.reset();
    if(pool_bytes(context,LLAMA_RESOURCE_MEMORY_HOST_JOURNAL)!=journal_before)return 1;
    }
    llama_backend_free();
    return 0;
}
