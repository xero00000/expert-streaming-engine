#include "llama-qwen4exp-ple-storage.h"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>

using namespace qwen4exp_ple_storage;

int main() {
    bounded_block_cache cache(180, 90);
    const auto empty = cache.stats();
    if (empty.resident_bytes != 0 || empty.arena_allocated_bytes < 180 ||
            empty.slot_allocated_bytes == 0 || empty.index_allocated_bytes != 0) {
        std::fprintf(stderr, "empty PLE cache allocation accounting is incomplete\n");return 1;
    }
    std::array<uint8_t, 90> row1 = {};
    std::array<uint8_t, 90> row2 = {};
    std::array<uint8_t, 90> row3 = {};
    std::array<uint8_t, 90> row4 = {};
    row1.fill(1);
    row2.fill(2);
    row3.fill(3);
    row4.fill(4);

    auto lease1 = cache.admit(1, row1.data(), row1.size());
    const uint64_t node_bytes = cache.stats().index_allocated_bytes;
    if (node_bytes == 0) { std::fprintf(stderr, "PLE index allocation was not counted\n");return 1; }
    {
        auto lease2 = cache.admit(2, row2.data(), row2.size());
        if (lease2.data()[0] != 2 || lease2.size() != 90 || lease2.key() != 2) {
            std::fprintf(stderr, "PLE cache lease returned incorrect data\n");
            return 1;
        }
    }

    // Row 1 is pinned; admitting row 3 must evict unpinned row 2.
    auto lease3 = cache.admit(3, row3.data(), row3.size());
    if (cache.acquire(2)) {
        std::fprintf(stderr, "PLE cache retained the wrong LRU entry\n");
        return 1;
    }
    {
        auto hit = cache.acquire(1);
        if (!hit || hit.data()[89] != 1) {
            std::fprintf(stderr, "PLE cache failed to acquire a resident row\n");
            return 1;
        }
    }

    bool pinned_failure = false;
    try {
        (void) cache.admit(4, row4.data(), row4.size());
    } catch (const block_cache_capacity_error &) {
        pinned_failure = true;
    }
    if (!pinned_failure) {
        std::fprintf(stderr, "PLE cache reused a pinned slot\n");
        return 1;
    }

    lease3 = {};
    auto lease4 = cache.admit(4, row4.data(), row4.size());
    if (lease4.data()[0] != 4) {
        std::fprintf(stderr, "PLE cache failed to reuse an unpinned slot\n");
        return 1;
    }

    bool changed_data_rejected = false;
    try {
        (void) cache.admit(4, row3.data(), row3.size());
    } catch (const std::runtime_error &) {
        changed_data_rejected = true;
    }
    if (!changed_data_rejected) {
        std::fprintf(stderr, "PLE cache accepted changed bytes for a resident key\n");
        return 1;
    }

    const block_cache_stats stats = cache.stats();
    if (stats.capacity_bytes != 180 || stats.resident_bytes > stats.capacity_bytes ||
            stats.peak_resident_bytes > stats.capacity_bytes || stats.admissions != 4 ||
            stats.evictions != 2 || stats.hits != 1 || stats.misses != 1 ||
            stats.pinned_capacity_failures != 1 || stats.rejected_admissions != 2 ||
            stats.active_leases != 2) {
        std::fprintf(stderr, "incorrect bounded PLE cache telemetry\n");
        return 1;
    }

    if (stats.arena_allocated_bytes != empty.arena_allocated_bytes ||
            stats.slot_allocated_bytes != empty.slot_allocated_bytes ||
            stats.index_allocated_bytes != 2 * node_bytes || stats.index_peak_allocated_bytes < stats.index_allocated_bytes) {
        std::fprintf(stderr, "PLE allocation counts changed incorrectly across admission/eviction\n");return 1;
    }

    // A real allocator quota failure at index insertion must preserve both an
    // empty slot and an unpinned eviction candidate, rather than orphaning pins.
    for (bool full : {false, true}) {
        bounded_block_cache limited(180,90,(full ? 2 : 1) * node_bytes);
        auto first=limited.admit(1,row1.data(),row1.size());
        bounded_block_cache::lease second;
        if(full) { second=limited.admit(2,row2.data(),row2.size()); first={}; }
        const auto before=limited.stats();bool failed=false;
        try { (void)limited.admit(3,row3.data(),row3.size()); }
        catch(const std::bad_alloc &) { failed=true; }
        const auto after=limited.stats();
        if(!failed || after.admissions!=before.admissions || after.evictions!=before.evictions ||
                after.resident_bytes!=before.resident_bytes || after.active_leases!=before.active_leases ||
                after.index_allocated_bytes!=before.index_allocated_bytes ||
                after.index_peak_allocated_bytes!=before.index_peak_allocated_bytes ||
                after.arena_allocated_bytes!=before.arena_allocated_bytes ||
                after.slot_allocated_bytes!=before.slot_allocated_bytes) {
            std::fprintf(stderr,"PLE index allocation failure changed storage/ownership\n");return 1;
        }
        auto retained=limited.acquire(1);
        if(!retained || std::memcmp(retained.data(),row1.data(),row1.size())!=0 || limited.acquire(3)) {
            std::fprintf(stderr,"PLE failed index admission lost old bytes/index\n");return 1;
        }
        retained={};
        bool failed_again=false;
        try { (void)limited.admit(4,row4.data(),row4.size()); }
        catch(const std::bad_alloc &) { failed_again=true; }
        catch(const std::exception &) {}
        if(!failed_again) {
            std::fprintf(stderr,"PLE failed admission left an unusable slot\n");return 1;
        }
    }
    std::printf("PLE allocations: arena=%llu slots=%llu index=%llu peak_index=%llu; quota failures preserve empty/evictable slots\n",
        (unsigned long long)stats.arena_allocated_bytes,(unsigned long long)stats.slot_allocated_bytes,
        (unsigned long long)stats.index_allocated_bytes,(unsigned long long)stats.index_peak_allocated_bytes);

    return 0;
}
