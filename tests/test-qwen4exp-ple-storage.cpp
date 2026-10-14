#include "llama-qwen4exp-ple-storage.h"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace qwen4exp_ple_storage;

class blocking_source final : public llama_expert_cache::source {
public:
    explicit blocking_source(size_t size) : bytes_(size) {
        for (size_t index = 0; index < bytes_.size(); ++index) bytes_[index] = uint8_t(index);
    }

    uint32_t id() const override { return 7; }
    llama_expert_cache::source_identity identity() const override { return {0x1234, 0x5678}; }
    uint64_t size() const override { return bytes_.size(); }
    llama_expert_cache::read_backend backend() const override {
        return llama_expert_cache::read_backend::pread;
    }

    void read(uint64_t offset, void * destination, size_t length) const override {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            read_threads_.push_back(std::this_thread::get_id());
            ++reads_;
            entered_ = true;
            entered_cv_.notify_all();
            release_cv_.wait(lock, [&] { return released_; });
        }
        if (offset > bytes_.size() || length > bytes_.size() - offset) {
            throw std::out_of_range("test source read out of range");
        }
        if (fail_reads_.load()) throw std::runtime_error("injected source failure");
        std::memcpy(destination, bytes_.data() + offset, length);
    }

    void wait_until_entered() const {
        std::unique_lock<std::mutex> lock(mutex_);
        entered_cv_.wait(lock, [&] { return entered_; });
    }

    void release() const {
        std::lock_guard<std::mutex> lock(mutex_);
        released_ = true;
        release_cv_.notify_all();
    }

    size_t reads() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return reads_;
    }

    bool read_on_thread(std::thread::id value) const {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto & thread : read_threads_) {
            if (thread == value) return true;
        }
        return false;
    }
    void fail_reads(bool value) { fail_reads_.store(value); }

private:
    std::vector<uint8_t> bytes_;
    mutable std::mutex mutex_;
    mutable std::condition_variable entered_cv_;
    mutable std::condition_variable release_cv_;
    mutable bool entered_ = false;
    mutable bool released_ = false;
    mutable size_t reads_ = 0;
    mutable std::vector<std::thread::id> read_threads_;
    std::atomic<bool> fail_reads_{false};
};

static bool wait_for_data_release(async_reader & reader, bool controls = true) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    do {
        const auto stats=reader.stats();
        if(stats.queued_reads==0 && stats.active_reads==0 && stats.staging_resident_bytes==0 &&
                stats.staging_allocated_bytes==0 && (!controls || stats.staging_control_allocated_bytes==0))return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }while(std::chrono::steady_clock::now()<deadline);
    return false;
}

static bool allocation_cases() {
    auto source=std::make_shared<blocking_source>(64);source->release();
    uint64_t control_bytes=0;
    {
        async_reader reader(source,1,4);
        auto ticket=reader.submit(100,{{0,4}});auto result=ticket.get();
        control_bytes=reader.stats().staging_control_allocated_bytes;
        if(control_bytes==0 || reader.stats().staging_allocated_bytes!=4)return false;
        std::weak_ptr<const std::vector<uint8_t>> weak=result.buffers[0];
        result={};ticket={};
        if(!wait_for_data_release(reader,false) || !weak.expired() ||
                reader.stats().staging_control_allocated_bytes!=control_bytes)return false;
        weak.reset();if(!wait_for_data_release(reader))return false;
    }
    // A zero quota fails the actual shared_ptr control allocation, after raw
    // data allocation. A one-control quota fails partway through a two-span batch.
    for(uint64_t quota:{uint64_t(0),control_bytes}) {
        async_reader reader(source,1,8,2,quota);
        auto ticket=reader.submit(101,{{0,4},{4,4}});bool failed=false;
        try { (void)ticket.get(); }catch(const std::bad_alloc &) { failed=true; }
        if(!failed || !wait_for_data_release(reader))return false;
        auto stats=reader.stats();
        if(stats.failed_reads!=2 || stats.source_reads!=0 || stats.staging_peak_allocated_bytes==0 ||
                stats.staging_peak_allocated_bytes>8 || stats.staging_control_peak_allocated_bytes>quota)return false;
        // Reuse proves cleanup did not leave a consumed reservation. With one
        // available control, the next single-span read must succeed.
        auto retry=reader.submit(102,{{8,4}});
        if(quota) {
            auto result=retry.get();
            if(*result.buffers[0]!=std::vector<uint8_t>({8,9,10,11}))return false;
            result={};retry={};
        }else {
            bool again=false;try{(void)retry.get();}catch(const std::bad_alloc &){again=true;}
            if(!again)return false;
        }
        if(!wait_for_data_release(reader))return false;
    }
    {
        auto blocked=std::make_shared<blocking_source>(64);async_reader reader(blocked,1,8);
        auto first=reader.submit(103,{{0,4}});blocked->wait_until_entered();
        auto second=reader.submit(104,{{4,4}});auto stats=reader.stats();
        const bool distinct=stats.staging_resident_bytes==8 && stats.staging_allocated_bytes==4 &&
            stats.queued_reads==1 && stats.active_reads==1 && stats.staging_control_allocated_bytes>0;
        blocked->release();auto a=first.get();auto b=second.get();
        if(!distinct || *a.buffers[0]!=std::vector<uint8_t>({0,1,2,3}) ||
                *b.buffers[0]!=std::vector<uint8_t>({4,5,6,7}))return false;
        first={};second={};a={};b={};if(!wait_for_data_release(reader))return false;
    }
    {
        source->fail_reads(true);async_reader reader(source,1,12,2);
        auto ticket=reader.submit(105,{{0,4},{4,4},{8,4}});bool failed=false;
        try{(void)ticket.get();}catch(const std::runtime_error &){failed=true;}
        ticket={};if(!failed || !wait_for_data_release(reader))return false;
        if(reader.stats().failed_reads!=3)return false;
        source->fail_reads(false);auto retry=reader.submit(106,{{12,4}});auto result=retry.get();
        if(*result.buffers[0]!=std::vector<uint8_t>({12,13,14,15}))return false;
        retry={};result={};if(!wait_for_data_release(reader))return false;
    }
    std::printf("PLE staging allocations: controls=%llu queued/allocated distinction, weak lifetime, partial control failures, source failure and retry passed\n",
        (unsigned long long)control_bytes);
    return true;
}

static bool publication_cases() {
    size_t rejected=0,accepted=0;
    // Keep an existing operation blocked while sweeping quotas through ticket,
    // future, operation, index, queue and owner allocation sites.
    for(uint64_t slack=0;slack<=2048;slack+=16) {
        auto source=std::make_shared<blocking_source>(64);async_reader reader(source,1,16,4);
        auto first=reader.submit(200,{{0,4}});source->wait_until_entered();
        const auto before=reader.stats();
        if(reader.set_metadata_allocation_budget(before.metadata_allocated_bytes-1)) {source->release();return false;}
        if(!reader.set_metadata_allocation_budget(before.metadata_allocated_bytes+slack)) {source->release();return false;}
        async_reader::ticket candidate;bool failed=false;
        try{candidate=reader.submit(201,{{0,4},{4,4},{8,4},{4,4}});}catch(const std::bad_alloc &){failed=true;}
        const auto after=reader.stats();
        const bool intact=!failed || (after.submitted_batches==before.submitted_batches &&
            after.logical_reads==before.logical_reads && after.logical_bytes==before.logical_bytes &&
            after.coalesced_reads==before.coalesced_reads && after.queued_reads==before.queued_reads &&
            after.active_reads==before.active_reads && after.staging_resident_bytes==before.staging_resident_bytes &&
            after.registered_owners==before.registered_owners && source->reads()==1);
        reader.set_metadata_allocation_budget(UINT64_MAX);source->release();
        auto old=first.get();
        if(!intact || *old.buffers[0]!=std::vector<uint8_t>({0,1,2,3})) {
            std::fprintf(stderr,"publication failure slack=%llu rejected=%d submitted=%llu/%llu queue=%llu/%llu reserved=%llu/%llu owners=%llu/%llu\n",
                (unsigned long long)slack,failed,(unsigned long long)after.submitted_batches,(unsigned long long)before.submitted_batches,
                (unsigned long long)after.queued_reads,(unsigned long long)before.queued_reads,
                (unsigned long long)after.staging_resident_bytes,(unsigned long long)before.staging_resident_bytes,
                (unsigned long long)after.registered_owners,(unsigned long long)before.registered_owners);return false;
        }
        if(failed)++rejected;
        else { ++accepted;auto result=candidate.get();if(result.buffers.size()!=4 ||
            *result.buffers[1]!=std::vector<uint8_t>({4,5,6,7}) || result.buffers[1]!=result.buffers[3])return false; }
        candidate={};first={};old={};if(!wait_for_data_release(reader))return false;
        auto retry=reader.submit(202,{{12,4}});auto result=retry.get();
        if(*result.buffers[0]!=std::vector<uint8_t>({12,13,14,15}))return false;
    }
    if(!rejected || !accepted){std::fprintf(stderr,"quota sweep had no split: rejected=%zu accepted=%zu\n",rejected,accepted);return false;}
    for(auto setting:{std::pair<size_t,uint64_t>{64,0},std::pair<size_t,uint64_t>{16,128}}) {
        const auto count=setting.first;const auto headroom=setting.second;
        auto source=std::make_shared<blocking_source>(1024);async_reader reader(source,1,4*(count+1),count);
        auto first=reader.submit(300,{{0,4}});source->wait_until_entered();
        std::vector<read_span> spans;for(uint64_t i=1;i<=count;++i)spans.push_back({i*4,4});
        auto batch=reader.submit(301,spans);
        const auto before=reader.stats();
        if(!reader.set_metadata_allocation_budget(before.metadata_allocated_bytes+headroom)) {source->release();return false;}
        source->release();
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(reader.stats().active_reads || reader.stats().queued_reads) {
            if(std::chrono::steady_clock::now()>deadline)return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Returned batch vectors now use the same quota. Restore it only after
        // worker failure completes so get() observes the stored worker error.
        reader.set_metadata_allocation_budget(UINT64_MAX);
        auto old=first.get();bool failed=false;
        try{(void)batch.get();}catch(const std::bad_alloc &){failed=true;}
        batch={};first={};old={};
        if(!failed || !wait_for_data_release(reader) || reader.stats().failed_reads!=count || source->reads()!=1) {
            std::fprintf(stderr,"worker failure headroom=%llu rejected=%d failures=%llu source_reads=%zu\n",
                (unsigned long long)headroom,failed,(unsigned long long)reader.stats().failed_reads,source->reads());return false;
        }
        auto retry=reader.submit(302,{{512,4}});auto result=retry.get();
        if(*result.buffers[0]!=std::vector<uint8_t>({0,1,2,3}))return false;
    }
    {
        auto source=std::make_shared<blocking_source>(64);source->release();async_reader reader(source,1,8,2);
        const uint64_t limit=reader.stats().metadata_allocated_bytes+4096;
        if(!reader.set_metadata_allocation_budget(limit))return false;
        for(uint64_t owner=0;owner<1000;++owner) {
            { auto ticket=reader.submit(owner,{{0,4}});auto result=ticket.get();
              if(*result.buffers[0]!=std::vector<uint8_t>({0,1,2,3}))return false; }
            if(!wait_for_data_release(reader))return false;
            const auto stats=reader.stats();
            if(stats.registered_owners>1 || stats.metadata_allocated_bytes>limit) {
                std::fprintf(stderr,"owner sweep id=%llu registry=%llu allocated=%llu limit=%llu\n",(unsigned long long)owner,
                    (unsigned long long)stats.registered_owners,(unsigned long long)stats.metadata_allocated_bytes,(unsigned long long)limit);return false;
            }
        }
    }
    {
        auto source=std::make_shared<blocking_source>(64);source->release();async_reader reader(source,1,8,2);
        auto ticket=reader.submit(400,{{0,4}});
        const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
        while(reader.stats().active_reads || reader.stats().queued_reads) {
            if(std::chrono::steady_clock::now()>deadline)return false;
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        auto before=reader.stats();if(!reader.set_metadata_allocation_budget(before.metadata_allocated_bytes))return false;
        bool rejected=false;try{(void)ticket.get();}catch(const std::bad_alloc &){rejected=true;}
        if(!rejected || reader.stats().failed_reads || reader.stats().metadata_allocated_bytes!=before.metadata_allocated_bytes) {
            std::fprintf(stderr,"returned reader batch escaped quota: rejected=%d metadata=%llu/%llu\n",rejected,
                    (unsigned long long)reader.stats().metadata_allocated_bytes,(unsigned long long)before.metadata_allocated_bytes);
            return false;
        }
        reader.set_metadata_allocation_budget(UINT64_MAX);auto result=ticket.get();
        if(*result.buffers[0]!=std::vector<uint8_t>({0,1,2,3}))return false;
        auto held=reader.stats();
        if(held.metadata_allocated_bytes-before.metadata_allocated_bytes!=result.buffers.capacity()*sizeof(result.buffers[0]))return false;
        auto copied=result;auto copy_stats=reader.stats();
        if(copy_stats.metadata_allocated_bytes-held.metadata_allocated_bytes!=copied.buffers.capacity()*sizeof(copied.buffers[0]))return false;
        copied={};result={};ticket={};reader.cancel_owner(400);
        if(!wait_for_data_release(reader))return false;
    }
    std::printf("PLE reader publication: quota_rejections=%zu quota_successes=%zu existing owner preserved; worker prepare/buffer failures recover; 1000 owner IDs bounded\n",rejected,accepted);
    return true;
}

int main() {
    const auto caller = std::this_thread::get_id();
    auto source = std::make_shared<blocking_source>(64);
    async_reader reader(source, 2, 8);

    const auto cancelled_ticket = reader.submit(11, {{4, 4}});
    source->wait_until_entered();
    const auto live_ticket = reader.submit(22, {{4, 4}, {4, 4}, {8, 4}});
    if (reader.cancel_owner(11) != 1) {
        std::fprintf(stderr, "failed to cancel the PLE read owner\n");
        source->release();
        return 1;
    }
    source->release();

    bool cancelled = false;
    try {
        (void) cancelled_ticket.get();
    } catch (const std::runtime_error &) {
        cancelled = true;
    }
    if (!cancelled) {
        std::fprintf(stderr, "cancelled PLE read ticket produced data\n");
        return 1;
    }

    const batch_result result = live_ticket.get();
    if (result.buffers.size() != 3 || *result.buffers[0] != *result.buffers[1] ||
            *result.buffers[0] != std::vector<uint8_t>({4, 5, 6, 7}) ||
            *result.buffers[2] != std::vector<uint8_t>({8, 9, 10, 11})) {
        std::fprintf(stderr, "asynchronous PLE batch returned incorrect bytes\n");
        return 1;
    }
    if (source->read_on_thread(caller)) {
        std::fprintf(stderr, "PLE storage work ran on the submitting thread\n");
        return 1;
    }

    const async_read_stats stats = reader.stats();
    if (stats.submitted_batches != 2 || stats.logical_reads != 4 || stats.logical_bytes != 16 ||
            stats.source_reads != 2 || stats.source_bytes != 8 || stats.coalesced_reads != 2 ||
            stats.source_batches != 2 || stats.peak_source_batch_width != 1 ||
            stats.cancelled_batches != 1 || stats.queued_reads != 0 || stats.active_reads != 0 ||
            stats.peak_queue_depth < 1 || stats.submit_to_ready_p50_ns == 0 ||
            stats.staging_capacity_bytes != 8 || stats.staging_resident_bytes != 8 ||
            stats.staging_peak_resident_bytes != 8 || stats.staging_rejected_batches != 0 ||
            stats.staging_allocated_bytes != 8 || stats.staging_peak_allocated_bytes != 8 ||
            stats.staging_control_allocated_bytes == 0 ||
            source->reads() != 2) {
        std::fprintf(stderr, "incorrect asynchronous PLE telemetry\n");
        return 1;
    }

    bool rejected = false;
    try {
        (void) reader.submit(33, {{0, 1}, {63, 2}});
    } catch (const std::out_of_range &) {
        rejected = true;
    }
    if (!rejected) {
        std::fprintf(stderr, "PLE reader accepted an out-of-bounds span\n");
        return 1;
    }
    const async_read_stats after_rejection = reader.stats();
    if (after_rejection.submitted_batches != stats.submitted_batches ||
            after_rejection.logical_reads != stats.logical_reads || source->reads() != 2) {
        std::fprintf(stderr, "invalid PLE batch was not rejected atomically\n");
        return 1;
    }

    bool staging_rejected = false;
    try {
        (void) reader.submit(44, {{12, 4}});
    } catch (const staging_capacity_error &) {
        staging_rejected = true;
    }
    const async_read_stats after_staging_rejection = reader.stats();
    if (!staging_rejected || after_staging_rejection.submitted_batches != stats.submitted_batches ||
            after_staging_rejection.logical_reads != stats.logical_reads ||
            after_staging_rejection.staging_rejected_batches != 1 || source->reads() != 2) {
        std::fprintf(stderr, "PLE staging capacity rejection was not failure-atomic\n");
        return 1;
    }

    {
        async_reader batch_reader(source, 1, 12, 2);
        const auto batch_ticket = batch_reader.submit(55, {{16, 4}, {20, 4}, {24, 4}});
        const batch_result batch = batch_ticket.get();
        const async_read_stats batch_stats = batch_reader.stats();
        if (batch.buffers.size() != 3 || batch_stats.source_reads != 3 ||
                batch_stats.source_batches != 2 || batch_stats.peak_source_batch_width != 2 ||
                batch_stats.source_bytes != 12 || batch_stats.failed_reads != 0) {
            std::fprintf(stderr, "PLE source batch width was not bounded\n");
            return 1;
        }
    }

    bool invalid_width_rejected = false;
    try {
        async_reader invalid_reader(source, 1, 8, 65);
    } catch (const std::invalid_argument &) {
        invalid_width_rejected = true;
    }
    if (!invalid_width_rejected) {
        std::fprintf(stderr, "PLE reader accepted an invalid source batch width\n");
        return 1;
    }

    if(!allocation_cases()) {
        std::fprintf(stderr,"PLE staging allocation/ownership cases failed\n");return 1;
    }
    if(!publication_cases()) {
        std::fprintf(stderr,"PLE reader metadata/publication cases failed\n");return 1;
    }
    return 0;
}
