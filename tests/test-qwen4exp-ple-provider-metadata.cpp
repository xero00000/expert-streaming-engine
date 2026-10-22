#include "llama-qwen4exp-ple-storage.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <thread>
#include <string>

using namespace qwen4exp_ple_storage;

class metadata_source final : public llama_expert_cache::source {
public:
    metadata_source() : bytes(128*90) {
        for (size_t row=0;row<128;++row) for(size_t block=0;block<5;++block) {
            bytes[row*90+block*18]=0;
            bytes[row*90+block*18+1]=0x3c; // finite F16 scale 1
            for(size_t q=2;q<18;++q)bytes[row*90+block*18+q]=uint8_t(row+q);
        }
    }
    uint32_t id() const override { return 18; }
    llama_expert_cache::source_identity identity() const override { return {18,19}; }
    uint64_t size() const override { return bytes.size(); }
    llama_expert_cache::read_backend backend() const override { return llama_expert_cache::read_backend::pread; }
    void read(uint64_t offset,void * dst,size_t length) const override {
        if(offset>bytes.size() || length>bytes.size()-offset)throw std::out_of_range("metadata source");
        ++reads;
        std::memcpy(dst,bytes.data()+offset,length);
    }
    std::vector<uint8_t> bytes;
    mutable std::atomic<uint64_t> reads{0};
};

template<class Store> static bool idle(Store & store) {
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    do {
        auto s=store.stats().reads;
        if(!s.active_reads && !s.queued_reads)return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while(std::chrono::steady_clock::now()<deadline);
    return false;
}

template<class Batch> static bool exact(const Batch & batch,const metadata_source & source,uint64_t row) {
    if(batch.size()!=1)return false;
    float actual[160],expected[160];batch.dequantize(actual,160);
    qwen4exp_ple_dequantize_iq4_nl(source.bytes.data()+row*90,90,expected,160);
    return std::memcmp(actual,expected,sizeof(actual))==0;
}

template<class Store,class Factory> static bool qualify(const char * name,Factory factory) {
    size_t rejected=0,admitted=0;
    for(uint64_t owner:{7,8}) for(uint64_t slack=0;slack<=1024;slack+=16) {
        auto source=std::make_shared<metadata_source>();auto store=factory(source);
        auto original=store->submit(7,{0});auto warm=original.get();
        if(!exact(warm,*source,0) || !idle(*store))return false;
        auto before=store->stats();const auto physical=source->reads.load();
        if(store->set_metadata_allocation_budget(before.metadata.allocated_bytes-1) ||
                !store->set_metadata_allocation_budget(before.metadata.allocated_bytes+slack))return false;
        typename Store::ticket candidate;bool failed=false;
        try { candidate=store->submit(owner,{50}); } catch(const std::bad_alloc &) { failed=true; }
        if(failed) {
            ++rejected;auto after=store->stats();
            if(after.reads.submitted_batches!=before.reads.submitted_batches ||
                    after.reads.logical_reads!=before.reads.logical_reads ||
                    after.reads.staging_resident_bytes!=before.reads.staging_resident_bytes ||
                    after.reads.active_reads || after.reads.queued_reads ||
                    after.metadata.registered_owners!=1 || source->reads.load()!=physical) {
                std::fprintf(stderr,"%s rejected dispatch owner=%llu slack=%llu submitted=%llu/%llu\n",name,
                        (unsigned long long)owner,(unsigned long long)slack,
                        (unsigned long long)after.reads.submitted_batches,(unsigned long long)before.reads.submitted_batches);
                return false;
            }
        } else {
            ++admitted;auto batch=candidate.get();if(!exact(batch,*source,50))return false;
        }
        candidate={};if(!idle(*store))return false;
        if(!store->set_metadata_allocation_budget(UINT64_MAX) || store->cancel_owner(7)!=1 ||
                store->cancel_owner(7)!=0)return false;
        original={};warm={};(void)store->cancel_owner(8);
        if(!idle(*store) || store->stats().metadata.allocated_bytes!=0)return false;
        auto retry=store->submit(9,{50});auto batch=retry.get();if(!exact(batch,*source,50))return false;
    }
    if(!rejected || !admitted)return false;
    // A failed child admission must erase only the unpublished outer ticket.
    {
        auto source=std::make_shared<metadata_source>();auto store=factory(source);
        auto original=store->submit(7,{0});auto warm=original.get();if(!idle(*store))return false;
        auto before=store->stats();bool failed=false;
        try {
            if(std::string(name)=="rows") (void)store->submit(7,{1,2,3,4});
            else (void)store->submit(7,{50,100});
        } catch(const staging_capacity_error &) {failed=true;}
        auto after=store->stats();
        if(!failed || after.metadata.registered_owners!=1 ||
                after.reads.submitted_batches!=before.reads.submitted_batches || store->cancel_owner(7)!=1)return false;
    }
    // Distinct expired owners cannot grow the retained registry/weak controls.
    {
        auto source=std::make_shared<metadata_source>();auto store=factory(source);
        auto batch=store->submit(1,{0}).get();if(!idle(*store))return false;
        const uint64_t limit=store->stats().metadata.allocated_bytes+1024;
        if(!store->set_metadata_allocation_budget(limit))return false;
        for(uint64_t owner=100;owner<1100;++owner) {
            { auto ticket=store->submit(owner,{0});auto result=ticket.get();if(!exact(result,*source,0))return false; }
            auto s=store->stats();
            if(s.metadata.registered_owners>1 || s.metadata.allocated_bytes>limit) {
                std::fprintf(stderr,"%s expired owner registry grew: owner=%llu entries=%llu allocated=%llu limit=%llu\n",
                        name,(unsigned long long)owner,(unsigned long long)s.metadata.registered_owners,
                        (unsigned long long)s.metadata.allocated_bytes,(unsigned long long)limit);
                return false;
            }
        }
    }
    // Lease/vector backing stays in its originating ledger after consuming and
    // dropping tickets, and after moves between independent providers.
    {
        auto source=std::make_shared<metadata_source>();auto left=factory(source);auto right=factory(source);
        auto a_ticket=left->submit(7,{0});auto a=a_ticket.get();
        auto b_ticket=right->submit(7,{0,0,0});auto b=b_ticket.get();
        a_ticket={};b_ticket={};left->cancel_owner(7);right->cancel_owner(7);
        if(!idle(*left) || !idle(*right))return false;
        const auto left_bytes=left->stats().metadata.allocated_bytes;
        const auto right_bytes=right->stats().metadata.allocated_bytes;
        if(!left_bytes || right_bytes<=left_bytes) {
            std::fprintf(stderr,"%s returned vector accounting lost: left=%llu right=%llu\n",name,
                    (unsigned long long)left_bytes,(unsigned long long)right_bytes);return false;
        }
        if(!left->set_metadata_allocation_budget(left_bytes))return false;
        bool rejected=false;auto before=left->stats();
        try{(void)left->submit(8,{50});}catch(const std::bad_alloc &){rejected=true;}
        if(!rejected || left->stats().reads.submitted_batches!=before.reads.submitted_batches)return false;
        std::swap(a,b);
        if(left->stats().metadata.allocated_bytes!=left_bytes || right->stats().metadata.allocated_bytes!=right_bytes)return false;
        b={};
        if(left->stats().metadata.allocated_bytes!=0 || right->stats().metadata.allocated_bytes!=right_bytes)return false;
        a={};if(right->stats().metadata.allocated_bytes!=0)return false;
        left->set_metadata_allocation_budget(UINT64_MAX);
        auto retry=left->submit(9,{50});auto result=retry.get();if(!exact(result,*source,50))return false;
    }
    {
        auto source=std::make_shared<metadata_source>();typename Store::staged_batch retained;
        {auto store=factory(source);retained=store->submit(9,{50}).get();}
        if(!exact(retained,*source,50))return false;
        retained={}; // Must release via the still-live originating allocator.
    }
    // Pointer/count inputs reject nonempty null spans and preserve the original
    // vector API; successful internal forwarding does not make an untracked copy.
    {
        auto source=std::make_shared<metadata_source>();auto store=factory(source);bool rejected=false;
        try{(void)store->submit(10,nullptr,1);}catch(const std::invalid_argument &){rejected=true;}
        if(!rejected || store->stats().reads.submitted_batches)return false;
    }
    std::printf("%s provider metadata: %zu quota rejections, %zu admissions; exact retry, child rollback, retained/moved vectors and 1000 owner IDs pass\n",name,rejected,admitted);
    return true;
}

int main() {
    qwen4exp_ple_hash_config config;qwen4exp_ple_extent_index extents(config,0,0,90,128);
    if(!qualify<row_store>("rows",[&](auto source){return std::make_unique<row_store>(source,extents,1,360);}) ||
       !qualify<page_store>("pages",[&](auto source){return std::make_unique<page_store>(source,extents,1,8192);}) ||
       !qualify<two_tier_store>("two-tier",[&](auto source){return std::make_unique<two_tier_store>(source,extents,1,8192,360);})) {
        std::fprintf(stderr,"Provider metadata/publication qualification failed\n");return 1;
    }
    return 0;
}
