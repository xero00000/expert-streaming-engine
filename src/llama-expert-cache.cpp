#include "llama-expert-cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <utility>

#if defined(_WIN32)
#include <malloc.h>
#endif

#if defined(__linux__)
#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace llama_expert_cache {
namespace {

bool checked_add_u64(uint64_t a, uint64_t b, uint64_t & result) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    result = a + b;
    return true;
}

bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) return false;
    result = a*b;
    return true;
}

bool identity_is_zero(const source_identity & value) {
    return value.high == 0 && value.low == 0;
}

uint64_t identity_hash(uint64_t hash, const void * data, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(data);
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

std::string key_string(const expert_key & key) {
    std::ostringstream out;
    out << "layer=" << key.layer << ", expert=" << key.expert << ", component=" << unsigned(key.part);
    return out.str();
}

} // namespace

descriptor::descriptor(descriptor_spec spec, uint64_t bytes) :
    key_(spec.key),
    ggml_type_(spec.ggml_type),
    rank_(spec.rank),
    quant_axis_(spec.quant_axis),
    block_elements_(spec.block_elements),
    block_bytes_(spec.block_bytes),
    source_alignment_(spec.source_alignment),
    dimensions_(spec.dimensions),
    strides_(spec.strides),
    model_identity_(spec.model_identity),
    extents_(std::move(spec.extents)),
    bytes_(bytes) {
}

struct file_source::impl {
    uint32_t source_id;
    source_identity source_identity_value;
    std::string source_path;
    read_backend source_backend;
    uint64_t source_size = 0;

#if defined(__linux__)
    int fd = -1;
    const uint8_t * mapping = nullptr;
    size_t mapping_size = 0;

    int ring_fd = -1;
    void * sq_ring = MAP_FAILED;
    void * cq_ring = MAP_FAILED;
    io_uring_sqe * sqes = nullptr;
    size_t sq_ring_size = 0;
    size_t cq_ring_size = 0;
    size_t sqes_size = 0;
    bool single_ring_mapping = false;

    unsigned * sq_head = nullptr;
    unsigned * sq_tail = nullptr;
    unsigned * sq_mask = nullptr;
    unsigned * sq_array = nullptr;
    unsigned * cq_head = nullptr;
    unsigned * cq_tail = nullptr;
    unsigned * cq_mask = nullptr;
    io_uring_cqe * cqes = nullptr;
    mutable std::mutex ring_mutex;
#endif

    impl(uint32_t id, source_identity identity, std::string path, read_backend backend) :
        source_id(id), source_identity_value(identity), source_path(std::move(path)), source_backend(backend) {
        if (identity_is_zero(source_identity_value)) {
            throw std::invalid_argument("expert file source identity cannot be zero");
        }
        try {
#if defined(__linux__)
        fd = ::open(source_path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) throw std::runtime_error("cannot open expert source '" + source_path + "': " + std::strerror(errno));
        struct stat info = {};
        if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
            const int saved = errno;
            ::close(fd);
            fd = -1;
            throw std::runtime_error("cannot stat non-empty expert source '" + source_path + "': " + std::strerror(saved));
        }
        source_size = uint64_t(info.st_size);

        // Recompute the bounded source identity from the descriptor's actual
        // file descriptor. This catches path replacement or mutation between
        // GGUF indexing and cache construction instead of trusting a stale
        // identity supplied by the caller.
        constexpr size_t identity_sample_bytes = 64*1024;
        std::vector<uint8_t> identity_sample(std::min<uint64_t>(identity_sample_bytes, source_size));
        auto read_identity_sample = [&](uint64_t offset) {
            size_t completed = 0;
            while (completed < identity_sample.size()) {
                const ssize_t result = ::pread(
                        fd, identity_sample.data() + completed,
                        identity_sample.size() - completed, off_t(offset + completed));
                if (result < 0 && errno == EINTR) continue;
                if (result <= 0) {
                    throw std::runtime_error("cannot read expert source identity: " + std::string(std::strerror(errno)));
                }
                completed += size_t(result);
            }
        };
        read_identity_sample(0);
        uint64_t identity_high = 1469598103934665603ULL;
        uint64_t identity_low  = 7809847782465536322ULL;
        identity_high = identity_hash(identity_high, &source_id, sizeof(source_id));
        identity_high = identity_hash(identity_high, &source_size, sizeof(source_size));
        identity_high = identity_hash(identity_high, identity_sample.data(), identity_sample.size());
        if (source_size > identity_sample.size()) {
            read_identity_sample(source_size - identity_sample.size());
        }
        identity_low = identity_hash(identity_low, identity_sample.data(), identity_sample.size());
        identity_low = identity_hash(identity_low, &source_size, sizeof(source_size));
        identity_low = identity_hash(identity_low, &source_id, sizeof(source_id));
        if (source_identity({ identity_high, identity_low }) != source_identity_value) {
            throw std::runtime_error("expert source identity changed before cache construction");
        }

        if (source_backend == read_backend::mmap) {
            if (source_size > std::numeric_limits<size_t>::max()) {
                throw std::overflow_error("expert mmap source exceeds addressable size");
            }
            mapping_size = size_t(source_size);
            void * result = ::mmap(nullptr, mapping_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (result == MAP_FAILED) throw std::runtime_error("cannot mmap expert source: " + std::string(std::strerror(errno)));
            mapping = static_cast<const uint8_t *>(result);
        } else if (source_backend == read_backend::io_uring) {
            io_uring_params params = {};
            ring_fd = int(::syscall(__NR_io_uring_setup, 2u, &params));
            if (ring_fd < 0) throw std::runtime_error("io_uring is unavailable for expert source: " + std::string(std::strerror(errno)));

            sq_ring_size = params.sq_off.array + params.sq_entries*sizeof(unsigned);
            cq_ring_size = params.cq_off.cqes + params.cq_entries*sizeof(io_uring_cqe);
            if (params.features & IORING_FEAT_SINGLE_MMAP) {
                single_ring_mapping = true;
                sq_ring_size = cq_ring_size = std::max(sq_ring_size, cq_ring_size);
            }
            sq_ring = ::mmap(nullptr, sq_ring_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQ_RING);
            if (sq_ring == MAP_FAILED) throw std::runtime_error("cannot map io_uring submission ring: " + std::string(std::strerror(errno)));
            if (single_ring_mapping) {
                cq_ring = sq_ring;
            } else {
                cq_ring = ::mmap(nullptr, cq_ring_size, PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_CQ_RING);
                if (cq_ring == MAP_FAILED) throw std::runtime_error("cannot map io_uring completion ring: " + std::string(std::strerror(errno)));
            }
            sqes_size = params.sq_entries*sizeof(io_uring_sqe);
            void * sqe_mapping = ::mmap(nullptr, sqes_size, PROT_READ | PROT_WRITE,
                    MAP_SHARED | MAP_POPULATE, ring_fd, IORING_OFF_SQES);
            if (sqe_mapping == MAP_FAILED) throw std::runtime_error("cannot map io_uring entries: " + std::string(std::strerror(errno)));
            sqes = static_cast<io_uring_sqe *>(sqe_mapping);

            auto * sq = static_cast<uint8_t *>(sq_ring);
            auto * cq = static_cast<uint8_t *>(cq_ring);
            sq_head = reinterpret_cast<unsigned *>(sq + params.sq_off.head);
            sq_tail = reinterpret_cast<unsigned *>(sq + params.sq_off.tail);
            sq_mask = reinterpret_cast<unsigned *>(sq + params.sq_off.ring_mask);
            sq_array = reinterpret_cast<unsigned *>(sq + params.sq_off.array);
            cq_head = reinterpret_cast<unsigned *>(cq + params.cq_off.head);
            cq_tail = reinterpret_cast<unsigned *>(cq + params.cq_off.tail);
            cq_mask = reinterpret_cast<unsigned *>(cq + params.cq_off.ring_mask);
            cqes = reinterpret_cast<io_uring_cqe *>(cq + params.cq_off.cqes);
        } else if (source_backend != read_backend::pread) {
            throw std::invalid_argument("unknown expert file read backend");
        }
#else
        (void) source_backend;
        throw std::runtime_error("expert file sources currently require Linux");
#endif
        } catch (...) {
            cleanup();
            throw;
        }
    }

    void cleanup() noexcept {
#if defined(__linux__)
        if (sqes) { ::munmap(sqes, sqes_size); sqes = nullptr; }
        if (sq_ring != MAP_FAILED) { ::munmap(sq_ring, sq_ring_size); sq_ring = MAP_FAILED; }
        if (!single_ring_mapping && cq_ring != MAP_FAILED) ::munmap(cq_ring, cq_ring_size);
        cq_ring = MAP_FAILED;
        if (ring_fd >= 0) { ::close(ring_fd); ring_fd = -1; }
        if (mapping) { ::munmap(const_cast<uint8_t *>(mapping), mapping_size); mapping = nullptr; }
        if (fd >= 0) { ::close(fd); fd = -1; }
#endif
    }

    ~impl() { cleanup(); }

    void read_exact(uint64_t offset, void * destination, size_t length) const {
        uint64_t end = 0;
        if (!checked_add_u64(offset, length, end) || end > source_size) {
            throw std::out_of_range("expert file read is outside source bounds");
        }
        if (length == 0) return;
        if (!destination) throw std::invalid_argument("expert file read destination cannot be null");
#if defined(__linux__)
        if (source_backend == read_backend::mmap) {
            std::memcpy(destination, mapping + offset, length);
            return;
        }
        if (source_backend == read_backend::pread) {
            size_t completed = 0;
            while (completed < length) {
                const size_t chunk = std::min(length - completed, size_t(SSIZE_MAX));
                const ssize_t result = ::pread(fd, static_cast<uint8_t *>(destination) + completed, chunk,
                        off_t(offset + completed));
                if (result < 0 && errno == EINTR) continue;
                if (result <= 0) throw std::runtime_error("pread failed for expert source: " + std::string(std::strerror(errno)));
                completed += size_t(result);
            }
            return;
        }

        std::lock_guard<std::mutex> lock(ring_mutex);
        size_t completed = 0;
        while (completed < length) {
            const unsigned chunk = unsigned(std::min(length - completed, size_t(UINT_MAX)));
            const unsigned tail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
            const unsigned index = tail & *sq_mask;
            io_uring_sqe & sqe = sqes[index];
            std::memset(&sqe, 0, sizeof(sqe));
            sqe.opcode = IORING_OP_READ;
            sqe.fd = fd;
            sqe.off = offset + completed;
            sqe.addr = reinterpret_cast<uint64_t>(static_cast<uint8_t *>(destination) + completed);
            sqe.len = chunk;
            sqe.user_data = uint64_t(completed) + 1;
            sq_array[index] = index;
            __atomic_store_n(sq_tail, tail + 1, __ATOMIC_RELEASE);
            int entered;
            do {
                entered = int(::syscall(__NR_io_uring_enter, ring_fd, 1u, 1u, IORING_ENTER_GETEVENTS, nullptr, 0));
            } while (entered < 0 && errno == EINTR);
            if (entered < 0) throw std::runtime_error("io_uring submit failed for expert source: " + std::string(std::strerror(errno)));

            const unsigned head = __atomic_load_n(cq_head, __ATOMIC_ACQUIRE);
            const unsigned completion_tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
            if (head == completion_tail) throw std::runtime_error("io_uring returned without an expert read completion");
            const io_uring_cqe & cqe = cqes[head & *cq_mask];
            if (cqe.res < 0) throw std::runtime_error("io_uring expert read failed: " + std::string(std::strerror(-cqe.res)));
            if (unsigned(cqe.res) != chunk) throw std::runtime_error("io_uring expert read completed short");
            __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
            completed += chunk;
        }
#else
        (void) offset; (void) destination; (void) length;
#endif
    }
};

file_source::file_source(uint32_t id, source_identity identity, std::string path, read_backend backend) :
    impl_(new impl(id, identity, std::move(path), backend)) {
}

file_source::~file_source() = default;
uint32_t file_source::id() const { return impl_->source_id; }
source_identity file_source::identity() const { return impl_->source_identity_value; }
uint64_t file_source::size() const { return impl_->source_size; }
read_backend file_source::backend() const { return impl_->source_backend; }
void file_source::read(uint64_t offset, void * destination, size_t length) const { impl_->read_exact(offset, destination, length); }
const std::string & file_source::path() const { return impl_->source_path; }

std::shared_ptr<const descriptor> descriptor::make(
        descriptor_spec spec,
        const std::map<uint32_t, uint64_t> & source_sizes) {
    if (spec.rank == 0 || spec.rank > spec.dimensions.size()) {
        throw std::invalid_argument("expert descriptor rank must be in [1,4]");
    }
    if (spec.quant_axis >= spec.rank) {
        throw std::invalid_argument("expert descriptor quantization axis is outside its rank");
    }
    if (spec.block_elements == 0 || spec.block_bytes == 0) {
        throw std::invalid_argument("expert descriptor quantization geometry cannot contain zero");
    }
    if (spec.source_alignment == 0 || (spec.source_alignment & (spec.source_alignment - 1)) != 0) {
        throw std::invalid_argument("expert descriptor source alignment must be a power of two");
    }
    if (identity_is_zero(spec.model_identity)) {
        throw std::invalid_argument("expert descriptor model identity cannot be zero");
    }
    if (spec.extents.empty()) {
        throw std::invalid_argument("expert descriptor requires at least one source extent");
    }

    uint64_t expected_bytes = 1;
    uint64_t tensor_span = spec.block_bytes;
    for (uint8_t axis = 0; axis < spec.rank; ++axis) {
        if (spec.dimensions[axis] == 0) {
            throw std::invalid_argument("expert descriptor dimensions cannot contain zero");
        }
        uint64_t axis_units = spec.dimensions[axis];
        uint64_t axis_bytes = axis_units;
        if (axis == spec.quant_axis) {
            uint64_t rounded = 0;
            if (!checked_add_u64(axis_bytes, spec.block_elements - 1, rounded)) {
                throw std::overflow_error("expert descriptor quantized dimension overflows");
            }
            axis_units = rounded/spec.block_elements;
            if (!checked_mul_u64(axis_units, spec.block_bytes, axis_bytes)) {
                throw std::overflow_error("expert descriptor quantized row bytes overflow");
            }
        }
        if (!checked_mul_u64(expected_bytes, axis_bytes, expected_bytes)) {
            throw std::overflow_error("expert descriptor tensor geometry overflows");
        }
        if (spec.strides[axis] == 0) {
            throw std::invalid_argument("expert descriptor active strides cannot contain zero");
        }
        uint64_t axis_offset = 0;
        if (!checked_mul_u64(axis_units - 1, spec.strides[axis], axis_offset) ||
                !checked_add_u64(tensor_span, axis_offset, tensor_span)) {
            throw std::overflow_error("expert descriptor stride geometry overflows");
        }
    }
    for (uint8_t axis = spec.rank; axis < spec.dimensions.size(); ++axis) {
        if (spec.dimensions[axis] != 0 || spec.strides[axis] != 0) {
            throw std::invalid_argument("expert descriptor inactive dimensions and strides must be zero");
        }
    }
    if (tensor_span != expected_bytes) {
        throw std::invalid_argument("expert descriptor strides are inconsistent with packed tensor geometry");
    }

    uint64_t extent_bytes = 0;
    std::map<uint32_t, std::vector<std::pair<uint64_t, uint64_t>>> intervals;
    for (const auto & extent : spec.extents) {
        const auto source_size = source_sizes.find(extent.source);
        if (source_size == source_sizes.end()) {
            throw std::invalid_argument("expert descriptor references an unknown source");
        }
        if (identity_is_zero(extent.identity)) {
            throw std::invalid_argument("expert descriptor source identity cannot be zero");
        }
        if (extent.length == 0) {
            throw std::invalid_argument("expert descriptor source extents cannot be empty");
        }
        if (extent.offset % spec.source_alignment != 0 || extent.length % spec.source_alignment != 0) {
            throw std::invalid_argument("expert descriptor source extent violates declared alignment");
        }
        uint64_t end = 0;
        if (!checked_add_u64(extent.offset, extent.length, end) || end > source_size->second) {
            throw std::out_of_range("expert descriptor source extent is outside file bounds");
        }
        if (!checked_add_u64(extent_bytes, extent.length, extent_bytes)) {
            throw std::overflow_error("expert descriptor source extent bytes overflow");
        }
        intervals[extent.source].push_back({ extent.offset, end });
    }
    for (auto & source_intervals : intervals) {
        auto & ranges = source_intervals.second;
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                throw std::invalid_argument("expert descriptor source extents overlap");
            }
        }
    }
    if (extent_bytes != expected_bytes) {
        throw std::invalid_argument("expert descriptor extents do not match quantized tensor geometry");
    }

    return std::shared_ptr<const descriptor>(new descriptor(std::move(spec), expected_bytes));
}

ram_cache::lease::lease(std::shared_ptr<entry> value) : entry_(std::move(value)) {
    entry_->pins.fetch_add(1, std::memory_order_acq_rel);
}

ram_cache::lease::lease(lease && other) noexcept : entry_(std::move(other.entry_)) {
}

ram_cache::lease & ram_cache::lease::operator=(lease && other) noexcept {
    if (this != &other) {
        release();
        entry_ = std::move(other.entry_);
    }
    return *this;
}

ram_cache::lease::~lease() {
    release();
}

void ram_cache::lease::release() {
    if (entry_) {
        entry_->pins.fetch_sub(1, std::memory_order_acq_rel);
        entry_.reset();
    }
}

const uint8_t * ram_cache::lease::data() const {
    return entry_ ? entry_->data : nullptr;
}

size_t ram_cache::lease::size() const {
    return entry_ ? size_t(entry_->desc->bytes()) : 0;
}

ram_cache::ram_cache(cache_config config, std::vector<std::shared_ptr<const source>> sources) : config_(config) {
    if (config_.capacity_bytes == 0) {
        throw std::invalid_argument("expert RAM cache capacity must be non-zero");
    }
    if (config_.staging_bytes == 0 || config_.staging_bytes > config_.capacity_bytes) {
        throw std::invalid_argument("expert RAM cache staging capacity must be in [1, capacity]");
    }
    if (config_.capacity_bytes > std::numeric_limits<size_t>::max() ||
            config_.staging_bytes > std::numeric_limits<size_t>::max()) {
        throw std::overflow_error("expert RAM cache capacity exceeds addressable memory");
    }
    void * arena = nullptr;
#if defined(_WIN32)
    arena = _aligned_malloc(size_t(config_.capacity_bytes), 64);
    const int allocation_error = arena ? 0 : 1;
#else
    const int allocation_error = ::posix_memalign(&arena, 64, size_t(config_.capacity_bytes));
#endif
    if (allocation_error != 0 || !arena) {
        throw std::bad_alloc();
    }
    resident_arena_.reset(static_cast<uint8_t *>(arena));
    staging_.resize(size_t(config_.staging_bytes));
    for (auto & item : sources) {
        if (!item) throw std::invalid_argument("expert RAM cache source cannot be null");
        if (item->size() == 0 || identity_is_zero(item->identity())) {
            throw std::invalid_argument("expert RAM cache source size and identity must be non-zero");
        }
        if (!sources_.emplace(item->id(), std::move(item)).second) {
            throw std::invalid_argument("expert RAM cache source IDs must be unique");
        }
    }
    stats_.capacity_bytes = config_.capacity_bytes;
}

void ram_cache::aligned_free::operator()(uint8_t * pointer) const noexcept {
#if defined(_WIN32)
    _aligned_free(pointer);
#else
    std::free(pointer);
#endif
}

bool ram_cache::checked_add(uint64_t a, uint64_t b, uint64_t & result) {
    return checked_add_u64(a, b, result);
}

bool ram_cache::allocation_plan(
        uint64_t bytes,
        uint64_t & arena_offset,
        std::vector<std::shared_ptr<entry>> & victims) const {
    std::vector<std::shared_ptr<entry>> candidates;
    candidates.reserve(entries_.size());
    for (const auto & item : entries_) {
        if (item.second->pins.load(std::memory_order_acquire) == 0) {
            candidates.push_back(item.second);
        }
    }
    std::sort(candidates.begin(), candidates.end(), [](const std::shared_ptr<entry> & a, const std::shared_ptr<entry> & b) {
        if (a->last_used != b->last_used) return a->last_used < b->last_used;
        if (a->insertion_order != b->insertion_order) return a->insertion_order < b->insertion_order;
        return a->desc->key() < b->desc->key();
    });

    auto find_first_fit = [&]() {
        std::vector<std::pair<uint64_t, uint64_t>> occupied;
        occupied.reserve(entries_.size());
        for (const auto & item : entries_) {
            const bool removed = std::find(victims.begin(), victims.end(), item.second) != victims.end();
            if (removed) continue;
            uint64_t end = 0;
            if (!checked_add_u64(item.second->arena_offset, item.second->desc->bytes(), end) ||
                    end > config_.capacity_bytes) {
                throw std::logic_error("expert RAM cache arena metadata is outside capacity");
            }
            occupied.push_back({ item.second->arena_offset, end });
        }
        std::sort(occupied.begin(), occupied.end());
        uint64_t cursor = 0;
        auto aligned_cursor = [](uint64_t value, uint64_t & result) {
            uint64_t rounded = 0;
            if (!checked_add_u64(value, 63, rounded)) return false;
            result = rounded & ~uint64_t(63);
            return true;
        };
        for (const auto & interval : occupied) {
            if (interval.first < cursor) {
                throw std::logic_error("expert RAM cache arena entries overlap");
            }
            uint64_t aligned = 0;
            if (aligned_cursor(cursor, aligned) && aligned <= interval.first && interval.first - aligned >= bytes) {
                arena_offset = aligned;
                return true;
            }
            cursor = interval.second;
        }
        uint64_t aligned = 0;
        if (aligned_cursor(cursor, aligned) && aligned <= config_.capacity_bytes &&
                config_.capacity_bytes - aligned >= bytes) {
            arena_offset = aligned;
            return true;
        }
        return false;
    };

    victims.clear();
    if (find_first_fit()) return true;
    for (const auto & candidate : candidates) {
        victims.push_back(candidate);
        if (find_first_fit()) return true;
    }
    victims.clear();
    return false;
}

ram_cache::lease ram_cache::acquire(std::shared_ptr<const descriptor> value) {
    if (!value) throw std::invalid_argument("cannot acquire a null expert descriptor");
    if (value->bytes() > config_.capacity_bytes || value->bytes() > config_.staging_bytes ||
            value->bytes() > std::numeric_limits<size_t>::max()) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.rejected_admissions;
        throw capacity_error("expert component exceeds RAM cache or staging capacity: " + key_string(value->key()));
    }

    const auto wait_start = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    const bool waited = !lock.try_lock();
    if (waited) lock.lock();
    if (waited) {
        ++stats_.staging_waits;
        stats_.staging_wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_start).count());
    }
    const auto canonical = canonical_descriptors_.find(value->key());
    if (canonical == canonical_descriptors_.end()) {
        canonical_descriptors_.emplace(value->key(), value);
    } else if (canonical->second != value) {
        throw std::invalid_argument("expert key was reused with a different immutable descriptor");
    }
    const auto cached = entries_.find(value->key());
    if (cached != entries_.end()) {
        if (cached->second->desc != value) {
            throw std::logic_error("expert cache descriptor registry is inconsistent");
        }
        cached->second->last_used = ++clock_;
        ++stats_.hits;
        return lease(cached->second);
    }

    ++stats_.misses;
    auto loaded = std::make_shared<entry>();
    loaded->desc = std::move(value);
    const size_t loaded_bytes = size_t(loaded->desc->bytes());

    size_t destination_offset = 0;
    std::array<uint64_t, size_t(read_backend::count)> bytes_read = {{ 0, 0, 0 }};
    const auto read_start = std::chrono::steady_clock::now();
    for (const auto & extent : loaded->desc->extents()) {
        const auto source_it = sources_.find(extent.source);
        if (source_it == sources_.end() || source_it->second->identity() != extent.identity) {
            throw std::invalid_argument("expert descriptor source identity does not match the registered source");
        }
        source_it->second->read(extent.offset, staging_.data() + destination_offset, size_t(extent.length));
        destination_offset += size_t(extent.length);
        const size_t backend_index = size_t(source_it->second->backend());
        if (!checked_add_u64(bytes_read[backend_index], extent.length, bytes_read[backend_index])) {
            throw std::overflow_error("expert RAM cache backend byte count overflow");
        }
    }
    stats_.storage_read_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - read_start).count());
    for (size_t i = 0; i < bytes_read.size(); ++i) stats_.bytes_read[i] += bytes_read[i];

    uint64_t post_insert = 0;
    if (!checked_add_u64(stats_.resident_bytes, loaded->desc->bytes(), post_insert)) {
        throw std::overflow_error("expert RAM cache resident byte count overflow");
    }
    uint64_t arena_offset = 0;
    std::vector<std::shared_ptr<entry>> plan;
    if (!allocation_plan(loaded->desc->bytes(), arena_offset, plan)) {
        ++stats_.pinned_capacity_failures;
        throw capacity_error("all eviction candidates are protected by in-flight expert leases");
    }

    // The only allocation that can fail is the map node below. The resident
    // payload already has a fixed arena, so no old bytes are overwritten until
    // that allocation succeeds. Planned victims are unpinned and the mutex
    // prevents them from becoming observable during the non-throwing commit.
    loaded->arena_offset = arena_offset;
    loaded->data = resident_arena_.get() + size_t(arena_offset);
    loaded->last_used = ++clock_;
    loaded->insertion_order = ++insertion_clock_;
    const auto inserted = entries_.emplace(loaded->desc->key(), loaded);
    if (!inserted.second) {
        throw std::logic_error("expert RAM cache insertion raced with itself");
    }
    std::memcpy(const_cast<uint8_t *>(loaded->data), staging_.data(), loaded_bytes);
    for (const auto & victim : plan) {
        stats_.resident_bytes -= victim->desc->bytes();
        entries_.erase(victim->desc->key());
        ++stats_.evictions;
    }
    stats_.resident_bytes += loaded->desc->bytes();
    stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
    return lease(std::move(loaded));
}

bool ram_cache::erase(const expert_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found == entries_.end() || found->second->pins.load(std::memory_order_acquire) != 0) return false;
    stats_.resident_bytes -= found->second->desc->bytes();
    entries_.erase(found);
    return true;
}

cache_stats ram_cache::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    cache_stats result = stats_;
    result.active_leases = 0;
    for (const auto & item : entries_) {
        result.active_leases += item.second->pins.load(std::memory_order_acquire);
    }
    return result;
}

void ram_cache::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto & item : entries_) {
        if (item.second->pins.load(std::memory_order_acquire) != 0) {
            throw capacity_error("cannot clear expert RAM cache while leases are in flight");
        }
    }
    entries_.clear();
    stats_.resident_bytes = 0;
}

} // namespace llama_expert_cache
