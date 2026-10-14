#include "llama-expert-cache.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <set>
#include <iterator>
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

#if defined(__linux__)
source_identity identity_from_fd(int fd, uint32_t source_id, uint64_t source_size) {
    constexpr size_t identity_sample_bytes = 64*1024;
    std::vector<uint8_t> sample(std::min<uint64_t>(identity_sample_bytes, source_size));
    auto read_sample = [&](uint64_t offset) {
        size_t completed = 0;
        while (completed < sample.size()) {
            const ssize_t result = ::pread(
                    fd, sample.data() + completed, sample.size() - completed,
                    off_t(offset + completed));
            if (result < 0 && errno == EINTR) continue;
            if (result <= 0) {
                throw std::runtime_error("cannot read expert source identity: " + std::string(std::strerror(errno)));
            }
            completed += size_t(result);
        }
    };
    read_sample(0);
    uint64_t high = 1469598103934665603ULL;
    uint64_t low  = 7809847782465536322ULL;
    high = identity_hash(high, &source_id, sizeof(source_id));
    high = identity_hash(high, &source_size, sizeof(source_size));
    high = identity_hash(high, sample.data(), sample.size());
    if (source_size > sample.size()) read_sample(source_size - sample.size());
    low = identity_hash(low, sample.data(), sample.size());
    low = identity_hash(low, &source_size, sizeof(source_size));
    low = identity_hash(low, &source_id, sizeof(source_id));
    return {high, low};
}
#endif

std::string key_string(const expert_key & key) {
    std::ostringstream out;
    out << "layer=" << key.layer << ", expert=" << key.expert << ", component=" << unsigned(key.part);
    return out.str();
}

} // namespace

source_identity identify_file_source(const std::string & path, uint32_t source_id) {
#if defined(__linux__)
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) throw std::runtime_error("cannot open expert source '" + path + "': " + std::strerror(errno));
    struct fd_cleanup { int value; ~fd_cleanup() { ::close(value); } } cleanup{fd};
    struct stat info = {};
    if (::fstat(fd, &info) != 0 || info.st_size <= 0) {
        throw std::runtime_error("cannot stat non-empty expert source '" + path + "'");
    }
    return identity_from_fd(fd, source_id, uint64_t(info.st_size));
#else
    (void) path; (void) source_id;
    throw std::runtime_error("expert file identity currently requires Linux");
#endif
}

void source::read_many(const std::vector<read_request> & requests) const {
    for (const read_request & request : requests) {
        read(request.offset, request.destination, request.length);
    }
}

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
    unsigned ring_entries = 0;

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

        // Recompute from the descriptor's actual open file to catch replacement
        // or mutation between indexing and cache construction.
        if (identity_from_fd(fd, source_id, source_size) != source_identity_value) {
            throw std::runtime_error("expert source identity changed before cache construction");
        }
        if (const char * advice = std::getenv("ESE_EXPERT_FILE_ADVICE")) {
            if (std::strcmp(advice, "normal") != 0 && std::strcmp(advice, "random") != 0)
                throw std::invalid_argument("ESE_EXPERT_FILE_ADVICE must be normal or random");
            if (std::strcmp(advice, "random") == 0 && ::posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM) != 0)
                throw std::runtime_error("cannot set random-access advice for expert source");
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
            ring_fd = int(::syscall(__NR_io_uring_setup, 64u, &params));
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
            ring_entries = params.sq_entries;

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
            const io_uring_cqe cqe = cqes[head & *cq_mask];
            // Consume even a failed completion before reporting it. Otherwise
            // the next read observes this stale result instead of its own.
            __atomic_store_n(cq_head, head + 1, __ATOMIC_RELEASE);
            if (cqe.user_data != uint64_t(completed) + 1)
                throw std::runtime_error("io_uring expert read returned invalid completion identity");
            if (cqe.res < 0) throw std::runtime_error("io_uring expert read failed: " + std::string(std::strerror(-cqe.res)));
            if (unsigned(cqe.res) != chunk) throw std::runtime_error("io_uring expert read completed short");
            completed += chunk;
        }
#else
        (void) offset; (void) destination; (void) length;
#endif
    }

    void read_many(const std::vector<source::read_request> & requests) const {
        for (const source::read_request & request : requests) {
            uint64_t end = 0;
            if (!checked_add_u64(request.offset, request.length, end) || end > source_size) {
                throw std::out_of_range("expert file batch read is outside source bounds");
            }
            if (request.length != 0 && request.destination == nullptr) {
                throw std::invalid_argument("expert file batch read destination cannot be null");
            }
        }
        if (requests.empty()) return;
#if defined(__linux__)
        if (source_backend != read_backend::io_uring) {
            for (const source::read_request & request : requests) {
                read_exact(request.offset, request.destination, request.length);
            }
            return;
        }
        for (const source::read_request & request : requests) {
            if (request.length > UINT_MAX) {
                for (const source::read_request & fallback : requests) {
                    read_exact(fallback.offset, fallback.destination, fallback.length);
                }
                return;
            }
        }

        std::lock_guard<std::mutex> lock(ring_mutex);
        size_t first = 0;
        while (first < requests.size()) {
            const unsigned count = unsigned(std::min<size_t>(ring_entries, requests.size() - first));
            const unsigned tail = __atomic_load_n(sq_tail, __ATOMIC_RELAXED);
            for (unsigned index = 0; index < count; ++index) {
                const unsigned ring_index = (tail + index) & *sq_mask;
                const source::read_request & request = requests[first + index];
                io_uring_sqe & sqe = sqes[ring_index];
                std::memset(&sqe, 0, sizeof(sqe));
                sqe.opcode = IORING_OP_READ;
                sqe.fd = fd;
                sqe.off = request.offset;
                sqe.addr = reinterpret_cast<uint64_t>(request.destination);
                sqe.len = unsigned(request.length);
                sqe.user_data = uint64_t(index) + 1;
                sq_array[ring_index] = ring_index;
            }
            __atomic_store_n(sq_tail, tail + count, __ATOMIC_RELEASE);

            int entered;
            do {
                entered = int(::syscall(
                    __NR_io_uring_enter, ring_fd, count, count,
                    IORING_ENTER_GETEVENTS, nullptr, 0));
            } while (entered < 0 && errno == EINTR);
            if (entered < 0) {
                throw std::runtime_error("io_uring batch submit failed for expert source: " +
                    std::string(std::strerror(errno)));
            }
            if (unsigned(entered) != count) {
                throw std::runtime_error("io_uring batch submission completed short");
            }

            unsigned head = __atomic_load_n(cq_head, __ATOMIC_ACQUIRE);
            unsigned completion_tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
            while (completion_tail - head < count) {
                const unsigned missing = count - (completion_tail - head);
                int waited;
                do {
                    waited = int(::syscall(
                        __NR_io_uring_enter, ring_fd, 0u, missing,
                        IORING_ENTER_GETEVENTS, nullptr, 0));
                } while (waited < 0 && errno == EINTR);
                if (waited < 0) {
                    throw std::runtime_error("io_uring batch wait failed for expert source: " +
                        std::string(std::strerror(errno)));
                }
                completion_tail = __atomic_load_n(cq_tail, __ATOMIC_ACQUIRE);
            }
            // All requests are finished. Retire their CQ entries before
            // validating results, including an error or short read.
            std::array<io_uring_cqe, 64> results;
            for (unsigned completion = 0; completion < count; ++completion)
                results[completion] = cqes[(head + completion) & *cq_mask];
            __atomic_store_n(cq_head, head + count, __ATOMIC_RELEASE);
            uint64_t completed = 0;
            for (unsigned completion = 0; completion < count; ++completion) {
                const io_uring_cqe & cqe = results[completion];
                if (cqe.user_data == 0 || cqe.user_data > count ||
                        (completed & (uint64_t(1) << (cqe.user_data - 1)))) {
                    throw std::runtime_error("io_uring batch returned invalid completion identity");
                }
                const unsigned request_index = unsigned(cqe.user_data - 1);
                completed |= uint64_t(1) << request_index;
                if (cqe.res < 0) {
                    throw std::runtime_error("io_uring batch read failed for expert source: " +
                        std::string(std::strerror(-cqe.res)));
                }
                if (size_t(cqe.res) != requests[first + request_index].length) {
                    throw std::runtime_error("io_uring batch read completed short");
                }
            }
            first += count;
        }
#else
        (void) requests;
#endif
    }

    void evict_range_from_os_cache(uint64_t offset, uint64_t length) const {
        uint64_t end = 0;
        if (length == 0 || !checked_add_u64(offset, length, end) || end > source_size) {
            throw std::out_of_range("expert cache eviction range is outside source bounds");
        }
#if defined(__linux__)
        if (mapping) {
            const long system_page_size = ::sysconf(_SC_PAGESIZE);
            if (system_page_size <= 0) {
                throw std::runtime_error("cannot determine system page size for expert cache eviction");
            }
            const uint64_t page_size = uint64_t(system_page_size);
            const uint64_t aligned_offset = offset - offset % page_size;
            const uint64_t advised_length = end - aligned_offset;
            if (advised_length > std::numeric_limits<size_t>::max() ||
                    ::madvise(const_cast<uint8_t *>(mapping) + aligned_offset,
                        size_t(advised_length), MADV_DONTNEED) != 0) {
                throw std::runtime_error("madvise DONTNEED failed for expert source: " +
                    std::string(std::strerror(errno)));
            }
        }
        const int result = ::posix_fadvise(fd, off_t(offset), off_t(length), POSIX_FADV_DONTNEED);
        if (result != 0) {
            throw std::runtime_error("posix_fadvise DONTNEED failed for expert source: " +
                std::string(std::strerror(result)));
        }
#else
        (void) offset;
        (void) length;
        throw std::runtime_error("expert OS cache eviction currently requires Linux");
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
void file_source::read_many(const std::vector<read_request> & requests) const {
    impl_->read_many(requests);
}
bool file_source::advise_will_need(uint64_t offset, uint64_t length) const {
    if (offset > size() || length > size() - offset) throw std::out_of_range("expert advice outside source");
    if (!length) return true;
#if defined(__linux__)
    if (offset > uint64_t(std::numeric_limits<off_t>::max()) ||
            length > uint64_t(std::numeric_limits<off_t>::max()) - offset) return false;
    return ::posix_fadvise(impl_->fd, off_t(offset), off_t(length), POSIX_FADV_WILLNEED) == 0;
#else
    return false;
#endif
}
void file_source::evict_range_from_os_cache(uint64_t offset, uint64_t length) const {
    impl_->evict_range_from_os_cache(offset, length);
}
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

ram_cache::ram_cache(cache_config config, std::vector<std::shared_ptr<const source>> sources,
        std::shared_ptr<uint8_t> resident_arena) : config_(config) {
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
    if (resident_arena) {
        if (reinterpret_cast<uintptr_t>(resident_arena.get()) % 64 != 0)
            throw std::invalid_argument("expert RAM external arena must be aligned to 64 bytes");
        resident_arena_ = std::move(resident_arena);
    } else {
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
        resident_arena_ = std::shared_ptr<uint8_t>(static_cast<uint8_t *>(arena), aligned_free{});
    }
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
    stats_.staging_capacity_bytes = config_.staging_bytes;
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

namespace {

// Reservations can overlap their still-readable eviction candidates. Merge
// those intervals when finding gaps rather than treating them as extra bytes.
bool arena_first_fit(std::vector<std::pair<uint64_t, uint64_t>> occupied,
        uint64_t capacity, uint64_t bytes, uint64_t alignment, uint64_t & offset) {
    std::sort(occupied.begin(), occupied.end());
    uint64_t cursor = 0;
    auto fits = [&](uint64_t end) {
        uint64_t aligned = 0;
        if (!checked_add_u64(cursor, alignment - 1, aligned)) return false;
        aligned &= ~(alignment - 1);
        if (aligned > end || end - aligned < bytes) return false;
        offset = aligned;
        return true;
    };
    for (const auto & interval : occupied) {
        if (interval.first > interval.second || interval.second > capacity) {
            throw std::logic_error("expert cache arena metadata is outside capacity");
        }
        if (fits(interval.first)) return true;
        cursor = std::max(cursor, interval.second);
    }
    return fits(capacity);
}

} // namespace

ram_cache::pending_load::~pending_load() {
    if (state == phase::ready) result->pins.fetch_sub(1, std::memory_order_acq_rel);
}

bool ram_cache::lru_less::operator()(const std::shared_ptr<entry> & a, const std::shared_ptr<entry> & b) const noexcept {
    if (a->frequency != b->frequency) return a->frequency < b->frequency;
    if (a->last_used != b->last_used) return a->last_used < b->last_used;
    if (a->insertion_order != b->insertion_order) return a->insertion_order < b->insertion_order;
    return a->desc->key() < b->desc->key();
}

void ram_cache::observe(const expert_key & key) {
    if (!config_.frequency_retention) return;
    // Age cumulative observations so a previous request cannot permanently
    // monopolize the arena. Extract/reinsert nodes without allocating new tree
    // nodes; changing a key while it remains in the ordered set is invalid.
    if ((++frequency_clock_ & 16383) == 0) {
        std::vector<lru_index::node_type> nodes;
        nodes.reserve(by_recency_.size());
        while (!by_recency_.empty()) nodes.push_back(by_recency_.extract(by_recency_.begin()));
        for (auto & item : observations_) item.second >>= 1;
        for (auto & node : nodes) {
            node.value()->frequency = observations_.at(node.value()->desc->key());
            by_recency_.insert(std::move(node));
        }
    }
    auto & count = observations_[key];
    count = std::min<uint32_t>(count + 1, 64);
}

void ram_cache::touch(const std::shared_ptr<entry> & value, bool inserted) {
    auto node = by_recency_.extract(value);
    if (node.empty()) throw std::logic_error("expert cache LRU index is missing an entry");
    value->last_used = ++clock_;
    if (config_.frequency_retention) value->frequency = observations_.at(value->desc->key());
    if (inserted) value->insertion_order = ++insertion_clock_;
    const auto result = by_recency_.insert(std::move(node));
    if (!result.inserted) throw std::logic_error("expert cache LRU index duplicated an entry");
}

bool ram_cache::allocation_plan(
        uint64_t bytes,
        uint64_t & arena_offset,
        std::vector<std::shared_ptr<entry>> & victims,
        const pending_load * own_reservation) const {
    using interval = std::pair<uint64_t, uint64_t>;
    std::vector<interval> occupied;
    occupied.reserve(entries_.size() + pending_.size());
    for (const auto & item : by_offset_) {
        occupied.emplace_back(item.first, item.first + item.second->desc->bytes());
    }
    const size_t resident_count = occupied.size();
    std::vector<interval> reservations;
    reservations.reserve(pending_.size());
    for (const auto & item : pending_) {
        const auto & load = item.second;
        if (load.get() == own_reservation) continue;
        reservations.emplace_back(load->arena_offset,
                load->arena_offset + load->result->desc->bytes());
        occupied.push_back(reservations.back());
    }
    if (occupied.size() > resident_count) {
        const auto middle = occupied.begin() + resident_count;
        std::sort(middle, occupied.end());
        std::inplace_merge(occupied.begin(), middle, occupied.end());
    }
    // A free admission needs no victim ordering or gap-tree allocations.
    // This is also the common path while initially populating the arena.
    victims.clear();
    uint64_t free_cursor = 0;
    auto free_fits = [&](uint64_t end) {
        uint64_t aligned;
        if (!checked_add_u64(free_cursor, 63, aligned)) return false;
        aligned &= ~uint64_t(63);
        if (aligned > end || end - aligned < bytes) return false;
        arena_offset = aligned;
        return true;
    };
    for (const auto & region : occupied) {
        if (region.first > region.second || region.second > config_.capacity_bytes) {
            throw std::logic_error("expert cache arena metadata is outside capacity");
        }
        if (free_fits(region.first)) return true;
        free_cursor = std::max(free_cursor, region.second);
    }
    if (free_fits(config_.capacity_bytes)) return true;

    std::sort(reservations.begin(), reservations.end());
    std::vector<interval> blocked;
    for (const auto & region : reservations) {
        if (!blocked.empty() && region.first <= blocked.back().second) {
            blocked.back().second = std::max(blocked.back().second, region.second);
        } else {
            blocked.push_back(region);
        }
    }

    // Build free spans once. During LRU-prefix exploration, removing an entry
    // only extends these spans; keep their first-fit eligibility in a set.
    std::map<uint64_t, uint64_t> gaps;
    std::set<uint64_t> fitting;
    auto aligned_fit = [&](uint64_t start, uint64_t end, uint64_t & aligned) {
        if (!checked_add_u64(start, 63, aligned)) return false;
        aligned &= ~uint64_t(63);
        return aligned <= end && end - aligned >= bytes;
    };
    auto insert_gap = [&](uint64_t start, uint64_t end) {
        if (start >= end) return;
        auto next = gaps.lower_bound(start);
        if (next != gaps.begin()) {
            auto previous = std::prev(next);
            if (previous->second >= start) {
                start = previous->first;
                end = std::max(end, previous->second);
                fitting.erase(previous->first);
                gaps.erase(previous);
            }
        }
        while (next != gaps.end() && next->first <= end) {
            end = std::max(end, next->second);
            fitting.erase(next->first);
            next = gaps.erase(next);
        }
        gaps.emplace(start, end);
        uint64_t aligned;
        if (aligned_fit(start, end, aligned)) fitting.insert(start);
    };
    auto first_fit = [&]() {
        if (fitting.empty()) return false;
        const auto gap = gaps.find(*fitting.begin());
        return aligned_fit(gap->first, gap->second, arena_offset);
    };
    uint64_t cursor = 0;
    for (const auto & region : occupied) {
        if (region.first > region.second || region.second > config_.capacity_bytes) {
            throw std::logic_error("expert cache arena metadata is outside capacity");
        }
        if (region.first > cursor) insert_gap(cursor, region.first);
        cursor = std::max(cursor, region.second);
    }
    insert_gap(cursor, config_.capacity_bytes);
    victims.clear();
    if (first_fit()) return true;

    // Use the shortest prefix of the configured retention order that permits
    // the same lowest-address aligned placement. In-flight leases stay pinned.
    for (const auto & candidate : by_recency_) {
        if (!candidate->ready || candidate->pins.load(std::memory_order_acquire) != 0 ||
                (candidate->reserved_by && candidate->reserved_by != own_reservation)) continue;
        victims.push_back(candidate);
        uint64_t start = candidate->arena_offset;
        const uint64_t end = start + candidate->desc->bytes();
        // A pending fill may reserve bytes belonging to a still-readable
        // victim. Removing the victim must never free the other fill's span.
        for (const auto & region : blocked) {
            if (region.second <= start) continue;
            if (region.first >= end) break;
            if (start < region.first) insert_gap(start, std::min(end, region.first));
            start = std::max(start, region.second);
            if (start >= end) break;
        }
        if (start < end) insert_gap(start, end);
        if (first_fit()) return true;
    }
    victims.clear();
    return false;
}

bool ram_cache::staging_plan(uint64_t bytes, uint64_t & offset) const {
    if (pending_.empty()) {
        offset = 0;
        return bytes <= config_.staging_bytes;
    }
    std::vector<std::pair<uint64_t, uint64_t>> occupied;
    occupied.reserve(pending_.size());
    for (const auto & item : pending_) {
        const auto & load = item.second;
        occupied.push_back({load->staging_offset, load->staging_offset + load->result->desc->bytes()});
    }
    return arena_first_fit(std::move(occupied), config_.staging_bytes, bytes, 1, offset);
}

void ram_cache::release_reservation(const std::shared_ptr<pending_load> & load) {
    for (const auto & victim : load->victims) {
        if (victim->reserved_by == load.get()) victim->reserved_by = nullptr;
    }
    stats_.reserved_bytes -= load->result->desc->bytes();
    stats_.staging_used_bytes -= load->result->desc->bytes();
    --stats_.active_loads;
    pending_.erase(load->result->desc->key());
    changed_.notify_all();
}

ram_cache::lease ram_cache::acquire(std::shared_ptr<const descriptor> value) {
    if (!value) throw std::invalid_argument("cannot acquire a null expert descriptor");
    if (value->bytes() > config_.capacity_bytes || value->bytes() > config_.staging_bytes ||
            value->bytes() > std::numeric_limits<size_t>::max()) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++stats_.rejected_admissions;
        throw capacity_error("expert component exceeds RAM cache or staging capacity: " + key_string(value->key()));
    }

    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (!lock.try_lock()) {
        const auto lock_start = std::chrono::steady_clock::now();
        lock.lock();
        stats_.lock_wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - lock_start).count());
    }
    const auto canonical = canonical_descriptors_.find(value->key());
    if (canonical == canonical_descriptors_.end()) {
        canonical_descriptors_.emplace(value->key(), value);
    } else if (canonical->second != value) {
        throw std::invalid_argument("expert key was reused with a different immutable descriptor");
    }
    observe(value->key());

    std::shared_ptr<pending_load> load;
    for (;;) {
        const auto cached = entries_.find(value->key());
        if (cached != entries_.end() && cached->second->ready) {
            touch(cached->second);
            ++stats_.hits;
            return lease(cached->second);
        }
        const auto pending = pending_.find(value->key());
        if (pending != pending_.end()) {
            load = pending->second;
            ++stats_.coalesced_loads;
            changed_.wait(lock, [&] {
                return load->state == pending_load::phase::ready || load->state == pending_load::phase::failed;
            });
            if (load->error) std::rethrow_exception(load->error);
            touch(load->result);
            ++stats_.hits;
            return lease(load->result);
        }

        load = std::make_shared<pending_load>();
        load->result = std::make_shared<entry>();
        load->result->desc = value;
        const bool admissible = allocation_plan(value->bytes(), load->arena_offset, load->victims);
        if (!admissible && pending_.empty()) {
            ++stats_.pinned_capacity_failures;
            throw capacity_error("all eviction candidates are protected by in-flight expert leases");
        }
        if (admissible && staging_plan(value->bytes(), load->staging_offset)) break;
        ++stats_.staging_waits;
        const auto wait_start = std::chrono::steady_clock::now();
        changed_.wait(lock);
        stats_.staging_wait_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - wait_start).count());
    }

    // Validate before admission and I/O. The source table and descriptors are
    // immutable, so workers can subsequently access them without the mutex.
    for (const auto & extent : value->extents()) {
        const auto source_it = sources_.find(extent.source);
        if (source_it == sources_.end() || source_it->second->identity() != extent.identity) {
            throw std::invalid_argument("expert descriptor source identity does not match the registered source");
        }
    }
    pending_.emplace(value->key(), load);
    for (const auto & victim : load->victims) victim->reserved_by = load.get();
    ++stats_.misses;
    ++stats_.active_loads;
    stats_.peak_active_loads = std::max(stats_.peak_active_loads, stats_.active_loads);
    stats_.reserved_bytes += value->bytes();
    stats_.staging_used_bytes += value->bytes();
    stats_.peak_staging_used_bytes = std::max(stats_.peak_staging_used_bytes, stats_.staging_used_bytes);

    try {
        lock.unlock();
        const auto read_start = std::chrono::steady_clock::now();
        size_t destination_offset = size_t(load->staging_offset);
        std::array<uint64_t, size_t(read_backend::count)> bytes_read{};
        std::exception_ptr read_error;
        try {
            for (const auto & extent : value->extents()) {
                const auto & input = sources_.at(extent.source);
                input->read(extent.offset, staging_.data() + destination_offset, size_t(extent.length));
                destination_offset += size_t(extent.length);
                bytes_read[size_t(input->backend())] += extent.length;
            }
        } catch (...) {
            read_error = std::current_exception();
        }
        const auto read_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - read_start).count());
        lock.lock();
        stats_.storage_read_ns += read_ns;
        for (size_t i = 0; i < bytes_read.size(); ++i) stats_.bytes_read[i] += bytes_read[i];
        if (read_error) std::rethrow_exception(read_error);

        // Reserved victims remain readable during I/O. If a caller acquired a
        // lease meanwhile, find another region or fail without changing old
        // payloads. Never wait for a lease held by this request's caller.
        if (std::any_of(load->victims.begin(), load->victims.end(), [](const auto & victim) {
                return victim->pins.load(std::memory_order_acquire) != 0;
            })) {
            std::vector<std::shared_ptr<entry>> replacement;
            uint64_t offset = 0;
            if (!allocation_plan(value->bytes(), offset, replacement, load.get())) {
                ++stats_.pinned_capacity_failures;
                throw capacity_error("expert RAM cache reservation became leased during I/O");
            }
            for (const auto & victim : load->victims) victim->reserved_by = nullptr;
            load->victims.swap(replacement);
            load->arena_offset = offset;
            for (const auto & victim : load->victims) victim->reserved_by = load.get();
        }

        auto & loaded = load->result;
        loaded->arena_offset = load->arena_offset;
        loaded->data = resident_arena_.get() + size_t(load->arena_offset);
        loaded->ready = false;
        // Prepare all index nodes before retiring victims. Detached nodes can
        // be installed without allocating after the old spans are released.
        offset_index prepared_offset;
        prepared_offset.emplace(loaded->arena_offset, loaded.get());
        auto offset_node = prepared_offset.extract(prepared_offset.begin());
        lru_index prepared_lru;
        prepared_lru.insert(loaded);
        auto lru_node = prepared_lru.extract(prepared_lru.begin());
        const auto inserted = entries_.emplace(value->key(), loaded);
        if (!inserted.second) throw std::logic_error("expert RAM cache duplicate publication");
        for (const auto & victim : load->victims) {
            stats_.resident_bytes -= victim->desc->bytes();
            by_offset_.erase(victim->arena_offset);
            by_recency_.erase(victim);
            entries_.erase(victim->desc->key());
            ++stats_.evictions;
        }
        if (!by_offset_.insert(std::move(offset_node)).inserted ||
                !by_recency_.insert(std::move(lru_node)).inserted) {
            throw std::logic_error("expert cache publication duplicated an index entry");
        }
        load->state = pending_load::phase::copying;
        lock.unlock();
        std::memcpy(const_cast<uint8_t *>(loaded->data),
                staging_.data() + size_t(load->staging_offset), size_t(value->bytes()));
        lock.lock();
        touch(loaded, true);
        loaded->ready = true;
        loaded->pins.fetch_add(1, std::memory_order_acq_rel); // completion ownership
        load->state = pending_load::phase::ready;
        stats_.resident_bytes += value->bytes();
        stats_.peak_resident_bytes = std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
        release_reservation(load);
        return lease(loaded);
    } catch (...) {
        const auto error = std::current_exception();
        if (!lock.owns_lock()) lock.lock();
        load->error = error;
        load->state = pending_load::phase::failed;
        ++stats_.failed_loads;
        release_reservation(load);
        std::rethrow_exception(error);
    }
}

ram_cache::lease ram_cache::try_acquire(std::shared_ptr<const descriptor> value) {
    if (!value) throw std::invalid_argument("cannot acquire a null expert descriptor");
    std::lock_guard<std::mutex> lock(mutex_);
    const auto canonical = canonical_descriptors_.find(value->key());
    if (canonical != canonical_descriptors_.end() && canonical->second != value) {
        throw std::invalid_argument("expert key was reused with a different immutable descriptor");
    }
    const auto found = entries_.find(value->key());
    if (found == entries_.end() || !found->second->ready) return {};
    observe(value->key());
    touch(found->second);
    ++stats_.hits;
    return lease(found->second);
}

bool ram_cache::erase(const expert_key & key) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto found = entries_.find(key);
    if (found == entries_.end() || !found->second->ready || found->second->reserved_by ||
            found->second->pins.load(std::memory_order_acquire) != 0) return false;
    stats_.resident_bytes -= found->second->desc->bytes();
    by_offset_.erase(found->second->arena_offset);
    by_recency_.erase(found->second);
    entries_.erase(found);
    return true;
}

uint64_t ram_cache::advise_will_need(std::shared_ptr<const descriptor> value) const {
    if (!value) throw std::invalid_argument("cannot advise a null expert descriptor");
    {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto canonical = canonical_descriptors_.find(value->key());
        if (canonical != canonical_descriptors_.end() && canonical->second != value)
            throw std::invalid_argument("expert advice reused a key with a different descriptor");
        const auto cached = entries_.find(value->key());
        if ((cached != entries_.end() && cached->second->ready) || pending_.count(value->key())) return 0;
    }
    // Validate every extent before issuing even the first hint. Sources and
    // descriptors are immutable and outlive the advice worker.
    for (const auto & extent : value->extents()) {
        const auto found = sources_.find(extent.source);
        if (found == sources_.end() || found->second->identity() != extent.identity ||
                extent.offset > found->second->size() || extent.length > found->second->size() - extent.offset)
            throw std::invalid_argument("expert advice source identity or range mismatch");
    }
    uint64_t bytes = 0;
    for (const auto & extent : value->extents()) {
        if (sources_.at(extent.source)->advise_will_need(extent.offset, extent.length)) bytes += extent.length;
    }
    return bytes;
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
    if (!pending_.empty()) throw capacity_error("cannot clear expert RAM cache while fills are in flight");
    for (const auto & item : entries_) {
        if (item.second->pins.load(std::memory_order_acquire) != 0) {
            throw capacity_error("cannot clear expert RAM cache while leases are in flight");
        }
    }
    by_offset_.clear();
    by_recency_.clear();
    entries_.clear();
    observations_.clear();
    frequency_clock_ = 0;
    stats_.resident_bytes = 0;
}

load_queue::load_queue(ram_cache & cache, size_t workers, size_t requests, uint64_t byte_limit)
    : cache_(cache) {
    const auto limits = cache.stats();
    if (workers == 0 || workers > 8 || requests < workers || requests > 64 ||
            byte_limit == 0 || byte_limit > limits.staging_capacity_bytes) {
        throw std::invalid_argument("expert load queue exceeds worker, request, or staging bounds");
    }
    stats_.byte_limit = byte_limit;
    slots_.resize(requests);
    workers_.reserve(workers);
    try {
        for (size_t i = 0; i < workers; ++i) workers_.emplace_back(&load_queue::worker, this);
    } catch (...) {
        stop();
        throw;
    }
}

load_queue::~load_queue() {
    stop();
}

void load_queue::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        work_changed_.notify_all();
    }
    // Blocking source reads cannot be cancelled by dropping their buffers.
    // Join them before slots (and therefore any descriptor/lease) are freed.
    for (auto & thread : workers_) if (thread.joinable()) thread.join();
}

bool load_queue::try_submit(uint64_t id, std::shared_ptr<const descriptor> value) {
    if (!value) throw std::invalid_argument("cannot submit a null expert descriptor");
    std::lock_guard<std::mutex> lock(mutex_);
    slot * available = nullptr;
    for (auto & item : slots_) {
        if (item.state == phase::free) {
            if (!available) available = &item;
        } else if (item.id == id) {
            throw std::invalid_argument("expert load queue request ID is already outstanding");
        }
    }
    if (stopping_ || !available || value->bytes() > stats_.byte_limit - stats_.outstanding_bytes) return false;
    available->id = id;
    available->desc = std::move(value);
    ++stats_.submitted;
    ++stats_.outstanding;
    stats_.outstanding_bytes += available->desc->bytes();
    stats_.peak_outstanding_bytes = std::max(stats_.peak_outstanding_bytes, stats_.outstanding_bytes);
    // Ready RAM hits need neither worker dispatch nor a storage wakeup.
    try {
        available->payload = cache_.try_acquire(available->desc);
    } catch (...) {
        available->error = std::current_exception();
    }
    if (available->payload || available->error) {
        stats_.ready_hits += bool(available->payload);
        available->state = phase::ready;
        completion_changed_.notify_all();
    } else {
        available->state = phase::queued;
        work_changed_.notify_one();
    }
    return true;
}

bool load_queue::next(completion & result, bool wait) {
    std::unique_lock<std::mutex> lock(mutex_);
    auto ready = [&]() {
        return std::find_if(slots_.begin(), slots_.end(), [](const slot & item) {
            return item.state == phase::ready;
        });
    };
    if (wait) completion_changed_.wait(lock, [&] { return stats_.outstanding == 0 || ready() != slots_.end(); });
    const auto found = ready();
    if (found == slots_.end()) return false;
    result.id = found->id;
    result.payload = std::move(found->payload);
    result.error = std::move(found->error);
    --stats_.outstanding;
    stats_.outstanding_bytes -= found->desc->bytes();
    found->desc.reset();
    found->state = phase::free;
    completion_changed_.notify_all();
    return true;
}

load_queue::queue_stats load_queue::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void load_queue::worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        auto queued = [&]() {
            return std::find_if(slots_.begin(), slots_.end(), [](const slot & item) {
                return item.state == phase::queued;
            });
        };
        work_changed_.wait(lock, [&] { return stopping_ || queued() != slots_.end(); });
        if (stopping_) return;
        auto & item = *queued();
        item.state = phase::running;
        lock.unlock();
        ram_cache::lease payload;
        std::exception_ptr error;
        try {
            payload = cache_.acquire(item.desc);
        } catch (...) {
            error = std::current_exception();
        }
        lock.lock();
        item.payload = std::move(payload);
        item.error = std::move(error);
        item.state = phase::ready;
        completion_changed_.notify_all();
    }
}

read_ahead_queue::read_ahead_queue(ram_cache & cache, load_queue * demand, uint64_t pass_byte_limit)
    : cache_(cache), demand_(demand), pass_byte_limit_(pass_byte_limit) {
    if (pass_byte_limit == 0 || pass_byte_limit > (1ULL << 30))
        throw std::invalid_argument("expert read-ahead pass budget must be in (0, 1 GiB]");
    worker_ = std::thread(&read_ahead_queue::worker, this);
}

read_ahead_queue::~read_ahead_queue() {
    stop();
}

void read_ahead_queue::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stopping_ = true;
        for (const auto & value : pending_) {
            stats_.outstanding_bytes -= value->bytes();
            ++stats_.expired;
        }
        pending_.clear();
        changed_.notify_all();
    }
    if (worker_.joinable()) worker_.join();
}

void read_ahead_queue::advance(uintptr_t owner, uint64_t pass, uint32_t layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    const bool new_pass = owner != owner_ || pass != pass_;
    for (auto item = pending_.begin(); item != pending_.end();) {
        if (new_pass || (*item)->key().layer <= layer) {
            stats_.outstanding_bytes -= (*item)->bytes();
            item = pending_.erase(item);
            ++stats_.expired;
        } else ++item;
    }
    if (new_pass) {
        seen_.clear();
        admitted_bytes_ = 0;
    }
    owner_ = owner; pass_ = pass; layer_ = layer;
    changed_.notify_all();
}

bool read_ahead_queue::submit(std::shared_ptr<const descriptor> value) {
    if (!value) throw std::invalid_argument("cannot queue null expert advice");
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopping_ || value->key().layer <= layer_ || seen_.count(value->key()) ||
            pending_.size() >= 64 || seen_.size() >= 4096 || value->bytes() > pass_byte_limit_ - admitted_bytes_) {
        ++stats_.rejected;
        return false;
    }
    auto seen = seen_.insert(value->key());
    try { pending_.push_back(value); }
    catch (...) { seen_.erase(seen.first); throw; }
    admitted_bytes_ += value->bytes();
    stats_.outstanding_bytes += value->bytes();
    stats_.peak_outstanding_bytes = std::max(stats_.peak_outstanding_bytes, stats_.outstanding_bytes);
    ++stats_.submitted;
    changed_.notify_one();
    return true;
}

read_ahead_queue::queue_stats read_ahead_queue::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

void read_ahead_queue::worker() {
    std::unique_lock<std::mutex> lock(mutex_);
    for (;;) {
        changed_.wait(lock, [&] { return stopping_ || !pending_.empty(); });
        if (stopping_) return;
        if (demand_ && demand_->stats().outstanding) {
            changed_.wait_for(lock, std::chrono::milliseconds(1));
            continue;
        }
        auto value = pending_.front();
        pending_.pop_front();
        lock.unlock();
        uint64_t bytes = 0;
        bool failed = false;
        try { bytes = cache_.advise_will_need(value); } catch (...) { failed = true; }
        lock.lock();
        stats_.advised_bytes += bytes;
        stats_.failed += failed;
        stats_.skipped += !bytes && !failed;
        stats_.outstanding_bytes -= value->bytes();
    }
}

} // namespace llama_expert_cache
