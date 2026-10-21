#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

static void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        const size_t bytes = 64 * 1024 * 1024 + 13;
        auto * buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes);
        require(buffer, "owned allocation");
        auto * data = static_cast<unsigned char *>(ggml_backend_buffer_get_base(buffer));
        require(reinterpret_cast<uintptr_t>(data) % ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type()) == 0,
                "backend alignment");
        std::memset(data, 0x5a, bytes);
        ggml_backend_buffer_clear(buffer, 0);
#ifdef __linux__
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        if (reinterpret_cast<uintptr_t>(data) % page == 0) {
            std::vector<unsigned char> residency((bytes + page - 1) / page);
            if (mincore(data, bytes, residency.data()) == 0) {
                size_t resident = 0;
                for (const auto entry : residency) resident += entry & 1;
                require(resident == 0, "zero clear releases owned pages");
            }
        }
#endif
        for (size_t i = 0; i < bytes; i += 4096) require(data[i] == 0, "zero clear after direct writes");
        require(data[bytes - 1] == 0, "zero clear tail");
        ggml_backend_buffer_clear(buffer, 0x37);
        for (size_t i = 0; i < bytes; i += 4096) require(data[i] == 0x37, "nonzero clear");
        require(data[bytes - 1] == 0x37, "nonzero clear tail");
        ggml_backend_buffer_clear(buffer, 0);
        require(data[0] == 0 && data[bytes - 1] == 0, "repeated zero clear");
        ggml_backend_buffer_free(buffer);
#ifdef __linux__
        // A file-backed external pointer must be overwritten, not discarded:
        // MADV_DONTNEED would restore the file's original nonzero contents.
        auto * file = std::tmpfile();
        require(file, "external backing file");
        const int fd = fileno(file);
        require(ftruncate(fd, bytes) == 0, "external backing size");
        const unsigned char marker = 0x5a;
        require(pwrite(fd, &marker, 1, 0) == 1 && pwrite(fd, &marker, 1, bytes - 1) == 1, "external backing pattern");
        void * mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
        require(mapping != MAP_FAILED, "external mapping");
        auto * external = ggml_backend_cpu_buffer_from_ptr(mapping, bytes);
        require(external, "external buffer");
        ggml_backend_buffer_clear(external, 0);
        auto * external_data = static_cast<unsigned char *>(mapping);
        require(external_data[0] == 0 && external_data[bytes - 1] == 0, "external zero clear");
        unsigned char original = 0;
        require(pread(fd, &original, 1, 0) == 1 && original == marker, "external backing preserved");
        ggml_backend_buffer_free(external);
        require(munmap(mapping, bytes) == 0, "external ownership preserved");
        std::fclose(file);
#endif
        std::puts("CPU buffer clear: owned pages, direct writes, nonzero clears, tail and external ownership passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
