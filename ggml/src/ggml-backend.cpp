#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"
#include "ggml-rpc.h"
#include "ggml-moe-prefetch.h"
#include "ggml-ese.h"

#include <cassert>
#include <climits>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>
#include <set>
#include <array>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <barrier>
#include <map>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#ifdef GGML_USE_OPENMP
#include <omp.h>
#endif

#define IK_PRINT_TIMING 0

#define MAX(a, b) ((a) > (b) ? (a) : (b))

// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name(buft);
}

GGML_CALL ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

GGML_CALL size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        //assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

// backend buffer

GGML_CALL ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t      buft,
        struct ggml_backend_buffer_i           iface,
               ggml_backend_buffer_context_t   context,
               size_t                          size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name(buffer);
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
    //free(buffer);
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

GGML_CALL void ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    }
}

size_t ggml_backend_buffer_get_alignment (ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    buffer->iface.clear(buffer, value);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return buffer && ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    return backend->iface.get_default_buffer_type(backend);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    if (offset + size > ggml_nbytes(tensor)) fprintf(stderr, "%s(%s): offset = %zu, size = %zu, nbytes = %zu\n", __func__, tensor->name, offset, size, ggml_nbytes(tensor));
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_expert_cache_upload_async(
        ggml_backend_t backend, struct ggml_tensor * tensor,
        const void * data, size_t offset, size_t size) {
    ggml_backend_tensor_set_async(backend, tensor, data, offset, size);
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

GGML_CALL void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (!size) {
        return;
    }


#if IK_PRINT_TIMING
    int64_t tim1 = ggml_time_us();
#endif
    buf->iface.set_tensor(buf, tensor, data, offset, size);
#if IK_PRINT_TIMING
    int64_t tim2 = ggml_time_us();
    //printf("%s(%s) %zu %d us\n", __func__, tensor->name, size, (int)(tim2-tim1));
    printf("%s(%s): %d us\n", __func__, tensor->name, (int)(tim2-tim1));
#endif

}

GGML_CALL void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (!size) {
        return;
    }

#if IK_PRINT_TIMING
    int64_t tim1 = ggml_time_us();
#endif
    buf->iface.get_tensor(buf, tensor, data, offset, size);
#if IK_PRINT_TIMING
    int64_t tim2 = ggml_time_us();
    //printf("%s(%s) %zu %d us\n", __func__, tensor->name, size, (int)(tim2-tim1));
    printf("%s(%s): %d us\n", __func__, tensor->name, (int)(tim2-tim1));
#endif
}

static void ggml_backend_tensor_memset(struct ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    return backend->iface.supports_op(backend, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return backend->iface.supports_buft(backend, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    if (backend->iface.offload_op != NULL) {
        return backend->iface.offload_op(backend, op);
    }
    return false;
}

// backend copy

static bool ggml_are_same_layout(const struct ggml_tensor * a, const struct ggml_tensor * b) {
    if (a->type != b->type) {
        return false;
    }
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
        if (a->nb[i] != b->nb[i]) {
            return false;
        }
    }
    return true;
}

void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (!ggml_are_same_layout(src, dst)) {
        fprintf(stderr, "ggml backend copy layout mismatch: src=%s %s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu] dst=%s %s ne=[%lld,%lld,%lld,%lld] nb=[%zu,%zu,%zu,%zu]\n",
                ggml_get_name(src), ggml_type_name(src->type),
                (long long) src->ne[0], (long long) src->ne[1], (long long) src->ne[2], (long long) src->ne[3],
                src->nb[0], src->nb[1], src->nb[2], src->nb[3],
                ggml_get_name(dst), ggml_type_name(dst->type),
                (long long) dst->ne[0], (long long) dst->ne[1], (long long) dst->ne[2], (long long) dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
    }
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        fprintf(stderr, "%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_t backend) {
    if (backend->iface.event_new == NULL) {
        return NULL;
    }
    return backend->iface.event_new(backend);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->backend->iface.event_free(event);
}

void ggml_backend_event_record(ggml_backend_event_t event) {
    GGML_ASSERT(event->backend->iface.event_record != NULL);

    event->backend->iface.event_record(event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event->backend->iface.event_synchronize != NULL);

    event->backend->iface.event_synchronize(event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

// backend registry

#define GGML_REG_MAX_BACKENDS 64

struct ggml_backend_reg {
    char name[128];
    ggml_backend_init_fn init_fn;
    ggml_backend_buffer_type_t default_buffer_type;
    void * user_data;
};

static struct ggml_backend_reg ggml_backend_registry[GGML_REG_MAX_BACKENDS];
static size_t ggml_backend_registry_count = 0;

GGML_CALL static ggml_backend_t ggml_backend_reg_cpu_init(const char * params, void * user_data);

#ifdef GGML_USE_CUDA
extern "C" GGML_CALL void ggml_backend_cuda_reg_devices(void);
extern "C" GGML_CALL bool ggml_backend_is_cuda(ggml_backend_t backend);
extern "C" GGML_CALL int ggml_backend_cuda_get_device(ggml_backend_t backend);
extern "C" GGML_CALL ggml_backend_t ggml_backend_cuda_init(int device, const void * params, const void * model);
extern "C" GGML_CALL void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);
#endif
#ifdef GGML_USE_SYCL
extern "C" void ggml_backend_sycl_reg_devices(void);
#endif
#ifdef GGML_USE_METAL
extern "C" GGML_CALL ggml_backend_t ggml_backend_reg_metal_init(const char * params, void * user_data);
extern "C" GGML_CALL ggml_backend_buffer_type_t ggml_backend_metal_buffer_type(void);
#endif
#ifdef GGML_USE_VULKAN
extern "C" GGML_CALL int ggml_backend_vk_reg_devices(void);
#endif
#ifdef GGML_USE_CANN
extern "C" GGML_CALL int ggml_backend_cann_reg_devices(void);
#endif
#ifdef GGML_USE_RPC
extern "C" GGML_CALL void ggml_backend_rpc_reg_devices(void);
#endif

GGML_CALL static void ggml_backend_registry_init(void) {
    static bool initialized = false;

    if (initialized) {
        return;
    }

    initialized = true;

    ggml_backend_register("CPU", ggml_backend_reg_cpu_init, ggml_backend_cpu_buffer_type(), NULL);

    // add forward decls here to avoid including the backend headers
#ifdef GGML_USE_CUDA
    ggml_backend_cuda_reg_devices();
#endif

#ifdef GGML_USE_SYCL
    ggml_backend_sycl_reg_devices();
#endif

#ifdef GGML_USE_METAL
    ggml_backend_register("Metal", ggml_backend_reg_metal_init, ggml_backend_metal_buffer_type(), NULL);
#endif

#ifdef GGML_USE_VULKAN
    ggml_backend_vk_reg_devices();
#endif

#ifdef GGML_USE_CANN
    ggml_backend_cann_reg_devices();
#endif
#ifdef GGML_USE_RPC
    ggml_backend_rpc_reg_devices();
#endif
}

GGML_CALL void ggml_backend_register(const char * name, ggml_backend_init_fn init_fn, ggml_backend_buffer_type_t default_buffer_type, void * user_data) {
    GGML_ASSERT(ggml_backend_registry_count < GGML_REG_MAX_BACKENDS);

    size_t id = ggml_backend_registry_count;

    ggml_backend_registry[id] = ggml_backend_reg {
        /* .name                = */ {0},
        /* .fn                  = */ init_fn,
        /* .default_buffer_type = */ default_buffer_type,
        /* .user_data           = */ user_data
    };

    snprintf(ggml_backend_registry[id].name, sizeof(ggml_backend_registry[id].name), "%s", name);

#ifndef NDEBUG
    fprintf(stderr, "%s: registered backend %s\n", __func__, name);
#endif

    ggml_backend_registry_count++;
}

// Backend (reg) enumeration
static bool striequals(const char* a, const char* b) {
    for (; *a && *b; a++, b++) {
        if (std::tolower(*a) != std::tolower(*b)) {
            return false;
        }
    }
    return *a == *b;
}

size_t ggml_backend_reg_get_count(void) {
    ggml_backend_registry_init();

    return ggml_backend_registry_count;
}

size_t ggml_backend_reg_find_by_name(const char * name) {
    ggml_backend_registry_init();

    for (size_t i = 0; i < ggml_backend_registry_count; i++) {
        // TODO: case insensitive in a portable way
        if (striequals(ggml_backend_registry[i].name, name)) {
            return i;
        }
    }

    // not found
    return SIZE_MAX;
}

// init from backend:params string
ggml_backend_t ggml_backend_reg_init_backend_from_str(const char * backend_str) {
    ggml_backend_registry_init();

    const char * params = strchr(backend_str, ':');
    char backend_name[128];
    if (params == NULL) {
        snprintf(backend_name, sizeof(backend_name), "%s", backend_str);
        params = "";
    } else {
        snprintf(backend_name, sizeof(backend_name), "%.*s", (int)(params - backend_str), backend_str);
        params++;
    }

    size_t backend_i = ggml_backend_reg_find_by_name(backend_name);

    if (backend_i == SIZE_MAX) {
        fprintf(stderr, "%s: backend %s not found\n", __func__, backend_name);
        return NULL;
    }

    return ggml_backend_reg_init_backend(backend_i, params);
}

const char * ggml_backend_reg_get_name(size_t i) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_registry[i].name;
}

ggml_backend_t ggml_backend_reg_init_backend(size_t i, const char * params) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_registry[i].init_fn(params, ggml_backend_registry[i].user_data);
}

ggml_backend_buffer_type_t ggml_backend_reg_get_default_buffer_type(size_t i) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_registry[i].default_buffer_type;
}

ggml_backend_buffer_t ggml_backend_reg_alloc_buffer(size_t i, size_t size) {
    ggml_backend_registry_init();

    GGML_ASSERT(i < ggml_backend_registry_count);
    return ggml_backend_buft_alloc_buffer(ggml_backend_registry[i].default_buffer_type, size);
}

// backend CPU

static const size_t TENSOR_ALIGNMENT = 32; // required for mmap as gguf only guarantees 32-byte alignment

GGML_CALL static const char * ggml_backend_cpu_buffer_name(ggml_backend_buffer_t buffer) {
    return "CPU";

    GGML_UNUSED(buffer);
}

GGML_CALL static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

GGML_CALL static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    free(buffer->context);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, uint8_t value, size_t offset, size_t size) {
    memset((char*)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

GGML_CALL static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

GGML_CALL static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    memset(buffer->context, value, buffer->size);
}

static struct ggml_backend_buffer_i cpu_backend_buffer_i = {
    /* .get_name        = */ ggml_backend_cpu_buffer_name,
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// for buffers from ptr, free is not called
static struct ggml_backend_buffer_i cpu_backend_buffer_i_from_ptr = {
    /* .get_name        = */ ggml_backend_cpu_buffer_name,
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

GGML_CALL static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    size += TENSOR_ALIGNMENT;   // malloc may return an address that is not aligned
    void * data = malloc(size); // TODO: use GGML_ALIGNED_MALLOC (move to ggml-impl.h)
    if (data == NULL) {
        fprintf(stderr, "%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, cpu_backend_buffer_i, data, size);
}

GGML_CALL static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

GGML_CALL static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

#ifdef GGML_USE_CPU_HBM

// buffer type HBM

#include <hbwmalloc.h>

GGML_CALL static const char * ggml_backend_cpu_hbm_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_HBM";

    GGML_UNUSED(buft);
}

GGML_CALL static const char * ggml_backend_cpu_hbm_buffer_get_name(ggml_backend_buffer_t buf) {
    return "CPU_HBM";

    GGML_UNUSED(buf);
}

GGML_CALL static void ggml_backend_cpu_hbm_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    hbw_free(buffer->context);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_cpu_hbm_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    //void * ptr = hbw_malloc(size);
    void * ptr;
    int result = hbw_posix_memalign(&ptr, ggml_backend_cpu_buffer_type_get_alignment(buft), size);
    if (result != 0) {
        fprintf(stderr, "failed to allocate HBM buffer of size %zu\n", size);
        return NULL;
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.get_name = ggml_backend_cpu_hbm_buffer_get_name;
    buffer->iface.free_buffer = ggml_backend_cpu_hbm_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cpu_hbm_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type_hbm = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cpu_hbm_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_hbm_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .context  = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type_hbm;
}
#endif

struct ggml_backend_cpu_context {
    int n_threads;
    void * work_data;
    size_t work_size;

    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    bool moe_expert_prefetch;
    ggml_expert_acquire_callback expert_lease_acquire;
    ggml_expert_release_callback expert_lease_release;
    void * expert_lease_user_data;
    bool expert_lease_required;
};

GGML_CALL static const char * ggml_backend_cpu_name(ggml_backend_t backend) {
    return "CPU";

    GGML_UNUSED(backend);
}

GGML_CALL static void ggml_backend_cpu_free(ggml_backend_t backend) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;
    free(cpu_ctx->work_data);
    free(cpu_ctx);
    free(backend);
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_cpu_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_cpu_buffer_type();

    GGML_UNUSED(backend);
}

struct ggml_backend_plan_cpu {
    struct ggml_cplan cplan;
    struct ggml_cgraph cgraph;
};

GGML_CALL static ggml_backend_graph_plan_t ggml_backend_cpu_graph_plan_create(ggml_backend_t backend, const struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_backend_plan_cpu * cpu_plan = (ggml_backend_plan_cpu *)malloc(sizeof(struct ggml_backend_plan_cpu));

    cpu_plan->cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);
    cpu_plan->cgraph = *cgraph; // FIXME: deep copy

    if (cpu_plan->cplan.work_size > 0) {
        cpu_plan->cplan.work_data = (uint8_t *)malloc(cpu_plan->cplan.work_size);
        if (cpu_plan->cplan.work_data == NULL) {
            free(cpu_plan);
            return NULL;
        }
    }

    cpu_plan->cplan.abort_callback      = cpu_ctx->abort_callback;
    cpu_plan->cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cpu_plan->cplan.moe_expert_prefetch = cpu_ctx->moe_expert_prefetch;
    cpu_plan->cplan.expert_lease_acquire = cpu_ctx->expert_lease_acquire;
    cpu_plan->cplan.expert_lease_release = cpu_ctx->expert_lease_release;
    cpu_plan->cplan.expert_lease_user_data = cpu_ctx->expert_lease_user_data;
    cpu_plan->cplan.expert_lease_required = cpu_ctx->expert_lease_required;

    return cpu_plan;
}

GGML_CALL static void ggml_backend_cpu_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    free(cpu_plan->cplan.work_data);
    free(cpu_plan);

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_cpu_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    struct ggml_backend_plan_cpu * cpu_plan = (struct ggml_backend_plan_cpu *)plan;

    return ggml_graph_compute(&cpu_plan->cgraph, &cpu_plan->cplan);

    GGML_UNUSED(backend);
}

GGML_CALL static enum ggml_status ggml_backend_cpu_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    struct ggml_backend_cpu_context * cpu_ctx = (struct ggml_backend_cpu_context *)backend->context;

    struct ggml_cplan cplan = ggml_graph_plan(cgraph, cpu_ctx->n_threads);

    if (cpu_ctx->work_size < cplan.work_size) {
        free(cpu_ctx->work_data);
        cpu_ctx->work_data = malloc(cplan.work_size);
        if (cpu_ctx->work_data == NULL) {
            cpu_ctx->work_size = 0;
            return GGML_STATUS_ALLOC_FAILED;
        }
        cpu_ctx->work_size = cplan.work_size;
    }
    cplan.work_data = (uint8_t *)cpu_ctx->work_data;

    cplan.abort_callback      = cpu_ctx->abort_callback;
    cplan.abort_callback_data = cpu_ctx->abort_callback_data;
    cplan.moe_expert_prefetch = cpu_ctx->moe_expert_prefetch;
    cplan.expert_lease_acquire = cpu_ctx->expert_lease_acquire;
    cplan.expert_lease_release = cpu_ctx->expert_lease_release;
    cplan.expert_lease_user_data = cpu_ctx->expert_lease_user_data;
    cplan.expert_lease_required = cpu_ctx->expert_lease_required;

    return ggml_graph_compute(cgraph, &cplan);
}

GGML_CALL static bool ggml_backend_cpu_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_CPY:
            return
                op->type != GGML_TYPE_IQ2_XXS &&
                op->type != GGML_TYPE_IQ2_XS  &&
                op->type != GGML_TYPE_IQ1_S   &&
                op->type != GGML_TYPE_IQ1_M; // missing type_traits.from_float
        case GGML_OP_MUL_MAT:
            return true;
            //return op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == ggml_internal_get_type_traits(op->src[0]->type).vec_dot_type;
        case GGML_OP_INDEXER_TOPK:
#ifdef GGML_USE_IQK_MULMAT
            return true;
#else
            return false;
#endif
        case GGML_OP_LATENT_ATTN:
            // Scalar reference forward exists and is dispatched, so support is truthful.
            // Whether to ADOPT the op on a CPU-resident layer is performance policy, and
            // that lives in the openPangu builder gate, which requires a non-CPU backend.
            return true;
        default:
            return true;
    }

    GGML_UNUSED(backend);
}

GGML_CALL static bool ggml_backend_cpu_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    return ggml_backend_buft_is_host(buft);

    GGML_UNUSED(backend);
}

static struct ggml_backend_i cpu_backend_i = {
    /* .get_name                = */ ggml_backend_cpu_name,
    /* .free                    = */ ggml_backend_cpu_free,
    /* .get_default_buffer_type = */ ggml_backend_cpu_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ ggml_backend_cpu_graph_plan_create,
    /* .graph_plan_free         = */ ggml_backend_cpu_graph_plan_free,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ ggml_backend_cpu_graph_plan_compute,
    /* .graph_compute           = */ ggml_backend_cpu_graph_compute,
    /* .supports_op             = */ ggml_backend_cpu_supports_op,
    /* .supports_buft           = */ ggml_backend_cpu_supports_buft,
    /* .offload_op              = */ NULL,
    /* .event_new               = */ NULL,
    /* .event_free              = */ NULL,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .event_synchronize       = */ NULL,
};

static ggml_guid_t ggml_backend_cpu_guid(void) {
    static ggml_guid guid = { 0xaa, 0x67, 0xc7, 0x43, 0x96, 0xe6, 0xa3, 0x8a, 0xe3, 0xaf, 0xea, 0x92, 0x36, 0xbc, 0xfc, 0x89 };
    return &guid;
}

ggml_backend_t ggml_backend_cpu_init(void) {
    struct ggml_backend_cpu_context * ctx = (ggml_backend_cpu_context *)malloc(sizeof(struct ggml_backend_cpu_context));
    if (ctx == NULL) {
        return NULL;
    }

    ctx->n_threads           = GGML_DEFAULT_N_THREADS;
    ctx->work_data           = NULL;
    ctx->work_size           = 0;
    ctx->abort_callback      = NULL;
    ctx->abort_callback_data = NULL;
    ctx->moe_expert_prefetch = false;
    ctx->expert_lease_acquire = nullptr;
    ctx->expert_lease_release = nullptr;
    ctx->expert_lease_user_data = nullptr;
    ctx->expert_lease_required = false;

    ggml_backend_t cpu_backend = (ggml_backend_t)malloc(sizeof(struct ggml_backend));
    if (cpu_backend == NULL) {
        free(ctx);
        return NULL;
    }

    *cpu_backend = ggml_backend {
        /* .guid      = */ ggml_backend_cpu_guid(),
        /* .interface = */ cpu_backend_i,
        /* .context   = */ ctx
    };
    return cpu_backend;
}

GGML_CALL bool ggml_backend_is_cpu(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cpu_guid());
}

void ggml_backend_cpu_set_n_threads(ggml_backend_t backend_cpu, int n_threads) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->n_threads = n_threads;
}

void ggml_backend_cpu_set_moe_expert_prefetch(ggml_backend_t backend_cpu, bool enable) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->moe_expert_prefetch = enable;
}

void ggml_backend_cpu_set_expert_lease_callbacks(
        ggml_backend_t backend_cpu,
        ggml_expert_acquire_callback acquire,
        ggml_expert_release_callback release,
        void * user_data,
        bool required) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));
    GGML_ASSERT((acquire == nullptr) == (release == nullptr));
    auto * ctx = (struct ggml_backend_cpu_context *) backend_cpu->context;
    ctx->expert_lease_acquire = acquire;
    ctx->expert_lease_release = release;
    ctx->expert_lease_user_data = user_data;
    ctx->expert_lease_required = required;
}

void ggml_backend_cpu_set_abort_callback(ggml_backend_t backend_cpu, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(ggml_backend_is_cpu(backend_cpu));

    struct ggml_backend_cpu_context * ctx = (struct ggml_backend_cpu_context *)backend_cpu->context;
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

GGML_CALL ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_type(), cpu_backend_buffer_i_from_ptr, ptr, size);
}

GGML_CALL static ggml_backend_t ggml_backend_reg_cpu_init(const char * params, void * user_data) {
    return ggml_backend_cpu_init();

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

typedef struct ggml_backend_multi_buffer_context * ggml_backend_multi_buffer_context_t;

GGML_CALL static const char * ggml_backend_multi_buffer_get_name(ggml_backend_buffer_t buffer) {
    ggml_backend_multi_buffer_context_t ctx = (ggml_backend_multi_buffer_context_t) buffer->context;

    return ctx->buffers[0]->iface.get_name(ctx->buffers[0]);
}

GGML_CALL static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_multi_buffer_context_t ctx = (ggml_backend_multi_buffer_context_t) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

GGML_CALL static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_multi_buffer_context_t ctx = (ggml_backend_multi_buffer_context_t) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static struct ggml_backend_buffer_i ggml_backend_multi_buffer_context_interface(void) {
    static struct ggml_backend_buffer_i multi_backend_buffer_i = {
        /* .get_name        = */ ggml_backend_multi_buffer_get_name,
        /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
        /* .get_base        = */ NULL,
        /* .init_tensor     = */ NULL,
        /* .memset_tensor   = */ NULL,
        /* .set_tensor      = */ NULL,
        /* .get_tensor      = */ NULL,
        /* .cpy_tensor      = */ NULL,
        /* .clear           = */ ggml_backend_multi_buffer_clear,
        /* .reset           = */ NULL,
    };

    return multi_backend_buffer_i;
}

GGML_CALL ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context_t ctx = (ggml_backend_multi_buffer_context_t) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_context_interface(), ctx, total_size);
}

GGML_CALL bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    return buffer->iface.get_name == ggml_backend_multi_buffer_get_name;
}

GGML_CALL void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context_t ctx = (ggml_backend_multi_buffer_context_t) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLITS
#define GGML_SCHED_MAX_SPLITS 4096
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
// Gemma4 with per-layer embeddings and uses up to 32 inputs
#define GGML_SCHED_MAX_SPLIT_INPUTS 64
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 1
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

// The active-expert path normally uploads the selected slices into a temporary
// full-expert tensor on every decode step.  This opt-in cache retains a small
// number of complete expert triplets on each CUDA device instead.  The cache is
// intentionally generic at the scheduler level: the slot key is (layer,
// expert), while the tensors in a slot are the gate/up/down projections.
//
// The CUDA MoE kernels already index experts via the I32 top-k tensor.  For a
// cache hit (or a newly staged miss), we provide a compact replacement ID tensor
// and point the temporary input at the cache buffer.  Its logical 128-expert
// layout remains unchanged, but all active IDs are in the small cache-slot
// range, so CUDA never dereferences the unused tail of the logical tensor.
enum ggml_active_expert_cache_component {
    GGML_ACTIVE_EXPERT_CACHE_GATE = 0,
    GGML_ACTIVE_EXPERT_CACHE_UP   = 1,
    GGML_ACTIVE_EXPERT_CACHE_DOWN = 2,
    GGML_ACTIVE_EXPERT_CACHE_GATE_UP = 3,
    GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT = 4,
};

// Scheduler graph tensors are rebuilt in a scratch context on every split.
// Keep only the immutable expert geometry needed to prepare a replacement
// cache; retaining graph tensor pointers here would become a use-after-free on
// the next graph rebuild.  The model is fixed for a scheduler lifetime, so the
// catalog can monotonically collect every layout observed by prompt/decode
// graphs and remains available while the live cache is disabled.
struct ggml_active_expert_cache_layout {
    enum ggml_type type = GGML_TYPE_COUNT;
    int64_t ne0 = 0;
    int64_t ne1 = 0;
    int64_t logical_experts = 0;
    size_t nb0 = 0;
    size_t nb1 = 0;
    size_t nb2 = 0;
};

static constexpr size_t GGML_ACTIVE_EXPERT_CACHE_BASE_ROUTE_COUNT = 64;
static constexpr size_t GGML_ACTIVE_EXPERT_CACHE_ID_STRIDE = 256;
static constexpr size_t GGML_ACTIVE_EXPERT_CACHE_BASE_ID_BYTES =
        GGML_ACTIVE_EXPERT_CACHE_BASE_ROUTE_COUNT*GGML_ACTIVE_EXPERT_CACHE_ID_STRIDE;

struct ggml_active_expert_cache_entry {
    int layer = -1;
    int expert = -1;
    uint64_t last_used = 0;
    uint8_t ready_mask = 0;
};

struct ggml_active_expert_cache_component_state {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_tensor * tensor = nullptr;
    std::vector<ggml_tensor *> layouts;
    size_t expert_bytes = 0;
    size_t buffer_bytes = 0;
};

struct ggml_active_expert_cache_device {
    bool attempted = false;
    bool ready = false;
    int backend_id = -1;
    int slots = 0;
    int64_t logical_experts = 0;
    uint64_t clock = 0;
    uint64_t allocated_bytes = 0;
    uint64_t capacity_bytes = 0;
    std::array<ggml_active_expert_cache_component_state, GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT> components;
    std::vector<ggml_active_expert_cache_entry> entries;
    ggml_backend_buffer_t ids_buffer = nullptr;
    std::vector<ggml_tensor *> ids_tensors;
    // Stable host sources paired with ids_tensors. CUDA H2D submission may
    // outlive prepare_route(), so route-remap bytes must not come from a
    // temporary vector.
    std::vector<std::array<int32_t, 64>> ids_host;
    uint64_t ids_bytes = 0;
    size_t ids_next = 0;
    ggml_context * ctx = nullptr;
    ggml_backend_event_t lease_event = nullptr;
    ggml_backend_event_t compute_event = nullptr;
    ggml_backend_t transfer_backend = nullptr;
    bool lease_event_recorded = false;
    bool compute_event_recorded = false;
    bool used_this_pass = false;
    std::vector<void *> lease_handles;
    std::unordered_map<uint64_t, uint32_t> observations;
    std::unordered_map<uint64_t, uint64_t> admission_thresholds;
    std::unordered_set<uint64_t> observed_this_pass;
    std::unordered_map<uint64_t, uint64_t> last_observed_route;
    std::unordered_map<int, std::vector<int32_t>> previous_route;
    uint64_t route_clock = 0;
    int pending_upload_layer = -1;
};

struct ggml_active_expert_cache_route {
    ggml_tensor * ids_source = nullptr;
    ggml_active_expert_cache_device * cache = nullptr;
    int backend_id = -1;
    int layer = -1;
    std::vector<int32_t> experts;
    std::vector<int> slots;
    ggml_tensor * mapped_ids = nullptr;
};

struct ggml_active_expert_cache_layer_stats {
    uint64_t routes = 0;
    uint64_t route_positions = 0;
    uint64_t gpu_route_positions = 0;
    uint64_t route_readback_ns = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t lease_acquire_ns = 0;
    uint64_t lease_uploads = 0;
    uint64_t transfer_submit_ns = 0;
    uint64_t transfer_wait_ns = 0;
    uint64_t load_bytes = 0;
    uint64_t cpu_compute_ns = 0;
    uint64_t cpu_compute_calls = 0;
};

struct ggml_expert_prefill_lane {
    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_event_t transfer_event = nullptr;
    ggml_backend_event_t compute_event = nullptr;
    bool transfer_event_recorded = false;
    bool compute_event_recorded = false;
    bool transfers_queued = false;
    int layer = -1;
    std::vector<uint8_t> ready_masks;
    std::vector<void *> lease_handles;
};

struct ggml_expert_prefill_device {
    bool attempted = false;
    bool ready = false;
    int backend_id = -1;
    uint64_t lane_bytes = 0;
    uint64_t allocated_bytes = 0;
    ggml_backend_t transfer_backend = nullptr;
    ggml_backend_event_t prior_compute_event = nullptr;
    std::array<ggml_expert_prefill_lane, 2> lanes;
    std::array<bool, 2> used_this_split = {{ false, false }};
};

struct ggml_expert_prefill_layer_stats {
    uint64_t route_readback_ns = 0;
    uint64_t selected_components = 0;
    uint64_t h2d_components = 0;
    uint64_t d2d_components = 0;
    uint64_t h2d_batches = 0;
    uint64_t d2d_batches = 0;
    uint64_t h2d_bytes = 0;
    uint64_t d2d_bytes = 0;
    uint64_t lease_acquire_ns = 0;
    uint64_t transfer_submit_ns = 0;
    uint64_t transfer_wait_ns = 0;
    uint64_t fallbacks = 0;
};

struct ggml_active_expert_cache_redirect {
    ggml_tensor * input_cpy = nullptr;
    void * input_data = nullptr;
    ggml_backend_buffer_t input_buffer = nullptr;
    int64_t input_ne2 = 0;
    size_t input_nb3 = 0;
    ggml_tensor * node = nullptr;
    int ids_index = -1;
    ggml_tensor * node_ids = nullptr;
    ggml_tensor * source_ids = nullptr;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    size_t max_extra_alloc = 0;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    std::array<ggml_backend_buffer_t, GGML_SCHED_MAX_BACKENDS> input_memory_bufs = {{ nullptr }};

    uint32_t op_offload[(GGML_OP_COUNT + 31)/32];

    std::vector<std::thread> workers;
    std::vector<ggml_status> statuses;
    std::vector<std::vector<ggml_backend_sched_split*>> backend_splits;
    std::array<bool, GGML_SCHED_MAX_BACKENDS> needs_sync;
    std::array<bool, GGML_SCHED_MAX_BACKENDS> own_cpy;

    bool only_active_experts;
    bool active_expert_decode = true;
    std::array<uint64_t, GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT> active_expert_component_max_bytes = {};
    uint64_t active_expert_set_bytes = 0;
    bool split_mode_graph;
    bool is_async = false;
    bool debug;
    bool has_reduce = false;

    // LLAMA_EXPERT_GPU_CACHE_SLOTS=0 (the default) preserves the original
    // transfer-only behavior.  Values 4..64 enable that many slots per GPU.
    int active_expert_cache_slots = 0;
    std::array<ggml_active_expert_cache_device, GGML_SCHED_MAX_BACKENDS> active_expert_caches;
    std::array<std::vector<ggml_active_expert_cache_layout>,
            GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT> active_expert_cache_layout_catalog;
    size_t active_expert_cache_route_capacity = GGML_ACTIVE_EXPERT_CACHE_BASE_ROUTE_COUNT;
    std::vector<ggml_active_expert_cache_route> active_expert_routes;
    std::vector<ggml_active_expert_cache_redirect> active_expert_redirects;
    uint64_t active_expert_cache_hits = 0;
    uint64_t active_expert_cache_misses = 0;
    uint64_t active_expert_cache_admissions = 0;
    uint64_t active_expert_cache_evictions = 0;
    uint64_t active_expert_cache_uploads = 0;
    uint64_t active_expert_cache_lease_uploads = 0;
    uint64_t active_expert_cache_forced_fallbacks = 0;
    uint64_t active_expert_cache_rejected_admissions = 0;
    uint64_t active_expert_cache_lease_acquire_ns = 0;
    uint64_t active_expert_cache_transfer_submit_ns = 0;
    uint64_t active_expert_cache_transfer_wait_ns = 0;
    uint64_t active_expert_cache_route_observations = 0;
    uint64_t active_expert_cache_route_prediction_matches = 0;
    uint64_t active_expert_cache_prediction_admissions = 0;
    uint64_t active_expert_cache_reuse_distance_sum = 0;
    uint64_t active_expert_cache_load_bytes = 0;
    uint64_t active_expert_cache_eviction_cost_bytes = 0;
    uint64_t active_expert_cache_cpu_compute_ns = 0;
    uint64_t active_expert_cache_cpu_compute_calls = 0;
    std::map<int, ggml_active_expert_cache_layer_stats> active_expert_cache_layer_stats;
    ggml_backend_expert_acquire_callback expert_lease_acquire = nullptr;
    ggml_backend_expert_release_callback expert_lease_release = nullptr;
    void * expert_lease_user_data = nullptr;
    bool expert_lease_required = false;
    uint64_t active_expert_cache_capacity_bytes = 0;
    uint64_t active_expert_cache_reserve_bytes = 0;
    uint64_t expert_prefill_staging_capacity_bytes = 0;
    uint64_t expert_prefill_staging_reserve_bytes = 0;
    uint64_t expert_prefill_lane_bytes = 0;
    std::map<int, std::array<uint64_t, GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT>> expert_prefill_layer_bytes;
    std::array<ggml_expert_prefill_device, GGML_SCHED_MAX_BACKENDS> expert_prefill_devices;
    std::array<ggml_backend_event_t, GGML_SCHED_MAX_BACKENDS> expert_prefill_route_events = {{ nullptr }};
    std::map<int, ggml_expert_prefill_layer_stats> expert_prefill_layer_stats;
    uint64_t expert_prefill_fallbacks = 0;
    uint64_t expert_prefill_route_global_sync_fallbacks = 0;
    uint32_t active_expert_cache_min_observations = 1;
    uint64_t expert_hybrid_guard_cpu_ns_per_expert = 0;
    uint64_t expert_hybrid_guard_upload_ns_per_expert = 0;
    uint32_t expert_hybrid_guard_cpu_route_positions = 0;
    uint32_t expert_hybrid_guard_maximum_drift_ppm = 0;
    uint32_t expert_hybrid_guard_minimum_cpu_calls = 0;
    std::atomic<enum ggml_backend_expert_hybrid_guard_status> expert_hybrid_guard_status {
            GGML_BACKEND_EXPERT_HYBRID_GUARD_DISABLED };
    struct ggml_backend_expert_hybrid_guard_window expert_hybrid_guard_baseline = {};
};

static void ggml_expert_prefill_drain_lane(
        ggml_backend_sched_t sched,
        ggml_expert_prefill_lane & lane,
        int stats_layer);
static bool ggml_expert_prefill_init_device(
        ggml_backend_sched_t sched,
        int backend_id);

enum ggml_backend_expert_hybrid_guard_status ggml_backend_expert_hybrid_guard_evaluate(
        const struct ggml_backend_expert_hybrid_guard_window * window,
        uint64_t cpu_ns_per_expert,
        uint32_t cpu_route_positions,
        uint64_t upload_ns_per_expert,
        uint32_t maximum_drift_ppm,
        uint32_t minimum_cpu_calls) {
    if (!window || cpu_ns_per_expert == 0 || cpu_route_positions == 0 ||
            upload_ns_per_expert == 0 || maximum_drift_ppm < 1000000 ||
            minimum_cpu_calls == 0) {
        return GGML_BACKEND_EXPERT_HYBRID_GUARD_DISABLED;
    }
    if (window->forced_fallbacks > 0) {
        return GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_FALLBACK;
    }
    if (window->cpu_compute_calls >= minimum_cpu_calls) {
        const long double predicted = (long double) cpu_ns_per_expert *
                cpu_route_positions * window->cpu_compute_calls;
        const long double allowed = predicted * maximum_drift_ppm / 1000000.0L;
        if (window->cpu_compute_ns == 0 || (long double) window->cpu_compute_ns > allowed) {
            return GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_CPU_DRIFT;
        }
    }
    // Prefill and established-path cache traffic can occur before the first
    // mixed decode graph.  Only judge upload drift in a window that also
    // proves the hybrid CPU branch executed.
    if (window->cpu_compute_calls > 0 && window->misses >= 3) {
        const long double observed = (long double) window->lease_acquire_ns +
                window->transfer_submit_ns + window->transfer_wait_ns;
        const long double predicted = (long double) upload_ns_per_expert * window->misses;
        const long double allowed = predicted * maximum_drift_ppm / 1000000.0L;
        if (observed == 0 || observed > allowed) {
            return GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_UPLOAD_DRIFT;
        }
    }
    return GGML_BACKEND_EXPERT_HYBRID_GUARD_MONITORING;
}

void ggml_backend_sched_set_op_offload(ggml_backend_sched_t sched, enum ggml_op op, bool on_or_off) {
    int int_op = (int)op;
    if (!sched) return;
    if (int_op < 0 || int_op >= (int)GGML_OP_COUNT) {
        uint32_t mask = on_or_off ? 0xffffffff : 0;
        for (int i = 0; i < (GGML_OP_COUNT + 31)/32; ++i) sched->op_offload[i] = mask;
        return;
    }
    int i = int_op >> 5;
    int j = int_op & 31;
    if (on_or_off) {
        sched->op_offload[i] |= (1u << j);
    } else {
        sched->op_offload[i] &= (~(1u << j));
    }
}

void ggml_backend_sched_set_only_active_experts(ggml_backend_sched_t sched, bool on_or_off) {
    if (!sched) return;
    sched->only_active_experts = on_or_off;
}

void ggml_backend_sched_set_active_expert_decode(ggml_backend_sched_t sched, bool on_or_off) {
    sched->active_expert_decode = on_or_off;
}

void ggml_backend_sched_set_split_mode_graph(ggml_backend_sched_t sched, bool on_or_off, bool async) {
    if (!sched) return;
    sched->split_mode_graph = on_or_off;
    sched->is_async = async;
}

void ggml_backend_sched_set_max_extra_alloc(ggml_backend_sched_t sched, int extra_alloc_MiB) {
    if (!sched) return;
    if (extra_alloc_MiB >= 0) {
        sched->max_extra_alloc = size_t(extra_alloc_MiB)*1024*1024;
    }
}

void ggml_backend_sched_set_expert_lease_callbacks(
        ggml_backend_sched_t sched,
        ggml_backend_expert_acquire_callback acquire,
        ggml_backend_expert_release_callback release,
        void * user_data,
        bool required) {
    if (!sched) return;
    GGML_ASSERT((acquire == nullptr) == (release == nullptr));
    sched->expert_lease_acquire = acquire;
    sched->expert_lease_release = release;
    sched->expert_lease_user_data = user_data;
    sched->expert_lease_required = required;
}

void ggml_backend_sched_set_expert_cache_capacity(
        ggml_backend_sched_t sched,
        uint64_t bytes_per_device,
        uint64_t reserve_bytes_per_device,
        uint32_t minimum_observations) {
    if (!sched) return;
    sched->active_expert_cache_capacity_bytes = bytes_per_device;
    sched->active_expert_cache_reserve_bytes = reserve_bytes_per_device;
    sched->active_expert_cache_min_observations = std::max<uint32_t>(1, minimum_observations);
    if (bytes_per_device > 0) {
        // Marks the path enabled; the exact slot count is derived from each
        // tensor's checked expert stride before allocation.
        sched->active_expert_cache_slots = 64;
    }
}

void ggml_backend_sched_set_expert_prefill_staging(
        ggml_backend_sched_t sched,
        uint64_t bytes_per_device,
        uint64_t reserve_bytes_per_device) {
    if (!sched) return;
    sched->expert_prefill_staging_capacity_bytes = bytes_per_device;
    sched->expert_prefill_staging_reserve_bytes = reserve_bytes_per_device;
}

size_t ggml_backend_sched_get_resource_device_count(ggml_backend_sched_t sched) {
    return sched && sched->n_backends > 0 ? size_t(sched->n_backends - 1) : 0;
}

bool ggml_backend_sched_get_resource_device_stats(
        ggml_backend_sched_t sched,
        struct ggml_backend_sched_resource_device_stats * stats,
        size_t capacity) {
    const size_t count = ggml_backend_sched_get_resource_device_count(sched);
    if (!sched || (count > 0 && (!stats || capacity < count))) return false;
    for (size_t backend_id = 0; backend_id < count; ++backend_id) {
        const auto & cache = sched->active_expert_caches[backend_id];
        uint64_t resident_bytes = 0;
        if (cache.ready) {
            for (const auto & entry : cache.entries) {
                if (entry.layer < 0 || entry.expert < 0) continue;
                for (size_t component = 0; component < cache.components.size(); ++component) {
                    if ((entry.ready_mask & uint8_t(1u << component)) == 0) continue;
                    const uint64_t bytes = cache.components[component].expert_bytes;
                    resident_bytes = bytes > UINT64_MAX - resident_bytes
                        ? UINT64_MAX : resident_bytes + bytes;
                }
            }
        }
        const auto & prefill = sched->expert_prefill_devices[backend_id];
        stats[backend_id] = {
            int32_t(backend_id),
            cache.ready ? cache.capacity_bytes : 0,
            cache.ready ? cache.allocated_bytes : 0,
            resident_bytes,
            sched->expert_prefill_staging_capacity_bytes,
            prefill.ready ? prefill.allocated_bytes : 0,
        };
    }
    return true;
}

void ggml_backend_sched_set_expert_hybrid_guard(
        ggml_backend_sched_t sched,
        uint64_t cpu_ns_per_expert,
        uint32_t cpu_route_positions,
        uint64_t upload_ns_per_expert,
        uint32_t maximum_drift_ppm,
        uint32_t minimum_cpu_calls) {
    if (!sched) return;
    sched->expert_hybrid_guard_cpu_ns_per_expert = cpu_ns_per_expert;
    sched->expert_hybrid_guard_cpu_route_positions = cpu_route_positions;
    sched->expert_hybrid_guard_upload_ns_per_expert = upload_ns_per_expert;
    sched->expert_hybrid_guard_maximum_drift_ppm = maximum_drift_ppm;
    sched->expert_hybrid_guard_minimum_cpu_calls = minimum_cpu_calls;
    sched->expert_hybrid_guard_baseline = {
        sched->active_expert_cache_misses,
        sched->active_expert_cache_forced_fallbacks,
        sched->active_expert_cache_lease_acquire_ns,
        sched->active_expert_cache_transfer_submit_ns,
        sched->active_expert_cache_transfer_wait_ns,
        sched->active_expert_cache_cpu_compute_ns,
        sched->active_expert_cache_cpu_compute_calls,
    };
    const struct ggml_backend_expert_hybrid_guard_window empty = {};
    sched->expert_hybrid_guard_status.store(ggml_backend_expert_hybrid_guard_evaluate(
            &empty, cpu_ns_per_expert, cpu_route_positions, upload_ns_per_expert,
            maximum_drift_ppm, minimum_cpu_calls), std::memory_order_release);
}

enum ggml_backend_expert_hybrid_guard_status
ggml_backend_sched_get_expert_hybrid_guard_status(ggml_backend_sched_t sched) {
    return sched ? sched->expert_hybrid_guard_status.load(std::memory_order_acquire) :
            GGML_BACKEND_EXPERT_HYBRID_GUARD_DISABLED;
}

static void ggml_backend_sched_evaluate_expert_hybrid_guard(ggml_backend_sched_t sched) {
    if (!sched || sched->expert_hybrid_guard_status.load(std::memory_order_acquire) !=
            GGML_BACKEND_EXPERT_HYBRID_GUARD_MONITORING) {
        return;
    }
    auto & baseline = sched->expert_hybrid_guard_baseline;
    const struct ggml_backend_expert_hybrid_guard_window window = {
        sched->active_expert_cache_misses - baseline.misses,
        sched->active_expert_cache_forced_fallbacks - baseline.forced_fallbacks,
        sched->active_expert_cache_lease_acquire_ns - baseline.lease_acquire_ns,
        sched->active_expert_cache_transfer_submit_ns - baseline.transfer_submit_ns,
        sched->active_expert_cache_transfer_wait_ns - baseline.transfer_wait_ns,
        sched->active_expert_cache_cpu_compute_ns - baseline.cpu_compute_ns,
        sched->active_expert_cache_cpu_compute_calls - baseline.cpu_compute_calls,
    };
    const auto status = ggml_backend_expert_hybrid_guard_evaluate(
            &window,
            sched->expert_hybrid_guard_cpu_ns_per_expert,
            sched->expert_hybrid_guard_cpu_route_positions,
            sched->expert_hybrid_guard_upload_ns_per_expert,
            sched->expert_hybrid_guard_maximum_drift_ppm,
            sched->expert_hybrid_guard_minimum_cpu_calls);
    if (status != GGML_BACKEND_EXPERT_HYBRID_GUARD_MONITORING) {
        sched->expert_hybrid_guard_status.store(status, std::memory_order_release);
        const char * reason = status == GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_FALLBACK ?
                "forced-fallback" :
                status == GGML_BACKEND_EXPERT_HYBRID_GUARD_FAILED_CPU_DRIFT ?
                "cpu-drift" : "upload-drift";
        fprintf(stderr,
                "expert_hybrid_guard: {\"status\":\"revoked\",\"reason\":\"%s\","
                "\"status_code\":%d,\"misses\":%llu,"
                "\"forced_fallbacks\":%llu,\"cpu_compute_ns\":%llu,"
                "\"cpu_compute_calls\":%llu,\"lease_acquire_ns\":%llu,"
                "\"transfer_submit_ns\":%llu,\"transfer_wait_ns\":%llu,"
                "\"cpu_ns_per_expert\":%llu,\"cpu_route_positions\":%u,"
                "\"upload_ns_per_expert\":%llu,\"maximum_drift_ppm\":%u}\n",
                reason,
                (int) status,
                (unsigned long long) window.misses,
                (unsigned long long) window.forced_fallbacks,
                (unsigned long long) window.cpu_compute_ns,
                (unsigned long long) window.cpu_compute_calls,
                (unsigned long long) window.lease_acquire_ns,
                (unsigned long long) window.transfer_submit_ns,
                (unsigned long long) window.transfer_wait_ns,
                (unsigned long long) sched->expert_hybrid_guard_cpu_ns_per_expert,
                sched->expert_hybrid_guard_cpu_route_positions,
                (unsigned long long) sched->expert_hybrid_guard_upload_ns_per_expert,
                sched->expert_hybrid_guard_maximum_drift_ppm);
        return;
    }
    if (window.cpu_compute_calls >= sched->expert_hybrid_guard_minimum_cpu_calls) {
        baseline.cpu_compute_ns = sched->active_expert_cache_cpu_compute_ns;
        baseline.cpu_compute_calls = sched->active_expert_cache_cpu_compute_calls;
    }
    if (window.misses >= 3) {
        baseline.misses = sched->active_expert_cache_misses;
        baseline.lease_acquire_ns = sched->active_expert_cache_lease_acquire_ns;
        baseline.transfer_submit_ns = sched->active_expert_cache_transfer_submit_ns;
        baseline.transfer_wait_ns = sched->active_expert_cache_transfer_wait_ns;
    }
    baseline.forced_fallbacks = sched->active_expert_cache_forced_fallbacks;
}

bool ggml_backend_prefetch_init(int n_threads) {
    if (n_threads <= 0) {
        n_threads = std::max(1, std::min(8, (int) std::thread::hardware_concurrency()));
    }
    ggml_moe_prefetch_set_n_threads(n_threads);
    return ggml_moe_prefetch_enabled();
}

void ggml_backend_prefetch_register_mapping(const void * addr, size_t size) {
    ggml_moe_prefetch_register_mapping(addr, size);
}

void ggml_backend_prefetch_unregister_mapping(const void * addr) {
    ggml_moe_prefetch_unregister_mapping(addr);
}

static inline bool ggml_backend_sched_offload_enabled(ggml_backend_sched_t sched, enum ggml_op op) {
    int int_op = (int)op;
    if (!sched || op < 0 || op >= GGML_OP_COUNT) return false;
    return sched->op_offload[int_op >> 5] & (1u << (int_op & 31));
}

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

static int ggml_active_expert_cache_layer_from_name(const char * name) {
    static constexpr char prefix[] = "blk.";
    if (std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) {
        return -1;
    }

    char * end = nullptr;
    const long parsed_layer = std::strtol(name + sizeof(prefix) - 1, &end, 10);
    if (end == name + sizeof(prefix) - 1 || parsed_layer < 0 || parsed_layer > INT_MAX) {
        return -1;
    }

    return (int) parsed_layer;
}

static int ggml_ese_callback_layer_from_name(const char * name) {
    if (!name || !name[0]) return -1;
    const char * separator = std::strrchr(name, '-');
    if (!separator || !separator[1]) return -1;
    char * end = nullptr;
    const long layer = std::strtol(separator + 1, &end, 10);
    if (end == separator + 1 || *end != '\0' || layer < 0 || layer > INT_MAX) return -1;
    return (int) layer;
}

static int ggml_active_expert_cache_component_from_name(const ggml_tensor * tensor, int * layer) {
    const char * name = ggml_get_name(tensor);
    const int parsed_layer = ggml_active_expert_cache_layer_from_name(name);
    if (parsed_layer < 0) {
        return -1;
    }

    *layer = parsed_layer;
    if (std::strstr(name, ".ffn_gate_up_exps") != nullptr) return GGML_ACTIVE_EXPERT_CACHE_GATE_UP;
    if (std::strstr(name, ".ffn_gate_exps") != nullptr) return GGML_ACTIVE_EXPERT_CACHE_GATE;
    if (std::strstr(name, ".ffn_up_exps")   != nullptr) return GGML_ACTIVE_EXPERT_CACHE_UP;
    if (std::strstr(name, ".ffn_down_exps") != nullptr) return GGML_ACTIVE_EXPERT_CACHE_DOWN;
    return -1;
}

static bool ggml_active_expert_cache_is_decode_node(const ggml_tensor * node) {
    if (node->op == GGML_OP_MOE_FUSED_UP_GATE) {
        return node->src[3] && node->src[3]->type == GGML_TYPE_I32 &&
                node->src[3]->ne[0] >= 1 && node->src[3]->ne[0] <= 64 && node->src[3]->ne[1] == 1;
    }
    if (node->op == GGML_OP_MUL_MAT_ID) {
        return node->src[2] && node->src[2]->type == GGML_TYPE_I32 &&
                node->src[2]->ne[0] >= 1 && node->src[2]->ne[0] <= 64 && node->src[2]->ne[1] == 1;
    }
    return false;
}

static bool ggml_active_expert_cache_is_host_expert_weight(const ggml_tensor * tensor) {
    int layer = -1;
    return tensor && tensor->buffer && tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
            ggml_backend_buffer_is_host(tensor->buffer) &&
            ggml_active_expert_cache_component_from_name(tensor, &layer) >= 0 && tensor->ne[2] > 1;
}

// Decode is the mode where all routed experts can fit in a small cache. Keep
// prompt batches on their established CPU-MoE graph; when the
// graph reduces to a single decode token, select the GPU that already owns the
// activation and give the MoE op compact cache-backed expert inputs.
static int ggml_active_expert_cache_layer_backend(
        ggml_backend_sched_t sched,
        const ggml_cgraph * graph,
        int layer) {
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * graph_node = graph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            const ggml_tensor * source = graph_node->src[j];
            if (!source || !source->buffer || source->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
                    ggml_backend_buffer_is_host(source->buffer) ||
                    ggml_active_expert_cache_layer_from_name(ggml_get_name(source)) != layer) {
                continue;
            }
            for (int backend_id = 0; backend_id < sched->n_backends - 1; ++backend_id) {
                if (ggml_backend_supports_buft(sched->backends[backend_id], source->buffer->buft) &&
                        ggml_backend_supports_op(sched->backends[backend_id], graph_node)) {
                    return backend_id;
                }
            }
        }
    }
    return -1;
}

static int ggml_active_expert_cache_slots_for(
        ggml_backend_sched_t sched,
        const ggml_tensor * source);

static int ggml_active_expert_cache_backend_for_node(
        ggml_backend_sched_t sched,
        const ggml_cgraph * graph,
        ggml_tensor * node) {
    if (sched->active_expert_cache_slots < 1 || !sched->active_expert_decode) {
        return -1;
    }

    if (!ggml_active_expert_cache_is_decode_node(node)) {
        if (getenv("LLAMA_EXPERT_GPU_CACHE_DEBUG") &&
                (node->op == GGML_OP_MOE_FUSED_UP_GATE || node->op == GGML_OP_MUL_MAT_ID) &&
                ggml_active_expert_cache_is_host_expert_weight(node->src[0])) {
            const ggml_tensor * ids = node->op == GGML_OP_MOE_FUSED_UP_GATE ? node->src[3] : node->src[2];
            fprintf(stderr, "ggml active-expert cache: CPU graph candidate %s ids=[%lld,%lld] weight=%s\n",
                    ggml_op_name(node->op), (long long) ids->ne[0], (long long) ids->ne[1], ggml_get_name(node->src[0]));
        }
        return -1;
    }

    const ggml_tensor * branch_ids = node->op == GGML_OP_MOE_FUSED_UP_GATE ? node->src[3] : node->src[2];
    if (ggml_ese_route_get_role(branch_ids) == GGML_ESE_ROUTE_CPU) {
        return -1;
    }

    const ggml_tensor * route_ids = node->op == GGML_OP_MOE_FUSED_UP_GATE ? node->src[3] : node->src[2];
    if (!route_ids || !node->src[0] ||
            ggml_active_expert_cache_slots_for(sched, node->src[0]) < route_ids->ne[0]) {
        // Worst-case/prompt graphs deliberately route every expert even when
        // microbatched one token at a time.  A bounded decode cache must not
        // pull those nodes onto the GPU only to fall back to a full-tensor
        // copy; leave them on the established CPU-MoE path.
        return -1;
    }

    bool has_host_expert_weight = false;
    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        if (ggml_active_expert_cache_is_host_expert_weight(node->src[j])) {
            has_host_expert_weight = true;
            break;
        }
    }
    if (!has_host_expert_weight) {
        return -1;
    }

    if (getenv("LLAMA_EXPERT_GPU_CACHE_DEBUG")) {
        const ggml_tensor * ids = node->op == GGML_OP_MOE_FUSED_UP_GATE ? node->src[3] : node->src[2];
        fprintf(stderr, "ggml active-expert cache: decode graph candidate %s ids=[%lld,%lld] weight=%s backend src=[%d,%d,%d,%d]\n",
                ggml_op_name(node->op), (long long) ids->ne[0], (long long) ids->ne[1], ggml_get_name(node->src[0]),
                node->src[0] ? tensor_backend_id(node->src[0]) : -1,
                node->src[1] ? tensor_backend_id(node->src[1]) : -1,
                node->src[2] ? tensor_backend_id(node->src[2]) : -1,
                node->src[3] ? tensor_backend_id(node->src[3]) : -1);
    }

    for (int j = 0; j < GGML_MAX_SRC; ++j) {
        ggml_tensor * src = node->src[j];
        if (!src || ggml_active_expert_cache_is_host_expert_weight(src)) {
            continue;
        }
        const int backend_id = tensor_backend_id(src);
        if (backend_id >= 0 && backend_id < sched->n_backends - 1 &&
                ggml_backend_supports_op(sched->backends[backend_id], node)) {
            return backend_id;
        }
    }

    int layer = -1;
    ggml_active_expert_cache_component_from_name(node->src[0], &layer);
    if (layer >= 0) {
        return ggml_active_expert_cache_layer_backend(sched, graph, layer);
    }
    return -1;
}

static int ggml_expert_prefill_backend_for_node(
        ggml_backend_sched_t sched,
        const ggml_cgraph * graph,
        ggml_tensor * node) {
    if (sched->active_expert_decode || sched->expert_prefill_staging_capacity_bytes == 0 ||
            sched->expert_prefill_lane_bytes == 0 ||
            sched->expert_prefill_lane_bytes > UINT64_MAX/2 ||
            2*sched->expert_prefill_lane_bytes > sched->expert_prefill_staging_capacity_bytes ||
            (node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_MOE_FUSED_UP_GATE)) {
        return -1;
    }
    const int ids_index = node->op == GGML_OP_MOE_FUSED_UP_GATE ? 3 : 2;
    const ggml_tensor * ids = node->src[ids_index];
    if (!ids || ids->type != GGML_TYPE_I32 || ids->ne[0] < 1 || ids->ne[0] > 64 || ids->ne[1] <= 1) {
        return -1;
    }
    const bool debug = getenv("LLAMA_EXPERT_PREFILL_DEBUG") != nullptr;
    int debug_layer = -1;
    ggml_active_expert_cache_component_from_name(node->src[0], &debug_layer);
    bool has_host_expert_weight = false;
    for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
        if (ggml_active_expert_cache_is_host_expert_weight(node->src[source_index])) {
            has_host_expert_weight = true;
            break;
        }
    }
    if (!has_host_expert_weight) {
        if (debug) fprintf(stderr,
                "ggml expert prefill: rejected layer=%d op=%s without host expert weight src=[%s,%s,%s,%s]\n",
                debug_layer, ggml_op_name(node->op),
                node->src[0] ? ggml_get_name(node->src[0]) : "-",
                node->src[1] ? ggml_get_name(node->src[1]) : "-",
                node->src[2] ? ggml_get_name(node->src[2]) : "-",
                node->src[3] ? ggml_get_name(node->src[3]) : "-");
        return -1;
    }
    if (debug) {
        fprintf(stderr,
                "ggml expert prefill: candidate layer=%d op=%s ids=[%lld,%lld] src_backends=[%d,%d,%d,%d]\n",
                debug_layer, ggml_op_name(node->op),
                (long long) ids->ne[0], (long long) ids->ne[1],
                node->src[0] ? tensor_backend_id(node->src[0]) : -1,
                node->src[1] ? tensor_backend_id(node->src[1]) : -1,
                node->src[2] ? tensor_backend_id(node->src[2]) : -1,
                node->src[3] ? tensor_backend_id(node->src[3]) : -1);
    }
    for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
        ggml_tensor * source = node->src[source_index];
        if (!source || ggml_active_expert_cache_is_host_expert_weight(source)) continue;
        const int backend_id = tensor_backend_id(source);
        if (backend_id >= 0 && backend_id < sched->n_backends - 1 &&
                ggml_backend_supports_op(sched->backends[backend_id], node) &&
                ggml_expert_prefill_init_device(sched, backend_id)) {
            if (debug) fprintf(stderr, "ggml expert prefill: layer=%d selected activation backend %d\n", debug_layer, backend_id);
            return backend_id;
        }
    }
    int layer = -1;
    ggml_active_expert_cache_component_from_name(node->src[0], &layer);
    const int backend_id = layer >= 0
        ? ggml_active_expert_cache_layer_backend(sched, graph, layer) : -1;
    if (debug) fprintf(stderr, "ggml expert prefill: layer=%d fallback backend %d\n", layer, backend_id);
    return backend_id >= 0 && ggml_expert_prefill_init_device(sched, backend_id)
        ? backend_id : -1;
}

static int ggml_active_expert_cache_slots_for(
        ggml_backend_sched_t sched,
        const ggml_tensor * source) {
    if (sched->active_expert_cache_capacity_bytes == 0) return sched->active_expert_cache_slots;
    if (!source || source->nb[2] == 0 || source->ne[2] <= 1 || sched->active_expert_set_bytes == 0) return 0;
    const uint64_t payload_bytes = sched->active_expert_cache_capacity_bytes > GGML_ACTIVE_EXPERT_CACHE_BASE_ID_BYTES
            ? sched->active_expert_cache_capacity_bytes - GGML_ACTIVE_EXPERT_CACHE_BASE_ID_BYTES : 0;
    const uint64_t slots = payload_bytes/sched->active_expert_set_bytes;
    return int(std::min<uint64_t>(std::min<uint64_t>(slots, 64), uint64_t(source->ne[2])));
}

static ggml_tensor * ggml_active_expert_cache_compact_copy(
        ggml_backend_sched_t sched,
        const ggml_tensor * source) {
    ggml_tensor * compact = ggml_dup_tensor_layout(sched->ctx, source);
    compact->ne[2] = ggml_active_expert_cache_slots_for(sched, source);
    compact->nb[3] = compact->nb[2]*size_t(compact->ne[2]);
    return compact;
}

static bool ggml_active_expert_cache_same_layout(
        const ggml_tensor * left,
        const ggml_tensor * right);

static void ggml_active_expert_cache_drain_leases(
        ggml_backend_sched_t sched,
        ggml_active_expert_cache_device & cache) {
    if (cache.lease_handles.empty()) return;
    GGML_ASSERT(sched->expert_lease_release != nullptr);
    GGML_ASSERT(cache.lease_event != nullptr && cache.lease_event_recorded);
    const auto wait_start = std::chrono::steady_clock::now();
    ggml_backend_event_synchronize(cache.lease_event);
    const uint64_t wait_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_start).count());
    sched->active_expert_cache_transfer_wait_ns += wait_ns;
    if (cache.pending_upload_layer >= 0) {
        sched->active_expert_cache_layer_stats[cache.pending_upload_layer].transfer_wait_ns += wait_ns;
    }
    for (void * handle : cache.lease_handles) {
        sched->expert_lease_release(sched->expert_lease_user_data, handle);
    }
    cache.lease_handles.clear();
    cache.lease_event_recorded = false;
    cache.pending_upload_layer = -1;
}

static void ggml_active_expert_cache_release(ggml_active_expert_cache_device & cache) {
    GGML_ASSERT(cache.lease_handles.empty());
    if (cache.compute_event) {
        if (cache.compute_event_recorded) {
            ggml_backend_event_synchronize(cache.compute_event);
        }
        ggml_backend_event_free(cache.compute_event);
        cache.compute_event = nullptr;
        cache.compute_event_recorded = false;
    }
    if (cache.lease_event) {
        ggml_backend_event_free(cache.lease_event);
        cache.lease_event = nullptr;
    }
    if (cache.transfer_backend) {
        ggml_backend_free(cache.transfer_backend);
        cache.transfer_backend = nullptr;
    }
    for (auto & component : cache.components) {
        if (component.buffer) {
            ggml_backend_buffer_free(component.buffer);
            component.buffer = nullptr;
        }
        component.tensor = nullptr;
        component.layouts.clear();
        component.expert_bytes = 0;
        component.buffer_bytes = 0;
    }
    if (cache.ids_buffer) {
        ggml_backend_buffer_free(cache.ids_buffer);
        cache.ids_buffer = nullptr;
    }
    cache.ids_tensors.clear();
    cache.ids_host.clear();
    cache.ids_bytes = 0;
    if (cache.ctx) {
        ggml_free(cache.ctx);
        cache.ctx = nullptr;
    }
    cache.entries.clear();
    cache.logical_experts = 0;
    cache.allocated_bytes = 0;
    cache.capacity_bytes = 0;
    cache.used_this_pass = false;
    cache.ready = false;
}

static bool ggml_active_expert_cache_init(
        ggml_backend_sched_t sched,
        int backend_id,
        const ggml_tensor * source) {
    auto & cache = sched->active_expert_caches[backend_id];
    if (cache.attempted) {
        return cache.ready;
    }
    cache.attempted = true;
    cache.backend_id = backend_id;
    cache.logical_experts = source == nullptr ? 0 : source->ne[2];
    cache.capacity_bytes = sched->active_expert_cache_capacity_bytes;
#ifdef GGML_USE_CUDA
    if (cache.capacity_bytes != 0 && ggml_backend_is_cuda(sched->backends[backend_id])) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const int device = ggml_backend_cuda_get_device(sched->backends[backend_id]);
        ggml_backend_cuda_get_device_memory(device, &free_bytes, &total_bytes);
        GGML_UNUSED(total_bytes);
        const uint64_t reserve = sched->active_expert_cache_reserve_bytes;
        const uint64_t available = free_bytes > reserve ? uint64_t(free_bytes) - reserve : 0;
        cache.capacity_bytes = std::min(cache.capacity_bytes, available);
    }
#endif
    std::unordered_set<const ggml_tensor *> graph_routes;
    for (int i = 0; i < sched->graph.n_nodes; ++i) {
        const ggml_tensor * node = sched->graph.nodes[i];
        if (!ggml_active_expert_cache_is_decode_node(node)) continue;
        const int ids_index = node->op == GGML_OP_MOE_FUSED_UP_GATE ? 3 : 2;
        if (node->src[ids_index]) graph_routes.insert(node->src[ids_index]);
    }
    const size_t route_capacity = std::max(
            sched->active_expert_cache_route_capacity,
            std::max(GGML_ACTIVE_EXPERT_CACHE_BASE_ROUTE_COUNT, graph_routes.size()));
    if (route_capacity > UINT64_MAX/GGML_ACTIVE_EXPERT_CACHE_ID_STRIDE) return false;
    cache.ids_bytes = uint64_t(route_capacity)*GGML_ACTIVE_EXPERT_CACHE_ID_STRIDE;
    if (cache.ids_bytes > SIZE_MAX) return false;

    if (sched->active_expert_cache_capacity_bytes == 0) {
        cache.slots = ggml_active_expert_cache_slots_for(sched, source);
    } else {
        if (sched->active_expert_set_bytes == 0) return false;
        const uint64_t payload_bytes = cache.capacity_bytes > cache.ids_bytes
                ? cache.capacity_bytes - cache.ids_bytes : 0;
        cache.slots = int(std::min<uint64_t>(64, std::min<uint64_t>(
                source->ne[2], payload_bytes/sched->active_expert_set_bytes)));
    }

    if (cache.slots < 1 || cache.slots > 64 || ggml_backend_is_cpu(sched->backends[backend_id]) ||
            source->ne[2] < cache.slots || source->nb[2] == 0) {
        return false;
    }

    const size_t tensor_overhead = ggml_tensor_overhead();
    if (route_capacity > (SIZE_MAX - 128*1024)/tensor_overhead) return false;
    ggml_init_params params = {
        /*.mem_size   =*/ 128*1024 + route_capacity*tensor_overhead,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    cache.ctx = ggml_init(params);
    if (!cache.ctx) {
        fprintf(stderr, "ggml active-expert cache: unable to create tensor context for backend %d\n", backend_id);
        return false;
    }

    // Pre-allocate one top-k remap tensor per possible routed graph value so a
    // long server session never grows a per-token ggml context and models with
    // more than 64 MoE layers do not hit an unrelated route-count ceiling.
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(sched->backends[backend_id])) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const int device = ggml_backend_cuda_get_device(sched->backends[backend_id]);
        ggml_backend_cuda_get_device_memory(device, &free_bytes, &total_bytes);
        GGML_UNUSED(total_bytes);
        const uint64_t reserve = sched->active_expert_cache_reserve_bytes;
        if (free_bytes <= reserve || cache.ids_bytes > uint64_t(free_bytes) - reserve) {
            ggml_active_expert_cache_release(cache);
            cache.attempted = false;
            return false;
        }
    }
#endif
    cache.ids_buffer = ggml_backend_buft_alloc_buffer(
            sched->bufts[backend_id], size_t(cache.ids_bytes));
    if (!cache.ids_buffer) {
        fprintf(stderr, "ggml active-expert cache: unable to allocate ID staging on backend %d\n", backend_id);
        ggml_active_expert_cache_release(cache);
        return false;
    }
    cache.allocated_bytes = ggml_backend_buffer_get_size(cache.ids_buffer);
    if (cache.capacity_bytes != 0 && cache.allocated_bytes > cache.capacity_bytes) {
        ggml_active_expert_cache_release(cache);
        return false;
    }
    auto * ids_base = (char *) ggml_backend_buffer_get_base(cache.ids_buffer);
    cache.ids_tensors.resize(route_capacity, nullptr);
    cache.ids_host.resize(route_capacity);
    for (size_t i = 0; i < route_capacity; ++i) {
        cache.ids_tensors[i] = ggml_new_tensor_2d(cache.ctx, GGML_TYPE_I32, 64, 1);
        ggml_backend_tensor_alloc(
                cache.ids_buffer, cache.ids_tensors[i],
                ids_base + i*GGML_ACTIVE_EXPERT_CACHE_ID_STRIDE);
    }

    cache.entries.resize(cache.slots);
    ggml_backend_t event_backend = sched->backends[backend_id];
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(event_backend)) {
        cache.transfer_backend = ggml_backend_cuda_init(
                ggml_backend_cuda_get_device(event_backend), nullptr, cache.ctx);
        if (cache.transfer_backend) event_backend = cache.transfer_backend;
    }
#endif
    cache.lease_event = ggml_backend_event_new(event_backend);
    if (cache.transfer_backend && !cache.lease_event) {
        ggml_backend_free(cache.transfer_backend);
        cache.transfer_backend = nullptr;
        cache.lease_event = ggml_backend_event_new(sched->backends[backend_id]);
    }
    cache.compute_event = ggml_backend_event_new(sched->backends[backend_id]);
    if (cache.transfer_backend && !cache.compute_event) {
        fprintf(stderr, "ggml active-expert cache: unable to create compute-readiness event on backend %d\n", backend_id);
        ggml_active_expert_cache_release(cache);
        return false;
    }
    cache.ready = true;
    fprintf(stderr, "ggml active-expert cache: backend %d enabled with %d slots and %zu route remaps (capacity %.1f MiB, reserve %.1f MiB)\n",
            backend_id, cache.slots,
            cache.ids_tensors.size(),
            cache.capacity_bytes/1024.0/1024.0,
            sched->active_expert_cache_reserve_bytes/1024.0/1024.0);
    return true;
}

static bool ggml_active_expert_cache_init_component(
        ggml_backend_sched_t sched,
        ggml_active_expert_cache_device & cache,
        int component_index,
        const ggml_tensor * source) {
    auto & component = cache.components[component_index];
    const auto matching_layout = std::find_if(
            component.layouts.begin(), component.layouts.end(),
            [&](const ggml_tensor * layout) {
                return ggml_active_expert_cache_same_layout(layout, source);
            });
    ggml_tensor * layout = matching_layout == component.layouts.end() ? nullptr : *matching_layout;
    if (!layout) {
        layout = ggml_dup_tensor_layout(cache.ctx, source);
        layout->ne[2] = cache.slots;
        layout->ne[3] = 1;
        layout->nb[3] = layout->nb[2]*size_t(cache.slots);
        component.layouts.push_back(layout);
    }

    // A model may change quantization or geometry between layers. Reserve for
    // the largest layout seen in this graph, then alias each layout over the
    // same bounded component allocation.
    const uint64_t max_expert_bytes = sched->active_expert_component_max_bytes[component_index];
    if (max_expert_bytes == 0 || max_expert_bytes > SIZE_MAX/size_t(cache.slots)) return false;
    const size_t buffer_size = size_t(max_expert_bytes)*size_t(cache.slots);
#ifdef GGML_USE_CUDA
    if (!component.buffer && ggml_backend_is_cuda(sched->backends[cache.backend_id])) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const int device = ggml_backend_cuda_get_device(sched->backends[cache.backend_id]);
        ggml_backend_cuda_get_device_memory(device, &free_bytes, &total_bytes);
        GGML_UNUSED(total_bytes);
        const uint64_t reserve = sched->active_expert_cache_reserve_bytes;
        if (free_bytes <= reserve || buffer_size > uint64_t(free_bytes) - reserve) {
            return false;
        }
    }
#endif
    if (!component.buffer && cache.capacity_bytes != 0 &&
            (buffer_size > cache.capacity_bytes ||
             cache.allocated_bytes > cache.capacity_bytes - buffer_size)) {
        return false;
    }
    if (!component.buffer) {
        component.buffer = ggml_backend_buft_alloc_buffer(sched->bufts[cache.backend_id], buffer_size);
        if (!component.buffer) return false;
        component.buffer_bytes = ggml_backend_buffer_get_size(component.buffer);
        ggml_backend_buffer_clear(component.buffer, 0);
        cache.allocated_bytes += component.buffer_bytes;
    }
    if (ggml_nbytes(layout) > component.buffer_bytes) return false;
    if (!layout->buffer) {
        ggml_backend_tensor_alloc(component.buffer, layout, ggml_backend_buffer_get_base(component.buffer));
    }
    if (component.tensor != layout) {
        // Existing events and ready bits refer to the previous tensor metadata,
        // even though the backing allocation is shared.
        ggml_active_expert_cache_drain_leases(sched, cache);
        if (cache.compute_event_recorded) {
            ggml_backend_event_synchronize(cache.compute_event);
            cache.compute_event_recorded = false;
        }
        const uint8_t component_mask = uint8_t(1u << component_index);
        for (auto & entry : cache.entries) entry.ready_mask &= ~component_mask;
    }
    component.tensor = layout;
    component.expert_bytes = layout->nb[2];
    return true;
}

static int ggml_active_expert_cache_slot(
        ggml_active_expert_cache_device & cache,
        int layer,
        int expert,
        const std::vector<int> & protected_slots,
        bool * hit,
        bool * evicted,
        uint64_t * eviction_cost) {
    *evicted = false;
    *eviction_cost = 0;
    for (int slot = 0; slot < cache.slots; ++slot) {
        auto & entry = cache.entries[slot];
        if (entry.layer == layer && entry.expert == expert) {
            entry.last_used = ++cache.clock;
            *hit = true;
            return slot;
        }
    }

    auto entry_cost = [&](int slot) {
        uint64_t result = 0;
        const auto & entry = cache.entries[slot];
        for (int component = 0; component < GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT; ++component) {
            if ((entry.ready_mask & (1u << component)) != 0) {
                result += cache.components[component].expert_bytes;
            }
        }
        return result;
    };

    int selected = -1;
    for (int slot = 0; slot < cache.slots; ++slot) {
        if (std::find(protected_slots.begin(), protected_slots.end(), slot) != protected_slots.end()) {
            continue;
        }
        if (cache.entries[slot].layer < 0) {
            selected = slot;
            break;
        }
        if (selected < 0 || cache.entries[slot].last_used < cache.entries[selected].last_used) {
            selected = slot;
        }
    }
    GGML_ASSERT(selected >= 0 && "active expert route requires more unique slots than its cache provides");

    if (cache.entries[selected].layer >= 0) {
        // Preserve deterministic LRU behavior, but among entries with nearly
        // equal age prefer the cheaper demotion. This avoids discarding a
        // fully uploaded expert when a similarly old partial entry exists.
        const uint64_t oldest = cache.entries[selected].last_used;
        const uint64_t age_window = uint64_t(std::max(1, cache.slots/4));
        uint64_t selected_cost = entry_cost(selected);
        for (int slot = 0; slot < cache.slots; ++slot) {
            if (std::find(protected_slots.begin(), protected_slots.end(), slot) != protected_slots.end()) {
                continue;
            }
            const auto & candidate = cache.entries[slot];
            if (candidate.layer < 0 || candidate.last_used < oldest ||
                    candidate.last_used - oldest > age_window) {
                continue;
            }
            const uint64_t candidate_cost = entry_cost(slot);
            if (candidate_cost < selected_cost ||
                    (candidate_cost == selected_cost && candidate.last_used < cache.entries[selected].last_used)) {
                selected = slot;
                selected_cost = candidate_cost;
            }
        }
    }

    auto & entry = cache.entries[selected];
    *evicted = entry.layer >= 0;
    if (*evicted) {
        *eviction_cost = entry_cost(selected);
    }
    entry.layer = layer;
    entry.expert = expert;
    entry.last_used = ++cache.clock;
    entry.ready_mask = 0;
    *hit = false;
    return selected;
}

static ggml_active_expert_cache_route * ggml_active_expert_cache_find_route(
        ggml_backend_sched_t sched,
        ggml_tensor * ids_source,
        int backend_id) {
    for (auto & route : sched->active_expert_routes) {
        if (route.ids_source == ids_source && route.backend_id == backend_id) {
            return &route;
        }
    }
    return nullptr;
}

static ggml_tensor * ggml_active_expert_cache_source_ids(
        ggml_backend_sched_t sched,
        ggml_tensor * node,
        int ids_index) {
    for (const auto & redirect : sched->active_expert_redirects) {
        if (redirect.node == node && redirect.ids_index == ids_index) {
            return redirect.source_ids;
        }
    }
    return node->src[ids_index];
}

static void ggml_active_expert_cache_redirect_ids(
        ggml_backend_sched_t sched,
        ggml_tensor * node,
        int ids_index,
        ggml_tensor * source_ids,
        ggml_tensor * mapped_ids) {
    for (const auto & redirect : sched->active_expert_redirects) {
        if (redirect.node == node && redirect.ids_index == ids_index) {
            node->src[ids_index] = mapped_ids;
            return;
        }
    }
    sched->active_expert_redirects.push_back({
        /*.input_cpy  =*/ nullptr,
        /*.input_data =*/ nullptr,
        /*.input_buffer =*/ nullptr,
        /*.input_ne2  =*/ 0,
        /*.input_nb3  =*/ 0,
        /*.node       =*/ node,
        /*.ids_index  =*/ ids_index,
        /*.node_ids   =*/ node->src[ids_index],
        /*.source_ids =*/ source_ids,
    });
    node->src[ids_index] = mapped_ids;
}

static void ggml_active_expert_cache_redirect_input(
        ggml_backend_sched_t sched,
        ggml_tensor * input_cpy,
        const ggml_tensor * logical_source,
        void * cache_data,
        ggml_backend_buffer_t cache_buffer) {
    for (const auto & redirect : sched->active_expert_redirects) {
        if (redirect.input_cpy == input_cpy) {
            input_cpy->data = cache_data;
            input_cpy->buffer = cache_buffer;
            input_cpy->ne[2] = logical_source->ne[2];
            input_cpy->nb[3] = logical_source->nb[3];
            return;
        }
    }
    sched->active_expert_redirects.push_back({
        /*.input_cpy  =*/ input_cpy,
        /*.input_data =*/ input_cpy->data,
        /*.input_buffer =*/ input_cpy->buffer,
        /*.input_ne2  =*/ input_cpy->ne[2],
        /*.input_nb3  =*/ input_cpy->nb[3],
        /*.node       =*/ nullptr,
        /*.ids_index  =*/ -1,
        /*.node_ids   =*/ nullptr,
        /*.source_ids =*/ nullptr,
    });
    input_cpy->data = cache_data;
    input_cpy->buffer = cache_buffer;
    input_cpy->ne[2] = logical_source->ne[2];
    input_cpy->nb[3] = logical_source->nb[3];
}

static void ggml_active_expert_cache_reset_pass(ggml_backend_sched_t sched) {
    for (const auto & redirect : sched->active_expert_redirects) {
        if (redirect.input_cpy) {
            redirect.input_cpy->data = redirect.input_data;
            redirect.input_cpy->buffer = redirect.input_buffer;
            redirect.input_cpy->ne[2] = redirect.input_ne2;
            redirect.input_cpy->nb[3] = redirect.input_nb3;
        }
        if (redirect.node) {
            redirect.node->src[redirect.ids_index] = redirect.node_ids;
        }
    }
    sched->active_expert_redirects.clear();
    sched->active_expert_routes.clear();
    for (auto & cache : sched->active_expert_caches) {
        ggml_active_expert_cache_drain_leases(sched, cache);
        cache.ids_next = 0;
        cache.observed_this_pass.clear();
        cache.used_this_pass = false;
    }
    for (auto & device : sched->expert_prefill_devices) {
        if (!device.ready) continue;
        for (auto & lane : device.lanes) {
            ggml_expert_prefill_drain_lane(sched, lane, lane.layer);
        }
        device.used_this_split.fill(false);
    }
}

static int64_t ggml_active_expert_cache_replace_failure_step() {
    const char * value = getenv("ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_STEPS");
    if (value == nullptr || value[0] == '\0') return -1;
    char * end = nullptr;
    const long long parsed = strtoll(value, &end, 10);
    return end != value && *end == '\0' && parsed >= 0 ? parsed : -1;
}

static int64_t ggml_active_expert_cache_replace_failure_copy() {
    const char * value = getenv("ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_COPIES");
    if (value == nullptr || value[0] == '\0') return -1;
    char * end = nullptr;
    const long long parsed = strtoll(value, &end, 10);
    return end != value && *end == '\0' && parsed >= 0 ? parsed : -1;
}

static int64_t ggml_active_expert_cache_replace_failure_device() {
    const char * value = getenv("ESE_EXPERT_CACHE_REPLACE_FAIL_AFTER_DEVICES");
    if (value == nullptr || value[0] == '\0') return -1;
    char * end = nullptr;
    const long long parsed = strtoll(value, &end, 10);
    return end != value && *end == '\0' && parsed >= 0 ? parsed : -1;
}

static bool ggml_active_expert_cache_same_layout(
        const ggml_tensor * left,
        const ggml_tensor * right) {
    return left && right && left->type == right->type &&
            left->ne[0] == right->ne[0] && left->ne[1] == right->ne[1] &&
            left->nb[0] == right->nb[0] && left->nb[1] == right->nb[1] &&
            left->nb[2] == right->nb[2];
}

static ggml_active_expert_cache_layout ggml_active_expert_cache_describe_layout(
        const ggml_tensor * source) {
    return {
        source->type,
        source->ne[0],
        source->ne[1],
        source->ne[2],
        source->nb[0],
        source->nb[1],
        source->nb[2],
    };
}

static bool ggml_active_expert_cache_same_layout(
        const ggml_active_expert_cache_layout & left,
        const ggml_active_expert_cache_layout & right) {
    return left.type == right.type && left.ne0 == right.ne0 && left.ne1 == right.ne1 &&
            left.nb0 == right.nb0 && left.nb1 == right.nb1 && left.nb2 == right.nb2;
}

static bool ggml_active_expert_cache_materialize_layout(
        const ggml_active_expert_cache_layout & descriptor,
        ggml_tensor * layout) {
    if (!layout || int(descriptor.type) < 0 || descriptor.type >= GGML_TYPE_COUNT ||
            descriptor.ne0 < 1 || descriptor.ne1 < 1 || descriptor.logical_experts < 2 ||
            descriptor.nb0 == 0 || descriptor.nb1 == 0 || descriptor.nb2 == 0 ||
            uint64_t(descriptor.logical_experts) > SIZE_MAX/descriptor.nb2) {
        return false;
    }
    *layout = {};
    layout->type = descriptor.type;
    layout->ne[0] = descriptor.ne0;
    layout->ne[1] = descriptor.ne1;
    layout->ne[2] = descriptor.logical_experts;
    layout->ne[3] = 1;
    layout->nb[0] = descriptor.nb0;
    layout->nb[1] = descriptor.nb1;
    layout->nb[2] = descriptor.nb2;
    layout->nb[3] = descriptor.nb2*size_t(descriptor.logical_experts);
    return true;
}

static bool ggml_active_expert_cache_target_is_realized(
        ggml_backend_sched_t sched,
        uint64_t bytes_per_device) {
    if (bytes_per_device == 0) {
        if (sched->active_expert_cache_slots != 0) return false;
        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            if (ggml_backend_is_cpu(sched->backends[backend_id])) continue;
            const auto & cache = sched->active_expert_caches[backend_id];
            if (cache.ready || cache.allocated_bytes != 0) return false;
        }
        return true;
    }
    if (sched->n_backends <= 1 || sched->active_expert_cache_slots == 0) return false;
    for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
        if (ggml_backend_is_cpu(sched->backends[backend_id])) continue;
        const auto & cache = sched->active_expert_caches[backend_id];
        if (!cache.ready || cache.capacity_bytes != bytes_per_device ||
                cache.allocated_bytes > bytes_per_device ||
                cache.ids_tensors.size() < sched->active_expert_cache_route_capacity) {
            return false;
        }
        for (size_t component = 0;
                component < sched->active_expert_cache_layout_catalog.size(); ++component) {
            for (const auto & descriptor : sched->active_expert_cache_layout_catalog[component]) {
                ggml_tensor source = {};
                if (!ggml_active_expert_cache_materialize_layout(descriptor, &source) ||
                        std::none_of(
                            cache.components[component].layouts.begin(),
                            cache.components[component].layouts.end(),
                            [&](const ggml_tensor * layout) {
                                return ggml_active_expert_cache_same_layout(layout, &source);
                            })) {
                    return false;
                }
            }
        }
    }
    return true;
}

static bool ggml_backend_sched_replace_expert_cache_impl(
        ggml_backend_sched_t sched,
        uint64_t bytes_per_device,
        uint64_t reserve_bytes_per_device,
        uint32_t minimum_observations) {
    if (!sched || (bytes_per_device > 0 && sched->n_backends <= 1)) return false;
    minimum_observations = std::max<uint32_t>(1, minimum_observations);
    if (sched->active_expert_cache_capacity_bytes == bytes_per_device &&
            sched->active_expert_cache_reserve_bytes == reserve_bytes_per_device &&
            sched->active_expert_cache_min_observations == minimum_observations &&
            ggml_active_expert_cache_target_is_realized(sched, bytes_per_device)) {
        return true;
    }

    // Cache redirects and borrowed host leases belong to the completed graph
    // pass. Settle them before any replacement allocation or residency copy.
    ggml_backend_sched_synchronize(sched);
    ggml_active_expert_cache_reset_pass(sched);
    ggml_backend_sched_synchronize(sched);

    const uint64_t previous_capacity = sched->active_expert_cache_capacity_bytes;
    const uint64_t previous_reserve = sched->active_expert_cache_reserve_bytes;
    const uint32_t previous_minimum = sched->active_expert_cache_min_observations;
    const int previous_slots = sched->active_expert_cache_slots;
    sched->active_expert_cache_capacity_bytes = bytes_per_device;
    sched->active_expert_cache_reserve_bytes = reserve_bytes_per_device;
    sched->active_expert_cache_min_observations = minimum_observations;
    sched->active_expert_cache_slots = bytes_per_device > 0 ? 64 : 0;

    std::array<ggml_active_expert_cache_device, GGML_SCHED_MAX_BACKENDS> prepared;
    const int64_t failure_step = ggml_active_expert_cache_replace_failure_step();
    const int64_t failure_copy = ggml_active_expert_cache_replace_failure_copy();
    const int64_t failure_device = ggml_active_expert_cache_replace_failure_device();
    int64_t completed_steps = 0;
    int64_t completed_copies = 0;
    int prepared_devices = 0;
    int required_devices = 0;
    bool success = true;
    auto inject_failure = [&]() {
        if (completed_steps == failure_step) return true;
        ++completed_steps;
        return false;
    };

    if (bytes_per_device > 0) {
        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            if (ggml_backend_is_cpu(sched->backends[backend_id])) continue;
            ++required_devices;
            auto & current_slot = sched->active_expert_caches[backend_id];

            // A growing replacement needs the model's logical expert count,
            // not the compact slot count stored in the old cache tensor.
            ggml_tensor logical_layout = {};
            const ggml_tensor * logical_source = nullptr;
            for (const auto & component : sched->active_expert_cache_layout_catalog) {
                if (!component.empty() && ggml_active_expert_cache_materialize_layout(
                            component.front(), &logical_layout)) {
                    logical_source = &logical_layout;
                    break;
                }
            }
            if (!logical_source && current_slot.ready && current_slot.logical_experts > 1) {
                for (int component = 0; component < GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT; ++component) {
                    const ggml_tensor * active = current_slot.components[size_t(component)].tensor;
                    if (!active) continue;
                    if (active->nb[2] == 0 ||
                            uint64_t(current_slot.logical_experts) > SIZE_MAX/active->nb[2]) {
                        success = false;
                        break;
                    }
                    logical_layout = *active;
                    logical_layout.ne[2] = current_slot.logical_experts;
                    logical_layout.nb[3] =
                            active->nb[2]*size_t(current_slot.logical_experts);
                    logical_source = &logical_layout;
                    break;
                }
            }
            if (!success) break;
            if (!logical_source) {
                success = false;
                break;
            }

            // Temporarily detach the current owner so the existing allocation
            // remains untouched while the normal checked initializer builds a
            // second cache in the scheduler slot.
            ggml_active_expert_cache_device current;
            std::swap(current, current_slot);
            auto & building = sched->active_expert_caches[backend_id];
            bool device_success = false;
            try {
                device_success = ggml_active_expert_cache_init(sched, backend_id, logical_source);
                device_success &= building.ready && building.capacity_bytes == bytes_per_device;
                if (device_success && inject_failure()) device_success = false;

                // Allocate every layout used by this backend before publication so
                // a later layer cannot reveal a deferred capacity failure.
                for (int component = 0;
                        device_success && component < GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT;
                        ++component) {
                    for (const auto & descriptor :
                            sched->active_expert_cache_layout_catalog[size_t(component)]) {
                        ggml_tensor source = {};
                        if (!ggml_active_expert_cache_materialize_layout(descriptor, &source) ||
                                !ggml_active_expert_cache_init_component(
                                    sched, building, component, &source) ||
                                building.allocated_bytes > bytes_per_device || inject_failure()) {
                            device_success = false;
                            break;
                        }
                    }
                    const ggml_tensor * active = current.components[size_t(component)].tensor;
                    if (device_success && active &&
                            !ggml_active_expert_cache_init_component(
                                    sched, building, component, active)) {
                        device_success = false;
                    }
                    if (device_success && active && building.allocated_bytes > bytes_per_device) {
                        device_success = false;
                    }
                }

                if (device_success) {
                    building.clock = current.clock;
                    building.observations = current.observations;
                    building.admission_thresholds = current.admission_thresholds;
                    building.last_observed_route = current.last_observed_route;
                    building.previous_route = current.previous_route;
                    building.route_clock = current.route_clock;

                    std::vector<int> residents;
                    residents.reserve(current.entries.size());
                    for (int slot = 0; slot < int(current.entries.size()); ++slot) {
                        if (current.entries[size_t(slot)].layer >= 0 &&
                                current.entries[size_t(slot)].expert >= 0) {
                            residents.push_back(slot);
                        }
                    }
                    std::stable_sort(residents.begin(), residents.end(), [&](int left, int right) {
                        return current.entries[size_t(left)].last_used >
                                current.entries[size_t(right)].last_used;
                    });
                    if (residents.size() > size_t(building.slots)) residents.resize(size_t(building.slots));

                    for (size_t destination_slot = 0;
                            device_success && destination_slot < residents.size();
                            ++destination_slot) {
                        const int source_slot = residents[destination_slot];
                        const auto & source_entry = current.entries[size_t(source_slot)];
                        auto & destination_entry = building.entries[destination_slot];
                        destination_entry = source_entry;
                        destination_entry.ready_mask = 0;
                        for (int component = 0;
                                component < GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT;
                                ++component) {
                            const uint8_t mask = uint8_t(1u << component);
                            if ((source_entry.ready_mask & mask) == 0) continue;
                            const auto & source_component = current.components[size_t(component)];
                            const auto & destination_component = building.components[size_t(component)];
                            if (!source_component.tensor || !destination_component.tensor ||
                                    source_component.expert_bytes != destination_component.expert_bytes ||
                                    !ggml_active_expert_cache_same_layout(
                                        source_component.tensor, destination_component.tensor) ||
                                    inject_failure() || completed_copies == failure_copy) {
                                device_success = false;
                                break;
                            }
                            ggml_tensor source_view = *source_component.tensor;
                            ggml_tensor destination_view = *destination_component.tensor;
                            source_view.ne[2] = destination_view.ne[2] = 1;
                            source_view.nb[3] = source_view.nb[2];
                            destination_view.nb[3] = destination_view.nb[2];
                            source_view.data = (uint8_t *) source_component.tensor->data +
                                    size_t(source_slot)*source_component.expert_bytes;
                            destination_view.data = (uint8_t *) destination_component.tensor->data +
                                    destination_slot*destination_component.expert_bytes;
                            ggml_backend_tensor_copy(&source_view, &destination_view);
                            ++completed_copies;
                            destination_entry.ready_mask |= mask;
                        }
                    }
                }
            } catch (const std::exception & exception) {
                fprintf(stderr,
                        "ggml active-expert cache: replacement preparation exception on backend %d: %s\n",
                        backend_id, exception.what());
                device_success = false;
            } catch (...) {
                fprintf(stderr,
                        "ggml active-expert cache: replacement preparation exception on backend %d\n",
                        backend_id);
                device_success = false;
            }

            if (device_success && prepared_devices == failure_device) {
                device_success = false;
            }
            if (device_success) {
                std::swap(prepared[size_t(backend_id)], building);
                ++prepared_devices;
            } else {
                ggml_active_expert_cache_drain_leases(sched, building);
                ggml_active_expert_cache_release(building);
            }
            std::swap(current, sched->active_expert_caches[backend_id]);
            if (!device_success) {
                success = false;
                break;
            }
        }
        if (required_devices == 0 || prepared_devices != required_devices) success = false;
    }

    if (!success) {
        for (auto & cache : prepared) {
            ggml_active_expert_cache_drain_leases(sched, cache);
            ggml_active_expert_cache_release(cache);
        }
        sched->active_expert_cache_capacity_bytes = previous_capacity;
        sched->active_expert_cache_reserve_bytes = previous_reserve;
        sched->active_expert_cache_min_observations = previous_minimum;
        sched->active_expert_cache_slots = previous_slots;
        fprintf(stderr, "ggml active-expert cache: replacement preparation failed; prior cache retained\n");
        return false;
    }

    // All fallible work is complete. Swap every device first, publish the
    // target policy, then retire the detached old allocations.
    ggml_backend_sched_synchronize(sched);
    for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
        std::swap(sched->active_expert_caches[backend_id], prepared[size_t(backend_id)]);
    }
    sched->active_expert_cache_capacity_bytes = bytes_per_device;
    sched->active_expert_cache_reserve_bytes = reserve_bytes_per_device;
    sched->active_expert_cache_min_observations = minimum_observations;
    sched->active_expert_cache_slots = bytes_per_device > 0 ? 64 : 0;
    for (auto & cache : prepared) {
        ggml_active_expert_cache_drain_leases(sched, cache);
        ggml_active_expert_cache_release(cache);
    }
    fprintf(stderr,
            "ggml active-expert cache: committed prepared replacement at %.1f MiB per device (%d prepared devices)\n",
            bytes_per_device/1024.0/1024.0, prepared_devices);
    return true;
}

bool ggml_backend_sched_replace_expert_cache(
        ggml_backend_sched_t sched,
        uint64_t bytes_per_device,
        uint64_t reserve_bytes_per_device,
        uint32_t minimum_observations) {
    try {
        return ggml_backend_sched_replace_expert_cache_impl(
                sched, bytes_per_device, reserve_bytes_per_device, minimum_observations);
    } catch (const std::exception & exception) {
        fprintf(stderr, "ggml active-expert cache: replacement exception: %s\n", exception.what());
    } catch (...) {
        fprintf(stderr, "ggml active-expert cache: replacement exception\n");
    }
    return false;
}

static bool ggml_active_expert_cache_prepare_route(
        ggml_backend_sched_t sched,
        ggml_backend_t ids_backend,
        ggml_backend_t split_backend,
        int split_backend_id,
        ggml_tensor * input,
        ggml_tensor * ids_source,
        int layer,
        bool full_tensor_fallback_available,
        ggml_active_expert_cache_route ** out_route) {
    const int64_t n_ids = ids_source->ne[0];
    if (ids_source->type != GGML_TYPE_I32 || n_ids < 1 || n_ids > 64 || ids_source->ne[1] != 1 ||
            input->ne[2] <= 1 || input->nb[2] == 0) {
        return false;
    }

    if (auto * route = ggml_active_expert_cache_find_route(sched, ids_source, split_backend_id)) {
        *out_route = route;
        return true;
    }

    if (!ggml_active_expert_cache_init(sched, split_backend_id, input)) {
        return false;
    }
    auto & cache = sched->active_expert_caches[split_backend_id];
    if (cache.ids_next >= cache.ids_tensors.size()) {
        return false;
    }

    const auto route_readback_start = std::chrono::steady_clock::now();
    std::vector<int32_t> expert_ids(size_t(n_ids), -1);
    if (ids_source->buffer && ggml_backend_buffer_is_host(ids_source->buffer)) {
        ggml_backend_tensor_get(ids_source, expert_ids.data(), 0, expert_ids.size()*sizeof(expert_ids[0]));
    } else {
        ggml_backend_event_t route_event = ggml_backend_event_new(ids_backend);
        if (!route_event) {
            if (sched->expert_lease_required) {
                GGML_ABORT("required expert route readback has no event-capable backend\n");
            }
            return false;
        }
        ggml_backend_tensor_get_async(ids_backend, ids_source, expert_ids.data(), 0, expert_ids.size()*sizeof(expert_ids[0]));
        ggml_backend_event_record(route_event);
        ggml_backend_event_synchronize(route_event);
        ggml_backend_event_free(route_event);
    }
    auto & layer_stats = sched->active_expert_cache_layer_stats[layer];
    layer_stats.route_readback_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - route_readback_start).count());

    // Thresholded top-k routing pads the ID tensor with negative sentinels.
    // Preserve those positions in the remap tensor, but size the residency
    // requirement from the distinct valid experts rather than its padded
    // width (for example, top-2 represented as [8, 1]).
    std::unordered_set<int32_t> active_experts;
    for (const int32_t expert : expert_ids) {
        if (expert < 0) continue;
        if (expert >= input->ne[2]) return false;
        active_experts.insert(expert);
    }
    if (getenv("LLAMA_EXPERT_GPU_CACHE_DEBUG")) {
        fprintf(stderr, "ggml active-expert cache: layer %d route IDs", layer);
        for (const int32_t expert : expert_ids) fprintf(stderr, " %d", expert);
        fprintf(stderr, " (%zu active)\n", active_experts.size());
    }
    if (active_experts.empty() || active_experts.size() > size_t(cache.slots)) {
        return false;
    }
    ++layer_stats.routes;
    layer_stats.route_positions += active_experts.size();
    if (ggml_ese_route_get_role(ids_source) == GGML_ESE_ROUTE_GPU) {
        layer_stats.gpu_route_positions += active_experts.size();
    }

    bool admission_ready = true;
    const uint64_t route_clock = ++cache.route_clock;
    const auto previous = cache.previous_route.find(layer);
    std::unordered_set<int32_t> predicted_experts;
    if (previous != cache.previous_route.end()) {
        for (size_t i = 0; i < expert_ids.size() && i < previous->second.size(); ++i) {
            if (expert_ids[i] == previous->second[i]) {
                ++sched->active_expert_cache_route_prediction_matches;
                predicted_experts.insert(expert_ids[i]);
            }
        }
    }
    cache.previous_route[layer] = expert_ids;
    for (const int32_t expert : expert_ids) {
        if (expert < 0) continue;
        const uint64_t observation_key = (uint64_t(uint32_t(layer)) << 32) | uint32_t(expert);
        if (cache.observed_this_pass.insert(observation_key).second) {
            ++sched->active_expert_cache_route_observations;
            auto & count = cache.observations[observation_key];
            if (count != UINT32_MAX) ++count;
            const auto last = cache.last_observed_route.find(observation_key);
            uint64_t reuse_distance = UINT64_MAX;
            if (last != cache.last_observed_route.end()) {
                reuse_distance = route_clock - last->second;
                sched->active_expert_cache_reuse_distance_sum += reuse_distance;
            }
            cache.last_observed_route[observation_key] = route_clock;
            const bool predicted = predicted_experts.count(expert) != 0;
            const bool short_reuse = reuse_distance <= uint64_t(std::max(2, cache.slots));
            const uint64_t minimum = sched->active_expert_cache_min_observations;
            constexpr uint64_t load_cost_step = 16ULL*1024ULL*1024ULL;
            const uint64_t load_penalty = std::min<uint64_t>(3, uint64_t(input->nb[2])/load_cost_step);
            uint64_t required_observations = minimum + load_penalty;
            if (predicted || short_reuse) {
                required_observations = minimum + (load_penalty + 1)/2;
            } else {
                required_observations = std::max(required_observations, minimum*2);
            }
            if (predicted && count >= required_observations) {
                ++sched->active_expert_cache_prediction_admissions;
            }
            cache.admission_thresholds[observation_key] = required_observations;
        }
        const auto threshold = cache.admission_thresholds.find(observation_key);
        if (threshold == cache.admission_thresholds.end() ||
                uint64_t(cache.observations[observation_key]) < threshold->second) {
            admission_ready = false;
        }
    }
    const bool calibrated_gpu_partition = ggml_ese_route_get_role(ids_source) == GGML_ESE_ROUTE_GPU;
    // Admission hysteresis may defer a route only while the scheduler retained
    // the original full expert tensor.  A compact cache-backed graph has no
    // correct CPU fallback, so its first route must be admitted immediately.
    if (!admission_ready && !sched->expert_lease_required && !calibrated_gpu_partition &&
            sched->active_expert_cache_capacity_bytes == 0 && full_tensor_fallback_available) {
        ++sched->active_expert_cache_rejected_admissions;
        return false;
    }

    ggml_active_expert_cache_route route;
    route.ids_source = ids_source;
    route.cache = &cache;
    route.backend_id = split_backend_id;
    route.layer = layer;
    const size_t route_index = cache.ids_next++;
    route.mapped_ids = cache.ids_tensors[route_index];
    route.mapped_ids->ne[0] = n_ids;
    route.mapped_ids->ne[1] = 1;
    route.experts.reserve(expert_ids.size());
    route.slots.reserve(expert_ids.size());

    auto & mapped_ids = cache.ids_host[route_index];
    mapped_ids.fill(-1);
    for (size_t i = 0; i < expert_ids.size(); ++i) {
        const int32_t expert = expert_ids[i];
        if (expert < 0) continue;
        bool hit = false;
        bool evicted = false;
        uint64_t eviction_cost = 0;
        const int slot = ggml_active_expert_cache_slot(
                cache, layer, expert, route.slots, &hit, &evicted, &eviction_cost);
        if (hit) {
            ++sched->active_expert_cache_hits;
            ++layer_stats.hits;
        } else {
            ++sched->active_expert_cache_misses;
            ++sched->active_expert_cache_admissions;
            ++layer_stats.misses;
            if (evicted) {
                ++sched->active_expert_cache_evictions;
                sched->active_expert_cache_eviction_cost_bytes += eviction_cost;
            }
        }
        route.experts.push_back(expert);
        route.slots.push_back(slot);
        mapped_ids[i] = slot;
    }

    ggml_backend_tensor_set_async(
            split_backend, route.mapped_ids, mapped_ids.data(), 0,
            size_t(n_ids)*sizeof(mapped_ids[0]));
    sched->active_expert_routes.push_back(std::move(route));
    *out_route = &sched->active_expert_routes.back();
    return true;
}

static bool ggml_active_expert_cache_stage(
        ggml_backend_sched_t sched,
        ggml_backend_t split_backend,
        int split_backend_id,
        ggml_tensor * input,
        ggml_tensor * input_cpy,
        ggml_tensor * node,
        int ids_index,
        ggml_tensor * ids_source) {
    if (sched->active_expert_cache_slots < 1 || input->ne[2] <= 1 ||
            !ggml_backend_buffer_is_host(input->buffer)) {
        return false;
    }

    int layer = -1;
    const int component_index = ggml_active_expert_cache_component_from_name(input, &layer);
    if (component_index < 0) {
        return false;
    }

    if (!ggml_active_expert_cache_init(sched, split_backend_id, input)) {
        ++sched->active_expert_cache_rejected_admissions;
        return false;
    }
    auto & cache = sched->active_expert_caches[split_backend_id];
    if (!ggml_active_expert_cache_init_component(sched, cache, component_index, input)) {
        ++sched->active_expert_cache_rejected_admissions;
        return false;
    }

    auto ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids_source);
    ggml_active_expert_cache_route * route = nullptr;
    const bool full_tensor_fallback_available = input_cpy->ne[2] == input->ne[2];
    if (!ggml_active_expert_cache_prepare_route(
            sched, ids_backend, split_backend, split_backend_id, input, ids_source, layer,
            full_tensor_fallback_available, &route)) {
        return false;
    }

    auto & component = route->cache->components[component_index];

    const uint8_t component_mask = uint8_t(1u << component_index);
    const bool needs_upload = std::any_of(
            route->slots.begin(), route->slots.end(),
            [&](int slot) {
                return (route->cache->entries[slot].ready_mask & component_mask) == 0;
            });
    if (needs_upload && !route->cache->lease_handles.empty()) {
        // Leases only have to remain pinned until their asynchronous H2D copy
        // completes.  Drain the preceding upload batch before borrowing host
        // storage for another component; retaining every lease for the whole
        // graph pass can exhaust an otherwise correctly bounded RAM cache.
        ggml_active_expert_cache_drain_leases(sched, *route->cache);
    }
    bool staged_from_lease = false;
    bool uploaded = false;
    ggml_backend_t upload_backend = route->cache->transfer_backend ? route->cache->transfer_backend : split_backend;
    if (route->cache->compute_event_recorded) {
        // A cache slot may be reused only after the previous compute stream has
        // consumed it. Queue the dependency on the transfer stream instead of
        // globally synchronizing either backend.
        ggml_backend_event_wait(upload_backend, route->cache->compute_event);
        route->cache->compute_event_recorded = false;
    }
    for (size_t i = 0; i < route->experts.size(); ++i) {
        auto & entry = route->cache->entries[route->slots[i]];
        if ((entry.ready_mask & component_mask) != 0) {
            continue;
        }
        const size_t source_offset = size_t(route->experts[i])*input->nb[2];
        const size_t cache_offset = size_t(route->slots[i])*component.expert_bytes;
        const void * upload_source = (const uint8_t *) input->data + source_offset;
        if (sched->expert_lease_acquire && route->cache->lease_event) {
            ggml_backend_expert_lease lease = {};
            const auto acquire_start = std::chrono::steady_clock::now();
            const bool acquired = sched->expert_lease_acquire(
                    sched->expert_lease_user_data, layer, route->experts[i], component_index, &lease);
            const uint64_t acquire_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - acquire_start).count());
            sched->active_expert_cache_lease_acquire_ns += acquire_ns;
            sched->active_expert_cache_layer_stats[layer].lease_acquire_ns += acquire_ns;
            if (acquired && lease.data && lease.handle && lease.size == input->nb[2]) {
                upload_source = lease.data;
                route->cache->lease_handles.push_back(lease.handle);
                staged_from_lease = true;
                ++sched->active_expert_cache_lease_uploads;
                ++sched->active_expert_cache_layer_stats[layer].lease_uploads;
            } else {
                if (acquired && lease.handle) {
                    sched->expert_lease_release(sched->expert_lease_user_data, lease.handle);
                }
                if (sched->expert_lease_required) {
                    GGML_ABORT("required expert lease unavailable for layer=%d expert=%d component=%d\n",
                            layer, route->experts[i], component_index);
                }
            }
        } else if (sched->expert_lease_required) {
            GGML_ABORT("required expert lease path has no event-capable backend\n");
        }
        const auto submit_start = std::chrono::steady_clock::now();
        ggml_backend_expert_cache_upload_async(upload_backend, component.tensor,
                upload_source, cache_offset, input->nb[2]);
        const uint64_t submit_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - submit_start).count());
        sched->active_expert_cache_transfer_submit_ns += submit_ns;
        sched->active_expert_cache_load_bytes += input->nb[2];
        auto & layer_stats = sched->active_expert_cache_layer_stats[layer];
        layer_stats.transfer_submit_ns += submit_ns;
        layer_stats.load_bytes += input->nb[2];
        uploaded = true;
        entry.ready_mask |= component_mask;
        ++sched->active_expert_cache_uploads;
    }
    if (uploaded && route->cache->lease_event) {
        ggml_backend_event_record(route->cache->lease_event);
        ggml_backend_event_wait(split_backend, route->cache->lease_event);
        if (staged_from_lease) {
            route->cache->lease_event_recorded = true;
            route->cache->pending_upload_layer = layer;
        }
    }

    ggml_active_expert_cache_redirect_input(
            sched, input_cpy, input, component.tensor->data, component.tensor->buffer);
    ggml_active_expert_cache_redirect_ids(sched, node, ids_index, ids_source, route->mapped_ids);
    route->cache->used_this_pass = true;
    return true;
}

static void ggml_expert_prefill_drain_lane(
        ggml_backend_sched_t sched,
        ggml_expert_prefill_lane & lane,
        int stats_layer) {
    if (lane.lease_handles.empty()) return;
    GGML_ASSERT(sched->expert_lease_release != nullptr);
    GGML_ASSERT(lane.transfer_event && lane.transfer_event_recorded);
    const auto wait_start = std::chrono::steady_clock::now();
    ggml_backend_event_synchronize(lane.transfer_event);
    lane.transfer_event_recorded = false;
    const uint64_t wait_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wait_start).count());
    if (stats_layer >= 0) {
        sched->expert_prefill_layer_stats[stats_layer].transfer_wait_ns += wait_ns;
    }
    for (void * handle : lane.lease_handles) {
        sched->expert_lease_release(sched->expert_lease_user_data, handle);
    }
    lane.lease_handles.clear();
}

static void ggml_expert_prefill_release_device(
        ggml_backend_sched_t sched,
        ggml_expert_prefill_device & device) {
    for (auto & lane : device.lanes) {
        ggml_expert_prefill_drain_lane(sched, lane, lane.layer);
        if (lane.compute_event) {
            if (lane.compute_event_recorded) {
                ggml_backend_event_synchronize(lane.compute_event);
            }
            ggml_backend_event_free(lane.compute_event);
        }
        if (lane.transfer_event && lane.transfer_event_recorded) {
            // A failed optional staging attempt can have queued a partial
            // transfer without ever submitting a consumer graph.
            ggml_backend_event_synchronize(lane.transfer_event);
        }
        if (lane.transfer_event) ggml_backend_event_free(lane.transfer_event);
        if (lane.buffer) ggml_backend_buffer_free(lane.buffer);
        lane = {};
    }
    if (device.prior_compute_event) {
        ggml_backend_event_synchronize(device.prior_compute_event);
        ggml_backend_event_free(device.prior_compute_event);
    }
    if (device.transfer_backend) ggml_backend_free(device.transfer_backend);
    device = {};
}

static bool ggml_expert_prefill_init_device(
        ggml_backend_sched_t sched,
        int backend_id) {
    auto & device = sched->expert_prefill_devices[backend_id];
    const uint64_t lane_bytes = sched->expert_prefill_lane_bytes;
    if (device.ready && device.lane_bytes >= lane_bytes) return true;
    if (device.ready) {
        ggml_expert_prefill_release_device(sched, device);
    }
    if (device.attempted) return false;
    device.attempted = true;
    device.backend_id = backend_id;
    if (backend_id < 0 || backend_id >= sched->n_backends - 1 || lane_bytes == 0 ||
            lane_bytes > UINT64_MAX/2 ||
            2*lane_bytes > sched->expert_prefill_staging_capacity_bytes ||
            ggml_backend_is_cpu(sched->backends[backend_id])) {
        return false;
    }
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(sched->backends[backend_id])) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        const int cuda_device = ggml_backend_cuda_get_device(sched->backends[backend_id]);
        ggml_backend_cuda_get_device_memory(cuda_device, &free_bytes, &total_bytes);
        GGML_UNUSED(total_bytes);
        const uint64_t reserve = sched->expert_prefill_staging_reserve_bytes;
        if (free_bytes <= reserve || 2*lane_bytes > uint64_t(free_bytes) - reserve) return false;
        device.transfer_backend = ggml_backend_cuda_init(cuda_device, nullptr, &device);
    }
#endif
    if (!device.transfer_backend) return false;
    auto & route_event = sched->expert_prefill_route_events[backend_id];
    if (!route_event) route_event = ggml_backend_event_new(sched->backends[backend_id]);
    if (!route_event) {
        ggml_expert_prefill_release_device(sched, device);
        device.attempted = true;
        return false;
    }
    device.prior_compute_event = ggml_backend_event_new(sched->backends[backend_id]);
    if (!device.prior_compute_event) {
        ggml_expert_prefill_release_device(sched, device);
        device.attempted = true;
        return false;
    }
    // Queue the fence after any work already submitted to the main compute
    // stream, then make the independent upload stream wait on it. This orders
    // the first prompt transfer after preceding decode work without a host or
    // device-wide synchronization.
    ggml_backend_event_record(device.prior_compute_event);
    ggml_backend_event_wait(device.transfer_backend, device.prior_compute_event);
    device.lane_bytes = lane_bytes;
    for (auto & lane : device.lanes) {
        lane.buffer = ggml_backend_buft_alloc_buffer(sched->bufts[backend_id], size_t(lane_bytes));
        lane.transfer_event = ggml_backend_event_new(device.transfer_backend);
        lane.compute_event = ggml_backend_event_new(sched->backends[backend_id]);
        if (!lane.buffer || !lane.transfer_event || !lane.compute_event) {
            ggml_expert_prefill_release_device(sched, device);
            device.attempted = true;
            return false;
        }
        lane.layer = -1;
        device.allocated_bytes += ggml_backend_buffer_get_size(lane.buffer);
    }
    if (device.allocated_bytes > sched->expert_prefill_staging_capacity_bytes) {
        ggml_expert_prefill_release_device(sched, device);
        device.attempted = true;
        return false;
    }
#ifdef GGML_USE_CUDA
    if (ggml_backend_is_cuda(sched->backends[backend_id])) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_cuda_get_device_memory(
                ggml_backend_cuda_get_device(sched->backends[backend_id]),
                &free_bytes, &total_bytes);
        GGML_UNUSED(total_bytes);
        if (uint64_t(free_bytes) < sched->expert_prefill_staging_reserve_bytes) {
            ggml_expert_prefill_release_device(sched, device);
            device.attempted = true;
            return false;
        }
    }
#endif
    device.ready = true;
    fprintf(stderr,
            "ggml expert prefill: backend %d enabled with two %.1f MiB lanes "
            "(bounded total %.1f MiB, reserve %.1f MiB)\n",
            backend_id, lane_bytes/1024.0/1024.0,
            device.allocated_bytes/1024.0/1024.0,
            sched->expert_prefill_staging_reserve_bytes/1024.0/1024.0);
    return true;
}

static bool ggml_expert_prefill_find_resident(
        ggml_backend_sched_t sched,
        int backend_id,
        int layer,
        int expert,
        int component_index,
        size_t expert_bytes,
        ggml_tensor * source_view) {
    auto & cache = sched->active_expert_caches[backend_id];
    if (!cache.ready || component_index < 0 ||
            component_index >= GGML_ACTIVE_EXPERT_CACHE_COMPONENT_COUNT) return false;
    auto & component = cache.components[component_index];
    if (!component.tensor || component.expert_bytes != expert_bytes) return false;
    const uint8_t mask = uint8_t(1u << component_index);
    for (size_t slot = 0; slot < cache.entries.size(); ++slot) {
        const auto & entry = cache.entries[slot];
        if (entry.layer != layer || entry.expert != expert || (entry.ready_mask & mask) == 0) continue;
        *source_view = *component.tensor;
        source_view->ne[2] = 1;
        source_view->nb[3] = source_view->nb[2];
        source_view->data = (uint8_t *) component.tensor->data + slot*expert_bytes;
        return true;
    }
    return false;
}

static bool ggml_expert_prefill_stage(
        ggml_backend_sched_t sched,
        ggml_backend_t split_backend,
        int split_backend_id,
        ggml_tensor * input,
        ggml_tensor * input_cpy,
        const std::vector<uint32_t> & unique_ids) {
    if (sched->active_expert_decode || sched->expert_prefill_staging_capacity_bytes == 0 ||
            !input || input->ne[2] <= 1 || input->nb[2] == 0 ||
            !ggml_backend_buffer_is_host(input->buffer)) return false;
    int layer = -1;
    const int component_index = ggml_active_expert_cache_component_from_name(input, &layer);
    const auto layer_layout = sched->expert_prefill_layer_bytes.find(layer);
    if (component_index < 0 || layer_layout == sched->expert_prefill_layer_bytes.end() ||
            !ggml_expert_prefill_init_device(sched, split_backend_id)) return false;
    auto & device = sched->expert_prefill_devices[split_backend_id];
    const int lane_index = layer & 1;
    auto & lane = device.lanes[lane_index];
    auto & stats = sched->expert_prefill_layer_stats[layer];
#ifdef GGML_USE_CUDA
    if (!device.used_this_split[lane_index] && ggml_backend_is_cuda(split_backend)) {
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        ggml_backend_cuda_get_device_memory(
                ggml_backend_cuda_get_device(split_backend), &free_bytes, &total_bytes);
        GGML_UNUSED(total_bytes);
        if (uint64_t(free_bytes) < sched->expert_prefill_staging_reserve_bytes) {
            ggml_expert_prefill_release_device(sched, device);
            device.attempted = true;
            return false;
        }
    }
#endif
    if (lane.layer != layer) {
        ggml_expert_prefill_drain_lane(sched, lane, lane.layer);
        if (lane.compute_event_recorded) {
            ggml_backend_event_wait(device.transfer_backend, lane.compute_event);
            lane.compute_event_recorded = false;
            lane.transfer_event_recorded = false;
        }
        lane.layer = layer;
        lane.ready_masks.assign(size_t(input->ne[2]), 0);
    } else if (lane.ready_masks.size() != size_t(input->ne[2])) {
        return false;
    }
    uint64_t component_offset = 0;
    for (int i = 0; i < component_index; ++i) {
        if (layer_layout->second[i] > UINT64_MAX - component_offset) return false;
        component_offset += layer_layout->second[i];
    }
    if (uint64_t(input->ne[2]) > UINT64_MAX/uint64_t(input->nb[2])) return false;
    const uint64_t component_bytes = uint64_t(input->nb[2])*uint64_t(input->ne[2]);
    if (
            component_bytes > layer_layout->second[component_index] ||
            component_offset > device.lane_bytes ||
            component_bytes > device.lane_bytes - component_offset) return false;

    const uint8_t component_mask = uint8_t(1u << component_index);
    bool waited_for_cache = false;
    auto * lane_base = (uint8_t *) ggml_backend_buffer_get_base(lane.buffer);
    const uint8_t * pending_h2d_source = nullptr;
    size_t pending_h2d_destination = 0;
    size_t pending_h2d_bytes = 0;
    uint64_t pending_h2d_components = 0;
    auto flush_h2d = [&]() {
        if (pending_h2d_bytes == 0) return;
        ggml_tensor destination_view = *input_cpy;
        destination_view.data = lane_base + pending_h2d_destination;
        destination_view.buffer = lane.buffer;
        const auto submit_start = std::chrono::steady_clock::now();
        ggml_backend_tensor_set_async(
                device.transfer_backend, &destination_view,
                pending_h2d_source, 0, pending_h2d_bytes);
        stats.transfer_submit_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - submit_start).count());
        stats.h2d_components += pending_h2d_components;
        ++stats.h2d_batches;
        stats.h2d_bytes += pending_h2d_bytes;
        pending_h2d_source = nullptr;
        pending_h2d_destination = 0;
        pending_h2d_bytes = 0;
        pending_h2d_components = 0;
        lane.transfers_queued = true;
    };
    auto fail_staging = [&]() {
        flush_h2d();
        if (lane.transfers_queued) {
            ggml_backend_event_record(lane.transfer_event);
            lane.transfer_event_recorded = true;
        }
        if (!lane.lease_handles.empty()) {
            ggml_expert_prefill_drain_lane(sched, lane, layer);
        }
        lane.transfers_queued = false;
        ++stats.fallbacks;
        ++sched->expert_prefill_fallbacks;
        return false;
    };
    for (int expert = 0; expert < input->ne[2]; ++expert) {
        if (size_t(expert >> 5) >= unique_ids.size() ||
                (unique_ids[expert >> 5] & (1u << (expert & 31))) == 0 ||
                (lane.ready_masks[expert] & component_mask) != 0) continue;
        ++stats.selected_components;
        const size_t destination_offset = size_t(component_offset) + size_t(expert)*input->nb[2];
        ggml_tensor source_view = {};
        bool copied_from_device = ggml_expert_prefill_find_resident(
                sched, split_backend_id, layer, expert, component_index, input->nb[2], &source_view);
        if (copied_from_device) {
            auto & cache = sched->active_expert_caches[split_backend_id];
            if (!waited_for_cache && cache.compute_event_recorded) {
                ggml_backend_event_wait(device.transfer_backend, cache.compute_event);
                waited_for_cache = true;
            }
            ggml_tensor destination_view = *input_cpy;
            destination_view.ne[2] = 1;
            destination_view.nb[3] = destination_view.nb[2];
            destination_view.data = lane_base + destination_offset;
            destination_view.buffer = lane.buffer;
            flush_h2d();
            const auto submit_start = std::chrono::steady_clock::now();
            copied_from_device = device.transfer_backend->iface.cpy_tensor_async &&
                device.transfer_backend->iface.cpy_tensor_async(
                    device.transfer_backend, device.transfer_backend,
                    &source_view, &destination_view);
            stats.transfer_submit_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - submit_start).count());
        }
        if (copied_from_device) {
            ++stats.d2d_components;
            ++stats.d2d_batches;
            stats.d2d_bytes += input->nb[2];
            lane.transfers_queued = true;
        } else {
            const void * upload_source = (const uint8_t *) input->data + size_t(expert)*input->nb[2];
            if (sched->expert_lease_acquire) {
                ggml_backend_expert_lease lease = {};
                auto acquire = [&]() {
                    const auto acquire_start = std::chrono::steady_clock::now();
                    const bool result = sched->expert_lease_acquire(
                            sched->expert_lease_user_data, layer, expert, component_index, &lease);
                    stats.lease_acquire_ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - acquire_start).count());
                    return result;
                };
                bool acquired = acquire();
                if ((!acquired || !lease.data || !lease.handle || lease.size != input->nb[2]) &&
                        !lane.lease_handles.empty()) {
                    if (acquired && lease.handle) {
                        sched->expert_lease_release(sched->expert_lease_user_data, lease.handle);
                    }
                    lease = {};
                    flush_h2d();
                    if (lane.transfers_queued) {
                        ggml_backend_event_record(lane.transfer_event);
                        lane.transfer_event_recorded = true;
                    }
                    ggml_expert_prefill_drain_lane(sched, lane, layer);
                    lane.transfers_queued = false;
                    acquired = acquire();
                }
                if (acquired && lease.data && lease.handle && lease.size == input->nb[2]) {
                    upload_source = lease.data;
                    lane.lease_handles.push_back(lease.handle);
                } else {
                    if (acquired && lease.handle) {
                        sched->expert_lease_release(sched->expert_lease_user_data, lease.handle);
                    }
                    if (sched->expert_lease_required) return fail_staging();
                }
            } else if (sched->expert_lease_required) {
                return fail_staging();
            }
            const uintptr_t source_address = reinterpret_cast<uintptr_t>(upload_source);
            const uintptr_t pending_address = reinterpret_cast<uintptr_t>(pending_h2d_source);
            const bool source_contiguous = pending_h2d_bytes > 0 &&
                pending_address <= UINTPTR_MAX - pending_h2d_bytes &&
                pending_address + pending_h2d_bytes == source_address;
            const bool destination_contiguous = pending_h2d_bytes > 0 &&
                pending_h2d_destination <= SIZE_MAX - pending_h2d_bytes &&
                pending_h2d_destination + pending_h2d_bytes == destination_offset;
            if (pending_h2d_bytes > 0 && (!source_contiguous || !destination_contiguous)) {
                flush_h2d();
            }
            if (pending_h2d_bytes == 0) {
                pending_h2d_source = static_cast<const uint8_t *>(upload_source);
                pending_h2d_destination = destination_offset;
            }
            if (input->nb[2] > SIZE_MAX - pending_h2d_bytes) return fail_staging();
            pending_h2d_bytes += input->nb[2];
            ++pending_h2d_components;
        }
        lane.ready_masks[expert] |= component_mask;
    }
    flush_h2d();
    if (lane.transfers_queued) {
        ggml_backend_event_record(lane.transfer_event);
        lane.transfer_event_recorded = true;
        ggml_backend_event_wait(split_backend, lane.transfer_event);
        // The compute stream now owns the dependency. Releasing this
        // component batch after its transfer event keeps the host tier bounded
        // even when a prompt selects more experts than the RAM cache can hold.
        ggml_expert_prefill_drain_lane(sched, lane, layer);
        lane.transfers_queued = false;
    }
    ggml_active_expert_cache_redirect_input(
            sched, input_cpy, input, lane_base + component_offset, lane.buffer);
    device.used_this_split[lane_index] = true;
    return true;
}

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static void ggml_active_expert_cache_record_compute(
        ggml_backend_sched_t sched,
        ggml_backend_t backend) {
    const int backend_id = ggml_backend_sched_backend_id(sched, backend);
    if (backend_id < 0) return;
    auto & cache = sched->active_expert_caches[backend_id];
    if (!cache.ready || !cache.used_this_pass || !cache.compute_event) return;
    ggml_backend_event_record(cache.compute_event);
    cache.compute_event_recorded = true;
}

static void ggml_expert_prefill_record_compute(
        ggml_backend_sched_t sched,
        ggml_backend_t backend) {
    const int backend_id = ggml_backend_sched_backend_id(sched, backend);
    if (backend_id < 0) return;
    auto & device = sched->expert_prefill_devices[backend_id];
    if (!device.ready) return;
    for (size_t lane_index = 0; lane_index < device.lanes.size(); ++lane_index) {
        if (!device.used_this_split[lane_index]) continue;
        auto & lane = device.lanes[lane_index];
        GGML_ASSERT(lane.compute_event != nullptr);
        ggml_backend_event_record(lane.compute_event);
        lane.compute_event_recorded = true;
        device.used_this_split[lane_index] = false;
    }
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    //printf("%s: have %d backends, buffer is %s\n", __func__, sched->n_backends, ggml_backend_buffer_name(buffer));
    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        //printf("  Checking bacckend %d (%s)\n", i, ggml_backend_name(sched->backends[i]));
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    fprintf(stderr, "%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // TODO: use supports_op to check if the backend supports the op

    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    bool offload_enabled = ggml_backend_sched_offload_enabled(sched, tensor->op);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (offload_enabled && src_backend_id == sched->n_backends - 1) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            fprintf(stderr, "\n## SPLIT #%d: %s # %d inputs: ", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                fprintf(stderr, "[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            fprintf(stderr, "\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
        fprintf(stderr, "node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s]:", i, ggml_op_name(node->op), node->name,
            fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node));
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
            fprintf(stderr, " %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
        }
        fprintf(stderr, "\n");
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
static void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // Cache redirects point into the scheduler tensor context. Restore them
    // while that context is still alive; doing this after ggml_free() can
    // mutate unrelated tensors that reuse the same metadata address in the
    // next graph (for example, an expert layout overwriting top-k IDs).
    ggml_active_expert_cache_reset_pass(sched);

    // Size a slot from the worst-case component layouts in the live graph. The
    // first expert layer is not representative for architectures such as DSV4.
    sched->active_expert_component_max_bytes.fill(0);
    sched->expert_prefill_layer_bytes.clear();
    sched->expert_prefill_lane_bytes = 0;
    std::unordered_set<const ggml_tensor *> expert_routes;
    for (int node_index = 0; node_index < graph->n_nodes; ++node_index) {
        const ggml_tensor * node = graph->nodes[node_index];
        if (node->op == GGML_OP_MOE_FUSED_UP_GATE || node->op == GGML_OP_MUL_MAT_ID) {
            const int ids_index = node->op == GGML_OP_MOE_FUSED_UP_GATE ? 3 : 2;
            if (node->src[ids_index]) {
                expert_routes.insert(node->src[ids_index]);
            }
        }
        for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
            const ggml_tensor * source = node->src[source_index];
            int layer = -1;
            const int component = source ? ggml_active_expert_cache_component_from_name(source, &layer) : -1;
            if (component >= 0 && source->ne[2] > 1 && source->nb[2] > 0) {
                sched->active_expert_component_max_bytes[component] = std::max<uint64_t>(
                        sched->active_expert_component_max_bytes[component], source->nb[2]);
                if (ggml_active_expert_cache_is_host_expert_weight(source)) {
                    const auto descriptor = ggml_active_expert_cache_describe_layout(source);
                    auto & catalog = sched->active_expert_cache_layout_catalog[size_t(component)];
                    const auto known = std::find_if(
                            catalog.begin(), catalog.end(),
                            [&](const ggml_active_expert_cache_layout & candidate) {
                                return ggml_active_expert_cache_same_layout(candidate, descriptor);
                            });
                    if (known == catalog.end()) {
                        catalog.push_back(descriptor);
                    } else {
                        known->logical_experts = std::max(
                                known->logical_experts, descriptor.logical_experts);
                    }
                }
                auto & layer_bytes = sched->expert_prefill_layer_bytes[layer];
                layer_bytes[component] = std::max<uint64_t>(
                        layer_bytes[component], ggml_nbytes(source));
            }
        }
    }
    sched->active_expert_cache_route_capacity = std::max(
            sched->active_expert_cache_route_capacity, expert_routes.size());
    sched->active_expert_set_bytes = 0;
    for (const uint64_t bytes : sched->active_expert_component_max_bytes) {
        if (bytes > UINT64_MAX - sched->active_expert_set_bytes) {
            sched->active_expert_set_bytes = 0;
            break;
        }
        sched->active_expert_set_bytes += bytes;
    }
    for (const auto & item : sched->expert_prefill_layer_bytes) {
        uint64_t layer_total = 0;
        for (const uint64_t bytes : item.second) {
            if (bytes > UINT64_MAX - layer_total) {
                layer_total = 0;
                break;
            }
            layer_total += bytes;
        }
        sched->expert_prefill_lane_bytes = std::max(sched->expert_prefill_lane_bytes, layer_total);
    }

    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;
    sched->has_reduce = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        if (ggml_ese_tensor_get_role(node) == GGML_ESE_ROUTE_CPU) {
            *node_backend_id = sched->n_backends - 1;
        } else if (ggml_ese_tensor_get_role(node) == GGML_ESE_ROUTE_GPU) {
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                if (!node->src[source]) continue;
                const int source_backend_id = tensor_backend_id(node->src[source]);
                if (source_backend_id >= 0 && source_backend_id < sched->n_backends - 1) {
                    *node_backend_id = source_backend_id;
                    break;
                }
            }
        }
        if (node->op == GGML_OP_REDUCE) {
            auto view_src = node->view_src;
            if (node->op_params[3] == 2) {
                // ESE heterogeneous sources are not ordered by scheduler
                // backend ID (for example source 1 can be CPU backend 3 on a
                // three-GPU host). Their producing branches already carry the
                // correct assignment; the reduction follows its explicit
                // destination branch.
                const int destination = node->op_params[4];
                GGML_ASSERT(destination >= 0 && destination < node->op_params[1]);
                GGML_ASSERT(node->src[destination] == view_src);
                const int destination_backend_id = tensor_backend_id(view_src);
                if (destination_backend_id >= 0) {
                    *node_backend_id = destination_backend_id;
                }
                continue;
            }
            int src_id = -1;
            for (int j = 0; j < node->op_params[1]; ++j) {
                if (node->src[j]) {
                    int * this_node_backend_id = &tensor_backend_id(node->src[j]);
                    if (*this_node_backend_id == -1) {
                        *this_node_backend_id = j;
                    } else {
                        GGML_ASSERT(*this_node_backend_id == j);
                    }
                    if (view_src == node->src[j]) {
                        src_id = j;
                    }
                }
            }
            if (src_id >= 0) {
                int * this_node_backend_id = &tensor_backend_id(view_src);
                *this_node_backend_id = tensor_backend_id(node->src[src_id]);
                *node_backend_id = *this_node_backend_id;
            }
        }
        else if (node->op == GGML_OP_MUL && node->src[0]->op == GGML_OP_NORM) {
            // This is a hack for Cohere2. Without this hack the scheduler creates
            // totally nonsensical splits for that arch
            int * src1_id = &tensor_backend_id(node->src[1]);
            if (*src1_id >= 0) {
                int * src0_id = &tensor_backend_id(node->src[0]);
                int * dst_id  = &tensor_backend_id(node);
                *src0_id = *src1_id;
                *dst_id  = *src1_id;
                // For some reason that I don't understand, we can have norm backend already assigned
                // at this point. How? That's why this more logical approach of first checking is commented out
                //if (*src0_id < 0) {
                //    *src0_id = *src1_id;
                //} else {
                //    printf("Oops: backend_id_src0(%s) = %d, backend_id_src1(%s) = %d\n", node->src[0]->name, *src0_id, node->src[1]->name, *src1_id);
                //    //GGML_ASSERT(*src0_id == *src1_id);
                //}
                //if (*dst_id < 0) {
                //    *dst_id = *src1_id;
                //} else {
                //    printf("Oops: backend_id_dst(%s) = %d, backend_id_src1(%s) = %d\n", node->name, *dst_id, node->src[1]->name, *src1_id);
                //    //GGML_ASSERT(*dst_id == *src1_id);
                //}
            }
        }
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);
            //printf("Pass 1: assigned backend %d to node %d, %s(%s)\n", *node_backend_id, i, ggml_op_name(node->op), node->name);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }

        // CPU-MoE normally pins these nodes to the host because their expert
        // tensors live there.  A one-token decode can instead use four compact
        // GPU cache slots, so choose the GPU that owns the activation.  Prompt
        // batches intentionally keep the original CPU assignment.
        if (const int prefill_backend_id = ggml_expert_prefill_backend_for_node(sched, graph, node);
                prefill_backend_id >= 0) {
            *node_backend_id = prefill_backend_id;
        } else if (const int cache_backend_id = ggml_active_expert_cache_backend_for_node(sched, graph, node); cache_backend_id >= 0) {
            *node_backend_id = cache_backend_id;
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                //printf("(u1) invoking ggml_backend_sched_set_if_supported for node %d, %s with cur_backend_id = %d, node_backend_id = %d\n", i, node->name, cur_backend_id, *node_backend_id);
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                //printf("(d1) invoking ggml_backend_sched_set_if_supported for node %d, %s with cur_backend_id = %d, node_backend_id = %d\n", i, node->name, cur_backend_id, *node_backend_id);
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                //printf("(u2) invoking ggml_backend_sched_set_if_supported for node %d, %s with cur_backend_id = %d, node_backend_id = %d\n", i, node->name, cur_backend_id, *node_backend_id);
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                //printf("(d2) invoking ggml_backend_sched_set_if_supported for node %d, %s with cur_backend_id = %d, node_backend_id = %d\n", i, node->name, cur_backend_id, *node_backend_id);
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        //printf("Pass 3: assigned backend %d to unassigned node %d, %s\n", b, i, node->name);
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        //printf("Pass 3: assigned backend %d to node %d, %s previously assigned to backend %d\n", b, i, node->name, *node_backend_id);
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                    //printf("Pass 4: assigned backend %d to src %d, %s in node %d, %s frpm view_src\n", *src_backend_id, j, src->name, i, node->name);
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                    //printf("Pass 4: assigned backend %d to src %d, %s in node %d, %s frpm current\n", *src_backend_id, j, src->name, i, node->name);
                }
            }
        }
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        // Expansion can assign an initially unallocated reduction view from an
        // adjacent node. Re-assert the explicit ESE destination after every
        // producing branch has its final backend so the reduction kernel runs
        // on the device that owns its result storage.
        for (int node_index = 0; node_index < graph->n_nodes; ++node_index) {
            ggml_tensor * node = graph->nodes[node_index];
            if (node->op != GGML_OP_REDUCE || node->op_params[3] != 2) continue;
            const int destination = node->op_params[4];
            GGML_ASSERT(destination >= 0 && destination < node->op_params[1]);
            const int destination_backend_id = tensor_backend_id(node->src[destination]);
            GGML_ASSERT(destination_backend_id >= 0);
            tensor_backend_id(node) = destination_backend_id;
            if (getenv("GGML_ESE_REDUCE_DEBUG")) {
                fprintf(stderr, "ggml ESE reduce: node=%s destination=%d backend=%d sources=",
                        ggml_get_name(node), destination, destination_backend_id);
                for (int source = 0; source < node->op_params[1]; ++source) {
                    fprintf(stderr, "%s%d", source ? "," : "", node->src[source] ? tensor_backend_id(node->src[source]) : -1);
                }
                fprintf(stderr, "\n");
            }
        }

        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        int split_prefill_layer = -1;
        bool split_has_compute_node = false;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            assert(node_backend_id != -1); // all nodes should be assigned by now

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            int node_prefill_layer = -1;
            if (!sched->active_expert_decode &&
                    sched->expert_prefill_staging_capacity_bytes > 0 &&
                    sched->expert_prefill_lane_bytes > 0 &&
                    sched->expert_prefill_lane_bytes <= UINT64_MAX/2 &&
                    2*sched->expert_prefill_lane_bytes <=
                        sched->expert_prefill_staging_capacity_bytes &&
                    node_backend_id < sched->n_backends - 1 &&
                    (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MOE_FUSED_UP_GATE)) {
                for (int source_index = 0; source_index < GGML_MAX_SRC; ++source_index) {
                    int source_layer = -1;
                    if (ggml_active_expert_cache_is_host_expert_weight(node->src[source_index]) &&
                            ggml_active_expert_cache_component_from_name(
                                node->src[source_index], &source_layer) >= 0) {
                        node_prefill_layer = source_layer;
                        break;
                    }
                }
                // A layer boundary is also a submission boundary. Once split
                // N is queued on the compute stream, copy_inputs for split N+1
                // can fill the opposite lane on the transfer stream. Keeping
                // all components of one layer in one split prevents premature
                // lane reuse while retaining that overlap.
                if (node_prefill_layer >= 0 &&
                        node_prefill_layer != split_prefill_layer &&
                        split_has_compute_node) {
                    need_new_split = true;
                }
            }
            if (node->op == GGML_OP_REDUCE) {
                sched->has_reduce = true;
            }
            if ((node->op == GGML_OP_ADD && node->op_params[0] == 0xff) ||
                 node->op == GGML_OP_REDUCE ||
                 node->op == GGML_OP_FAKE_CPY ||
                 node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == 0xff) {
                need_new_split = true;
            }
            else if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            //printf("starting new split because of too many inputs: node %s, input %s\n", node->name, src->name);
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                GGML_ASSERT(i_split < GGML_SCHED_MAX_SPLITS);
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
                split_prefill_layer = -1;
                split_has_compute_node = false;
            }

            if (node_prefill_layer >= 0) {
                if (split_prefill_layer >= 0) {
                    GGML_ASSERT(split_prefill_layer == node_prefill_layer);
                }
                split_prefill_layer = node_prefill_layer;
            }
            split_has_compute_node = true;

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                assert(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        if (node->op == GGML_OP_REDUCE && node->op_params[3] != 2 &&
                                !ggml_backend_is_cpu(sched->backends[src_backend_id])) {
                            //printf("setting tensor_id_copy(reduce, %zu, %d, %s) to %s\n", src_id, cur_backend_id, node->name, src->name);
                            tensor_id_copy(src_id, cur_backend_id, 0) = src;
                        } else if (node->op == GGML_OP_FAKE_CPY && src->op == GGML_OP_REDUCE) {
                            //printf("setting tensor_id_copy(fake_cpy, %zu, %d, %s) to %s\n", src_id, cur_backend_id, node->name, src->src[j]->name);
                            tensor_id_copy(src_id, cur_backend_id, 0) = src->src[j];
                        } else {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        const ggml_tensor * active_ids = node->op == GGML_OP_MOE_FUSED_UP_GATE ? node->src[3] : node->src[2];
                        const bool compact_active_expert_copy =
                            ggml_active_expert_cache_is_decode_node(node) &&
                            sched->active_expert_cache_slots >= 1 && active_ids &&
                            ggml_active_expert_cache_slots_for(sched, src) >= active_ids->ne[0] &&
                            ggml_active_expert_cache_is_host_expert_weight(src);
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = compact_active_expert_copy
                                ? ggml_active_expert_cache_compact_copy(sched, src)
                                : ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        if (n_inputs >= GGML_SCHED_MAX_SPLIT_INPUTS) {
                            printf("======================== Oops, too many inputs (%d)\n", n_inputs+1);
                            for (int k = 0; k < n_inputs; ++k) printf("Input %2d: %s\n", k, split->inputs[k]->name);
                        }
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                        }
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = graph->n_nodes + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **)realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **)realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
        // the re-allocation may cause the split inputs to be moved to a different address
        ggml_backend_sched_synchronize(sched);
#ifndef NDEBUG
        fprintf(stderr, "%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif
        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            fprintf(stderr, "%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

static void ggml_backend_sched_copy_inputs(ggml_backend_sched_t sched, ggml_backend_sched_split * split, std::array<bool, GGML_SCHED_MAX_BACKENDS> & needs_sync,
        std::vector<int32_t> & ids, std::vector<uint32_t> & unique_ids, ggml_tensor * last_ids_tensor) {
    if (split->n_inputs < 1) return;
    constexpr bool k_set_sync = false;
    int split_backend_id = split->backend_id;
    ggml_backend_t split_backend = sched->backends[split_backend_id];
    ggml_backend_t last_input_backend = nullptr;
    bool synced_on_input = false;
    for (int j = 0; j < split->n_inputs; j++) {
        ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[j]);
        struct ggml_tensor * input = split->inputs[j];
        struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);

        if (input->flags & GGML_TENSOR_FLAG_INPUT) {
            // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
            // if there are multiple inputs for the split, and we have already synchronized this backend, no need to do it again.
            if (!synced_on_input) {
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                synced_on_input = true;
            }
            ggml_backend_tensor_copy(input, input_cpy);
        } else {
            // wait for the split backend to finish using the input before overwriting it
            if (needs_sync[split_backend_id]) {
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                needs_sync[split_backend_id] = k_set_sync;
            }

            ggml_tensor * node = split->graph.nodes[0];
            if (sched->active_expert_cache_slots >= 1 && getenv("LLAMA_EXPERT_GPU_CACHE_DEBUG") && input->ne[2] > 1) {
                fprintf(stderr, "ggml active-expert cache: input name=%s node=%s usage=%d host=%d ne=[%lld,%lld,%lld]\n",
                        ggml_get_name(input), ggml_op_name(node->op),
                        (int) ggml_backend_buffer_get_usage(input->buffer),
                        (int) ggml_backend_buffer_is_host(input->buffer),
                        (long long) input->ne[0], (long long) input->ne[1], (long long) input->ne[2]);
            }
            if (sched->only_active_experts && split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) &&
                    (node->op == GGML_OP_MUL_MAT_ID || node->op == GGML_OP_MOE_FUSED_UP_GATE)) {

                if (input_backend != last_input_backend) {
                    // Prompt expert sources are immutable model/lease data.
                    // Their selected upload path is ordered explicitly below;
                    // a backend-wide host synchronization would only create a
                    // scheduling bubble. Decode retains its established path.
                    if (sched->active_expert_decode ||
                            sched->expert_prefill_staging_capacity_bytes == 0) {
                        ggml_backend_synchronize(input_backend);
                    }
                    last_input_backend = input_backend;
                }

                const int ids_index = node->op == GGML_OP_MUL_MAT_ID ? 2 : 3;
                ggml_tensor * ids_tensor = ggml_active_expert_cache_source_ids(sched, node, ids_index);
                auto ids_backend = split_backend;

                // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                // in that case, we use the original ids tensor
                if (ids_tensor == node->src[ids_index]) {
                    for (int jj = j + 1; jj < split->n_inputs; ++jj) {
                        if (ids_tensor == tensor_copy(split->inputs[jj], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[jj];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[jj]);
                            break;
                        }
                    }
                } else {
                    ids_backend = ggml_backend_sched_get_tensor_backend(sched, ids_tensor);
                }

                int n_expert = node->src[0]->ne[2];

                // Stage the routed expert slices into the bounded resident cache
                // and remap this operation to its cache slots.  Prompt batches
                // and unsupported MoE layouts retain the established transfer
                // path below.
                if (sched->active_expert_cache_slots >= 1 && getenv("LLAMA_EXPERT_GPU_CACHE_DEBUG")) {
                    fprintf(stderr, "ggml active-expert cache: candidate name=%s op=%s ne=[%lld,%lld,%lld] ids=[%lld,%lld]\n",
                            ggml_get_name(input), ggml_op_name(node->op),
                            (long long) input->ne[0], (long long) input->ne[1], (long long) input->ne[2],
                            (long long) ids_tensor->ne[0], (long long) ids_tensor->ne[1]);
                }
                if (sched->active_expert_decode) {
                    if (ggml_active_expert_cache_stage(sched, split_backend, split_backend_id, input, input_cpy,
                            node, ids_index, ids_tensor)) {
                        continue;
                    }
                    ++sched->active_expert_cache_forced_fallbacks;
                    if (input_cpy->ne[2] != input->ne[2]) {
                        GGML_ABORT("bounded expert-cache staging failed for compact tensor %s\n",
                                ggml_get_name(input));
                    }
                    if (sched->expert_lease_required && sched->active_expert_cache_slots >= 1) {
                        GGML_ABORT("sidecar-only expert staging refused original-tensor fallback for %s\n",
                                ggml_get_name(input));
                    }
                }

                if (ids_tensor != last_ids_tensor) {
                    ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));

                    const auto route_readback_start = std::chrono::steady_clock::now();
                    ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                    bool route_event_waited = false;
                    if (!sched->active_expert_decode &&
                            sched->expert_prefill_staging_capacity_bytes > 0) {
                        const int ids_backend_id = ggml_backend_sched_backend_id(sched, ids_backend);
                        if (ids_backend_id >= 0) {
                            auto & route_event = sched->expert_prefill_route_events[ids_backend_id];
                            if (!route_event) route_event = ggml_backend_event_new(ids_backend);
                            if (route_event) {
                                ggml_backend_event_record(route_event);
                                ggml_backend_event_synchronize(route_event);
                                route_event_waited = true;
                            }
                        }
                    }
                    if (!route_event_waited) {
                        ggml_backend_synchronize(ids_backend);
                        if (!sched->active_expert_decode &&
                                sched->expert_prefill_staging_capacity_bytes > 0) {
                            ++sched->expert_prefill_route_global_sync_fallbacks;
                        }
                    }
                    if (!sched->active_expert_decode) {
                        int route_layer = -1;
                        ggml_active_expert_cache_component_from_name(node->src[0], &route_layer);
                        if (route_layer >= 0) {
                            sched->expert_prefill_layer_stats[route_layer].route_readback_ns +=
                                uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - route_readback_start).count());
                        }
                    }
                    if (auto id = tensor_backend_id(ids_tensor); id >= 0 && id < GGML_SCHED_MAX_BACKENDS) {
                        needs_sync[id] = k_set_sync;
                    }
                    //needs_sync[tensor_backend_id(ids_tensor)] = k_set_sync;

                    unique_ids.resize((n_expert + 31)/32);
                    std::memset(unique_ids.data(), 0, unique_ids.size()*sizeof(uint32_t));
                    for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                        for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                            int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                            if (id < 0 || id >= n_expert) {
                                continue;
                            }
                            unique_ids[id >> 5] |= (1u << (id & 31));
                        }
                    }

                    last_ids_tensor = ids_tensor;
                }

                const uint64_t prefill_fallbacks_before = sched->expert_prefill_fallbacks;
                if (ggml_expert_prefill_stage(
                        sched, split_backend, split_backend_id, input, input_cpy, unique_ids)) {
                    continue;
                }
                if (!sched->active_expert_decode && sched->expert_lease_required &&
                        sched->expert_prefill_staging_capacity_bytes > 0) {
                    GGML_ABORT("sidecar-only prefill staging refused original-tensor fallback for %s\n",
                            ggml_get_name(input));
                }
                if (!sched->active_expert_decode &&
                        sched->expert_prefill_staging_capacity_bytes > 0 &&
                        sched->expert_prefill_fallbacks == prefill_fallbacks_before) {
                    int prefill_layer = -1;
                    if (ggml_active_expert_cache_component_from_name(input, &prefill_layer) >= 0) {
                        ++sched->expert_prefill_layer_stats[prefill_layer].fallbacks;
                        ++sched->expert_prefill_fallbacks;
                    }
                }

                // when the expert prefetch engine streamed this tensor ahead
                // (see the lookahead in compute_splits), wait for it so the
                // host-side reads below hit warm page cache instead of faulting
                ggml_moe_prefetch_wait(input);

                const size_t expert_size = input->ne[2] > 1 ? input->nb[2] : input->nb[1];

                if (input->ne[2] > 1) {

                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = 512;
                        const size_t padding_end = last_id < n_expert - 1 ? std::min<size_t>(expert_size, padding) : 0;

                        ggml_backend_tensor_set_async(split_backend,
                                input_cpy,
                                (const uint8_t *)input->data + expert_offset, expert_offset,
                                // copy a bit extra to ensure there are no NaNs in the padding
                                expert_size_copy + padding_end);

                    };

                    auto next_on_id = [&unique_ids, n_expert] (int id) {
                        while (id < n_expert && (unique_ids[id >> 5] & (1u << (id & 31))) == 0) ++id;
                        return id;
                    };
                    auto next_off_id = [&unique_ids, n_expert] (int id) {
                        while (id < n_expert && (unique_ids[id >> 5] & (1u << (id & 31))) != 0) ++id;
                        return id;
                    };

                    int first_id = next_on_id(0);
                    while (first_id < n_expert) {
                        int last_id = next_off_id(first_id+1);
                        copy_experts(first_id, last_id-1);
                        first_id = next_on_id(last_id);
                    }

                } else {
                    auto copy_size = ggml_nbytes(input);
                    ggml_backend_tensor_set_async(split_backend, input_cpy, input->data, 0, copy_size);
                }

            } else
                // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                    ggml_backend_synchronize(input_backend);
                    if (needs_sync[split_backend_id]) {
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        needs_sync[split_backend_id] = k_set_sync;
                    }
                    ggml_backend_tensor_copy(input, input_cpy);
                }
        }
    }
}

static ggml_status ggml_backend_sched_eval(ggml_backend_sched_t sched, ggml_backend_t split_backend, ggml_backend_sched_split * split) {
    int cpu_hybrid_layer = -1;
    if (sched->has_reduce && sched->active_expert_cache_slots >= 1 &&
            ggml_backend_is_cpu(split_backend)) {
        for (int index = 0; index < split->graph.n_nodes; ++index) {
            const ggml_tensor * node = split->graph.nodes[index];
            if (ggml_ese_tensor_get_role(node) != GGML_ESE_ROUTE_CPU) continue;
            const int layer = ggml_ese_callback_layer_from_name(ggml_get_name(node));
            if (layer < 0 || (cpu_hybrid_layer >= 0 && cpu_hybrid_layer != layer)) {
                cpu_hybrid_layer = -2;
                break;
            }
            cpu_hybrid_layer = layer;
        }
    }
    std::chrono::steady_clock::time_point eval_start;
    if (cpu_hybrid_layer >= 0) {
        eval_start = std::chrono::steady_clock::now();
    }
    if (!sched->callback_eval) {
#if IK_PRINT_TIMING
        int64_t tim2 = ggml_time_us();
        printf("%s(.1.): %d us\n", __func__, (int)(tim2-tim1));
#endif
        enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
        if (ec != GGML_STATUS_SUCCESS) {
            return ec;
        }
    } else {
        // similar to ggml_backend_compare_graph_backend
        for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
            struct ggml_tensor * t = split->graph.nodes[j0];

            // check if the user needs data from this node
            int need = sched->callback_eval(t, true, sched->callback_eval_user_data);

            int j1 = j0;

            // determine the range [j0, j1] of nodes that can be computed together
            while (!need && j1 < split->graph.n_nodes - 1) {
                t = split->graph.nodes[++j1];
                need = sched->callback_eval(t, true, sched->callback_eval_user_data);
            }

            struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

#if IK_PRINT_TIMING
            int64_t tim2 = ggml_time_us();
            printf("%s(.2.): %d us\n", __func__, (int)(tim2-tim1));
#endif

            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }

            // TODO: pass backend to the callback, then the user can decide if they want to synchronize
            if (need == 1) {
                ggml_backend_synchronize(split_backend);
            }

            if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                break;
            }

            j0 = j1;
        }
    }
    if (cpu_hybrid_layer >= 0) {
        const auto found = sched->active_expert_cache_layer_stats.find(cpu_hybrid_layer);
        if (found != sched->active_expert_cache_layer_stats.end()) {
            const uint64_t compute_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - eval_start).count());
            found->second.cpu_compute_ns += compute_ns;
            ++found->second.cpu_compute_calls;
            sched->active_expert_cache_cpu_compute_ns += compute_ns;
            ++sched->active_expert_cache_cpu_compute_calls;
        }
    }
    ggml_active_expert_cache_record_compute(sched, split_backend);
    ggml_expert_prefill_record_compute(sched, split_backend);
    return GGML_STATUS_SUCCESS;
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {

    // Restore the graph's normal temporary expert buffers before preparing the
    // next evaluation.  CUDA has already consumed the previous pointer values;
    // this only changes the scheduler-side tensor metadata for this pass.
    ggml_active_expert_cache_reset_pass(sched);

    for (auto & item : sched->needs_sync) item = true;

    if (sched->is_async && sched->n_backends > 1 && sched->split_mode_graph && sched->has_reduce) {

        for (auto & s : sched->statuses) s = GGML_STATUS_SUCCESS;

        int first_reduce = -1;
        bool work_done = false;
#ifdef GGML_USE_OPENMP
        //This may not be available in old OpenMP versions
        //if (int nlevels = omp_get_max_active_levels(); nlevels < 2) {
        //    omp_set_max_active_levels(nlevels+1);
        //    //printf("%s: Setting omp max active levels to 2\n", __func__);
        //}
        bool has_cpu_work = false;
        for (int i = 0; i < sched->n_backends; ++i) {
            if (!sched->backend_splits[i].empty()) {
                auto split = sched->backend_splits[i].front();
                if (ggml_backend_is_cpu(sched->backends[split->backend_id])) {
                    // A CPU backend invokes its own OpenMP worker team. Run
                    // heterogeneous scheduler workers as std::threads so that
                    // nested OpenMP does not silently collapse the CPU MoE
                    // branch to one thread.
                    has_cpu_work = true;
                    break;
                }
            }
        }
        for (int i = 0; i < sched->n_splits; i++) {
            auto split = &sched->splits[i];
            if (split->graph.n_nodes == 1 && split->graph.nodes[0]->op == GGML_OP_REDUCE) {
                first_reduce = split->backend_id;
                break;
            }
        }

        if (!has_cpu_work) {
        #pragma omp parallel num_threads(sched->n_backends)
        {

            int last_reduce = first_reduce;
            int ith = omp_get_thread_num();

            struct ggml_backend_sched_split * splits = sched->splits;

            std::vector<int32_t> ids;
            std::vector<uint32_t> unique_ids;
            ggml_tensor * last_ids_tensor = nullptr;

            for (int i = 0; i < sched->n_splits; i++) {
#if IK_PRINT_TIMING
                int64_t tim1 = ggml_time_us();
#endif
                struct ggml_backend_sched_split * split = &splits[i];
                int split_backend_id = split->backend_id;
                ggml_backend_t split_backend = sched->backends[split_backend_id];

                bool needs_barrier = split->n_inputs > 0 || split->graph.nodes[0]->op == GGML_OP_REDUCE;

                if (needs_barrier) {
                    #pragma omp barrier
                }

                if (split->n_inputs > 0) {
                    int copy_thread = last_reduce >= 0 ? last_reduce : 0;
                    if (ith == copy_thread) {
                        ggml_backend_sched_copy_inputs(sched, split, sched->needs_sync, ids, unique_ids, last_ids_tensor);
                    }
                    #pragma omp barrier
                }

                if (ith == split_backend_id) {

                    sched->statuses[ith] = ggml_backend_sched_eval(sched, split_backend, split);

                    if (split->n_inputs > 0 && !sched->own_cpy[split_backend_id]) {
                        sched->needs_sync[split_backend_id] = true;
                    } else {
                        for (int j = 0; j < split->n_inputs; ++j) {
                            if (ggml_backend_buffer_is_host(split->inputs[j]->buffer)) {
                                sched->needs_sync[split_backend_id] = true;
                            }
                        }
                    }
                }

                if (split->graph.nodes[0]->op == GGML_OP_REDUCE && i < sched->n_splits - 1) {
                    last_reduce = split_backend_id;
                    if (ith == split_backend_id) {
                        auto node = split->graph.nodes[0];
                        int n = node->op_params[1];
                        for (int j = 0; j < n; ++j) {
                            if (node->src[j]) {
                                sched->needs_sync[j] = false;
                            }
                        }
                    }
                    #pragma omp barrier
                }

                // record the event of this copy
                if (split->n_inputs > 0) {
                    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                        ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy]);
                    }
                }
            }
        }
        work_done = true;
        }
#endif
        if (!work_done) {

        std::barrier barrier(sched->n_backends);
        auto compute = [sched, &barrier, first_reduce] (int ith) {

            struct ggml_backend_sched_split * splits = sched->splits;

            std::vector<int32_t> ids;
            std::vector<uint32_t> unique_ids;
            ggml_tensor * last_ids_tensor = nullptr;

            int last_reduce = first_reduce;

            for (int i = 0; i < sched->n_splits; i++) {
#if IK_PRINT_TIMING
                int64_t tim1 = ggml_time_us();
#endif
                struct ggml_backend_sched_split * split = &splits[i];
                int split_backend_id = split->backend_id;
                ggml_backend_t split_backend = sched->backends[split_backend_id];

                bool needs_barrier = split->n_inputs > 0 || split->graph.nodes[0]->op == GGML_OP_REDUCE;

                if (needs_barrier) {
                    barrier.arrive_and_wait();
                }

                if (split->n_inputs > 0) {
                    int copy_thread = last_reduce >= 0 ? last_reduce : 0;
                    if (ith == copy_thread) {
                        ggml_backend_sched_copy_inputs(sched, split, sched->needs_sync, ids, unique_ids, last_ids_tensor);
                    }
                    barrier.arrive_and_wait();
                }

                if (ith == split_backend_id) {

                    sched->statuses[ith] = ggml_backend_sched_eval(sched, split_backend, split);
                    if (split->n_inputs > 0 && !sched->own_cpy[split_backend_id]) {
                        sched->needs_sync[split_backend_id] = true;
                    } else {
                        for (int j = 0; j < split->n_inputs; ++j) {
                            if (ggml_backend_buffer_is_host(split->inputs[j]->buffer)) {
                                sched->needs_sync[split_backend_id] = true;
                            }
                        }
                    }
                }

                if (split->graph.nodes[0]->op == GGML_OP_REDUCE && i < sched->n_splits - 1) {
                    last_reduce = split_backend_id;
                    barrier.arrive_and_wait();
                    if (ith == split_backend_id) {
                        auto node = split->graph.nodes[0];
                        int n = node->op_params[1];
                        for (int j = 0; j < n; ++j) {
                            if (node->src[j]) {
                                sched->needs_sync[j] = false;
                            }
                        }
                    }
                }
                //if (needs_barrier) {
                //    barrier.arrive_and_wait();
                //}

                // record the event of this copy
                if (split->n_inputs > 0) {
                    if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                        printf("Recording event %d, %d\n", split_backend_id, sched->cur_copy);
                        ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy]);
                    }
                }
            }
        };

        for (int i = 0; i < sched->n_backends; ++i) sched->workers.emplace_back(compute, i);
        for (auto & w : sched->workers) w.join();
        sched->workers.clear();
        }
        for (auto status : sched->statuses) {
            if (status != GGML_STATUS_SUCCESS) {
                ggml_active_expert_cache_reset_pass(sched);
                return status;
            }
        }
        ggml_active_expert_cache_reset_pass(sched);
        ggml_backend_sched_evaluate_expert_hybrid_guard(sched);
        return GGML_STATUS_SUCCESS;

    }

    struct ggml_backend_sched_split * splits = sched->splits;

    std::vector<int32_t> ids;
    std::vector<uint32_t> unique_ids;
    ggml_tensor * last_ids_tensor = nullptr;

    // MoE expert prefetch; pre-scan the splits for expert-weight matmuls whose
    // weights live in a host mmap. Two behaviors, both page-cache warmers.
    //  - lookahead streams the next splits' expert tensors in full while split
    //    i computes (batch/PP graphs only, where most experts are hit)
    //  - selective enqueues just the selected expert slices of a split's
    //    up/gate/down tensors once its ids have been copied to the host
    struct moe_split_info {
        int split;
        int64_t n_tokens;
        std::vector<ggml_tensor *> host_weights; // originals (host mmap)
        std::vector<ggml_tensor *> nodes;        // MoE nodes computed on a host buffer
    };
    std::vector<moe_split_info> moe_infos;
    const bool moe_prefetch = ggml_moe_prefetch_enabled();
    static const size_t moe_ahead = [] {
        const char * env = getenv("GGML_MOE_PREFETCH_AHEAD");
        return env ? (size_t) std::max(0, atoi(env)) : (size_t) 3;
    }();
    if (moe_prefetch) {
        ggml_moe_prefetch_new_epoch();
        for (int i = 0; i < sched->n_splits; i++) {
            moe_split_info info;
            info.split = i;
            info.n_tokens = 0;
            for (int n = 0; n < splits[i].graph.n_nodes; ++n) {
                ggml_tensor * node = splits[i].graph.nodes[n];
                if (node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_MOE_FUSED_UP_GATE) {
                    continue;
                }
                ggml_tensor * node_ids = node->op == GGML_OP_MUL_MAT_ID ? node->src[2] : node->src[3];
                info.n_tokens = std::max(info.n_tokens, node_ids ? node_ids->ne[1] : 0);
                ggml_tensor * ws[2] = { node->src[0], node->op == GGML_OP_MOE_FUSED_UP_GATE ? node->src[1] : nullptr };
                bool node_on_host = false;
                for (ggml_tensor * w : ws) {
                    if (w && w->buffer && ggml_backend_buffer_is_host(w->buffer) &&
                            ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        info.host_weights.push_back(w);
                        node_on_host = true;
                    }
                }
                if (node_on_host) {
                    info.nodes.push_back(node);
                }
            }
            // in the offloaded case the node's weight srcs were rewritten to
            // device copies; the host originals arrive as split inputs
            for (int j = 0; j < splits[i].n_inputs; ++j) {
                ggml_tensor * input = splits[i].inputs[j];
                if (input->ne[2] > 1 && input->buffer && ggml_backend_buffer_is_host(input->buffer) &&
                        ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                    info.host_weights.push_back(input);
                }
            }
            if (!info.host_weights.empty()) {
                moe_infos.push_back(std::move(info));
            }
        }
    }
    size_t moe_next  = 0; // first moe_infos entry with split >= current split
    size_t moe_enq   = 0; // moe_infos entries already enqueued for lookahead

    for (int i = 0; i < sched->n_splits; i++) {
#if IK_PRINT_TIMING
        int64_t tim1 = ggml_time_us();
#endif
        struct ggml_backend_sched_split * split = &splits[i];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        if (moe_prefetch && !moe_infos.empty()) {
            while (moe_next < moe_infos.size() && moe_infos[moe_next].split < i) {
                moe_next++;
            }
            // keep the next `moe_ahead` MoE-bearing splits streaming in
            const size_t want_end = std::min(moe_next + moe_ahead, moe_infos.size());
            for (size_t k = std::max(moe_enq, moe_next); k < want_end; ++k) {
                if (moe_infos[k].n_tokens >= 32) { // batch graphs touch most experts (min batch offload)
                    for (ggml_tensor * w : moe_infos[k].host_weights) {
                        ggml_moe_prefetch_tensor(w);
                    }
                }
                moe_enq = k + 1;
            }
        }

        // copy the input tensors to the split backend
        ggml_backend_sched_copy_inputs(sched, split, sched->needs_sync, ids, unique_ids, last_ids_tensor);

        // ids are now final and host-visible; enqueue the selected expert
        // slices of this split's host-computed MoE matmuls (up/gate and down
        // share one ids tensor, so the down weights stream in during up/gate)
        if (moe_prefetch && moe_next < moe_infos.size() && moe_infos[moe_next].split == i) {
            for (ggml_tensor * node : moe_infos[moe_next].nodes) {
                ggml_moe_prefetch_node(node);
            }
        }

        if (split->n_inputs > 0 && !sched->own_cpy[split_backend_id]) {
            sched->needs_sync[split_backend_id] = true;
        } else {
            for (int j = 0; j < split->n_inputs; ++j) {
                if (ggml_backend_buffer_is_host(split->inputs[j]->buffer)) {
                    sched->needs_sync[split_backend_id] = true;
                }
            }
        }
        auto ec = ggml_backend_sched_eval(sched, split_backend, split);
        if (ec != GGML_STATUS_SUCCESS) {
            ggml_active_expert_cache_reset_pass(sched);
            return ec;
        }

        // the pages the lookahead streamer just read for this split are one-shot
        // streaming traffic; MADV_COLD them so the decode working set survives
        if (moe_prefetch && moe_next < moe_infos.size() && moe_infos[moe_next].split == i &&
                moe_infos[moe_next].n_tokens >= 32) {
            for (ggml_tensor * w : moe_infos[moe_next].host_weights) {
                ggml_moe_prefetch_cold(w);
            }
        }

        // record the event of this copy
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy]);
            }
        }
    }

    sched->cur_copy = (sched->cur_copy + 1) % sched->n_copies;

    ggml_active_expert_cache_reset_pass(sched);
    ggml_backend_sched_evaluate_expert_hybrid_guard(sched);
    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_is_cpu(backends[n_backends - 1])); // last backend must be CPU

    struct ggml_backend_sched * sched = new ggml_backend_sched{};

    for (int i = 0; i < (GGML_OP_COUNT + 31)/32; ++i) sched->op_offload[i] = 0xffffffff;

    sched->debug = getenv("GGML_SCHED_DEBUG") != NULL;
    if (const char * cache_slots = getenv("LLAMA_EXPERT_GPU_CACHE_SLOTS")) {
        char * end = nullptr;
        const long parsed = std::strtol(cache_slots, &end, 10);
        if (end != cache_slots && *end == '\0' && parsed >= 4 && parsed <= 64) {
            sched->active_expert_cache_slots = (int) parsed;
            fprintf(stderr, "ggml active-expert cache: requested %d slots per GPU\n", sched->active_expert_cache_slots);
        } else if (*cache_slots != '\0' && std::strcmp(cache_slots, "0") != 0) {
            fprintf(stderr, "ggml active-expert cache: ignoring invalid LLAMA_EXPERT_GPU_CACHE_SLOTS=%s (use 4..64)\n", cache_slots);
        }
    }
    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *)malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **)malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t nodes_size = graph_size + GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *)calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *)calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *)calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *)calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->context_buffer_size = GGML_SCHED_MAX_SPLITS*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *)malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *)calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));
        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);

    sched->workers.reserve(sched->n_backends);
    sched->statuses.resize(sched->n_backends, GGML_STATUS_SUCCESS);
    sched->backend_splits.resize(sched->n_backends);

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    for (int i = 0; i < sched->n_backends; ++i) {
        if (sched->input_memory_bufs[i]) {
            ggml_backend_buffer_free(sched->input_memory_bufs[i]);
        }
        if (sched->active_expert_caches[i].ready) {
            fprintf(stderr, "expert_cache_stats: {\"level\":\"vram\",\"device\":%d,"
                    "\"capacity_bytes\":%llu,\"resident_bytes\":%llu,\"route_id_bytes\":%llu,\"slots\":%d}\n",
                    i,
                    (unsigned long long) sched->active_expert_caches[i].capacity_bytes,
                    (unsigned long long) sched->active_expert_caches[i].allocated_bytes,
                    (unsigned long long) sched->active_expert_caches[i].ids_bytes,
                    sched->active_expert_caches[i].slots);
        }
        if (sched->expert_prefill_devices[i].ready) {
            fprintf(stderr,
                    "expert_prefill_stats: {\"level\":\"device\",\"device\":%d,"
                    "\"lanes\":2,\"lane_bytes\":%llu,\"allocated_bytes\":%llu}\n",
                    i,
                    (unsigned long long) sched->expert_prefill_devices[i].lane_bytes,
                    (unsigned long long) sched->expert_prefill_devices[i].allocated_bytes);
        }
        ggml_expert_prefill_release_device(sched, sched->expert_prefill_devices[i]);
        if (sched->expert_prefill_route_events[i]) {
            ggml_backend_event_synchronize(sched->expert_prefill_route_events[i]);
            ggml_backend_event_free(sched->expert_prefill_route_events[i]);
            sched->expert_prefill_route_events[i] = nullptr;
        }
        ggml_active_expert_cache_drain_leases(sched, sched->active_expert_caches[i]);
        ggml_active_expert_cache_release(sched->active_expert_caches[i]);
    }
    if (!sched->expert_prefill_layer_stats.empty() || sched->expert_prefill_fallbacks > 0) {
        uint64_t selected = 0;
        uint64_t route_readback_ns = 0;
        uint64_t h2d_components = 0;
        uint64_t d2d_components = 0;
        uint64_t h2d_batches = 0;
        uint64_t d2d_batches = 0;
        uint64_t h2d_bytes = 0;
        uint64_t d2d_bytes = 0;
        for (const auto & item : sched->expert_prefill_layer_stats) {
            const auto & stats = item.second;
            route_readback_ns += stats.route_readback_ns;
            selected += stats.selected_components;
            h2d_components += stats.h2d_components;
            d2d_components += stats.d2d_components;
            h2d_batches += stats.h2d_batches;
            d2d_batches += stats.d2d_batches;
            h2d_bytes += stats.h2d_bytes;
            d2d_bytes += stats.d2d_bytes;
            fprintf(stderr,
                    "expert_prefill_stats: {\"level\":\"layer\",\"layer\":%d,"
                    "\"route_readback_ns\":%llu,\"selected_components\":%llu,"
                    "\"h2d_components\":%llu,"
                    "\"d2d_components\":%llu,\"h2d_batches\":%llu,\"d2d_batches\":%llu,"
                    "\"h2d_bytes\":%llu,\"d2d_bytes\":%llu,"
                    "\"lease_acquire_ns\":%llu,\"transfer_submit_ns\":%llu,"
                    "\"transfer_wait_ns\":%llu,\"fallbacks\":%llu}\n",
                    item.first,
                    (unsigned long long) stats.route_readback_ns,
                    (unsigned long long) stats.selected_components,
                    (unsigned long long) stats.h2d_components,
                    (unsigned long long) stats.d2d_components,
                    (unsigned long long) stats.h2d_batches,
                    (unsigned long long) stats.d2d_batches,
                    (unsigned long long) stats.h2d_bytes,
                    (unsigned long long) stats.d2d_bytes,
                    (unsigned long long) stats.lease_acquire_ns,
                    (unsigned long long) stats.transfer_submit_ns,
                    (unsigned long long) stats.transfer_wait_ns,
                    (unsigned long long) stats.fallbacks);
        }
        fprintf(stderr,
                "expert_prefill_stats: {\"level\":\"total\",\"selected_components\":%llu,"
                "\"h2d_components\":%llu,\"d2d_components\":%llu,"
                "\"h2d_batches\":%llu,\"d2d_batches\":%llu,"
                "\"h2d_bytes\":%llu,\"d2d_bytes\":%llu,\"route_readback_ns\":%llu,"
                "\"route_global_sync_fallbacks\":%llu,\"fallbacks\":%llu}\n",
                (unsigned long long) selected,
                (unsigned long long) h2d_components,
                (unsigned long long) d2d_components,
                (unsigned long long) h2d_batches,
                (unsigned long long) d2d_batches,
                (unsigned long long) h2d_bytes,
                (unsigned long long) d2d_bytes,
                (unsigned long long) route_readback_ns,
                (unsigned long long) sched->expert_prefill_route_global_sync_fallbacks,
                (unsigned long long) sched->expert_prefill_fallbacks);
    }
    if (sched->active_expert_cache_slots >= 1) {
        uint64_t cpu_compute_ns = 0;
        uint64_t cpu_compute_calls = 0;
        for (const auto & item : sched->active_expert_cache_layer_stats) {
            const auto & stats = item.second;
            cpu_compute_ns += stats.cpu_compute_ns;
            cpu_compute_calls += stats.cpu_compute_calls;
            fprintf(stderr, "expert_cache_stats: {\"level\":\"vram-layer\",\"layer\":%d,"
                    "\"routes\":%llu,\"route_positions\":%llu,\"gpu_route_positions\":%llu,"
                    "\"route_readback_ns\":%llu,\"hits\":%llu,\"misses\":%llu,"
                    "\"lease_acquire_ns\":%llu,\"lease_uploads\":%llu,"
                    "\"transfer_submit_ns\":%llu,\"transfer_wait_ns\":%llu,\"load_bytes\":%llu,"
                    "\"cpu_compute_ns\":%llu,\"cpu_compute_calls\":%llu}\n",
                    item.first,
                    (unsigned long long) stats.routes,
                    (unsigned long long) stats.route_positions,
                    (unsigned long long) stats.gpu_route_positions,
                    (unsigned long long) stats.route_readback_ns,
                    (unsigned long long) stats.hits,
                    (unsigned long long) stats.misses,
                    (unsigned long long) stats.lease_acquire_ns,
                    (unsigned long long) stats.lease_uploads,
                    (unsigned long long) stats.transfer_submit_ns,
                    (unsigned long long) stats.transfer_wait_ns,
                    (unsigned long long) stats.load_bytes,
                    (unsigned long long) stats.cpu_compute_ns,
                    (unsigned long long) stats.cpu_compute_calls);
        }
        fprintf(stderr, "ggml active-expert cache: %llu hits, %llu misses, %llu host-to-GPU expert uploads\n",
                (unsigned long long) sched->active_expert_cache_hits,
                (unsigned long long) sched->active_expert_cache_misses,
                (unsigned long long) sched->active_expert_cache_uploads);
        if (sched->expert_lease_acquire) {
            fprintf(stderr, "ggml active-expert cache: %llu lease-backed uploads, %llu forced host-tensor fallbacks\n",
                    (unsigned long long) sched->active_expert_cache_lease_uploads,
                    (unsigned long long) sched->active_expert_cache_forced_fallbacks);
        }
        fprintf(stderr, "ggml active-expert cache: %llu admissions deferred by minimum-observation hysteresis\n",
                (unsigned long long) sched->active_expert_cache_rejected_admissions);
        fprintf(stderr, "expert_cache_stats: {\"level\":\"vram-total\",\"hits\":%llu,"
                "\"misses\":%llu,\"admissions\":%llu,\"evictions\":%llu,"
                "\"uploads\":%llu,\"lease_uploads\":%llu,"
                "\"forced_fallbacks\":%llu,\"rejected_admissions\":%llu,"
                "\"transfer_submit_ns\":%llu,\"transfer_wait_ns\":%llu,"
                "\"route_observations\":%llu,\"route_prediction_matches\":%llu,\"prediction_admission_contributions\":%llu,"
                "\"reuse_distance_sum\":%llu,\"load_bytes\":%llu,\"eviction_cost_bytes\":%llu,"
                "\"cpu_compute_ns\":%llu,\"cpu_compute_calls\":%llu}\n",
                (unsigned long long) sched->active_expert_cache_hits,
                (unsigned long long) sched->active_expert_cache_misses,
                (unsigned long long) sched->active_expert_cache_admissions,
                (unsigned long long) sched->active_expert_cache_evictions,
                (unsigned long long) sched->active_expert_cache_uploads,
                (unsigned long long) sched->active_expert_cache_lease_uploads,
                (unsigned long long) sched->active_expert_cache_forced_fallbacks,
                (unsigned long long) sched->active_expert_cache_rejected_admissions,
                (unsigned long long) sched->active_expert_cache_transfer_submit_ns,
                (unsigned long long) sched->active_expert_cache_transfer_wait_ns,
                (unsigned long long) sched->active_expert_cache_route_observations,
                (unsigned long long) sched->active_expert_cache_route_prediction_matches,
                (unsigned long long) sched->active_expert_cache_prediction_admissions,
                (unsigned long long) sched->active_expert_cache_reuse_distance_sum,
                (unsigned long long) sched->active_expert_cache_load_bytes,
                (unsigned long long) sched->active_expert_cache_eviction_cost_bytes,
                (unsigned long long) cpu_compute_ns,
                (unsigned long long) cpu_compute_calls);
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    delete sched;
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);
    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

static void ggml_sched_prepare_graph(ggml_backend_sched_t sched) {

    for (auto & item : sched->own_cpy   ) item = false;
    for (auto & item : sched->needs_sync) item = true;

    if (sched->split_mode_graph) {
        auto tensor_size = [] (const ggml_tensor * t) {
            auto nbytes = ggml_nbytes(t);
            nbytes = 256*((nbytes + 255)/256);
            return nbytes;
        };
        //auto tim1 = std::chrono::steady_clock::now();
        for (auto & b : sched->backend_splits) b.clear();
        for (int i = 0; i < sched->n_splits; i++) {
            sched->backend_splits[sched->splits[i].backend_id].push_back(&sched->splits[i]);
        }
        for (int backend_id = 0; backend_id < sched->n_backends; ++backend_id) {
            if (ggml_backend_is_cpu(ggml_backend_sched_get_backend(sched, backend_id))) continue;
            if (sched->backend_splits[backend_id].empty()) continue;
            size_t input_size = 0;
            size_t max_input_size = 0;
            int last_split = 0;
            bool can_alloc = true;
            for (int i = 0; i < int(sched->backend_splits[backend_id].size()); ++i) {
                auto split = sched->backend_splits[backend_id][i];
                if (split->n_inputs < 1) continue;
                size_t this_size = 0;
                for (int j = 0; j < split->n_inputs; ++j) {
                    if (!ggml_backend_buffer_is_host(split->inputs[j]->buffer)) {
                        this_size += tensor_size(split->inputs[j]);
                    }
                }
                if (input_size + this_size > sched->max_extra_alloc) {
                    if (i - last_split < 3) {
                        can_alloc = false;
                        break;
                    }
                    max_input_size = std::max(max_input_size, input_size);
                    input_size = 0;
                    last_split = i - 1;
                }
                input_size += this_size;
            }
            max_input_size = std::max(max_input_size, input_size);
            if (!can_alloc || !max_input_size) continue;
            if (sched->input_memory_bufs[backend_id] && sched->input_memory_bufs[backend_id]->size < max_input_size) {
                ggml_backend_buffer_free(sched->input_memory_bufs[backend_id]);
                sched->input_memory_bufs[backend_id] = nullptr;
            }
            if (!sched->input_memory_bufs[backend_id]) {
                sched->input_memory_bufs[backend_id] = ggml_backend_alloc_buffer(sched->backends[backend_id], max_input_size);
            }
            auto ptr = (char *)ggml_backend_buffer_get_base(sched->input_memory_bufs[backend_id]);
            input_size = 0;
            for (int i = 0; i < int(sched->backend_splits[backend_id].size()); ++i) {
                auto split = sched->backend_splits[backend_id][i];
                size_t this_size = 0;
                for (int j = 0; j < split->n_inputs; ++j) {
                    if (!ggml_backend_buffer_is_host(split->inputs[j]->buffer)) {
                        this_size += tensor_size(split->inputs[j]);
                    }
                }
                if (input_size + this_size > max_input_size) {
                    ptr = (char *)ggml_backend_buffer_get_base(sched->input_memory_bufs[backend_id]);
                    input_size = 0;
                }
                for (int j = 0; j < split->n_inputs; ++j) {
                    if (ggml_backend_buffer_is_host(split->inputs[j]->buffer)) continue;
                    auto input_cpy = tensor_copy(split->inputs[j], backend_id, sched->cur_copy);
                    for (int k = 0; k < split->graph.n_nodes; ++k) {
                        auto node = split->graph.nodes[k];
                        for (int l = 0; l < GGML_MAX_SRC; ++l) {
                            if (node->src[l] && node->src[l]->data == input_cpy->data) node->src[l]->data = ptr;
                        }
                    }
                    input_cpy->data = ptr;
                    ptr += tensor_size(split->inputs[j]);
                }
                input_size += this_size;
            }
            sched->needs_sync[backend_id] = false;
            sched->own_cpy[backend_id] = true;
        }
    }
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }
    ggml_sched_prepare_graph(sched);

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

int ggml_backend_sched_get_backend_idx(ggml_backend_sched_t sched, ggml_backend_buffer_t buffer) {
    if (!buffer || !buffer->buft) return -1;
    if (buffer && buffer->buft) {
        for (int i = 0; i < sched->n_backends; ++i) {
            if (ggml_backend_get_default_buffer_type(sched->backends[i]) == buffer->buft) return i;
        }
    }
    return -1;
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        if (node->buffer && node->buffer->buft) {
            for (int i = 0; i < sched->n_backends; ++i) {
                if (sched->backends[i]->iface.get_default_buffer_type(sched->backends[i]) == node->buffer->buft) {
                    return sched->backends[i];
                }
            }
        }
        return nullptr;
    }
    return sched->backends[backend_index];
}

// utils

void ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

void ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT((char *)addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        ggml_backend_view_init(dst);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **)calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *)calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        fprintf(stderr, "failed to allocate context for graph copy\n");
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        fprintf(stderr, "failed to allocate buffer for graph copy\n");
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    for (int i = 0; i < g1->n_nodes; i++) {
        //printf("eval %d/%d\n", i, g1->n_nodes);
        struct ggml_tensor * t1 = g1->nodes[i];
        struct ggml_tensor * t2 = g2->nodes[i];

        assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

        struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
        struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

        ggml_backend_graph_compute(backend1, &g1v);
        ggml_backend_graph_compute(backend2, &g2v);

        if (ggml_is_view_op(t1->op)) {
            continue;
        }

        // compare results, calculate rms etc
        if (!callback(i, t1, t2, user_data)) {
            break;
        }
    }

    ggml_backend_graph_copy_free(copy);

    return true;
}
