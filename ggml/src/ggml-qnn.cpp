#include "ggml-qnn.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <QnnInterface.h>
#include <QnnTypes.h>

#include <dlfcn.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// This backend deliberately uses QNN client buffers first. On Android these are
// valid with HTP and give us a correctness-first path without making the ggml
// allocator depend on rpcmem/ION/DMABUF details. The Android app can therefore
// run with the ordinary QAIRT redistributables. A shared-buffer fast path can be
// layered on top later without changing the ggml backend ABI.

namespace {

static std::mutex g_status_mutex;
static std::string g_status = "QNN backend not probed";

static void set_status(const std::string & s) {
    std::lock_guard<std::mutex> lock(g_status_mutex);
    g_status = s;
}

static const char * qnn_backend_library() {
    const char * env = std::getenv("GGML_QNN_BACKEND_LIB");
    return env && *env ? env : "libQnnHtp.so";
}

static uint16_t fp32_to_fp16(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));

    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x007fffffu;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) {
            return static_cast<uint16_t>(sign);
        }
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        if (mantissa & 0x00001000u) {
            mantissa += 0x00002000u;
        }
        return static_cast<uint16_t>(sign | (mantissa >> 13));
    }

    if (exponent >= 31) {
        if (((x >> 23) & 0xffu) == 0xffu && mantissa != 0) {
            return static_cast<uint16_t>(sign | 0x7c00u | (mantissa >> 13) | 1u);
        }
        return static_cast<uint16_t>(sign | 0x7c00u);
    }

    if (mantissa & 0x00001000u) {
        mantissa += 0x00002000u;
        if (mantissa & 0x00800000u) {
            mantissa = 0;
            ++exponent;
            if (exponent >= 31) {
                return static_cast<uint16_t>(sign | 0x7c00u);
            }
        }
    }

    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

static float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t out;

    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            exponent = 1;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x03ffu;
            const uint32_t e = exponent + (127 - 15);
            out = sign | (e << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        const uint32_t e = exponent + (127 - 15);
        out = sign | (e << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

static bool tensor_is_plain_2d(const ggml_tensor * t) {
    if (!t || !t->data || !ggml_is_contiguous(t)) {
        return false;
    }
    if (t->ne[2] != 1 || t->ne[3] != 1) {
        return false;
    }
    return t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_F32;
}

static void tensor_to_half(const ggml_tensor * src, uint16_t * dst, size_t count) {
    if (src->type == GGML_TYPE_F16) {
        std::memcpy(dst, src->data, count * sizeof(uint16_t));
        return;
    }

    const float * in = static_cast<const float *>(src->data);
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp32_to_fp16(in[i]);
    }
}

static void half_to_tensor(const uint16_t * src, ggml_tensor * dst, size_t count) {
    if (dst->type == GGML_TYPE_F16) {
        std::memcpy(dst->data, src, count * sizeof(uint16_t));
        return;
    }

    float * out = static_cast<float *>(dst->data);
    for (size_t i = 0; i < count; ++i) {
        out[i] = fp16_to_fp32(src[i]);
    }
}

struct matmul_key {
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t m = 0;

    bool operator==(const matmul_key & other) const {
        return n == other.n && k == other.k && m == other.m;
    }
};

struct matmul_key_hash {
    size_t operator()(const matmul_key & key) const {
        uint64_t x = (static_cast<uint64_t>(key.n) << 42) ^
                     (static_cast<uint64_t>(key.k) << 21) ^
                      static_cast<uint64_t>(key.m);
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        return static_cast<size_t>(x);
    }
};

struct qnn_matmul_graph {
    Qnn_GraphHandle_t graph = nullptr;

    uint32_t a_dims[2] = {};
    uint32_t b_dims[2] = {};
    uint32_t o_dims[2] = {};

    Qnn_Tensor_t graph_a = QNN_TENSOR_INIT;
    Qnn_Tensor_t graph_b = QNN_TENSOR_INIT;
    Qnn_Tensor_t graph_o = QNN_TENSOR_INIT;

    std::vector<uint16_t> a;
    std::vector<uint16_t> b;
    std::vector<uint16_t> o;

    std::mutex execute_mutex;
};

class qnn_runtime {
public:
    ~qnn_runtime() {
        shutdown();
    }

    bool initialize() {
        if (ready_) {
            return true;
        }

        lib_ = dlopen(qnn_backend_library(), RTLD_NOW | RTLD_LOCAL);
        if (!lib_) {
            set_status(std::string("QNN HTP unavailable: ") + (dlerror() ? dlerror() : "dlopen failed"));
            return false;
        }

        using get_providers_fn = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
        auto get_providers = reinterpret_cast<get_providers_fn>(dlsym(lib_, "QnnInterface_getProviders"));
        if (!get_providers) {
            set_status("QNN HTP unavailable: QnnInterface_getProviders missing");
            shutdown();
            return false;
        }

        const QnnInterface_t ** providers = nullptr;
        uint32_t provider_count = 0;
        if (get_providers(&providers, &provider_count) != QNN_SUCCESS || !providers || provider_count == 0) {
            set_status("QNN HTP unavailable: no interface providers");
            shutdown();
            return false;
        }

        bool found = false;
        for (uint32_t i = 0; i < provider_count; ++i) {
            const QnnInterface_t * provider = providers[i];
            if (!provider) {
                continue;
            }
            if (provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
                provider->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
                interface_ = provider->QNN_INTERFACE_VER_NAME;
                found = true;
                break;
            }
        }

        if (!found) {
            set_status("QNN HTP unavailable: incompatible QNN API provider");
            shutdown();
            return false;
        }

        if (!interface_.backendCreate || !interface_.contextCreate || !interface_.graphCreate ||
            !interface_.tensorCreateGraphTensor || !interface_.graphAddNode ||
            !interface_.graphFinalize || !interface_.graphExecute) {
            set_status("QNN HTP unavailable: required graph API is missing");
            shutdown();
            return false;
        }

        const Qnn_ErrorHandle_t backend_error = interface_.backendCreate(nullptr, nullptr, &backend_);
        if (QNN_GET_ERROR_CODE(backend_error) != QNN_SUCCESS || !backend_) {
            set_status("QNN HTP unavailable: backendCreate failed");
            shutdown();
            return false;
        }

        const Qnn_ErrorHandle_t context_error = interface_.contextCreate(backend_, nullptr, nullptr, &context_);
        if (QNN_GET_ERROR_CODE(context_error) != QNN_SUCCESS || !context_) {
            set_status("QNN HTP unavailable: contextCreate failed");
            shutdown();
            return false;
        }

        ready_ = true;
        set_status(std::string("QNN HTP ready via ") + qnn_backend_library());
        return true;
    }

    bool ready() const {
        return ready_;
    }

    bool execute_matmul(const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * output) {
        if (!ready_ || !tensor_is_plain_2d(weight) || !tensor_is_plain_2d(input) || !tensor_is_plain_2d(output)) {
            return false;
        }

        const int64_t k64 = weight->ne[0];
        const int64_t m64 = weight->ne[1];
        const int64_t n64 = input->ne[1];
        if (input->ne[0] != k64 || output->ne[0] != m64 || output->ne[1] != n64 ||
            k64 <= 0 || m64 <= 0 || n64 <= 0 ||
            k64 > std::numeric_limits<uint32_t>::max() ||
            m64 > std::numeric_limits<uint32_t>::max() ||
            n64 > std::numeric_limits<uint32_t>::max()) {
            return false;
        }

        const matmul_key key {
            static_cast<uint32_t>(n64),
            static_cast<uint32_t>(k64),
            static_cast<uint32_t>(m64),
        };

        std::shared_ptr<qnn_matmul_graph> compiled = get_or_create_matmul(key);
        if (!compiled) {
            return false;
        }

        std::lock_guard<std::mutex> execute_lock(compiled->execute_mutex);

        const size_t n = key.n;
        const size_t k = key.k;
        const size_t m = key.m;

        tensor_to_half(input, compiled->a.data(), n * k);

        // ggml stores a mul_mat weight as M rows of K contiguous values while
        // QNN MatMul consumes [N,K] x [K,M]. Transpose the streamed/current
        // weight into the graph's reusable B staging buffer. Keeping this
        // staging bounded per shape is important for MoE models.
        if (weight->type == GGML_TYPE_F16) {
            const uint16_t * w = static_cast<const uint16_t *>(weight->data);
            for (size_t row = 0; row < m; ++row) {
                for (size_t col = 0; col < k; ++col) {
                    compiled->b[col * m + row] = w[row * k + col];
                }
            }
        } else {
            const float * w = static_cast<const float *>(weight->data);
            for (size_t row = 0; row < m; ++row) {
                for (size_t col = 0; col < k; ++col) {
                    compiled->b[col * m + row] = fp32_to_fp16(w[row * k + col]);
                }
            }
        }

        Qnn_Tensor_t inputs[2] = { compiled->graph_a, compiled->graph_b };
        Qnn_Tensor_t outputs[1] = { compiled->graph_o };

        inputs[0].v1.clientBuf = Qnn_ClientBuffer_t { compiled->a.data(), compiled->a.size() * sizeof(uint16_t) };
        inputs[1].v1.clientBuf = Qnn_ClientBuffer_t { compiled->b.data(), compiled->b.size() * sizeof(uint16_t) };
        outputs[0].v1.clientBuf = Qnn_ClientBuffer_t { compiled->o.data(), compiled->o.size() * sizeof(uint16_t) };

        const Qnn_ErrorHandle_t result = interface_.graphExecute(
            compiled->graph,
            inputs, 2,
            outputs, 1,
            nullptr,
            nullptr);

        if (QNN_GET_ERROR_CODE(result) != QNN_SUCCESS) {
            set_status("QNN HTP graphExecute failed");
            return false;
        }

        half_to_tensor(compiled->o.data(), output, n * m);
        return true;
    }

private:
    std::shared_ptr<qnn_matmul_graph> get_or_create_matmul(const matmul_key & key) {
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            auto it = matmuls_.find(key);
            if (it != matmuls_.end()) {
                return it->second;
            }
        }

        std::shared_ptr<qnn_matmul_graph> graph(new (std::nothrow) qnn_matmul_graph());
        if (!graph) {
            return nullptr;
        }

        graph->a_dims[0] = key.n;
        graph->a_dims[1] = key.k;
        graph->b_dims[0] = key.k;
        graph->b_dims[1] = key.m;
        graph->o_dims[0] = key.n;
        graph->o_dims[1] = key.m;

        graph->a.resize(static_cast<size_t>(key.n) * key.k);
        graph->b.resize(static_cast<size_t>(key.k) * key.m);
        graph->o.resize(static_cast<size_t>(key.n) * key.m);

        char graph_name[96];
        std::snprintf(graph_name, sizeof(graph_name), "ggml_matmul_%ux%ux%u", key.n, key.k, key.m);
        if (QNN_GET_ERROR_CODE(interface_.graphCreate(context_, graph_name, nullptr, &graph->graph)) != QNN_SUCCESS || !graph->graph) {
            set_status("QNN HTP graphCreate failed for MatMul");
            return nullptr;
        }

        graph->graph_a = make_tensor("a", QNN_TENSOR_TYPE_APP_WRITE, graph->a_dims, 2);
        graph->graph_b = make_tensor("b", QNN_TENSOR_TYPE_APP_WRITE, graph->b_dims, 2);
        graph->graph_o = make_tensor("out", QNN_TENSOR_TYPE_APP_READ, graph->o_dims, 2);

        if (QNN_GET_ERROR_CODE(interface_.tensorCreateGraphTensor(graph->graph, &graph->graph_a)) != QNN_SUCCESS ||
            QNN_GET_ERROR_CODE(interface_.tensorCreateGraphTensor(graph->graph, &graph->graph_b)) != QNN_SUCCESS ||
            QNN_GET_ERROR_CODE(interface_.tensorCreateGraphTensor(graph->graph, &graph->graph_o)) != QNN_SUCCESS) {
            set_status("QNN HTP tensorCreateGraphTensor failed for MatMul");
            return nullptr;
        }

        Qnn_Tensor_t op_inputs[2] = { graph->graph_a, graph->graph_b };
        Qnn_Tensor_t op_outputs[1] = { graph->graph_o };

        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1 = QNN_OPCONFIG_V1_INIT;
        op.v1.name = "ggml_matmul";
        op.v1.packageName = "qti.aisw";
        op.v1.typeName = "MatMul";
        op.v1.numOfParams = 0;
        op.v1.params = nullptr;
        op.v1.numOfInputs = 2;
        op.v1.inputTensors = op_inputs;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = op_outputs;

        if (QNN_GET_ERROR_CODE(interface_.graphAddNode(graph->graph, op)) != QNN_SUCCESS) {
            set_status("QNN HTP graphAddNode(MatMul) failed");
            return nullptr;
        }

        if (QNN_GET_ERROR_CODE(interface_.graphFinalize(graph->graph, nullptr, nullptr)) != QNN_SUCCESS) {
            set_status("QNN HTP graphFinalize(MatMul) failed");
            return nullptr;
        }

        std::lock_guard<std::mutex> lock(graph_mutex_);
        auto inserted = matmuls_.emplace(key, graph);
        if (!inserted.second) {
            return inserted.first->second;
        }
        return graph;
    }

    static Qnn_Tensor_t make_tensor(const char * name, Qnn_TensorType_t type, uint32_t * dims, uint32_t rank) {
        Qnn_Tensor_t tensor = QNN_TENSOR_INIT;
        tensor.version = QNN_TENSOR_VERSION_1;
        tensor.v1 = QNN_TENSOR_V1_INIT;
        tensor.v1.id = 0;
        tensor.v1.name = name;
        tensor.v1.type = type;
        tensor.v1.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
        tensor.v1.dataType = QNN_DATATYPE_FLOAT_16;
        tensor.v1.quantizeParams = QNN_QUANTIZE_PARAMS_INIT;
        tensor.v1.rank = rank;
        tensor.v1.dimensions = dims;
        tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        tensor.v1.clientBuf = Qnn_ClientBuffer_t { nullptr, 0 };
        return tensor;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            matmuls_.clear();
        }

        if (context_ && interface_.contextFree) {
            interface_.contextFree(context_, nullptr);
            context_ = nullptr;
        }
        if (backend_ && interface_.backendFree) {
            interface_.backendFree(backend_);
            backend_ = nullptr;
        }
        if (lib_) {
            dlclose(lib_);
            lib_ = nullptr;
        }
        ready_ = false;
    }

    void * lib_ = nullptr;
    QNN_INTERFACE_VER_TYPE interface_ = {};
    Qnn_BackendHandle_t backend_ = nullptr;
    Qnn_ContextHandle_t context_ = nullptr;
    bool ready_ = false;

    std::mutex graph_mutex_;
    std::unordered_map<matmul_key, std::shared_ptr<qnn_matmul_graph>, matmul_key_hash> matmuls_;
};

struct qnn_buffer_context {
    void * ptr = nullptr;
    size_t size = 0;
};

static const char * qnn_buffer_name(ggml_backend_buffer_t) {
    return "QNN-HTP host-visible";
}

static void qnn_buffer_free(ggml_backend_buffer_t buffer) {
    qnn_buffer_context * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    if (!ctx) {
        return;
    }
    std::free(ctx->ptr);
    delete ctx;
}

static void * qnn_buffer_base(ggml_backend_buffer_t buffer) {
    qnn_buffer_context * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    return ctx ? ctx->ptr : nullptr;
}

static void qnn_buffer_memset_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

static void qnn_buffer_set_tensor(ggml_backend_buffer_t, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

static void qnn_buffer_get_tensor(ggml_backend_buffer_t, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

static void qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    qnn_buffer_context * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    if (ctx && ctx->ptr && ctx->size) {
        std::memset(ctx->ptr, value, ctx->size);
    }
}

static ggml_backend_buffer_i qnn_buffer_i = {
    /* .get_name       = */ qnn_buffer_name,
    /* .free_buffer    = */ qnn_buffer_free,
    /* .get_base       = */ qnn_buffer_base,
    /* .init_tensor    = */ nullptr,
    /* .memset_tensor  = */ qnn_buffer_memset_tensor,
    /* .set_tensor     = */ qnn_buffer_set_tensor,
    /* .get_tensor     = */ qnn_buffer_get_tensor,
    /* .cpy_tensor     = */ nullptr,
    /* .clear          = */ qnn_buffer_clear,
    /* .reset          = */ nullptr,
};

static const char * qnn_buft_name(ggml_backend_buffer_type_t) {
    return "QNN-HTP";
}

static ggml_backend_buffer_t qnn_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    qnn_buffer_context * ctx = new (std::nothrow) qnn_buffer_context();
    if (!ctx) {
        return nullptr;
    }

    const size_t actual = std::max<size_t>(size, 64);
    void * ptr = nullptr;
    if (posix_memalign(&ptr, 64, actual) != 0 || !ptr) {
        delete ctx;
        return nullptr;
    }

    ctx->ptr = ptr;
    ctx->size = size;
    return ggml_backend_buffer_init(buft, qnn_buffer_i, ctx, size);
}

static size_t qnn_buft_alignment(ggml_backend_buffer_type_t) {
    return 64;
}

static size_t qnn_buft_max_size(ggml_backend_buffer_type_t) {
    return SIZE_MAX;
}

static bool qnn_buft_is_host(ggml_backend_buffer_type_t) {
    // Host-visible on purpose: unsupported ggml ops can be scheduled on CPU
    // without a second copy even when a layer's weights were assigned to QNN.
    return true;
}

static ggml_backend_buffer_type qnn_buft = {
    {
        qnn_buft_name,
        qnn_buft_alloc,
        qnn_buft_alignment,
        qnn_buft_max_size,
        nullptr,
        qnn_buft_is_host,
    },
    nullptr,
};

static ggml_guid_t qnn_guid() {
    static ggml_guid guid = { 0x71, 0x6e, 0x6e, 0x2d, 0x68, 0x74, 0x70, 0x2d, 0x73, 0x32, 0x35, 0x2d, 0x30, 0x30, 0x30, 0x31 };
    return &guid;
}

struct qnn_backend_context {
    std::unique_ptr<qnn_runtime> runtime;
};

static const char * qnn_backend_name(ggml_backend_t) {
    return "QNN-HTP";
}

static void qnn_backend_free(ggml_backend_t backend) {
    if (!backend) {
        return;
    }
    delete static_cast<qnn_backend_context *>(backend->context);
    delete backend;
}

static ggml_backend_buffer_type_t qnn_backend_default_buft(ggml_backend_t) {
    return &qnn_buft;
}

static bool qnn_backend_supports_op(ggml_backend_t, const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1]) {
        return false;
    }
    if (!tensor_is_plain_2d(op->src[0]) || !tensor_is_plain_2d(op->src[1]) || !tensor_is_plain_2d(op)) {
        return false;
    }
    return op->src[0]->ne[0] == op->src[1]->ne[0] &&
           op->ne[0] == op->src[0]->ne[1] &&
           op->ne[1] == op->src[1]->ne[1];
}

static bool qnn_backend_supports_buft(ggml_backend_t, ggml_backend_buffer_type_t buft) {
    return buft == &qnn_buft || ggml_backend_buft_is_host(buft);
}

static bool qnn_backend_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    if (!qnn_backend_supports_op(backend, op)) {
        return false;
    }

    int min_batch = 1;
    if (const char * env = std::getenv("GGML_QNN_MIN_BATCH")) {
        min_batch = std::max(1, std::atoi(env));
    }
    return op->src[1]->ne[1] >= min_batch;
}

static enum ggml_status qnn_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    qnn_backend_context * ctx = static_cast<qnn_backend_context *>(backend->context);
    if (!ctx || !ctx->runtime || !ctx->runtime->ready()) {
        return GGML_STATUS_FAILED;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (!qnn_backend_supports_op(backend, node)) {
            return GGML_STATUS_FAILED;
        }
        if (!ctx->runtime->execute_matmul(node->src[0], node->src[1], node)) {
            return GGML_STATUS_FAILED;
        }
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i qnn_backend_i = {
    /* .get_name                = */ qnn_backend_name,
    /* .free                    = */ qnn_backend_free,
    /* .get_default_buffer_type = */ qnn_backend_default_buft,
    /* .set_tensor_async        = */ nullptr,
    /* .get_tensor_async        = */ nullptr,
    /* .cpy_tensor_async        = */ nullptr,
    /* .synchronize             = */ nullptr,
    /* .graph_plan_create       = */ nullptr,
    /* .graph_plan_free         = */ nullptr,
    /* .graph_plan_update       = */ nullptr,
    /* .graph_plan_compute      = */ nullptr,
    /* .graph_compute           = */ qnn_backend_graph_compute,
    /* .supports_op             = */ qnn_backend_supports_op,
    /* .supports_buft           = */ qnn_backend_supports_buft,
    /* .offload_op              = */ qnn_backend_offload_op,
    /* .event_new               = */ nullptr,
    /* .event_free              = */ nullptr,
    /* .event_record            = */ nullptr,
    /* .event_wait              = */ nullptr,
    /* .event_synchronize       = */ nullptr,
};

static ggml_backend_t qnn_reg_init(const char *, void *) {
    return ggml_backend_qnn_init(0);
}

static std::once_flag qnn_register_once;

} // namespace

int ggml_backend_qnn_get_device_count(void) {
    qnn_runtime runtime;
    return runtime.initialize() ? 1 : 0;
}

void ggml_backend_qnn_reg_devices(void) {
    std::call_once(qnn_register_once, [] {
        qnn_runtime probe;
        if (!probe.initialize()) {
            return;
        }
        ggml_backend_register("QNN0", qnn_reg_init, &qnn_buft, nullptr);
    });
}

ggml_backend_t ggml_backend_qnn_init(int device) {
    if (device != 0) {
        set_status("QNN HTP: only device 0 is supported by the Android backend");
        return nullptr;
    }

    std::unique_ptr<qnn_runtime> runtime(new (std::nothrow) qnn_runtime());
    if (!runtime || !runtime->initialize()) {
        return nullptr;
    }

    qnn_backend_context * ctx = new (std::nothrow) qnn_backend_context();
    if (!ctx) {
        return nullptr;
    }
    ctx->runtime = std::move(runtime);

    ggml_backend * backend = new (std::nothrow) ggml_backend {
        qnn_guid(),
        qnn_backend_i,
        ctx,
    };
    if (!backend) {
        delete ctx;
        return nullptr;
    }
    return backend;
}

ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(int device) {
    return device == 0 ? &qnn_buft : nullptr;
}

bool ggml_backend_is_qnn(ggml_backend_t backend) {
    return backend && ggml_backend_guid(backend) == qnn_guid();
}

const char * ggml_backend_qnn_status(void) {
    std::lock_guard<std::mutex> lock(g_status_mutex);
    return g_status.c_str();
}
