#include "ggml-qnn.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "iqk/iqk_quantize.h"

#include <QnnInterface.h>
#include <QnnTypes.h>

#include <dlfcn.h>

#include <algorithm>
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
#include <vector>

// Android Qualcomm HTP backend for the Expert Streaming Engine.
//
// The model remains GGUF/mmap-backed. HTP receives only the operation currently
// being evaluated. For routed MoE decode, only router-selected experts are
// dequantized/staged, so QNN does not require a second full-model copy.

namespace {

std::mutex g_status_mutex;
std::string g_status = "QNN backend not probed";
std::mutex g_registration_mutex;
bool g_registered = false;

void set_status(const std::string & value) {
    std::lock_guard<std::mutex> lock(g_status_mutex);
    g_status = value;
}

const char * backend_library() {
    const char * value = std::getenv("GGML_QNN_BACKEND_LIB");
    return value && *value ? value : "libQnnHtp.so";
}

size_t max_stage_bytes() {
    long mib = 256;
    if (const char * value = std::getenv("GGML_QNN_MAX_STAGE_MIB")) {
        char * end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0' && parsed >= 16 && parsed <= 4096) {
            mib = parsed;
        }
    }
    return static_cast<size_t>(mib) * 1024u * 1024u;
}

bool safe_mul(size_t a, size_t b, size_t & out) {
    if (a != 0 && b > std::numeric_limits<size_t>::max()/a) return false;
    out = a*b;
    return true;
}

bool safe_add(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    out = a+b;
    return true;
}

bool stage_allowed(uint32_t n, uint32_t k, uint32_t m, size_t * total_elements = nullptr) {
    size_t nk, km, nm;
    if (!safe_mul(n, k, nk) || !safe_mul(k, m, km) || !safe_mul(n, m, nm)) return false;

    size_t total;
    if (!safe_add(nk, km, total) || !safe_add(total, nm, total)) return false;

    size_t bytes;
    if (!safe_mul(total, sizeof(uint16_t), bytes) || bytes > max_stage_bytes()) return false;

    // Keep each QNN client buffer representable by the portable 32-bit size.
    size_t a_bytes, b_bytes, o_bytes;
    if (!safe_mul(nk, sizeof(uint16_t), a_bytes) ||
        !safe_mul(km, sizeof(uint16_t), b_bytes) ||
        !safe_mul(nm, sizeof(uint16_t), o_bytes)) return false;
    if (a_bytes > UINT32_MAX || b_bytes > UINT32_MAX || o_bytes > UINT32_MAX) return false;

    if (total_elements) *total_elements = total;
    return true;
}

uint16_t f32_to_f16(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t mantissa = x & 0x007fffffu;
    int32_t exponent = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;

    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa = (mantissa | 0x00800000u) >> (1 - exponent);
        if (mantissa & 0x00001000u) mantissa += 0x00002000u;
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
            if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

float f16_to_f32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t out;

    if (exponent == 0) {
        if (mantissa == 0) {
            out = sign;
        } else {
            int32_t e = -14;
            while ((mantissa & 0x0400u) == 0) {
                mantissa <<= 1;
                --e;
            }
            mantissa &= 0x03ffu;
            out = sign | (static_cast<uint32_t>(e + 127) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        out = sign | 0x7f800000u | (mantissa << 13);
    } else {
        out = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    }

    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

bool activation_type(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

bool weight_type(enum ggml_type type) {
    return activation_type(type) || type == GGML_TYPE_MXFP4;
}

// supports_op() runs before graph allocation, so these helpers must not depend
// on tensor->data being non-null.
bool tensor_layout_2d(const ggml_tensor * tensor) {
    return tensor && activation_type(tensor->type) &&
           tensor->ne[0] > 0 && tensor->ne[1] > 0 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

bool tensor_ready_2d(const ggml_tensor * tensor) {
    return tensor && tensor->data && tensor_layout_2d(tensor);
}

bool weight_layout_2d(const ggml_tensor * tensor) {
    return tensor && weight_type(tensor->type) &&
           tensor->ne[0] > 0 && tensor->ne[1] > 0 &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

bool weight_ready_2d(const ggml_tensor * tensor) {
    return tensor && tensor->data && weight_layout_2d(tensor);
}

void activation_to_f16(const ggml_tensor * src, uint16_t * dst, size_t count) {
    if (src->type == GGML_TYPE_F16) {
        std::memcpy(dst, src->data, count*sizeof(uint16_t));
        return;
    }
    const float * values = static_cast<const float *>(src->data);
    for (size_t i = 0; i < count; ++i) dst[i] = f32_to_f16(values[i]);
}

void f16_to_output(const uint16_t * src, ggml_tensor * dst, size_t count) {
    if (dst->type == GGML_TYPE_F16) {
        std::memcpy(dst->data, src, count*sizeof(uint16_t));
        return;
    }
    float * values = static_cast<float *>(dst->data);
    for (size_t i = 0; i < count; ++i) values[i] = f16_to_f32(src[i]);
}

bool weight_to_transposed_f16(const ggml_tensor * weight, uint16_t * dst, size_t k, size_t m) {
    if (weight->type == GGML_TYPE_F16) {
        for (size_t row = 0; row < m; ++row) {
            const auto * src = reinterpret_cast<const uint16_t *>(
                static_cast<const char *>(weight->data) + row*weight->nb[1]);
            for (size_t col = 0; col < k; ++col) dst[col*m + row] = src[col];
        }
        return true;
    }

    if (weight->type == GGML_TYPE_F32) {
        for (size_t row = 0; row < m; ++row) {
            const auto * src = reinterpret_cast<const float *>(
                static_cast<const char *>(weight->data) + row*weight->nb[1]);
            for (size_t col = 0; col < k; ++col) dst[col*m + row] = f32_to_f16(src[col]);
        }
        return true;
    }

    if (weight->type == GGML_TYPE_MXFP4) {
        std::vector<float> row_buffer(k);
        for (size_t row = 0; row < m; ++row) {
            const auto * src = reinterpret_cast<const block_mxfp4 *>(
                static_cast<const char *>(weight->data) + row*weight->nb[1]);
            dequantize_row_mxfp4(src, row_buffer.data(), static_cast<int64_t>(k));
            for (size_t col = 0; col < k; ++col) dst[col*m + row] = f32_to_f16(row_buffer[col]);
        }
        return true;
    }

    return false;
}

struct matmul_key {
    uint32_t n, k, m;
    bool operator==(const matmul_key & rhs) const {
        return n == rhs.n && k == rhs.k && m == rhs.m;
    }
};

struct matmul_hash {
    size_t operator()(const matmul_key & key) const {
        uint64_t value = (static_cast<uint64_t>(key.n) << 42) ^
                         (static_cast<uint64_t>(key.k) << 21) ^ key.m;
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33;
        return static_cast<size_t>(value);
    }
};

struct qnn_matmul_graph {
    Qnn_GraphHandle_t graph = nullptr;
    uint32_t a_dims[2] = {};
    uint32_t b_dims[2] = {};
    uint32_t o_dims[2] = {};
    Qnn_Tensor_t a = QNN_TENSOR_INIT;
    Qnn_Tensor_t b = QNN_TENSOR_INIT;
    Qnn_Tensor_t out = QNN_TENSOR_INIT;
};

class qnn_runtime {
public:
    ~qnn_runtime() { shutdown(); }

    bool initialize() {
        if (ready_) return true;

        library_ = dlopen(backend_library(), RTLD_NOW | RTLD_LOCAL);
        if (!library_) {
            const char * error = dlerror();
            set_status(std::string("QNN HTP unavailable: ") + (error ? error : "dlopen failed"));
            return false;
        }

        using get_providers_fn = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
        auto get_providers = reinterpret_cast<get_providers_fn>(
            dlsym(library_, "QnnInterface_getProviders"));
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

        bool compatible = false;
        for (uint32_t i = 0; i < provider_count; ++i) {
            const QnnInterface_t * provider = providers[i];
            if (provider &&
                provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
                provider->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
                api_ = provider->QNN_INTERFACE_VER_NAME;
                compatible = true;
                break;
            }
        }
        if (!compatible) {
            set_status("QNN HTP unavailable: incompatible QNN API provider");
            shutdown();
            return false;
        }

        if (!api_.backendCreate || !api_.backendFree || !api_.contextCreate || !api_.contextFree ||
            !api_.graphCreate || !api_.tensorCreateGraphTensor || !api_.graphAddNode ||
            !api_.graphFinalize || !api_.graphExecute) {
            set_status("QNN HTP unavailable: required graph API missing");
            shutdown();
            return false;
        }

        Qnn_ErrorHandle_t error = api_.backendCreate(nullptr, nullptr, &backend_);
        if (QNN_GET_ERROR_CODE(error) != QNN_SUCCESS || !backend_) {
            set_status("QNN HTP unavailable: backendCreate failed");
            shutdown();
            return false;
        }

        error = api_.contextCreate(backend_, nullptr, nullptr, &context_);
        if (QNN_GET_ERROR_CODE(error) != QNN_SUCCESS || !context_) {
            set_status("QNN HTP unavailable: contextCreate failed");
            shutdown();
            return false;
        }

        ready_ = true;
        char text[192];
        std::snprintf(text, sizeof(text), "QNN HTP ready via %s (staging cap %.0f MiB)",
                      backend_library(), max_stage_bytes()/(1024.0*1024.0));
        set_status(text);
        return true;
    }

    bool ready() const { return ready_; }

    bool matmul(const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * output) {
        if (!ready_ || !weight_ready_2d(weight) || !tensor_ready_2d(input) || !tensor_ready_2d(output)) {
            return false;
        }

        const int64_t k64 = weight->ne[0];
        const int64_t m64 = weight->ne[1];
        const int64_t n64 = input->ne[1];
        if (input->ne[0] != k64 || output->ne[0] != m64 || output->ne[1] != n64 ||
            k64 <= 0 || m64 <= 0 || n64 <= 0 ||
            k64 > UINT32_MAX || m64 > UINT32_MAX || n64 > UINT32_MAX) return false;

        const matmul_key key {
            static_cast<uint32_t>(n64),
            static_cast<uint32_t>(k64),
            static_cast<uint32_t>(m64),
        };

        size_t total_elements;
        if (!stage_allowed(key.n, key.k, key.m, &total_elements)) return false;

        std::shared_ptr<qnn_matmul_graph> graph = get_matmul_graph(key);
        if (!graph) return false;

        // A single arena is reused for all graph shapes. It grows only to the
        // largest operation allowed by GGML_QNN_MAX_STAGE_MIB.
        std::lock_guard<std::mutex> lock(execute_mutex_);
        staging_.resize(total_elements);

        const size_t nk = static_cast<size_t>(key.n)*key.k;
        const size_t km = static_cast<size_t>(key.k)*key.m;
        const size_t nm = static_cast<size_t>(key.n)*key.m;
        uint16_t * a = staging_.data();
        uint16_t * b = a + nk;
        uint16_t * out = b + km;

        activation_to_f16(input, a, nk);
        if (!weight_to_transposed_f16(weight, b, key.k, key.m)) return false;

        Qnn_Tensor_t inputs[2] = { graph->a, graph->b };
        Qnn_Tensor_t outputs[1] = { graph->out };
        inputs[0].v1.clientBuf = Qnn_ClientBuffer_t {
            a, static_cast<uint32_t>(nk*sizeof(uint16_t))
        };
        inputs[1].v1.clientBuf = Qnn_ClientBuffer_t {
            b, static_cast<uint32_t>(km*sizeof(uint16_t))
        };
        outputs[0].v1.clientBuf = Qnn_ClientBuffer_t {
            out, static_cast<uint32_t>(nm*sizeof(uint16_t))
        };

        const Qnn_ErrorHandle_t error = api_.graphExecute(
            graph->graph, inputs, 2, outputs, 1, nullptr, nullptr);
        if (QNN_GET_ERROR_CODE(error) != QNN_SUCCESS) {
            set_status("QNN HTP graphExecute failed");
            return false;
        }

        f16_to_output(out, output, nm);
        return true;
    }

    bool mul_mat_id(const ggml_tensor * weights, const ggml_tensor * input,
                    const ggml_tensor * ids, ggml_tensor * output) {
        if (!ready_ || !weights || !weights->data || !weight_type(weights->type) ||
            !tensor_ready_2d(input) || !ids || !ids->data || ids->type != GGML_TYPE_I32 ||
            !output || !output->data || !activation_type(output->type)) return false;

        // Decode path only. Large prompt batches continue to use the fork's
        // established CPU MoE path, which avoids one HTP launch per routing row.
        if (input->ne[1] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
            ids->ne[1] != 1 || ids->ne[0] <= 0 || ids->ne[0] > 64 ||
            output->ne[1] != 1 || output->ne[2] < ids->ne[0] || output->ne[3] != 1) return false;

        const int64_t k = weights->ne[0];
        const int64_t m = weights->ne[1];
        const int64_t n_experts = weights->ne[2];
        if (input->ne[0] != k || output->ne[0] != m || n_experts <= 0 ||
            k <= 0 || m <= 0 || k > UINT32_MAX || m > UINT32_MAX) return false;
        if (!stage_allowed(1, static_cast<uint32_t>(k), static_cast<uint32_t>(m))) return false;

        std::memset(output->data, 0, ggml_nbytes(output));

        for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
            const int32_t expert = *reinterpret_cast<const int32_t *>(
                static_cast<const char *>(ids->data) + slot*ids->nb[0]);
            if (expert < 0 || expert >= n_experts) continue;

            ggml_tensor selected = *weights;
            selected.data = static_cast<char *>(weights->data) + static_cast<size_t>(expert)*weights->nb[2];
            selected.ne[2] = 1;
            selected.ne[3] = 1;

            ggml_tensor selected_output = *output;
            selected_output.data = static_cast<char *>(output->data) + static_cast<size_t>(slot)*output->nb[2];
            selected_output.ne[1] = 1;
            selected_output.ne[2] = 1;
            selected_output.ne[3] = 1;
            selected_output.nb[1] = static_cast<size_t>(m)*ggml_type_size(output->type);
            selected_output.nb[2] = selected_output.nb[1];
            selected_output.nb[3] = selected_output.nb[1];

            if (!matmul(&selected, input, &selected_output)) return false;
        }

        return true;
    }

private:
    static Qnn_Tensor_t make_tensor(const char * name, Qnn_TensorType_t type,
                                    uint32_t * dimensions, uint32_t rank) {
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
        tensor.v1.dimensions = dimensions;
        tensor.v1.memType = QNN_TENSORMEMTYPE_RAW;
        tensor.v1.clientBuf = Qnn_ClientBuffer_t { nullptr, 0 };
        return tensor;
    }

    std::shared_ptr<qnn_matmul_graph> get_matmul_graph(const matmul_key & key) {
        // QNN graph creation mutates a shared context; serialize creation and
        // cache finalized graphs by logical shape.
        std::lock_guard<std::mutex> lock(graph_mutex_);
        auto found = matmuls_.find(key);
        if (found != matmuls_.end()) return found->second;

        auto graph = std::make_shared<qnn_matmul_graph>();
        graph->a_dims[0] = key.n;
        graph->a_dims[1] = key.k;
        graph->b_dims[0] = key.k;
        graph->b_dims[1] = key.m;
        graph->o_dims[0] = key.n;
        graph->o_dims[1] = key.m;

        char name[96];
        std::snprintf(name, sizeof(name), "ggml_matmul_%ux%ux%u", key.n, key.k, key.m);
        if (QNN_GET_ERROR_CODE(api_.graphCreate(context_, name, nullptr, &graph->graph)) != QNN_SUCCESS ||
            !graph->graph) {
            set_status("QNN HTP graphCreate failed for MatMul");
            return nullptr;
        }

        graph->a   = make_tensor("a",   QNN_TENSOR_TYPE_APP_WRITE, graph->a_dims, 2);
        graph->b   = make_tensor("b",   QNN_TENSOR_TYPE_APP_WRITE, graph->b_dims, 2);
        graph->out = make_tensor("out", QNN_TENSOR_TYPE_APP_READ,  graph->o_dims, 2);

        if (QNN_GET_ERROR_CODE(api_.tensorCreateGraphTensor(graph->graph, &graph->a)) != QNN_SUCCESS ||
            QNN_GET_ERROR_CODE(api_.tensorCreateGraphTensor(graph->graph, &graph->b)) != QNN_SUCCESS ||
            QNN_GET_ERROR_CODE(api_.tensorCreateGraphTensor(graph->graph, &graph->out)) != QNN_SUCCESS) {
            set_status("QNN HTP tensorCreateGraphTensor failed for MatMul");
            return nullptr;
        }

        Qnn_Tensor_t inputs[2] = { graph->a, graph->b };
        Qnn_Tensor_t outputs[1] = { graph->out };
        Qnn_OpConfig_t op = QNN_OPCONFIG_INIT;
        op.version = QNN_OPCONFIG_VERSION_1;
        op.v1 = QNN_OPCONFIG_V1_INIT;
        op.v1.name = "ggml_matmul";
        op.v1.packageName = "qti.aisw";
        op.v1.typeName = "MatMul";
        op.v1.numOfParams = 0;
        op.v1.params = nullptr;
        op.v1.numOfInputs = 2;
        op.v1.inputTensors = inputs;
        op.v1.numOfOutputs = 1;
        op.v1.outputTensors = outputs;

        if (QNN_GET_ERROR_CODE(api_.graphAddNode(graph->graph, op)) != QNN_SUCCESS) {
            set_status("QNN HTP graphAddNode(MatMul) failed");
            return nullptr;
        }
        if (QNN_GET_ERROR_CODE(api_.graphFinalize(graph->graph, nullptr, nullptr)) != QNN_SUCCESS) {
            set_status("QNN HTP graphFinalize(MatMul) failed");
            return nullptr;
        }

        matmuls_.emplace(key, graph);
        return graph;
    }

    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            matmuls_.clear();
        }
        staging_.clear();
        staging_.shrink_to_fit();

        if (context_ && api_.contextFree) {
            api_.contextFree(context_, nullptr);
            context_ = nullptr;
        }
        if (backend_ && api_.backendFree) {
            api_.backendFree(backend_);
            backend_ = nullptr;
        }
        if (library_) {
            dlclose(library_);
            library_ = nullptr;
        }
        ready_ = false;
    }

    void * library_ = nullptr;
    QNN_INTERFACE_VER_TYPE api_ = {};
    Qnn_BackendHandle_t backend_ = nullptr;
    Qnn_ContextHandle_t context_ = nullptr;
    bool ready_ = false;

    std::mutex graph_mutex_;
    std::unordered_map<matmul_key, std::shared_ptr<qnn_matmul_graph>, matmul_hash> matmuls_;
    std::mutex execute_mutex_;
    std::vector<uint16_t> staging_;
};

// ggml buffer facade. Backing storage is host-visible for correctness/fallback,
// but is_host=false intentionally identifies QNN as an accelerator backend.
struct qnn_buffer_context {
    void * ptr = nullptr;
    size_t size = 0;
};

const char * qnn_buffer_name(ggml_backend_buffer_t) { return "QNN-HTP"; }

void qnn_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    if (!ctx) return;
    std::free(ctx->ptr);
    delete ctx;
}

void * qnn_buffer_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    return ctx ? ctx->ptr : nullptr;
}

void qnn_buffer_memset(ggml_backend_buffer_t, ggml_tensor * tensor,
                       uint8_t value, size_t offset, size_t size) {
    std::memset(static_cast<char *>(tensor->data) + offset, value, size);
}

void qnn_buffer_set(ggml_backend_buffer_t, ggml_tensor * tensor,
                    const void * data, size_t offset, size_t size) {
    std::memcpy(static_cast<char *>(tensor->data) + offset, data, size);
}

void qnn_buffer_get(ggml_backend_buffer_t, const ggml_tensor * tensor,
                    void * data, size_t offset, size_t size) {
    std::memcpy(data, static_cast<const char *>(tensor->data) + offset, size);
}

void qnn_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    if (ctx && ctx->ptr && ctx->size) std::memset(ctx->ptr, value, ctx->size);
}

ggml_backend_buffer_i qnn_buffer_iface = {
    qnn_buffer_name,
    qnn_buffer_free,
    qnn_buffer_base,
    nullptr,
    qnn_buffer_memset,
    qnn_buffer_set,
    qnn_buffer_get,
    nullptr,
    qnn_buffer_clear,
    nullptr,
};

const char * qnn_buft_name(ggml_backend_buffer_type_t) { return "QNN-HTP"; }

size_t qnn_buft_alignment(ggml_backend_buffer_type_t) { return 64; }
size_t qnn_buft_max_size(ggml_backend_buffer_type_t) { return SIZE_MAX; }
bool qnn_buft_is_host(ggml_backend_buffer_type_t) { return false; }

ggml_backend_buffer_t qnn_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new (std::nothrow) qnn_buffer_context();
    if (!ctx) return nullptr;

    void * ptr = nullptr;
    if (posix_memalign(&ptr, 64, std::max<size_t>(size, 64)) != 0 || !ptr) {
        delete ctx;
        return nullptr;
    }
    ctx->ptr = ptr;
    ctx->size = size;
    return ggml_backend_buffer_init(buft, qnn_buffer_iface, ctx, size);
}

ggml_backend_buffer_type qnn_buft = {
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

ggml_guid_t qnn_guid() {
    static ggml_guid guid = {
        0x71, 0x6e, 0x6e, 0x2d, 0x68, 0x74, 0x70, 0x2d,
        0x73, 0x32, 0x35, 0x2d, 0x30, 0x30, 0x30, 0x31,
    };
    return &guid;
}

struct qnn_backend_context {
    std::unique_ptr<qnn_runtime> runtime;
};

const char * qnn_backend_name(ggml_backend_t) { return "QNN-HTP"; }

void qnn_backend_free(ggml_backend_t backend) {
    if (!backend) return;
    delete static_cast<qnn_backend_context *>(backend->context);
    delete backend;
}

ggml_backend_buffer_type_t qnn_default_buft(ggml_backend_t) { return &qnn_buft; }

bool supports_mul_mat_id(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID || !op->src[0] || !op->src[1] || !op->src[2]) return false;
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    const ggml_tensor * ids = op->src[2];

    if (!weight_type(weights->type) || weights->ne[0] <= 0 || weights->ne[1] <= 0 || weights->ne[2] <= 0 ||
        !tensor_layout_2d(input) || input->ne[1] != 1 ||
        ids->type != GGML_TYPE_I32 || ids->ne[1] != 1 || ids->ne[0] <= 0 || ids->ne[0] > 64 ||
        !activation_type(op->type) || op->ne[0] <= 0 || op->ne[1] != 1 ||
        op->ne[2] < ids->ne[0] || op->ne[3] != 1) return false;

    if (weights->ne[0] > UINT32_MAX || weights->ne[1] > UINT32_MAX) return false;
    return stage_allowed(1,
        static_cast<uint32_t>(weights->ne[0]),
        static_cast<uint32_t>(weights->ne[1]));
}

bool qnn_supports_op(ggml_backend_t, const ggml_tensor * op) {
    if (!op) return false;
    if (op->op == GGML_OP_MUL_MAT_ID) return supports_mul_mat_id(op);
    if (op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1]) return false;

    const ggml_tensor * weight = op->src[0];
    const ggml_tensor * input = op->src[1];
    if (!weight_layout_2d(weight) || !tensor_layout_2d(input) || !tensor_layout_2d(op) ||
        weight->ne[0] != input->ne[0] || op->ne[0] != weight->ne[1] || op->ne[1] != input->ne[1]) {
        return false;
    }

    if (weight->ne[0] > UINT32_MAX || weight->ne[1] > UINT32_MAX || input->ne[1] > UINT32_MAX) return false;
    return stage_allowed(
        static_cast<uint32_t>(input->ne[1]),
        static_cast<uint32_t>(weight->ne[0]),
        static_cast<uint32_t>(weight->ne[1]));
}

bool qnn_supports_buft(ggml_backend_t, ggml_backend_buffer_type_t buft) {
    // This is what lets a QNN split consume mmap'd/deferred expert weights in
    // place instead of asking ggml to copy the complete expert tensor first.
    return buft == &qnn_buft || ggml_backend_buft_is_host(buft);
}

bool qnn_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    if (!qnn_supports_op(backend, op)) return false;
    if (op->op == GGML_OP_MUL_MAT_ID) return true;

    int min_batch = 1;
    if (const char * value = std::getenv("GGML_QNN_MIN_BATCH")) {
        min_batch = std::max(1, std::atoi(value));
    }
    return op->src[1]->ne[1] >= min_batch;
}

enum ggml_status qnn_compute(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<qnn_backend_context *>(backend->context);
    if (!ctx || !ctx->runtime || !ctx->runtime->ready()) return GGML_STATUS_FAILED;

    for (int i = 0; i < graph->n_nodes; ++i) {
        ggml_tensor * node = graph->nodes[i];
        bool ok = false;
        if (node->op == GGML_OP_MUL_MAT) {
            ok = ctx->runtime->matmul(node->src[0], node->src[1], node);
        } else if (node->op == GGML_OP_MUL_MAT_ID) {
            ok = ctx->runtime->mul_mat_id(node->src[0], node->src[1], node->src[2], node);
        }
        if (!ok) return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

ggml_backend_i qnn_backend_iface = {
    qnn_backend_name,
    qnn_backend_free,
    qnn_default_buft,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    qnn_compute,
    qnn_supports_op,
    qnn_supports_buft,
    qnn_offload_op,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

ggml_backend_t qnn_registry_init(const char *, void *) {
    return ggml_backend_qnn_init(0);
}

} // namespace

int ggml_backend_qnn_get_device_count(void) {
    qnn_runtime runtime;
    return runtime.initialize() ? 1 : 0;
}

void ggml_backend_qnn_reg_devices(void) {
    std::lock_guard<std::mutex> lock(g_registration_mutex);
    if (g_registered) return;

    qnn_runtime probe;
    if (!probe.initialize()) return;

    ggml_backend_register("QNN0", qnn_registry_init, &qnn_buft, nullptr);
    g_registered = true;
}

ggml_backend_t ggml_backend_qnn_init(int device) {
    if (device != 0) {
        set_status("QNN HTP: only device 0 is supported");
        return nullptr;
    }

    auto runtime = std::make_unique<qnn_runtime>();
    if (!runtime->initialize()) return nullptr;

    auto * ctx = new (std::nothrow) qnn_backend_context();
    if (!ctx) return nullptr;
    ctx->runtime = std::move(runtime);

    auto * backend = new (std::nothrow) ggml_backend {
        qnn_guid(), qnn_backend_iface, ctx,
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
    thread_local std::string copy;
    std::lock_guard<std::mutex> lock(g_status_mutex);
    copy = g_status;
    return copy.c_str();
}
