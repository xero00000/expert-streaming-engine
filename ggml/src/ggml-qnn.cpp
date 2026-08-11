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
#include <utility>
#include <vector>

// Qualcomm HTP backend for the Android expert-streaming port.
//
// Design constraints:
//   * QAIRT is optional at build time and libQnnHtp.so is optional at runtime.
//   * GGUF remains the source of truth. No model-wide QNN conversion is needed.
//   * Dense F16/F32 MUL_MAT can run on HTP.
//   * Decode-time MUL_MAT_ID can consume F16/F32/MXFP4 expert tensors directly
//     from mmap/host buffers, dequantizing only the selected expert into a
//     bounded reusable FP16 QNN staging buffer.
//   * QNN allocations are reported as accelerator memory to ggml even though
//     the correctness-first implementation currently uses host-visible backing
//     memory. This keeps the existing active-expert cache and scheduler logic
//     treating HTP as a non-CPU backend.

namespace {

static std::mutex g_status_mutex;
static std::string g_status = "QNN backend not probed";

static void set_status(const std::string & status) {
    std::lock_guard<std::mutex> lock(g_status_mutex);
    g_status = status;
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

static float fp16_to_fp32(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    uint32_t exponent = (value >> 10) & 0x1fu;
    uint32_t mantissa = value & 0x03ffu;
    uint32_t out = 0;
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
        out = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

static bool activation_type_supported(enum ggml_type type) {
    return type == GGML_TYPE_F16 || type == GGML_TYPE_F32;
}

static bool weight_type_supported(enum ggml_type type) {
    return activation_type_supported(type) || type == GGML_TYPE_MXFP4;
}

static bool tensor_is_plain_2d(const ggml_tensor * tensor) {
    return tensor && tensor->data && activation_type_supported(tensor->type) &&
           tensor->ne[2] == 1 && tensor->ne[3] == 1 && ggml_is_contiguous(tensor);
}

static bool weight_is_2d(const ggml_tensor * tensor) {
    return tensor && tensor->data && weight_type_supported(tensor->type) &&
           tensor->ne[0] > 0 && tensor->ne[1] > 0 && tensor->ne[2] == 1 && tensor->ne[3] == 1;
}

static void tensor_to_half(const ggml_tensor * src, uint16_t * dst, size_t count) {
    if (src->type == GGML_TYPE_F16) {
        std::memcpy(dst, src->data, count * sizeof(uint16_t));
        return;
    }
    const float * input = static_cast<const float *>(src->data);
    for (size_t i = 0; i < count; ++i) dst[i] = fp32_to_fp16(input[i]);
}

static void half_to_tensor(const uint16_t * src, ggml_tensor * dst, size_t count) {
    if (dst->type == GGML_TYPE_F16) {
        std::memcpy(dst->data, src, count * sizeof(uint16_t));
        return;
    }
    float * output = static_cast<float *>(dst->data);
    for (size_t i = 0; i < count; ++i) output[i] = fp16_to_fp32(src[i]);
}

static bool transpose_weight_to_half(const ggml_tensor * weight, uint16_t * dst, size_t k, size_t m) {
    if (weight->type == GGML_TYPE_F16) {
        for (size_t row = 0; row < m; ++row) {
            const uint16_t * source = reinterpret_cast<const uint16_t *>(
                static_cast<const char *>(weight->data) + row * weight->nb[1]);
            for (size_t col = 0; col < k; ++col) dst[col * m + row] = source[col];
        }
        return true;
    }
    if (weight->type == GGML_TYPE_F32) {
        for (size_t row = 0; row < m; ++row) {
            const float * source = reinterpret_cast<const float *>(
                static_cast<const char *>(weight->data) + row * weight->nb[1]);
            for (size_t col = 0; col < k; ++col) dst[col * m + row] = fp32_to_fp16(source[col]);
        }
        return true;
    }
    if (weight->type == GGML_TYPE_MXFP4) {
        std::vector<float> row_buffer(k);
        for (size_t row = 0; row < m; ++row) {
            const block_mxfp4 * source = reinterpret_cast<const block_mxfp4 *>(
                static_cast<const char *>(weight->data) + row * weight->nb[1]);
            dequantize_row_mxfp4(source, row_buffer.data(), static_cast<int64_t>(k));
            for (size_t col = 0; col < k; ++col) dst[col * m + row] = fp32_to_fp16(row_buffer[col]);
        }
        return true;
    }
    return false;
}

struct matmul_key {
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t m = 0;
    bool operator==(const matmul_key & other) const { return n == other.n && k == other.k && m == other.m; }
};

struct matmul_key_hash {
    size_t operator()(const matmul_key & key) const {
        uint64_t value = (static_cast<uint64_t>(key.n) << 42) ^
                         (static_cast<uint64_t>(key.k) << 21) ^ static_cast<uint64_t>(key.m);
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
    ~qnn_runtime() { shutdown(); }

    bool initialize() {
        if (ready_) return true;
        lib_ = dlopen(qnn_backend_library(), RTLD_NOW | RTLD_LOCAL);
        if (!lib_) {
            const char * error = dlerror();
            set_status(std::string("QNN HTP unavailable: ") + (error ? error : "dlopen failed"));
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
            if (provider && provider->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
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
            set_status("QNN HTP unavailable: required graph API missing");
            shutdown();
            return false;
        }
        Qnn_ErrorHandle_t result = interface_.backendCreate(nullptr, nullptr, &backend_);
        if (QNN_GET_ERROR_CODE(result) != QNN_SUCCESS || !backend_) {
            set_status("QNN HTP unavailable: backendCreate failed");
            shutdown();
            return false;
        }
        result = interface_.contextCreate(backend_, nullptr, nullptr, &context_);
        if (QNN_GET_ERROR_CODE(result) != QNN_SUCCESS || !context_) {
            set_status("QNN HTP unavailable: contextCreate failed");
            shutdown();
            return false;
        }
        ready_ = true;
        set_status(std::string("QNN HTP ready via ") + qnn_backend_library());
        return true;
    }

    bool ready() const { return ready_; }

    bool execute_matmul(const ggml_tensor * weight, const ggml_tensor * input, ggml_tensor * output) {
        if (!ready_ || !weight_is_2d(weight) || !tensor_is_plain_2d(input) || !tensor_is_plain_2d(output)) return false;
        const int64_t k64 = weight->ne[0];
        const int64_t m64 = weight->ne[1];
        const int64_t n64 = input->ne[1];
        if (input->ne[0] != k64 || output->ne[0] != m64 || output->ne[1] != n64 ||
            k64 <= 0 || m64 <= 0 || n64 <= 0 || k64 > std::numeric_limits<uint32_t>::max() ||
            m64 > std::numeric_limits<uint32_t>::max() || n64 > std::numeric_limits<uint32_t>::max()) return false;
        const matmul_key key { static_cast<uint32_t>(n64), static_cast<uint32_t>(k64), static_cast<uint32_t>(m64) };
        std::shared_ptr<qnn_matmul_graph> compiled = get_or_create_matmul(key);
        if (!compiled) return false;
        std::lock_guard<std::mutex> execute_lock(compiled->execute_mutex);
        const size_t n = key.n, k = key.k, m = key.m;
        tensor_to_half(input, compiled->a.data(), n * k);
        if (!transpose_weight_to_half(weight, compiled->b.data(), k, m)) return false;
        Qnn_Tensor_t inputs[2] = { compiled->graph_a, compiled->graph_b };
        Qnn_Tensor_t outputs[1] = { compiled->graph_o };
        inputs[0].v1.clientBuf = Qnn_ClientBuffer_t { compiled->a.data(), compiled->a.size() * sizeof(uint16_t) };
        inputs[1].v1.clientBuf = Qnn_ClientBuffer_t { compiled->b.data(), compiled->b.size() * sizeof(uint16_t) };
        outputs[0].v1.clientBuf = Qnn_ClientBuffer_t { compiled->o.data(), compiled->o.size() * sizeof(uint16_t) };
        const Qnn_ErrorHandle_t result = interface_.graphExecute(compiled->graph, inputs, 2, outputs, 1, nullptr, nullptr);
        if (QNN_GET_ERROR_CODE(result) != QNN_SUCCESS) {
            set_status("QNN HTP graphExecute failed");
            return false;
        }
        half_to_tensor(compiled->o.data(), output, n * m);
        return true;
    }

    bool execute_mul_mat_id(const ggml_tensor * weights, const ggml_tensor * input,
                            const ggml_tensor * ids, ggml_tensor * output) {
        if (!ready_ || !weights || !weights->data || !weight_type_supported(weights->type) ||
            !tensor_is_plain_2d(input) || !ids || !ids->data || ids->type != GGML_TYPE_I32 ||
            !output || !output->data || !activation_type_supported(output->type)) return false;
        if (input->ne[1] != 1 || input->ne[2] != 1 || input->ne[3] != 1 ||
            ids->ne[1] != 1 || ids->ne[0] <= 0 || ids->ne[0] > 64 ||
            output->ne[1] != 1 || output->ne[2] < ids->ne[0] || output->ne[3] != 1) return false;
        const int64_t k = weights->ne[0];
        const int64_t m = weights->ne[1];
        const int64_t n_experts = weights->ne[2];
        if (input->ne[0] != k || output->ne[0] != m || n_experts <= 0) return false;
        std::memset(output->data, 0, ggml_nbytes(output));
        for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
            const int32_t expert = *reinterpret_cast<const int32_t *>(
                static_cast<const char *>(ids->data) + slot * ids->nb[0]);
            if (expert < 0 || expert >= n_experts) continue;
            ggml_tensor expert_weight = *weights;
            expert_weight.data = static_cast<char *>(weights->data) + static_cast<size_t>(expert) * weights->nb[2];
            expert_weight.ne[2] = 1;
            expert_weight.ne[3] = 1;
            ggml_tensor expert_output = *output;
            expert_output.data = static_cast<char *>(output->data) + static_cast<size_t>(slot) * output->nb[2];
            expert_output.ne[1] = 1;
            expert_output.ne[2] = 1;
            expert_output.ne[3] = 1;
            expert_output.nb[1] = static_cast<size_t>(m) * ggml_type_size(output->type);
            expert_output.nb[2] = expert_output.nb[1];
            expert_output.nb[3] = expert_output.nb[1];
            if (!execute_matmul(&expert_weight, input, &expert_output)) return false;
        }
        return true;
    }

private:
    std::shared_ptr<qnn_matmul_graph> get_or_create_matmul(const matmul_key & key) {
        {
            std::lock_guard<std::mutex> lock(graph_mutex_);
            auto it = matmuls_.find(key);
            if (it != matmuls_.end()) return it->second;
        }
        std::shared_ptr<qnn_matmul_graph> graph(new (std::nothrow) qnn_matmul_graph());
        if (!graph) return nullptr;
        graph->a_dims[0] = key.n; graph->a_dims[1] = key.k;
        graph->b_dims[0] = key.k; graph->b_dims[1] = key.m;
        graph->o_dims[0] = key.n; graph->o_dims[1] = key.m;
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
        return inserted.second ? graph : inserted.first->second;
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
        { std::lock_guard<std::mutex> lock(graph_mutex_); matmuls_.clear(); }
        if (context_ && interface_.contextFree) { interface_.contextFree(context_, nullptr); context_ = nullptr; }
        if (backend_ && interface_.backendFree) { interface_.backendFree(backend_); backend_ = nullptr; }
        if (lib_) { dlclose(lib_); lib_ = nullptr; }
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

struct qnn_buffer_context { void * ptr = nullptr; size_t size = 0; };
static const char * qnn_buffer_name(ggml_backend_buffer_t) { return "QNN-HTP"; }
static void qnn_buffer_free(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    if (!ctx) return;
    std::free(ctx->ptr);
    delete ctx;
}
static void * qnn_buffer_base(ggml_backend_buffer_t buffer) {
    auto * ctx = static_cast<qnn_buffer_context *>(buffer->context);
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
    auto * ctx = static_cast<qnn_buffer_context *>(buffer->context);
    if (ctx && ctx->ptr && ctx->size) std::memset(ctx->ptr, value, ctx->size);
}
static ggml_backend_buffer_i qnn_buffer_i = {
    qnn_buffer_name, qnn_buffer_free, qnn_buffer_base, nullptr, qnn_buffer_memset_tensor,
    qnn_buffer_set_tensor, qnn_buffer_get_tensor, nullptr, qnn_buffer_clear, nullptr,
};
static const char * qnn_buft_name(ggml_backend_buffer_type_t) { return "QNN-HTP"; }
static ggml_backend_buffer_t qnn_buft_alloc(ggml_backend_buffer_type_t buft, size_t size) {
    auto * ctx = new (std::nothrow) qnn_buffer_context();
    if (!ctx) return nullptr;
    const size_t actual = std::max<size_t>(size, 64);
    void * ptr = nullptr;
    if (posix_memalign(&ptr, 64, actual) != 0 || !ptr) { delete ctx; return nullptr; }
    ctx->ptr = ptr;
    ctx->size = size;
    return ggml_backend_buffer_init(buft, qnn_buffer_i, ctx, size);
}
static size_t qnn_buft_alignment(ggml_backend_buffer_type_t) { return 64; }
static size_t qnn_buft_max_size(ggml_backend_buffer_type_t) { return SIZE_MAX; }
static bool qnn_buft_is_host(ggml_backend_buffer_type_t) { return false; }
static ggml_backend_buffer_type qnn_buft = {
    { qnn_buft_name, qnn_buft_alloc, qnn_buft_alignment, qnn_buft_max_size, nullptr, qnn_buft_is_host }, nullptr,
};

static ggml_guid_t qnn_guid() {
    static ggml_guid guid = { 0x71, 0x6e, 0x6e, 0x2d, 0x68, 0x74, 0x70, 0x2d, 0x73, 0x32, 0x35, 0x2d, 0x30, 0x30, 0x30, 0x31 };
    return &guid;
}
struct qnn_backend_context { std::unique_ptr<qnn_runtime> runtime; };
static const char * qnn_backend_name(ggml_backend_t) { return "QNN-HTP"; }
static void qnn_backend_free(ggml_backend_t backend) {
    if (!backend) return;
    delete static_cast<qnn_backend_context *>(backend->context);
    delete backend;
}
static ggml_backend_buffer_type_t qnn_backend_default_buft(ggml_backend_t) { return &qnn_buft; }

static bool qnn_mul_mat_id_supported(const ggml_tensor * op) {
    if (!op || op->op != GGML_OP_MUL_MAT_ID || !op->src[0] || !op->src[1] || !op->src[2]) return false;
    const ggml_tensor * weights = op->src[0];
    const ggml_tensor * input = op->src[1];
    const ggml_tensor * ids = op->src[2];
    return weights->data && weight_type_supported(weights->type) && weights->ne[2] > 0 &&
           tensor_is_plain_2d(input) && input->ne[1] == 1 &&
           ids->data && ids->type == GGML_TYPE_I32 && ids->ne[1] == 1 && ids->ne[0] > 0 && ids->ne[0] <= 64 &&
           op->data && activation_type_supported(op->type) && op->ne[1] == 1 && op->ne[2] >= ids->ne[0] && op->ne[3] == 1;
}

static bool qnn_backend_supports_op(ggml_backend_t, const ggml_tensor * op) {
    if (!op) return false;
    if (op->op == GGML_OP_MUL_MAT_ID) return qnn_mul_mat_id_supported(op);
    if (op->op != GGML_OP_MUL_MAT || !op->src[0] || !op->src[1]) return false;
    return weight_is_2d(op->src[0]) && tensor_is_plain_2d(op->src[1]) && tensor_is_plain_2d(op) &&
           op->src[0]->ne[0] == op->src[1]->ne[0] && op->ne[0] == op->src[0]->ne[1] && op->ne[1] == op->src[1]->ne[1];
}
static bool qnn_backend_supports_buft(ggml_backend_t, ggml_backend_buffer_type_t buft) {
    return buft == &qnn_buft || ggml_backend_buft_is_host(buft);
}
static bool qnn_backend_offload_op(ggml_backend_t backend, const ggml_tensor * op) {
    if (!qnn_backend_supports_op(backend, op)) return false;
    if (op->op == GGML_OP_MUL_MAT_ID) return true;
    int min_batch = 1;
    if (const char * env = std::getenv("GGML_QNN_MIN_BATCH")) min_batch = std::max(1, std::atoi(env));
    return op->src[1]->ne[1] >= min_batch;
}
static enum ggml_status qnn_backend_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = static_cast<qnn_backend_context *>(backend->context);
    if (!ctx || !ctx->runtime || !ctx->runtime->ready()) return GGML_STATUS_FAILED;
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        if (node->op == GGML_OP_MUL_MAT_ID) {
            if (!ctx->runtime->execute_mul_mat_id(node->src[0], node->src[1], node->src[2], node)) return GGML_STATUS_FAILED;
        } else if (node->op == GGML_OP_MUL_MAT) {
            if (!ctx->runtime->execute_matmul(node->src[0], node->src[1], node)) return GGML_STATUS_FAILED;
        } else {
            return GGML_STATUS_FAILED;
        }
    }
    return GGML_STATUS_SUCCESS;
}
static ggml_backend_i qnn_backend_i = {
    qnn_backend_name, qnn_backend_free, qnn_backend_default_buft,
    nullptr, nullptr, nullptr, nullptr,
    nullptr, nullptr, nullptr, nullptr,
    qnn_backend_graph_compute, qnn_backend_supports_op, qnn_backend_supports_buft, qnn_backend_offload_op,
    nullptr, nullptr, nullptr, nullptr, nullptr,
};
static ggml_backend_t qnn_reg_init(const char *, void *) { return ggml_backend_qnn_init(0); }
static std::once_flag qnn_register_once;

} // namespace

int ggml_backend_qnn_get_device_count(void) {
    qnn_runtime runtime;
    return runtime.initialize() ? 1 : 0;
}
void ggml_backend_qnn_reg_devices(void) {
    std::call_once(qnn_register_once, [] {
        qnn_runtime probe;
        if (probe.initialize()) ggml_backend_register("QNN0", qnn_reg_init, &qnn_buft, nullptr);
    });
}
ggml_backend_t ggml_backend_qnn_init(int device) {
    if (device != 0) {
        set_status("QNN HTP: only device 0 is supported by the Android backend");
        return nullptr;
    }
    std::unique_ptr<qnn_runtime> runtime(new (std::nothrow) qnn_runtime());
    if (!runtime || !runtime->initialize()) return nullptr;
    auto * ctx = new (std::nothrow) qnn_backend_context();
    if (!ctx) return nullptr;
    ctx->runtime = std::move(runtime);
    auto * backend = new (std::nothrow) ggml_backend { qnn_guid(), qnn_backend_i, ctx };
    if (!backend) { delete ctx; return nullptr; }
    return backend;
}
ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(int device) { return device == 0 ? &qnn_buft : nullptr; }
bool ggml_backend_is_qnn(ggml_backend_t backend) { return backend && ggml_backend_guid(backend) == qnn_guid(); }
const char * ggml_backend_qnn_status(void) {
    std::lock_guard<std::mutex> lock(g_status_mutex);
    return g_status.c_str();
}
