#include <android/log.h>
#include <jni.h>

#include <algorithm>
#include <string>
#include <unistd.h>

#include "llama.h"
#include "ggml-backend.h"
#include "qnn-probe.h"

#ifdef GGML_USE_QNN
#include "ggml-qnn.h"
#endif

#define TAG "expert-android-jni"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

int auto_threads() {
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    if (online <= 2) return 1;
    return std::max(1, std::min(8, static_cast<int>(online) - 2));
}

void throw_state(JNIEnv * env, const char * message) {
    jclass cls = env->FindClass("java/lang/IllegalStateException");
    if (cls) env->ThrowNew(cls, message);
}

bool configure_qnn(bool requested) {
    if (!requested) return false;
#ifdef GGML_USE_QNN
    ggml_backend_qnn_reg_devices();
    const size_t index = ggml_backend_reg_find_by_name("QNN0");
    const bool ready = index < ggml_backend_reg_get_count();
    LOGI("QNN requested; registered=%d status=%s", ready ? 1 : 0, ggml_backend_qnn_status());
    return ready;
#else
    LOGI("QNN requested but this APK was built without EXPERT_ANDROID_QNN");
    return false;
#endif
}

std::string backend_summary() {
    std::string result;
    const size_t count = ggml_backend_reg_get_count();
    for (size_t i = 0; i < count; ++i) {
        if (!result.empty()) result += ", ";
        const char * name = ggml_backend_reg_get_name(i);
        result += name ? name : "?";
    }
    if (result.empty()) result = "none";
    return result;
}

} // namespace

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_load_1engine_1model(
        JNIEnv * env,
        jobject,
        jstring filename,
        jboolean defer_experts,
        jint n_gpu_layers,
        jint n_cpu_moe,
        jboolean use_qnn) {
    if (!filename) {
        throw_state(env, "Model path is null");
        return 0;
    }

    const bool qnn_ready = configure_qnn(use_qnn == JNI_TRUE);

    llama_model_params params = llama_model_default_params();
    params.use_mmap      = true;
    params.use_mlock     = false;
    params.defer_experts = defer_experts == JNI_TRUE;
    params.n_gpu_layers  = std::max(0, static_cast<int>(n_gpu_layers));

    static const char qnn_device[] = "QNN0";
    if (qnn_ready) params.devices = qnn_device;
    if (n_cpu_moe >= 0) params.ncmoe = static_cast<int32_t>(n_cpu_moe);

    const char * path = env->GetStringUTFChars(filename, nullptr);
    if (!path) {
        throw_state(env, "Unable to read model path");
        return 0;
    }

    LOGI("Loading model path=%s mmap=1 defer_experts=%d accel=%s gpu_layers=%d cpu_moe=%d",
         path,
         params.defer_experts ? 1 : 0,
         qnn_ready ? "QNN0" : "auto/CPU",
         params.n_gpu_layers,
         params.ncmoe);

    llama_model * model = llama_load_model_from_file(path, params);
    env->ReleaseStringUTFChars(filename, path);

    if (!model) {
        LOGE("llama_load_model_from_file failed");
        throw_state(env, "Failed to load GGUF model");
        return 0;
    }

    return reinterpret_cast<jlong>(model);
}

extern "C"
JNIEXPORT jlong JNICALL
Java_android_llama_cpp_LLamaAndroid_new_1engine_1context(
        JNIEnv * env,
        jobject,
        jlong model_ptr,
        jint n_ctx,
        jint n_threads,
        jint n_batch,
        jint n_ubatch,
        jboolean prefetch_experts,
        jint prefetch_threads,
        jboolean use_qnn) {
    auto * model = reinterpret_cast<llama_model *>(model_ptr);
    if (!model) {
        throw_state(env, "Model is null");
        return 0;
    }

    const int threads = n_threads > 0 ? static_cast<int>(n_threads) : auto_threads();
    const int batch   = std::max(32, static_cast<int>(n_batch));
    const int ubatch  = std::max(1, std::min(batch, static_cast<int>(n_ubatch)));

    llama_context_params params = llama_context_default_params();
    params.seed                     = 1234;
    params.n_ctx                    = std::max(512, static_cast<int>(n_ctx));
    params.n_batch                  = batch;
    params.n_ubatch                 = ubatch;
    params.n_threads                = threads;
    params.n_threads_batch          = threads;
    params.prefetch_experts         = prefetch_experts == JNI_TRUE;
    params.prefetch_experts_threads = prefetch_threads;

    // HTP implements standard MUL_MAT_ID for routed decode. Keep that graph
    // form while QNN is selected; unsupported graph nodes remain on CPU/Vulkan.
    if (use_qnn == JNI_TRUE) params.fused_moe_up_gate = false;

    LOGI("Creating context ctx=%u threads=%u batch=%u ubatch=%u prefetch=%d prefetch_threads=%d qnn=%d",
         params.n_ctx,
         params.n_threads,
         params.n_batch,
         params.n_ubatch,
         params.prefetch_experts ? 1 : 0,
         params.prefetch_experts_threads,
         use_qnn == JNI_TRUE ? 1 : 0);

    llama_context * ctx = llama_new_context_with_model(model, params);
    if (!ctx) {
        LOGE("llama_new_context_with_model failed");
        throw_state(env, "Failed to create llama context");
        return 0;
    }

    return reinterpret_cast<jlong>(ctx);
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_qnn_1probe(JNIEnv * env, jobject) {
#ifdef GGML_USE_QNN
    // Probe without registering. This is intentionally retryable so the UI can
    // query status before ADSP_LIBRARY_PATH is configured, then try again after
    // bundled Hexagon skeletons have been extracted.
    (void) ggml_backend_qnn_get_device_count();
    return env->NewStringUTF(ggml_backend_qnn_status());
#else
    const std::string status = expert_android_qnn_probe();
    return env->NewStringUTF(status.c_str());
#endif
}

extern "C"
JNIEXPORT jstring JNICALL
Java_android_llama_cpp_LLamaAndroid_backend_1summary(JNIEnv * env, jobject) {
    const std::string summary = backend_summary();
    return env->NewStringUTF(summary.c_str());
}

extern "C"
JNIEXPORT void JNICALL
Java_android_llama_cpp_LLamaAndroid_set_1dsp_1library_1path(JNIEnv * env, jobject, jstring path) {
    if (!path) return;
    const char * value = env->GetStringUTFChars(path, nullptr);
    if (!value) return;
    const char * old = getenv("ADSP_LIBRARY_PATH");
    std::string merged(value);
    if (old && *old) {
        merged += ";";
        merged += old;
    }
    setenv("ADSP_LIBRARY_PATH", merged.c_str(), 1);
    env->ReleaseStringUTFChars(path, value);
    LOGI("Configured ADSP_LIBRARY_PATH");
}
