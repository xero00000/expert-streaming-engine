#pragma once

#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Qualcomm AI Engine Direct / QAIRT HTP backend.
//
// The backend is intentionally optional. It is only compiled by the Android
// target when EXPERT_ANDROID_QNN=ON and the QAIRT/QNN headers are available.
// At runtime libQnnHtp.so is loaded with dlopen so the app can still start on
// builds where the redistributable HTP runtime is not packaged.

GGML_API int  ggml_backend_qnn_get_device_count(void);
GGML_API void ggml_backend_qnn_reg_devices(void);

GGML_API ggml_backend_t             ggml_backend_qnn_init(int device);
GGML_API ggml_backend_buffer_type_t ggml_backend_qnn_buffer_type(int device);
GGML_API bool                       ggml_backend_is_qnn(ggml_backend_t backend);

// Human readable status of the last probe/initialization attempt.
GGML_API const char * ggml_backend_qnn_status(void);

#ifdef __cplusplus
}
#endif
