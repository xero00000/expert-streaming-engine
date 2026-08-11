#pragma once

#include <string>

// Lightweight runtime probe used before the full ggml-qnn backend is linked.
// It deliberately uses dlopen so the Android app can run without QAIRT/QNN
// libraries installed. A future backend can replace this probe without
// changing the Kotlin/JNI surface.
std::string expert_android_qnn_probe();
