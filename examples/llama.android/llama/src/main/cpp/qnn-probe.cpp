#include "qnn-probe.h"

#include <dlfcn.h>
#include <sstream>

namespace {

bool can_open(const char * name) {
    void * handle = dlopen(name, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        return false;
    }
    dlclose(handle);
    return true;
}

} // namespace

std::string expert_android_qnn_probe() {
    const bool qnn_system = can_open("libQnnSystem.so");
    const bool qnn_htp    = can_open("libQnnHtp.so");
    const bool qnn_cpu    = can_open("libQnnCpu.so");
    const bool qnn_gpu    = can_open("libQnnGpu.so");

    std::ostringstream out;
    out << "QNN runtime: system=" << (qnn_system ? "yes" : "no")
        << ", htp=" << (qnn_htp ? "yes" : "no")
        << ", gpu=" << (qnn_gpu ? "yes" : "no")
        << ", cpu=" << (qnn_cpu ? "yes" : "no");

    if (qnn_system && qnn_htp) {
        out << " (HTP candidate available)";
    } else {
        out << " (HTP libraries not visible to app)";
    }

    return out.str();
}
