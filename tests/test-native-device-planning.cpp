#include "common.h"
#include "llama.h"
#include <cuda.h>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

static void check(bool result, const char * message) {
    if (!result) throw std::runtime_error(message);
}

static void check_native_single_device_plan(const std::filesystem::path & path,
        const std::string & selected, int main_gpu, int expected_device) {
    gpt_params params;
    params.model = path.string();
    params.devices = selected;
    params.main_gpu = main_gpu;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.n_gpu_layers = expected_device < 0 ? 0 : 99;
    params.n_ctx = 512;
    params.n_batch = 16;
    params.n_ubatch = 16;
    params.n_parallel = 1;
    params.n_threads = 1;
    params.n_threads_batch = 1;
    params.flash_attn = true;
    params.use_mmap = false;
    params.warmup = false;
    params.memory_policy = "cache";
    params.min_kv_quality = "f16";
    params.max_ram_bytes = 1024ULL*1024ULL*1024ULL;
    params.reserve_vram_bytes = 1024ULL*1024ULL*1024ULL;
    params.expert_ram_cache_mib = 0;
    params.expert_vram_cache_mib = 0;
    params.expert_prefill_staging_mib = 0;
    params.cache_ram_mib = 0;
    params.transient_mtp_mib = 1;
    auto initialized = llama_init_from_gpt_params(params);
    std::unique_ptr<llama_model, decltype(&llama_free_model)> model(initialized.model, llama_free_model);
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx(initialized.context, llama_free);
    check(model != nullptr && ctx != nullptr, "native selected-device resource plan initialization");
    const auto plan = nlohmann::json::parse(params.resolved_resource_plan_json);
    check(plan.at("transient_device").get<int>() == expected_device,
          "native transient reservation uses a local rather than physical device ID");
    check(plan.at("devices").size() == 1 &&
          plan.at("devices").at(0).at("id").get<int>() == expected_device,
          "single-device resource plan includes an unselected GPU");
    check(plan.at("devices").at(0).at("transient_bytes").get<uint64_t>() == 1024ULL*1024ULL,
          "transient reservation is not charged to the selected physical device");
    const uint32_t count = llama_resource_device_count(ctx.get());
    check(count == (expected_device < 0 ? 0U : 1U), "native single-device snapshot count");
    llama_resource_snapshot snapshot = {};
    std::vector<llama_resource_device_snapshot> devices(count);
    check(llama_resource_get_snapshot(ctx.get(), &snapshot, devices.data(), count), "native snapshot");
    if (count) check(devices[0].device_id == expected_device, "native snapshot has the wrong physical GPU");
    llama_token token = 1;
    auto batch = llama_batch_get_one(&token, 1, 0, 0);
    check(llama_decode(ctx.get(), batch) == 0, "native single-device planned decode");
    check(llama_resource_device_count(ctx.get()) == count, "decode changed the selected device set");
    std::printf("selected_plan: devices=%s main=%d physical=%d transient=1048576 exact\n",
        selected.c_str(), main_gpu, expected_device);
}

void check_native_device_plans(const std::filesystem::path & path, int count) {
    const std::string first = "CUDA0";
    const std::string last = "CUDA" + std::to_string(count - 1);
    check_native_single_device_plan(path, first, 0, 0);
    check_native_single_device_plan(path, last, 0, count - 1);
    if (count > 1) {
        check_native_single_device_plan(path, first + "," + last, 0, 0);
        check_native_single_device_plan(path, first + "," + last, 1, count - 1);
        check_native_single_device_plan(path, last + "," + first, 0, count - 1);
        check_native_single_device_plan(path, last + "," + first, 1, 0);
    }
    check_native_single_device_plan(path, "none", 0, -1);
}

void check_native_primary_contexts(const std::filesystem::path & path, int count) {
    check(cuInit(0) == CUDA_SUCCESS, "CUDA driver initialization");
    const auto primary_active = [](int id) {
        CUdevice device;
        unsigned flags = 0;
        int active = 0;
        check(cuDeviceGet(&device, id) == CUDA_SUCCESS &&
              cuDevicePrimaryCtxGetState(device, &flags, &active) == CUDA_SUCCESS,
              "CUDA primary context state query");
        return active != 0;
    };
    for (int i = 0; i < count; ++i)
        check(!primary_active(i), "enumeration created a primary context");
    const int main = count - 1;
    // Leave the native selected list at its all-visible default. NONE mode
    // must query and initialize only the main row, including its dry probe.
    check_native_single_device_plan(path, "", main, main);
    for (int i = 0; i < count; ++i)
        check(primary_active(i) == (i == main), "native planning created an unselected primary context");
    std::printf("native_primary_contexts: visible=%d main=%d selected_only=true\n", count, main);
}
