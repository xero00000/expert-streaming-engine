# Expert Streaming Engine — Android / Galaxy S25 Ultra

This Android target builds the repository itself through the NDK and exposes the fork-specific expert-streaming controls through JNI.

## Current branch

`android-s25-qnn`

## Implemented in the first Android milestone

- arm64-v8a Android build target
- direct NDK/JNI integration with this fork (not a remote-server wrapper)
- GGUF mmap loading
- `defer_experts` exposed to Kotlin
- route-aware expert prefetch exposed to Kotlin
- configurable prefetch workers, context, batch, ubatch, CPU threads, GPU layers and CPU-MoE layer count
- runtime QNN/HTP library probe using `dlopen`
- optional Vulkan build switch (`-PexpertVulkan=true`)
- CPU-first defaults that keep the port usable before Vulkan/QNN are enabled

## Not claimed complete yet

The QNN probe only detects whether QAIRT/QNN libraries are visible to the Android process. It does **not** yet execute ggml operators on HTP. The next backend milestone is a real `ggml-qnn` device/backend implementation and HTP graph execution.

## Build

Open `examples/llama.android` in Android Studio, or use the Gradle wrapper with an installed Android SDK/NDK:

```bash
cd examples/llama.android
./gradlew :app:assembleDebug
```

CPU-only is the default. To attempt the repository's Vulkan backend as well, make sure the host build machine has the Vulkan SDK and `glslc`, then run:

```bash
./gradlew :app:assembleDebug -PexpertVulkan=true
```

The initial target is `arm64-v8a` and minSdk 33, appropriate for the Galaxy S25 Ultra.

## Engine configuration

`LLamaAndroid.EngineConfig` currently exposes:

```text
contextSize
threads
batchSize
ubatchSize
maxTokens
deferExperts
prefetchExperts
prefetchThreads
gpuLayers
cpuMoeLayers
```

Defaults are intentionally conservative for first device validation:

```text
contextSize      = 4096
threads          = auto
batchSize        = 512
ubatchSize       = 256
deferExperts     = true
prefetchExperts  = true
prefetchThreads  = 2
gpuLayers        = 0
cpuMoeLayers     = auto/default
```

## QNN / Snapdragon HTP plan

The HTP backend will remain optional so the application still launches when Qualcomm's redistributable runtime is absent.

Planned path:

1. Add `GGML_QNN` backend registration and device discovery.
2. Load the QAIRT/QNN HTP provider from the Qualcomm SDK/runtime.
3. Implement tensor/buffer mapping for ggml tensors.
4. Start with FP16 dense GEMM / RMSNorm / projection graphs.
5. Add bounded reusable QNN buffers for selected MoE expert tensors.
6. Connect router-selected expert extents to the Android mmap/prefetch path.
7. Add INT8/INT4 paths where QNN supports the required encoding without permanent GGUF conversion.
8. Keep CPU/Vulkan fallback for unsupported operators.

The intended final data path is:

```text
GGUF on UFS
   -> mmap / selected expert prefetch
   -> bounded staging/shared buffer
   -> QNN HTP graph
   -> ggml scheduler
```

## Device validation gates

Before calling the NPU port complete, validate on a physical S25 Ultra:

- model load with `deferExperts=true`
- bounded resident memory under repeated expert churn
- cold/warm expert-prefetch behavior
- token parity against CPU reference for deterministic prompts
- QNN HTP device discovery
- QNN operator parity per implemented op
- mixed CPU/QNN graph correctness
- Vulkan fallback correctness
- thermal throttling behavior over sustained generation
- app lifecycle unload/reload without leaked native buffers
