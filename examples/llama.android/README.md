# Expert Streaming Engine — Android / Galaxy S25 Ultra

Branch: `android-s25-qnn`

This is the native Android port of the Expert Streaming Engine. It builds the fork itself with the Android NDK; it is not a remote-server wrapper and it does not convert GGUF into a different model format.

> Source status: implementation complete for the Android/hybrid-QNN port described below. It has intentionally **not** been claimed as compiled or device-validated yet; the next step is a local Android Studio/NDK + physical S25 Ultra validation pass.

## Architecture

```text
GGUF selected with Android SAF
        |
        |  /proc/self/fd/<fd> (descriptor held for model lifetime)
        v
mmap / deferred expert pages on UFS
        |
        +--> route-aware madvise/mincore prefetch
        |
        v
GGML scheduler
   |              |                 |
   |              |                 +--> ARM CPU fallback / CPU MoE prefill
   |              +--> Adreno Vulkan backend (optional)
   |
   +--> QNN0 / Snapdragon Hexagon HTP (optional)
            |
            +-- dense MUL_MAT
            +-- decode MUL_MAT_ID
            +-- selected MXFP4/F16/F32 expert -> bounded FP16 staging
```

The full GGUF remains on storage. For routed MXFP4 experts, only the selected expert projection is dequantized/transposed into the reusable FP16 QNN input buffer for that HTP MatMul. This preserves the engine's disk-native design rather than creating a second NPU-specific copy of the model.

## Android port features

- `arm64-v8a` Android target, minSdk 33
- direct NDK/JNI integration with this fork
- native GGUF mmap loading
- Android Storage Access Framework model picker
- zero-copy SAF path using `/proc/self/fd/<fd>` while keeping the descriptor alive
- `defer_experts` exposed to Kotlin/UI
- route-aware expert prefetch exposed to Kotlin/UI
- configurable prefetch workers, context, batch, ubatch, CPU threads, accelerator layers and CPU-MoE layers
- mobile control panel for load/unload, generation, benchmark and backend status
- optional Vulkan/Adreno build
- optional QAIRT/QNN HTP build
- automatic CPU fallback for QNN-unsupported graph operations
- optional bundled Hexagon skeleton extraction + `ADSP_LIBRARY_PATH`
- Qualcomm runtime binaries ignored by Git

## QNN / HTP backend

Core files:

```text
ggml/include/ggml-qnn.h
ggml/src/ggml-qnn.cpp
```

The backend:

1. Loads `libQnnHtp.so` dynamically.
2. Resolves `QnnInterface_getProviders`.
3. Selects a provider whose core API is compatible with the headers used to build the APK.
4. Creates a QNN HTP backend and context.
5. Registers `QNN0` with ggml's existing backend registry.
6. Compiles and caches QNN MatMul graphs by `(N, K, M)` shape.
7. Executes QNN graphs with FP16 client buffers.
8. Converts F32 activations/results at the backend boundary when necessary.
9. Supports F16, F32 and MXFP4 weight sources.
10. Implements decode-time `GGML_OP_MUL_MAT_ID` by executing only the selected router experts.

For HTP MoE decode the Android context disables the fork's fused up/gate operator so the routed graph remains expressible as standard `MUL_MAT_ID` operations. CPU MoE remains available for prefill and as a correctness fallback.

### Current HTP operator policy

| Operation | HTP | Fallback |
| --- | --- | --- |
| `GGML_OP_MUL_MAT` | F16/F32 activation/output; F16/F32/MXFP4 weights | CPU/Vulkan |
| `GGML_OP_MUL_MAT_ID` | decode path, selected experts, F16/F32/MXFP4 weights | CPU/Vulkan |
| fused MoE up/gate | disabled while QNN is selected | standard routed graph |
| everything else | not claimed by QNN backend | CPU/Vulkan |

This is deliberately a hybrid backend. QNN never claims an operation it does not implement, so the existing ggml scheduler remains responsible for mixed-backend execution.

## Build modes

Open `examples/llama.android` in Android Studio or use its Gradle wrapper.

### CPU / ARM

```bash
./gradlew :app:assembleDebug
```

### Vulkan / Adreno

The host build machine needs a Vulkan SDK and `glslc` for shader generation.

```bash
./gradlew :app:assembleDebug -PexpertVulkan=true
```

### Qualcomm QNN / HTP

Install a Qualcomm QAIRT/QNN SDK locally, then stage the matching redistributable Android + Hexagon runtime files from that SDK:

```bash
export QNN_SDK_ROOT=/path/to/qairt
bash prepare-qnn-runtime.sh
```

Then build:

```bash
./gradlew :app:assembleDebug \
  -PexpertQnn=true \
  -PqnnSdkRoot="$QNN_SDK_ROOT"
```

### QNN + Vulkan fallback

```bash
./gradlew :app:assembleDebug \
  -PexpertQnn=true \
  -PexpertVulkan=true \
  -PqnnSdkRoot="$QNN_SDK_ROOT"
```

`prepare-qnn-runtime.sh` copies from your local SDK only. It does not download Qualcomm binaries. The staged `libQnn*.so` and Hexagon skeleton `.so` files are ignored by `.gitignore` and should not be committed.

## QAIRT runtime layout

The helper script targets:

```text
llama/src/main/jniLibs/arm64-v8a/
    libQnn*.so                 # Android-side QAIRT libraries

app/src/main/assets/qnn/
    *Skel.so                   # Hexagon-side HTP skeletons
```

On launch, `QnnRuntimeInstaller` copies optional `assets/qnn` files to app-private storage. The JNI bridge prepends that directory to `ADSP_LIBRARY_PATH` before the first real QNN registration/model load.

Use runtime files from the same QAIRT release as the headers used for the build. Do not mix arbitrary HTP skeleton versions.

## Engine controls

`LLamaAndroid.EngineConfig` exposes:

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
useQnn
qnnDspLibraryPath
```

Default Android profile:

```text
contextSize       = 4096
threads           = 0      # auto; leaves cores for Android
batchSize         = 512
ubatchSize        = 256
deferExperts      = true
prefetchExperts   = true
prefetchThreads   = 2
gpuLayers         = 0
cpuMoeLayers      = -1     # engine default/auto
useQnn            = true
```

`gpuLayers` is the fork's existing accelerator-layer parameter. When QNN is requested and available the native bridge restricts the requested accelerator device to `QNN0`; otherwise normal CPU/Vulkan discovery is retained.

## Model selection

The new UI uses Android's document picker instead of requiring the model to be copied into the app's Downloads folder.

For a single GGUF:

1. Tap **Pick GGUF**.
2. Select the model from internal storage, SD storage, or a document provider that supplies a seekable file descriptor.
3. The app opens it read-only.
4. Native code receives `/proc/self/fd/<fd>`.
5. The descriptor stays open until **Unload**.

This is important for very large files because no second app-private copy is made.

A document provider must expose a seekable descriptor for mmap. For unusual cloud-only providers, download the GGUF locally first. Multi-file/split GGUF is best supplied through a normal filesystem-visible location during the first validation pass because sibling-shard discovery from a single SAF descriptor is not emulated yet.

## Recommended S25 Ultra validation sequence

### Gate 1 — CPU correctness

Build without QNN/Vulkan and verify:

- app starts
- SAF model selection succeeds
- GGUF mmap load succeeds
- generation works
- unload/reload works
- repeated generation does not leak descriptors/native buffers

### Gate 2 — expert streaming

Use an MoE GGUF and verify with `deferExperts=true` and prefetch enabled:

- model starts without materializing all deferred experts
- cold expert pages fault/read from storage
- warm repeated routes reuse page cache
- resident memory remains bounded during route churn
- deterministic output matches the same build with prefetch disabled

### Gate 3 — QNN discovery

Build with `-PexpertQnn=true` and the staged QAIRT runtime:

- **Backends** reports the QNN runtime probe successfully
- model load registers `QNN0`
- Logcat shows `QNN requested; registered=1`
- no QAIRT/Hexagon skeleton version error appears

### Gate 4 — dense HTP parity

For a small F16 model or test graph:

- run a deterministic CPU baseline
- enable QNN
- compare logits/tokens
- verify HTP `MUL_MAT` execution via Logcat/profiling

### Gate 5 — routed HTP parity

For an MoE model:

- confirm `MUL_MAT_ID` is selected for decode
- confirm the router IDs are valid
- compare CPU vs QNN deterministic tokens
- test both F16 and MXFP4 expert tensors when available
- force route churn and verify QNN staging memory remains bounded by an expert projection/shape, not total model size

### Gate 6 — mixed backend

If Vulkan is compiled too:

- QNN-compatible operations execute on HTP
- Vulkan-capable fallback operations execute on Adreno where selected
- remaining operations execute on ARM CPU
- token parity remains stable

### Gate 7 — sustained phone behavior

Run a long generation and record:

- prompt processing tok/s
- decode tok/s
- RSS / Android memory pressure
- expert cache hit/miss behavior
- UFS read bandwidth on cold routes
- temperature / thermal throttling
- power draw if available

## Useful runtime knobs

Native QNN backend:

```text
GGML_QNN_BACKEND_LIB   override the HTP backend library name/path
GGML_QNN_MIN_BATCH     minimum batch rows before dense MUL_MAT is claimed by QNN
```

The routed decode path ignores `GGML_QNN_MIN_BATCH`; if `MUL_MAT_ID` satisfies the supported decode geometry it is eligible for HTP.

## Source map

```text
# Android app / lifecycle / UI
app/src/main/java/com/example/llama/MainActivity.kt
app/src/main/java/com/example/llama/MainViewModel.kt
app/src/main/java/com/example/llama/QnnRuntimeInstaller.kt

# Kotlin native API
llama/src/main/java/android/llama/cpp/LLamaAndroid.kt

# JNI
llama/src/main/cpp/expert-android-jni.cpp
llama/src/main/cpp/qnn-probe.cpp

# HTP backend
ggml/include/ggml-qnn.h
ggml/src/ggml-qnn.cpp

# Android native build
llama/src/main/cpp/CMakeLists.txt
llama/build.gradle.kts

# QAIRT staging
prepare-qnn-runtime.sh
```

## Important implementation notes

- The QNN backend currently uses QNN raw/client buffers. This is a valid HTP execution path and keeps the port self-contained. Qualcomm's shared-buffer registration path is a future zero-copy optimization, not a prerequisite for HTP execution.
- QNN MatMul graph handles are cached by shape until the QNN context is destroyed.
- Expert weight conversion is per selected expert, not model-wide.
- MXFP4 is dequantized with the fork's existing `dequantize_row_mxfp4` implementation and staged as FP16 for QNN.
- The app does not request broad external-storage permissions; SAF grants access only to the user-selected document.
- No Qualcomm binaries are committed by this branch.
- No performance number is claimed until the APK is compiled and run on a physical device.
