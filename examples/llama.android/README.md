# Expert Streaming Engine for Android

Native Android port of the Expert Streaming Engine, targeting the **Samsung Galaxy S25 Ultra / Snapdragon 8 Elite** first.

Branch: `android-s25-qnn`

This is a real NDK/JNI port of the engine in this repository. It is **not** a remote-server wrapper, and it does not require converting GGUF models into a separate NPU model format.

> **Status:** the Android + Qualcomm QNN/HTP source port is implemented and source-audited. It has intentionally **not** been claimed as compiled, device-validated, token-parity validated, or benchmarked on a physical S25 Ultra yet. The first local Android SDK/NDK + QAIRT build is the next validation step.

---

## What this port does

The Android build keeps the original Expert Streaming Engine design:

- models stay as GGUF files;
- GGUF data remains mmap-backed;
- MoE experts can remain deferred on storage;
- router-selected expert pages are prefetched from storage;
- supported compute can run on Qualcomm Hexagon HTP through QNN;
- Adreno Vulkan can be built as an additional accelerator backend;
- unsupported operations fall back to the ARM CPU or Vulkan backend;
- the full model is never permanently converted into a second NPU-specific copy.

For routed MXFP4 MoE decode, only the **selected expert projection** is dequantized/transposed into a reusable FP16 QNN staging arena for the current HTP MatMul.

---

## Architecture

```text
Android Storage Access Framework
        |
        | selected GGUF
        | held ParcelFileDescriptor
        v
/proc/self/fd/<fd>
        |
        v
GGUF mmap on UFS
        |
        +--> deferred experts
        +--> route-aware expert prefetch
        +--> page-cache reuse
        |
        v
GGML scheduler
   |
   +--> QNN0 / Hexagon HTP
   |       +-- dense MUL_MAT
   |       +-- decode MUL_MAT_ID
   |       +-- F16/F32 activations
   |       +-- F16/F32/MXFP4 weights
   |       +-- bounded reusable FP16 staging
   |
   +--> Adreno Vulkan (optional)
   |
   +--> ARM CPU
           +-- unsupported ops
           +-- normal CPU fallback
           +-- CPU MoE/prefill where appropriate
```

The QNN backend is deliberately hybrid. It only reports support for operations it actually implements, allowing the existing ggml scheduler to split graphs between HTP, Vulkan, and CPU.

---

## Implemented Android features

- native `arm64-v8a` Android/NDK build target;
- minSdk 33;
- direct JNI integration with this fork;
- GGUF mmap loading;
- Android Storage Access Framework model picker;
- model access through `/proc/self/fd/<fd>` without making a second app-private model copy;
- held file descriptor for the entire loaded-model lifetime;
- `defer_experts` support from the Android UI/native bridge;
- route-aware expert prefetch;
- configurable expert-prefetch worker count;
- configurable context, batch, ubatch, CPU thread, accelerator-layer, and CPU-MoE settings;
- chat/generation UI;
- native benchmark button;
- backend-status reporting;
- explicit load/unload controls;
- deterministic native model/context cleanup;
- deterministic SAF descriptor cleanup;
- optional Vulkan backend;
- optional Qualcomm QNN/HTP backend;
- automatic CPU/Vulkan fallback when QNN is unavailable or an operation is unsupported;
- runtime Hexagon skeleton extraction and `ADSP_LIBRARY_PATH` setup;
- Qualcomm runtime binaries excluded from Git.

---

# Qualcomm QNN / Hexagon HTP backend

Core backend files:

```text
ggml/include/ggml-qnn.h
ggml/src/ggml-qnn.cpp
```

The backend registers with the existing ggml backend registry as:

```text
QNN0
```

## QNN initialization

The implementation:

1. dynamically loads `libQnnHtp.so`;
2. resolves `QnnInterface_getProviders`;
3. enumerates QNN interface providers;
4. selects a provider compatible with the QNN headers used to build the APK;
5. creates the QNN backend;
6. creates the QNN context;
7. registers `QNN0` with ggml;
8. lets the existing scheduler assign compatible graph operations to HTP.

Registration is retryable. If HTP discovery initially fails because the DSP library path has not been configured yet, the app can configure the extracted Hexagon runtime and try registration again.

If QNN was requested but cannot be registered, the engine retains its normal CPU/Vulkan graph behavior rather than forcing the QNN-specific MoE graph shape.

---

## HTP operator support

| ggml operation | HTP support | Weight formats | Fallback |
| --- | --- | --- | --- |
| `GGML_OP_MUL_MAT` | Yes | F16, F32, MXFP4 | CPU / Vulkan |
| `GGML_OP_MUL_MAT_ID` | Yes, routed decode path | F16, F32, MXFP4 | CPU / Vulkan |
| fused MoE up/gate | Not executed directly by QNN | — | standard routed graph while QNN is active |
| other ggml operations | Not claimed by QNN | — | CPU / Vulkan |

### Dense MatMul

The QNN backend supports standard dense matrix multiplication with:

- F16 or F32 activation input;
- F16 or F32 output;
- F16, F32, or MXFP4 weights;
- shape-cached QNN MatMul graphs.

### Routed MoE decode

The backend also implements decode-time `GGML_OP_MUL_MAT_ID`.

The router ID tensor selects the active experts. Instead of converting every expert or every layer, the backend stages only the expert(s) used by that decode operation.

This is the important Android data path:

```text
GGUF expert on UFS
        |
        v
mmap / prefetched expert extent
        |
        v
selected expert only
        |
        +-- F16/F32 -> FP16 staging
        |
        +-- MXFP4 -> dequantize selected rows -> FP16 staging
        |
        v
QNN HTP MatMul
```

---

# Bounded QNN staging memory

QNN currently uses raw/client buffers for execution.

To avoid accumulating one FP16 copy for every graph shape encountered during a long session, the backend uses **one reusable staging arena** shared across MatMul executions.

Default maximum staging memory:

```text
256 MiB
```

The limit is controlled by:

```text
GGML_QNN_MAX_STAGE_MIB
```

Accepted range:

```text
16 .. 4096 MiB
```

If an operation would require more staging memory than the configured ceiling, QNN rejects that operation during backend selection and the scheduler uses another backend instead of attempting an unbounded allocation.

QNN graph handles themselves are cached by logical MatMul shape `(N, K, M)` until the QNN context is destroyed. The large temporary FP16 buffers are **not** cached per shape.

---

# Model storage and Android SAF

The app uses Android's document picker rather than requiring broad storage permissions or forcing models into the app's private directory.

## Loading a model

1. Tap **Pick GGUF**.
2. Select a local GGUF file.
3. Android returns a `ParcelFileDescriptor`.
4. The descriptor remains open while the model is loaded.
5. Native code loads:

```text
/proc/self/fd/<fd>
```

6. llama/ggml mmap the selected file normally.
7. Tapping **Unload**, or destroying the owning Android runtime, releases both native resources and the descriptor.

This avoids duplicating multi-gigabyte GGUF files into app-private storage.

### SAF limitation

The document provider must expose a seekable file descriptor for mmap.

Cloud-only providers that do not expose a normal seekable descriptor should be downloaded locally first.

For the first physical-device validation pass, multi-file/split GGUF models are best placed in a normal filesystem-visible location because sibling-shard discovery is not emulated through a single SAF descriptor.

---

# Build prerequisites

## CPU-only build

You need:

- Android Studio or Android command-line tools;
- Android SDK;
- Android NDK;
- CMake supported by the Android project;
- JDK/Gradle environment expected by the project.

## Vulkan build

The host machine also needs a Vulkan SDK and `glslc` because ggml's Vulkan shaders are generated during the build.

## QNN / HTP build

You additionally need a local Qualcomm **QAIRT/QNN SDK** containing Android ARM64 QNN headers/runtime files compatible with the Snapdragon HTP runtime you intend to deploy.

Use one QAIRT release consistently. Do not mix arbitrary QNN host libraries and Hexagon skeletons from different releases.

---

# Build commands

All commands below are run from:

```bash
cd examples/llama.android
```

## CPU / ARM only

```bash
./gradlew :app:assembleDebug
```

## Vulkan / Adreno

```bash
./gradlew :app:assembleDebug \
  -PexpertVulkan=true
```

## Qualcomm QNN / HTP

Point the project at your local QAIRT SDK:

```bash
export QNN_SDK_ROOT=/path/to/qairt
```

Stage the runtime files:

```bash
bash prepare-qnn-runtime.sh
```

Then build:

```bash
./gradlew :app:assembleDebug \
  -PexpertQnn=true \
  -PqnnSdkRoot="$QNN_SDK_ROOT"
```

## QNN + Vulkan

```bash
./gradlew :app:assembleDebug \
  -PexpertQnn=true \
  -PexpertVulkan=true \
  -PqnnSdkRoot="$QNN_SDK_ROOT"
```

No APK build or device result is claimed by this branch yet; these commands are the intended local handoff for the first compile.

---

# QAIRT runtime staging

Helper:

```text
prepare-qnn-runtime.sh
```

The script copies files from the builder's **existing local Qualcomm SDK**. It does not download QAIRT and does not commit Qualcomm binaries.

Android-side runtime libraries are staged under:

```text
llama/src/main/jniLibs/arm64-v8a/
```

Example contents:

```text
libQnnHtp.so
libQnnSystem.so
other required ARM64 QAIRT runtime libraries
```

Hexagon-side runtime/skeleton files are staged under:

```text
app/src/main/assets/qnn/
```

At runtime, `QnnRuntimeInstaller` extracts the DSP-side files to an app-private directory. The JNI layer adds that directory to:

```text
ADSP_LIBRARY_PATH
```

before QNN registration/model execution.

The staging directories are ignored by Git so licensed Qualcomm runtime binaries are not accidentally committed.

---

# Engine configuration

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

Current Android defaults:

```text
contextSize       = 4096
threads           = 0      # automatic
batchSize         = 512
ubatchSize        = 256
maxTokens         = 128
deferExperts      = true
prefetchExperts   = true
prefetchThreads   = 2
gpuLayers         = 0
cpuMoeLayers      = -1     # engine default/automatic
useQnn            = true
qnnDspLibraryPath = null   # populated by Android runtime installer when available
```

The automatic CPU thread count leaves cores available for Android/UI/background work rather than consuming every online core by default.

---

# Native QNN runtime knobs

## Backend library

```text
GGML_QNN_BACKEND_LIB
```

Default:

```text
libQnnHtp.so
```

Use this only when you need to override the HTP backend library name/path.

## Maximum temporary staging memory

```text
GGML_QNN_MAX_STAGE_MIB
```

Default:

```text
256
```

Allowed:

```text
16 .. 4096
```

## Minimum dense MatMul batch

```text
GGML_QNN_MIN_BATCH
```

Controls when dense `GGML_OP_MUL_MAT` becomes eligible for HTP.

The routed decode `GGML_OP_MUL_MAT_ID` path is not gated by this dense-MatMul threshold; routed offload is decided from its supported decode geometry and staging-memory limit.

---

# Recommended Galaxy S25 Ultra validation

The source port should not be called device-complete until these gates pass on a physical phone.

## Gate 1 — CPU correctness

Build without QNN or Vulkan.

Verify:

- application launches;
- GGUF picker opens;
- SAF model loading succeeds;
- mmap succeeds;
- generation succeeds;
- unload/reload succeeds;
- exiting the owning UI releases native model/context resources;
- the SAF descriptor is released;
- repeated generation/load cycles do not leak descriptors or native buffers.

## Gate 2 — expert streaming

Use an MoE GGUF with:

```text
deferExperts = true
prefetchExperts = true
```

Verify:

- deferred experts are not all materialized into resident RAM;
- cold expert routes generate storage/page-cache activity;
- repeated routes benefit from warm pages;
- resident memory remains bounded during route churn;
- deterministic output matches the same build with expert prefetch disabled.

## Gate 3 — QNN discovery

Build with QNN and staged QAIRT runtime files.

Verify:

- backend status can register `QNN0`;
- Logcat reports successful QNN registration;
- the selected HTP provider is compatible with the build headers;
- no Hexagon skeleton/runtime version mismatch is reported.

## Gate 4 — dense HTP parity

Use a small F16 model or controlled graph.

Verify:

- deterministic CPU baseline;
- QNN-enabled output matches within the expected FP16 tolerance;
- HTP executes compatible `MUL_MAT` operations;
- unsupported operations remain on CPU/Vulkan.

## Gate 5 — routed HTP parity

Use an MoE model.

Verify:

- decode uses `MUL_MAT_ID` while QNN is actually active;
- router IDs are valid;
- CPU and QNN deterministic token streams agree;
- F16 expert path works;
- MXFP4 selected-expert path works;
- route churn does not increase QNN staging memory beyond the configured arena limit.

## Gate 6 — mixed HTP + Vulkan + CPU

Build both accelerator backends.

Verify:

- QNN-compatible operations execute on Hexagon HTP;
- suitable fallback GPU operations can execute on Adreno Vulkan;
- remaining operations execute on ARM CPU;
- graph transitions remain correct;
- token parity remains stable.

## Gate 7 — sustained mobile behavior

Run a long generation and record:

- prompt-processing tok/s;
- decode tok/s;
- RSS / Android memory pressure;
- QNN staging high-water mark;
- expert cache/page-cache behavior;
- UFS read bandwidth on cold routes;
- SoC temperature;
- Android thermal-throttling state;
- sustained performance after warm-up;
- power consumption if available.

---

# Troubleshooting

## `QNN0` does not appear

Check:

- the APK was built with `-PexpertQnn=true`;
- `qnnSdkRoot` points to the intended QAIRT SDK;
- `libQnnHtp.so` was packaged for ARM64 Android;
- Hexagon-side runtime files were staged/extracted;
- `ADSP_LIBRARY_PATH` contains the extracted runtime directory;
- host and DSP QNN files come from compatible QAIRT versions.

QNN registration is retryable, so fixing the DSP path before model load does not require changing the model format or engine configuration.

## QNN is available but an operation runs on CPU

That can be correct.

The backend intentionally rejects operations when:

- the ggml op is unsupported;
- the tensor type/layout is unsupported;
- the geometry is unsupported;
- the required QNN staging would exceed `GGML_QNN_MAX_STAGE_MIB`;
- the operation does not satisfy the dense offload policy.

The scheduler should then choose Vulkan or CPU.

## Large model fails through a cloud document provider

Move/download the GGUF to normal local storage and select the local file. The current Android path expects a seekable descriptor suitable for mmap.

## Split GGUF cannot find sibling shards

For the initial Android validation, place split shards in a normal filesystem-visible directory and load from a path where the engine can discover sibling files normally. A single SAF descriptor does not emulate sibling-file discovery.

---

# Project layout

```text
examples/llama.android/
|
+-- app/
|   +-- src/main/java/com/example/llama/
|       +-- MainActivity.kt
|       +-- MainViewModel.kt
|       +-- QnnRuntimeInstaller.kt
|
+-- llama/
|   +-- src/main/java/android/llama/cpp/
|   |   +-- LLamaAndroid.kt
|   |
|   +-- src/main/cpp/
|       +-- CMakeLists.txt
|       +-- expert-android-jni.cpp
|       +-- llama-android.cpp
|       +-- qnn-probe.cpp
|       +-- qnn-probe.h
|
+-- prepare-qnn-runtime.sh

ggml/
+-- include/ggml-qnn.h
+-- src/ggml-qnn.cpp
```

The root repository also contains:

```text
ANDROID_PORT.md
```

for a shorter project-level handoff summary.

---

# Current limitations / intentionally unfinished validation

The following are **not** being claimed yet:

- successful Android SDK/NDK build on the final branch;
- successful APK installation on the S25 Ultra;
- confirmed HTP execution on the physical device;
- CPU ↔ QNN token/logit parity results;
- measured QNN speedup;
- measured UFS expert-streaming throughput;
- measured thermals or power efficiency;
- validated zero-copy QNN shared-buffer execution;
- complete QNN coverage of all transformer operations;
- transparent split-GGUF sibling discovery from a single SAF document descriptor.

QNN currently uses raw/client buffers. Qualcomm shared-buffer registration is a future copy-reduction optimization; it is not required for the current HTP execution architecture.

---

# Design rule

The Android port should continue to preserve the same core property as the desktop Expert Streaming Engine:

> **Keep the full model on storage, keep resident memory bounded, and move only the weights needed for the work being executed.**

For the S25 Ultra QNN path, that means:

```text
UFS GGUF
  -> mmap/deferred experts
  -> router-selected expert extent
  -> bounded reusable FP16 staging
  -> Hexagon HTP
  -> mixed-backend ggml scheduler
```

No full-model NPU conversion is required.