# Android / Galaxy S25 Ultra port

The Android port lives on the `android-s25-qnn` branch and is implemented under:

- [`examples/llama.android/README.md`](examples/llama.android/README.md) — architecture, build modes and physical-device validation gates
- [`examples/llama.android/`](examples/llama.android/) — Android application / JNI target
- [`ggml/src/ggml-qnn.cpp`](ggml/src/ggml-qnn.cpp) — optional Qualcomm QAIRT/QNN HTP backend
- [`ggml/include/ggml-qnn.h`](ggml/include/ggml-qnn.h) — QNN backend API

## What is implemented

- native arm64-v8a Android/NDK build of this fork
- Storage Access Framework GGUF selection without making a second model copy
- mmap/deferred-expert loading and route-aware expert prefetch
- CPU backend and optional Adreno Vulkan backend
- optional `QNN0` backend for Qualcomm Hexagon HTP
- HTP dense `GGML_OP_MUL_MAT`
- HTP decode-time routed `GGML_OP_MUL_MAT_ID`
- F16/F32 activation/output support
- F16/F32/MXFP4 weight input support
- selected-expert MXFP4 -> FP16 staging rather than model-wide conversion
- one reusable QNN staging arena, default maximum 256 MiB
- automatic CPU/Vulkan fallback for unsupported or oversized QNN operations
- QAIRT runtime staging helper and Android-side DSP skeleton extraction
- Android UI for backend status, streaming controls, chat and benchmark
- deterministic native teardown on model unload / ViewModel destruction

## Deliberately not claimed

This branch has not been compiled or benchmarked on a physical S25 Ultra as part of the source-port work. No performance or parity result should be treated as validated until the physical-device gates in `examples/llama.android/README.md` are run.

The QNN backend currently uses QNN raw/client buffers. Qualcomm shared-buffer registration can be added later as a copy-reduction optimization without changing the Android/ggml backend architecture.

## QAIRT build handoff

```bash
cd examples/llama.android
export QNN_SDK_ROOT=/path/to/qairt
bash prepare-qnn-runtime.sh
./gradlew :app:assembleDebug -PexpertQnn=true -PqnnSdkRoot="$QNN_SDK_ROOT"
```

For Vulkan + QNN:

```bash
./gradlew :app:assembleDebug \
  -PexpertQnn=true \
  -PexpertVulkan=true \
  -PqnnSdkRoot="$QNN_SDK_ROOT"
```

Advanced QNN environment knobs used by the native backend:

```text
GGML_QNN_BACKEND_LIB    default: libQnnHtp.so
GGML_QNN_MAX_STAGE_MIB  default: 256, allowed: 16..4096
GGML_QNN_MIN_BATCH      default: 1 for dense MUL_MAT
```

`MUL_MAT_ID` routed decode is not gated by `GGML_QNN_MIN_BATCH`; it is selected by supported decode geometry and the staging-memory ceiling.
