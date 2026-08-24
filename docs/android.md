# Android status

Android is not currently a supported ESE target. `main` does not contain a
validated QNN/NPU backend, an Android package, or physical-device evidence for
the bounded expert hierarchy. Historical references to an `android-s25-qnn`
branch describe an earlier draft; that branch is no longer present on the
remote.

## Experimental Termux CPU build

Advanced users can attempt a native CPU-only build in Termux, but it is an
unsupported development path and does not provide ESE Studio, CUDA, QNN, or the
Linux-only bounded expert RAM cache.

Install the basic tools:

```bash
pkg update
pkg install git cmake ninja clang python
```

Clone this repository—not an upstream `llama.cpp` fork—and configure a portable
CPU build:

```bash
git clone https://github.com/xero00000/expert-streaming-engine.git
cd expert-streaming-engine
cmake -S . -B build-android -G Ninja \
  -DGGML_CUDA=OFF \
  -DGGML_NATIVE=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build-android --target llama-server
```

Termux package availability and Android linker behavior can change. Treat a
successful compile as an experiment, not as proof of model parity, bounded
memory, thermal stability, or supported operation.

## Requirements before Android can be supported

An Android backend must provide all of the following before this page can claim
support:

- reproducible on-device build and packaging;
- exact token/output and route parity against a supported CPU reference;
- bounded RAM/device-memory accounting and failure-atomic lifecycle behavior;
- proof that the intended GPU/NPU kernels execute without hidden CPU fallback;
- sustained performance and thermal measurements on physical devices; and
- CI or retained device evidence covering the published configuration.

Track new Android work through a current pull request or issue from `main`; do
not rely on deleted branch names as installation instructions.
