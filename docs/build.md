# Build ESE from source

This guide builds Expert Streaming Engine from this repository. ESE is derived
from `ik_llama.cpp` and `llama.cpp`, but cloning either upstream project does
not install ESE's launcher, resource controller, or bounded expert hierarchy.

## Requirements

- Python 3.10 or newer
- CMake and a C/C++ compiler
- Git
- Optional: an NVIDIA CUDA toolkit for the CUDA backend

ESE currently supports source builds directly on Linux and Windows. The native
bounded expert RAM cache is Linux-only; Windows Studio and native builds support
the packaged CUDA/CPU runtime without claiming that Linux-only cache path.

## Linux

Clone ESE, inspect the host, and let the launcher choose CUDA when a usable
toolchain is present:

```bash
git clone https://github.com/xero00000/expert-streaming-engine.git
cd expert-streaming-engine

./ese doctor
./ese build
```

Force a backend when reproducibility matters:

```bash
./ese build --backend cpu
./ese build --backend cuda --clean
```

Useful build controls are:

```text
--backend auto|cuda|cpu
--build-dir PATH
--build-type TYPE
--jobs N
--clean
```

The default build directory is `build/`. Verify the result with:

```bash
./ese --version
build/bin/llama-server --version
./ese plan /models/model.gguf
```

## Windows

Open PowerShell in the cloned repository. The checked-in `ese.cmd` wrapper runs
the same launcher interface:

```powershell
.\ese.cmd doctor
.\ese.cmd build --backend cpu
```

For CUDA, install a supported NVIDIA driver and CUDA toolkit, then run:

```powershell
.\ese.cmd build --backend cuda --clean
```

Windows compilation requires the Visual Studio 2022 C++ Build Tools. ESE Studio
contributors can run the dependency preflight and package build from `studio`:

```powershell
cd studio
.\install.ps1 -Check
.\install.ps1
```

The script displays missing prerequisites and asks before using `winget`.

## Manual CMake build

Use this only when developing the native runtime directly; normal users should
prefer `./ese build` so hardware detection and paths remain consistent.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j2
```

Set `-DGGML_CUDA=ON` for a CUDA build. Architecture-specific CMake and backend
options remain available for native development, but published packages use
portable targets rather than `GGML_NATIVE=ON`.

## Build ESE Studio locally

For a user-local Linux installation containing Studio, the launcher, and the
native runtime:

```bash
./studio/scripts/install-local.sh
ese doctor
```

For development without installation:

```bash
cd studio
pnpm install --frozen-lockfile
pnpm tauri dev
```

See the [installation guide](install.md), [Studio guide](../studio/README.md),
and [contribution evidence requirements](../CONTRIBUTING.md).
