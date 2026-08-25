# Install Expert Streaming Engine

ESE Studio installers contain the matching `ese` launcher and native
`llama-server` runtime. Do not install a generic `llama.cpp` package when you
need ESE's planner, resource controller, or bounded expert hierarchy.

## Published packages

Download the package and checksum file from the
[latest GitHub release](https://github.com/xero00000/expert-streaming-engine/releases/latest).
Package filenames include the release version; the examples below use `0.2.0`.

### Linux

The AppImage supports signed in-app updates:

```bash
sha256sum --check --ignore-missing SHA256SUMS
chmod +x ese-studio_0.2.0_amd64.AppImage
./ese-studio_0.2.0_amd64.AppImage
```

Distribution packages integrate with APT or DNF:

```bash
sudo apt install ./ese-studio_0.2.0_amd64.deb
sudo dnf install ./ese-studio-0.2.0-1.x86_64.rpm
```

Published Linux packages use a portable CPU runtime. Build from source when you
want CUDA acceleration on Linux.

### Windows

Download `ese-studio_0.2.0_x64-setup.exe` (recommended) or the MSI, compare its
SHA-256 value with `SHA256SUMS-windows.txt`, and run it. The package includes a
CUDA-enabled runtime for supported NVIDIA GPUs and retains CPU fallback. Python,
CMake, Visual Studio, and the CUDA toolkit are not required to run the installed
application.

The NSIS installation supports signed update checks from **Settings → Updates**.

## Install from source

On Linux:

```bash
git clone https://github.com/xero00000/expert-streaming-engine.git
cd expert-streaming-engine
./studio/scripts/install-local.sh
ese doctor
```

On Windows, clone the repository and run from PowerShell:

```powershell
cd expert-streaming-engine\studio
.\install.ps1 -Check
.\install.ps1
```

Both installers show missing dependencies and ask before installing anything.
For a command-line-only build or manual CMake controls, see the
[source-build guide](build.md).

## Verify an installation

```bash
ese --version
ese doctor
ese plan /models/model.gguf
```

On Windows, use a Windows model path, for example:

```powershell
ese.exe plan C:\Models\model.gguf
```

The server listens on `http://127.0.0.1:8080` by default after `ese serve`.
