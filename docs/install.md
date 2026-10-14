# Install Expert Streaming Engine

ESE Studio gives you a model library, local chat and measured configuration
sweeps. Each package includes the matching ESE launcher and native server.

These instructions describe the prepared **v0.2.1 candidate**. Publication and
signing remain pending; the latest public release may still be an earlier
version. Check the version on the [GitHub release page](https://github.com/xero00000/expert-streaming-engine/releases)
before choosing a file.

## Pick the package for your PC

| System | Download | Requirements |
| --- | --- | --- |
| Linux, CPU-only | `ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage`, `.deb` or `.rpm` | x86_64, Python 3.10+, system OpenMP and a graphical session for Studio |
| Linux, NVIDIA GPU | The same filename with `_cuda` instead of `_cpu` | Compatible NVIDIA driver; CUDA toolkit not required to run the package |
| Windows | `ESE-Studio_0.2.1_windows-x64-setup.exe` or `ESE-Studio_0.2.1_windows-x64.msi` | x64 Windows; NVIDIA driver for CUDA, or automatic selection of the separate CPU runtime |

Windows packages include both runtimes. Python, CMake, MSVC and the CUDA toolkit
are not required to run the installed Windows application.

## Verify and install

Download `SHA256SUMS` with your package. Check its SHA-256 before installation.
The current candidate has no verified publisher/updater signatures; apply
updates by installing the next package manually.

### Linux

The AppImage is convenient for trying Studio without a system-wide install:

```bash
sha256sum --check --ignore-missing SHA256SUMS
chmod +x ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage
./ESE-Studio_0.2.1_linux-x86_64_cpu.AppImage
```

Replace `_cpu` with `_cuda` for NVIDIA acceleration. For package-manager
integration use either:

```bash
sudo apt install ./ESE-Studio_0.2.1_linux-x86_64_cpu.deb
sudo dnf install ./ESE-Studio_0.2.1_linux-x86_64_cpu.rpm
```

DEB/RPM packages declare Python/OpenMP dependencies. AppImage users provide
`python3` and `libgomp1` on Ubuntu/Debian, or `python3` and `libgomp` on Fedora.
The package checks cover Ubuntu 24.04 and Fedora 44; other distribution versions
are not established by those installation checks.

### Windows

In PowerShell, compare the hash with the matching line in `SHA256SUMS`:

```powershell
Get-FileHash .\ESE-Studio_0.2.1_windows-x64-setup.exe -Algorithm SHA256
```

Run the setup EXE (or MSI), open ESE Studio and add your model folder. Current
installers are unsigned. CPU executable and launcher selection checks passed in
CI; actual installer execution and Windows GPU model inference remain
unverified by that hosted run.

## Check your PC and first model

Use [What can my PC run?](WHAT_CAN_MY_PC_RUN.md) before choosing a large download.
For the command line:

```bash
ese --version
ese doctor
ese plan /models/model.gguf --context 65536
```

On Windows use `ese.exe` and `C:\Models\model.gguf`. `plan` inspects the model;
it does not benchmark a 64K prompt. The [demo record](LAUNCH_DEMO.md) identifies
which exact model/context combinations have actually been measured.

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

Both source installers show missing dependencies and ask before installing
them. See the [source-build guide](build.md) for manual CMake controls.
