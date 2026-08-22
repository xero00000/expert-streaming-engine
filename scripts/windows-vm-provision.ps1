#Requires -RunAsAdministrator

param(
    [string]$NodeVersion = "22.23.2",
    [string]$InstallRoot = "C:\ese-tools"
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
New-Item -ItemType Directory -Path $InstallRoot -Force | Out-Null

function Invoke-CheckedInstaller {
    param(
        [Parameter(Mandatory)] [string]$FilePath,
        [Parameter(Mandatory)] [string[]]$ArgumentList,
        [int[]]$AllowedExitCodes = @(0, 3010)
    )

    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -Wait -PassThru
    if ($process.ExitCode -notin $AllowedExitCodes) {
        throw "$FilePath failed with exit code $($process.ExitCode)"
    }
}

$nodeMsi = Join-Path $InstallRoot "node-v$NodeVersion-x64.msi"
if (-not (Get-Command node.exe -ErrorAction SilentlyContinue)) {
    Invoke-WebRequest "https://nodejs.org/dist/v$NodeVersion/node-v$NodeVersion-x64.msi" -OutFile $nodeMsi
    Invoke-CheckedInstaller msiexec.exe @("/i", $nodeMsi, "/qn", "/norestart")
}

$rustup = Join-Path $InstallRoot "rustup-init.exe"
if (-not (Test-Path "$env:USERPROFILE\.cargo\bin\cargo.exe")) {
    Invoke-WebRequest "https://win.rustup.rs/x86_64" -OutFile $rustup
    Invoke-CheckedInstaller $rustup @(
        "-y",
        "--profile", "minimal",
        "--default-host", "x86_64-pc-windows-msvc",
        "--default-toolchain", "stable"
    ) @(0)
}

$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasCppTools = (Test-Path $vswhere) -and [bool](& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if (-not $hasCppTools) {
    $buildTools = Join-Path $InstallRoot "vs_BuildTools.exe"
    Invoke-WebRequest "https://aka.ms/vs/17/release/vs_BuildTools.exe" -OutFile $buildTools
    Invoke-CheckedInstaller $buildTools @(
        "--quiet",
        "--wait",
        "--norestart",
        "--nocache",
        "--installPath", "C:\BuildTools",
        "--add", "Microsoft.VisualStudio.Workload.VCTools",
        "--includeRecommended"
    )
}

$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath;$env:USERPROFILE\.cargo\bin"

& "$env:ProgramFiles\nodejs\corepack.cmd" enable
& "$env:ProgramFiles\nodejs\corepack.cmd" prepare pnpm@10 --activate

$report = [ordered]@{
    node = (& node.exe --version)
    npm = (& npm.cmd --version)
    pnpm = (& pnpm.cmd --version)
    rustc = (& rustc.exe --version)
    cargo = (& cargo.exe --version)
    cppBuildTools = (& $vswhere -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
}
$report | ConvertTo-Json | Set-Content -LiteralPath "C:\ese-windows-toolchain.json" -Encoding utf8
$report | Format-List
