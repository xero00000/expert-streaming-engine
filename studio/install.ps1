[CmdletBinding()]
param(
    [switch]$Check
)

$ErrorActionPreference = "Stop"
$studioRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$missing = [System.Collections.Generic.List[string]]::new()

foreach ($command in @("node", "cargo", "rustc")) {
    if (-not (Get-Command $command -ErrorAction SilentlyContinue)) {
        $missing.Add($command)
    }
}

$pnpmMissing = -not (Get-Command "pnpm" -ErrorAction SilentlyContinue)
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$cppReady = (Test-Path $vswhere) -and [bool](& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath)
if (-not $cppReady) {
    $missing.Add("Visual Studio C++ Build Tools")
}

$webViewReady = (Test-Path "${env:ProgramFiles(x86)}\Microsoft\EdgeWebView\Application") -or
    (Test-Path "$env:ProgramFiles\Microsoft\EdgeWebView\Application")
if (-not $webViewReady) {
    $missing.Add("Microsoft Edge WebView2 Runtime")
}

Write-Host "ESE Studio dependency preflight"
Write-Host "  Windows: $([Environment]::OSVersion.VersionString)"
if ($missing.Count -eq 0 -and -not $pnpmMissing) {
    Write-Host "  Required build dependencies: ready"
} else {
    if ($missing.Count -gt 0) {
        Write-Host "  Missing: $($missing -join ', ')"
    }
    if ($pnpmMissing) {
        Write-Host "  Missing JavaScript package manager: pnpm"
    }
}

foreach ($optional in @("ese", "nvidia-smi")) {
    if (-not (Get-Command $optional -ErrorAction SilentlyContinue)) {
        Write-Host "  Optional runtime tool not found: $optional"
    }
}

if ($Check) {
    if ($missing.Count -gt 0 -or $pnpmMissing) { exit 2 }
    exit 0
}

if ($missing.Count -gt 0) {
    if (-not (Get-Command "winget" -ErrorAction SilentlyContinue)) {
        throw "Install the missing prerequisites listed above, then run this script again. winget is unavailable for assisted installation."
    }
    $reply = Read-Host "Install missing prerequisites with winget? [y/N]"
    if ($reply -notmatch '^[Yy]$') {
        Write-Host "No packages were installed."
        exit 2
    }
    if ($missing -contains "node") {
        winget install --exact --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
    }
    if (($missing -contains "cargo") -or ($missing -contains "rustc")) {
        winget install --exact --id Rustlang.Rustup --accept-package-agreements --accept-source-agreements
    }
    if ($missing -contains "Visual Studio C++ Build Tools") {
        winget install --exact --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended" --accept-package-agreements --accept-source-agreements
    }
    if ($missing -contains "Microsoft Edge WebView2 Runtime") {
        winget install --exact --id Microsoft.EdgeWebView2Runtime --accept-package-agreements --accept-source-agreements
    }
    Write-Host "Dependency installation finished. Open a new PowerShell window, then rerun install.ps1."
    exit 0
}

if ($pnpmMissing) {
    $reply = Read-Host "Enable pnpm with Corepack? [y/N]"
    if ($reply -notmatch '^[Yy]$') {
        Write-Host "pnpm was not enabled."
        exit 2
    }
    corepack enable pnpm
}

Push-Location $studioRoot
try {
    pnpm install --frozen-lockfile
    pnpm tauri build --bundles nsis,msi
} finally {
    Pop-Location
}
Write-Host "Build complete. Windows installers are under src-tauri\target\release\bundle\."
