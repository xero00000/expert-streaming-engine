$ErrorActionPreference = "Stop"

$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
$downloadRoot = Join-Path $env:RUNNER_TEMP "ese-cuda-12.4"
New-Item -ItemType Directory -Force -Path $cudaRoot, $downloadRoot | Out-Null

$packages = @(
    @("cuda_cudart", "12.4.127"),
    @("cuda_nvcc", "12.4.131"),
    @("cuda_nvrtc", "12.4.127"),
    @("libcublas", "12.4.5.8"),
    @("cuda_cccl", "12.4.127"),
    @("visual_studio_integration", "12.4.127")
)

foreach ($package in $packages) {
    $name = $package[0]
    $version = $package[1]
    $archiveName = "$name-windows-x86_64-$version-archive"
    $archive = Join-Path $downloadRoot "$archiveName.zip"
    $extract = Join-Path $downloadRoot $archiveName
    $url = "https://developer.download.nvidia.com/compute/cuda/redist/$name/windows-x86_64/$archiveName.zip"

    Write-Host "Downloading $name $version"
    Invoke-WebRequest -Uri $url -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $extract -Force
    $contents = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
    if (-not $contents) {
        throw "$archiveName did not contain an archive root"
    }
    Copy-Item -Path (Join-Path $contents.FullName "*") -Destination $cudaRoot -Recurse -Force
}

Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$cudaRoot"
Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH_V12_4=$cudaRoot"
Add-Content -Path $env:GITHUB_PATH -Value (Join-Path $cudaRoot "bin")

& (Join-Path $cudaRoot "bin\nvcc.exe") --version
