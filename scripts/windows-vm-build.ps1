param(
    [string]$SourceRoot = "C:\ese-src"
)

$ErrorActionPreference = "Stop"
$env:Path = "$env:ProgramFiles\nodejs;$env:USERPROFILE\.cargo\bin;$env:Path"
$studio = Join-Path $SourceRoot "studio"
Set-Location $studio

function Invoke-Checked {
    param(
        [Parameter(Mandatory)] [string]$Command,
        [Parameter(ValueFromRemainingArguments)] [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Command failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked node.exe --version
Invoke-Checked pnpm.cmd --version
Invoke-Checked rustc.exe --version
Invoke-Checked pnpm.cmd install --frozen-lockfile
Invoke-Checked pnpm.cmd build
Invoke-Checked cargo.exe check --manifest-path "src-tauri\Cargo.toml" --all-targets
