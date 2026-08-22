#Requires -RunAsAdministrator

param(
    [string]$AuthorizedKeyPath = (Join-Path $PSScriptRoot "authorized_key.pub")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $AuthorizedKeyPath -PathType Leaf)) {
    throw "Authorized SSH key not found: $AuthorizedKeyPath"
}

$capability = Get-WindowsCapability -Online -Name "OpenSSH.Server~~~~0.0.1.0"
if ($capability.State -ne "Installed") {
    Add-WindowsCapability -Online -Name $capability.Name | Out-Null
}

$sshDirectory = Join-Path $env:ProgramData "ssh"
$authorizedKeys = Join-Path $sshDirectory "administrators_authorized_keys"
New-Item -ItemType Directory -Path $sshDirectory -Force | Out-Null
(Get-Content -LiteralPath $AuthorizedKeyPath -Raw).Trim() |
    Set-Content -LiteralPath $authorizedKeys -Encoding ascii

& icacls.exe $authorizedKeys /inheritance:r /grant "SYSTEM:F" "Administrators:F" | Out-Null

Set-Service -Name sshd -StartupType Automatic
Start-Service -Name sshd

if (-not (Get-NetFirewallRule -Name "ESE-VM-OpenSSH" -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule `
        -Name "ESE-VM-OpenSSH" `
        -DisplayName "ESE Windows VM OpenSSH" `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 22 `
        -Profile Any | Out-Null
}

$windows = Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber, OsArchitecture
$windows | ConvertTo-Json | Set-Content -LiteralPath "C:\ese-windows-info.json" -Encoding utf8

Write-Host "ESE Windows VM remote control is ready."
