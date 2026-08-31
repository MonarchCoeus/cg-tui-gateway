# CG — AI TUI Gateway installer for Windows (PowerShell 5.1+ / 7).
# Downloads the latest release zip, installs to %LOCALAPPDATA%\cg-tui-gateway
# and adds a `cg.cmd` wrapper (in %USERPROFILE%\bin, added to your user PATH).
#
# Usage:  irm https://github.com/MonarchCoeus/cg-tui-gateway/releases/latest/download/install.ps1 | iex
param(
    [string]$Base = "https://github.com/MonarchCoeus/cg-tui-gateway/releases/latest/download",
    [string]$InstallDir = "$env:LOCALAPPDATA\cg-tui-gateway"
)
$ErrorActionPreference = "Stop"

Write-Host "==> CG — AI TUI Gateway installer"

# --- python check ---
$pyLauncher = $null
foreach ($c in @("py", "python")) {
    if (Get-Command $c -ErrorAction SilentlyContinue) { $pyLauncher = $c; break }
}
if (-not $pyLauncher) {
    Write-Host "error: Python 3.9+ required — install it from https://www.python.org/downloads/ (tick 'Add to PATH')"
    exit 1
}
Write-Host "==> using python launcher: $pyLauncher"

# --- download + extract ---
$zip = Join-Path $env:TEMP "cg-tui-gateway.zip"
$ex  = Join-Path $env:TEMP "cg-install"
Write-Host "==> downloading $Base/cg-tui-gateway.zip"
Invoke-WebRequest -Uri "$Base/cg-tui-gateway.zip" -OutFile $zip
if (Test-Path $ex) { Remove-Item $ex -Recurse -Force }
Expand-Archive -Path $zip -DestinationPath $ex -Force

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Copy-Item (Join-Path $ex "cg-tui-gateway\*") $InstallDir -Recurse -Force
# py.exe is picky about extension-less scripts; a .py copy always runs
Copy-Item (Join-Path $InstallDir "cg") (Join-Path $InstallDir "cg.py") -Force

# --- cg.cmd wrapper + PATH ---
$bin = Join-Path $HOME "bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$cmd = "@echo off`r`n$pyLauncher `"$InstallDir\cg.py`" %*`r`n"
Set-Content -Path (Join-Path $bin "cg.cmd") -Value $cmd -Encoding Ascii

$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$bin*") {
    [Environment]::SetEnvironmentVariable("Path", "$bin;$userPath", "User")
    Write-Host "==> added $bin to your user PATH (open a NEW terminal to pick it up)"
}

Write-Host "==> installed to $InstallDir"
Write-Host "==> done. Open a new terminal and run:  cg tui   (or:  cg serve)"
