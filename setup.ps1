#Requires -Version 5.1
<#
.SYNOPSIS
    Helix Transvoicer - Windows 11 Setup Script

.DESCRIPTION
    Installs Helix Transvoicer with all dependencies on Windows 11.
    Supports both CUDA (NVIDIA GPU) and CPU-only installations.

.PARAMETER CudaVersion
    CUDA version to install (11.8, 12.1, or "cpu" for CPU-only)

.PARAMETER PythonPath
    Path to Python executable (optional, will auto-detect)

.EXAMPLE
    .\setup.ps1
    .\setup.ps1 -CudaVersion "12.1"
    .\setup.ps1 -CudaVersion "cpu"
#>

param(
    [ValidateSet("11.8", "12.1", "cpu")]
    [string]$CudaVersion = "12.1",

    [string]$PythonPath = ""
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
function Write-Header { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Step { param($msg) Write-Host "  -> $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "     $msg" -ForegroundColor Gray }
function Write-Warn { param($msg) Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "  [X] $msg" -ForegroundColor Red }

# Banner
Write-Host @"

    ██╗  ██╗███████╗██╗     ██╗██╗  ██╗
    ██║  ██║██╔════╝██║     ██║╚██╗██╔╝
    ███████║█████╗  ██║     ██║ ╚███╔╝
    ██╔══██║██╔══╝  ██║     ██║ ██╔██╗
    ██║  ██║███████╗███████╗██║██╔╝ ██╗
    ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝╚═╝  ╚═╝
    T R A N S V O I C E R

    Windows 11 Setup Script

"@ -ForegroundColor Cyan

# Check Windows version
Write-Header "Checking System Requirements"

$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Build -lt 22000) {
    Write-Warn "Windows 11 recommended (Build 22000+). Current: $($osVersion.Build)"
}
Write-Step "Windows Build: $($osVersion.Build)"

# Check for Python
Write-Header "Checking Python Installation"

if ($PythonPath -eq "") {
    # Try to find Python
    $pythonCandidates = @(
        "python",
        "python3",
        "py -3",
        "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    )

    foreach ($candidate in $pythonCandidates) {
        try {
            $version = & $candidate.Split()[0] $candidate.Split()[1..$candidate.Split().Length] --version 2>&1
            if ($version -match "Python 3\.(1[0-2])") {
                $PythonPath = $candidate
                break
            }
        } catch {
            continue
        }
    }
}

if ($PythonPath -eq "") {
    Write-Err "Python 3.10+ not found!"
    Write-Info "Please install Python 3.10 or later from https://www.python.org/downloads/"
    Write-Info "Make sure to check 'Add Python to PATH' during installation."
    exit 1
}

$pythonVersion = & $PythonPath.Split()[0] $PythonPath.Split()[1..$PythonPath.Split().Length] --version 2>&1
Write-Step "Found: $pythonVersion"
Write-Info "Path: $PythonPath"

# Check for NVIDIA GPU
Write-Header "Checking GPU"

$hasNvidiaGpu = $false
try {
    $nvidiaSmi = & nvidia-smi --query-gpu=name --format=csv,noheader 2>&1
    if ($LASTEXITCODE -eq 0) {
        $hasNvidiaGpu = $true
        Write-Step "NVIDIA GPU detected: $nvidiaSmi"
    }
} catch {
    Write-Info "No NVIDIA GPU detected or nvidia-smi not available"
}

if (-not $hasNvidiaGpu -and $CudaVersion -ne "cpu") {
    Write-Warn "No NVIDIA GPU detected. Switching to CPU-only installation."
    $CudaVersion = "cpu"
}

# Create virtual environment
Write-Header "Creating Virtual Environment"

$venvPath = Join-Path $PSScriptRoot "venv"

if (Test-Path $venvPath) {
    Write-Info "Virtual environment already exists"
    $response = Read-Host "  Recreate? (y/N)"
    if ($response -eq "y" -or $response -eq "Y") {
        Remove-Item -Recurse -Force $venvPath
        & $PythonPath.Split()[0] $PythonPath.Split()[1..$PythonPath.Split().Length] -m venv $venvPath
        Write-Step "Virtual environment recreated"
    }
} else {
    & $PythonPath.Split()[0] $PythonPath.Split()[1..$PythonPath.Split().Length] -m venv $venvPath
    Write-Step "Virtual environment created"
}

# Activate virtual environment
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript
Write-Step "Virtual environment activated"

# Upgrade pip
Write-Header "Upgrading pip"
python -m pip install --upgrade pip wheel setuptools
Write-Step "pip upgraded"

# Install PyTorch
Write-Header "Installing PyTorch"

if ($CudaVersion -eq "cpu") {
    Write-Step "Installing CPU-only version..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
} elseif ($CudaVersion -eq "11.8") {
    Write-Step "Installing CUDA 11.8 version..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Step "Installing CUDA 12.1 version..."
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
}

# Install requirements
Write-Header "Installing Dependencies"
pip install -r requirements.txt
Write-Step "Dependencies installed"

# Install package
Write-Header "Installing Helix Transvoicer"
pip install -e .
Write-Step "Helix Transvoicer installed"

# Create data directories
Write-Header "Creating Data Directories"

$appDataPath = Join-Path $env:LOCALAPPDATA "HelixTransvoicer"
$directories = @(
    (Join-Path $appDataPath "models"),
    (Join-Path $appDataPath "cache"),
    (Join-Path $env:USERPROFILE "Documents\HelixTransvoicer")
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Step "Created: $dir"
    } else {
        Write-Info "Exists: $dir"
    }
}

# Create desktop shortcut
Write-Header "Creating Shortcuts"

$WshShell = New-Object -ComObject WScript.Shell
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path $desktopPath "Helix Transvoicer.lnk"

$shortcut = $WshShell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = Join-Path $venvPath "Scripts\pythonw.exe"
$shortcut.Arguments = "-m helix_transvoicer.main"
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.Description = "Helix Transvoicer - Voice Conversion & TTS"
$shortcut.Save()

Write-Step "Desktop shortcut created"

# Create start menu shortcut
$startMenuPath = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs"
$startShortcutPath = Join-Path $startMenuPath "Helix Transvoicer.lnk"

$shortcut = $WshShell.CreateShortcut($startShortcutPath)
$shortcut.TargetPath = Join-Path $venvPath "Scripts\pythonw.exe"
$shortcut.Arguments = "-m helix_transvoicer.main"
$shortcut.WorkingDirectory = $PSScriptRoot
$shortcut.Description = "Helix Transvoicer - Voice Conversion & TTS"
$shortcut.Save()

Write-Step "Start menu shortcut created"

# Verify installation
Write-Header "Verifying Installation"

try {
    $testOutput = python -c "import torch; print(f'PyTorch {torch.__version__}')" 2>&1
    Write-Step $testOutput

    if ($CudaVersion -ne "cpu") {
        $cudaAvailable = python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" 2>&1
        Write-Step $cudaAvailable

        if ($cudaAvailable -match "True") {
            $cudaDevice = python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')" 2>&1
            Write-Step $cudaDevice
        }
    }

    python -c "from helix_transvoicer import AudioProcessor; print('Helix Transvoicer: OK')" 2>&1
    Write-Step "Helix Transvoicer modules loaded successfully"
} catch {
    Write-Err "Verification failed: $_"
}

# Done
Write-Host @"

╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║   Helix Transvoicer installation complete!                    ║
║                                                                ║
║   To start the application:                                   ║
║   - Double-click the desktop shortcut, or                     ║
║   - Run: .\run.bat                                            ║
║                                                                ║
║   Data locations:                                              ║
║   - Models:  %LOCALAPPDATA%\HelixTransvoicer\models           ║
║   - Exports: %USERPROFILE%\Documents\HelixTransvoicer         ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green
