# Auto-detect CUDA version, pick the correct PyTorch wheel index URL,
# create a local venv, and install all dependencies.
#
# Usage (from prediction_models/):
#   .\setup.ps1
#
# Falls back to CPU wheels if no NVIDIA GPU is detected.

$ErrorActionPreference = "Stop"

# --- Detect CUDA version via nvidia-smi -------------------------------------
$indexUrl = "https://download.pytorch.org/whl/cpu"
$cudaLabel = "CPU"

$smi = & nvidia-smi 2>$null
if ($LASTEXITCODE -eq 0) {
    $match = [regex]::Match($smi, "CUDA Version:\s+(\d+)\.(\d+)")
    if ($match.Success) {
        $cudaMajor = [int]$match.Groups[1].Value
        $cudaMinor = [int]$match.Groups[2].Value
        $cuda = [version]"$cudaMajor.$cudaMinor"

        # PyTorch wheel index URLs ship for specific CUDA versions. Pick the
        # newest one not above the host driver, since CUDA drivers are
        # forward-compatible (a 13.x driver can run cu130 wheels).
        if     ($cuda -ge [version]"13.0") { $indexUrl = "https://download.pytorch.org/whl/cu130"; $cudaLabel = "CUDA 13.0" }
        elseif ($cuda -ge [version]"12.6") { $indexUrl = "https://download.pytorch.org/whl/cu126"; $cudaLabel = "CUDA 12.6" }
        elseif ($cuda -ge [version]"12.4") { $indexUrl = "https://download.pytorch.org/whl/cu124"; $cudaLabel = "CUDA 12.4" }
        elseif ($cuda -ge [version]"12.1") { $indexUrl = "https://download.pytorch.org/whl/cu121"; $cudaLabel = "CUDA 12.1" }
        elseif ($cuda -ge [version]"11.8") { $indexUrl = "https://download.pytorch.org/whl/cu118"; $cudaLabel = "CUDA 11.8" }
        else {
            Write-Host "CUDA $cuda is older than 11.8 -> using CPU wheels." -ForegroundColor Yellow
        }
        Write-Host "Detected CUDA $cuda -> $cudaLabel" -ForegroundColor Green
    } else {
        Write-Host "nvidia-smi found but CUDA Version not parsed -> CPU wheels." -ForegroundColor Yellow
    }
} else {
    Write-Host "No NVIDIA GPU detected -> CPU wheels." -ForegroundColor Yellow
}

# --- Pick a Python interpreter (prefer 3.11, then 3.10) ---------------------
# Modern PyTorch/Optuna wheels target 3.10+. Python 3.9 will break on
# PEP-604 union syntax used in this codebase.
$pyExe = $null
foreach ($v in @("3.11", "3.10")) {
    # Probe with Stop disabled so a missing version doesn't abort the script.
    $oldErr = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    $probe = & py "-$v" -c "import sys; sys.stdout.write(sys.executable)" 2>$null
    $code = $LASTEXITCODE
    $ErrorActionPreference = $oldErr
    if ($code -eq 0 -and $probe) {
        $pyExe = "$probe".Trim()
        Write-Host "Using Python $v at $pyExe" -ForegroundColor Cyan
        break
    }
}
if (-not $pyExe) {
    throw "No Python 3.10 or 3.11 found. Install one (https://www.python.org/downloads/) and retry."
}

# --- Create venv ------------------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "`nCreating venv at .\.venv ..." -ForegroundColor Cyan
    & $pyExe -m venv .venv
}

$pip = ".\.venv\Scripts\python.exe"
& $pip -m pip install --upgrade pip wheel

# --- Install dependencies ---------------------------------------------------
Write-Host "`nInstalling from requirements.txt with index $indexUrl ..." -ForegroundColor Cyan
& $pip -m pip install --extra-index-url $indexUrl -r requirements.txt

# --- Install package in editable mode ---------------------------------------
Write-Host "`nInstalling ports_dfl in editable mode ..." -ForegroundColor Cyan
& $pip -m pip install -e .

# --- Done -------------------------------------------------------------------
Write-Host "`n=== Setup complete ===" -ForegroundColor Green
Write-Host "Activate venv:  .\.venv\Scripts\Activate.ps1"
Write-Host "Verify GPU:     python scripts\check_gpu.py"
Write-Host "Run tests:      pytest -q"
