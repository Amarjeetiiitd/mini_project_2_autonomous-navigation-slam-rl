# Autonomous Navigation using SLAM and RL - Windows Environment Setup
# Usage in PowerShell: .\scripts\setup_windows.ps1

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Autonomous Navigation (SLAM + RL) - Windows Setup Script" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Verify Python Installation
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "[ERROR] Python was not found in PATH! Please install Python 3.10-3.13 from python.org." -ForegroundColor Red
    Exit 1
}
$pyVer = python --version
Write-Host "[OK] Detected: $pyVer" -ForegroundColor Green

# 2. Virtual Environment Setup
if (-not (Test-Path ".venv")) {
    Write-Host "[INFO] Creating virtual environment (.venv)..." -ForegroundColor Cyan
    python -m venv .venv
} else {
    Write-Host "[OK] Existing virtual environment found (.venv)." -ForegroundColor Green
}

# 3. Activate Virtual Environment
Write-Host "[INFO] Activating virtual environment..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# 4. Install / Update Dependencies
Write-Host "[INFO] Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# 5. Run Verification Checks
Write-Host "[INFO] Verifying Gymnasium navigation environment..." -ForegroundColor Cyan
python -m rl_training.check_env
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Environment verification failed!" -ForegroundColor Red
    Exit 1
}

Write-Host "[INFO] Running pytest test suite..." -ForegroundColor Cyan
python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARN] Pytest had failures or warnings. Check logs." -ForegroundColor Yellow
} else {
    Write-Host "[OK] All core tests passed successfully!" -ForegroundColor Green
}

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Setup complete! Quick commands:" -ForegroundColor Yellow
Write-Host "   Launch Interactive Demo : python run_demo.py" -ForegroundColor White
Write-Host "   Run Evaluation Pipeline : .\scripts\run_pipeline_windows.ps1" -ForegroundColor White
Write-Host "   WSL2 & ROS2 Setup Guide : Get-Content docs\ROS_INTEGRATION.md" -ForegroundColor White
Write-Host "=================================================================" -ForegroundColor Cyan
