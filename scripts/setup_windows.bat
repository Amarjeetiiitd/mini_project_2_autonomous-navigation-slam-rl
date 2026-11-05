@echo off
echo =================================================================
echo    Autonomous Navigation (SLAM + RL) - Windows Setup Batch
echo =================================================================

where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python not found in PATH! Install Python 3.10-3.13.
    pause
    exit /b 1
)

python --version

if not exist ".venv" (
    echo [INFO] Creating virtual environment .venv...
    python -m venv .venv
)

call .venv\Scripts\activate.bat

echo [INFO] Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo [INFO] Verifying environment...
python -m rl_training.check_env
python -m pytest -q

echo =================================================================
echo Setup complete! To run the interactive visualizer:
echo    python run_demo.py
echo =================================================================
pause
