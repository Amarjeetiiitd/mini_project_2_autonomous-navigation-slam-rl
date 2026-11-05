@echo off
echo =================================================================
echo    Autonomous Navigation (SLAM + RL) - GitHub Push Helper
echo =================================================================

git remote get-url origin >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo No remote origin configured.
    set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/user/repo.git): "
    if "%REPO_URL%"=="" (
        echo [ERROR] No URL entered. Aborting.
        pause
        exit /b 1
    )
    git remote add origin %REPO_URL%
)

git branch -M main
git add .
git commit -m "chore: Update project configuration" 2>nul
echo [INFO] Pushing to GitHub...
git push -u origin main
echo =================================================================
pause
