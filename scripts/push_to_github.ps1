# Autonomous Navigation - One-Click GitHub Push Helper
# Usage: .\scripts\push_to_github.ps1 [-RemoteUrl "https://github.com/username/repo.git"]

param (
    [string]$RemoteUrl = ""
)

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Autonomous Navigation (SLAM + RL) - GitHub Push Helper" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check existing remotes
$existingRemote = git remote get-url origin 2>$null

if (-not $existingRemote) {
    if (-not $RemoteUrl) {
        Write-Host "No remote 'origin' is configured yet." -ForegroundColor Yellow
        Write-Host "Please enter your GitHub repository URL:" -ForegroundColor White
        Write-Host "Example: https://github.com/Amarjeetiiitd/autonomous-navigation-slam-rl.git" -ForegroundColor Gray
        $RemoteUrl = Read-Host "GitHub Repository URL"
    }

    if ($RemoteUrl) {
        Write-Host "[INFO] Adding remote origin: $RemoteUrl" -ForegroundColor Cyan
        git remote add origin $RemoteUrl
    } else {
        Write-Host "[ERROR] No repository URL provided. Aborting." -ForegroundColor Red
        Exit 1
    }
} else {
    Write-Host "[OK] Detected remote origin: $existingRemote" -ForegroundColor Green
}

# 2. Ensure on main branch
Write-Host "[INFO] Ensuring branch is 'main'..." -ForegroundColor Cyan
git branch -M main

# 3. Stage any pending updates
$status = git status --porcelain
if ($status) {
    Write-Host "[INFO] Staging and committing pending changes..." -ForegroundColor Cyan
    git add .
    git commit -m "chore: Update project configuration and documentation"
}

# 4. Push to GitHub
Write-Host "[INFO] Pushing code to GitHub (main branch)..." -ForegroundColor Cyan
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host "=================================================================" -ForegroundColor Green
    Write-Host " [SUCCESS] Repository successfully pushed to GitHub!" -ForegroundColor Green
    Write-Host "=================================================================" -ForegroundColor Green
} else {
    Write-Host "[WARN] Push failed. If the remote repository has existing commits, try:" -ForegroundColor Yellow
    Write-Host "   git pull origin main --rebase" -ForegroundColor White
    Write-Host "   git push -u origin main" -ForegroundColor White
}
