# Autonomous Navigation - Interactive Mission Control Launcher
# Usage: .\scripts\run_demo.ps1

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   Autonomous Navigation - Launching Interactive Mission Control" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " Interactive Features for Demonstration:" -ForegroundColor White
Write-Host "   - Click ANY point on the map to set an immediate destination" -ForegroundColor Green
Write-Host "   - Select 'Draw Multi-Waypoint' to click a sequential route" -ForegroundColor Green
Write-Host "   - Select 'Draw Obstacle' to add custom blocks on the floor" -ForegroundColor Green
Write-Host "   - Hot-swap controllers live (PPO / DWA / Reactive)" -ForegroundColor Green
Write-Host "   - Live 24-sector LiDAR rays with green/amber/red safety hazard coloring" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan

python run_ui.py
