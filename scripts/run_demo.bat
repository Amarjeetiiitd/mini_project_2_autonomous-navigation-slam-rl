@echo off
echo =================================================================
echo   Launching Autonomous Navigation Interactive Mission Control UI
echo =================================================================
echo Interactive Features:
echo   - Click ANY point on the map to navigate there in real-time
echo   - Draw multi-waypoint routes (Point 1 -^> Point 2 -^> Goal)
echo   - Draw custom obstacle blocks on the floor
echo   - Hot-swap controllers (PPO / DWA / Reactive)
echo   - Live 24-sector LiDAR rays with hazard color coding
echo =================================================================

python run_ui.py
