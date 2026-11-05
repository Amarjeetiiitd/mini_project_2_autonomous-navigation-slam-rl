# Autonomous Navigation - WSL2 & ROS 2 Jazzy Setup Automation for Windows
# Run this from Windows PowerShell as Administrator

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   WSL2 + Ubuntu 24.04 + ROS 2 Jazzy Automation Helper" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan

# Step 1: Check WSL2 Status
Write-Host "[1/4] Checking WSL status on Windows..." -ForegroundColor Cyan
try {
    wsl --status
} catch {
    Write-Host "[INFO] WSL might need installation. Run: wsl --install -d Ubuntu-24.04" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host " RECOMMENDED WORKFLOW FOR TRACK B (ROS 2 & GAZEBO HARMONIC):" -ForegroundColor Yellow
Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "1. Install Ubuntu 24.04 in WSL2 (if not already installed):" -ForegroundColor White
Write-Host "   wsl --install -d Ubuntu-24.04" -ForegroundColor Green
Write-Host ""
Write-Host "2. Open Ubuntu terminal and copy project to Linux home to avoid slow I/O:" -ForegroundColor White
Write-Host "   mkdir -p ~/autonomous-navigation" -ForegroundColor Green
Write-Host "   cp -r /mnt/d/Mini_project_autonomous-navigation-slam-rl/* ~/autonomous-navigation/" -ForegroundColor Green
Write-Host "   cd ~/autonomous-navigation" -ForegroundColor Green
Write-Host ""
Write-Host "3. Run the automated Ubuntu setup script inside WSL2:" -ForegroundColor White
Write-Host "   chmod +x scripts/*.sh" -ForegroundColor Green
Write-Host "   ./scripts/setup_ubuntu.sh" -ForegroundColor Green
Write-Host "   (This installs ROS 2 Jazzy, Gazebo Harmonic, SLAM Toolbox, Nav2, TurtleBot3 packages)" -ForegroundColor Gray
Write-Host ""
Write-Host "4. Launch TurtleBot3 SLAM Simulation in Gazebo:" -ForegroundColor White
Write-Host "   ./scripts/run_tb3_slam_sim.sh" -ForegroundColor Green
Write-Host "   (If WSLg GUI rendering lags, use HEADLESS=True ./scripts/run_tb3_slam_sim.sh)" -ForegroundColor Gray
Write-Host ""
Write-Host "5. Verify topics and frame tree in a second terminal:" -ForegroundColor White
Write-Host "   ./scripts/check_ros_stack.sh" -ForegroundColor Green
Write-Host ""
Write-Host "6. Run the PPO Policy Node in Gazebo:" -ForegroundColor White
Write-Host "   source /opt/ros/jazzy/setup.bash && source .venv/bin/activate" -ForegroundColor Green
Write-Host "   python ros2_integration/policy_node.py --ros-args -p use_sim_time:=true -p plan_source:=nav2" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Cyan
