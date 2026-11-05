#!/usr/bin/env bash
# Launch TurtleBot3 (waffle) in Gazebo with SLAM Toolbox + Nav2 (Stage 3 of the roadmap).
# Launch files can differ between package versions: if this fails, follow the official Nav2
# Jazzy docs (https://docs.nav2.org/jazzy/) - see docs/ROS_INTEGRATION.md.
set -euo pipefail
source /opt/ros/jazzy/setup.bash
export TURTLEBOT3_MODEL="${TURTLEBOT3_MODEL:-waffle}"
HEADLESS="${HEADLESS:-False}"     # HEADLESS=True disables the Gazebo GUI (useful on WSL2)
exec ros2 launch nav2_bringup tb3_simulation_launch.py slam:=True headless:="$HEADLESS"
