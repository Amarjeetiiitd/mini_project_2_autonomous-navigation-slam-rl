#!/usr/bin/env bash
# One-shot setup for Ubuntu 24.04 (native or WSL2): ROS 2 Jazzy, Gazebo Harmonic, Nav2,
# SLAM Toolbox, TurtleBot3, plus the Python environment for this project.
# Review before running - it uses sudo. Run from the project root.
set -euo pipefail

echo "==> Checking OS"
. /etc/os-release
[[ "$VERSION_ID" == "24.04" ]] || { echo "Expected Ubuntu 24.04, found $VERSION_ID"; exit 1; }

echo "==> ROS 2 apt repository (follow https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html if this fails)"
sudo apt update
sudo apt install -y software-properties-common curl gnupg lsb-release unzip python3-venv python3-pip
sudo add-apt-repository -y universe
if [ ! -f /usr/share/keyrings/ros-archive-keyring.gpg ]; then
  sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list >/dev/null
fi
sudo apt update

echo "==> ROS 2 Jazzy + simulation + navigation packages"
sudo apt install -y ros-jazzy-desktop ros-dev-tools \
  ros-jazzy-ros-gz ros-jazzy-navigation2 ros-jazzy-nav2-bringup ros-jazzy-slam-toolbox \
  ros-jazzy-turtlebot3 ros-jazzy-turtlebot3-msgs ros-jazzy-turtlebot3-bringup \
  ros-jazzy-turtlebot3-gazebo ros-jazzy-tf2-tools ros-jazzy-nav2-map-server

grep -q "source /opt/ros/jazzy/setup.bash" ~/.bashrc || echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc

echo "==> Python venv (with system site-packages so rclpy is importable)"
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

echo "==> Verifying"
python -m rl_training.check_env
python -m pytest -q
echo "Done. Next: 'source .venv/bin/activate', then scripts/run_standalone_pipeline.sh"
