#!/usr/bin/env bash
# Verify the running simulation exposes what the policy node needs (Stage 3 checks).
set -uo pipefail
source /opt/ros/jazzy/setup.bash
ok() { printf '  [ok]   %s\n' "$1"; }
bad() { printf '  [FAIL] %s\n' "$1"; }

echo "ROS_DISTRO=${ROS_DISTRO:-unset}"
for t in /scan /odom /map /tf /cmd_vel; do
  if ros2 topic list 2>/dev/null | grep -qx "$t"; then ok "topic $t"; else bad "topic $t not found"; fi
done
for t in /scan /odom /map; do
  if timeout 8 ros2 topic echo "$t" --once >/dev/null 2>&1; then ok "$t publishing"; else bad "$t no message in 8 s"; fi
done
if timeout 8 ros2 run tf2_ros tf2_echo map base_link 2>&1 | grep -q "Translation"; then ok "TF map -> base_link"; else bad "TF map -> base_link missing"; fi
ros2 action list 2>/dev/null | grep -q navigate_to_pose && ok "action navigate_to_pose" || bad "action navigate_to_pose"
ros2 action list 2>/dev/null | grep -q compute_path_to_pose && ok "action compute_path_to_pose" || bad "action compute_path_to_pose"
echo "Scan details:"; timeout 8 ros2 topic echo /scan --once --field angle_min 2>/dev/null | head -1
ros2 topic info /cmd_vel -v 2>/dev/null | grep -E "Type|Node name" | head -6
