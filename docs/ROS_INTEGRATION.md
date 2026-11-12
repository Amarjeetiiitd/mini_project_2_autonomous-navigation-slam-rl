# ROS 2 Jazzy / Gazebo Harmonic integration

> **Status.** The standalone benchmark (`rl_training/`) was executed and unit-tested when this
> project was generated. The ROS 2 files in `ros2_integration/` were written against the Jazzy
> APIs and syntax-checked, but **not run** (no ROS installation was available). Expect a short
> bring-up phase; use the checks below to localise problems quickly.

## Architecture

```
 Gazebo (TurtleBot3 waffle)                 SLAM Toolbox                      Nav2
   /scan  /odom  /tf ───────────────────►  /map, TF map->odom  ─────►  planner_server (global plan)
        ▲                                         │                           │  ComputePathToPose
        │ /cmd_vel                                ▼                           ▼
        └────────────────────────── policy_node.py (PPO)  ◄──────  global path (lookahead waypoint)
                                         │ obs = [24 LiDAR sectors | wp dist | sin | cos | v | w]
                                         └─► safety layer ─► /cmd_vel,  /rl_status
```

* SLAM Toolbox supplies the map and the `map -> odom` transform; `policy_node.py` reads the
  robot pose from TF `map -> base_link` (so it consumes the SLAM pose, not raw odometry).
* Nav2 is used **only as the global planner** (`plan_source:=nav2`). PPO is the local controller
  and the only node publishing `/cmd_vel` during an RL trial. Do not send Nav2 navigation goals
  at the same time (except in the runner's *staging* phase, when PPO is idle).
* The observation layout and action mapping are imported from `rl_training/obs.py` and
  `rl_training/config.py`, the same code the training environment uses.

## Bring-up checklist

1. `scripts/run_tb3_slam_sim.sh` (set `HEADLESS=True` if Gazebo's GUI struggles under WSLg).
2. `scripts/check_ros_stack.sh` - confirms `/scan`, `/odom`, `/map`, TF, and the Nav2 actions.
3. Check the **scan frame**: the adapter assumes angle 0 = robot front, CCW positive. If your
   LiDAR is rotated, pass `-p scan_yaw_offset:=<rad>`.
4. Check **cmd_vel type**. Jazzy's Nav2 stack uses `geometry_msgs/Twist`. If your bridge expects
   `TwistStamped`, pass `-p stamped_cmd_vel:=true`. Verify with `ros2 topic info /cmd_vel -v`.
5. Dry run first:
   ```bash
   source .venv/bin/activate && source /opt/ros/jazzy/setup.bash
   python ros2_integration/policy_node.py --ros-args -p use_sim_time:=true -p dry_run:=true -p plan_source:=nav2
   ros2 topic pub --once /rl_goal geometry_msgs/PoseStamped "{header: {frame_id: map}, pose: {position: {x: 0.5, y: 1.0}}}"
   ros2 topic echo /rl_status
   ```
6. Real run: same command without `dry_run`. The node stops on stale `/scan`, missing TF, a
   front obstacle closer than `stop_distance` (forward motion inhibited, rotation allowed),
   goal reached, and a wall-clock `goal_timeout`.

## Known gaps and honest caveats

| Issue | Consequence / what to do |
|---|---|
| Train/deploy dynamics gap | Training uses a kinematic model with 10 Hz control, accel limits, `V_MAX=0.26`, `W_MAX=1.0`. Gazebo adds wheel dynamics and latency. Evaluate in Gazebo before trusting the policy; consider domain randomisation (`--lidar-noise`, `--dropout`, `--loc-drift`). |
| LiDAR offset | Env casts rays from the robot centre; the real LiDAR is offset a few cm. Minor, but visible at close range. |
| Invalid readings & sensor dropout | `resample_scan()` detects valid beams ($r \in [r_{\min}, r_{\max}]$) and free-space $+inf$ returns. Sectors with zero valid readings are marked as invalid (NaN in range array, False in validity mask). If valid sector coverage drops below $50\%$, or if forward sectors (8..16) lose visibility during forward motion, the safety filter triggers an immediate fail-safe stop (`sensor_loss_stop` / `forward_sensor_loss_stop`). Any isolated invalid sectors are conservatively clamped to the closest observed obstacle distance rather than assuming wide-open free space. |
| Action space & recovery | The trained PPO policy operates in a continuous forward action space ($v \in [0, V_{\max}]$, $\omega \in [-W_{\max}, W_{\max}]$). The supervisory safety filter provides Layer 4 auto-recovery, which executes rear-clearance verified backup maneuvers ($v = -0.10\,\text{m/s}$) and footprint-damped pivots when stagnation is detected, eliminating dead-end freezing. |
| SPL in Gazebo | Shortest path = Nav2 global-plan length at trial start; path length is integrated from TF at 10 Hz. |
| Collision proxy | `min(/scan) < collision_range` (default 0.17 m) - calibrate; not a contact sensor. |
| Map-frame reproducibility | With online SLAM the map origin depends on the start pose. For matched Nav2-vs-PPO trials, build and save a map once, then run both controllers with localisation on that fixed map (`nav2_bringup` `localization_launch.py` with `map:=...`), or at least spawn at the identical pose every run. |
| Ground truth for E6b | Gazebo does not publish ground truth by default; bridge a pose/odometry topic and pass `gt_topic`. |

## Running matched Nav2 vs PPO trials

```bash
# 1. Edit configs/ros_scenarios_tb3_world.json with verified free-space poses (map frame).
# 2. Nav2 baseline
python ros2_integration/ros_eval_runner.py --controller nav2 \
    --scenarios configs/ros_scenarios_tb3_world.json --repeats 5 --out results/ros_nav2.csv
# 3. PPO (policy node running with plan_source:=nav2)
ros2 launch ros2_integration/launch/rl_nav.launch.py model_path:=models/ppo_navigation plan_source:=nav2
python ros2_integration/ros_eval_runner.py --controller rl \
    --scenarios configs/ros_scenarios_tb3_world.json --repeats 5 --out results/ros_rl.csv
```
