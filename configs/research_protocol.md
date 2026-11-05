# Research protocol

## Research question

How does an RL-based local navigation controller (PPO) compare with a conventional
global-plan + local-planner stack in goal-reaching success, collision avoidance, path
efficiency and generalisation to unseen indoor layouts, when localisation and the map
come from SLAM?

## Two evidence tracks (never mix their numbers)

| Track | Where | What it can support | What it cannot |
|---|---|---|---|
| A. Standalone benchmark | `rl_training/` (2D sim, no ROS) | RL algorithm quality, generalisation to unseen layouts / families, robustness to synthetic sensor noise and synthetic localisation drift, controller comparison on matched scenarios | Anything about real SLAM, Gazebo physics, Nav2 |
| B. ROS 2 / Gazebo | `ros2_integration/` | SLAM-in-the-loop behaviour, Nav2 vs PPO on the same robot and map, real localisation error vs ground truth | Large-scale statistics (slow) |

## Controllers

| Name | Description |
|---|---|
| `reactive` | Map-free goal seeking + obstacle avoidance (weak baseline) |
| `dwa_global` | Global geodesic plan from the believed pose + DWA local planner on LiDAR points (Nav2-*style* baseline in Track A) |
| `ppo_waypoint` | PPO local controller; observation target = lookahead waypoint on the global plan (default, matches the ROS integration) |
| `ppo_direct` | PPO given only the final goal (train with `--goal-mode direct`) - ablation |
| Nav2 (Track B) | Real Nav2 (planner + controller) driving the Gazebo robot |

## Data splits (disjoint *layouts*)

| Split | Layout seeds | Use |
|---|---|---|
| train | 0-999 | PPO training |
| val | 1000-1049 | checkpoint selection (EvalCallback) |
| test | 10000-10099 | headline results: unseen layouts, seen families |
| heldout | 20000-20039 | unseen layouts **and** unseen family (`zigzag`) |

## Experiments

| ID | Track | Test | How |
|---|---|---|---|
| E1 | A | Basic navigation | per-family table in `results/compare_by_family.csv` (open, corridor) |
| E2 | A | Obstacle avoidance / narrow passages | same table (rooms, clutter, zigzag) |
| E3 | A | Generalisation | `test` and `heldout` splits |
| E4 | A | Sensor robustness | condition `lidar_noise` (sigma 3 cm, 5% dropouts) |
| E5 | A+B | Controller comparison | `rl_training.compare`; `ros_eval_runner.py` |
| E6a | A (synthetic) | Localisation-error sensitivity | condition `loc_drift` (random-walk pose error) |
| E6b | B | Real SLAM localisation error | `localization_error_logger.py` vs Gazebo ground truth |

## Metrics

Success rate and collision rate (Wilson 95% CI); timeout rate; SPL (bootstrap 95% CI, 2000
resamples); path length and time of successful episodes; training return and learning
curves (seed-wise); localisation error (mean / RMSE / max / final drift).
SPL denominator is the 8-connected 0.1 m grid geodesic (can be up to ~8% above the
true Euclidean shortest path in open space).

## Reproducibility rules

1. Fixed seeds: training `--seed`, scenario generation `--seed`; report both.
2. Same scenario list for every controller (scenarios are generated once per split/seed).
3. Keep `models/*.json` (metadata), raw per-episode CSVs, and Monitor logs with the results.
4. Train >= 3 seeds before claiming differences between PPO variants; report mean +- CI over seeds.
5. Do not tune on `test` / `heldout`; use `val`.
6. Never report Track A numbers as SLAM results.

## Threats to validity (state these in the report)

- The 2D simulator has kinematic dynamics, perfect LiDAR geometry and static obstacles.
- The `dwa_global` baseline is a re-implementation, not Nav2; Track B uses real Nav2.
- Synthetic drift is a random walk, not the structured error SLAM produces (loop-closure jumps, scan-matching failures).
- Forward-only action space: the policy cannot reverse out of dead ends.
- Procedural layouts are rectilinear; real indoor environments are richer.
