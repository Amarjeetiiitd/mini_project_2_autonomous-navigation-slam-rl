# Autonomous Navigation using SLAM and Reinforcement Learning

[![CI](https://github.com/Amarjeetiiitd/mini_project_2_autonomous-navigation-slam-rl/actions/workflows/ci.yml/badge.svg)](https://github.com/Amarjeetiiitd/mini_project_2_autonomous-navigation-slam-rl/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade robotics project designed for final-year engineering capstone and research evaluations:
a differential-drive mobile robot navigating complex indoor environments using a **PPO local controller** with lookahead waypoint conditioning and an **active 4-layer predictive safety shield**, compared rigorously against conventional navigation baselines (**DWA + global plan** and **reactive**), integrated with **ROS 2 Jazzy, Gazebo Harmonic, SLAM Toolbox, and Nav2**.

**GitHub Repository:** [https://github.com/Amarjeetiiitd/mini_project_2_autonomous-navigation-slam-rl](https://github.com/Amarjeetiiitd/mini_project_2_autonomous-navigation-slam-rl)

```bash
git clone https://github.com/Amarjeetiiitd/mini_project_2_autonomous-navigation-slam-rl.git
cd mini_project_2_autonomous-navigation-slam-rl
```

---

## 🌟 Key Highlights & Academic Deliverables

- **Dual-Track Research Architecture:** Decouples high-throughput 2D Gymnasium RL training (Track A: ~10,000 steps/sec) from embodied ROS 2 Jazzy / Gazebo Harmonic physical simulation (Track B) via a strictly synchronized observation contract (`rl_training/config.py` & `obs.py`).
- **Native Windows & WSL2 Support:** Runs 100% natively on Windows (Python 3.10-3.13 / PowerShell / CMD) without requiring WSL for Track A training, evaluation, analytics, and live GUI demos.
- **Interactive Real-Time Mission Control UI (`python run_ui.py` / `python run_demo.py`):** Real-time interactive simulation window displaying a **scale-accurate TurtleBot3 Waffle Industrial AMR model** (beveled chassis, tread wheels, RealSense camera, elevated LiDAR turret), dynamic 24-sector LiDAR rays with danger coloring, **optimal Dijkstra wave-front geodesic paths with line-of-sight string pulling**, **adaptive boundary standoff handling**, **unreachable/sealed destination detection with visual hazard alerts**, non-sticky boundary disengagement, live HUD telemetry, **click-and-draw any target point**, multi-waypoint route sequencing, dynamic obstacle placement, and on-the-fly controller hot-swapping (PPO vs. DWA vs. Reactive).
- **Comprehensive Final-Year Project Report:** Formatted academic report ready for submission (`docs/FINAL_PROJECT_REPORT.md`).
- **Viva-Voce & Defense Preparation Guide:** 25+ high-yield examiner questions with mathematically sound answers and a 10-minute presentation script (`docs/VIVA_DEFENSE_GUIDE.md`).
- **Publication-Grade Figures & Academic Tables:** Automated scripts generate high-DPI bar charts, radar charts, topology breakdowns, and LaTeX tables (`results/figures/`, `results/table_comparison.tex`).

---

## 📁 Repository Layout

```
├── run_ui.py                       # One-click launcher for Interactive Mission Control UI (Draw points & waypoints)
├── run_demo.py                     # Interactive demo launcher (UI mode & GIF export)
├── rl_training/                    # Track A: Standalone 2D Gymnasium RL benchmark
│   ├── config.py                   # Single source of truth (29-D observation, robot & sensor params)
│   ├── nav_env.py                  # Fast Gymnasium environment with line-segment raycasting
│   ├── obs.py                      # Observation normalization, 24-sector pooling, frame transforms
│   ├── planning.py                 # 8-connected geodesic potential field & lookahead waypoint logic
│   ├── maps.py                     # Procedural layout generator (open, corridor, clutter, rooms, zigzag)
│   ├── scenarios.py                # Deterministic start/goal scenario sampling & seed splitting
│   ├── baselines.py                # Classical controllers: ReactiveBaseline & DWAController
│   ├── train.py                    # Stable-Baselines3 PPO training with EvalCallback
│   ├── evaluate.py                 # Policy evaluation script on matched scenarios
│   ├── compare.py                  # Full factorial comparison across nominal & perturbed conditions
│   ├── gui_demo.py                 # Real-time interactive TkAgg/Matplotlib simulation dashboard
│   ├── check_env.py                # Environment verification & action/observation space checks
│   └── metrics.py                  # Wilson Score 95% CIs and Bootstrap SPL estimation
├── ros2_integration/               # Track B: ROS 2 Jazzy & Gazebo Harmonic integration
│   ├── policy_node.py              # PPO local controller ROS 2 node (TF2 lookup, safety watchdog)
│   ├── ros_eval_runner.py          # Automated scenario runner for Nav2 vs PPO in Gazebo
│   ├── localization_error_logger.py# Records SLAM pose error against Gazebo ground truth
│   └── launch/rl_nav.launch.py     # Launch file for policy node with parameter overrides
├── analysis/                       # Academic analytics and publication generator
│   ├── generate_visual_comparison.py # Generates high-res PNG charts (radar, grouped bars, breakdown)
│   └── export_academic_tables.py   # Exports LaTeX (.tex) and Markdown tables for reports
├── scripts/                        # Automated turnkey scripts
│   ├── setup_windows.ps1 / .bat    # One-click Windows virtual environment & dependency setup
│   ├── run_pipeline_windows.ps1    # Full benchmark execution pipeline on Windows
│   ├── run_demo.ps1 / .bat         # Launcher for interactive GUI simulation
│   ├── setup_wsl2_ros2.ps1         # WSL2 + Ubuntu 24.04 + ROS 2 Jazzy automation guide
│   ├── setup_ubuntu.sh             # Ubuntu setup script for ROS 2 Jazzy & Gazebo
│   ├── run_tb3_slam_sim.sh         # Launches TurtleBot3 simulation with SLAM Toolbox
│   └── check_ros_stack.sh          # Verifies /scan, /odom, /map, and TF frames
├── docs/                           # Academic documentation & guides
│   ├── FINAL_PROJECT_REPORT.md     # Complete Final Year Mini-Project Research Report
│   ├── VIVA_DEFENSE_GUIDE.md       # Viva-Voce preparation guide, examiner Q&A & presentation script
│   ├── ROS_INTEGRATION.md          # In-depth ROS 2 Jazzy / Gazebo Harmonic bringup guide
│   └── EXPERIMENTS.md              # Detailed experimentation protocol & seed matrix
├── models/                         # Saved checkpoints
│   ├── ppo_navigation.zip / .json  # Trained PPO checkpoint (1,000,000 timesteps)
│   └── best/best_model.zip         # Best checkpoint selected on validation layouts
├── results/                        # Generated benchmark results, tables, and figures
│   ├── figures/                    # High-DPI publication charts & trajectory plots
│   ├── compare_summary.csv / .md   # Factorial summary table (95% Wilson & Bootstrap CIs)
│   ├── compare_by_family.csv       # Per-topology breakdown (open, corridor, clutter, rooms, zigzag)
│   └── table_comparison.tex        # Formatted LaTeX table ready for IEEE papers
└── tests/                          # Automated pytest suite (17 unit tests)
```

---

## 🚀 Quick Start on Windows (Native - No WSL Required for Track A)

### 1. Automated Setup
Open PowerShell or Command Prompt in the project directory:
```powershell
# Run the automated setup script:
.\scripts\setup_windows.ps1

# Or manually:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 2. Verify Environment & Tests
```powershell
python -m rl_training.check_env
python -m pytest -v
```
*(All 19 core geometry, observation adapter, raycasting, safety governor, and metric tests pass).*

### 3. Launch the Live Interactive Mission Control UI
```powershell
python run_ui.py
# (or: python run_demo.py)
```
**Interactive GUI Features:**
- **🎯 Exact Point-and-Click Navigation:** Click anywhere on the floor plan; the robot immediately computes the mathematically optimal Dijkstra shortest path and navigates to the exact millimeter coordinate without artificial jumps.
- **🛡️ Adaptive Near-Boundary Standoff Handling:** When destinations are set adjacent to walls ($d < R_{\text{robot}}$), the robot safely navigates to the physical collision-free standoff position and completes the goal cleanly.
- **🚫 Unreachable Destination Detection:** If a target is placed inside an obstacle or completely walled off, a prominent red warning banner (`🚫 UNREACHABLE (BLOCKED)`) alerts the user and safely halts the robot.
- **📍 Multi-Waypoint Route Sequencing:** Queue multiple waypoints in sequence with green checkmarks on reached destinations.
- **🧱 Dynamic Obstacle Drawing:** Drag custom obstacle boxes on the floor; the occupancy grid and Dijkstra shortest path automatically update in real-time ($< 4\,\text{ms}$) to route around them.
- **🤖 Controller Hot-Swapping:** Switch between **MAS-RL (Full 4-Layer Shield)**, **PPO Governor**, **Raw PPO**, **DWA**, and **Reactive** baselines on the fly.
- **🗺️ Map Topologies:** Switch between multi-room apartments, corridors, warehouse clutter, open arenas, and held-out labyrinths.
- **📊 Live Telemetry HUD:** Real-time velocity, yaw rate, distance, 24-sector LiDAR rays with green/amber/red safety coloring, and SPL metrics.

### 4. Run the Full Benchmark Evaluation Pipeline
```powershell
.\scripts\run_pipeline_windows.ps1 -Episodes 100 -Seed 0
```
This runs the full evaluation of PPO, evaluates DWA and Reactive baselines across nominal, sensor noise, and localization drift conditions, and automatically generates publication charts and LaTeX tables.

---

## 🐧 Running Track B: ROS 2 Jazzy & Gazebo Harmonic (WSL2)

For physical simulation with **TurtleBot3 Waffle**, **SLAM Toolbox**, and **Nav2**:

1. **Install WSL2 with Ubuntu 24.04:**
   ```powershell
   wsl --install -d Ubuntu-24.04
   ```
2. **Copy code to Linux home to avoid filesystem overhead:**
   ```bash
   mkdir -p ~/autonomous-navigation
   cp -r /mnt/d/Mini_project_autonomous-navigation-slam-rl/* ~/autonomous-navigation/
   cd ~/autonomous-navigation
   ```
3. **Run automated Ubuntu setup:**
   ```bash
   chmod +x scripts/*.sh
   ./scripts/setup_ubuntu.sh
   ```
4. **Launch TurtleBot3 Simulation with SLAM in Gazebo:**
   ```bash
   ./scripts/run_tb3_slam_sim.sh
   # (If WSLg GUI rendering lags, pass: HEADLESS=True ./scripts/run_tb3_slam_sim.sh)
   ```
5. **Verify topics and transform tree:**
   ```bash
   ./scripts/check_ros_stack.sh
   ```
6. **Launch PPO Policy Node:**
   ```bash
   source /opt/ros/jazzy/setup.bash && source .venv/bin/activate
   python ros2_integration/policy_node.py --ros-args -p use_sim_time:=true -p plan_source:=nav2
   ```

---

## 📊 Benchmark Results & Scientific Insights

Evaluated across **100 matched scenario trials per experimental condition** (seed 0, 95% Wilson Score CIs for success, 2,000-resample bootstrap CIs for SPL, frozen from `results/compare_summary.csv`):

### 1. Nominal Condition Benchmark ($N=100$ matched episodes per split)
| Dataset Split | Controller | Navigation Success (95% Wilson CI) | Collision Rate | Timeout Rate | SPL (95% Bootstrap CI) | Mean Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Test (Unseen Layouts, Known Families)** | Reactive Baseline | 0.430 [0.34, 0.53] | 0.560 | 0.010 | 0.430 [0.33, 0.53] | 17.0 s |
| **Test (Unseen Layouts, Known Families)** | DWA + Global Plan | **0.930 [0.86, 0.97]** | **0.000** | 0.070 | **0.911 [0.86, 0.96]** | 20.6 s |
| **Test (Unseen Layouts, Known Families)** | Raw PPO (Waypoint) | 0.840 [0.76, 0.90] | 0.160 | **0.000** | 0.824 [0.75, 0.89] | 20.6 s |
| **Test (Unseen Layouts, Known Families)** | PPO Governor (Stopping Governor) | 0.880 [0.80, 0.93] | **0.000** | 0.120 | 0.862 [0.80, 0.92] | 21.0 s |
| **Test (Unseen Layouts, Known Families)** | **MAS-RL (Full 4-Layer Shield)** | **0.900 [0.83, 0.94]** | 0.070 | 0.030 | **0.881 [0.82, 0.94]** | 21.1 s |
| **Held-Out (Unseen ZigZag, 100 Seeds)** | Reactive Baseline | 0.360 [0.27, 0.46] | 0.620 | 0.020 | 0.360 [0.27, 0.46] | 16.3 s |
| **Held-Out (Unseen ZigZag, 100 Seeds)** | DWA + Global Plan | **0.860 [0.78, 0.91]** | **0.000** | 0.140 | **0.850 [0.78, 0.91]** | 20.6 s |
| **Held-Out (Unseen ZigZag, 100 Seeds)** | Raw PPO (Waypoint) | 0.500 [0.40, 0.60] | 0.500 | **0.000** | 0.498 [0.40, 0.60] | 17.0 s |
| **Held-Out (Unseen ZigZag, 100 Seeds)** | PPO Governor (Stopping Governor) | 0.580 [0.48, 0.67] | **0.000** | 0.420 | 0.578 [0.48, 0.68] | 18.9 s |
| **Held-Out (Unseen ZigZag, 100 Seeds)** | **MAS-RL (Full 4-Layer Shield)** | **0.840 [0.76, 0.90]** | 0.160 | **0.000** | **0.830 [0.76, 0.90]** | 21.8 s |

### 2. Multi-Condition Factorial Robustness (Success Rates across 30 Conditions)
| Condition | Dataset Split | Reactive | DWA Baseline | Raw PPO | PPO Governor | MAS-RL (Ours) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Nominal** | Test (Unseen) | 43.0% | **93.0%** | 84.0% | 88.0% | **90.0%** |
| **Nominal** | Held-Out (ZigZag) | 36.0% | **86.0%** | 50.0% | 58.0% | **84.0%** |
| **LiDAR Noise ($\sigma=0.03\,\text{m}$)** | Test (Unseen) | 37.0% | **89.0%** | 79.0% | 83.0% | **83.0%** |
| **LiDAR Noise ($\sigma=0.03\,\text{m}$)** | Held-Out (ZigZag) | 31.0% | 61.0% | 43.0% | 47.0% | **66.0%** |
| **Localization Drift ($\sigma=0.01\,\text{m}$)** | Test (Unseen) | 40.0% | **87.0%** | 79.0% | 83.0% | **87.0%** |
| **Localization Drift ($\sigma=0.01\,\text{m}$)** | Held-Out (ZigZag) | 33.0% | 62.0% | 43.0% | 46.0% | **59.0%** |

### 🔬 Core Scientific Findings:
1. **The Safety vs. Conservatism Trade-Off Formalized:**
   - On the held-out `zigzag` topology, unshielded Raw PPO collapses to **50.0% success** with a **50.0% collision rate** (zero timeouts) due to forward-velocity saturation ($v \to 0.26\,\text{m/s}$).
   - Applying a predictive stopping governor (`PPO Governor`) eliminates all collisions ($0.0\%$), but dramatically inflates conservative timeouts to **42.0%** (freezing robot problem).
   - Our full **MAS-RL architecture** reconciles this trade-off by combining medial-axis doorway centering with an autonomous reverse-and-pivot recovery state machine, restoring held-out success to **84.0%** with **0.0% timeouts** (+34% over Raw PPO).
2. **Comparison with Classical DWA:**
   - DWA achieves 93% nominal and 86% held-out success with 0% collisions due to exhaustive forward trajectory rollout over static costmaps, but fails strictly via timeouts (7% on test, 14% on zigzag) and requires $\approx 8.5\,\text{ms}$ CPU time per cycle.
   - MAS-RL executes in **$1.2\,\text{ms}$ CPU time** ($< 2\%$ of a 10 Hz period) and narrows the performance gap with DWA to within $3\%$ on test ($90\%$ vs $93\%$) and $2\%$ on unseen switchbacks ($84\%$ vs $86\%$).
3. **Decoupled Dual-Track Methodology:**
   - **Track A:** High-throughput 2D procedural Gymnasium simulation benchmark ($N=100$ matched trials per condition across 100 unique seeds, with 95% Wilson Score CIs and bootstrap SPL intervals).
   - **Track B:** Embodied ROS 2 Jazzy, Gazebo Harmonic, and SLAM Toolbox software integration architecture with verified message structures, TF2 lookups, and standalone CPU policy inference latency ($1.2\,\text{ms}$).

---

## 🎓 Academic Documents

- [📄 Full Project Report (`docs/FINAL_PROJECT_REPORT.md`)](docs/FINAL_PROJECT_REPORT.md): Complete IEEE-style project capstone report with mathematical derivations, literature review, and comprehensive discussion.
- [🎯 Viva Defense Guide (`docs/VIVA_DEFENSE_GUIDE.md`)](docs/VIVA_DEFENSE_GUIDE.md): 25+ real examiner questions and answers, presentation script, and examiner cheat sheet.
- [🤖 ROS 2 Bring-Up Guide (`docs/ROS_INTEGRATION.md`)](docs/ROS_INTEGRATION.md): Detailed instructions on frame transforms, topics, and Nav2 global planner coupling.

---

## 📜 Citation & License
This project is licensed under the MIT License. Developed for research and educational purposes in autonomous mobile robotics and reinforcement learning.
