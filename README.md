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
- **Interactive Real-Time Mission Control UI (`python run_ui.py` / `python run_demo.py`):** Real-time interactive simulation window displaying a **scale-accurate TurtleBot3 Waffle Industrial AMR model** (beveled chassis, tread wheels, RealSense camera, elevated LiDAR turret), dynamic 24-sector LiDAR rays with danger coloring, **clearance-weighted medial-axis global paths**, **dynamic passage impasse detection & alternative path rerouting** (`⚠️ IMPASSE`), non-sticky boundary disengagement, live HUD telemetry, **click-and-draw any target point**, multi-waypoint route sequencing, dynamic obstacle placement, and on-the-fly controller hot-swapping (PPO vs. DWA vs. Reactive).
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
python -m pytest -q
```
*(All 17 core geometry, observation adapter, raycasting, and metric tests pass).*

### 3. Launch the Live Interactive Mission Control UI
```powershell
python run_ui.py
# (or: python run_demo.py)
```
**Interactive GUI Features for Professor Demonstration:**
- **🎯 Click Any Point on Map:** Click anywhere on the floor plan; the robot immediately computes a collision-free geodesic route and navigates directly to your clicked coordinate.
- **📍 Draw Multi-Waypoint Route:** Select "Draw Multi-Waypoint Route" to click a sequence of points (Point 1 -> Point 2 -> Goal). The robot executes the route sequentially with green checkmarks on reached waypoints!
- **🧱 Draw Custom Obstacles:** Select "Draw Custom Obstacle Box" and drag on the floor to spawn new wall blocks. The robot's LiDAR detects it in real-time and dynamically re-routes!
- **🤖 Controller Hot-Swapping:** Switch between **PPO Neural Network**, **DWA**, and **Reactive** live.
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

Evaluated across **100 matched scenarios per experimental cell** (nominal condition, seed 0, with 95% confidence intervals):

| Dataset Split | Controller | Navigation Success (95% Wilson CI) | Collision Rate | Timeout Rate | SPL (95% Bootstrap CI) | Mean Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **Test (Unseen Layouts)** | Reactive Baseline | 0.450 [0.36, 0.55] | 0.540 | 0.010 | 0.450 [0.35, 0.55] | 16.5 s |
| **Test (Unseen Layouts)** | DWA + Global Plan | 0.730 [0.64, 0.81] | **0.000** | 0.270 | 0.720 [0.63, 0.80] | 19.2 s |
| **Test (Unseen Layouts)** | PPO (Unshielded) | 0.760 [0.67, 0.83] | 0.240 | **0.000** | 0.754 [0.67, 0.83] | 19.3 s |
| **Test (Unseen Layouts)** | **PPO + Safety Shield** | **0.820 [0.73, 0.88]** | **0.000** | 0.180 | **0.812 [0.72, 0.88]** | 20.4 s |
| **Held-Out (ZigZag)** | Reactive Baseline | 0.380 [0.29, 0.48] | 0.540 | 0.080 | 0.378 [0.29, 0.48] | 15.7 s |
| **Held-Out (ZigZag)** | **DWA + Global Plan** | **0.660 [0.56, 0.75]** | **0.000** | 0.340 | **0.653 [0.56, 0.75]** | 19.2 s |
| **Held-Out (ZigZag)** | PPO (Unshielded) | 0.430 [0.34, 0.53] | 0.570 | **0.000** | 0.428 [0.34, 0.53] | 15.6 s |
| **Held-Out (ZigZag)** | **PPO + Safety Shield** | **0.710 [0.61, 0.79]** | **0.000** | 0.290 | **0.702 [0.61, 0.79]** | 18.8 s |

### 🔬 Core Scientific Findings:
1. **The Safety vs. Conservatism Trade-Off:**
   - DWA achieves zero collisions (**0.0%**), but suffers a **27.0% timeout rate** due to trajectory invalidation in narrow doorways ("freezing robot" problem).
   - PPO achieves zero timeouts (**0.0%**), but experiences a **24.0% collision rate** due to saturated forward velocity ($v \to 0.26\,\text{m/s}$).
2. **Multi-Room Doorway Superiority:** In multi-room layouts (`rooms`), PPO significantly outperforms DWA (**44.0% vs. 32.0%**), learning smooth steering adjustments through narrow doorways where DWA stalls.
3. **Out-of-Distribution Generalization Deficit:** On unseen sharp switchbacks (`zigzag`), PPO’s success rate drops from **76% to 43%**, proving that model-free RL requires braking curriculum or safety shielding before zero-shot transfer to radical new architectural layouts.

---

## 🎓 Academic Documents

- [📄 Full Project Report (`docs/FINAL_PROJECT_REPORT.md`)](docs/FINAL_PROJECT_REPORT.md): Complete IEEE-style project capstone report with mathematical derivations, literature review, and comprehensive discussion.
- [🎯 Viva Defense Guide (`docs/VIVA_DEFENSE_GUIDE.md`)](docs/VIVA_DEFENSE_GUIDE.md): 25+ real examiner questions and answers, presentation script, and examiner cheat sheet.
- [🤖 ROS 2 Bring-Up Guide (`docs/ROS_INTEGRATION.md`)](docs/ROS_INTEGRATION.md): Detailed instructions on frame transforms, topics, and Nav2 global planner coupling.

---

## 📜 Citation & License
This project is licensed under the MIT License. Developed for research and educational purposes in autonomous mobile robotics and reinforcement learning.
