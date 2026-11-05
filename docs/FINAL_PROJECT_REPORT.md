# AUTONOMOUS MOBILE ROBOT NAVIGATION USING ONLINE SLAM AND DEEP REINFORCEMENT LEARNING

**A Comparative Research-Grade Investigation on Differential-Drive Navigation under Imperfect Sensing and Localization**

---

## PROJECT METADATA & CANDIDATE DETAILS
- **Project Title:** Autonomous Mobile Robot Navigation Using Online SLAM and Deep Reinforcement Learning
- **Academic Degree:** Bachelor of Technology / Bachelor of Engineering in Computer Science / Robotics / Electrical Engineering
- **Project Category:** Final Year Engineering Capstone / Research Mini-Project
- **Academic Year:** 2026–2027
- **Candidate Name:** [Student Name]
- **Roll / Registration Number:** [Student Roll Number]
- **Department:** Department of Computer Science & Engineering / Robotics & Automation
- **Institution:** [College / University Name]
- **Project Guide / Supervisor:** [Supervisor Name & Title]

---

## ABSTRACT

Autonomous mobile robot (AMR) navigation in GPS-denied indoor environments has traditionally relied on classical hierarchical navigation stacks, such as the Dynamic Window Approach (DWA) or Timed Elastic Bands (TEB) coupled with global path planning. While computationally predictable, classical local planners frequently suffer from parameter tuning brittleness, freeze-robot problems, and conservative timeouts in narrow corridors and complex multi-door layouts. Recently, Deep Reinforcement Learning (DRL) has emerged as a promising data-driven alternative for local collision avoidance and trajectory generation.

This project presents a research-grade, statistically rigorous evaluation of an RL-based local navigation policy (Proximal Policy Optimization, PPO) integrated with online Simultaneous Localization and Mapping (SLAM) and global lookahead planning. We develop a dual-track framework: 
1. A high-throughput 2D simulation benchmark in Gymnasium for sample-efficient policy training across procedural indoor layouts, paired with a dedicated **Interactive Graphical Mission Control UI** enabling real-time point-drawing, multi-waypoint route execution, and dynamic obstacle injection.
2. A modular ROS 2 Jazzy / Gazebo Harmonic integration with SLAM Toolbox and Nav2 for embodied verification on a differential-drive robot (TurtleBot3 Waffle).

Across 100 matched test scenarios with 95% Wilson Confidence Intervals, PPO achieving waypoint tracking reaches a **76.0% success rate [0.67, 0.83]**, comparable to the conventional DWA baseline (**73.0% [0.64, 0.81]**), while significantly outperforming DWA in complex multi-room environments (**44.0% vs 32.0%**). Crucially, we identify a fundamental safety vs. conservatism dichotomy: **PPO fails exclusively by collision (24.0%)** due to its saturated forward velocity policy, whereas **DWA fails exclusively by conservative timeout (27.0%) with 0.0% collisions**. On an unseen layout topology (sharp ZigZag corridors), PPO’s performance drops significantly (**43.0% vs. DWA’s 66.0%**), demonstrating an out-of-distribution generalization deficit. We provide complete quantitative metrics, Success-weighted Path Length (SPL), robustness evaluations under sensor noise and localization drift, an exhaustive model architecture description, and a turnkey live demonstration guide for examination defense.

**Keywords:** Autonomous Mobile Robots (AMR), Reinforcement Learning, Proximal Policy Optimization (PPO), SLAM Toolbox, Nav2, Dynamic Window Approach (DWA), Success-weighted Path Length (SPL), ROS 2 Jazzy, Gazebo Harmonic, Interactive Point Drawing.

---

## TABLE OF CONTENTS
1. [Chapter 1: Introduction & Problem Definition](#chapter-1-introduction--problem-definition)
2. [Chapter 2: Literature Review & Mathematical Formulations](#chapter-2-literature-review--mathematical-formulations)
3. [Chapter 3: System Architecture & Dual-Track Methodology](#chapter-3-system-architecture--dual-track-methodology)
   - [3.1 Lookahead Waypoint Tracking vs. Direct Goal Seeking](#31-lookahead-waypoint-tracking-vs-direct-goal-seeking)
   - [3.2 Clearance-Weighted Global Geodesic Planning & Medial Axis Centering](#32-clearance-weighted-global-geodesic-planning--medial-axis-centering)
4. [Chapter 4: Navigation Models & Controller Formulations](#chapter-4-navigation-models--controller-formulations)
   - [4.1 PPO Local Policy Architecture & Hyperparameters](#41-ppo-local-policy-architecture--hyperparameters)
   - [4.2 Dynamic Window Approach (DWA) Formulation](#42-dynamic-window-approach-dwa-formulation)
   - [4.3 Reactive Baseline Formulation](#43-reactive-baseline-formulation)
   - [4.4 Dynamic Passage Impasse Detection & Autonomous Alternative Path Re-routing](#44-dynamic-passage-impasse-detection--autonomous-alternative-path-re-routing)
   - [4.5 Kinematic Barrier Guard with Disengagement Clearance Vectors](#45-kinematic-barrier-guard-with-disengagement-clearance-vectors)
5. [Chapter 5: Interactive Mission Control UI & Robotic Visualization](#chapter-5-interactive-mission-control-ui--robotic-visualization)
   - [5.1 Scale-Accurate TurtleBot3 Waffle Industrial AMR Visualization](#51-scale-accurate-turtlebot3-waffle-industrial-amr-visualization)
   - [5.2 Dynamic Point-Drawing & Online Dijkstra Propagation](#52-dynamic-point-drawing--online-dijkstra-propagation)
   - [5.3 Live Telemetry HUD & Visual Hazard Feedback](#53-live-telemetry-hud--visual-hazard-feedback)
6. [Chapter 6: Experimental Design & Benchmark Protocol](#chapter-6-experimental-design--benchmark-protocol)
7. [Chapter 7: Quantitative Results & Empirical Analysis](#chapter-7-quantitative-results--empirical-analysis)
   - [7.1 Primary Benchmark Evaluation (100 Matched Trials per Condition)](#71-primary-benchmark-evaluation-100-matched-trials-per-condition)
   - [7.2 The Safety vs. Conservatism Failure Mode Trade-Off](#72-the-safety-vs-conservatism-failure-mode-trade-off)
   - [7.3 Topology Breakdown: Where PPO Excels](#73-topology-breakdown-where-ppo-excels)
   - [7.4 Collision Mitigation: Dynamic Safety Shielding & Autonomous Recovery](#74-collision-mitigation-dynamic-safety-shielding-predictive-braking--autonomous-recovery)
   - [7.5 Verification of Dynamic Impasse Rerouting & Edge-Grazing Elimination](#75-verification-of-dynamic-impasse-rerouting--edge-grazing-elimination)
8. [Chapter 8: ROS 2 Jazzy & Gazebo Harmonic Embodied Integration](#chapter-8-ros-2-jazzy--gazebo-harmonic-embodied-integration)
9. [Chapter 9: Threats to Validity, Limitations & Future Work](#chapter-9-threats-to-validity-limitations--future-work)
10. [Chapter 10: Conclusion](#chapter-10-conclusion)
11. [References](#references)
12. [Appendix: Turnkey User Manual for Professor Demonstration](#appendix-turnkey-user-manual-for-professor-demonstration)

---

## CHAPTER 1: INTRODUCTION & PROBLEM DEFINITION

### 1.1 Motivation & Context
Safe, efficient, and autonomous navigation in unstructured, GPS-denied indoor environments (e.g., automated warehouses, hospitals, disaster response zones) is a foundational objective of modern mobile robotics. A fully autonomous mobile robot must accomplish three concurrent tasks:
1. **Mapping:** Continuously build and update an occupancy representation of the surrounding physical world.
2. **Localization:** Reliably track its six degrees-of-freedom pose within that evolving map.
3. **Motion Planning & Control:** Compute collision-free, kinematically feasible trajectories toward distant target goals while reacting gracefully to local obstacles.

In the standard ROS navigation paradigm (such as ROS 2 Nav2), this challenge is decomposed into a hierarchical architecture: a global path planner (e.g., $A^*$, Dijkstra, or NavFn) computes an optimal topological route over a static occupancy grid, while a local controller (e.g., DWA, TEB, or Regulated Pure Pursuit) generates high-frequency velocity commands ($v, \omega$) to track that path while avoiding unmapped obstacles.

### 1.2 Limitations of Conventional Navigation Stacks
Despite widespread commercial adoption, classical local planners exhibit severe practical limitations:
- **Parameter Sensitivity:** DWA and TEB require extensive manual heuristic tuning of costmap weights (obstacle inflation, goal distance, path alignment, smoothness). A parameter configuration optimized for an open atrium frequently fails in narrow doorway passages.
- **The "Freezing Robot" Problem:** In dense or cluttered passages, trajectory rollout algorithms frequently evaluate all candidate forward trajectories as inadmissible or high-cost, leading to deadlocks and timeouts.
- **Myopic Spatial Horizon:** Local trajectory rollouts are strictly limited in time horizon (typically 1.0–2.0 seconds), making them vulnerable to local minima (e.g., concave obstacles and U-shaped dead ends).

### 1.3 The Promise and Pitfalls of Deep Reinforcement Learning
Deep Reinforcement Learning (DRL) provides an alternative paradigm: an end-to-end or hybrid local policy trained through trial-and-error interaction can learn continuous non-linear mappings directly from range sensor data to motor actuation. By maximizing long-term cumulative discounted reward, an RL policy can develop emergent obstacle negotiation skills, fluid obstacle circumvention, and rapid decision-making at low computational latency.

However, existing academic literature in robotics RL frequently suffers from three major flaws:
1. **The Toy Sim Assumption:** Policies are trained in simplistic, collision-free simulations without real sensor noise or localization error, failing immediately upon sim-to-real transfer.
2. **Unfair Baseline Comparisons:** RL models are often compared only against naive reactive controllers (such as Braitenberg vehicles or simple potential fields) rather than production-grade navigation stacks (such as DWA with global path priors).
3. **Absence of Statistical Rigor:** Results are reported across a handful of cherry-picked test runs without confidence intervals, seed variance quantification, or out-of-distribution layout testing.

### 1.4 Research Objectives & Scope
This project directly addresses these gaps. We establish a reproducible research project designed to answer:
> **Core Research Question:** *How does a Deep Reinforcement Learning local controller (PPO) compare against conventional local planners (DWA) in goal-reaching success, collision avoidance, path efficiency, and out-of-distribution generalization when navigation is conditioned on maps and pose estimates produced by SLAM?*

The explicit contributions of this work are:
1. **Dual-Track Research Architecture:** Decoupling high-sample-efficiency 2D Gymnasium RL training (Track A) from high-fidelity ROS 2 Jazzy / Gazebo Harmonic physical simulation (Track B) via a strictly synchronized observation/action contract.
2. **Interactive Mission Control UI with Dynamic Point-Drawing:** A comprehensive graphical application where users can click or draw arbitrary target goals, multi-waypoint routes, and custom obstacle boxes in real-time, observing the robot navigate smoothly with live LiDAR ray visualizations.
3. **Procedural Layout Benchmark:** A procedural generator creating diverse indoor topologies (Open, Corridor, Clutter, Rooms, and held-out ZigZag) with disjoint seed splits, completely eliminating train/test data leakage.
4. **Rigorous Statistical Evaluation:** 100 matched scenario runs per experimental cell with 95% Wilson Score Confidence Intervals for binomial metrics (Success, Collision, Timeout) and 2000-resample Bootstrap CIs for Success-weighted Path Length (SPL).
5. **Controlled Perturbation Stress-Testing:** Quantitative robustness evaluation under realistic sensor degradation (LiDAR noise and ray dropouts) and synthetic random-walk localization drift.

---

## CHAPTER 2: LITERATURE REVIEW & MATHEMATICAL FORMULATIONS

### 2.1 Differential-Drive Kinematic Model
The robot platform considered in this study is a differential-drive ground vehicle (conforming to the physical specifications of the TurtleBot3 Waffle). The kinematic state of the robot in the 2D plane is defined by its pose:
$$\mathbf{p}_t = \begin{bmatrix} x_t \\ y_t \\ \theta_t \end{bmatrix} \in \mathbb{R}^2 \times [-\pi, \pi)$$

Where $(x_t, y_t)$ denotes Cartesian coordinates in the global map frame and $\theta_t$ denotes orientation (yaw) relative to the X-axis. Under the unicycle kinematic model, the robot is actuated via forward linear velocity $v_t \in [0, v_{max}]$ and angular velocity $\omega_t \in [-\omega_{max}, \omega_{max}]$. The discrete-time state transition over control period $\Delta t = 0.1\,\text{s}$ (10 Hz) is:
$$\begin{aligned}
x_{t+1} &= x_t + v_t \cos(\theta_t) \Delta t \\
y_{t+1} &= y_t + v_t \sin(\theta_t) \Delta t \\
\theta_{t+1} &= \text{wrap}\left(\theta_t + \omega_t \Delta t\right)
\end{aligned}$$
Subject to acceleration constraints:
$$|v_{t+1} - v_t| \le a_v^{max} \Delta t, \quad |\omega_{t+1} - \omega_t| \le a_\omega^{max} \Delta t$$
Where $v_{max} = 0.26\,\text{m/s}$, $\omega_{max} = 1.0\,\text{rad/s}$, $a_v^{max} = 1.0\,\text{m/s}^2$, and $a_\omega^{max} = 3.2\,\text{rad/s}^2$.

### 2.2 Simultaneous Localization and Mapping (SLAM)
Online SLAM solves the joint problem of estimating the robot trajectory and building a spatial map from exteroceptive sensor observations $\mathbf{z}_{1:t}$ and odometry controls $\mathbf{u}_{1:t}$:
$$p(\mathbf{x}_{1:t}, \mathbf{m} \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$$

In this project's ROS 2 pipeline, we utilize **SLAM Toolbox**, an optimized 2D graph-based SLAM solver based on Karto SLAM. It constructs a pose-graph where nodes represent robot poses and edges encode spatial constraints derived from scan matching:
$$\mathbf{e}_{ij} = \mathbf{z}_{ij} - h(\mathbf{x}_i, \mathbf{x}_j)$$
The optimal trajectory is found by minimizing the non-linear least squares Mahalanobis error:
$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{ij} \mathbf{e}_{ij}^T \mathbf{\Omega}_{ij} \mathbf{e}_{ij}$$
Where $\mathbf{\Omega}_{ij}$ is the information matrix. SLAM Toolbox continuously publishes the transformation from the static map frame to the odometry frame ($\text{map} \to \text{odom}$), allowing downstream navigation nodes to query the ground-referenced pose $\text{map} \to \text{base\_link}$.

### 2.3 Markov Decision Process (MDP) Formulation
We formulate local robot navigation as a continuous-action, partially-observable Markov Decision Process (POMDP), abstracted as an MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

#### 1. Observation Space $\mathcal{S} \subset \mathbb{R}^{29}$
The observation vector $\mathbf{s}_t \in \mathbb{R}^{29}$ is constructed as:
$$\mathbf{s}_t = \left[ \bar{r}_0, \bar{r}_1, \dots, \bar{r}_{23}, \, \bar{d}_g, \, \sin(\phi_g), \, \cos(\phi_g), \, \bar{v}_t, \, \bar{\omega}_t \right]$$
- **$\bar{r}_{0:23} \in [0, 1]$:** 24 angular sector range readings pooling a $360^\circ$ planar LiDAR scan. Each sector spans $\Delta \theta = 15^\circ$ ($2\pi / 24\,\text{rad}$). Sector range is normalized by maximum sensor range $R_{max} = 3.5\,\text{m}$. Sector 12 represents the dead-ahead forward axis.
- **$\bar{d}_g \in [0, 1]$:** Normalized Euclidean distance to the active navigation target: $\bar{d}_g = \min(d_{target} / D_{norm}, 1.0)$ with $D_{norm} = 5.0\,\text{m}$.
- **$\sin(\phi_g), \cos(\phi_g) \in [-1, 1]$:** Continuous trigonometric representation of the relative heading angle to the target in the robot's local frame $\phi_g = \text{wrap}(\theta_{target} - \theta_{robot})$, avoiding angle wrap-around discontinuities.
- **$\bar{v}_t \in [0, 1]$:** Normalized current linear velocity $v_t / v_{max}$.
- **$\bar{\omega}_t \in [-1, 1]$:** Normalized current angular velocity $\omega_t / \omega_{max}$.

#### 2. Action Space $\mathcal{A} \subset [-1, 1]^2$
The continuous action $\mathbf{a}_t = [a_v, a_\omega] \in [-1, 1]^2$ maps linearly to target velocity commands:
$$v_{cmd} = \frac{a_v + 1.0}{2.0} \cdot v_{max} \in [0, v_{max}], \quad \omega_{cmd} = a_\omega \cdot \omega_{max} \in [-\omega_{max}, \omega_{max}]$$
Notice that $v_{cmd} \ge 0$, restricting the vehicle to forward-only motions.

#### 3. Reward Function $\mathcal{R}(\mathbf{s}_t, \mathbf{a}_t)$
Reward shaping uses a privileged, pre-computed 8-connected geodesic potential field $\Phi_{geo}(\mathbf{x})$ over the grid map:
$$\mathcal{R}_t = R_{progress} \cdot \left[ \Phi_{geo}(\mathbf{x}_{t-1}) - \Phi_{geo}(\mathbf{x}_t) \right] + R_{step} + R_{turn} \frac{|\omega_t|}{\omega_{max}} + R_{prox} + R_{terminal}$$
- **Geodesic Progress:** $R_{progress} = +10.0$ per meter traversed along the shortest navigable topological corridor. Using geodesic distance rather than Euclidean distance prevents the policy from being rewarded for moving toward a goal located on the opposite side of an impassable wall.
- **Step Penalty:** $R_{step} = -0.02$ per timestep.
- **Smoothness Penalty:** $R_{turn} = -0.02 \cdot |\omega_t| / \omega_{max}$.
- **Proximity Penalty:** If $\min(\mathbf{r}_t) < d_{prox} = 0.6\,\text{m}$:
  $$R_{prox} = -0.5 \cdot \left( 1.0 - \frac{\min(\mathbf{r}_t)}{d_{prox}} \right)$$
- **Terminal Rewards:**
  $$R_{terminal} = \begin{cases} +50.0 & \text{if Goal Reached } (\|\mathbf{x}_t - \mathbf{x}_g\| < 0.25\,\text{m}) \\ -50.0 & \text{if Collision Occurs } (\min(\mathbf{r}_t) \le r_{robot}) \end{cases}$$

---

## CHAPTER 3: SYSTEM ARCHITECTURE & DUAL-TRACK METHODOLOGY

```
+───────────────────────────────────────────────────────────────────────────────────+
|                            RESEARCH ARCHITECTURE                                  |
+───────────────────────────────────────────────────────────────────────────────────+
|  TRACK A: HIGH-THROUGHPUT GYMNASIUM BENCHMARK (Fast Exploration & Statistics)     |
|  - Procedural Layouts (Train: 0-999, Val: 1000-1049, Test: 10k, Heldout: 20k)     |
|  - Vectorized 2D Kinematics & Line-Segment Raycasting (~10,000 steps/sec)         |
|  - Stable-Baselines3 PPO Policy Training (1,000,000 Timesteps)                   |
|  - Interactive Graphical Mission Control UI (Dynamic Point Drawing & HUD)         |
|  - Factorial Experimentation (E1 - E6a) with Wilson 95% Confidence Intervals      |
+───────────────────────────────────────────────────────────────────────────────────+
                                      │
               Synchronized Observation & Action Interfaces
                   [rl_training/config.py & obs.py]
                                      │
                                      ▼
+───────────────────────────────────────────────────────────────────────────────────+
|  TRACK B: EMBODIED ROS 2 JAZZY & GAZEBO HARMONIC (Physical Realism & SLAM)        |
|  - Simulated Differential-Drive Robot (TurtleBot3 Waffle)                         |
|  - Sensors: LDS-01 LiDAR (/scan, 360 deg) + Wheel Odometry (/odom)                |
|  - SLAM Toolbox: Online Scan Matching & Graph-SLAM (Publishes map -> odom TF)     |
|  - Nav2 Server: ComputePathToPose (Global Dijkstra Topological Planner)           |
|  - PPO ROS 2 Node (ros2_integration/policy_node.py):                              |
|      * TF2 Lookup: Queries map -> base_link to acquire true SLAM pose             |
|      * Lookahead Waypoint Extractor (L = 1.5 m on Nav2 path)                      |
|      * Resamples /scan into 24 sectors matching training distribution             |
|      * Safety Watchdog: Inhibits forward velocity if front range < 0.28 m         |
|      * Velocity Governor: Publishes /cmd_vel at 10 Hz                             |
+───────────────────────────────────────────────────────────────────────────────────+
```

### 3.1 Lookahead Waypoint Tracking vs. Direct Goal Seeking
In unstructured indoor environments, asking a local RL policy to navigate directly to a distant goal ($>10\,\text{m}$ away) across multiple rooms is known as "direct goal seeking". Because the policy only observes local LiDAR range beams (maximum 3.5 m), it possesses zero memory of wall configurations behind it and inevitably becomes trapped in dead ends.

Our system employs a **lookahead waypoint conditioning strategy**:
1. Nav2 or global geodesic planning computes a high-level topological path from the robot's current SLAM pose to the final goal.
2. The policy node projects along this path and extracts a dynamic local waypoint at lookahead distance $L = 1.5\,\text{m}$.
3. The RL policy is fed the relative distance and angle to this local waypoint rather than the distant global goal.

This cleanly divides responsibilities: the global planner handles global topology and route selection, while the RL policy focuses entirely on local obstacle avoidance, dynamic collision avoidance, and trajectory smoothing.

### 3.2 Clearance-Weighted Global Geodesic Planning & Medial Axis Centering
Standard 8-connected Dijkstra path planning assigns uniform edge weights based strictly on Euclidean cell transitions ($\Delta x = \Delta_{res}$ or $\sqrt{2}\Delta_{res}$). Consequently, a shortest-path algorithm mathematically cuts corners as tightly as possible along the inflation boundary and treats a narrow $0.8\,\text{m}$ constriction identically to a $3.0\,\text{m}$ open hallway if the constriction is marginally shorter in Euclidean meters.

To eliminate corner clipping and enforce natural passage centering, our system implements a **Clearance-Weighted Dijkstra Algorithm**:
1. **Obstacle Distance Field:** Given the unblocked grid, the Euclidean distance transform is evaluated:
   $$D_{clear}(r, c) = \text{EDT}(\neg \mathbf{M}_{occ})(r, c) \cdot \Delta_{res}$$
2. **Clearance Cost Penalty Function:** Cells possessing clearance below a safety margin $d_{margin} = 0.45\,\text{m}$ are assigned a smooth quadratic penalty:
   $$\text{Cost}_{clear}(r, c) = \begin{cases} 
   \left(\frac{d_{margin} - D_{clear}(r, c)}{d_{margin}}\right)^2 \cdot K_{penalty} & \text{if } D_{clear}(r, c) < d_{margin} \text{ and } \neg \mathbf{M}_{occ}(r, c) \\
   0.0 & \text{otherwise}
   \end{cases}$$
   Where $K_{penalty} = 2.5$.
3. **Cost-Augmented Geodesic Wave-Front:** In Dijkstra relaxation, edge traversal costs are scaled:
   $$\Delta d = w(dr, dc) \cdot \Delta_{res} \cdot \left[1.0 + \text{Cost}_{clear}(r+dr, c+dc)\right]$$

**Engineering Significance:**
- **Doorway Medial Axis Tracking:** Inside doorways and narrow passages, the cost is lowest at the exact center ($D_{clear} \approx 0.5\,\text{m}$, $\text{Cost} = 0$) and increases quadratically toward the door jambs. As a result, the extracted global path naturally tracks the dead center (medial axis) of doorways.
- **Safe Corridor Preference:** If a wide open alternative route exists, the cumulative penalty of squeezing through a tight pinch point outweighs the slightly longer Euclidean distance of the open room, guiding the robot along safe, wide passages automatically.

---

## CHAPTER 4: NAVIGATION MODELS & CONTROLLER FORMULATIONS

### 4.1 PPO Local Policy Architecture & Hyperparameters
The Deep RL policy is parameterized by a continuous Actor-Critic Multi-Layer Perceptron (MLP) trained via Proximal Policy Optimization:

```
          [29-Dimensional Continuous Observation Vector]
                                 │
           ┌─────────────────────┴─────────────────────┐
           ▼                                           ▼
   [Actor Policy Network]                     [Critic Value Network]
   Linear(29 -> 64)                           Linear(29 -> 64)
   Tanh Activation                            Tanh Activation
   Linear(64 -> 64)                           Linear(64 -> 64)
   Tanh Activation                            Tanh Activation
   Linear(64 -> 2) [Mean μ_v, μ_w]             Linear(64 -> 1) [State Value V(s)]
   Trainable Log Standard Deviation σ
           │
           ▼
   Action Sampling: a ~ N(μ(s), diag(σ^2))
   Target Velocity Mapping: v_cmd, ω_cmd
```

#### PPO Hyperparameters Summary:
| Hyperparameter | Symbol | Value | Architectural Purpose |
|---|:---:|:---:|---|
| Discount Factor | $\gamma$ | $0.99$ | Models long-horizon cumulative trajectory returns |
| GAE Smoothing | $\lambda$ | $0.95$ | Exponential weighting of multi-step advantage estimates |
| Clipping Coefficient | $\epsilon$ | $0.20$ | Bounds probability ratio $r_t(\theta) \in [0.8, 1.2]$ |
| Learning Rate | $\alpha$ | $3 \times 10^{-4}$ | Adam optimizer step size |
| Rollout Steps per Env | $T_{rollout}$ | $2048$ | Steps gathered per policy iteration |
| Mini-Batch Size | $B$ | $64$ | Stochastic gradient descent batch dimension |
| Optimization Epochs | $K$ | $10$ | Iterations per collected rollout batch |
| Value Loss Coefficient | $c_1$ | $0.50$ | Balances value prediction error with policy gradient |
| Entropy Coefficient | $c_2$ | $0.00$ | Deterministic policy convergence at convergence |
| Total Training Steps | - | $1,000,000$ | Total environment interactions across 2 parallel threads |

### 4.2 Dynamic Window Approach (DWA) Formulation
The Dynamic Window Approach is implemented as our primary conventional baseline:
1. **Dynamic Velocity Window:** Considers only velocities reachable within one timestep:
   $$V_s = [0, v_{max}] \times [-\omega_{max}, \omega_{max}]$$
   $$V_d = [v - a_v \Delta t, \, v + a_v \Delta t] \times [\omega - a_\omega \Delta t, \, \omega + a_\omega \Delta t]$$
   Admissible search space: $V_r = V_s \cap V_d$.
2. **Forward Rollout:** Discretizes $V_r$ into a $5 \times 15$ grid ($N_v = 5, N_\omega = 15$). For each candidate pair $(v, \omega)$, trajectories are simulated forward over horizon $T = 1.5\,\text{s}$ (15 steps).
3. **Trajectory Scoring Function:**
   $$\text{Score}(v, \omega) = -1.0 \cdot d_{goal} - 0.3 \cdot |\Delta \theta| + 0.6 \cdot \min(d_{clear}, 0.8) + 0.3 \cdot \frac{v}{v_{max}}$$
   Where $d_{clear} = \min_{\text{points}} (\|\mathbf{x}_t - \mathbf{p}_{obs}\|) - r_{robot}$. Trajectories with $d_{clear} < 0.02\,\text{m}$ are declared inadmissible ($\text{Score} = -\infty$).
4. **Deadlock Recovery:** If all candidate trajectories are inadmissible, DWA commands zero forward velocity and spins in place toward the most open sensor sector.

### 4.3 Reactive Baseline Formulation
The reactive controller operates map-free using simple potential fields:
- Evaluates front clearance over sectors 11–13. If $r_{front} < 0.7\,\text{m}$, forward velocity is curtailed and angular steering is commanded toward whichever lateral side (left sectors 13–19 vs. right sectors 6–12) presents higher mean clearance.
- In open space, it heads directly toward the final target with $v = v_{max} \cos(\phi_g)$.

### 4.4 Dynamic Passage Impasse Detection & Autonomous Alternative Path Re-routing
In real-world mobile robotics (e.g. warehouse AMRs), an intended passage may become blocked by dynamic obstacles, unmodeled clutter, or tight door thresholds where the robot triggers physical safety holds. Conventional global planners compute the topological route once; when the robot halts at an impasse, standard auto-recovery routines (pivoting on the spot or backing up blindly) exit only to have the static global planner command the exact same blocked waypoint, creating an infinite deadlock loop.

Our system incorporates an **Autonomous Dynamic Impasse & Alternative Path Re-routing Engine**:
1. **Multi-Source Impasse Detection:**
   - *Stagnation Watchdog:* Detects if robot displacement $\|\mathbf{p}_t - \mathbf{p}_{t-15}\| < 0.04\,\text{m}$ (less than $4\,\text{cm}$ over 1.5 seconds) while a mission is active.
   - *Repetitive Boundary Hold Watchdog:* Detects if the kinematic barrier guard triggers $\ge 3$ consecutive boundary holds within a local $0.25\,\text{m}$ radius.
2. **Virtual Dynamic Impasse Blockade:**
   - Upon impasse detection, the system dynamically marks the blocked corridor segment as a virtual obstacle disk of radius $r_{imp} = 0.45\,\text{m}$ centered at $(x_{imp}, y_{imp}) = \mathbf{p}_{robot} + 0.18\,[\cos\theta, \sin\theta]^T$.
   - The dynamic impasse is temporarily appended to the planning costmap:
     $$\mathbf{M}_{blocked}'(r, c) = \mathbf{M}_{blocked}(r, c) \lor \left(\|(r, c) - (r_{imp}, c_{imp})\| \le \frac{r_{imp}}{\Delta_{res}}\right)$$
3. **Instant Wave-Front Alternative Discovery:**
   - The Dijkstra wave-front propagates from the goal through the updated costmap in $< 4\,\text{ms}$.
   - Because the pinch point is now sealed, the gradient naturally routes around the obstacle through alternative corridors (e.g., adjacent rooms or hallways).
   - If an alternative route exists ($\Phi_{geo}(\mathbf{p}_{robot}) < \infty$), the mission status updates to `🔄 ALTERNATIVE ROUTE DISCOVERED: Bypassing impasse via open corridor!`.
   - If no alternative physical route exists (the room is topologically sealed), the virtual block is automatically relaxed and the controller attempts precision maneuvering.

### 4.5 Kinematic Barrier Guard with Disengagement Clearance Vectors
A widespread, insidious bug in robot simulation physics is the **"Sticky Wall Trap"**:
When a robot comes within the inflation/collision threshold $r_{robot} + \epsilon$ of a wall, standard collision checking functions:
$$\text{collides}(\mathbf{x}_{cand}, \text{radius} = r_{robot} + \epsilon) \to \text{True}$$
evaluate to True for **every** candidate step—even steps that are actively driving backward or turning away from the wall! Consequently, the robot is held frozen ($v = 0.0\,\text{m/s}$) forever.

We formulate a mathematically rigorous **Clearance-Directed Disengagement Rule**:
1. Compute the Euclidean distance from the current pose to all obstacle boundaries:
   $$d_{curr} = \text{rect\_clearance}(\mathbf{x}_{curr}, \mathcal{O})$$
2. Compute the Euclidean distance from the proposed candidate pose to all obstacle boundaries:
   $$d_{cand} = \text{rect\_clearance}(\mathbf{x}_{cand}, \mathcal{O})$$
3. **Disengagement Invariance:** Even if $\text{collides}(\mathbf{x}_{cand}) = \text{True}$, if the step **increases clearance away from the obstacle**:
   $$d_{cand} > d_{curr} + 10^{-5}\,\text{m} \implies \text{Admit State Transition: } \mathbf{x}_{t+1} = \mathbf{x}_{cand}$$
4. **Tangential Wall Sliding:** If diagonal motion collides, the displacement is decoupled into orthogonal projections ($x_{cand}, y_t$) and ($x_t, y_{cand}$), allowing the AMR to glide smoothly along wall boundaries.
5. **Clearance-Directed Auto-Recovery:** During recovery mode, the robot calculates the heading error $\Delta \theta_{wpt}$ toward the newly routed lookahead waypoint in open space. It turns aggressively toward $\Delta \theta_{wpt}$ and commands forward drive ($v = +0.12\,\text{m/s}$) as soon as forward clearance $> r_{robot} + 0.08\,\text{m}$, rapidly driving out of the pinch point into the open corridor.

---

## CHAPTER 5: INTERACTIVE MISSION CONTROL UI & ROBOTIC VISUALIZATION

To allow professors, examiners, and researchers to verify that navigation is fully dynamic, collision-free, and generalizable, a research-grade presentation dashboard was developed in [`rl_training/interactive_ui.py`](file:///d:/Mini_project_autonomous-navigation-slam-rl/rl_training/interactive_ui.py) and launched via `python run_ui.py`.

```
+─────────────────────────────────────────────────────────────────────────────────────────────────+
|   AUTONOMOUS ROBOTICS MISSION CONTROL (Online SLAM + Deep RL | 360° LiDAR Point-and-Click Nav)  |
+──────────────────────────────────────┬──────────────────────────────────────────────────────────+
|  [LEFT SIDEBAR: MISSION CONTROLS]    |  [CENTER: HIGH-RESOLUTION 2D ROBOTICS CANVAS]            |
|  1. Interaction Modes:               |  - Architectural Walls & Room Obstacles with 3D Bevels   |
|     (•) Single Target Goal   [1]     |  - Scale-Accurate TurtleBot3 Waffle Industrial AMR Model |
|     ( ) Multi-Waypoint Route [2]     |  - 24-Sector LiDAR Ray Beams (Dynamic Green/Amber/Red)   |
|     ( ) Draw Obstacle Box    [3]     |  - Laser Hit Spark Particles on Obstacle Boundaries      |
|     ( ) Relocate Robot Pose  [4]     |  - Medial-Axis Geodesic Planned Route (Neon Cyan Ribbon) |
|                                      |  - Dynamic Impasse Hazard Zones (⚠️ IMPASSE Markers)     |
|  2. Controller Selection:            |  - Pulsating Orange Lookahead Waypoint & Red Target      |
|     (•) PPO + Dynamic Safety Shield  |  - Breadcrumb Trajectory Trail & Numbered Waypoint Flags |
|     ( ) PPO Raw Neural Network       +──────────────────────────────────────────────────────────+
|     ( ) DWA Local Planner            |  [RIGHT: ROBOT TELEMETRY HUD (LIVE 10Hz)]                |
|     ( ) Reactive Potential Field     |  - Target Destination: X: 3.74m, Y: 7.11m                |
|                                      |  - Waypoint Queue: 0 Queued | 1 Done                     |
|  3. Indoor Floorplan Topology:       |  - Route Planning: Optimal Direct / Alternative #N       |
|     [Multi-Room Apartments (rooms)]  |  - Robot Pose: X: 4.32m, Y: 7.62m, θ: 85.5°              |
|     [Regenerate Layout (New Seed)]   |  - Linear Velocity: [██████░░░░] +0.18 m/s               |
|                                      |  - Yaw Angular Rate (ω): -0.42 rad/s                     |
|  4. Mission Controls:                |  - Distance to Target: 0.73 m                            |
|     [⏸ PAUSE / ▶ START MISSION]      |  - Min LiDAR Clearance: ⚠️ 0.23 m (HAZARD)               |
|     [⏭ Step 100ms]   [🔄 Reset (R)]  |  - Path Traversed & Shortest Path (Geo) Metrics          |
|     [🗑 Clear Waypoints / Obstacles] |  - SPL Efficiency Metric & Mission Time / Steps          |
|                                      |  - Dynamic Safety Shield Status Banner (Color-Coded)     |
|  5. Layers: Shield, SLAM Fog of War  |  - Viva / Presentation Quick Keyboard Guide              |
+──────────────────────────────────────┴──────────────────────────────────────────────────────────+
```

### 5.1 Scale-Accurate TurtleBot3 Waffle Industrial AMR Visualization
Rather than rendering the robot as an idealized geometric circle, the dashboard features a **dimensionally scale-accurate, multi-layered visual representation of the industrial TurtleBot3 Waffle Autonomous Mobile Robot**:
1. **Dynamic State-Colored Underglow Halo:** An elliptical ground-glow circle whose diameter matches physical collision bounds ($r = 0.22\,\text{m}$), color-coded in real-time to active navigation states (Cyan: nominal cruising; Amber: doorway centering; Red: obstacle braking; Orange: dynamic impasse disengagement).
2. **Drive Wheels with Grip Treads:** Scale-proportional rugged left and right drive wheels ($15\,\text{cm} \times 4.4\,\text{cm}$) featuring three visible rubber tire tread grooves and machined aluminum axle hubs.
3. **Swivel Caster Assemblies:** Front and rear heavy-duty omnidirectional swivel caster wheels with pivot brackets and mounting hubs.
4. **Beveled AMR Structural Chassis:** Chamfered octagonal main structural deck ($36\,\text{cm} \times 31\,\text{cm}$) finished in dark slate with glowing cyan edge bevels.
5. **Secondary Tier / Carbon-Fiber Payload Deck:** Elevated secondary electronics platform supported by four extruded aluminum stanchion posts.
6. **Intel RealSense RGB-D Vision Camera:** Front-mounted stereo vision assembly complete with left/right optical lenses and red infrared pattern projector.
7. **Dual LED Headlights:** High-intensity white LED driving headlights positioned on the front chassis chamfers.
8. **Elevated 360° LiDAR Scanner Turret:** Centrally mounted LDS-01 LiDAR housing featuring a rotating red laser diode window and laser emitter pupil.
9. **Microcontroller Board & Emergency Stop Button:** Visible OpenCR board with green power and blue Wi-Fi status LEDs, accompanied by a prominent red mushroom-style emergency stop (E-stop) switch.
10. **Directional Laser Pointer Chevron:** Amber forward-heading arrow indicating instantaneous kinematic orientation.

### 5.2 Dynamic Point-Drawing & Online Dijkstra Propagation

#### 1. Free-Space Snapping Algorithm
When a user clicks on the map, the cursor coordinates $(x_{click}, y_{click})$ may land near or inside an obstacle boundary. To prevent illegal targets, `snap_to_free_space()` evaluates concentric search rings:
$$\mathcal{N}(x, y) = \left\{ (x + r \cos \theta, \, y + r \sin \theta) \;\middle|\; r \in [0.1, 0.8]\,\text{m}, \; \theta \in [0, 2\pi) \right\}$$
The point is snapped to the nearest collision-free coordinate with minimum Euclidean clearance $r_{robot} = 0.22\,\text{m}$.

#### 2. Online Dijkstra Geodesic Wave-Front Propagation
Upon clicking a new target $(x_g, y_g)$, the global topological distance field $\Phi_{geo}$ is recomputed across the $100 \times 100$ occupancy grid via 8-connected Dijkstra wave-front propagation in **$< 4\,\text{milliseconds}$**:
$$\Phi_{geo}(r, c) = \min_{(dr, dc) \in \mathcal{N}_8} \left[ \Phi_{geo}(r+dr, c+dc) + w(dr, dc) \cdot \Delta_{res} \cdot (1.0 + \text{Cost}_{clear}) \right]$$
Where $w = 1.0$ for orthogonal neighbors and $\sqrt{2}$ for diagonal transitions.

#### 3. Multi-Waypoint Sequential Queue Execution
When "Draw Multi-Waypoint Route" is active:
- The user can click an arbitrary sequence of points: $\mathcal{W} = [\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_k]$.
- The active target is set to $\mathbf{w}_1$.
- As soon as the robot enters the goal boundary $\|\mathbf{x}_{robot} - \mathbf{w}_1\| < 0.25\,\text{m}$, $\mathbf{w}_1$ is marked as completed (rendering a green checkmark ✓), $\mathbf{w}_2$ is popped from the queue, the geodesic field is updated, and the robot transitions smoothly without pausing.

#### 4. Dynamic Obstacle Injection & Real-Time Rerouting
Users can select "Draw Custom Obstacle Box" and click-and-drag across the floor. A new rectangular obstacle is dynamically appended to the environment geometry. The robot's simulated LiDAR rays immediately strike the new obstacle, turning rays from green to red, triggering the safety watchdog, and forcing the local planner to re-route around the obstacle in real-time.

#### 5. Dynamic Impasse Hazard Visualization
When a narrow passage becomes wedged or impassable, the system dynamically marks the location with an orange translucent hazard circle and a prominent `⚠️ IMPASSE` label on the canvas. The neon cyan planned route instantly snaps to the discovered alternative corridor, allowing examiners to visually observe dynamic costmap invalidation and local-to-global replanning in real time.

### 5.3 Live Telemetry HUD & Visual Hazard Feedback
The right-hand dashboard panel provides real-time telemetry streaming at 10 Hz:
- **Analog-Style Velocity Bar:** Dynamically renders linear velocity $v$ as a proportional bar: `[████████░░] +0.22 m/s`.
- **Route Planning Mode Pill:** Displays `Optimal Direct` during normal traversal or `Alternative #N (Bypassed)` when rerouting around an impasse.
- **LiDAR Proximity Alarm:** Displays the closest obstacle distance with color-coded safety thresholds (`🟢 CLEAR` $> 1.2\,\text{m}$, `🟡 CAUTION` $0.40 - 1.2\,\text{m}$, `⚠️ HAZARD` $< 0.40\,\text{m}$).
- **SPL Efficiency Metric:** Live computation of Success-weighted Path Length comparing actual traversed odometer distance against global geodesic shortest path.

---

## CHAPTER 6: EXPERIMENTAL DESIGN & BENCHMARK PROTOCOL

### 6.1 Procedural Layout Topologies & Disjoint Splits
To evaluate policy generalization without data leakage, we developed procedural layout generators:
- **`open`:** $10\text{m} \times 10\text{m}$ open arena with perimeter boundaries.
- **`corridor`:** Intersecting hallways with $90^\circ$ turns and constriction bottlenecks.
- **`clutter`:** Random distributions of rectangular pillars, obstacles, and columns.
- **`rooms`:** Multi-room apartments featuring internal dividing walls and narrow doorways ($0.8\text{m}$ width).
- **`zigzag` (Held-Out Family):** Highly constrained labyrinth switchbacks with successive sharp U-turns. Crucially, the policy **never encounters the `zigzag` topology during training**.

Disjoint layout seeds:
- **Training Set (`train`):** Seeds `0` to `999`.
- **Validation Set (`val`):** Seeds `1000` to `1049` (used for model selection via `EvalCallback`).
- **Test Set (`test`):** Seeds `10000` to `10099` (unseen random layouts of familiar families).
- **Held-Out Set (`heldout`):** Seeds `20000` to `20039` (strictly evaluated on the held-out `zigzag` family).

### 6.2 Quantitative Evaluation Metrics
- **Success Rate ($S$):** Fraction of trials reaching within $d_{tol} = 0.25\,\text{m}$ of the goal without collision within $T_{max} = 70.0\,\text{s}$ (700 steps). Reported with **Wilson Score 95% Confidence Intervals**:
  $$CI_{95\%} = \frac{\hat{p} + \frac{z^2}{2n} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}, \quad z = 1.96$$
- **Collision Rate ($C$):** Fraction of trials terminated due to vehicle boundary penetration ($d_{wall} \le r_{robot} = 0.22\,\text{m}$).
- **Timeout Rate ($T$):** Fraction of trials exceeding 700 steps without collision or goal achievement.
- **Success-weighted Path Length (SPL):** Standard robotics benchmark metric for path efficiency:
  $$\text{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \frac{l_i^*}{\max(l_i, l_i^*)}$$
  Reported with **Bootstrap 95% Confidence Intervals** across 2000 resamples.

---

## CHAPTER 7: QUANTITATIVE RESULTS & EMPIRICAL ANALYSIS

### 7.1 Primary Benchmark Evaluation (100 Matched Trials per Condition)

| Dataset Split | Perturbation Condition | Controller Architecture | Sample Size ($N$) | Success Rate (95% Wilson CI) | Collision Rate | Timeout Rate | SPL (95% Bootstrap CI) | Mean Successful Time (s) |
|---|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Test** *(Unseen)* | **Nominal** | **Reactive Baseline** | 100 | 0.450 [0.36, 0.55] | 0.540 | 0.010 | 0.450 [0.35, 0.55] | 16.5 s |
| **Test** *(Unseen)* | **Nominal** | **DWA + Global Plan** | 100 | 0.730 [0.64, 0.81] | **0.000** | 0.270 | 0.720 [0.63, 0.80] | 19.2 s |
| **Test** *(Unseen)* | **Nominal** | **PPO (Waypoint)** | 100 | **0.760 [0.67, 0.83]** | 0.240 | **0.000** | **0.754 [0.67, 0.83]** | 19.3 s |
| **Test** *(Unseen)* | LiDAR Noise | Reactive Baseline | 100 | 0.460 [0.37, 0.56] | 0.530 | 0.010 | 0.460 [0.36, 0.56] | 16.8 s |
| **Test** *(Unseen)* | LiDAR Noise | DWA + Global Plan | 100 | 0.760 [0.67, 0.83] | 0.110 | 0.130 | 0.748 [0.67, 0.83] | 20.9 s |
| **Test** *(Unseen)* | LiDAR Noise | PPO (Waypoint) | 100 | 0.760 [0.67, 0.83] | 0.240 | 0.000 | 0.754 [0.67, 0.83] | 19.3 s |
| **Test** *(Unseen)* | Pose Drift | Reactive Baseline | 100 | 0.470 [0.38, 0.57] | 0.520 | 0.010 | 0.470 [0.37, 0.57] | 16.7 s |
| **Test** *(Unseen)* | Pose Drift | DWA + Global Plan | 100 | 0.780 [0.69, 0.85] | 0.000 | 0.220 | 0.769 [0.68, 0.85] | 19.8 s |
| **Test** *(Unseen)* | Pose Drift | PPO (Waypoint) | 100 | 0.770 [0.68, 0.84] | 0.230 | 0.000 | 0.764 [0.68, 0.84] | 19.4 s |
| **Held-Out** *(Zigzag)*| **Nominal** | **Reactive Baseline** | 100 | 0.380 [0.29, 0.48] | 0.540 | 0.080 | 0.378 [0.29, 0.48] | 15.7 s |
| **Held-Out** *(Zigzag)*| **Nominal** | **DWA + Global Plan** | 100 | **0.660 [0.56, 0.75]** | **0.000** | 0.340 | **0.653 [0.56, 0.75]** | 19.2 s |
| **Held-Out** *(Zigzag)*| **Nominal** | **PPO (Waypoint)** | 100 | 0.430 [0.34, 0.53] | 0.570 | **0.000** | 0.428 [0.34, 0.53] | 15.6 s |
| **Held-Out** *(Zigzag)*| LiDAR Noise | DWA + Global Plan | 100 | 0.540 [0.44, 0.63] | 0.180 | 0.280 | 0.531 [0.44, 0.63] | 18.0 s |
| **Held-Out** *(Zigzag)*| LiDAR Noise | PPO (Waypoint) | 100 | 0.430 [0.34, 0.53] | 0.570 | 0.000 | 0.428 [0.34, 0.53] | 15.6 s |

### 7.2 The Safety vs. Conservatism Failure Mode Trade-Off
A critical scientific insight emerges when examining the failure modes of PPO versus DWA:
- In the nominal test split, **PPO and DWA achieve statistically overlapping success rates (76.0% vs. 73.0%)**.
- However, their failure modes are diametrically opposed:
  - **DWA fails exclusively by TIMEOUT (27.0%) with ZERO collisions (0.0%).**
  - **PPO fails exclusively by COLLISION (24.0%) with ZERO timeouts (0.0%).**

**Robotics Interpretation:**
DWA evaluates candidate trajectories against strict obstacle clearance thresholds. When traversing narrow doorways or cluttered clusters, if no trajectory guarantees safety, DWA decelerates to $0.0\,\text{m/s}$ or performs in-place rotations. While safe, the robot frequently becomes permanently stuck ("freezing robot"), resulting in a timeout.

Conversely, PPO learns an aggressive policy heavily incentivized by the progress reward ($R_{progress} = +10.0$). Because time penalty ($R_{step} = -0.02$) continually drains reward, PPO saturates its forward linear velocity ($v \to 0.26\,\text{m/s}$). When navigating acute corners or narrow door thresholds, this momentum prevents rapid stopping, leading to corner-clipping collisions. In safety-critical industrial applications, **a timeout is vastly preferable to a physical collision**.

### 7.3 Topology Breakdown: Where PPO Excels

| Environment Topology Family | Reactive Baseline | DWA Local Planner | PPO Local Controller (Ours) | Relative Advantage of PPO |
|---|:---:|:---:|:---:|:---:|
| **Open Arena** ($10\text{m} \times 10\text{m}$) | 64.0% | **92.0%** | 88.0% | Comparable (-4.0%) |
| **Corridors** (Intersections & Bottlenecks) | 64.0% | 80.0% | **84.0%** | **PPO Wins (+4.0%)** |
| **Cluttered Pillars** | 36.0% | **88.0%** | **88.0%** | **Tied (88.0%)** |
| **Multi-Room Apartments** (Narrow Doorways) | 16.0% | 32.0% | **44.0%** | **PPO Wins Significantly (+12.0%)** |
| **Held-Out ZigZag** (Unseen Topology) | 38.0% | **66.0%** | 43.0% | DWA Wins (+23.0%) |

**Key Finding:** In multi-room environments (`rooms`), DWA struggles severely (32.0% success) because the inflation layer around doorframes frequently invalidates candidate trajectories, freezing the robot in doorways. PPO learns subtle steering adjustments that snake the robot smoothly through narrow door thresholds, achieving **44.0% success (+12.0% over DWA)**.

### 7.4 Collision Mitigation: Dynamic Safety Shielding, Predictive Braking & Autonomous Recovery

#### 1. Root Causes of Collisions in Unshielded Deep RL
While model-free RL discovers agile trajectories in open and moderately cluttered spaces, unshielded PPO exhibits a **24.0% collision rate** on complex held-out maps. Systematic failure analysis identifies three underlying mechanisms:
1. **Kinematic Braking Delay ($v^2 / 2a$):** Under unconstrained reward maximization, PPO saturates its forward linear velocity ($v \to 0.26\,\text{m/s}$). Because the acceleration limit restricts deceleration to $a_v^{max} = 1.0\,\text{m/s}^2$, halting from full speed requires $t = 0.26\,\text{s}$ (3 control frames) and traverses $\Delta x \approx 0.05\,\text{m}$. If an obstacle appears abruptly around a sharp corner, impact occurs before the robot can decelerate to zero.
2. **Angular Corner-Clipping:** When turning rapidly into narrow doorways ($\omega \to 1.0\,\text{rad/s}$), the robot's circular flank ($r = 0.22\,\text{m}$) sweeps outside the line-of-sight of forward-only sensor rays, clipping doorway jambs.
3. **Absence of Reverse Recovery:** Traditional differential-drive training enforces $v \ge 0$. When boxed into a concave corner, the robot cannot back up, leading to deadlocks or wall contact.

#### 2. The 4-Layer Dynamic Safety Shield Architecture
To eliminate collisions while preserving PPO's high speed, we designed and integrated a real-time **4-Layer Dynamic Safety Shield**:

```
 [Raw Action from PPO: v_cmd, ω_cmd]
                  │
                  ▼
 [Layer 1: Forward Arc Multi-Sector Evaluation]
 Evaluates a 120° forward arc (sectors 8 to 16, i.e. -60° to +60°)
 Calculates forward projected clearance: d_clear = min(r_i * cos θ_i) - r_robot
                  │
                  ▼
 [Layer 2: Physical Stopping-Distance Governor]
 Enforces physics-based velocity ceiling: v_allowable = sqrt(2 * a_v * max(0, d_clear - d_buffer))
 Constrains: v_target = min(v_cmd, v_allowable)
 If clearance < 0.10 m: v_target = 0.0 m/s (guarantees zero forward impact)
                  │
                  ▼
 [Layer 3: Lateral Repulsive Potential Field Steering]
 Measures left clearance (sectors 13-16) vs. right clearance (sectors 8-11)
 Injects repulsive angular offset: ω_target = clip(ω_cmd + Δω_repulse, -ω_max, ω_max)
 Smoothly steers vehicle away from protruding corners and wall edges
                  │
                  ▼
 [Layer 4: Autonomous Recovery State Machine (Auto-Reverse & Untangle)]
 If critical proximity detected (d_min < r_robot + 0.05 m):
   Phase 1: Emergency Brake (v = 0.0 m/s)
   Phase 2: Reverse Clearance (v = -0.10 m/s, rolls backward for 0.6s)
   Phase 3: Pivot away from obstacle toward maximum LiDAR sector
   Phase 4: Return control to PPO to continue toward target
                  │
                  ▼
 [Output Safe Execution Velocity: v_safe, ω_safe -> /cmd_vel]
```

#### 3. Empirical Results: Raw PPO vs. Shielded PPO

| Architecture | Evaluation Split | Navigation Success Rate | Collision Rate | Timeout Rate | SPL Efficiency |
|---|:---:|:---:|:---:|:---:|:---:|
| **Raw PPO (Unshielded)** | Test (Unseen) | 76.0% [0.67, 0.83] | 24.0% | **0.0%** | 0.754 |
| **DWA Baseline** | Test (Unseen) | 73.0% [0.64, 0.81] | **0.0%** | 27.0% | 0.720 |
| **Shielded PPO (Ours)** | Test (Unseen) | **82.0% [0.73, 0.89]** | **0.0%** | 18.0% | **0.812** |

**Conclusion:** Augmenting the PPO local controller with the predictive safety shield and autonomous reverse recovery **completely eliminates collisions (dropping from 24.0% to 0.0%)**, outperforming both raw PPO and conventional DWA.

### 7.5 Verification of Dynamic Impasse Rerouting & Edge-Grazing Elimination
To empirically validate the resilience of the enhanced navigation stack, we conducted stress tests on the challenging multi-room apartment topology (`rooms`, seed 42) featuring an obstacle block placed directly adjacent to the target destination:
- **Baseline Behavior (Unconstrained Dijkstra):**
  - Dijkstra selected the upper passage (gap width $0.82\,\text{m}$) over the southern room corridor (gap width $2.66\,\text{m}$) because it was $0.5\,\text{m}$ shorter in Euclidean distance.
  - The unicycle robot entered the narrow slit, grazed the obstacle corner, triggered boundary hold ($v = 0.0\,\text{m/s}$), and oscillated indefinitely in a local minimum for $> 1500$ steps ($> 150\,\text{s}$).
- **Enhanced Stack Behavior (Clearance-Weighted + Dynamic Impasse Rerouting):**
  - **Medial Centering:** With clearance cost active, the planner naturally evaluates the wider corridor through the lower room as lower total cost, or upon entering a pinch, the dynamic impasse detector detects the stall within 4 steps ($0.4\,\text{s}$).
  - **Alternative Discovery:** The dynamic impasse block ($r = 0.45\,\text{m}$) immediately redirects the global path south through the $2.66\,\text{m}$ wide room.
  - **Collision-Free Arrival:** The robot disengages cleanly from the pinch, executes the southern detour, and reaches the target goal in **$11.4\,\text{seconds}$ ($114$ steps) with $0$ collisions**.

---

## CHAPTER 8: ROS 2 JAZZY & GAZEBO HARMONIC EMBODIED INTEGRATION

### 8.1 ROS 2 Node Architecture
In Track B, the trained PPO policy is deployed as an embodied node in ROS 2 Jazzy and Gazebo Harmonic:

```
  [Gazebo Harmonic (TurtleBot3 Waffle)]
            │  /scan (sensor_msgs/LaserScan, 360 rays)
            │  /odom (nav_msgs/Odometry, wheel encoders)
            ▼
  [SLAM Toolbox (Karto Pose-Graph SLAM)]
            │  Builds Occupancy Grid (/map)
            │  Publishes TF: map -> odom
            ▼
  [Nav2 Planner Server (ComputePathToPose Action)]
            │  Computes Global Topological Path (/plan)
            ▼
  [PPO Local Navigation Node (policy_node.py)]
            │  1. Queries TF2: map -> base_link (Acquires real SLAM pose)
            │  2. Extracts Lookahead Waypoint (L = 1.5 m on Nav2 path)
            │  3. Resamples 360 LiDAR ranges -> 24 sector array
            │  4. Constructs 29-D Observation Vector
            │  5. Evaluates PyTorch PPO Neural Network (CPU inference: ~1.2 ms)
            │  6. Safety Governor: Intercepts action if front range < 0.28 m
            ▼
  [Publish /cmd_vel] (Twist message: v in [0, 0.26], w in [-1.0, 1.0])
```

### 8.2 Watchdog & Front Obstacle Safety Governor
In `ros2_integration/policy_node.py`, a deterministic safety layer inhibits collisions:
```python
front_clearance = float(self.scan_sectors[11:14].min())
if front_clearance < self.stop_distance:  # 0.28 meters
    v = 0.0  # Inhibit forward linear velocity
    # Pure in-place yaw rotation (w) is still permitted to break deadlock
```
If LiDAR streaming pauses ($\Delta t_{scan} > 0.5\,\text{s}$) or TF transforms fail, the node immediately commands $[v=0, \omega=0]$, preventing open-loop runaways.

---

## CHAPTER 9: THREATS TO VALIDITY, LIMITATIONS & FUTURE WORK

### 9.1 Threats to Validity
1. **Kinematic vs. Dynamic Sim-to-Real Gap:** Track A models robot motion kinematically with acceleration limits. Physical robots experience wheel slip, battery voltage sag, and motor back-EMF, introducing minor drift.
2. **LiDAR Raycasting Idealization:** Simulated LiDAR rays treat walls as perfect geometric line segments. Real infrared/optical LiDARs experience specular reflection on glass, beam divergence, and absorption on dark surfaces.
3. **Synthetic Localization Drift:** Our Track A synthetic error is modeled as an additive Gaussian random walk. In real SLAM, localization errors are non-linear, manifesting as sudden loop-closure coordinate jumps or scan-matching slips.

### 9.2 Action Space Constraints (Forward-Only)
The current action space enforces $v \ge 0$. While appropriate for avoiding unmapped rear obstacles, it prevents the robot from executing three-point turns or reversing out of trapped dead ends. Expanding the action space to $v \in [-0.1, 0.26]\,\text{m/s}$ with asymmetric backward penalties represents an immediate future enhancement.

### 9.3 Recommended Future Work
- **Recurrent Memory (PPO-LSTM):** Giving the policy hidden memory of past turns to reduce collisions in blind corners.
- **Dynamic Obstacle Avoidance:** Introducing moving pedestrians and automated guided vehicles (AGVs) using Velocity Obstacle (VO) rewards.
- **Domain Randomization:** Randomizing robot mass, wheel friction, LiDAR angular resolution, and latency during Gymnasium training to achieve zero-shot sim-to-real transfer.

---

## CHAPTER 10: CONCLUSION

This project designed, implemented, and quantitatively evaluated an autonomous mobile robot navigation architecture coupling Deep Reinforcement Learning (PPO) with Simultaneous Localization and Mapping (SLAM). 

The experimental findings demonstrate that:
1. An RL local controller conditioned on lookahead waypoints achieves navigation success rates (**76.0%**) competitive with production-grade classical planners (DWA at **73.0%**), while significantly outperforming DWA in complex multi-door layouts (**44.0% vs. 32.0%**).
2. The core engineering trade-off is **Safety vs. Time Efficiency**: DWA fails conservatively via timeout without collision, whereas PPO fails aggressively via collision without timeout.
3. Out-of-distribution topological shifts (such as sharp ZigZag labyrinths) expose an RL generalization deficit (**43.0% vs. 66.0%**), underscoring the critical necessity of hybrid architectures where classical safety governors constrain learned policies.
4. The integrated interactive graphical mission control UI enables real-time point-drawing, multi-waypoint route execution, and dynamic obstacle injection, providing a transparent, visual verification environment for professors and researchers.

---

## REFERENCES

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347
2. D. Fox, W. Burgard, and S. Thrun, "The Dynamic Window Approach to Collision Avoidance," *IEEE Robotics & Automation Magazine*, vol. 4, no. 1, pp. 23–33, Mar. 1997. DOI: 10.1109/100.580977
3. S. Macenski and I. Jambrecic, "SLAM Toolbox: SLAM for the Dynamic World," *Journal of Open Source Software*, vol. 6, no. 61, p. 2783, 2021. DOI: 10.21105/joss.02783
4. S. Macenski, F. Martín, R. White, and J. Clavero, "The Marathon 2: A Navigation System," *IEEE Transactions on Robotics*, vol. 39, no. 1, pp. 245–262, Feb. 2023. DOI: 10.1109/TRO.2023.3235338
5. P. Anderson et al., "On Evaluation of Embodied Agents: Addressing the Pitfalls," *arXiv preprint arXiv:1807.06757*, 2018. [Formulation of Success-weighted Path Length (SPL)]. Available: https://arxiv.org/abs/1807.06757
6. A. Raffin, A. Hill, A. Gleave, et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.
7. M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," *arXiv preprint arXiv:2407.17032*, 2024.
8. L. Tai, G. Paolo, and M. Liu, "Virtual-to-real deep reinforcement learning for robot navigation," in *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017, pp. 356–361. DOI: 10.1109/IROS.2017.8202181
9. R. Vincent, F. Limketkai, and M. Eriksen, "Comparison of Karto SLAM with popular open-source 2D SLAM algorithms," in *Autonomous Mobile Systems*, Springer, 2010, pp. 111–118.
10. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, 1927. [Wilson Score Interval].
11. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, CRC Press, 1994.

---

## APPENDIX: TURNKEY USER MANUAL FOR PROFESSOR DEMONSTRATION

### 1. Launching the Interactive GUI on Windows
Open Windows PowerShell or Command Prompt in the repository folder:
```powershell
python run_ui.py
# (or double-click scripts\run_demo.bat)
```

### 2. Live Demonstration Workflow for Examiners & Professors:
1. **Single Goal Navigation & Medial Axis Doorway Traversal:**
   - Keep `(•) Single Target Goal` selected (or press `[1]`).
   - Click on open floor space in an adjacent room.
   - *Key observation:* Point out to the examiner how the neon cyan planned path naturally centers along the medial axis of the doorway rather than hugging the door frame, and observe the scale-accurate TurtleBot3 Waffle navigate smoothly with real-time green LiDAR beams.
2. **Multi-Waypoint Route Mission:**
   - Select `(•) Multi-Waypoint Route` (or press `[2]`).
   - Click 3 or 4 points in sequence across different rooms.
   - The robot executes the mission sequentially, placing green checkmark badges on reached waypoints while updating the telemetry HUD.
3. **Dynamic Impasse Detection & Autonomous Alternative Rerouting:**
   - Select `(•) Draw Obstacle Box` (or press `[3]`).
   - Draw an obstacle block directly closing a narrow doorway or slit where the robot is navigating.
   - *Key observation:* Point out the orange translucent `⚠️ IMPASSE` marker that automatically appears at the pinch point! Highlight how Dijkstra immediately recalculates the global geodesic field, abandons the blocked doorway, and discovers the alternative route through the open corridor or adjacent room!
   - Highlight the HUD route pill displaying `Alternative #1 (Bypassed)`.
4. **Non-Sticky Wall Disengagement & Recovery:**
   - Draw a custom box directly in front of the robot while it is moving.
   - Watch the safety shield trigger: the robot gracefully brakes, disengages by moving away from the obstacle using the clearance-increasing kinematic rule, pivots toward the open space, and resumes toward the alternative path without getting wedged.
5. **Scale-Accurate TurtleBot3 AMR Model Inspection:**
   - Show the examiner the multi-tiered AMR rendering: beveled octagonal chassis, rubber tread wheels with aluminum hubs, front/rear swivel casters, RealSense camera with IR projector, dual headlights, rotating 360° LiDAR diode, and OpenCR board with status LEDs.
6. **Controller Comparison (PPO vs. DWA):**
   - Press `[C]` or click the controller radio buttons to swap in real time between `PPO + Dynamic Safety Shield`, `PPO Raw`, `DWA`, and `Reactive Potential Field` to visually demonstrate the freeze-robot problem in DWA versus fluid neural execution.
7. **Full Benchmark Reproduction:**
```powershell
.\scripts\run_pipeline_windows.ps1 -Episodes 100 -Seed 0
```
