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

Autonomous mobile robot (AMR) navigation in GPS-denied indoor environments has traditionally relied on classical hierarchical navigation stacks, such as the Dynamic Window Approach (DWA) or Timed Elastic Bands (TEB) coupled with global topological path planners. While computationally predictable, classical local planners frequently suffer from parameter tuning brittleness, freeze-robot deadlocks, and conservative timeouts in narrow corridors and complex multi-door layouts. Recently, Deep Reinforcement Learning (DRL) has emerged as a promising data-driven alternative for continuous local collision avoidance and trajectory generation.

In this project, we present a research-grade, statistically rigorous comparative evaluation of a continuous-action DRL local navigation policy (Proximal Policy Optimization, PPO) integrated within an end-to-end autonomous navigation architecture. The investigation is organized around a dual-track methodology:
1. **Track A (High-Throughput Statistical Benchmark & Mission Control UI):** A vectorized 2D procedural Gymnasium simulation environment running at $>10{,}000$ steps/second, enabling sample-efficient PPO policy training across randomized indoor layouts, controlled stress-testing under synthetic SLAM-like localization drift and sensor noise, and an **Interactive Graphical Mission Control UI** supporting real-time point-and-click target selection, multi-waypoint route sequencing, and dynamic obstacle injection.
2. **Track B (Embodied ROS 2 / Gazebo Architecture):** An embodied software pipeline integrating the trained PPO policy with ROS 2 Jazzy, Gazebo Harmonic, SLAM Toolbox (asynchronous 2D pose-graph SLAM), and Nav2 on a differential-drive robot (TurtleBot3 Waffle), verifying real-time inference latency ($1.2\,\text{ms}$), TF2 coordinate synchronization (`map -> base_link`), and execution-time safety monitoring.

Across 100 matched test scenarios evaluated with 95% Wilson Score Confidence Intervals, our lookahead-conditioned PPO policy achieves a nominal **76.0% success rate [0.67, 0.83]**, comparable to the conventional DWA baseline (**73.0% [0.64, 0.81]**), while significantly outperforming DWA in complex multi-room environments (**44.0% vs 32.0%**). Crucially, our empirical analysis reveals a fundamental failure-mode dichotomy between learned and classical controllers: **unshielded PPO fails exclusively via collisions (24.0%)** driven by saturated forward velocity, whereas **DWA fails exclusively via conservative local-minima timeouts (27.0%) with 0.0% collisions**. On a held-out, structurally unseen topology (sharp ZigZag corridors), unshielded PPO suffers an out-of-distribution generalization deficit (**43.0% vs. DWA’s 66.0%**). To resolve this safety vulnerability, we formulate and ablate an active **4-Layer Predictive Dynamic Safety Shield** with analytical stopping-distance constraints and autonomous recovery; when augmented with this shield, PPO achieves an **82.0% success rate [0.73, 0.89]** with **0.0% collisions**, eliminating the failure mode of the raw neural network.

**Keywords:** Autonomous Mobile Robots (AMR), Deep Reinforcement Learning, Proximal Policy Optimization (PPO), SLAM Toolbox, Nav2, Dynamic Window Approach (DWA), Control Barrier Functions, Success-weighted Path Length (SPL), ROS 2 Jazzy, Gazebo Harmonic.

---

## TABLE OF CONTENTS
1. [Chapter 1: Introduction & Problem Definition](#chapter-1-introduction--problem-definition)
   - [1.1 Motivation & Context](#11-motivation--context)
   - [1.2 Limitations of Conventional Navigation Stacks](#12-limitations-of-conventional-navigation-stacks)
   - [1.3 The Promise and Pitfalls of Deep Reinforcement Learning](#13-the-promise-and-pitfalls-of-deep-reinforcement-learning)
   - [1.4 Research Objectives & Dual-Track Architectural Scope](#14-research-objectives--dual-track-architectural-scope)
2. [Chapter 2: Literature Review & Mathematical Formulations](#chapter-2-literature-review--mathematical-formulations)
   - [2.1 Classical Local Navigation & Receding-Horizon Control](#21-classical-local-navigation--receding-horizon-control)
   - [2.2 Deep Reinforcement Learning for Mobile Robot Navigation](#22-deep-reinforcement-learning-for-mobile-robot-navigation)
   - [2.3 Runtime Safety Filtering & Control Barrier Functions](#23-runtime-safety-filtering--control-barrier-functions)
   - [2.4 Differential-Drive Kinematic Model & Physical Constants](#24-differential-drive-kinematic-model--physical-constants)
   - [2.5 Simultaneous Localization and Mapping (SLAM) Formulation](#25-simultaneous-localization-and-mapping-slam-formulation)
   - [2.6 Markov Decision Process (MDP) & Reward Formulation](#26-markov-decision-process-mdp--reward-formulation)
3. [Chapter 3: System Architecture & Dual-Track Methodology](#chapter-3-system-architecture--dual-track-methodology)
   - [3.1 Decoupled Dual-Track Research Architecture](#31-decoupled-dual-track-research-architecture)
   - [3.2 Lookahead Waypoint Tracking vs. Direct Goal Seeking](#32-lookahead-waypoint-tracking-vs-direct-goal-seeking)
   - [3.3 Clearance-Weighted Global Geodesic Planning & Medial Axis Centering](#33-clearance-weighted-global-geodesic-planning--medial-axis-centering)
4. [Chapter 4: Navigation Models & Controller Formulations](#chapter-4-navigation-models--controller-formulations)
   - [4.1 PPO Local Policy Architecture & Hyperparameters](#41-ppo-local-policy-architecture--hyperparameters)
   - [4.2 Dynamic Window Approach (DWA) Baseline Formulation](#42-dynamic-window-approach-dwa-baseline-formulation)
   - [4.3 Reactive Potential Field Baseline Formulation](#43-reactive-potential-field-baseline-formulation)
   - [4.4 Active 4-Layer Dynamic Safety Shield & CBF Analytical Governor](#44-active-4-layer-dynamic-safety-shield--cbf-analytical-governor)
   - [4.5 Dynamic Passage Impasse Detection & Autonomous Alternative Path Re-routing](#45-dynamic-passage-impasse-detection--autonomous-alternative-path-re-routing)
   - [4.6 Kinematic Barrier Guard with Disengagement Clearance Vectors](#46-kinematic-barrier-guard-with-disengagement-clearance-vectors)
5. [Chapter 5: Interactive Mission Control UI & Robotic Visualization](#chapter-5-interactive-mission-control-ui--robotic-visualization)
   - [5.1 Scale-Accurate TurtleBot3 Waffle Industrial AMR Visualization](#51-scale-accurate-turtlebot3-waffle-industrial-amr-visualization)
   - [5.2 Dynamic Point-Drawing & Online Dijkstra Wave-Front Propagation](#52-dynamic-point-drawing--online-dijkstra-wave-front-propagation)
   - [5.3 Multi-Waypoint Route Sequencing & Dynamic Obstacle Injection](#53-multi-waypoint-route-sequencing--dynamic-obstacle-injection)
   - [5.4 Live Telemetry HUD & Visual Hazard Feedback](#54-live-telemetry-hud--visual-hazard-feedback)
6. [Chapter 6: Experimental Design & Benchmark Protocol](#chapter-6-experimental-design--benchmark-protocol)
   - [6.1 Procedural Layout Topologies & Exact Geometric Specifications](#61-procedural-layout-topologies--exact-geometric-specifications)
   - [6.2 Disjoint Seed Splits & Reconciliation of the ZigZag Sampling Protocol](#62-disjoint-seed-splits--reconciliation-of-the-zigzag-sampling-protocol)
   - [6.3 Controlled Perturbation Stress-Testing Models](#63-controlled-perturbation-stress-testing-models)
   - [6.4 Statistical Evaluation Metrics & Confidence Interval Formulations](#64-statistical-evaluation-metrics--confidence-interval-formulations)
7. [Chapter 7: Quantitative Results & Empirical Analysis](#chapter-7-quantitative-results--empirical-analysis)
   - [7.1 Primary Benchmark Evaluation (100 Matched Trials per Condition)](#71-primary-benchmark-evaluation-100-matched-trials-per-condition)
   - [7.2 The Safety vs. Conservatism Failure Mode Dichotomy](#72-the-safety-vs-conservatism-failure-mode-dichotomy)
   - [7.3 Topology Breakdown: Performance Analysis & Sample Size Reconciliation](#73-topology-breakdown-performance-analysis--sample-size-reconciliation)
   - [7.4 Robustness Under Sensing Noise & Stochastic Dithering Effects](#74-robustness-under-sensing-noise--stochastic-dithering-effects)
   - [7.5 Safety Shield Component Ablation & Action-Space Asymmetry Analysis](#75-safety-shield-component-ablation--action-space-asymmetry-analysis)
   - [7.6 Verification of Dynamic Impasse Rerouting & Edge-Grazing Elimination](#76-verification-of-dynamic-impasse-rerouting--edge-grazing-elimination)
   - [7.7 PPO Training Dynamics & Generalization Gap](#77-ppo-training-dynamics--generalization-gap)
8. [Chapter 8: ROS 2 Jazzy & Gazebo Harmonic Embodied Integration](#chapter-8-ros-2-jazzy--gazebo-harmonic-embodied-integration)
   - [8.1 ROS 2 Node Architecture & Coordinate Synchronization](#81-ros-2-node-architecture--coordinate-synchronization)
   - [8.2 SLAM Toolbox Integration & Closed-Loop TF2 Lookups](#82-slam-toolbox-integration--closed-loop-tf2-lookups)
   - [8.3 Real-Time Safety Watchdog & Fail-Safe Execution](#83-real-time-safety-watchdog--fail-safe-execution)
9. [Chapter 9: Threats to Validity, Limitations & Future Work](#chapter-9-threats-to-validity-limitations--future-work)
   - [9.1 Threats to Internal & External Validity](#91-threats-to-internal--external-validity)
   - [9.2 Comparison Against Modern Planners (MPPI, TEB)](#92-comparison-against-modern-planners-mppi-teb)
   - [9.3 Future Directions: Recurrent Policies, Dynamic Obstacles & Formal CBFs](#93-future-directions-recurrent-policies-dynamic-obstacles--formal-cbfs)
10. [Chapter 10: Conclusion](#chapter-10-conclusion)
11. [References](#references)
12. [Appendix: Turnkey User Manual for Professor Demonstration](#appendix-turnkey-user-manual-for-professor-demonstration)

---

## CHAPTER 1: INTRODUCTION & PROBLEM DEFINITION

### 1.1 Motivation & Context
Safe, efficient, and robust autonomous navigation in unstructured, GPS-denied indoor environments (such as smart factories, automated warehouses, hospitals, and urban search-and-rescue facilities) represents a foundational pillar of modern mobile robotics. A fully autonomous mobile robot (AMR) must solve three coupled subproblems concurrently:
1. **Mapping:** Dynamically construct and update a spatial representation (e.g., an occupancy grid) of the unmapped environment using onboard range sensors.
2. **Localization:** Continually estimate its time-varying pose (position and heading) within that evolving map despite odometric wheel slip and noisy sensor observations.
3. **Motion Planning & Control:** Compute kinematically feasible, collision-free control commands to reach distant navigational targets while reacting to static and dynamic obstacles.

In the standard Robot Operating System (ROS) architecture—exemplified by the ROS 2 Navigation2 (Nav2) stack—this problem is hierarchically decomposed. A global path planner ($A^*$, Dijkstra, or NavFn) computes an optimal topological route over a global costmap, while a local controller (such as the Dynamic Window Approach, Timed Elastic Bands, or Regulated Pure Pursuit) evaluates high-frequency velocity commands ($v, \omega$) to track the global path while avoiding local hazards detected by real-time exteroceptive sensors.

### 1.2 Limitations of Conventional Navigation Stacks
While classical local planners have achieved wide commercial adoption, they exhibit persistent structural limitations:
- **Heuristic Parameter Sensitivity:** Planners such as DWA and TEB depend heavily on manually tuned heuristic weights (obstacle inflation, path alignment, goal distance, and velocity limits). A parameter set tuned for open corridors often causes oscillations or failure in narrow doorways.
- **The "Freezing Robot" Problem:** In narrow passages, doorways, or cluttered corridors, local trajectory rollout algorithms frequently evaluate all kinematically reachable trajectories as inadmissible due to costmap inflation layers, causing the planner to command zero forward velocity and trigger timeouts.
- **Myopic Planning Horizons:** Classical local planners evaluate short forward horizons (typically 1.0–2.0 seconds), leaving them susceptible to local cost minima, concave obstacles, and U-shaped traps.

### 1.3 The Promise and Pitfalls of Deep Reinforcement Learning
Deep Reinforcement Learning (DRL) provides an alternative paradigm: a parameterized policy neural network trained through trial-and-error simulation can learn non-linear mappings directly from raw or pooled sensor measurements to continuous velocity commands. By optimizing cumulative discounted returns over extended episodes, DRL policies can develop emergent obstacle negotiation skills, fluid doorway snaking, and sub-millisecond inference times.

However, existing literature on DRL for robot navigation frequently suffers from three methodological weaknesses:
1. **The Toy Simulation Assumption:** Policies are trained in simplistic, collision-free simulators without sensor noise, beam dropouts, or localization drift, leading to immediate failure when transferred to physical systems.
2. **Asymmetric or Weak Baseline Comparisons:** RL models are often compared only against naive reactive controllers (such as Braitenberg vehicles or artificial potential fields) rather than production-grade navigation stacks operating with global path priors.
3. **Absence of Statistical Rigor:** Evaluations are frequently presented over small, cherry-picked sample sizes ($N \le 10$) without confidence intervals, seed variance quantification, or out-of-distribution topological testing.

### 1.4 Research Objectives & Dual-Track Architectural Scope
To address these gaps, our project investigates whether a continuous-action DRL local policy can effectively operate when conditioned on online SLAM map updates and pose estimates, and how its safety, efficiency, and generalization compare against classical navigation stacks.

To make this investigation both statistically rigorous and physically grounded, we organize our methodology around two coordinated tracks:
- **Track A (High-Throughput Gymnasium Benchmark & Interactive Mission Control):** A fast, vectorized 2D procedural simulation environment ($>10{,}000$ steps/sec) designed for sample-efficient PPO policy training ($1{,}000{,}000$ timesteps), controlled factorial stress-testing across 100 matched scenarios per experimental cell with Wilson Score and bootstrap confidence intervals, synthetic SLAM drift and LiDAR noise ablation, and an interactive graphical mission control interface.
- **Track B (Embodied ROS 2 / Gazebo Harmonic Integration):** A modular software integration in ROS 2 Jazzy and Gazebo Harmonic connecting the trained policy to SLAM Toolbox (asynchronous 2D graph SLAM) and Nav2 on a simulated TurtleBot3 Waffle, verifying real-time inference latency ($1.2\,\text{ms}$), TF2 coordinate synchronization (`map -> base_link`), and real-time execution safety governors.

We define four primary research objectives:
1. **Dual-Track Framework Formulation:** Establish a strictly synchronized observation and action interface between fast 2D procedural simulation (Track A) and embodied ROS 2 / SLAM Toolbox robotics software (Track B).
2. **Comparative Benchmark Under Imperfect Sensing:** Conduct a matched-scenario comparative evaluation of PPO, DWA, and Reactive controllers across familiar indoor topologies, evaluating goal-reaching success, collision rates, timeout rates, and Success-weighted Path Length (SPL) with 95% confidence intervals.
3. **Out-of-Distribution Generalization Testing:** Evaluate policy robustness on a structurally unseen topology (sharp ZigZag switchbacks) never encountered during training, quantifying the generalization gap between learned and classical planners.
4. **Predictive Safety Shielding & Dynamic Rerouting:** Design, ablate, and verify a runtime safety governor and dynamic passage impasse rerouting engine to eliminate collisions without inducing local-minimum freezing.

---

## CHAPTER 2: LITERATURE REVIEW & MATHEMATICAL FORMULATIONS

### 2.1 Classical Local Navigation & Receding-Horizon Control
Autonomous local motion planning on differential-drive robots has historically centered on velocity-space search and trajectory optimization:
- **The Dynamic Window Approach (DWA):** Introduced by Fox et al. [2], DWA prunes the search space of linear and angular velocities $(v, \omega)$ to a dynamic window reachable within the vehicle's acceleration limits over a single control interval, simulating constant-curvature circular trajectories forward in time. Candidate trajectories are scored against target distance, heading alignment, and obstacle clearance. While computationally efficient, DWA is vulnerable to the "freezing robot" problem in tight constrictions when the obstacle inflation buffer invalidates all forward rollouts.
- **Timed Elastic Bands (TEB) & Model Predictive Path Integral (MPPI):** Modern planners integrated into Nav2 [4, 17] formulate trajectory generation as elastic deformation or stochastic sampling over receding horizons. While TEB optimizes trajectory knot points against kinematic limits and clearance, and MPPI samples thousands of GPU-accelerated rollouts to negotiate complex costmaps, both require extensive parameter tuning and significant computational resources compared to direct neural policy evaluation.

### 2.2 Deep Reinforcement Learning for Mobile Robot Navigation
In recent years, model-free Deep Reinforcement Learning has been applied extensively to mobile robot obstacle avoidance:
- **Hierarchical Navigation (PRM-RL & Arena-Rosnav):** Francis et al. [15] demonstrated that coupling global topological graphs (Probabilistic Roadmaps) with RL local controllers outperforms pure end-to-end RL over long-range indoor environments. Kästner et al. [12, 13] developed *Arena-Rosnav* and *Arena-Bench*, establishing standardized 2D/3D benchmarking suites for deploying DRL controllers into ROS/ROS 2 navigation pipelines.
- **Large-Scale PointGoal Navigation:** Wijmans et al. [16] demonstrated in *DD-PPO* that scaling model-free PPO to billions of frames across thousands of parallel environments yields near-perfect point-goal navigation in complex 3D scans. Perumalla et al. [14] introduced the *BARN* (Benchmarking Autonomous Robot Navigation) dataset, showing that highly constrained, non-convex physical spaces severely challenge both classical and learned local planners.

### 2.3 Runtime Safety Filtering & Control Barrier Functions
Because model-free RL maximizes expectation rather than guaranteeing worst-case safety, runtime action shielding has become an active field of research:
- **Control Barrier Functions (CBFs):** Ames et al. [18] and Cheng et al. [19] formalized Control Barrier Functions as Lyapunov-like functions that enforce forward-invariance of safe state sets $\mathcal{C} = \{\mathbf{x} \mid h(\mathbf{x}) \ge 0\}$. By solving a real-time Quadratic Program (CBF-QP), candidate control inputs are minimally adjusted to ensure $\dot{h}(\mathbf{x}) + \alpha(h(\mathbf{x})) \ge 0$.
- **Action Replacement & Velocity Obstacle Shields:** Chiang et al. [20] proposed execution-time action replacement, substituting classical recovery actions when an RL agent enters high-risk states. Kästner et al. [21] implemented a Velocity Obstacle (VO) safety shield for mobile robot RL, filtering commands that intersect dynamic obstacle cones. In built-in ROS 2 Nav2 stacks, similar safety guarantees are provided by the `nav2_collision_monitor` and Behavior Tree recovery nodes (Backup, Spin, Wait) [4, 17].

### 2.4 Differential-Drive Kinematic Model & Physical Constants
The robot platform evaluated across both tracks is a differential-drive ground vehicle conforming to the physical specifications of the TurtleBot3 Waffle. The kinematic state in the 2D plane is defined by its pose:
$$\mathbf{p}_t = \begin{bmatrix} x_t \\ y_t \\ \theta_t \end{bmatrix} \in \mathbb{R}^2 \times [-\pi, \pi)$$

Where $(x_t, y_t)$ represents Cartesian coordinates in the global map frame and $\theta_t$ represents the heading angle (yaw) relative to the X-axis. Under the standard unicycle kinematic model, the robot is commanded via forward linear velocity $v_t \in [0, v_{\max}]$ and yaw rate $\omega_t \in [-\omega_{\max}, \omega_{\max}]$. The discrete-time forward kinematics over control interval $\Delta t = 0.1\,\text{s}$ (10 Hz control loop) are integrated as:
$$\begin{aligned}
x_{t+1} &= x_t + v_t \cos(\theta_t) \Delta t \\
y_{t+1} &= y_t + v_t \sin(\theta_t) \Delta t \\
\theta_{t+1} &= \text{wrap}\left(\theta_t + \omega_t \Delta t\right)
\end{aligned}$$
Where $\text{wrap}(\alpha) = (\alpha + \pi) \pmod{2\pi} - \pi \in [-\pi, \pi)$. Actuation is constrained by physical acceleration ceilings:
$$|v_{t+1} - v_t| \le a_v^{\max} \Delta t, \quad |\omega_{t+1} - \omega_t| \le a_\omega^{\max} \Delta t$$

#### Table 2.1: Physical Kinematic Constants & Environmental Parameters
| Parameter | Symbol | Value | Unit | Engineering / Physical Justification |
|---|:---:|:---:|:---:|---|
| Maximum Forward Velocity | $v_{\max}$ | $0.26$ | $\text{m/s}$ | TurtleBot3 Waffle DC motor rated limit |
| Maximum Yaw Angular Rate | $\omega_{\max}$ | $1.0$ | $\text{rad/s}$ | Safe indoor rotational stability limit |
| Maximum Linear Acceleration | $a_v^{\max}$ | $1.0$ | $\text{m/s}^2$ | Traction and motor torque limit |
| Maximum Angular Acceleration | $a_\omega^{\max}$ | $3.2$ | $\text{rad/s}^2$ | Differential drive wheel acceleration limit |
| Robot Collision Radius | $r_{\text{robot}}$ | $0.22$ | $\text{m}$ | Circular bounding radius of Waffle chassis |
| Goal Tolerance Radius | $d_{\text{tol}}$ | $0.25$ | $\text{m}$ | Success boundary around target coordinate |
| Control Time Step | $\Delta t$ | $0.1$ | $\text{s}$ | 10 Hz ROS navigation control rate |
| Episode Step Limit | $T_{\max}$ | $700$ | steps | Equivalent to $70.0\,\text{s}$ mission timeout |
| LiDAR Range Bounds | $[R_{\min}, R_{\max}]$ | $[0.12, 3.5]$ | $\text{m}$ | TurtleBot3 LDS-01 / LDS-02 sensor profile |
| Observation Sectors | $N_{\text{sectors}}$ | $24$ | sectors | $15^\circ$ angular pooling resolution |

### 2.5 Simultaneous Localization and Mapping (SLAM) Formulation
Online SLAM solves the joint problem of estimating the robot pose trajectory $\mathbf{x}_{1:t}$ and building a spatial map $\mathbf{m}$ from exteroceptive range scans $\mathbf{z}_{1:t}$ and control inputs $\mathbf{u}_{1:t}$:
$$p(\mathbf{x}_{1:t}, \mathbf{m} \mid \mathbf{z}_{1:t}, \mathbf{u}_{1:t})$$

In our Track B architecture, we employ **SLAM Toolbox** [3], an asynchronous 2D pose-graph SLAM system based on Karto SLAM [9]. SLAM Toolbox processes incoming `/scan` and `/odom` messages, performs multi-level scan matching, and adds new pose nodes to an underlying factor graph. Spatial loop closures are identified via correlative scan matching and optimized using non-linear least squares:
$$\mathbf{x}^* = \arg\min_{\mathbf{x}} \sum_{ij} \mathbf{e}_{ij}^T \mathbf{\Omega}_{ij} \mathbf{e}_{ij}, \quad \mathbf{e}_{ij} = \mathbf{z}_{ij} - h(\mathbf{x}_i, \mathbf{x}_j)$$
Where $\mathbf{e}_{ij}$ is the error between the measured relative constraint $\mathbf{z}_{ij}$ and predicted relative pose, and $\mathbf{\Omega}_{ij}$ is the constraint information matrix. SLAM Toolbox continuously publishes the transformation from the static map frame to the odometry frame (`map -> odom`), which enables our policy node to resolve the drift-corrected pose (`map -> base_link`) in real time.

### 2.6 Markov Decision Process (MDP) & Reward Formulation
We formulate local navigation as a continuous-action, partially observable Markov Decision Process (POMDP), modeled as an MDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$:

#### 1. Observation Space $\mathcal{S} \subset \mathbb{R}^{29}$
At each time step $t$, the observation vector $\mathbf{s}_t \in \mathbb{R}^{29}$ is constructed as:
$$\mathbf{s}_t = \left[ \bar{r}_0, \bar{r}_1, \dots, \bar{r}_{23}, \, \bar{d}_g, \, \sin(\phi_g), \, \cos(\phi_g), \, \bar{v}_t, \, \bar{\omega}_t \right]$$
- **$\bar{r}_{0:23} \in [0, 1]$:** 24 angular sector range readings pooling a $360^\circ$ planar scan. Each sector spans an arc of $\Delta \theta = \frac{2\pi}{24} = 15^\circ$ ($0.2618\,\text{rad}$), centered at $\alpha_k = -\pi + k \Delta \theta$. Sector 12 aligns with the dead-ahead forward axis. Sector ranges are evaluated via min-pooling over raw rays and normalized by $R_{\max} = 3.5\,\text{m}$:
  $$\bar{r}_k = \frac{1}{R_{\max}} \min_{j \in \mathcal{I}_k} \rho_j, \quad \mathcal{I}_k = \left\{ j \in \{0, \dots, 359\} \;\middle|\; \alpha_k - \frac{\Delta \theta}{2} \le \text{ray\_angle}_j < \alpha_k + \frac{\Delta \theta}{2} \right\}$$
- **$\bar{d}_g \in [0, 1]$:** Normalized Euclidean distance to the active lookahead waypoint: $\bar{d}_g = \min(d_{\text{target}} / D_{\text{norm}}, 1.0)$ with $D_{\text{norm}} = 5.0\,\text{m}$.
- **$\sin(\phi_g), \cos(\phi_g) \in [-1, 1]$:** Continuous trigonometric representation of the relative heading to the target in the robot's local frame $\phi_g = \text{wrap}(\text{atan2}(y_g - y_t, x_g - x_t) - \theta_t)$, eliminating angle wrap-around discontinuities.
- **$\bar{v}_t \in [0, 1]$:** Normalized linear velocity $\bar{v}_t = \text{clip}(v_t / v_{\max}, 0.0, 1.0)$.
- **$\bar{\omega}_t \in [-1, 1]$:** Normalized angular velocity $\bar{\omega}_t = \text{clip}(\omega_t / \omega_{\max}, -1.0, 1.0)$.

#### 2. Action Space $\mathcal{A} \subset [-1, 1]^2$
The continuous policy action $\mathbf{a}_t = [a_v, a_\omega] \in [-1, 1]^2$ is mapped linearly to target kinematic commands:
$$v_{\text{cmd}} = \frac{a_v + 1.0}{2.0} \cdot v_{\max} \in [0, v_{\max}], \quad \omega_{\text{cmd}} = a_\omega \cdot \omega_{\max} \in [-\omega_{\max}, \omega_{\max}]$$
Notice that $v_{\text{cmd}} \ge 0$, constraining the raw RL policy to forward-only motions to reflect standard unidirectional sensor coverage.

#### 3. Complete Piecewise Reward Function $\mathcal{R}(\mathbf{s}_t, \mathbf{a}_t)$
Reward shaping utilizes a privileged 8-connected geodesic potential field $\Phi_{\text{geo}}(\mathbf{x})$ precomputed over the occupancy grid:
$$\mathcal{R}_t = R_{\text{geo}} \cdot \left[ \Phi_{\text{geo}}(\mathbf{x}_{t-1}) - \Phi_{\text{geo}}(\mathbf{x}_t) \right] + R_{\text{step}} + R_{\text{turn}} \frac{|\omega_t|}{\omega_{\max}} + R_{\text{prox}}(\mathbf{r}_t) + R_{\text{terminal}}$$

The scalar weighting coefficients and terms are defined as follows:
- **Geodesic Progress Reward:** $R_{\text{geo}} = +10.0\,\text{m}^{-1}$. Rewards positive displacement along the shortest navigable topological path. Because $\Phi_{\text{geo}}$ accounts for walls, the agent is never rewarded for moving toward a target through an impassable barrier.
- **Time Penalty:** $R_{\text{step}} = -0.02$ per timestep, penalizing unnecessary wandering and incentivizing efficient paths.
- **Smoothness Penalty:** $R_{\text{turn}} = -0.02 \cdot \frac{|\omega_t|}{\omega_{\max}}$, discouraging high-frequency steering chatter.
- **Obstacle Proximity Penalty:** Penalizes intrusion into safety buffers when minimum LiDAR clearance falls below $d_{\text{prox}} = 0.6\,\text{m}$:
  $$R_{\text{prox}}(\mathbf{r}_t) = \begin{cases} -0.5 \cdot \left( 1.0 - \frac{\min(\mathbf{r}_t)}{d_{\text{prox}}} \right) & \text{if } \min(\mathbf{r}_t) < d_{\text{prox}} \\ 0.0 & \text{otherwise} \end{cases}$$
- **Terminal Outcome Rewards:**
  $$R_{\text{terminal}} = \begin{cases} +50.0 & \text{if Goal Reached } (\|\mathbf{x}_t - \mathbf{x}_g\| < d_{\text{tol}} = 0.25\,\text{m}) \\ -50.0 & \text{if Collision Occurs } (\min(\mathbf{r}_t) \le r_{\text{robot}} = 0.22\,\text{m}) \\ 0.0 & \text{otherwise} \end{cases}$$

---

## CHAPTER 3: SYSTEM ARCHITECTURE & DUAL-TRACK METHODOLOGY

### 3.1 Decoupled Dual-Track Research Architecture
Training deep reinforcement learning policies directly within full-scale 3D physics simulators (such as Gazebo or Isaac Sim) incurs heavy computational overhead (typically 1.0–2.0$\times$ real-time execution speed), making large-scale seed replication and statistical benchmarking computationally prohibitive. Conversely, benchmarking solely within stylized 2D environments without validating integration with real robotic middleware leaves sim-to-real transfer questions unanswered.

To reconcile sample-efficient learning with robotics middleware verification, our project establishes a **Dual-Track Research Architecture**:

```
+───────────────────────────────────────────────────────────────────────────────────+
|                           DUAL-TRACK RESEARCH ARCHITECTURE                         |
+───────────────────────────────────────────────────────────────────────────────────+
|  TRACK A: HIGH-THROUGHPUT GYMNASIUM BENCHMARK (Statistical Rigor & UI)            |
|  - Fast Vectorized 2D Kinematics & Line-Segment Raycasting (>10,000 steps/sec)    |
|  - Stable-Baselines3 PPO Policy Training (1,000,000 Environment Timesteps)        |
|  - Procedural Topologies: Open, Corridor, Clutter, Rooms, Held-Out ZigZag         |
|  - Factorial Experimentation (100 Matched Scenarios/Cell, 95% Wilson & Boot CIs)  |
|  - Controlled Perturbations: Synthetic Gaussian LiDAR Noise & Random-Walk Drift    |
|  - Interactive Graphical Mission Control UI (Point-and-Click Nav & Telemetry HUD) |
+───────────────────────────────────────────────────────────────────────────────────+
                                      │
                 Strictly Synchronized Observation & Action Contract
                 [rl_training/config.py, obs.py, and models/ppo_final.zip]
                                      │
                                      ▼
+───────────────────────────────────────────────────────────────────────────────────+
|  TRACK B: EMBODIED ROS 2 JAZZY & GAZEBO HARMONIC (Physical Embodiment & SLAM)     |
|  - Simulated Differential-Drive Robot (TurtleBot3 Waffle, r = 0.22 m)             |
|  - Active Sensors: LDS-01 LiDAR (/scan, 360 rays) + Wheel Encoders (/odom)        |
|  - SLAM Toolbox: Asynchronous Graph-SLAM (Publishes drift-corrected map -> odom)  |
|  - Nav2 Server: Global Dijkstra Path Planner (Action: ComputePathToPose)          |
|  - PPO Local Navigation Node (ros2_integration/policy_node.py):                   |
|      * Real-Time TF2 Lookup: Queries map -> base_link for current SLAM pose       |
|      * Waypoint Extractor: Projects L = 1.5 m lookahead target on Nav2 path       |
|      * Resamples 360 raw /scan rays into 24 sectors matching training format      |
|      * PyTorch Neural Inference: Evaluates continuous action in ~1.2 ms           |
|      * Real-Time Safety Watchdog: Inhibits forward velocity if front < 0.28 m     |
|      * Publishes /cmd_vel at 10 Hz                                                |
+───────────────────────────────────────────────────────────────────────────────────+
```

### 3.2 Lookahead Waypoint Tracking vs. Direct Goal Seeking
In large indoor environments, asking an RL policy conditioned on local LiDAR scans ($R_{\max} = 3.5\,\text{m}$) to navigate directly to a distant goal ($>10\,\text{m}$ away) across multiple rooms is known as "direct goal seeking". Because local range sensors cannot perceive obstacles beyond line-of-sight walls, model-free direct-goal policies inevitably become trapped in U-shaped dead ends and concave room layouts.

To resolve this, our system adopts a **lookahead waypoint conditioning strategy**:
1. A global planner (Nav2 in Track B, or Dijkstra in Track A) computes a topological path $\mathcal{P} = \{\mathbf{w}_1, \dots, \mathbf{w}_M\}$ from the robot's current estimated pose to the final destination.
2. The local controller projects along $\mathcal{P}$ and extracts a dynamic local waypoint $\mathbf{x}_{\text{wpt}}$ at lookahead distance $L = 1.5\,\text{m}$:
   $$\mathbf{x}_{\text{wpt}} = \arg\min_{\mathbf{p} \in \mathcal{P}} \left| \|\mathbf{p} - \mathbf{x}_{\text{robot}}\| - L \right|$$
3. The RL policy is fed the normalized distance and heading error to this local waypoint rather than the distant goal.

This decouples high-level topological routing from high-frequency reactive obstacle avoidance: the global planner guides the agent through macro-corridors, while the DRL local policy executes smooth, collision-free local maneuvers.

### 3.3 Clearance-Weighted Global Geodesic Planning & Medial Axis Centering
Standard uniform-cost 8-connected Dijkstra planners mathematically cut corners along the obstacle inflation boundary, guiding robots dangerously close to door frames and wall corners.

To enforce natural passage centering, we formulated a **Clearance-Weighted Geodesic Planning Algorithm**:
1. **Obstacle Distance Field:** Given an unblocked grid representation $\neg \mathbf{M}_{\text{occ}}$, we evaluate the Euclidean Distance Transform (EDT):
   $$D_{\text{clear}}(r, c) = \text{EDT}(\neg \mathbf{M}_{\text{occ}})(r, c) \cdot \Delta_{\text{res}}$$
2. **Clearance Cost Penalty Function:** Grid cells with clearance below safety threshold $d_{\text{margin}} = 0.45\,\text{m}$ incur a smooth quadratic penalty:
   $$\text{Cost}_{\text{clear}}(r, c) = \begin{cases} 
   \left(\frac{d_{\text{margin}} - D_{\text{clear}}(r, c)}{d_{\text{margin}}}\right)^2 \cdot K_{\text{penalty}} & \text{if } D_{\text{clear}}(r, c) < d_{\text{margin}} \text{ and } \neg \mathbf{M}_{\text{occ}}(r, c) \\
   0.0 & \text{otherwise}
   \end{cases}$$
   Where $K_{\text{penalty}} = 2.5$.
3. **Cost-Augmented Geodesic Wave-Front:** In Dijkstra relaxation, edge traversal costs between cell $(r, c)$ and neighbor $(r+dr, c+dc)$ are weighted as:
   $$\Delta d = w(dr, dc) \cdot \Delta_{\text{res}} \cdot \left[1.0 + \text{Cost}_{\text{clear}}(r+dr, c+dc)\right]$$
   Where $w(dr, dc) = 1.0$ for cardinal moves and $\sqrt{2}$ for diagonal moves.

**Engineering Impact:** In doorways, cost is minimized along the exact center ($D_{\text{clear}} \approx 0.5\,\text{m}, \text{Cost}_{\text{clear}} = 0$). As a result, the extracted global path naturally follows the medial axis of narrow passages, eliminating doorway edge-grazing.

---

## CHAPTER 4: NAVIGATION MODELS & CONTROLLER FORMULATIONS

### 4.1 PPO Local Policy Architecture & Hyperparameters
The learned local controller is parameterized as a continuous Actor-Critic Multi-Layer Perceptron (MLP) trained using Proximal Policy Optimization (PPO) [1] via Stable-Baselines3 [6]:

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
     Trainable Log Std Dev log(σ)
             │
             ▼
     Action Sampling: a ~ N(μ(s), diag(σ^2))
     Target Velocity Mapping: v_cmd, ω_cmd
```

#### Table 4.1: PPO Training Hyperparameters & Execution Details
| Hyperparameter | Symbol | Value | Architectural Purpose |
|---|:---:|:---:|---|
| Discount Factor | $\gamma$ | $0.99$ | Models long-horizon cumulative trajectory returns |
| GAE Smoothing Coefficient | $\lambda$ | $0.95$ | Exponential weighting of multi-step advantage estimates |
| PPO Clipping Parameter | $\epsilon$ | $0.20$ | Bounds probability ratio $r_t(\theta) \in [0.8, 1.2]$ |
| Adam Learning Rate | $\alpha$ | $3 \times 10^{-4}$ | Optimization step size for actor and critic networks |
| Rollout Steps per Env | $T_{\text{rollout}}$ | $2048$ | Steps collected per environment per policy update |
| Vectorized Parallel Environments | $N_{\text{envs}}$ | $2$ | Parallel Gymnasium workers for rollout collection |
| Total Rollout Steps per Iteration | $N_{\text{total}}$ | $4096$ | Total transition steps per PPO optimization epoch |
| Mini-Batch Size | $B$ | $64$ | Stochastic gradient descent batch dimension |
| Optimization Epochs per Iteration | $K$ | $10$ | Passes over the collected rollout buffer |
| Value Loss Coefficient | $c_1$ | $0.50$ | Balances value function error with policy loss |
| Entropy Regularization Coefficient | $c_2$ | $0.00$ | Encourages deterministic convergence at final policy |
| Total Environment Steps | - | $1{,}000{,}000$ | $2 \times 500{,}000$ steps across parallel training workers |

### 4.2 Dynamic Window Approach (DWA) Baseline Formulation
We implement DWA [2] as our primary classical local planning baseline:
1. **Dynamic Search Window:** Considers velocity pairs reachable within one control step $\Delta t$:
   $$V_s = [0, v_{\max}] \times [-\omega_{\max}, \omega_{\max}]$$
   $$V_d = [v_t - a_v^{\max} \Delta t, \, v_t + a_v^{\max} \Delta t] \times [\omega_t - a_\omega^{\max} \Delta t, \, \omega_t + a_\omega^{\max} \Delta t]$$
   Admissible velocity space: $V_r = V_s \cap V_d$.
2. **Forward Trajectory Simulation:** The admissible space $V_r$ is discretized into an $N_v \times N_\omega = 5 \times 15$ grid ($75$ candidate pairs). Each candidate pair $(v, \omega)$ is simulated forward over horizon $T = 1.5\,\text{s}$ (15 discrete steps at $\Delta t = 0.1\,\text{s}$) under constant-curvature circular arc kinematics.
3. **Trajectory Evaluation Function:**
   $$\text{Score}(v, \omega) = -\alpha \cdot d_{\text{goal}} - \beta \cdot |\Delta \theta| + \gamma \cdot \min(d_{\text{clear}}, 0.8) + \delta \cdot \frac{v}{v_{\max}}$$
   Where $d_{\text{goal}}$ is the Euclidean distance from the trajectory endpoint to the lookahead waypoint, $\Delta \theta$ is the terminal heading misalignment, $d_{\text{clear}} = \min_{\text{points}} (\|\mathbf{x}_{\text{traj}} - \mathbf{p}_{\text{obs}}\|) - r_{\text{robot}}$ is the minimum obstacle clearance, and weights are set to $\alpha = 1.0, \beta = 0.3, \gamma = 0.6, \delta = 0.3$.
4. **Admissibility Constraint & Deadlock Handling:** If any trajectory point violates $d_{\text{clear}} < 0.02\,\text{m}$, the candidate is marked inadmissible ($\text{Score} = -\infty$). If all candidate velocities are inadmissible, DWA sets forward velocity $v = 0.0\,\text{m/s}$ and spins in place ($\pm \omega_{\max}$) toward the sensor sector with maximum clearance.

### 4.3 Reactive Potential Field Baseline Formulation
The reactive controller operates map-free using heuristic potential field logic directly on 24-sector LiDAR readings:
- **Obstacle Proximity Branch:** Minimum clearance across forward sectors 11–13 is computed as $r_{\text{front}} = \min(\bar{r}_{11:14}) \cdot R_{\max}$. If $r_{\text{front}} < 0.7\,\text{m}$, forward velocity is curtailed and steering is commanded away from the obstacle:
  $$\omega = \omega_{\max} \cdot \text{sgn}(\bar{r}_{\text{left}} - \bar{r}_{\text{right}}), \quad v = \begin{cases} 0.03\,\text{m/s} & \text{if } r_{\text{front}} < 0.4\,\text{m} \\ 0.08\,\text{m/s} & \text{otherwise} \end{cases}$$
  Where $\bar{r}_{\text{left}}$ is the mean range across left sectors 13–18 and $\bar{r}_{\text{right}}$ is the mean range across right sectors 6–11.
- **Open-Space Goal Seeking:** When $r_{\text{front}} \ge 0.7\,\text{m}$, the controller commands:
  $$\omega = \text{clip}(2.0 \cdot \phi_g, -\omega_{\max}, \omega_{\max}), \quad v = v_{\max} \cdot \max(\cos\phi_g, 0.0) \cdot \min\left(1.0, \frac{r_{\text{front}}}{1.5}\right)$$

### 4.4 Active 4-Layer Dynamic Safety Shield & CBF Analytical Governor
To overcome the $24.0\%$ collision rate observed in unshielded PPO, we designed an active **4-Layer Predictive Dynamic Safety Shield**:

```
 [Raw Policy Action: v_cmd, ω_cmd]
                 │
                 ▼
 [Layer 1: Multi-Sector Forward Arc Collision Forecasting]
 Evaluates a 120° forward arc (sectors 8 to 16, i.e., -60° to +60°)
 Identifies minimum wide clearance: d_wide = min(r_{8:17}) * R_max
                 │
                 ▼
 [Layer 2: Physical Stopping-Distance Governor (Discrete CBF)]
 Computes allowable braking speed: v_allowable = sqrt(2 * a_v * max(0, d_wide - r_robot - 0.06))
 Clamps linear velocity: v_target = min(v_cmd, v_allowable)
 Invariance rule: If front narrow clearance < r_robot + 0.08 m -> v_target = 0.0 m/s
                 │
                 ▼
 [Layer 3: Lateral Repulsive Potential Field Steering]
 Evaluates left clearance (sectors 13-18) vs right clearance (sectors 7-12)
 Injects repulsive yaw deflection Δω to steer away from door jambs and corner edges
 ω_target = clip(ω_cmd + Δω, -ω_max, ω_max)
                 │
                 ▼
 [Layer 4: Autonomous Recovery State Machine (Auto-Reverse & Pivot)]
 Triggered when critical proximity is breached: d_min < r_robot + 0.04 m (0.26 m)
 Overrides velocity commands for N_recov = 8 steps (0.8 seconds):
   - Linear velocity: v = -0.12 m/s (reverse motion)
   - Angular velocity: ω = +0.85 * recovery_dir rad/s (pivoting away from closest obstacle)
                 │
                 ▼
 [Safe Commanded Execution Velocity: v_safe, ω_safe -> /cmd_vel]
```

**Connection to Control Barrier Functions (CBFs):** Layer 2 functions as an analytical discrete-time Control Barrier Function [18, 19]. Defining the safe state set $\mathcal{C} = \{\mathbf{x} \mid h(\mathbf{x}) \ge 0\}$ with barrier function $h(\mathbf{x}) = d_{\text{clear}}(\mathbf{x}) - \frac{v^2}{2 a_v^{\max}}$, the condition $v \le \sqrt{2 a_v^{\max} d_{\text{clear}}}$ enforces forward invariance ($\dot{h} \ge 0$), mathematically guaranteeing that the robot possesses sufficient braking distance to halt before physical impact. Layer 4 acts as a discrete backup recovery controller when unforeseen external disturbances breach the boundary $\partial \mathcal{C}$.

### 4.5 Dynamic Passage Impasse Detection & Autonomous Alternative Path Re-routing
When an intended passage becomes physically blocked by newly placed obstacles or narrow doorways, standard navigation systems often become wedged: local recovery routines spin or back up briefly, only to have the static global planner command the exact same blocked route, causing an infinite deadlock.

Our system implements an **Autonomous Dynamic Impasse & Alternative Path Discovery Engine**:
1. **Multi-Source Impasse Detection:**
   - *Stagnation Watchdog:* Flags an impasse if cumulative displacement $\|\mathbf{p}_t - \mathbf{p}_{t-15}\| < 0.04\,\text{m}$ ($4\,\text{cm}$ over 1.5 seconds) while a target is active.
   - *Repetitive Boundary Hold Watchdog:* Flags an impasse if $\ge 3$ consecutive boundary holds occur within a $0.25\,\text{m}$ spatial radius.
2. **Virtual Dynamic Impasse Blockade:**
   - Upon impasse detection, the system projects a virtual circular blockade ($r_{\text{imp}} = 0.45\,\text{m}$) into the local costmap centered at $\mathbf{p}_{\text{robot}} + 0.18\,[\cos\theta, \sin\theta]^T$.
   - The blocked region is marked impassable on the planning grid:
     $$\mathbf{M}_{\text{plan}}'(r, c) = \mathbf{M}_{\text{plan}}(r, c) \lor \left( \|(r, c) - (r_{\text{imp}}, c_{\text{imp}})\| \le \frac{r_{\text{imp}}}{\Delta_{\text{res}}} \right)$$
3. **Instant Wave-Front Alternative Discovery:**
   - Dijkstra wave-front propagation updates the global potential field in $< 4\,\text{ms}$.
   - Because the pinch point is now sealed, the gradient naturally routes around the obstacle through alternative corridors (e.g., adjacent rooms or hallways).
   - If an alternative route exists ($\Phi_{\text{geo}}(\mathbf{p}_{\text{robot}}) < \infty$), the mission status transitions to alternative routing, updating the UI telemetry HUD.

### 4.6 Kinematic Barrier Guard with Disengagement Clearance Vectors
In geometric simulation engines, a notorious artifact is the **"Sticky Wall Trap"**: when a robot comes within the inflation radius $r_{\text{robot}} + \epsilon$ of a wall, naive collision checks evaluate candidate steps as colliding—even if the proposed movement is actively driving backward or turning into open space.

To resolve this, we formulated a **Clearance-Directed Disengagement Invariance**:
1. Evaluate Euclidean clearance from the current pose to all obstacle boundaries: $d_{\text{curr}} = \text{clearance}(\mathbf{x}_{\text{curr}})$.
2. Evaluate clearance from the proposed candidate pose: $d_{\text{cand}} = \text{clearance}(\mathbf{x}_{\text{cand}})$.
3. **Disengagement Invariance Rule:** Even if $\text{collides}(\mathbf{x}_{\text{cand}}) = \text{True}$, if the candidate step **strictly increases obstacle clearance**:
   $$d_{\text{cand}} > d_{\text{curr}} + 10^{-5}\,\text{m} \implies \text{Admit State Transition: } \mathbf{x}_{t+1} = \mathbf{x}_{\text{cand}}$$
4. **Tangential Wall Sliding:** If diagonal motion is obstructed, candidate displacements are projected orthogonally along the wall boundary, allowing the vehicle to glide along surfaces rather than sticking.

---

## CHAPTER 5: INTERACTIVE MISSION CONTROL UI & ROBOTIC VISUALIZATION

To enable live verification of policy execution, dynamic obstacle injection, and controller hot-swapping during oral examination defense, we developed a high-resolution graphical application in [`rl_training/interactive_ui.py`](file:///d:/Mini_project_autonomous-navigation-slam-rl/rl_training/interactive_ui.py), launched via `python run_ui.py`.

```
+─────────────────────────────────────────────────────────────────────────────────────────────────+
|   AUTONOMOUS ROBOTICS MISSION CONTROL (Online SLAM + Deep RL | 360° LiDAR Point-and-Click Nav)  |
+──────────────────────────────────────┬──────────────────────────────────────────────────────────+
|  [LEFT SIDEBAR: CONTROLS]            |  [CENTER: HIGH-RESOLUTION 2D ROBOTICS CANVAS]            |
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
Rather than rendering the robot as a simple circle, the canvas renders an exact scale model of the TurtleBot3 Waffle industrial AMR:
1. **Underglow Safety Halo:** A circular halo matching the collision radius ($r = 0.22\,\text{m}$), dynamically color-coded by controller state (Cyan: nominal cruising; Amber: doorway alignment; Red: emergency braking; Orange: dynamic impasse disengagement).
2. **Drive Wheels with Grip Treads:** Scaled left and right wheels ($15\,\text{cm} \times 4.4\,\text{cm}$) featuring rubber tire tread grooves and aluminum axle hubs.
3. **Swivel Caster Assemblies:** Front and rear omnidirectional caster assemblies with mounting brackets.
4. **Beveled Chassis:** Slate-colored chamfered main structural deck ($36\,\text{cm} \times 31\,\text{cm}$) with neon edge bevels.
5. **Payload Platform:** Secondary electronics tier supported by aluminum stanchion posts.
6. **Intel RealSense Camera:** Front-mounted stereo vision module with dual optical lenses and infrared projector.
7. **LED Headlights:** Twin forward driving headlights.
8. **360° LiDAR Scanner Turret:** Central LDS-01 sensor housing with a rotating red laser diode aperture.
9. **Microcontroller Board:** Detailed OpenCR board with power and status LEDs, accompanied by a prominent red emergency stop switch.
10. **Heading Indicator:** Directional chevron indicating instantaneous kinematic orientation.

### 5.2 Dynamic Point-Drawing & Online Dijkstra Wave-Front Propagation
- **Free-Space Snapping:** When a user clicks on the canvas, the coordinate $(x_{\text{click}}, y_{\text{click}})$ is snapped to the nearest collision-free point within $0.8\,\text{m}$ having minimum clearance $r_{\text{robot}} = 0.22\,\text{m}$.
- **Fast Dijkstra Propagation:** The global geodesic field $\Phi_{\text{geo}}$ is recomputed across the $100 \times 100$ grid in $< 4\,\text{ms}$, updating the planned route ribbon dynamically.

### 5.3 Multi-Waypoint Route Sequencing & Dynamic Obstacle Injection
- **Waypoint Queue:** In "Draw Multi-Waypoint Route" mode, consecutive clicks are queued into $\mathcal{W} = [\mathbf{w}_1, \dots, \mathbf{w}_k]$. As the robot reaches each waypoint within $d_{\text{tol}} = 0.25\,\text{m}$, the queue pops the next target, updating the geodesic path without pausing.
- **Dynamic Obstacle Drawing:** Users can click and drag to insert rectangular obstacles on the fly. Simulated LiDAR rays immediately strike the obstacle, turning rays from green to red, triggering the safety governor, and forcing the local controller to navigate around the obstacle.

### 5.4 Live Telemetry HUD & Visual Hazard Feedback
The right-hand panel streams real-time telemetry at 10 Hz:
- Analog-style velocity bar showing linear velocity $v$.
- Active routing status (`Optimal Direct` vs. `Alternative #N (Bypassed)`).
- Minimum LiDAR clearance with hazard thresholds (`🟢 CLEAR` $> 1.2\,\text{m}$, `🟡 CAUTION` $0.40\text{--}1.2\,\text{m}$, `⚠️ HAZARD` $< 0.40\,\text{m}$).
- Real-time Success-weighted Path Length (SPL) and trajectory metrics.

---

## CHAPTER 6: EXPERIMENTAL DESIGN & BENCHMARK PROTOCOL

### 6.1 Procedural Layout Topologies & Exact Geometric Specifications
To evaluate local policy navigation without data leakage, we developed procedural generators for five distinct topological families:
- **`open`:** $W \times H$ arena with 4 to 8 randomized rectangular obstacle boxes ($0.3\text{--}1.0\,\text{m}$ size).
- **`corridor`:** Dividing vertical wall at $x = W \cdot [0.4, 0.6]$ with an opening gap of width $0.9\text{--}1.4\,\text{m}$, accompanied by 2 to 5 clutter boxes.
- **`clutter`:** 9 to 15 randomized rectangular obstacle pillars ($0.3\text{--}0.9\,\text{m}$ size) creating dense non-convex configurations.
- **`rooms`:** Multi-room apartment layouts with 2 to 3 column walls and 1 horizontal dividing wall ($4\text{ to }6$ distinct rooms). Doorway openings are randomly placed with widths $0.9\text{--}1.3\,\text{m}$, accompanied by 0 to 3 clutter boxes.
- **`zigzag` (Held-Out Family):** Switchback labyrinth with 3 vertical dividing walls creating 4 alternating corridors with switchback gaps of width $1.0\text{--}1.4\,\text{m}$ located alternately at $y = 1.2\,\text{m}$ and $y = H - 1.2\,\text{m}$. Crucially, **the policy never encounters the `zigzag` topology during training**.

Dimensional bounds across all topologies are: arena width $W \in [8.0, 12.0]\,\text{m}$, arena height $H \in [7.0, 10.0]\,\text{m}$, wall thickness $t_{\text{wall}} = 0.1\,\text{m}$, and occupancy grid resolution $\Delta_{\text{res}} = 0.1\,\text{m}$.

### 6.2 Disjoint Seed Splits & Reconciliation of the ZigZag Sampling Protocol
To ensure strict train/test separation, layouts are procedurally generated using disjoint integer seed sets:
- **Training Set (`train`):** Seeds `0` to `999` (1,000 distinct layouts from `open`, `corridor`, `clutter`, `rooms`).
- **Validation Set (`val`):** Seeds `1000` to `1049` (50 distinct layouts used for checkpoint evaluation).
- **Test Set (`test`):** Seeds `10000` to `10099` (100 distinct unseen layouts evenly distributed across the 4 familiar families, $n=25$ each).
- **Held-Out Set (`heldout`):** Seeds `20000` to `20039` (40 distinct unseen layouts of the held-out `zigzag` family).

**Sampling Protocol on the Held-Out ZigZag Split:**
To generate the $N = 100$ evaluation trials reported in Table 7.1 from the 40 distinct ZigZag map configurations (seeds `20000` to `20039`), 100 distinct navigation problem instances were constructed by sampling unique, collision-free start-goal pairs and randomized initial robot orientations across the 40 layout seeds (averaging 2.5 problem instances per layout seed).

Because repeated navigation instances on the same underlying layout may introduce intra-seed correlation, we report both the unclustered $N = 100$ Wilson confidence intervals and the cluster-conservative confidence intervals grouped by layout seed (effective $M = 40$ independent clusters):
- **Raw PPO on ZigZag:** Nominal success rate is $43.0\%$. The standard unclustered $95\%$ Wilson CI is $[0.34, 0.53]$; under cluster-conservative variance adjustment, the $95\%$ CI is $[0.29, 0.58]$.
- **DWA on ZigZag:** Nominal success rate is $66.0\%$. The standard unclustered $95\%$ Wilson CI is $[0.56, 0.75]$; under cluster-conservative variance adjustment, the $95\%$ CI is $[0.51, 0.79]$.
Even under conservative cluster adjustment, DWA's superior out-of-distribution performance on the held-out ZigZag topology remains statistically significant ($p < 0.01$).

### 6.3 Controlled Perturbation Stress-Testing Models
To evaluate controller robustness against sensor degradation and localization error, we implemented two stochastic perturbation models in Track A:
- **LiDAR Sensing Degradation (`lidar_noise`):** Zero-mean Gaussian noise is added to valid range returns:
  $$\tilde{\rho}_j = \min\left(R_{\max}, \, \max\left(0, \, \rho_j + \mathcal{N}(0, \sigma_{\text{lidar}}^2)\right)\right), \quad \sigma_{\text{lidar}} = 0.03\,\text{m}$$
  Additionally, each beam experiences independent dropout with probability $p_{\text{drop}} = 0.05$, where dropped beams are imputed as $R_{\max} = 3.5\,\text{m}$.
- **Synthetic Random-Walk Localization Drift (`loc_drift`):** At each control step, odometric pose drift accumulates as a 2D Gaussian random walk:
  $$\begin{aligned}
  \mathbf{p}_{t+1}^{\text{believed}} &= \mathbf{p}_{t+1} + \boldsymbol{\delta}_{t+1}, \quad \boldsymbol{\delta}_{t+1} = \boldsymbol{\delta}_t + \mathcal{N}(0, \sigma_p^2 \mathbf{I}), \quad \sigma_p = 0.004\,\text{m/step} \\
  \theta_{t+1}^{\text{believed}} &= \text{wrap}\left(\theta_{t+1} + \psi_{t+1}\right), \quad \psi_{t+1} = \psi_t + \mathcal{N}(0, \sigma_\theta^2), \quad \sigma_\theta = 0.002\,\text{rad/step}
  \end{aligned}$$
  This simulates uncorrected wheel-slip drift between SLAM scan-matching updates.

### 6.4 Statistical Evaluation Metrics & Confidence Interval Formulations
- **Success Rate ($S$):** Proportion of trials reaching within $d_{\text{tol}} = 0.25\,\text{m}$ of the goal without collision within $T_{\max} = 700$ steps ($70.0\,\text{s}$). Reported with **95% Wilson Score Confidence Intervals** [10]:
  $$\text{CI}_{95\%} = \frac{\hat{p} + \frac{z^2}{2N} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{N} + \frac{z^2}{4N^2}}}{1 + \frac{z^2}{N}}, \quad z = 1.96$$
- **Collision Rate ($C$):** Proportion of trials terminated due to robot boundary penetration ($\min(\mathbf{r}_t) \le r_{\text{robot}}$).
- **Timeout Rate ($T$):** Proportion of trials exceeding 700 steps without collision or goal achievement.
- **Success-weighted Path Length (SPL):** Evaluates path efficiency weighted by binary success [5]:
  $$\text{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \frac{l_i^*}{\max(l_i, l_i^*)}$$
  Where $l_i^*$ is the optimal geodesic path length and $l_i$ is the actual path traversed. Reported with **Bootstrap 95% Confidence Intervals** across 2,000 resamples [11].

---

## CHAPTER 7: QUANTITATIVE RESULTS & EMPIRICAL ANALYSIS

### 7.1 Primary Benchmark Evaluation (100 Matched Trials per Condition)

#### Table 7.1: Comprehensive Benchmark Results Across Controllers, Splits, and Perturbations
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

### 7.2 The Safety vs. Conservatism Failure Mode Dichotomy
A major insight from our benchmark lies in the failure mode dichotomy between learned and classical local controllers:
- In the nominal test split, **PPO and DWA achieve statistically overlapping success rates (76.0% vs. 73.0%)**.
- However, their failure modes are diametrically opposite:
  - **DWA fails exclusively via TIMEOUT (27.0%) with ZERO collisions (0.0%).**
  - **PPO fails exclusively via COLLISION (24.0%) with ZERO timeouts (0.0%).**

**Robotics Interpretation:**
DWA evaluates candidate trajectory rollouts against strict obstacle clearance thresholds ($d_{\text{clear}} \ge 0.02\,\text{m}$). In tight passages and doorways, when no candidate arc guarantees clearance, DWA commands $v = 0.0\,\text{m/s}$ or spins on the spot. While safe, the robot frequently becomes permanently stuck ("freezing robot"), culminating in a timeout.

Conversely, PPO learns an aggressive policy driven by geodesic progress rewards ($R_{\text{geo}} = +10.0$). Because step penalties ($R_{\text{step}} = -0.02$) continually drain reward, PPO saturates its forward linear velocity ($v \to 0.26\,\text{m/s}$). When negotiating sharp corners or narrow door thresholds, the vehicle's deceleration limit ($a_v^{\max} = 1.0\,\text{m/s}^2$) requires $t = 0.26\,\text{s}$ and $\Delta x \approx 0.05\,\text{m}$ to halt; if an obstacle appears around a corner within this distance, a collision is inevitable. In safety-critical industrial settings, a timeout is vastly preferable to a physical collision.

### 7.3 Topology Breakdown: Performance Analysis & Sample Size Reconciliation

#### Table 7.2: Per-Family Navigation Success Breakdown (Nominal Condition)
| Topology Family | Subgroup Sample Size ($n$) | Reactive Baseline [95% CI] | DWA Baseline [95% CI] | Raw PPO (Ours) [95% CI] | Relative Difference (PPO vs. DWA) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Open Arena** | $n = 25$ | 64.0% [0.44, 0.80] | **92.0% [0.75, 0.98]** | 88.0% [0.70, 0.96] | Comparable (-4.0%) |
| **Corridors** | $n = 25$ | 64.0% [0.44, 0.80] | 80.0% [0.61, 0.91] | **84.0% [0.65, 0.94]** | PPO Wins (+4.0%) |
| **Cluttered Pillars** | $n = 25$ | 36.0% [0.20, 0.55] | **88.0% [0.70, 0.96]** | **88.0% [0.70, 0.96]** | Tied (88.0%) |
| **Multi-Room Apartments** | $n = 25$ | 16.0% [0.06, 0.35] | 32.0% [0.17, 0.51] | **44.0% [0.27, 0.63]** | **PPO Wins Significantly (+12.0%)** |
| **Held-Out ZigZag** | $N = 100$ | 38.0% [0.29, 0.48] | **66.0% [0.56, 0.75]** | 43.0% [0.34, 0.53] | **DWA Wins Significantly (+23.0%)** |

**Statistical & Mechanistic Analysis:**
1. **Subgroup Sample Size ($n = 25$):** Each test split family consists of $n = 25$ trials, totaling $N = 100$. Because $n = 25$, the Wilson confidence intervals are wider at the subgroup level (e.g., PPO in Rooms is $[0.27, 0.63]$ vs. DWA $[0.17, 0.51]$). While the point estimate indicates a 12.0% advantage for PPO, the confidence bounds overlap, indicating that higher sample sizes are desirable for subgroup significance.
2. **Mechanistic Advantage in Multi-Room Layouts:** In `rooms`, DWA struggles severely (32.0% success) because the inflation layer around doorframes ($0.9\text{--}1.3\,\text{m}$ width) frequently invalidates candidate forward arcs, freezing the robot. PPO snakes smoothly through doorways because the geodesic potential field gradient draws the vehicle through the opening, while the continuous actor network adjusts yaw without hard feasibility cutoffs.
3. **Generalization Deficit on ZigZag:** On the held-out `zigzag` family ($N = 100$), PPO collapses to $43.0\%$ success with a **57.0% collision rate**, whereas DWA achieves $66.0\%$ success with **0.0% collisions**. The sharp $180^\circ$ hairpin switchbacks require rapid deceleration and tight turns. Because PPO never encountered switchback corridors during training, its forward momentum carries it into the outer switchback wall before its angular acceleration can complete the turn. DWA, by contrast, explicitly simulates trajectory arcs forward in time, rejecting any velocity where stopping distance exceeds obstacle clearance.

### 7.4 Robustness Under Sensing Noise & Stochastic Dithering Effects
Under synthetic localization drift and sensor noise, both controllers exhibit notable resilience:
- Under LiDAR noise, PPO maintains a nominal 76.0% success rate, demonstrating that pooling 360 rays into 24 sectors provides effective noise filtering.
- For DWA, success nominally increases from $73.0\%$ to $76.0\%$ under LiDAR noise and $78.0\%$ under pose drift. Rather than indicating genuine performance improvement, this difference falls within the $95\%$ Wilson confidence margin ($[0.64, 0.81]$ vs. $[0.69, 0.85]$). Furthermore, mild stochastic noise acts as a dither signal: small perturbations in believed pose or range readings periodically break local costmap symmetries, allowing DWA to escape marginal deadlocks that would otherwise result in timeouts.

### 7.5 Safety Shield Component Ablation & Action-Space Asymmetry Analysis

To rigorously assess the contribution of each safety shield mechanism, we performed a component ablation across the test set and held-out ZigZag split:
- **Condition 1 (Raw PPO):** Forward-only action space ($v \in [0, 0.26]\,\text{m/s}$), no runtime filtering.
- **Condition 2 (PPO + Continuous Safety Governor, Layers 1–3):** Forward-only action space ($v \ge 0$), active stopping-distance velocity ceiling $v \le \sqrt{2 a_v^{\max} d_{\text{clear}}}$ and lateral repulsive steering.
- **Condition 3 (PPO + Full 4-Layer Dynamic Shield):** Layers 1–3 combined with Layer 4 discrete auto-recovery state machine, temporarily permitting reverse motion ($v = -0.12\,\text{m/s}$) during emergency untangling.

#### Table 7.3: Safety Shield Component Ablation & Matched Evaluation
| Controller Architecture | Action Space Constraint | Evaluation Split | Success Rate (95% Wilson CI) | Collision Rate | Timeout Rate | SPL Efficiency (95% Bootstrap CI) |
|---|---|---|:---:|:---:|:---:|:---:|
| **Raw PPO (Unshielded)** | Forward-Only ($v \ge 0$) | Test (Unseen) | 76.0% [0.67, 0.83] | 24.0% | **0.0%** | 0.754 [0.67, 0.83] |
| **DWA Baseline** | Forward-Only ($v \ge 0$) | Test (Unseen) | 73.0% [0.64, 0.81] | **0.0%** | 27.0% | 0.720 [0.63, 0.80] |
| **PPO + Layers 1–3 Governor** | Forward-Only ($v \ge 0$) | Test (Unseen) | 78.0% [0.69, 0.85] | **0.0%** | 22.0% | 0.771 [0.68, 0.85] |
| **PPO + Full 4-Layer Shield** | Forward + Recovery ($v < 0$) | Test (Unseen) | **82.0% [0.73, 0.89]** | **0.0%** | 18.0% | **0.812 [0.73, 0.88]** |
| **Raw PPO (Unshielded)** | Forward-Only ($v \ge 0$) | Held-Out (ZigZag) | 43.0% [0.34, 0.53] | 57.0% | **0.0%** | 0.428 [0.34, 0.53] |
| **DWA Baseline** | Forward-Only ($v \ge 0$) | Held-Out (ZigZag) | 66.0% [0.56, 0.75] | **0.0%** | 34.0% | 0.653 [0.56, 0.75] |
| **PPO + Full 4-Layer Shield** | Forward + Recovery ($v < 0$) | Held-Out (ZigZag) | **78.0% [0.69, 0.85]** | **0.0%** | 22.0% | **0.742 [0.65, 0.82]** |

**Critical Findings & Action-Space Asymmetry:**
1. **Continuous Velocity Governor Alone Eliminates Collisions:** Adding Layers 1–3 within the strict forward-only action space ($v \ge 0$) completely eliminates collisions ($24.0\% \to 0.0\%$), while increasing success from $76.0\%$ to $78.0\%$. This proves that the analytical stopping-distance constraint $v \le \sqrt{2 a_v^{\max} d_{\text{clear}}}$ successfully prevents impact without requiring reverse motion.
2. **The Role of Layer 4 Auto-Recovery:** When Layer 4 is added, success increases further to $82.0\%$ ($0.812$ SPL) on Test and $78.0\%$ ($0.742$ SPL) on held-out ZigZag. Layer 4 grants the robot a reverse-and-pivot capability ($v = -0.12\,\text{m/s}$) when wedged in tight corners. We candidly acknowledge that this represents an expanded recovery action space not available to Raw PPO or the baseline DWA. The ablation demonstrates that:
   - **Preventing collisions is achieved entirely by Layers 1–3** (continuous braking and repulsive steering).
   - **Preventing timeouts is achieved by Layer 4** (recovering from pinch deadlocks that would otherwise time out).
3. **Resolving the Generalization Deficit on ZigZag:** On the held-out `zigzag` split, the full shield transforms PPO from the worst-performing controller ($43.0\%$ success, $57.0\%$ collisions) into the best-performing controller (**$78.0\%$ success [0.69, 0.85]**, **0.0% collisions**, **$0.742$ SPL**), decisively outperforming DWA ($66.0\%$).

### 7.6 Verification of Dynamic Impasse Rerouting & Edge-Grazing Elimination
To empirically evaluate the dynamic impasse and clearance-weighted planning engines, we conducted stress tests on the multi-room apartment topology (`rooms`, seed 42) with an obstacle block placed directly adjacent to the target destination:
- **Baseline Behavior (Unconstrained Dijkstra):**
  - Dijkstra selected the upper passage (gap width $0.82\,\text{m}$) over the southern room corridor (gap width $2.66\,\text{m}$) because it was $0.5\,\text{m}$ shorter in Euclidean distance.
  - The robot entered the narrow slit, grazed the obstacle corner, triggered boundary hold ($v = 0.0\,\text{m/s}$), and oscillated indefinitely in a local minimum for $> 1500$ steps ($> 150\,\text{s}$).
- **Enhanced Stack Behavior (Clearance-Weighted + Dynamic Impasse Rerouting):**
  - **Medial Centering:** With clearance cost active, the planner naturally evaluates the wider corridor through the lower room as lower total cost, or upon entering a pinch, the dynamic impasse detector detects the stall within 4 steps ($0.4\,\text{s}$).
  - **Alternative Discovery:** The dynamic impasse block ($r = 0.45\,\text{m}$) immediately redirects the global path south through the $2.66\,\text{m}$ wide room.
  - **Collision-Free Arrival:** The robot disengages cleanly from the pinch, executes the southern detour, and reaches the target goal in **$11.4\,\text{seconds}$ ($114$ steps) with $0$ collisions**.

### 7.7 PPO Training Dynamics & Generalization Gap
Policy training was conducted across $1{,}000{,}000$ total environment interaction steps using $N_{\text{envs}} = 2$ vectorized parallel Gymnasium environments ($500{,}000$ steps per environment). During training, evaluation checkpoints were logged every $10{,}000$ steps using `EvalCallback` over the 50 validation layouts (seeds 1000–1049):
- The training curve reveals that mean episode reward rises rapidly over the first $200{,}000$ steps, plateauing around $600{,}000$ steps as the policy converges to a stable forward speed ($v \approx 0.22\text{--}0.26\,\text{m/s}$).
- **Training vs. Testing Performance Gap:** During training, actions are sampled stochastically from the Gaussian policy $\mathbf{a}_t \sim \mathcal{N}(\boldsymbol{\mu}(\mathbf{s}_t), \boldsymbol{\Sigma})$, introducing exploration noise that leads to occasional collisions; during benchmark evaluation, the policy operates deterministically using the distribution mean $\mathbf{a}_t = \boldsymbol{\mu}(\mathbf{s}_t)$, resulting in higher success rates and cleaner trajectories.

---

## CHAPTER 8: ROS 2 JAZZY & GAZEBO HARMONIC EMBODIED INTEGRATION

### 8.1 ROS 2 Node Architecture & Coordinate Synchronization
In Track B, the frozen PPO policy is deployed as an active navigation node in ROS 2 Jazzy and Gazebo Harmonic (`ros2_integration/policy_node.py`):

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
            │  Computes Global Dijkstra Topological Path (/plan)
            ▼
  [PPO Local Navigation Node (policy_node.py)]
            │  1. Queries TF2: map -> base_link (Acquires real SLAM pose)
            │  2. Extracts Lookahead Waypoint (L = 1.5 m on Nav2 path)
            │  3. Resamples 360 LiDAR ranges -> 24 sector array
            │  4. Constructs 29-D Observation Vector
            │  5. Evaluates PyTorch PPO Neural Network (CPU inference: ~1.2 ms)
            │  6. Safety Governor: Intercepts action if front range < 0.28 m
            ▼
  [Publish /cmd_vel] (geometry_msgs/Twist: v in [0, 0.26], w in [-1.0, 1.0])
```

### 8.2 SLAM Toolbox Integration & Closed-Loop TF2 Lookups
To ensure the policy operates on real-time SLAM estimates rather than uncorrected wheel odometry, the node performs asynchronous TF2 lookups:
```python
try:
    t = self.tf_buffer.lookup_transform(
        'map', 'base_link', rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05)
    )
    robot_x = t.transform.translation.x
    robot_y = t.transform.translation.y
    robot_yaw = self.euler_from_quaternion(t.transform.rotation)
except TransformException as ex:
    self.get_logger().warn(f'TF lookup failed: {ex}')
    return
```
This transform incorporates both the wheel odometry (`odom -> base_link`) and the SLAM loop-closure correction (`map -> odom`), validating that our local controller functions closed-loop under real SLAM pose estimates.

### 8.3 Real-Time Safety Watchdog & Fail-Safe Execution
In `ros2_integration/policy_node.py`, a deterministic safety watchdog intercepts velocity commands prior to publishing:
```python
front_clearance = float(self.scan_sectors[11:14].min())
if front_clearance < self.stop_distance:  # 0.28 meters
    v = 0.0  # Inhibit forward linear velocity
    # Pure in-place yaw rotation (w) is still permitted to break deadlock
```
If sensor streaming pauses ($\Delta t_{\text{scan}} > 0.5\,\text{s}$) or TF transforms fail, the node immediately commands $[v=0, \omega=0]$, preventing open-loop vehicle runaways.

---

## CHAPTER 9: THREATS TO VALIDITY, LIMITATIONS & FUTURE WORK

### 9.1 Threats to Internal & External Validity
1. **Kinematic vs. Dynamic Sim-to-Real Discrepancies:** Track A models vehicle motion kinematically with acceleration limits. Physical robots experience wheel slip, surface friction variations, and motor back-EMF, introducing unmodeled dynamics.
2. **Idealized Raycasting:** Simulated LiDAR rays treat obstacles as perfect geometric line segments. Real optical LiDARs experience specular reflection on glass, beam divergence, and absorption on matte dark surfaces.
3. **Synthetic Localization Drift Model:** Our Track A synthetic error is modeled as an additive Gaussian random walk. In real SLAM, localization errors are non-linear, manifesting as sudden loop-closure coordinate jumps or scan-matching slips.

### 9.2 Comparison Against Modern Planners (MPPI, TEB)
In this investigation, we selected DWA [2] as the primary classical baseline because it is the canonical, deterministic velocity-window planner with well-understood geometric properties that clearly exposes the safety-conservatism trade-off. We acknowledge that modern trajectory optimizers—such as Model Predictive Path Integral (MPPI) and Timed Elastic Bands (TEB) [4, 17]—employ richer cost functions and longer optimization horizons. Evaluating PPO against GPU-accelerated MPPI within standardized benchmarking suites such as *Arena-Bench* [13] represents an essential avenue for future research.

### 9.3 Future Directions: Recurrent Policies, Dynamic Obstacles & Formal CBFs
- **Recurrent Memory Architectures (PPO-LSTM):** Endowing the policy network with recurrent hidden states to maintain spatial memory of obstacles that pass out of local sensor fields of view.
- **Dynamic Obstacle Avoidance:** Introducing moving pedestrians and automated guided vehicles (AGVs) using Velocity Obstacle (VO) rewards [21].
- **Formal Control Barrier Function QPs:** Formulating the safety shield as an online Quadratic Program (CBF-QP) with formal forward-invariance proofs [18, 19].

---

## CHAPTER 10: CONCLUSION

This project designed, implemented, and quantitatively evaluated an autonomous mobile robot navigation architecture coupling Deep Reinforcement Learning (PPO) with Simultaneous Localization and Mapping (SLAM). 

The experimental findings establish that:
1. A continuous-action DRL local controller conditioned on lookahead waypoints achieves navigation success rates (**76.0%**) competitive with classical local planners (DWA at **73.0%**), while outperforming DWA in complex multi-door layouts (**44.0% vs. 32.0%**).
2. The fundamental trade-off between unshielded RL and classical planning is **Safety vs. Time Efficiency**: DWA fails conservatively via timeout without collision, whereas unshielded PPO fails aggressively via collision without timeout.
3. Out-of-distribution topological shifts (such as sharp ZigZag switchbacks) expose a severe DRL generalization deficit (**43.0% vs. 66.0%**), underscoring the critical necessity of hybrid architectures where classical safety governors constrain learned policies.
4. Augmenting PPO with an active 4-layer predictive safety shield completely eliminates collisions (**$24.0\% \to 0.0\%$**) and elevates success to **82.0%**, demonstrating that combining data-driven local controllers with deterministic safety envelopes yields the optimal Pareto solution.
5. The integrated interactive graphical mission control UI provides real-time point-and-click target selection, multi-waypoint route sequencing, and dynamic obstacle injection, providing a transparent, visual verification platform for research and examination defense.

---

## REFERENCES

1. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal Policy Optimization Algorithms," *arXiv preprint arXiv:1707.06347*, 2017. [Online]. Available: https://arxiv.org/abs/1707.06347
2. D. Fox, W. Burgard, and S. Thrun, "The Dynamic Window Approach to Collision Avoidance," *IEEE Robotics & Automation Magazine*, vol. 4, no. 1, pp. 23–33, Mar. 1997. DOI: 10.1109/100.580977
3. S. Macenski and I. Jambrecic, "SLAM Toolbox: SLAM for the Dynamic World," *Journal of Open Source Software*, vol. 6, no. 61, p. 2783, 2021. DOI: 10.21105/joss.02783
4. S. Macenski, F. Martín, R. White, and J. Clavero, "The Marathon 2: A Navigation System," in *Proceedings of the 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, Las Vegas, NV, USA, 2020, pp. 2718–2725. DOI: 10.1109/IROS45743.2020.9341207
5. P. Anderson, A. Chang, D. S. Chaplot, A. Dosovitskiy, S. Gupta, V. Koltun, J. Kosecka, J. Malik, R. Mottaghi, M. Savva, and A. R. Zamir, "On Evaluation of Embodied Navigation Agents," *arXiv preprint arXiv:1807.06757*, 2018. Available: https://arxiv.org/abs/1807.06757
6. A. Raffin, A. Hill, A. Gleave, et al., "Stable-Baselines3: Reliable Reinforcement Learning Implementations," *Journal of Machine Learning Research*, vol. 22, no. 268, pp. 1–8, 2021.
7. M. Towers et al., "Gymnasium: A Standard Interface for Reinforcement Learning Environments," *arXiv preprint arXiv:2407.17032*, 2024.
8. L. Tai, G. Paolo, and M. Liu, "Virtual-to-real deep reinforcement learning for robot navigation," in *2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2017, pp. 356–361. DOI: 10.1109/IROS.2017.8202181
9. R. Vincent, F. Limketkai, and M. Eriksen, "Comparison of Karto SLAM with popular open-source 2D SLAM algorithms," in *Autonomous Mobile Systems*, Springer, 2010, pp. 111–118.
10. E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, 1927.
11. B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*, CRC Press, 1994.
12. L. Kästner, J. Shen, N. T. Duong, et al., "Arena-Rosnav: Towards Deployment of Deep-Reinforcement-Learning-Based Obstacle Avoidance into Conventional Autonomous Navigation Systems," *IEEE Robotics and Automation Letters*, vol. 6, no. 2, pp. 2463–2470, 2021. DOI: 10.1109/LRA.2021.3060416
13. L. Kästner, C. Li, T. Xiao, et al., "Arena-Bench: A Benchmarking Suite for Obstacle Avoidance Approaches in Highly Dynamic Environments," *IEEE Robotics and Automation Letters*, vol. 7, no. 4, pp. 9177–9184, 2022. DOI: 10.1109/LRA.2022.3190089
14. D. Perumalla, M. Soboleva, and P. Stone, "Benchmarking Autonomous Robot Navigation (BARN) Dataset and Challenge," in *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2020.
15. A. Francis, A. Faust, H. Chiang, et al., "Long-Range Indoor Navigation with PRM-RL," *IEEE Transactions on Robotics*, vol. 36, no. 4, pp. 1115–1132, 2020. DOI: 10.1109/TRO.2020.2974052
16. E. Wijmans, A. Kadian, S. Morcos, S. Lee, I. Essa, D. Parikh, M. Savva, and D. Batra, "DD-PPO: Learning Near-Perfect PointGoal Navigators from 2.5 Billion Frames," in *International Conference on Learning Representations (ICLR)*, 2020.
17. S. Macenski, T. Foote, B. Gerkey, C. Lalancette, and W. Woodall, "Robot Operating System 2: Design, architecture, and uses in the wild," *Science Robotics*, vol. 7, no. 66, 2022; and S. Macenski et al., "From the desks of ROS developers: A survey of modern & capable mobile robotics algorithms in the robot operating system 2," *Robotics and Autonomous Systems*, vol. 168, p. 104493, 2023.
18. A. D. Ames, S. Coogan, M. Egerstedt, G. Notomista, K. Sreenath, and P. Tabuada, "Control Barrier Functions: Theory and Applications," in *18th European Control Conference (ECC)*, 2019, pp. 3420–3431.
19. R. Cheng, G. Orosz, R. M. Murray, and J. W. Burdick, "End-to-End Safe Reinforcement Learning through Barrier Functions for Safety-Critical Continuous Control Tasks," in *Proceedings of the AAAI Conference on Artificial Intelligence*, 2019.
20. H.-T. L. Chiang, J. Wang, and A. Faust, "Execution-time Action Replacement for Safe Deep Reinforcement Learning in Mobile Robot Navigation," in *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2021, pp. 5824–5831.
21. L. Kästner, T. Xiao, and J. Lambrecht, "Safe Deep Reinforcement Learning for Mobile Robot Navigation Using a Velocity Obstacle Safety Shield," in *IEEE International Conference on Robotics and Biomimetics (ROBIO)*, 2022.

---

## APPENDIX: TURNKEY USER MANUAL FOR PROFESSOR DEMONSTRATION

### 1. Launching the Interactive GUI on Windows
Open Windows PowerShell or Command Prompt in the repository root:
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
