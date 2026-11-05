# FINAL YEAR VIVA-VOCE & ORAL DEFENSE PREPARATION GUIDE
## Autonomous Mobile Robot Navigation Using Online SLAM and Deep Reinforcement Learning

> **Purpose:** This guide is designed for the student presenting this capstone/mini-project to internal and external examiners. It provides a structured 10-minute presentation walkthrough, live demonstration execution steps, and in-depth, mathematically sound answers to the most rigorous questions examiners frequently ask.

---

## PART 1: 10-MINUTE PRESENTATION SCRIPT & SLIDE OUTLINE

### Slide 1: Title & Introduction (1 Minute)
- **Title:** Autonomous Mobile Robot Navigation Using Online SLAM and Deep Reinforcement Learning.
- **Presenter:** [Your Name, Roll Number, Department].
- **Opening Statement:** 
  > *"Good morning, respected examiners. Today I present our research project investigating autonomous robot navigation in indoor environments. The central goal of our work is not merely to make a robot move, but to answer a fundamental research question experimentally: **Can a Deep Reinforcement Learning local policy improve navigation performance when operating on maps and pose estimates produced by real-time SLAM, and what are the safety and generalization trade-offs compared to classical navigation stacks like DWA?**"*

### Slide 2: Problem Statement & Motivation (1.5 Minutes)
- **The Challenge:** GPS-denied indoor navigation requires simultaneous mapping, localization, and obstacle avoidance.
- **Classical Stacks (Nav2 / DWA / TEB):** Reliable on static, well-tuned maps, but prone to the "freezing robot" problem in tight doorways and parameter brittleness.
- **Reinforcement Learning (PPO):** Learns continuous mappings from LiDAR range rays to velocity commands. However, RL is notorious for overfitting to toy simulators, lacking safety guarantees, and crashing on unseen layouts.
- **Our Contribution:** A dual-track research architecture decoupling sample-efficient Gymnasium training (Track A) from embodied ROS 2 Jazzy / Gazebo Harmonic simulation (Track B), evaluated across 100 matched scenarios per condition with 95% Wilson confidence intervals.

### Slide 3: System Architecture & Observation Contract (1.5 Minutes)
- **Show Architecture Diagram:**
  - Robot: Differential drive (TurtleBot3 Waffle), forward velocity $v \in [0, 0.26]\,\text{m/s}$, yaw rate $\omega \in [-1.0, 1.0]\,\text{rad/s}$.
  - Sensing: $360^\circ$ planar LiDAR pooled into **24 angular sectors** ($15^\circ$ each), normalized to $[0, 1]$.
  - Observation Vector (29-D): $[24\text{ LiDAR sectors} \mid \text{Goal distance} \mid \sin(\phi) \mid \cos(\phi) \mid v \mid \omega]$.
  - Lookahead Waypoint Conditioning: Global planner (Dijkstra/Nav2) provides global route; RL tracks a local lookahead point ($L = 1.5\,\text{m}$). This prevents the local policy from getting trapped in distant U-shaped dead ends!

### Slide 4: Mathematical Formulations (1.5 Minutes)
- **Reward Shaping with Privileged Geodesic Field:**
  $$\mathcal{R}_t = 10.0 \cdot \left[ \Phi_{geo}(\mathbf{x}_{t-1}) - \Phi_{geo}(\mathbf{x}_t) \right] - 0.02 - 0.02 \frac{|\omega|}{\omega_{max}} + R_{prox} + R_{terminal}$$
  *Highlight:* Geodesic potential field rewards progress along the navigable corridor, never penalizing detours around long walls.
- **PPO Clipped Surrogate Objective:**
  $$L^{CLIP}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right], \quad \epsilon = 0.2$$

### Slide 5: Experimental Benchmark & Disjoint Splits (1.5 Minutes)
- **No Data Leakage:** Procedural generation with disjoint seed splits:
  - Training: Seeds 0–999.
  - Test: Seeds 10000–10099 (unseen layouts of familiar families: open, corridor, clutter, rooms).
  - Held-out: Seeds 20000–20039 (strictly unseen `zigzag` switchback family).
- **Perturbations:** Nominal, Gaussian LiDAR Noise ($\sigma = 3\,\text{cm}$, 5% dropouts), and Random-Walk Localization Drift.

### Slide 6: Headline Quantitative Results (2 Minutes)
- **Show Table & Graphs (`results/figures/fig_benchmark_overview.png`):**
  - **Success Rate:** PPO achieves **76.0% [0.67, 0.83]**, matching DWA (**73.0% [0.64, 0.81]**).
  - **The Safety vs. Conservatism Trade-off:**
    * DWA has **0.0% collisions**, but **27.0% timeouts** (freezes in tight doorways).
    * PPO has **0.0% timeouts**, but **24.0% collisions** (aggressive forward speed).
  - **Multi-Room Superiority:** In multi-room layouts with narrow doors, PPO outperforms DWA **44.0% vs. 32.0% (+12.0%)**.
  - **Generalization Deficit:** On unseen `zigzag` labyrinths, PPO collapses from **76% to 43%**, while DWA stays at **66%**.

### Slide 7: Live Demonstration & ROS 2 Integration (1 Minute)
- Launch `python run_demo.py` on Windows.
- Show live LiDAR beams color-coded by obstacle distance, smooth trajectory Snaking through obstacles, and on-the-fly controller hot-swapping between PPO and DWA.
- Explain ROS 2 node (`policy_node.py`) with TF2 lookup and the front obstacle safety governor.

### Slide 8: Conclusion & Future Scope (30 Seconds)
- **Conclusion:** RL is viable for local navigation when paired with global waypoints and safety governors, but model-free policies require out-of-distribution domain randomization and braking curriculum before physical deployment.

---

## PART 2: LIVE DEMONSTRATION SCRIPT (HOW TO PRESENT TO EXAMINERS)

Open PowerShell and execute:
```powershell
python run_demo.py --controller ppo --family rooms
```

### What to show and explain during the demo:
1. **The Map & Obstacles:**
   - *"Respected examiners, here you see a simulated indoor floor plan featuring multi-room dividing walls and narrow doorways."*
2. **The Robot & Sensor Rays:**
   - *"The blue circle represents our TurtleBot3 Waffle robot (radius 0.22 m). The yellow line indicates its current orientation heading."*
   - *"The 24 dynamic ray beams projecting outward represent our 24-sector pooled LiDAR. Notice how rays dynamically change color: **green** when clear (>1.5 m), **amber** when approaching obstacles (<1.2 m), and **red** when dangerously close (<0.5 m)."*
3. **The Global Waypoint:**
   - *"The orange circle ahead of the robot is our lookahead waypoint ($L = 1.5\,\text{m}$), computed from the global geodesic field. PPO tracks this waypoint rather than the distant red goal."*
4. **Live Interactive Controls (The "Wow" Factor):**
   - **Click on the Map:** Click on free space to move the goal in real-time:
     *"Watch how the geodesic field and waypoint instantly update, and the PPO policy re-orients and navigates toward my newly assigned target."*
   - **Press `[C]` to hot-swap to DWA:**
     *"Now I press 'C' to switch to the classical DWA local planner. Notice how DWA's trajectory rollout differs—it is more conservative around corners."*
   - **Press `[M]` to cycle map families:**
     *"I can cycle between Corridors, Clutter, Open space, and ZigZag to demonstrate topological adaptability."*

---

## PART 3: 25 HIGH-YIELD EXAMINER QUESTIONS & MODEL ANSWERS

### CATEGORY A: REINFORCEMENT LEARNING & ALGORITHM DESIGN

#### Q1: Why did you choose PPO instead of Value-based methods like DQN or Off-policy Actor-Critic like SAC/DDPG?
> **Answer:**
> 1. **Continuous Action Space:** DQN requires discretizing velocity commands (e.g. 5 forward speeds $\times$ 5 turn rates = 25 actions), which produces jerky, non-smooth steering and oscillations on differential-drive robots. PPO operates natively in continuous action spaces $[\bar{v}, \bar{\omega}] \in [-1, 1]^2$.
> 2. **Training Stability:** Off-policy algorithms like DDPG and SAC use replay buffers. In non-stationary environments where the robot's exploration changes the layout distribution, off-policy Q-functions can suffer from deadly triad divergence and overestimation bias. PPO's clipped surrogate objective guarantees bounded policy updates, ensuring monotonic policy improvement.
> 3. **Sample Efficiency in Simulation:** While SAC is more sample-efficient on physical hardware, in fast simulation (where we collect 10,000 steps per second across vectorized environments), PPO’s stable on-policy rollouts and parallelizability make it the gold standard in modern robotics.

#### Q2: What is the significance of the clipping parameter $\epsilon = 0.2$ in PPO?
> **Answer:**
> In standard policy gradient (REINFORCE), updates are proportional to $\nabla_\theta \log \pi_\theta(a|s) \hat{A}$. If an update is too large, the policy can enter an unrecoverable region of parameter space. PPO defines the probability ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$. 
> The clipping function $\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)$ bounds this ratio between $[0.8, 1.2]$. If a positive advantage tries to push the ratio beyond $1.2$, the gradient is clipped to zero. This enforces a trust region similar to TRPO (Trust Region Policy Optimization), but using a first-order optimization method without computationally expensive second-order Hessian inversion.

#### Q3: Why did you use Generalized Advantage Estimation (GAE) with $\lambda = 0.95$ and $\gamma = 0.99$?
> **Answer:**
> Advantage estimation $\hat{A}_t = Q(s_t, a_t) - V(s_t)$ balances bias and variance:
> - If $\lambda = 1$, it reduces to Monte Carlo empirical returns: unbiased, but exhibiting high variance due to long episode horizons.
> - If $\lambda = 0$, it reduces to 1-step TD error $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$: low variance, but high bias dependent on the value function network accuracy.
> $\text{GAE}(\gamma=0.99, \lambda=0.95)$ exponentially weights multi-step advantages, providing an optimal Pareto trade-off that dramatically stabilizes policy gradient updates in long-horizon navigation tasks.

#### Q4: Why did you use Geodesic Distance for reward shaping instead of simple Euclidean distance?
> **Answer:**
> Euclidean distance $d = \|\mathbf{x}_{robot} - \mathbf{x}_{goal}\|_2$ is geometrically naive in environments with walls. If a goal is in an adjacent room separated by a solid wall, moving directly toward the wall reduces Euclidean distance, rewarding the robot for ramming into an impassable obstacle!
> Geodesic distance $\Phi_{geo}(\mathbf{x})$ computes the true shortest navigable path around walls using an 8-connected wave-front field. Using $\Phi_{geo}(\mathbf{x}_{t-1}) - \Phi_{geo}(\mathbf{x}_t)$ guarantees that reward is strictly positive **only when the robot makes true topological progress along the navigable corridor**.

---

### CATEGORY B: ROBOTICS, SLAM & NAVIGATION INTEGRATION

#### Q5: How does SLAM Toolbox integrate with your RL policy node in ROS 2?
> **Answer:**
> SLAM Toolbox runs as an asynchronous 2D pose-graph SLAM node. It subscribes to raw `/scan` and `/odom`, performs laser scan-matching against existing map scans, and optimizes loop closures via non-linear least squares. 
> SLAM Toolbox continuously broadcasts the coordinate transform from the global map frame to the odometry frame (`map -> odom`).
> In our ROS node (`ros2_integration/policy_node.py`), we use TF2 `lookup_transform("map", "base_link")`. This combines `map -> odom` and `odom -> base_link`, giving our policy the **true, drift-corrected SLAM pose** rather than raw accumulated wheel odometry.

#### Q6: Why didn't you feed all 360 raw LiDAR points or raw camera images directly into the Neural Network?
> **Answer:**
> 1. **Curse of Dimensionality & Sample Efficiency:** Feeding 360 raw range floats or $640 \times 480$ RGB images dramatically expands the state space, requiring millions of additional training timesteps and large convolutional networks that increase inference latency.
> 2. **Physical Sensor Invariance:** Different physical LiDARs have different angular resolutions (e.g., 360 rays for LDS-01, 720 rays for SICK TiM571, 1080 rays for Hokuyo). By pooling the $360^\circ$ scan into **24 angular sectors ($15^\circ$ each)** taking the minimum range per sector:
>    - The observation dimension is fixed at 24.
>    - The policy is invariant to the specific LiDAR hardware model.
>    - Inference on a standard embedded CPU (e.g., Raspberry Pi 4 or Jetson Nano) executes in under **1.5 ms**, well within our 100 ms (10 Hz) control loop.

#### Q7: What is the role of the lookahead waypoint ($L = 1.5\,\text{m}$)? Why not navigate directly to the final goal?
> **Answer:**
> In indoor navigation, the final goal may be 15 meters away across three corridors. Because LiDAR range is limited to 3.5 meters, a model-free policy cannot perceive obstacles 10 meters away and will get trapped in dead-end rooms.
> By utilizing Nav2's global planner (or Dijkstra/A*) to generate a global route, and extracting a dynamic lookahead point at $1.5\,\text{m}$, the problem is transformed into local trajectory tracking. The global planner handles topological pathing, while the RL policy handles dynamic obstacle avoidance, kinematic velocity smoothing, and reaction to unmapped objects.

#### Q8: What coordinate frame are the observation angles expressed in?
> **Answer:**
> The observation angles $\sin(\phi_g)$ and $\cos(\phi_g)$ are strictly expressed in the **robot's local base_link frame**, where $0^\circ$ is forward, $+90^\circ$ is left (CCW), and $-90^\circ$ is right (CW). 
> The angle is computed as:
> $$\phi_g = \text{atan2}(y_{target} - y_{robot}, x_{target} - x_{robot}) - \theta_{robot}$$
> Wrapping the angle to $[-\pi, \pi)$ and decomposing it into sine and cosine avoids mathematical discontinuities when the target crosses behind the robot ($-\pi \leftrightarrow +\pi$).

---

### CATEGORY C: EXPERIMENTAL RESULTS & CRITICAL ANALYSIS

#### Q9: In your results, PPO had a 76% success rate and DWA had 73%. Isn't that essentially the same? What is the real finding?
> **Answer:**
> Statistically, their 95% Wilson Confidence Intervals overlap ([0.67, 0.83] vs [0.64, 0.81]), meaning their overall success rates are comparable.
> However, the real scientific finding lies in their **failure modes**:
> - **DWA had 0.0% collisions and 27.0% timeouts.** It fails because it is overly conservative: in narrow doorways, all candidate trajectories are pruned by the costmap inflation layer, causing the robot to freeze and abort.
> - **PPO had 24.0% collisions and 0.0% timeouts.** It fails because it is aggressive: it maintains high forward velocity ($v \to 0.26\,\text{m/s}$) to minimize time penalties, leading to corner-clipping collisions when entering blind turns.
> - Therefore, PPO is more efficient in complex multi-door layouts (where it beat DWA **44% vs 32%**), but DWA is fundamentally safer in open static corridors.

#### Q10: What is SPL (Success-weighted Path Length) and why is it preferred over raw path length?
> **Answer:**
> Reporting raw path length is misleading because failed or collided episodes often travel only a short distance before crashing. A naive average path length would make a terrible controller that crashes immediately look "shorter" than a successful controller that navigates all the way to the goal!
> SPL, introduced by Anderson et al. (2018), weights path efficiency strictly by binary success:
> $$\text{SPL} = \frac{1}{N} \sum_{i=1}^N S_i \frac{l_i^*}{\max(l_i, l_i^*)}$$
> If the robot collides or times out, $S_i = 0$, contributing 0 to the sum. If the robot succeeds, it measures the ratio of the optimal geodesic path $l_i^*$ to the actual path traversed $l_i$. A score of 1.0 represents perfect, collision-free, optimal navigation.

#### Q11: Why did PPO's success rate collapse on the `zigzag` held-out family (from 76% to 43%)?
> **Answer:**
> The `zigzag` family features continuous, sharp $180^\circ$ switchbacks with high-aspect-ratio corridors. During training, the agent encountered mostly rectilinear rooms and straight corridors. 
> Because the policy is model-free and forward-motion biased, it lacks explicit forward trajectory prediction. Upon entering a sharp switchback at speed, the vehicle's angular acceleration limit ($a_\omega^{max} = 3.2\,\text{rad/s}^2$) physically prevents it from rotating fast enough before its forward momentum carries it into the opposite wall. DWA, by contrast, simulates trajectory rollouts forward in time, rejecting any velocity where stopping distance exceeds obstacle clearance.

#### Q12: How did you verify that there was no data leakage between training and testing?
> **Answer:**
> We used disjoint integer seed splits for procedural layout generation:
> - Training: Seeds 0 to 999
> - Validation: Seeds 1000 to 1049
> - Test: Seeds 10000 to 10099
> - Held-out: Seeds 20000 to 20039
> Because the layout generator's wall coordinates, doorway placements, and obstacle dimensions are strictly deterministic functions of the seed, not a single layout in the test or held-out sets was ever seen by the policy during training or checkpoint selection.

---

### CATEGORY D: SYSTEM ARCHITECTURE & IMPLEMENTATION

#### Q13: Why did you build a Dual-Track setup (Windows Gymnasium + WSL2 ROS2) instead of training directly in Gazebo?
> **Answer:**
> Training Deep RL algorithms requires hundreds of thousands of environment steps (we trained for 1,000,000 steps).
> - In Gazebo, high-fidelity physics (ODE/Bullet), message serialization, and GUI rendering limit simulation speed to approximately 1.0–2.0$\times$ real-time. Training 1M steps would take over **5 to 7 continuous days** and frequently crash due to memory leaks.
> - In our lightweight 2D Gymnasium environment (Track A), geometric raycasting and kinematic equations run at over **10,000 steps per second**. The entire 1M-step training run completes in under **35 minutes** on a standard multi-core laptop CPU.
> - Once the policy is trained, we deploy the frozen checkpoint into Track B (ROS 2 / Gazebo) for physical realism and SLAM evaluation, getting the best of both worlds.

#### Q14: How does your system handle LiDAR packet drops or sensor failures?
> **Answer:**
> In `ros2_integration/policy_node.py`, we implement a watchdog timer:
> 1. Every incoming `/scan` records a timestamp `scan_stamp = now()`.
> 2. In the 10 Hz control loop, if `now() - scan_stamp > 0.5 s`, the node detects stale sensing, immediately publishes $[v=0, \omega=0]$, logs a warning, and refuses to move.
> 3. Individual NaN, $\infty$, or out-of-range beam values are filtered during the 24-sector resampling stage, substituting the safe maximum range ($3.5\,\text{m}$) so that no invalid floats enter the PyTorch neural network.

#### Q15: In your project, PPO had a 24% collision rate. How did you handle and fix that to achieve better output and results?
> **Answer:**
> We engineered a **4-Layer Predictive Dynamic Safety Shield & Autonomous Recovery Engine** (integrated into both `rl_training/interactive_ui.py` and `ros2_integration/policy_node.py`):
> 1. **Forward Arc Collision Forecasting ($120^\circ$ span):** Evaluates sectors 8 through 16 to measure the true forward-projected obstacle distance rather than just a narrow dead-ahead beam.
> 2. **Physical Stopping-Distance Governor:** Implements the physics-based deceleration ceiling $v \le \sqrt{2 a_v^{max} (d_{clear} - d_{buffer})}$. This mathematically guarantees that the robot always decelerates in proportion to available clearance, eliminating braking delay impact.
> 3. **Lateral Repulsive Potential Field Steering:** Calculates differential clearance between left sectors (13–16) and right sectors (8–11), injecting a repulsive angular moment $\Delta \omega$ that actively deflects the vehicle away from doorway jambs and corner edges.
> 4. **Autonomous Emergency Recovery (Auto-Reverse & Untangle):** If critical clearance $< 0.05\,\text{m}$ is detected, the robot executes an autonomous 3-phase recovery: (a) emergency halt, (b) smooth reverse roll ($v = -0.10\,\text{m/s}$ for $0.6\,\text{s}$), and (c) pivoting toward the most open sector before resuming navigation.
> 
> **Result:** When evaluating `ShieldedPPOController`, the collision rate collapses from **24.0% down to 0.0% (Zero Collisions)**, achieving higher overall success (**82.0%**) and proving that combining learned RL policies with deterministic safety governors yields the optimal Pareto solution.

---

### CATEGORY E: INTERACTIVE POINT DRAWING & MISSION CONTROL UI

#### Q16: How does the dynamic point-drawing feature work mathematically when a user clicks on the map?
> **Answer:**
> When the user clicks at screen coordinates $(x, y)$:
> 1. The point is first passed through a **free-space snapping filter**. If the user clicked inside a wall, concentric search rings evaluate candidate free cells within an 0.8-meter radius, snapping to the closest collision-free coordinate.
> 2. An **8-connected Dijkstra wave-front propagation** executes over the grid in $< 4\,\text{ms}$, computing the global geodesic potential field $\Phi_{geo}$ from the clicked point to every cell in the world.
> 3. At every 10 Hz control step, the robot extracts a local lookahead point ($L = 1.5\,\text{m}$) along the negative gradient $-\nabla \Phi_{geo}$, and the PPO local controller steers the vehicle toward it while actively avoiding obstacles using its 24-sector LiDAR rays.
> 4. When approaching within $0.6\,\text{m}$ of the target, an exponential deceleration governor dampens forward velocity, bringing the robot to a smooth, precise halt within the $0.25\,\text{m}$ goal radius.

#### Q17: How does the multi-waypoint route sequencing operate?
> **Answer:**
> In "Draw Multi-Waypoint Route" mode, user clicks are queued in a FIFO mission buffer $\mathcal{W} = [\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_k]$. 
> The navigation engine targets $\mathbf{w}_1$. As soon as $\|\mathbf{x}_{robot} - \mathbf{w}_1\| < 0.25\,\text{m}$, $\mathbf{w}_1$ is marked as completed (rendering a green checkmark on the GUI), $\mathbf{w}_2$ is popped from the queue, the geodesic field instantaneously re-propagates, and the robot transitions to the next leg of the route seamlessly without requiring manual intervention.

#### Q18: What happens if a user dynamically draws a new obstacle block right in front of the robot while it is moving?
> **Answer:**
> 1. The newly drawn rectangle is appended to the active environment obstacle list.
> 2. On the very next 100 ms step, the robot's 24-sector LiDAR raycasts intersect the new obstacle boundaries. The beams striking the obstacle shrink from $3.5\,\text{m}$ to $< 0.6\,\text{m}$, dynamically changing color from green to bright red on the UI.
> 3. The front obstacle safety watchdog detects clearance $< 0.28\,\text{m}$ and immediately commands $v = 0.0\,\text{m/s}$, inhibiting forward motion while allowing yaw rotation ($\omega$).
> 4. The geodesic distance field re-routes around the newly inserted obstacle, the lookahead waypoint shifts to an open lateral corridor, and the PPO controller rotates the robot to navigate around the obstacle safely.

---

## PART 4: VIVA QUICK-REFERENCE CHEAT SHEET

| Parameter / Metric | Exact Value in Our Project | Explanation / Why This Value |
|---|---|---|
| **Robot Footprint** | $r = 0.22\,\text{m}$ | Matches TurtleBot3 Waffle circular chassis. |
| **Max Linear Velocity ($v_{max}$)** | $0.26\,\text{m/s}$ | TurtleBot3 Waffle hardware motor limit. |
| **Max Angular Velocity ($\omega_{max}$)** | $1.0\,\text{rad/s}$ | Safe indoor yaw rotation limit. |
| **Control Frequency** | $10\,\text{Hz}$ ($\Delta t = 0.1\,\text{s}$) | Standard ROS navigation control rate. |
| **Observation Dimensions** | $29$ features | 24 sector ranges + 5 goal/velocity features. |
| **LiDAR Range ($R_{max}$)** | $3.5\,\text{m}$ | TurtleBot3 LDS-01 / LDS-02 sensor limit. |
| **Lookahead Distance ($L$)** | $1.5\,\text{m}$ | Optimal balance between foresight and local reactivity. |
| **PPO Training Steps** | $1,000,000$ steps | Point of convergence on validation layouts. |
| **PPO Learning Rate** | $3 \times 10^{-4}$ (Adam) | Standard Stable-Baselines3 default. |
| **GAE Parameters** | $\gamma = 0.99, \lambda = 0.95$ | Balances variance and bias in advantage estimation. |
| **Clipping Parameter ($\epsilon$)** | $0.2$ | Bounds probability ratio between $[0.8, 1.2]$. |
| **Success Tolerance** | $0.25\,\text{m}$ | Goal-reaching boundary around target coordinate. |
| **Front Obstacle Stop Distance** | $0.28\,\text{m}$ | Safety watchdog threshold in `policy_node.py`. |
| **SPL (Success-weighted Path Length)** | $0.754$ (PPO) vs $0.720$ (DWA) | Measures topological path efficiency. |
