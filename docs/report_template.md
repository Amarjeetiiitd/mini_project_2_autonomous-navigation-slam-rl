# Autonomous Navigation using SLAM and Reinforcement Learning - Report

*Author, date, code version (git hash), seeds used.*

## 1. Abstract
One paragraph: question, method, headline result with CI, main limitation.

## 2. Research question and hypotheses
> How does an RL-based local navigation controller compare with conventional navigation in
> goal-reaching success, collision avoidance, path efficiency, and generalisation to unseen
> layouts when localisation and mapping are provided by SLAM?

H1: PPO matches DWA-with-global-plan on success rate in unseen layouts.
H2: PPO is more robust to LiDAR noise / dropouts than the reactive baseline.
H3: Localisation drift degrades the waypoint-conditioned policy less than the direct-goal policy.
*(Rewrite to match what you actually test; state them before looking at test results.)*

## 3. System
Robot and sensor model; SLAM Toolbox config; Nav2 role; observation / action spec (copy from `rl_training/config.py`); reward; PPO hyper-parameters (copy `models/ppo_navigation.json` and `train.py` defaults).

## 4. Experimental protocol
Splits, scenario counts, conditions, seeds (see `configs/research_protocol.md`). Track A vs Track B.

## 5. Results
### 5.1 Training (`results/figures/training_curves.png`)
### 5.2 Track A - controller comparison (paste `results/compare_summary.md`)
| split | condition | controller | n | success (95% CI) | collision | timeout | SPL |
|---|---|---|---|---|---|---|---|
### 5.3 Per-family breakdown (`results/compare_by_family.csv`)
### 5.4 Qualitative trajectories (`results/figures/trajectories_*.png`)
### 5.5 Track B - Gazebo with SLAM (Nav2 vs PPO) and localisation error
Report n, success, collisions, SPL, RMSE of localisation error. Keep separate from 5.2.

## 6. Analysis
Where does PPO fail (timeouts vs collisions)? Which families? Effect of noise and drift? Seeds variance.

## 7. Limitations / threats to validity
Copy and extend the list in `configs/research_protocol.md`.

## 8. Conclusion and future work
Reverse actions, dynamic obstacles, domain randomisation, recurrent policies, real-robot validation.

## Appendix
Commands to reproduce; environment versions (`pip freeze`, `ros2 --version`).
