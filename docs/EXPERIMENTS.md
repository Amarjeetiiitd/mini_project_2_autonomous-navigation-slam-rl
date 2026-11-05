# Running the experiments

All commands run from the project root with the venv active.

## Track A - standalone benchmark (no ROS)

| Goal | Command |
|---|---|
| Learning curves | `python -m rl_training.plot_training --log-dir logs/ppo_navigation` |
| E3 unseen layouts | `python -m rl_training.evaluate --split test --episodes 200` |
| E3 unseen family | `python -m rl_training.evaluate --split heldout --episodes 100` |
| E4 + E5 + E6a (all controllers x conditions x splits) | `python -m rl_training.compare --model models/ppo_navigation --episodes 100` |
| E1/E2 per-family | `results/compare_by_family.csv` (written by `compare`) |
| Trajectory figures | `python -m rl_training.visualize --model models/ppo_navigation --split heldout --n 6` |

### Ablation: direct-goal vs waypoint-conditioned policy

```bash
python -m rl_training.train --timesteps 1000000 --goal-mode direct --run-name ppo_direct
python -m rl_training.compare --model models/ppo_direct --out-dir results/direct
```

### Domain-randomised policy (noise + drift in training)

```bash
python -m rl_training.train --timesteps 1000000 --lidar-noise 0.03 --dropout 0.05 --loc-drift 0.004 \
    --run-name ppo_robust
python -m rl_training.compare --model models/ppo_robust --out-dir results/robust
```

### Several seeds (needed before claiming PPO-variant differences)

```bash
for s in 0 1 2; do
  python -m rl_training.train --timesteps 1000000 --seed $s --run-name ppo_s$s --out-dir models/s$s --log-dir logs
  python -m rl_training.compare --model models/s$s/ppo_s$s --seed 0 --out-dir results/seed$s
done
```
(Keep `compare --seed` fixed so every training seed sees the same scenarios.)

## Track B - ROS 2 / Gazebo
See `docs/ROS_INTEGRATION.md` (matched Nav2 vs PPO trials, localisation error logger).
Keep these results in a separate table from Track A.

## Changing the benchmark
Everything that defines the task lives in `rl_training/config.py` (sensor, robot, rewards,
seed splits, perturbation conditions). Changing observation- or action-related values
invalidates trained models.
