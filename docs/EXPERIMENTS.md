# Running the experiments

All commands run from the project root with the venv active.

## Track A - standalone benchmark (no ROS)

| Goal | Command |
|---|---|
| Learning curves | `python -m rl_training.plot_training --log-dir logs/ppo_navigation` |
| Checkpoint selection audit | `python scripts/compare_checkpoints.py` (evaluates candidates on disjoint validation split; records `results/checkpoint_selection.json`) |
| E3 unseen layouts | `python -m rl_training.evaluate --split test --episodes 100` (evaluates 100 unique layouts; passing `--episodes 200` evaluates 2 trials per layout) |
| E3 unseen family | `python -m rl_training.evaluate --split heldout --episodes 100` |
| Full 7-controller benchmark (E4+E5+E6) | `python -m rl_training.compare --model models/best/best_model.zip --episodes 100 --include-ablations` (evaluates 4,200 episodes across 7 controllers, 3 conditions, 2 splits) |
| Ablation synchronization & pairwise tests | `python -m rl_training.ablation` (re-syncs `results/ablation_summary.csv` and `results/ablation_pairwise_tests.csv`) |
| CPU latency benchmark | `python scripts/benchmark_inference_latency.py` (measures microbenchmarks, full obs-to-cmd pipeline, and DWA latency) |
| Multi-seed training & evaluation | `python scripts/train_multi_seed.py --seeds 0 1 2` |
| E1/E2 per-family breakdown | `results/compare_by_family.csv` (written by `compare`) |
| Trajectory figures | `python -m rl_training.visualize --model models/best/best_model.zip --split heldout --n 6` |

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
