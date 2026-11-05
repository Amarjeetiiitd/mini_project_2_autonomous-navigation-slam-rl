#!/usr/bin/env bash
# Full standalone (no ROS) experiment pipeline: check -> train -> evaluate -> compare -> figures.
#   scripts/run_standalone_pipeline.sh [TIMESTEPS] [EPISODES] [SEED]
set -euo pipefail
cd "$(dirname "$0")/.."
STEPS="${1:-1000000}"; EPISODES="${2:-100}"; SEED="${3:-0}"

python -m rl_training.check_env
python -m pytest -q
python -m rl_training.train --timesteps "$STEPS" --n-envs 4 --subproc --n-steps 1024 --seed "$SEED"
python -m rl_training.plot_training --log-dir logs/ppo_navigation
python -m rl_training.evaluate --model models/ppo_navigation --episodes "$EPISODES" --split test
python -m rl_training.compare --model models/ppo_navigation --episodes "$EPISODES" --seed "$SEED"
python -m rl_training.visualize --model models/ppo_navigation --split test --n 6 --out results/figures/trajectories_test.png
python -m rl_training.visualize --model models/ppo_navigation --split heldout --n 6 --out results/figures/trajectories_heldout.png
echo "Done. See results/compare_summary.md and results/figures/"
