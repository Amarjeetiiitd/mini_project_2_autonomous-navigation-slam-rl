.PHONY: test check smoke train evaluate compare figures all

test:
	python -m pytest -q

check:
	python -m rl_training.check_env

smoke:
	python -m rl_training.train --timesteps 10000 --n-envs 2 --eval-freq 5000 --run-name smoke --out-dir /tmp/smoke_models --log-dir /tmp/smoke_logs

train:
	python -m rl_training.train --timesteps 1000000 --n-envs 4 --subproc --n-steps 1024

evaluate:
	python -m rl_training.evaluate --episodes 100 --model models/ppo_navigation
	python -m rl_training.baselines --episodes 100

compare:
	python -m rl_training.compare --model models/ppo_navigation --episodes 100

figures:
	python -m rl_training.plot_training --log-dir logs/ppo_navigation
	python -m rl_training.visualize --model models/ppo_navigation --split test --n 6 --out results/figures/trajectories_test.png
	python -m rl_training.visualize --model models/ppo_navigation --split heldout --n 6 --out results/figures/trajectories_heldout.png

all: check test train evaluate compare figures
