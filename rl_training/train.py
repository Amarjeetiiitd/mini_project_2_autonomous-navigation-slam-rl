"""Train a PPO local-navigation policy (Stable-Baselines3).

    python -m rl_training.train --timesteps 10000          # smoke test
    python -m rl_training.train --timesteps 1000000 --n-envs 4 --seed 0

Outputs (under --out-dir, default `models/`):
    ppo_navigation.zip     final policy
    ppo_navigation.json    metadata (goal_mode, env kwargs, seeds, versions) read by evaluate/ROS
    best/best_model.zip    best checkpoint on the validation layouts
and per-env Monitor logs under `logs/<run-name>/` (for learning curves).
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import gymnasium
import numpy as np
import stable_baselines3 as sb3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from . import config as C
from .nav_env import NavEnv


def make_env_fn(layout_seeds, env_kwargs, seed, log_path):
    def _init():
        env = NavEnv(layout_seeds=layout_seeds, **env_kwargs)
        env = Monitor(env, filename=log_path, info_keywords=("is_success", "collision", "spl"))
        env.reset(seed=seed)
        return env
    return _init


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--timesteps", type=int, default=300_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--subproc", action="store_true", help="use SubprocVecEnv (multi-core)")
    p.add_argument("--goal-mode", choices=["waypoint", "direct"], default="waypoint")
    p.add_argument("--lidar-noise", type=float, default=0.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--loc-drift", type=float, default=0.0, help="per-step position random-walk std (m)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--ent-coef", type=float, default=0.005)
    p.add_argument("--eval-freq", type=int, default=25_000, help="total env steps between validations")
    p.add_argument("--run-name", default="ppo_navigation")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--log-dir", default="logs")
    a = p.parse_args()

    set_random_seed(a.seed)
    torch.set_num_threads(1)
    out, logs = Path(a.out_dir), Path(a.log_dir) / a.run_name
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"goal_mode": a.goal_mode, "lidar_noise_std": a.lidar_noise,
                  "lidar_dropout": a.dropout, "loc_drift_pos": a.loc_drift,
                  "loc_drift_yaw": a.loc_drift / 2}
    Vec = SubprocVecEnv if a.subproc else DummyVecEnv
    venv = Vec([make_env_fn(C.TRAIN_SEEDS, env_kwargs, a.seed + i, str(logs / f"monitor_{i}"))
                for i in range(a.n_envs)])
    # validation: held-out layouts, nominal conditions (no perturbations)
    val_kwargs = {"goal_mode": a.goal_mode}
    veval = DummyVecEnv([make_env_fn(C.VAL_SEEDS, val_kwargs, 10_000 + a.seed, str(logs / "val_monitor"))])

    model = PPO("MlpPolicy", venv, learning_rate=a.lr, n_steps=a.n_steps, batch_size=a.batch_size,
                gamma=0.99, gae_lambda=0.95, ent_coef=a.ent_coef, clip_range=0.2, n_epochs=10,
                policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
                seed=a.seed, device="cpu", verbose=1)

    per_env = max(a.eval_freq // a.n_envs, 1)
    cbs = [EvalCallback(veval, best_model_save_path=str(out / "best"), log_path=str(logs / "eval"),
                        eval_freq=per_env, n_eval_episodes=30, deterministic=True),
           CheckpointCallback(save_freq=per_env * 4, save_path=str(out / "checkpoints"),
                              name_prefix=a.run_name)]
    model.learn(total_timesteps=a.timesteps, callback=cbs, progress_bar=False)

    model.save(str(out / a.run_name))
    meta = {"goal_mode": a.goal_mode, "env_kwargs": env_kwargs, "timesteps": a.timesteps,
            "seed": a.seed, "n_envs": a.n_envs, "obs_dim": C.OBS_DIM, "lookahead": C.LOOKAHEAD,
            "v_max": C.V_MAX, "w_max": C.W_MAX, "train_layout_seeds": [C.TRAIN_SEEDS[0], C.TRAIN_SEEDS[-1]],
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "sb3": sb3.__version__, "gymnasium": gymnasium.__version__,
                         "numpy": np.__version__}}
    (out / f"{a.run_name}.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {out / a.run_name}.zip")


if __name__ == "__main__":
    main()
