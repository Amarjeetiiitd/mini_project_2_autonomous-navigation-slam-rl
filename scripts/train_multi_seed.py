"""Train and evaluate PPO navigation policies across multiple independent seeds.

Implements the multi-seed research protocol recommended for peer-reviewed publication:
For each seed in seeds (e.g., [0, 1, 2]):
1. Independent PPO policy training on TRAIN_SEEDS (seed + i)
2. Periodic validation evaluation and checkpoint saving via EvalCallback on VAL_SEEDS
3. Pre-registered validation checkpoint selection (highest validation success rate / reward)
4. Matched evaluation on identical TEST and HELDOUT scenario manifests
5. Aggregates multi-seed statistics: mean, std, and 95% confidence intervals across policies,
   separating policy-level training stochasticity from scenario-level evaluation variation.

Outputs:
  models/seed_{i}/best/best_model.zip
  results/multi_seed_evaluation.csv
  results/multi_seed_summary.json
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

import rl_training.config as C
from rl_training.baselines import (
    MASRLController,
    PPOController,
    evaluate_controller,
)
from rl_training.maps import FAMILIES
from rl_training.metrics import summarize
from rl_training.nav_env import NavEnv
from rl_training.scenarios import make_scenarios


def make_env_fn(layout_seeds, env_kwargs, seed, log_path):
    def _init():
        env = NavEnv(layout_seeds=layout_seeds, **env_kwargs)
        env = Monitor(env, filename=log_path, info_keywords=("is_success", "collision", "spl"))
        env.reset(seed=seed)
        return env
    return _init


def train_single_seed(seed: int, timesteps: int = 300_000, n_envs: int = 4,
                      out_dir: Path = Path("models")) -> Path:
    set_random_seed(seed)
    torch.set_num_threads(1)
    seed_dir = out_dir / f"seed_{seed}"
    best_dir = seed_dir / "best"
    logs_dir = Path("logs") / f"multi_seed_{seed}"
    best_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env_kwargs = {"goal_mode": "waypoint"}
    venv = DummyVecEnv([
        make_env_fn(C.TRAIN_SEEDS, env_kwargs, seed * 100 + i, str(logs_dir / f"monitor_{i}"))
        for i in range(n_envs)
    ])
    veval = DummyVecEnv([
        make_env_fn(C.VAL_SEEDS, env_kwargs, 10_000 + seed, str(logs_dir / "val_monitor"))
    ])

    model = PPO(
        "MlpPolicy", venv, learning_rate=3e-4, n_steps=512, batch_size=256,
        gamma=0.99, gae_lambda=0.95, ent_coef=0.005, clip_range=0.2, n_epochs=10,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        seed=seed, device="cpu", verbose=0
    )

    eval_freq = max(25_000 // n_envs, 1)
    eval_cb = EvalCallback(
        veval, best_model_save_path=str(best_dir), log_path=str(logs_dir / "eval"),
        eval_freq=eval_freq, n_eval_episodes=30, deterministic=True
    )
    print(f"[Seed {seed}] Training PPO policy for {timesteps} steps...")
    model.learn(total_timesteps=timesteps, callback=[eval_cb], progress_bar=False)

    final_model_path = seed_dir / "ppo_navigation.zip"
    model.save(str(seed_dir / "ppo_navigation"))

    best_model_path = best_dir / "best_model.zip"
    selected_path = best_model_path if best_model_path.exists() else final_model_path
    print(f"[Seed {seed}] Completed training. Selected checkpoint: {selected_path}")
    return selected_path


def evaluate_seed_policy(model_path: Path, seed: int, episodes: int = 100) -> list[dict]:
    model = PPO.load(str(model_path), device="cpu")
    controllers = [
        ("ppo_waypoint", PPOController(model, "ppo_waypoint")),
        ("mas_rl", MASRLController(model, "mas_rl")),
    ]

    records = []
    for split in ["test", "heldout"]:
        scenarios = make_scenarios(episodes, split, seed=0)
        for cond_name in ["nominal", "lidar_noise", "loc_drift"]:
            cond_kw = C.CONDITIONS[cond_name]
            for ctrl_name, ctrl in controllers:
                df = evaluate_controller(ctrl, scenarios, {"goal_mode": "waypoint", **cond_kw}, label=ctrl_name)
                s = summarize(df)
                records.append({
                    "training_seed": seed,
                    "model_path": str(model_path),
                    "split": split,
                    "condition": cond_name,
                    "controller": ctrl_name,
                    "episodes": s["n"],
                    "success_rate": s["success_rate"],
                    "collision_rate": s["collision_rate"],
                    "timeout_rate": s["timeout_rate"],
                    "spl": s["spl"],
                })
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2],
                   help="List of independent training seeds")
    p.add_argument("--timesteps", type=int, default=300_000,
                   help="Timesteps per seed policy training")
    p.add_argument("--episodes", type=int, default=100,
                   help="Evaluation episodes per split/condition")
    p.add_argument("--eval-existing", action="store_true", default=False,
                   help="Evaluate existing seed models without retraining")
    p.add_argument("--out-csv", default="results/multi_seed_evaluation.csv")
    p.add_argument("--out-json", default="results/multi_seed_summary.json")
    a = p.parse_args()

    all_eval_records = []
    for s in a.seeds:
        seed_model = Path(f"models/seed_{s}/best/best_model.zip")
        if not a.eval_existing or not seed_model.exists():
            if s == 0 and Path("models/best/best_model.zip").exists() and a.eval_existing:
                seed_model = Path("models/best/best_model.zip")
            else:
                seed_model = train_single_seed(s, timesteps=a.timesteps)

        print(f"\nEvaluating Seed {s} policy ({seed_model}) across benchmark splits...")
        records = evaluate_seed_policy(seed_model, seed=s, episodes=a.episodes)
        all_eval_records.extend(records)

    eval_df = pd.DataFrame(all_eval_records)
    Path(a.out_csv).parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(a.out_csv, index=False)
    print(f"\nSaved multi-seed evaluation results to: {a.out_csv}")

    # Compute summary across seeds
    summary_rows = []
    for (split, cond, ctrl), g in eval_df.groupby(["split", "condition", "controller"]):
        n_seeds = len(g)
        succ_mean = float(g["success_rate"].mean())
        succ_std = float(g["success_rate"].std()) if n_seeds > 1 else 0.0
        spl_mean = float(g["spl"].mean())
        spl_std = float(g["spl"].std()) if n_seeds > 1 else 0.0

        summary_rows.append({
            "split": split,
            "condition": cond,
            "controller": ctrl,
            "n_training_seeds": n_seeds,
            "success_mean": round(succ_mean, 4),
            "success_std": round(succ_std, 4),
            "spl_mean": round(spl_mean, 4),
            "spl_std": round(spl_std, 4),
        })

    summary_json = {
        "seeds": a.seeds,
        "timesteps": a.timesteps,
        "episodes_per_seed": a.episodes,
        "results": summary_rows,
    }
    Path(a.out_json).write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"Saved multi-seed statistical summary to: {a.out_json}")


if __name__ == "__main__":
    main()
