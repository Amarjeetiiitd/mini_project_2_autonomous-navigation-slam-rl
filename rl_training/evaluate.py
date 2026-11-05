"""Evaluate a trained PPO policy on matched, held-out scenarios.

    python -m rl_training.evaluate --episodes 100 --model models/ppo_navigation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from stable_baselines3 import PPO

from .baselines import PPOController, evaluate_controller
from .metrics import summarize
from .scenarios import make_scenarios


def load_policy(model_path: str, goal_mode: str | None = None):
    """Returns (PPOController, goal_mode). goal_mode is read from <model>.json if present."""
    path = Path(model_path)
    meta_path = path.with_suffix(".json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    mode = goal_mode or meta.get("goal_mode", "waypoint")
    return PPOController(PPO.load(str(path), device="cpu"), "ppo"), mode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppo_navigation")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["val", "test", "heldout"])
    ap.add_argument("--goal-mode", default=None, choices=[None, "waypoint", "direct"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ppo_evaluation.csv")
    a = ap.parse_args()

    ctrl, mode = load_policy(a.model, a.goal_mode)
    scns = make_scenarios(a.episodes, a.split, a.seed)
    df = evaluate_controller(ctrl, scns, {"goal_mode": mode})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    s = summarize(df)
    print(f"PPO ({mode}) on '{a.split}' n={s['n']}: success={s['success_rate']:.3f} "
          f"[{s['success_ci_lo']:.3f}, {s['success_ci_hi']:.3f}]  collision={s['collision_rate']:.3f} "
          f"timeout={s['timeout_rate']:.3f}  SPL={s['spl']:.3f} [{s['spl_ci_lo']:.3f}, {s['spl_ci_hi']:.3f}]")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
