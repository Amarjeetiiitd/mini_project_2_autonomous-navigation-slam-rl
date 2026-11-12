"""Evaluate and compare PPO checkpoints on the disjoint validation split.

Implements the pre-registered checkpoint selection protocol:
1. Load candidates: final 1M-step checkpoint ('models/ppo_navigation.zip') and
   callback validation peak checkpoint ('models/best/best_model.zip').
2. Evaluate on 50 disjoint validation scenarios (VAL_SEEDS: 1000-1049, seed 0).
3. Selection criterion: Validation Success Rate (tie-break: SPL and validation reward).
4. Freeze selected checkpoint and record SHA-256 and selection audit in results/checkpoint_selection.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import rl_training.config as C
from rl_training.baselines import evaluate_controller
from rl_training.evaluate import load_policy
from rl_training.metrics import summarize
from rl_training.scenarios import make_scenarios


def get_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def evaluate_checkpoint(path: Path | str, scenarios, goal_mode: str | None = None) -> dict:
    ctrl, mode = load_policy(str(path), goal_mode)
    df = evaluate_controller(ctrl, scenarios, {"goal_mode": mode})
    s = summarize(df)
    return {
        "model_path": str(path),
        "sha256": get_sha256(path),
        "success_rate": round(float(s["success_rate"]), 4),
        "collision_rate": round(float(s["collision_rate"]), 4),
        "timeout_rate": round(float(s["timeout_rate"]), 4),
        "spl": round(float(s["spl"]), 4),
        "episodes": int(s["n"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-model", default="models/ppo_navigation.zip")
    p.add_argument("--best-model", default="models/best/best_model.zip")
    p.add_argument("--val-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/checkpoint_selection.json")
    a = p.parse_args()

    print(f"Generating {a.val_episodes} validation scenarios on split 'val' (seeds 1000..)...")
    val_scns = make_scenarios(a.val_episodes, "val", seed=a.seed)

    print(f"Evaluating Candidate 1: Final checkpoint ({a.final_model})...")
    res_final = evaluate_checkpoint(a.final_model, val_scns)
    print(f"  Final model val success: {res_final['success_rate']*100:.1f}%, collision: {res_final['collision_rate']*100:.1f}%, SPL: {res_final['spl']:.4f}")

    print(f"Evaluating Candidate 2: Best callback checkpoint ({a.best_model})...")
    res_best = evaluate_checkpoint(a.best_model, val_scns)
    print(f"  Best model val success:  {res_best['success_rate']*100:.1f}%, collision: {res_best['collision_rate']*100:.1f}%, SPL: {res_best['spl']:.4f}")

    # Decision rule: Validation success rate first, then SPL
    if res_best["success_rate"] > res_final["success_rate"]:
        winner = "best_model"
        winner_path = a.best_model
        rationale = f"Higher validation success rate ({res_best['success_rate']*100:.1f}% vs {res_final['success_rate']*100:.1f}%)"
    elif res_best["success_rate"] < res_final["success_rate"]:
        winner = "final_model"
        winner_path = a.final_model
        rationale = f"Higher validation success rate ({res_final['success_rate']*100:.1f}% vs {res_best['success_rate']*100:.1f}%)"
    else:
        # Tie-break by SPL
        if res_best["spl"] >= res_final["spl"]:
            winner = "best_model"
            winner_path = a.best_model
            rationale = f"Equal validation success rate; higher SPL ({res_best['spl']:.4f} vs {res_final['spl']:.4f})"
        else:
            winner = "final_model"
            winner_path = a.final_model
            rationale = f"Equal validation success rate; higher SPL ({res_final['spl']:.4f} vs {res_best['spl']:.4f})"

    print(f"\n[DECISION] Selected Checkpoint: {winner} ({winner_path})")
    print(f"Rationale: {rationale}")

    record = {
        "selection_metric": "validation_success_rate (tie-break: SPL)",
        "validation_split": "val",
        "validation_layout_seeds": [C.VAL_SEEDS[0], C.VAL_SEEDS[-1]],
        "validation_episodes": a.val_episodes,
        "selected_candidate": winner,
        "selected_model_path": winner_path,
        "selected_model_sha256": get_sha256(winner_path),
        "rationale": rationale,
        "candidates": {
            "final_1m_steps": res_final,
            "callback_best_300k": res_best,
        },
    }

    out_file = Path(a.out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved checkpoint selection audit to: {out_file}")


if __name__ == "__main__":
    main()
