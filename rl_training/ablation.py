"""Component Ablation Study for Safety-Aware Reinforcement Learning Navigation.

Isolates individual contributions of each safety layer:
1. Raw PPO (Unshielded)
2. Layer 2: Predictive Stopping-Distance Governor only (v^2 <= 2*a*d)
3. Layer 2+3: Stopping Governor + Lateral Repulsion & Doorway Midline Centering
4. Layer 2+4: Stopping Governor + Stagnation Watchdog & Reverse Recovery
5. Full MAS-RL: Layers 1-4 combined
6. Classical DWA + Geodesic Global Planner

Computes paired McNemar's exact tests against unshielded PPO and saves results to results/ablation_summary.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO

from . import config as C
from .baselines import (
    DWAController,
    GovernorPPOController,
    MASRLController,
    PPOBrakingRecoveryController,
    PPOBrakingSteeringController,
    PPOController,
    evaluate_controller,
)
from .metrics import summarize
from .scenarios import make_scenarios


def compute_mcnemar_pvalue(b_successes: np.ndarray, a_successes: np.ndarray) -> float:
    """Exact two-sided McNemar test using binomial distribution on discordant pairs.
    
    Discordant pairs:
    b: True in model A, False in model B
    c: False in model A, True in model B
    """
    b = int(np.sum((a_successes == 1) & (b_successes == 0)))
    c = int(np.sum((a_successes == 0) & (b_successes == 1)))
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    res = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
    return float(res.pvalue)


def run_ablation(
    model_path: str = "models/ppo_navigation",
    episodes: int = 100,
    split: str = "test",
    seed: int = 0,
    out_csv: str = "results/ablation_summary.csv",
) -> pd.DataFrame:
    print(f"============================================================")
    print(f"Running Safety Layer Ablation Study: split='{split}', N={episodes}")
    print(f"============================================================")

    model = PPO.load(model_path, device="cpu")
    scenarios = make_scenarios(episodes, split, seed)

    controllers = [
        ("ppo_raw", PPOController(model, "ppo_raw")),
        ("ppo_governor", GovernorPPOController(model, "ppo_governor")),
        ("ppo_gov_steering", PPOBrakingSteeringController(model, "ppo_gov_steering")),
        ("ppo_gov_recovery", PPOBrakingRecoveryController(model, "ppo_gov_recovery")),
        ("mas_rl", MASRLController(model, "mas_rl")),
        ("dwa_global", DWAController()),
    ]

    dfs = {}
    for name, ctrl in controllers:
        print(f"Evaluating {name:18s} ...", end="", flush=True)
        df_ctrl = evaluate_controller(ctrl, scenarios, {"goal_mode": "waypoint"}, label=name)
        dfs[name] = df_ctrl
        s = summarize(df_ctrl)
        print(f" Success: {s['success_rate']*100:.1f}% | Coll: {s['collision_rate']*100:.1f}% | "
              f"Timeout: {s['timeout_rate']*100:.1f}% | SPL: {s['spl']:.3f}")

    raw_succ = dfs["ppo_raw"]["success"].to_numpy(dtype=int)

    summary_rows = []
    for name, _ in controllers:
        df_ctrl = dfs[name]
        s = summarize(df_ctrl)
        ctrl_succ = df_ctrl["success"].to_numpy(dtype=int)
        p_val = compute_mcnemar_pvalue(raw_succ, ctrl_succ) if name != "ppo_raw" else 1.0

        summary_rows.append({
            "split": split,
            "controller": name,
            "n": s["n"],
            "success_rate": s["success_rate"],
            "success_ci_lo": s["success_ci_lo"],
            "success_ci_hi": s["success_ci_hi"],
            "collision_rate": s["collision_rate"],
            "timeout_rate": s["timeout_rate"],
            "spl": s["spl"],
            "spl_ci_lo": s["spl_ci_lo"],
            "spl_ci_hi": s["spl_ci_hi"],
            "mcnemar_p_vs_raw": p_val,
        })

    out_df = pd.DataFrame(summary_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved component ablation summary to: {out_csv}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Run 5-Way Safety-Component Ablation.")
    parser.add_argument("--model", default="models/ppo_navigation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--split", default="test", choices=["val", "test", "heldout"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="results/ablation_summary.csv")
    args = parser.parse_args()

    run_ablation(args.model, args.episodes, args.split, args.seed, args.out)


if __name__ == "__main__":
    main()
