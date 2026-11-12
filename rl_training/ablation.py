"""Component Ablation Study for Safety-Aware Reinforcement Learning Navigation.

Isolates individual contributions of each safety layer:
1. Raw PPO (Unshielded Waypoint Tracking)
2. Layer 2: Predictive Stopping-Distance Governor (v <= sqrt(2*a*d))
3. Layer 2+3: Stopping Governor + Lateral Repulsion & Doorway Midline Centering
4. Layer 2+4: Stopping Governor + Stagnation Watchdog & Reverse Recovery
5. Full MAS-RL: Layers 1-4 combined
6. Classical DWA + Geodesic Global Planner

Computes paired McNemar's exact tests against unshielded PPO with Holm-Bonferroni FWER
corrections and outputs results/ablation_summary.csv.
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
from .metrics import holm_bonferroni_correction, summarize
from .scenarios import make_scenarios


def compute_mcnemar_pvalue(raw_successes: np.ndarray, ctrl_successes: np.ndarray) -> float:
    """Exact two-sided McNemar test using binomial distribution on discordant pairs.

    Discordant pairs:
    b: True in ctrl, False in raw (ctrl-only win)
    c: False in ctrl, True in raw (raw-only win)
    """
    b = int(np.sum((ctrl_successes == 1) & (raw_successes == 0)))
    c = int(np.sum((ctrl_successes == 0) & (raw_successes == 1)))
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0
    res = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
    return float(res.pvalue)


def generate_ablation_from_episodes(episodes_csv: str = "results/compare_episodes.csv",
                                   out_csv: str = "results/ablation_summary.csv") -> pd.DataFrame:
    """Generate canonical ablation summary directly from matched compare_episodes.csv."""
    df = pd.read_csv(episodes_csv)
    nominal = df[df["condition"] == "nominal"]

    ctrl_order = [
        "ppo_waypoint",
        "ppo_governor",
        "ppo_gov_steering",
        "ppo_gov_recovery",
        "mas_rl",
        "dwa_global",
    ]

    summary_rows = []
    for split in ["test", "heldout"]:
        sub_split = nominal[nominal["split"] == split]
        # Get raw baseline successes indexed by scenario
        raw_df = sub_split[sub_split["controller"] == "ppo_waypoint"].sort_values("scenario")
        if raw_df.empty:
            continue
        raw_succ = raw_df["success"].to_numpy(dtype=int)

        p_vals = []
        split_records = []
        for name in ctrl_order:
            ctrl_df = sub_split[sub_split["controller"] == name].sort_values("scenario")
            if ctrl_df.empty:
                continue
            s = summarize(ctrl_df)
            ctrl_succ = ctrl_df["success"].to_numpy(dtype=int)
            p_val = compute_mcnemar_pvalue(raw_succ, ctrl_succ) if name != "ppo_waypoint" else 1.0
            p_vals.append(p_val)

            split_records.append({
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

        # Apply Holm-Bonferroni correction across comparators
        adj_p, reject = holm_bonferroni_correction(p_vals, alpha=0.05)
        for r, adj, rej in zip(split_records, adj_p, reject):
            adj_val = float(adj)
            r["mcnemar_p_holm"] = adj_val
            r["significant_holm_fwer"] = bool(rej and r["controller"] != "ppo_waypoint")
            summary_rows.append(r)

    out_df = pd.DataFrame(summary_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved canonical component ablation summary to: {out_csv}")
    return out_df


def run_ablation(
    model_path: str = "models/ppo_navigation",
    episodes: int = 100,
    splits: list[str] | None = None,
    seed: int = 0,
    out_csv: str = "results/ablation_summary.csv",
) -> pd.DataFrame:
    splits = splits or ["test", "heldout"]
    model = PPO.load(model_path, device="cpu")

    controllers = [
        ("ppo_waypoint", PPOController(model, "ppo_waypoint")),
        ("ppo_governor", GovernorPPOController(model, "ppo_governor")),
        ("ppo_gov_steering", PPOBrakingSteeringController(model, "ppo_gov_steering")),
        ("ppo_gov_recovery", PPOBrakingRecoveryController(model, "ppo_gov_recovery")),
        ("mas_rl", MASRLController(model, "mas_rl")),
        ("dwa_global", DWAController()),
    ]

    all_rows = []
    for split in splits:
        print(f"\n============================================================")
        print(f"Running Safety Layer Ablation Study: split='{split}', N={episodes}")
        print(f"============================================================")
        scenarios = make_scenarios(episodes, split, seed)

        dfs = {}
        for name, ctrl in controllers:
            print(f"Evaluating {name:18s} ...", end="", flush=True)
            df_ctrl = evaluate_controller(ctrl, scenarios, {"goal_mode": "waypoint", **C.CONDITIONS["nominal"]}, label=name)
            dfs[name] = df_ctrl
            s = summarize(df_ctrl)
            print(f" Success: {s['success_rate']*100:.1f}% | Coll: {s['collision_rate']*100:.1f}% | "
                  f"Timeout: {s['timeout_rate']*100:.1f}% | SPL: {s['spl']:.3f}")

        raw_succ = dfs["ppo_waypoint"]["success"].to_numpy(dtype=int)
        p_vals = []
        split_records = []
        for name, _ in controllers:
            df_ctrl = dfs[name]
            s = summarize(df_ctrl)
            ctrl_succ = df_ctrl["success"].to_numpy(dtype=int)
            p_val = compute_mcnemar_pvalue(raw_succ, ctrl_succ) if name != "ppo_waypoint" else 1.0
            p_vals.append(p_val)

            split_records.append({
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

        adj_p, reject = holm_bonferroni_correction(p_vals, alpha=0.05)
        for r, adj, rej in zip(split_records, adj_p, reject):
            adj_val = float(adj)
            r["mcnemar_p_holm"] = adj_val
            r["significant_holm_fwer"] = bool(rej and r["controller"] != "ppo_waypoint")
            all_rows.append(r)

    out_df = pd.DataFrame(all_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved component ablation summary to: {out_csv}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Run 5-Way Safety-Component Ablation.")
    parser.add_argument("--model", default="models/ppo_navigation")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--splits", nargs="+", default=["test", "heldout"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from-episodes", action="store_true", default=False,
                        help="Generate summary directly from existing compare_episodes.csv")
    parser.add_argument("--out", default="results/ablation_summary.csv")
    args = parser.parse_args()

    if args.from_episodes or Path("results/compare_episodes.csv").exists():
        generate_ablation_from_episodes("results/compare_episodes.csv", args.out)
    else:
        run_ablation(args.model, args.episodes, args.splits, args.seed, args.out)


if __name__ == "__main__":
    main()
