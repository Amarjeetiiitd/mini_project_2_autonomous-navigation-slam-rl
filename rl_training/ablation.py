"""Component Ablation Study for Safety-Aware Reinforcement Learning Navigation.

Isolates individual contributions of each safety layer:
1. Raw PPO (Unshielded Waypoint Tracking)
2. Layer 2: Predictive Stopping-Distance Governor (v <= sqrt(2*a*d))
3. Layer 2+3: Stopping Governor + Lateral Repulsion & Doorway Midline Centering
4. Layer 2+4: Stopping Governor + Stagnation Watchdog & Reverse Recovery
5. Full MAS-RL: Layers 1-4 combined
6. Classical DWA + Geodesic Global Planner

Computes:
1. Paired McNemar exact tests against unshielded PPO with Holm-Bonferroni FWER corrections (ablation_summary.csv).
2. Pairwise intermediate layer McNemar tests with Holm-Bonferroni corrections (ablation_pairwise_tests.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import PPO

import rl_training.config as C
from rl_training.baselines import (
    DWAController,
    GovernorPPOController,
    MASRLController,
    PPOBrakingRecoveryController,
    PPOBrakingSteeringController,
    PPOController,
    evaluate_controller,
)
from rl_training.metrics import holm_bonferroni_correction, summarize
from rl_training.scenarios import make_scenarios


def compute_mcnemar_pvalue(succ_a: np.ndarray, succ_b: np.ndarray) -> tuple[float, int, int]:
    """Exact two-sided McNemar test using binomial distribution on discordant pairs.

    Returns:
    (pvalue, b, c) where:
    b: True in B, False in A (B-only win)
    c: False in B, True in A (A-only win)
    """
    b = int(np.sum((succ_b == 1) & (succ_a == 0)))
    c = int(np.sum((succ_b == 0) & (succ_a == 1)))
    n_discordant = b + c
    if n_discordant == 0:
        return 1.0, b, c
    res = stats.binomtest(b, n_discordant, 0.5, alternative="two-sided")
    return float(res.pvalue), b, c


def compute_pairwise_ablation_tests(df_nominal: pd.DataFrame,
                                   out_csv: str = "results/ablation_pairwise_tests.csv") -> pd.DataFrame:
    """Compute exact paired McNemar tests between intermediate safety layers.

    Evaluates:
    1. Governor vs Governor + Steering
    2. Governor vs Governor + Recovery
    3. Governor + Steering vs Full MAS-RL
    4. Governor + Recovery vs Full MAS-RL
    5. Governor vs Full MAS-RL
    """
    pairs = [
        ("ppo_governor", "ppo_gov_steering", "Governor vs Governor+Steering"),
        ("ppo_governor", "ppo_gov_recovery", "Governor vs Governor+Recovery"),
        ("ppo_gov_steering", "mas_rl", "Governor+Steering vs Full MAS-RL"),
        ("ppo_gov_recovery", "mas_rl", "Governor+Recovery vs Full MAS-RL"),
        ("ppo_governor", "mas_rl", "Governor vs Full MAS-RL"),
    ]

    records = []
    for split in ["test", "heldout"]:
        sub = df_nominal[df_nominal["split"] == split]
        ctrl_succs = {}
        for name in ["ppo_governor", "ppo_gov_steering", "ppo_gov_recovery", "mas_rl"]:
            c_df = sub[sub["controller"] == name].sort_values("scenario")
            if not c_df.empty:
                ctrl_succs[name] = c_df["success"].to_numpy(dtype=int)

        split_pvals = []
        split_rows = []
        for ctrl_a, ctrl_b, desc in pairs:
            if ctrl_a in ctrl_succs and ctrl_b in ctrl_succs:
                p_val, b, c = compute_mcnemar_pvalue(ctrl_succs[ctrl_a], ctrl_succs[ctrl_b])
                succ_a = float(np.mean(ctrl_succs[ctrl_a]))
                succ_b = float(np.mean(ctrl_succs[ctrl_b]))
                split_pvals.append(p_val)
                split_rows.append({
                    "split": split,
                    "comparison": desc,
                    "controller_a": ctrl_a,
                    "controller_b": ctrl_b,
                    "success_a": succ_a,
                    "success_b": succ_b,
                    "delta_b_minus_a": round(succ_b - succ_a, 4),
                    "discordant_b_wins": b,
                    "discordant_a_wins": c,
                    "mcnemar_p_unadjusted": p_val,
                })

        if split_pvals:
            adj_p, reject = holm_bonferroni_correction(split_pvals, alpha=0.05)
            for r, adj, rej in zip(split_rows, adj_p, reject):
                r["mcnemar_p_holm"] = float(adj)
                r["significant_holm_fwer"] = bool(rej)
                records.append(r)

    res_df = pd.DataFrame(records)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"Saved pairwise ablation tests to: {out_csv}")
    return res_df


def generate_ablation_from_episodes(episodes_csv: str = "results/compare_episodes.csv",
                                    out_csv: str = "results/ablation_summary.csv",
                                    out_pairwise_csv: str = "results/ablation_pairwise_tests.csv") -> pd.DataFrame:
    """Generate canonical ablation summary and pairwise tests directly from matched compare_episodes.csv."""
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
            p_val, _, _ = compute_mcnemar_pvalue(raw_succ, ctrl_succ) if name != "ppo_waypoint" else (1.0, 0, 0)
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
            summary_rows.append(r)

    out_df = pd.DataFrame(summary_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved canonical component ablation summary to: {out_csv}")

    # Generate pairwise tests
    compute_pairwise_ablation_tests(nominal, out_pairwise_csv)
    return out_df


def run_ablation(
    model_path: str = "models/best/best_model.zip",
    fallback_model: str = "models/ppo_navigation.zip",
    episodes: int = 100,
    splits: list[str] | None = None,
    seed: int = 0,
    out_csv: str = "results/ablation_summary.csv",
    out_pairwise_csv: str = "results/ablation_pairwise_tests.csv",
) -> pd.DataFrame:
    splits = splits or ["test", "heldout"]
    target_path = model_path if Path(model_path).exists() else fallback_model
    print(f"Loading ablation model from: {target_path}")
    model = PPO.load(target_path, device="cpu")

    controllers = [
        ("ppo_waypoint", PPOController(model, "ppo_waypoint")),
        ("ppo_governor", GovernorPPOController(model, "ppo_governor")),
        ("ppo_gov_steering", PPOBrakingSteeringController(model, "ppo_gov_steering")),
        ("ppo_gov_recovery", PPOBrakingRecoveryController(model, "ppo_gov_recovery")),
        ("mas_rl", MASRLController(model, "mas_rl")),
        ("dwa_global", DWAController()),
    ]

    all_dfs = []
    for split in splits:
        print(f"\n============================================================")
        print(f"Running Safety Layer Ablation Study: split='{split}', N={episodes}")
        print(f"============================================================")
        scenarios = make_scenarios(episodes, split, seed)

        for name, ctrl in controllers:
            print(f"Evaluating {name:18s} ...", end="", flush=True)
            df_ctrl = evaluate_controller(ctrl, scenarios, {"goal_mode": "waypoint", **C.CONDITIONS["nominal"]}, label=name)
            df_ctrl["split"] = split
            df_ctrl["condition"] = "nominal"
            all_dfs.append(df_ctrl)
            s = summarize(df_ctrl)
            print(f" Success: {s['success_rate']*100:.1f}% | Coll: {s['collision_rate']*100:.1f}% | "
                  f"Timeout: {s['timeout_rate']*100:.1f}% | SPL: {s['spl']:.3f}")

    combined_df = pd.concat(all_dfs, ignore_index=True)

    summary_rows = []
    for split in splits:
        sub_split = combined_df[combined_df["split"] == split]
        raw_df = sub_split[sub_split["controller"] == "ppo_waypoint"].sort_values("scenario")
        raw_succ = raw_df["success"].to_numpy(dtype=int)
        p_vals = []
        split_records = []
        for name, _ in controllers:
            df_ctrl = sub_split[sub_split["controller"] == name].sort_values("scenario")
            s = summarize(df_ctrl)
            ctrl_succ = df_ctrl["success"].to_numpy(dtype=int)
            p_val, _, _ = compute_mcnemar_pvalue(raw_succ, ctrl_succ) if name != "ppo_waypoint" else (1.0, 0, 0)
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
            summary_rows.append(r)

    out_df = pd.DataFrame(summary_rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved component ablation summary to: {out_csv}")

    compute_pairwise_ablation_tests(combined_df, out_pairwise_csv)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Run 5-Way Safety-Component Ablation.")
    parser.add_argument("--model", default="models/best/best_model.zip")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--splits", nargs="+", default=["test", "heldout"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from-episodes", action="store_true", default=False,
                        help="Generate summary directly from existing compare_episodes.csv")
    parser.add_argument("--run", action="store_true", default=False,
                        help="Force execution of fresh ablation simulations")
    parser.add_argument("--out", default="results/ablation_summary.csv")
    parser.add_argument("--out-pairwise", default="results/ablation_pairwise_tests.csv")
    args = parser.parse_args()

    if args.run:
        run_ablation(args.model, episodes=args.episodes, splits=args.splits, seed=args.seed,
                     out_csv=args.out, out_pairwise_csv=args.out_pairwise)
    elif args.from_episodes or Path("results/compare_episodes.csv").exists():
        print("Using existing results/compare_episodes.csv for canonical ablation synchronization...")
        generate_ablation_from_episodes("results/compare_episodes.csv", args.out, args.out_pairwise)
    else:
        run_ablation(args.model, episodes=args.episodes, splits=args.splits, seed=args.seed,
                     out_csv=args.out, out_pairwise_csv=args.out_pairwise)


if __name__ == "__main__":
    main()
