"""Experiment E5 (+E3/E4/E6-synthetic): compare controllers on *identical* scenarios.

    python -m rl_training.compare --model models/ppo_navigation --episodes 100

For each condition in config.CONDITIONS (nominal / lidar_noise / loc_drift) and each
split (test, heldout) every controller runs the same scenario list. Outputs:

    results/compare_episodes.csv          per-episode rows
    results/compare_summary.csv           per controller x split x condition, with 95% CIs
    results/compare_summary.md            the same as a Markdown table
    results/paired_statistical_tests.csv  paired McNemar & bootstrap difference tests
    results/manifest_test.csv             scenario manifest for test split
    results/manifest_heldout.csv          scenario manifest for held-out split
    results/experiment_fingerprint.json   configuration hash & seed tracking
    results/figures/compare_*.png         success-rate / SPL bar charts with CIs

NOTE: perturbations here are synthetic. Localisation drift is a random walk added to
the pose the controller believes; it is NOT a SLAM result. SLAM-specific numbers must
come from the ROS 2 / Gazebo runs (see docs/EXPERIMENTS.md).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .baselines import (DWAController, GovernorPPOController, MASRLController,
                        PPOBrakingRecoveryController, PPOBrakingSteeringController,
                        ReactiveController, evaluate_controller)  # noqa: E402
from .evaluate import load_policy  # noqa: E402
from .metrics import (clustered_bootstrap_ci, mcnemar_test, paired_bootstrap_ci,
                      summarize)  # noqa: E402
from .scenarios import make_scenarios  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/ppo_navigation.zip", help="PPO model path (omit for baselines only)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["test", "heldout"])
    ap.add_argument("--conditions", nargs="+", default=list(C.CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--merge-existing", action="store_true", default=False,
                    help="Attempt to reuse existing runs if configuration fingerprint and seeds match")
    ap.add_argument("--include-ablations", action="store_true", default=False,
                    help="Include intermediate ablation controllers (ppo_gov_steering, ppo_gov_recovery)")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    ppo, mode = (load_policy(a.model) if a.model and Path(a.model).exists() else (None, "waypoint"))
    controllers = [("reactive", ReactiveController(), {}), ("dwa_global", DWAController(), {})]
    if ppo is not None:
        controllers.append((f"ppo_{mode}", ppo, {"goal_mode": mode}))
        controllers.append(("ppo_governor", GovernorPPOController(ppo.model), {"goal_mode": mode}))
        if a.include_ablations:
            controllers.append(("ppo_gov_steering", PPOBrakingSteeringController(ppo.model), {"goal_mode": mode}))
            controllers.append(("ppo_gov_recovery", PPOBrakingRecoveryController(ppo.model), {"goal_mode": mode}))
        controllers.append(("mas_rl", MASRLController(ppo.model), {"goal_mode": mode}))

    # Save scenario manifests & track unique seeds
    scenarios_by_split = {}
    unique_seeds_by_split = {}
    for split in a.splits:
        scns = make_scenarios(a.episodes, split, a.seed)
        scenarios_by_split[split] = scns
        unique_seeds = sorted(list(set(s.layout_seed for s in scns)))
        unique_seeds_by_split[split] = len(unique_seeds)

        manifest_rows = []
        for s in scns:
            manifest_rows.append({
                "scenario_id": s.sid,
                "split": split,
                "layout_seed": s.layout_seed,
                "family": s.family_override or "procedural",
                "start_x": s.start[0],
                "start_y": s.start[1],
                "start_yaw": s.start[2],
                "goal_x": s.goal[0],
                "goal_y": s.goal[1],
                "shortest_len": s.shortest_len,
            })
        pd.DataFrame(manifest_rows).to_csv(out / f"manifest_{split}.csv", index=False)
        print(f"[{split}] Saved manifest with {len(scns)} scenarios ({len(unique_seeds)} unique layout seeds).")

    # Record experiment configuration fingerprint
    fingerprint = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "episodes_per_condition": a.episodes,
        "splits": a.splits,
        "conditions": a.conditions,
        "controllers": [c[0] for c in controllers],
        "unique_layout_seeds": unique_seeds_by_split,
        "test_seeds_config": {"start": C.TEST_SEEDS[0], "end": C.TEST_SEEDS[-1], "count": len(C.TEST_SEEDS)},
        "heldout_seeds_config": {"start": C.HELDOUT_SEEDS[0], "end": C.HELDOUT_SEEDS[-1], "count": len(C.HELDOUT_SEEDS)},
        "robot_kinematics": {
            "robot_radius": C.ROBOT_RADIUS,
            "v_max": C.V_MAX,
            "w_max": C.W_MAX,
            "a_v_max": C.A_V_MAX,
            "a_w_max": C.A_W_MAX,
            "reaction_time_s": 0.15,
            "max_recovery_attempts": 4,
            "min_sensor_coverage": 0.50,
        },
        "model_path": str(a.model),
    }
    (out / "experiment_fingerprint.json").write_text(json.dumps(fingerprint, indent=2))
    print(f"Saved experiment configuration fingerprint: {out / 'experiment_fingerprint.json'}")

    existing_df = None
    if a.merge_existing and (out / "compare_episodes.csv").exists():
        existing_df = pd.read_csv(out / "compare_episodes.csv")

    frames = []
    for split in a.splits:
        scns = scenarios_by_split[split]
        expected_seeds = [s.layout_seed for s in scns]
        for cond in a.conditions:
            for name, ctrl, kw in controllers:
                if existing_df is not None and not existing_df.empty:
                    match = existing_df[(existing_df.split == split) &
                                        (existing_df.condition == cond) &
                                        (existing_df.controller == name)]
                    if len(match) >= a.episodes:
                        matched_seeds = match.iloc[:a.episodes]["layout_seed"].tolist()
                        if matched_seeds == expected_seeds:
                            print(f"[{split}/{cond}] {name} matches current 100-seed manifest; reusing {len(match)} episodes.", flush=True)
                            frames.append(match.iloc[:a.episodes].copy())
                            continue
                        else:
                            print(f"[{split}/{cond}] {name}: Stale seeds detected in existing CSV ({len(set(matched_seeds))} vs {len(set(expected_seeds))} unique); regenerating.", flush=True)

                print(f"[{split}/{cond}] Evaluating {name} ({len(scns)} scenarios)...", flush=True)
                df = evaluate_controller(ctrl, scns, {**kw, **C.CONDITIONS[cond]}, label=name)
                df["split"], df["condition"] = split, cond
                frames.append(df)

    ep = pd.concat(frames, ignore_index=True)
    ep.to_csv(out / "compare_episodes.csv", index=False)
    print(f"Saved: {out / 'compare_episodes.csv'} ({len(ep)} total episodes)")

    rows = []
    for (split, cond, name), g in ep.groupby(["split", "condition", "controller"], sort=False):
        s = summarize(g)
        # Also compute layout-clustered bootstrap SPL interval
        clust_lo, clust_hi = clustered_bootstrap_ci(g, value_col="spl", cluster_col="layout_seed")
        rows.append({
            "split": split,
            "condition": cond,
            "controller": name,
            **s,
            "spl_cluster_ci_lo": clust_lo,
            "spl_cluster_ci_hi": clust_hi,
        })
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "compare_summary.csv", index=False)
    print(f"Saved: {out / 'compare_summary.csv'}")

    # Compute paired statistical hypothesis tests (MAS-RL vs baselines on matched episodes)
    paired_rows = []
    for split in a.splits:
        for cond in a.conditions:
            sub = ep[(ep.split == split) & (ep.condition == cond)]
            mas_ep = sub[sub.controller == "mas_rl"].sort_values("scenario")
            if mas_ep.empty:
                continue

            for baseline_name in [c for c in dict.fromkeys(sub.controller) if c != "mas_rl"]:
                base_ep = sub[sub.controller == baseline_name].sort_values("scenario")
                if len(mas_ep) != len(base_ep) or len(mas_ep) == 0:
                    continue

                mcn = mcnemar_test(mas_ep["success"].to_numpy(), base_ep["success"].to_numpy())
                boot_spl = paired_bootstrap_ci(mas_ep["spl"].to_numpy(), base_ep["spl"].to_numpy())

                paired_rows.append({
                    "split": split,
                    "condition": cond,
                    "comparator": f"mas_rl vs {baseline_name}",
                    "mas_rl_success": float(mas_ep["success"].mean()),
                    "base_success": float(base_ep["success"].mean()),
                    "b_mas_only_wins": mcn["b_wins"],
                    "c_base_only_wins": mcn["c_wins"],
                    "mcnemar_p_exact": mcn["p_value_exact"],
                    "success_diff_significant": mcn["significant"],
                    "mean_spl_diff": round(boot_spl["mean_diff"], 4),
                    "spl_diff_ci_lo": round(boot_spl["ci_lo"], 4),
                    "spl_diff_ci_hi": round(boot_spl["ci_hi"], 4),
                    "spl_diff_significant": boot_spl["significant"],
                })

    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(out / "paired_statistical_tests.csv", index=False)
    print(f"Saved: {out / 'paired_statistical_tests.csv'}")

    # Per-family breakdown (nominal only)
    fam = []
    for (split, name, family), g in ep[ep.condition == "nominal"].groupby(["split", "controller", "family"]):
        s = summarize(g)
        fam.append({"split": split, "controller": name, "family": family, "n": s["n"],
                    "success_rate": s["success_rate"], "spl": s["spl"]})
    pd.DataFrame(fam).to_csv(out / "compare_by_family.csv", index=False)

    md = [
        "# Benchmark Evaluation Summary (Full Factorial Matrix)",
        "",
        "| split | condition | controller | n | success (95% Wilson CI) | collision | timeout | SPL (95% CI) | SPL (Clustered CI) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(f"| {r['split']} | {r['condition']} | {r['controller']} | {r['n']} | "
                  f"{r['success_rate']:.3f} [{r['success_ci_lo']:.3f}, {r['success_ci_hi']:.3f}] | "
                  f"{r['collision_rate']:.3f} | {r['timeout_rate']:.3f} | "
                  f"{r['spl']:.3f} [{r['spl_ci_lo']:.3f}, {r['spl_ci_hi']:.3f}] | "
                  f"[{r['spl_cluster_ci_lo']:.3f}, {r['spl_cluster_ci_hi']:.3f}] |")
    (out / "compare_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # Generate figures
    for split in a.splits:
        d = summ[summ.split == split]
        conds = [c for c in a.conditions if c in set(d.condition)]
        names = list(dict.fromkeys(d.controller))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        w = 0.8 / max(len(names), 1)
        for ax, key, lo, hi, title in ((axes[0], "success_rate", "success_ci_lo", "success_ci_hi", "Success rate"),
                                       (axes[1], "spl", "spl_ci_lo", "spl_ci_hi", "SPL")):
            for i, n in enumerate(names):
                dd = d[d.controller == n].set_index("condition").loc[conds]
                y = dd[key].to_numpy()
                err = np.vstack([y - dd[lo].to_numpy(), dd[hi].to_numpy() - y])
                ax.bar(np.arange(len(conds)) + i * w, y, w, yerr=err, capsize=3, label=n)
            ax.set_xticks(np.arange(len(conds)) + w * (len(names) - 1) / 2)
            ax.set_xticklabels(conds)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{title} - {split} layouts")
            ax.grid(axis="y", alpha=0.3)
        axes[0].legend()
        fig.tight_layout()
        fig.savefig(out / "figures" / f"compare_{split}.png", dpi=150)
        plt.close(fig)
    print(f"results written to {out}/")


if __name__ == "__main__":
    main()
