"""Experiment E5 (+E3/E4/E6-synthetic): compare controllers on *identical* scenarios.

    python -m rl_training.compare --model models/ppo_navigation --episodes 100

For each condition in config.CONDITIONS (nominal / lidar_noise / loc_drift) and each
split (test, heldout) every controller runs the same scenario list. Outputs:

    results/compare_episodes.csv   per-episode rows
    results/compare_summary.csv    per controller x split x condition, with 95% CIs
    results/compare_summary.md     the same as a Markdown table
    results/figures/compare_*.png  success-rate / SPL bar charts with CIs

NOTE: perturbations here are synthetic. Localisation drift is a random walk added to
the pose the controller believes; it is NOT a SLAM result. SLAM-specific numbers must
come from the ROS 2 / Gazebo runs (see docs/EXPERIMENTS.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .baselines import DWAController, ReactiveController, evaluate_controller  # noqa: E402
from .evaluate import load_policy  # noqa: E402
from .metrics import summarize  # noqa: E402
from .scenarios import make_scenarios  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="PPO model path (omit for baselines only)")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--splits", nargs="+", default=["test", "heldout"])
    ap.add_argument("--conditions", nargs="+", default=list(C.CONDITIONS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    out = Path(a.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ppo, mode = (load_policy(a.model) if a.model else (None, "waypoint"))
    controllers = [("reactive", ReactiveController(), {}), ("dwa_global", DWAController(), {})]
    if ppo is not None:
        controllers.append((f"ppo_{mode}", ppo, {"goal_mode": mode}))

    frames = []
    for split in a.splits:
        scns = make_scenarios(a.episodes, split, a.seed)
        for cond in a.conditions:
            for name, ctrl, kw in controllers:
                print(f"[{split}/{cond}] {name} ...", flush=True)
                df = evaluate_controller(ctrl, scns, {**kw, **C.CONDITIONS[cond]}, label=name)
                df["split"], df["condition"] = split, cond
                frames.append(df)
    ep = pd.concat(frames, ignore_index=True)
    ep.to_csv(out / "compare_episodes.csv", index=False)

    rows = []
    for (split, cond, name), g in ep.groupby(["split", "condition", "controller"], sort=False):
        rows.append({"split": split, "condition": cond, "controller": name, **summarize(g)})
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "compare_summary.csv", index=False)

    # per-family breakdown (nominal only)
    fam = []
    for (split, name, family), g in ep[ep.condition == "nominal"].groupby(["split", "controller", "family"]):
        s = summarize(g)
        fam.append({"split": split, "controller": name, "family": family, "n": s["n"],
                    "success_rate": s["success_rate"], "spl": s["spl"]})
    pd.DataFrame(fam).to_csv(out / "compare_by_family.csv", index=False)

    md = ["| split | condition | controller | n | success (95% CI) | collision | timeout | SPL (95% CI) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['split']} | {r['condition']} | {r['controller']} | {r['n']} | "
                  f"{r['success_rate']:.3f} [{r['success_ci_lo']:.3f}, {r['success_ci_hi']:.3f}] | "
                  f"{r['collision_rate']:.3f} | {r['timeout_rate']:.3f} | "
                  f"{r['spl']:.3f} [{r['spl_ci_lo']:.3f}, {r['spl_ci_hi']:.3f}] |")
    (out / "compare_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # figures
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
