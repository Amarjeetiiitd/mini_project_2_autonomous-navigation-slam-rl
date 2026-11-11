"""Plot trajectories of controllers on the same scenarios (for the report).

    python -m rl_training.visualize --model models/ppo_navigation --split heldout --n 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from . import config as C  # noqa: E402
from .baselines import (DWAController, GovernorPPOController, MASRLController,
                        ReactiveController, run_episode)  # noqa: E402
from .evaluate import load_policy  # noqa: E402
from .maps import get_bundle  # noqa: E402
from .nav_env import NavEnv  # noqa: E402
from .scenarios import make_scenarios  # noqa: E402


def draw(ax, scn, traj, title):
    w = get_bundle(scn.layout_seed, scn.family_override).world
    for x0, y0, x1, y1 in w.rects:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="0.25"))
    ax.plot(traj[:, 0], traj[:, 1], "-", lw=1.5, color="tab:blue")
    ax.add_patch(Circle(scn.start[:2], C.ROBOT_RADIUS, color="tab:green", alpha=0.6))
    ax.add_patch(Circle(scn.goal, C.GOAL_TOLERANCE, color="tab:red", alpha=0.6))
    ax.set_xlim(0, w.width); ax.set_ylim(0, w.height); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--split", default="test", choices=["test", "heldout", "val"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    scns = make_scenarios(a.n, a.split, a.seed)
    ctrls = [("reactive", ReactiveController(), {}), ("dwa_global", DWAController(), {})]
    if a.model:
        ppo, mode = load_policy(a.model)
        ctrls.append((f"ppo_{mode}", ppo, {"goal_mode": mode}))
        ctrls.append(("ppo_governor", GovernorPPOController(ppo.model), {"goal_mode": mode}))
        ctrls.append(("mas_rl", MASRLController(ppo.model), {"goal_mode": mode}))
    out_file = a.out or f"results/figures/trajectories_{a.split}.png"
    fig, axes = plt.subplots(len(ctrls), a.n, figsize=(2.6 * a.n, 2.6 * len(ctrls)), squeeze=False)
    for r, (name, ctrl, kw) in enumerate(ctrls):
        env = NavEnv(**kw)
        for c, scn in enumerate(scns):
            res = run_episode(env, ctrl, scn, seed=scn.sid, record=True)
            tag = "OK" if res["success"] else ("COLL" if res["collision"] else "TIME")
            draw(axes[r, c], scn, res["traj"], f"{name} #{scn.sid} {tag}")
    fig.tight_layout()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=130)
    print(f"saved {out_file}")


if __name__ == "__main__":
    main()
