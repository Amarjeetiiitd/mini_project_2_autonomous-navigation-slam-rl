"""Learning curves from SB3 Monitor logs:  python -m rl_training.plot_training --log-dir logs/ppo_navigation"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_monitors(log_dir: str) -> pd.DataFrame:
    frames = []
    for f in sorted(glob.glob(str(Path(log_dir) / "monitor_*.monitor.csv"))):
        df = pd.read_csv(f, skiprows=1)
        df["env"] = Path(f).stem
        frames.append(df)
    if not frames:
        raise SystemExit(f"no monitor files in {log_dir}")
    df = pd.concat(frames, ignore_index=True)
    df["steps"] = df.groupby("env")["l"].cumsum()
    return df.sort_values("steps")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="logs/ppo_navigation")
    ap.add_argument("--window", type=int, default=100)
    ap.add_argument("--out", default="results/figures/training_curves.png")
    a = ap.parse_args()
    df = load_monitors(a.log_dir)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for axi, col, title in zip(ax, ["r", "is_success", "collision"],
                               ["Episode return", "Success rate (train)", "Collision rate (train)"]):
        axi.plot(df["steps"], df[col].rolling(a.window, min_periods=10).mean())
        axi.set_title(f"{title} (rolling {a.window} eps)")
        axi.set_xlabel("env steps (per-env counter)")
        axi.grid(alpha=0.3)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
