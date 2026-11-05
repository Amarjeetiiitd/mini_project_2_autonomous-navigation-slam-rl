"""Statistics helpers: Wilson interval for proportions, bootstrap CI for means."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def summarize(df: pd.DataFrame) -> dict:
    """One row of summary statistics for a block of per-episode results."""
    n = len(df)
    k = int(df["success"].sum())
    lo, hi = wilson_ci(k, n)
    succ = df[df["success"]]
    spl_lo, spl_hi = bootstrap_ci(df["spl"])
    return {
        "n": n,
        "success_rate": k / n if n else float("nan"), "success_ci_lo": lo, "success_ci_hi": hi,
        "collision_rate": float(df["collision"].mean()) if n else float("nan"),
        "timeout_rate": float(df["timeout"].mean()) if n else float("nan"),
        "spl": float(df["spl"].mean()) if n else float("nan"), "spl_ci_lo": spl_lo, "spl_ci_hi": spl_hi,
        "path_len_succ": float(succ["path_length"].mean()) if len(succ) else float("nan"),
        "time_s_succ": float(succ["time_s"].mean()) if len(succ) else float("nan"),
    }
