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


def mcnemar_test(succ_a, succ_b) -> dict:
    """Paired McNemar test with Edwards continuity correction and exact binomial p-value.

    Evaluates whether the discordant success/failure outcomes between two controllers
    evaluated on identical matched scenarios are statistically significant.
    """
    a = np.asarray(succ_a, dtype=bool)
    b = np.asarray(succ_b, dtype=bool)
    if len(a) != len(b):
        raise ValueError(f"Lengths must match: {len(a)} vs {len(b)}")
    b_wins = int(np.sum(a & (~b)))  # A succeeded, B failed
    c_wins = int(np.sum((~a) & b))  # A failed, B succeeded
    total_discordant = b_wins + c_wins
    if total_discordant == 0:
        return {
            "b_wins": b_wins,
            "c_wins": c_wins,
            "n_discordant": 0,
            "chi2": 0.0,
            "p_value_exact": 1.0,
            "significant": False,
        }

    chi2_stat = max(0.0, abs(b_wins - c_wins) - 1.0) ** 2 / total_discordant
    from scipy.stats import binomtest, chi2
    p_chi2 = float(chi2.sf(chi2_stat, df=1))
    p_exact = float(binomtest(min(b_wins, c_wins), total_discordant, p=0.5).pvalue)
    return {
        "b_wins": b_wins,
        "c_wins": c_wins,
        "n_discordant": total_discordant,
        "chi2": float(chi2_stat),
        "p_value_chi2": p_chi2,
        "p_value_exact": p_exact,
        "significant": p_exact < 0.05,
    }


def paired_bootstrap_ci(
    x_a, x_b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> dict:
    """Paired bootstrap for difference in continuous metrics (SPL, path length, time)."""
    a = np.asarray(x_a, dtype=float)
    b = np.asarray(x_b, dtype=float)
    diff = a - b
    diff = diff[np.isfinite(diff)]
    if len(diff) == 0:
        return {"mean_diff": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "significant": False}

    rng = np.random.default_rng(seed)
    mean_diff = float(np.mean(diff))
    boot_means = rng.choice(diff, (n_boot, len(diff)), replace=True).mean(axis=1)
    ci_lo = float(np.quantile(boot_means, alpha / 2))
    ci_hi = float(np.quantile(boot_means, 1.0 - alpha / 2))
    return {
        "mean_diff": mean_diff,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "significant": (ci_lo > 0.0) or (ci_hi < 0.0),
    }


def clustered_bootstrap_ci(
    df: pd.DataFrame,
    value_col: str = "spl",
    cluster_col: str = "layout_seed",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Cluster-level bootstrap resampling over layout seeds rather than individual episodes."""
    clusters = df[cluster_col].unique()
    if len(clusters) == 0:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    # Group df by cluster
    clust_groups = {c: g[value_col].to_numpy() for c, g in df.groupby(cluster_col)}

    boot_means = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        sampled_vals = np.concatenate([clust_groups[c] for c in sampled_clusters])
        boot_means[b] = np.mean(sampled_vals)

    return float(np.quantile(boot_means, alpha / 2)), float(np.quantile(boot_means, 1.0 - alpha / 2))
