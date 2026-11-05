"""Publication-Grade Visual Analytics and Comparative Graph Generator.

Generates high-resolution figures for the final year project report and defense:
1. fig_benchmark_overview.png: Grouped bars for Success, Collision, and Timeout rates (95% Wilson CI).
2. fig_family_breakdown.png: Per-family navigation success across rooms, corridors, clutter, open, zigzag.
3. fig_radar_tradeoff.png: Multi-metric radar chart contrasting PPO vs DWA vs Reactive.
4. fig_robustness_perturbation.png: Sensitivity to LiDAR sensor noise and localization drift.

Usage:
    python -m analysis.generate_visual_comparison
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set high-quality IEEE/ACM publication style
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_benchmark_overview(df_summary: pd.DataFrame):
    """Plot Success, Collision, and Timeout rates with 95% Wilson Confidence Intervals."""
    nominal = df_summary[df_summary["condition"] == "nominal"].copy()
    
    controllers = ["reactive", "dwa_global", "ppo_waypoint"]
    ctrl_labels = ["Reactive Baseline", "DWA + Global Plan", "PPO (Lookahead Waypoint)"]
    colors = ["#e53e3e", "#3182ce", "#38a169"]
    
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    splits = [("test", "Unseen Layouts (Test Split)"), ("heldout", "Unseen Family: ZigZag (Held-Out Split)")]
    
    bar_width = 0.24
    
    for ax_idx, (split_name, split_title) in enumerate(splits):
        ax = axes[ax_idx]
        split_data = nominal[nominal["split"] == split_name]
        
        x_indices = np.arange(len(controllers))
        
        # Extract metrics
        successes = []
        err_lo = []
        err_hi = []
        collisions = []
        timeouts = []
        
        for ctrl in controllers:
            row = split_data[split_data["controller"] == ctrl].iloc[0]
            s = row["success_rate"]
            successes.append(s)
            err_lo.append(s - row["success_ci_lo"])
            err_hi.append(row["success_ci_hi"] - s)
            collisions.append(row["collision_rate"])
            timeouts.append(row["timeout_rate"])
            
        yerr = [err_lo, err_hi]
        
        # Plot grouped bars
        b1 = ax.bar(x_indices - bar_width, successes, bar_width, yerr=yerr, capsize=4,
                    label="Success Rate", color="#2b6cb0", edgecolor="black", linewidth=0.8)
        b2 = ax.bar(x_indices, collisions, bar_width,
                    label="Collision Rate", color="#c53030", edgecolor="black", linewidth=0.8)
        b3 = ax.bar(x_indices + bar_width, timeouts, bar_width,
                    label="Timeout Rate", color="#dd6b20", edgecolor="black", linewidth=0.8)
        
        # Add value labels on top of bars
        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.03, f"{h:.2f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color="#2b6cb0")
        for bar in b2:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f}",
                        ha='center', va='bottom', fontsize=8, color="#c53030")
        for bar in b3:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.2f}",
                        ha='center', va='bottom', fontsize=8, color="#dd6b20")
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(ctrl_labels, rotation=12, ha="right")
        ax.set_title(split_title, fontweight="bold", pad=8)
        ax.set_ylim(0, 1.08)
        
        if ax_idx == 0:
            ax.set_ylabel("Metric Proportion [0.0 - 1.0]")
            ax.legend(loc="upper right", framealpha=0.9)
            
    fig.tight_layout()
    out_path = FIG_DIR / "fig_benchmark_overview.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_family_breakdown(df_family: pd.DataFrame):
    """Plot per-family success rate breakdown."""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    
    families = ["open", "corridor", "clutter", "rooms", "zigzag"]
    fam_labels = ["Open Space", "Corridors", "Clutter / Columns", "Multi-Room Doors", "ZigZag (Held-Out)"]
    
    controllers = ["reactive", "dwa_global", "ppo_waypoint"]
    ctrl_names = ["Reactive Baseline", "DWA Local Planner", "PPO Local Controller"]
    ctrl_colors = ["#e53e3e", "#3182ce", "#38a169"]
    
    n_fams = len(families)
    n_ctrls = len(controllers)
    indices = np.arange(n_fams)
    bar_width = 0.26
    
    for c_idx, ctrl in enumerate(controllers):
        rates = []
        for fam in families:
            sub = df_family[(df_family["controller"] == ctrl) & (df_family["family"] == fam)]
            if len(sub) > 0:
                rates.append(sub.iloc[0]["success_rate"])
            else:
                rates.append(0.0)
                
        offset = (c_idx - (n_ctrls - 1) / 2) * bar_width
        bars = ax.bar(indices + offset, rates, bar_width, label=ctrl_names[c_idx],
                      color=ctrl_colors[c_idx], edgecolor="black", linewidth=0.8, alpha=0.9)
        
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.015, f"{h:.2f}",
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
    ax.set_xticks(indices)
    ax.set_xticklabels(fam_labels, fontweight="bold")
    ax.set_ylabel("Navigation Success Rate")
    ax.set_title("Navigation Success Rate Across Indoor Layout Topologies", fontweight="bold", pad=10)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", framealpha=0.95)
    
    # Add vertical divider before held-out family
    ax.axvline(3.5, color="grey", linestyle=":", linewidth=1.5)
    ax.text(3.6, 0.98, "Held-Out Family →", color="#4a5568", fontsize=9, fontstyle="italic")
    
    fig.tight_layout()
    out_path = FIG_DIR / "fig_family_breakdown.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_radar_tradeoff(df_summary: pd.DataFrame):
    """Generate Radar/Spider chart illustrating the multi-objective trade-offs."""
    # Axes:
    # 1. Success Rate (Test)
    # 2. Safety (1 - Collision Rate on Test)
    # 3. Path Efficiency (SPL on Test)
    # 4. Out-of-Distribution Generalization (Success on Held-Out)
    # 5. Sensor Noise Robustness (Success on Test Lidar Noise / Success Nominal)
    
    categories = [
        "Success Rate\n(Test Layouts)",
        "Collision Safety\n(1 - Collision Rate)",
        "Path Efficiency\n(SPL Metric)",
        "OOD Generalization\n(Held-out Zigzag)",
        "Sensor Robustness\n(Under Lidar Noise)",
    ]
    N = len(categories)
    
    controllers = ["reactive", "dwa_global", "ppo_waypoint"]
    ctrl_names = ["Reactive Baseline", "DWA + Global Plan", "PPO Local Controller"]
    colors = ["#e53e3e", "#3182ce", "#38a169"]
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close polygon
    
    fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
    
    for c_idx, ctrl in enumerate(controllers):
        test_nom = df_summary[(df_summary["split"] == "test") & (df_summary["condition"] == "nominal") & (df_summary["controller"] == ctrl)].iloc[0]
        held_nom = df_summary[(df_summary["split"] == "heldout") & (df_summary["condition"] == "nominal") & (df_summary["controller"] == ctrl)].iloc[0]
        test_noise = df_summary[(df_summary["split"] == "test") & (df_summary["condition"] == "lidar_noise") & (df_summary["controller"] == ctrl)].iloc[0]
        
        succ = test_nom["success_rate"]
        safety = 1.0 - test_nom["collision_rate"]
        spl = test_nom["spl"]
        ood = held_nom["success_rate"]
        robust = test_noise["success_rate"] / max(succ, 1e-6)
        robust = min(robust, 1.0)
        
        values = [succ, safety, spl, ood, robust]
        values += values[:1]  # close loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=ctrl_names[c_idx], color=colors[c_idx])
        ax.fill(angles, values, color=colors[c_idx], alpha=0.15)
        
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    ax.set_title("Multi-Objective Performance Comparison", fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.95)
    
    fig.tight_layout()
    out_path = FIG_DIR / "fig_radar_tradeoff.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_robustness_perturbation(df_summary: pd.DataFrame):
    """Plot performance across nominal, lidar_noise, and loc_drift conditions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    splits = [("test", "Test Split (Familiar Families)"), ("heldout", "Held-Out Split (ZigZag)")]
    
    conditions = ["nominal", "lidar_noise", "loc_drift"]
    cond_labels = ["Nominal", "LiDAR Noise (σ=3cm, 5% drop)", "Pose Drift (Random Walk)"]
    
    controllers = ["reactive", "dwa_global", "ppo_waypoint"]
    ctrl_names = ["Reactive", "DWA", "PPO (Waypoint)"]
    colors = ["#e53e3e", "#3182ce", "#38a169"]
    markers = ["o", "s", "^"]
    
    for ax_idx, (split_name, split_title) in enumerate(splits):
        ax = axes[ax_idx]
        split_data = df_summary[df_summary["split"] == split_name]
        
        for c_idx, ctrl in enumerate(controllers):
            c_data = split_data[split_data["controller"] == ctrl]
            y_vals = []
            for cond in conditions:
                row = c_data[c_data["condition"] == cond]
                y_vals.append(row.iloc[0]["success_rate"] if len(row) > 0 else np.nan)
                
            ax.plot(range(len(conditions)), y_vals, marker=markers[c_idx], markersize=7,
                    linewidth=2.0, label=ctrl_names[c_idx], color=colors[c_idx])
            
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(cond_labels, rotation=10, ha="right", fontsize=9)
        ax.set_title(split_title, fontweight="bold", pad=8)
        ax.set_ylim(0.2, 0.9)
        
        if ax_idx == 0:
            ax.set_ylabel("Success Rate")
            ax.legend(loc="lower left", framealpha=0.9)
            
    fig.tight_layout()
    out_path = FIG_DIR / "fig_robustness_perturbation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    summary_path = Path("results/compare_summary.csv")
    family_path = Path("results/compare_by_family.csv")
    
    if not summary_path.exists() or not family_path.exists():
        print("[ERROR] results/compare_summary.csv or results/compare_by_family.csv missing.")
        return
    
    df_summary = pd.read_csv(summary_path)
    df_family = pd.read_csv(family_path)
    
    print("[INFO] Generating academic publication-grade graphs...")
    plot_benchmark_overview(df_summary)
    plot_family_breakdown(df_family)
    plot_radar_tradeoff(df_summary)
    plot_robustness_perturbation(df_summary)
    print("[SUCCESS] All figures successfully generated in results/figures/")


if __name__ == "__main__":
    main()
