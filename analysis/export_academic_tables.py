"""Export Academic Benchmark Tables in LaTeX and Markdown for Reports.

Reads `results/compare_summary.csv` and `results/compare_by_family.csv`
and outputs formatted tables ready for inclusion in research reports and IEEE papers.

Usage:
    python -m analysis.export_academic_tables
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def export_latex_main_table(df: pd.DataFrame, out_path: Path):
    nominal = df[df["condition"] == "nominal"]
    
    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Quantitative Comparison of Navigation Policies under Nominal Conditions (95\% Confidence Intervals).}",
        r"\label{tab:controller_comparison}",
        r"\begin{tabular}{llcccccc}",
        r"\hline\hline",
        r"\textbf{Split} & \textbf{Controller} & \textbf{Trials ($N$)} & \textbf{Success Rate} $\uparrow$ & \textbf{Collision Rate} $\downarrow$ & \textbf{Timeout Rate} $\downarrow$ & \textbf{SPL} $\uparrow$ & \textbf{Time to Goal (s)} \\",
        r"\hline"
    ]
    
    ctrl_map = {
        "reactive": "Reactive Baseline",
        "dwa_global": "DWA + Global Plan",
        "ppo_waypoint": "Unshielded PPO",
        "ppo_governor": "PPO + Safety Governor (L1)",
        "mas_rl": r"\textbf{MAS-RL (Full 4-Layer)}"
    }
    
    for split in ["test", "heldout"]:
        sub = nominal[nominal["split"] == split]
        split_label = "Test (Unseen)" if split == "test" else "Held-Out (ZigZag)"
        first = True
        for _, row in sub.iterrows():
            c_name = ctrl_map.get(row["controller"], row["controller"])
            s_ci = f"{row['success_rate']:.3f} [{row['success_ci_lo']:.2f}, {row['success_ci_hi']:.2f}]"
            coll = f"{row['collision_rate']:.3f}"
            tout = f"{row['timeout_rate']:.3f}"
            spl = f"{row['spl']:.3f}"
            t_succ = f"{row['time_s_succ']:.1f}" if pd.notna(row['time_s_succ']) else "-"
            
            s_col = split_label if first else ""
            latex.append(f"{s_col} & {c_name} & {int(row['n'])} & {s_ci} & {coll} & {tout} & {spl} & {t_succ} \\\\")
            first = False
        latex.append(r"\hline")
        
    latex.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    out_path.write_text("\n".join(latex), encoding="utf-8")
    print(f"Saved: {out_path}")


def export_latex_family_table(df_fam: pd.DataFrame, out_path: Path):
    latex = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Per-Topology Success Rate Comparison across Environment Families.}",
        r"\label{tab:family_breakdown}",
        r"\begin{tabular}{lccccc}",
        r"\hline\hline",
        r"\textbf{Controller} & \textbf{Open} & \textbf{Corridor} & \textbf{Clutter} & \textbf{Rooms} & \textbf{ZigZag (Held-out)} \\",
        r"\hline"
    ]
    
    ctrls = ["reactive", "dwa_global", "ppo_waypoint", "ppo_governor", "mas_rl"]
    ctrl_map = {
        "reactive": "Reactive Baseline",
        "dwa_global": "DWA + Global Plan",
        "ppo_waypoint": "Unshielded PPO",
        "ppo_governor": "PPO + Safety Governor (L1)",
        "mas_rl": r"\textbf{MAS-RL (Full 4-Layer)}"
    }
    
    fams = ["open", "corridor", "clutter", "rooms", "zigzag"]
    
    for c in ctrls:
        row_str = [ctrl_map[c]]
        for f in fams:
            match = df_fam[(df_fam["controller"] == c) & (df_fam["family"] == f)]
            if len(match) > 0:
                val = match.iloc[0]["success_rate"]
                row_str.append(f"{val:.2f}")
            else:
                row_str.append("-")
        latex.append(" & ".join(row_str) + r" \\")
        
    latex.extend([
        r"\hline\hline",
        r"\end{tabular}",
        r"\end{table}"
    ])
    out_path.write_text("\n".join(latex), encoding="utf-8")
    print(f"Saved: {out_path}")


def export_markdown_formatted_table(df: pd.DataFrame, out_path: Path):
    nominal = df[df["condition"] == "nominal"]
    md = [
        "# Consolidated Navigation Benchmark Results (Nominal)",
        "",
        "| Split | Controller | N | Success Rate (95% CI) | Collision Rate | Timeout Rate | SPL (95% CI) | Avg Time (s) |",
        "|---|---|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    
    for _, row in nominal.iterrows():
        split = f"**{row['split']}**"
        ctrl = f"**{row['controller']}**" if "ppo" in row['controller'] else row['controller']
        succ = f"{row['success_rate']:.3f} [{row['success_ci_lo']:.2f}, {row['success_ci_hi']:.2f}]"
        coll = f"{row['collision_rate']:.3f}"
        tout = f"{row['timeout_rate']:.3f}"
        spl = f"{row['spl']:.3f} [{row['spl_ci_lo']:.2f}, {row['spl_ci_hi']:.2f}]"
        t_succ = f"{row['time_s_succ']:.1f}" if pd.notna(row['time_s_succ']) else "-"
        md.append(f"| {split} | {ctrl} | {int(row['n'])} | {succ} | {coll} | {tout} | {spl} | {t_succ} |")
        
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Saved: {out_path}")


def main():
    summary_path = Path("results/compare_summary.csv")
    family_path = Path("results/compare_by_family.csv")
    
    if not summary_path.exists() or not family_path.exists():
        print("[ERROR] CSV files not found.")
        return
        
    df_summary = pd.read_csv(summary_path)
    df_family = pd.read_csv(family_path)
    
    out_dir = Path("results")
    export_latex_main_table(df_summary, out_dir / "table_comparison.tex")
    export_latex_family_table(df_family, out_dir / "table_per_family.tex")
    export_markdown_formatted_table(df_summary, out_dir / "table_comparison_formatted.md")
    print("[SUCCESS] All academic tables exported successfully.")


if __name__ == "__main__":
    main()
