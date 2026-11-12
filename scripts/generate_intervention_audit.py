"""Generate intervention audit summary for safety-filtered controllers.

Evaluates MAS-RL and PPO Governor across test and held-out scenarios
to quantify the exact intervention frequencies of each safety layer:
1. Governor Braking (predictive stopping-distance governor with latency delay)
2. Lateral Centering (corridor and doorway midline bias)
3. Flank Repulsion (repulsive potential field against close sidewalls)
4. Recovery Reverse (rear-clearance verified backward backing)
5. Recovery Pivot (zero-linear in-place rotation away from obstacle)
6. Deadlock Spin (breakout rotation when trapped)
"""
from pathlib import Path
import numpy as np
import pandas as pd

from rl_training.baselines import (
    DWAController,
    GovernorPPOController,
    MASRLController,
    ReactiveController,
    evaluate_controller,
)
from rl_training.evaluate import load_policy
from rl_training.scenarios import make_scenarios
import rl_training.config as C


def main():
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    ppo_policy, mode = load_policy("models/ppo_navigation.zip")
    controllers = [
        ("ppo_governor", GovernorPPOController(ppo_policy.model), {"goal_mode": mode}),
        ("mas_rl", MASRLController(ppo_policy.model), {"goal_mode": mode}),
    ]

    splits = ["test", "heldout"]
    episodes = 100
    rows = []

    for split in splits:
        scns = make_scenarios(episodes, split, seed=0)
        for name, ctrl, kw in controllers:
            print(f"Evaluating {name} on {split} ({episodes} episodes)...", flush=True)
            df = evaluate_controller(ctrl, scns, {**kw, **C.CONDITIONS["nominal"]}, label=name)
            df["split"] = split
            df["condition"] = "nominal"

            # Compute mean intervention metrics per episode
            mean_gov = float(df["governor_brakes"].mean())
            mean_cent = float(df["centering_interventions"].mean())
            mean_flank = float(df["flank_repulsions"].mean())
            mean_rev = float(df["recovery_reverses"].mean())
            mean_piv = float(df["recovery_pivots"].mean())
            mean_spin = float(df["deadlock_spins"].mean())
            mean_sensor = float(df.get("sensor_loss_stops", 0).mean()) if "sensor_loss_stops" in df else 0.0
            mean_safe = float(df.get("safe_stops", 0).mean()) if "safe_stops" in df else 0.0
            mean_tot = float(df["total_interventions"].mean())
            mean_steps = float(df["steps"].mean())
            density = (mean_tot / mean_steps * 100.0) if mean_steps > 0 else 0.0

            rows.append({
                "split": split,
                "controller": name,
                "episodes": episodes,
                "success_rate": float(df["success"].mean()),
                "collision_rate": float(df["collision"].mean()),
                "mean_steps": round(mean_steps, 1),
                "governor_brakes": round(mean_gov, 2),
                "centering": round(mean_cent, 2),
                "flank_repulsions": round(mean_flank, 2),
                "recovery_reverses": round(mean_rev, 2),
                "recovery_pivots": round(mean_piv, 2),
                "deadlock_spins": round(mean_spin, 2),
                "sensor_loss_stops": round(mean_sensor, 2),
                "safe_stops": round(mean_safe, 2),
                "total_interventions": round(mean_tot, 2),
                "intervention_density_pct": round(density, 2),
            })

    summary_df = pd.DataFrame(rows)
    csv_path = out_dir / "interventions_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Generate Markdown table
    md = [
        "# Runtime Safety Intervention Audit Summary",
        "",
        "Breakdown of predictive safety filter layer activations per episode (Nominal condition, N=100 episodes per split):",
        "",
        "| Split | Controller | Success Rate | Mean Steps | Governor Brakes | Centering | Flank Repulse | Recovery Rev | Recovery Pivot | Deadlock Spin | Safe Stops | Total Interventions | Density (%) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['split']} | {r['controller']} | {r['success_rate']*100:.1f}% | {r['mean_steps']} | "
            f"{r['governor_brakes']} | {r['centering']} | {r['flank_repulsions']} | {r['recovery_reverses']} | "
            f"{r['recovery_pivots']} | {r['deadlock_spins']} | {r['safe_stops']} | {r['total_interventions']} | {r['intervention_density_pct']}% |"
        )
    md_path = out_dir / "interventions_summary.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved: {md_path}")
    print("\n".join(md))


if __name__ == "__main__":
    main()
