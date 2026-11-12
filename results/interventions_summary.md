# Runtime Safety Intervention Audit Summary

Breakdown of predictive safety filter layer activations per episode (Nominal condition, N=100 episodes per split):

| Split | Controller | Success Rate | Mean Steps | Governor Brakes | Centering | Flank Repulse | Recovery Rev | Recovery Pivot | Deadlock Spin | Safe Stops | Total Interventions | Density (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | ppo_governor | 83.0% | 280.8 | 99.88 | 0.0 | 0.0 | 0.0 | 0.0 | 0.18 | 0.0 | 100.06 | 35.64% |
| test | mas_rl | 84.0% | 283.7 | 80.74 | 5.81 | 7.67 | 3.32 | 0.0 | 0.01 | 10.23 | 107.78 | 37.99% |
| heldout | ppo_governor | 56.0% | 398.2 | 259.85 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 259.85 | 65.25% |
| heldout | mas_rl | 60.0% | 391.1 | 198.04 | 0.45 | 4.99 | 7.15 | 0.0 | 0.0 | 41.04 | 251.67 | 64.35% |
