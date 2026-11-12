# Runtime Safety Intervention Audit Summary

Breakdown of predictive safety filter layer activations per episode (Nominal condition, N=100 episodes per split):

| Split | Controller | Success Rate | Mean Steps | Governor Brakes | Centering | Flank Repulse | Recovery Rev | Recovery Pivot | Deadlock Spin | Safe Stops | Total Interventions | Density (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | ppo_governor | 83.0% | 280.8 | 99.88 | 0.0 | 0.0 | 0.0 | 0.0 | 0.18 | 0.0 | 100.06 | 35.64% |
| test | mas_rl | 84.0% | 283.7 | 80.74 | 5.81 | 7.67 | 3.32 | 0.0 | 0.01 | 10.21 | 107.76 | 37.98% |
| heldout | ppo_governor | 56.0% | 398.2 | 259.85 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 259.85 | 65.25% |
| heldout | mas_rl | 60.0% | 390.4 | 197.42 | 0.45 | 4.98 | 7.15 | 0.0 | 0.0 | 41.01 | 251.01 | 64.3% |
