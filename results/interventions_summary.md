# Runtime Safety Intervention Audit Summary

Breakdown of predictive safety filter layer activations per episode (Nominal condition, N=100 episodes per split):

| Split | Controller | Success Rate | Mean Steps | Governor Brakes | Centering | Flank Repulse | Recovery Rev | Recovery Pivot | Deadlock Spin | Total Interventions | Density (%) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | ppo_governor | 83.0% | 280.8 | 99.88 | 0.0 | 0.0 | 0.0 | 0.0 | 0.18 | 100.06 | 35.64% |
| test | mas_rl | 85.0% | 280.2 | 85.72 | 6.15 | 7.72 | 3.32 | 0.64 | 0.01 | 103.56 | 36.95% |
| heldout | ppo_governor | 56.0% | 398.2 | 259.85 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 259.85 | 65.25% |
| heldout | mas_rl | 65.0% | 376.6 | 203.83 | 0.45 | 6.91 | 7.79 | 9.11 | 0.08 | 228.17 | 60.59% |
