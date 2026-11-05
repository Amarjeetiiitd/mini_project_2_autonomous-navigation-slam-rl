# Consolidated Navigation Benchmark Results (Nominal)

| Split | Controller | N | Success Rate (95% CI) | Collision Rate | Timeout Rate | SPL (95% CI) | Avg Time (s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **test** | reactive | 100 | 0.450 [0.36, 0.55] | 0.540 | 0.010 | 0.450 [0.35, 0.55] | 16.5 |
| **test** | dwa_global | 100 | 0.730 [0.64, 0.81] | 0.000 | 0.270 | 0.720 [0.63, 0.80] | 19.2 |
| **test** | **ppo_waypoint** | 100 | 0.760 [0.67, 0.83] | 0.240 | 0.000 | 0.754 [0.67, 0.83] | 19.3 |
| **heldout** | reactive | 100 | 0.380 [0.29, 0.48] | 0.540 | 0.080 | 0.378 [0.29, 0.48] | 15.7 |
| **heldout** | dwa_global | 100 | 0.660 [0.56, 0.75] | 0.000 | 0.340 | 0.653 [0.56, 0.74] | 19.2 |
| **heldout** | **ppo_waypoint** | 100 | 0.430 [0.34, 0.53] | 0.570 | 0.000 | 0.428 [0.34, 0.53] | 15.6 |