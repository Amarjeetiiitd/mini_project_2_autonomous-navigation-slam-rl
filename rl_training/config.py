"""Single source of truth for every constant shared by the training environment,
the baselines and the ROS 2 policy node.

If you change anything that affects the observation vector or the action
mapping, you MUST retrain the policy. The ROS node imports this module, so the
two sides cannot silently drift apart.
"""
from __future__ import annotations

import math

# ----------------------------------------------------------------- sensing
N_SECTORS = 24                      # angular sectors in the observation
RAYS_PER_SECTOR = 3                 # simulated rays per sector (min is taken)
SECTOR_WIDTH = 2.0 * math.pi / N_SECTORS
RANGE_MIN = 0.12                    # m   (TurtleBot3 LDS-01 / LDS-02 class)
RANGE_MAX = 3.5                     # m

# ------------------------------------------------------------- observation
# [0:24]  sector ranges / RANGE_MAX           in [0, 1]
# [24]    min(goal_dist / GOAL_DIST_NORM, 1)  in [0, 1]
# [25]    sin(relative goal angle)            in [-1, 1]
# [26]    cos(relative goal angle)            in [-1, 1]
# [27]    v / V_MAX                           in [0, 1]
# [28]    w / W_MAX                           in [-1, 1]
OBS_DIM = N_SECTORS + 5
GOAL_DIST_NORM = 5.0                # m
LOOKAHEAD = 1.5                     # m, waypoint lookahead on the global plan

# ------------------------------------------------------------------- robot
ROBOT_RADIUS = 0.22                 # m (TurtleBot3 waffle ~0.22)
GOAL_TOLERANCE = 0.25               # m
V_MAX = 0.26                        # m/s  forward only (see README)
W_MAX = 1.0                         # rad/s
A_V_MAX = 1.0                       # m/s^2
A_W_MAX = 3.2                       # rad/s^2
DT = 0.1                            # s per control step (10 Hz)

# --------------------------------------------------------------- episodes
MAX_STEPS = 700                     # 70 s
GRID_RES = 0.1                      # m, planning / occupancy resolution
PLAN_INFLATION = 0.8 * ROBOT_RADIUS  # planner inflation radius
START_CLEARANCE = ROBOT_RADIUS + 0.15  # clearance for sampled starts/goals

# ------------------------------------------------------ layout seed splits
# Layouts are generated procedurally from an integer seed, so these splits are
# disjoint sets of *layouts* (no leakage between train / val / test).
TRAIN_SEEDS = tuple(range(0, 1000))
VAL_SEEDS = tuple(range(1000, 1050))
TEST_SEEDS = tuple(range(10000, 10100))
HELDOUT_SEEDS = tuple(range(20000, 20040))   # held-out *family* ("zigzag")

# --------------------------------------------------------------- rewards
R_PROGRESS = 10.0                   # per metre of geodesic progress
R_GOAL = 50.0
R_COLLISION = -50.0
R_STEP = -0.02
R_TURN = -0.02                      # * |w| / W_MAX
R_PROXIMITY = -0.5                  # * (1 - min_range / PROX_DIST) if closer
PROX_DIST = 0.6

# Named perturbation conditions used by experiments E4 / E6 (synthetic).
CONDITIONS = {
    "nominal": {},
    "lidar_noise": {"lidar_noise_std": 0.03, "lidar_dropout": 0.05},
    "loc_drift": {"loc_drift_pos": 0.004, "loc_drift_yaw": 0.002},
}
