"""Observation / action adapters shared by the Gym environment and the ROS 2 node.

Keeping them here (pure NumPy, no ROS imports) guarantees that the policy sees
exactly the same feature layout in simulation-training and on the robot.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def sector_center(i: int) -> float:
    """Sector i is centred at -pi + i*15deg. Sector 12 is straight ahead,
    13..18 are to the left (CCW, +angle), 6..11 to the right."""
    return -math.pi + i * C.SECTOR_WIDTH


def sector_index(angle: np.ndarray | float) -> np.ndarray | int:
    a = (np.asarray(angle) + math.pi) % (2.0 * math.pi) - math.pi
    return np.floor((a + math.pi + C.SECTOR_WIDTH / 2) / C.SECTOR_WIDTH).astype(int) % C.N_SECTORS


def ray_angles() -> np.ndarray:
    """Robot-frame angles of the simulated rays, shape (N_SECTORS, RAYS_PER_SECTOR)."""
    offs = (np.arange(C.RAYS_PER_SECTOR) - (C.RAYS_PER_SECTOR - 1) / 2) * (
        C.SECTOR_WIDTH / C.RAYS_PER_SECTOR)
    centers = -math.pi + np.arange(C.N_SECTORS) * C.SECTOR_WIDTH
    return centers[:, None] + offs[None, :]


def resample_scan(ranges, angle_min: float, angle_increment: float,
                  range_min: float = C.RANGE_MIN, range_max: float = C.RANGE_MAX,
                  yaw_offset: float = 0.0) -> np.ndarray:
    """Reduce a sensor_msgs/LaserScan-style array to N_SECTORS minimum ranges (m).

    NaN, +-inf, readings < range_min and readings > range_max are *invalid* and
    ignored; a sector with no valid reading reports RANGE_MAX ("free").
    `yaw_offset` rotates scan angles into the base_link frame.
    """
    r = np.asarray(ranges, dtype=np.float64)
    angles = angle_min + np.arange(r.size) * angle_increment + yaw_offset
    valid = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    out = np.full(C.N_SECTORS, C.RANGE_MAX)
    if valid.any():
        idx = sector_index(angles[valid])
        np.minimum.at(out, idx, np.minimum(r[valid], C.RANGE_MAX))
    return out


def build_observation(sector_ranges, goal_dist: float, goal_angle: float,
                      v: float, w: float) -> np.ndarray:
    """Assemble the 29-D observation. v, w are the robot's current velocities."""
    obs = np.empty(C.OBS_DIM, dtype=np.float32)
    obs[:C.N_SECTORS] = np.clip(np.asarray(sector_ranges) / C.RANGE_MAX, 0.0, 1.0)
    obs[C.N_SECTORS] = min(goal_dist / C.GOAL_DIST_NORM, 1.0)
    obs[C.N_SECTORS + 1] = math.sin(goal_angle)
    obs[C.N_SECTORS + 2] = math.cos(goal_angle)
    obs[C.N_SECTORS + 3] = np.clip(v / C.V_MAX, 0.0, 1.0)
    obs[C.N_SECTORS + 4] = np.clip(w / C.W_MAX, -1.0, 1.0)
    return obs


def action_to_cmd(action) -> tuple[float, float]:
    """Policy action in [-1, 1]^2 -> (v [m/s] in [0, V_MAX], w [rad/s])."""
    a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    return float((a[0] + 1.0) / 2.0 * C.V_MAX), float(a[1] * C.W_MAX)


def cmd_to_action(v: float, w: float) -> np.ndarray:
    return np.array([np.clip(2.0 * v / C.V_MAX - 1.0, -1, 1), np.clip(w / C.W_MAX, -1, 1)],
                    dtype=np.float32)


def goal_in_robot_frame(pose_xy_yaw, target_xy) -> tuple[float, float]:
    x, y, yaw = pose_xy_yaw
    dx, dy = target_xy[0] - x, target_xy[1] - y
    return math.hypot(dx, dy), wrap_angle(math.atan2(dy, dx) - yaw)


def lookahead_point(path_xy: np.ndarray, robot_xy, lookahead: float = C.LOOKAHEAD):
    """Point `lookahead` metres along `path_xy` (N,2) from the path point closest to
    the robot. Returns the final point if the remaining path is shorter."""
    path_xy = np.asarray(path_xy, dtype=np.float64)
    if len(path_xy) == 0:
        return None
    i0 = int(np.argmin(np.hypot(path_xy[:, 0] - robot_xy[0], path_xy[:, 1] - robot_xy[1])))
    seg = np.hypot(*np.diff(path_xy[i0:], axis=0).T) if len(path_xy) - i0 > 1 else np.zeros(0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    j = int(np.searchsorted(cum, lookahead))
    return path_xy[min(i0 + j, len(path_xy) - 1)]
