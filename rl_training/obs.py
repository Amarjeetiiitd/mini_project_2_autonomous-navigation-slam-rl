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
                  yaw_offset: float = 0.0,
                  return_mask: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Reduce a sensor_msgs/LaserScan-style array to N_SECTORS minimum ranges (m).

    Valid readings within [range_min, range_max] or genuine free space (+inf)
    are mapped to angular sectors.
    Sectors with NO valid raw readings (complete dropout / blind spots) are marked as
    invalid (NaN in ranges, False in mask) rather than being falsely imputed as free space.

    Args:
        ranges: Sequence of raw distance returns.
        angle_min: Angle of first beam (rad).
        angle_increment: Angular distance between beams (rad).
        range_min: Minimum reliable sensor range (m).
        range_max: Maximum sensor range (m).
        yaw_offset: Offset in radians rotating scan into base_link frame.
        return_mask: If True, returns (sector_ranges, sector_valid_mask).

    Returns:
        np.ndarray of shape (N_SECTORS,), or tuple (sector_ranges, sector_valid_mask).
    """
    r = np.asarray(ranges, dtype=np.float64)
    out = np.full(C.N_SECTORS, np.nan, dtype=np.float64)
    valid_mask = np.zeros(C.N_SECTORS, dtype=bool)

    if r.size == 0:
        return (out, valid_mask) if return_mask else out

    angles = angle_min + np.arange(r.size) * angle_increment + yaw_offset
    beam_sectors = sector_index(angles)

    finite_hits = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    inf_returns = np.isposinf(r)
    valid_beams = finite_hits | inf_returns

    if not valid_beams.any():
        return (out, valid_mask) if return_mask else out

    # Mark sectors containing at least one valid beam (hit or free-space inf)
    valid_sectors = np.unique(beam_sectors[valid_beams])
    valid_mask[valid_sectors] = True
    out[valid_sectors] = range_max

    # For valid finite obstacle hits, aggregate minimum distance per sector
    if finite_hits.any():
        hit_sectors = beam_sectors[finite_hits]
        hit_ranges = np.minimum(r[finite_hits], range_max)
        np.minimum.at(out, hit_sectors, hit_ranges)

    return (out, valid_mask) if return_mask else out


def build_observation(sector_ranges, goal_dist: float, goal_angle: float,
                      v: float, w: float) -> np.ndarray:
    """Assemble the 29-D observation. v, w are the robot's current velocities."""
    obs = np.empty(C.OBS_DIM, dtype=np.float32)
    sr = np.asarray(sector_ranges, dtype=np.float32)
    if not np.isfinite(sr).all():
        # Sanitize missing/NaN sectors conservatively to 0.0 (obstacle immediately adjacent)
        sr = np.where(np.isfinite(sr), sr, 0.0)
    obs[:C.N_SECTORS] = np.clip(sr / C.RANGE_MAX, 0.0, 1.0)
    obs[C.N_SECTORS] = min(goal_dist / C.GOAL_DIST_NORM, 1.0)
    obs[C.N_SECTORS + 1] = math.sin(goal_angle)
    obs[C.N_SECTORS + 2] = math.cos(goal_angle)
    obs[C.N_SECTORS + 3] = np.clip(v / C.V_MAX, 0.0, 1.0)
    obs[C.N_SECTORS + 4] = np.clip(w / C.W_MAX, -1.0, 1.0)
    return obs


def action_to_cmd(action) -> tuple[float, float]:
    """Policy action -> (v [m/s], w [rad/s]).
    Standard policy action a in [-1, 1]^2 maps linearly to v in [0, V_MAX], w in [-W_MAX, W_MAX].
    Clips strictly within [-1.0, 1.0] to enforce the declared Gymnasium action space contract.
    """
    a = np.asarray(action, dtype=np.float64)
    a0 = float(np.clip(a[0], -1.0, 1.0))
    a1 = float(np.clip(a[1], -1.0, 1.0))
    v = (a0 + 1.0) / 2.0 * C.V_MAX
    return float(v), float(a1 * C.W_MAX)


def cmd_to_action(v: float, w: float) -> np.ndarray:
    """Velocity commands (v, w) -> policy action in standard Box(-1.0, 1.0, (2,)).
    Clamped strictly to [-1, 1] to preserve Gymnasium action space contract.
    Forward velocities map linearly from [0, V_MAX] to [-1, 1].
    Any non-positive velocity maps to -1.0 (zero forward throttle).
    """
    a0 = np.clip(2.0 * max(0.0, v) / C.V_MAX - 1.0, -1.0, 1.0)
    a1 = np.clip(w / C.W_MAX, -1.0, 1.0)
    return np.array([a0, a1], dtype=np.float32)


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
