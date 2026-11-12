"""Fast Gymnasium environment: differential-drive robot with a simulated 2D LiDAR
in procedurally generated indoor layouts.

* Observation (29-D): see rl_training/config.py and rl_training/obs.py.
* Action: Box(-1, 1, (2,)) -> (v in [0, V_MAX], w in [-W_MAX, W_MAX]).
* The robot only ever *believes* it is at `believed_pose` (true pose + optional
  random-walk drift). This is a synthetic stand-in for SLAM localisation error.
  Real SLAM results must come from the ROS 2 / Gazebo experiments.
* Reward shaping uses a privileged geodesic distance field (never observed).
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from . import config as C
from .maps import collides, get_bundle, raycast
from .obs import (action_to_cmd, build_observation, goal_in_robot_frame, ray_angles,
                  wrap_angle)
from .planning import (extract_path, geodesic_field, make_clearance_penalty,
                       rc_to_xy, smooth_sampler, xy_to_rc)
from .scenarios import Scenario, sample_start_goal


class NavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, layout_seeds=None, goal_mode: str = "waypoint", family: str | None = None,
                 lidar_noise_std: float = 0.0, lidar_dropout: float = 0.0,
                 loc_drift_pos: float = 0.0, loc_drift_yaw: float = 0.0,
                 min_goal_dist: float = 2.0, max_goal_dist: float = 9.0,
                 max_steps: int = C.MAX_STEPS, clearance_weighted: bool = True):
        super().__init__()
        assert goal_mode in ("waypoint", "direct")
        self.layout_seeds = tuple(layout_seeds) if layout_seeds is not None else C.TRAIN_SEEDS
        self.goal_mode, self.family = goal_mode, family
        self.noise_std, self.dropout = lidar_noise_std, lidar_dropout
        self.drift_pos, self.drift_yaw = loc_drift_pos, loc_drift_yaw
        self.dmin, self.dmax, self.max_steps = min_goal_dist, max_goal_dist, max_steps
        self.clearance_weighted = clearance_weighted

        lo = np.array([0.0] * C.N_SECTORS + [0.0, -1.0, -1.0, 0.0, -1.0], dtype=np.float32)
        hi = np.ones(C.OBS_DIM, dtype=np.float32)
        self.observation_space = spaces.Box(lo, hi, dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self._angles = ray_angles()  # (24, 3)
        self.bundle = None

    # ---------------------------------------------------------------- reset
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        scn = options.get("scenario")
        if scn is not None:
            self.bundle = get_bundle(scn.layout_seed, scn.family_override)
            sx, sy, yaw = scn.start
            self.goal = np.array(scn.goal, dtype=np.float64)
            pen = make_clearance_penalty(self.bundle.plan_blocked) if self.clearance_weighted else None
            self.field = geodesic_field(self.bundle.plan_blocked, xy_to_rc(*self.goal, self.bundle.occ.shape),
                                        clearance_cost=pen)
            self.shortest = scn.shortest_len
        else:
            for _ in range(50):
                layout = options.get("layout_seed", self.layout_seeds[self.np_random.integers(len(self.layout_seeds))])
                self.bundle = get_bundle(int(layout), self.family)
                res = sample_start_goal(self.bundle, self.np_random, self.dmin, self.dmax)
                if res is not None:
                    break
            else:
                raise RuntimeError("could not sample a valid start/goal")
            (sx, sy), g, self.shortest, self.field = res
            self.goal = np.array(g, dtype=np.float64)
            if self.clearance_weighted:
                pen = make_clearance_penalty(self.bundle.plan_blocked)
                self.field = geodesic_field(self.bundle.plan_blocked, xy_to_rc(*self.goal, self.bundle.occ.shape),
                                            clearance_cost=pen)
            yaw = float(self.np_random.uniform(-math.pi, math.pi))
        self.rects = self.bundle.world.rects
        self._geo = smooth_sampler(self.field)
        self.pose = np.array([sx, sy, yaw], dtype=np.float64)
        self.v = self.w = 0.0
        self.off = np.zeros(3)
        self.steps, self.path_len = 0, 0.0
        self._prev_geo = self._geo(sx, sy)
        obs = self._observe()
        return obs, self._info(False, False, False)

    # ------------------------------------------------------------- sensing
    def scan_rays(self, angles_rel: np.ndarray, noisy: bool = True) -> np.ndarray:
        x, y, th = self.pose
        r = raycast((x, y), angles_rel.ravel() + th, self.rects).reshape(angles_rel.shape)
        if noisy:
            if self.noise_std > 0:
                hit = r < C.RANGE_MAX
                r = np.where(hit, r + self.np_random.normal(0, self.noise_std, r.shape), r)
            if self.dropout > 0:
                r = np.where(self.np_random.random(r.shape) < self.dropout, C.RANGE_MAX, r)
            r = np.clip(r, 0.0, C.RANGE_MAX)
        return r

    def scan_full(self, n: int = 360) -> tuple[np.ndarray, float, float]:
        """Noise-free LaserScan-like data (ranges, angle_min, angle_increment)."""
        inc = 2 * math.pi / n
        ang = -math.pi + np.arange(n) * inc
        return self.scan_rays(ang, noisy=False), -math.pi, inc

    @property
    def believed_pose(self) -> np.ndarray:
        p = self.pose + self.off
        return np.array([p[0], p[1], wrap_angle(p[2])])

    def waypoint_world(self, lookahead: float = C.LOOKAHEAD) -> np.ndarray:
        """Lookahead point on the global plan from the *believed* pose."""
        bx, by, _ = self.believed_pose
        rc = xy_to_rc(bx, by, self.field.shape)
        cells = extract_path(self.field, rc, lookahead)
        if not cells or self.field[cells[-1]] <= 0.0:
            return self.goal.copy()
        return np.array(rc_to_xy(*cells[-1]))

    def goal_rel(self, direct: bool = False) -> tuple[float, float]:
        target = self.goal if (direct or self.goal_mode == "direct") else self.waypoint_world()
        return goal_in_robot_frame(self.believed_pose, target)

    def _observe(self) -> np.ndarray:
        rays = self.scan_rays(self._angles)
        self._last_angles, self._last_ranges = self._angles.copy(), rays.copy()
        d, a = self.goal_rel()
        return build_observation(rays.min(axis=1), d, a, self.v, self.w)

    # ----------------------------------------------------------- dynamics
    def step_cmd(self, v_t: float, w_t: float):
        """Execute physical velocity commands (v_t [m/s], w_t [rad/s]) directly with acceleration limits."""
        dv, dw = C.A_V_MAX * C.DT, C.A_W_MAX * C.DT
        self.v = float(np.clip(v_t, self.v - dv, self.v + dv))
        self.w = float(np.clip(w_t, self.w - dw, self.w + dw))
        x, y, th = self.pose
        nx, ny = x + self.v * math.cos(th) * C.DT, y + self.v * math.sin(th) * C.DT
        self.path_len += math.hypot(nx - x, ny - y)
        self.pose = np.array([nx, ny, wrap_angle(th + self.w * C.DT)])
        self.steps += 1
        if self.drift_pos or self.drift_yaw:
            self.off += self.np_random.normal(0, [self.drift_pos, self.drift_pos, self.drift_yaw])

        collided = collides(nx, ny, self.rects)
        success = (not collided) and math.hypot(nx - self.goal[0], ny - self.goal[1]) < C.GOAL_TOLERANCE
        truncated = (not collided) and (not success) and self.steps >= self.max_steps

        geo = self._geo(nx, ny)
        obs = self._observe()
        min_rng = float(self._last_ranges.min())
        reward = C.R_PROGRESS * (self._prev_geo - geo) + C.R_STEP + C.R_TURN * abs(self.w) / C.W_MAX
        if min_rng < C.PROX_DIST:
            reward += C.R_PROXIMITY * (1.0 - min_rng / C.PROX_DIST)
        self._prev_geo = geo
        if success:
            reward += C.R_GOAL
        if collided:
            reward += C.R_COLLISION
        return obs, float(reward), bool(success or collided), bool(truncated), \
            self._info(success, collided, truncated)

    def step(self, action):
        """Gymnasium standard step interface: action in Box(-1.0, 1.0, (2,))."""
        v_t, w_t = action_to_cmd(action)
        return self.step_cmd(v_t, w_t)

    def _info(self, success: bool, collided: bool, timeout: bool) -> dict:
        done = success or collided or timeout
        spl = 0.0
        if success:
            spl = self.shortest / max(self.path_len, self.shortest, 1e-9)
        info = {"is_success": bool(success), "success": bool(success), "collision": bool(collided),
                "timeout": bool(timeout and not success), "path_length": self.path_len,
                "shortest_len": self.shortest, "spl": spl, "steps": self.steps,
                "time_s": self.steps * C.DT, "layout_seed": self.bundle.world.seed,
                "family": self.bundle.world.family}
        if done:
            info["terminal"] = True
        return info
