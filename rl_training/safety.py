"""Autonomous Mobile Robot Navigation Safety Architecture.

Unified supervisory runtime safety filter providing:
1. Multi-sector forward arc collision forecasting (sectors 8 to 16, +/-60 deg).
2. Predictive stopping-distance governor incorporating discrete-time reaction latency:
      d_stop(v) = v * t_delay + v^2 / (2 * a_brake) <= d_clearance
      v_allowable = -a * t_delay + sqrt((a * t_delay)^2 + 2 * a * d_clearance)
3. Lateral repulsive potential field steering & doorway midline centering.
4. Non-privileged anti-deadlock watchdog with rear-clearance-verified recovery.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import config as C


@dataclass
class SafetyConfig:
    """Configurable safety thresholds conforming to TurtleBot3 Waffle kinematics."""
    robot_radius: float = C.ROBOT_RADIUS           # 0.22 m
    max_linear_accel: float = C.A_V_MAX           # 1.0 m/s^2
    max_angular_accel: float = C.A_W_MAX          # 3.2 rad/s^2
    reaction_time: float = 0.15                   # s, control/sensing latency delay (t_delay)
    stop_distance_margin: float = 0.04            # m, safety buffer over robot radius (d_margin)
    corridor_detect_dist: float = 0.65            # m, threshold to detect bilateral constriction
    centering_gain: float = 1.15                  # P-gain for doorway midline tracking
    flank_repulse_dist: float = 0.45              # m, threshold for single-wall repulsion
    stagnation_history_len: int = 15              # frames (~1.5 s at 10 Hz)
    stagnation_disp_threshold: float = 0.04       # m, displacement below which stagnation triggers
    reverse_speed: float = -0.10                  # m/s, recovery backup velocity
    recovery_duration: int = 8                    # control steps (~0.8 s)
    max_recovery_attempts: int = 4                # maximum repeated recovery cycles before timeout
    min_sensor_coverage: float = 0.50             # fraction of finite sectors required to avoid sensor loss stop
    forward_coverage_min: float = 0.50            # fraction of forward sectors (8..16) required for forward drive
    enable_governor: bool = True                  # Layer 2: Stopping-distance braking ceiling
    enable_centering: bool = True                 # Layer 3: Lateral repulsion & doorway centering
    enable_recovery: bool = True                  # Layer 4: Stagnation watchdog & reverse recovery


class NavigationSafetyFilter:
    """Unified supervisory safety filter for mobile robot local navigation."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.cfg = config or SafetyConfig()
        self.recovery_steps = 0
        self.recovery_direction = 1.0
        self.recovery_attempts = 0
        self.last_recovery_pos: tuple[float, float] | None = None
        self.history: list[tuple[float, float]] = []
        self.last_intervention = "none"
        self.recovery_failed_latched = False
        self.stats = {
            "governor_brakes": 0,
            "centering_interventions": 0,
            "flank_repulsions": 0,
            "recovery_reverses": 0,
            "recovery_pivots": 0,
            "deadlock_spins": 0,
            "sensor_loss_stops": 0,
            "safe_stops": 0,
            "total_steps": 0,
        }

    def reset(self) -> None:
        """Reset internal recovery state machine, pose history, and intervention stats between episodes."""
        self.recovery_steps = 0
        self.recovery_direction = 1.0
        self.recovery_attempts = 0
        self.last_recovery_pos = None
        self.history.clear()
        self.last_intervention = "none"
        self.recovery_failed_latched = False
        self.reset_stats()

    def reset_stats(self) -> None:
        """Reset cumulative intervention audit counters."""
        for k in self.stats:
            self.stats[k] = 0

    def filter_command(
        self,
        raw_v: float,
        raw_w: float,
        sector_ranges: np.ndarray,
        estimated_pose: tuple[float, float] | np.ndarray | None = None,
        dt: float = C.DT,
        valid_mask: np.ndarray | None = None,
    ) -> tuple[float, float, dict]:
        """Apply cascading safety layers to nominal velocity commands (v, w).

        Args:
            raw_v: Nominal linear velocity (m/s) from PPO or reactive planner.
            raw_w: Nominal angular velocity (rad/s).
            sector_ranges: 24-sector 360-degree LiDAR range array (in meters).
            estimated_pose: (x, y) believed robot coordinate from odometry or SLAM TF2.
                            MUST NOT be simulator ground truth!
            dt: Control cycle interval (s).
            valid_mask: Optional boolean array (shape 24) indicating which sectors contain valid readings.
                        If omitted, inferred from np.isfinite(sector_ranges).

        Returns:
            (safe_v, safe_w, diagnostics_dict)
        """
        ranges = np.asarray(sector_ranges, dtype=np.float64)
        if ranges.shape[0] != C.N_SECTORS:
            raise ValueError(f"Expected {C.N_SECTORS} LiDAR sectors, got {ranges.shape[0]}")

        # Check explicit or inferred sector validity mask
        if valid_mask is None:
            finite_mask = np.isfinite(ranges)
        else:
            finite_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(ranges)

        coverage = float(np.mean(finite_mask))
        forward_mask = finite_mask[8:17]  # sectors 8..16 (+/- 60 deg)
        forward_coverage = float(np.mean(forward_mask))

        # Check sensor coverage fail-safes
        if coverage < self.cfg.min_sensor_coverage:
            diag = {
                "intervention": "sensor_loss_stop",
                "sensor_loss": True,
                "coverage": coverage,
                "forward_coverage": forward_coverage,
                "v_raw": float(raw_v),
                "w_raw": float(raw_w),
                "v_safe": 0.0,
                "w_safe": 0.0,
                "min_front": 0.0,
                "min_wide": 0.0,
                "min_rear": 0.0,
                "min_overall": 0.0,
                "is_stagnated": False,
                "recovery_active": False,
                "recovery_attempts": self.recovery_attempts,
                "recovery_failed": False,
            }
            self.last_intervention = "sensor_loss_stop"
            self.stats["sensor_loss_stops"] += 1
            self.stats["total_steps"] += 1
            return 0.0, 0.0, diag
        elif raw_v > 0 and forward_coverage < self.cfg.forward_coverage_min:
            diag = {
                "intervention": "forward_sensor_loss_stop",
                "sensor_loss": True,
                "coverage": coverage,
                "forward_coverage": forward_coverage,
                "v_raw": float(raw_v),
                "w_raw": float(raw_w),
                "v_safe": 0.0,
                "w_safe": 0.0,
                "min_front": 0.0,
                "min_wide": 0.0,
                "min_rear": 0.0,
                "min_overall": 0.0,
                "is_stagnated": False,
                "recovery_active": False,
                "recovery_attempts": self.recovery_attempts,
                "recovery_failed": False,
            }
            self.last_intervention = "forward_sensor_loss_stop"
            self.stats["sensor_loss_stops"] += 1
            self.stats["total_steps"] += 1
            return 0.0, 0.0, diag

        # Latched safe stop check: once recovery budget is permanently exhausted,
        # remain in stationary safe stop until an explicit reset() is called
        if self.recovery_failed_latched:
            diag = {
                "intervention": "safe_stop",
                "sensor_loss": False,
                "coverage": coverage,
                "forward_coverage": forward_coverage,
                "v_raw": float(raw_v),
                "w_raw": float(raw_w),
                "v_safe": 0.0,
                "w_safe": 0.0,
                "min_front": 0.0,
                "min_wide": 0.0,
                "min_rear": 0.0,
                "min_overall": 0.0,
                "is_stagnated": True,
                "recovery_active": False,
                "recovery_attempts": self.recovery_attempts,
                "recovery_failed": True,
            }
            self.last_intervention = "safe_stop"
            self.stats["safe_stops"] += 1
            self.stats["total_steps"] += 1
            return 0.0, 0.0, diag

        # Conservative handling for any remaining non-finite sectors:
        # Instead of assuming wide-open free space (RANGE_MAX), default conservatively to min observed range
        if not finite_mask.all():
            min_valid = float(ranges[finite_mask].min()) if finite_mask.any() else self.cfg.robot_radius
            ranges = np.where(finite_mask, ranges, min_valid)
        ranges = np.clip(ranges, 0.0, C.RANGE_MAX)

        # Sector indexing convention:
        # Sector 12 is 0 rad (straight ahead)
        # Sectors 13..18 are left flank (+angle, CCW)
        # Sectors 6..11 are right flank (-angle, CW)
        # Sector 0 is -pi (straight behind), flanked by sectors 22, 23, 1, 2
        front_narrow = ranges[10:15]       # +/-30 deg
        front_wide = ranges[8:17]          # +/-60 deg
        min_front = float(front_narrow.min())
        min_wide = float(front_wide.min())
        min_overall = float(ranges.min())

        left_flank = float(ranges[14:18].min())
        right_flank = float(ranges[6:10].min())
        left_dist = float(ranges[13:18].min())
        right_dist = float(ranges[7:12].min())

        rear_sectors = ranges[[22, 23, 0, 1, 2]]
        min_rear = float(rear_sectors.min())
        rear_clear = min_rear > (self.cfg.robot_radius + 0.06)

        diag = {
            "intervention": "none",
            "v_raw": float(raw_v),
            "w_raw": float(raw_w),
            "v_safe": float(raw_v),
            "w_safe": float(raw_w),
            "min_front": min_front,
            "min_wide": min_wide,
            "min_rear": min_rear,
            "min_overall": min_overall,
            "is_stagnated": False,
            "recovery_active": self.recovery_steps > 0,
            "recovery_attempts": self.recovery_attempts,
            "sensor_loss": False,
            "recovery_failed": False,
        }

        # -------------------------------------------------------------
        # LAYER 4: ACTIVE RECOVERY STATE MACHINE
        # -------------------------------------------------------------
        if self.cfg.enable_recovery:
            # Stagnation monitoring using non-privileged estimated pose
            if estimated_pose is not None:
                ex, ey = float(estimated_pose[0]), float(estimated_pose[1])
                self.history.append((ex, ey))
                if len(self.history) > self.cfg.stagnation_history_len:
                    self.history.pop(0)

                if len(self.history) == self.cfg.stagnation_history_len:
                    disp = math.hypot(ex - self.history[0][0], ey - self.history[0][1])
                    if disp < self.cfg.stagnation_disp_threshold:
                        diag["is_stagnated"] = True

                # Reset recovery attempt count if robot has achieved net displacement >= 0.20 m
                if self.last_recovery_pos is not None:
                    if math.hypot(ex - self.last_recovery_pos[0], ey - self.last_recovery_pos[1]) >= 0.20:
                        self.recovery_attempts = 0
                        self.last_recovery_pos = None

            # Trigger condition: direction-aware imminent collision or forward freeze stagnation
            front_is_blocked = min_front < (self.cfg.robot_radius + 0.05)
            imminent_collision = min_overall < (self.cfg.robot_radius + 0.015)
            stagnated_freeze = diag["is_stagnated"] and (front_is_blocked or raw_v < 0.05)

            if (imminent_collision or stagnated_freeze) and self.recovery_steps == 0:
                if self.recovery_attempts >= self.cfg.max_recovery_attempts:
                    # Finite recovery budget exhausted: latch safe stop rather than spinning forever
                    self.recovery_attempts += 1
                    self.recovery_failed_latched = True
                    safe_v = 0.0
                    safe_w = 0.0
                    diag["intervention"] = "safe_stop"
                    diag["recovery_failed"] = True
                    diag["v_safe"] = safe_v
                    diag["w_safe"] = safe_w
                    self.last_intervention = "safe_stop"
                    self.stats["safe_stops"] += 1
                    self.stats["total_steps"] += 1
                    return safe_v, safe_w, diag

                self.recovery_steps = self.cfg.recovery_duration
                self.recovery_attempts += 1
                if estimated_pose is not None:
                    self.last_recovery_pos = (float(estimated_pose[0]), float(estimated_pose[1]))
                closest_idx = int(np.argmin(ranges))
                self.recovery_direction = -1.0 if closest_idx >= 12 else 1.0
                self.history.clear()

            if self.recovery_steps > 0:
                self.recovery_steps -= 1
                diag["recovery_active"] = True

                # Verified rear clearance check & attempt budget
                if rear_clear and self.recovery_attempts <= self.cfg.max_recovery_attempts:
                    rear_clearance = max(0.0, min_rear - self.cfg.robot_radius - self.cfg.stop_distance_margin)
                    v_rev_allowable = math.sqrt(2.0 * self.cfg.max_linear_accel * rear_clearance)
                    safe_v = -min(abs(self.cfg.reverse_speed), v_rev_allowable)
                    safe_w = 0.85 * self.recovery_direction
                    diag["intervention"] = "recovery_reverse"
                elif self.recovery_attempts <= self.cfg.max_recovery_attempts:
                    # Rear is obstructed: execute footprint-damped in-place pivot rotation
                    safe_v = 0.0
                    rot_clearance = min(left_flank, right_flank)
                    w_scale = float(np.clip((rot_clearance - self.cfg.robot_radius) / 0.15, 0.35, 1.0))
                    safe_w = 0.95 * self.recovery_direction * w_scale
                    diag["intervention"] = "recovery_pivot"
                else:
                    # Exhausted attempts: latch stationary safe halt
                    self.recovery_failed_latched = True
                    safe_v = 0.0
                    safe_w = 0.0
                    diag["intervention"] = "safe_stop"
                    diag["recovery_failed"] = True

                diag["v_safe"] = safe_v
                diag["w_safe"] = safe_w
                self.last_intervention = diag["intervention"]
                self.stats["total_steps"] += 1
                if diag["intervention"] == "recovery_reverse":
                    self.stats["recovery_reverses"] += 1
                elif diag["intervention"] == "recovery_pivot":
                    self.stats["recovery_pivots"] += 1
                else:
                    self.stats["safe_stops"] += 1
                return safe_v, safe_w, diag

        # -------------------------------------------------------------
        # LAYER 2: PREDICTIVE STOPPING-DISTANCE GOVERNOR
        # -------------------------------------------------------------
        v = float(raw_v)
        w = float(raw_w)

        if self.cfg.enable_governor:
            clearance = max(0.0, min_wide - self.cfg.robot_radius - self.cfg.stop_distance_margin)
            a = self.cfg.max_linear_accel
            t_delay = self.cfg.reaction_time
            # Realistic stopping distance model: d_stop(v) = v * t_delay + v^2 / (2 * a) <= clearance
            # Solving v^2 + 2 * a * t_delay * v - 2 * a * clearance = 0
            if clearance > 0.0:
                v_allowable = -a * t_delay + math.sqrt((a * t_delay) ** 2 + 2.0 * a * clearance)
                v_allowable = max(0.0, v_allowable)
            else:
                v_allowable = 0.0

            if v > v_allowable:
                v = v_allowable
                diag["intervention"] = "governor_brake"

            # Inhibit all forward velocity if front obstacle is within immediate braking horizon
            if min_front < (self.cfg.robot_radius + 0.05):
                v = 0.0
                diag["intervention"] = "governor_brake"

        # -------------------------------------------------------------
        # LAYER 3: LATERAL REPULSION & DOORWAY CENTERING TORQUE
        # -------------------------------------------------------------
        if self.cfg.enable_centering:
            repulse_w = 0.0

            # Bilateral constriction detected (doorway or narrow corridor)
            if left_dist < self.cfg.corridor_detect_dist and right_dist < self.cfg.corridor_detect_dist:
                centering_bias = self.cfg.centering_gain * (left_dist - right_dist)
                repulse_w += centering_bias
                v = min(v, 0.12)  # Throttle velocity for stable portal transit
                if abs(centering_bias) > 0.05 and diag["intervention"] == "none":
                    diag["intervention"] = "centering"
            else:
                # Unilateral wall repulsion
                if left_dist < self.cfg.flank_repulse_dist:
                    repulse_w -= 0.7 * (1.0 - left_dist / self.cfg.flank_repulse_dist)
                if right_dist < self.cfg.flank_repulse_dist:
                    repulse_w += 0.7 * (1.0 - right_dist / self.cfg.flank_repulse_dist)
                if abs(repulse_w) > 0.05 and diag["intervention"] == "none":
                    diag["intervention"] = "flank_repulse"

            w = float(np.clip(w + repulse_w, -C.W_MAX, C.W_MAX))

            # Flank pinch crawl protection
            if min(left_flank, right_flank) < (self.cfg.robot_radius + 0.035):
                v = min(v, 0.08)

        # -------------------------------------------------------------
        # ANTI-DEADLOCK SYMMETRY BREAKING
        # -------------------------------------------------------------
        if v == 0.0:
            preferred_dir = 1.0 if left_dist >= right_dist else -1.0
            if abs(w) < 0.4:
                w = 0.85 * preferred_dir
                if diag["intervention"] == "none":
                    diag["intervention"] = "deadlock_spin"

        diag["v_safe"] = v
        diag["w_safe"] = w
        self.last_intervention = diag["intervention"]

        # Cumulative intervention audit counters
        self.stats["total_steps"] += 1
        if diag["intervention"] == "governor_brake":
            self.stats["governor_brakes"] += 1
        elif diag["intervention"] == "centering":
            self.stats["centering_interventions"] += 1
        elif diag["intervention"] == "flank_repulse":
            self.stats["flank_repulsions"] += 1
        elif diag["intervention"] == "deadlock_spin":
            self.stats["deadlock_spins"] += 1

        return v, w, diag
