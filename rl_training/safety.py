"""Unified Robotics Safety Filter and Predictive Stopping-Distance Governor.

Shared across Track A (Gymnasium Benchmarks, baselines.py, interactive_ui.py)
and Track B (ROS 2 Jazzy node ros2_integration/policy_node.py).

Guarantees 100% parity between simulation benchmarking and physical robot deployment:
1. Layer 1: Multi-sector LiDAR scan monitoring (forward arc, flanks, rear).
2. Layer 2: Predictive Stopping-Distance Governor derived from kinematic braking physics (v^2 <= 2*a*d).
3. Layer 3: Lateral repulsive potential steering & active doorway/corridor midline centering.
4. Layer 4: Autonomous stagnation watchdog and verified rear-clearance reverse-and-pivot recovery state machine.

CRITICAL RESEARCH DESIGN:
- Stagnation is tracked using ESTIMATED/BELIEVED pose (odometry or SLAM pose), NEVER ground-truth simulation pose!
- Reverse velocity commands (v < 0) are strictly conditioned on rear LiDAR clearance (> ROBOT_RADIUS + 0.06m).
  If the rear is obstructed, the robot executes an in-place rotation pivot toward the maximum clearance sector.
- Post-policy supervisory filtering preserves the declared Gymnasium action space contract for the nominal RL actor.
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
    stop_distance_margin: float = 0.04            # m, safety buffer over robot radius
    corridor_detect_dist: float = 0.65            # m, threshold to detect bilateral constriction
    centering_gain: float = 1.15                  # P-gain for doorway midline tracking
    flank_repulse_dist: float = 0.45              # m, threshold for single-wall repulsion
    stagnation_history_len: int = 15              # frames (~1.5 s at 10 Hz)
    stagnation_disp_threshold: float = 0.04       # m, displacement below which stagnation triggers
    reverse_speed: float = -0.10                  # m/s, recovery backup velocity
    recovery_duration: int = 8                    # control steps (~0.8 s)
    enable_governor: bool = True                  # Layer 2: Stopping-distance braking ceiling
    enable_centering: bool = True                 # Layer 3: Lateral repulsion & doorway centering
    enable_recovery: bool = True                  # Layer 4: Stagnation watchdog & reverse recovery


class NavigationSafetyFilter:
    """Unified supervisory safety filter for mobile robot local navigation."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.cfg = config or SafetyConfig()
        self.recovery_steps = 0
        self.recovery_direction = 1.0
        self.history: list[tuple[float, float]] = []
        self.last_intervention = "none"

    def reset(self) -> None:
        """Reset internal recovery state machine and pose history between episodes."""
        self.recovery_steps = 0
        self.recovery_direction = 1.0
        self.history.clear()
        self.last_intervention = "none"

    def filter_command(
        self,
        raw_v: float,
        raw_w: float,
        sector_ranges: np.ndarray,
        estimated_pose: tuple[float, float] | np.ndarray | None = None,
        dt: float = C.DT,
    ) -> tuple[float, float, dict]:
        """Apply cascading safety layers to nominal velocity commands (v, w).

        Args:
            raw_v: Nominal linear velocity (m/s) from PPO or reactive planner.
            raw_w: Nominal angular velocity (rad/s).
            sector_ranges: 24-sector 360-degree LiDAR range array (in meters).
            estimated_pose: (x, y) believed robot coordinate from odometry or SLAM TF2.
                            MUST NOT be simulator ground truth!
            dt: Control cycle interval (s).

        Returns:
            (safe_v, safe_w, diagnostics_dict)
        """
        ranges = np.asarray(sector_ranges, dtype=np.float64)
        if ranges.shape[0] != C.N_SECTORS:
            raise ValueError(f"Expected {C.N_SECTORS} LiDAR sectors, got {ranges.shape[0]}")

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

            # Trigger condition: imminent collision or forward freeze stagnation
            front_is_blocked = min_front < (self.cfg.robot_radius + 0.05)
            imminent_collision = min_overall < (self.cfg.robot_radius + 0.015)
            stagnated_freeze = diag["is_stagnated"] and (front_is_blocked or raw_v < 0.05)

            if (imminent_collision or stagnated_freeze) and self.recovery_steps == 0:
                self.recovery_steps = self.cfg.recovery_duration
                closest_idx = int(np.argmin(ranges))
                self.recovery_direction = -1.0 if closest_idx >= 12 else 1.0
                self.history.clear()

            if self.recovery_steps > 0:
                self.recovery_steps -= 1
                diag["recovery_active"] = True

                # Verified rear clearance check before commanding reverse motion
                if rear_clear:
                    safe_v = self.cfg.reverse_speed
                    safe_w = 0.85 * self.recovery_direction
                    diag["intervention"] = "recovery_reverse"
                else:
                    # Rear is obstructed: execute zero-linear in-place pivot rotation
                    safe_v = 0.0
                    safe_w = 0.95 * self.recovery_direction
                    diag["intervention"] = "recovery_pivot"

                diag["v_safe"] = safe_v
                diag["w_safe"] = safe_w
                self.last_intervention = diag["intervention"]
                return safe_v, safe_w, diag

        # -------------------------------------------------------------
        # LAYER 2: PREDICTIVE STOPPING-DISTANCE GOVERNOR
        # -------------------------------------------------------------
        v = float(raw_v)
        w = float(raw_w)

        if self.cfg.enable_governor:
            clearance = max(0.0, min_wide - self.cfg.robot_radius - self.cfg.stop_distance_margin)
            # Analytical kinematic braking limit: v <= sqrt(2 * a * d)
            v_allowable = math.sqrt(2.0 * self.cfg.max_linear_accel * clearance)
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
        return v, w, diag
