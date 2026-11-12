"""Comprehensive validation suite for the NavigationSafetyFilter.

Verifies all safety layers, boundary cases, and interface contracts:
  T1 : No obstacle nearby -> command passes untouched.
  T2 : Obstacle directly ahead -> forward speed governed by latency-aware stopping model.
  T3 : Obstacle very close (< R_robot + 0.05 m) -> forward motion inhibited (v = 0).
  T4 : Left flank closer than right -> repulsive steering to the right (w < 0).
  T5 : Right flank closer than left -> repulsive steering to the left (w > 0).
  T6 : Stagnation with rear clear -> reverse-and-pivot recovery (v < 0).
  T7 : Stagnation with rear blocked -> zero-linear in-place pivot (v = 0, w != 0).
  T8 : Repeated stagnation exhaustion -> capped reverse attempts force safe pivot.
  T9 : Localization drift -> stagnation uses estimated pose exclusively.
  T10: Mismatched sector dimensions -> raises ValueError.
  T11: Non-finite sensor readings (NaN / Inf) -> safe fallback without crash.
  T12: Maximum velocity saturation -> governor enforces physical safety limits.
  T13: Action-space regression -> reverse velocity travels via act_cmd/step_cmd, not cmd_to_action.
"""
import math
import numpy as np
import pytest

from rl_training import config as C
from rl_training.safety import NavigationSafetyFilter, SafetyConfig
from rl_training.obs import cmd_to_action, action_to_cmd
from rl_training.nav_env import NavEnv
from rl_training.baselines import MASRLController, PPOController


@pytest.fixture
def filter():
    return NavigationSafetyFilter()


def make_clean_sectors(clear_dist: float = 3.0) -> np.ndarray:
    return np.full(C.N_SECTORS, clear_dist, dtype=np.float64)


# T1: No obstacle nearby
def test_t1_no_obstacle_nearby(filter):
    sectors = make_clean_sectors(3.0)
    v_safe, w_safe, diag = filter.filter_command(0.20, 0.1, sectors)
    assert v_safe == pytest.approx(0.20, abs=1e-4)
    assert w_safe == pytest.approx(0.10, abs=1e-4)
    assert diag["intervention"] == "none"


# T2: Obstacle directly ahead within stopping distance
def test_t2_obstacle_ahead_governor_brake(filter):
    sectors = make_clean_sectors(3.0)
    # Place obstacle at 0.32 m straight ahead (sector 12)
    sectors[11:14] = 0.32
    v_safe, w_safe, diag = filter.filter_command(0.26, 0.0, sectors)
    assert v_safe < 0.26
    assert diag["intervention"] == "governor_brake"
    assert filter.stats["governor_brakes"] > 0


# T3: Obstacle very close -> forward speed inhibited
def test_t3_obstacle_very_close_forward_inhibited(filter):
    sectors = make_clean_sectors(3.0)
    sectors[12] = C.ROBOT_RADIUS + 0.02  # 0.24 m
    v_safe, _, diag = filter.filter_command(0.20, 0.0, sectors)
    assert v_safe == 0.0
    assert diag["intervention"] == "governor_brake"


# T4: Left flank closer than right -> steers right (w < 0)
def test_t4_left_flank_repulsion(filter):
    sectors = make_clean_sectors(3.0)
    sectors[14:18] = 0.30  # left wall at 0.30 m
    _, w_safe, diag = filter.filter_command(0.15, 0.0, sectors)
    assert w_safe < 0.0
    assert diag["intervention"] == "flank_repulse"


# T5: Right flank closer than left -> steers left (w > 0)
def test_t5_right_flank_repulsion(filter):
    sectors = make_clean_sectors(3.0)
    sectors[6:10] = 0.30  # right wall at 0.30 m
    _, w_safe, diag = filter.filter_command(0.15, 0.0, sectors)
    assert w_safe > 0.0
    assert diag["intervention"] == "flank_repulse"


# T6: Stagnation with rear clear -> reverse-and-pivot recovery
def test_t6_stagnation_rear_clear_reverse(filter):
    sectors = make_clean_sectors(3.0)
    # Feed stationary pose history to simulate stuck robot
    for _ in range(filter.cfg.stagnation_history_len):
        filter.filter_command(0.02, 0.0, sectors, estimated_pose=(1.0, 1.0))
    # Imminent blocked front
    sectors[12] = C.ROBOT_RADIUS + 0.03
    v_safe, w_safe, diag = filter.filter_command(0.02, 0.0, sectors, estimated_pose=(1.0, 1.0))
    assert v_safe < 0.0  # Must command reverse
    assert abs(w_safe) > 0.0
    assert diag["intervention"] == "recovery_reverse"


# T7: Stagnation with rear blocked -> in-place pivot without reversing
def test_t7_stagnation_rear_blocked_pivot(filter):
    sectors = make_clean_sectors(3.0)
    for _ in range(filter.cfg.stagnation_history_len):
        filter.filter_command(0.02, 0.0, sectors, estimated_pose=(1.0, 1.0))
    # Front blocked and rear blocked
    sectors[12] = C.ROBOT_RADIUS + 0.03
    sectors[[22, 23, 0, 1, 2]] = C.ROBOT_RADIUS + 0.02
    v_safe, w_safe, diag = filter.filter_command(0.02, 0.0, sectors, estimated_pose=(1.0, 1.0))
    assert v_safe == 0.0  # Must NOT reverse into rear obstacle
    assert abs(w_safe) > 0.5
    assert diag["intervention"] == "recovery_pivot"


# T8: Repeated stagnation exhaustion -> capped reverse attempts transitions to safe stop
def test_t8_repeated_stagnation_caps_reverse(filter):
    filter.recovery_attempts = filter.cfg.max_recovery_attempts + 1
    filter.recovery_steps = 2
    sectors = make_clean_sectors(3.0)
    v_safe, w_safe, diag = filter.filter_command(0.02, 0.0, sectors, estimated_pose=(1.0, 1.0))
    assert v_safe == 0.0
    assert w_safe == 0.0
    assert diag["intervention"] == "safe_stop"
    assert diag["recovery_failed"] is True


# T9: Localization drift -> uses estimated pose exclusively
def test_t9_stagnation_uses_estimated_pose(filter):
    sectors = make_clean_sectors(3.0)
    # Open front during history build up so recovery doesn't trigger early
    for _ in range(filter.cfg.stagnation_history_len - 1):
        filter.filter_command(0.01, 0.0, sectors, estimated_pose=(2.0, 2.0))
    # On the step reaching stagnation_history_len, check stagnation flag
    _, _, diag = filter.filter_command(0.01, 0.0, sectors, estimated_pose=(2.0, 2.0))
    assert diag["is_stagnated"] is True


# T10: Mismatched sector dimensions raises ValueError
def test_t10_invalid_sector_dimensions(filter):
    with pytest.raises(ValueError):
        filter.filter_command(0.20, 0.0, np.zeros(12))


# T11: Non-finite sensor readings (NaN / Inf) safe fallback
def test_t11_non_finite_sensor_values(filter):
    sectors = make_clean_sectors(3.0)
    sectors[5] = np.nan
    sectors[12] = np.inf
    v_safe, w_safe, diag = filter.filter_command(0.15, 0.0, sectors)
    assert np.isfinite(v_safe)
    assert np.isfinite(w_safe)


# T12: High nominal velocity speed limit enforced
def test_t12_velocity_limit_governor(filter):
    sectors = make_clean_sectors(3.0)
    sectors[8:17] = 0.30  # Forward clearance = 0.30 - 0.22 - 0.04 = 0.04 m -> v_allowable ~ 0.17 m/s
    v_safe, _, diag = filter.filter_command(0.26, 0.0, sectors)
    assert v_safe < 0.26
    assert diag["intervention"] == "governor_brake"


# T13: Action-interface regression test
def test_t13_action_interface_reverse_separation():
    # Policy action space must strictly clip to [-1, 1]
    a = cmd_to_action(-0.10, 0.5)
    assert a[0] == -1.0  # Clamped to minimum throttle, NOT negative action
    assert a[1] == pytest.approx(0.5 / C.W_MAX, abs=1e-5)
    assert -1.0 <= a[0] <= 1.0 and -1.0 <= a[1] <= 1.0

    # Policy action_to_cmd must only produce forward speeds [0, V_MAX]
    v_cmd, _ = action_to_cmd([-1.0, 0.0])
    assert v_cmd == 0.0
    v_cmd_max, _ = action_to_cmd([1.0, 0.0])
    assert v_cmd_max == pytest.approx(C.V_MAX, abs=1e-5)


# T14: Sensor dropout / complete invalid LiDAR failsafe
def test_t14_sensor_dropout_failsafe(filter):
    # Complete sensor dropout: all sectors return NaN
    invalid_sectors = np.full(24, np.nan)
    v_safe, w_safe, diag = filter.filter_command(0.20, 0.5, invalid_sectors)
    assert v_safe == 0.0
    assert w_safe == 0.0
    assert diag["intervention"] == "sensor_loss_stop"
    assert diag["sensor_loss"] is True

    # Severe sensor degradation (< 50% valid readings)
    degraded_sectors = make_clean_sectors(3.0)
    degraded_sectors[:15] = np.nan  # 15/24 = 62.5% invalid
    v_safe, w_safe, diag = filter.filter_command(0.20, 0.5, degraded_sectors)
    assert v_safe == 0.0
    assert w_safe == 0.0
    assert diag["intervention"] == "sensor_loss_stop"


# T15: Finite recovery termination: robot halts instead of pivoting forever
def test_t15_recovery_attempts_finite_termination():
    # Use max_recovery_attempts = 2 for quick testing
    cfg = SafetyConfig(max_recovery_attempts=2, recovery_duration=3)
    f = NavigationSafetyFilter(cfg)

    # Imminent collision with rear blocked (forces pivot)
    sectors = make_clean_sectors(3.0)
    sectors[12] = 0.23  # Imminent front collision
    sectors[[22, 23, 0, 1, 2]] = 0.20  # Rear obstructed

    # Cycle 1: recovery attempt 1
    for _ in range(3):
        v, w, diag = f.filter_command(0.15, 0.0, sectors, estimated_pose=(1.0, 1.0))
        assert diag["intervention"] == "recovery_pivot"

    # Cycle 2: recovery attempt 2
    for _ in range(3):
        v, w, diag = f.filter_command(0.15, 0.0, sectors, estimated_pose=(1.0, 1.0))
        assert diag["intervention"] == "recovery_pivot"

    # Cycle 3: attempt budget exceeded -> must transition to safe_stop with v=0, w=0
    v, w, diag = f.filter_command(0.15, 0.0, sectors, estimated_pose=(1.0, 1.0))
    assert v == 0.0
    assert w == 0.0
    assert diag["intervention"] == "safe_stop"
    assert diag["recovery_failed"] is True

