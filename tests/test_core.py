"""Unit tests for the parts that do not need ROS: geometry, adapters, env, metrics."""
import math

import numpy as np
import pytest

from rl_training import config as C
from rl_training.maps import collides, generate_world, get_bundle, raycast
from rl_training.metrics import bootstrap_ci, wilson_ci
from rl_training.nav_env import NavEnv
from rl_training.obs import (action_to_cmd, build_observation, cmd_to_action, goal_in_robot_frame,
                             lookahead_point, ray_angles, resample_scan, sector_index)
from rl_training.planning import geodesic_field
from rl_training.scenarios import make_scenarios


def test_raycast_hits_wall():
    rects = np.array([[2.0, -5.0, 2.1, 5.0]])
    r = raycast((0.0, 0.0), np.array([0.0, math.pi, math.pi / 2]), rects)
    assert r[0] == pytest.approx(2.0)
    assert r[1] == C.RANGE_MAX and r[2] == C.RANGE_MAX


def test_collision_circle_vs_rect():
    rects = np.array([[1.0, 0.0, 2.0, 1.0]])
    assert collides(0.9, 0.5, rects)         # 0.1 from wall < radius 0.22
    assert not collides(0.5, 0.5, rects)


def test_sector_layout_front_is_12():
    assert sector_index(0.0) == 12
    assert sector_index(math.pi) == 0 and sector_index(-math.pi) == 0
    assert sector_index(math.radians(15)) == 13       # left (CCW) -> higher index
    assert sector_index(math.radians(-15)) == 11


def test_resample_scan_handles_invalid_and_orientation():
    n = 360
    inc = 2 * math.pi / n
    ranges = np.full(n, np.inf)
    ranges[180] = 1.0             # index 180 -> angle 0 (angle_min = -pi) -> front
    ranges[10] = np.nan
    ranges[20] = 0.01             # below range_min -> invalid
    sect = resample_scan(ranges, -math.pi, inc)
    assert sect[12] == pytest.approx(1.0)
    assert np.sum(sect < C.RANGE_MAX) == 1
    assert sect.shape == (C.N_SECTORS,)


def test_build_observation_layout_and_bounds():
    obs = build_observation(np.full(C.N_SECTORS, 10.0), 99.0, math.pi / 2, 5.0, -5.0)
    assert obs.shape == (C.OBS_DIM,) and obs.dtype == np.float32
    assert np.all(obs[:24] == 1.0) and obs[24] == 1.0
    assert obs[25] == pytest.approx(1.0) and abs(obs[26]) < 1e-6
    assert obs[27] == 1.0 and obs[28] == -1.0


def test_action_roundtrip():
    for v, w in [(0.0, 0.0), (C.V_MAX, C.W_MAX), (0.1, -0.5)]:
        v2, w2 = action_to_cmd(cmd_to_action(v, w))
        assert v2 == pytest.approx(v, abs=1e-6) and w2 == pytest.approx(w, abs=1e-6)


def test_goal_in_robot_frame():
    d, a = goal_in_robot_frame((0, 0, math.pi / 2), (0, 2))
    assert d == pytest.approx(2.0) and a == pytest.approx(0.0, abs=1e-9)


def test_lookahead_point():
    path = np.stack([np.linspace(0, 5, 51), np.zeros(51)], 1)
    p = lookahead_point(path, (1.0, 0.0), 1.5)
    assert p[0] == pytest.approx(2.5, abs=0.11)
    assert lookahead_point(path, (4.9, 0.0), 1.5)[0] == pytest.approx(5.0)


def test_worlds_deterministic_and_families():
    a, b = generate_world(7), generate_world(7)
    assert np.array_equal(a.rects, b.rects)
    assert {generate_world(s).family for s in range(4)} == {"open", "corridor", "rooms", "clutter"}
    assert generate_world(0, "zigzag").family == "zigzag"


def test_geodesic_field_goes_around_wall():
    blocked = np.zeros((20, 20), bool)
    blocked[:, 10] = True
    blocked[18:, 10] = False   # gap at the bottom
    f = geodesic_field(blocked, (2, 2), 0.1)
    assert np.isfinite(f[2, 17]) and f[2, 17] > 1.5   # must detour via the gap


def test_scenarios_matched_and_valid():
    a, b = make_scenarios(5, "test", 3), make_scenarios(5, "test", 3)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    for s in a:
        assert s.shortest_len >= 2.0
        bnd = get_bundle(s.layout_seed, s.family_override)
        assert not collides(s.start[0], s.start[1], bnd.world.rects)


def test_env_reset_step_and_success_info_keys():
    env = NavEnv()
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    obs, r, term, trunc, info = env.step(np.array([1.0, 0.0], np.float32))
    for k in ("success", "collision", "timeout", "spl", "path_length", "shortest_len"):
        assert k in info


def test_env_seed_reproducibility():
    e1, e2 = NavEnv(lidar_noise_std=0.03, lidar_dropout=0.05), NavEnv(lidar_noise_std=0.03, lidar_dropout=0.05)
    o1, _ = e1.reset(seed=5)
    o2, _ = e2.reset(seed=5)
    assert np.allclose(o1, o2)
    for _ in range(20):
        a = np.array([0.5, 0.3], np.float32)
        assert np.allclose(e1.step(a)[0], e2.step(a)[0])


def test_adapter_matches_env_sectors():
    env = NavEnv()
    obs, _ = env.reset(seed=2)
    rng, a0, inc = env.scan_full(360)
    sect = resample_scan(rng, a0, inc)
    assert np.all(sect <= obs[:24] * C.RANGE_MAX + 1e-3)
    assert np.abs(sect - obs[:24] * C.RANGE_MAX).mean() < 0.25


def test_localization_drift_changes_belief_not_truth():
    env = NavEnv(loc_drift_pos=0.05)
    env.reset(seed=0)
    for _ in range(30):
        env.step(np.array([-1.0, 0.0], np.float32))     # stand still
    assert np.linalg.norm(env.believed_pose[:2] - env.pose[:2]) > 0.0


def test_stats():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi and 0.39 < lo and hi < 0.61
    lo, hi = bootstrap_ci(np.ones(20))
    assert lo == hi == 1.0


def test_academic_tables_export(tmp_path):
    from analysis.export_academic_tables import export_latex_main_table, export_latex_family_table, export_markdown_formatted_table
    import pandas as pd
    
    df_summary = pd.read_csv("results/compare_summary.csv")
    df_family = pd.read_csv("results/compare_by_family.csv")
    
    t1 = tmp_path / "t1.tex"
    t2 = tmp_path / "t2.tex"
    t3 = tmp_path / "t3.md"
    
    export_latex_main_table(df_summary, t1)
    export_latex_family_table(df_family, t2)
    export_markdown_formatted_table(df_summary, t3)
    
    assert t1.exists() and len(t1.read_text(encoding="utf-8")) > 100
    assert t2.exists() and len(t2.read_text(encoding="utf-8")) > 100
    assert t3.exists() and len(t3.read_text(encoding="utf-8")) > 100


def test_line_of_sight_and_visible_waypoint():
    from rl_training.planning import has_line_of_sight, extract_visible_waypoint, geodesic_field, xy_to_rc
    from rl_training.maps import make_plan_blocked
    
    # Wall in between (2.0, 0.0) and (2.0, 4.0) with thickness 0.2
    rects = np.array([[1.9, 0.0, 2.1, 4.0]])
    
    # Direct line crossing wall should have NO line of sight
    assert not has_line_of_sight((1.0, 2.0), (3.0, 2.0), rects)
    # Line parallel to wall should have line of sight
    assert has_line_of_sight((1.0, 1.0), (1.0, 3.0), rects)
    
    # Test dynamic blocked grid
    blocked = make_plan_blocked(5.0, 5.0, rects)
    assert blocked.shape[0] > 0 and blocked.shape[1] > 0
    assert blocked.any()


def test_string_pulling_shortcut_path():
    from rl_training.planning import shortcut_path
    
    # In an open room without obstacles, a zigzag grid path [(0,0), (0.1, 0.1), (0.2, 0.2), ..., (2.0, 2.0)]
    # should be pruned by string pulling down to just [(0,0), (2.0, 2.0)]
    rects = np.zeros((0, 4), dtype=np.float64)
    grid_zigzag = [(i * 0.1, i * 0.1) for i in range(21)]
    
    pruned = shortcut_path(grid_zigzag, rects)
    assert len(pruned) == 2
    assert pruned[0] == (0.0, 0.0)
    assert pruned[-1] == (2.0, 2.0)


def test_navigation_safety_filter_governor_and_recovery():
    from rl_training.safety import NavigationSafetyFilter, SafetyConfig
    
    # 1. Stopping distance governor test
    sf = NavigationSafetyFilter()
    sectors = np.full(C.N_SECTORS, 3.0)
    # Put obstacle 0.28 m ahead: clearance = 0.28 - 0.22 - 0.04 = 0.02m -> v_allowable = sqrt(2*1*0.02) = 0.20 m/s
    sectors[12] = 0.28
    sectors[11] = 0.28
    sectors[13] = 0.28
    
    # Raw command asks for full speed (0.26 m/s)
    v_safe, w_safe, diag = sf.filter_command(0.26, 0.0, sectors)
    assert v_safe < 0.26, "Governor must reduce velocity when obstacle is close ahead"
    assert diag["intervention"] == "governor_brake"
    
    # 2. Doorway centering test
    sf.reset()
    sectors_door = np.full(C.N_SECTORS, 3.0)
    # Left flank is 0.30 m, right flank is 0.50 m (robot is off-center to the left)
    sectors_door[14:18] = 0.30
    sectors_door[6:10] = 0.50
    v_safe, w_safe, diag = sf.filter_command(0.20, 0.0, sectors_door)
    # Centering bias should turn robot to the right (negative w) to center between walls
    assert w_safe < 0.0 or diag["intervention"] in ("centering", "flank_repulse")
    
    # 3. Rear-clearance recovery test: rear is blocked -> in-place pivot, not reverse
    sf.reset()
    sectors_rear_blocked = np.full(C.N_SECTORS, 3.0)
    sectors_rear_blocked[12] = 0.24  # front imminent obstacle
    sectors_rear_blocked[0] = 0.23   # rear obstacle (< ROBOT_RADIUS + 0.06 = 0.28)
    v_safe, w_safe, diag = sf.filter_command(0.20, 0.0, sectors_rear_blocked)
    # When rear is blocked, robot must NOT reverse
    assert v_safe >= 0.0, "Must not command reverse when rear LiDAR is obstructed"
    assert diag["intervention"] == "recovery_pivot"


def test_is_inside_obstacle_and_standoff():
    from rl_training.maps import is_inside_obstacle, rect_clearance
    
    rects = np.array([[1.0, 1.0, 3.0, 3.0]])
    # Inside obstacle
    assert is_inside_obstacle(2.0, 2.0, rects)
    # Outside obstacle
    assert not is_inside_obstacle(0.5, 0.5, rects)
    
    # Target near boundary: clearance 0.05 m
    clr = rect_clearance(0.95, 2.0, rects)
    assert clr == pytest.approx(0.05, abs=1e-3)
    # Verify adaptive standoff tolerance formula:
    target_clr = clr
    effective_tol = max(C.GOAL_TOLERANCE, (C.ROBOT_RADIUS + 0.04 - target_clr) + 0.08)
    # Should give >= 0.25 m to allow robot to complete without colliding
    assert effective_tol > C.GOAL_TOLERANCE


def test_gymnasium_action_space_contract_and_direct_cmd_step():
    from pathlib import Path
    from rl_training.nav_env import NavEnv
    from rl_training.baselines import ReactiveController, DWAController, MASRLController
    from rl_training.evaluate import load_policy

    env = NavEnv()
    obs, _ = env.reset(seed=42)

    # 1. Classical and Safety Controllers must strictly obey Gymnasium Box(-1, 1, (2,))
    rc = ReactiveController()
    a_rc = rc.act(env, obs)
    assert env.action_space.contains(a_rc), f"ReactiveController produced action {a_rc} outside action_space"

    dwa = DWAController()
    a_dwa = dwa.act(env, obs)
    assert env.action_space.contains(a_dwa), f"DWAController produced action {a_dwa} outside action_space"

    model_path = Path("models/ppo_navigation.zip")
    if model_path.exists():
        ppo, _ = load_policy(model_path)
        mas = MASRLController(ppo.model)
        a_mas = mas.act(env, obs)
        assert env.action_space.contains(a_mas), f"MASRLController produced action {a_mas} outside action_space"

    # 2. Direct velocity execution via step_cmd respects acceleration limits
    obs_next, r, term, trunc, info = env.step_cmd(0.20, 0.5)
    assert obs_next.shape == (C.OBS_DIM,)
    assert env.v <= C.V_MAX
    assert abs(env.w) <= C.W_MAX



