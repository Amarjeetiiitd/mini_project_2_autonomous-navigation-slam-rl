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

