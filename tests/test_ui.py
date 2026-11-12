"""Unit tests for AutonomousNavApp Interactive UI telemetry and safety deadlock mechanisms."""
from __future__ import annotations

import math
import tkinter as tk
import pytest

from rl_training import config as C
from rl_training.interactive_ui import AutonomousNavApp


@pytest.fixture(scope="module")
def tk_app():
    root = tk.Tk()
    root.withdraw()
    app = AutonomousNavApp(root)
    app.is_running = False
    yield app
    try:
        app.on_close()
    except Exception:
        pass


def test_ui_telemetry_hud_fields_populated(tk_app):
    """Verify all 12 HUD cards are updated without throwing exceptions."""
    tk_app.update_telemetry_hud()
    
    assert tk_app.hud_cards["target_xy"].cget("text") == "(Click Map to Set)"
    assert "0.00 m" in tk_app.hud_cards["dist_metric"].cget("text")
    assert "Queued" in tk_app.hud_cards["wp_queue"].cget("text")
    assert "Optimal Direct" in tk_app.hud_cards["route_status"].cget("text")
    
    # Robot pose must show real float coordinates, not stuck default
    pose_text = tk_app.hud_cards["robot_pose"].cget("text")
    assert "X:" in pose_text and "Y:" in pose_text and "θ:" in pose_text
    
    # Speed and yaw rate
    assert "m/s" in tk_app.hud_cards["v_metric"].cget("text")
    assert "rad/s" in tk_app.hud_cards["w_metric"].cget("text")
    
    # LiDAR clearance must reflect raycast
    lidar_text = tk_app.hud_cards["lidar_metric"].cget("text")
    assert "m" in lidar_text
    assert tk_app.min_lidar_range < 3.5  # true raycast computed
    
    # Path metrics
    assert "m" in tk_app.hud_cards["path_metric"].cget("text")
    assert "m" in tk_app.hud_cards["shortest_metric"].cget("text")
    assert "0.000" in tk_app.hud_cards["spl_metric"].cget("text")
    assert "0 steps (0.0 s)" in tk_app.hud_cards["time_metric"].cget("text")


def test_ui_set_single_goal_and_step(tk_app):
    """Verify setting single goal plans path and stepping updates telemetry."""
    tk_app.set_single_goal(3.0, 3.0)
    assert tk_app.active_target == (3.0, 3.0)
    assert tk_app.target_dist > 0.0
    assert "X:3.00m, Y:3.00m" in tk_app.hud_cards["target_xy"].cget("text")
    
    # Step one control frame
    tk_app.step_one_frame()
    assert tk_app.step_count == 1
    assert "1 steps (0.1 s)" in tk_app.hud_cards["time_metric"].cget("text")


def test_ui_rotation_deadlock_halt(tk_app):
    """Verify persistent in-place spinning for > 6.0s triggers safe deadlock halt."""
    tk_app.set_single_goal(5.0, 5.0)
    tk_app.rotation_stuck_steps = 59
    tk_app.current_w = 0.5
    tk_app.execute_control_step()
    
    assert tk_app.is_running is False
    assert tk_app.current_v == 0.0
    assert tk_app.current_w == 0.0
    assert "DEADLOCK HALT" in tk_app.mission_status


def test_ui_mission_horizon_timeout(tk_app):
    """Verify mission terminates at C.MAX_STEPS (70.0s)."""
    tk_app.set_single_goal(5.0, 5.0)
    tk_app.rotation_stuck_steps = 0
    tk_app.step_count = C.MAX_STEPS
    tk_app.execute_control_step()
    
    assert tk_app.is_running is False
    assert tk_app.current_v == 0.0
    assert tk_app.current_w == 0.0
    assert "MISSION TIMEOUT" in tk_app.mission_status
