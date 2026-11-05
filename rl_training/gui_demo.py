"""Interactive Real-Time Graphical Simulator and Live Controller Comparison Demo.

Runs natively on Windows, Linux, and macOS without requiring ROS or Gazebo.
Displays real-time robot navigation, 24-sector LiDAR rays with danger coloring,
global geodesic path, lookahead waypoint, trajectory trail, and live HUD telemetry.

Usage:
    python -m rl_training.gui_demo
    python run_demo.py --controller ppo --family rooms
    python run_demo.py --save-gif results/figures/demo.gif --frames 200

Interactive Controls:
    [Space] : Pause / Resume
    [C]     : Cycle Controller (PPO -> DWA -> Reactive)
    [N]     : Load Next Scenario
    [M]     : Cycle Map Family (rooms -> corridor -> clutter -> open -> zigzag)
    [R]     : Reset Current Scenario
    [Click] : Click anywhere on free space to set a new goal dynamically!
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

# Ensure Matplotlib config dir is writable on Windows
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge, Arrow
from matplotlib.lines import Line2D
import numpy as np

from . import config as C
from .baselines import DWAController, ReactiveController, PPOController
from .evaluate import load_policy
from .maps import get_bundle, collides
from .nav_env import NavEnv
from .obs import cmd_to_action
from .planning import geodesic_field, xy_to_rc, rc_to_xy, smooth_sampler
from .scenarios import make_scenarios, Scenario


FAMILIES = ["rooms", "corridor", "clutter", "open", "zigzag"]


class InteractiveNavDemo:
    def __init__(
        self,
        model_path: str = "models/ppo_navigation",
        initial_controller: str = "ppo",
        family: str = "rooms",
        split: str = "test",
        seed: int = 42,
        save_gif: str | None = None,
        max_gif_frames: int = 250,
    ):
        self.split = split
        self.family_idx = FAMILIES.index(family) if family in FAMILIES else 0
        self.scenario_seed = seed
        self.save_gif = save_gif
        self.max_gif_frames = max_gif_frames
        self.gif_frames = []

        # Load controllers
        self.controllers = {}
        self.controller_names = []

        # PPO
        self.ppo_mode = "waypoint"
        try:
            ppo_ctrl, self.ppo_mode = load_policy(model_path)
            self.controllers["ppo"] = (ppo_ctrl, {"goal_mode": self.ppo_mode})
            self.controller_names.append("ppo")
        except Exception as e:
            print(f"[WARN] Could not load PPO model from {model_path}: {e}")

        # Baselines
        self.controllers["dwa"] = (DWAController(), {"goal_mode": "waypoint"})
        self.controller_names.append("dwa")
        self.controllers["reactive"] = (ReactiveController(), {"goal_mode": "direct"})
        self.controller_names.append("reactive")

        if initial_controller in self.controllers:
            self.active_ctrl_idx = self.controller_names.index(initial_controller)
        else:
            self.active_ctrl_idx = 0

        # Scenario management
        self.scenario_idx = 0
        self.scenarios = make_scenarios(50, split=self.split, seed=self.scenario_seed)
        self.paused = False
        self.step_delay = 0.03  # ~30 fps visual loop

        # Initialize Environment
        self.setup_scenario()

        # Initialize Matplotlib Figure
        self.init_figure()

    @property
    def current_family(self) -> str:
        return FAMILIES[self.family_idx]

    @property
    def active_controller_name(self) -> str:
        return self.controller_names[self.active_ctrl_idx]

    def setup_scenario(self, force_family: str | None = None):
        fam = force_family or self.current_family
        # Find or create a scenario with the specified family
        scn = self.scenarios[self.scenario_idx % len(self.scenarios)]
        scn.family_override = fam

        ctrl, kw = self.controllers[self.active_controller_name]
        self.env = NavEnv(family=fam, **kw)
        self.obs, self.info = self.env.reset(options={"scenario": scn})
        ctrl.reset(self.env)

        self.bundle = self.env.bundle
        self.world = self.bundle.world
        self.traj = [[float(self.env.pose[0]), float(self.env.pose[1])]]
        self.current_scn = scn
        self.terminated = False
        self.status_text = "NAVIGATING"
        self.current_reward = 0.0

    def init_figure(self):
        # Configure dark aesthetic styling for high-tech robotics feel
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(13, 8), dpi=100)
        self.fig.canvas.manager.set_window_title(
            "Autonomous Navigation: SLAM & RL Interactive Benchmarking Simulator"
        )

        # 2 Subplots: Left is 2D Map Simulation, Right is Real-time Telemetry Dashboard
        gs = self.fig.add_gridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.15)
        self.ax_map = self.fig.add_subplot(gs[0])
        self.ax_dash = self.fig.add_subplot(gs[1])

        # Hook interactive events
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

        self.draw_static_elements()

    def draw_static_elements(self):
        self.ax_map.clear()
        self.ax_dash.clear()

        # Background and Obstacles
        self.ax_map.set_facecolor("#1a1e24")
        w = self.world
        for x0, y0, x1, y1 in w.rects:
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#3a4454", edgecolor="#5c6b82", linewidth=1.2, zorder=2
            )
            self.ax_map.add_patch(rect)

        # Grid styling
        self.ax_map.set_xlim(-0.2, w.width + 0.2)
        self.ax_map.set_ylim(-0.2, w.height + 0.2)
        self.ax_map.set_aspect("equal")
        self.ax_map.grid(True, color="#2b3340", linestyle="--", linewidth=0.6, alpha=0.7)
        self.ax_map.set_xlabel("X Position (meters)", color="#a0aec0", fontsize=9)
        self.ax_map.set_ylabel("Y Position (meters)", color="#a0aec0", fontsize=9)
        self.ax_map.tick_params(colors="#718096", labelsize=8)

        # Title
        self.title_text = self.ax_map.set_title(
            f"Map Family: {self.current_family.upper()} | Controller: {self.active_controller_name.upper()}",
            color="#63b3ed", fontsize=12, fontweight="bold", pad=10
        )

        # Start & Goal Markers
        sx, sy = self.current_scn.start[:2]
        gx, gy = self.env.goal
        self.start_patch = Circle((sx, sy), C.ROBOT_RADIUS, color="#38a169", alpha=0.5, zorder=3, label="Start")
        self.goal_outer = Circle((gx, gy), C.GOAL_TOLERANCE, color="#e53e3e", alpha=0.35, zorder=3, label="Goal")
        self.goal_inner = Circle((gx, gy), 0.08, color="#e53e3e", alpha=0.9, zorder=4)
        self.ax_map.add_patch(self.start_patch)
        self.ax_map.add_patch(self.goal_outer)
        self.ax_map.add_patch(self.goal_inner)

        # Trajectory Trail Line
        self.line_traj, = self.ax_map.plot([], [], color="#4299e1", linewidth=2.0, alpha=0.8, zorder=4)

        # Robot Visual Representation
        self.robot_patch = Circle((sx, sy), C.ROBOT_RADIUS, facecolor="#3182ce", edgecolor="#63b3ed",
                                  linewidth=2.0, zorder=6)
        self.ax_map.add_patch(self.robot_patch)

        # Robot Heading Pointer
        self.heading_line, = self.ax_map.plot([], [], color="#f6e05e", linewidth=2.5, zorder=7)

        # Lookahead Waypoint Marker
        self.waypoint_patch = Circle((sx, sy), 0.12, facecolor="#ed8936", edgecolor="#dd6b20",
                                     linewidth=1.5, alpha=0.85, zorder=5)
        self.ax_map.add_patch(self.waypoint_patch)

        # LiDAR Ray LineCollection / Lines
        self.lidar_lines = []
        for _ in range(C.N_SECTORS):
            line, = self.ax_map.plot([], [], linewidth=0.9, alpha=0.75, zorder=3)
            self.lidar_lines.append(line)

        # Dashboard configuration
        self.ax_dash.set_facecolor("#14181d")
        self.ax_dash.axis("off")
        self.render_dashboard()

    def render_dashboard(self):
        self.ax_dash.clear()
        self.ax_dash.set_facecolor("#14181d")
        self.ax_dash.axis("off")

        # Telemetry Text Display
        v = self.env.v
        w = self.env.w
        steps = self.env.steps
        dist_goal = math.hypot(self.env.pose[0] - self.env.goal[0], self.env.pose[1] - self.env.goal[1])
        min_lidar = float(self.env._last_ranges.min()) if hasattr(self.env, "_last_ranges") else 0.0
        spl = self.info.get("spl", 0.0)

        status_color = "#68d391" if "REACHED" in self.status_text else (
            "#fc8181" if "COLLISION" in self.status_text else (
                "#f6ad55" if "TIMEOUT" in self.status_text else "#63b3ed"
            )
        )

        hud_text = [
            ("TELEMETRY HUD", "#e2e8f0", 13, "bold"),
            ("────────────────────────────────", "#4a5568", 9, "normal"),
            (f"Status:       {self.status_text}", status_color, 11, "bold"),
            (f"Controller:   {self.active_controller_name.upper()}", "#90cdf4", 10, "bold"),
            (f"Map Family:   {self.current_family.upper()}", "#cbd5e0", 10, "normal"),
            (f"Scenario ID:  #{self.current_scn.sid}", "#a0aec0", 10, "normal"),
            ("", "#000", 6, "normal"),
            ("DYNAMIC STATE", "#e2e8f0", 11, "bold"),
            (f"Velocity v:   {v:+.2f} m/s  (max {C.V_MAX})", "#68d391", 9.5, "normal"),
            (f"Yaw rate w:   {w:+.2f} rad/s (max {C.W_MAX})", "#f6e05e", 9.5, "normal"),
            (f"Goal Dist:    {dist_goal:.2f} m", "#63b3ed", 9.5, "normal"),
            (f"Min LiDAR:    {min_lidar:.2f} m", "#fc8181" if min_lidar < 0.35 else "#68d391", 9.5, "normal"),
            (f"Elapsed Time: {steps * C.DT:.1f} s ({steps} steps)", "#cbd5e0", 9.5, "normal"),
            (f"Path Length:  {self.env.path_len:.2f} m", "#cbd5e0", 9.5, "normal"),
            (f"Shortest Geo: {self.env.shortest:.2f} m", "#cbd5e0", 9.5, "normal"),
            (f"Current SPL:  {spl:.3f}", "#d6bcfa", 10, "bold"),
            ("", "#000", 6, "normal"),
            ("INTERACTIVE KEYBINDS", "#e2e8f0", 11, "bold"),
            ("────────────────────────────────", "#4a5568", 9, "normal"),
            ("[Space] : Pause / Resume", "#e2e8f0", 8.5, "normal"),
            ("[C]     : Toggle Controller", "#f6ad55", 8.5, "bold"),
            ("[M]     : Switch Map Family", "#63b3ed", 8.5, "normal"),
            ("[N]     : Next Scenario", "#68d391", 8.5, "normal"),
            ("[R]     : Reset Episode", "#fc8181", 8.5, "normal"),
            ("[Click] : Set New Goal Pose", "#f6e05e", 8.5, "bold"),
        ]

        y_pos = 0.96
        for text, col, fsz, weight in hud_text:
            self.ax_dash.text(0.05, y_pos, text, color=col, fontsize=fsz, fontweight=weight,
                              transform=self.ax_dash.transAxes, family="monospace")
            y_pos -= 0.038

    def update_visuals(self):
        x, y, th = self.env.pose
        self.traj.append([x, y])
        traj_arr = np.array(self.traj)

        # Update Robot and Heading
        self.robot_patch.center = (x, y)
        hx = x + (C.ROBOT_RADIUS + 0.15) * math.cos(th)
        hy = y + (C.ROBOT_RADIUS + 0.15) * math.sin(th)
        self.heading_line.set_data([x, hx], [y, hy])

        # Update Trajectory Trail
        self.line_traj.set_data(traj_arr[:, 0], traj_arr[:, 1])

        # Update Lookahead Waypoint
        wp = self.env.waypoint_world()
        self.waypoint_patch.center = (wp[0], wp[1])

        # Update Goal Marker position
        gx, gy = self.env.goal
        self.goal_outer.center = (gx, gy)
        self.goal_inner.center = (gx, gy)

        # Update LiDAR ray lines
        if hasattr(self.env, "_last_angles") and hasattr(self.env, "_last_ranges"):
            ang_rel = self.env._last_angles.mean(axis=1)  # 24 sectors
            rng_sec = self.env._last_ranges.min(axis=1)   # sector ranges
            ang_world = ang_rel + th

            for i in range(C.N_SECTORS):
                r = rng_sec[i]
                rx = x + r * math.cos(ang_world[i])
                ry = y + r * math.sin(ang_world[i])
                self.lidar_lines[i].set_data([x, rx], [y, ry])
                # Color code: red if < 0.5m, orange if < 1.2m, green otherwise
                if r < 0.5:
                    self.lidar_lines[i].set_color("#f56565")
                elif r < 1.2:
                    self.lidar_lines[i].set_color("#ecc94b")
                else:
                    self.lidar_lines[i].set_color("#48bb78")

        self.title_text.set_text(
            f"Map Family: {self.current_family.upper()} | Controller: {self.active_controller_name.upper()} | Steps: {self.env.steps}"
        )
        self.render_dashboard()
        self.fig.canvas.draw_idle()

    def step(self):
        if self.paused or self.terminated:
            return

        ctrl, _ = self.controllers[self.active_controller_name]
        action = ctrl.act(self.env, self.obs)
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.current_reward += reward

        if terminated or truncated:
            self.terminated = True
            if self.info["success"]:
                self.status_text = "GOAL REACHED! (+50 Reward)"
            elif self.info["collision"]:
                self.status_text = "COLLISION! (-50 Penalty)"
            elif self.info["timeout"]:
                self.status_text = "TIMEOUT! (> 70s Limit)"

        self.update_visuals()

        # Capture GIF frame if requested
        if self.save_gif and len(self.gif_frames) < self.max_gif_frames:
            self.fig.canvas.draw()
            try:
                buf = np.asarray(self.fig.canvas.buffer_rgba())
                self.gif_frames.append(buf[:, :, :3].copy())
            except Exception:
                try:
                    w, h = self.fig.canvas.get_width_height()
                    buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape((h, w, 3))
                    self.gif_frames.append(buf.copy())
                except Exception:
                    pass

    def export_gif(self):
        try:
            from PIL import Image
            print(f"[INFO] Exporting {len(self.gif_frames)} frames to {self.save_gif}...")
            imgs = [Image.fromarray(f) for f in self.gif_frames]
            Path(self.save_gif).parent.mkdir(parents=True, exist_ok=True)
            imgs[0].save(
                self.save_gif,
                save_all=True,
                append_images=imgs[1:],
                optimize=True,
                duration=int(C.DT * 1000),
                loop=0,
            )
            print(f"[SUCCESS] Saved animated demonstration to {self.save_gif}")
        except Exception as e:
            print(f"[ERROR] Could not save GIF: {e}")

    # ---------------------------------------------------- Interactive handlers
    def on_key_press(self, event):
        key = (event.key or "").lower()
        if key == " ":
            self.paused = not self.paused
            self.status_text = "PAUSED" if self.paused else "NAVIGATING"
            self.render_dashboard()
            self.fig.canvas.draw_idle()
        elif key == "c":
            # Cycle controller
            self.active_ctrl_idx = (self.active_ctrl_idx + 1) % len(self.controller_names)
            print(f"[SWITCH] Controller: {self.active_controller_name.upper()}")
            self.setup_scenario()
            self.draw_static_elements()
            self.update_visuals()
        elif key == "m":
            # Switch map family
            self.family_idx = (self.family_idx + 1) % len(FAMILIES)
            print(f"[SWITCH] Map Family: {self.current_family.upper()}")
            self.setup_scenario()
            self.draw_static_elements()
            self.update_visuals()
        elif key == "n":
            # Next scenario
            self.scenario_idx += 1
            print(f"[NEXT] Scenario #{self.scenario_idx}")
            self.setup_scenario()
            self.draw_static_elements()
            self.update_visuals()
        elif key == "r":
            # Reset
            print("[RESET] Restarting current scenario")
            self.setup_scenario()
            self.draw_static_elements()
            self.update_visuals()

    def on_click(self, event):
        # Click on map to reassign goal!
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return
        gx, gy = float(event.xdata), float(event.ydata)

        # Validate that clicked location does not collide with obstacles
        if collides(gx, gy, self.env.rects, radius=C.ROBOT_RADIUS):
            print(f"[WARN] Clicked point ({gx:.2f}, {gy:.2f}) inside obstacle or too close to wall!")
            return

        print(f"[GOAL] New goal dynamically set to ({gx:.2f}, {gy:.2f})")
        self.env.goal = np.array([gx, gy], dtype=np.float64)
        # Recompute geodesic potential field
        self.env.field = geodesic_field(
            self.env.bundle.plan_blocked, xy_to_rc(gx, gy, self.env.bundle.occ.shape)
        )
        self.env._geo = smooth_sampler(self.env.field)
        self.env._prev_geo = self.env._geo(self.env.pose[0], self.env.pose[1])
        self.env.shortest = float(self.env._prev_geo)
        self.terminated = False
        self.status_text = "NAVIGATING TO USER GOAL"
        self.update_visuals()

    def run(self):
        plt.ion()
        plt.show(block=False)
        self.update_visuals()

        try:
            while plt.fignum_exists(self.fig.number):
                self.step()
                if self.save_gif and len(self.gif_frames) >= self.max_gif_frames:
                    break
                plt.pause(self.step_delay)
        except KeyboardInterrupt:
            pass
        finally:
            if self.save_gif and self.gif_frames:
                self.export_gif()
            plt.close(self.fig)


def main():
    parser = argparse.ArgumentParser(description="Interactive Real-Time Autonomous Navigation Demo")
    parser.add_argument("--model", default="models/ppo_navigation", help="Path to trained PPO model")
    parser.add_argument("--controller", default="ppo", choices=["ppo", "dwa", "reactive"],
                        help="Initial controller to run")
    parser.add_argument("--family", default="rooms", choices=FAMILIES,
                        help="Initial indoor layout family")
    parser.add_argument("--split", default="test", choices=["test", "heldout", "val"],
                        help="Data split to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random scenario seed")
    parser.add_argument("--save-gif", default=None, help="Optional path to export animated GIF")
    parser.add_argument("--frames", type=int, default=200, help="Max frames for GIF export")
    args = parser.parse_args()

    demo = InteractiveNavDemo(
        model_path=args.model,
        initial_controller=args.controller,
        family=args.family,
        split=args.split,
        seed=args.seed,
        save_gif=args.save_gif,
        max_gif_frames=args.frames,
    )
    demo.run()


if __name__ == "__main__":
    main()
