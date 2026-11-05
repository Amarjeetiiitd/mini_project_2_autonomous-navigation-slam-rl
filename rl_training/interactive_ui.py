"""Interactive Autonomous Navigation GUI with Dynamic Point Drawing & Telemetry Dashboard.

Allows professors and researchers to:
1. Click or draw ANY target point on the map after launching.
2. Draw multi-waypoint routes (Point 1 -> Point 2 -> Point 3 -> Target).
3. Draw/place custom dynamic obstacles to test real-time LiDAR obstacle avoidance.
4. Hot-swap controllers live (PPO Neural Network, DWA Local Planner, Reactive).
5. Switch map environments (Multi-room, Corridors, Clutter, Open, Zigzag).
6. View real-time 24-sector LiDAR rays with safety color-coding, trajectory trail,
   global geodesic paths, lookahead waypoints, and detailed telemetry HUD.

Usage:
    python run_ui.py
    python -m rl_training.interactive_ui
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure Matplotlib config dir is writable on Windows
if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_cache")

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, Rectangle, Polygon
from matplotlib.lines import Line2D

from . import config as C
from .baselines import DWAController, ReactiveController, PPOController
from .evaluate import load_policy
from .maps import get_bundle, collides, generate_world
from .nav_env import NavEnv
from .obs import cmd_to_action, action_to_cmd, goal_in_robot_frame, build_observation
from .planning import geodesic_field, xy_to_rc, rc_to_xy, smooth_sampler, extract_path, nearest_finite


FAMILIES = ["rooms", "corridor", "clutter", "open", "zigzag"]


class AutonomousNavApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Autonomous Mobile Robot Navigation (Online SLAM & Deep RL) - Interactive Mission Control")
        self.root.geometry("1420x920")
        self.root.minsize(1100, 750)
        self.root.configure(bg="#12151a")

        # System State Variables
        self.active_controller_name = "ppo_shielded"
        self.active_family = "rooms"
        self.current_seed = 42
        self.interaction_mode = tk.StringVar(value="set_goal")  # set_goal | multi_waypoint | add_obstacle | set_robot
        self.is_running = False
        self.sim_speed_factor = tk.DoubleVar(value=1.0)
        self.show_lidar_rays = tk.BooleanVar(value=True)
        self.show_global_path = tk.BooleanVar(value=True)
        self.show_trajectory_trail = tk.BooleanVar(value=True)
        self.slam_exploration_mode = tk.BooleanVar(value=False)
        self.safety_shield_enabled = tk.BooleanVar(value=True)
        self.recovery_mode = False
        self.recovery_steps = 0
        self.recovery_direction = 1.0

        # Mission Waypoints Queue
        self.waypoint_queue = []  # list of (x, y) tuples
        self.completed_waypoints = []
        self.custom_obstacles = []  # list of (x0, y0, x1, y1)

        # Navigation telemetry
        self.mission_status = "READY - Click anywhere on the map to draw a target point!"
        self.step_count = 0
        self.path_traversed = 0.0
        self.min_lidar_range = 3.5
        self.current_v = 0.0
        self.current_w = 0.0
        self.spl_metric = 0.0
        self.target_dist = 0.0

        # Load PPO Model and Controllers
        self.controllers = {}
        self.load_all_controllers()

        # Build World and Environment
        self.setup_environment()

        # Create GUI Layout
        self.build_user_interface()

        # Start GUI Update Loop
        self.last_frame_time = time.time()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(30, self.simulation_step_loop)

    def on_close(self):
        self.is_running = False
        try:
            plt.close(self.fig)
        except Exception:
            pass
        self.root.destroy()

    def load_all_controllers(self):
        from .baselines import ShieldedPPOController
        model_path = Path("models/ppo_navigation")
        if model_path.with_suffix(".zip").exists():
            try:
                ppo_ctrl, self.ppo_mode = load_policy(str(model_path))
                self.controllers["ppo"] = ppo_ctrl
                self.controllers["ppo_shielded"] = ShieldedPPOController(ppo_ctrl.model)
            except Exception as e:
                print(f"[WARN] Error loading PPO model: {e}")
                self.controllers["ppo"] = None
                self.controllers["ppo_shielded"] = None
        else:
            self.controllers["ppo"] = None
            self.controllers["ppo_shielded"] = None

        self.controllers["dwa"] = DWAController()
        self.controllers["reactive"] = ReactiveController()

    def setup_environment(self):
        # Generate procedural layout bundle
        self.bundle = get_bundle(self.current_seed, self.active_family)
        self.world = self.bundle.world
        self.rects = list(self.world.rects)

        # Merge custom obstacles if any
        if self.custom_obstacles:
            self.rects.extend(self.custom_obstacles)
        self.rects_arr = np.array(self.rects)

        # Robot initial pose (spawn at clear location)
        # Search for a safe clear spawn point
        w = self.world
        sx, sy = 1.0, 1.0
        found_spawn = False
        for test_x in np.linspace(0.8, w.width - 0.8, 15):
            for test_y in np.linspace(0.8, w.height - 0.8, 15):
                if not collides(test_x, test_y, self.rects_arr, radius=C.ROBOT_RADIUS + 0.15):
                    sx, sy = test_x, test_y
                    found_spawn = True
                    break
            if found_spawn:
                break

        self.robot_pose = np.array([sx, sy, 0.0], dtype=np.float64)  # x, y, yaw
        self.current_v = 0.0
        self.current_w = 0.0
        self.trajectory_history = [[sx, sy]]
        self.step_count = 0
        self.path_traversed = 0.0

        # Goal and Field initialization
        self.active_target = None
        self.geodesic_field_map = None
        self.field_sampler = None
        self.shortest_geo_len = 0.0

        # Discovered occupancy grid (for SLAM exploration view)
        ny, nx = self.bundle.occ.shape
        self.discovered_grid = np.zeros((ny, nx), dtype=bool)

        # Create underlying Gymnasium env for raycasting & helper functions
        self.env = NavEnv(family=self.active_family, goal_mode="waypoint")
        self.env.reset(options={"layout_seed": self.current_seed})
        self.env.rects = self.rects_arr
        self.env.pose = self.robot_pose.copy()

    def build_user_interface(self):
        # Top Header Bar
        header_frame = tk.Frame(self.root, bg="#1a1e24", height=60, relief="flat", bd=0)
        header_frame.pack(fill="x", side="top", padx=0, pady=0)

        title_label = tk.Label(
            header_frame,
            text="AUTONOMOUS ROBOT NAVIGATION WITH SLAM & DEEP RL",
            font=("Segoe UI", 14, "bold"),
            fg="#63b3ed",
            bg="#1a1e24",
        )
        title_label.pack(side="left", padx=20, pady=10)

        sub_label = tk.Label(
            header_frame,
            text="Research-Grade Benchmark | Dynamic Point Drawing & Real-Time LiDAR Control",
            font=("Segoe UI", 9, "italic"),
            fg="#a0aec0",
            bg="#1a1e24",
        )
        sub_label.pack(side="left", padx=5, pady=14)

        # Main Body Split: Left is Control Sidebar, Center/Right is Map Canvas & Telemetry HUD
        main_split = tk.Frame(self.root, bg="#12151a")
        main_split.pack(fill="both", expand=True, padx=10, pady=10)

        # ------------------ LEFT SIDEBAR: CONTROL PANEL ------------------
        sidebar = tk.Frame(main_split, bg="#1a1e24", width=340, padx=15, pady=15, relief="ridge", bd=1)
        sidebar.pack(side="left", fill="y", padx=(0, 10))
        sidebar.pack_propagate(False)

        # Mission Interaction Mode Section
        self.create_section_label(sidebar, "1. INTERACTION MODE")
        mode_frame = tk.Frame(sidebar, bg="#1a1e24")
        mode_frame.pack(fill="x", pady=5)

        modes = [
            ("🎯 Set Single Target Goal", "set_goal", "#4299e1"),
            ("📍 Draw Multi-Waypoint Route", "multi_waypoint", "#48bb78"),
            ("🧱 Draw Custom Obstacle Box", "add_obstacle", "#ed8936"),
            ("🤖 Relocate Robot Pose", "set_robot", "#9f7aea"),
        ]
        for text, val, col in modes:
            rb = tk.Radiobutton(
                mode_frame,
                text=text,
                value=val,
                variable=self.interaction_mode,
                font=("Segoe UI", 9, "bold"),
                fg=col,
                bg="#1a1e24",
                selectcolor="#2d3748",
                activebackground="#1a1e24",
                activeforeground=col,
                command=self.on_mode_change,
            )
            rb.pack(anchor="w", pady=2)

        # Controller Selection Section
        self.create_section_label(sidebar, "2. LOCAL CONTROLLER")
        ctrl_frame = tk.Frame(sidebar, bg="#1a1e24")
        ctrl_frame.pack(fill="x", pady=5)

        self.ctrl_var = tk.StringVar(value="ppo_shielded")
        ctrls = [
            ("🛡️ PPO + Dynamic Safety Shield", "ppo_shielded"),
            ("🧠 PPO Local Policy (Raw Neural Net)", "ppo"),
            ("⚙️ DWA Planner (Dynamic Window)", "dwa"),
            ("⚡ Reactive Baseline (LiDAR)", "reactive"),
        ]
        for text, val in ctrls:
            rb = tk.Radiobutton(
                ctrl_frame,
                text=text,
                value=val,
                variable=self.ctrl_var,
                font=("Segoe UI", 9),
                fg="#e2e8f0",
                bg="#1a1e24",
                selectcolor="#2d3748",
                activebackground="#1a1e24",
                command=self.on_controller_change,
            )
            rb.pack(anchor="w", pady=2)

        # Map Environment Selection Section
        self.create_section_label(sidebar, "3. ENVIRONMENT TOPOLOGY")
        map_frame = tk.Frame(sidebar, bg="#1a1e24")
        map_frame.pack(fill="x", pady=5)

        map_label = tk.Label(map_frame, text="Select Floorplan Preset:", font=("Segoe UI", 8), fg="#cbd5e0", bg="#1a1e24")
        map_label.pack(anchor="w")

        self.map_combobox = ttk.Combobox(
            map_frame,
            values=[
                "Multi-Room Apartments (rooms)",
                "Intersecting Corridors (corridor)",
                "Warehouse Obstacle Clutter (clutter)",
                "Open Test Arena (open)",
                "Held-Out ZigZag Labyrinth (zigzag)",
            ],
            state="readonly",
            font=("Segoe UI", 9),
        )
        self.map_combobox.current(0)
        self.map_combobox.pack(fill="x", pady=4)
        self.map_combobox.bind("<<ComboboxSelected>>", self.on_map_preset_change)

        btn_new_seed = tk.Button(
            map_frame,
            text="🎲 Generate New Random Layout",
            font=("Segoe UI", 8, "bold"),
            bg="#2d3748",
            fg="#e2e8f0",
            relief="flat",
            command=self.generate_new_layout,
        )
        btn_new_seed.pack(fill="x", pady=4)

        # Mission Execution Controls Section
        self.create_section_label(sidebar, "4. MISSION CONTROLS")
        exec_frame = tk.Frame(sidebar, bg="#1a1e24")
        exec_frame.pack(fill="x", pady=5)

        self.btn_run = tk.Button(
            exec_frame,
            text="▶ START NAVIGATION",
            font=("Segoe UI", 10, "bold"),
            bg="#38a169",
            fg="white",
            relief="flat",
            pady=6,
            command=self.toggle_navigation,
        )
        self.btn_run.pack(fill="x", pady=3)

        btn_step = tk.Button(
            exec_frame,
            text="⏭ Step 1 Frame (100ms)",
            font=("Segoe UI", 8),
            bg="#2d3748",
            fg="#e2e8f0",
            relief="flat",
            command=self.step_one_frame,
        )
        btn_step.pack(fill="x", pady=2)

        btn_clear_wp = tk.Button(
            exec_frame,
            text="🗑 Clear Drawn Points / Reset Route",
            font=("Segoe UI", 8),
            bg="#c53030",
            fg="white",
            relief="flat",
            command=self.clear_all_waypoints,
        )
        btn_clear_wp.pack(fill="x", pady=2)

        btn_clear_obs = tk.Button(
            exec_frame,
            text="🧹 Clear Custom Obstacles",
            font=("Segoe UI", 8),
            bg="#4a5568",
            fg="white",
            relief="flat",
            command=self.clear_custom_obstacles,
        )
        btn_clear_obs.pack(fill="x", pady=2)

        # Simulation Speed Slider
        speed_label = tk.Label(sidebar, text="Simulation Execution Speed:", font=("Segoe UI", 8), fg="#cbd5e0", bg="#1a1e24")
        speed_label.pack(anchor="w", pady=(8, 0))
        speed_slider = tk.Scale(
            sidebar,
            from_=0.2,
            to=3.0,
            resolution=0.1,
            orient="horizontal",
            variable=self.sim_speed_factor,
            bg="#1a1e24",
            fg="#e2e8f0",
            highlightthickness=0,
            troughcolor="#2d3748",
        )
        speed_slider.pack(fill="x", pady=2)

        # Visual Layer Toggles Section
        self.create_section_label(sidebar, "5. SAFETY & VISUAL LAYERS")
        vis_frame = tk.Frame(sidebar, bg="#1a1e24")
        vis_frame.pack(fill="x", pady=2)

        chk_shield = tk.Checkbutton(
            vis_frame, text="🛡️ Collision Shield & Auto-Recovery", variable=self.safety_shield_enabled,
            font=("Segoe UI", 8, "bold"), fg="#48bb78", bg="#1a1e24", selectcolor="#2d3748",
            activebackground="#1a1e24"
        )
        chk_shield.pack(anchor="w")

        chk_lidar = tk.Checkbutton(
            vis_frame, text="24-Sector LiDAR Ray Beams", variable=self.show_lidar_rays,
            font=("Segoe UI", 8), fg="#e2e8f0", bg="#1a1e24", selectcolor="#2d3748",
            activebackground="#1a1e24", command=self.redraw_canvas
        )
        chk_lidar.pack(anchor="w")

        chk_path = tk.Checkbutton(
            vis_frame, text="Global Geodesic Planned Path", variable=self.show_global_path,
            font=("Segoe UI", 8), fg="#e2e8f0", bg="#1a1e24", selectcolor="#2d3748",
            activebackground="#1a1e24", command=self.redraw_canvas
        )
        chk_path.pack(anchor="w")

        chk_trail = tk.Checkbutton(
            vis_frame, text="Robot Historic Trajectory Trail", variable=self.show_trajectory_trail,
            font=("Segoe UI", 8), fg="#e2e8f0", bg="#1a1e24", selectcolor="#2d3748",
            activebackground="#1a1e24", command=self.redraw_canvas
        )
        chk_trail.pack(anchor="w")

        chk_slam = tk.Checkbutton(
            vis_frame, text="SLAM Exploration (Fog of War)", variable=self.slam_exploration_mode,
            font=("Segoe UI", 8), fg="#63b3ed", bg="#1a1e24", selectcolor="#2d3748",
            activebackground="#1a1e24", command=self.redraw_canvas
        )
        chk_slam.pack(anchor="w")

        # ------------------ CENTER/RIGHT: CANVAS & DASHBOARD ------------------
        right_container = tk.Frame(main_split, bg="#12151a")
        right_container.pack(side="right", fill="both", expand=True)

        # Status Notification Banner
        self.banner_frame = tk.Frame(right_container, bg="#2b6cb0", height=38)
        self.banner_frame.pack(fill="x", pady=(0, 8))
        self.banner_label = tk.Label(
            self.banner_frame,
            text=self.mission_status,
            font=("Segoe UI", 10, "bold"),
            fg="white",
            bg="#2b6cb0",
            pady=6,
        )
        self.banner_label.pack(fill="both")

        # Top Canvas + Sidebar Split
        canvas_dashboard_split = tk.Frame(right_container, bg="#12151a")
        canvas_dashboard_split.pack(fill="both", expand=True)

        # Embedded Matplotlib Canvas for High-Performance Rendering
        self.fig, self.ax = plt.subplots(figsize=(8.5, 7.0), dpi=100)
        self.fig.patch.set_facecolor("#16191f")
        self.ax.set_facecolor("#1a1e24")

        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_dashboard_split)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side="left", fill="both", expand=True)

        # Bind Mouse Click and Drag Events for Drawing Points and Obstacles
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)
        self.canvas.mpl_connect("button_release_event", self.on_canvas_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_canvas_drag)

        self.drag_start_point = None

        # Right Telemetry HUD Panel
        telemetry_panel = tk.Frame(canvas_dashboard_split, bg="#1a1e24", width=300, padx=12, pady=10, relief="ridge", bd=1)
        telemetry_panel.pack(side="right", fill="y", padx=(10, 0))
        telemetry_panel.pack_propagate(False)

        self.build_telemetry_hud(telemetry_panel)

        # Initial Draw
        self.redraw_canvas()

    def create_section_label(self, parent, text):
        lbl = tk.Label(parent, text=text, font=("Segoe UI", 9, "bold"), fg="#63b3ed", bg="#1a1e24")
        lbl.pack(anchor="w", pady=(12, 2))
        sep = tk.Frame(parent, bg="#2d3748", height=1)
        sep.pack(fill="x", pady=(0, 4))

    def build_telemetry_hud(self, parent):
        hud_title = tk.Label(parent, text="LIVE ROBOT TELEMETRY", font=("Segoe UI", 11, "bold"), fg="#e2e8f0", bg="#1a1e24")
        hud_title.pack(anchor="w", pady=(0, 6))

        # Dynamic Metrics Cards
        self.hud_cards = {}
        fields = [
            ("Target Coordinates:", "target_xy", "(Not Set)"),
            ("Waypoints Remaining:", "wp_queue", "0 points"),
            ("Robot Pose (x, y, θ):", "robot_pose", "0.00 m, 0.00 m, 0.0°"),
            ("Linear Velocity (v):", "v_metric", "0.00 m/s"),
            ("Angular Velocity (ω):", "w_metric", "0.00 rad/s"),
            ("Distance to Goal:", "dist_metric", "0.00 m"),
            ("Min LiDAR Clearance:", "lidar_metric", "3.50 m"),
            ("Path Traversed:", "path_metric", "0.00 m"),
            ("Shortest Path (Geo):", "shortest_metric", "0.00 m"),
            ("SPL Path Efficiency:", "spl_metric", "0.000"),
            ("Mission Steps / Time:", "time_metric", "0 steps (0.0 s)"),
        ]

        for label_text, key, init_val in fields:
            card = tk.Frame(parent, bg="#14171d", padx=8, pady=5, relief="flat")
            card.pack(fill="x", pady=2)
            lbl = tk.Label(card, text=label_text, font=("Segoe UI", 8), fg="#a0aec0", bg="#14171d")
            lbl.pack(anchor="w")
            val_lbl = tk.Label(card, text=init_val, font=("Consolas", 9, "bold"), fg="#68d391", bg="#14171d")
            val_lbl.pack(anchor="e")
            self.hud_cards[key] = val_lbl

        # Safety Governor Indicator
        self.create_section_label(parent, "SAFETY WATCHDOG")
        self.safety_status_lbl = tk.Label(
            parent,
            text="🟢 CLEAR (Full Velocity Permitted)",
            font=("Segoe UI", 8, "bold"),
            fg="#48bb78",
            bg="#14171d",
            pady=6,
            relief="sunken",
            bd=1,
        )
        self.safety_status_lbl.pack(fill="x", pady=4)

        # Instructions Help Box
        self.create_section_label(parent, "HOW TO DEMO TO PROFESSOR")
        instructions = (
            "1. Click ANY point on the map to set an immediate destination.\n\n"
            "2. Select 'Draw Multi-Waypoint' and click multiple points to create a sequential mission.\n\n"
            "3. Select 'Draw Obstacle' and drag on the map to spawn new walls. The robot's LiDAR detects it instantly!\n\n"
            "4. Switch to DWA or Reactive to demonstrate why PPO outperforms in multi-room doorways."
        )
        help_box = tk.Label(
            parent,
            text=instructions,
            font=("Segoe UI", 8),
            fg="#cbd5e0",
            bg="#14171d",
            justify="left",
            wraplength=270,
            padx=6,
            pady=6,
        )
        help_box.pack(fill="x", pady=4)

    # ------------------ EVENT HANDLERS & MOUSE DRAWING ------------------
    def on_mode_change(self):
        mode = self.interaction_mode.get()
        if mode == "set_goal":
            self.update_status("Click anywhere on the map floor to set a new target goal!", "#2b6cb0")
        elif mode == "multi_waypoint":
            self.update_status("Multi-Waypoint Mode: Click multiple points on the map to create a route!", "#276749")
        elif mode == "add_obstacle":
            self.update_status("Draw Obstacle Mode: Click and drag on the floor to draw a new obstacle box!", "#c05621")
        elif mode == "set_robot":
            self.update_status("Relocate Robot Mode: Click anywhere on free space to place the robot!", "#6b46c1")

    def on_controller_change(self):
        ctrl = self.ctrl_var.get()
        self.active_controller_name = ctrl
        print(f"[UI] Active Controller Switched to: {ctrl.upper()}")
        self.update_status(f"Switched active controller to {ctrl.upper()}", "#2b6cb0")
        if self.controllers.get(ctrl) is not None:
            self.controllers[ctrl].reset(self.env)
        self.redraw_canvas()

    def on_map_preset_change(self, event=None):
        idx = self.map_combobox.current()
        self.active_family = FAMILIES[idx]
        self.generate_new_layout(keep_seed=True)

    def generate_new_layout(self, keep_seed=False):
        if not keep_seed:
            self.current_seed = int(np.random.randint(1000, 99999))
        self.custom_obstacles.clear()
        self.waypoint_queue.clear()
        self.completed_waypoints.clear()
        self.active_target = None
        self.is_running = False
        self.btn_run.config(text="▶ START NAVIGATION", bg="#38a169")
        self.setup_environment()
        self.update_status(f"Generated {self.active_family.upper()} layout (Seed {self.current_seed})", "#2b6cb0")
        self.redraw_canvas()

    def on_canvas_click(self, event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)

        mode = self.interaction_mode.get()
        if mode == "set_goal":
            self.set_single_goal(x, y)
        elif mode == "multi_waypoint":
            self.add_waypoint_to_queue(x, y)
        elif mode == "add_obstacle":
            self.drag_start_point = (x, y)
        elif mode == "set_robot":
            self.teleport_robot(x, y)

    def on_canvas_drag(self, event):
        if self.interaction_mode.get() == "add_obstacle" and self.drag_start_point is not None:
            if event.xdata is not None and event.ydata is not None:
                # Live preview of rectangle being dragged could be drawn if desired
                pass

    def on_canvas_release(self, event):
        if self.interaction_mode.get() == "add_obstacle" and self.drag_start_point is not None:
            if event.xdata is not None and event.ydata is not None:
                x0, y0 = self.drag_start_point
                x1, y1 = float(event.xdata), float(event.ydata)
                min_x, max_x = min(x0, x1), max(x0, x1)
                min_y, max_y = min(y0, y1), max(y0, y1)
                if (max_x - min_x) > 0.2 and (max_y - min_y) > 0.2:
                    new_obs = (min_x, min_y, max_x, max_y)
                    self.custom_obstacles.append(new_obs)
                    self.rects.append(new_obs)
                    self.rects_arr = np.array(self.rects)
                    self.env.rects = self.rects_arr
                    print(f"[OBSTACLE] Added custom obstacle: {new_obs}")
                    # Recompute path if target exists
                    if self.active_target is not None:
                        self.plan_path_to_target(self.active_target[0], self.active_target[1])
                    self.update_status(f"Added custom obstacle box! Total obstacles: {len(self.rects)}", "#c05621")
                    self.redraw_canvas()
            self.drag_start_point = None

    def set_single_goal(self, x: float, y: float):
        # Snap to free space if user clicked slightly close to a wall
        tx, ty = self.snap_to_free_space(x, y)
        if tx is None:
            self.update_status("Clicked inside an impassable obstacle! Please click on free floor space.", "#c53030")
            return

        self.waypoint_queue.clear()
        self.completed_waypoints.clear()
        self.active_target = (tx, ty)
        self.plan_path_to_target(tx, ty)

        self.update_status(f"Target goal set at ({tx:.2f} m, {ty:.2f} m). Navigating now!", "#38a169")
        self.is_running = True
        self.btn_run.config(text="⏸ PAUSE NAVIGATION", bg="#dd6b20")
        self.redraw_canvas()

    def add_waypoint_to_queue(self, x: float, y: float):
        tx, ty = self.snap_to_free_space(x, y)
        if tx is None:
            self.update_status("Waypoint inside obstacle! Please click on free floor space.", "#c53030")
            return

        self.waypoint_queue.append((tx, ty))
        print(f"[WAYPOINT] Added waypoint #{len(self.waypoint_queue)} at ({tx:.2f}, {ty:.2f})")

        # If no active target is running, start navigating toward the first waypoint!
        if self.active_target is None:
            self.pop_next_waypoint()

        self.update_status(f"Waypoint #{len(self.waypoint_queue) + len(self.completed_waypoints)} queued at ({tx:.2f}, {ty:.2f})", "#276749")
        self.redraw_canvas()

    def pop_next_waypoint(self):
        if not self.waypoint_queue:
            self.active_target = None
            self.is_running = False
            self.btn_run.config(text="▶ START NAVIGATION", bg="#38a169")
            self.update_status("🎉 MISSION COMPLETED! All user-drawn waypoints successfully reached!", "#38a169")
            return False

        next_wp = self.waypoint_queue.pop(0)
        self.active_target = next_wp
        self.plan_path_to_target(next_wp[0], next_wp[1])
        self.is_running = True
        self.btn_run.config(text="⏸ PAUSE NAVIGATION", bg="#dd6b20")
        return True

    def teleport_robot(self, x: float, y: float):
        if collides(x, y, self.rects_arr, radius=C.ROBOT_RADIUS):
            self.update_status("Cannot place robot inside an obstacle!", "#c53030")
            return

        self.robot_pose[0] = x
        self.robot_pose[1] = y
        self.current_v = 0.0
        self.current_w = 0.0
        self.trajectory_history = [[x, y]]
        self.path_traversed = 0.0
        self.step_count = 0
        self.env.pose = self.robot_pose.copy()

        if self.active_target is not None:
            self.plan_path_to_target(self.active_target[0], self.active_target[1])

        self.update_status(f"Robot relocated to ({x:.2f} m, {y:.2f} m)", "#6b46c1")
        self.redraw_canvas()

    def snap_to_free_space(self, x: float, y: float, search_radius: float = 0.8) -> tuple[float, float] | None:
        """If (x, y) is clear, return it. Otherwise, search surrounding neighborhood for closest free space."""
        if not collides(x, y, self.rects_arr, radius=C.ROBOT_RADIUS):
            return x, y

        # Search nearby points in concentric rings
        best_pt, best_dist = None, 1e9
        for r in np.linspace(0.1, search_radius, 8):
            for ang in np.linspace(0, 2 * math.pi, 16, endpoint=False):
                cand_x = x + r * math.cos(ang)
                cand_y = y + r * math.sin(ang)
                if not collides(cand_x, cand_y, self.rects_arr, radius=C.ROBOT_RADIUS):
                    d = (cand_x - x)**2 + (cand_y - y)**2
                    if d < best_dist:
                        best_dist = d
                        best_pt = (cand_x, cand_y)
        return best_pt

    def plan_path_to_target(self, gx: float, gy: float):
        # Update underlying env goal
        self.env.goal = np.array([gx, gy], dtype=np.float64)

        # Compute geodesic distance field (Dijkstra 8-connected shortest paths)
        occ_shape = self.bundle.occ.shape
        target_rc = xy_to_rc(gx, gy, occ_shape)
        self.geodesic_field_map = geodesic_field(self.bundle.plan_blocked, target_rc)
        self.field_sampler = smooth_sampler(self.geodesic_field_map)
        self.env.field = self.geodesic_field_map
        self.env._geo = self.field_sampler

        rx, ry, _ = self.robot_pose
        self.shortest_geo_len = float(self.field_sampler(rx, ry))
        self.env.shortest = self.shortest_geo_len
        self.env._prev_geo = self.shortest_geo_len

    def clear_all_waypoints(self):
        self.waypoint_queue.clear()
        self.completed_waypoints.clear()
        self.active_target = None
        self.is_running = False
        self.btn_run.config(text="▶ START NAVIGATION", bg="#38a169")
        self.update_status("Cleared all waypoints. Draw any point on the map to begin!", "#2b6cb0")
        self.redraw_canvas()

    def clear_custom_obstacles(self):
        self.custom_obstacles.clear()
        self.rects = list(self.world.rects)
        self.rects_arr = np.array(self.rects)
        self.env.rects = self.rects_arr
        if self.active_target is not None:
            self.plan_path_to_target(self.active_target[0], self.active_target[1])
        self.update_status("Cleared all custom obstacles.", "#2b6cb0")
        self.redraw_canvas()

    def toggle_navigation(self):
        if self.active_target is None and not self.waypoint_queue:
            self.update_status("Please click on the map to set a target point first!", "#c53030")
            return

        self.is_running = not self.is_running
        if self.is_running:
            self.btn_run.config(text="⏸ PAUSE NAVIGATION", bg="#dd6b20")
            self.update_status("Navigation active...", "#38a169")
        else:
            self.btn_run.config(text="▶ RESUME NAVIGATION", bg="#38a169")
            self.update_status("Navigation paused by user.", "#dd6b20")

    def update_status(self, text: str, color: str = "#2b6cb0"):
        self.mission_status = text
        self.banner_label.config(text=text, bg=color)
        self.banner_frame.config(bg=color)

    # ------------------ SIMULATION ENGINE & CONTROL STEP ------------------
    def simulation_step_loop(self):
        try:
            if not self.root.winfo_exists():
                return
        except Exception:
            return

        now = time.time()
        dt_elapsed = now - self.last_frame_time
        self.last_frame_time = now

        if self.is_running and self.active_target is not None:
            # Execute step(s) based on speed factor
            speed = self.sim_speed_factor.get()
            n_substeps = max(1, int(round(speed)))
            for _ in range(n_substeps):
                if not self.is_running or self.active_target is None:
                    break
                self.execute_control_step()

        self.redraw_canvas()
        self.update_telemetry_hud()

        # Schedule next frame at ~30 FPS
        try:
            if self.root.winfo_exists():
                self.root.after(30, self.simulation_step_loop)
        except Exception:
            pass

    def step_one_frame(self):
        if self.active_target is not None:
            self.execute_control_step()
            self.redraw_canvas()
            self.update_telemetry_hud()
        else:
            self.update_status("Set a target point first before stepping!", "#c53030")

    def execute_control_step(self):
        rx, ry, rth = self.robot_pose
        gx, gy = self.active_target
        dist_to_goal = math.hypot(gx - rx, gy - ry)
        self.target_dist = dist_to_goal

        # 1. Check Goal Tolerance Reached
        if dist_to_goal < C.GOAL_TOLERANCE:
            print(f"[REACHED] Waypoint ({gx:.2f}, {gy:.2f}) reached successfully!")
            self.completed_waypoints.append(self.active_target)
            self.current_v = 0.0
            self.current_w = 0.0

            # Advance to next waypoint if queued
            has_next = self.pop_next_waypoint()
            if not has_next:
                return

        # 2. Extract Lookahead Waypoint from Geodesic Field
        lookahead_target = np.array([gx, gy])
        if self.geodesic_field_map is not None:
            occ_shape = self.bundle.occ.shape
            rc = xy_to_rc(rx, ry, occ_shape)
            path_cells = extract_path(self.geodesic_field_map, rc, max_len=C.LOOKAHEAD)
            if path_cells and self.geodesic_field_map[path_cells[-1]] > 0.0:
                lookahead_target = np.array(rc_to_xy(*path_cells[-1]))

        # 3. Compute LiDAR Sensor Rays
        angles_rel = self.env._angles  # (24, 3)
        rays = self.env.scan_rays(angles_rel, noisy=False)
        self.env._last_angles = angles_rel
        self.env._last_ranges = rays
        sector_ranges = rays.min(axis=1)  # 24 sectors
        self.min_lidar_range = float(sector_ranges.min())

        # Update SLAM Discovered Grid (Fog of war)
        if self.slam_exploration_mode.get():
            self.update_slam_map(rx, ry, rth, sector_ranges)

        # 4. Construct 29-D Observation Vector
        rel_dist, rel_ang = goal_in_robot_frame(self.robot_pose, lookahead_target)
        obs = build_observation(sector_ranges, rel_dist, rel_ang, self.current_v, self.current_w)

        # 5. Query Active Controller
        ctrl_name = self.active_controller_name
        v_target, w_target = 0.0, 0.0

        if ctrl_name == "ppo_shielded" and self.controllers.get("ppo_shielded") is not None:
            action = self.controllers["ppo_shielded"].act(self.env, obs)
            v_target, w_target = action_to_cmd(action)
        elif ctrl_name == "ppo" and self.controllers.get("ppo") is not None:
            action = self.controllers["ppo"].act(self.env, obs)
            v_target, w_target = action_to_cmd(action)
        elif ctrl_name == "dwa":
            action = self.controllers["dwa"].act(self.env, obs)
            v_target, w_target = action_to_cmd(action)
        else:  # reactive
            action = self.controllers["reactive"].act(self.env, obs)
            v_target, w_target = action_to_cmd(action)

        # Smooth Goal Deceleration: Scale down linear velocity when approaching final target
        if dist_to_goal < 0.6:
            v_target = min(v_target, C.V_MAX * (dist_to_goal / 0.6))

        # 6. Intelligent Safety Shield & Autonomous Recovery Layer
        if self.safety_shield_enabled.get():
            # 6.1 Check if currently executing an Autonomous Recovery Maneuver
            if self.recovery_mode:
                self.recovery_steps -= 1
                v_target = -0.10  # Back up smoothly
                w_target = 0.8 * self.recovery_direction  # Pivot away from obstacle
                self.safety_status_lbl.config(
                    text=f"🔄 AUTO-RECOVERY: Reversing & Untangling ({self.recovery_steps})",
                    fg="#f6e05e",
                    bg="#744210",
                )
                if self.recovery_steps <= 0:
                    self.recovery_mode = False
            else:
                # 6.2 Forward Arc Multi-Sector Evaluation (Sectors 8 to 16, i.e. -60° to +60°)
                front_sectors = sector_ranges[8:17]
                min_front_range = float(front_sectors.min())
                min_overall_range = float(sector_ranges.min())

                # If dangerously close -> Initiate Recovery Immediately
                if min_overall_range < C.ROBOT_RADIUS + 0.05:
                    self.recovery_mode = True
                    self.recovery_steps = 7  # 7 steps = 0.7s of reverse clearance
                    closest_idx = int(np.argmin(sector_ranges))
                    self.recovery_direction = -1.0 if closest_idx >= 12 else 1.0
                    v_target = -0.10
                    w_target = 0.8 * self.recovery_direction
                else:
                    # 6.3 Physical Stopping Distance Governor: v^2 <= 2 * a * clearance
                    clearance = max(0.0, min_front_range - C.ROBOT_RADIUS - 0.08)
                    v_allowable = math.sqrt(2.0 * C.A_V_MAX * clearance)
                    if v_target > v_allowable:
                        v_target = v_allowable

                    if min_front_range < C.ROBOT_RADIUS + 0.10:
                        v_target = 0.0

                    # 6.4 Lateral Repulsive Potential Field Steering
                    left_dist = float(sector_ranges[13:17].min())
                    right_dist = float(sector_ranges[8:12].min())
                    repulse_w = 0.0
                    if left_dist < 0.45:
                        repulse_w -= 0.6 * (1.0 - left_dist / 0.45)
                    if right_dist < 0.45:
                        repulse_w += 0.6 * (1.0 - right_dist / 0.45)

                    w_target = float(np.clip(w_target + repulse_w, -C.W_MAX, C.W_MAX))

                    # 6.5 Forward Trajectory Lookahead Forecast (3 steps = 0.3s)
                    for step_i in [1, 2, 3]:
                        pred_x = rx + step_i * v_target * math.cos(rth) * C.DT
                        pred_y = ry + step_i * v_target * math.sin(rth) * C.DT
                        if collides(pred_x, pred_y, self.rects_arr, radius=C.ROBOT_RADIUS + 0.03):
                            v_target = 0.0
                            w_target = C.W_MAX if left_dist >= right_dist else -C.W_MAX
                            break

                    # Update status badge
                    if v_target == 0.0:
                        self.safety_status_lbl.config(
                            text=f"🛑 SHIELD BRAKING: Obstacle {min_front_range:.2f} m",
                            fg="#fc8181",
                            bg="#742a2a",
                        )
                    elif abs(repulse_w) > 0.1:
                        self.safety_status_lbl.config(
                            text="🟡 SHIELD DEFLECTION: Repelling Wall",
                            fg="#f6e05e",
                            bg="#744210",
                        )
                    else:
                        self.safety_status_lbl.config(
                            text=f"🟢 SHIELD ACTIVE: Safe Clearance {min_front_range:.2f} m",
                            fg="#68d391",
                            bg="#1c4532",
                        )
        else:
            # Unshielded baseline check
            front_clearance = float(sector_ranges[11:14].min())
            if front_clearance < 0.28:
                v_target = 0.0
                self.safety_status_lbl.config(
                    text=f"⚠️ BRAKING: Front Obstacle {front_clearance:.2f} m!",
                    fg="#fc8181",
                    bg="#742a2a",
                )
            else:
                self.safety_status_lbl.config(
                    text=f"🟢 CLEAR: Clearance {front_clearance:.2f} m",
                    fg="#68d391",
                    bg="#1c4532",
                )

        # 7. Apply Acceleration Limits
        dv = C.A_V_MAX * C.DT
        dw = C.A_W_MAX * C.DT
        min_v = -0.15 if self.safety_shield_enabled.get() else 0.0
        self.current_v = float(np.clip(v_target, max(min_v, self.current_v - dv), min(C.V_MAX, self.current_v + dv)))
        self.current_w = float(np.clip(w_target, self.current_w - dw, self.current_w + dw))

        # 8. Integrate Unicycle Kinematics
        nx = rx + self.current_v * math.cos(rth) * C.DT
        ny = ry + self.current_v * math.sin(rth) * C.DT
        nth = rth + self.current_w * C.DT
        nth = (nth + math.pi) % (2 * math.pi) - math.pi  # wrap angle

        # 9. Collision Check
        if collides(nx, ny, self.rects_arr, radius=C.ROBOT_RADIUS):
            if self.safety_shield_enabled.get():
                # Shield prevents hard freeze: initiate auto-reverse recovery!
                self.recovery_mode = True
                self.recovery_steps = 8
                closest_idx = int(np.argmin(sector_ranges))
                self.recovery_direction = -1.0 if closest_idx >= 12 else 1.0
                self.current_v = -0.10
                self.current_w = 0.8 * self.recovery_direction
                # Back away slightly to clear contact
                nx = rx - 0.02 * math.cos(rth)
                ny = ry - 0.02 * math.sin(rth)
                if collides(nx, ny, self.rects_arr, radius=C.ROBOT_RADIUS):
                    nx, ny = rx, ry
                self.update_status("🛡️ SAFETY SHIELD: Imminent collision prevented! Auto-reversing...", "#dd6b20")
            else:
                self.current_v = 0.0
                self.current_w = 0.0
                self.is_running = False
                self.btn_run.config(text="▶ RESUME NAVIGATION", bg="#38a169")
                self.update_status("⚠️ COLLISION DETECTED! Robot reached obstacle boundary.", "#c53030")
                return

        # Advance state
        step_dist = math.hypot(nx - rx, ny - ry)
        self.path_traversed += step_dist
        self.robot_pose = np.array([nx, ny, nth])
        self.env.pose = self.robot_pose.copy()
        self.trajectory_history.append([nx, ny])
        self.step_count += 1

        # Calculate SPL
        if self.shortest_geo_len > 0:
            self.spl_metric = self.shortest_geo_len / max(self.path_traversed, self.shortest_geo_len)

    def update_slam_map(self, rx: float, ry: float, rth: float, sector_ranges: np.ndarray):
        """Simulate SLAM occupancy discovery by ray-sweeping visited cells."""
        res = C.GRID_RES
        ny, nx = self.bundle.occ.shape
        ang_rel = self.env._angles.mean(axis=1)
        ang_world = ang_rel + rth

        for i in range(C.N_SECTORS):
            r = min(sector_ranges[i], 3.0)
            for d in np.arange(0.1, r, 0.15):
                cx = rx + d * math.cos(ang_world[i])
                cy = ry + d * math.sin(ang_world[i])
                rc = xy_to_rc(cx, cy, (ny, nx))
                if 0 <= rc[0] < ny and 0 <= rc[1] < nx:
                    self.discovered_grid[rc[0], rc[1]] = True

    # ------------------ CANVAS DRAWING & RENDERING ------------------
    def redraw_canvas(self):
        self.ax.clear()
        self.ax.set_facecolor("#181b20")
        w = self.world

        # Draw Walls and Obstacles
        for x0, y0, x1, y1 in self.rects:
            rect = Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                facecolor="#374151", edgecolor="#4b5563", linewidth=1.2, zorder=2
            )
            self.ax.add_patch(rect)

        # SLAM Fog of War Overlay
        if self.slam_exploration_mode.get():
            ny, nx = self.bundle.occ.shape
            unexplored = ~self.discovered_grid
            self.ax.imshow(
                unexplored, cmap="gray", origin="lower",
                extent=[0, w.width, 0, w.height], alpha=0.6, zorder=3
            )

        # Draw Global Geodesic Planned Path
        if self.show_global_path.get() and self.active_target is not None and self.geodesic_field_map is not None:
            rx, ry, _ = self.robot_pose
            path_cells = extract_path(self.geodesic_field_map, xy_to_rc(rx, ry, self.bundle.occ.shape), max_len=15.0)
            if len(path_cells) > 1:
                pts = [rc_to_xy(*c) for c in path_cells]
                pts_arr = np.array(pts)
                self.ax.plot(pts_arr[:, 0], pts_arr[:, 1], color="#38bdf8", linestyle="--", linewidth=2.0, alpha=0.7, zorder=4)

        # Draw Trajectory Trail
        if self.show_trajectory_trail.get() and len(self.trajectory_history) > 1:
            traj_arr = np.array(self.trajectory_history)
            self.ax.plot(traj_arr[:, 0], traj_arr[:, 1], color="#818cf8", linewidth=2.2, alpha=0.85, zorder=5)

        # Draw Completed Waypoints (Green checkmarks)
        for idx, (cx, cy) in enumerate(self.completed_waypoints):
            self.ax.plot(cx, cy, "o", color="#22c55e", markersize=10, zorder=6)
            self.ax.text(cx, cy + 0.25, f"✓ #{idx+1}", color="#4ade80", fontsize=8, fontweight="bold", ha="center", zorder=7)

        # Draw Active Target Goal (Glowing Bullseye)
        if self.active_target is not None:
            gx, gy = self.active_target
            self.ax.add_patch(Circle((gx, gy), C.GOAL_TOLERANCE, color="#ef4444", alpha=0.3, zorder=6))
            self.ax.add_patch(Circle((gx, gy), 0.12, color="#ef4444", alpha=0.9, zorder=7))
            self.ax.text(gx, gy + 0.3, "🎯 TARGET", color="#f87171", fontsize=9, fontweight="bold", ha="center", zorder=8)

        # Draw Queued Waypoints (Numbered Flags)
        for idx, (qx, qy) in enumerate(self.waypoint_queue):
            pin_num = len(self.completed_waypoints) + idx + 2
            self.ax.plot(qx, qy, "s", color="#eab308", markersize=8, zorder=6)
            self.ax.text(qx, qy + 0.25, f"🚩 #{pin_num}", color="#fde047", fontsize=8, fontweight="bold", ha="center", zorder=7)

        # Draw 24-Sector LiDAR Ray Beams
        rx, ry, rth = self.robot_pose
        if self.show_lidar_rays.get() and hasattr(self.env, "_last_angles") and hasattr(self.env, "_last_ranges"):
            ang_rel = self.env._last_angles.mean(axis=1)
            rng_sec = self.env._last_ranges.min(axis=1)
            ang_world = ang_rel + rth

            for i in range(C.N_SECTORS):
                r = rng_sec[i]
                bx = rx + r * math.cos(ang_world[i])
                by = ry + r * math.sin(ang_world[i])

                # Color-code: Red < 0.5m, Amber < 1.2m, Green >= 1.2m
                beam_col = "#f87171" if r < 0.5 else ("#facc15" if r < 1.2 else "#4ade80")
                self.ax.plot([rx, bx], [ry, by], color=beam_col, linewidth=0.8, alpha=0.7, zorder=6)
                if r < C.RANGE_MAX * 0.98:
                    self.ax.plot(bx, by, ".", color=beam_col, markersize=3, zorder=6)

        # Draw Robot Chassis & Orientation
        self.ax.add_patch(Circle((rx, ry), C.ROBOT_RADIUS, facecolor="#3b82f6", edgecolor="#60a5fa", linewidth=2.0, zorder=8))
        hx = rx + (C.ROBOT_RADIUS + 0.18) * math.cos(rth)
        hy = ry + (C.ROBOT_RADIUS + 0.18) * math.sin(rth)
        self.ax.plot([rx, hx], [ry, hy], color="#fde047", linewidth=2.8, zorder=9)

        # Plot Settings
        self.ax.set_xlim(-0.3, w.width + 0.3)
        self.ax.set_ylim(-0.3, w.height + 0.3)
        self.ax.set_aspect("equal")
        self.ax.grid(True, color="#27272a", linestyle="--", linewidth=0.5, alpha=0.7)
        self.ax.set_title(
            f"Layout: {self.active_family.upper()} | Controller: {self.active_controller_name.upper()}",
            color="#93c5fd", fontsize=10, fontweight="bold", pad=8
        )
        self.ax.tick_params(colors="#71717a", labelsize=8)

        self.canvas.draw_idle()

    def update_telemetry_hud(self):
        rx, ry, rth = self.robot_pose
        if self.active_target is not None:
            gx, gy = self.active_target
            self.hud_cards["target_xy"].config(text=f"({gx:.2f} m, {gy:.2f} m)")
            self.hud_cards["dist_metric"].config(text=f"{self.target_dist:.2f} m")
        else:
            self.hud_cards["target_xy"].config(text="(No Target Set)")
            self.hud_cards["dist_metric"].config(text="0.00 m")

        self.hud_cards["wp_queue"].config(text=f"{len(self.waypoint_queue)} queued | {len(self.completed_waypoints)} reached")
        self.hud_cards["robot_pose"].config(text=f"X:{rx:.2f}m Y:{ry:.2f}m θ:{math.degrees(rth):.1f}°")
        self.hud_cards["v_metric"].config(text=f"{self.current_v:+.2f} m/s")
        self.hud_cards["w_metric"].config(text=f"{self.current_w:+.2f} rad/s")
        self.hud_cards["lidar_metric"].config(
            text=f"{self.min_lidar_range:.2f} m",
            fg="#f87171" if self.min_lidar_range < 0.35 else "#4ade80"
        )
        self.hud_cards["path_metric"].config(text=f"{self.path_traversed:.2f} m")
        self.hud_cards["shortest_metric"].config(text=f"{self.shortest_geo_len:.2f} m")
        self.hud_cards["spl_metric"].config(text=f"{self.spl_metric:.3f}")
        self.hud_cards["time_metric"].config(text=f"{self.step_count} steps ({self.step_count * C.DT:.1f} s)")


def launch_interactive_ui():
    root = tk.Tk()
    # Configure ttk styling for clean dark aesthetic
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TCombobox", fieldbackground="#1e232b", background="#2d3748", foreground="#ffffff")
    app = AutonomousNavApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_interactive_ui()
