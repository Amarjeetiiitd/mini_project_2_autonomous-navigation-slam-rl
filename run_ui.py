#!/usr/bin/env python3
"""Autonomous Mobile Robot Navigation (SLAM & Reinforcement Learning)
Interactive Mission Control & Graphical Demonstration Dashboard.

Run this script to launch the graphical user interface:
    python run_ui.py

Interactive Features for Professor Demonstration:
- Click ANY point on the map to set an immediate target destination.
- Switch to "Draw Multi-Waypoint Route" to click multiple waypoints (Point 1 -> Point 2 -> Goal).
- Switch to "Draw Custom Obstacle Box" to add custom walls and test real-time obstacle avoidance.
- Live controller switching: PPO (Deep RL), DWA (Dynamic Window Approach), and Reactive.
- Real-time 24-sector LiDAR rays with dynamic color coding (Green/Amber/Red).
- Live Telemetry HUD: velocity, yaw rate, distance, SPL, steps, safety governor status.
"""
from rl_training.interactive_ui import launch_interactive_ui

if __name__ == "__main__":
    launch_interactive_ui()
