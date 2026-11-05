#!/usr/bin/env python3
"""Autonomous Navigation: Live Interactive Demonstration & Mission Control Launcher.

Run this script directly from PowerShell, Windows Command Prompt, or Linux Terminal:
    python run_demo.py

Options:
    python run_demo.py                  # Launches the Full Interactive Mission Control UI (Draw points, waypoints, obstacles)
    python run_demo.py --ui             # Explicitly launch Mission Control UI
    python run_demo.py --save-gif out.gif # Export animated demonstration GIF
"""
import sys

def main():
    if "--save-gif" in sys.argv:
        from rl_training.gui_demo import main as run_gif_demo
        run_gif_demo()
    else:
        from rl_training.interactive_ui import launch_interactive_ui
        launch_interactive_ui()

if __name__ == "__main__":
    main()
