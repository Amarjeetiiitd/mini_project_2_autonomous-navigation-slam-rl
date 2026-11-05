"""Launch the PPO policy node (and optionally the localisation-error logger).

    ros2 launch ros2_integration/launch/rl_nav.launch.py model_path:=models/ppo_navigation \
        plan_source:=nav2 use_sim_time:=true

Run from the project root inside the activated venv so `python` finds torch/SB3.
"""
import os
import sys

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration as LC

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def generate_launch_description():
    args = [
        DeclareLaunchArgument("model_path", default_value=os.path.join(ROOT, "models", "ppo_navigation")),
        DeclareLaunchArgument("plan_source", default_value="nav2"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        DeclareLaunchArgument("log_localization", default_value="false"),
        DeclareLaunchArgument("gt_topic", default_value="/ground_truth/odom"),
    ]
    policy = ExecuteProcess(
        cmd=[sys.executable, os.path.join(ROOT, "ros2_integration", "policy_node.py"), "--ros-args",
             "-p", ["model_path:=", LC("model_path")], "-p", ["plan_source:=", LC("plan_source")],
             "-p", ["use_sim_time:=", LC("use_sim_time")], "-p", ["dry_run:=", LC("dry_run")]],
        output="screen")
    loc = ExecuteProcess(
        cmd=[sys.executable, os.path.join(ROOT, "ros2_integration", "localization_error_logger.py"),
             "--ros-args", "-p", ["gt_topic:=", LC("gt_topic")], "-p", ["use_sim_time:=", LC("use_sim_time")],
             "-p", ["out:=", os.path.join(ROOT, "results", "loc_error.csv")]],
        output="screen", condition=IfCondition(LC("log_localization")))
    return LaunchDescription(args + [policy, loc])
