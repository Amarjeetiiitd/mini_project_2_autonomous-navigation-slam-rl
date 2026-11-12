#!/usr/bin/env python3
"""PPO local-navigation controller as a ROS 2 (Jazzy) node.

Pipeline (10 Hz):  /scan + /odom + TF(map->base_link) + goal [+ global plan]
                   -> 29-D observation (rl_training.obs) -> PPO -> safety layer -> /cmd_vel

Run (from the project root, venv created with --system-site-packages, ROS sourced):

    python ros2_integration/policy_node.py --ros-args -p model_path:=models/ppo_navigation \
        -p use_sim_time:=true

Send a goal:
    ros2 topic pub --once /rl_goal geometry_msgs/PoseStamped \
      "{header: {frame_id: map}, pose: {position: {x: 1.5, y: 0.5}}}"

Status is published on /rl_status (std_msgs/String): idle | active | reached | timeout |
safety_stop. Do NOT run this while Nav2's controller is actively navigating: two nodes
publishing /cmd_vel will fight. Nav2 may still be used as *global planner only*
(plan_source:=nav2), which is the intended integration.

STATUS: written against the ROS 2 Jazzy API but NOT executed inside the sandbox where this
project was generated (no ROS there). Treat the first run as a bring-up, start with
`dry_run:=true` (computes actions, publishes nothing), and verify frames/topics.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> rl_training

import rclpy  # noqa: E402
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped  # noqa: E402
from nav_msgs.msg import Odometry, Path as PathMsg  # noqa: E402
from nav2_msgs.action import ComputePathToPose  # noqa: E402
from rclpy.action import ActionClient  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data  # noqa: E402
from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import String  # noqa: E402
from tf2_ros import Buffer, TransformException, TransformListener  # noqa: E402

from rl_training import config as C  # noqa: E402
from rl_training.obs import (action_to_cmd, build_observation, goal_in_robot_frame,  # noqa: E402
                             lookahead_point, resample_scan)
from rl_training.safety import NavigationSafetyFilter, SafetyConfig  # noqa: E402


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("rl_policy_node")
        P = self.declare_parameter
        self.model_path = P("model_path", "models/ppo_navigation").value
        self.goal_mode = P("goal_mode", "").value            # "" -> read from <model>.json
        self.plan_source = P("plan_source", "none").value     # none | nav2 | topic
        self.global_frame = P("global_frame", "map").value
        self.base_frame = P("base_frame", "base_link").value
        self.scan_topic = P("scan_topic", "/scan").value
        self.odom_topic = P("odom_topic", "/odom").value
        self.cmd_topic = P("cmd_vel_topic", "/cmd_vel").value
        self.stamped = P("stamped_cmd_vel", False).value
        self.goal_topic = P("goal_topic", "/rl_goal").value
        self.plan_topic = P("plan_topic", "/plan").value
        self.rate_hz = P("rate_hz", 10.0).value
        self.scan_yaw_offset = P("scan_yaw_offset", 0.0).value
        self.max_linear = min(P("max_linear", C.V_MAX).value, C.V_MAX)
        self.max_angular = min(P("max_angular", C.W_MAX).value, C.W_MAX)
        self.stop_distance = P("stop_distance", 0.28).value   # m, front sectors
        self.scan_timeout = P("scan_timeout", 0.5).value      # s
        self.odom_timeout = P("odom_timeout", 0.5).value      # s
        self.goal_timeout = P("goal_timeout", 120.0).value    # s
        self.goal_tol = P("goal_tolerance", C.GOAL_TOLERANCE).value
        self.replan_period = P("replan_period", 2.0).value
        self.dry_run = P("dry_run", False).value
        self.enable_safety_shield = P("enable_safety_shield", True).value
        self.safety_filter = NavigationSafetyFilter()

        from stable_baselines3 import PPO
        mp = Path(self.model_path)
        meta = json.loads(mp.with_suffix(".json").read_text()) if mp.with_suffix(".json").exists() else {}
        if not self.goal_mode:
            self.goal_mode = meta.get("goal_mode", "waypoint")
        self.model = PPO.load(str(mp), device="cpu")
        assert self.model.observation_space.shape == (C.OBS_DIM,), "model/observation mismatch"
        self.get_logger().info(f"loaded {mp} (goal_mode={self.goal_mode}, plan_source={self.plan_source})")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.scan_sectors, self.scan_stamp = None, 0.0
        self.odom_stamp = 0.0
        self.v = self.w = 0.0
        self.goal = None            # np.array([x, y]) in global frame
        self.goal_t0 = 0.0
        self.path_xy = None
        self.state = "idle"
        self.plan_pending, self.last_plan_t = False, 0.0

        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, self.odom_topic, self.on_odom, 10)
        self.create_subscription(PoseStamped, self.goal_topic, self.on_goal, 10)
        if self.plan_source == "topic":
            self.create_subscription(PathMsg, self.plan_topic, self.on_plan_msg, 10)
        self.cmd_pub = self.create_publisher(TwistStamped if self.stamped else Twist, self.cmd_topic, 10)
        self.status_pub = self.create_publisher(String, "/rl_status", 10)
        self.plan_client = ActionClient(self, ComputePathToPose, "compute_path_to_pose") \
            if self.plan_source == "nav2" else None
        self.create_timer(1.0 / self.rate_hz, self.on_timer)

    # ------------------------------------------------------------ callbacks
    def now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_scan(self, msg: LaserScan) -> None:
        self.scan_sectors, self.scan_mask = resample_scan(
            msg.ranges, msg.angle_min, msg.angle_increment,
            max(msg.range_min, C.RANGE_MIN), min(msg.range_max, C.RANGE_MAX),
            self.scan_yaw_offset, return_mask=True
        )
        self.scan_stamp = self.now_s()

    def on_odom(self, msg: Odometry) -> None:
        self.v, self.w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        self.odom_stamp = self.now_s()

    def on_goal(self, msg: PoseStamped) -> None:
        if msg.header.frame_id and msg.header.frame_id != self.global_frame:
            self.get_logger().warn(f"goal frame '{msg.header.frame_id}' != '{self.global_frame}'; ignoring")
            return
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.goal_t0, self.path_xy, self.state = self.now_s(), None, "active"
        self.last_plan_t = -1e9
        self.get_logger().info(f"new goal {self.goal}")

    def on_plan_msg(self, msg: PathMsg) -> None:
        self.path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in msg.poses])

    # ---------------------------------------------------------- global plan
    def request_plan(self, pose) -> None:
        if self.plan_client is None or self.plan_pending or not self.plan_client.server_is_ready():
            return
        goal = ComputePathToPose.Goal()
        goal.goal.header.frame_id = self.global_frame
        goal.goal.pose.position.x, goal.goal.pose.position.y = float(self.goal[0]), float(self.goal[1])
        goal.goal.pose.orientation.w = 1.0
        goal.use_start = False  # planner uses the current robot pose
        self.plan_pending, self.last_plan_t = True, self.now_s()
        self.plan_client.send_goal_async(goal).add_done_callback(self._plan_goal_resp)

    def _plan_goal_resp(self, fut) -> None:
        h = fut.result()
        if not h.accepted:
            self.plan_pending = False
            return
        h.get_result_async().add_done_callback(self._plan_result)

    def _plan_result(self, fut) -> None:
        self.plan_pending = False
        path = fut.result().result.path
        if path.poses:
            self.path_xy = np.array([[p.pose.position.x, p.pose.position.y] for p in path.poses])

    # ------------------------------------------------------------- control
    def robot_pose(self):
        t = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, rclpy.time.Time())
        tr = t.transform
        return np.array([tr.translation.x, tr.translation.y, yaw_from_quat(tr.rotation)])

    def publish_cmd(self, v: float, w: float) -> None:
        if self.dry_run:
            return
        if self.stamped:
            m = TwistStamped()
            m.header.stamp = self.get_clock().now().to_msg()
            m.header.frame_id = self.base_frame
            m.twist.linear.x, m.twist.angular.z = float(v), float(w)
        else:
            m = Twist()
            m.linear.x, m.angular.z = float(v), float(w)
        self.cmd_pub.publish(m)

    def set_state(self, s: str) -> None:
        if s != self.state:
            self.get_logger().info(f"state: {self.state} -> {s}")
        self.state = s

    def on_timer(self) -> None:
        self.status_pub.publish(String(data=self.state))
        if self.state != "active" or self.goal is None:
            return
        # ---- safety: stale sensing / localisation -> stop
        if self.scan_sectors is None or self.now_s() - self.scan_stamp > self.scan_timeout:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn("no fresh /scan; holding", throttle_duration_sec=2.0)
            return
        if self.odom_stamp == 0.0 or self.now_s() - self.odom_stamp > self.odom_timeout:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn("no fresh /odom; holding", throttle_duration_sec=2.0)
            return
        try:
            pose = self.robot_pose()
        except TransformException as e:
            self.publish_cmd(0.0, 0.0)
            self.get_logger().warn(f"TF {self.global_frame}->{self.base_frame} unavailable: {e}",
                                   throttle_duration_sec=2.0)
            return
        if self.now_s() - self.goal_t0 > self.goal_timeout:
            self.publish_cmd(0.0, 0.0)
            self.set_state("timeout")
            return
        dist_goal = math.hypot(self.goal[0] - pose[0], self.goal[1] - pose[1])
        if dist_goal < self.goal_tol:
            self.publish_cmd(0.0, 0.0)
            self.set_state("reached")
            return

        # ---- target for the observation: waypoint on the global plan, or the goal itself
        target = self.goal
        if self.goal_mode == "waypoint":
            if self.plan_source == "nav2" and self.now_s() - self.last_plan_t > self.replan_period:
                self.request_plan(pose)
            if self.path_xy is not None and len(self.path_xy) > 1:
                wp = lookahead_point(self.path_xy, pose[:2], C.LOOKAHEAD)
                target = wp if wp is not None else self.goal
            elif dist_goal > C.LOOKAHEAD:
                # no plan yet: straight-line waypoint. Only safe in open space; wait if possible.
                if self.plan_source != "none":
                    self.publish_cmd(0.0, 0.0)
                    return
                u = (self.goal - pose[:2]) / dist_goal
                target = pose[:2] + u * C.LOOKAHEAD
        d, ang = goal_in_robot_frame(pose, target)
        obs = build_observation(self.scan_sectors, d, ang, self.v, self.w)
        action, _ = self.model.predict(obs, deterministic=True)
        v, w = action_to_cmd(action)
        v, w = min(v, self.max_linear), float(np.clip(w, -self.max_angular, self.max_angular))

        # ---- safety: unified predictive stopping-distance governor, repulsive deflection & recovery
        if self.enable_safety_shield:
            v, w, diag = self.safety_filter.filter_command(
                v, w, self.scan_sectors, estimated_pose=pose[:2], dt=1.0 / self.rate_hz,
                valid_mask=getattr(self, "scan_mask", None)
            )
            if diag["intervention"] != "none":
                self.get_logger().info(
                    f"Safety shield intervention: {diag['intervention']} "
                    f"(min_front={diag['min_front']:.2f}m, min_rear={diag['min_rear']:.2f}m, "
                    f"v: {diag['v_raw']:.2f}->{diag['v_safe']:.2f}, w: {diag['w_raw']:.2f}->{diag['w_safe']:.2f})",
                    throttle_duration_sec=1.0,
                )
        else:
            front = float(self.scan_sectors[11:14].min())
            if front < self.stop_distance:
                v = 0.0
                self.get_logger().warn(f"front obstacle {front:.2f} m: forward motion inhibited",
                                       throttle_duration_sec=1.0)
        self.publish_cmd(v, w)


def main() -> None:
    rclpy.init()
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_cmd(0.0, 0.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
