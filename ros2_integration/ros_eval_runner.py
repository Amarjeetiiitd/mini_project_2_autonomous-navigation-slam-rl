#!/usr/bin/env python3
"""Run matched start/goal trials in Gazebo for Nav2 (baseline) or the PPO policy node.

For every scenario the runner (1) drives the robot to the scenario's *start* pose with
Nav2 (staging, not recorded), then (2) runs the controller under test to the *goal*
and records success, time, path length, a collision proxy and minimum clearance.
This keeps trials matched without needing to teleport the robot (teleporting would
break the SLAM map frame).

    # Nav2 baseline (bringup running with slam:=True, headless or not)
    python ros2_integration/ros_eval_runner.py --controller nav2 \
        --scenarios configs/ros_scenarios_tb3_world.json --out results/ros_nav2.csv
    # PPO (policy_node.py running, plan_source:=nav2)
    python ros2_integration/ros_eval_runner.py --controller rl \
        --scenarios configs/ros_scenarios_tb3_world.json --out results/ros_rl.csv

Scenario JSON: [{"name": "s1", "start": [x, y, yaw], "goal": [x, y, yaw]}, ...] in the
`map` frame. NOTE: with SLAM the map origin is where mapping started, so coordinates are
only reproducible if you start each run from the same spawn pose; for fair comparisons
pre-build a map with SLAM Toolbox, save it, and run both controllers on that fixed map
(see docs/EXPERIMENTS.md).

Collision proxy: a collision event is counted when min(/scan) < --collision-range.
Calibrate that threshold for your robot model (the LiDAR sits inside the footprint).

STATUS: untested in the generation sandbox (no ROS available). Bring up step by step.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def make_pose(x, y, yaw, stamp) -> PoseStamped:
    p = PoseStamped()
    p.header.frame_id, p.header.stamp = "map", stamp
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


class Runner(Node):
    def __init__(self, a) -> None:
        super().__init__("ros_eval_runner")
        self.a = a
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.plan = ActionClient(self, ComputePathToPose, "compute_path_to_pose")
        self.min_scan, self.rl_status = float("inf"), "idle"
        self.create_subscription(LaserScan, a.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.create_subscription(String, "/rl_status", lambda m: setattr(self, "rl_status", m.data), 10)
        self.goal_pub = self.create_publisher(PoseStamped, "/rl_goal", 10)

    def on_scan(self, msg: LaserScan) -> None:
        r = np.asarray(msg.ranges)
        r = r[np.isfinite(r) & (r >= msg.range_min) & (r <= msg.range_max)]
        self.min_scan = float(r.min()) if r.size else float("inf")

    def spin_for(self, secs: float) -> None:
        end = time.time() + secs
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def pose(self):
        t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time()).transform
        return np.array([t.translation.x, t.translation.y, yaw_from_quat(t.rotation)])

    def wait_future(self, fut, timeout: float):
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def nav2_goto(self, x, y, yaw, timeout, on_tick=None) -> bool:
        goal = NavigateToPose.Goal()
        goal.pose = make_pose(x, y, yaw, self.get_clock().now().to_msg())
        h = self.wait_future(self.nav.send_goal_async(goal), 10.0)
        if h is None or not h.accepted:
            return False
        res_fut = h.get_result_async()
        end = time.time() + timeout
        while not res_fut.done() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if on_tick:
                on_tick()
        if not res_fut.done():
            h.cancel_goal_async()
            self.spin_for(1.0)
            return False
        return res_fut.result().status == 4  # GoalStatus.STATUS_SUCCEEDED

    def plan_length(self, x, y) -> float:
        g = ComputePathToPose.Goal()
        g.goal = make_pose(x, y, 0.0, self.get_clock().now().to_msg())
        g.use_start = False
        h = self.wait_future(self.plan.send_goal_async(g), 5.0)
        if h is None or not h.accepted:
            return float("nan")
        r = self.wait_future(h.get_result_async(), 10.0)
        if r is None or not r.result.path.poses:
            return float("nan")
        p = np.array([[q.pose.position.x, q.pose.position.y] for q in r.result.path.poses])
        return float(np.hypot(*np.diff(p, axis=0).T).sum())

    def run_trial(self, scn: dict) -> dict:
        a = self.a
        sx, sy, syaw = scn["start"]
        gx, gy = scn["goal"][:2]
        if not self.nav2_goto(sx, sy, syaw, a.stage_timeout):
            return {"name": scn["name"], "success": False, "note": "staging_failed"}
        self.spin_for(1.0)
        shortest = self.plan_length(gx, gy)
        samples, collisions, min_clear = [], 0, float("inf")
        state = {"in_col": False, "last_t": 0.0}

        def tick():
            nonlocal collisions, min_clear
            if self.min_scan < a.collision_range and not state["in_col"]:
                collisions += 1
            state["in_col"] = self.min_scan < a.collision_range
            min_clear = min(min_clear, self.min_scan)
            if time.time() - state["last_t"] > 0.1:
                state["last_t"] = time.time()
                try:
                    samples.append(self.pose()[:2])
                except TransformException:
                    pass

        t0 = time.time()
        if a.controller == "nav2":
            ok = self.nav2_goto(gx, gy, 0.0, a.timeout, tick)
        else:
            self.rl_status = "active_wait"
            self.goal_pub.publish(make_pose(gx, gy, 0.0, self.get_clock().now().to_msg()))
            ok = False
            while time.time() - t0 < a.timeout:
                rclpy.spin_once(self, timeout_sec=0.05)
                tick()
                if self.rl_status in ("reached", "timeout"):
                    ok = self.rl_status == "reached"
                    break
        dt = time.time() - t0
        path_len = float(np.hypot(*np.diff(np.array(samples), axis=0).T).sum()) if len(samples) > 1 else 0.0
        spl = (shortest / max(path_len, shortest)) if (ok and shortest == shortest and path_len > 0) else 0.0
        if a.sim_time:  # wall time != sim time; warn the reader
            pass
        return {"name": scn["name"], "success": ok, "collision_events": collisions,
                "collision": collisions > 0, "min_clearance": min_clear, "time_s": dt,
                "path_length": path_len, "shortest_len": shortest, "spl": spl, "note": ""}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["nav2", "rl"], required=True)
    ap.add_argument("--scenarios", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--timeout", type=float, default=120.0, help="wall-clock s per trial")
    ap.add_argument("--stage-timeout", type=float, default=180.0)
    ap.add_argument("--collision-range", type=float, default=0.17)
    ap.add_argument("--scan-topic", default="/scan")
    ap.add_argument("--sim-time", action="store_true", help="note: times are wall-clock")
    ap.add_argument("--repeats", type=int, default=1)
    a = ap.parse_args()
    scns = json.loads(Path(a.scenarios).read_text())

    rclpy.init()
    node = Runner(a)
    node.get_logger().info("waiting for navigate_to_pose ...")
    node.nav.wait_for_server()
    rows = []
    for rep in range(a.repeats):
        for scn in scns:
            r = node.run_trial(scn)
            r.update(controller=a.controller, repeat=rep)
            rows.append(r)
            node.get_logger().info(f"{a.controller} {scn['name']}: {r}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    node.get_logger().info(f"saved {a.out}")
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
