#!/usr/bin/env python3
"""Experiment E6: localisation error of the SLAM pose against simulator ground truth.

Logs the difference between the SLAM pose (TF map->base_link) and a ground-truth
nav_msgs/Odometry topic at `rate_hz`, after aligning the two frames with a single SE(2)
transform estimated from the first `align_samples` samples (the SLAM map origin is
arbitrary, so a constant offset must be removed before comparing).

    python ros2_integration/localization_error_logger.py --ros-args \
        -p gt_topic:=/ground_truth/odom -p out:=results/loc_error.csv -p use_sim_time:=true

Ctrl-C prints mean / RMSE / max position error and final drift.

You need a ground-truth source. Gazebo does not publish one on a ROS topic by default;
bridge it (e.g. an OdometryPublisher system in the world SDF bridged with ros_gz_bridge)
and pass its topic via `gt_topic`. Caveats: the alignment assumes the error is ~0 during
the first samples (robot standing still at the start), so this measures *drift*, not an
absolute offset.
STATUS: untested in the generation sandbox (no ROS available).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import Buffer, TransformException, TransformListener


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def se2_mul(a, b):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([a[0] + c * b[0] - s * b[1], a[1] + s * b[0] + c * b[1], wrap(a[2] + b[2])])


def se2_inv(a):
    c, s = math.cos(a[2]), math.sin(a[2])
    return np.array([-(c * a[0] + s * a[1]), s * a[0] - c * a[1], wrap(-a[2])])


class Logger(Node):
    def __init__(self) -> None:
        super().__init__("localization_error_logger")
        P = self.declare_parameter
        self.gt_topic = P("gt_topic", "/ground_truth/odom").value
        self.out = P("out", "results/loc_error.csv").value
        self.align_n = int(P("align_samples", 20).value)
        rate = P("rate_hz", 10.0).value
        self.gt = None
        self.gt_stamp = None
        self.T = None
        self.align_buf = []
        self.rows = []
        self.buf = Buffer()
        self.tfl = TransformListener(self.buf, self)
        self.create_subscription(Odometry, self.gt_topic, self.on_gt, 10)
        self.create_timer(1.0 / rate, self.tick)

    def on_gt(self, m: Odometry) -> None:
        p = m.pose.pose
        self.gt = np.array([p.position.x, p.position.y, yaw_from_quat(p.orientation)])
        self.gt_stamp = m.header.stamp

    def estimate_se2_alignment(self, slam_pts: np.ndarray, gt_pts: np.ndarray) -> np.ndarray:
        """Closed-form least-squares SE(2) transformation estimation over N samples."""
        s_mean = np.mean(slam_pts[:, :2], axis=0)
        g_mean = np.mean(gt_pts[:, :2], axis=0)

        # Average orientation difference across all alignment samples
        yaw_diffs = [wrap(s[2] - g[2]) for s, g in zip(slam_pts, gt_pts)]
        yaw_avg = math.atan2(float(np.sum(np.sin(yaw_diffs))), float(np.sum(np.cos(yaw_diffs))))

        # Optimal translation aligning centroids under optimal yaw rotation
        c, s = math.cos(yaw_avg), math.sin(yaw_avg)
        R = np.array([[c, -s], [s, c]])
        t = s_mean - R @ g_mean
        return np.array([t[0], t[1], yaw_avg])

    def tick(self) -> None:
        if self.gt is None:
            return
        try:
            # Strictly query TF synchronized to ground truth message timestamp
            stamp = self.gt_stamp if self.gt_stamp is not None else rclpy.time.Time()
            t = self.buf.lookup_transform(
                "map", "base_link", stamp, timeout=rclpy.duration.Duration(seconds=0.05)
            ).transform
        except TransformException:
            # Drop sample if synchronized transform is unavailable to prevent pairing poses from different times
            return

        slam = np.array([t.translation.x, t.translation.y, yaw_from_quat(t.rotation)])
        if self.T is None:
            self.align_buf.append((slam, self.gt.copy()))
            if len(self.align_buf) >= self.align_n:
                slam_arr = np.array([pair[0] for pair in self.align_buf])
                gt_arr = np.array([pair[1] for pair in self.align_buf])
                self.T = self.estimate_se2_alignment(slam_arr, gt_arr)
                self.get_logger().info(f"aligned frames over {len(self.align_buf)} samples, T_map_world = {self.T}")
            return
        gt_in_map = se2_mul(self.T, self.gt)
        e = slam - gt_in_map
        e[2] = wrap(e[2])
        self.rows.append([self.get_clock().now().nanoseconds * 1e-9, *e, math.hypot(e[0], e[1])])

    def save(self) -> None:
        if not self.rows:
            self.get_logger().warn("no samples collected")
            return
        Path(self.out).parent.mkdir(parents=True, exist_ok=True)
        with open(self.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "ex", "ey", "eyaw", "pos_err"])
            w.writerows(self.rows)
        pe = np.array([r[4] for r in self.rows])
        print(f"samples={len(pe)} mean={pe.mean():.3f} m RMSE={math.sqrt((pe**2).mean()):.3f} m "
              f"max={pe.max():.3f} m final={pe[-1]:.3f} m  -> {self.out}")


def main() -> None:
    rclpy.init()
    n = Logger()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.save()
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
