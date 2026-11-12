"""Benchmark policy, safety filter, full observation pipeline, and DWA latency on CPU.

Measures:
1. PPO Policy forward pass: MLP Actor forward pass on pre-computed observation.
2. Safety Filter pass: 4-layer predictive stopping governor, flank repulsion, recovery checks.
3. Policy + Safety Filter: PPO model.predict + action_to_cmd + safety_filter.filter_command.
4. Full Observation-to-Command Pipeline:
   Raw LaserScan (360 beams) -> sector downsampling (24 sectors) + validity mask
   -> 29-D observation construction -> PPO predict -> action_to_cmd
   -> safety_filter.filter_command -> safe velocity command.
5. Matched Classical DWA: dynamic window trajectory generation, circular arc rollout,
   LiDAR point cloud clearance evaluation, and objective scoring.

Reports: mean, std, median (p50), p90, p95, p99, min, max, with CPU hardware and dependency metadata.
Saves: results/latency_benchmark.json
"""
from __future__ import annotations

import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import rl_training.config as C
from rl_training.baselines import DWAController, action_to_cmd
from rl_training.obs import resample_scan
from rl_training.safety import NavigationSafetyFilter, SafetyConfig


class MockDWAEnv:
    """Mock environment providing realistic robot state and sensor history for DWA."""
    def __init__(self):
        self.v = 0.15
        self.w = 0.05
        self.believed_pose = (1.0, 1.0, 0.1)
        self._last_angles = np.linspace(-math.pi, math.pi, C.N_SECTORS, endpoint=False)
        self._last_ranges = np.full(C.N_SECTORS, 1.5, dtype=np.float32)
        self._last_ranges[12] = 0.85

    def waypoint_world(self, lookahead: float = 1.2):
        return (2.2, 1.3)


def benchmark_latencies(model_path: str = "models/best/best_model.zip",
                        fallback_model: str = "models/ppo_navigation.zip",
                        n_warmup: int = 100,
                        n_eval: int = 1000) -> dict:
    device = "cpu"
    target_path = model_path if Path(model_path).exists() else fallback_model
    print(f"Loading model '{target_path}' on {device}...")
    model = PPO.load(target_path, device=device)
    safety_filter = NavigationSafetyFilter(SafetyConfig())
    dwa = DWAController(horizon_steps=15, n_v=5, n_w=15)

    # 1. Setup realistic mock inputs
    n_beams = 360
    angle_min = -math.pi
    angle_inc = 2.0 * math.pi / n_beams
    raw_scan = np.full(n_beams, 2.5, dtype=np.float32)
    # Add an obstacle directly ahead and at flanks
    raw_scan[170:190] = 0.85  # Obstacle ahead
    raw_scan[80:100] = 1.10   # Right obstacle
    raw_scan[260:280] = 1.10  # Left obstacle

    mock_dwa_env = MockDWAEnv()

    # Pre-computed sector observation for microbenchmarks
    dummy_obs = np.zeros((1, C.OBS_DIM), dtype=np.float32)
    dummy_ranges = np.full(C.N_SECTORS, 1.5, dtype=np.float32)
    dummy_ranges[12] = 0.85
    dummy_mask = np.ones(C.N_SECTORS, dtype=bool)

    # 2. Warm-up
    print(f"Running {n_warmup} warm-up iterations across all pipelines...")
    for _ in range(n_warmup):
        action, _ = model.predict(dummy_obs, deterministic=True)
        raw_v, raw_w = action_to_cmd(action[0] if action.ndim > 1 else action)
        safety_filter.filter_command(raw_v, raw_w, dummy_ranges, estimated_pose=(1.0, 1.0), valid_mask=dummy_mask)
        safety_filter.reset()
        _ = dwa.act_cmd(mock_dwa_env, dummy_obs)

    # 3. Microbenchmark: PPO Neural Network Forward Pass
    print(f"Timing {n_eval} iterations of PPO neural forward pass...")
    ppo_times_ms = []
    for _ in range(n_eval):
        t0 = time.perf_counter_ns()
        _ = model.predict(dummy_obs, deterministic=True)
        t1 = time.perf_counter_ns()
        ppo_times_ms.append((t1 - t0) / 1e6)

    # 4. Microbenchmark: Safety Filter Evaluation
    print(f"Timing {n_eval} iterations of Safety Filter pass...")
    filter_times_ms = []
    for _ in range(n_eval):
        safety_filter.reset()
        t0 = time.perf_counter_ns()
        _ = safety_filter.filter_command(0.22, 0.15, dummy_ranges, estimated_pose=(1.0, 1.0), valid_mask=dummy_mask)
        t1 = time.perf_counter_ns()
        filter_times_ms.append((t1 - t0) / 1e6)

    # 5. Microbenchmark: Policy + Safety Filter
    print(f"Timing {n_eval} iterations of Policy + Safety Filter...")
    policy_filter_times_ms = []
    for _ in range(n_eval):
        safety_filter.reset()
        t0 = time.perf_counter_ns()
        act, _ = model.predict(dummy_obs, deterministic=True)
        raw_v, raw_w = action_to_cmd(act[0] if act.ndim > 1 else act)
        _ = safety_filter.filter_command(raw_v, raw_w, dummy_ranges, estimated_pose=(1.0, 1.0), valid_mask=dummy_mask)
        t1 = time.perf_counter_ns()
        policy_filter_times_ms.append((t1 - t0) / 1e6)

    # 6. Full Pipeline Benchmark: LaserScan -> Sectors -> Obs -> PPO -> action_to_cmd -> Filter -> cmd_vel
    print(f"Timing {n_eval} iterations of Full Obs-to-Command Pipeline...")
    full_pipeline_times_ms = []
    wp_relative = (1.5, 0.2)  # 1.5m ahead, slight left offset
    for _ in range(n_eval):
        safety_filter.reset()
        t0 = time.perf_counter_ns()
        # a. Sector downsampling and validity masking
        sector_ranges, valid_mask = resample_scan(raw_scan, angle_min, angle_inc, return_mask=True)
        # b. Observation feature assembly (29-D)
        norm_ranges = sector_ranges / C.RANGE_MAX
        d_goal = math.hypot(wp_relative[0], wp_relative[1])
        alpha = math.atan2(wp_relative[1], wp_relative[0])
        obs_vec = np.empty((1, C.OBS_DIM), dtype=np.float32)
        obs_vec[0, :C.N_SECTORS] = norm_ranges
        obs_vec[0, 24] = min(d_goal / C.GOAL_DIST_NORM, 1.0)
        obs_vec[0, 25] = math.sin(alpha)
        obs_vec[0, 26] = math.cos(alpha)
        obs_vec[0, 27] = 0.15 / C.V_MAX
        obs_vec[0, 28] = 0.05 / C.W_MAX
        # c. PPO Inference
        act, _ = model.predict(obs_vec, deterministic=True)
        # d. Action unscaling
        cmd_v, cmd_w = action_to_cmd(act[0] if act.ndim > 1 else act)
        # e. Safety filter verification
        safe_v, safe_w, _ = safety_filter.filter_command(
            cmd_v, cmd_w, sector_ranges, estimated_pose=(1.0, 1.0), valid_mask=valid_mask
        )
        t1 = time.perf_counter_ns()
        full_pipeline_times_ms.append((t1 - t0) / 1e6)

    # 7. Matched Classical DWA Trajectory Evaluation Benchmark
    print(f"Timing {n_eval} iterations of Classical DWA Evaluation...")
    dwa_times_ms = []
    for _ in range(n_eval):
        t0 = time.perf_counter_ns()
        _ = dwa.act_cmd(mock_dwa_env, dummy_obs)
        t1 = time.perf_counter_ns()
        dwa_times_ms.append((t1 - t0) / 1e6)

    def stats_dict(arr: list[float]) -> dict:
        a = np.array(arr)
        return {
            "mean_ms": round(float(np.mean(a)), 3),
            "std_ms": round(float(np.std(a)), 3),
            "median_ms": round(float(np.median(a)), 3),
            "p90_ms": round(float(np.percentile(a, 90)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3),
            "p99_ms": round(float(np.percentile(a, 99)), 3),
            "min_ms": round(float(np.min(a)), 3),
            "max_ms": round(float(np.max(a)), 3),
        }

    dwa_med = float(np.median(dwa_times_ms))
    e2e_med = float(np.median(full_pipeline_times_ms))
    micro_med = float(np.median(policy_filter_times_ms))

    results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "platform": platform.platform(),
            "processor": platform.processor() or "x86_64",
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "n_warmup": n_warmup,
            "n_eval": n_eval,
            "device": device,
            "evaluated_model": target_path,
        },
        "ppo_forward_pass": stats_dict(ppo_times_ms),
        "safety_filter_pass": stats_dict(filter_times_ms),
        "policy_and_safety_filter": stats_dict(policy_filter_times_ms),
        "full_pipeline_obs_to_cmd": stats_dict(full_pipeline_times_ms),
        "dwa_trajectory_evaluation": stats_dict(dwa_times_ms),
        "comparisons": {
            "dwa_to_mas_rl_pipeline_speedup": round(dwa_med / e2e_med, 2) if e2e_med > 0 else 0.0,
            "dwa_to_policy_filter_speedup": round(dwa_med / micro_med, 2) if micro_med > 0 else 0.0,
        },
    }

    out_path = Path("results/latency_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nLatency benchmark saved to: {out_path}")

    print("\n======================== CPU LATENCY BENCHMARK RESULTS ========================")
    print(f"Platform: {results['metadata']['platform']} ({results['metadata']['processor']})")
    print(f"Iterations: {n_eval} timed cycles ({n_warmup} warmup)")
    print(f"{'Component':<32} | {'Mean (ms)':<10} | {'Median':<10} | {'p95':<10} | {'p99':<10}")
    print("-" * 80)
    for comp in ["ppo_forward_pass", "safety_filter_pass", "policy_and_safety_filter",
                 "full_pipeline_obs_to_cmd", "dwa_trajectory_evaluation"]:
        d = results[comp]
        print(f"{comp:<32} | {d['mean_ms']:<10.3f} | {d['median_ms']:<10.3f} | {d['p95_ms']:<10.3f} | {d['p99_ms']:<10.3f}")
    print("=" * 80)
    print(f"DWA vs MAS-RL Full Pipeline Speedup (Median): {results['comparisons']['dwa_to_mas_rl_pipeline_speedup']}x")
    print(f"DWA vs Policy+Filter Speedup (Median):         {results['comparisons']['dwa_to_policy_filter_speedup']}x")
    print("=" * 80)

    return results


if __name__ == "__main__":
    benchmark_latencies()
