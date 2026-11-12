"""Benchmark policy and safety filter inference latency on CPU.

Measures:
1. PPO Policy forward pass: observation normalization + Actor-Critic MLP inference.
2. Safety Filter pass: 4-layer predictive stopping governor, flank repulsion, recovery checks.
3. End-to-end MAS-RL cycle: observation build + PPO model.predict + safety filtering.

Reports: mean, std, median (p50), p95, p99, min, max, with CPU hardware and dependency metadata.
Saves: results/latency_benchmark.json
"""
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

import rl_training.config as C
from rl_training.safety import NavigationSafetyFilter, SafetyConfig


def benchmark_latencies(model_path: str = "models/ppo_navigation.zip",
                        n_warmup: int = 100,
                        n_eval: int = 1000) -> dict:
    device = "cpu"
    print(f"Loading model '{model_path}' on {device}...")
    model = PPO.load(model_path, device=device)
    safety_filter = NavigationSafetyFilter(SafetyConfig())

    # Create dummy observation (29-D) and sector ranges (24 sectors)
    dummy_obs = np.zeros((1, C.OBS_DIM), dtype=np.float32)
    dummy_ranges = np.full(C.N_SECTORS, 1.5, dtype=np.float32)
    dummy_ranges[12] = 0.8  # Front sector obstacle
    dummy_mask = np.ones(C.N_SECTORS, dtype=bool)

    # 1. Warm-up
    print(f"Running {n_warmup} warm-up iterations...")
    for _ in range(n_warmup):
        action, _ = model.predict(dummy_obs, deterministic=True)
        raw_v = float(action[0, 0] * C.V_MAX) if action.ndim > 1 else float(action[0] * C.V_MAX)
        raw_w = float(action[0, 1] * C.W_MAX) if action.ndim > 1 else float(action[1] * C.W_MAX)
        safety_filter.filter_command(raw_v, raw_w, dummy_ranges, estimated_pose=(1.0, 1.0), valid_mask=dummy_mask)

    # 2. Benchmark PPO Neural Network Forward Pass
    print(f"Timing {n_eval} iterations of PPO neural forward pass...")
    ppo_times_ms = []
    for _ in range(n_eval):
        t0 = time.perf_counter_ns()
        _ = model.predict(dummy_obs, deterministic=True)
        t1 = time.perf_counter_ns()
        ppo_times_ms.append((t1 - t0) / 1e6)

    # 3. Benchmark Safety Filter Evaluation
    print(f"Timing {n_eval} iterations of Safety Filter pass...")
    filter_times_ms = []
    for _ in range(n_eval):
        t0 = time.perf_counter_ns()
        _ = safety_filter.filter_command(0.22, 0.15, dummy_ranges, estimated_pose=(1.0, 1.0), valid_mask=dummy_mask)
        t1 = time.perf_counter_ns()
        filter_times_ms.append((t1 - t0) / 1e6)

    # 4. Benchmark Full End-to-End Cycle
    print(f"Timing {n_eval} iterations of Full End-to-End MAS-RL cycle...")
    e2e_times_ms = []
    for _ in range(n_eval):
        t0 = time.perf_counter_ns()
        act, _ = model.predict(dummy_obs, deterministic=True)
        v_cmd = float(act[0, 0] * C.V_MAX) if act.ndim > 1 else float(act[0] * C.V_MAX)
        w_cmd = float(act[0, 1] * C.W_MAX) if act.ndim > 1 else float(act[1] * C.W_MAX)
        _ = safety_filter.filter_command(v_cmd, w_cmd, dummy_ranges, estimated_pose=(1.0, 1.0), valid_mask=dummy_mask)
        t1 = time.perf_counter_ns()
        e2e_times_ms.append((t1 - t0) / 1e6)

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
        },
        "ppo_forward_pass": stats_dict(ppo_times_ms),
        "safety_filter_pass": stats_dict(filter_times_ms),
        "end_to_end_mas_rl": stats_dict(e2e_times_ms),
    }

    out_path = Path("results/latency_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nLatency benchmark saved to: {out_path}")

    print("\n================== CPU LATENCY BENCHMARK RESULTS ==================")
    print(f"Platform: {results['metadata']['platform']} ({results['metadata']['processor']})")
    print(f"Iterations: {n_eval} timed cycles ({n_warmup} warmup)")
    print(f"{'Component':<25} | {'Mean (ms)':<10} | {'Median':<10} | {'p95':<10} | {'p99':<10}")
    print("-" * 72)
    for comp in ["ppo_forward_pass", "safety_filter_pass", "end_to_end_mas_rl"]:
        d = results[comp]
        print(f"{comp:<25} | {d['mean_ms']:<10.3f} | {d['median_ms']:<10.3f} | {d['p95_ms']:<10.3f} | {d['p99_ms']:<10.3f}")
    print("=" * 72)

    return results


if __name__ == "__main__":
    benchmark_latencies()
