"""Controllers behind one interface, plus an episode runner.

* ReactiveController : map-free goal seeking + obstacle avoidance (weak baseline).
* DWAController      : global plan (geodesic field from the believed pose) + Dynamic Window
                       local planner over LiDAR points. A Nav2-*style* baseline in the
                       standalone environment; the real Nav2 baseline is run in Gazebo.
* PPOController      : trained Stable-Baselines3 policy.

Run `python -m rl_training.baselines` to evaluate the two classical controllers.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .metrics import summarize
from .nav_env import NavEnv
from .obs import cmd_to_action, goal_in_robot_frame, action_to_cmd
from .scenarios import make_scenarios


class Controller:
    name = "controller"

    def reset(self, env: NavEnv) -> None:
        pass

    def act(self, env: NavEnv, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ReactiveController(Controller):
    name = "reactive"

    def act(self, env, obs):
        s = obs[:C.N_SECTORS] * C.RANGE_MAX
        _, ang = env.goal_rel(direct=True)
        front = float(s[11:14].min())
        left, right = float(s[13:19].mean()), float(s[6:12].mean())
        if front < 0.7:
            w = C.W_MAX * (1.0 if left >= right else -1.0)
            v = 0.03 if front < 0.4 else 0.08
        else:
            w = float(np.clip(2.0 * ang, -C.W_MAX, C.W_MAX))
            v = C.V_MAX * max(math.cos(ang), 0.0) * min(1.0, front / 1.5)
        return cmd_to_action(v, w)


class DWAController(Controller):
    name = "dwa_global"

    def __init__(self, horizon_steps: int = 15, n_v: int = 5, n_w: int = 15):
        self.T, self.n_v, self.n_w = horizon_steps, n_v, n_w

    def act(self, env, obs):
        wp = env.waypoint_world(1.2)
        # waypoint in robot frame (believed pose)
        bx, by, bth = env.believed_pose
        dx, dy = wp[0] - bx, wp[1] - by
        c, s = math.cos(-bth), math.sin(-bth)
        wp_r = np.array([c * dx - s * dy, s * dx + c * dy])
        # obstacle points from the (noisy) scan, robot frame
        ang, rng = env._last_angles.ravel(), env._last_ranges.ravel()
        hit = rng < C.RANGE_MAX * 0.999
        pts = np.stack([rng[hit] * np.cos(ang[hit]), rng[hit] * np.sin(ang[hit])], 1)

        dv, dw = C.A_V_MAX * C.DT, C.A_W_MAX * C.DT
        vs = np.linspace(max(0.0, env.v - dv), min(C.V_MAX, env.v + dv), self.n_v)
        ws = np.linspace(max(-C.W_MAX, env.w - dw), min(C.W_MAX, env.w + dw), self.n_w)
        V, Wm = np.meshgrid(vs, ws, indexing="ij")
        V, Wm = V.ravel(), Wm.ravel()
        # roll out constant (v, w); same integration order as the environment
        th = np.cumsum(np.tile(Wm[:, None] * C.DT, (1, self.T)), axis=1)
        th_prev = np.concatenate([np.zeros((len(V), 1)), th[:, :-1]], axis=1)
        X = np.cumsum(V[:, None] * np.cos(th_prev) * C.DT, axis=1)
        Y = np.cumsum(V[:, None] * np.sin(th_prev) * C.DT, axis=1)
        if len(pts):
            d = np.hypot(X[:, :, None] - pts[None, None, :, 0], Y[:, :, None] - pts[None, None, :, 1])
            clear = d.min(axis=(1, 2)) - C.ROBOT_RADIUS
        else:
            clear = np.full(len(V), 5.0)
        dist_end = np.hypot(X[:, -1] - wp_r[0], Y[:, -1] - wp_r[1])
        head_end = np.abs(np.arctan2(wp_r[1] - Y[:, -1], wp_r[0] - X[:, -1]) - th[:, -1])
        head_end = np.abs((head_end + math.pi) % (2 * math.pi) - math.pi)
        score = -1.0 * dist_end - 0.3 * head_end + 0.6 * np.minimum(clear, 0.8) + 0.3 * V / C.V_MAX
        score = np.where(clear < 0.02, -1e6 + clear, score)  # inadmissible
        k = int(np.argmax(score))
        if score[k] < -1e5:  # boxed in: rotate on the spot toward more open side
            left = float(env._last_ranges[13:19].mean()); right = float(env._last_ranges[6:12].mean())
            return cmd_to_action(0.0, C.W_MAX if left >= right else -C.W_MAX)
        return cmd_to_action(V[k], Wm[k])


class PPOController(Controller):
    def __init__(self, model, name: str = "ppo"):
        self.model, self.name = model, name

    def act(self, env, obs):
        return self.model.predict(obs, deterministic=True)[0]


class GovernorPPOController(Controller):
    """PPO Local Policy augmented with Layers 1-3 Continuous CBF Safety Governor.
    
    Ablation controller: strictly forward-only action space (v in [0, V_MAX]).
    No reverse recovery maneuvers. Proves that analytical stopping-distance constraints
    alone eliminate collisions without expanding the action space.
    """
    def __init__(self, model, name: str = "ppo_governor"):
        self.model, self.name = model, name

    def reset(self, env):
        pass

    def act(self, env, obs):
        raw_act = self.model.predict(obs, deterministic=True)[0]
        v, w = action_to_cmd(raw_act)

        sector_ranges = obs[:C.N_SECTORS] * C.RANGE_MAX
        front_narrow = sector_ranges[10:15]  # ±30°
        front_wide = sector_ranges[8:17]     # ±60°
        min_front = float(front_narrow.min())
        min_wide = float(front_wide.min())

        # Layer 2: Analytical Discrete-Time CBF Stopping-Distance Governor
        clearance = max(0.0, min_wide - C.ROBOT_RADIUS - 0.06)
        v_allowable = math.sqrt(2.0 * C.A_V_MAX * clearance)
        v = min(v, v_allowable)
        if min_front < C.ROBOT_RADIUS + 0.08:
            v = 0.0

        # Layer 3: Lateral Repulsive Potential Field Steering
        left_dist = float(sector_ranges[13:18].min())
        right_dist = float(sector_ranges[7:12].min())
        repulse_w = 0.0
        if left_dist < 0.45:
            repulse_w -= 0.7 * (1.0 - left_dist / 0.45)
        if right_dist < 0.45:
            repulse_w += 0.7 * (1.0 - right_dist / 0.45)
        w = float(np.clip(w + repulse_w, -C.W_MAX, C.W_MAX))

        # Anti-Deadlock Symmetry Breaking
        if v == 0.0:
            preferred_dir = 1.0 if left_dist >= right_dist else -1.0
            if abs(w) < 0.4:
                w = 0.85 * preferred_dir

        return cmd_to_action(max(0.0, v), w)


class MASRLController(Controller):
    """Medial-Axis Shielded Reinforcement Learning (MAS-RL) Controller.
    
    Full 4-Layer Architecture:
    1. Multi-sector forward arc collision forecasting (sectors 8 to 16, ±60°)
    2. Discrete-Time Control Barrier Function (CBF) stopping-distance governor
    3. Lateral repulsive potential field steering
    4. Anti-deadlock symmetry breaking & stagnation watchdog
    5. Autonomous escape reverse-and-pivot recovery (v = -0.12 m/s) when trapped
    """
    def __init__(self, model, name: str = "mas_rl"):
        self.model, self.name = model, name
        self.recovery_steps = 0
        self.recovery_dir = 1.0
        self.history = []

    def reset(self, env):
        self.recovery_steps = 0
        self.recovery_dir = 1.0
        self.history.clear()

    def act(self, env, obs):
        sector_ranges = obs[:C.N_SECTORS] * C.RANGE_MAX
        front_narrow = sector_ranges[10:15]  # ±30°
        front_wide = sector_ranges[8:17]     # ±60°
        min_front = float(front_narrow.min())
        min_wide = float(front_wide.min())
        min_overall = float(sector_ranges.min())
        left_dist = float(sector_ranges[13:18].min())
        right_dist = float(sector_ranges[7:12].min())

        # 1. Active Recovery State Machine
        if self.recovery_steps > 0:
            self.recovery_steps -= 1
            # Check rear clearance before backing up
            rear_sectors = sector_ranges[[22, 23, 0, 1, 2]]
            if float(rear_sectors.min()) > C.ROBOT_RADIUS + 0.08:
                return cmd_to_action(-0.12, 0.85 * self.recovery_dir)
            else:
                return cmd_to_action(0.0, 0.95 * self.recovery_dir)

        # 2. Query Raw PPO Policy
        raw_act = self.model.predict(obs, deterministic=True)[0]
        v, w = action_to_cmd(raw_act)

        # 3. Emergency Proximity & Stagnation Detection
        rx, ry = float(env.pose[0]), float(env.pose[1])
        self.history.append((rx, ry))
        if len(self.history) > 15:
            self.history.pop(0)

        is_stagnated = False
        if len(self.history) == 15:
            disp = math.hypot(rx - self.history[0][0], ry - self.history[0][1])
            if disp < 0.04 and env.steps > 15:
                is_stagnated = True

        if min_overall < C.ROBOT_RADIUS + 0.04 or is_stagnated:
            self.recovery_steps = 8
            closest_idx = int(np.argmin(sector_ranges))
            self.recovery_dir = -1.0 if closest_idx >= 12 else 1.0
            self.history.clear()
            rear_sectors = sector_ranges[[22, 23, 0, 1, 2]]
            if float(rear_sectors.min()) > C.ROBOT_RADIUS + 0.08:
                return cmd_to_action(-0.12, 0.85 * self.recovery_dir)
            else:
                return cmd_to_action(0.0, 0.95 * self.recovery_dir)

        # 4. Predictive Stopping-Distance Governor (Discrete CBF)
        clearance = max(0.0, min_wide - C.ROBOT_RADIUS - 0.06)
        v_allowable = math.sqrt(2.0 * C.A_V_MAX * clearance)
        v = min(v, v_allowable)
        if min_front < C.ROBOT_RADIUS + 0.08:
            v = 0.0

        # 5. Lateral Repulsive Potential Field Steering
        repulse_w = 0.0
        if left_dist < 0.45:
            repulse_w -= 0.7 * (1.0 - left_dist / 0.45)
        if right_dist < 0.45:
            repulse_w += 0.7 * (1.0 - right_dist / 0.45)
        w = float(np.clip(w + repulse_w, -C.W_MAX, C.W_MAX))

        # 6. Anti-Deadlock Symmetry Breaking
        if v == 0.0:
            preferred_dir = 1.0 if left_dist >= right_dist else -1.0
            if abs(w) < 0.4:
                w = 0.85 * preferred_dir

        return cmd_to_action(v, w)


# Backward-compatible alias
ShieldedPPOController = MASRLController


def run_episode(env: NavEnv, ctrl: Controller, scenario=None, seed: int | None = None,
                record: bool = False) -> dict:
    obs, info = env.reset(seed=seed, options={"scenario": scenario} if scenario else None)
    ctrl.reset(env)
    traj = [env.pose.copy()] if record else None
    while True:
        obs, _, term, trunc, info = env.step(ctrl.act(env, obs))
        if record:
            traj.append(env.pose.copy())
        if term or trunc:
            break
    out = dict(info)
    if record:
        out["traj"] = np.array(traj)
    return out


def evaluate_controller(ctrl: Controller, scenarios, env_kwargs: dict | None = None,
                        label: str | None = None) -> pd.DataFrame:
    env = NavEnv(**(env_kwargs or {}))
    rows = []
    for scn in scenarios:
        res = run_episode(env, ctrl, scn, seed=scn.sid)
        rows.append({"controller": label or ctrl.name, "scenario": scn.sid,
                     **{k: res[k] for k in ("layout_seed", "family", "success", "collision", "timeout",
                                            "steps", "time_s", "path_length", "shortest_len", "spl")}})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the classical baselines.")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--split", default="test", choices=["val", "test", "heldout"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/baselines_evaluation.csv")
    a = ap.parse_args()
    scns = make_scenarios(a.episodes, a.split, a.seed)
    dfs = [evaluate_controller(c, scns) for c in (ReactiveController(), DWAController())]
    df = pd.concat(dfs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    for name, g in df.groupby("controller"):
        s = summarize(g)
        print(f"{name:12s} n={s['n']} success={s['success_rate']:.3f} "
              f"[{s['success_ci_lo']:.3f},{s['success_ci_hi']:.3f}] collision={s['collision_rate']:.3f} "
              f"SPL={s['spl']:.3f}")
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
