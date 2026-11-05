"""Sanity checks for the Gymnasium environment and the shared observation adapter."""
from __future__ import annotations

import numpy as np
from gymnasium.utils.env_checker import check_env

from . import config as C
from .nav_env import NavEnv
from .obs import resample_scan


def main() -> None:
    env = NavEnv()
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (C.OBS_DIM,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)

    # Real-robot adapter consistency: a 360-ray scan resampled by the ROS adapter must
    # match the environment's native sectors (adapter can only be <= by finer sampling).
    ranges, a0, inc = env.scan_full(360)
    sect = resample_scan(ranges, a0, inc)
    native = obs[:C.N_SECTORS] * C.RANGE_MAX
    assert np.all(sect <= native + 1e-3), "adapter should never see farther than native rays"
    assert np.abs(sect - native).mean() < 0.25, "adapter/env sector mismatch"

    for ep in range(3):
        env.reset(seed=ep)
        for _ in range(50):
            obs, r, te, tr, info = env.step(env.action_space.sample())
            assert env.observation_space.contains(obs) and np.isfinite(r)
            if te or tr:
                break
    print("Environment checks passed.")


if __name__ == "__main__":
    main()
