"""Reproducible evaluation scenarios (layout + start + goal) shared by every
controller so that comparisons are matched."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import config as C
from .maps import HELDOUT_FAMILIES, WorldBundle, get_bundle
from .planning import geodesic_field, rc_to_xy

SPLITS = {
    "train": (C.TRAIN_SEEDS, None),
    "val": (C.VAL_SEEDS, None),
    "test": (C.TEST_SEEDS, None),                       # unseen layouts, seen families
    "heldout": (C.HELDOUT_SEEDS, HELDOUT_FAMILIES[0]),  # unseen layouts AND family
}


@dataclass
class Scenario:
    layout_seed: int
    family_override: str | None
    start: tuple[float, float, float]   # x, y, yaw
    goal: tuple[float, float]
    shortest_len: float                 # 8-connected grid geodesic, metres
    sid: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(d["layout_seed"], d.get("family_override"), tuple(d["start"]),
                        tuple(d["goal"]), d["shortest_len"], d.get("sid", 0))


def sample_start_goal(bundle: WorldBundle, rng: np.random.Generator,
                      dmin: float, dmax: float, tries: int = 30):
    """Returns (start_xy, goal_xy, shortest_len, field) or None."""
    free = np.argwhere(bundle.clear_free)
    if len(free) < 50:
        return None
    for _ in range(tries):
        g = tuple(int(v) for v in free[rng.integers(len(free))])
        field = geodesic_field(bundle.plan_blocked, g)
        cand = np.argwhere(bundle.clear_free & np.isfinite(field) & (field >= dmin) & (field <= dmax))
        if len(cand) == 0:
            continue
        s = cand[rng.integers(len(cand))]
        return rc_to_xy(*s), rc_to_xy(*g), float(field[s[0], s[1]]), field
    return None


def make_scenarios(n: int, split: str = "test", seed: int = 0,
                   dmin: float = 2.0, dmax: float = 9.0) -> list[Scenario]:
    seeds, fam = SPLITS[split]
    rng = np.random.default_rng(seed)
    out: list[Scenario] = []
    i = 0
    while len(out) < n:
        layout = seeds[i % len(seeds)]
        i += 1
        res = sample_start_goal(get_bundle(layout, fam), rng, dmin, dmax)
        if res is None:
            continue
        (sx, sy), goal, shortest, _ = res
        out.append(Scenario(layout, fam, (sx, sy, float(rng.uniform(-np.pi, np.pi))),
                            goal, shortest, len(out)))
    return out


def save_scenarios(scns: list[Scenario], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([s.to_dict() for s in scns], indent=1))


def load_scenarios(path: str | Path) -> list[Scenario]:
    return [Scenario.from_dict(d) for d in json.loads(Path(path).read_text())]
