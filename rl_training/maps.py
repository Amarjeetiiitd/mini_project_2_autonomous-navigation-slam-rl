"""Procedural 2D indoor layouts built from axis-aligned rectangles, plus the
geometry primitives (ray casting, collision) used by the environment."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import ndimage

from . import config as C

FAMILIES = ("open", "corridor", "rooms", "clutter")
HELDOUT_FAMILIES = ("zigzag",)
WALL_T = 0.1


@dataclass
class World:
    seed: int
    family: str
    width: float
    height: float
    rects: np.ndarray  # (N, 4): x0, y0, x1, y1

    def grid(self, res: float = C.GRID_RES) -> np.ndarray:
        """Boolean occupancy grid, shape (ny, nx), True = occupied."""
        nx, ny = int(round(self.width / res)), int(round(self.height / res))
        occ = np.zeros((ny, nx), dtype=bool)
        for x0, y0, x1, y1 in self.rects:
            i0, i1 = max(int(np.floor(x0 / res)), 0), min(int(np.ceil(x1 / res)), nx)
            j0, j1 = max(int(np.floor(y0 / res)), 0), min(int(np.ceil(y1 / res)), ny)
            occ[j0:j1, i0:i1] = True
        return occ


# ------------------------------------------------------------- generation
def _wall(orient: str, pos: float, a: float, b: float, gap_c: float | None = None,
          gap_w: float = 0.0) -> list[list[float]]:
    """Wall along 'v' (x = pos, spanning y in [a, b]) or 'h' with an optional gap."""
    t = WALL_T / 2
    segs = [(a, b)]
    if gap_c is not None:
        segs = [(a, gap_c - gap_w / 2), (gap_c + gap_w / 2, b)]
    out = []
    for s0, s1 in segs:
        if s1 - s0 < 0.05:
            continue
        out.append([pos - t, s0, pos + t, s1] if orient == "v" else [s0, pos - t, s1, pos + t])
    return out


def _boxes(rng, n, W, H, smin=0.3, smax=1.0):
    out = []
    for _ in range(n):
        w, h = rng.uniform(smin, smax, 2)
        cx, cy = rng.uniform(1.0, W - 1.0), rng.uniform(1.0, H - 1.0)
        out.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return out


def generate_world(seed: int, family: str | None = None) -> World:
    rng = np.random.default_rng(seed)
    fam = family or FAMILIES[seed % len(FAMILIES)]
    W, H = float(rng.uniform(8, 12)), float(rng.uniform(7, 10))
    t = WALL_T
    rects: list[list[float]] = [[0, 0, W, t], [0, H - t, W, H], [0, 0, t, H], [W - t, 0, W, H]]

    if fam == "open":
        rects += _boxes(rng, int(rng.integers(4, 9)), W, H)
    elif fam == "corridor":
        x = W * rng.uniform(0.4, 0.6)
        rects += _wall("v", x, 0, H, rng.uniform(1.0, H - 1.0), rng.uniform(0.9, 1.4))
        rects += _boxes(rng, int(rng.integers(2, 6)), W, H)
    elif fam == "rooms":
        ncols, nrows = int(rng.integers(2, 4)), 2
        for k in range(1, ncols):
            for r in range(nrows):
                a, b = r * H / nrows, (r + 1) * H / nrows
                rects += _wall("v", W * k / ncols, a, b,
                               rng.uniform(a + 0.7, b - 0.7), rng.uniform(0.9, 1.3))
        for c in range(ncols):
            a, b = c * W / ncols, (c + 1) * W / ncols
            rects += _wall("h", H / 2, a, b, rng.uniform(a + 0.7, b - 0.7), rng.uniform(0.9, 1.3))
        rects += _boxes(rng, int(rng.integers(0, 4)), W, H, 0.3, 0.6)
    elif fam == "clutter":
        rects += _boxes(rng, int(rng.integers(9, 16)), W, H, 0.3, 0.9)
    elif fam == "zigzag":
        bottom = bool(rng.integers(2))
        for k in range(3):
            gap_c = 1.2 if bottom else H - 1.2
            rects += _wall("v", W * (k + 1) / 4, 0, H, gap_c, rng.uniform(1.0, 1.4))
            bottom = not bottom
    else:
        raise ValueError(f"unknown family {fam}")
    return World(seed, fam, W, H, np.asarray(rects, dtype=np.float64))


# --------------------------------------------------------------- geometry
def raycast(origin, angles: np.ndarray, rects: np.ndarray,
            max_range: float = C.RANGE_MAX) -> np.ndarray:
    """Distance from `origin` along each absolute angle to the nearest rectangle,
    clipped to `max_range`. Slab method, fully vectorised over rays x rects."""
    ox, oy = origin
    dx, dy = np.cos(angles), np.sin(angles)
    dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)[:, None]
    dy = np.where(np.abs(dy) < 1e-12, 1e-12, dy)[:, None]
    t1x, t2x = (rects[None, :, 0] - ox) / dx, (rects[None, :, 2] - ox) / dx
    t1y, t2y = (rects[None, :, 1] - oy) / dy, (rects[None, :, 3] - oy) / dy
    tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
    tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
    hit = (tmax >= tmin) & (tmax > 0)
    t = np.where(hit, np.maximum(tmin, 0.0), np.inf).min(axis=1)
    return np.minimum(t, max_range)


def collides(x: float, y: float, rects: np.ndarray, radius: float = C.ROBOT_RADIUS) -> bool:
    ddx = np.maximum(np.maximum(rects[:, 0] - x, x - rects[:, 2]), 0.0)
    ddy = np.maximum(np.maximum(rects[:, 1] - y, y - rects[:, 3]), 0.0)
    return bool(np.any(ddx * ddx + ddy * ddy < radius * radius))


def rect_clearance(x: float, y: float, rects: np.ndarray) -> float:
    """Distance (m) from point (x, y) to the nearest rectangle edge."""
    if len(rects) == 0:
        return 10.0
    ddx = np.maximum(np.maximum(rects[:, 0] - x, x - rects[:, 2]), 0.0)
    ddy = np.maximum(np.maximum(rects[:, 1] - y, y - rects[:, 3]), 0.0)
    return float(np.sqrt(np.min(ddx * ddx + ddy * ddy)))


# ----------------------------------------------------------- cached bundle
@dataclass
class WorldBundle:
    world: World
    occ: np.ndarray           # raw occupancy
    plan_blocked: np.ndarray  # inflated for planning
    clear_free: np.ndarray    # cells with enough clearance to start/goal in


def _inflate(occ: np.ndarray, radius: float, res: float) -> np.ndarray:
    dist = ndimage.distance_transform_edt(~occ) * res
    return dist <= radius


def make_plan_blocked(width: float, height: float, rects: np.ndarray,
                      radius: float = C.PLAN_INFLATION,
                      res: float = C.GRID_RES) -> np.ndarray:
    """Generate inflated boolean occupancy grid (True = blocked) from an array of rectangles."""
    nx, ny = int(round(width / res)), int(round(height / res))
    occ = np.zeros((ny, nx), dtype=bool)
    if len(rects) > 0:
        for x0, y0, x1, y1 in rects:
            i0, i1 = max(int(np.floor(x0 / res)), 0), min(int(np.ceil(x1 / res)), nx)
            j0, j1 = max(int(np.floor(y0 / res)), 0), min(int(np.ceil(y1 / res)), ny)
            occ[j0:j1, i0:i1] = True
    return _inflate(occ, radius, res)


@lru_cache(maxsize=4096)
def get_bundle(seed: int, family: str | None = None) -> WorldBundle:
    w = generate_world(seed, family)
    occ = w.grid()
    return WorldBundle(w, occ, _inflate(occ, C.PLAN_INFLATION, C.GRID_RES),
                       ~_inflate(occ, C.START_CLEARANCE, C.GRID_RES))
