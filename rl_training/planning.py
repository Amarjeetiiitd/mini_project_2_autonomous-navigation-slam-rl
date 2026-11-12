"""Grid-based global planning: geodesic distance field (Dijkstra, 8-connected),
path extraction by gradient descent, and smooth interpolation of the field."""
from __future__ import annotations

import heapq
import math

import numpy as np
from scipy import ndimage

from . import config as C
from .maps import collides, rect_clearance

_SQ2 = math.sqrt(2.0)
_NBRS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
         (-1, -1, _SQ2), (-1, 1, _SQ2), (1, -1, _SQ2), (1, 1, _SQ2)]


def make_clearance_penalty(blocked: np.ndarray, res: float = C.GRID_RES,
                           margin: float = 0.25, max_penalty: float = 0.35) -> np.ndarray:
    """Compute clearance penalty using Euclidean Distance Transform (EDT).

    Acts as a medial-axis-inspired clearance-aware tie-breaker that guides Dijkstra
    along corridor and doorway midlines without distorting Euclidean path length metrics
    or inducing non-optimal room detours.
    """
    dist = ndimage.distance_transform_edt(~blocked) * res
    penalty = np.where((dist < margin) & ~blocked, ((margin - dist) / margin) ** 2 * max_penalty, 0.0)
    return penalty


def geodesic_field(blocked: np.ndarray, goal_rc: tuple[int, int],
                   res: float = C.GRID_RES,
                   clearance_cost: np.ndarray | None = None) -> np.ndarray:
    """Shortest-path distance (m) from every free cell to `goal_rc`; inf if unreachable."""
    ny, nx = blocked.shape
    blk = blocked.tolist()
    dist = [[math.inf] * nx for _ in range(ny)]
    gr, gc = int(goal_rc[0]), int(goal_rc[1])
    if blk[gr][gc]:
        # Target coordinate is inside inflation margin; find closest valid free cell
        best_seed, best_d = None, 1e9
        for dr in range(-6, 7):
            for dc in range(-6, 7):
                rr, cc = gr + dr, gc + dc
                if 0 <= rr < ny and 0 <= cc < nx and not blk[rr][cc]:
                    d = dr * dr + dc * dc
                    if d < best_d:
                        best_seed, best_d = (rr, cc), d
        if best_seed is not None:
            gr, gc = best_seed
        else:
            return np.full((ny, nx), np.inf)
    dist[gr][gc] = 0.0
    heap = [(0.0, gr, gc)]
    has_cost = clearance_cost is not None
    while heap:
        d, r, c = heapq.heappop(heap)
        if d > dist[r][c]:
            continue
        for dr, dc, w in _NBRS:
            rr, cc = r + dr, c + dc
            if rr < 0 or cc < 0 or rr >= ny or cc >= nx or blk[rr][cc]:
                continue
            if dr and dc and (blk[r + dr][c] or blk[r][c + dc]):
                continue  # no corner cutting
            cost_mult = 1.0 + (float(clearance_cost[rr, cc]) if has_cost else 0.0)
            nd = d + w * res * cost_mult
            if nd < dist[rr][cc]:
                dist[rr][cc] = nd
                heapq.heappush(heap, (nd, rr, cc))
    return np.asarray(dist)


def xy_to_rc(x: float, y: float, shape, res: float = C.GRID_RES) -> tuple[int, int]:
    r = int(np.clip(int(y / res), 0, shape[0] - 1))
    c = int(np.clip(int(x / res), 0, shape[1] - 1))
    return r, c


def rc_to_xy(r: int, c: int, res: float = C.GRID_RES) -> tuple[float, float]:
    return (c + 0.5) * res, (r + 0.5) * res


def nearest_finite(field: np.ndarray, rc: tuple[int, int], radius: int = 6):
    r0, c0 = rc
    if np.isfinite(field[r0, c0]):
        return rc
    best, best_d = None, 1e9
    for r in range(max(r0 - radius, 0), min(r0 + radius + 1, field.shape[0])):
        for c in range(max(c0 - radius, 0), min(c0 + radius + 1, field.shape[1])):
            if np.isfinite(field[r, c]):
                d = (r - r0) ** 2 + (c - c0) ** 2
                if d < best_d:
                    best, best_d = (r, c), d
    return best


def extract_path(field: np.ndarray, rc: tuple[int, int], max_len: float,
                 res: float = C.GRID_RES) -> list[tuple[int, int]]:
    """Greedy steepest descent of the geodesic field starting at `rc` until `max_len` metres
    of path were collected or the goal (field == 0) is reached."""
    start = nearest_finite(field, rc)
    if start is None or not np.isfinite(field[start[0], start[1]]):
        return []
    path, length = [start], 0.0
    r, c = start
    ny, nx = field.shape
    visited = {(r, c)}
    while length < max_len and field[r, c] > 1e-4:
        best, best_grad, bw = None, 0.0, 1.0
        cur_v = field[r, c]
        for dr, dc, w in _NBRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < ny and 0 <= cc < nx and (rr, cc) not in visited:
                val = field[rr, cc]
                if np.isfinite(val) and val < cur_v:
                    grad = (cur_v - val) / w
                    if grad > best_grad:
                        best, best_grad, bw = (rr, cc), grad, w
        if best is None:
            break
        r, c = best
        visited.add((r, c))
        length += bw * res
        path.append(best)
    return path


def smooth_sampler(field: np.ndarray):
    """Return f(x, y) -> geodesic distance with bilinear interpolation. Unreachable
    cells are replaced by (max finite + 5 m) so the reward stays well-behaved."""
    finite = np.isfinite(field)
    filled = np.where(finite, field, (field[finite].max() if finite.any() else 0.0) + 5.0)

    def f(x: float, y: float) -> float:
        coords = np.array([[y / C.GRID_RES - 0.5], [x / C.GRID_RES - 0.5]])
        return float(ndimage.map_coordinates(filled, coords, order=1, mode="nearest")[0])

    return f


def has_line_of_sight(p1: tuple[float, float], p2: tuple[float, float],
                      rects: np.ndarray, radius: float = C.ROBOT_RADIUS + 0.03,
                      step: float = 0.05) -> bool:
    """True if the straight line segment between p1 and p2 has at least `radius` clearance."""
    if len(rects) == 0:
        return True
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1e-4:
        return not collides(x1, y1, rects, radius=radius)
    n_pts = max(2, int(math.ceil(dist / step)))
    xs = np.linspace(x1, x2, n_pts)
    ys = np.linspace(y1, y2, n_pts)
    for x, y in zip(xs, ys):
        if collides(float(x), float(y), rects, radius=radius):
            return False
    return True


def shortcut_path(path_xy: list[tuple[float, float]], rects: np.ndarray,
                  clearance_radius: float = C.ROBOT_RADIUS + 0.04) -> list[tuple[float, float]]:
    """Taut string-pulling (Theta*-style line-of-sight shortcutting):
    Replaces jagged 8-connected grid staircase segments with the true straight Euclidean
    chords having at least `clearance_radius` obstacle clearance."""
    if len(path_xy) <= 2:
        return list(path_xy)

    pruned = [path_xy[0]]
    curr_i = 0
    n = len(path_xy)

    while curr_i < n - 1:
        # Greedily search furthest reachable waypoint with direct collision-free line of sight
        furthest_j = curr_i + 1
        for j in range(n - 1, curr_i, -1):
            if has_line_of_sight(path_xy[curr_i], path_xy[j], rects, radius=clearance_radius):
                furthest_j = j
                break
        pruned.append(path_xy[furthest_j])
        curr_i = furthest_j

    return pruned


def extract_visible_waypoint(field: np.ndarray, rc: tuple[int, int],
                             rx: float, ry: float, rects: np.ndarray,
                             max_len: float = C.LOOKAHEAD * 1.5,
                             clearance_radius: float = C.ROBOT_RADIUS + 0.03) -> tuple[float, float]:
    """Extract path cells from geodesic field, returning the furthest waypoint with unobstructed line of sight.

    Includes Constrained Corridor Lookahead (Portal Invariance):
    When negotiating a narrow constriction or doorway, lookahead is adaptively contracted
    to guide the robot squarely through the portal before turning around blind room corners.
    """
    local_clr = rect_clearance(rx, ry, rects)
    if local_clr < C.ROBOT_RADIUS + 0.12:
        eff_max_len = min(max_len, max(0.65, 2.2 * (local_clr - C.ROBOT_RADIUS + 0.04)))
    else:
        eff_max_len = max_len

    path_cells = extract_path(field, rc, max_len=eff_max_len)
    if not path_cells:
        return rc_to_xy(*rc)

    # Check from furthest cell back to nearest for unobstructed line of sight
    for cell in reversed(path_cells):
        cx, cy = rc_to_xy(*cell)
        if has_line_of_sight((rx, ry), (cx, cy), rects, radius=clearance_radius):
            return cx, cy

    # Fallback to nearest path cell
    idx = min(1, len(path_cells) - 1)
    return rc_to_xy(*path_cells[idx])
