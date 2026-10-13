"""
guard_warehouse.py -- THE INTENT GUARD (build log entry 29).

WHAT IT IS
  The last, exact safety check of a step. By the time it runs, every fleet's
  move for this step is decided (step.simultaneous) and every fleet has its
  braked speed. For every pair of MOBILE fleets close enough to meet this step,
  the guard takes both actual moves -- the intent each fleet would broadcast to
  its neighbours -- and computes where both would land. If the gap after the
  step would fall below `min_gap` hops, the fleets closing the gap are slowed
  just enough to stop at it, down to standing still. Slowing one fleet can make
  another pair short (a follower behind it), so it repeats until no pair is
  short; whatever is still short after `rounds` stops outright.

  So no moving fleet can come within `min_gap` (1 hop) of another moving fleet,
  whatever the policy, the ladder or exploration chose. The policy decides how
  efficiently the fleet moves; the guard decides that it is safe. The Simplex
  pattern (build log entry 23), and what a hardware safety field does on a real
  vehicle.

WHAT IT DOES NOT DO
  * It never slows a fleet whose own move opens (or keeps) the gap: a fleet
    backing away is never held back, so a pair already too close can always
    separate.
  * A pair already closer than min_gap at the start of the step (a re-entry, a
    recovery move) only has to not get closer.
  * Immobile fleets (parked, stopped) are not threats, exactly as in
    check_integrity: a rescuer must be able to reach a dead fleet's cell.
  * People are not in here: the action mask's hard veto already stops any move
    into a human (node_warehouse.get_valid_action_mask).

UNITS: HOPS, ALWAYS
  Distances are graph hops between continuous positions, measured the same way
  as GraphProximity (proximity_warehouse.py): each position resolves to one or
  two anchor nodes with a residual in cells; distance is the minimum of
  residual + hops + residual, except on the interior of one shared edge, where
  the direct gap along the edge is exact. Landing positions come from
  node_warehouse.reach_position(), the computation the move itself uses, so the
  guard judges the move that will actually happen (stop on the next node, turn
  snap, refused moves stay put).

Pure numpy / networkx: test_intent_guard.py runs it without torch.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import node_warehouse as _nw

ON_NODE_TOL = 1e-6


# ----------------------------------------------------------------------------- landing
def landing_position(node: Any, action: int, speed: float) -> np.ndarray:
    """Where apply_discrete_action(action) at `speed` would leave `node`,
    without moving it: reach_position(), then the same validity test the move
    applies (an invalid move leaves the fleet where it is)."""
    here = np.asarray(node.current_pos, dtype=np.float64)
    if int(action) == 0 or float(speed) <= 0.0:
        return here.copy()
    pos = _nw.reach_position(node, int(action), float(speed))
    if not node.is_structurally_valid(pos, int(action)):
        return here.copy()
    return np.asarray(pos, dtype=np.float64)


# ----------------------------------------------------------------------------- hops
class HopMetric:
    """Graph distance in hops between two continuous positions. Node-to-node
    hops come from a BFS cut off at `cutoff` and cached (the graph is static)."""

    def __init__(self, graph: Any, grid_pos_dict: Dict[Tuple[int, int, int], str],
                 cutoff: float = 6.0):
        self.G = graph
        self.grid = grid_pos_dict
        self.cutoff = int(np.ceil(float(cutoff)))
        self._bfs: Dict[str, Dict[str, int]] = {}

    def anchors(self, pos: np.ndarray) -> Optional[List[Tuple[str, float]]]:
        """[(node_id, residual)], as GraphProximity._anchor."""
        p = np.asarray(pos, dtype=np.float64)
        snapped = np.round(p).astype(int)
        resid = p - snapped
        axis = int(np.argmax(np.abs(resid)))
        r = float(resid[axis])
        leak = float(np.sum(np.abs(resid))) - abs(r)
        base = self.grid.get((int(snapped[0]), int(snapped[1]), int(snapped[2])))
        if abs(r) <= ON_NODE_TOL:
            return [(base, leak)] if base is not None else None
        far_c = snapped.copy()
        far_c[axis] += 1 if r > 0 else -1
        far = self.grid.get((int(far_c[0]), int(far_c[1]), int(far_c[2])))
        out: List[Tuple[str, float]] = []
        if base is not None:
            out.append((base, abs(r) + leak))
        if far is not None:
            out.append((far, (1.0 - abs(r)) + leak))
        return out or None

    def hops(self, a: str, b: str) -> float:
        if a == b:
            return 0.0
        d = self._bfs.get(a)
        if d is None:
            d = {a: 0}
            q = deque([a])
            while q:
                u = q.popleft()
                if d[u] >= self.cutoff:
                    continue
                for v in self.G.neighbors(u):
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            self._bfs[a] = d
        return float(d.get(b, np.inf))

    def distance(self, pa: np.ndarray, pb: np.ndarray) -> float:
        A, B = self.anchors(pa), self.anchors(pb)
        if A is None or B is None:              # off the graph: as GraphProximity's fallback
            return float(np.sum(np.abs(np.asarray(pa, float) - np.asarray(pb, float))))
        na, nb = {n for n, _ in A}, {n for n, _ in B}
        if len(na) == 2 and na == nb:           # interior of one shared edge: exact
            return float(np.sum(np.abs(np.asarray(pa, float) - np.asarray(pb, float))))
        best = np.inf
        for x, rx in A:
            for y, ry in B:
                h = self.hops(x, y)
                if np.isfinite(h):
                    best = min(best, rx + h + ry)
        return float(best)


def crossed(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray,
            tol: float = 0.1 + 1e-6) -> bool:
    """True if two fleets on one aisle line swap order during the step: one
    drove THROUGH the other. Distance alone misses this -- a fleet overtaking
    another that is 0.25 ahead, or two overlapping fleets swapping, can land
    as far apart as they started. Only fleets on the same line (the other two
    coordinates within the turn tolerance, at both ends) can pass through
    each other; around a corner they meet at the junction node, which distance
    catches."""
    for k in range(3):
        o = [j for j in range(3) if j != k]
        if all(abs(a0[j] - b0[j]) <= tol and abs(a1[j] - b1[j]) <= tol for j in o):
            if (b0[k] - a0[k]) * (b1[k] - a1[k]) < -1e-12:
                return True
    return False


# ----------------------------------------------------------------------------- guard
def guard_speeds(movers: Dict[str, Tuple[Any, int, float]],
                 pairs: Iterable[Tuple[str, str]],
                 metric: HopMetric,
                 min_gap: float = 1.0,
                 rounds: int = 8,
                 bisect_iters: int = 14,
                 eps: float = 1e-6) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    movers: fleet id -> (node, action, speed) for every MOBILE fleet this step.
    pairs:  candidate pairs (both ids in movers), e.g. proximity.pairs() within
            min_gap + the most two fleets can close in one step.
    Returns (speed per fleet id, stats). A speed is only ever lowered.
    """
    speed = {i: max(0.0, float(v[2])) for i, v in movers.items()}
    pos0 = {i: np.asarray(v[0].current_pos, dtype=np.float64) for i, v in movers.items()}
    # CANONICAL ORDER. Pairs are slowed one after another, so the order could
    # matter; sorting by fleet id makes the result independent of the order of
    # self.nodes, as the simultaneous step requires (SIMULTANEOUS_STEP.md).
    plist = sorted({(min(str(a), str(b)), max(str(a), str(b)))
                    for a, b in pairs if a in movers and b in movers and a != b})
    if movers and not isinstance(next(iter(movers)), str):
        back = {str(i): i for i in movers}
        plist = [(back[a], back[b]) for a, b in plist]
    d0 = {(a, b): metric.distance(pos0[a], pos0[b]) for a, b in plist}
    need = {k: min(float(min_gap), v) - eps for k, v in d0.items()}

    def land(i: str, s: float) -> np.ndarray:
        node, act, _ = movers[i]
        return landing_position(node, act, s)

    def gap(a: str, b: str, sa: float, sb: float) -> float:
        la, lb = land(a, sa), land(b, sb)
        if crossed(pos0[a], la, pos0[b], lb):
            return 0.0                          # drove through each other
        return metric.distance(la, lb)

    def closes(i: str, j: str, si: float) -> bool:
        """Does fleet i's own move (j standing still) shrink the gap, or take
        it through j?"""
        li = land(i, si)
        return (d0[(a_b(i, j))] - metric.distance(li, pos0[j]) > eps
                or crossed(pos0[i], li, pos0[j], pos0[j]))

    def a_b(i: str, j: str) -> Tuple[str, str]:
        return (i, j) if (i, j) in d0 else (j, i)

    stats = {"pairs_checked": len(plist), "pairs_short": 0, "fleets_slowed": 0,
             "fleets_stopped": 0, "rounds": 0, "fallback_stops": 0}
    short_once = set()
    for r in range(int(rounds)):
        changed = False
        for a, b in plist:
            sa, sb = speed[a], speed[b]
            if gap(a, b, sa, sb) >= need[(a, b)]:
                continue
            short_once.add((a, b))
            # Who is closing? A fleet whose own move (the other standing still)
            # does not shrink the gap is never slowed: it is opening the gap.
            ca = closes(a, b, sa)
            cb = closes(b, a, sb)
            if not (ca or cb):
                ca = cb = True                  # only jointly closing: slow both

            def ok(f: float, ca=ca, cb=cb) -> bool:
                return gap(a, b, sa * (f if ca else 1.0), sb * (f if cb else 1.0)) >= need[(a, b)]

            if not ok(0.0):                     # the closers stopped is not enough:
                ca = cb = True                  # slow both (both stopped always works)
            lo, hi = 0.0, 1.0
            for _ in range(int(bisect_iters)):
                mid = 0.5 * (lo + hi)
                if ok(mid, ca, cb):
                    lo = mid
                else:
                    hi = mid
            if ca:
                speed[a] = sa * lo
            if cb:
                speed[b] = sb * lo
            changed = True
        stats["rounds"] = r + 1
        if not changed:
            break
    # Anything still short after the rounds stops outright (always safe).
    for a, b in plist:
        if gap(a, b, speed[a], speed[b]) < need[(a, b)]:
            for i in (a, b):
                if speed[i] > 0.0:
                    speed[i] = 0.0
                    stats["fallback_stops"] += 1
    stats["pairs_short"] = len(short_once)
    for i, v in movers.items():
        if speed[i] < float(v[2]) - 1e-12:
            stats["fleets_slowed"] += 1
            if speed[i] <= 1e-9 and int(v[1]) != 0 and float(v[2]) > 0.0:
                stats["fleets_stopped"] += 1
    return speed, stats


def candidate_radius(min_gap: float, max_speed: float) -> float:
    """Pairs farther apart than this cannot come within min_gap in one step:
    each fleet moves at most max_speed cells."""
    return float(min_gap) + 2.0 * float(max_speed) + 0.25
