"""
conflict_warehouse.py -- CONFLICT_DESIGN.md component 1: path-based warnings.

WHY. The warning zone was a distance: any two fleets within 2.0 hops warned,
whatever they were doing. On cold_run24 57% of warnings were convoys sitting 1.5-2
hops apart on the same route -- no conflict at all -- while a head-on pair three
hops apart in a single-lane shaft, which WILL meet, raised nothing until it was
too late to step aside. Warning on distance flags the harmless and misses the
dangerous.

WHAT. Every pair within `radius` hops is classified by where the two fleets are
going: their next `route_horizon` cells, from the same goal-map descent the
density field already uses for its projections (ties branch; any branch counts).

    head_on     each route contains the other's cell
    blocked     one route contains the other's cell, and that fleet is stationary
                (held, waiting, stopped, at its goal, or did not move last step)
    following   one route contains the other's cell, and the other moves on
    contested   the routes share a cell reached within one step of each other
    sequential  the routes share a cell, reached further apart than that
    none        no shared cell, neither heads for the other
    unknown     a route could not be computed (no goal map, off the graph)

and the warning follows from the kind, not the distance:

    within `floor` hops              always a warning (routes are predictions)
    head_on, contested, blocked      warning
    following                        warning only closer than `follow_gap`
    sequential, none                 no warning
    unknown                          today's rule: warning within the old
                                     warning_threshold

TRAFFIC STAYS LOCAL (the FLOWRRA principle). Only pairs already within `radius`
hops are looked at, and each fleet's route is its own intent -- what a fleet
would broadcast to its neighbours. Nothing here plans for anyone.

THE SPLAT. A warned head_on / contested / blocked pair gets its density splat on
the cell where the two would actually meet: the contested cell reached soonest
(or, for a blocker, the blocker's own cell). The old splat went to the pair's
midpoint -- often a cell neither fleet was going to enter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import numpy as np

Cell = Tuple[int, int, int]
KINDS = ("head_on", "blocked", "following", "contested", "sequential", "none", "unknown")
MEETING_KINDS = ("head_on", "contested", "blocked")


@dataclass
class Verdict:
    a: str
    b: str
    dist: float
    kind: str
    warn: bool
    why: str                                  # "floor", "kind", "gap", "fallback", ""
    meet: Optional[Cell] = None               # where to splat, for MEETING_KINDS
    follower: Optional[str] = None            # for "following": who is behind


@dataclass
class ConflictSettings:
    radius: float = 3.0
    floor: float = 1.0
    follow_gap: float = 1.5
    route_horizon: int = 8
    fallback_threshold: float = 2.0           # today's warning distance, for "unknown"

    @classmethod
    def from_config(cls, cfg: Dict[str, Any], warning_threshold: float) -> "ConflictSettings":
        return cls(radius=float(cfg.get("radius", 3.0)),
                   floor=float(cfg.get("floor", 1.0)),
                   follow_gap=float(cfg.get("follow_gap", 1.5)),
                   route_horizon=int(cfg.get("route_horizon", 8)),
                   fallback_threshold=float(warning_threshold))


class PathConflicts:
    """
    Classifies fleet pairs by route. Stateless apart from its counters: call
    evaluate() whenever a warning set is needed (start of step, after the
    action), with the proximity index already refreshed for that moment.
    """

    def __init__(self, density: Any, proximity: Any, settings: ConflictSettings):
        self.density = density            # _intended_path(), _coords_by_id
        self.proximity = proximity        # _anchor(), pairs()
        self.s = settings
        self.kind_counts = {k: 0 for k in KINDS}
        self.warn_counts = {k: 0 for k in KINDS}
        self.evaluations = 0

    # ------------------------------------------------------------ geometry
    def occupied(self, fleet: Any) -> Set[Cell]:
        """Cells a fleet occupies: its node, or both ends of the edge it is on."""
        anchors = self.proximity._anchor(np.asarray(fleet.current_pos, dtype=np.float64))
        out: Set[Cell] = set()
        for nid, _res in anchors or []:
            c = self.density._coords_by_id.get(nid)
            if c is not None:
                out.add(tuple(int(v) for v in c))
        if not out:
            out.add(tuple(int(v) for v in np.round(np.asarray(fleet.current_pos))))
        return out

    def route(self, fleet: Any) -> Optional[List[Set[Cell]]]:
        """Next route_horizon cells as one set per step (ties: every branch).
        None when it cannot be computed; [] when the fleet is at its goal."""
        levels = self.density._intended_path(fleet, horizon=self.s.route_horizon)
        if levels is None:
            return None
        return [{tuple(int(v) for v in c) for c, _w in lvl} for lvl in levels]

    # ------------------------------------------------------------ one pair
    @staticmethod
    def _first_hit(route: List[Set[Cell]], cells: Set[Cell]) -> Optional[int]:
        for t, lvl in enumerate(route, start=1):
            if lvl & cells:
                return t
        return None

    def classify(self, ra: Optional[List[Set[Cell]]], rb: Optional[List[Set[Cell]]],
                 oa: Set[Cell], ob: Set[Cell], stat_a: bool, stat_b: bool
                 ) -> Tuple[str, Optional[Cell], Optional[str]]:
        """(kind, meeting cell, follower 'a'/'b'/None) for one pair."""
        if ra is None or rb is None:
            return "unknown", None, None
        ta = self._first_hit(ra, ob)          # a reaches b's cell at step ta
        tb = self._first_hit(rb, oa)
        if ta is not None and tb is not None:
            meet = self._meeting(ra, rb)
            if meet is None:                  # adjacent swap: they meet on the edge
                meet = min(sorted(ob)) if ta <= tb else min(sorted(oa))
            return "head_on", meet, None
        if ta is not None:
            if stat_b:
                return "blocked", min(sorted(ob)), None
            return "following", None, "a"
        if tb is not None:
            if stat_a:
                return "blocked", min(sorted(oa)), None
            return "following", None, "b"
        meet = self._meeting(ra, rb)
        if meet is not None:
            return "contested", meet, None
        if self._shares(ra, rb):
            return "sequential", None, None
        return "none", None, None

    @staticmethod
    def _meeting(ra: List[Set[Cell]], rb: List[Set[Cell]]) -> Optional[Cell]:
        """Soonest cell both routes reach within one step of each other."""
        best = None
        for t1, la in enumerate(ra, start=1):
            for t2 in (t1 - 1, t1, t1 + 1):
                if 1 <= t2 <= len(rb):
                    for c in la & rb[t2 - 1]:
                        key = (max(t1, t2), c)
                        if best is None or key < best:
                            best = key
        return None if best is None else best[1]

    @staticmethod
    def _shares(ra: List[Set[Cell]], rb: List[Set[Cell]]) -> bool:
        ua: Set[Cell] = set().union(*ra) if ra else set()
        ub: Set[Cell] = set().union(*rb) if rb else set()
        return bool(ua & ub)

    def decide(self, kind: str, dist: float) -> Tuple[bool, str]:
        if dist <= self.s.floor:
            return True, "floor"
        if kind in MEETING_KINDS:
            return True, "kind"
        if kind == "following":
            return (dist < self.s.follow_gap), ("gap" if dist < self.s.follow_gap else "")
        if kind == "unknown":
            ok = dist <= self.s.fallback_threshold
            return ok, ("fallback" if ok else "")
        return False, ""

    # ------------------------------------------------------------ all pairs
    def evaluate(self, nodes_by_id: Dict[str, Any], stationary: Set[str],
                 collision_threshold: float) -> List[Verdict]:
        """
        Every non-colliding pair within `radius`, classified and decided. Pairs
        at collision distance are the loop's business and are not returned.
        """
        self.evaluations += 1
        routes: Dict[str, Optional[List[Set[Cell]]]] = {}
        occ: Dict[str, Set[Cell]] = {}

        def r(fid):
            if fid not in routes:
                routes[fid] = self.route(nodes_by_id[fid])
            return routes[fid]

        def o(fid):
            if fid not in occ:
                occ[fid] = self.occupied(nodes_by_id[fid])
            return occ[fid]

        out: List[Verdict] = []
        for a, b, d in sorted(self.proximity.pairs(radius=self.s.radius)):
            if d <= collision_threshold or a not in nodes_by_id or b not in nodes_by_id:
                continue
            ra, rb = r(a), r(b)
            sa = a in stationary or (ra is not None and len(ra) == 0)
            sb = b in stationary or (rb is not None and len(rb) == 0)
            kind, meet, fol = self.classify(ra, rb, o(a), o(b), sa, sb)
            warn, why = self.decide(kind, d)
            self.kind_counts[kind] += 1
            if warn:
                self.warn_counts[kind] += 1
            out.append(Verdict(a, b, d, kind, warn, why, meet,
                               {"a": a, "b": b}.get(fol) if fol else None))
        return out

    def statistics(self) -> Dict[str, Any]:
        out = {"conflict_evaluations": self.evaluations}
        for k in KINDS:
            out[f"conflict_pairs_{k}"] = self.kind_counts[k]
            out[f"conflict_warned_{k}"] = self.warn_counts[k]
        return out
