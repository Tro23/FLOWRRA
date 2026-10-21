"""
proximity_warehouse.py

NEW MODULE (2026-09-13). Graph-distance peer proximity, replacing Manhattan
distance on raw coordinates at the four sites that measure "how close is the
nearest fleet that could actually hit me".

WHAT THIS REPLACES
------------------
Four independent copies of the same computation, all of them Manhattan on
coordinates:

  core_warehouse.py:~1265   sf_peer_proximity   (situation feature, state vector)
  core_warehouse.py:~1570   braking min_dist    (speed throttle)
  core_warehouse.py:~1841   safety reward       (reward_warning_zone scaling)
  loop_warehouse.py         check_integrity     (fatal / warning classification)

WHY IT WAS WRONG
----------------
`np.sum(np.abs(pos_a - pos_b))` measures straight-line grid distance through
racks. Two fleets in adjacent aisles separated by a solid shelf are Manhattan
distance 2 apart and graph distance 20+ apart: they cannot reach each other,
cannot collide, and have no business braking each other or scoring a warning.
This is the "phantom pair" failure. density_warehouse.py already fixed its own
version of this in the August rewrite (`_structure_mask` runs a depth-L BFS
through the graph) -- so the DENSITY FIELD the network sees has been
graph-aware for months while the BRAKING AND COLLISION LOGIC that acts on the
same geometry stayed Manhattan. This module closes that gap.

EXPECTED EFFECT SIZE: SMALL. The earlier diagnostic measured phantom_pct at
0.0 on the map that was failing and low on the other two. This change is made
because the metric should be the right one, not because it is expected to
unlock performance. If collision counts move a LOT after this lands, the
phantom measurement was wrong and that is itself the finding.

HOW DISTANCE IS DEFINED HERE
----------------------------
Fleets sit at continuous coordinates (base_speed 0.5, and braking produces
arbitrary fractions), so they are usually BETWEEN two graph nodes. Each fleet
is resolved to one or two "anchors": (node_id, residual), where residual is the
distance in cells from that node to the fleet.

  on a node          -> [(n, 0.0)]
  on edge u--v at t  -> [(u, t), (v, 1-t)]

Distance between fleets A and B is then

  min over a in anchors(A), b in anchors(B) of  resid_a + hops(a, b) + resid_b

with `hops` the true graph distance. This is exact EXCEPT when both fleets sit
on the same edge, where the anchor form over-estimates: two fleets 0.4 apart on
edge (10)--(11) would score min(0.3+0+0.7, 0.7+0+0.3) = 1.0 instead of 0.4, and
a genuine collision would go unreported. That case is special-cased below and
measured directly. See test_proximity.py::test_same_edge_pair.

COST
----
Cheaper than what it replaces. The old sf_peer_proximity and safety-reward
sites were each an O(N^2) pure-Python generator expression (40,000 iterations
per step at 200 fleets, twice per step), and check_integrity was an O(N^2)
pairwise numpy loop. This runs one bounded Dijkstra per fleet out to
`search_radius` cells, which on a degree-~2.27 corridor graph settles a few
dozen nodes: O(N * d) with a small d.

REPRODUCIBILITY
---------------
metric="manhattan" reproduces the previous behaviour bit-for-bit (it calls the
same np.sum(np.abs(...)) with the same exclusion set). Every published FLOWRRA
number was measured under "manhattan". The flag exists so those runs can be
re-created after this module lands; it is not a fallback for when "graph"
misbehaves.
"""

from __future__ import annotations

import heapq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# A fleet is treated as sitting exactly ON a node when its offset from the
# nearest integer is below this. is_structurally_valid() in node_warehouse.py
# uses 0.1 as its own on-axis tolerance, so anything looser than that would
# disagree with the movement engine about where a fleet is.
ON_NODE_TOL = 1e-6


class GraphProximity:
    """
    Per-step index of fleet-to-fleet distances measured THROUGH the graph.

    Usage, once per simulation step, before anything reads a peer distance:

        prox.refresh(self.nodes, excluded_ids=self.immobile_nodes)
        d = prox.nearest(node.id)               # float, inf if nothing in range
        for a, b, dist in prox.pairs():         # deduped, dist <= search_radius
            ...

    `excluded_ids` are fleets that cannot participate as a THREAT: parked and
    stopped fleets. They still get their own `nearest()` computed (the
    situation-feature site needs a value for every fleet, including immobile
    ones) but they never appear as somebody else's nearest peer. This mirrors
    the exclusion the four replaced call sites already applied via
    `self.immobile_nodes`.
    """

    def __init__(
        self,
        graph: Any,
        grid_pos_dict: Dict[Tuple[int, int, int], str],
        search_radius: float = 4.0,
        metric: str = "graph",
    ):
        self.graph = graph
        self.grid_pos_dict = grid_pos_dict
        self.search_radius = float(search_radius)
        if metric not in ("graph", "manhattan"):
            raise ValueError(
                f"metric must be 'graph' or 'manhattan', got {metric!r}. "
                f"'manhattan' reproduces pre-2026-09-13 behaviour exactly."
            )
        self.metric = metric

        # Populated by refresh().
        self._anchors: Dict[str, List[Tuple[str, float]]] = {}
        self._nodeset: Dict[str, frozenset] = {}
        self._pos: Dict[str, np.ndarray] = {}
        self._occupancy: Dict[str, List[Tuple[str, float]]] = {}
        self._within: Dict[str, Dict[str, float]] = {}
        self._ids: List[str] = []
        # Fleets that cannot act as a THREAT. They still get their own reading
        # (the situation-feature site needs one for every fleet) which is why
        # pairs() has to filter on this rather than trusting _within.
        self._excluded: Set[str] = set()

        # Diagnostics. off_grid_fleets counts fleets whose rounded position is
        # not a graph node at all (possible mid-move on a malformed edge); those
        # fall back to Manhattan for that step rather than reporting "no peers",
        # which would silently disable braking.
        self.off_grid_fleets = 0
        self.last_refresh_settled = 0

    # ------------------------------------------------------------------
    # Anchor resolution
    # ------------------------------------------------------------------
    def _anchor(self, pos: np.ndarray) -> Optional[List[Tuple[str, float]]]:
        """
        Resolve a continuous position to [(node_id, residual_in_cells), ...].

        Returns None if the position cannot be placed on the graph at all.

        A fleet only ever has ONE meaningfully fractional axis: is_structurally_valid()
        refuses a move whose two off-axis coordinates are not within 0.1 of an
        integer, so a fleet cannot turn off a corridor until it is essentially on
        a node. The residual carried by the other two axes is therefore <= 0.1 in
        the worst case and is folded into `leak` rather than ignored.
        """
        snapped = np.round(pos).astype(int)
        resid = np.asarray(pos, dtype=np.float64) - snapped
        axis = int(np.argmax(np.abs(resid)))
        r = float(resid[axis])
        leak = float(np.sum(np.abs(resid))) - abs(r)

        base = self.grid_pos_dict.get((int(snapped[0]), int(snapped[1]), int(snapped[2])))

        if abs(r) <= ON_NODE_TOL:
            return [(base, leak)] if base is not None else None

        far_coords = snapped.copy()
        far_coords[axis] += 1 if r > 0 else -1
        far = self.grid_pos_dict.get(
            (int(far_coords[0]), int(far_coords[1]), int(far_coords[2]))
        )

        out: List[Tuple[str, float]] = []
        if base is not None:
            out.append((base, abs(r) + leak))
        if far is not None:
            out.append((far, (1.0 - abs(r)) + leak))
        return out or None

    # ------------------------------------------------------------------
    # Per-step build
    # ------------------------------------------------------------------
    def refresh(self, nodes: Sequence[Any], excluded_ids: Optional[Set[str]] = None) -> None:
        """Rebuild the index. Call exactly once per simulation step."""
        excluded = set(excluded_ids or ())
        self._excluded = excluded
        self._anchors.clear()
        self._nodeset.clear()
        self._pos.clear()
        self._occupancy.clear()
        self._within.clear()
        self._ids = [n.id for n in nodes]
        self.off_grid_fleets = 0
        self.last_refresh_settled = 0

        if self.metric == "manhattan":
            self._refresh_manhattan(nodes, excluded)
            return

        fallback: List[Any] = []
        for n in nodes:
            pos = np.asarray(n.current_pos, dtype=np.float64)
            self._pos[n.id] = pos
            anchors = self._anchor(pos)
            if anchors is None:
                self.off_grid_fleets += 1
                fallback.append(n)
                continue
            self._anchors[n.id] = anchors
            self._nodeset[n.id] = frozenset(a for a, _ in anchors)
            if n.id not in excluded:
                for nid, resid in anchors:
                    self._occupancy.setdefault(nid, []).append((n.id, resid))

        for n in nodes:
            self._within[n.id] = self._search(n.id, excluded)

        # Fleets that could not be placed on the graph keep working, on the old
        # metric, rather than silently reporting an empty neighbourhood.
        for n in fallback:
            d: Dict[str, float] = {}
            for o in nodes:
                if o.id == n.id or o.id in excluded:
                    continue
                dist = float(np.sum(np.abs(self._pos[n.id] - self._pos[o.id])))
                if dist <= self.search_radius:
                    d[o.id] = dist
            self._within[n.id] = d

    def _refresh_manhattan(self, nodes: Sequence[Any], excluded: Set[str]) -> None:
        """Bit-for-bit reproduction of the pre-2026-09-13 metric."""
        for n in nodes:
            self._pos[n.id] = np.asarray(n.current_pos, dtype=np.float64)
        for n in nodes:
            d: Dict[str, float] = {}
            for o in nodes:
                if o.id == n.id or o.id in excluded:
                    continue
                dist = float(np.sum(np.abs(self._pos[n.id] - self._pos[o.id])))
                if dist <= self.search_radius:
                    d[o.id] = dist
            self._within[n.id] = d

    def _search(self, fid: str, excluded: Set[str]) -> Dict[str, float]:
        """
        Bounded multi-source Dijkstra from one fleet's anchors.

        Every edge in this warehouse graph has weight 1 (see
        density_warehouse._structure_mask), but the SOURCES start at fractional
        residuals, so a plain BFS queue would settle nodes out of order. heapq
        keeps it exact for the cost of a log factor on a handful of nodes.
        """
        anchors = self._anchors.get(fid)
        if not anchors:
            return {}

        own_nodes = self._nodeset.get(fid, frozenset())
        found: Dict[str, float] = {}

        # SAME-EDGE CASE. The anchor-min formula over-estimates when both fleets
        # occupy the interior of the same edge: it must route out to an endpoint
        # and back. Measured directly instead. This is the only case where the
        # general formula is not exact, and it is precisely the case where two
        # fleets are closest -- so getting it wrong would hide real collisions.
        if len(own_nodes) == 2:
            for nid in own_nodes:
                for peer, _ in self._occupancy.get(nid, ()):
                    if peer == fid or peer in excluded:
                        continue
                    if self._nodeset.get(peer) == own_nodes:
                        found[peer] = float(
                            np.sum(np.abs(self._pos[fid] - self._pos[peer]))
                        )

        dist: Dict[str, float] = {}
        heap: List[Tuple[float, str]] = []
        for nid, resid in anchors:
            if resid < dist.get(nid, float("inf")):
                dist[nid] = resid
                heapq.heappush(heap, (resid, nid))

        settled = 0
        while heap:
            d, nid = heapq.heappop(heap)
            if d > dist.get(nid, float("inf")) + 1e-12:
                continue
            if d > self.search_radius:
                break
            settled += 1

            for peer, resid in self._occupancy.get(nid, ()):
                if peer == fid or peer in excluded:
                    continue
                cand = d + resid
                if cand <= self.search_radius and cand < found.get(peer, float("inf")):
                    found[peer] = cand

            if d + 1.0 > self.search_radius:
                continue
            for nb in self.graph.neighbors(nid):
                nd = d + 1.0
                if nd < dist.get(nb, float("inf")):
                    dist[nb] = nd
                    heapq.heappush(heap, (nd, nb))

        self.last_refresh_settled += settled
        return found

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def nearest(self, fid: str) -> float:
        """
        Graph distance to the nearest non-excluded peer, or inf if none is
        within search_radius.

        CALLER NOTE: the replaced code returned the TRUE distance however large,
        or float('inf') when every other fleet was excluded. Every decision site
        clamps at warning_threshold, so truncating at search_radius is
        behaviourally identical for all of them -- but NOT for the mean_peer_gap
        METRIC, which summed the raw value. See nearest_censored().
        """
        d = self._within.get(fid)
        return min(d.values()) if d else float("inf")

    def nearest_censored(self, fid: str) -> float:
        """
        nearest(), with "nothing in range" reported as search_radius rather than
        inf, for accumulating mean_peer_gap.

        BUG THIS FIXES (pre-existing, not introduced here): core_warehouse.py set
        `min_dist = float('inf')` when every other fleet was immobile -- common
        in the last third of an episode -- and then did
        `self._min_dist_sum += float(min_dist)`. One such step poisons
        mean_peer_gap to inf for the rest of the run. Reported mean_peer_gap is
        therefore a mean CENSORED at search_radius, and that has to be stated
        wherever the number is published.
        """
        d = self.nearest(fid)
        return self.search_radius if d == float("inf") else min(d, self.search_radius)

    def peers_within(self, fid: str, radius: float) -> List[Tuple[str, float]]:
        """All non-excluded peers within `radius` cells, nearest first."""
        d = self._within.get(fid, {})
        out = [(p, v) for p, v in d.items() if v <= radius]
        out.sort(key=lambda x: x[1])
        return out

    def pairs(self, radius: Optional[float] = None) -> List[Tuple[str, str, float]]:
        """
        Every unordered pair of non-excluded fleets within `radius`, deduped.

        Used by loop_warehouse.check_integrity, which previously ran a full
        O(N^2) pairwise sweep -- 20,000 numpy calls per step at 200 fleets, of
        which essentially all were of pairs nowhere near each other.
        """
        r = self.search_radius if radius is None else float(radius)
        seen: Dict[Tuple[str, str], float] = {}
        for a, peers in self._within.items():
            # BUG THIS FIXES (caught by test_excluded_peer_invisible): excluded
            # fleets still have their own _within populated, so iterating it
            # unfiltered emitted pairs in which one side was parked or stopped.
            # check_integrity() has exempted those since the "IF EITHER FLEET IS
            # PARKED, IT CEASES TO EXIST FOR COLLISIONS" fix, and re-introducing
            # them here would have resurrected the corpse-collision failure mode.
            if a in self._excluded:
                continue
            for b, d in peers.items():
                if d > r or b in self._excluded:
                    continue
                key = (a, b) if a < b else (b, a)
                prev = seen.get(key)
                if prev is None or d < prev:
                    seen[key] = d
        return [(a, b, d) for (a, b), d in seen.items()]

    def describe(self) -> str:
        return (
            f"[Proximity] metric={self.metric} search_radius={self.search_radius} "
            f"fleets={len(self._ids)} off_grid={self.off_grid_fleets} "
            f"settled={self.last_refresh_settled}"
        )