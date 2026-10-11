"""
obstacles_warehouse.py

Unregistered obstacles: humans and debris.

WHY A SEPARATE MODULE. These are categorically unlike every other obstacle in
the system. A fleet is in the registry, so its position is reported over VDA
5050, its goal is known, and the near-goal discount can make it transparent to
whoever needs to reach it. An obstacle is something NOBODY TOLD THE
ORCHESTRATOR ABOUT. It has no id, no goal, no velocity report, and no
transparency rule -- nobody's goal is ever a person.

COSTS NO STATE DIMENSIONS. The agent already has ray_hit_unknown (6 dims), added
in Phase 2 and reserved for exactly this. Obstacles are perceived the only way
an unregistered thing can be: by RAY CAST. Position reports cover the registry,
the graph covers static structure, and neither knows about anything nobody
registered. So this module feeds the world, not the state vector.

HUMANS AND DEBRIS ARE ONE CATEGORY TO THE POLICY. They differ in how they arrive
and how long they last, not in what a fleet should do about them. The module
tracks the distinction for spawning and logging; the state vector does not.

HOW THEY ENTER THE WORLD. The module maintains a set of occupied cells and the
orchestrator reads it each step into `static_obstacles`. From there:

  * FleetNode.get_valid_action_mask() HARD VETOES a move into one.
  * The density field zeroes the structure MASK at that cell and stamps
    repulsion around it -- so an obstacle reads as (mask 0, R > 0) while a wall
    reads as (mask 0, R = 0), distinguishable in two channels.
  * sense_6_axis_rays() terminates a ray on one and sets ray_hit_unknown.

Repulsion alone would be a preference, and a large enough reward can outbid a
preference. The veto is what makes it absolute.

TWO MODES (CONFIG["obstacles"]["mode"])

  "wander" (default; the original behaviour, unchanged)
      n_humans wander at random, n_debris never move, fixed_cells are debris
      placed by hand. Humans here do NOT avoid fleets.

  "cleanup" (added 2026-10-11, for stream mode)
      Debris APPEARS during the episode (a dropped tote, a spill), and a human
      comes to clear it:
        1. debris appears at a random free cell: never on a fleet, never on a
           cell the orchestrator asked to keep clear (fleet starts, goals,
           docks and their neighbours), at most max_debris at once, on average
           once every debris_interval steps;
        2. after response_delay steps a human enters at a random free cell on
           the EDGE of the debris's floor and walks a shortest path to it;
        3. standing next to it, they clean for cleanup_steps, and it is gone;
        4. they walk a shortest path to another random edge cell and leave.
      A human NEVER steps into a fleet's cell: blocked, they wait, and after
      human_patience blocked attempts they look for a way around (shortest
      path avoiding fleets and obstacles). With no way round (a dead end, a
      fleet that will not move), after human_give_way blocked attempts they
      step aside to any free neighbour and try again from there. Nothing is
      permanent: debris nobody could reach is cleared after max_debris_life
      steps, and a human boxed in for max_human_life steps leaves.

      Humans have absolute priority. The fleets need no new rule for that: the
      veto keeps them out of a human's cell, the density repulsion keeps them
      a little away, and the ladder (fleet-to-fleet only) never asks a human
      to move.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx


class ObstacleField:
    """
    A small population of humans and debris on the warehouse graph.

    Deliberately simple: this exists to exercise the unregistered-obstacle
    channel end to end, not to model pedestrian behaviour.
    """

    def __init__(
        self,
        graph: Any,
        grid_pos_dict: Dict[Tuple[int, int, int], str],
        n_humans: int = 0,
        n_debris: int = 0,
        human_move_period: int = 4,
        fixed_cells: Optional[List[Tuple[int, int, int]]] = None,
        seed: Optional[int] = None,
        mode: str = "wander",
        debris_interval: int = 200,
        max_debris: int = 2,
        response_delay: int = 10,
        cleanup_steps: int = 5,
        human_patience: int = 2,
        max_debris_life: int = 1000,
        human_give_way: int = 6,
        max_human_life: int = 2000,
    ):
        self.graph = graph
        self.grid_pos_dict = grid_pos_dict
        self.coords_by_id = {v: k for k, v in grid_pos_dict.items()}
        self.n_humans = int(n_humans)
        self.n_debris = int(n_debris)
        self.human_move_period = max(1, int(human_move_period))
        # Cells placed by hand rather than sampled. Used to construct a specific
        # situation -- the entropy test needs an obstacle at a KNOWN cell.
        self.fixed_cells = list(fixed_cells or [])
        self.rng = random.Random(seed)

        self.mode = str(mode or "wander")
        if self.mode not in ("wander", "cleanup"):
            raise ValueError(f"obstacles.mode must be 'wander' or 'cleanup', not {mode!r}")
        self.debris_interval = max(1, int(debris_interval))
        self.max_debris = max(0, int(max_debris))
        self.response_delay = max(0, int(response_delay))
        self.cleanup_steps = max(0, int(cleanup_steps))
        self.human_patience = max(1, int(human_patience))
        self.max_debris_life = max(1, int(max_debris_life))
        self.human_give_way = max(self.human_patience, int(human_give_way))
        self.max_human_life = max(1, int(max_human_life))

        self.humans: List[str] = []     # wander mode: node ids, they move
        self.debris: List[str] = []     # wander mode: node ids, they do not
        self.step_count = 0
        self.human_moves = 0

        # cleanup mode
        self._sorted_ids: List[str] = sorted(self.coords_by_id)
        self._edge_by_floor: Dict[Any, List[str]] = {}
        self._floor_of: Dict[str, Any] = {}
        if self.mode == "cleanup":
            self._index_floors()
        self.no_spawn: Set[str] = set()
        self.c_debris: Dict[str, Dict[str, Any]] = {}     # cell -> record
        self.c_humans: List[Dict[str, Any]] = []
        self._stats = self._zero_stats()

    @staticmethod
    def _zero_stats() -> Dict[str, float]:
        return {"debris_spawned": 0, "debris_cleaned": 0, "debris_expired": 0,
                "debris_life_sum": 0, "humans_entered": 0, "humans_exited": 0,
                "human_wait_steps": 0, "human_reroutes": 0, "human_give_ways": 0,
                "humans_timed_out": 0, "near_miss_steps": 0}

    # ------------------------------------------------------------------ floors
    def _index_floors(self) -> None:
        """Floors and their edge cells. The vertical axis is the one the fewest
        edges run along (lifts are rare); a floor is one value on it. A floor's
        edge is every cell at the min or max of either horizontal axis."""
        coords = self.coords_by_id
        counts = [0, 0, 0]
        for a, b in self.graph.edges():
            ca, cb = coords.get(a), coords.get(b)
            if ca is None or cb is None:
                continue
            for ax in range(3):
                if ca[ax] != cb[ax]:
                    counts[ax] += 1
        # As stream mode does it (core_warehouse._init_stream_exits): the axis
        # with the fewest edges. On a single-floor map that is the flat one
        # (no edges at all), so the whole map is one floor.
        va = min(range(3), key=lambda ax: (counts[ax], ax))
        h = [ax for ax in range(3) if ax != va]
        floors: Dict[Any, List[str]] = {}
        for nid in self._sorted_ids:
            c = coords[nid]
            floors.setdefault(c[va], []).append(nid)
            self._floor_of[nid] = c[va]
        for f, ids in floors.items():
            lo = [min(coords[i][ax] for i in ids) for ax in h]
            hi = [max(coords[i][ax] for i in ids) for ax in h]
            edge = [i for i in ids
                    if any(coords[i][ax] in (lo[k], hi[k]) for k, ax in enumerate(h))]
            self._edge_by_floor[f] = edge or list(ids)

    # ------------------------------------------------------------------ reset
    def reset(self, avoid: Optional[Set[Tuple[int, int, int]]] = None) -> None:
        """
        Place the population. `avoid` holds cells that must stay clear -- fleet
        start positions and goals, so an episode cannot begin with a fleet
        standing inside a human or with an unreachable goal. In cleanup mode it
        is also where debris may never appear.
        """
        avoid = avoid or set()
        self.step_count = 0
        self.human_moves = 0
        self.humans = []
        self.debris = []
        self.c_debris = {}
        self.c_humans = []
        self._stats = self._zero_stats()
        self.no_spawn = {self.grid_pos_dict[c] for c in avoid if c in self.grid_pos_dict}

        candidates = [nid for cell, nid in self.grid_pos_dict.items()
                      if cell not in avoid]
        self.rng.shuffle(candidates)

        for cell in self.fixed_cells:
            nid = self.grid_pos_dict.get(tuple(cell))
            if nid is not None:
                self.debris.append(nid)
        if self.mode == "cleanup":
            return                      # nothing is on the floor at the start

        take = iter(candidates)
        for _ in range(self.n_humans):
            nid = next(take, None)
            if nid is not None:
                self.humans.append(nid)
        for _ in range(self.n_debris):
            nid = next(take, None)
            if nid is not None:
                self.debris.append(nid)

    # ------------------------------------------------------------------ step
    def step(self, fleet_cells: Optional[Set[str]] = None) -> None:
        """Advance one simulation step. `fleet_cells`: node ids fleets stand on
        (both cells for a fleet between two); cleanup-mode humans never enter
        one. Wander mode ignores it, as it always did."""
        self.step_count += 1
        if self.mode == "cleanup":
            self._step_cleanup(set(fleet_cells or ()))
            return
        if not self.humans or self.step_count % self.human_move_period:
            return
        occupied = set(self.humans) | set(self.debris)
        for i, nid in enumerate(self.humans):
            nbrs = [n for n in self.graph.neighbors(nid) if n not in occupied]
            if not nbrs:
                continue
            nxt = self.rng.choice(nbrs)
            occupied.discard(nid)
            occupied.add(nxt)
            self.humans[i] = nxt
            self.human_moves += 1

    # ------------------------------------------------------------------ cleanup mode
    def _occupied_ids(self) -> Set[str]:
        out = set(self.debris) | set(self.c_debris)
        out |= {h["cell"] for h in self.c_humans}
        return out

    def _path(self, src: str, dst: str, blocked: Set[str]) -> Optional[List[str]]:
        """Shortest path src -> dst avoiding `blocked` (src and dst themselves
        allowed). The cells AFTER src, so path[0] is the next step."""
        block = set(blocked) - {src, dst}
        try:
            view = nx.restricted_view(self.graph, block, []) if block else self.graph
            p = nx.shortest_path(view, src, dst)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
        return list(p[1:])

    def _step_cleanup(self, fleets: Set[str]) -> None:
        st = self._stats
        occ = self._occupied_ids()

        # 1. Debris appears.
        if (len(self.c_debris) < self.max_debris
                and self.rng.random() < 1.0 / self.debris_interval):
            free = [i for i in self._sorted_ids
                    if i not in fleets and i not in occ and i not in self.no_spawn]
            if free:
                cell = self.rng.choice(free)
                self.c_debris[cell] = {"born": self.step_count, "human": False}
                st["debris_spawned"] += 1
                occ.add(cell)

        # 2. Debris nobody could reach is cleared after max_debris_life.
        for cell in sorted(self.c_debris):
            rec = self.c_debris[cell]
            if self.step_count - rec["born"] >= self.max_debris_life:
                self._clear(cell, expired=True)
        occ = self._occupied_ids()

        # 3. A human is sent to each debris once its response delay has passed.
        for cell in sorted(self.c_debris):
            rec = self.c_debris[cell]
            if rec["human"] or self.step_count - rec["born"] < self.response_delay:
                continue
            edge = [i for i in self._edge_by_floor.get(self._floor_of.get(cell), [])
                    if i not in fleets and i not in occ]
            if not edge:
                continue                            # try again next step
            entry = self.rng.choice(edge)
            path = self._path(entry, cell, fleets | (occ - {cell}))
            if path is None:
                continue
            path = path[:-1]                        # stand NEXT to the debris
            self.c_humans.append({"cell": entry, "entry": entry, "target": cell,
                                  "phase": "to_debris", "path": path, "wait": 0,
                                  "clean_left": self.cleanup_steps, "exit": None,
                                  "born": self.step_count})
            rec["human"] = True
            st["humans_entered"] += 1
            occ.add(entry)

        # 4. Humans act. A human still on the floor after max_human_life steps
        #    (boxed in for good) leaves, so nothing is permanent.
        move_now = self.step_count % self.human_move_period == 0
        leaving: List[Dict[str, Any]] = []
        for h in self.c_humans:
            if self.step_count - h["born"] >= self.max_human_life:
                if h["phase"] != "leaving" and h["target"] in self.c_debris:
                    self.c_debris[h["target"]]["human"] = False   # send someone else
                h["phase"], h["path"] = "leaving", []
                st["humans_timed_out"] += 1
                leaving.append(h)
                continue
            if h["phase"] == "to_debris" and h["target"] not in self.c_debris:
                self._head_out(h, fleets)           # it expired meanwhile
            if h["phase"] == "to_debris" and not h["path"]:
                h["phase"] = "cleaning"
            if h["phase"] == "cleaning":
                h["clean_left"] -= 1
                if h["clean_left"] <= 0:
                    self._clear(h["target"], expired=False)
                    self._head_out(h, fleets)
                continue
            if h["phase"] == "leaving" and not h["path"]:
                leaving.append(h)
                continue
            if not move_now or not h["path"]:
                continue
            occ = self._occupied_ids()
            nxt = h["path"][0]
            if nxt in fleets or nxt in occ:
                h["wait"] += 1
                st["human_wait_steps"] += self.human_move_period
                if h["wait"] >= self.human_patience:
                    goal = h["exit"] if h["phase"] == "leaving" else h["target"]
                    alt = self._path(h["cell"], goal, fleets | (occ - {goal}))
                    if alt is not None:
                        if h["phase"] != "leaving":
                            alt = alt[:-1]
                        h["path"] = alt
                        h["wait"] = 0
                        st["human_reroutes"] += 1
                    elif h["wait"] >= self.human_give_way:
                        # No way round (a dead end, a fleet that will not move):
                        # step aside to any free neighbour, as a person would,
                        # and try again from there.
                        side = [n for n in sorted(self.graph.neighbors(h["cell"]))
                                if n not in fleets and n not in occ and n != nxt]
                        if side:
                            h["cell"] = self.rng.choice(side)
                            p = self._path(h["cell"], goal, set())
                            if p is not None:
                                h["path"] = p if h["phase"] == "leaving" else p[:-1]
                            elif h["phase"] == "leaving":
                                h["path"] = []                 # leave from here
                            else:
                                self.c_debris.get(h["target"], {})["human"] = False
                                self._head_out(h, fleets)
                            h["wait"] = 0
                            self.human_moves += 1
                            st["human_give_ways"] += 1
                continue
            h["cell"] = nxt
            h["path"].pop(0)
            h["wait"] = 0
            self.human_moves += 1
        for h in leaving:
            self.c_humans.remove(h)
            st["humans_exited"] += 1

        # 5. Near misses: a fleet on a cell next to a human, per human per step.
        for h in self.c_humans:
            if any(n in fleets for n in self.graph.neighbors(h["cell"])):
                st["near_miss_steps"] += 1

    def _clear(self, cell: str, expired: bool) -> None:
        rec = self.c_debris.pop(cell, None)
        if rec is None:
            return
        self._stats["debris_life_sum"] += self.step_count - rec["born"]
        self._stats["debris_expired" if expired else "debris_cleaned"] += 1

    def _head_out(self, h: Dict[str, Any], fleets: Set[str]) -> None:
        """Walk to another random edge cell of the current floor, and leave."""
        h["phase"] = "leaving"
        h["wait"] = 0
        edge = self._edge_by_floor.get(self._floor_of.get(h["cell"]), [])
        choices = [i for i in edge if i != h["entry"]] or edge or [h["cell"]]
        occ = self._occupied_ids() - {h["cell"]}
        for _ in range(8):
            exit_cell = self.rng.choice(choices)
            h["exit"] = exit_cell
            if exit_cell == h["cell"]:
                h["path"] = []
                return
            p = self._path(h["cell"], exit_cell, fleets | (occ - {exit_cell}))
            if p is not None:
                h["path"] = p
                return
        # Every way out blocked right now: head for the last one tried anyway,
        # through fleets if need be; blocked steps wait, reroute or give way.
        p = self._path(h["cell"], h["exit"], set())
        h["path"] = p if p is not None else []

    # ------------------------------------------------------------------ outputs
    def occupied_cells(self) -> Set[Tuple[int, int, int]]:
        """
        The cell set the orchestrator reads into `static_obstacles`.

        One category, as the policy sees it: a human and a piece of debris are
        the same fact -- something is in that cell and it is not a fleet.
        """
        out: Set[Tuple[int, int, int]] = set()
        for nid in self._occupied_ids() | set(self.humans):
            c = self.coords_by_id.get(nid)
            if c is not None:
                out.add(c)
        return out

    def statistics(self) -> Dict[str, Any]:
        base = {
            "humans": len(self.humans) + len(self.c_humans),
            "debris": len(self.debris) + len(self.c_debris),
            "obstacle_cells": len(self.occupied_cells()),
            "human_moves": self.human_moves,
        }
        if self.mode == "cleanup":
            s = dict(self._stats)
            gone = s["debris_cleaned"] + s["debris_expired"]
            s["debris_life_mean"] = s.pop("debris_life_sum") / gone if gone else 0.0
            base.update(s)
        return base


def from_config(graph, grid_pos_dict, config: Dict[str, Any],
                seed: Optional[int] = None) -> Optional["ObstacleField"]:
    """Build from CONFIG["obstacles"], or None when the feature is off."""
    cfg = (config or {}).get("obstacles", {})
    if not cfg.get("enabled", False):
        return None
    return ObstacleField(
        graph=graph,
        grid_pos_dict=grid_pos_dict,
        n_humans=cfg.get("n_humans", 0),
        n_debris=cfg.get("n_debris", 0),
        human_move_period=cfg.get("human_move_period", 4),
        fixed_cells=cfg.get("fixed_cells", None),
        seed=seed,
        mode=cfg.get("mode", "wander"),
        debris_interval=cfg.get("debris_interval", 200),
        max_debris=cfg.get("max_debris", 2),
        response_delay=cfg.get("response_delay", 10),
        cleanup_steps=cfg.get("cleanup_steps", 5),
        human_patience=cfg.get("human_patience", 2),
        max_debris_life=cfg.get("max_debris_life", 1000),
        human_give_way=cfg.get("human_give_way", 6),
        max_human_life=cfg.get("max_human_life", 2000),
    )