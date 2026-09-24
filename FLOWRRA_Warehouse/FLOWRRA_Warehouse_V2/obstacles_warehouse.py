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

  humans  move, slowly, along the graph, and wander
  debris  does not move, and persists until cleared

HOW THEY ENTER THE WORLD. The module maintains a set of occupied cells and the
orchestrator reads it each step into `static_obstacles`. From there:

  * FleetNode.get_valid_action_mask() HARD VETOES a move into one.
  * The density field zeroes the structure MASK at that cell and stamps
    repulsion around it -- so an obstacle reads as (mask 0, R > 0) while a wall
    reads as (mask 0, R = 0), distinguishable in two channels.
  * sense_6_axis_rays() terminates a ray on one and sets ray_hit_unknown.

Repulsion alone would be a preference, and a large enough reward can outbid a
preference. The veto is what makes it absolute.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple


class ObstacleField:
    """
    A small population of humans and debris on the warehouse graph.

    Deliberately simple: this exists to exercise the unregistered-obstacle
    channel end to end, not to model pedestrian behaviour. A human takes a
    random step along the graph every `human_move_period` simulation steps,
    which is slower than a fleet and enough to make the cell set non-stationary.
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

        self.humans: List[str] = []     # node ids, they move
        self.debris: List[str] = []     # node ids, they do not
        self.step_count = 0
        self.human_moves = 0

    # ------------------------------------------------------------------
    def reset(self, avoid: Optional[Set[Tuple[int, int, int]]] = None) -> None:
        """
        Place the population. `avoid` holds cells that must stay clear -- fleet
        start positions and goals, so an episode cannot begin with a fleet
        standing inside a human or with an unreachable goal.
        """
        avoid = avoid or set()
        candidates = [nid for cell, nid in self.grid_pos_dict.items()
                      if cell not in avoid]
        self.rng.shuffle(candidates)

        self.humans = []
        self.debris = []
        for cell in self.fixed_cells:
            nid = self.grid_pos_dict.get(tuple(cell))
            if nid is not None:
                self.debris.append(nid)

        take = iter(candidates)
        for _ in range(self.n_humans):
            nid = next(take, None)
            if nid is not None:
                self.humans.append(nid)
        for _ in range(self.n_debris):
            nid = next(take, None)
            if nid is not None:
                self.debris.append(nid)
        self.step_count = 0
        self.human_moves = 0

    # ------------------------------------------------------------------
    def step(self) -> None:
        """Advance humans. Debris never moves."""
        self.step_count += 1
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

    # ------------------------------------------------------------------
    def occupied_cells(self) -> Set[Tuple[int, int, int]]:
        """
        The cell set the orchestrator reads into `static_obstacles`.

        One category, as the policy sees it: a human and a piece of debris are
        the same fact -- something is in that cell and it is not a fleet.
        """
        out: Set[Tuple[int, int, int]] = set()
        for nid in self.humans:
            c = self.coords_by_id.get(nid)
            if c is not None:
                out.add(c)
        for nid in self.debris:
            c = self.coords_by_id.get(nid)
            if c is not None:
                out.add(c)
        return out

    def statistics(self) -> Dict[str, Any]:
        return {
            "humans": len(self.humans),
            "debris": len(self.debris),
            "obstacle_cells": len(self.occupied_cells()),
            "human_moves": self.human_moves,
        }


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
    )