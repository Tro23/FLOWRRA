"""
collapse_warehouse.py -- the collapse rung (Part B1 of JAM_COLLAPSE_DESIGN.md).

WHAT IT DOES. When fleets in path warnings form a cluster (linked within the
warning radius) that lasts `grace_steps` steps, the rung gives the cluster ONE
joint plan for the next `horizon` hops, and its fleets follow it together.
It sits above L1: while a fleet follows a plan, no retrace is ordered for it,
and L1, L2 and L3 stay underneath as the fallback.

THE PLAN (the same rules as the offline study)
    * each tick (one hop) a fleet stays or moves one hop;
    * no two fleets in one cell;
    * no entering a cell another fleet held one tick earlier (no swaps, no
      following nose to tail inside the plan);
    * fleets outside the cluster are clamped to where they may be: at tick k,
      anywhere from their cell to k hops along their route ahead (they may
      brake, so the cells behind them are not freed); failed, waiting and
      parked fleets stay where they are. People and debris are no-go.
    * PEOPLE: a plan keeps a hop from every person and from the next cells of
      their path; when a person comes within 2 hops of a member the plan ends
      and FLOWRRA's own rules take over, and no plan is made there.
    * THE FAR BOUNDARY, live: each fleet's goal-distance map. The cost of a
      plan is every fleet's hops-to-goal added over the ticks -- the area under
      its distance-to-goal line. Waiting costs, backing off costs more, and a
      zigzag costs most.
    A tiny cost per hop moved breaks ties: waiting beats a sideways move that
    gains nothing.
    Up to `max_exact` fleets: exact A* search. Larger (up to `max_fleets`):
    sampled with thrml (jam_thrml.collapse), or left to the ladder (big=ladder).

FOLLOWING IT: LOCKSTEP. Tick k+1 starts only when every member stands on its
tick-k cell; a member that is early waits. So the plan's rules hold while it is
driven, whatever braking does to each fleet's speed. Braking and the intent
guard stay on.

A PLAN ENDS when its last tick is reached (a new one is made if the cluster is
still there), or it is CANCELLED: a member fails, leaves, gets a new goal, or
L3 fires on it; the cluster grows (re-planned with the new fleets at once); or
a tick is not completed within `tick_timeout` steps (blocked).

THE HOLD TEST (solver="hold", 23 October). The seed-1 replay showed most plans
are remade within about 2 steps, as their cluster changes, and that their fleets
mostly stand still meanwhile. So is the gain the joint plan, or just holding a
cluster still and giving it one authority (no retrace orders)? "hold" keeps
everything the rung does -- the same clusters, grace, people rule, one
authority, lockstep -- but the "plan" is to stay put: each member waits on its
cell. No search, no sampling. A fleet is held for at most `horizon` steps in a
row, then its own rules get `retry_after` steps before it can be held again
(without that cap, a cluster that keeps changing would be held for ever).

THE EXACT-PLUS-HOLD TEST (big="hold"): exact search for 2-4 fleets as in the
rung, and hold wherever thrml would be asked (5-8 fleets, and 3-4 that exact
search could not settle). Against the rung, it asks whether sampling the big
clusters adds anything beyond holding them.

INSTALL. Nothing in core_warehouse.py is edited. install(core_warehouse) wraps
four methods; the rung is created only while ENABLED["on"] is True when a
FLOWRRA env is built (collapse_eval.py sets it per run).
"""
from __future__ import annotations

import heapq
import itertools
import time
from collections import deque
from typing import Any, Dict, List, Optional, Set

import numpy as np

DEFAULTS = {
    "horizon": 8,            # ticks (hops) per plan: the horizon path awareness looks ahead
    "grace_steps": 2,        # a cluster must be seen on this many consecutive steps
    "max_exact": 4,          # exact search up to this many fleets
    "max_fleets": 8,         # larger clusters are left to the ladder
    "big": "thrml",          # 5..max_fleets fleets: "thrml", "ladder", or "hold" (the exact-plus-hold test)
    "budget": 20000,         # exact search: states expanded before giving up
    "time_limit": 1.0,       # exact search: seconds before giving up
    "tick_timeout": 24,      # steps allowed to complete one tick (braked to 0.1, a hop takes 10)
    "retry_after": 10,       # a cluster with no plan is tried again after this many steps
    # PEOPLE (found on AWS, 22 October: near misses with people rose on 2 of 4
    # thrml days). A near miss is a fleet on a cell next to a person, counted
    # per person per step, and people wait when a fleet blocks their way -- so
    # a plan that parks fleets beside a person's path piles them up.
    "people_margin": 1,      # plans keep this many hops from a person and from the path ahead of them
    "people_path": 4,        # how many cells of a person's path ahead count
    "people_cancel": 2,      # a person this close to a member ends its plan, and no plan is made there
    "reach": 10,             # fleets within this many hops of a member are clamped
    "move_cost": 0.01,       # per hop moved: breaks ties between waiting and a sideways move
    "chains": 16, "steps": 300, "sweeps": 2, "penalty": 20.0,   # thrml, live: half the offline budget
    "solver": "plan",        # "plan": exact search and thrml; "hold": the hold test (members just wait)
}
ENABLED = {"on": False, "cfg": {}}
CURRENT: Dict[str, Any] = {}          # the env being stepped (for the decision timer)
# THE PLAN LOG (decision time by jam size). With PLAN_LOG["path"] set, every
# planning attempt of a run is appended to that CSV when the run ends: the day
# (RUN_INFO, set by collapse_eval.py), the step, the solver, the jam's size and
# the milliseconds it took. collapse_eval.py --plans summarises it.
PLAN_LOG: Dict[str, Any] = {"path": None}
RUN_INFO: Dict[str, Any] = {}
PLAN_LOG_COLS = ["map", "k", "seed", "step", "solver", "fleets", "variables", "choices_max",
                 "size_fleets", "size_choices", "found", "ms", "compile_ms", "sample_ms"]
UNREACHABLE = 1000


def _tail(d: int, m: int) -> int:
    """Least cost a fleet d hops from its goal can still pay over m more ticks."""
    j = min(m, d)
    return j * d - j * (j + 1) // 2


class CollapseRung:
    def __init__(self, env: Any, cfg: Optional[Dict[str, Any]] = None):
        self.env = env
        self.c = dict(DEFAULTS, **(cfg or {}))
        self.H = int(self.c["horizon"])
        self.adj = {u: sorted(env.G.neighbors(u)) for u in env.G.nodes()}
        self.plans: Dict[int, Dict[str, Any]] = {}
        self.member: Dict[str, int] = {}
        self.prev: List[tuple] = []           # last step's clusters, with their ages
        self.next_id = 0
        self.actions: Dict[str, int] = {}
        self.stats = {k: 0 for k in (
            "clusters", "plans", "plans_exact", "plans_thrml", "plans_hold", "hold_released", "no_plan", "too_big",
            "completed", "replanned", "cancel_blocked", "cancel_goal", "cancel_gone", "cancel_l3",
            "cancel_people", "near_people",
            "fleets_planned", "ticks", "fleet_steps")}
        self.max_size = 0
        self.plan_seconds = 0.0
        self.thrml_seconds = 0.0
        # DECISION TIME. One record per planning attempt: (solver, fleets,
        # milliseconds, found a plan?, thrml compile ms, thrml sampling ms).
        self.attempts: List[tuple] = []
        self.plan_log: List[Dict[str, Any]] = []   # one row per attempt (see PLAN_LOG)
        self._last_model: Dict[str, Any] = {}      # the last thrml model's size
        self._logged = False
        self.rung_seconds = 0.0                   # all of prepare(): plans and following them
        self.cool: Dict[frozenset, int] = {}   # clusters with no plan -> step to try again
        self._no_go_cells: Set[str] = set()       # refreshed every step by _people_view()
        self._people_near: Set[str] = set()
        self.hold_start: Dict[str, int] = {}      # hold test: when each held fleet's current hold began
        self.hold_cool: Dict[str, int] = {}       # hold test: a released fleet is not held again before this step

    # ------------------------------------------------------------ helpers
    def _anchors(self, node) -> List[str]:
        return self.env.rules._anchors(node)

    def _on(self, node, cell) -> bool:
        an = self._anchors(node)
        return len(an) == 1 and an[0] == cell

    def _cell_now(self, node) -> Optional[str]:
        """The cell a fleet counts as being in: its node, or between two the
        nearer one (a tie goes to the one it is driving towards)."""
        an = self._anchors(node)
        if len(an) == 1:
            return an[0]
        if not an:
            return None
        coords = self.env.density._coords_by_id
        pos = np.asarray(node.current_pos, dtype=np.float64)
        d = [float(np.abs(pos - np.asarray(coords[c], dtype=np.float64)).sum()) for c in an]
        if abs(d[0] - d[1]) > 1e-6:
            return an[int(np.argmin(d))]
        heading = np.asarray(getattr(node, "direction", np.zeros(3)), dtype=np.float64)
        ahead = [float(np.dot(np.asarray(coords[c], dtype=np.float64) - pos, heading)) for c in an]
        return an[int(np.argmax(ahead))]

    def _still(self) -> Set[str]:
        """Fleets that will not move: failed, parked without work, or leaving.
        NOT the env's waiting_nodes: that is a fleet held still with a peer
        close by -- inside a jam, which is exactly where a plan is needed."""
        e = self.env
        out = set(e.immobile_nodes)
        for name in ("frozen_nodes", "stopped_nodes", "_despawning"):
            out |= set(getattr(e, name, set()) or set())
        return out

    def _ball(self, cells: Set[str], r: int) -> Set[str]:
        """Every cell within r hops of any of these cells, through the floor graph."""
        dist = {c: 0 for c in cells}
        q = deque(cells)
        while q:
            u = q.popleft()
            if dist[u] >= r:
                continue
            for w in self.adj.get(u, ()):
                if w not in dist:
                    dist[w] = dist[u] + 1
                    q.append(w)
        return set(dist)

    def _people_view(self) -> None:
        """This step's people and debris: where plans may not go (_no_go_cells),
        and where no plan may be followed or made (_people_near)."""
        of = getattr(self.env, "obstacle_field", None)
        people, ahead, debris = set(), set(), set()
        if of is not None:
            for h in (getattr(of, "c_humans", None) or []):
                if h.get("cell") is not None:
                    people.add(h["cell"])
                    ahead |= set((h.get("path") or [])[: int(self.c["people_path"])])
            people |= set(getattr(of, "humans", None) or [])
            debris |= set(getattr(of, "c_debris", {}) or {})
            debris |= set(getattr(of, "debris", None) or [])
        self._no_go_cells = debris | self._ball(people | ahead, int(self.c["people_margin"]))
        self._people_near = self._ball(people, int(self.c["people_cancel"])) if people else set()

    def _no_go(self) -> Set[str]:
        return self._no_go_cells

    # ------------------------------------------------------------ the step
    def prepare(self) -> None:
        """Called once per step, before the ladder's retrace orders are read."""
        env = self.env
        t = env.step_count
        by_id = {n.id: n for n in env.nodes}
        still = self._still()
        self._people_view()
        near = self._people_near
        # 1. Plans that can no longer be followed.
        for pid, P in list(self.plans.items()):
            why = None
            for f in P["members"]:
                n = by_id.get(f)
                if n is None or f in still:
                    why = "gone"
                elif n.current_goal_id != P["goal"][f]:
                    why = "goal"
                elif 0 <= t - env._l3_recent.get(f, -10 ** 9) <= 1:
                    why = "l3"
                elif near and set(self._anchors(n)) & near:
                    why = "people"                      # a person close by: FLOWRRA's own rules take it
                if why:
                    break
            if why is None and t - P["since"] > self.c["tick_timeout"]:
                why = "blocked"
            if why:
                self._drop(pid, "cancel_" + why)
        # 2. This step's clusters, chained to last step's for their age.
        clusters = self._clusters(by_id, still)
        aged = []
        for C in clusters:
            age = 1 + max([a for S, a in self.prev if S & C], default=0)
            aged.append((C, age))
        self.prev = aged
        for C, age in aged:
            pids = {self.member[f] for f in C if f in self.member}
            planned = set()
            for p in pids:
                planned |= set(self.plans[p]["members"])
            if len(pids) == 1 and C <= planned:
                continue                              # already following one plan
            if not pids and age < self.c["grace_steps"]:
                continue                              # grace: the reflexes get a chance first
            if near and any(set(self._anchors(by_id[f])) & near for f in C if f in by_id):
                self.stats["near_people"] += 1
                continue                              # a person close by: no plan here
            group = C | planned
            if t < self.cool.get(frozenset(group), -1):
                continue                              # no plan a moment ago: the ladder has it
            if self.c["solver"] == "hold" and any(self.hold_cool.get(f, -1) > t for f in group):
                continue                              # hold test: just released, its own rules have it
            for p in pids:
                self._drop(p, "replanned")
            self.stats["clusters"] += 1
            if not self._plan(group, by_id, still):
                self.cool[frozenset(group)] = t + int(self.c["retry_after"])
        if len(self.cool) > 500:
            self.cool = {k: v for k, v in self.cool.items() if v > t}
        # 3. This step's move for every fleet on a plan.
        self.actions = {}
        for pid, P in list(self.plans.items()):
            self._follow(pid, P, by_id)
        # One authority: a fleet on a plan takes no retrace order.
        for f in self.member:
            env._retrace.pop(f, None)
        if self.hold_start:
            for f in [f for f in self.hold_start if f not in self.member]:
                del self.hold_start[f]                # no longer held: its next hold starts afresh

    def _clusters(self, by_id, still) -> List[Set[str]]:
        env = self.env
        warned = env._warn_set(env._verdicts_start or [])
        link: Dict[str, Set[str]] = {}
        for pair in warned:
            a, b = tuple(pair)
            if a in still or b in still or a not in by_id or b not in by_id:
                continue
            if by_id[a].current_goal_id is None or by_id[b].current_goal_id is None:
                continue
            link.setdefault(a, set()).add(b)
            link.setdefault(b, set()).add(a)
        seen, out = set(), []
        for f in sorted(link):
            if f in seen:
                continue
            comp, q = set(), [f]
            while q:
                x = q.pop()
                if x in comp:
                    continue
                comp.add(x)
                q.extend(link[x] - comp)
            seen |= comp
            out.append(comp)
        return out

    def _drop(self, pid: int, why: str) -> None:
        P = self.plans.pop(pid, None)
        if P is None:
            return
        self.stats[why] = self.stats.get(why, 0) + 1
        for f in P["members"]:
            if self.member.get(f) == pid:
                del self.member[f]

    # ------------------------------------------------------------ planning
    def _plan(self, group: Set[str], by_id, still) -> bool:
        env = self.env
        members = sorted(group)
        n = len(members)
        if n > self.c["max_fleets"] or (n > self.c["max_exact"] and self.c["big"] not in ("thrml", "hold")):
            self.stats["too_big"] += 1
            return False
        H = self.H
        starts, held0, dmaps = [], set(), []
        for f in members:
            node = by_id[f]
            c = self._cell_now(node)
            if c is None:
                self.stats["no_plan"] += 1
                return False
            starts.append(c)
            held0 |= set(self._anchors(node))
            dmaps.append(node.goal_distance_map)
        # Where everyone else may be, tick by tick.
        reach = {}
        q = deque()
        for c in held0:
            reach[c] = 0
            q.append(c)
        while q:
            u = q.popleft()
            if reach[u] >= self.c["reach"]:
                continue
            for w in self.adj.get(u, ()):
                if w not in reach:
                    reach[w] = reach[u] + 1
                    q.append(w)
        blocked = [set() for _ in range(H + 1)]
        for node in env.nodes:
            if node.id in group:
                continue
            an = set(self._anchors(node))
            if not an & set(reach):
                continue
            levels = []
            if node.id not in still:
                r = env.conflicts.route(node) or []
                gp = env.grid_pos_dict
                levels = [{gp.get(c) for c in lvl} - {None} for lvl in r]
            seen = set(an)
            for k in range(H + 1):
                if 1 <= k <= len(levels):
                    seen |= levels[k - 1]
                blocked[k] |= seen
        no_go = self._no_go()                       # debris, and people with a hop of margin
        t0 = time.perf_counter()
        paths, how = None, None
        mode = self.c["solver"]
        if mode == "hold":
            paths = self._hold(members, starts)
            if paths is None:
                return False
            how = "hold"
        if mode == "plan" and n <= self.c["max_exact"]:
            te = time.perf_counter()
            paths = self._exact(starts, held0, dmaps, blocked, no_go)
            ms = 1000 * (time.perf_counter() - te)
            self.attempts.append(("exact", n, ms, paths is not None, None, None))
            self.plan_log.append({"step": env.step_count, "solver": "exact", "fleets": n,
                                  "variables": n * self.H, "found": int(paths is not None), "ms": round(ms, 2)})
            how = "exact"
        if mode == "plan" and paths is None and n >= 3 and self.c["big"] == "hold":
            # The exact-plus-hold test: wherever thrml would be asked (5-8 fleets,
            # and 3-4 that exact search could not settle in time), hold instead.
            paths = self._hold(members, starts)
            if paths is None:
                return False
            how = "hold"
        if mode == "plan" and paths is None and n >= 3 and self.c["big"] == "thrml":
            # Clear the last call's timing and size first: an attempt that stops
            # before sampling (no room at some tick, too many choices) must not
            # be logged with the previous jam's numbers.
            try:
                from jam_thrml import TIMING as _T
                _T.clear()
            except Exception:
                pass
            self._last_model = {}
            t1 = time.perf_counter()
            paths = self._thrml(members, starts, held0, dmaps, blocked, no_go)
            dt = time.perf_counter() - t1
            self.thrml_seconds += dt
            try:
                from jam_thrml import TIMING
                tm = dict(TIMING)
            except Exception:
                tm = {}
            self.attempts.append(("thrml", n, 1000 * dt, paths is not None,
                                  1000 * tm["compile"] if "compile" in tm else None,
                                  1000 * tm["sample"] if "sample" in tm else None))
            lm = self._last_model
            size = tm.get("bucket") if isinstance(tm.get("bucket"), tuple) else (None, None)
            self.plan_log.append({"step": env.step_count, "solver": "thrml", "fleets": n,
                                  "variables": lm.get("variables", n * self.H), "choices_max": lm.get("K"),
                                  "size_fleets": size[0], "size_choices": size[1],
                                  "found": int(paths is not None), "ms": round(1000 * dt, 2),
                                  "compile_ms": round(1000 * tm["compile"], 2) if "compile" in tm else None,
                                  "sample_ms": round(1000 * tm["sample"], 2) if "sample" in tm else None})
            how = "thrml"
        self.plan_seconds += time.perf_counter() - t0
        if paths is None:
            self.stats["no_plan"] += 1
            return False
        pid = self.next_id
        self.next_id += 1
        self.plans[pid] = {"members": members, "cells": dict(zip(members, paths)), "tick": 0,
                           "since": env.step_count, "how": how,
                           "goal": {f: by_id[f].current_goal_id for f in members}}
        for f in members:
            self.member[f] = pid
        self.stats["plans"] += 1
        self.stats["plans_" + how] += 1
        self.stats["fleets_planned"] += n
        self.max_size = max(self.max_size, n)
        return True

    def _hold(self, members, starts) -> Optional[List[List[str]]]:
        """The hold test's "plan": every member waits on its cell. A fleet held
        `horizon` steps in a row is released, and not held again for
        `retry_after` steps (its own rules have it meanwhile)."""
        now = self.env.step_count
        if any(self.hold_cool.get(f, -1) > now for f in members):
            return None
        began = min(self.hold_start.get(f, now) for f in members)
        if now - began >= self.H:
            for f in members:
                self.hold_cool[f] = now + int(self.c["retry_after"])
                self.hold_start.pop(f, None)
            self.stats["hold_released"] += 1
            return None
        for f in members:
            self.hold_start.setdefault(f, now)
        return [[c] * (self.H + 1) for c in starts]

    @staticmethod
    def _d(dmap, c) -> int:
        v = dmap.get(c) if dmap is not None else None
        return UNREACHABLE if v is None or v < 0 else int(v)

    def _exact(self, starts, held0, dmaps, blocked, no_go) -> Optional[List[List[str]]]:
        """A* over the members' cells together, tick by tick, to tick H."""
        H, adj, d = self.H, self.adj, self._d
        mc = float(self.c["move_cost"])
        n = len(starts)
        s0 = tuple(starts)
        if len(set(s0)) < n or any(c in blocked[0] - held0 for c in s0):
            return None
        h = lambda t, pos: sum(_tail(d(dmaps[i], p), H - t) for i, p in enumerate(pos))
        openq = [(h(0, s0), 0, 0, s0)]
        parent = {(0, s0): None}
        best = {(0, s0): 0}
        expanded, t_start = 0, time.time()
        while openq:
            _f, g, t, pos = heapq.heappop(openq)
            if t == H:
                seq, key = [], (t, pos)
                while key is not None:
                    seq.append(key[1])
                    key = parent[key]
                seq.reverse()
                return [[s[i] for s in seq] for i in range(n)]
            if g > best.get((t, pos), 10 ** 12):
                continue
            expanded += 1
            if expanded > self.c["budget"] or time.time() - t_start > self.c["time_limit"]:
                return None
            held = set(pos) | blocked[t] | (held0 if t == 0 else set())
            options = []
            for p in pos:
                opts = [q for q in [p] + adj.get(p, []) if q not in no_go and q not in blocked[t + 1]
                        and (q == p or q not in held)]
                if not opts:
                    break
                options.append(opts)
            else:
                for nxt in itertools.product(*options):
                    if len(set(nxt)) < n:
                        continue
                    g2 = g + sum(d(dmaps[i], q) for i, q in enumerate(nxt)) + mc * sum(
                        1 for p, q in zip(pos, nxt) if q != p)
                    key = (t + 1, nxt)
                    if g2 < best.get(key, 10 ** 12):
                        best[key] = g2
                        parent[key] = (t, pos)
                        heapq.heappush(openq, (g2 + h(t + 1, nxt), g2, t + 1, nxt))
        return None

    def _thrml(self, members, starts, held0, dmaps, blocked, no_go) -> Optional[List[List[str]]]:
        """The same plan sampled with thrml: one variable per fleet per tick."""
        try:
            from jam_thrml import collapse_bucketed
        except Exception:
            return None
        H, adj, n = self.H, self.adj, len(members)
        dom = {}
        for i, s in enumerate(starts):
            dist = {s: 0}
            q = deque([s])
            while q:
                u = q.popleft()
                if dist[u] >= H:
                    continue
                for w in adj.get(u, ()):
                    if w not in dist:
                        dist[w] = dist[u] + 1
                        q.append(w)
            others0 = (held0 | set(starts)) - {s}
            for k in range(1, H + 1):
                # blocked[k] grows with k (a clamped fleet's cells are never
                # freed), so it already holds every cell blocked at k - 1.
                cells = [c for c, dc in dist.items() if dc <= k and c not in no_go and c not in blocked[k]]
                if k == 1:
                    cells = [c for c in cells if (c == s or c in adj.get(s, ())) and c not in others0]
                if not cells:
                    return None
                dom[i, k] = sorted(cells)
        K = max(len(v) for v in dom.values())
        self._last_model = {"K": K, "variables": n * H}
        if K > 192:                                 # larger than the largest size (B2)
            return None
        keys = [(i, k) for i in range(n) for k in range(1, H + 1)]
        where = {key: r for r, key in enumerate(keys)}
        cost = np.zeros((len(keys), K), dtype=np.float32)
        pad = np.ones((len(keys), K), dtype=bool)
        for r, (i, k) in enumerate(keys):
            for a, c in enumerate(dom[i, k]):
                cost[r, a] = self._d(dmaps[i], c)
                pad[r, a] = False
        pl, pr, tables = [], [], []

        def add(kl, kr, ok):
            W = np.zeros((K, K), dtype=bool)
            for a, ca in enumerate(dom[kl]):
                for b, cb in enumerate(dom[kr]):
                    W[a, b] = ok(ca, cb)
            pl.append(where[kl]); pr.append(where[kr]); tables.append(W)

        for i in range(n):
            for k in range(1, H):
                add((i, k), (i, k + 1), lambda a, b: a == b or b in adj.get(a, ()))
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(1, H + 1):
                    add((i, k), (j, k), lambda a, b: a != b)
                for k in range(1, H):
                    add((i, k + 1), (j, k), lambda a, b: a != b)
                    add((j, k + 1), (i, k), lambda a, b: a != b)
        M = {"keys": keys, "dom": dom, "K": K, "cost": cost, "pad": pad,
             "pl": np.array(pl), "pr": np.array(pr), "ok": np.stack(tables)}
        # Starts: half the chains drive each fleet down its goal map after a
        # random wait; half are random.
        chains = int(self.c["chains"])
        rng = np.random.default_rng(self.env.step_count)
        sizes = np.array([len(dom[key]) for key in keys])
        init = (rng.random((chains, len(keys))) * sizes).astype(np.uint8)
        for c in range(0, chains, 2):
            for i, s in enumerate(starts):
                wait = 0 if c == 0 else int(rng.integers(0, 3))
                seq, cur = [s] * (wait + 1), s
                while len(seq) <= H:
                    nxt = [w for w in adj.get(cur, ()) if self._d(dmaps[i], w) < self._d(dmaps[i], cur)]
                    cur = (nxt[0] if c == 0 else nxt[int(rng.integers(len(nxt)))]) if nxt else cur
                    seq.append(cur)
                for k in range(1, H + 1):
                    if seq[k] in dom[i, k]:
                        init[c, where[(i, k)]] = dom[i, k].index(seq[k])
        # B2: padded to a fixed size (exact fleets; choices 32 ... 192), so each
        # size is compiled once per run and every later jam of that size reuses it.
        res = collapse_bucketed(M, n, H, chains, int(self.c["steps"]), int(self.c["sweeps"]),
                                int(self.env.step_count), float(self.c["penalty"]), init)
        if res is None:
            return None
        idx, (breaks, _e) = res
        if breaks:
            return None
        return [[starts[i]] + [dom[i, k][idx[where[(i, k)]]] for k in range(1, H + 1)] for i in range(n)]

    # ------------------------------------------------------------ following
    def _follow(self, pid: int, P: Dict[str, Any], by_id) -> None:
        env = self.env
        k = P["tick"]
        cells = P["cells"]
        if all(self._on(by_id[f], cells[f][k]) for f in P["members"]):
            if k == self.H:
                if P.get("how") == "hold":            # hold test: held long enough, release
                    for f in P["members"]:
                        self.hold_cool[f] = env.step_count + int(self.c["retry_after"])
                        self.hold_start.pop(f, None)
                self._drop(pid, "completed")
                return
            k += 1
            P["tick"] = k
            P["since"] = env.step_count
            self.stats["ticks"] += 1
        for f in P["members"]:
            node = by_id[f]
            target = cells[f][k]
            self.stats["fleet_steps"] += 1
            if self._on(node, target):
                self.actions[f] = 0                    # here: wait for the others
            else:
                self.actions[f] = int(env.rules._toward(node, {target: 0}))

    # ------------------------------------------------------------ reporting
    def summary(self) -> Dict[str, float]:
        out = {f"ladder_collapse_{k}": float(v) for k, v in self.stats.items()}
        out["ladder_collapse_max_size"] = float(self.max_size)
        out["ladder_collapse_plan_seconds"] = round(self.plan_seconds, 2)
        out["ladder_collapse_thrml_seconds"] = round(self.thrml_seconds, 2)
        out["ladder_collapse_big_thrml"] = 1.0 if self.c["big"] == "thrml" else 0.0
        out["ladder_collapse_rung_seconds"] = round(self.rung_seconds, 2)
        # Decision time per planning attempt, by solver (all attempts, found or not).
        for how in ("exact", "thrml"):
            ms = sorted(a[2] for a in self.attempts if a[0] == how)
            out[f"ladder_collapse_{how}_attempts"] = float(len(ms))
            if ms:
                out[f"ladder_collapse_{how}_ms_median"] = round(ms[len(ms) // 2], 1)
                out[f"ladder_collapse_{how}_ms_p95"] = round(ms[min(len(ms) - 1, int(0.95 * len(ms)))], 1)
                out[f"ladder_collapse_{how}_ms_max"] = round(ms[-1], 1)
                out[f"ladder_collapse_{how}_ms_mean"] = round(sum(ms) / len(ms), 1)
        comp = sorted(a[4] for a in self.attempts if a[0] == "thrml" and a[4] is not None)
        samp = sorted(a[5] for a in self.attempts if a[0] == "thrml" and a[5] is not None)
        if comp:
            out["ladder_collapse_thrml_compile_ms_median"] = round(comp[len(comp) // 2], 1)
        if samp:
            out["ladder_collapse_thrml_sample_ms_median"] = round(samp[len(samp) // 2], 1)
        return out


# ---------------------------------------------------------------- installing
def install(core_module) -> None:
    """Wrap FLOWRRA's constructor (attach the rung), _ladder_plan_retreats (no
    retrace for a cycle on a plan), _ladder_retrace_actions (plans first, then
    the plan's moves outrank any other order) and get_error_statistics (the
    rung's counters, and every fleet's distance driven, in each CSV row)."""
    F = core_module.FLOWRRA
    if getattr(F, "_collapse_installed", False):
        return
    _init, _plan_retreats = F.__init__, F._ladder_plan_retreats
    _retrace_actions, _stats = F._ladder_retrace_actions, F.get_error_statistics
    _step = F.step

    def __init__(self, *a, **kw):
        _init(self, *a, **kw)
        self._collapse = (CollapseRung(self, ENABLED.get("cfg"))
                          if ENABLED["on"] and getattr(self, "ladder_retrace", False) else None)

    def _ladder_plan_retreats(self, cycles):
        rung = getattr(self, "_collapse", None)
        if rung is not None and rung.member:
            on_plan = set(rung.member)
            cycles = [c for c in cycles if not (set(c) & on_plan)]
        return _plan_retreats(self, cycles)

    def _ladder_retrace_actions(self):
        rung = getattr(self, "_collapse", None)
        if rung is not None:
            t = time.perf_counter()
            rung.prepare()
            rung.rung_seconds += time.perf_counter() - t
        out = _retrace_actions(self)
        if rung is not None:
            out.update(rung.actions)
        return out

    def step(self, *a, **kw):
        # DECISION TIME, every day (rung on or off): the whole step, and the
        # fleets' own decision inside it (network and route, or RULES' driver).
        CURRENT["env"] = self
        g = getattr(self, "gnn", None)
        if g is not None and not getattr(getattr(g, "choose_actions", None), "_ct_timer", False):
            inner = g.choose_actions

            def choose_actions(*ca, **ckw):
                t = time.perf_counter()
                try:
                    return inner(*ca, **ckw)
                finally:
                    e = CURRENT.get("env")
                    if e is not None:
                        e._ct_decide = getattr(e, "_ct_decide", 0.0) + time.perf_counter() - t

            choose_actions._ct_timer = True
            g.choose_actions = choose_actions
        t = time.perf_counter()
        try:
            return _step(self, *a, **kw)
        finally:
            self._ct_step = getattr(self, "_ct_step", 0.0) + time.perf_counter() - t
            self._ct_steps = getattr(self, "_ct_steps", 0) + 1

    def get_error_statistics(self, *a, **kw):
        st = _stats(self, *a, **kw)
        st["ladder_collapse_on"] = 1.0 if getattr(self, "_collapse", None) is not None else 0.0
        n = max(1, int(getattr(self, "_ct_steps", 0)))
        st["ladder_collapse_step_ms"] = round(1000 * getattr(self, "_ct_step", 0.0) / n, 2)
        st["ladder_collapse_decide_ms"] = round(1000 * getattr(self, "_ct_decide", 0.0) / n, 2)
        rung = getattr(self, "_collapse", None)
        st["ladder_collapse_rung_ms"] = round(1000 * (rung.rung_seconds if rung is not None else 0.0) / n, 2)
        travel = getattr(self, "_fleet_travel", {}) or {}
        st["ladder_collapse_travel_all"] = round(float(sum(travel.values())), 2)
        if getattr(self, "_collapse", None) is not None:
            st.update(self._collapse.summary())
            rung = self._collapse
            if PLAN_LOG.get("path") and not rung._logged and rung.plan_log:
                import csv
                import os as _os
                new = not _os.path.exists(PLAN_LOG["path"])
                with open(PLAN_LOG["path"], "a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=PLAN_LOG_COLS, extrasaction="ignore")
                    if new:
                        w.writeheader()
                    for row in rung.plan_log:
                        w.writerow({**{k: RUN_INFO.get(k) for k in ("map", "k", "seed")}, **row})
                rung._logged = True
        return st

    F.__init__ = __init__
    F._ladder_plan_retreats = _ladder_plan_retreats
    F._ladder_retrace_actions = _ladder_retrace_actions
    F.get_error_statistics = get_error_statistics
    F.step = step
    F._collapse_installed = True
