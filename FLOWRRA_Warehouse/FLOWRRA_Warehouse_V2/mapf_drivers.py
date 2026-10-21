"""
mapf_drivers.py -- the MAPF baselines driving on FLOWRRA's own floor
(build log entry 44, the final deploy_eval).

SAME FLOOR, DIFFERENT BRAIN
baselines_mapf.run_rolling_horizon_instance runs its planners in a world of its
own: robots jump cell to cell with no gap and no braking, there are no people,
the failure model is its own, and there is no order book and no docks. Put
beside deploy_eval's rows, that would compare a point-robot algorithm with a
physical fleet. Here the SAME planners are the brain of a driver inside the real
simulator, exactly where RULES and the networks sit:

  the world, the same for every arm   map, order book, greedy dispatcher, docks,
                                      rescue handover, failures and people on
                                      keyed days, half a hop per step
  physical safety, on for every arm   1-hop minimum gap and the intent guard,
                                      braking bubbles, direction-aware braking,
                                      never into a robot that stays put, people a
                                      hard no-go, collision recovery
  our brain, OFF for these arms       route by default, corridor entry, priority,
                                      the conflict ladder, livelock escape
                                      (BRAIN_OFF below; the env reads them once,
                                      when it is built)

The planner decides each robot's next cell; our physics moves it. A planner that
asks a robot to follow nose to tail gets the same braking as any of ours, and a
robot the guard holds back simply arrives late.

ARMS
  PIBT          per-step PIBT (Okumura et al. 2022): plan_pibt's rule, applied
                at every step to the robots standing on a node. Candidates are
                the neighbours and the robot's own cell, nearest the goal first
                (ties by cell id); a lower-priority robot in the way inherits the
                priority and is pushed, recursively, with backtracking; no swap
                with the pusher. Priority grows by one per timestep (half per
                step) away from the goal and resets when the goal changes.
                Heuristic: the robot's own distance map from the env, so exits
                and rescue pickups mean exactly what they mean for RULES.
  RHCR-PP       rolling horizon (Li et al., AAAI 2021, the mechanism
                reimplemented, NOT the authors' C++): every `replan_every`
                timesteps, and whenever the floor changes (a robot enters or
                leaves, a goal changes, a robot dies, a rescue is assigned) or a
                robot is found off its plan, plan a `window`-timestep window
                with windowed prioritized planning -- baselines_mapf's
                space_time_astar_windowed, unchanged, in plan_windowed's own
                loop -- from where the robots are to the goals the env gives
                them.
                With a ONE-STEP BUFFER (k-robust, k = 1): a cell stays reserved for
                one timestep after a robot leaves it, so convoys keep one empty cell
                -- plans our floor can carry out. Recorded as planner_robust_k.
  RHCR-PIBT     the same loop with baselines_mapf.plan_pibt over the window (no
                buffer: PIBT's push needs following; forbid it and PIBT freezes).
  RHCR-*+naive  the same planner with the naive rescue instead of ours
                (errors.naive_rescue): rescuers chosen exactly as ours, but no
                re-dispatch -- a rescuer that dies while fetching leaves its
                pickup abandoned, and a robot that dies carrying a rescued load
                gets no pickup. Against the same planner without it, this
                measures what re-dispatch is worth.

TIME. One planning timestep is one hop: 1 / base_speed steps (2). A planned
wait is kept while a robot is on time; a robot that is late (braked, held by the
guard) skips waits it has already missed and moves on; a robot found anywhere
off its remaining plan triggers a replan.

WHAT THE PLANNERS SEE: the whole floor (they are centralised; our robots see
their vision range). Blocked cells: people and debris, and every robot that will
not move (dead, parked) -- except that a rescuer may enter its own pickup cell,
the dead robot's cell. RHCR-PP blocks per robot (its heuristic is computed
without the blocked cells); plan_pibt takes one graph, so for RHCR-PIBT a
pickup cell is open to everyone. The simulator waits for the planner: planning
time is reported (planner_ms_per_step), never charged.

A ROBOT BETWEEN TWO NODES finishes the move it started (its committed target);
the planners count it on both cells, and RHCR plans it from the cell it is
entering.
"""

from __future__ import annotations

import time
import zlib
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from baselines_mapf import bfs_distances, plan_pibt, space_time_astar_windowed
from config_warehouse import CONFIG
from drive_shortest_path import ShortestPathDriver

BASELINE_ARMS = ("PIBT", "RHCR-PP", "RHCR-PIBT", "RHCR-PP+naive", "RHCR-PIBT+naive")

# Our brain: off while a baseline arm's env is built (read once, at construction).
BRAIN_OFF: Dict[Tuple[str, ...], Any] = {
    ("conflict", "route_by_default"): False,
    ("conflict", "corridor_entry"): False,
    ("conflict", "priority"): False,
    ("conflict", "ladder", "enabled"): False,
    ("recovery", "livelock_patience"): 10 ** 9,      # livelock escape never fires
}
_MISSING = object()


def baseline_name(spec: str) -> Optional[str]:
    """The canonical arm name if `spec` names a baseline, else None."""
    for a in BASELINE_ARMS:
        if spec.strip().upper() == a.upper():
            return a
    return None


class brain_off:
    """Our brain's switches off while a baseline arm's env is built; restored after."""

    def __enter__(self):
        self.saved = []
        for path, v in BRAIN_OFF.items():
            node = CONFIG
            for k in path[:-1]:
                node = node.setdefault(k, {})
            self.saved.append((node, path[-1], node.get(path[-1], _MISSING)))
            node[path[-1]] = v
        return self

    def __exit__(self, *exc):
        for node, k, old in reversed(self.saved):
            if old is _MISSING:
                node.pop(k, None)
            else:
                node[k] = old
        return False


# ----------------------------------------------------------------------------- the floor
def _coords(env) -> Dict[str, Any]:
    c = getattr(env, "_coords_by_id", None)
    return c if c is not None else env.coords_by_id


def _cell(env, pos) -> Optional[str]:
    return env.grid_pos_dict.get(tuple(int(round(float(v))) for v in pos))


def _on_node(env, node) -> Optional[str]:
    p = np.asarray(node.current_pos, dtype=float)
    r = np.round(p)
    if float(np.max(np.abs(p - r))) > 1e-4:
        return None
    return env.grid_pos_dict.get(tuple(int(v) for v in r))


def _bounds(env, node) -> List[str]:
    """The node(s) a robot occupies: one on a node, two between nodes."""
    p = np.asarray(node.current_pos, dtype=float)
    out: List[str] = []
    for c in (np.floor(p + 1e-6), np.ceil(p - 1e-6)):
        nid = env.grid_pos_dict.get(tuple(int(v) for v in c))
        if nid is not None and nid not in out:
            out.append(nid)
    return out


def _action_toward(env, node, target: Optional[str]) -> int:
    if target is None:
        return 0
    d = np.asarray(_coords(env)[target], dtype=float) - np.asarray(node.current_pos, dtype=float)
    ax = int(np.argmax(np.abs(d)))
    if abs(d[ax]) < 1e-6:
        return 0
    return 1 + 2 * ax + (0 if d[ax] > 0 else 1)          # 1:+X 2:-X 3:+Y 4:-Y 5:+Z 6:-Z


def _guess_target(env, node, bounds: List[str]) -> Optional[str]:
    """For a robot between nodes with no recorded target: the bound it is moving toward."""
    if not bounds:
        return None
    if len(bounds) == 1:
        return bounds[0]
    p = np.asarray(node.current_pos, dtype=float)
    d = np.asarray(getattr(node, "direction", np.zeros(3)), dtype=float)
    if float(np.abs(d).sum()) > 0:
        return max(bounds, key=lambda c: float(np.dot(np.asarray(_coords(env)[c], float) - p, d)))
    return min(bounds, key=lambda c: float(np.abs(np.asarray(_coords(env)[c], float) - p).sum()))


def _blocked_cells(env) -> Tuple[Set[str], Dict[str, str]]:
    """Cells no planned robot may enter (people, debris, robots that will not
    move), and each rescuer's pickup cell, which is open to that rescuer only."""
    blocked: Set[str] = set()
    for c in (getattr(env, "static_obstacles", None) or ()):
        nid = env.grid_pos_dict.get(tuple(int(v) for v in c))
        if nid is not None:
            blocked.add(nid)
    immobile = env.immobile_nodes
    for n in env.nodes:
        if n.id in immobile:
            blocked.update(_bounds(env, n))
    pickup_of: Dict[str, str] = {}
    for fid, pid in (getattr(env, "_pickup_assignment", None) or {}).items():
        pk = env.open_pickups.get(pid)
        if pk is not None:
            nid = _cell(env, pk["pos"])
            if nid is not None:
                pickup_of[fid] = nid
    return blocked, pickup_of


def _tie(fid: str) -> int:
    return zlib.crc32(str(fid).encode())


class _BaselineDriver(ShortestPathDriver):
    """The env calls exactly what it calls on RULES (no preemptive recovery)."""

    def __init__(self, env):
        super().__init__(env, "never", 0.5, 0)
        self.target: Dict[str, str] = {}          # committed next node, robots between nodes
        self.plan_s = 0.0
        self.steps = 0

    def _between(self, env, node, out) -> List[str]:
        """A robot between nodes finishes its move; returns the cells it holds."""
        b = _bounds(env, node)
        t = self.target.get(node.id)
        if t not in b:
            t = _guess_target(env, node, b)
        out[node.id] = _action_toward(env, node, t)
        return b

    def planner_stats(self) -> Dict[str, Any]:
        return {"planner_ms_per_step": round(1000.0 * self.plan_s / max(1, self.steps), 3)}


# ----------------------------------------------------------------------------- per-step PIBT
class PIBTDriver(_BaselineDriver):
    def __init__(self, env):
        super().__init__(env)
        self.pri: Dict[str, float] = {}
        self.goal_key: Dict[str, Tuple] = {}
        self.at_goal: Dict[str, bool] = {}
        self.decisions = 0
        self.dt = float(CONFIG["warehouse"]["base_speed"])  # one timestep = 1 / base_speed steps

    @staticmethod
    def _h(node, v) -> float:
        m = node.goal_distance_map
        d = m.get(v) if m is not None else None
        return 1e9 if d is None else float(d)

    def _update_priority(self, node, here: Optional[str]) -> None:
        """PIBT's rule: priority grows while away from the goal and resets when
        the goal is REACHED. A goal changed mid-route (a rescue dispatch) keeps
        the priority earned so far; the env hands out the next goal on arrival,
        so a change seen right after arriving is a reset."""
        key = tuple(np.round(np.asarray(node.goal_pos, dtype=float), 3))
        h = self._h(node, here) if here is not None else 1e9
        if self.goal_key.get(node.id) != key:
            first = node.id not in self.goal_key
            self.goal_key[node.id] = key
            n = max(1, self.env.G.number_of_nodes())
            if first:
                self.pri[node.id] = (h / n) if h < 1e9 else 0.0
            elif self.at_goal.get(node.id, False):
                self.pri[node.id] = 0.0
            self.at_goal[node.id] = h == 0
            return
        self.at_goal[node.id] = h == 0
        if h > 0:
            self.pri[node.id] = self.pri.get(node.id, 0.0) + self.dt

    def choose_actions(self, node_features=None, adj_matrix=None, episode_number=None,
                       total_episodes=None, node_ids=None, valid_action_masks=None, **_):
        t0 = time.perf_counter()
        env = self.env
        G = env.G
        by_id = {n.id: n for n in env.nodes}
        immobile = env.immobile_nodes
        held = {f for f, u in env._yield_until.items() if u > env.step_count}
        blocked, pickup_of = _blocked_cells(env)
        out: Dict[str, int] = {fid: 0 for fid in node_ids}
        claimed: Dict[str, str] = {}
        cur: Dict[str, str] = {}
        for fid in node_ids:
            if fid in immobile:
                continue
            node = by_id[fid]
            u = _on_node(env, node)
            self._update_priority(node, u)
            if u is None:
                for c in self._between(env, node, out):
                    claimed[c] = fid
                continue
            if fid in held or pickup_of.get(fid) == u:      # held by recovery, or transferring
                claimed[u] = fid
                continue
            cur[fid] = u
        occ = {u: f for f, u in cur.items()}
        nxt: Dict[str, Optional[str]] = {f: None for f in cur}

        def pibt(i: str, blocker: Optional[str]) -> bool:
            node = by_id[i]
            u = cur[i]
            nbrs = list(G.neighbors(u)) + [u]
            if node.goal_distance_map is None:               # no goal: prefer to stay
                cands = [u] + sorted(v for v in nbrs if v != u)
            else:
                cands = sorted(nbrs, key=lambda v: (self._h(node, v), v))
            for v in cands:
                if v in claimed:
                    continue
                if v in blocked and pickup_of.get(i) != v:
                    continue
                if blocker is not None and v == cur[blocker]:
                    continue                                 # no swap with the pusher
                claimed[v] = i
                k = occ.get(v)
                if k is not None and k != i and nxt[k] is None:
                    if not pibt(k, i):
                        del claimed[v]
                        continue
                nxt[i] = v
                return True
            claimed[u] = i
            nxt[i] = u
            return False

        for i in sorted(cur, key=lambda f: (-self.pri.get(f, 0.0), _tie(f))):
            if nxt[i] is None:
                pibt(i, None)
        for i, v in nxt.items():
            self.decisions += 1
            if v is not None and v != cur[i]:
                out[i] = _action_toward(env, by_id[i], v)
                self.target[i] = v
        self.plan_s += time.perf_counter() - t0
        self.steps += 1
        return np.array([out[f] for f in node_ids], dtype=np.int64)

    def planner_stats(self) -> Dict[str, Any]:
        return {**super().planner_stats(), "planner_decisions": self.decisions}


# ----------------------------------------------------------------------------- rolling horizon
class RHCRDriver(_BaselineDriver):
    def __init__(self, env, method: str, window: int = 20, replan_every: int = 5):
        super().__init__(env)
        if method not in ("PP", "PIBT"):
            raise ValueError(method)
        self.method = method
        self.window = int(window)
        self.replan_every = int(replan_every)
        self.sph = max(1, int(round(1.0 / float(CONFIG["warehouse"]["base_speed"]))))
        self.paths: Dict[str, List[str]] = {}
        self.k: Dict[str, int] = {}
        self.s0: Optional[int] = None
        self.sig = None
        self.replans = 0
        self.offplan = 0
        self.robust_k = 1 if method == "PP" else 0      # PIBT's push needs following

    # -- planning
    def _plan_pp(self, G, starts, goals, blocked, pickup_of, fids):
        """plan_windowed's PP loop, with each robot's heuristic computed on the
        floor without the blocked cells (its own pickup open to a rescuer)."""
        Gf = G.copy()
        Gf.remove_nodes_from(blocked - set(starts))
        hs = []
        for f, s, g in zip(fids, starts, goals):
            pk = pickup_of.get(f)
            if pk is not None and pk not in Gf:
                Gi = G.subgraph(set(Gf.nodes) | {pk}).copy()
            else:
                Gi = Gf
            hs.append(bfs_distances(Gi, g) if g in Gi else {})
        vertex_res: Dict[Tuple[str, int], int] = {}
        edge_res: Dict[Tuple[str, str, int], int] = {}
        paths = []
        for i in range(len(starts)):
            p = space_time_astar_windowed(G, starts[i], goals[i], hs[i],
                                          vertex_res, edge_res, self.window)
            if p is None:
                p = [starts[i]]
            paths.append(p)
            for t, v in enumerate(p):
                vertex_res[(v, t)] = i
                # ONE-STEP BUFFER (k-robust, k = 1; build log entry 44): a cell
                # stays reserved for one timestep after this robot leaves it, so
                # no plan follows nose to tail -- convoys keep one empty cell, a
                # spacing our floor can drive at full speed. Without it RHCR-PP's
                # plans assumed point robots and the guard broke them ~25,000
                # times on an 80-robot Kiva day.
                vertex_res.setdefault((v, t + self.robust_k), i) if self.robust_k else None
                if t + 1 < len(p):
                    edge_res[(v, p[t + 1], t)] = i
            for t in range(len(p), self.window + 1):
                vertex_res[(p[-1], t)] = i
        return paths

    def _plan_pibt(self, G, starts, goals, blocked, pickup_of):
        Gf = G.copy()
        Gf.remove_nodes_from(blocked - set(starts) - set(pickup_of.values()))
        goals = [g if g in Gf else s for s, g in zip(starts, goals)]
        return plan_pibt(Gf, starts, goals, self.window)

    def _here(self, env, node, u, b) -> Optional[str]:
        """The node a robot counts as standing on: its node, or between two the
        nearer one (exactly halfway: the one it is heading for)."""
        if u is not None:
            return u
        if not b:
            return None
        p = np.asarray(node.current_pos, dtype=float)
        dist = {c: float(np.abs(np.asarray(_coords(env)[c], float) - p).sum()) for c in b}
        best = min(dist.values())
        near = [c for c in b if dist[c] - best < 1e-6]
        if len(near) > 1:
            t = self.target.get(node.id)
            return t if t in near else _guess_target(env, node, near)
        return near[0]

    def _replan(self, env, by_id, active, where, goals, blocked, pickup_of, sig):
        fids = sorted(active, key=lambda f: (_tie(f), f))   # fixed PP order
        # Starts: a robot between two nodes is planned from the nearer one (it can
        # still go either way), and no two robots share a start -- if two are
        # nearest the same node, the nearer keeps it and the other takes its other
        # bound. (Planning both from the node they were heading for put them on
        # one cell, head-on, and the guard held them there forever.)
        taken: Dict[str, str] = {}
        order = sorted(fids, key=lambda f: (0 if where[f][0] is not None else 1, f))
        for f in order:
            u, b = where[f]
            h = self._here(env, by_id[f], u, b)
            if h in taken and u is None:
                alt = [c for c in b if c != h and c not in taken]
                h = alt[0] if alt else h
            taken.setdefault(h, f)
            self._start = getattr(self, "_start", {})
            self._start[f] = h
        starts = [self._start[f] for f in fids]
        gl = [goals[f] or s for f, s in zip(fids, starts)]
        t0 = time.perf_counter()
        if self.method == "PP":
            paths = self._plan_pp(env.G, starts, gl, blocked, pickup_of, fids)
        else:
            paths = self._plan_pibt(env.G, starts, gl, blocked, pickup_of)
        self.plan_s += time.perf_counter() - t0
        self.replans += 1
        self.paths = dict(zip(fids, paths))
        self.k = {f: 0 for f in fids}
        self.s0 = env.step_count
        self.sig = sig

    def choose_actions(self, node_features=None, adj_matrix=None, episode_number=None,
                       total_episodes=None, node_ids=None, valid_action_masks=None, **_):
        env = self.env
        step = env.step_count
        by_id = {n.id: n for n in env.nodes}
        immobile = env.immobile_nodes
        blocked, pickup_of = _blocked_cells(env)
        active = [f for f in node_ids if f not in immobile]
        where: Dict[str, Tuple[Optional[str], List[str]]] = {}
        for f in active:
            u = _on_node(env, by_id[f])
            where[f] = (u, [u] if u is not None else _bounds(env, by_id[f]))
        # A rescuer standing on its pickup is transferring a load: it does not
        # move, and nobody may push it (planned as an obstacle, not an agent).
        dwell = {f for f in active
                 if where[f][0] is not None and pickup_of.get(f) == where[f][0]}
        if dwell:
            blocked = blocked | {where[f][0] for f in dwell}
            active = [f for f in active if f not in dwell]
        goals = {f: _cell(env, by_id[f].goal_pos) for f in active}
        sig = (frozenset(active), frozenset(goals.items()),
               frozenset(pickup_of.items()), frozenset(env.stopped_nodes),
               frozenset(dwell))
        need = (self.s0 is None or sig != self.sig
                or step - self.s0 >= self.replan_every * self.sph)
        here = {f: self._here(env, by_id[f], *where[f]) for f in active}
        if not need:
            off = [f for f in active
                   if here[f] not in self.paths.get(f, [])[self.k.get(f, 0):]]
            if off:
                self.offplan += len(off)
                need = True
        if need:
            self._replan(env, by_id, active, where, goals, blocked, pickup_of, sig)
            here = {f: self._start.get(f, here[f]) for f in active}

        out: Dict[str, int] = {fid: 0 for fid in node_ids}
        for f in active:
            node = by_id[f]
            u, b = where[f]
            h = here[f]
            p = self.paths.get(f)
            k = self.k.get(f, 0)
            settle = 0 if u is not None else _action_toward(env, node, h)
            if not p or h not in p[k:]:
                out[f] = settle                              # replans next step
                continue
            j0 = p.index(h, k)
            self.k[f] = j0
            j = next((x for x in range(j0 + 1, len(p)) if p[x] != h), None)
            if j is None or step < self.s0 + self.sph * (j - 1):
                out[f] = settle                              # home, end of window, or a planned wait
                continue
            if u is None and p[j] not in b:
                out[f] = settle                              # reach the node before turning
                continue
            out[f] = _action_toward(env, node, p[j])         # on time or late: go
            self.target[f] = p[j]
        self.steps += 1
        return np.array([out[f] for f in node_ids], dtype=np.int64)

    def planner_stats(self) -> Dict[str, Any]:
        return {**super().planner_stats(), "planner_replans": self.replans,
                "planner_offplan": self.offplan,
                "planner_window": self.window, "planner_replan_every": self.replan_every,
                "planner_robust_k": self.robust_k}


# ----------------------------------------------------------------------------- factory
def make_driver(arm: str, env, window: int = 20, replan_every: int = 5):
    # "+naive" changes the rescue (errors.naive_rescue, set by deploy_eval while
    # the env is built), not the driver.
    arm = arm[: -len("+naive")] if arm.endswith("+naive") else arm
    if arm == "PIBT":
        return PIBTDriver(env)
    if arm == "RHCR-PP":
        return RHCRDriver(env, "PP", window, replan_every)
    if arm == "RHCR-PIBT":
        return RHCRDriver(env, "PIBT", window, replan_every)
    raise ValueError(f"unknown baseline arm {arm!r}")