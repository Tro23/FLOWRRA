"""
corridor_warehouse.py -- CONFLICT_DESIGN.md components 3 (corridor entry) and
4 (priority), with the corridor rule decided 2026-09-29.

A CORRIDOR is a single-lane chain of two-neighbour cells with a junction at
each end (no junction or side edge along it). Static: found once per map.

COMPONENT 3 -- ENTRY. A fleet standing on a junction whose chosen move enters a
corridor WAITS if the nearest fleet it can see inside is heading toward it. What
it can see: straight corridors -- max_vision_range edges along the corridor,
exactly what its ray would see (the nearest fleet stops the ray); bent
corridors -- neighbours' route intents within `radius` hops. A fleet inside
heading away (a convoy) does not stop entry. Two fleets at the two ends that can
see each other and both want in: priority picks one; the other waits.

COMPONENT 4 -- PRIORITY. One key everywhere, lower goes first:
    (hops to goal - aging x steps this fleet was made to wait,
     step it entered its current corridor   <- seniority,
     a fixed per-fleet tiebreak)
"Waited" accumulates over a trip and resets when the fleet's goal changes
(delivery). Applied where fleets compete:
  * FACING INSIDE A CORRIDOR (the meeting entry could not prevent -- beyond
    sight when the second fleet entered): the loser, and every fleet following
    it, BACKS OUT to the junction behind and PULLS OVER onto a free cell off the
    corridor's line (never the winners' next cell), then holds until the
    winners have cleared the corridor and that junction. Then it is released:
    back into the emptied corridor, or another way -- the policy decides.
  * THE SAME CELL NEXT: when two fleets' chosen moves claim the same cell, the
    lower priority waits this step.

Everything is decided ONCE per step, from start-of-step positions, before any
fleet moves -- so it is order-independent -- and returned as action overrides.
"""

from __future__ import annotations

import zlib
from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from node_warehouse import ACTION_DELTAS


# ============================================================== static index
class CorridorIndex:
    def __init__(self, G: Any, coords_by_id: Dict[str, Tuple[int, int, int]]):
        self.G = G
        self.coords = coords_by_id
        deg = dict(G.degree())
        self.sid_of: Dict[str, int] = {}
        self.cells: List[List[str]] = []          # ordered, end 0 -> end 1
        self.ends: List[Tuple[Optional[str], Optional[str]]] = []
        self.pos_in: List[Dict[str, int]] = []
        self.straight: List[bool] = []
        for v in G.nodes():
            if deg[v] != 2 or v in self.sid_of:
                continue
            chain = self._walk(v, deg)
            if chain is None:
                continue
            sid = len(self.cells)
            cells, e0, e1 = chain
            self.cells.append(cells)
            self.ends.append((e0, e1))
            self.pos_in.append({c: i for i, c in enumerate(cells)})
            for c in cells:
                self.sid_of[c] = sid
            pts = [coords_by_id.get(c) for c in cells]
            axes = {tuple(int(a != b) for a, b in zip(p, q)) for p, q in zip(pts, pts[1:])
                    if p is not None and q is not None}
            self.straight.append(len(axes) <= 1)

    def _walk(self, v, deg):
        """Ordered chain of degree-2 cells through v, and its two end junctions.
        None for a closed loop of degree-2 cells (no junction to enter from)."""
        a, b = list(self.G.neighbors(v))
        left, right = [], []
        for start, out in ((a, left), (b, right)):
            prev, cur = v, start
            while deg[cur] == 2:
                if cur == v:
                    return None
                out.append(cur)
                nxt = [w for w in self.G.neighbors(cur) if w != prev]
                prev, cur = cur, nxt[0]
            out.append(cur)                         # the end junction
        e0, e1 = left[-1], right[-1]
        cells = list(reversed(left[:-1])) + [v] + right[:-1]
        return cells, e0, e1

    def length(self, sid: int) -> int:
        return len(self.cells[sid])

    def end_pos(self, sid: int, end: str) -> Optional[float]:
        """Position of an end junction along the corridor: -1 or length."""
        e0, e1 = self.ends[sid]
        if end == e0:
            return -1.0
        if end == e1:
            return float(self.length(sid))
        return None


# ============================================================== dynamic rules
class ConflictRules:
    def __init__(self, env: Any, cfg: Dict[str, Any], vision: int):
        self.env = env
        self.entry_on = bool(cfg.get("corridor_entry", False))
        self.priority_on = bool(cfg.get("priority", False))
        self.aging = float(cfg.get("aging", 0.25))
        self.radius = float(cfg.get("radius", 3.0))
        # A fleet waiting ON the junction is standing on the one cell the
        # oncoming fleet must cross: when that fleet is this close, the waiter
        # pulls over (same mechanism as a retreat) and comes back after.
        self.pull_at = float(cfg.get("pullover_at", 2.0))
        # YIELD TO STOPPED FLEETS (conflict.yield_to_stopped): never move into
        # a cell held by a fleet that stays put this step -- queue behind it.
        self.stopped_on = bool(cfg.get("yield_to_stopped", False))
        self._waits_for: Dict[str, str] = {}      # entry waiter -> oncoming fleet
        self.vision = int(vision)
        self.idx = CorridorIndex(env.G, env._coords_by_id)
        self.entered: Dict[str, Tuple[int, int]] = {}      # fid -> (sid, step)
        self.waited: Dict[str, int] = {}
        self.goal_seen: Dict[str, Any] = {}
        # fid -> {"target", "dmap", "sid", "winners", "until"}
        self.orders: Dict[str, Dict[str, Any]] = {}
        # (step, fleet, event, corridor) for every retreat, pull-over, release
        # and timeout -- who backed out and when, for tests and debugging.
        self.log: List[Tuple[int, str, str, int]] = []
        self.stats = {k: 0 for k in (
            "entry_waits", "entry_contests", "meetings", "retreaters",
            "released", "timeouts", "no_pullover", "claim_waits", "hold_steps",
            "entry_pullovers", "stopped_waits", "stopped_cycles_left")}

    # ---------------------------------------------------------- geometry
    def _anchors(self, node) -> List[str]:
        a = self.env.proximity._anchor(np.asarray(node.current_pos, dtype=np.float64))
        return [nid for nid, _r in (a or []) if nid is not None]

    def _locate(self, node) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """(corridor id, position along it, junction id if standing on one)."""
        an = self._anchors(node)
        pos = np.asarray(node.current_pos, dtype=np.float64)
        if len(an) == 1:
            sid = self.idx.sid_of.get(an[0])
            if sid is not None:
                return sid, float(self.idx.pos_in[sid][an[0]]), None
            return None, None, an[0]
        if len(an) == 2:
            pts = []
            sid = None
            for nid in an:
                s = self.idx.sid_of.get(nid)
                if s is not None:
                    sid = s
            if sid is None:
                return None, None, None
            for nid in an:
                p = self.idx.pos_in[sid].get(nid)
                if p is None:
                    p = self.idx.end_pos(sid, nid)
                if p is None:
                    return None, None, None
                pts.append((p, np.asarray(self.idx.coords[nid], dtype=np.float64)))
            (p0, c0), (p1, c1) = pts
            span = float(np.abs(c1 - c0).sum()) or 1.0
            frac = float(np.abs(pos - c0).sum()) / span
            return sid, p0 + (p1 - p0) * frac, None
        return None, None, None

    def _next_cell(self, node) -> Optional[str]:
        levels = self.env.density._intended_path(node, horizon=1)
        if not levels:
            return None
        cells = {self.env.grid_pos_dict.get(tuple(int(v) for v in c)) for c, _w in levels[0]}
        cells.discard(None)
        return next(iter(cells)) if len(cells) == 1 else None

    def _heading(self, sid: int, here: float, nxt: Optional[str]) -> int:
        if nxt is None:
            return 0
        p = self.idx.pos_in[sid].get(nxt)
        if p is None:
            p = self.idx.end_pos(sid, nxt)
        if p is None:
            return 0
        return int(np.sign(p - here))

    def _target_cell(self, node, action: int) -> Optional[str]:
        """The graph node a move with `action` heads to (as get_goal_gradient
        probes it), or None if the move is not structurally valid."""
        if action == 0:
            return None
        delta = ACTION_DELTAS[action]
        if not node.is_structurally_valid(node.current_pos + delta * node.speed, action):
            return None
        axis = int(np.argmax(np.abs(delta)))
        probe = np.round(node.current_pos).astype(np.float64)
        v = float(node.current_pos[axis])
        if delta[axis] > 0:
            probe[axis] = np.ceil(v) + (1 if np.ceil(v) == v else 0)
        else:
            probe[axis] = np.floor(v) - (1 if np.floor(v) == v else 0)
        return self.env.grid_pos_dict.get(tuple(int(x) for x in probe))

    def _toward(self, node, dmap: Dict[str, int]) -> int:
        best, best_a = None, 0
        for a in range(1, 7):
            t = self._target_cell(node, a)
            if t is None or t not in dmap:
                continue
            if best is None or dmap[t] < best:
                best, best_a = dmap[t], a
        return best_a

    @staticmethod
    def _bfs(G, source: str, banned: Set[str], cutoff: int = 400) -> Dict[str, int]:
        d, q = {source: 0}, deque([source])
        while q:
            u = q.popleft()
            if d[u] >= cutoff:
                continue
            for w in G.neighbors(u):
                if w not in d and w not in banned:
                    d[w] = d[u] + 1
                    q.append(w)
        return d

    # ---------------------------------------------------------- priority
    def key(self, node) -> Tuple[float, float, int]:
        hops = node.get_graph_distance_to_goal()
        hops = float(hops) if hops is not None and np.isfinite(hops) else 1e6
        eff = hops - self.aging * self.waited.get(node.id, 0)
        ent = self.entered.get(node.id)
        return (eff, float(ent[1]) if ent else float("inf"),
                zlib.crc32(str(node.id).encode()))

    # ---------------------------------------------------------- the plan
    def plan(self, actions, node_ids: List[str]) -> Dict[str, int]:
        env = self.env
        step = env.step_count
        by_id = {n.id: n for n in env.nodes}
        active = [n for n in env.nodes if n.id not in env.immobile_nodes]
        chosen = {fid: int(a) for fid, a in zip(node_ids, actions)}

        for n in active:                                   # aging resets on delivery
            g = getattr(n, "current_goal_id", None)
            if self.goal_seen.get(n.id, g) != g:
                self.waited[n.id] = 0
            self.goal_seen[n.id] = g

        loc: Dict[str, Tuple] = {}
        occ: Dict[int, List[Tuple[float, str, int]]] = {}
        for n in active:
            sid, p, junc = self._locate(n)
            loc[n.id] = (sid, p, junc)
            if sid is not None:
                if self.entered.get(n.id, (None,))[0] != sid:
                    self.entered[n.id] = (sid, step)
                o = self.orders.get(n.id)
                if o is not None and o["sid"] == sid:
                    # backing out: it is heading for the junction behind, whatever
                    # its route says -- a fleet waiting there must see it coming
                    h = int(np.sign(self.idx.end_pos(sid, o["back"]) - p))
                else:
                    h = self._heading(sid, p, self._next_cell(n))
                occ.setdefault(sid, []).append((p, n.id, h))
            else:
                self.entered.pop(n.id, None)
        for lst in occ.values():
            lst.sort()

        out: Dict[str, int] = {}
        self._waits_for = {}
        if self.priority_on:
            self._advance_orders(by_id, loc, out)
            self._new_meetings(by_id, occ, out)
        if self.entry_on:
            self._entry(active, chosen, loc, occ, by_id, out)
        if self.priority_on:
            self._claims(active, chosen, out, by_id)
        if self.stopped_on:
            self._yield_to_stopped(active, chosen, out, by_id)

        for fid, a in out.items():
            if a == 0:
                self.waited[fid] = self.waited.get(fid, 0) + 1
        return out

    # ---- component 3
    def _visible(self, sid: int, occupants, from_end: float):
        """Occupants in sight from a junction at `from_end` (-1 or L), nearest
        first, as (distance, pos, fid, heading)."""
        reach = self.vision if self.idx.straight[sid] else self.radius
        vis = [(abs(p - from_end), p, fid, h) for p, fid, h in occupants]
        return sorted(v for v in vis if v[0] <= reach)

    def _entry(self, active, chosen, loc, occ, by_id, out):
        wants: Dict[Tuple[int, float], List[str]] = {}
        for n in active:
            sid0, _p, junc = loc[n.id]
            if junc is None or n.id in out:
                continue
            t = self._target_cell(n, chosen.get(n.id, 0))
            sid = self.idx.sid_of.get(t) if t is not None else None
            if sid is None:
                continue
            end = self.idx.end_pos(sid, junc)
            if end is None:
                continue
            wants.setdefault((sid, end), []).append(n.id)

        for (sid, end), fids in wants.items():
            L = self.idx.length(sid)
            vis = self._visible(sid, occ.get(sid, []), end)
            toward = -1 if end == L else 1          # heading of a fleet entering here
            if vis:
                d0, _p, _fid, h = vis[0]
                if h == -toward:                    # coming at the junction
                    oncoming = []
                    for _d, _pp, f, hh in vis:
                        if hh != -toward:
                            break
                        oncoming.append(f)
                    e0, e1 = self.idx.ends[sid]
                    junc, other = (e0, e1) if end == -1.0 else (e1, e0)
                    for fid in fids:
                        self._waits_for[fid] = oncoming[0] if oncoming else _fid
                        if d0 <= self.pull_at and fid not in self.orders:
                            self.stats["entry_pullovers"] += 1
                            self._order_retreat(sid, junc, other, [fid], oncoming, by_id, out)
                            if fid in out:
                                continue
                        out[fid] = 0
                        self.stats["entry_waits"] += 1
                continue
            reach = self.vision if self.idx.straight[sid] else self.radius
            if L + 1 > reach:
                continue
            other = -1.0 if end == L else float(L)
            rivals = wants.get((sid, other), [])
            if rivals and fids:
                if end == -1.0:
                    self.stats["entry_contests"] += 1
                pool = [by_id[f] for f in fids + rivals]
                win = min(pool, key=self.key).id
                for fid in fids:
                    if fid != win:
                        out[fid] = 0
                        self.stats["entry_waits"] += 1

    # ---- component 4: facing inside a corridor
    def _new_meetings(self, by_id, occ, out):
        for sid, lst in occ.items():
            reach = self.vision if self.idx.straight[sid] else self.radius
            for (p1, f1, h1), (p2, f2, h2) in zip(lst, lst[1:]):
                if not (h1 == 1 and h2 == -1) or p2 - p1 > reach:
                    continue
                if f1 in self.orders or f2 in self.orders:
                    continue
                a, b = by_id[f1], by_id[f2]
                loser = max((a, b), key=self.key)
                self.stats["meetings"] += 1
                i1 = [f for _p, f, _h in lst].index(f1)
                if loser.id == f1:                  # heading +1: its followers are below it
                    group = self._run(lst, i1, -1, 1)
                    winners = self._run(lst, i1 + 1, 1, -1)
                    back, front = self.idx.ends[sid]
                else:
                    group = self._run(lst, i1 + 1, 1, -1)
                    winners = self._run(lst, i1, -1, 1)
                    front, back = self.idx.ends[sid]
                self._order_retreat(sid, back, front, group, winners, by_id, out)

    @staticmethod
    def _run(lst, i, step, heading):
        """From index i outward (step -1 or +1), the unbroken run of fleets with
        this heading -- the one at i and every fleet directly behind it. A fleet
        facing another way ends the run."""
        out = []
        while 0 <= i < len(lst) and lst[i][2] == heading:
            out.append(lst[i][1])
            i += step
        return out

    def _order_retreat(self, sid, back, front, group, winners, by_id, out):
        env = self.env
        if back is None:
            self.stats["no_pullover"] += len(group)
            return
        corridor = set(self.idx.cells[sid])
        banned = corridor | ({front} if front is not None else set())
        # the winners' next cells beyond the junction stay clear
        for w in winners[:1]:
            levels = env.density._intended_path(by_id[w], horizon=self.idx.length(sid) + 3) or []
            for lvl in levels:
                for c, _w in lvl:
                    nid = env.grid_pos_dict.get(tuple(int(v) for v in c))
                    if nid is not None and nid not in corridor and nid != back:
                        banned.add(nid)
        occupied = set()
        for n in env.nodes:
            occupied.update(self._anchors(n))
        free = [c for c, _d in sorted(self._bfs(env.G, back, banned, cutoff=6).items(),
                                      key=lambda kv: (kv[1], str(kv[0])))
                if c != back and c not in occupied]
        if len(free) < len(group):
            self.stats["no_pullover"] += len(group) - len(free)
        # the fleet nearest the junction exits first and goes furthest
        exit_order = list(reversed(group))
        targets = list(reversed(free[:len(group)]))
        until = env.step_count + int(4 * self.idx.length(sid) / max(1e-6, float(
            by_id[group[0]].speed or 0.5))) + 40
        for fid, tgt in zip(exit_order, targets):
            dmap = self._bfs(env.G, tgt, {front} if front is not None else set())
            self.orders[fid] = {"target": tgt, "dmap": dmap, "sid": sid, "back": back,
                                "winners": set(winners), "until": until}
            self.stats["retreaters"] += 1
            self.log.append((env.step_count, fid, "retreat", sid))
            out[fid] = self._toward(by_id[fid], dmap)

    def _advance_orders(self, by_id, loc, out):
        for fid in list(self.orders):
            o = self.orders[fid]
            n = by_id.get(fid)
            if n is None or fid in self.env.immobile_nodes:
                del self.orders[fid]
                continue
            clear = all(
                w not in by_id or w in self.env.immobile_nodes
                or (loc.get(w, (None,))[0] != o["sid"] and loc.get(w, (None, None, None))[2] != o["back"])
                for w in o["winners"])
            if clear:
                del self.orders[fid]
                self.stats["released"] += 1
                self.log.append((self.env.step_count, fid, "released", o["sid"]))
                continue
            if self.env.step_count >= o["until"]:
                del self.orders[fid]
                self.stats["timeouts"] += 1
                self.log.append((self.env.step_count, fid, "timeout", o["sid"]))
                continue
            an = self._anchors(n)
            if an == [o["target"]]:
                out[fid] = 0
                self.stats["hold_steps"] += 1
            else:
                out[fid] = self._toward(n, o["dmap"])

    # ---- component 4: two fleets claiming the same cell next
    def _claims(self, active, chosen, out, by_id):
        claims: Dict[str, List[str]] = {}
        for n in active:
            a = out.get(n.id, chosen.get(n.id, 0))
            t = self._target_cell(n, a)
            if t is not None:
                claims.setdefault(t, []).append(n.id)
        for cell, fids in claims.items():
            if len(fids) < 2:
                continue
            # a fleet backing out always goes: stopping it keeps the corridor shut
            backing = [f for f in fids if f in self.orders]
            win = min((by_id[f] for f in (backing or fids)), key=self.key).id
            for f in fids:
                if f != win and f not in self.orders:
                    out[f] = 0
                    self.stats["claim_waits"] += 1

    # ---- yield to stopped fleets
    def _yield_to_stopped(self, active, chosen, out, by_id):
        """
        THE REMAINING COLLISIONS (CONFLICT_DESIGN.md, quick check): 62% were a
        moving fleet driving into a STOPPED one on a mesh junction -- often one
        the rules themselves had told to wait. Braking never reaches zero (it
        floors at 0.1), so a fleet heading for an occupied cell creeps into it.

        Rule: a fleet whose move this step heads for a cell occupied by a fleet
        that stays put this step waits instead -- a queue. Stays put: told to
        wait by a rule, held by recovery, waiting, chose idle, or chose a move
        that cannot be made. Applied to a fixpoint, so a queue propagates back.

        DEADLOCK GUARD. A wait is never imposed if it would close a loop in
        who-waits-for-whom -- e.g. a fleet waiting ON a junction to enter a
        corridor (it waits for the oncoming fleet) and that oncoming fleet
        waiting for it to clear the junction. Such a pair is left to today's
        machinery and counted (stopped_cycles_left).
        """
        env = self.env
        step = env.step_count
        held = {f for f, u in env._yield_until.items() if u > step}
        waiting = set(getattr(env, "waiting_nodes", set()))

        def effective(fid):
            return out.get(fid, chosen.get(fid, 0))

        # only waiters that really are waiting (not sent to pull over)
        edges: Dict[str, str] = {k: v for k, v in self._waits_for.items()
                                 if out.get(k) == 0}
        for _ in range(10):                        # fixpoint: queues propagate
            stays: Dict[str, str] = {}
            for n in active:
                a = effective(n.id)
                if (a == 0 or n.id in held or n.id in waiting
                        or self._target_cell(n, a) is None):
                    for c in self._anchors(n):
                        stays[c] = n.id
            changed = False
            for n in active:
                if n.id in held:
                    continue
                a = effective(n.id)
                if a == 0:
                    continue
                t = self._target_cell(n, a)
                s = stays.get(t) if t is not None else None
                if s is None or s == n.id:
                    continue
                # would this wait close a loop?
                seen, cur, loop = {n.id}, s, False
                while cur in edges:
                    cur = edges[cur]
                    if cur in seen:
                        loop = cur == n.id or loop
                        break
                    seen.add(cur)
                if loop or cur == n.id:
                    self.stats["stopped_cycles_left"] += 1
                    continue
                out[n.id] = 0
                edges[n.id] = s
                self.stats["stopped_waits"] += 1
                changed = True
            if not changed:
                break

    def statistics(self) -> Dict[str, Any]:
        return {f"corridor_{k}": v for k, v in self.stats.items()} | {
            "corridor_count": len(self.idx.cells),
            "corridor_orders_open": len(self.orders)}
