"""
jam_collapse.py -- the two-boundary collapse (JAM_COLLAPSE_DESIGN.md, Rohit's
design of 22 October). Steps 1 and 2: the boundaries from a recorded day, and
the EXACT best joint plan between them, compared with what the fleets did.

THE QUESTION
    Given where each fleet of a jam started and where it actually came out of
    the jam, could the jam have been untangled more directly?

THE BOUNDARIES (from the recording)
    start  each jam fleet's cell when the cluster forms (t0)
    exit   each fleet's cell at the first step it is no longer within --radius
           hops of any other jam fleet: where it came out of the cluster
    A jam is used only if every fleet comes out within --max-steps steps and
    none fails meanwhile.

TICKS. A plan moves in ticks of one hop. Tick k is matched to recorded step
t0 + 2k (two steps per hop at full speed): the fleets outside the jam are
clamped to their recorded cells at those steps, and people and debris too.

THE RULES (every plan, exact or sampled):
    * each tick a fleet stays or moves one hop;
    * no two fleets in one cell (jam fleets or clamped neighbours);
    * no entering a cell another fleet held one tick earlier: this rules out
      passing through each other. Stricter than FLOWRRA (following lags a
      tick), so it handicaps the plan, not the ladder.
    * no people, debris or racks.
    * a fleet that reaches its exit may drive on (it is "out"): from then on
      it holds no cell, as the recording stops following it once it is out
      of the cluster. The plan ends when every fleet is at its exit or out.
THE COST: every tick a fleet has not yet reached its exit costs 1, so a plan's
cost is the fleets' arrival times added up (sum of costs).

COMPARED, per fleet: the distance it actually drove (from the raw positions,
half-hops included, so dithering inside one edge counts at its true length)
against the plan's hops; the distance driven away from its exit (retreats);
detour against the shortest path; per jam: ticks until everyone is
out (the recorded time is in steps and includes braking; noted, not equated).

USE
    python jam_collapse.py DAY.json.gz JAMS.json.gz [--max-size 3] [--max-steps 40]
"""
from __future__ import annotations

import argparse
import gzip
import heapq
import itertools
import json
import math
from collections import Counter, defaultdict, deque

DEAD = 5


def load(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as fh:
        return json.load(fh)


class Floor:
    def __init__(self, static):
        self.cell, self.pos, self.adj = {}, {}, defaultdict(set)
        for nid, c in static["cells"].items():
            k = tuple(int(round(v)) for v in c)
            self.cell[k] = nid
            self.pos[nid] = k
        for u, v in static["edges"]:
            if u in self.pos and v in self.pos:
                self.adj[u].add(v)
                self.adj[v].add(u)
        self._d = {}

    def snap(self, p, p_next):
        on = lambda q: all(abs(v - round(v)) < 1e-6 for v in q)
        if on(p):
            return self.cell.get(tuple(int(round(v)) for v in p))
        if p_next is not None and on(p_next):
            return self.cell.get(tuple(int(round(v)) for v in p_next))
        return self.cell.get(tuple(math.floor(v + 0.5) for v in p))

    def dist(self, src):
        if src not in self._d:
            d = {src: 0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if v not in d:
                        d[v] = d[u] + 1
                        q.append(v)
            self._d[src] = d
        return self._d[src]


class Day:
    def __init__(self, day):
        self.floor = Floor(day["static"])
        self.by_t = {f["t"]: f for f in day["frames"]}
        self._cells = {}

    def cells(self, t):
        """fleet id -> (cell, state) at step t, mid-hop fleets snapped forward."""
        if t not in self._cells:
            fr, nx = self.by_t.get(t), self.by_t.get(t + 1)
            out = {}
            if fr is not None:
                nxt = {r[0]: r for r in nx["fleets"]} if nx else {}
                for r in fr["fleets"]:
                    n = nxt.get(r[0])
                    out[r[0]] = (self.floor.snap(r[1:4], n[1:4] if n else None), r[4])
            self._cells[t] = out
        return self._cells[t]

    def no_go(self, t):
        fr = self.by_t.get(t)
        s = set()
        if fr:
            for h in fr["humans"] + fr["debris"]:
                c = self.floor.cell.get(tuple(math.floor(v + 0.5) for v in h[:3]))
                if c:
                    s.add(c)
        return s


def build_problem(day, jam, radius, max_steps, slack, cap):
    F = day.floor
    t0 = jam["t0"]
    c0 = day.cells(t0)
    members = [f for f in jam["members"] if f in c0 and c0[f][0] is not None and c0[f][1] != DEAD]
    if len(members) < 2:
        return None, "fewer than 2 fleets on cells"
    start = {f: c0[f][0] for f in members}
    exit_, t_exit, path = {}, {}, {f: [start[f]] for f in members}
    for t in range(t0 + 1, t0 + max_steps + 1):
        ct = day.cells(t)
        for f in members:
            if f in exit_:
                continue
            cf = ct.get(f)
            if cf is None or cf[1] == DEAD:
                return None, "a fleet failed or left"
            if cf[0] is not None and cf[0] != path[f][-1]:
                path[f].append(cf[0])
            others = [ct[g][0] for g in members if g != f and g in ct and ct[g][0] is not None]
            d = F.dist(cf[0]) if cf[0] is not None else {}
            if cf[0] is not None and all(d.get(o, 10 ** 6) > radius for o in others):
                exit_[f], t_exit[f] = cf[0], t
        if len(exit_) == len(members):
            break
    if len(exit_) < len(members):
        return None, f"did not separate within {max_steps} steps"
    shortest = {f: F.dist(exit_[f]).get(start[f], 10 ** 6) for f in members}
    T = min(cap, max(shortest.values()) + slack)
    nb = []                                  # per tick: cells of fleets outside the jam
    ng = []
    for k in range(T + 1):
        ck = day.cells(t0 + 2 * k)
        nb.append({c for f, (c, st) in ck.items() if f not in start and c is not None})
        ng.append(day.no_go(t0 + 2 * k))
    return {"jam": jam["jam"], "t0": t0, "members": members, "start": start, "exit": exit_,
            "t_exit": t_exit, "recorded_path": path, "shortest": shortest, "T": T,
            "nb": nb, "no_go": ng}, None


def raw_track(day, fid, t0, t1):
    """The fleet's actual positions, step by step, from t0 to t1."""
    out = []
    for t in range(t0, t1 + 1):
        fr = day.by_t.get(t)
        r = next((x for x in fr["fleets"] if x[0] == fid), None) if fr else None
        if r is not None:
            out.append(tuple(r[1:4]))
    return out


def physical_stats(F, track, exit_cell):
    """Distance actually driven (hops, half-hops and all), and how much of it
    took the fleet further from its exit. A position between two cells reads
    its distance to the exit by interpolating between the two cells."""
    d = F.dist(exit_cell)

    def dist_at(p):
        lo = [math.floor(v + 1e-9) for v in p]
        frac = [v - l for v, l in zip(p, lo)]
        ax = max(range(3), key=lambda i: frac[i])
        a = F.cell.get(tuple(lo))
        if frac[ax] < 1e-6:
            return d.get(a, 0)
        hi = list(lo); hi[ax] += 1
        b = F.cell.get(tuple(hi))
        da, db = d.get(a, 0), d.get(b, 0)
        return (1 - frac[ax]) * da + frac[ax] * db

    driven = sum(sum(abs(x - y) for x, y in zip(p, q)) for p, q in zip(track, track[1:]))
    ds = [dist_at(p) for p in track]
    away = sum(max(0.0, b_ - a_) for a_, b_ in zip(ds, ds[1:]))
    return driven, away


def path_stats(F, path, exit_cell):
    d = F.dist(exit_cell)
    hops = len(path) - 1
    retreats = sum(1 for a, b in zip(path, path[1:]) if d.get(b, 0) > d.get(a, 0))
    return hops, retreats


OUT = "out"   # a fleet that reached its exit and drove on: it holds no cell any more


def exact_plan(F, P, budget=300000):
    """A* over joint positions and time.

    A fleet that reaches its exit may leave (OUT): it drives on and stops
    holding any cell, just as the recording stops following it once it is out
    of the cluster. The plan ends when every fleet is at its exit or out.
    Returns ((cost, makespan, paths), None) or (None, reason); each path runs
    tick by tick from the start to the exit (ticks after leaving are dropped)."""
    mem = P["members"]
    ex = [P["exit"][f] for f in mem]
    dist = [F.dist(e) for e in ex]
    T = P["T"]
    s0 = tuple(P["start"][f] for f in mem)
    if len(set(s0)) < len(s0) or any(c in P["nb"][0] for c in s0):
        return None, "start cells clash with recorded neighbours"
    done = lambda pos: all(p == OUT or p == e for p, e in zip(pos, ex))
    h = lambda pos: sum(0 if p == OUT else dist[i].get(p, 10 ** 6) for i, p in enumerate(pos))
    openq = [(h(s0), 0, 0, s0)]
    parent = {(0, s0): None}
    gbest = {(0, s0): 0}
    n = 0
    while openq:
        fval, g, t, pos = heapq.heappop(openq)
        if done(pos):
            seq, key = [], (t, pos)
            while key is not None:
                seq.append(key[1])
                key = parent[key]
            seq.reverse()
            paths = [[s[i] for s in seq if s[i] != OUT] for i in range(len(mem))]
            return (g, t, paths), None
        n += 1
        if n > budget:
            return None, "search budget exceeded"
        if t >= T:
            continue
        held = {p for p in pos if p != OUT} | P["nb"][t]
        options = []
        for i, p in enumerate(pos):
            if p == OUT:
                options.append([OUT])
                continue
            opts = [OUT] if p == ex[i] else []
            for q in [p] + sorted(F.adj[p]):
                if q in P["no_go"][t + 1] or q in P["nb"][t + 1]:
                    continue
                if q != p and q in held:
                    continue                    # entering a cell held one tick earlier
                if dist[i].get(q, 10 ** 6) > T - (t + 1):
                    continue
                opts.append(q)
            if not opts:
                break
            options.append(opts)
        else:
            for nxt in itertools.product(*options):
                cells = [q for q in nxt if q != OUT]
                if len(set(cells)) < len(cells):
                    continue
                g2 = g + sum(1 for q, e in zip(nxt, ex) if q != OUT and q != e)
                key = (t + 1, nxt)
                if g2 < gbest.get(key, 10 ** 9):
                    gbest[key] = g2
                    parent[key] = (t, pos)
                    heapq.heappush(openq, (g2 + h(nxt), g2, t + 1, nxt))
    return None, "no legal plan within the horizon"


def zone_windows(day, J, radius, every, cap):
    """Windows inside the standing congestion zones: at each step where the
    ladder starts acting on a fleet in the zone, the cluster around it."""
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from extract_jams import clusters_at, Floor as JFloor
    jf = JFloor({"cells": {k: list(v) for k, v in day.floor.pos.items()},
                 "edges": [[u, v] for u in day.floor.adj for v in day.floor.adj[u] if u < v]})
    out, n, taken = [], 0, set()
    for j in J:
        if j["duration"] <= 100:
            continue
        last = -10 ** 9
        prev = {}
        for t in range(j["t0"], j["t_end"] + 1):
            fr = day.by_t.get(t)
            if fr is None:
                continue
            rung = {r[0]: (r[10] if len(r) > 10 else 0) for r in fr["fleets"]}
            started = [f for f, rg in rung.items() if rg >= 1 and prev.get(f, 0) == 0]
            prev = rung
            if not started or t - last < every:
                continue
            for c in clusters_at(fr, jf, radius, 2):
                # Long jams can overlap in time; the same cluster at the same
                # step is taken once, under the first zone that finds it.
                if c & set(started) and (t, frozenset(c)) not in taken:
                    taken.add((t, frozenset(c)))
                    out.append({"jam": 10000 + n, "t0": t, "members": {f: {} for f in sorted(c)},
                                "zone_of": j["jam"], "duration": 0, "max_rung": 1, "size_at_t0": len(c)})
                    n += 1
                    last = t
                    break
            if n >= cap:
                return out
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("day")
    ap.add_argument("jams")
    ap.add_argument("--radius", type=int, default=3)
    ap.add_argument("--min-size", type=int, default=2, help="fleets per jam, at least")
    ap.add_argument("--max-size", type=int, default=3, help="fleets per jam, at most")
    ap.add_argument("--max-steps", type=int, default=40, help="steps for every fleet to come out")
    ap.add_argument("--slack", type=int, default=8, help="ticks beyond the longest shortest path")
    ap.add_argument("--cap", type=int, default=24, help="most ticks in a plan")
    ap.add_argument("--zones", type=int, default=0,
                    help="also take up to N windows inside the standing congestion zones (jams over 100 steps)")
    ap.add_argument("--zone-every", type=int, default=20, help="steps between zone windows")
    ap.add_argument("--no-exact", action="store_true", help="build the problems only; thrml solves them")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    day = Day(load(a.day))
    J = load(a.jams)["jams"]
    F = day.floor
    cands = [j for j in J if (j["duration"] >= 3 or j["max_rung"] >= 1) and j["duration"] <= 100
             and a.min_size <= j["size_at_t0"] <= a.max_size]
    if a.zones:
        cands += [w for w in zone_windows(day, J, a.radius, a.zone_every, a.zones)
                  if a.min_size <= w["size_at_t0"] <= max(a.max_size, 8)]
    skipped, rows, problems = Counter(), [], []
    for j in cands:
        P, why = build_problem(day, j, a.radius, a.max_steps, a.slack, a.cap)
        if P is None:
            skipped[why] += 1
            continue
        rec = {}
        for f in P["members"]:
            drv, away = physical_stats(F, raw_track(day, f, P["t0"], P["t_exit"][f]), P["exit"][f])
            rec[f] = {"driven": drv, "away": away}
        prob = {"P": {k: (sorted(v) if isinstance(v, set) else v) for k, v in P.items() if k not in ("nb", "no_go")},
                "nb": [sorted(x) for x in P["nb"]], "no_go": [sorted(x) for x in P["no_go"]],
                "recorded": rec, "zone_of": j.get("zone_of")}
        if a.no_exact:
            problems.append(prob)
            continue
        res, why = exact_plan(F, P)
        if res is None:
            skipped[why] += 1
            continue
        cost, makespan, plans = res
        r = {"jam": P["jam"], "t0": P["t0"], "size": len(P["members"]), "ticks_exact": makespan,
             "cost_exact": cost, "steps_recorded": max(P["t_exit"].values()) - P["t0"]}
        hr = rr = he = re_ = sh = we = 0
        for i, f in enumerate(P["members"]):
            h1, r1 = rec[f]["driven"], rec[f]["away"]
            seq = plans[i]
            moves = [q for p_, q in zip(seq, seq[1:]) if q != p_]
            hops_e = len(moves)
            ret_e = sum(1 for p_, q in zip(seq, seq[1:]) if q != p_ and
                        F.dist(P["exit"][f]).get(q, 0) > F.dist(P["exit"][f]).get(p_, 0))
            arrive = next(k for k in range(len(seq)) if all(c == P["exit"][f] for c in seq[k:]))
            hr += h1; rr += r1; he += hops_e; re_ += ret_e; sh += P["shortest"][f]
            we += arrive - hops_e
        r.update(hops_recorded=hr, hops_exact=he, retreats_recorded=rr, retreats_exact=re_, shortest=sh,
                 waits_exact=we)
        rows.append(r)
        prob.update(exact=plans, cost_exact=cost, ticks_exact=makespan)
        problems.append(prob)

    sizes = Counter(len(p["P"]["members"]) for p in problems)
    print(f"[collapse] {len(cands)} candidate jams ({a.min_size}-{a.max_size} fleets"
          + (f", plus zone windows" if a.zones else "") + f"); problems built: {len(problems)}"
          + f" (sizes: " + ", ".join(f"{k}: {v}" for k, v in sorted(sizes.items())) + ")")
    for k, c in skipped.most_common():
        print(f"[collapse]   skipped, {k}: {c}")
    n = len(rows)
    if n:
        S = lambda k: sum(r[k] for r in rows)
        print(f"[collapse] solved exactly: {n}")
        print(f"[collapse] distance driven, in hops: recorded {S('hops_recorded'):.1f}, exact {S('hops_exact')} "
              f"(shortest possible {S('shortest')})")
        print(f"[collapse] driven away from the exit, in hops: recorded {S('retreats_recorded'):.1f}, exact {S('retreats_exact')}")
        fewer = sum(r["hops_exact"] < r["hops_recorded"] - 1e-6 for r in rows)
        same = sum(abs(r["hops_exact"] - r["hops_recorded"]) <= 1e-6 for r in rows)
        print(f"[collapse] per jam: exact used fewer hops in {fewer}, the same in {same}, more in {n - fewer - same}")
        print(f"[collapse] exact plans waited {S('waits_exact')} fleet-ticks in all; median ticks until everyone "
              f"is out {sorted(r['ticks_exact'] for r in rows)[n // 2]} (recorded median "
              f"{sorted(r['steps_recorded'] for r in rows)[n // 2]} steps, braking included)")
    tag = "collapse_problems_" if a.no_exact else "collapse_exact_"
    out = a.out or a.jams.replace("jams_", tag)
    with gzip.open(out, "wt") as fh:
        json.dump({"rows": rows, "problems": problems}, fh)
    print(f"[collapse] -> {out}")


if __name__ == "__main__":
    main()
