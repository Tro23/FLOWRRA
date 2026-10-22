"""
extract_jams.py -- find the jams in a day recorded by record_day.py, and save
each one with everything the sampler will need (JAM_COLLAPSE_DESIGN.md, step 2).

WHAT A JAM IS
    At every step, take the fleets in a path warning. Link two of them when
    they are within --radius hops of each other through the floor graph (racks
    in between do not count as near). Every linked group of at least
    --min-size fleets is a cluster: a place where the density field piles up.
    A cluster that persists over consecutive steps, sharing at least one fleet
    from one step to the next, is ONE jam, from the step it forms to the step
    it dissolves.

WHAT IS SAVED PER JAM (at the step it forms, t0)
    members      the jam's fleets: cell and goal cell at t0
    ladder       what the ladder did: each member's recorded cell for the next
                 --horizon steps, and the highest rung any member reached
                 (1 = retreat, 3 = safety net) while the jam lasted
    neighbours   every other fleet within --context hops at t0, with its
                 recorded cells for the next --horizon steps. The sampler treats
                 these as fixed (clamped): its plan must be legal against them.
    no_go        people and debris cells for each of those steps
    patch        the floor cells within --context hops of the members, and the
                 edges between them (the sampler works on this patch only)

USE
    python extract_jams.py day_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz
        -> jams_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz, and a summary
    --radius 3  --min-size 2  --horizon 8  --context 12
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter, defaultdict, deque

DEAD, OFF = 5, None


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        return json.load(fh)


def node_key(x, y, z):
    """A position to the floor cell it is on, or nearest to while between two."""
    return (math.floor(x + 0.5), math.floor(y + 0.5), math.floor(z + 0.5))


class Floor:
    def __init__(self, static):
        self.cell = {}                       # (x, y, z) -> node id
        self.pos = {}                        # node id -> (x, y, z)
        for nid, c in static["cells"].items():
            k = node_key(*c)
            self.cell[k] = nid
            self.pos[nid] = k
        self.adj = defaultdict(set)
        for u, v in static["edges"]:
            if u in self.pos and v in self.pos:
                self.adj[u].add(v)
                self.adj[v].add(u)

    def at(self, x, y, z):
        return self.cell.get(node_key(x, y, z))

    def within(self, sources, hops):
        """Every cell within `hops` of any source, with its distance."""
        dist = {s: 0 for s in sources if s is not None}
        q = deque(dist)
        while q:
            u = q.popleft()
            if dist[u] >= hops:
                continue
            for v in self.adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist


def clusters_at(frame, floor, radius, min_size):
    """Linked groups of warned fleets in one frame."""
    warned = {}
    for r in frame["fleets"]:
        if r[6] and r[4] != DEAD:
            c = floor.at(r[1], r[2], r[3])
            if c is not None:
                warned[r[0]] = c
    ids = list(warned)
    reach = {f: floor.within([warned[f]], radius) for f in ids}
    link = defaultdict(set)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if warned[b] in reach[a]:
                link[a].add(b)
                link[b].add(a)
    seen, out = set(), []
    for f in ids:
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
        if len(comp) >= min_size:
            out.append(comp)
    return out


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("day")
    ap.add_argument("--radius", type=int, default=3, help="hops that link two warned fleets")
    ap.add_argument("--min-size", type=int, default=2, help="fleets in a cluster")
    ap.add_argument("--horizon", type=int, default=8, help="steps of the ladder's answer and of context saved")
    ap.add_argument("--context", type=int, default=12, help="hops around the jam: neighbours and floor patch")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    day = load(a.day)
    if day.get("stride", 1) != 1:
        raise SystemExit("record the day with --stride 1: jams need every step")
    frames = day["frames"]
    floor = Floor(day["static"])
    by_t = {f["t"]: f for f in frames}
    ts = sorted(by_t)

    # 1. Clusters per step, chained into jams.
    open_jams, jams = [], []
    for t in ts:
        cl = clusters_at(by_t[t], floor, a.radius, a.min_size)
        rung = {r[0]: (r[10] if len(r) > 10 else 0) for r in by_t[t]["fleets"]}
        next_open = []
        used = set()
        for j in open_jams:
            match = [i for i, c in enumerate(cl) if i not in used and c & j["last"]]
            if match:
                for i in match:
                    used.add(i)
                    j["members"] |= cl[i]
                j["last"] = set().union(*[cl[i] for i in match])
                j["end"] = t
                j["max_size"] = max(j["max_size"], len(j["last"]))
                j["max_rung"] = max([j["max_rung"]] + [rung.get(f, 0) for f in j["last"]])
                next_open.append(j)
            else:
                jams.append(j)
        for i, c in enumerate(cl):
            if i not in used:
                next_open.append({"t0": t, "end": t, "first": set(c), "last": set(c), "members": set(c),
                                  "max_size": len(c),
                                  "max_rung": max(rung.get(f, 0) for f in c)})
        open_jams = next_open
    jams += open_jams

    # 2. The snapshot each sampler run needs, taken at the step the jam forms.
    out = []
    for k, j in enumerate(jams):
        t0, f0 = j["t0"], by_t[j["t0"]]
        rows = {r[0]: r for r in f0["fleets"]}
        members = sorted(j["first"])
        cells = {f: floor.at(*rows[f][1:4]) for f in members}
        goals = {f: (floor.at(*rows[f][7:10]) if len(rows[f]) > 9 and rows[f][7] is not None else None)
                 for f in members}
        patch = floor.within(list(cells.values()), a.context)
        window = [by_t[t] for t in ts if t0 <= t <= t0 + a.horizon]

        def track(fid):
            out_ = []
            for fr in window:
                r = next((x for x in fr["fleets"] if x[0] == fid), None)
                out_.append(floor.at(*r[1:4]) if r is not None else OFF)
            return out_

        neighbours = {}
        for fid, r in rows.items():
            if fid in cells:                  # failed fleets stay: they are obstacles too
                continue
            c = floor.at(*r[1:4])
            if c in patch:
                neighbours[fid] = {"track": track(fid), "failed": r[4] == DEAD}
        no_go = []
        for fr in window:
            s = set()
            for h in fr["humans"]:
                c = floor.at(*h[:3])
                if c in patch:
                    s.add(c)
            for d in fr["debris"]:
                c = floor.at(*d[:3])
                if c in patch:
                    s.add(c)
            no_go.append(sorted(s))
        patch_cells = sorted(patch)
        pset = set(patch_cells)
        out.append({
            "jam": k, "t0": t0, "t_end": j["end"], "duration": j["end"] - t0 + 1,
            "size_at_t0": len(members), "max_size": j["max_size"], "fleets_involved": len(j["members"]),
            "max_rung": j["max_rung"],
            "members": {f: {"cell": cells[f], "goal": goals[f], "ladder_track": track(f)} for f in members},
            "neighbours": neighbours, "no_go": no_go,
            "patch": {"cells": {c: list(floor.pos[c]) for c in patch_cells},
                      "edges": sorted([u, v] for u in patch_cells for v in floor.adj[u] if v in pset and u < v)},
        })

    base = a.day[:-8] if a.day.endswith(".json.gz") else a.day.rsplit(".", 1)[0]
    import os
    head, tail = os.path.split(base)
    dst = a.out or os.path.join(head, (tail.replace("day_", "jams_", 1) if tail.startswith("day_") else "jams_" + tail)
                                + ".json.gz")
    with gzip.open(dst, "wt") as fh:
        json.dump({"source": a.day, "arm": day["arm"], "map": day["map"], "k": day["k"], "seed": day["seed"],
                   "cond": day["cond"], "radius": a.radius, "min_size": a.min_size, "horizon": a.horizon,
                   "context": a.context, "jams": out}, fh, separators=(",", ":"))

    # 3. Summary.
    n = len(out)
    sizes = Counter(min(x["size_at_t0"], 6) for x in out)
    dur = sorted(x["duration"] for x in out)
    rungs = Counter(x["max_rung"] for x in out)
    print(f"[jams] {day['arm']} {day['map']} k={day['k']} seed={day['seed']} {day['cond']}: "
          f"{len(frames)} steps, {n} jams (radius {a.radius} hops, at least {a.min_size} fleets)")
    if n:
        print(f"[jams] size when formed: " + ", ".join(f"{s}{'+' if s == 6 else ''} fleets: {c}"
                                                    for s, c in sorted(sizes.items())))
        print(f"[jams] duration in steps: median {dur[n // 2]}, 90th percentile {dur[int(0.9 * (n - 1))]}, "
              f"longest {dur[-1]}")
        print(f"[jams] ladder: none {rungs.get(0, 0)}, retreat (L1/L2) {rungs.get(1, 0)}, "
              f"safety net (L3) {rungs.get(3, 0)}")
        nb = sorted(len(x["neighbours"]) for x in out)
        print(f"[jams] neighbours to clamp within {a.context} hops: median {nb[n // 2]}, most {nb[-1]}")
    print(f"[jams] -> {dst}")


if __name__ == "__main__":
    main()
