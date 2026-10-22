"""
jam_figures.py -- pictures of jams: what the fleets actually drove, against the
collapsed plan between the same start and exit cells.

Each jam gets two panels:
    left   the floor around the jam: racks in grey, each fleet's recorded path
           (thin) and the plan (dashed), start (dot) and exit (square)
    right  each fleet's distance to its exit, over time. The recorded line is
           step by step, so every back-and-forth shows as a zigzag; the plan
           moves one hop per tick, drawn at two steps per tick (full speed).

USE
    python jam_figures.py DAY.json.gz PROBLEMS.json.gz RESULTS.json.gz --jams 130,284 --out jams.png
    RESULTS can be the thrml output (jam_thrml.py) or the exact file (jam_collapse.py).
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, MUTED, RACK = "#1a1a19", "#f0efec", "#c3c2b7", "#55554f"
COLS = ["#3987e5", "#d55181", "#199e70", "#c98500", "#9085e9", "#d95926", "#e66767", "#e0c341"]


def load(p):
    with (gzip.open(p, "rt") if p.endswith(".gz") else open(p)) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("day")
    ap.add_argument("problems")
    ap.add_argument("results")
    ap.add_argument("--jams", required=True, help="jam numbers, e.g. 130,284")
    ap.add_argument("--out", default="jams.png")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from jam_collapse import Day, raw_track, physical_stats  # noqa: F401
    day = Day(load(a.day))
    F = day.floor
    probs = {p["P"]["jam"]: p for p in load(a.problems)["problems"]}
    R = load(a.results)
    if isinstance(R, dict):                      # the exact file
        plans = {p["P"]["jam"]: p["exact"] for p in R["problems"] if "exact" in p}
    else:                                        # thrml output
        plans = {r["jam"]: r["plan"] for r in R if r.get("legal") and "plan" in r}
    ids = [int(x) for x in a.jams.split(",")]
    ids = [j for j in ids if j in probs and j in plans]
    if not ids:
        raise SystemExit("none of those jams has both a problem and a plan")

    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 4.2 * len(ids)), dpi=110, squeeze=False,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(SURFACE)
    for row, jid in zip(axes, ids):
        P = probs[jid]["P"]
        plan = plans[jid]
        axm, axt = row
        for ax in row:
            ax.set_facecolor(SURFACE)
        # --- map
        pts = [F.pos[c] for i, f in enumerate(P["members"]) for c in plan[i]]
        tracks = {f: raw_track(day, f, P["t0"], P["t_exit"][f]) for f in P["members"]}
        xs = [q[0] for q in pts] + [p[0] for tr in tracks.values() for p in tr]
        ys = [q[1] for q in pts] + [p[1] for tr in tracks.values() for p in tr]
        x0, x1, y0, y1 = int(min(xs)) - 3, int(max(xs)) + 3, int(min(ys)) - 2, int(max(ys)) + 2
        while x1 - x0 < 12:                      # small jams: show some floor around them
            x0, x1 = x0 - 1, x1 + 1
        while y1 - y0 < 8:
            y0, y1 = y0 - 1, y1 + 1
        z = F.pos[P["start"][P["members"][0]]][2]
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if (x, y, z) not in F.cell:
                    axm.add_patch(plt.Rectangle((x - .46, y - .46), .92, .92, color=RACK, lw=0))
        for i, f in enumerate(P["members"]):
            c = COLS[i % len(COLS)]
            tr = tracks[f]
            axm.plot([p[0] for p in tr], [p[1] for p in tr], "-", color=c, alpha=.5, lw=1.2)
            q = [F.pos[x] for x in plan[i]]
            axm.plot([p[0] for p in q], [p[1] for p in q], "--", color=c, lw=2.4)
            s, e = F.pos[P["start"][f]], F.pos[P["exit"][f]]
            axm.plot(s[0], s[1], "o", color=c, ms=8)
            axm.plot(e[0], e[1], "s", color=c, ms=9, mfc="none", mew=2)
        axm.set_xlim(x0 - .5, x1 + .5); axm.set_ylim(y0 - .5, y1 + .5)
        axm.set_aspect("equal"); axm.axis("off")
        # --- distance to exit over time
        drv_rec = drv_plan = 0.0
        for i, f in enumerate(P["members"]):
            c = COLS[i % len(COLS)]
            d = F.dist(P["exit"][f])
            tr = tracks[f]
            # interpolated distance for mid-hop positions, as physical_stats does
            def dist_at(p):
                lo = [math.floor(v + 1e-9) for v in p]
                fr = [v - l for v, l in zip(p, lo)]
                ax_ = max(range(3), key=lambda k: fr[k])
                a_ = F.cell.get(tuple(lo))
                if fr[ax_] < 1e-6:
                    return d.get(a_, 0)
                hi = list(lo); hi[ax_] += 1
                return (1 - fr[ax_]) * d.get(a_, 0) + fr[ax_] * d.get(F.cell.get(tuple(hi)), 0)
            axt.plot(range(len(tr)), [dist_at(p) for p in tr], "-", color=c, alpha=.6, lw=1.3)
            nudge = (i - (len(P["members"]) - 1) / 2) * 0.04     # keeps equal plan lines from hiding each other
            axt.plot([2 * k for k in range(len(plan[i]))], [d.get(x, 0) + nudge for x in plan[i]], "--", color=c,
                     lw=2.4)
            drv_rec += physical_stats(F, tr, P["exit"][f])[0]
            drv_plan += sum(1 for p_, q_ in zip(plan[i], plan[i][1:]) if q_ != p_)
        axt.set_xlabel("steps since the jam formed", color=MUTED, fontsize=8)
        axt.set_ylabel("hops to its exit", color=MUTED, fontsize=8)
        axt.tick_params(colors=MUTED, labelsize=7)
        for sp in axt.spines.values():
            sp.set_color("#3a3a37")
        axt.grid(color="#2b2b29", lw=.6)
        zone = " (inside a congestion zone)" if probs[jid].get("zone_of") is not None else ""
        axm.set_title(f"jam {jid}{zone}, step {P['t0']}, {len(P['members'])} fleets\n"
                      f"driven {drv_rec:.1f} hops (recorded) vs {drv_plan:.0f} (collapsed plan), "
                      f"shortest {sum(P['shortest'].values())}", color=INK, fontsize=9, loc="left")
    fig.text(0.5, 0.005, "thin: what each fleet actually drove   ·   dashed: the collapsed plan, drawn at full speed "
             "(2 steps per hop; FLOWRRA brakes near others, so compare distance, not time)   ·   ● start  □ exit   ·   "
             "grey: racks", color=MUTED, ha="center", fontsize=7.5)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(a.out, facecolor=SURFACE)
    print(f"[figures] {len(ids)} jams -> {a.out}")


if __name__ == "__main__":
    main()
