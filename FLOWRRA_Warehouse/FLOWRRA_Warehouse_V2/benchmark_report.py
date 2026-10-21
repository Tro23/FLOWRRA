"""
benchmark_report.py -- what a map allows, and where FLOWRRA sits inside it.

    # the full report for one map
    python benchmark_report.py --map 50_5_5_10_5_2 \\
        --run cold_run25=curriculum_metrics_cold_run25.csv \\
        --run cold_run24=curriculum_metrics_cold_run24.csv \\
        --capacity "driver_results/driver_*.csv" --out report_50

    # anatomy only, several maps side by side (e.g. kiva against 50_)
    python benchmark_report.py --anatomy kiva,kiva_3f,50_5_5_10_5_2 --out anatomy

THREE QUESTIONS, IN ORDER

1. ANATOMY -- what kind of map is this? Floors and lift shafts; junctions (three
   or more neighbours); CORRIDORS, the single-lane chains between junctions,
   where two fleets cannot pass; AISLES, the straight lines of cells, which may
   have junctions along them (side exits); dead ends; cut cells whose loss
   splits the map; and MANOEUVRABILITY -- how far a fleet must back out to
   reach a junction and yield, and where two fleets can pass side by side.

2. CEILINGS -- what could N fleets deliver?
     the IDEAL      fleets x steps x speed / conflict-free cycle: what they
                    would deliver if no fleet ever met another. An upper bound
                    nobody reaches in dense traffic.
     the CAPACITY   what the shortest-path driver with the conflict rules
                    actually delivers at several fleet counts
                    (drive_shortest_path.py), with and without failures: where
                    the map saturates. Shortest paths do not spread load, so a
                    policy that routes around congestion can beat it.

3. FLOWRRA -- every run CSV (curriculum_metrics.csv), in blocks of episodes:
   deliveries, efficiency against the ideal, collisions, holds, how often the
   rules and recovery overrode the policy, the policy's own wait share, and
   failures and rescues when the run has them. Paired by episode against the
   first run, and placed against both ceilings.

Writes <out>.md (the report), <out>_summary.csv, and <out>_episodes.png and
<out>_capacity.png when there is something to draw.
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import re
from collections import Counter, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


# ====================================================================== anatomy
def load_map(maps_dir: str, name: str):
    nodes = pd.read_csv(os.path.join(maps_dir, f"{name}_Nodes.csv"))
    edges = pd.read_csv(os.path.join(maps_dir, f"{name}_Edges.csv"))
    pos = {r.NodeId: (int(r.X), int(r.Y), int(r.Z)) for r in nodes.itertuples()}
    G = nx.Graph()
    G.add_nodes_from(pos)
    for r in edges.itertuples():
        if r.nodeFrom in pos and r.nodeTo in pos:
            G.add_edge(r.nodeFrom, r.nodeTo)
    return G, pos


def _axis(pa, pb) -> int:
    return next(i for i in range(3) if pa[i] != pb[i])


def anatomy(G, pos, vision: int = 10, fleets: Optional[int] = None) -> Dict[str, float]:
    n = G.number_of_nodes()
    deg = dict(G.degree())
    out: Dict[str, float] = {"cells": n, "edges": G.number_of_edges(),
                             "mean_degree": round(2 * G.number_of_edges() / max(1, n), 2)}

    # floors: Z levels holding a real share of the cells; the rest are shafts
    zc = Counter(p[2] for p in pos.values())
    big = max(zc.values())
    floors = sorted(z for z, c in zc.items() if c >= 0.2 * big)
    shaft_cells = [v for v, p in pos.items() if p[2] not in floors]
    shafts = list(nx.connected_components(G.subgraph(shaft_cells))) if shaft_cells else []
    out.update(floors=len(floors), shaft_segments=len(shafts),
               shaft_cells_pct=round(100 * len(shaft_cells) / n, 1),
               vertical_edges=sum(1 for a, b in G.edges() if pos[a][2] != pos[b][2]))

    junctions = [v for v in G if deg[v] >= 3]
    out.update(junction_pct=round(100 * len(junctions) / n, 1),
               corridor_cell_pct=round(100 * sum(1 for v in G if deg[v] == 2) / n, 1),
               dead_end_pct=round(100 * sum(1 for v in G if deg[v] == 1) / n, 1))

    # corridors: chains of two-neighbour cells between junctions
    lens, bent = [], 0
    for comp in nx.connected_components(G.subgraph([v for v in G if deg[v] == 2])):
        comp = list(comp)
        lens.append(len(comp))
        axes = {_axis(pos[a], pos[b]) for a, b in G.subgraph(comp).edges()}
        bent += len(axes) > 1
    lens = np.array(lens or [0])
    out.update(corridors=int((lens > 0).sum()), corridor_len_mean=round(float(lens.mean()), 1),
               corridor_len_max=int(lens.max()), corridors_bent_pct=round(100 * bent / max(1, len(lens)), 1),
               cells_in_lanes_5plus_pct=round(100 * lens[lens >= 5].sum() / n, 1),
               corridors_seen_whole_pct=round(100 * float((lens + 1 <= vision).mean()), 1))

    # aisles: maximal straight runs along X or Y within a level; side exits = junctions on them
    by = {p: v for v, p in pos.items()}
    aisles = []
    for ax in (0, 1):
        seen = set()
        for v, p in pos.items():
            if (v, ax) in seen:
                continue
            q = list(p); q[ax] -= 1
            prev = by.get(tuple(q))
            if prev is not None and G.has_edge(prev, v):
                continue                              # not the start of a run
            run, cur = [v], v
            while True:
                q = list(pos[cur]); q[ax] += 1
                nxt = by.get(tuple(q))
                if nxt is None or not G.has_edge(cur, nxt):
                    break
                run.append(nxt); cur = nxt
            for c in run:
                seen.add((c, ax))
            if len(run) >= 3:
                exits = sum(1 for c in run if any(_axis(pos[c], pos[w]) != ax for w in G[c]))
                aisles.append((len(run), exits))
    if aisles:
        al = np.array(aisles)
        out.update(aisles=len(al), aisle_len_mean=round(float(al[:, 0].mean()), 1),
                   aisle_len_max=int(al[:, 0].max()),
                   aisle_side_exit_spacing=round(float(al[:, 0].sum() / max(1, al[:, 1].sum())), 1))

    # manoeuvrability
    if junctions:
        dist = nx.multi_source_dijkstra_path_length(G, junctions)
        d = np.array([dist.get(v, np.inf) for v in G])
        d = d[np.isfinite(d)]
        out.update(backout_hops_mean=round(float(d.mean()), 2), backout_hops_p90=round(float(np.percentile(d, 90)), 1),
                   backout_hops_max=int(d.max()),
                   room_to_yield_pct=round(100 * float((d <= 1).mean()), 1))
    passing = set()
    for v, (x, y, z) in pos.items():
        sq = [by.get((x + dx, y + dy, z)) for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1))]
        if None in sq:
            continue
        a, b, c, e = sq
        if G.has_edge(a, b) and G.has_edge(a, c) and G.has_edge(b, e) and G.has_edge(c, e):
            passing.update(sq)
    out["passing_room_pct"] = round(100 * len(passing) / n, 1)
    arts = list(nx.articulation_points(G))
    out.update(cut_cells_pct=round(100 * len(arts) / n, 1),
               largest_biconnected_pct=round(100 * max((len(b) for b in nx.biconnected_components(G)),
                                                       default=0) / n, 1))
    rng = np.random.default_rng(0)
    sample = rng.choice(list(G.nodes()), min(300, n), replace=False)
    tot = far = 0
    for v in sample:
        hd = nx.single_source_shortest_path_length(G, v, cutoff=12)
        x, y, z = pos[v]
        for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (2, 0, 0), (-2, 0, 0), (0, 2, 0),
                           (0, -2, 0), (1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0)):
            w = by.get((x + dx, y + dy, z + dz))
            if w is None:
                continue
            tot += 1
            far += hd.get(w, 99) > 2 * (abs(dx) + abs(dy) + abs(dz))
    out["phantom_neighbour_pct"] = round(100 * far / max(1, tot), 1)
    if fleets:
        out["fleets"] = fleets
        out["occupancy_pct"] = round(100 * fleets / n, 2)
    return out


def describe(a: Dict[str, float]) -> List[str]:
    """Plain sentences about what the anatomy means for traffic."""
    s = []
    lvl = "one level" if a["floors"] == 1 else (f"{a['floors']} floors joined by {a['shaft_segments']} "
                                                f"single-lane shaft segments ({a['shaft_cells_pct']}% of cells)")
    s.append(f"{a['cells']:,} cells on {lvl}; mean degree {a['mean_degree']}.")
    s.append(f"{a['corridor_cell_pct']}% of cells have exactly two neighbours; {a['corridors']} corridors "
             f"(single-lane chains between junctions), mean {a['corridor_len_mean']} cells, longest "
             f"{a['corridor_len_max']}; {a['cells_in_lanes_5plus_pct']}% of all cells lie in lanes of five or more.")
    if "aisles" in a:
        s.append(f"{a['aisles']} straight aisles, mean {a['aisle_len_mean']} cells (longest {a['aisle_len_max']}), "
                 f"one side exit every {a['aisle_side_exit_spacing']} cells.")
    if "backout_hops_mean" in a:
        s.append(f"To yield, a fleet backs out {a['backout_hops_mean']} hops on average to the nearest junction "
                 f"(90% within {a['backout_hops_p90']}, worst {a['backout_hops_max']}); "
                 f"{a['room_to_yield_pct']}% of cells are on or next to one.")
    s.append(f"Two fleets can pass side by side on {a['passing_room_pct']}% of cells; "
             f"{a['cut_cells_pct']}% of cells would split the map if blocked; "
             f"{a['phantom_neighbour_pct']}% of grid-near cells are more than twice as far by path.")
    single = a["corridor_cell_pct"] >= 50 and a["passing_room_pct"] < 20
    s.append("Traffic character: " + (
        "single-lane dominated -- conflicts are head-on and must be resolved by waiting, backing out, "
        "or routing around, not by stepping aside." if single else
        "open -- most conflicts can be resolved by stepping aside; the challenge is weaving many fleets."))
    return s


# ====================================================================== ceilings
def load_capacity(patterns: List[str]) -> pd.DataFrame:
    files = sorted({f for p in patterns for f in glob.glob(p) if "driver_events" not in f})
    if not files:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d["family"] = d["arm"].astype(str).str.replace(r"_N\d+(?=$|_)", "", regex=True)
    d["family"] = d["family"] + np.where(d.get("recovery", "never") != "never",
                                         " (preempt " + d.get("recovery", "").astype(str) + ")", "")
    # the driver is deterministic: the same arm, size, seed and length twice is one run
    return d.drop_duplicates(["family", "fleets", "seed", "steps"]).reset_index(drop=True)


def capacity_table(cap: pd.DataFrame) -> pd.DataFrame:
    if cap.empty:
        return cap
    g = cap.groupby(["family", "fleets"])
    t = g.agg(steps=("steps", "first"), seeds=("seed", "nunique"),
              deliveries=("deliveries", "mean"), deliveries_sd=("deliveries", "std"),
              ideal=("ideal_deliveries", "mean"), collisions=("collisions", "mean"),
              holds=("holds", "mean")).reset_index()
    if "errors_injected" in cap.columns:
        t = t.merge(g.agg(failures=("errors_injected", "mean"), rescues=("handovers", "mean")).reset_index(),
                    on=["family", "fleets"])
    t["efficiency_pct"] = 100 * t["deliveries"] / t["ideal"]
    t["per_fleet"] = t["deliveries"] / t["fleets"]
    return t


# ====================================================================== runs
def load_run(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    return d.sort_values("episode").reset_index(drop=True)


def block_table(d: pd.DataFrame, block: int) -> pd.DataFrame:
    d = d.copy()
    d["block"] = (d["episode"] - 1) // block
    def col(c, f="mean"):
        return (c, f) if c in d.columns else ("episode", "size")
    agg = {"episodes": ("episode", lambda e: f"{e.min()}-{e.max()}"),
           "deliveries": ("completed", "mean"), "collisions": ("collisions", "mean")}
    for name, c in (("efficiency", "stream_efficiency"), ("ideal", "stream_ideal_deliveries"),
                    ("window", "stream_order_window"), ("holds", "conflict_holds"),
                    ("holds_at_cap", "conflict_holds_at_cap"), ("repeats", "conflict_repeat_offences"),
                    ("rule_overrides", "choice_over_rule"), ("hold_overrides", "choice_over_hold"),
                    ("policy_wait", "choice_policy_wait_share"), ("done_wait", "choice_exec_wait_share"),
                    ("failures", "errors_injected"), ("rescues", "handovers_completed")):
        if c in d.columns:
            agg[name] = (c, "mean")
    t = d.groupby("block").agg(**agg).reset_index(drop=True)
    for c in ("failures", "rescues"):            # only when the run has them
        if c in t.columns and not (t[c] > 0).any():
            t = t.drop(columns=c)
    t["coll_per_100_del"] = 100 * t["collisions"] / t["deliveries"].clip(lower=1)
    return t


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str) -> Optional[Dict[str, float]]:
    m = a[["episode", col]].merge(b[["episode", col]], on="episode", suffixes=("_a", "_b"))
    if len(m) < 3:
        return None
    diff = m[f"{col}_a"] - m[f"{col}_b"]
    se = diff.std(ddof=1) / math.sqrt(len(diff))
    return {"n": len(m), "mean_a": m[f"{col}_a"].mean(), "mean_b": m[f"{col}_b"].mean(),
            "diff": diff.mean(), "t": diff.mean() / se if se > 0 else float("nan"),
            "a_higher": int((diff > 0).sum())}


# ====================================================================== report
def fmt(x, p=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    if isinstance(x, (float, np.floating)) and float(x).is_integer() and abs(x) >= 1:
        return f"{int(x):,}"
    return f"{x:,.{p}f}" if isinstance(x, (float, np.floating)) else f"{x:,}" if isinstance(x, (int, np.integer)) else str(x)


def df_md(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional `tabulate` dependency."""
    cols = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, r in df.iterrows():
        cells = ([str(idx)] if index else []) + [fmt(v, 1) if isinstance(v, (float, np.floating)) else str(v)
                                                for v in r.values]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_table(df: pd.DataFrame, cols: List[Tuple[str, str, int]]) -> str:
    cols = [c for c in cols if c[0] in df.columns]
    lines = ["| " + " | ".join(h for _, h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, _h, p in cols:
            v = r[c]
            if c in ("policy_wait", "done_wait", "efficiency") and isinstance(v, (float, np.floating)):
                cells.append(f"{100 * v:.0f}%" if np.isfinite(v) else "--")
            else:
                cells.append(fmt(v, p))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--map", default=None, help="the map the runs were trained on")
    ap.add_argument("--anatomy", default=None, help="comma list: anatomy only, maps side by side")
    ap.add_argument("--run", action="append", default=[], help="label=path/to/curriculum_metrics.csv")
    ap.add_argument("--capacity", action="append", default=[], help="glob of driver CSVs")
    ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--vision", type=int, default=10)
    ap.add_argument("--out", default="benchmark_report")
    args = ap.parse_args()
    md: List[str] = []
    summary: List[Dict] = []

    if args.anatomy:
        rows = {}
        for name in [m.strip() for m in args.anatomy.split(",") if m.strip()]:
            G, pos = load_map(args.maps_dir, name)
            rows[name] = anatomy(G, pos, args.vision)
        t = pd.DataFrame(rows)
        md.append("# Map anatomy, side by side\n")
        md.append(df_md(t))
        for name, a in rows.items():
            md.append(f"\n**{name}.** " + " ".join(describe(a)))
        t.T.to_csv(f"{args.out}_summary.csv")
        open(f"{args.out}.md", "w").write("\n".join(md) + "\n")
        print("\n".join(md))
        print(f"\nwrote {args.out}.md, {args.out}_summary.csv")
        return

    runs = []
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs.append((label, load_run(path)))
    fleets = int(runs[0][1]["agents"].iloc[0]) if runs and "agents" in runs[0][1].columns else None

    md.append(f"# Benchmark report -- {args.map or 'map'}\n")
    # ---- 1 anatomy
    if args.map:
        G, pos = load_map(args.maps_dir, args.map)
        a = anatomy(G, pos, args.vision, fleets)
        md.append("## 1. The map\n")
        md.extend(f"- {x}" for x in describe(a))
        md.append("\n" + df_md(pd.DataFrame([a]).T.rename(columns={0: "value"})))
        summary.append({"section": "anatomy", **a})

    # ---- 2 ceilings
    cap = load_capacity(args.capacity)
    ct = capacity_table(cap)
    if not ct.empty:
        md.append("\n## 2. Ceilings: what N fleets could deliver\n")
        md.append("The IDEAL assumes no fleet ever meets another. The CAPACITY is what the "
                  "shortest-path driver with the conflict rules delivers: where the map saturates. "
                  "A policy that spreads load across routes can beat the capacity; nothing reaches the ideal "
                  "in dense traffic.\n")
        md.append(md_table(ct, [("family", "arm", 0), ("fleets", "fleets", 0), ("seeds", "seeds", 0),
                                ("steps", "steps", 0), ("deliveries", "deliveries", 1),
                                ("deliveries_sd", "sd", 1), ("ideal", "ideal", 0),
                                ("efficiency_pct", "% of ideal", 0), ("per_fleet", "per fleet", 2),
                                ("collisions", "collisions", 1), ("holds", "holds", 1),
                                ("failures", "failures", 1), ("rescues", "rescues", 1)]))
        for fam, g in ct.groupby("family"):
            g = g.sort_values("fleets")
            if len(g) >= 2:
                steps = [f"{int(r.fleets)} fleets {r.deliveries:.0f} ({r.efficiency_pct:.0f}% of ideal, "
                         f"{r.per_fleet:.2f} each)" for r in g.itertuples()]
                gain = [(b.deliveries - a_.deliveries) / max(1, b.fleets - a_.fleets)
                        for a_, b in zip(g.itertuples(), list(g.itertuples())[1:])]
                md.append(f"\n- **{fam}:** " + "; ".join(steps) + ". Each extra fleet adds "
                          + ", ".join(f"{x:.2f}" for x in gain) + " deliveries between those sizes.")
        for r in ct.itertuples():
            summary.append({"section": "capacity", **r._asdict()})

    # ---- 3 runs
    if runs:
        md.append(f"\n## 3. FLOWRRA runs, in blocks of {args.block} episodes\n")
        for label, d in runs:
            bt = block_table(d, args.block)
            n_ag = int(d["agents"].iloc[0]) if "agents" in d.columns else None
            md.append(f"\n### {label} ({len(d)} episodes, {n_ag} fleets)\n")
            md.append(md_table(bt, [("episodes", "episodes", 0), ("window", "window", 1),
                                    ("deliveries", "deliveries", 1), ("ideal", "ideal", 0),
                                    ("efficiency", "of ideal", 0), ("collisions", "collisions", 1),
                                    ("coll_per_100_del", "coll/100 del", 1), ("holds", "holds", 0),
                                    ("holds_at_cap", "at cap", 0), ("rule_overrides", "rule overrides", 0),
                                    ("hold_overrides", "hold overrides", 0),
                                    ("policy_wait", "policy wait", 0), ("done_wait", "done wait", 0),
                                    ("failures", "failures", 1), ("rescues", "rescues", 1)]))
            if "failures" in bt.columns and bt["failures"].sum() > 0:
                md.append(f"\nFailures injected: {d['errors_injected'].sum()} over the run; "
                          f"orders recovered by handover: {d['handovers_completed'].sum()}.")
            for r in bt.to_dict("records"):
                summary.append({"section": f"run:{label}", **r})
            # place against capacity at the same fleet count
            if not ct.empty and n_ag is not None:
                same = ct[ct["fleets"] == n_ag]
                for r in same.itertuples():
                    last = bt.iloc[-1]
                    eff = last.get("efficiency", np.nan)
                    md.append(f"\n- Against the capacity at {n_ag} fleets ({r.family}, {r.steps} steps): "
                              f"{r.deliveries:.0f} deliveries, {r.efficiency_pct:.0f}% of its ideal. "
                              f"{label}'s last block: {last['deliveries']:.0f} deliveries, "
                              f"{100 * eff:.0f}% of its ideal"
                              + (f" (order window {last['window']:.0f}; the capacity runs use the final "
                                 f"window, so compare efficiency, not raw deliveries)." if "window" in last else "."))
        base_label, base = runs[0]
        for label, d in runs[1:]:
            md.append(f"\n### {base_label} against {label}, paired by episode\n")
            for col, name in (("completed", "deliveries"), ("collisions", "collisions")):
                p = paired(base, d, col)
                if p:
                    md.append(f"- {name}: {p['mean_a']:.1f} vs {p['mean_b']:.1f} over {p['n']} episodes, "
                              f"difference {p['diff']:+.1f} (t = {p['t']:.1f}); {base_label} higher in "
                              f"{p['a_higher']}/{p['n']}.")

    # ---- charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        if runs:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for label, d in runs:
                ax.plot(d["episode"], d["completed"], marker=".", label=label)
            if "stream_ideal_deliveries" in runs[0][1].columns:
                ax.plot(runs[0][1]["episode"], runs[0][1]["stream_ideal_deliveries"], "k--", lw=1,
                        label="conflict-free ideal")
            if not ct.empty and fleets is not None:
                for r in ct[ct["fleets"] == fleets].itertuples():
                    ax.axhline(r.deliveries, ls=":", lw=1.5, label=f"capacity, {r.family}")
            ax.set_xlabel("episode"); ax.set_ylabel("deliveries per episode")
            ax.set_title(f"{args.map or ''}: deliveries against the ceilings"); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(f"{args.out}_episodes.png", dpi=120); plt.close(fig)
        if not ct.empty:
            fig, ax = plt.subplots(figsize=(7, 4.5))
            for fam, g in ct.groupby("family"):
                g = g.sort_values("fleets")
                ax.errorbar(g["fleets"], g["deliveries"], yerr=g["deliveries_sd"].fillna(0), marker="o",
                            capsize=3, label=f"capacity: {fam}")
            g0 = ct.sort_values("fleets").drop_duplicates("fleets")
            ax.plot(g0["fleets"], g0["ideal"], "k--", lw=1, label="conflict-free ideal")
            star_colours = ["crimson", "forestgreen", "purple", "black", "darkorange"]
            for (label, d), colour in zip(runs, star_colours):
                bt = block_table(d, args.block)
                if "agents" in d.columns:
                    w = bt["window"].iloc[-1] if "window" in bt.columns else None
                    ax.scatter([int(d["agents"].iloc[0])], [bt["deliveries"].iloc[-1]], marker="*", s=180,
                               color=colour, zorder=5,
                               label=f"{label}, last block" + (f" (window {w:.0f})" if w is not None and np.isfinite(w) else ""))
            fig.text(0.01, 0.005, "Capacity runs use the final order window; a run's star sits at its own "
                     "window, so compare efficiency (% of each ideal), not height alone.", fontsize=7)
            ax.set_xlabel("fleets"); ax.set_ylabel("deliveries per episode")
            ax.set_title("Where the map saturates"); ax.legend(fontsize=8)
            fig.tight_layout(); fig.savefig(f"{args.out}_capacity.png", dpi=120); plt.close(fig)
    except ImportError:
        md.append("\n(matplotlib not installed: no charts)")

    open(f"{args.out}.md", "w").write("\n".join(md) + "\n")
    pd.DataFrame(summary).to_csv(f"{args.out}_summary.csv", index=False)
    print("\n".join(md))
    print(f"\nwrote {args.out}.md, {args.out}_summary.csv and charts")


if __name__ == "__main__":
    main()
