"""
final_report.py -- tables, figures and the prediction score for the final
deploy_eval (build log entry 44).

    python final_report.py --clean final_clean.csv --full final_full.csv --out report

WRITES (into --out)
  tables.md                     every table below, in plain Markdown
  t1_headline_<cond>.csv        per arm: book delivered, lead time, collisions,
                                guard stops, L3, near misses with people, rescues,
                                planning time
  t2_by_map_<cond>.csv          per arm x map: delivered and lead time
  t3_by_fleet_<cond>.csv        per arm x fleet size: delivered and lead time
  t4_paired_<cond>.csv          every arm against ladder4 and against RULES, paired
                                on (map, fleet, seed): mean difference, days
                                better / worse, Wilcoxon p
  t5_predictions.md             P1-P6, each scored PASS / FAIL with its evidence
  fig_delivered.png             book delivered against fleet size, one panel per
                                map, clean on top and full below
  fig_lead.png                  the same for lead time
  fig_people.png                near misses with people per day (full)
  fig_guard.png                 guard stops per day, both conditions

LEAD TIME, HONESTLY. stream_book_lead_mean averages DELIVERED orders only, so an
arm that delivers less can look faster. Every table here uses lead_cens: each
undelivered order is counted at the least it can have waited -- from the end of
the release window (500 steps) to the end of the day. It is a lower bound for
those orders, so it still flatters an arm that leaves orders undelivered, only
much less. The delivered-only mean is kept beside it (lead_delivered).

FIRM = Wilcoxon signed-rank p < 0.05 on the paired days, with the mean in the
stated direction. Comparisons are paired on (map, k, seed).
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd

ARMS = ["ladder4", "ladder6", "ladder7", "RULES", "PIBT", "RHCR-PP", "RHCR-PIBT", "RHCR-PIBT+naive"]
BASELINES = ["PIBT", "RHCR-PP", "RHCR-PIBT", "RHCR-PIBT+naive"]
# The validated reference palette, light mode, in its fixed slot order (dataviz
# skill; checked with validate_palette.js: all checks pass, contrast WARN on three
# light hues -> every figure also has distinct markers and a legend, and every
# number is in tables.md).
COLOR = dict(zip(ARMS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                        "#e87ba4", "#008300", "#6250d6", "#e34948"]))
MARKER = dict(zip(ARMS, ["o", "s", "^", "D", "v", "P", "X", "*"]))
INK, INK2, MUTED, GRID, SURFACE = "#1f1f1d", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
RELEASE_STEPS = 500
MAP_NAMES = {"25_5_2_5_2_1": "small", "25_10_5_10_5_2": "medium",
             "50_20_5_10_5_2": "large", "Kiva_2D_46_33_1": "Kiva-style"}
KEY = ["map", "k", "seed"]


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, low_memory=False).copy()
    orders = d["stream_book_orders"].astype(float)
    deliv = d["stream_book_delivered"].astype(float)
    und = (orders - deliv).clip(lower=0)
    wait_min = (d["steps_run"].astype(float) - RELEASE_STEPS).clip(lower=0)
    d["delivered_pct"] = 100.0 * d["stream_book_fill"].astype(float)
    d["lead_delivered"] = d["stream_book_lead_mean"].astype(float)
    d["lead_cens"] = ((deliv * d["lead_delivered"].fillna(0.0) + und * wait_min)
                      / orders.where(orders > 0))
    d["map_name"] = d["map"].map(MAP_NAMES).fillna(d["map"])
    return d


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import wilcoxon
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.allclose(a, b):
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def paired(d: pd.DataFrame, arm: str, ref: str, col: str, lower_better: bool,
           subset=None) -> Dict[str, float]:
    x = d[d.arm == arm].set_index(KEY)[col]
    y = d[d.arm == ref].set_index(KEY)[col]
    idx = x.index.intersection(y.index)
    if subset is not None:
        idx = [i for i in idx if subset(i)]
    if len(idx) == 0:
        return {"n": 0}
    a, b = x.loc[idx].astype(float).values, y.loc[idx].astype(float).values
    diff = a - b
    better = int(np.sum(diff < 0)) if lower_better else int(np.sum(diff > 0))
    worse = int(np.sum(diff > 0)) if lower_better else int(np.sum(diff < 0))
    return {"n": len(idx), "arm_mean": float(np.nanmean(a)), "ref_mean": float(np.nanmean(b)),
            "diff": float(np.nanmean(diff)), "better": better, "worse": worse,
            "p": wilcoxon_p(a, b)}


def firm(r: Dict[str, float], lower_better: bool) -> bool:
    if r.get("n", 0) == 0 or not np.isfinite(r.get("p", float("nan"))):
        return False
    good = r["diff"] < 0 if lower_better else r["diff"] > 0
    return bool(good and r["p"] < 0.05)


# ----------------------------------------------------------------------------- tables
def headline(d: pd.DataFrame, cond: str) -> pd.DataFrame:
    cols = {"delivered_pct": "delivered %", "lead_cens": "lead time",
            "lead_delivered": "lead (delivered only)", "collisions": "collisions/day",
            "guard_stops": "guard stops/day", "ladder_l3_fired": "L3/day"}
    if cond == "full":
        cols.update({"obstacle_near_miss_steps": "near misses with people/day",
                     "errors_injected": "failures/day", "handovers_completed": "rescues/day",
                     "orders_lost_naive": "orders lost (naive)/day"})
    if "planner_ms_per_step" in d:
        cols["planner_ms_per_step"] = "planning ms/step"
    t = d.groupby("arm")[list(cols)].mean().rename(columns=cols)
    return t.reindex([a for a in ARMS if a in t.index]).round(2)


def by(d: pd.DataFrame, what: str) -> pd.DataFrame:
    t = d.pivot_table(index="arm", columns=what, values=["delivered_pct", "lead_cens"],
                      aggfunc="mean").round(1)
    return t.reindex([a for a in ARMS if a in t.index])


def paired_table(d: pd.DataFrame, cond: str) -> pd.DataFrame:
    rows = []
    measures = [("delivered_pct", False), ("lead_cens", True), ("guard_stops", True),
                ("collisions", True)]
    if cond == "full":
        measures.append(("obstacle_near_miss_steps", True))
    for ref in ["ladder4", "RULES"]:
        for arm in ARMS:
            if arm == ref or arm not in set(d.arm):
                continue
            for col, lower in measures:
                r = paired(d, arm, ref, col, lower)
                if r.get("n", 0):
                    rows.append({"against": ref, "arm": arm, "measure": col,
                                 "arm_mean": round(r["arm_mean"], 2),
                                 "ref_mean": round(r["ref_mean"], 2),
                                 "diff": round(r["diff"], 2), "days_better": r["better"],
                                 "days_worse": r["worse"], "n": r["n"],
                                 "p": (round(r["p"], 4) if np.isfinite(r["p"]) else np.nan),
                                 "firm": firm(r, lower)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- predictions
def predictions(c: pd.DataFrame, f: pd.DataFrame) -> List[str]:
    out = []
    # P1
    worst = {cond: d.groupby("arm")["collisions"].mean() for cond, d in (("clean", c), ("full", f))}
    ok = all((s < 1.0).all() for s in worst.values())
    mx = max(float(s.max()) for s in worst.values())
    out.append(f"**P1 {'PASS' if ok else 'FAIL'}.** Collisions per day: highest arm mean "
               f"{mx:.3f} (clean max {worst['clean'].max():.3f}, full max {worst['full'].max():.3f}).")
    # P2
    hi = lambda i: i[1] in (60, 80)
    res = []
    for arm in ("ladder4", "ladder6"):
        for b in BASELINES:
            r = paired(f, arm, b, "delivered_pct", False, hi)
            res.append((arm, b, r, firm(r, False)))
    ok = all(x[3] for x in res)
    out.append(f"**P2 {'PASS' if ok else 'FAIL'}.** Full, 60-80 robots, delivered %: " + "; ".join(
        f"{a} vs {b} {r['arm_mean']:.1f} / {r['ref_mean']:.1f} (p {r['p']:.3f}){'' if fm else ' -- not firm'}"
        for a, b, r, fm in res))
    # P3
    gs = {cond: d.groupby("arm")["guard_stops"].mean() for cond, d in (("clean", c), ("full", f))}
    nm = f.groupby("arm")["obstacle_near_miss_steps"].mean()
    ok = (all(s.idxmin() == "PIBT" for s in gs.values()) and nm.idxmin() == "PIBT")
    out.append(f"**P3 {'PASS' if ok else 'FAIL'}.** Fewest guard stops: clean {gs['clean'].idxmin()} "
               f"({gs['clean'].min():.1f}), full {gs['full'].idxmin()} ({gs['full'].min():.1f}); "
               f"fewest near misses with people (full): {nm.idxmin()} ({nm.min():.1f}; "
               f"next {nm.drop(nm.idxmin()).idxmin()} {nm.drop(nm.idxmin()).min():.1f}).")
    # P4
    parts, ok = [], True
    for k in (20, 40):
        r = paired(c, "PIBT", "RULES", "lead_cens", True, lambda i, k=k: i[1] == k)
        fm = firm(r, True)
        ok &= fm
        parts.append(f"{k} robots: PIBT {r['arm_mean']:.1f} vs RULES {r['ref_mean']:.1f} (p {r['p']:.3f})"
                     + ("" if fm else " -- not firm"))
    out.append(f"**P4 {'PASS' if ok else 'FAIL'}.** Clean lead time: " + "; ".join(parts) + ".")
    # P5
    ratios, ok = [], True
    for cond, d in (("clean", c), ("full", f)):
        g = d.groupby("arm")["guard_stops"].mean()
        ratio = g["RHCR-PIBT"] / max(g["RHCR-PP"], 1e-9)
        ok &= ratio >= 5
        r = paired(d, "RHCR-PP", "RHCR-PIBT", "delivered_pct", False, hi)
        fm = firm(r, False)
        ok &= fm
        ratios.append(f"{cond}: guard stops {g['RHCR-PP']:.0f} vs {g['RHCR-PIBT']:.0f} ({ratio:.1f}x); "
                      f"delivered at 60-80 {r['arm_mean']:.1f} vs {r['ref_mean']:.1f} (p {r['p']:.3f})"
                      + ("" if fm else " -- not firm"))
    out.append(f"**P5 {'PASS' if ok else 'FAIL'}.** " + "; ".join(ratios) + ".")
    # P6
    nv = f[f.arm == "RHCR-PIBT+naive"].set_index(KEY)
    rp = f[f.arm == "RHCR-PIBT"].set_index(KEY)
    idx = nv.index.intersection(rp.index)
    more = int((nv.loc[idx, "delivered_pct"] > rp.loc[idx, "delivered_pct"] + 1e-9).sum())
    quiet = idx[(nv.loc[idx, "orders_lost_naive"].fillna(0) == 0)
                & (nv.loc[idx, "orders_pickups_abandoned"].fillna(0) == 0)]
    same = int((nv.loc[quiet, "delivered_pct"] == rp.loc[quiet, "delivered_pct"]).sum())
    others_lost = float(f[f.arm != "RHCR-PIBT+naive"]["orders_lost_naive"].fillna(0).sum())
    ok = more == 0 and same == len(quiet) and others_lost == 0
    out.append(f"**P6 {'PASS' if ok else 'FAIL'}.** Naive delivered more than RHCR-PIBT on {more} of "
               f"{len(idx)} days; on the {len(quiet)} days with nothing lost or abandoned it was identical on "
               f"{same}; orders lost by any other arm: {others_lost:.0f}.")
    return out


# ----------------------------------------------------------------------------- figures
def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)


def fig_by_fleet(c: pd.DataFrame, f: pd.DataFrame, col: str, ylabel: str, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    maps = [m for m in MAP_NAMES if m in set(c["map"]) | set(f["map"])]
    fig, axes = plt.subplots(2, len(maps), figsize=(4.2 * len(maps), 7.4), sharex=True,
                             facecolor=SURFACE)
    for r, (cond, d) in enumerate((("clean", c), ("full", f))):
        for j, m in enumerate(maps):
            ax = axes[r][j]
            _style(ax)
            g = d[d["map"] == m].groupby(["arm", "k"])[col].mean()
            for arm in ARMS:
                if arm not in g.index.get_level_values(0):
                    continue
                s = g.loc[arm]
                ax.plot(s.index, s.values, color=COLOR[arm], linewidth=2, marker=MARKER[arm],
                        markersize=8, markeredgecolor=SURFACE, markeredgewidth=1.5, label=arm)
            ax.set_title(f"{MAP_NAMES[m]} map, {cond}", color=INK, fontsize=11, loc="left")
            if j == 0:
                ax.set_ylabel(ylabel, color=INK2)
            if r == 1:
                ax.set_xlabel("robots", color=INK2)
            ax.set_xticks(sorted(d["k"].unique()))
    h, l = axes[1][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=8, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_bars(series: Dict[str, pd.Series], path: str, title: str, xlabel: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(series)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.2), facecolor=SURFACE, squeeze=False)
    for ax, (name, s) in zip(axes[0], series.items()):
        _style(ax)
        ax.grid(True, axis="x", color=GRID)
        ax.grid(False, axis="y")
        s = s.reindex([a for a in ARMS if a in s.index])
        y = np.arange(len(s))[::-1]
        ax.barh(y, s.values, color=[COLOR[a] for a in s.index], height=0.62,
                edgecolor=SURFACE, linewidth=2)
        ax.set_yticks(y)
        ax.set_yticklabels(s.index, color=INK)
        for yi, v in zip(y, s.values):
            ax.text(v, yi, f"  {v:,.0f}", va="center", ha="left", color=INK2, fontsize=9)
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_xlim(0, max(1.0, float(np.nanmax(s.values))) * 1.18)
    fig.suptitle(title, color=INK, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown()
    except ImportError:                      # tabulate missing: a plain fallback
        return "```\n" + df.to_string() + "\n```"


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--clean", default="final_clean.csv")
    ap.add_argument("--full", default="final_full.csv")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    c, f = load(args.clean), load(args.full)
    lines = ["# Final deploy_eval: tables (build log entry 44)", "",
             "Lead time counts each undelivered order at its minimum wait (end of the 500-step "
             "release window to the end of the day); `lead (delivered only)` is the plain mean. "
             "Firm = Wilcoxon p < 0.05 on days paired by map, fleet size and seed.", ""]
    for cond, d in (("clean", c), ("full", f)):
        t1 = headline(d, cond); t1.to_csv(os.path.join(args.out, f"t1_headline_{cond}.csv"))
        t2 = by(d, "map_name"); t2.to_csv(os.path.join(args.out, f"t2_by_map_{cond}.csv"))
        t3 = by(d, "k"); t3.to_csv(os.path.join(args.out, f"t3_by_fleet_{cond}.csv"))
        t4 = paired_table(d, cond); t4.to_csv(os.path.join(args.out, f"t4_paired_{cond}.csv"), index=False)
        n_days = d.groupby("arm").size().max()
        lines += [f"## {cond.capitalize()} ({n_days} days per arm)", "",
                  "### Headline (means per day)", "", md(t1), "",
                  "### By map", "", md(t2), "", "### By fleet size", "", md(t3), "",
                  "### Paired against ladder4 and against RULES", "",
                  md(t4.set_index(["against", "arm", "measure"])), ""]
    preds = predictions(c, f)
    with open(os.path.join(args.out, "t5_predictions.md"), "w") as fh:
        fh.write("# Predictions P1-P6, scored\n\n" + "\n\n".join(preds) + "\n")
    lines += ["## Predictions P1-P6, scored", ""] + [p + "\n" for p in preds]
    with open(os.path.join(args.out, "tables.md"), "w") as fh:
        fh.write("\n".join(lines))
    fig_by_fleet(c, f, "delivered_pct", "book delivered (%)",
                 os.path.join(args.out, "fig_delivered.png"),
                 "Book delivered by fleet size: clean (top), full (bottom). Each panel has its own y-scale")
    fig_by_fleet(c, f, "lead_cens", "lead time (steps)",
                 os.path.join(args.out, "fig_lead.png"),
                 "Lead time by fleet size (undelivered orders at their minimum wait): clean (top), full (bottom). Lower is better")
    fig_bars({"full condition": f.groupby("arm")["obstacle_near_miss_steps"].mean()},
             os.path.join(args.out, "fig_people.png"),
             "Near misses with people per day (full condition). Lower is better", "steps per day")
    fig_bars({"clean": c.groupby("arm")["guard_stops"].mean(),
              "full": f.groupby("arm")["guard_stops"].mean()},
             os.path.join(args.out, "fig_guard.png"),
             "Guard stops per day: how often the safety guard had to stop a move", "stops per day")
    print("\n".join(preds))
    print(f"\nwrote {args.out}/: tables.md, t1-t5, fig_delivered.png, fig_lead.png, "
          f"fig_people.png, fig_guard.png")


if __name__ == "__main__":
    main()
