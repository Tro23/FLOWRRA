"""
collapse_eval.py -- Part B1: the same days with and without the collapse rung.

Each day is run by deploy_eval itself, with exactly the flags final_jobs.py
used (same world, keyed failures and people, book, assignment, world fixes),
once with the rung off and once with it on. Nothing in deploy_eval or
core_warehouse is edited: collapse_warehouse.install() wraps four methods, and
the rung exists only in the "on" runs. Pairs are matched on (map, fleets, seed).

USE (in FLOWRRA_Warehouse_V2, next to collapse_warehouse.py and jam_thrml.py)
    python collapse_eval.py --arm ladder4 --fleets 60 --seeds 0 --cond full
        -> collapse_ladder4_Kiva_2D_46_33_1_k60_full_thrml_s0.csv  (both runs, one row each)
    python collapse_eval.py --compare "collapse_ladder4_*_thrml_s*.csv"   # pairs, means, per-day table
    python collapse_eval.py --plans "collapse_ladder4_*_plans.csv"       # decision time by jam size
    python collapse_eval.py --head "aws_s*_on3.csv" "aws_s*_hold.csv"    # two rung versions, same days

    Two variants of the rung, compared on the same "off" rows: run the second with
    --collapse on, then --compare FILE_WITH_OFF FILE_OF_VARIANT (a later file's
    "on" row replaces an earlier one's for the same day).

    --collapse off|on|both   which runs (default both, one after the other)
    --big thrml|ladder|hold  clusters of 5-8 fleets: sampled with thrml, left to the ladder, or
                             held (the exact-plus-hold test)
    --solver plan|hold       plan: the rung; hold: the hold test (same clusters and one authority,
                             but members just wait -- no search, no sampling)
    --horizon 8  --max-exact 4  --max-fleets 8  --grace 2
    --steps N                a shorter day (testing only; not the scored day)

Arms: ladder4, ladder6, ladder7, RULES (the rung needs the ladder, so not the baselines).
"""
from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import sys

import numpy as np
import pandas as pd

SHOW = [("stream_book_delivered", "orders delivered", "{:.1f}"),
        ("ladder_collapse_travel_all", "distance driven, all fleets (cells)", "{:.0f}"),
        ("collisions", "collisions", "{:.2f}"),
        ("obstacle_near_miss_steps", "near-miss steps with people", "{:.1f}"),
        ("ladder_l3_fired", "L3 fired", "{:.1f}"),
        ("ladder_retrace_orders", "L1 retrace orders", "{:.1f}"),
        ("guard_stops", "intent-guard stops", "{:.1f}"),
        ("stream_book_lead_mean", "mean lead time per order (steps)", "{:.1f}"),
        ("runtime_s", "runtime (s)", "{:.0f}"),
        # DECISION TIME, per simulated step (wall clock on this machine)
        ("ladder_collapse_step_ms", "time per step, whole floor (ms)", "{:.1f}"),
        ("ladder_collapse_decide_ms", "  the fleets' own decision (ms)", "{:.1f}"),
        ("ladder_collapse_rung_ms", "  the collapse rung (ms)", "{:.1f}")]
RUNG = ["plans", "plans_exact", "plans_thrml", "plans_hold", "hold_released", "no_plan", "too_big", "completed", "replanned",
        "cancel_blocked", "cancel_goal", "cancel_gone", "cancel_l3", "cancel_people", "near_people", "max_size", "plan_seconds",
        "thrml_seconds", "ticks", "fleet_steps",
        # DECISION TIME per planning attempt, in milliseconds (found a plan or not)
        "exact_attempts", "exact_ms_median", "exact_ms_p95", "exact_ms_max",
        "thrml_attempts", "thrml_ms_median", "thrml_ms_p95", "thrml_ms_max",
        "thrml_compile_ms_median", "thrml_sample_ms_median"]


def compare(paths):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).copy()
    df["collapse"] = df.get("ladder_collapse_on", 0).fillna(0).astype(int)
    key = ["arm_base", "map", "k", "seed"]
    df["arm_base"] = df["arm"].str.replace("+collapse", "", regex=False)
    off = df[df.collapse == 0].drop_duplicates(key, keep="last").set_index(key)
    on = df[df.collapse == 1].drop_duplicates(key, keep="last").set_index(key)
    idx = off.index.intersection(on.index)
    if len(idx) == 0:
        print("[compare] no pairs: each day needs a row with the rung off and one with it on")
        return
    print(f"[compare] {len(idx)} paired days: " + ", ".join(f"{a} {m} k={k} seed={s}" for a, m, k, s in idx))
    table(off, on, idx, "rung off", "rung on")


def head(a_paths, b_paths):
    """Two versions of the rung, head to head on the same days (--head A B): each
    file set's rung-on rows, paired by day. E.g. the hold test against the rung."""
    key = ["arm_base", "map", "k", "seed"]

    def load(paths):
        df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True).copy()
        df = df[df.get("ladder_collapse_on", 0).fillna(0).astype(int) == 1].copy()
        df["arm_base"] = df["arm"].str.replace("+collapse", "", regex=False)
        return df.drop_duplicates(key, keep="last").set_index(key)
    A, B = load(a_paths), load(b_paths)
    idx = A.index.intersection(B.index)
    if len(idx) == 0:
        print("[head] no days in both sets")
        return
    print(f"[head] {len(idx)} paired days; A = {a_paths[0]}..., B = {b_paths[0]}...")
    table(A, B, idx, "A", "B")
    print("\nthe rung in B (means per day):")
    for c in RUNG:
        col = "ladder_collapse_" + c
        if col in B:
            print(f"  {c:24} {B.loc[idx, col].astype(float).mean():.1f}")


def table(off, on, idx, la, lb):
    # Paired Wilcoxon signed-rank test on the per-day differences, as in
    # deploy_eval (the agreed rule: "firm" means p < 0.05). With fewer than 5
    # days it cannot reach 0.05, so it is not shown.
    def pval(a, b):
        d = (b - a).to_numpy()
        if len(d) < 5 or np.allclose(d, 0):
            return float("nan")
        try:
            from scipy.stats import wilcoxon
            return float(wilcoxon(d).pvalue)
        except Exception:
            return float("nan")

    def row(label, a, b, fmt):
        p = pval(a, b)
        ps = "" if p != p else (f"{p:.3f}" + (" firm" if p < 0.05 else ""))
        per = ", ".join(fmt.format(x) for x in (b - a))
        print(f"{label:40} {fmt.format(a.mean()):>10} {fmt.format(b.mean()):>10} "
              f"{fmt.format((b - a).mean()):>10} {ps:>10}   {per}")

    print(f"\n{'':40} {la:>10} {lb:>10} {'diff':>10} {'p':>10}   per day ({lb} - {la})")
    for col, label, fmt in SHOW:
        if col not in off or col not in on:
            continue
        row(label, off.loc[idx, col].astype(float), on.loc[idx, col].astype(float), fmt)
    if "ladder_collapse_travel_all" in off and "stream_book_delivered" in off:
        per = lambda T: (T.loc[idx, "ladder_collapse_travel_all"].astype(float)
                         / T.loc[idx, "stream_book_delivered"].astype(float).clip(lower=1))
        row("distance driven per delivery (cells)", per(off), per(on), "{:.1f}")
    if lb != "rung on":
        return
    print("\nthe rung (means per day):")
    for c in RUNG:
        col = "ladder_collapse_" + c
        if col in on:
            print(f"  {c:24} {on.loc[idx, col].astype(float).mean():.1f}")


def plans(paths):
    """Decision time by solver and jam size, from the plan logs (--plans)."""
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    days = df[["map", "k", "seed"]].drop_duplicates().shape[0]
    print(f"[plans] {len(df)} planning attempts over {days} days")
    q = lambda v, f: float(np.percentile(v, f)) if len(v) else float("nan")
    for solver in ("exact", "thrml"):
        d = df[df.solver == solver]
        if d.empty:
            continue
        print(f"\n{solver}: time per attempt (ms), by number of fleets")
        cols = "  sampling median ms" if solver == "thrml" else ""
        print(f"  {'fleets':>6} {'attempts':>9} {'found':>7} {'median':>9} {'p95':>9} {'max':>9}" + cols)
        for n, g in d.groupby("fleets"):
            extra = (f" {g['sample_ms'].dropna().median():>9.0f}" if solver == "thrml" else "")
            print(f"  {int(n):>6} {len(g):>9} {g['found'].mean():>6.0%} {q(g.ms, 50):>9.1f} {q(g.ms, 95):>9.1f} "
                  f"{g.ms.max():>9.1f}" + extra)
        if solver == "thrml" and d["choices_max"].notna().any():
            print("\nthrml: sampling time (ms) by size (fleets x most choices per variable)")
            print(f"  {'fleets':>6} {'choices':>8} {'attempts':>9} {'median':>9} {'p95':>9}")
            for (n, kb), g in d.groupby(["size_fleets", "size_choices"]):
                s_ = g["sample_ms"].dropna()
                print(f"  {int(n):>6} {int(kb):>8} {len(g):>9} {q(s_, 50):>9.0f} {q(s_, 95):>9.0f}")
            first = d[d.compile_ms > 1000]
            print(f"\n  compiles (first jam of a size): {len(first)}, median {first.compile_ms.median():.0f} ms; "
                  f"all other jams: compile median {d[d.compile_ms <= 1000].compile_ms.median():.0f} ms")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--compare", nargs="*", help="CSV files to pair and compare (no runs)")
    ap.add_argument("--head", nargs=2, metavar=("A", "B"),
                    help="two versions of the rung head to head, e.g. \"aws_s*_on3.csv\" \"aws_s*_hold.csv\"")
    ap.add_argument("--plans", nargs="*", help="plan logs (*_plans.csv) to summarise: decision time by jam size")
    ap.add_argument("--arm", default="ladder4")
    ap.add_argument("--map", default="Kiva_2D_46_33_1")
    ap.add_argument("--fleets", type=int, default=60)
    ap.add_argument("--seeds", default="0", help="comma-separated, e.g. 0,1,2,3")
    ap.add_argument("--cond", choices=["clean", "full"], default="full")
    ap.add_argument("--scens-dir", default="")
    ap.add_argument("--collapse", choices=["off", "on", "both"], default="both")
    ap.add_argument("--big", choices=["thrml", "ladder", "hold"], default="thrml",
                    help="5-8 fleets (and 3-4 exact search cannot settle): thrml, left to the ladder, or held")
    ap.add_argument("--solver", choices=["plan", "hold"], default="plan",
                    help="plan: the rung (exact search and thrml); hold: the hold test (members just wait)")
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--max-exact", type=int, default=4)
    ap.add_argument("--max-fleets", type=int, default=8)
    ap.add_argument("--grace", type=int, default=2)
    ap.add_argument("--thrml-chains", type=int, default=16)
    ap.add_argument("--thrml-steps", type=int, default=300)
    ap.add_argument("--steps", type=int, default=0, help="a shorter day (testing only)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    if a.head is not None:
        A = sorted(glob.glob(a.head[0]))
        B = sorted(glob.glob(a.head[1]))
        if not A or not B:
            sys.exit("[head] no files")
        head(A, B)
        return
    if a.plans is not None:
        paths = [p for pat in a.plans for p in sorted(glob.glob(pat))]
        if not paths:
            sys.exit("[plans] no files")
        plans(paths)
        return
    if a.compare is not None:
        paths = [p for pat in a.compare for p in sorted(glob.glob(pat))]
        if not paths:
            sys.exit("[compare] no files")
        compare(paths)
        return

    from final_jobs import ARMS, CONDS, GROUPS, SCENS
    if a.arm not in ("ladder4", "ladder6", "ladder7", "RULES"):
        sys.exit("the rung needs the ladder: --arm ladder4, ladder6, ladder7 or RULES")
    group = next((g for g, v in GROUPS.items() if a.map in v["maps"].split(",")), None)
    if group is None:
        sys.exit(f"{a.map} is not one of the final run's maps")
    g = GROUPS[group]
    fail, people = CONDS[a.cond]
    steps = a.steps or g["steps"][a.cond]
    out = a.out or (f"collapse_{a.arm}_{a.map}_k{a.fleets}_{a.cond}_"
                    + (a.big if a.solver == "plan" else a.solver) + f"_s{a.seeds.replace(',', '-')}"
                    + (f"_steps{a.steps}" if a.steps else "") + ".csv")

    import core_warehouse
    import collapse_warehouse as cw
    cw.install(core_warehouse)
    cw.ENABLED["cfg"] = {"horizon": a.horizon, "max_exact": a.max_exact, "max_fleets": a.max_fleets,
                         "grace_steps": a.grace, "big": a.big, "solver": a.solver,
                         "chains": a.thrml_chains, "steps": a.thrml_steps}
    import deploy_eval
    # The plan log: every planning attempt of the "on" runs, with its day.
    cw.PLAN_LOG["path"] = out.replace(".csv", "_plans.csv")
    _run_one = deploy_eval.run_one

    def run_one(inst, *ra, **rkw):
        cw.RUN_INFO.update(map=inst["map"], k=inst["k"], seed=inst["seed"])
        return _run_one(inst, *ra, **rkw)

    deploy_eval.run_one = run_one

    modes = ["off", "on"] if a.collapse == "both" else [a.collapse]
    parts = []
    for mode in modes:
        cw.ENABLED["on"] = (mode == "on")
        part = out.replace(".csv", f"_{mode}.part.csv")
        argv = ["deploy_eval.py", "--arms", ARMS[a.arm], "--maps", a.map,
                "--scens-dir", a.scens_dir or SCENS, "--agents", str(a.fleets), "--seeds", a.seeds,
                "--failures", fail, "--people", people, "--book", str(g["book"]),
                "--release", str(g["release"]), "--max-steps", str(steps),
                "--assignment", "hungarian", "--world-fixes", "on", "--orphans", "pickup",
                "--keyed-draws", "on", "--out", part]
        print(f"[collapse] rung {mode.upper()}: deploy_eval " + " ".join(argv[1:]), flush=True)
        sys.argv = argv
        try:
            deploy_eval.main()
        except SystemExit:
            pass
        except Exception as e:                     # the closing summary only (scipy etc.)
            print(f"[collapse] note: deploy_eval ended with {type(e).__name__}: {e}")
        if os.path.exists(part):
            df = pd.read_csv(part)
            if mode == "on":
                df["arm"] = df["arm"].astype(str) + "+collapse"
            parts.append(df)
            os.remove(part)
    if not parts:
        sys.exit("[collapse] no rows")
    old = pd.read_csv(out) if os.path.exists(out) else None
    allrows = pd.concat(([old] if old is not None else []) + parts, ignore_index=True)
    allrows.to_csv(out, index=False)
    print(f"[collapse] {sum(len(p) for p in parts)} rows -> {out}")
    if a.collapse == "both":
        compare([out])


if __name__ == "__main__":
    main()
