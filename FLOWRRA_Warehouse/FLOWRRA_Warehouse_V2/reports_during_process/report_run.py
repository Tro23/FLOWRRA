"""
report_run.py -- tables and graphs for a training run and for its graduation
tests (build log entry 43). Reads the CSVs only; runs in seconds; no torch.

Two modes:

  1. A training run (the curriculum_metrics CSV), optionally against an earlier
     run as the baseline, and optionally scoring run 10's six predictions:

     python report_run.py train --csv ladder_run6/curriculum_metrics_ladder_run6.csv \
         --base ladder_run5/curriculum_metrics_ladder_run5.csv --score-run10 --out report_run6

  2. Graduation tests (deploy_eval CSVs; several may be given):

     python report_run.py deploy --csv deploy_eval_clean10.csv --ref ladder6 --out report_clean10

Writes <out>.txt (every table, also printed) and <out>_*.png (graphs; skipped,
with a note, if matplotlib is missing). Paths to CSVs: adjust to where they are.

Reading guide:
  * "small" is map 25_5_2_5_2_1, "large" is 50_20_5_10_5_2.
  * Fleet size is the stream_fleets column (training) or k (deploy). The log's
    "k=" is the robots still on the floor at the end, not the fleet size.
  * Near misses come in spike days, so every near-miss table shows the median
    beside the mean.
  * Graduation: the reference arm (--ref) against each other arm, paired over
    the same (map, fleet, seed) days, on the five measures: share of the book
    delivered, lead time, collisions, L3 fires, near misses with people. A
    measure is won when the paired mean is better. 3 of 5 graduates, unless the
    safety gate fails: firmly worse (Wilcoxon p < 0.05) on collisions or on near
    misses with people.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

MAPS = {"25_5_2_5_2_1": "small", "50_20_5_10_5_2": "large"}

# column, label, which way is better (None: no direction)
TRAIN_COLS = [
    ("router_learner_agree", "Learner agrees with the route (live)", "high"),
    ("router_q_differs", "Heads alone differ from the route", "low"),
    ("router_choice_differs", "Chosen moves differ from the route", "low"),
    ("stream_book_lead_mean", "Lead time (steps)", "low"),
    ("stream_book_delivered", "Orders delivered (of 80)", "high"),
    ("guard_stops", "Guard stops", "low"),
    ("ladder_l3_fired", "L3 fires", "low"),
    ("obstacle_near_miss_steps", "Near misses with people", "low"),
    ("collisions", "Collisions", "low"),
    ("errors_injected", "Robots failed (stress)", None),
    ("qval_safety", "Safety value", None),
    ("loss_safety", "Safety loss", "low"),
    ("qval_efficiency", "Efficiency value", None),
    ("loss_efficiency", "Efficiency loss", "low"),
    ("router_scale", "Router scale", None),
]

FIVE = [
    ("stream_book_fill", "Share of book delivered", "high"),
    ("stream_book_lead_mean", "Lead time", "low"),
    ("collisions", "Collisions", "low"),
    ("ladder_l3_fired", "L3 fires", "low"),
    ("obstacle_near_miss_steps", "Near misses with people", "low"),
]
EXTRA = [("guard_stops", "Guard stops", "low"), ("orders_per_hour", "Orders per hour", "high")]

OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def table(df):
    say(df.to_string())
    say()


def plt_or_none():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        say("(matplotlib not installed: tables only. pip install matplotlib for the graphs)")
        return None


def wilcoxon_p(a, b):
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return float("nan")
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.allclose(a, b):
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


# ============================================================== training run
def load_train(path):
    df = pd.read_csv(path)
    df["size"] = df["map"].map(MAPS).fillna(df["map"])
    df["fleet"] = df["stream_fleets"] if "stream_fleets" in df else df.get("agents")
    on = "router_route_on" if "router_route_on" in df else "route_on"
    df["route"] = df[on].fillna(1).astype(float) > 0.5
    off = df.loc[~df["route"], "episode"]
    sw = int(off.min()) if len(off) else int(df["episode"].max()) + 1
    df["window"] = np.where(df["route"], "warm-up",
                            df["episode"].apply(lambda e: f"{sw + 15 * ((e - sw) // 15)}-"
                                                          f"{min(sw + 15 * ((e - sw) // 15) + 14, df['episode'].max())}"))
    last = int(df["episode"].max())
    df["last15"] = df["episode"] > last - 15
    return df, sw, last


def cols_in(df, cols):
    return [(c, l, b) for c, l, b in cols if c in df]


def train_mode(args):
    run, sw, last = load_train(args.csv)
    name = os.path.basename(args.csv)
    say(f"=== {name}: {len(run)} episodes, route off from episode {sw}, last 15 = {last - 14}-{last}")
    base = None
    if args.base:
        base, bsw, blast = load_train(args.base)
        say(f"=== baseline {os.path.basename(args.base)}: {len(base)} episodes, route off from {bsw}")
    say()
    cols = cols_in(run, TRAIN_COLS)
    main = [c for c, _, _ in cols if c not in ("router_scale",)]

    say("--- 1. By map and window (means; route-off windows of 15 episodes) ---")
    g = run.groupby(["size", "window"], sort=False)[main].mean()
    order = sorted(g.index, key=lambda t: (t[0], -1 if t[1] == "warm-up" else int(t[1].split("-")[0])))
    table(g.loc[order].T.round(3))
    if "obstacle_near_miss_steps" in run:
        say("Near misses with people, median by map and window:")
        table(run.groupby(["size", "window"], sort=False)["obstacle_near_miss_steps"].median()
              .loc[order].to_frame("median").T)

    say("--- 2. Route off, by map and fleet size (means; near misses also median) ---")
    keyc = ["router_q_differs", "stream_book_lead_mean", "stream_book_delivered", "guard_stops",
            "ladder_l3_fired", "obstacle_near_miss_steps", "collisions"]
    keyc = [c for c in keyc if c in run]

    def by_fleet(df, lo, hi):
        d = df[(~df["route"]) & (df["episode"] >= lo) & (df["episode"] <= hi)]
        m = d.groupby(["size", "fleet"])[keyc].mean()
        m["near_median"] = d.groupby(["size", "fleet"])["obstacle_near_miss_steps"].median()
        m["n"] = d.groupby(["size", "fleet"]).size()
        return m.round(3)

    R = by_fleet(run, sw, last)
    if base is not None:
        B = by_fleet(base, bsw, min(blast, last))
        both = pd.concat({"run": R, "base": B}, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
        say(f"(run: episodes {sw}-{last}; base: episodes {bsw}-{min(blast, last)}, the same span)")
        table(both)
    else:
        table(R)

    say("--- 3. Last 15 episodes by map (mean / median) ---")
    L = run[run["last15"]]
    rows = {}
    for c in keyc + [c for c in ("router_learner_agree", "router_choice_differs") if c in run]:
        for sz, d in L.groupby("size"):
            rows.setdefault(c, {})[f"{sz} mean"] = d[c].mean()
            rows[c][f"{sz} median"] = d[c].median()
            if base is not None:
                bd = base[base["last15"] & (base["size"] == sz)]
                rows[c][f"{sz} base mean"] = bd[c].mean() if c in bd else np.nan
                rows[c][f"{sz} base median"] = bd[c].median() if c in bd else np.nan
    table(pd.DataFrame(rows).T.round(3))

    if "obstacle_near_miss_steps" in run:
        say("--- 4. The 10 worst near-miss days (route off) ---")
        cc = [c for c in ["episode", "size", "fleet", "obstacle_near_miss_steps", "obstacle_humans",
                          "errors_injected", "wave_failures", "guard_stops", "ladder_l3_fired",
                          "stream_book_lead_mean", "stream_book_delivered"] if c in run]
        table(run[~run["route"]].nlargest(10, "obstacle_near_miss_steps")[cc].set_index("episode"))

    if args.score_run10:
        score_run10(run, base)

    plt = plt_or_none()
    if plt is not None:
        n = len(cols)
        nc = 3
        nr = int(np.ceil(n / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(16, 3.2 * nr), squeeze=False)
        colors = {"small": "#c0392b", "large": "#2c7fb8"}
        for ax, (c, lab, better) in zip(axes.flat, cols):
            for sz, d in run.groupby("size"):
                d = d.sort_values("episode")
                ax.scatter(d["episode"], d[c], s=10, color=colors.get(sz, "gray"), alpha=0.55)
                ax.plot(d["episode"], d[c].rolling(5, min_periods=1).median(),
                        color=colors.get(sz, "gray"), lw=1.8, label=f"{sz} (rolling median of 5)")
                if base is not None and c in base:
                    bd = base[base["size"] == sz].sort_values("episode")
                    ax.plot(bd["episode"], bd[c].rolling(5, min_periods=1).median(),
                            color=colors.get(sz, "gray"), lw=1.0, ls="--", alpha=0.6,
                            label=f"{sz} baseline")
            ax.axvspan(0.5, sw - 0.5, color="0.9", zorder=0)
            ax.axvspan(last - 14.5, last + 0.5, color="#fff3c4", zorder=0)
            if c == "obstacle_near_miss_steps":
                ax.set_yscale("symlog", linthresh=10)
            ax.set_title(lab + ("" if better is None else f"  ({better}er is better)"), fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].legend(fontsize=7)
        fig.suptitle(f"{name}: grey = route on (warm-up), yellow = last 15; dots = episodes, "
                     f"solid = rolling median, dashed = baseline", fontsize=10)
        fig.tight_layout()
        p = f"{args.out}_training.png"
        fig.savefig(p, dpi=110)
        say(f"wrote {p}")

        # by fleet size, route off: run against baseline
        show = [c for c in ("stream_book_lead_mean", "stream_book_delivered", "guard_stops",
                            "obstacle_near_miss_steps", "router_q_differs", "ladder_l3_fired") if c in run]
        fig, axes = plt.subplots(2, len(show), figsize=(3.2 * len(show), 6), squeeze=False)
        for j, c in enumerate(show):
            for i, sz in enumerate(["small", "large"]):
                ax = axes[i, j]
                fl = sorted(set(R.loc[sz].index) if sz in R.index.get_level_values(0) else [])
                if base is not None and sz in B.index.get_level_values(0):
                    fl = sorted(set(fl) | set(B.loc[sz].index))
                x = np.arange(len(fl))
                agg = "near_median" if c == "obstacle_near_miss_steps" else c
                rv = [R.loc[(sz, f), agg] if (sz, f) in R.index else np.nan for f in fl]
                ax.bar(x - 0.2, rv, 0.4, label="run", color="#c0392b" if sz == "small" else "#2c7fb8")
                if base is not None:
                    bv = [B.loc[(sz, f), agg] if (sz, f) in B.index else np.nan for f in fl]
                    ax.bar(x + 0.2, bv, 0.4, label="baseline", color="0.65")
                ax.set_xticks(x)
                ax.set_xticklabels([str(int(f)) for f in fl], fontsize=8)
                ax.set_title(f"{sz}: {c if agg == c else 'near misses (median)'}", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle("Route off, by fleet size (x axis): run against baseline, same span of episodes",
                     fontsize=10)
        fig.tight_layout()
        p = f"{args.out}_by_fleet.png"
        fig.savefig(p, dpi=110)
        say(f"wrote {p}")


def score_run10(run, base):
    say("--- 5. Run 10's six predictions (entry 43), on the last 15 episodes ---")
    L = run[run["last15"]]
    S, Lg = L[L["size"] == "small"], L[L["size"] == "large"]

    def m(d, c):
        return float(d[c].mean()) if c in d and len(d) else float("nan")

    def md(d, c):
        return float(d[c].median()) if c in d and len(d) else float("nan")

    res = []
    a_s, a_l = m(S, "router_learner_agree"), m(Lg, "router_learner_agree")
    res.append(("1 Learner reads the route", f"small {a_s:.3f} (>= 0.95), large {a_l:.3f} (>= 0.97)",
                a_s >= 0.95 and a_l >= 0.97))
    c_s, c_l = m(S, "router_choice_differs"), m(Lg, "router_choice_differs")
    res.append(("2 Chosen moves follow the route", f"small {c_s:.3f} (<= 0.35), large {c_l:.3f} (<= 0.20)",
                c_s <= 0.35 and c_l <= 0.20))
    if base is not None:
        BL0 = base[base["last15"]]
        shared = set(BL0["fleet"].unique())
        # Scored twice: all of run 10's last 15, and like for like (only the fleet
        # sizes the baseline also trained on; run 9 had no 15-robot days).
        for tag, LL in (("all fleets", L), ("like for like", L[L["fleet"].isin(shared)])):
            S2, Lg2 = LL[LL["size"] == "small"], LL[LL["size"] == "large"]
            BL = BL0
            bs, bl = BL[BL["size"] == "small"], BL[BL["size"] == "large"]
            ls, lb = m(S2, "stream_book_lead_mean"), m(bs, "stream_book_lead_mean")
            ll, lbl = m(Lg2, "stream_book_lead_mean"), m(bl, "stream_book_lead_mean")
            res.append((f"3 The small map is fixed [{tag}, {len(LL)} days]",
                        f"small lead {ls:.1f} against base {lb:.1f} ({100 * (ls / lb - 1):+.0f}%, needs <= -40%); "
                        f"large {ll:.1f} against {lbl:.1f} (must not be worse)",
                        ls <= 0.6 * lb and ll <= lbl))
            nm, nmd = m(LL, "obstacle_near_miss_steps"), md(LL, "obstacle_near_miss_steps")
            bnm, bnmd = m(BL, "obstacle_near_miss_steps"), md(BL, "obstacle_near_miss_steps")
            l3, l3d = m(LL, "ladder_l3_fired"), md(LL, "ladder_l3_fired")
            bl3, bl3d = m(BL, "ladder_l3_fired"), md(BL, "ladder_l3_fired")
            coll = float(LL["collisions"].sum()) if "collisions" in LL else float("nan")
            near_up = nm > bnm and nmd > bnmd
            l3_up = l3 > bl3 and l3d > bl3d
            res.append((f"4 Safety holds [{tag}]",
                        f"collisions {coll:.0f}; near misses mean/median {nm:.1f}/{nmd:.1f} against base "
                        f"{bnm:.1f}/{bnmd:.1f}; L3 {l3:.2f}/{l3d:.1f} against {bl3:.2f}/{bl3d:.1f} "
                        f"(fails if above on both mean and median)",
                        coll == 0 and not near_up and not l3_up))
    else:
        res.append(("3 and 4", "need --base", None))
    q = run.sort_values("episode")
    qv = m(L, "qval_safety")
    last30 = q[q["episode"] > q["episode"].max() - 30]
    if "loss_safety" in last30 and len(last30) > 5:
        slope = float(np.polyfit(last30["episode"], last30["loss_safety"], 1)[0]) * 30
    else:
        slope = float("nan")
    res.append(("5 The safety value is steady",
                f"safety value (last 15) {qv:.3f} (band -0.10 to -0.20); safety loss trend over the last 30 "
                f"episodes {slope:+.5f} (fails if clearly climbing)",
                (-0.20 <= qv <= -0.10) and not (slope > 0.0005)))
    h = m(S, "router_q_differs")
    res.append(("6 The heads learn the clear (allowed to fail)", f"small-map heads differ {h:.3f} (<= 0.45)",
                h <= 0.45))
    for k, txt, ok in res:
        mark = "?" if ok is None else ("PASS" if ok else "FAIL")
        say(f"  {mark:4}  {k}: {txt}")
    say("  (Prediction 5's 'clearly climbing' line here is a rise of more than 0.0005 over 30 episodes;"
        " judge it with the graph.)")
    say()


# ============================================================== graduation
def deploy_mode(args):
    df = pd.concat([pd.read_csv(p).assign(source=os.path.basename(p)) for p in args.csv],
                   ignore_index=True)
    df["size"] = df["map"].map(MAPS).fillna(df["map"])
    if "stream_book_fill" not in df and "stream_book_delivered" in df:
        df["stream_book_fill"] = df["stream_book_delivered"] / 80.0
    arms = list(dict.fromkeys(df["arm"]))
    ref = args.ref or arms[0]
    if ref not in arms:
        raise SystemExit(f"--ref {ref} not among the arms {arms}")
    five = cols_in(df, FIVE)
    extra = cols_in(df, EXTRA)
    allc = [c for c, _, _ in five + extra]
    say(f"=== {', '.join(args.csv)}: arms {arms}; reference {ref}; {len(df)} rows")
    say()
    say("--- 1. By arm, map and fleet (means) ---")
    table(df.groupby(["size", "k", "arm"])[allc].mean().round(3))
    say("--- 2. By arm and map (means; near misses also median) ---")
    m = df.groupby(["size", "arm"])[allc].mean()
    if "obstacle_near_miss_steps" in df:
        m["near_median"] = df.groupby(["size", "arm"])["obstacle_near_miss_steps"].median()
    table(m.round(3))

    say(f"--- 3. Graduation: {ref} against each arm, paired over the same days ---")
    missing = [lab for c, lab, _ in FIVE if c not in df]
    if missing:
        say(f"(not in these CSVs, so not won by anyone: {', '.join(missing)}; e.g. near misses with "
            f"people in the clean condition, where people are off. 3 wins are still needed.)")
    key = ["map", "k", "seed"]
    A = df[df.arm == ref].drop_duplicates(key).set_index(key)
    for other in [a for a in arms if a != ref]:
        Bo = df[df.arm == other].drop_duplicates(key).set_index(key)
        for scope in ["all", "small", "large"]:
            idx = A.index.intersection(Bo.index)
            if scope != "all":
                idx = [i for i in idx if MAPS.get(i[0], i[0]) == scope]
            if len(idx) == 0:
                continue
            say(f"{ref} vs {other} | {scope} | {len(idx)} paired days")
            won, gate = 0, True
            for c, lab, better in five + extra:
                a = A.loc[idx, c].astype(float).to_numpy()
                b = Bo.loc[idx, c].astype(float).to_numpy()
                ma, mb = np.nanmean(a), np.nanmean(b)
                p = wilcoxon_p(a, b)
                if better == "low":
                    w = (a < b).sum(); lo = (a > b).sum()
                    res = "won" if ma < mb else ("lost" if ma > mb else "tie")
                else:
                    w = (a > b).sum(); lo = (a < b).sum()
                    res = "won" if ma > mb else ("lost" if ma < mb else "tie")
                in5 = (c, lab, better) in five
                if in5 and res == "won":
                    won += 1
                if c in ("collisions", "obstacle_near_miss_steps") and res == "lost" and p < 0.05:
                    gate = False
                say(f"   {lab:26} {ref} {ma:9.3f} | {other} {mb:9.3f} | {res:4} | days better/worse "
                    f"{w}/{lo} | p {p:.3f}" + ("" if in5 else "   (reported, not scored)"))
            verdict = ("GRADUATES" if won >= 3 and gate else
                       ("held by the safety gate" if won >= 3 else "does not graduate"))
            say(f"   => {won} of 5 won; safety gate {'passed' if gate else 'FAILED'}: {ref} {verdict} "
                f"against {other} ({scope})")
            say()

    plt = plt_or_none()
    if plt is not None:
        show = five + [e for e in extra if e[0] == "guard_stops"]
        fig, axes = plt.subplots(2, len(show), figsize=(3.3 * len(show), 6.2), squeeze=False)
        pal = ["#c0392b", "#2c7fb8", "#7f8c8d", "#27ae60", "#8e44ad"]
        for j, (c, lab, better) in enumerate(show):
            for i, sz in enumerate(["small", "large"]):
                ax = axes[i, j]
                d = df[df["size"] == sz]
                ks = sorted(d["k"].unique())
                x = np.arange(len(ks))
                wdt = 0.8 / max(1, len(arms))
                for t, arm in enumerate(arms):
                    agg = d[d.arm == arm].groupby("k")[c]
                    v = (agg.median() if c == "obstacle_near_miss_steps" else agg.mean()).reindex(ks)
                    ax.bar(x - 0.4 + wdt * (t + 0.5), v.to_numpy(), wdt, label=arm, color=pal[t % len(pal)])
                ax.set_xticks(x)
                ax.set_xticklabels([str(k) for k in ks], fontsize=8)
                ax.set_title(f"{sz}: {lab}" + (" (median)" if c == "obstacle_near_miss_steps" else "")
                             + f" ({better}er is better)", fontsize=8)
                ax.tick_params(labelsize=7)
        axes[0, 0].legend(fontsize=7)
        fig.suptitle(f"{', '.join(os.path.basename(p) for p in args.csv)}: by fleet size (x axis), "
                     f"mean over seeds", fontsize=10)
        fig.tight_layout()
        p = f"{args.out}_deploy.png"
        fig.savefig(p, dpi=110)
        say(f"wrote {p}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=["train", "deploy"])
    ap.add_argument("--csv", required=True, nargs="+")
    ap.add_argument("--base", default=None, help="train: an earlier run's CSV to compare against")
    ap.add_argument("--score-run10", action="store_true", help="train: score run 10's six predictions")
    ap.add_argument("--ref", default=None, help="deploy: the arm under test (default: the first)")
    ap.add_argument("--out", default="report")
    args = ap.parse_args()
    if args.mode == "train":
        args.csv = args.csv[0]
        train_mode(args)
    else:
        deploy_mode(args)
    with open(f"{args.out}.txt", "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    print(f"wrote {args.out}.txt")


if __name__ == "__main__":
    main()
