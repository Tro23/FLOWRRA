"""
analyse_failure_benchmark.py

Turns failurebenchmark_epsilon.csv into the tables a write-up can stand on.

WHY A SCRIPT AND NOT A groupby
Three things about this data will mislead a plain mean, and all three have
already bitten during development:

  1. UNMATCHED FAILURE COUNTS. Arms can receive different numbers of orphaned
     orders when their episode lengths differ by an order of magnitude (RHCR
     alone finished in 34 steps where FLOWRRA took 428). recovery_RATE is
     comparable; raw counts are not. This script reports the orphan counts per
     arm first, so the reader can see whether the matching held before reading
     anything downstream.

  2. THE MEAN HIDES THE FAILURE MODE. In an earlier run the naive arm averaged
     31.9 collisions -- which was one instance at 287 and nine at 1-8. "Naive
     collides slightly more" and "naive has a catastrophic failure mode that
     fires on 1 in 10 instances" are different claims, and only the second is
     true. Per-instance worst cases are reported alongside every mean.

  3. UNPAIRED TESTS UNDERSTATE. Every arm ran the SAME instances, so paired
     tests apply and are far more powerful at these sample sizes.

Usage:
    python analyse_failure_benchmark.py failurebenchmark_epsilon.csv
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "failurebenchmark_epsilon.csv"
    d = pd.read_csv(path)
    d["instance"] = d["map"].astype(str) + "|s" + d["seed"].astype(str) + \
                    "|k" + d["requested_agents"].astype(str)
    n_inst = d.instance.nunique()
    print(f"{len(d)} rows | {n_inst} instances | {d.algorithm.nunique()} arms")
    print(d.algorithm.value_counts().to_string())

    # ---- 0. did the harness hold? ----------------------------------------
    section("0. HARNESS CHECK -- were the arms given the same failures?")
    orph = d.groupby("algorithm").orders_orphaned.agg(["mean", "min", "max", "sum"])
    print(orph.round(2).to_string())
    spread = orph["mean"].max() - orph["mean"].min()
    print(f"\nspread across arms: {spread:.2f} orphans "
          f"({spread / max(orph['mean'].max(), 1e-9) * 100:.0f}% of the largest)")
    if spread > 0.5:
        print("  -> NOT well matched. Use recovery_RATE, not counts, and state the")
        print("     residual. Note which direction it biases: an arm that received")
        print("     MORE failures and still scored better is a conservative result.")
    else:
        print("  -> matched; absolute counts are comparable.")

    # ---- 1. headline ------------------------------------------------------
    section("1. RECOVERY")
    cols = [c for c in ["orders_orphaned", "orders_recovered", "recovered_via_handover",
                        "orders_lost", "recovery_rate", "rescuer_deaths",
                        "completion_rate", "success"] if c in d.columns]
    print(d.groupby("algorithm")[cols].mean().round(3).to_string())

    # ---- 2. cost of recovering -------------------------------------------
    section("2. WHAT RECOVERY COST -- means AND worst cases")
    for c in ["collisions", "mean_integrity", "soc_hops", "steps_run"]:
        if c not in d.columns:
            continue
        g = d.groupby("algorithm")[c]
        t = pd.DataFrame({"mean": g.mean(), "median": g.median(),
                          "worst": g.max() if c != "mean_integrity" else g.min()})
        print(f"\n{c}:")
        print(t.round(3).to_string())
        # a mean far from the median means one instance is carrying it
        for a in t.index:
            sub = d[d.algorithm == a][c]
            if len(sub) > 2 and sub.std() > 0:
                z = (sub - sub.mean()).abs() / sub.std()
                if (z > 2.5).any():
                    bad = d[d.algorithm == a].loc[z.idxmax()]
                    print(f"   OUTLIER {a}: instance {bad['instance']} = {bad[c]:.3g} "
                          f"(mean {sub.mean():.3g}) -- the mean is carrying one case")

    # ---- 3. paired tests vs each baseline ---------------------------------
    section("3. PAIRED TESTS (same instances, so pairing applies)")
    piv = {c: d.pivot_table(index="instance", columns="algorithm", values=c)
           for c in ["recovery_rate", "completion_rate", "collisions", "soc_hops"]
           if c in d.columns}
    arms = sorted(d.algorithm.unique())
    flow = [a for a in arms if a.startswith("FLOWRRA")]
    base = [a for a in arms if not a.startswith("FLOWRRA")]
    for b in base:
        for f in flow:
            print(f"\n{f}  vs  {b}")
            for c, tab in piv.items():
                if f not in tab or b not in tab:
                    continue
                x, y = tab[f].dropna(), tab[b].dropna()
                idx = x.index.intersection(y.index)
                if len(idx) < 3:
                    continue
                x, y = x.loc[idx], y.loc[idx]
                diff = x - y
                try:
                    p = stats.wilcoxon(x, y).pvalue
                except Exception:
                    p = float("nan")
                w, l = int((diff > 0).sum()), int((diff < 0).sum())
                print(f"   {c:<18} {x.mean():9.3f} vs {y.mean():9.3f}  "
                      f"diff {diff.mean():+8.3f}  W/L {w}/{l}  wilcoxon p={p:.4f}")

    # ---- 4. does epsilon do anything? -------------------------------------
    if "eval_epsilon" in d.columns and d.eval_epsilon.notna().any():
        section("4. EPSILON SWEEP -- is FLOWRRA locking deterministically?")
        e = d[d.algorithm.str.startswith("FLOWRRA")]
        t = e.groupby("eval_epsilon")[
            [c for c in ["recovery_rate", "completion_rate", "success",
                         "collisions", "soc_hops"] if c in e.columns]].mean()
        print(t.round(3).to_string())
        if len(t) > 2:
            for c in t.columns:
                sl, _, r, p, _ = stats.linregress(t.index.astype(float), t[c])
                print(f"   trend {c:<18} slope {sl:+.4f} per unit eps  r={r:+.2f}  p={p:.3f}")
            print("\n   Rising completion with epsilon means the policy was locking and")
            print("   needs a stochastic floor at deployment. Flat or falling means")
            print("   epsilon is not the constraint and the shipped value stays low.")

    section("NOTE FOR THE WRITE-UP")
    print("decision_ms_per_step is NOT comparable across arms: for the planners it")
    print("includes amortised planning and replanning, for FLOWRRA it is a pure")
    print("forward pass. Any compute claim needs its own microbenchmark of")
    print("forward-pass latency against map size, with planning reported separately.")


if __name__ == "__main__":
    main()