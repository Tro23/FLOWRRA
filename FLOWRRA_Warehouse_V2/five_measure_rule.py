"""
five_measure_rule.py -- score a dry run against the baseline with the 5-measure rule.

WHAT IT DOES (plain language)
  You give it the BASELINE dry run's CSV (the one to beat) and one or more
  CANDIDATE dry-run CSVs (same map, same seed, same number of episodes, only the
  Learner's beta settings changed). For each candidate it:
    1. adds up the five measures over the whole dry run,
    2. says who wins each measure (a tie is a win for nobody),
    3. gives the verdict fixed BEFORE any run: the candidate goes into the next
       long run only if it wins AT LEAST 3 of the 5,
    4. prints extra context that is NOT scored (the Learner's share, its
       effective beta, self back-offs, overrides, the last episodes).

THE FIVE MEASURES (where each comes from in curriculum_metrics_*.csv)
  completion     mean of completion_rate              higher is better
  collisions     sum of collisions                    lower is better
  L3 fires       sum of ladder_l3_fired               lower is better
  retreats       sum of ladder_retrace_orders         lower is better
  longest repeat max of conflict_max_pair_repeat      lower is better

RUN
  python five_measure_rule.py curriculum_metrics_dry10_pr_off.csv curriculum_metrics_dry10_b008.csv
  python five_measure_rule.py BASELINE.csv CAND1.csv CAND2.csv CAND3.csv     (several at once)
  python five_measure_rule.py BASELINE.csv CAND.csv --last 5                  (size of the "last episodes" window)

Needs only pandas. Takes about a second.
"""

import argparse
import os
import sys

import pandas as pd

# (label, column, how to combine over episodes, True if higher is better)
MEASURES = [
    ("Completion",          "completion_rate",          "mean", True),
    ("Collisions",          "collisions",               "sum",  False),
    ("L3 fires",            "ladder_l3_fired",          "sum",  False),
    ("Retreats ordered",    "ladder_retrace_orders",    "sum",  False),
    ("Longest pair repeat", "conflict_max_pair_repeat", "max",  False),
]
NEED_TO_WIN = 3


def combine(df, col, how):
    return float(getattr(df[col], how)())


def show(label, col, v):
    return f"{v * 100:.1f}%" if col == "completion_rate" else f"{v:g}"


def learner_share(df):
    """Of the conflicted choices the Learner was asked about, how many it changed."""
    used = df["learner_used"].sum() if "learner_used" in df else 0
    return 100.0 * df["learner_changed"].sum() / used if used else float("nan")


def context(df, last):
    tail = df.tail(last)
    share_ep = (100.0 * df["learner_changed"] / df["learner_used"].where(df["learner_used"] > 0)) \
        if "learner_used" in df else None
    beta = df["learner_beta_eff"] if "learner_beta_eff" in df else None
    return {
        "Learner share (all eps)": f"{learner_share(df):.1f}%",
        "Learner share per ep (min-max)":
            f"{share_ep.min():.0f}-{share_ep.max():.0f}%" if share_ep is not None else "n/a",
        "beta_eff first -> last": f"{beta.iloc[0]:.3f} -> {beta.iloc[-1]:.3f}" if beta is not None else "n/a",
        "Learner agreement with RULES": f"{df['learner_agree'].mean():.3f}" if "learner_agree" in df else "n/a",
        "Self back-offs": f"{df['ladder_self_backoffs'].sum():g}",
        "Retreat timeouts": f"{df['ladder_retrace_timeouts'].sum():g}",
        "Retreat 'no room'": f"{df['ladder_retrace_no_room'].sum():g}",
        "qval_safety (last ep)": f"{df['qval_safety'].iloc[-1]:.4f}",
        f"Last {last}: completion": f"{tail['completion_rate'].mean() * 100:.1f}%",
        f"Last {last}: collisions": f"{tail['collisions'].sum():g}",
        f"Last {last}: L3 fires": f"{tail['ladder_l3_fired'].sum():g}",
    }


def name_of(path):
    n = os.path.basename(path)
    for cut in ("curriculum_metrics_", ".csv"):
        n = n.replace(cut, "")
    return n


def score(base, cand, bname, cname, last):
    n = min(len(base), len(cand))
    if len(base) != len(cand):
        print(f"  !! episode counts differ ({len(base)} vs {len(cand)}); comparing the first {n} only."
              " A fair test needs the same number of episodes.")
    b, c = base.head(n), cand.head(n)

    rows, wins_c, wins_b = [], 0, 0
    for label, col, how, higher in MEASURES:
        vb, vc = combine(b, col, how), combine(c, col, how)
        if vb == vc:
            who = "tie"
        elif (vc > vb) == higher:
            who, wins_c = cname, wins_c + 1
        else:
            who, wins_b = bname, wins_b + 1
        rows.append((label, show(label, col, vb), show(label, col, vc), who))

    w = max(len(bname), len(cname), 12)
    print(f"\n  {'Measure (' + str(n) + ' episodes)':<26}{bname:>{w}}{cname:>{w}}   winner")
    print("  " + "-" * (26 + 2 * w + 12))
    for label, vb, vc, who in rows:
        print(f"  {label:<26}{vb:>{w}}{vc:>{w}}   {who}")
    verdict = ("ADOPT -- goes into the next long run" if wins_c >= NEED_TO_WIN
               else "KEEP BASELINE -- not adopted")
    print(f"\n  Score: {cname} {wins_c} - {wins_b} {bname}  "
          f"(ties {5 - wins_c - wins_b})  ->  {verdict}")

    print("\n  Context (NOT scored, for reading the result):")
    cb, cc = context(b, last), context(c, last)
    w2 = max(w, 17)
    for k in cb:
        print(f"    {k:<32}{cb[k]:>{w2}}{cc[k]:>{w2}}")
    return wins_c, wins_b


def main():
    ap = argparse.ArgumentParser(description="5-measure rule: candidate dry run(s) vs a baseline.")
    ap.add_argument("baseline", help="the dry run to beat, e.g. curriculum_metrics_dry10_pr_off.csv")
    ap.add_argument("candidates", nargs="+", help="one or more candidate dry-run CSVs")
    ap.add_argument("--last", type=int, default=5, help="size of the 'last episodes' window (default 5)")
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    bname = name_of(args.baseline)
    results = []
    for path in args.candidates:
        cname = name_of(path)
        print("\n" + "=" * 78 + f"\n  {cname}  vs  {bname}  (baseline)\n" + "=" * 78)
        wc, wb = score(base, pd.read_csv(path), bname, cname, args.last)
        results.append((cname, wc, wb))

    if len(results) > 1:
        print("\n" + "=" * 78 + "\n  SUMMARY\n" + "=" * 78)
        for cname, wc, wb in sorted(results, key=lambda r: -r[1]):
            print(f"  {cname:<30} {wc} - {wb}   {'ADOPT' if wc >= NEED_TO_WIN else 'keep baseline'}")
        print("\n  Note: with several candidates, one can win 3-2 by luck alone. Before a long")
        print("  run, repeat the winner's dry run once; adopt it only if it wins again.")


if __name__ == "__main__":
    sys.exit(main())