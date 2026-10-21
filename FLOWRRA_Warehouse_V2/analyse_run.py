"""
analyse_run.py -- what moves with what in a training run, and what only appears to.

    python analyse_run.py cold_run5/curriculum_metrics.csv
    python analyse_run.py cold_run5/curriculum_metrics.csv --target completion_rate
    python analyse_run.py a.csv b.csv --label-a flat --label-b conv
    python analyse_run.py run.csv --out correlations.csv

THE TRAP THIS EXISTS TO AVOID

Epsilon decays monotonically, so EVERY metric with a trend correlates with every
other metric with a trend. Loss falls and completion rises: r = -0.8, and it
means nothing -- both are just functions of how far into training you are. A raw
correlation matrix over training metrics is close to useless and extremely
convincing, which is the worst combination.

So three views, in increasing order of how much you should trust them:

  raw        Pearson/Spearman. Mostly measures "both trend with episode".
  partial    Correlation after removing the linear effect of EPISODE from both
             variables. Answers: do these move together BEYOND the shared trend?
  delta      Correlation of episode-to-episode CHANGES. Removes trend entirely
             and is the strictest view. If a relationship survives here, it is
             about the dynamics rather than the schedule.

A pair that is strong in `raw` and absent in `partial` and `delta` is a trend
artifact. A pair strong in all three is worth thinking about.

SAMPLE SIZE. At n=60 you need |r| > 0.25 for p<0.05, and consecutive episodes
are NOT independent -- the policy barely changes between them -- so the
effective n is smaller than the row count. Everything here is a hint for where
to look, never a result on its own.
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import sys

import numpy as np
import pandas as pd


SKIP = {"episode", "map", "scen", "agents"}


def numeric_columns(d: pd.DataFrame) -> list:
    cols = [c for c in d.columns
            if c not in SKIP and pd.api.types.is_numeric_dtype(d[c])]
    # Constant columns have undefined correlation and pollute the output.
    return [c for c in cols if d[c].notna().sum() > 2 and d[c].std() > 1e-12]


def resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """y with the linear effect of x removed."""
    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < 3:
        return np.full_like(y, np.nan, dtype=float)
    b, a = np.polyfit(x[ok], y[ok], 1)
    out = np.full_like(y, np.nan, dtype=float)
    out[ok] = y[ok] - (a + b * x[ok])
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    return corr(ra, rb)


def crit(n: int, p: float = 0.05) -> float:
    """|r| needed for significance at n, two-tailed, normal approximation."""
    if n < 4:
        return 1.0
    z = 1.96 if p == 0.05 else 2.576
    return float(np.tanh(z / math.sqrt(n - 3)))


def trend_table(d: pd.DataFrame, cols: list, window: int) -> None:
    """First vs last window of episodes -- the blunt question, asked plainly."""
    n = len(d)
    w = min(window, n // 2)
    if w < 2:
        return
    head, tail = d.iloc[:w], d.iloc[-w:]
    print(f"\nFIRST {w} EPISODES vs LAST {w}")
    print(f"   {'metric':<26}{'first':>10}{'last':>10}{'change':>11}")
    rows = []
    for c in cols:
        a, b = head[c].mean(), tail[c].mean()
        if not np.isfinite(a) or not np.isfinite(b):
            continue
        pct = (b - a) / abs(a) * 100 if abs(a) > 1e-9 else float("nan")
        rows.append((abs(pct) if np.isfinite(pct) else 0, c, a, b, pct))
    for _, c, a, b, pct in sorted(rows, reverse=True):
        arrow = "up" if b > a else "down"
        print(f"   {c:<26}{a:>10.4f}{b:>10.4f}{pct:>9.1f}% {arrow}")


def correlations(d: pd.DataFrame, cols: list, target: str | None,
                 top: int, out: str | None) -> None:
    n = len(d)
    ep = d["episode"].to_numpy(dtype=float) if "episode" in d else np.arange(n, dtype=float)
    r_crit = crit(n)

    pairs = ([(target, c) for c in cols if c != target] if target
             else list(itertools.combinations(cols, 2)))

    rows = []
    for a, b in pairs:
        ya, yb = d[a].to_numpy(dtype=float), d[b].to_numpy(dtype=float)
        ra, rb = resid(ya, ep), resid(yb, ep)
        da, db = np.diff(ya), np.diff(yb)
        rows.append({
            "a": a, "b": b,
            "raw": corr(ya, yb),
            "spearman": spearman(ya, yb),
            "partial_ep": corr(ra, rb),
            "delta": corr(da, db),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("\nnothing to correlate")
        return

    # NEAR-PERFECT PAIRS ARE DEFINITIONS, NOT FINDINGS.
    #
    # completed vs completion_rate is one divided by a constant.
    # risk_steps vs risk_steps_acted turned out to be the SAME column, which is
    # why the log prints "risk 122/122 (100%)" -- it divides a number by itself.
    # Left in the main list, these crowd out everything real.
    df["duplicate"] = df["raw"].abs() > 0.995

    # A pair survives if it is significant AFTER the trend is removed. Raw
    # significance alone is what the trap looks like.
    df["survives"] = ((df["partial_ep"].abs() > r_crit)
                      & (df["delta"].abs() > r_crit)
                      & ~df["duplicate"])
    df["trend_artifact"] = ((df["raw"].abs() > r_crit)
                           & (df["partial_ep"].abs() <= r_crit)
                           & ~df["duplicate"])

    print(f"\nn = {n} episodes   |r| > {r_crit:.3f} needed for p<0.05")
    print("   (episodes are autocorrelated, so the true bar is higher)")

    dup = df[df["duplicate"]]
    if len(dup):
        print(f"\nDEFINITIONAL DUPLICATES ({len(dup)}) -- |r| > 0.995, the same quantity twice")
        for r in dup.head(8).itertuples():
            print(f"   {r.a:<24}{r.b:<24}{r.raw:>8.3f}")
        print("   Excluded below. If a pair here is NOT meant to be identical,")
        print("   that is a bug in how one of them is computed.")

    keep = df[df["survives"]].reindex(
        df[df["survives"]]["delta"].abs().sort_values(ascending=False).index)
    print(f"\nSURVIVES the trend ({len(keep)} pair(s)) -- moves together beyond training progress")
    if len(keep):
        print(f"   {'a':<24}{'b':<24}{'raw':>8}{'partial':>9}{'delta':>8}")
        for r in keep.head(top).itertuples():
            print(f"   {r.a:<24}{r.b:<24}{r.raw:>8.2f}{r.partial_ep:>9.2f}{r.delta:>8.2f}")
    else:
        print("   none -- every strong pair here is explained by training progress")

    art = df[df["trend_artifact"]].reindex(
        df[df["trend_artifact"]]["raw"].abs().sort_values(ascending=False).index)
    print(f"\nTREND ARTIFACTS ({len(art)} pair(s)) -- strong raw, nothing left after removing episode")
    if len(art):
        print(f"   {'a':<24}{'b':<24}{'raw':>8}{'partial':>9}")
        for r in art.head(top).itertuples():
            print(f"   {r.a:<24}{r.b:<24}{r.raw:>8.2f}{r.partial_ep:>9.2f}")

    if out:
        df.to_csv(out, index=False)
        print(f"\nwrote {out} ({len(df)} pairs, all four views)")


def compare(a: pd.DataFrame, b: pd.DataFrame, la: str, lb: str, cols: list) -> None:
    """Two runs, same columns -- the comparison the correlations cannot make."""
    print(f"\nRUN COMPARISON: {la} (n={len(a)}) vs {lb} (n={len(b)})")
    print(f"   {'metric':<26}{la:>12}{lb:>12}{'diff':>11}{'t-ish':>8}")
    for c in cols:
        if c not in b.columns:
            continue
        xa, xb = a[c].dropna().to_numpy(), b[c].dropna().to_numpy()
        if len(xa) < 3 or len(xb) < 3:
            continue
        ma, mb = xa.mean(), xb.mean()
        se = math.sqrt(xa.var(ddof=1) / len(xa) + xb.var(ddof=1) / len(xb))
        t = (mb - ma) / se if se > 1e-12 else float("nan")
        flag = "  *" if abs(t) > 2 else ""
        print(f"   {c:<26}{ma:>12.4f}{mb:>12.4f}{mb-ma:>11.4f}{t:>8.2f}{flag}")
    print("   * |t| > 2. Episodes are autocorrelated, so treat this as a hint.")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("csv", nargs="+", help="one run, or two to compare")
    ap.add_argument("--target", default=None,
                    help="correlate everything against this column only")
    ap.add_argument("--window", type=int, default=10,
                    help="episodes in the first/last trend comparison")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None, help="write the full pair table here")
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    args = ap.parse_args()

    for p in args.csv:
        if not os.path.exists(p):
            sys.exit(f"not found: {p}")

    d = pd.read_csv(args.csv[0])
    cols = numeric_columns(d)
    print(f"{args.csv[0]}: {len(d)} episodes, {len(cols)} numeric columns")

    if args.target and args.target not in cols:
        sys.exit(f"--target {args.target!r} is not a numeric column.\n"
                 f"available: {', '.join(cols)}")

    trend_table(d, cols, args.window)
    correlations(d, cols, args.target, args.top, args.out)

    if len(args.csv) > 1:
        d2 = pd.read_csv(args.csv[1])
        compare(d, d2,
                args.label_a or os.path.basename(os.path.dirname(args.csv[0])) or "A",
                args.label_b or os.path.basename(os.path.dirname(args.csv[1])) or "B",
                cols)

    print("\nREADING THIS")
    print("  A pair strong in raw but absent in partial/delta is the schedule,")
    print("  not a relationship. Both variables just move with training.")
    print("  A pair strong in all three is worth a mechanism -- and a mechanism")
    print("  is a hypothesis, which still needs its own test.")
    print("  Correlations point at where to look. They never settle anything.")


if __name__ == "__main__":
    main()