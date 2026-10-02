"""
plot_runs.py -- one run against another, in six panels, plus what moves deliveries.

    python plot_runs.py --run cold_run25=curriculum_metrics_cold_run25.csv \\
                        --ref cold_run24=curriculum_metrics_cold_run24.csv \\
                        --capacity-eff 0.42 --capacity-window 3 --out cold_run25

  <out>_overview.png   deliveries, efficiency, collisions per 100 deliveries,
                       preemption and the recovery head's value, holds, wait share
  <out>_drivers.png    how strongly each mechanism moves with deliveries from one
                       episode to the next (correlation of CHANGES -- the training
                       trend removed, as in analyse_run.py's strictest view)

--capacity-eff is the rule-based controller's efficiency at the same fleet count
(drive_shortest_path.py; 50_ at 60 fleets on the +-3 window: 0.42), drawn from
the first episode at --capacity-window.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def roll(s, k=5):
    return s.rolling(k, min_periods=1, center=True).mean()


def shade_windows(ax, d):
    if "stream_order_window" not in d.columns:
        return
    colours = {1: "#f2f2f2", 2: "#e6eef7", 3: "#dde8d9"}
    for w, g in d.groupby("stream_order_window"):
        ax.axvspan(g.episode.min() - 0.5, g.episode.max() + 0.5, color=colours.get(int(w), "#eee"), zorder=0,
                   label=f"order window +-{int(w)}")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--run", required=True, help="label=path")
    ap.add_argument("--ref", default=None, help="label=path, drawn in grey")
    ap.add_argument("--capacity-eff", type=float, default=None)
    ap.add_argument("--capacity-window", type=int, default=3)
    ap.add_argument("--out", default="run")
    args = ap.parse_args()
    lab, path = args.run.split("=", 1)
    d = pd.read_csv(path).sort_values("episode").reset_index(drop=True)
    r = None
    if args.ref:
        rlab, rpath = args.ref.split("=", 1)
        r = pd.read_csv(rpath).sort_values("episode").reset_index(drop=True)

    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    # (a) deliveries
    ax = axs[0, 0]; shade_windows(ax, d)
    if r is not None:
        ax.plot(r.episode, r.completed, color="grey", alpha=0.35, lw=1)
        ax.plot(r.episode, roll(r.completed), color="grey", lw=2, label=f"{rlab} (5-ep mean)")
    ax.plot(d.episode, d.completed, color="tab:blue", alpha=0.35, lw=1)
    ax.plot(d.episode, roll(d.completed), color="tab:blue", lw=2.5, label=f"{lab} (5-ep mean)")
    b = d.completed.idxmax()
    ax.annotate(f"best {d.completed[b]}", (d.episode[b], d.completed[b]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color="tab:blue")
    ax.set_title("Deliveries per episode"); ax.set_xlabel("episode"); ax.legend(fontsize=7, loc="lower right")
    # (b) efficiency
    ax = axs[0, 1]; shade_windows(ax, d)
    if "stream_efficiency" in d.columns:
        ax.plot(d.episode, 100 * d.stream_efficiency, color="tab:blue", alpha=0.35, lw=1)
        ax.plot(d.episode, 100 * roll(d.stream_efficiency), color="tab:blue", lw=2.5, label=f"{lab} (5-ep mean)")
    if args.capacity_eff is not None and "stream_order_window" in d.columns:
        e0 = d.loc[d.stream_order_window >= args.capacity_window, "episode"]
        if len(e0):
            ax.hlines(100 * args.capacity_eff, e0.min() - 0.5, d.episode.max() + 0.5, colors="black", linestyles="--",
                      label=f"rule-based controller, window +-{args.capacity_window}")
    ax.set_title("Efficiency: % of the conflict-free ideal"); ax.set_xlabel("episode"); ax.set_ylabel("%")
    ax.legend(fontsize=7, loc="lower right")
    # (c) collisions per 100 deliveries
    ax = axs[0, 2]
    if r is not None:
        ax.plot(r.episode, roll(100 * r.collisions / r.completed.clip(lower=1)), color="grey", lw=2, label=rlab)
    ax.plot(d.episode, roll(100 * d.collisions / d.completed.clip(lower=1)), color="tab:red", lw=2.5, label=lab)
    ax.set_title("Collisions per 100 deliveries (5-ep mean)"); ax.set_xlabel("episode"); ax.legend(fontsize=8)
    # (d) preemption and the recovery head's value
    ax = axs[1, 0]
    ax.bar(d.episode, d.recovery_preemptive, color="tab:purple", alpha=0.5, label="preemptive recoveries")
    ax.plot(d.episode, roll(d.collisions), color="tab:red", lw=2, label="collisions (5-ep mean)")
    ax.set_xlabel("episode"); ax.set_title("Preemption falls, collisions creep up")
    ax2 = ax.twinx()
    if "qval_recovery" in d.columns:
        ax2.plot(d.episode, d.qval_recovery, color="black", lw=1.5, ls=":", label="recovery head value")
        ax2.set_ylabel("recovery head value", fontsize=8)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    # (e) holds
    ax = axs[1, 1]
    if "conflict_holds" in d.columns:
        ax.bar(d.episode, d.conflict_holds, color="tab:orange", alpha=0.45, label="holds")
        ax.bar(d.episode, d.conflict_holds_at_cap, color="tab:brown", alpha=0.9, label="holds at the 30-step cap")
        big = d.conflict_holds_at_cap.idxmax()
        ax.annotate(f"lockup, ep {d.episode[big]}", (d.episode[big], d.conflict_holds[big]), textcoords="offset points",
                    xytext=(10, -10), fontsize=8)
    ax.set_title("Recovery holds per episode"); ax.set_xlabel("episode"); ax.legend(fontsize=8)
    # (f) wait share
    ax = axs[1, 2]
    if "choice_policy_wait_share" in d.columns:
        ax.plot(d.episode, 100 * roll(d.choice_exec_wait_share), color="tab:gray", lw=2.5, label="waits executed")
        ax.plot(d.episode, 100 * roll(d.choice_policy_wait_share), color="tab:green", lw=2.5, label="waits the policy proposed")
        ax.fill_between(d.episode, 100 * roll(d.choice_policy_wait_share), 100 * roll(d.choice_exec_wait_share),
                        color="tab:gray", alpha=0.15, label="imposed by rules and holds")
    ax.set_title("Fleets at risk: share of decisions that waited"); ax.set_xlabel("episode"); ax.set_ylabel("%")
    ax.legend(fontsize=8)
    fig.suptitle(f"{lab}" + (f" against {rlab}" if r is not None else ""), fontsize=14)
    fig.tight_layout(); fig.savefig(f"{args.out}_overview.png", dpi=110); plt.close(fig)

    # drivers
    mech = [("choice_over_hold", "actions overridden by recovery holds"), ("conflict_holds", "recovery holds"),
            ("conflict_repeat_offences", "repeat offences"), ("conflict_holds_at_cap", "holds at the 30-step cap"),
            ("choice_over_rule", "actions overridden by the rules"), ("recovery_preemptive", "preemptive recoveries"),
            ("collisions", "collisions"), ("choice_policy_wait_share", "waits the policy proposed")]
    rows = []
    dd = d.completed.diff()
    for c, name in mech:
        if c in d.columns:
            rows.append((name, float(np.corrcoef(dd.iloc[1:], d[c].diff().iloc[1:])[0, 1])))
    if rows:
        rows.sort(key=lambda x: x[1])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh([x[0] for x in rows], [x[1] for x in rows],
                color=["tab:red" if v < -0.3 else "tab:gray" for _, v in rows])
        ax.axvline(0, color="black", lw=0.8)
        n = len(d) - 1
        crit = float(np.tanh(1.96 / np.sqrt(max(1, n - 3))))
        ax.axvline(-crit, color="black", ls=":", lw=0.8); ax.axvline(crit, color="black", ls=":", lw=0.8)
        ax.set_xlim(-1, 1)
        ax.set_title(f"{lab}: what moves with deliveries from one episode to the next\n"
                     f"(correlation of changes; dotted: p<0.05 at n={n})", fontsize=10)
        fig.tight_layout(); fig.savefig(f"{args.out}_drivers.png", dpi=110); plt.close(fig)
    print(f"wrote {args.out}_overview.png and {args.out}_drivers.png")


if __name__ == "__main__":
    main()
