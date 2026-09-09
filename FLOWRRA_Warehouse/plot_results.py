"""
plot_results.py

Figures for the FLOWRRA warehouse write-up, from the benchmark and training CSVs.

DESIGN RULE THROUGHOUT: show the distribution, not just the mean.

That is not a stylistic preference. In this data the mean actively lies. The
naive-recovery arm averages 38.6 collisions per instance, and that number is one
instance at 736 and nineteen between 0 and 8. A bar chart of means would draw a
tall bar and imply "collides somewhat more", when the truth is "behaves like
FLOWRRA on 19 of 20 instances and catastrophically on the twentieth". Those are
different claims and only the second is true, so every figure here plots the
per-instance points on top of whatever summary it shows.

Usage:
    # benchmark figures
    python plot_results.py benchmark benchmark_failure_all.csv --out figs

    # training curves; pass any number of runs with labels
    python plot_results.py training \
        ckpt_pilot2/curriculum_metrics.csv=Pilot2 \
        ckpt_pilot3/curriculum_metrics.csv=Pilot3 --out figs
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# Colour-blind-safe, and consistent across every figure so the arms are
# recognisable without reading the legend each time.
C_FLOW = "#0072B2"     # FLOWRRA
C_NAIVE = "#D55E00"    # heuristic recovery
C_PLAN = "#666666"     # planner alone
C_ACC = "#009E73"      # accent / positive
C_WARN = "#CC79A7"

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def arm_colour(name):
    if name.startswith("FLOWRRA"):
        return C_FLOW
    if "naive" in name:
        return C_NAIVE
    return C_PLAN


def short(name):
    return (name.replace("FLOWRRA(eps=", "FLOWRRA ε=").replace(")", "")
                .replace("RHCR-PIBT+naive", "RHCR + naive recovery")
                .replace("RHCR-PIBT", "RHCR alone"))


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name)
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}")


# =============================================================================
# BENCHMARK FIGURES
# =============================================================================

def fig_headline(d, out):
    """Recovery and completion by arm, with every instance shown as a point."""
    arms = list(d.groupby("algorithm").recovery_rate.mean().sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, col, title, ylab in [
        (axes[0], "recovery_rate", "Orphaned orders recovered", "fraction recovered"),
        (axes[1], "completion_rate", "Orders delivered overall", "completion rate"),
    ]:
        means = [d[d.algorithm == a][col].mean() for a in arms]
        cols = [arm_colour(a) for a in arms]
        ax.barh(range(len(arms)), means, color=cols, alpha=0.75, height=0.6)
        for i, a in enumerate(arms):
            v = d[d.algorithm == a][col].values
            ax.scatter(v, np.full_like(v, i, dtype=float) + np.random.uniform(-.13, .13, len(v)),
                       s=9, color="black", alpha=0.45, zorder=3, linewidths=0)
            # Label INSIDE the bar when it fits. Placed outside, these run into
            # the next subplot's y-tick labels and render as "0.86FLOWRRA e=0.10".
            inside = means[i] > 0.18
            ax.text(means[i] - 0.02 if inside else means[i] + 0.02, i,
                    f"{means[i]:.3f}", va="center",
                    ha="right" if inside else "left", fontsize=9,
                    color="white" if inside else "black",
                    fontweight="bold" if inside else "normal")
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([short(a) for a in arms], fontsize=9)
        ax.set_xlabel(ylab)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlim(0, max(1.05, max(means) * 1.12))

    fig.subplots_adjust(wspace=0.42)
    fig.suptitle("20 instances · 2 maps · 50 vehicles · 12 induced failures each",
                 fontsize=10, y=1.03, x=0.01, ha="left", color="#555")
    save(fig, out, "fig1_headline.png")


def fig_collisions(d, out):
    """
    The figure that stops the mean from lying.

    Log-scaled strip plot of every instance. The naive arm's median sits with
    everyone else's; its worst case is two orders of magnitude away. A bar of
    means would render that as a uniformly worse arm, which is not what happened.
    """
    arms = list(d.groupby("algorithm").collisions.median().sort_values().index)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, a in enumerate(arms):
        v = d[d.algorithm == a].collisions.values.astype(float)
        ax.scatter(np.clip(v, 0.5, None), np.full_like(v, i) + np.random.uniform(-.14, .14, len(v)),
                   s=26, color=arm_colour(a), alpha=0.7, linewidths=0)
        med, mx = np.median(v), v.max()
        ax.plot([max(med, 0.5)], [i], marker="|", ms=22, color="black", mew=2, zorder=4)
        if mx > 50:
            # Below the point, not above: above collides with the title.
            ax.annotate(f"worst case: {int(mx)} collision-steps\n(fleet integrity 0.06)",
                        xy=(mx, i), xytext=(mx * 0.22, i - 0.55), fontsize=8.5,
                        color=C_NAIVE, ha="right", va="top",
                        arrowprops=dict(arrowstyle="->", color=C_NAIVE, lw=1))
    ax.set_xscale("log")
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([short(a) for a in arms], fontsize=9)
    ax.set_xlabel("collision-steps per instance (log scale; 0 plotted at 0.5)")
    ax.set_ylim(-0.9, len(arms) - 0.4)
    ax.set_title("Every instance, not the mean. Black bar = median.",
                 fontsize=11, loc="left", pad=12)
    save(fig, out, "fig2_collisions.png")


def fig_paired(d, out, flow="FLOWRRA(eps=0.10)", base="RHCR-PIBT+naive"):
    """
    Per-instance paired differences. Every arm ran the SAME instances, so this
    is the honest view: how often did FLOWRRA win, and by how much, rather than
    a difference of two averages that could hide a split decision.
    """
    d = d.copy()
    d["instance"] = d["map"] + "|s" + d.seed.astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, lab in [(axes[0], "recovery_rate", "recovery rate"),
                         (axes[1], "completion_rate", "completion rate")]:
        p = d.pivot_table(index="instance", columns="algorithm", values=col)
        if flow not in p or base not in p:
            continue
        diff = (p[flow] - p[base]).sort_values()
        cols = [C_FLOW if v > 0 else (C_NAIVE if v < 0 else "#999") for v in diff]
        ax.barh(range(len(diff)), diff.values, color=cols, height=0.72)
        ax.axvline(0, color="black", lw=1)
        w, l, t = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
        ax.set_title(f"{lab}: {w} wins / {t} ties / {l} losses", fontsize=11, loc="left")
        ax.set_xlabel(f"FLOWRRA ε=0.10  −  RHCR + naive   ({lab})")
        ax.set_yticks([])
        ax.set_ylabel("instances, sorted")
    fig.suptitle("Paired per-instance differences — same instances, same failures",
                 fontsize=10, y=1.02, x=0.01, ha="left", color="#555")
    save(fig, out, "fig3_paired.png")


def fig_cost(d, out):
    """Travel cost and episode length: what recovery cost the rest of the fleet."""
    arms = list(d.groupby("algorithm").soc_hops.mean().sort_values().index)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    fig.subplots_adjust(wspace=0.42)   # right panel's tick labels were sitting on
                                       # top of the left panel's data points
    for ax, col, title, xlab in [
        (axes[0], "soc_hops", "Total travel cost", "sum of costs (hops)"),
        (axes[1], "steps_run", "Episode length", "simulator steps (cap = 780)"),
    ]:
        for i, a in enumerate(arms):
            v = d[d.algorithm == a][col].values.astype(float)
            ax.scatter(v, np.full_like(v, i) + np.random.uniform(-.13, .13, len(v)),
                       s=22, color=arm_colour(a), alpha=0.7, linewidths=0)
            ax.plot([np.median(v)], [i], marker="|", ms=20, color="black", mew=2, zorder=4)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([short(a) for a in arms], fontsize=9)
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=11, loc="left")
    axes[1].axvline(780, color=C_NAIVE, ls="--", lw=1)
    axes[1].text(780, len(arms) - 0.4, " step cap", color=C_NAIVE, fontsize=8, va="top")
    # FLOWRRA's episode lengths are BIMODAL -- an instance either finishes in
    # 150-430 steps or runs into the cap -- so its median falls in the empty gap
    # between the two clusters and sits where no instance actually landed. Worth
    # saying, because a median with no points near it reads as a plotting error.
    # Caption on the FIGURE, not the right axes -- anchored to the axes it ran
    # off the right edge of the image.
    fig.text(0.01, -0.03,
             "FLOWRRA's episode length is bimodal: an instance either finishes or "
             "hits the cap, so the median lands between the two clusters.",
             fontsize=8.5, color="#555", va="top", ha="left")
    save(fig, out, "fig4_cost.png")


def fig_epsilon(d, out):
    """The exploration sweep. Included because it is a NEGATIVE result."""
    e = d[d.algorithm.str.startswith("FLOWRRA")]
    if "eval_epsilon" not in e.columns or e.eval_epsilon.isna().all():
        return
    g = e.groupby("eval_epsilon")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    fig.subplots_adjust(wspace=0.34)
    for ax, col, lab in [(axes[0], "recovery_rate", "recovery rate"),
                         (axes[1], "completion_rate", "completion rate"),
                         (axes[2], "soc_hops", "travel cost")]:
        m, se = g[col].mean(), g[col].sem()
        ax.errorbar(m.index, m.values, yerr=se.values, marker="o", color=C_FLOW,
                    capsize=3, lw=1.5)
        ax.set_xlabel("evaluation exploration rate (ε)")
        ax.set_title(lab, fontsize=10, loc="left")   # title only; the y-label
                                                     # duplicated it and ran into
                                                     # the neighbouring panel
    axes[0].text(0.02, 0.06, "flat: p=0.99", transform=axes[0].transAxes, fontsize=9, color="#555")
    axes[1].text(0.02, 0.06, "flat: p=0.85", transform=axes[1].transAxes, fontsize=9, color="#555")
    axes[2].text(0.02, 0.88, "rising: p=0.027", transform=axes[2].transAxes, fontsize=9, color=C_NAIVE)
    fig.suptitle("Exploration sweep — a negative result. Only travel cost moves, and it moves the wrong way.",
                 fontsize=10, y=1.04, x=0.01, ha="left", color="#555")
    save(fig, out, "fig5_epsilon.png")


def fig_rescuer_deaths(d, out):
    """The mechanism behind the recovery gap, in one column."""
    if "rescuer_deaths" not in d.columns:
        return
    arms = [a for a in d.algorithm.unique() if d[d.algorithm == a].rescuer_deaths.sum() > 0]
    if not arms:
        return
    arms = sorted(arms, key=lambda a: d[d.algorithm == a].rescuer_deaths.mean())
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    means = [d[d.algorithm == a].rescuer_deaths.mean() for a in arms]
    ax.barh(range(len(arms)), means, color=[arm_colour(a) for a in arms], alpha=0.8, height=0.55)
    for i, m in enumerate(means):
        ax.text(m + 0.06, i, f"{m:.2f}", va="center", fontsize=9)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([short(a) for a in arms], fontsize=9)
    ax.set_xlabel("rescuers lost per instance")
    ax.set_title("Why the heuristic recovers fewer: it loses its rescuers en route",
                 fontsize=11, loc="left")
    save(fig, out, "fig6_rescuer_deaths.png")


# =============================================================================
# TRAINING FIGURES
# =============================================================================

def fig_training(runs, out):
    """
    Training curves across one or more runs.

    Runs are drawn as SEPARATE PANELS rather than concatenated into one
    timeline, because consecutive pilots differed in more than episode count --
    reward constants and injection rates changed between them. Splicing them
    into a single x-axis would imply a continuous experiment that never
    happened.
    """
    n = len(runs)
    metrics = [
        ("recovery_wasted", "wasted recovery invocations", True),
        ("completion_rate", "completion rate", False),
        ("gradient_agreement", "agreement with shortest-path field", False),
        ("collisions", "collisions per episode", False),
    ]
    fig, axes = plt.subplots(len(metrics), n, figsize=(5.6 * n, 2.7 * len(metrics)),
                             squeeze=False, sharex="col")
    for j, (label, d) in enumerate(runs):
        for i, (col, ylab, logy) in enumerate(metrics):
            ax = axes[i][j]
            if col not in d.columns:
                ax.set_visible(False)
                continue
            y = d[col].astype(float)
            ax.plot(d.episode, y, lw=0.7, color="#bbb")
            ax.plot(d.episode, y.rolling(15, min_periods=1).mean(), lw=2, color=C_FLOW)
            if logy:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_ylabel(ylab, fontsize=8.5)
            if i == 0:
                ax.set_title(label, fontsize=11)
            if i == len(metrics) - 1:
                ax.set_xlabel("episode")
    fig.suptitle("Training. Thin line = per episode, thick = 15-episode rolling mean.",
                 fontsize=10, y=1.005, x=0.01, ha="left", color="#555")
    save(fig, out, "fig7_training.png")


def fig_heads(runs, out):
    """
    Per-head TD loss. The point of decomposing the reward was that a head
    sitting at zero is diagnosable: it means either its reward never fires, or
    nothing in the state lets it tell its situation apart. Both are real bugs
    and both are invisible in a single summed loss.
    """
    heads = ["loss_goal", "loss_safety", "loss_integrity", "loss_rescue", "loss_time"]
    cols = {"loss_goal": C_FLOW, "loss_safety": C_NAIVE, "loss_integrity": C_ACC,
            "loss_rescue": C_WARN, "loss_time": "#666"}
    runs = [(l, d) for l, d in runs if any(h in d.columns for h in heads)]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(6.2 * len(runs), 4), squeeze=False)
    for j, (label, d) in enumerate(runs):
        ax = axes[0][j]
        for h in heads:
            if h not in d.columns:
                continue
            ax.plot(d.episode, d[h].rolling(10, min_periods=1).mean(),
                    lw=1.8, color=cols[h], label=h.replace("loss_", ""))
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("episode")
        ax.set_ylabel("TD loss (10-episode rolling mean, symlog)")
        ax.set_title(label, fontsize=11)
        ax.legend(fontsize=8.5, ncol=2, frameon=False)
    fig.suptitle("Per-head learning. A head pinned near zero is starved, not converged.",
                 fontsize=10, y=1.02, x=0.01, ha="left", color="#555")
    save(fig, out, "fig8_heads.png")


def fig_handover(runs, out):
    """Handover completion as a RATE -- error injection is stochastic, so raw
    counts would track how many failures an episode happened to get."""
    runs = [(l, d) for l, d in runs
            if {"errors_injected", "handovers_completed"} <= set(d.columns)]
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 25
    thin = []
    for (label, d), c in zip(runs, [C_FLOW, C_ACC, C_NAIVE, C_WARN]):
        e = d.errors_injected.rolling(w, min_periods=5).sum()
        h = d.handovers_completed.rolling(w, min_periods=5).sum()
        # EVENTS PER WINDOW is the number that decides whether this curve means
        # anything. A run at the original failure rate of 0.0008/step sees ~0.28
        # failures an episode, so a 25-episode window holds about 7 events and a
        # single handover moves the rate by 14 points -- the line lurches between
        # 0.12 and 1.0 and none of it is behaviour. At 0.006/step the same window
        # holds ~58 events and the curve becomes readable. Putting the count in
        # the legend stops the jagged run being read as instability.
        per_win = float(d.errors_injected.mean()) * w
        thin.append(per_win < 20)
        ax.plot(d.episode, (h / e.replace(0, np.nan)),
                lw=(1.3 if per_win < 20 else 2.2),
                ls=("--" if per_win < 20 else "-"),
                alpha=(0.65 if per_win < 20 else 1.0),
                color=c, label=f"{label}  (~{per_win:.0f} events/window)")
    ax.set_xlabel("episode")
    ax.set_ylabel("handovers completed / failures injected")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    ax.set_title(f"Handover completion rate ({w}-episode rolling window)",
                 fontsize=11, loc="left")
    if any(thin):
        ax.text(0.0, -0.28,
                "Dashed = too few failures per window to read. The rescue reward and its "
                "Q-head were present in every run;\nwhat changed was the failure "
                "injection rate (0.0008/step to 0.006/step), and with it the number of "
                "events the head had to learn from.",
                transform=ax.transAxes, fontsize=8.5, color="#555", va="top")
    save(fig, out, "fig9_handover.png")


# =============================================================================

def fig_distance(frames, out):
    """
    Recovery advantage against map size.

    The point of plotting against NODES rather than occupancy is that occupancy
    was the wrong axis. It moved with fleet count, and fleet count is exactly
    what the heuristic baseline is sensitive to -- a denser fleet puts an idle
    vehicle closer to every failure, so nearest-idle reassignment gets BETTER as
    density rises. Map size does the opposite: it lengthens the rescue, and a
    rescuer in transit is a rescuer exposed to the next failure wave.

    So the two baselines move in opposite directions with fleet count, and any
    single-map comparison confounds them. Three map sizes at matched fleet
    counts separates the two.
    """
    d = pd.concat(frames, ignore_index=True)
    if "map_nodes" not in d.columns:
        return
    order = d.groupby("map").map_nodes.first().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    for ax, col, lab, invert in [
        (axes[0], "recovery_rate", "orphaned orders recovered", False),
        (axes[1], "completion_rate", "orders delivered overall", False)]:
        for arm in ["FLOWRRA", "RHCR-PIBT+naive", "RHCR-PIBT"]:
            sub = d[d.algorithm == arm]
            if sub.empty:
                continue
            ys = [sub[sub["map"] == m][col].mean() for m in order.index]
            es = [sub[sub["map"] == m][col].sem() for m in order.index]
            ax.errorbar(order.values, ys, yerr=es, marker="o", lw=2, capsize=4,
                        color=arm_colour(arm), label=short(arm))
            for m, y in zip(order.index, ys):
                pts = sub[sub["map"] == m][col].values
                x = np.full_like(pts, order[m], dtype=float)
                ax.scatter(x * np.random.uniform(0.93, 1.07, len(pts)), pts,
                           s=8, color=arm_colour(arm), alpha=0.25, linewidths=0)
        # Log scale draws its own minor ticks (2x10^3, 3x10^3 ...) which
        # overprint the three labels we care about. Clear both locators.
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.FixedLocator(order.values))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticklabels([f"{int(v/1000)}k" if v >= 1000 else str(int(v))
                            for v in order.values], rotation=45, ha="right")
        ax.set_xlabel("map size (navigable nodes, log scale)")
        ax.set_ylabel(lab)
        ax.set_title(lab, fontsize=11, loc="left")
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(fontsize=9, frameon=False, loc="center right")
    fig.suptitle("Fleet counts matched at 25/40/60 on every map; 10 seeds each. "
                 "Bars are standard error, points are instances.",
                 fontsize=9.5, y=1.03, x=0.01, ha="left", color="#555")
    save(fig, out, "fig10_distance.png")


def fig_rescue_mechanism(frames, out):
    """Why the gap opens: how far the rescue is, and how many rescuers die making it."""
    d = pd.concat(frames, ignore_index=True)
    order = d.groupby("map").map_nodes.first().sort_values()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    # Both arms, now that FLOWRRA reports latency too. It previously showed only
    # the heuristic -- not because FLOWRRA never finished a rescue, but because
    # the measurement was only implemented on one side.
    w0 = 0.36
    for i, arm in enumerate(["FLOWRRA", "RHCR-PIBT+naive"]):
        sub = d[d.algorithm == arm]
        if "mean_recovery_steps" not in sub.columns or sub.mean_recovery_steps.isna().all():
            continue
        axes[0].bar(np.arange(len(order)) + (i - 0.5) * w0,
                    [sub[sub["map"] == m].mean_recovery_steps.mean() for m in order.index],
                    width=w0, color=arm_colour(arm), alpha=0.85, label=short(arm))
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([f"{int(v/1000)}k" if v >= 1000 else str(int(v))
                             for v in order.values])
    axes[0].set_xlabel("map size (nodes)")
    axes[0].set_ylabel("steps from failure to delivery")
    axes[0].set_title("Rescue takes longer on bigger maps", fontsize=11, loc="left")
    # Only the heuristic reports this -- FLOWRRA's rescues are handled inside the
    # orchestrator and never surface as a dispatch-to-delivery latency. Without
    # saying so the orange bars look like an unlabelled comparison.
    axes[0].legend(fontsize=9, frameon=False, loc="upper left")

    w = 0.36
    for i, arm in enumerate(["FLOWRRA", "RHCR-PIBT+naive"]):
        sub = d[d.algorithm == arm]
        axes[1].bar(np.arange(len(order)) + (i - 0.5) * w,
                    [sub[sub["map"] == m].rescuer_deaths.mean() for m in order.index],
                    width=w, color=arm_colour(arm), alpha=0.85, label=short(arm))
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([f"{int(v/1000)}k" if v >= 1000 else str(int(v))
                             for v in order.values])
    axes[1].set_xlabel("map size (nodes)")
    axes[1].set_ylabel("rescuers lost per instance")
    axes[1].set_title("And the heuristic loses more of them", fontsize=11, loc="left")
    axes[1].legend(fontsize=9, frameon=False)
    save(fig, out, "fig11_rescue_mechanism.png")


def fig_v4_by_map(d, out):
    """Per-map comparison on the columns that survived the measurement audit."""
    maps = sorted(d["map"].unique(), key=lambda m: d[d["map"] == m].map_nodes.iloc[0])
    labs = [f"{int(d[d['map']==m].map_nodes.iloc[0]/1000)}k" if
            d[d["map"]==m].map_nodes.iloc[0] >= 1000 else str(int(d[d["map"]==m].map_nodes.iloc[0]))
            for m in maps]
    panels = [("recovery_rate", "orphaned orders recovered", False),
              ("completion_rate", "orders delivered", False),
              ("rescuer_deaths", "rescuers lost per instance", True),
              ("mean_recovery_hops", "hops from failure to delivery", True)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.9))
    w = 0.26
    for ax, (col, lab, _) in zip(axes, panels):
        for i, arm in enumerate(["FLOWRRA", "RHCR-PIBT+naive", "RHCR-PIBT"]):
            sub = d[d.algorithm == arm]
            if col not in sub.columns or sub[col].isna().all():
                continue
            ys = [sub[sub["map"] == m][col].mean() for m in maps]
            es = [sub[sub["map"] == m][col].sem() for m in maps]
            ax.bar(np.arange(len(maps)) + (i - 1) * w, ys, width=w, yerr=es,
                   capsize=3, color=arm_colour(arm), alpha=0.85, label=short(arm))
        ax.set_xticks(range(len(maps)))
        ax.set_xticklabels(labs)
        ax.set_xlabel("map size (nodes)")
        ax.set_title(lab, fontsize=10.5, loc="left")
    axes[0].legend(fontsize=8.5, frameon=False)
    fig.suptitle("Fleet counts matched at 25/40/60 on every map; 10 seeds each. "
                 "Bars are standard error.",
                 fontsize=9.5, y=1.04, x=0.01, ha="left", color="#555")
    save(fig, out, "fig12_v4_by_map.png")


def fig_v4_trend(d, out):
    """
    The result, as a trend rather than three separate comparisons.

    Each panel is FLOWRRA relative to the heuristic on one axis, plotted against
    map size. Plotting the RATIO rather than the two raw lines is what makes the
    story legible: the interesting fact is not the level on any single map, it is
    that every axis moves the same way as the building grows, and two of them
    cross the break-even line.
    """
    maps = sorted(d["map"].unique(), key=lambda m: d[d["map"] == m].map_nodes.iloc[0])
    x = [d[d["map"] == m].map_nodes.iloc[0] for m in maps]
    F = lambda s: s[s.algorithm == "FLOWRRA"]
    N = lambda s: s[s.algorithm == "RHCR-PIBT+naive"]

    series = [
        ("orphaned orders recovered", lambda s: F(s).recovery_rate.mean() / N(s).recovery_rate.mean(), True),
        ("rescuers kept alive",       lambda s: N(s).rescuer_deaths.mean() / F(s).rescuer_deaths.mean(), True),
        ("rescue speed",              lambda s: N(s).mean_recovery_hops.mean() / F(s).mean_recovery_hops.mean(), True),
        ("distance efficiency",       lambda s: N(s).distance_travelled.mean() / F(s).distance_travelled.mean(), True),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5))
    cols = [C_FLOW, C_ACC, C_WARN, C_NAIVE]
    _dy = []
    for (lab, fn, _), c in zip(series, cols):
        ys = [fn(d[d["map"] == m]) for m in maps]
        ax.plot(x, ys, marker="o", lw=2, color=c, label=lab)
        _dy.append(ys[-1])
        # nudge apart when two end-points land within 0.04 of each other
        off = 0
        for prev in _dy[:-1]:
            if abs(prev - ys[-1]) < 0.04:
                off = 9 if ys[-1] >= prev else -9
        ax.annotate(f"{ys[-1]:.2f}x", (x[-1], ys[-1]), xytext=(9, off),
                    textcoords="offset points", fontsize=9, color=c, va="center")
    # 1.0 is parity. Above it FLOWRRA wins that axis, below it the heuristic does.
    ax.axhline(1.0, color="black", lw=1.2, ls="--")
    ax.text(x[0], 1.02, "parity", fontsize=9, color="#444")
    # Same log-locator problem as fig_distance: matplotlib adds its own minor
    # ticks (2x10^3, 3x10^3 ...) on top of the three labels that matter.
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(mticker.FixedLocator(x))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xticklabels([f"{int(v/1000)}k" if v >= 1000 else str(int(v)) for v in x])
    ax.set_xlabel("map size (navigable nodes, log scale)")
    ax.set_xlim(x[0]*0.85, x[-1]*1.45)
    ax.set_ylabel("FLOWRRA relative to the heuristic  (>1 = FLOWRRA better)")
    ax.legend(fontsize=9.5, frameon=False, loc="upper left")
    ax.set_title("Every axis moves the same way as the floor gets bigger",
                 fontsize=12, loc="left")
    save(fig, out, "fig13_v4_trend.png")


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("mode", choices=["benchmark", "training", "distance", "v4"])
    ap.add_argument("files", nargs="+",
                    help="benchmark: one CSV. training: one or more PATH=Label pairs.")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--flowrra-arm", default="FLOWRRA(eps=0.10)")
    args = ap.parse_args()
    np.random.seed(0)   # jitter in the strip plots is cosmetic; keep it reproducible

    if args.mode == "v4":
        d = pd.read_csv(args.files[0])
        print(f"v4: {len(d)} rows, {d['map'].nunique()} maps")
        fig_v4_by_map(d, args.out)
        fig_v4_trend(d, args.out)
        print("done."); return

    if args.mode == "distance":
        frames = [pd.read_csv(f) for f in args.files]
        print(f"distance: {sum(len(f) for f in frames)} rows across {len(frames)} files")
        fig_distance(frames, args.out)
        fig_rescue_mechanism(frames, args.out)
        print("done."); return

    if args.mode == "benchmark":
        d = pd.read_csv(args.files[0])
        print(f"benchmark: {len(d)} rows, {d.algorithm.nunique()} arms")
        fig_headline(d, args.out)
        fig_collisions(d, args.out)
        fig_paired(d, args.out, flow=args.flowrra_arm)
        fig_cost(d, args.out)
        fig_epsilon(d, args.out)
        fig_rescuer_deaths(d, args.out)
    else:
        runs = []
        for spec in args.files:
            path, _, label = spec.partition("=")
            runs.append((label or os.path.basename(os.path.dirname(path)) or path,
                         pd.read_csv(path)))
            print(f"training run '{runs[-1][0]}': {len(runs[-1][1])} episodes")
        fig_training(runs, args.out)
        fig_heads(runs, args.out)
        fig_handover(runs, args.out)
    print("done.")


if __name__ == "__main__":
    main()