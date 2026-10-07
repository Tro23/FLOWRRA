"""
FLOWRRA in Warehouse, v2: the shock benchmark after the Conflict Ladder.

Reads (in the current directory):
  - benchmark_ladder_run1.csv    REQUIRED. Three arms on held-out all_scens_v2:
                                 RHCR-PIBT+naive, RULES, and FLOWRRA (the policy
                                 + RULES, ladder_run1's checkpoint).
  - benchmark_new_all_2.csv      optional. v1 benchmark (the original policy).
  - benchmark_teacher_run_1.csv  optional. teacher_run1's policy arm.
With both optional files present, a sixth panel and a second table compare the
three policy checkpoints on the instances all three benchmarks share.

Outputs:
  - flowrra_benchmark_v2_ladder.png      (300 DPI, six panels)
  - summary_table_v2_ladder.csv / .md    (the three arms, small / large map)
  - checkpoint_progression_v2.csv / .md  (only when the optional files exist)

Every comparison is paired by instance (map, seed, fleet count); p-values are
two-sided Wilcoxon signed-rank tests. All arms faced the same failure waves.
The original script (generate_benchmark_figures.py) is unchanged, so the
article's first figure can still be reproduced.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

SMALL, LARGE = "25_5_2_5_2_1", "50_20_5_10_5_2"
MAP_LABEL = {SMALL: "Small map\n1,435 nodes", LARGE: "Large map\n27,000 nodes"}
KEY = ["map", "seed", "num_agents"]

# Arms in a FIXED order; colour follows the arm (validated light palette).
ARMS = [("RULES", "RULES alone"),
        ("FLOWRRA", "Policy + RULES (ladder_run1)"),
        ("RHCR-PIBT+naive", "RHCR-PIBT + naive")]
COLOR = {"RULES": "#2a78d6", "FLOWRRA": "#eb6834", "RHCR-PIBT+naive": "#1baf7a"}
# Checkpoints are one arm over time: a single hue, light -> dark.
CKPT = [("v1", "v1 policy", "#f6b89a"),
        ("teacher", "teacher_run1", "#eb6834"),
        ("ladder", "ladder_run1", "#a23d12")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(path):
    return pd.read_csv(path, dtype={"map": str}) if os.path.exists(path) else None


def paired_p(a, b):
    a, b = a.align(b, join="inner")
    ok = a.notna() & b.notna()
    a, b = a[ok], b[ok]
    if len(a) < 5 or np.allclose(a.values, b.values):
        return float("nan")
    try:
        return float(wilcoxon(a, b).pvalue)
    except ValueError:
        return float("nan")


def fmt_p(p):
    if not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK, pad=10)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=9.5)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def grouped_bars(ax, means, fmt, title, ylabel, ylim=None):
    """means: {arm: [small, large]}; thin bars, 2px gaps, every value labelled."""
    x = np.arange(2)
    w = 0.24
    top = max(max(v) for v in means.values()) or 1.0
    for i, (arm, label) in enumerate(ARMS):
        vals = means[arm]
        bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=COLOR[arm], label=label,
                      edgecolor="white", linewidth=1.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + top * 0.015,
                    fmt(v), ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(x, [MAP_LABEL[SMALL], MAP_LABEL[LARGE]])
    if ylim:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(0, top * 1.18)
    style(ax, title, ylabel)


def main():
    new = load("benchmark_ladder_run1.csv")
    if new is None:
        raise FileNotFoundError("benchmark_ladder_run1.csv not found in the current directory.")
    v1, teacher = load("benchmark_new_all_2.csv"), load("benchmark_teacher_run_1.csv")

    arm = {a: new[new.algorithm == a].set_index(KEY) for a, _ in ARMS}
    n_inst = {m: len(arm["RULES"].xs(m, level="map")) for m in (SMALL, LARGE)}

    def mean(a, col, m):
        return float(arm[a].xs(m, level="map")[col].mean())

    def both(col, scale=1.0):
        return {a: [mean(a, col, SMALL) * scale, mean(a, col, LARGE) * scale] for a, _ in ARMS}

    # ---------------------------------------------------------------- figure
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(2, 3, figsize=(19, 11), facecolor="#fcfcfb")
    for ax in axs.flat:
        ax.set_facecolor("#fcfcfb")

    grouped_bars(axs[0, 0], both("recovery_rate", 100), lambda v: f"{v:.1f}%",
                 "A  Stranded orders recovered", "Recovery rate (%)", (0, 108))
    grouped_bars(axs[0, 1], both("rescuer_deaths"), lambda v: f"{v:.2f}",
                 "B  Rescuers lost to later failure waves", "Rescuer deaths per episode")
    grouped_bars(axs[0, 2], both("completion_rate", 100), lambda v: f"{v:.1f}%",
                 "C  Orders delivered", "Completion (%)", (0, 108))
    grouped_bars(axs[1, 0], both("mean_recovery_hops"), lambda v: f"{v:.0f}",
                 "D  Distance to reach a stranded order", "Mean recovery hops")
    grouped_bars(axs[1, 1], both("replan_s"), lambda v: f"{v:.1f} s",
                 "E  Time frozen re-planning after failures", "Replan downtime (s per episode)")

    ax = axs[1, 2]
    prog = None
    if v1 is not None and teacher is not None:
        sets = {"v1": v1[v1.algorithm == "FLOWRRA"].set_index(KEY),
                "teacher": teacher[teacher.algorithm == "FLOWRRA"].set_index(KEY),
                "ladder": arm["FLOWRRA"], "rules": arm["RULES"]}
        shared = sets["v1"].index
        for s in sets.values():
            shared = shared.intersection(s.index)
        prog = {k: s.loc[shared] for k, s in sets.items()}
        x = np.arange(2)
        w = 0.24
        for i, (k, label, c) in enumerate(CKPT):
            vals = [prog[k].xs(m, level="map")["recovery_rate"].mean() * 100 for m in (SMALL, LARGE)]
            bars = ax.bar(x + (i - 1) * w, vals, w - 0.02, color=c, label=label,
                          edgecolor="white", linewidth=1.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                        ha="center", va="bottom", fontsize=8.5, color=INK)
        for j, m in enumerate((SMALL, LARGE)):
            r = prog["rules"].xs(m, level="map")["recovery_rate"].mean() * 100
            ax.hlines(r, j - 1.6 * w, j + 1.6 * w, colors=COLOR["RULES"], linestyles="--", linewidth=2)
            # Label at the LEFT end, above the v1 bar, clear of the ladder bar's label.
            ax.text(j - 1.6 * w, r + 1.2, f"RULES alone {r:.1f}%", ha="left", va="bottom",
                    fontsize=8.5, color=COLOR["RULES"], fontweight="bold")
        ax.set_xticks(x, [MAP_LABEL[SMALL], MAP_LABEL[LARGE]])
        ax.set_ylim(0, 122)
        ax.set_yticks(range(0, 101, 20))
        n_sh = len(shared) // 2
        style(ax, f"F  Policy + RULES by checkpoint ({n_sh} shared instances/map)",
              "Recovery rate (%)")
        # Legend in the empty band above the bars, never over a same-coloured bar.
        ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3,
                  bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Panel F needs benchmark_new_all_2.csv\nand benchmark_teacher_run_1.csv",
                ha="center", va="center", color=MUTED)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("FLOWRRA v2 shock benchmark: held-out instances (all_scens_v2), "
                 f"{n_inst[SMALL]} per map, 3 failure waves of 3 vehicles each",
                 fontsize=14, fontweight="bold", color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_png = "flowrra_benchmark_v2_ladder.png"
    fig.savefig(out_png, dpi=300, facecolor=fig.get_facecolor())
    print(f"[figure] {out_png}")

    # ---------------------------------------------------------------- table 1
    rows = [("Stranded orders recovered", "recovery_rate", lambda v: f"{v * 100:.1f}%"),
            ("Rescuers lost per episode", "rescuer_deaths", lambda v: f"{v:.2f}"),
            ("Orders delivered", "completion_rate", lambda v: f"{v * 100:.1f}%"),
            ("Hops to reach a stranded order", "mean_recovery_hops", lambda v: f"{v:.1f}"),
            ("Collisions per episode", "collisions", lambda v: f"{v:.2f}"),
            ("Distance travelled (cells)", "distance_travelled", lambda v: f"{v:.0f}"),
            ("Replan downtime (s per episode)", "replan_s", lambda v: f"{v:.1f}")]
    t1 = []
    for name, col, f in rows:
        r = {"Metric": name}
        for a, label in ARMS:
            r[f"{label} (small / large)"] = f"{f(mean(a, col, SMALL))} / {f(mean(a, col, LARGE))}"
        ps = [paired_p(arm["FLOWRRA"].xs(m, level="map")[col], arm["RULES"].xs(m, level="map")[col])
              for m in (SMALL, LARGE)]
        r["Policy vs RULES, paired p (small / large)"] = f"{fmt_p(ps[0])} / {fmt_p(ps[1])}"
        t1.append(r)
    t1 = pd.DataFrame(t1)
    t1.to_csv("summary_table_v2_ladder.csv", index=False)
    with open("summary_table_v2_ladder.md", "w") as fh:
        fh.write(t1.to_markdown(index=False) + "\n")
    print("[table] summary_table_v2_ladder.csv / .md\n")
    print(t1.to_markdown(index=False))

    # ---------------------------------------------------------------- table 2
    if prog is not None:
        t2 = []
        for name, col, f in rows[:4]:
            r = {"Metric": name}
            for k, label, _ in CKPT + [("rules", "RULES alone", None)]:
                r[f"{label} (small / large)"] = " / ".join(
                    f(prog[k].xs(m, level="map")[col].mean()) for m in (SMALL, LARGE))
            ps = [paired_p(prog["ladder"].xs(m, level="map")[col], prog["v1"].xs(m, level="map")[col])
                  for m in (SMALL, LARGE)]
            r["ladder_run1 vs v1, paired p (small / large)"] = f"{fmt_p(ps[0])} / {fmt_p(ps[1])}"
            t2.append(r)
        t2 = pd.DataFrame(t2)
        t2.to_csv("checkpoint_progression_v2.csv", index=False)
        with open("checkpoint_progression_v2.md", "w") as fh:
            fh.write(t2.to_markdown(index=False) + "\n")
        print("\n[table] checkpoint_progression_v2.csv / .md\n")
        print(t2.to_markdown(index=False))


if __name__ == "__main__":
    main()