"""
render_pair.py -- one day with the jam-collapse rung off and on, side by side.

Both recordings come from record_day.py, same arm, map, fleets and seed:
    python record_day.py --arm ladder4 --fleets 60 --seed 1 --cond full --collapse off
    python record_day.py --arm ladder4 --fleets 60 --seed 1 --cond full --collapse on

THREE VIEWS
    python render_pair.py OFF.json.gz ON.json.gz
        -> pair_..._day.gif     the whole day, both panels at the same step
                                (--from/--to for a part of it)
    python render_pair.py OFF.json.gz ON.json.gz --close fork
        -> pair_..._fork.gif    a close-up of the rung's first joint plan. Same
                                seed, failures and people: the two days are
                                identical step for step up to this plan, so both
                                panels show the same jam. Left, the fleets sort it
                                out on their own; right, they follow the plan.
    python render_pair.py OFF.json.gz ON.json.gz --close thrml
        -> pair_..._thrml.gif   a close-up of the first plan sampled by thrml (a
                                jam of 5-8 fleets, or one exact search could not
                                settle in time). By then the days have parted, so
                                the left panel shows the same place and time
                                without the rung, not the same jam; the title
                                says so.
    --close-at T                a close-up of the plan that starts nearest step T
    --list                      the plans that ran furthest (id, steps, hops got through)
    --plan ID                   a close-up of that plan, from 6 steps before it to 10 after it ends
    --html                      also an interactive page (play, pause, scrub)

ON SCREEN (as render_day.py, plus the rung)
    aqua square ring   on a joint plan made by exact search (2-4 fleets)
    aqua diamond ring  on a joint plan sampled by thrml
    aqua line          the hops left in that fleet's plan
    fleet numbers      in a close-up, the plan's fleets are numbered in both
                       panels, so the same fleets can be followed side by side
Under each panel: orders delivered, intent-guard stops, near-miss steps with
people and cells driven so far (and, on the right, plans made so far).

Needs matplotlib and pillow (GIF), plotly (HTML). Reads only the recordings.
"""
from __future__ import annotations

import argparse
import os
import textwrap

from render_day import (AQUA, BLUE, GREEN, INK, MAGENTA, MUTED, ORANGE, SHELF, STATE_STYLE, SURFACE,
                        VIOLET, YELLOW, Layout, load)

PLAN = AQUA
RING = {"exact": ("s", "square-open", "joint plan, exact search"),
        "thrml": ("D", "diamond-open", "joint plan, thrml")}


# ----------------------------------------------------------------------------- the pair
class Pair:
    def __init__(self, off_path, on_path):
        self.off, self.on = load(off_path), load(on_path)
        for k in ("map", "k", "seed", "cond", "arm"):
            if self.off.get(k) != self.on.get(k):
                raise SystemExit(f"[pair] the recordings differ in {k}: {self.off.get(k)} vs {self.on.get(k)}")
        if self.on.get("collapse") != "on":
            print("[pair] note: the second recording was not made with --collapse on")
        self.by_t = [{f["t"]: f for f in d["frames"]} for d in (self.off, self.on)]
        self.ts = sorted(set(self.by_t[0]) & set(self.by_t[1]))
        self.lay = Layout(self.off["static"]["cells"])
        self.cells = self.off["static"]["cells"]
        self.blk = self.lay.blocked(self.cells)
        self.fork_t = next((t for t in self.ts if self._pos(self.by_t[0][t]) != self._pos(self.by_t[1][t])), None)

    @staticmethod
    def _pos(f):
        return sorted((r[0], r[1], r[2], r[3]) for r in f["fleets"])

    def P(self, c):
        return self.lay.xy(*c)

    def frame(self, side, t):
        return self.by_t[side][t]

    def first_plan(self, how=None, near=None):
        """(step, plan) for the first plan (of a solver), or the one starting nearest step `near`."""
        seen, best = set(), None
        for t in self.ts:
            for p in self.by_t[1][t].get("plans", []) or []:
                if p["id"] in seen or (how and p["how"] != how):
                    continue
                seen.add(p["id"])
                if near is None:
                    return t, p
                if best is None or abs(t - near) < abs(best[0] - near):
                    best = (t, p)
        return best

    def plan_life(self, pid):
        """(first step, last step, the plan as first seen, hops it got through, hops planned, remade?)."""
        H1 = max((len(v) for f in self.by_t[1].values() for p in f.get("plans", []) or []
                  for v in p["path"].values()), default=9)          # cells in a fresh plan: horizon + 1
        t0 = t1 = first = last = None
        for t in self.ts:
            for p in self.by_t[1][t].get("plans", []) or []:
                if p["id"] == pid:
                    if t0 is None:
                        t0, first = t, p
                    t1, last = t, p
        if t0 is None:
            return None
        hops = H1 - max(len(v) for v in last["path"].values())
        remade = any(p["id"] != pid and p["id"] > pid and set(p["members"]) & set(first["members"])
                     for t in self.ts if t1 <= t <= t1 + 2 for p in self.by_t[1][t].get("plans", []) or [])
        return t0, t1, first, hops, H1 - 1, remade

    def same_place(self, plan, t, within=3):
        """How many of the plan's fleets are within `within` cells of the same spot in the day without the rung."""
        a = {r[0]: r[1:4] for r in self.by_t[0][t]["fleets"]}
        b = {r[0]: r[1:4] for r in self.by_t[1][t]["fleets"]}
        return sum(1 for m in plan["members"] if m in a and m in b
                   and sum(abs(x - y) for x, y in zip(a[m], b[m])) <= within)

    def box(self, plan, t, pad=5, min_w=18, min_h=12):
        f = self.by_t[1][t]
        pts = [self.P(r[1:4]) for r in f["fleets"] if r[0] in plan["members"]]
        for cells in plan["path"].values():
            pts += [self.P(c) for c in cells]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
        if x1 - x0 < min_w:
            c = (x0 + x1) / 2
            x0, x1 = c - min_w / 2, c + min_w / 2
        if y1 - y0 < min_h:
            c = (y0 + y1) / 2
            y0, y1 = c - min_h / 2, c + min_h / 2
        return x0, x1, y0, y1

    def last_t(self):
        """Shortly after the last delivery on either side; the floor is still after that."""
        last = 0
        for side in (0, 1):
            fr = [self.by_t[side][t] for t in self.ts]
            for i in range(1, len(fr)):
                if fr[i]["delivered"] > fr[i - 1]["delivered"]:
                    last = max(last, fr[i]["t"])
        return min(self.ts[-1], last + 40) if last else self.ts[-1]

    def head(self):
        d = self.off
        h = f"FLOWRRA · {d['arm']} · {d['map']} · {d['k']} fleets · seed {d['seed']} · {d['cond']}"
        # record_day.py --check: say so when a recording is a rerun rather than the scored day.
        rerun = [name for name, r in (("rung off", self.off), ("rung on", self.on)) if r.get("check") == "differs"]
        if rerun:
            h += "  ·  " + " and ".join(rerun) + ": a rerun, not the scored day"
        return h

    def counts(self, side, f):
        book = self.off["static"].get("book_total") or 0
        g = f.get("guard")
        s = (f"delivered {f['delivered']}/{book}  ·  guard stops {'–' if g is None else g}  ·  "
             f"near misses {f.get('near', '–')}  ·  driven {int(f['driven']) if 'driven' in f else '–'}")
        if side == 1 and "n_exact" in f:
            s += f"\nplans so far: {f['n_exact']} exact, {f['n_thrml']} thrml"
        return s


# ----------------------------------------------------------------------------- GIF
def render_gif(pair, out, ts, box=None, focus=(), focus_plan=None, note="", fps=12, dpi=80):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import PatchCollection
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

    P, cells = pair.P, pair.cells
    allx = [P(c)[0] for c in cells.values()] + [b[0] for b in pair.blk]
    ally = [P(c)[1] for c in cells.values()] + [b[1] for b in pair.blk]
    x0, x1, y0, y1 = box or (min(allx) - 1, max(allx) + 1, min(ally) - 1, max(ally) + 1)
    W, H = x1 - x0, y1 - y0
    # Size the figure to the panels, so there is no empty band above them.
    fw = 15.6 if box is not None else max(13.0, 2 * W * 0.15 + 0.6)
    pw = (fw * 0.98 - 0.15) / 2                  # one panel's width, inches
    ph = pw * H / max(W, 1)
    head, foot = 1.6, 0.75                       # inches for the titles and the legend
    fh = ph + head + foot
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - head / fh, bottom=foot / fh, wspace=0.03)
    big = box is not None
    k = max(1.0, min(4.0, (pw / max(W, 1) / 0.15) ** 2))   # marker scale: bigger cells, bigger marks
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_collection(PatchCollection([Rectangle((x - 0.46, y - 0.46), 0.92, 0.92) for x, y in pair.blk
                                           if x0 - 1 <= x <= x1 + 1 and y0 - 1 <= y <= y1 + 1],
                                          facecolor=SHELF, edgecolor="none"))
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        dk = [P(cells[d]) for d in pair.off["static"]["docks"] if d in cells]
        ax.scatter([p[0] for p in dk], [p[1] for p in dk], s=30 * k, marker="s", c=GREEN,
                   linewidths=0)

    def h(label, mk, col, filled=True, ms=6, ls="none"):
        return Line2D([], [], linestyle=ls, marker=mk, markersize=ms, label=label, color=col,
                      markerfacecolor=col if filled else "none", markeredgecolor=col,
                      markeredgewidth=1.5 if not filled or mk == "x" else 0)
    handles = [h("driving", "o", BLUE), h("waiting", "o", BLUE, False), h("parked", "o", MUTED),
               h("to a dock", "o", AQUA), h("rescuer", "o", VIOLET), h("failed", "x", STATE_STYLE[5][1]),
               h("carrying", "o", INK, False, 8), h("path warning", "o", ORANGE, False, 9),
               h(RING["exact"][2], "s", PLAN, False, 9), h(RING["thrml"][2], "D", PLAN, False, 8),
               h("hops left in the plan", None, PLAN, ls="-"),
               h("order", "s", YELLOW, False), h("person", "*", MAGENTA, ms=9), h("debris", "^", ORANGE),
               h("dock", "s", GREEN), h("rack", "s", SHELF, ms=8)]
    fig.legend(handles=handles, loc="lower center", ncol=8, frameon=False, fontsize=8.5, labelcolor=MUTED,
               handletextpad=0.3, columnspacing=0.9)
    sup = fig.suptitle("", color=INK, fontsize=12, x=0.01, ha="left", y=0.99, va="top")
    titles = [axes[0].set_title("", color=INK, fontsize=10.5, loc="left"),
              axes[1].set_title("", color=INK, fontsize=10.5, loc="left")]
    arts = []

    def draw(i):
        for a in arts:
            a.remove()
        arts.clear()
        t = ts[i]
        for side, ax in enumerate(axes):
            f = pair.frame(side, t)

            def sc(pts, **kw):
                if pts:
                    arts.append(ax.scatter([p[0] for p in pts], [p[1] for p in pts], **kw))
            sc([P(r[:3]) for r in f["orders"] if not r[3]], s=16 * k, marker="s", facecolors="none",
               edgecolors=YELLOW)
            sc([P(r[:3]) for r in f["orders"] if r[3]], s=16 * k, marker="s", c=YELLOW)
            sc([P(r) for r in f["pickups"]], s=40 * k, marker="D", c=VIOLET)
            sc([P(r) for r in f["debris"]], s=45 * k, marker="^", c=ORANGE, linewidths=0)
            sc([P(r[:3]) for r in f["humans"]], s=90 * k, marker="*", c=MAGENTA, edgecolors=INK, linewidths=0.5)
            fl = f["fleets"]
            pos = {r[0]: P(r[1:4]) for r in fl}
            for p in f.get("plans", []) or []:
                # In a close-up, the plan in focus is drawn bold; other plans nearby stay faint.
                dim = focus_plan is not None and p["id"] != focus_plan
                for fid, cells_left in p["path"].items():
                    if fid in pos and cells_left:
                        xy = [pos[fid]] + [P(c) for c in cells_left]
                        arts.extend(ax.plot([q[0] for q in xy], [q[1] for q in xy], color=PLAN,
                                            alpha=0.3 if dim else 0.8,
                                            linewidth=1.2 if not big else (1.4 if dim else 2.4),
                                            solid_capstyle="round"))
            for how, (mk, _, _) in RING.items():
                mem = {m for p in f.get("plans", []) or [] if p["how"] == how for m in p["members"]}
                sc([pos[m] for m in mem if m in pos], s=(150 if mk == "s" else 170) * k, marker=mk,
                   facecolors="none", edgecolors=PLAN, linewidths=1.6)
            sc([P(r[1:4]) for r in fl if r[5] > 0 and r[4] != 5], s=95 * k, marker="o", facecolors="none",
               edgecolors=INK, linewidths=1)
            sc([P(r[1:4]) for r in fl if r[6]], s=125 * k, marker="o", facecolors="none", edgecolors=ORANGE,
               linewidths=1.5)
            for code, (_, col, _, mk, filled) in STATE_STYLE.items():
                pts = [P(r[1:4]) for r in fl if r[4] == code]
                if mk == "x":
                    sc(pts, s=40 * k, marker="x", c=col, linewidths=2)
                elif filled:
                    sc(pts, s=34 * k, marker=mk, c=col, linewidths=0)
                else:
                    sc(pts, s=34 * k, marker=mk, facecolors="none", edgecolors=col, linewidths=1.5)
            for fid in focus:
                if fid in pos and x0 <= pos[fid][0] <= x1 and y0 <= pos[fid][1] <= y1:
                    arts.append(ax.annotate(fid, pos[fid], xytext=(7, 7), textcoords="offset points",
                                            color=INK, fontsize=9.5))
            label = "Rung off: each fleet decides on its own" if side == 0 else \
                "Rung on: joint plans (exact search 2–4 fleets, thrml 5–8)"
            titles[side].set_text(f"{label}\n{pair.counts(side, f)}")
        sup.set_text(f"{pair.head()}  ·  step {t}" + (f"\n{textwrap.fill(note, 170)}" if note else ""))
        return arts

    anim = FuncAnimation(fig, draw, frames=len(ts), interval=1000 / fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)


# ----------------------------------------------------------------------------- HTML
def render_html(pair, out, ts, cdn=True, box=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    P = pair.P
    cells = pair.cells
    W = pair.lay.w * len(pair.lay.floors) + pair.lay.gap * (len(pair.lay.floors) - 1)
    H = max(c[1] for c in cells.values()) - min(c[1] for c in cells.values()) + 1
    if box:
        W, H = box[1] - box[0], box[3] - box[2]
    cell_px = 15 if not box else max(15, min(40, 800 / max(W, 1)))
    k = cell_px / 15                              # marker scale: a close-up draws bigger cells
    titles = ["Rung off: each fleet decides on its own", "Rung on: joint plans (exact 2–4, thrml 5–8)"]

    def dyn(side, f):
        ax = dict(xaxis="x" if side == 0 else "x2", yaxis="y" if side == 0 else "y2")
        show = side == 1
        tr = []
        o = f["orders"]
        for claimed, sym, name in ((0, "square-open", "order released"), (1, "square", "order assigned")):
            pts = [P(r[:3]) for r in o if r[3] == claimed]
            tr.append(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=name,
                                 marker=dict(symbol=sym, size=6 * k, color=YELLOW), hoverinfo="skip",
                                 showlegend=show, **ax))
        hm = [P(r[:3]) for r in f["humans"]]
        tr.append(go.Scatter(x=[p[0] for p in hm], y=[p[1] for p in hm], mode="markers", name="person",
                             marker=dict(symbol="star", size=13 * k, color=MAGENTA, line=dict(width=1, color=INK)),
                             hoverinfo="skip", showlegend=show, **ax))
        db = [P(r) for r in f["debris"]]
        tr.append(go.Scatter(x=[p[0] for p in db], y=[p[1] for p in db], mode="markers", name="debris",
                             marker=dict(symbol="triangle-up", size=10 * k, color=ORANGE), hoverinfo="skip",
                             showlegend=show, **ax))
        fl = f["fleets"]
        pos = {r[0]: P(r[1:4]) for r in fl}
        lx, ly = [], []
        for p in f.get("plans", []) or []:
            for fid, cl in p["path"].items():
                if fid in pos and cl:
                    xy = [pos[fid]] + [P(c) for c in cl]
                    lx += [q[0] for q in xy] + [None]
                    ly += [q[1] for q in xy] + [None]
        tr.append(go.Scatter(x=lx, y=ly, mode="lines", name="hops left in the plan", hoverinfo="skip",
                             line=dict(color=PLAN, width=1.5), opacity=0.75, showlegend=show, **ax))
        for how, (_, sym, name) in RING.items():
            mem = [m for p in f.get("plans", []) or [] if p["how"] == how for m in p["members"] if m in pos]
            tr.append(go.Scatter(x=[pos[m][0] for m in mem], y=[pos[m][1] for m in mem], mode="markers",
                                 name=name, hoverinfo="skip", showlegend=show,
                                 marker=dict(symbol=sym, size=20 * k, color=PLAN, line=dict(width=2)), **ax))
        wr = [r for r in fl if r[6]]
        tr.append(go.Scatter(x=[P(r[1:4])[0] for r in wr], y=[P(r[1:4])[1] for r in wr], mode="markers",
                             name="path warning", hoverinfo="skip", showlegend=show,
                             marker=dict(symbol="circle-open", size=16 * k, color=ORANGE, line=dict(width=2)), **ax))
        for code, (label, col, sym, _, _) in STATE_STYLE.items():
            rs = [r for r in fl if r[4] == code]
            tr.append(go.Scatter(
                x=[P(r[1:4])[0] for r in rs], y=[P(r[1:4])[1] for r in rs], mode="markers", name=label,
                marker=dict(symbol=sym, size=9 * k, color=col, line=dict(width=2 if sym != "circle" else 0, color=col)),
                text=[f"fleet {r[0]}<br>{label}" for r in rs], hoverinfo="text", showlegend=show, **ax))
        for t_ in tr:
            if not t_.x:
                t_.x, t_.y = [None], [None]
        return tr

    def anns(t):
        out = []
        for side in (0, 1):
            f = pair.frame(side, t)
            out.append(dict(text=f"<b>{titles[side]}</b><br>" + pair.counts(side, f).replace("\n", "<br>"),
                            x=0.0 if side == 0 else 0.515, y=1.0, xref="paper", yref="paper", xanchor="left",
                            yanchor="bottom", showarrow=False, align="left", font=dict(color=INK, size=12)))
        return out

    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.03)
    blk = pair.blk
    docks = [P(cells[d]) for d in pair.off["static"]["docks"] if d in cells]
    static = []
    for side in (0, 1):
        ax = dict(xaxis="x" if side == 0 else "x2", yaxis="y" if side == 0 else "y2")
        static += [go.Scatter(x=[b[0] for b in blk], y=[b[1] for b in blk], mode="markers", name="racks",
                              hoverinfo="skip", showlegend=side == 1,
                              marker=dict(symbol="square", size=12 * k, color=SHELF), **ax),
                   go.Scatter(x=[d[0] for d in docks], y=[d[1] for d in docks], mode="markers", name="dock",
                              hoverinfo="skip", showlegend=side == 1,
                              marker=dict(symbol="square", size=7 * k, color=GREEN), **ax)]
    first = dyn(0, pair.frame(0, ts[0])) + dyn(1, pair.frame(1, ts[0]))
    for tr in static + first:
        fig.add_trace(tr)
    idx = list(range(len(static), len(static) + len(first)))
    fig.frames = [go.Frame(data=dyn(0, pair.frame(0, t)) + dyn(1, pair.frame(1, t)), traces=idx, name=str(t),
                           layout=go.Layout(annotations=anns(t),
                                            title=dict(text=f"{pair.head()}  ·  step {t}")))
                  for t in ts]
    for i in (1, 2):
        fig.update_xaxes(visible=False, scaleanchor=f"y{'' if i == 1 else 2}", scaleratio=1, row=1, col=i,
                         **({"range": [box[0], box[1]]} if box else {}))
        fig.update_yaxes(visible=False, row=1, col=i, **({"range": [box[2], box[3]]} if box else {}))
    fig.update_layout(
        title=dict(text=f"{pair.head()}  ·  step {ts[0]}", x=0.01, font=dict(size=14, color=INK)),
        annotations=anns(ts[0]), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=MUTED, size=12),
        width=min(2000, int(200 + 2 * cell_px * W)), height=int(260 + cell_px * H),
        legend=dict(orientation="h", y=-0.02, x=0, font=dict(color=MUTED)),
        margin=dict(l=10, r=10, t=120, b=90),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.1, xanchor="left", yanchor="top",
                          bgcolor="#2b2b29", font=dict(color=INK), showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=80, redraw=True), transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.16, len=0.84, y=-0.08, yanchor="top", currentvalue=dict(prefix="step "),
                      font=dict(color=MUTED), steps=[
            dict(method="animate", label=str(t),
                 args=[[str(t)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                      transition=dict(duration=0))]) for t in ts])])
    fig.write_html(out, include_plotlyjs="cdn" if cdn else True, auto_play=False)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("off")
    ap.add_argument("on")
    ap.add_argument("--close", choices=["fork", "thrml"], default=None, help="a close-up instead of the whole day")
    ap.add_argument("--close-at", type=int, default=None, help="close-up of the plan starting nearest this step")
    ap.add_argument("--plan", type=int, default=None, help="close-up of this plan (its id, from --list)")
    ap.add_argument("--list", action="store_true", help="list the plans that ran furthest, for --plan")
    ap.add_argument("--before", type=int, default=6, help="close-up: steps shown before the plan")
    ap.add_argument("--after", type=int, default=None,
                    help="close-up: steps shown after the plan starts (default: until 10 steps after it ends)")
    ap.add_argument("--from", dest="t_from", type=int, default=0, help="whole day: first step shown")
    ap.add_argument("--to", dest="t_to", type=int, default=0, help="whole day: last step shown (0 = end of the work)")
    ap.add_argument("--every", type=int, default=0, help="keep every Nth step (default 6 whole day, 1 close-up)")
    ap.add_argument("--fps", type=int, default=0, help="default 12 whole day, 5 close-up")
    ap.add_argument("--dpi", type=int, default=0, help="default 85 whole day, 100 close-up")
    ap.add_argument("--html", action="store_true", help="also an interactive page")
    ap.add_argument("--embed-plotly", action="store_true", help="HTML: embed plotly.js (works offline, larger)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    pair = Pair(a.off, a.on)
    d = pair.off
    stem = a.out or f"pair_{d['arm']}_{d['map']}_k{d['k']}_s{d['seed']}_{d['cond']}"
    print(f"[pair] {len(pair.ts)} common steps; the days part at step "
          f"{pair.fork_t if pair.fork_t is not None else '(never)'}")
    if a.list:
        ids = sorted({p["id"] for t in pair.ts for p in pair.by_t[1][t].get("plans", []) or []})
        lives = [(pid, pair.plan_life(pid)) for pid in ids]
        lives.sort(key=lambda x: (-x[1][3], x[0]))
        print(f"[pair] {len(lives)} plans; those that got furthest:")
        print("  plan  solver  fleets  steps        held  hops")
        for pid, (t0, t1, p, hops, H, remade) in lives[:30]:
            print(f"  {pid:4d}  {p['how']:6s}  {len(p['members']):6d}  {t0:4d}-{t1:<6d}  {t1 - t0 + 1:4d}  {hops}/{H}")
        return
    close = a.close or ("plan" if a.plan is not None else "at" if a.close_at is not None else None)
    if close is None:
        every = a.every or 6
        t_to = a.t_to or pair.last_t()
        ts = [t for t in pair.ts if a.t_from <= t <= t_to][::every]
        out = stem + ("_day.gif" if not (a.t_from or a.t_to) else f"_day{a.t_from}-{t_to}.gif")
        render_gif(pair, out, ts, note="Same day, same failures and people; the two days part at the rung's "
                                       f"first plan (step {pair.fork_t}).", fps=a.fps or 12, dpi=a.dpi or 85)
        print(f"[pair] {len(ts)} frames -> {out}")
        if a.html:
            render_html(pair, out.replace(".gif", ".html"), ts, cdn=not a.embed_plotly)
            print(f"[pair] -> {out.replace('.gif', '.html')}")
        return
    if close == "plan":
        life = pair.plan_life(a.plan)
        found = (life[0], life[2]) if life else None
    else:
        found = (pair.first_plan() if close == "fork" else
                 pair.first_plan("thrml") if close == "thrml" else pair.first_plan(near=a.close_at))
    if found is None:
        raise SystemExit("[pair] no such plan in the 'on' recording")
    t0, plan = found
    _, t1, _, hops, H, remade = pair.plan_life(plan["id"])
    box = pair.box(plan, t0)
    after = a.after if a.after is not None else (t1 - t0) + 10
    ts = [t for t in pair.ts if t0 - a.before <= t <= t0 + after][::a.every or 1]
    same = pair.fork_t is None or t0 <= pair.fork_t
    n = len(plan["members"])
    near = pair.same_place(plan, t0)
    who = ", ".join(plan["members"])
    life_s = (f"It ran all {H} hops in {t1 - t0 + 1} steps." if hops >= H else
              f"It got through {hops} of {H} hops in {t1 - t0 + 1} steps, then was "
              + ("remade as its cluster changed." if remade else "ended."))
    if same:
        where = "Up to here the two days are identical: both panels show the same jam."
    elif near >= max(2, (n + 1) // 2):
        where = (f"The days parted at step {pair.fork_t}, but {near} of these {n} fleets are within 3 cells of "
                 "the same spot without the rung (left): nearly the same jam.")
    else:
        where = (f"The days parted at step {pair.fork_t}: the left panel shows the same place and time "
                 "without the rung, not the same jam.")
    note = f"Plan {plan['id']} ({plan['how']}, {n} fleets: {who}) starts at step {t0}. {life_s} {where}"
    tag = {"at": f"at{t0}", "plan": f"plan{plan['id']}"}.get(close, close)
    out = f"{stem}_{tag}.gif"
    render_gif(pair, out, ts, box=box, focus=plan["members"], focus_plan=plan["id"], note=note,
               fps=a.fps or 5, dpi=a.dpi or 100)
    print(f"[pair] {note}\n[pair] {len(ts)} frames -> {out}")
    if a.html:
        render_html(pair, f"{stem}_{tag}.html", ts, cdn=not a.embed_plotly, box=box)
        print(f"[pair] -> {stem}_{tag}.html")


if __name__ == "__main__":
    main()