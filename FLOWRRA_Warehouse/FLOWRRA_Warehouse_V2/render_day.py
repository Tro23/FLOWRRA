"""
render_day.py -- animate a day recorded by record_day.py.

    python render_day.py day_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz
        -> ..._full.html   interactive, top-down: play, pause, scrub, hover
    python render_day.py DAY.json.gz --gif --gif-every 4
        -> ..._full.gif    for a README (matplotlib + pillow)

Needs: plotly for the HTML; matplotlib and pillow for the GIF. No torch, no
checkpoint: it only reads the recording, so it runs anywhere.

WHAT IS ON SCREEN (top-down; multi-level maps are laid out floor by floor)
    fleets     blue: driving · hollow blue: waiting · grey: parked
               aqua: heading to a dock · violet: rescuer on its way to a pickup
               red x: failed (VDA 5050 error), its load waiting for a rescuer
    rings      white ring: carrying orders (larger = 2, a rescued load on board)
               orange ring: in a path warning (paths closing; §3.5)
    orders     yellow square: released, not yet delivered (filled: assigned)
    pickups    violet diamond: a failed fleet's load, waiting for its rescuer
    people     magenta star: a person walking in, cleaning, or leaving
    debris     orange triangle: debris on the floor, a no-go cell until cleared
    docks      green squares: where fleets leave and re-enter the floor
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter

# Reference dark palette (dataviz skill, validated dark steps); identity is
# never colour alone -- every element type also has its own marker shape.
SURFACE, INK, MUTED = "#1a1a19", "#f0efec", "#c3c2b7"
CELL, EDGE = "#2b2b29", "#3a3a37"
SHELF = "#55554f"          # racks, workstations, anything a fleet cannot enter
BLUE, ORANGE, AQUA, YELLOW = "#3987e5", "#d95926", "#199e70", "#c98500"
MAGENTA, GREEN, VIOLET, RED = "#d55181", "#008300", "#9085e9", "#e66767"
STATE_STYLE = {   # code -> (label, colour, plotly symbol, mpl marker, filled)
    0: ("driving", BLUE, "circle", "o", True),
    1: ("waiting", BLUE, "circle-open", "o", False),
    2: ("parked", MUTED, "circle", "o", True),
    3: ("heading to a dock", AQUA, "circle", "o", True),
    4: ("rescuer", VIOLET, "circle", "o", True),
    5: ("failed (VDA 5050 error)", RED, "x", "x", True),
}


def load(path):
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as fh:
        return json.load(fh)


class Layout:
    """Top-down coordinates. Main floors (z levels holding at least 10% of the
    cells) sit side by side; shaft cells between floors go to the floor below."""

    def __init__(self, cells):
        zs = Counter(round(c[2], 2) for c in cells.values())
        top = max(zs.values())
        self.floors = sorted(z for z, n in zs.items() if n >= 0.1 * top) or sorted(zs)
        xs = [c[0] for c in cells.values()]
        self.x0, self.w = min(xs), max(xs) - min(xs) + 1
        self.gap = 3

    def floor_index(self, z):
        below = [i for i, f in enumerate(self.floors) if f <= z + 1e-6]
        return below[-1] if below else 0

    def blocked(self, cells):
        """Grid cells inside each floor's footprint that are not floor: the racks
        and workstations. They are drawn; the drivable floor is left dark."""
        open_by_floor = {}
        for c in cells.values():
            fi = self.floor_index(c[2])
            open_by_floor.setdefault(fi, set()).add((round(c[0]), round(c[1])))
        out = []
        for fi, cells_open in open_by_floor.items():
            xs = [c[0] for c in cells_open]
            ys = [c[1] for c in cells_open]
            for x in range(min(xs), max(xs) + 1):
                for y in range(min(ys), max(ys) + 1):
                    if (x, y) not in cells_open:
                        out.append((x - self.x0 + fi * (self.w + self.gap), y))
        return out

    def xy(self, x, y, z):
        return (x - self.x0 + self.floor_index(z) * (self.w + self.gap), y)


def title_of(day, f):
    book = day["static"].get("book_total") or 0
    extra = "  ·  past the scored day" if f["t"] > day["scored_steps"] else ""
    var = []
    if day.get("book") and day.get("scored_book") and day["book"] != day["scored_book"]:
        var.append(f"book of {day['book']} (scored: {day['scored_book']})")
    if (day.get("max_failures") or -1) >= 0:
        var.append(f"at most {day['max_failures']} failures")
    if var:
        extra += "  ·  variant of the scored day: " + ", ".join(var)
    elif day.get("check") == "differs":
        extra += "  ·  a re-run of a scored day"
    return (f"FLOWRRA · {day['arm']} · {day['map']} · {day['k']} fleets · seed {day['seed']} · "
            f"{day['cond']}<br><sup>step {f['t']}{extra}  ·  delivered {f['delivered']}/{book}  ·  "
            f"released {f['released']}  ·  failures {f['failures']}  ·  "
            f"people {len(f['humans'])}  ·  debris {len(f['debris'])}  ·  "
            f"collisions {f['collisions']}</sup>")


# ----------------------------------------------------------------------------- HTML
def render_html(day, out, every, cdn):
    import plotly.graph_objects as go

    lay = Layout(day["static"]["cells"])
    cells = day["static"]["cells"]
    P = lambda c: lay.xy(*c)                                        # noqa: E731
    blk = lay.blocked(cells)
    docks = [P(cells[d]) for d in day["static"]["docks"] if d in cells]
    frames_in = day["frames"][::max(1, every)]
    if frames_in[-1] is not day["frames"][-1]:
        frames_in.append(day["frames"][-1])

    def dyn(f):
        tr = []
        o = f["orders"]
        for claimed, sym, name in ((0, "square-open", "order released"), (1, "square", "order assigned")):
            pts = [P(r[:3]) for r in o if r[3] == claimed]
            tr.append(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=name,
                                 marker=dict(symbol=sym, size=7, color=YELLOW), hoverinfo="skip"))
        pk = [P(r) for r in f["pickups"]]
        tr.append(go.Scatter(x=[p[0] for p in pk], y=[p[1] for p in pk], mode="markers", name="pickup waiting",
                             marker=dict(symbol="diamond", size=11, color=VIOLET), hoverinfo="skip"))
        db = [P(r) for r in f["debris"]]
        tr.append(go.Scatter(x=[p[0] for p in db], y=[p[1] for p in db], mode="markers", name="debris",
                             marker=dict(symbol="triangle-up", size=11, color=ORANGE), hoverinfo="skip"))
        hm = [(P(r[:3]), r[3]) for r in f["humans"]]
        tr.append(go.Scatter(x=[p[0][0] for p in hm], y=[p[0][1] for p in hm], mode="markers", name="person",
                             marker=dict(symbol="star", size=15, color=MAGENTA, line=dict(width=1, color=INK)),
                             text=[f"person: {ph.replace('_', ' ')}" for _, ph in hm], hoverinfo="text"))
        fl = f["fleets"]
        ld = [r for r in fl if r[5] > 0 and r[4] != 5]
        tr.append(go.Scatter(x=[P(r[1:4])[0] for r in ld], y=[P(r[1:4])[1] for r in ld], mode="markers",
                             name="carrying orders", hoverinfo="skip",
                             marker=dict(symbol="circle-open", size=[15 + 6 * (r[5] - 1) for r in ld],
                                         color=INK, line=dict(width=1.5))))
        wr = [r for r in fl if r[6]]
        tr.append(go.Scatter(x=[P(r[1:4])[0] for r in wr], y=[P(r[1:4])[1] for r in wr], mode="markers",
                             name="path warning", hoverinfo="skip",
                             marker=dict(symbol="circle-open", size=21, color=ORANGE, line=dict(width=2))))
        for code, (label, col, sym, _, _) in STATE_STYLE.items():
            rs = [r for r in fl if r[4] == code]
            tr.append(go.Scatter(
                x=[P(r[1:4])[0] for r in rs], y=[P(r[1:4])[1] for r in rs], mode="markers", name=label,
                marker=dict(symbol=sym, size=10, color=col, line=dict(width=2 if sym != "circle" else 0, color=col)),
                text=[f"fleet {r[0]}<br>{label}<br>orders on board: {r[5]}"
                      + ("<br>in a path warning" if r[6] else "") for r in rs],
                hoverinfo="text"))
        for t in tr:                     # empty traces would drop out of the legend
            if not t.x:
                t.x, t.y = [None], [None]
        return tr

    static = [
        go.Scatter(x=[b[0] for b in blk], y=[b[1] for b in blk], mode="markers", name="racks",
                   hoverinfo="skip", marker=dict(symbol="square", size=17, color=SHELF)),
        go.Scatter(x=[d[0] for d in docks], y=[d[1] for d in docks], mode="markers", name="dock",
                   hoverinfo="skip", marker=dict(symbol="square", size=9, color=GREEN)),
    ]
    first = dyn(frames_in[0])
    idx = list(range(len(static), len(static) + len(first)))
    frames = [go.Frame(data=dyn(f), traces=idx, name=str(f["t"]),
                       layout=go.Layout(title=dict(text=title_of(day, f))))
              for f in frames_in]
    fig = go.Figure(data=static + first, frames=frames)
    W = lay.w * len(lay.floors) + lay.gap * (len(lay.floors) - 1)
    H = max(c[1] for c in cells.values()) - min(c[1] for c in cells.values()) + 1
    fig.update_layout(
        title=dict(text=title_of(day, frames_in[0]), x=0.01, font=dict(size=15, color=INK)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=MUTED, size=12),
        width=min(1600, int(140 + 22 * W)), height=int(170 + 22 * H),
        xaxis=dict(visible=False, scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=-0.02, x=0, font=dict(color=MUTED)),
        margin=dict(l=10, r=10, t=70, b=90),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.1, xanchor="left", yanchor="top",
                          bgcolor="#2b2b29", font=dict(color=INK), showactive=False, buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, dict(frame=dict(duration=60, redraw=True), transition=dict(duration=0),
                                  fromcurrent=True, mode="immediate")]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(x=0.16, len=0.84, y=-0.08, yanchor="top", currentvalue=dict(prefix="step "),
                      font=dict(color=MUTED), steps=[
            dict(method="animate", label=str(f["t"]),
                 args=[[str(f["t"])], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                           transition=dict(duration=0))]) for f in frames_in])])
    fig.write_html(out, include_plotlyjs="cdn" if cdn else True, auto_play=False)
    return len(frames_in)


# ----------------------------------------------------------------------------- GIF
def render_gif(day, out, every, fps, max_frames, t_from=0, t_to=0, dpi=90):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    lay = Layout(day["static"]["cells"])
    cells = day["static"]["cells"]
    P = lambda c: lay.xy(*c)                                        # noqa: E731
    if not t_to:
        # End shortly after the last delivery: once the book stops moving, the
        # floor empties and the rest of the day is a still picture.
        last = max((f["t"] for i, f in enumerate(day["frames"][1:], 1)
                    if f["delivered"] > day["frames"][i - 1]["delivered"]), default=0)
        if last:
            t_to = last + 40
    fr = [f for f in day["frames"] if f["t"] >= t_from and (not t_to or f["t"] <= t_to)]
    fr = fr[::max(1, every)]
    if max_frames and len(fr) > max_frames:
        step = len(fr) / max_frames
        fr = [fr[int(i * step)] for i in range(max_frames)]
    W = lay.w * len(lay.floors) + lay.gap * (len(lay.floors) - 1)
    H = max(c[1] for c in cells.values()) - min(c[1] for c in cells.values()) + 1
    fig, ax = plt.subplots(figsize=(max(6, W * 0.19), max(4.5, H * 0.19 + 1.4)), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_aspect("equal")
    ax.axis("off")
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    blk = lay.blocked(cells)
    ax.add_collection(PatchCollection([Rectangle((x - 0.46, y - 0.46), 0.92, 0.92) for x, y in blk],
                                      facecolor=SHELF, edgecolor="none"))
    allx = [P(c)[0] for c in cells.values()] + [b[0] for b in blk]
    ally = [P(c)[1] for c in cells.values()] + [b[1] for b in blk]
    ax.set_xlim(min(allx) - 1, max(allx) + 1)
    ax.set_ylim(min(ally) - 1, max(ally) + 1)
    dk = [P(cells[d]) for d in day["static"]["docks"] if d in cells]
    ax.scatter([p[0] for p in dk], [p[1] for p in dk], s=30, marker="s", c=GREEN, linewidths=0)
    from matplotlib.lines import Line2D
    def h(label, mk, col, filled=True, ms=6):
        return Line2D([], [], linestyle="none", marker=mk, markersize=ms, label=label,
                      markerfacecolor=col if filled else "none", markeredgecolor=col,
                      markeredgewidth=1.5 if not filled or mk == "x" else 0)
    handles = [h("driving", "o", BLUE), h("waiting", "o", BLUE, False), h("parked", "o", MUTED),
               h("to a dock", "o", AQUA), h("rescuer", "o", VIOLET), h("failed", "x", RED),
               h("carrying", "o", INK, False, 8), h("path warning", "o", ORANGE, False, 9),
               h("order", "s", YELLOW, False), h("pickup", "D", VIOLET), h("person", "*", MAGENTA, ms=9),
               h("debris", "^", ORANGE), h("dock", "s", GREEN), h("rack", "s", SHELF, ms=8)]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=7, frameon=False,
                    fontsize=7, labelcolor=MUTED, handletextpad=0.3, columnspacing=0.9)
    arts = []
    title = ax.set_title("", color=INK, fontsize=9, loc="left")

    def draw(i):
        for a in arts:
            a.remove()
        arts.clear()
        f = fr[i]
        def sc(pts, **kw):
            if pts:
                arts.append(ax.scatter([p[0] for p in pts], [p[1] for p in pts], **kw))
        sc([P(r[:3]) for r in f["orders"] if not r[3]], s=16, marker="s", facecolors="none", edgecolors=YELLOW)
        sc([P(r[:3]) for r in f["orders"] if r[3]], s=16, marker="s", c=YELLOW)
        sc([P(r) for r in f["pickups"]], s=40, marker="D", c=VIOLET)
        sc([P(r) for r in f["debris"]], s=45, marker="^", c=ORANGE, linewidths=0)
        sc([P(r[:3]) for r in f["humans"]], s=90, marker="*", c=MAGENTA, edgecolors=INK, linewidths=0.5)
        fl = f["fleets"]
        sc([P(r[1:4]) for r in fl if r[5] > 0 and r[4] != 5], s=95, marker="o", facecolors="none",
           edgecolors=INK, linewidths=1)
        sc([P(r[1:4]) for r in fl if r[6]], s=170, marker="o", facecolors="none", edgecolors=ORANGE,
           linewidths=1.5)
        for code, (_, col, _, mk, filled) in STATE_STYLE.items():
            pts = [P(r[1:4]) for r in fl if r[4] == code]
            if filled and mk != "x":
                sc(pts, s=34, marker=mk, c=col, linewidths=0)
            elif mk == "x":
                sc(pts, s=40, marker="x", c=col, linewidths=2)
            else:
                sc(pts, s=34, marker=mk, facecolors="none", edgecolors=col, linewidths=1.5)
        # Three short lines, so nothing runs off the right edge of a small GIF.
        head, sub = title_of(day, f).replace("</sup>", "").split("<br><sup>")
        parts = sub.split("  ·  ")
        notes = [x for x in parts if x.startswith(("past the scored", "variant of", "a re-run"))]
        counts = [x for x in parts if x not in notes]
        title.set_text("\n".join([head, "  ·  ".join(counts)] + (["  ·  ".join(notes)] if notes else [])))
        return arts

    anim = FuncAnimation(fig, draw, frames=len(fr), interval=1000 / fps, blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), savefig_kwargs={"facecolor": SURFACE})
    plt.close(fig)
    return len(fr)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("day")
    ap.add_argument("--every", type=int, default=2, help="HTML: keep every Nth recorded frame")
    ap.add_argument("--cdn", action="store_true", help="HTML: load plotly.js from the CDN (smaller file)")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--gif-every", type=int, default=4)
    ap.add_argument("--gif-fps", type=int, default=15)
    ap.add_argument("--gif-max-frames", type=int, default=400,
                    help="frames / fps = seconds on screen (600 at 15 fps = 40 s)")
    ap.add_argument("--gif-from", type=int, default=0, help="first step to show (a close-up window)")
    ap.add_argument("--gif-to", type=int, default=0, help="last step to show (0 = the end of the day)")
    ap.add_argument("--gif-dpi", type=int, default=90, help="lower (e.g. 72) for a smaller file")
    ap.add_argument("--gif-out", default="", help="file name (default: the recording's name + .gif)")
    ap.add_argument("--no-html", action="store_true")
    a = ap.parse_args()
    day = load(a.day)
    base = a.day[:-8] if a.day.endswith(".json.gz") else os.path.splitext(a.day)[0]
    s = day.get("summary", {})
    print(f"[render] {day['arm']} {day['map']} k={day['k']} seed={day['seed']} {day['cond']}: "
          f"{len(day['frames'])} frames; delivered {s.get('stream_book_delivered')}/{s.get('stream_book_orders')}")
    if not a.no_html:
        n = render_html(day, base + ".html", a.every, a.cdn)
        print(f"[render] {n} frames -> {base}.html")
    if a.gif:
        gout = a.gif_out or base + ".gif"
        n = render_gif(day, gout, a.gif_every, a.gif_fps, a.gif_max_frames, a.gif_from, a.gif_to, a.gif_dpi)
        print(f"[render] {n} frames, {n / max(1, a.gif_fps):.0f} s -> {gout} "
              f"({os.path.getsize(gout) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()