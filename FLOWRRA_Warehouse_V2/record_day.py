"""
record_day.py -- record one evaluated day of the final deploy test, step by step,
for render_day.py to animate.

WHY THIS WAY. The day is run by deploy_eval itself, with exactly the flags
final_jobs.py used (same world, keyed failures and people, book, assignment,
world fixes). Nothing in deploy_eval or core_warehouse is edited: this script
wraps FLOWRRA.step for the length of the run and takes a snapshot after each
step. So the animation shows the day that was scored, and --check proves it by
comparing this run's CSV row with the one in final_full.csv / final_clean.csv.

USE (in FLOWRRA_Warehouse_V2, where the checkpoints and all_scens_final live)
    python record_day.py --arm ladder4 --map Kiva_2D_46_33_1 --fleets 60 --seed 3 \
        --cond full --check final_full.csv
    -> day_ladder4_Kiva_2D_46_33_1_k60_s3_full.json.gz   (then: render_day.py)

    --extra-steps N   let the day run N steps past the evaluated length, so the
                      book can finish on screen. The env is still told the
                      scored length (throughput pressure T and the failure
                      window are computed from it), so the first steps are the
                      scored day's; --check is skipped.
    --stride N        snapshot every Nth step (default 1; 2 halves the file).
    --max-failures N  at most N hardware failures in the day (scored days had
                      about 11-12). Also a variant, labelled as such.
    --book N          a smaller (or larger) order book, e.g. 150, for a shorter
                      animation. That is a variant of the scored day, so --check
                      is skipped and the rendered title says so.

Arms: ladder4, ladder6, ladder7, RULES, PIBT, RHCR-PP, RHCR-PIBT, RHCR-PIBT+naive.
"""
from __future__ import annotations

import os
# One thread, as final_jobs.py ran every job: multi-threaded maths can break a
# near-tie between two moves differently, and one different move changes the day.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gzip
import json
import sys
import tempfile

from final_jobs import ARMS, CONDS, GROUPS, SCENS

# Fleet states, in order of precedence (render_day.py reads these codes).
TRANSIT, WAITING, PARKED, LEAVING, RESCUER, DEAD = 0, 1, 2, 3, 4, 5
STATE_NAMES = ["in transit", "waiting", "parked", "heading to a dock", "rescuer", "failed (VDA 5050 error)"]


def _r(v):
    return round(float(v), 2)


class Recorder:
    def __init__(self, stride: int):
        self.stride = max(1, int(stride))
        self.env = None
        self.frames = []
        self.static = None

    # -- the floor, once ----------------------------------------------------
    def _static(self, env):
        cell_of = {}
        for cell, nid in getattr(env, "grid_pos_dict", {}).items():
            cell_of[str(nid)] = [_r(c) for c in cell]
        edges = []
        G = getattr(env, "G", None) or getattr(env, "graph", None)
        if G is not None:
            for u, v in G.edges():
                if str(u) in cell_of and str(v) in cell_of:
                    edges.append([str(u), str(v)])
        docks = [str(d) for d in sorted(getattr(env, "_exit_set", set()) or set(), key=str)]
        self.cell_of = cell_of
        return {"cells": cell_of, "edges": edges, "docks": docks,
                "book_total": int(getattr(env, "_book_total", 0) or 0)}

    def _xyz(self, nid):
        return self.cell_of.get(str(nid))

    # -- one snapshot -------------------------------------------------------
    def capture(self, env):
        stopped = getattr(env, "stopped_nodes", set()) or set()
        frozen = getattr(env, "frozen_nodes", set()) or set()
        immobile = getattr(env, "immobile_nodes", set()) or set()
        waiting = getattr(env, "waiting_nodes", set()) or set()
        leaving = getattr(env, "_despawning", set()) or set()
        rescuers = getattr(env, "_pickup_assignment", {}) or {}
        warned = getattr(env, "_step_warning", set()) or set()
        pending = getattr(env, "_pending_goals", {}) or {}
        fleets = []
        for n in env.nodes:
            if n.id in stopped:
                st = DEAD
            elif n.id in rescuers:
                st = RESCUER
            elif n.id in leaving:
                st = LEAVING
            elif n.id in frozen or n.id in immobile:
                st = PARKED
            elif n.id in waiting:
                st = WAITING
            else:
                st = TRANSIT
            load = 0 if st == DEAD else ((1 if getattr(n, "current_goal_id", None) else 0)
                                         + len(pending.get(n.id, []) or []))
            p = n.current_pos
            fleets.append([str(n.id), _r(p[0]), _r(p[1]), _r(p[2]), st, int(load),
                           int(n.id in warned)])
        humans, debris = [], []
        of = getattr(env, "obstacle_field", None)
        if of is not None:
            coords = getattr(of, "coords_by_id", {}) or {}
            for h in (getattr(of, "c_humans", None) or []):
                c = coords.get(h.get("cell"))
                if c is not None:
                    humans.append([_r(c[0]), _r(c[1]), _r(c[2]), str(h.get("phase", ""))])
            for nid in (getattr(of, "humans", None) or []):
                c = coords.get(nid)
                if c is not None:
                    humans.append([_r(c[0]), _r(c[1]), _r(c[2]), "wander"])
            for nid in list(getattr(of, "c_debris", {}) or {}) + list(getattr(of, "debris", None) or []):
                c = coords.get(nid)
                if c is not None:
                    debris.append([_r(c[0]), _r(c[1]), _r(c[2])])
        claimed = getattr(env, "claimed_goals", set()) or set()
        orders = []
        for g, pos in (getattr(env, "goal_pool", {}) or {}).items():
            orders.append([_r(pos[0]), _r(pos[1]), _r(pos[2]), int(g in claimed)])
        pickups = [[_r(p["pos"][0]), _r(p["pos"][1]), _r(p["pos"][2])]
                   for p in (getattr(env, "open_pickups", {}) or {}).values() if "pos" in p]
        ss = getattr(env, "_sstats", {}) or {}
        self.frames.append({
            "t": int(getattr(env, "step_count", len(self.frames))),
            "fleets": fleets, "humans": humans, "debris": debris,
            "orders": orders, "pickups": pickups,
            "delivered": int(ss.get("deliveries", 0)),
            "released": int(getattr(env, "_book_released", 0) or 0),
            "off_floor": len(getattr(env, "_offfloor", []) or []),
            "failures": int(getattr(env, "total_errors", 0) or 0),
            "collisions": int(getattr(getattr(env, "loop", None), "total_collisions", 0) or 0),
        })

    def before(self, env):
        if self.env is None:
            self.env = env
            self.static = self._static(env)
            self.capture(env)                      # t = 0

    def after(self, env):
        t = int(getattr(env, "step_count", 0))
        if t % self.stride == 0 or env.is_episode_over():
            self.capture(env)


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--map", default="Kiva_2D_46_33_1")
    ap.add_argument("--fleets", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--cond", choices=sorted(CONDS), default="full")
    ap.add_argument("--scens-dir", default=SCENS)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--extra-steps", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0, help="override the day's length (testing only)")
    ap.add_argument("--max-failures", type=int, default=-1,
                    help="at most this many hardware failures in the day (default: as scored, about 11-12). "
                         "Waves stop once one could overshoot the cap; single failures fill up to it. "
                         "A variant of the scored day, like --book")
    ap.add_argument("--book", type=int, default=0,
                    help="orders in the day's book (default: the scored size, 100 or 200). A different size is a "
                         "variant of the scored day, not the day itself: --check is skipped and the title says so")
    ap.add_argument("--check", default="", help="final_full.csv or final_clean.csv: confirm this is the scored day")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    group = next((g for g, v in GROUPS.items() if a.map in v["maps"].split(",")), None)
    if group is None:
        sys.exit(f"{a.map} is not one of the final run's maps")
    g = GROUPS[group]
    fail, people = CONDS[a.cond]
    steps = a.steps or (g["steps"][a.cond] + max(0, a.extra_steps))
    book = a.book or g["book"]
    variant = book != g["book"] or a.max_failures >= 0
    row_csv = os.path.join(tempfile.mkdtemp(prefix="record_day_"), "row.csv")
    argv = ["deploy_eval.py", "--arms", ARMS[a.arm], "--maps", a.map, "--scens-dir", a.scens_dir,
            "--agents", str(a.fleets), "--seeds", str(a.seed),
            "--failures", fail, "--people", people,
            "--book", str(book), "--release", str(g["release"]), "--max-steps", str(steps),
            "--assignment", "hungarian", "--world-fixes", "on", "--orphans", "pickup",
            "--keyed-draws", "on", "--out", row_csv]
    print("[record] deploy_eval", " ".join(argv[1:]), flush=True)

    import core_warehouse
    rec = Recorder(a.stride)
    _orig = core_warehouse.FLOWRRA.step

    def _step(self, *args, **kw):
        if a.max_failures >= 0:
            # Waves fire 2-4 at once: stop them while one could overshoot the cap,
            # and stop everything once the cap is reached.
            done = int(getattr(self, "total_errors", 0) or 0)
            if done + int(getattr(self, "wave_size_max", 4) or 4) > a.max_failures:
                self.waves_enabled = False
            if done >= a.max_failures:
                self.errors_enabled = False
        rec.before(self)
        out = _orig(self, *args, **kw)
        rec.after(self)
        return out

    core_warehouse.FLOWRRA.step = _step
    # The env reads the day's length from the step budget for throughput
    # pressure T = 1 / (1 + orders left / steps left) and for the window in
    # which failures can happen. A longer run must not change either, or the
    # whole day changes from step 1: pin both to the scored length.
    _orig_init = core_warehouse.FLOWRRA.__init__
    scored_len = int(g["steps"][a.cond])

    def _init(self, *args, **kw):
        _orig_init(self, *args, **kw)
        self.max_steps_per_episode = scored_len
        self._max_steps_hint = scored_len

    core_warehouse.FLOWRRA.__init__ = _init
    import deploy_eval
    sys.argv = argv
    try:
        deploy_eval.main()
    except SystemExit:
        pass
    except Exception as e:                         # the closing summary only (scipy etc.)
        print(f"[record] note: deploy_eval ended with {type(e).__name__}: {e}")
    finally:
        core_warehouse.FLOWRRA.step = _orig
        core_warehouse.FLOWRRA.__init__ = _orig_init
    if rec.env is None or not rec.frames:
        sys.exit("[record] nothing was recorded")

    import pandas as pd
    row = pd.read_csv(row_csv).iloc[0].to_dict() if os.path.exists(row_csv) else {}
    keep = ["stream_book_delivered", "stream_book_orders", "stream_book_done_step", "collisions",
            "errors_injected", "handovers_completed", "obstacle_near_miss_steps",
            "obstacle_debris_spawned", "obstacle_debris_cleaned", "ladder_l3_fired", "steps_run"]
    summary = {k: (None if pd.isna(row.get(k)) else float(row.get(k))) for k in keep if k in row}

    check = "not checked"
    if a.check and variant:
        print(f"[record] --check skipped: a variant of the scored day (book {book}"
              + (f", at most {a.max_failures} failures" if a.max_failures >= 0 else "") + ")")
    if a.check and a.extra_steps == 0 and not a.steps and not variant:
        ref = pd.read_csv(a.check, low_memory=False)
        ref = ref[(ref["arm"] == a.arm) & (ref["map"] == a.map) & (ref["k"] == a.fleets) & (ref["seed"] == a.seed)]
        if len(ref) != 1:
            print(f"[record] --check: {len(ref)} matching rows in {a.check}; not compared")
        else:
            r = ref.iloc[0]
            diffs = [k for k in ("stream_book_delivered", "collisions", "errors_injected",
                                 "handovers_completed", "steps_run")
                     if k in r and k in row and float(r[k]) != float(row[k])]
            check = "identical" if not diffs else "differs"
            print("[record] --check: " + ("IDENTICAL to the scored day" if not diffs
                                         else f"DIFFERS from the scored day in {diffs} "
                                              f"(delivered {row.get('stream_book_delivered')} here, "
                                              f"{r.get('stream_book_delivered')} scored)"))

    out = a.out or (f"day_{a.arm}_{a.map}_k{a.fleets}_s{a.seed}_{a.cond}"
                    + (f"_book{book}" if book != g["book"] else "")
                    + (f"_fail{a.max_failures}" if a.max_failures >= 0 else "") + ".json.gz").replace("+", "_")
    payload = {"arm": a.arm, "map": a.map, "k": a.fleets, "seed": a.seed, "cond": a.cond,
               "scored_steps": g["steps"][a.cond], "extra_steps": a.extra_steps,
               "book": book, "scored_book": g["book"], "check": check,
               "max_failures": a.max_failures,
               "stride": a.stride, "state_names": STATE_NAMES, "summary": summary,
               "static": rec.static, "frames": rec.frames}
    with gzip.open(out, "wt") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    print(f"[record] {len(rec.frames)} frames -> {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"[record] summary: {summary}")


if __name__ == "__main__":
    main()