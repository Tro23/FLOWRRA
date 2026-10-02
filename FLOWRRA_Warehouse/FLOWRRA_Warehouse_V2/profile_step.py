"""
profile_step.py -- where the time actually goes in one FLOWRRA step.

THE HYPOTHESIS THIS EXISTS TO KILL OR CONFIRM
---------------------------------------------
sense_6_axis_rays() sizes its trace loop as

    int(self.max_vision_range * (5.0 / max(self.speed, 1e-3)))

which is 100 iterations per ray at base_speed 0.5 and 1000 under braking, where
speed floors at 0.05. Each iteration calls is_structurally_valid(), whose fast
path does list(aisle_index[axis][...]) then .sort() then a linear scan of the
whole aisle line -- O(A log A) in aisle length, not O(1).

So per-ray compute is inversely proportional to speed, which means it SPIKES
UNDER CONGESTION: exactly the regime the scaling work is trying to reach. And
state vectors are built for EVERY node twice per step (core_warehouse.py ~1386
and ~1878), including parked and stopped fleets, whose speed is never restored
to base because the immobile branch `continue`s before `node.speed = base_speed`.

Back-of-envelope at 200 fleets: 200 x 2 x 6 x ~100 iterations x a sort over a
few-hundred-node aisle lands in the several-seconds range. The measured figure
was 7.4 s/step. That is consistent, which is not the same as true.

This script settles it. Run it BEFORE anything else, because if the answer is
yes then a one-line change to the loop bound is worth more than any amount of
architecture, and if the answer is no then that belief has been steering
decisions for no reason.

WHAT TO LOOK FOR
----------------
In the cumulative table:
  * is_structurally_valid high in ncalls AND tottime -> hypothesis confirmed
  * sense_6_axis_rays dominating cumtime                -> confirmed
  * get_local_affordance / _structure_mask on top       -> the field is the cost,
                                                           not the rays
  * choose_actions / torch internals on top             -> it is the network, and
                                                           none of this matters

The per-speed histogram at the end is the decisive one: it reports the ray
iteration budget actually used, bucketed by fleet speed. If the braked buckets
carry most of the total, the 1/speed bound is the problem by construction.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import random
import time
from collections import Counter

import numpy as np

from config_warehouse import CONFIG


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--map", default="", help="map name; default = the largest discovered")
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=40,
                    help="steps to run before profiling, so the profile covers a "
                         "populated field and some braking rather than a cold start")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sort", default="tottime", choices=["tottime", "cumtime", "ncalls"])
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("--out", default="profile_results",
                    help="directory for the CSV log; nothing is overwritten, each "
                         "run gets its own timestamped file")
    ap.add_argument("--tag", default="",
                    help="short label written into every row, e.g. 'after-roundtrip-fix'. "
                         "Makes runs comparable after the terminal has scrolled away.")
    ap.add_argument("--block", type=int, default=10,
                    help="report unprofiled ms/step every N warmup steps, to show "
                         "whether step cost is flat or climbing with episode phase")
    ap.add_argument("--resume", default="",
                    help="checkpoint to profile with. Optional: a randomly "
                         "initialised network profiles the same code paths, but "
                         "picks different actions, so it lands the fleet in a "
                         "different congestion regime. Since the whole question "
                         "is how compute behaves UNDER CONGESTION, prefer a real "
                         "checkpoint when you have one.")
    args = ap.parse_args()

    import torch
    import csv
    import datetime as _dt
    import platform
    import subprocess
    from main_runner_warehouse import MapCache, sample_instance
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent

    random.seed(args.seed)
    np.random.seed(args.seed)

    cache = MapCache(args.maps_dir, args.scens_dir)
    names = cache.discover([args.map] if args.map else None)
    if not names:
        raise SystemExit(f"no usable maps in {args.maps_dir} / {args.scens_dir}")

    rng = random.Random(args.seed)
    name = args.map or names[-1]
    inst = sample_instance(cache, name, args.agents, rng)
    if inst is None:
        raise SystemExit(f"could not build an instance for {name} at k={args.agents}")

    env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                  goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                  goal_pool=inst["goal_pool"])

    # BUG FIXED 2026-09-13: FLOWRRA does not construct its own agent -- the
    # runner and the lesion harness both assign env.gnn afterwards, and this
    # script did not, so step() hit `'NoneType' object has no attribute
    # choose_actions` on the first warmup step.
    n0 = env.nodes[0]
    input_dim = (len(n0.get_state_vector(env.nodes))
                 + len(env.density.get_local_affordance(
                       n0.current_pos, env.nodes, env.frozen_nodes)))
    rd = CONFIG["reward_decomposition"]
    agent = GNNAgent(
        node_feature_dim=input_dim, edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        reward_heads=rd["heads"], head_weights=rd["weights"],
        dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
        gamma=CONFIG["training"]["gamma"],
        buffer_capacity=CONFIG["training"]["buffer_capacity"],
        batch_size=CONFIG["training"]["batch_size"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
    )
    if args.resume:
        agent.load(args.resume)
        print(f"loaded {args.resume}")
    else:
        print("NO CHECKPOINT -- profiling with a randomly initialised network. "
              "Same code paths, different congestion. Pass --resume for the real thing.")
    # Greedy, so two runs of this script are comparable and the profile is not
    # partly measuring the exploration schedule.
    agent.epsilon_gaussian = lambda *a, **k: 0.0
    env.gnn = agent

    print(f"\nmap={name}  nodes={len(inst['pos_dict'])}  "
          f"fleets={len(env.nodes)} (requested {args.agents})")
    if len(env.nodes) < args.agents:
        print(f"  NOTE: the scenario capped the fleet count at {len(env.nodes)}. "
              f"The ray-budget question is about SCALE, so profile on the largest "
              f"map and highest fleet count you can actually instantiate -- a run "
              f"at {len(env.nodes)} fleets on {len(inst['pos_dict'])} nodes is a "
              f"much lighter regime than the 200-fleet / 120k-node case where "
              f"7.4 s/step was measured.")
    print(f"input_dim={input_dim}")
    print(env.density.describe())
    print("(proximity index is empty until the first refresh, inside step())")

    # Defined BEFORE the CSV metadata below reads it. It used to be set just
    # above the warmup loop, and inserting the CSV block in between left the
    # metadata dict referencing an unbound local -- a crash that only fires
    # AFTER map discovery and the 120k-node BFS precompute, so it costs ten
    # minutes to discover.
    block = max(1, args.block)

    # ---- CSV log -------------------------------------------------------
    # The terminal keeps only the newest run, so anything not written to disk
    # is gone. One file per run, timestamped, never overwritten, plus a single
    # appended index so runs can be compared later without hunting.
    os.makedirs(args.out, exist_ok=True)
    # Milliseconds, not seconds. Two runs started inside the same second would
    # otherwise share a run_id and become indistinguishable in the appended
    # index -- caught by test_profile_step, which runs twice in a row.
    stamp_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    tag = args.tag or "untagged"
    csv_path = os.path.join(args.out, f"warmup_{tag}_{stamp_id}.csv")

    try:
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        git_rev = ""

    meta = {
        "run_id": stamp_id, "tag": tag, "git": git_rev,
        "map": name, "map_nodes": len(inst["pos_dict"]),
        "fleets": len(env.nodes), "requested_fleets": args.agents,
        "warmup": args.warmup, "block": block, "seed": args.seed,
        "checkpoint": os.path.basename(args.resume) if args.resume else "random-init",
        "kernel_metric": CONFIG["density"].get("kernel_metric", "?"),
        "proximity_metric": CONFIG.get("proximity", {}).get("metric", "?"),
        "project_stationary": CONFIG["density"].get("project_stationary", "?"),
        "idle_mode": CONFIG.get("perception", {}).get("idle_mode", "?"),
        "idle_refresh_every": CONFIG.get("perception", {}).get("idle_refresh_every", "?"),
        "python": platform.python_version(),
    }

    fields = ["run_id", "tag", "git", "block_start", "block_end", "ms_per_step",
              "active", "frozen", "immobile", "collisions", "stamps_per_obs",
              "stamp_early_outs", "mean_speed", "min_speed", "braked_samples", "mem_cells",
              "ray_origin_recovered", "ray_origin_blind",
              "idle_reused", "idle_computed",
              "proj_fallbacks", "kernel_fallbacks", "phantom_rejected",
              "elapsed_s"]

    csv_fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(csv_fh, fieldnames=fields)

    # ---- runs_index.csv: EVERY BLOCK from EVERY RUN, appended ------------
    # One file to open. run_id and tag distinguish the runs, so two runs can be
    # compared block-for-block without hunting through per-run files.
    #
    # If an existing index was written with a different set of columns -- which
    # happens whenever a driver is added -- it is rotated aside rather than
    # appended to. Appending mismatched rows to a CSV produces a file that still
    # opens and is silently wrong, which is worse than two files.
    idx_path = os.path.join(args.out, "runs_index.csv")
    idx_new = True
    if os.path.exists(idx_path):
        with open(idx_path, newline="") as _fh:
            _first = _fh.readline().strip()
        if _first == ",".join(fields):
            idx_new = False
        else:
            _rot = idx_path.replace(".csv", f"_schema_changed_{stamp_id}.csv")
            os.rename(idx_path, _rot)
            print(f"index schema changed; previous index moved to {os.path.basename(_rot)}")
    idx_fh = open(idx_path, "a", newline="")
    idx_writer = csv.DictWriter(idx_fh, fieldnames=fields)
    if idx_new:
        idx_writer.writeheader()
    # Metadata as leading comment lines: readable by eye, and pandas skips them
    # with comment="#".
    for k, v in meta.items():
        csv_fh.write(f"# {k},{v}\n")
    writer.writeheader()
    print(f"\nlogging to {csv_path}")

    # ---- TAG vs CONFIG sanity check -------------------------------------
    # runs_index.csv carries only the tag; the per-run file carries the real
    # config in its metadata header. So a tag that contradicts the config
    # produces an index row that is silently wrong, and the only way back is to
    # open the right file and read the header.
    #
    # This happened on 2026-09-14: a run tagged "idle-cached" executed with
    # idle_mode=full. Cheap to catch here, ten minutes wasted if caught after.
    _modes = {
        "idle_mode": (CONFIG.get("perception", {}).get("idle_mode", "full"),
                      ("full", "cached", "skip")),
        "kernel_metric": (CONFIG["density"].get("kernel_metric", "graph"),
                          ("graph", "manhattan")),
        "proximity_metric": (CONFIG.get("proximity", {}).get("metric", "graph"),
                             ("graph", "manhattan")),
    }
    _tag_lower = tag.lower()
    _warn = []
    for _key, (_actual, _valid) in _modes.items():
        for _v in _valid:
            if _v in _tag_lower and _v != _actual:
                _warn.append(f"tag mentions {_v!r} but {_key}={_actual!r}")
    print("\nactive experimental config:")
    for _key, (_actual, _) in _modes.items():
        print(f"   {_key:<18} {_actual}")
    if _warn:
        print("\n  *** TAG DOES NOT MATCH CONFIG ***")
        for _w in _warn:
            print(f"      {_w}")
        print("      The index row will be labelled wrongly. Ctrl-C now if that")
        print("      matters -- the per-run CSV header keeps the truth either way.")
        time.sleep(5)

    print(f"\nwarmup {args.warmup} steps (UNPROFILED -- these are the honest numbers)...")
    t0 = time.time()
    tb = t0
    done = 0
    curve = []
    for i in range(args.warmup):
        env.step(episode_step=1, total_episodes=1)
        done += 1
        if done % block == 0:
            now = time.time()
            ms = (now - tb) / block * 1000
            curve.append((done, ms))
            active = len(env.get_active_nodes())
            # Read the BRAKED speed accumulated inside step()'s action loop.
            # Reading node.speed here would report base_speed for every fleet:
            # core_warehouse restores it at the end of that loop, so the braked
            # value no longer exists by the time step() returns. That is exactly
            # what the 2026-09-13 run measured -- 0.500 in every block, for 200
            # steps, while collisions were happening.
            _n = getattr(env, "_braked_speed_n", 0)
            mean_speed = (env._braked_speed_sum / _n) if _n else float("nan")
            min_speed = (env._braked_speed_min) if _n else float("nan")
            speeds = [min_speed]
            # collapse_memory is a DICT of cell -> severity, not an array.
            mem_cells = len(getattr(env.density, "collapse_memory", ()))
            stamps = getattr(env.density, "stamps_this_step", 0)
            print(f"  steps {done-block+1:>4}-{done:<4}  {ms:8.0f} ms/step   "
                  f"active={active:<4} frozen={len(env.frozen_nodes):<4} "
                  f"coll={env.loop.total_collisions:<3} "
                  f"stamps/obs={stamps // max(1, len(env.nodes)):<5} "
                  f"speed={mean_speed:.3f}/{min_speed:.3f} memcells={mem_cells}")
            _row = {
                "run_id": stamp_id, "tag": tag, "git": git_rev,
                "block_start": done - block + 1, "block_end": done,
                "ms_per_step": round(ms, 1),
                "active": active, "frozen": len(env.frozen_nodes),
                "immobile": len(getattr(env, "immobile_nodes", ())),
                "collisions": env.loop.total_collisions,
                "stamps_per_obs": stamps // max(1, len(env.nodes)),
                "stamp_early_outs": getattr(env.density, "stamp_early_outs", 0),
                "mean_speed": round(mean_speed, 4),
                "min_speed": round(min_speed, 4) if _n else "",
                "braked_samples": _n,
                "mem_cells": mem_cells,
                "ray_origin_recovered": sum(
                    getattr(n, "ray_origin_recovered", 0) for n in env.nodes),
                "ray_origin_blind": sum(
                    getattr(n, "ray_origin_blind", 0) for n in env.nodes),
                "idle_reused": getattr(env, "_idle_perception_reused", 0),
                "idle_computed": getattr(env, "_idle_perception_computed", 0),
                "proj_fallbacks": getattr(env.density, "_projection_fallbacks", 0),
                "kernel_fallbacks": getattr(env.density, "_kernel_manhattan_fallbacks", 0),
                "phantom_rejected": getattr(env.loop, "phantom_pairs_rejected", 0),
                "elapsed_s": round(now - t0, 2),
            }
            writer.writerow(_row)
            idx_writer.writerow(_row)
            csv_fh.flush()   # flush per block: a run killed at step 180 still
                             # leaves 170 steps of usable data on disk.
            idx_fh.flush()
            env._braked_speed_sum = 0.0
            env._braked_speed_n = 0
            env._braked_speed_min = float("inf")
            env.density.stamps_this_step = 0
            env.density.stamp_early_outs = 0
            tb = now
        if env.is_episode_over():
            print(f"  episode ended early at warmup step {i}")
            break
    warm = time.time() - t0
    print(f"warmup: {warm:.2f}s total, {warm/max(1,done)*1000:.0f} ms/step mean")

    csv_fh.close()
    idx_fh.close()

    # One line per RUN, separate from the per-block index above.
    sum_path = os.path.join(args.out, "runs_summary.csv")
    new_idx = not os.path.exists(sum_path)
    with open(sum_path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new_idx:
            w.writerow(["run_id", "tag", "git", "map", "fleets", "warmup",
                        "mean_ms_per_step", "first_block_ms", "last_block_ms",
                        "trend_pct", "csv"])
        w.writerow([
            stamp_id, tag, git_rev, name, len(env.nodes), done,
            round(warm / max(1, done) * 1000, 1),
            round(curve[0][1], 1) if curve else "",
            round(curve[-1][1], 1) if curve else "",
            round((curve[-1][1] - curve[0][1]) / curve[0][1] * 100, 1)
            if len(curve) >= 2 and curve[0][1] > 0 else "",
            os.path.basename(csv_path),
        ])
    print(f"appended {len(curve)} block rows to {idx_path}")
    print(f"appended one summary row to {sum_path}")

    # WHY THE CURVE MATTERS. A single window early in an episode measures the
    # cheapest phase. Collapse memory accumulates and decays at x0.7/step but is
    # re-stamped, fleets park and stay parked, and Tier 2/3 events only start
    # firing at density -- so step 200 is not step 30 with a different seed, it
    # is a different workload. Comparing a 5-step window at step 35 against an
    # 80-step window at step 120 is not a before/after, and reading one point as
    # if it were the whole episode is how a 30% "speedup" turns out to be phase.
    if len(curve) >= 2:
        first, last = curve[0][1], curve[-1][1]
        trend = (last - first) / first * 100 if first > 0 else 0.0
        print(f"\n  cost trend across warmup: {first:.0f} -> {last:.0f} ms/step "
              f"({trend:+.0f}%)")
        if abs(trend) > 25:
            print("  NOT FLAT. Step cost depends on episode phase, so any "
                  "before/after must hold warmup, steps and seed fixed.")
        print("  Read the DRIVERS, not just ms/step. A flat curve with falling")
        print("  stamps/obs means something else grew to cancel it -- watch")
        print("  mean speed (the ray budget is 1/speed, so braking multiplies")
        print("  ray iterations) and memcells (collapse memory accumulating).")

    # Speed histogram BEFORE profiling: this is what sets the ray budget.
    speeds = Counter()
    budget_total = 0
    for n in env.nodes:
        sp = float(getattr(n, "speed", 0.5))
        bucket = round(sp, 2)
        speeds[bucket] += 1
        budget_total += int(n.max_vision_range * (5.0 / max(sp, 1e-3))) * 6
    print(f"\nray iteration budget across all fleets, one state build: {budget_total:,}")
    print("  (x2 per step -- current state and next state are built separately)")
    print("  speed histogram:")
    for sp, cnt in sorted(speeds.items()):
        per = int(env.nodes[0].max_vision_range * (5.0 / max(sp, 1e-3))) * 6
        print(f"    speed {sp:<5} x{cnt:<4} fleets  ->  {per:>6,} ray iters each  "
              f"= {per*cnt:>9,}  ({100*per*cnt/max(1,budget_total):4.1f}% of total)")

    print(f"\nprofiling {args.steps} steps...")
    pr = cProfile.Profile()
    t0 = time.time()
    pr.enable()
    for _ in range(args.steps):
        env.step(episode_step=1, total_episodes=1)
        if env.is_episode_over():
            break
    pr.disable()
    elapsed = time.time() - t0

    print(f"\n{elapsed:.2f}s for {args.steps} steps = "
          f"{elapsed/max(1,args.steps)*1000:.0f} ms/step\n")

    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats(args.sort).print_stats(args.rows)
    print(buf.getvalue())

    print("=" * 72)
    print("READING THIS")
    print("=" * 72)
    print("is_structurally_valid high in ncalls AND tottime -> ray budget confirmed;")
    print("  fix is unit-step tracing capped at ray_range, ~10-16 iters not 100-1000.")
    print("get_local_affordance / _structure_mask on top    -> the field is the cost;")
    print("  the shared-field refactor is the right next move, not the ray loop.")
    print("torch internals on top                           -> it is the network,")
    print("  and none of the geometry work will move the wall clock.")
    print()
    print("NOTE: this profiles env.step() only. learn() is called by the runner,")
    print("not by step(), so the optimiser and backward pass are NOT in this")
    print("table. That is deliberate -- the question is whether the GEOMETRY is")
    print("the bottleneck -- but it means the numbers here are not a full")
    print("training step and should not be compared against one.")


if __name__ == "__main__":
    main()