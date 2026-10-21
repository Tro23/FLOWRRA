"""
deploy_eval.py -- frozen checkpoints in deployment conditions.

WHAT IT ANSWERS
  "How does this fleet do on the job?" -- not how it learns. Every arm drives
  FROZEN (no learning, a small fixed exploration) through the same fixed, seeded
  instances of the order stream with a fixed ORDER BOOK, so arms are compared
  like for like:

    --arms champion=ladder_run1/flowrra_curriculum.pth,warm=warm_stream1/flowrra_curriculum.pth,RULES

  RULES is the orchestrator with shortest-path driving instead of the learned
  policy (the benchmark's RULES arm, drive_shortest_path.ShortestPathDriver).

  --route-arms champion adds "champion+route": the same network, but a robot with
  no other robot within the warning distance (and in no warning or deadlock set)
  takes its shortest-path move; the network decides only near other robots
  (install_route_default; build log entry 25). Rules, ladder, holds and the veto
  apply after it, as for every arm.

MAPF BASELINES ON OUR FLOOR (build log entry 44; mapf_drivers.py)
  --arms ...,RULES,PIBT,RHCR-PP,RHCR-PIBT   (RHCR-*+naive: the naive rescue, once
  its rule is agreed). The planners from baselines_mapf.py are the drivers; the
  world (map, book, greedy dispatcher, docks, rescue handover, keyed failures
  and people) and physical safety (gap, guard, bubbles, direction-aware braking,
  never into a robot that stays put, people a hard no-go, collision recovery)
  are the same for every arm. Our brain (route by default, corridor entry,
  priority, the ladder, livelock escape) is off for a baseline arm only. Rows
  gain arm_kind, planner_ms_per_step (reported, not charged), and for RHCR
  planner_replans / planner_offplan; L3 is NaN for a baseline (no ladder).
  --rhcr-window 20 --rhcr-replan-every 5 (timesteps, one hop each; benchmark_all's).

THE SAME DAY FOR EVERY ARM
  An instance is (map, fleet count, seed). From it come: the scenario file (the
  seed-th file of that map, sorted), the first k fleets of that file, the order
  book's order sequence, and the random streams of the failure waves, the single
  failures, the debris and the humans. Python's random, numpy and torch are
  seeded from it too, before every arm. So every arm starts from exactly the
  same day; once their choices differ, later details drift apart, as they would
  on a real floor.

CONDITIONS (command line, not the config)
  --failures on|off   robots failing (errors + failure waves, as configured)
  --people on|off     debris and the humans who clear it (obstacles "cleanup")
  --book 80 --release 400 --max-steps 800 --epsilon 0.015
  Everything else comes from config_warehouse.py: run it with the config the
  checkpoints were trained under (run 5's config works for run 2 and run 5).

OUTPUT
  --out deploy_eval.csv: one row per (arm, instance), written after every run.
  At the end, a summary by arm and map x fleets, and paired comparisons of each
  arm against the first one (Wilcoxon signed-rank over shared instances).
  Orders per hour assume one step = one second (0.5 m/s, 1 m cells); scale by
  1 / seconds-per-step for other robots.

EXAMPLES
  # quick check that it runs and that repeats are identical (a few minutes)
  python -u deploy_eval.py --arms champion=ladder_run1/flowrra_curriculum.pth,RULES --route-arms champion \\
      --maps 25_5_2_5_2_1 --agents 25 --seeds 0 --max-steps 120 --check --out deploy_check2.csv

  # clean floor: no failures, no people, 3 seeds, four arms (72 runs; roughly
  # 3-5 hours -- the progress line prints an ETA after the first run)
  python -u deploy_eval.py --arms champion=ladder_run1/flowrra_curriculum.pth,warm=warm_stream1/flowrra_curriculum.pth,RULES \\
      --route-arms champion --failures off --people off --seeds 0,1,2 --out deploy_eval_clean.csv

  # run 5's conditions, four arms, 5 seeds (120 runs; roughly 6-9 hours)
  python -u deploy_eval.py --arms champion=ladder_run1/flowrra_curriculum.pth,warm=warm_stream1/flowrra_curriculum.pth,RULES \\
      --route-arms champion --failures on --people on --seeds 0,1,2,3,4 --out deploy_eval_full.csv
"""

import argparse
import os
import random
import time
import zlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from config_warehouse import CONFIG
import core_warehouse
from core_warehouse import FLOWRRA
from node_warehouse import precompute_goal_distances_compact

KEYED_DRAWS = True      # --keyed-draws (main sets it); the order-leak fix, entry 43
RHCR_WINDOW = 20        # --rhcr-window, timesteps (benchmark_all's default)
RHCR_REPLAN = 5         # --rhcr-replan-every, timesteps (benchmark_all's default)

FAIR_COLS = ["stream_book_fill", "stream_book_lead_mean", "collisions", "ladder_l3_fired",
             "obstacle_near_miss_steps", "orders_per_hour", "guard_stops", "guard_pairs_short"]


# ----------------------------------------------------------------------------- safety mode
def apply_safety_mode(mode: str) -> None:
    """--safety: "config" uses config_warehouse.py as it is. "off" switches off
    everything build log entry 29 added -- the intent guard, bubbles by kind,
    hops everywhere, route by default inside the env -- and restores the
    presence charge: the step exactly as before it (an old checkpoint's check
    must reproduce yesterday's numbers). "on" switches them all on."""
    # conflict.route_explore (build log entry 34, Fix 3) is TRAINING ONLY: a
    # deployed system never lets a fleet explore through the route. Forced
    # off here, whatever the config says and in every mode, so every frozen
    # test (and trace_fleet, which calls this) runs the route as deployed.
    CONFIG.setdefault("conflict", {})["route_explore"] = False
    if mode == "config":
        return
    on = (mode == "on")
    sf = CONFIG.setdefault("safety", {})
    for k in ("intent_guard", "bubbles_by_kind", "hops_everywhere"):
        sf[k] = on
    CONFIG.setdefault("conflict", {})["route_by_default"] = on
    CONFIG.setdefault("reward_decomposition", {})["presence_charge"] = not on


def apply_world(fixes: str, orphans: str) -> None:
    """Build log entry 44, for every arm alike. fixes "on": safety.node_snap_eps
    1e-4 and safety.bubble_floor_fix (the float stall and the convoy crawl);
    "off": the old world. orphans "pickup": errors.orphans_need_pickup."""
    sf = CONFIG.setdefault("safety", {})
    sf["node_snap_eps"] = 1e-4 if fixes == "on" else 0.0
    sf["bubble_floor_fix"] = (fixes == "on")
    CONFIG.setdefault("errors", {})["orphans_need_pickup"] = (orphans == "pickup")


# ----------------------------------------------------------------------------- instances
def instance_seed(map_name: str, seed: int, k: int) -> int:
    return (zlib.crc32(map_name.encode()) ^ (int(seed) * 1_000_003) ^ (int(k) * 7_919)) & 0x7FFFFFFF


def seed_everything(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def build_instance(cache, map_name: str, k: int, seed: int) -> Optional[Dict[str, Any]]:
    """The seed-th scenario file of this map (sorted), its first k usable rows."""
    e = cache.get(map_name)
    files = e["scen_files"]
    if not files:
        return None
    scen = files[seed % len(files)]
    df = pd.read_csv(os.path.join(e["scen_dir"], scen), index_col=False)
    df["startNodeId"] = df["startNodeId"].astype(str).str.strip()
    df["goalNodeId"] = df["goalNodeId"].astype(str).str.strip()
    pos = e["pos_dict"]
    df = df[df["startNodeId"].isin(pos) & df["goalNodeId"].isin(pos)].head(k)
    if len(df) < k:
        return None
    missions = [{"id": str(r["agentId"]).strip(), "start_node": r["startNodeId"],
                 "start_pos": np.array([pos[r["startNodeId"]]["X"], pos[r["startNodeId"]]["Y"],
                                        pos[r["startNodeId"]]["Z"]], dtype=np.float32)}
                for _, r in df.iterrows()]
    goal_pool = {g: np.array([pos[g]["X"], pos[g]["Y"], pos[g]["Z"]], dtype=np.float32)
                 for g in df["goalNodeId"].unique()}
    missing = [g for g in goal_pool if g not in e["goal_distance_maps"]]
    if missing:                                   # as main_runner's sample_instance
        extra, _ = precompute_goal_distances_compact(e["G"], missing, node_index=e["node_index"])
        e["goal_distance_maps"].update(extra)
    return {"map": map_name, "k": k, "seed": seed, "scen": scen, "G": e["G"], "pos_dict": pos,
            "missions": missions, "goal_pool": goal_pool, "gdm": e["goal_distance_maps"],
            "bank": e["bank"]}


# ----------------------------------------------------------------------------- route by default
def install_route_default(driver, env, radius: float):
    """
    ROUTE BY DEFAULT (build log entry 25). The network decides only where it is
    meant to: in contention. Every step, after the network has proposed, a fleet
    that is mobile, in no warning or deadlock set, and has no live peer within
    RADIUS (the warning distance) takes its shortest-path move instead -- the
    goal gradient's best move, if that move is open in the action mask (so the
    hard veto on people still holds). Everywhere else the network's choice
    stands. Rules, ladder orders, holds, the livelock escape and the tabu swap
    all still apply afterwards, in the env, exactly as for any other arm.

    Why: the trace (trace_fleet.py) showed the champion's choice close to random
    far from its goal -- 32% toward, 37% away, 31% wait, more than 10 hops out in
    the clear -- while RULES, which reads the same BFS map, goes straight there.

    Returns a function that removes the hook (the agent is shared between arms)
    and leaves counts on driver._route_counts.
    """
    orig = driver.choose_actions
    counts = {"steps": 0, "routed": 0, "network_differed": 0, "no_route_move": 0}

    def choose(*a, **kw):
        acts = orig(*a, **kw)
        raw = np.array(acts, copy=True)
        routed = np.zeros(len(acts), dtype=bool)
        vm = kw.get("valid_action_masks")
        imm = env.immobile_nodes
        busy = set(getattr(env.loop, "warning_nodes", ()) or ()) | set(
            getattr(env.loop, "deadlocked_nodes", ()) or ())
        for i, node in enumerate(env.nodes):
            if node.id in imm:
                continue
            counts["steps"] += 1
            if node.id in busy or env.proximity.nearest(node.id) <= radius:
                continue                                  # contention: the network decides
            g = node.get_goal_gradient()
            if not np.any(g > 0):
                counts["no_route_move"] += 1              # at the goal, or no way shown
                continue
            move = int(np.argmax(g)) + 1
            if vm is not None and not bool(vm[i][move]):
                counts["no_route_move"] += 1
                continue
            counts["routed"] += 1
            counts["network_differed"] += int(int(raw[i]) != move)
            acts[i] = move
            routed[i] = True
        driver._route_raw, driver._route_mask = raw, routed
        return acts

    driver.choose_actions = choose
    driver._route_counts = counts

    def remove():
        for k in ("choose_actions", "_route_raw", "_route_mask", "_route_counts"):
            driver.__dict__.pop(k, None)
    return remove


# ----------------------------------------------------------------------------- one run
def run_one(inst: Dict[str, Any], arm: str, agent, epsilon: float, max_steps: int,
            route: bool = False, env_route_off: bool = False) -> Dict[str, Any]:
    s = instance_seed(inst["map"], inst["seed"], inst["k"])
    seed_everything(s)
    # The env's own streams (waves, obstacles) count episodes in module globals:
    # pin them per instance so every arm draws the same waves, failures and debris.
    core_warehouse._WAVE_EPISODES = s % 1_000_000
    core_warehouse._OBSTACLE_EPISODES = s % 1_000_000
    # THE ORDER-LEAK FIX (build log entry 43): every failure, wave and people
    # draw keyed by (instance seed, what, step, who), so arms whose robots
    # differ still meet the same events wherever their floors allow.
    CONFIG.setdefault("keyed_draws", {})["seed"] = s if KEYED_DRAWS else None
    # A ":noroute" arm (build log entry 42, the router's graduation): route by
    # default OFF in the env for this arm only -- the network drives the clear.
    _ccfg = CONFIG.setdefault("conflict", {})
    _route_saved = _ccfg.get("route_by_default", False)
    if env_route_off:
        _ccfg["route_by_default"] = False
    # A MAPF baseline (mapf_drivers.py, build log entry 44): the same world and
    # physical safety, our brain (route, corridor entry, priority, ladder,
    # livelock escape) off while its env is built.
    from mapf_drivers import BASELINE_ARMS, brain_off, make_driver
    is_base = isinstance(agent, str) and agent in BASELINE_ARMS
    _brain = brain_off() if is_base else None
    if _brain is not None:
        _brain.__enter__()
    # "+naive" arms: the same rescuer choice, no re-dispatch (errors.naive_rescue).
    _ecfg = CONFIG.setdefault("errors", {})
    _naive_saved = _ecfg.get("naive_rescue", False)
    _ecfg["naive_rescue"] = bool(is_base and agent.endswith("+naive"))
    try:
        if inst.get("lifelong") is not None:
            # LIFELONG (kiva_eval.py, Follower's protocol): each fleet's goal
            # sequence is fixed in advance; on arrival it takes the next one at
            # once. No order book, no docks, no pool.
            env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                          goal_distance_maps=inst["gdm"], shared_pool_mode=False,
                          lifelong_goals=inst["lifelong"])
        else:
            env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                          goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                          goal_pool=dict(inst["goal_pool"]), order_bank=inst["bank"],
                          order_seed=s, order_floor_window=None)
    finally:
        if _brain is not None:
            _brain.__exit__(None, None, None)
        _ecfg["naive_rescue"] = _naive_saved
        _ccfg["route_by_default"] = _route_saved
    if agent == "RULES":
        from drive_shortest_path import ShortestPathDriver
        driver = ShortestPathDriver(env, "never", 0.5, 0)
        eps = 0.0                                 # a rules backup does not explore on the job
    elif is_base:
        driver = make_driver(agent, env, RHCR_WINDOW, RHCR_REPLAN)
        eps = 0.0                                 # planners do not explore
    else:
        driver = agent
        eps = epsilon
    driver.reset_episode_state()
    driver.epsilon_gaussian = lambda *a, **kw: eps
    env.gnn = driver
    remove_route = (install_route_default(driver, env, float(CONFIG["warehouse"]["warning_threshold"]))
                    if route else None)

    t0 = time.time()
    steps = 0
    try:
        for _ in range(max_steps):
            env.step(episode_step=1, total_episodes=1)      # frozen: no learn()
            steps += 1
            if env.is_episode_over():
                break
        rc = dict(getattr(driver, "_route_counts", {}) or {})
    finally:
        if remove_route is not None:
            remove_route()
    est = env.get_error_statistics()
    b = env._book_stats() if getattr(env, "_book", False) else {}
    done = b.get("stream_book_done_step", -1)
    delivered = b.get("stream_book_delivered", est.get("stream_deliveries", 0))
    row = {"arm": arm, "map": inst["map"], "k": inst["k"], "seed": inst["seed"],
           "scen": inst["scen"], "epsilon": eps, "route_default": bool(route),
           "route_env_off": bool(env_route_off),
           "route_share": (rc["routed"] / rc["steps"]) if rc.get("steps") else float("nan"),
           "route_network_differed": (rc["network_differed"] / rc["routed"]) if rc.get("routed") else float("nan"),
           "steps_run": steps,
           "runtime_s": round(time.time() - t0, 1),
           "collisions": env.loop.total_collisions,
           "orders_per_hour": delivered * 3600.0 / max(1, max_steps),
           "orders_per_robot_hour": delivered * 3600.0 / max(1, max_steps) / inst["k"],
           "minutes_to_finish_book": (done / 60.0) if done >= 0 else float("nan")}
    row.update(b)
    keep = ("ladder_", "obstacle_", "stream_", "orders_", "wave_", "incomplete_", "guard_",
            "router_", "lifelong_")
    row.update({k: v for k, v in est.items() if k.startswith(keep) and k not in row})
    # Route by default inside the env (conflict.route_by_default, build log
    # entry 29): its own counters, when this arm did not add the outside hook.
    if not route and est.get("route_on"):
        row.update(route_default=True, route_share=est.get("route_share", float("nan")),
                   route_network_differed=est.get("route_network_differed", float("nan")))
    for k in ("errors_injected", "handovers_completed", "brake_duty_cycle"):
        if k in est:
            row[k] = est[k]
    row["arm_kind"] = "baseline" if is_base else ("rules" if agent == "RULES" else "network")
    if is_base:
        # No ladder runs for a baseline: L3 is not zero, it does not exist.
        row["ladder_l3_fired"] = float("nan")
        row.update(driver.planner_stats())
    return row


# ----------------------------------------------------------------------------- summary
def summarise(df: pd.DataFrame, arms: List[str]) -> None:
    from scipy.stats import wilcoxon
    print("\n=== by arm, map and fleet count (means) ===")
    cols = [c for c in ["stream_book_fill", "orders_per_hour", "stream_book_lead_mean",
                        "collisions", "ladder_l3_fired", "obstacle_near_miss_steps",
                        "guard_stops"] if c in df]
    print(df.groupby(["arm", "map", "k"])[cols].mean().round(3).to_string())
    base = arms[0]
    key = ["map", "k", "seed"]
    A = df[df.arm == base].set_index(key)
    for other in arms[1:]:
        B = df[df.arm == other].set_index(key)
        idx = A.index.intersection(B.index)
        if len(idx) == 0:
            continue
        print(f"\n=== {other} vs {base}, paired over {len(idx)} instances ===")
        for c in [c for c in FAIR_COLS if c in df]:
            a, b = A.loc[idx, c].astype(float), B.loc[idx, c].astype(float)
            ok = a.notna() & b.notna()
            try:
                p = wilcoxon(a[ok].to_numpy(), b[ok].to_numpy()).pvalue if ok.sum() >= 5 and not np.allclose(a[ok], b[ok]) else float("nan")
            except ValueError:
                p = float("nan")
            print(f"  {c:28} {base} {a.mean():9.3f} | {other} {b.mean():9.3f} | p {p:.3f}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arms", required=True,
                    help="comma-separated label=checkpoint pairs, and/or RULES; the first is the reference")
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2", help="held-out instances by default")
    ap.add_argument("--maps", default="25_5_2_5_2_1,50_20_5_10_5_2")
    ap.add_argument("--agents", default="25,40,60")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--failures", choices=["on", "off"], default="on")
    ap.add_argument("--people", choices=["on", "off"], default="on")
    ap.add_argument("--book", type=int, default=80)
    ap.add_argument("--release", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=800)
    ap.add_argument("--epsilon", type=float, default=0.015)
    ap.add_argument("--route-arms", default="",
                    help="labels of checkpoint arms to ALSO run with route by default, as "
                         "<label>+route (same checkpoint; see install_route_default)")
    ap.add_argument("--safety", choices=["config", "on", "off"], default="config",
                    help="build log entry 29's switches: as in the config, all on, or all off "
                         "(off reproduces the step before them exactly)")
    ap.add_argument("--keyed-draws", choices=["on", "off"], default="on",
                    help="on: failures, waves and people keyed per instance, step and robot "
                         "(the order-leak fix, entry 43); off: the old shared streams, to "
                         "reproduce CSVs written before the fix")
    ap.add_argument("--check", action="store_true",
                    help="run the first instance of the first arm twice and report whether they match")
    ap.add_argument("--world-fixes", choices=["on", "off"], default="on",
                    help="build log entry 44's two physics fixes, for every arm: snap a robot "
                         "within 0.0001 hops of a node onto it, and a bubble that ends at "
                         "min_gap does not brake a pair at min_gap or more. off: the old "
                         "world, to reproduce CSVs written before")
    ap.add_argument("--assignment", choices=["config", "greedy", "hungarian"], default="config",
                    help="the spawn-time goal assignment for every arm (CONFIG assignment.mode): "
                         "config = as the config says (what every earlier run used); greedy = "
                         "each fleet takes its nearest open goal; hungarian = one optimal match")
    ap.add_argument("--orphans", choices=["pickup", "book"], default="pickup",
                    help="pickup: a stranded order can be delivered only after its load is "
                         "fetched from the dead robot (entry 44, 'handover, not reissue'); "
                         "book: the old world, where it went back to the book")
    ap.add_argument("--rhcr-window", type=int, default=20,
                    help="RHCR-* arms: planning window, timesteps (one hop each)")
    ap.add_argument("--rhcr-replan-every", type=int, default=5,
                    help="RHCR-* arms: replan every this many timesteps (and whenever the floor changes)")
    ap.add_argument("--out", default="deploy_eval.csv")
    args = ap.parse_args()
    global KEYED_DRAWS, RHCR_WINDOW, RHCR_REPLAN
    KEYED_DRAWS = (args.keyed_draws == "on")
    RHCR_WINDOW, RHCR_REPLAN = args.rhcr_window, args.rhcr_replan_every

    # Conditions: the command line, not the config.
    CONFIG.setdefault("stream", {})["enabled"] = True
    CONFIG["stream"]["order_book"] = {"enabled": True, "orders": args.book,
                                      "release_steps": args.release}
    CONFIG.setdefault("errors", {})["enabled"] = (args.failures == "on")
    CONFIG.setdefault("obstacles", {})["enabled"] = (args.people == "on")
    if args.people == "on":
        CONFIG["obstacles"].setdefault("mode", "cleanup")
    # The env reads its step budget from the config (failure timing, throughput
    # pressure): keep it equal to the steps we actually run.
    CONFIG.setdefault("training", {})["max_steps_per_episode"] = args.max_steps
    apply_safety_mode(args.safety)
    apply_world(args.world_fixes, args.orphans)
    if args.assignment != "config":
        CONFIG.setdefault("assignment", {})["mode"] = args.assignment

    from main_runner_warehouse import MapCache
    from benchmark_flowrra import build_agent
    cache = MapCache(args.maps_dir, args.scens_dir)
    maps = [m.strip() for m in args.maps.split(",") if m.strip()]
    ks = [int(v) for v in args.agents.split(",") if v.strip()]
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    insts = []
    for m in maps:
        for k in ks:
            for sd in seeds:
                inst = build_instance(cache, m, k, sd)
                if inst is None:
                    print(f"[deploy] skip {m} k={k} seed={sd}: not enough usable rows")
                    continue
                insts.append(inst)
    if not insts:
        raise SystemExit("no instances")

    route_labels = {a.strip() for a in args.route_arms.split(",") if a.strip()}
    arms: List[Tuple[str, Any, bool]] = []
    from mapf_drivers import baseline_name, BRAIN_OFF, BASELINE_ARMS
    BASELINE_ARMS_SET = set(BASELINE_ARMS)
    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if spec.upper() == "RULES":
            arms.append(("RULES", "RULES", False, False))
            continue
        base = baseline_name(spec)
        if base is not None:
            arms.append((base, base, False, False))
            print(f"[deploy] arm {base}: MAPF baseline on our floor (mapf_drivers.py)"
                  + (f", window {args.rhcr_window}, replan every {args.rhcr_replan_every} timesteps"
                     if base.startswith("RHCR") else ", per step"))
            continue
        label, path = spec.split("=", 1)
        # "label=path:noroute": the env's route by default OFF for this arm
        # (build log entry 42) -- the router's graduation exam.
        noroute = path.endswith(":noroute")
        if noroute:
            path = path[: -len(":noroute")]
        i0 = insts[0]
        agent = build_agent(i0["G"], i0["pos_dict"], i0["missions"], i0["goal_pool"], path)
        arms.append((label, agent, False, noroute))
        print(f"[deploy] arm {label}: {path}" + (" | route OFF for this arm" if noroute else ""))
        if label in route_labels:
            arms.append((label + "+route", agent, True, False))    # same network, route by default
            print(f"[deploy] arm {label}+route: {path}, shortest-path move unless a robot is "
                  f"within {CONFIG['warehouse']['warning_threshold']} cells")
            route_labels.discard(label)
    if route_labels:
        raise SystemExit(f"--route-arms names no checkpoint arm: {sorted(route_labels)}")
    _sf = CONFIG.get("safety", {}) or {}
    print(f"[deploy] safety={args.safety}: guard {bool(_sf.get('intent_guard'))}, bubbles "
          f"{bool(_sf.get('bubbles_by_kind'))}, hops {bool(_sf.get('hops_everywhere'))}, "
          f"route in env {bool((CONFIG.get('conflict') or {}).get('route_by_default'))}, "
          f"route explore {bool((CONFIG.get('conflict') or {}).get('route_explore'))} (training only)")
    print(f"[deploy] keyed draws {args.keyed_draws} (failures, waves and people keyed per "
          f"instance, step and robot; entry 43)")
    _lrn = ((CONFIG.get("training", {}) or {}).get("learner", {}) or {})
    print(f"[deploy] world fixes {args.world_fixes} (node snap "
          f"{CONFIG['safety']['node_snap_eps']}, bubble floor fix "
          f"{CONFIG['safety']['bubble_floor_fix']}) | orphans {args.orphans} "
          f"(need a pickup: {CONFIG['errors']['orphans_need_pickup']}) | entry 44")
    print(f"[deploy] assignment {CONFIG['assignment']['mode']} | router floor (scale_end) "
          f"{(_lrn.get('router', {}) or {}).get('scale_end')} | next_state_edge_features "
          f"{CONFIG['gnn'].get('next_state_edge_features')} | base_speed "
          f"{CONFIG['warehouse']['base_speed']}")
    if any(a[0] in BASELINE_ARMS_SET for a in arms):
        print("[deploy] baseline arms: our brain OFF ("
              + ", ".join(".".join(k) + "=" + str(v) for k, v in BRAIN_OFF.items())
              + "); world and physical safety as above, the same for every arm")
    print(f"[deploy] {len(insts)} instances x {len(arms)} arms | failures={args.failures} "
          f"people={args.people} | book {args.book} over {args.release} steps | "
          f"{args.max_steps} steps | epsilon {args.epsilon} (RULES 0)")

    if args.check:
        # The first arm, and the first route arm if there is one, each twice.
        to_check = [arms[0]] + [a for a in arms if a[2]][:1]
        for label, agent, route, noroute in to_check:
            r1 = run_one(insts[0], label, agent, args.epsilon, args.max_steps, route, noroute)
            r2 = run_one(insts[0], label, agent, args.epsilon, args.max_steps, route, noroute)
            same = all(r1.get(c) == r2.get(c) for c in
                       ["collisions", "stream_book_delivered", "stream_book_lead_mean",
                        "ladder_l3_fired", "obstacle_debris_spawned", "errors_injected"])
            print(f"[check] repeat of {label} on {insts[0]['map']} k={insts[0]['k']} seed={insts[0]['seed']}: "
                  f"{'IDENTICAL' if same else 'DIFFERENT'} "
                  f"(delivered {r1.get('stream_book_delivered')} / {r2.get('stream_book_delivered')}, "
                  f"collisions {r1['collisions']} / {r2['collisions']})")

    rows, t0 = [], time.time()
    total = len(insts) * len(arms)
    for inst in insts:
        for label, agent, route, noroute in arms:
            r = run_one(inst, label, agent, args.epsilon, args.max_steps, route, noroute)
            r.update(failures=args.failures, people=args.people)
            rows.append(r)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            n = len(rows)
            eta = (time.time() - t0) / n * (total - n) / 60
            print(f"[deploy] {n}/{total} | {label:9} {inst['map']} k={inst['k']} seed={inst['seed']} | "
                  f"delivered {r.get('stream_book_delivered')}/{args.book} | coll {r['collisions']} | "
                  f"L3 {r.get('ladder_l3_fired')} | {r['runtime_s']:.0f}s | eta {eta:.0f}m"
                  + (f" | routed {100 * r['route_share']:.0f}% of moves" if route else ""), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[deploy] wrote {len(df)} rows -> {args.out}")
    summarise(df, [a[0] for a in arms])


if __name__ == "__main__":
    main()