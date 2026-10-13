"""
trace_fleet.py -- why does a robot take so long? Replays one deploy_eval instance
and follows every decision, for every arm.

WHAT IT RECORDS, for every robot on every step
  1. what the driver PROPOSED (the network, or RULES' shortest-path driver), and
     how that move relates to the robot's goal: toward / wait / sideways / away
     (from the shortest-path gradient, as the env's own diagnostics classify it);
  2. what was EXECUTED, and what decided it, in the env's own order of authority:
     rules order (L0/L2) or ladder retreat (L1)  >  recovery hold  >  livelock
     escape  >  tabu swap  >  the driver's own choice;
  3. the speed it moved at (braking near a peer slows it, down to 10%), and how
     far it actually moved;
  4. for a network: its Q-values -- for waiting, for the best move toward the
     goal, and for the move it chose -- so we can see whether a detour was a
     clear preference or a near-tie.
  It also notes whether another robot was within the warning distance (2 cells)
  before the move ("near"), so choices near peers and in the clear are split.

  Nothing is changed: the hooks only read. The runs match deploy_eval's exactly
  (same seed, same day). Before the line-end fix (node_warehouse.reach_position,
  build log entry 25) the check's instance gave champion 16 delivered and 3
  collisions, RULES 33 and 0; after it, different numbers are expected.

  --route-arms champion also traces "champion+route" (deploy_eval's route by
  default): steps the route decided are labelled "route default", and
  "proposed" is still the network's own choice.

OUTPUT
  1. a per-arm table: what the driver proposed and what ran, near peers and in
     the clear, who overrode it, how much it was braked;
  2. a step-by-step timeline of one robot (--fleet, its later lives too);
  3. --out: every robot-step as a CSV, for a closer look later.

RUN (a minute or two; uses the same files as deploy_eval.py):
  python -u trace_fleet.py --arms champion=ladder_run1/flowrra_curriculum.pth,RULES --route-arms champion --fleet 22

  The defaults are the check's instance: small map, 25 fleets, seed 0, 120 steps,
  failures and people on. Add warm=warm_stream1/flowrra_curriculum.pth to --arms
  to see the warm run too, or --max-steps 800 for a whole shift.
"""

import argparse
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from config_warehouse import CONFIG
import core_warehouse
from core_warehouse import FLOWRRA
from node_warehouse import FleetNode
import deploy_eval as de

ACTION_NAMES = {0: "wait", 1: "+X", 2: "-X", 3: "+Y", 4: "-Y", 5: "+Z", 6: "-Z"}
CLASSES = ["toward", "wait", "sideways", "away"]


def classify(a: int, grad: np.ndarray) -> str:
    """As the env's own diagnostic (_decided): wait, or the move's gradient sign."""
    a = int(a)
    if a == 0:
        return "wait"
    v = float(grad[a - 1])
    return "toward" if v > 0 else ("away" if v < 0 else "sideways")


# ----------------------------------------------------------------------------- hooks
class Recorder:
    """Reads the env at two points of a step: when the driver has proposed
    (inside choose_actions) and when each robot's move is executed
    (FleetNode.apply_discrete_action). Writes nothing back."""

    def __init__(self):
        self.env = None
        self.active = False
        self.pre = {}
        self.post = {}
        self.retrace_ids = set()
        self.step = -1
        self.want_q = False

    def on_proposed(self, actions, node_ids):
        env = self.env
        self.step = int(env.step_count)
        self.pre, self.post, self.retrace_ids = {}, {}, set()
        imm = set(env.immobile_nodes)
        idx = env._prox_all if getattr(env, "_prox_all", None) is not None else env.proximity
        # Q-values only for a network: RULES' driver may leave a ranking of its
        # own there (the tabu swap reads it), which is not a value.
        q = getattr(env.gnn, "last_q_values", None) if self.want_q else None
        # Route by default (deploy_eval.install_route_default): the network's own
        # proposal, before the route replaced it, and which fleets it replaced.
        raw = getattr(env.gnn, "_route_raw", None)
        rmask = getattr(env.gnn, "_route_mask", None)
        for i, node in enumerate(env.nodes):
            if node.id in imm:
                continue
            try:
                near = float(idx.nearest(node.id))
            except Exception:
                near = float("nan")
            self.pre[node.id] = {
                "i": i,
                "cell": tuple(int(round(float(v))) for v in node.current_pos),
                "goal": getattr(node, "current_goal_id", None),
                "hops": float(node.get_graph_distance_to_goal()),
                "grad": np.asarray(node.get_goal_gradient(), dtype=np.float64).copy(),
                "near": near,
                "prop": int(raw[i]) if raw is not None else int(actions[i]),
                "after_route": int(actions[i]),
                "routed": bool(rmask[i]) if rmask is not None else False,
                "q": None if q is None else np.asarray(q[i], dtype=np.float64).copy(),
            }
        self.active = True

    def on_apply(self, node, action):
        env = self.env
        if not self.active or node.id in self.post or node.id not in self.pre:
            return None
        # Route by default inside the env (conflict.route_by_default): applied
        # after choose_actions, so read here, when every decision is made.
        if getattr(env, "route_by_default", False) and getattr(env, "_route_mask", None) is not None:
            pre = self.pre[node.id]
            pre["routed"] = bool(env._route_mask[pre["i"]])
            pre["after_route"] = int(env._route_actions[pre["i"]])
        held = env.step_count < env._yield_until.get(node.id, -1)
        rule_actions = getattr(env, "_rule_actions", {}) or {}
        ruled = node.id in rule_actions and (not held or getattr(env, "ladder_on", False))
        livelock = env.step_count < env._override_until.get(node.id, -1)
        rec = {"exec": int(action), "speed": float(node.speed),
               # the intent guard turned this fleet's move into a wait
               "guard_stop": bool(getattr(env, "intent_guard", False) and int(action) == 0
                                  and float(node.speed) <= 0.0),
               "pos0": np.asarray(node.current_pos, dtype=np.float64).copy(),
               "ruled": ruled, "retreat": node.id in self.retrace_ids,
               "held": held, "livelock": livelock}
        self.post[node.id] = rec
        return rec


REC = Recorder()
_ORIG_APPLY = FleetNode.apply_discrete_action


def _apply_hook(self, action):
    rec = REC.on_apply(self, action)
    out = _ORIG_APPLY(self, action)
    if rec is not None:
        rec["moved"] = float(np.sum(np.abs(np.asarray(self.current_pos, dtype=np.float64) - rec["pos0"])))
    return out


FleetNode.apply_discrete_action = _apply_hook


def decided_by(pre, post, driver_word):
    if post["ruled"]:
        return "ladder retreat" if post["retreat"] else "rules order"
    if post["held"]:
        return "recovery hold"
    if post["livelock"]:
        return "livelock escape"
    if post.get("guard_stop") and pre["after_route"] != 0:
        return "safety guard"
    if pre["routed"] and post["exec"] == pre["after_route"]:
        return "route default"
    if post["exec"] != pre["after_route"]:
        return "tabu swap"
    return driver_word


# ----------------------------------------------------------------------------- one run
def traced_run(inst, arm, agent, epsilon, max_steps, base_speed, warn, route=False):
    """deploy_eval.run_one, step for step, with the recorder attached."""
    s = de.instance_seed(inst["map"], inst["seed"], inst["k"])
    de.seed_everything(s)
    core_warehouse._WAVE_EPISODES = s % 1_000_000
    core_warehouse._OBSTACLE_EPISODES = s % 1_000_000
    env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                  goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                  goal_pool=dict(inst["goal_pool"]), order_bank=inst["bank"],
                  order_seed=s, order_floor_window=None)
    if agent == "RULES":
        from drive_shortest_path import ShortestPathDriver
        driver = ShortestPathDriver(env, "never", 0.5, 0)
        eps = 0.0
    else:
        driver = agent
        eps = epsilon
    driver.reset_episode_state()
    driver.epsilon_gaussian = lambda *a, **kw: eps
    env.gnn = driver
    remove_route = de.install_route_default(driver, env, warn) if route else None

    orig_choose = driver.choose_actions

    def choose_hook(*a, **kw):
        acts = orig_choose(*a, **kw)
        REC.on_proposed(acts, kw.get("node_ids"))
        return acts

    driver.choose_actions = choose_hook
    orig_retrace = getattr(env, "_ladder_retrace_actions", None)
    if orig_retrace is not None:
        def retrace_hook():
            out = orig_retrace()
            REC.retrace_ids = set(out.keys())
            return out
        env._ladder_retrace_actions = retrace_hook

    REC.env = env
    REC.want_q = agent != "RULES"
    driver_word = "rules driver" if agent == "RULES" else "own choice"
    rows = []
    try:
        for _ in range(max_steps):
            REC.active = False
            env.step(episode_step=1, total_episodes=1)
            REC.active = False
            for fid, pre in REC.pre.items():
                post = REC.post.get(fid)
                if post is None:
                    continue
                q = pre["q"]
                g = pre["grad"]
                row = {"arm": arm, "step": REC.step, "fleet": fid, "cell": pre["cell"],
                       "goal": pre["goal"], "hops": pre["hops"], "near_peer": pre["near"],
                       "is_near": bool(np.isfinite(pre["near"]) and pre["near"] <= warn),
                       "proposed": pre["prop"], "executed": post["exec"],
                       "prop_class": classify(pre["prop"], g),
                       "exec_class": classify(post["exec"], g),
                       "decided_by": decided_by(pre, post, driver_word),
                       "routed": pre["routed"],
                       "speed": post["speed"], "braked": post["speed"] < base_speed - 1e-6,
                       "moved": post.get("moved", float("nan"))}
                if q is not None:
                    fin = np.isfinite(q)
                    toward = [a for a in range(1, 7) if g[a - 1] > 0 and fin[a]]
                    row["explored"] = bool(fin.any() and pre["prop"] != int(np.nanargmax(np.where(fin, q, -np.inf))))
                    row["q_wait"] = float(q[0]) if fin[0] else float("nan")
                    row["q_best_toward"] = max((float(q[a]) for a in toward), default=float("nan"))
                    row["q_chosen"] = float(q[pre["prop"]]) if fin[pre["prop"]] else float("nan")
                    row["q_spread"] = float(q[fin].max() - q[fin].min()) if fin.sum() > 1 else float("nan")
                rows.append(row)
            if env.is_episode_over():
                break
    finally:
        REC.active = False
        try:
            del driver.choose_actions        # the agent is reused: drop the hook
        except AttributeError:
            pass
        if remove_route is not None:
            remove_route()
    est = env.get_error_statistics()
    b = env._book_stats() if getattr(env, "_book", False) else {}
    print(f"[trace] {arm}: delivered {b.get('stream_book_delivered')} | collisions "
          f"{env.loop.total_collisions} | L3 {est.get('ladder_l3_fired')} "
          )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- reports
def pct(n, d):
    return f"{100.0 * n / d:5.1f}%" if d else "   - "


def summary(df: pd.DataFrame, arm: str) -> None:
    d = df[df.arm == arm]
    n = len(d)
    near, clear = d[d.is_near], d[~d.is_near]
    print(f"\n=== {arm}: {n} robot-steps ({len(near)} with another robot within 2 cells, "
          f"{len(clear)} in the clear) ===")
    print(f"{'':16}{'PROPOSED':>24}{'EXECUTED':>24}")
    print(f"{'':16}{'clear':>12}{'near peer':>12}{'clear':>12}{'near peer':>12}")
    for c in CLASSES:
        print(f"  {c:14}"
              f"{pct((clear.prop_class == c).sum(), len(clear)):>12}"
              f"{pct((near.prop_class == c).sum(), len(near)):>12}"
              f"{pct((clear.exec_class == c).sum(), len(clear)):>12}"
              f"{pct((near.exec_class == c).sum(), len(near)):>12}")
    by = Counter(d.decided_by)
    print("  what decided the executed move: " + ", ".join(
        f"{k} {pct(v, n).strip()}" for k, v in by.most_common()))
    over = d[d.proposed != d.executed]
    print(f"  overridden (executed differs from proposed): {pct(len(over), n).strip()}"
          + ("" if over.empty else " -- " + ", ".join(
              f"{k} {v}" for k, v in Counter(over.decided_by).most_common())))
    print(f"  braked (a peer close enough to slow it): {pct(int(d.braked.sum()), n).strip()}, "
          f"mean speed {d.speed.mean():.3f} (full speed {d.speed.max():.3f}); "
          f"did not move at all: {pct(int((d.moved < 1e-6).sum()), n).strip()}")
    if "q_chosen" in d and d.q_chosen.notna().any():
        nt = d[(d.prop_class != "toward") & d.q_best_toward.notna() & d.q_chosen.notna()
               & (d.explored == False)]
        if len(nt):
            m = (nt.q_chosen - nt.q_best_toward)
            sp = nt.q_spread
            r = (m / sp.replace(0, np.nan)).dropna()
            print(f"  when it chose NOT to head for its goal although a move toward it existed "
                  f"({len(nt)} robot-steps, exploration excluded):")
            print(f"    its choice beat the best move toward the goal by a median "
                  f"{m.median():.4f} (quartiles {m.quantile(.25):.4f} to {m.quantile(.75):.4f}),")
            print(f"    against a median spread of {sp.median():.4f} across its valid moves: "
                  f"the margin is {100 * r.median():.0f}% of the spread (median).")
            for c in ["wait", "sideways", "away"]:
                k = (nt.prop_class == c).sum()
                if k:
                    print(f"      {c:9} {k:5} robot-steps, near a peer in {pct(int(nt[nt.prop_class == c].is_near.sum()), k).strip()}")
        print(f"  explored (random move, epsilon): {pct(int(d.explored.sum()), n).strip()}")


def timeline(df: pd.DataFrame, arm: str, fleet: str, limit: int) -> None:
    d = df[(df.arm == arm) & ((df.fleet == fleet) | df.fleet.astype(str).str.startswith(fleet + "#"))]
    if d.empty:
        print(f"\n--- {arm}: fleet {fleet} never moved ---")
        return
    print(f"\n--- {arm}: fleet {fleet}, step by step ({len(d)} steps on the floor"
          f"{'' if len(d) <= limit else f', first {limit} shown'}) ---")
    hasq = "q_chosen" in d and d.q_chosen.notna().any()
    head = (f"{'step':>4} {'fleet':>6} {'cell':>13} {'goal':>6} {'hops':>5} {'peer':>5}  "
            f"{'proposed':15} {'executed':15} {'decided by':16} {'speed':>5} {'moved':>5}")
    if hasq:
        head += "   Q wait / toward / chosen"
    print(head)
    for _, r in d.head(limit).iterrows():
        line = (f"{r.step:>4} {str(r.fleet):>6} {str(r.cell):>13} {str(r.goal):>6} "
                f"{r.hops:>5.1f} {r.near_peer:>5.1f}  "
                f"{ACTION_NAMES[r.proposed] + ' (' + r.prop_class + ')':15} "
                f"{ACTION_NAMES[r.executed] + ' (' + r.exec_class + ')':15} "
                f"{r.decided_by:16} {r.speed:>5.2f} {r.moved:>5.2f}")
        if hasq and np.isfinite(r.q_chosen):
            line += (f"   {r.q_wait:7.3f} / {r.q_best_toward:7.3f} / {r.q_chosen:7.3f}"
                     + ("  (explored)" if r.explored else ""))
        print(line)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arms", required=True, help="as deploy_eval: label=checkpoint pairs and/or RULES")
    ap.add_argument("--fleet", default="22", help="the robot to follow step by step")
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--map", default="25_5_2_5_2_1")
    ap.add_argument("--agents", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--failures", choices=["on", "off"], default="on")
    ap.add_argument("--people", choices=["on", "off"], default="on")
    ap.add_argument("--book", type=int, default=80)
    ap.add_argument("--release", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=120)
    ap.add_argument("--epsilon", type=float, default=0.015)
    ap.add_argument("--route-arms", default="",
                    help="as deploy_eval: also run these checkpoint arms with route by default")
    ap.add_argument("--safety", choices=["config", "on", "off"], default="config",
                    help="as deploy_eval: build log entry 29's switches as configured, on, or off")
    ap.add_argument("--timeline-steps", type=int, default=150, help="max lines per arm in the timeline")
    ap.add_argument("--out", default="trace_steps.csv")
    args = ap.parse_args()

    # The same conditions as deploy_eval.main(), in the same order.
    CONFIG.setdefault("stream", {})["enabled"] = True
    CONFIG["stream"]["order_book"] = {"enabled": True, "orders": args.book,
                                      "release_steps": args.release}
    CONFIG.setdefault("errors", {})["enabled"] = (args.failures == "on")
    CONFIG.setdefault("obstacles", {})["enabled"] = (args.people == "on")
    if args.people == "on":
        CONFIG["obstacles"].setdefault("mode", "cleanup")
    CONFIG.setdefault("training", {})["max_steps_per_episode"] = args.max_steps
    de.apply_safety_mode(args.safety)
    base_speed = float(CONFIG["warehouse"].get("base_speed", 0.5))
    warn = float(CONFIG["warehouse"].get("warning_threshold", 2.0))

    from main_runner_warehouse import MapCache
    from benchmark_flowrra import build_agent
    cache = MapCache(args.maps_dir, args.scens_dir)
    inst = de.build_instance(cache, args.map, args.agents, args.seed)
    if inst is None:
        raise SystemExit("instance not available")

    route_labels = {a.strip() for a in args.route_arms.split(",") if a.strip()}
    frames, labels = [], []
    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if spec.upper() == "RULES":
            label, agent = "RULES", "RULES"
        else:
            label, path = spec.split("=", 1)
            agent = build_agent(inst["G"], inst["pos_dict"], inst["missions"], inst["goal_pool"], path)
        labels.append(label)
        frames.append(traced_run(inst, label, agent, args.epsilon, args.max_steps, base_speed, warn))
        if label in route_labels:
            labels.append(label + "+route")
            frames.append(traced_run(inst, label + "+route", agent, args.epsilon, args.max_steps,
                                     base_speed, warn, route=True))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out, index=False)

    for label in labels:
        summary(df, label)
    for label in labels:
        timeline(df, label, str(args.fleet), args.timeline_steps)
    print(f"\n[trace] wrote {len(df)} robot-steps -> {args.out}")


if __name__ == "__main__":
    main()
