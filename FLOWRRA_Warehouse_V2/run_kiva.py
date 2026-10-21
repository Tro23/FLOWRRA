"""
run_kiva.py -- FLOWRRA on Follower's kiva protocol (lifelong, 512 timesteps).

    python kiva_instances.py                               # once: the instances
    python run_kiva.py --policy rules                      # the rule-based orchestrator
    python run_kiva.py --policy checkpoint --checkpoint checkpoints/flowrra_warehouse_gnn.pth
    python run_kiva.py --policy rules --agents 64 --seeds 0,1

Each instance (kiva_instances/kiva_n<agents>_s<seed>.json) gives every fleet a
start at a home cell and a fixed goal sequence drawn by POGEMA exactly as in
Follower's evaluation. FLOWRRA runs in lifelong mode: on reaching a goal, a
fleet takes the next one at once.

UNITS. POGEMA agents move one cell per timestep; FLOWRRA moves base_speed (0.5)
cells per step, so 512 timesteps are 1,024 FLOWRRA steps, and throughput is
goals reached / 512 -- goals per timestep, the published unit.

WHAT DIFFERS, STATED PLAINLY. POGEMA resolves moves into the same cell with a
"soft" collision rule and never lets agents overlap; FLOWRRA moves in half-cell
steps, can collide, and recovers. So the maps, tasks, horizon and metric are
theirs; the physics is ours. Report collisions next to throughput.

POLICIES
    rules       the shortest-path driver with the six conflict rules on: a
                rule-based, reservation-style controller (the orchestrator
                baseline)
    checkpoint  a trained FLOWRRA network, greedy (on 50_-trained weights this is a
                zero-shot transfer test: kiva is an open grid it has never seen)
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import random
import time

import numpy as np

import drive_shortest_path as drv
from config_warehouse import CONFIG

RULES = {"conflict.path_warnings": True, "conflict.directional_braking": True,
         "conflict.corridor_entry": True, "conflict.priority": True,
         "conflict.node_aligned_moves": True, "conflict.yield_to_stopped": True,
         "stream.enabled": False, "errors.enabled": False}


def build_env(inst: dict, maps_dir: str, scens_dir: str):
    from main_runner_warehouse import MapCache
    from core_warehouse import FLOWRRA
    e = MapCache(maps_dir, scens_dir).get("kiva")
    pos = e["pos_dict"]
    coords = lambda nid: np.array([pos[nid]["X"], pos[nid]["Y"], pos[nid]["Z"]], dtype=np.float32)
    starts = [str(v) for v in inst["starts"]]
    seqs = [[str(g) for g in q] for q in inst["goals"]]
    missions = [{"id": f"a{i}", "start_node": s, "goal_node": q[0],
                 "start_pos": coords(s), "goal_pos": coords(q[0])} for i, (s, q) in enumerate(zip(starts, seqs))]
    return FLOWRRA(e["G"], pos, missions, mode="training", goal_distance_maps=e["goal_distance_maps"],
                   shared_pool_mode=False, lifelong_goals={f"a{i}": q for i, q in enumerate(seqs)})


def attach_checkpoint(env, path: str):
    from agent_warehouse import GNNAgent
    from main_runner_warehouse import _recovery_value_bound
    base = len(env.nodes[0].get_state_vector(env.nodes))
    rd, g = CONFIG["reward_decomposition"], CONFIG["gnn"]
    agent = GNNAgent(node_feature_dim=base + env.density.output_dim, edge_feature_dim=g["edge_feature_dim"],
                     action_size=g["action_size"], hidden_dim=g["hidden_dim"], num_layers=g["num_layers"],
                     n_heads=g["num_heads"], reward_heads=rd["heads"], head_weights=rd["weights"],
                     dropout=g["dropout"], lr=g["learning_rate"], gamma=CONFIG["training"]["gamma"],
                     buffer_capacity=1000, batch_size=CONFIG["training"]["batch_size"],
                     stability_coef=g["stability_coef"], encoder_mode=g["encoder"], base_feature_dim=base,
                     density_diamond_mask=env.density._diamond_mask,
                     recovery_value_bound=_recovery_value_bound())
    agent.load(path)
    agent.epsilon_gaussian = lambda *a, **k: 0.0          # greedy
    agent.learn = lambda *a, **k: None
    agent.reset_episode_state()
    env.gnn = agent


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--policy", choices=("rules", "checkpoint"), default="rules")
    ap.add_argument("--checkpoint", default="checkpoints/flowrra_warehouse_gnn.pth")
    ap.add_argument("--instances", default="kiva_instances")
    ap.add_argument("--agents", default="32,64,96,128,160,192")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    ap.add_argument("--timesteps", type=int, default=512)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v4")
    ap.add_argument("--out", default="kiva_results.csv")
    ap.add_argument("--set", action="append", default=[],
                    help="dotted.key=value over the policy's defaults (repeatable), "
                         "e.g. warehouse.braking=false or conflict.priority=false")
    ap.add_argument("--label", default=None, help="name for this configuration in the CSV")
    args = ap.parse_args()

    speed = float(CONFIG["warehouse"]["base_speed"])
    steps = int(round(args.timesteps / speed))
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["label", "policy", "agents", "seed", "timesteps", "steps", "goals", "throughput",
                    "collisions", "ms_per_step", "settings"])
    extra = drv.parse_arm("x:" + ",".join(args.set))[1] if args.set else {}
    label = args.label or args.policy
    for n in [int(a) for a in args.agents.split(",")]:
        tps = []
        for s in [int(x) for x in args.seeds.split(",")]:
            path = os.path.join(args.instances, f"kiva_n{n}_s{s}.json")
            if not os.path.exists(path):
                print(f"  missing {path} -- run kiva_instances.py first"); continue
            inst = json.load(open(path))
            base = RULES if args.policy == "rules" else {"stream.enabled": False, "errors.enabled": False}
            drv.apply_config({**base, **extra})
            random.seed(s); np.random.seed(s)
            with contextlib.redirect_stdout(io.StringIO()):
                env = build_env(inst, args.maps_dir, args.scens_dir)
            if args.policy == "rules":
                env.gnn = drv.ShortestPathDriver(env, "never", 0.5, s)
            else:
                attach_checkpoint(env, args.checkpoint)
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                for _ in range(steps):
                    env.step(episode_step=1, total_episodes=1)
            st = env.get_error_statistics()
            ms = 1000 * (time.perf_counter() - t0) / steps
            tp = st["lifelong_goals_reached"] / args.timesteps
            tps.append(tp)
            w.writerow([label, args.policy, n, s, args.timesteps, steps, st["lifelong_goals_reached"],
                        round(tp, 4), env.loop.total_collisions, round(ms, 1), json.dumps(extra, sort_keys=True)])
            fh.flush()
            print(f"  {label:<14} {n:>3} agents seed {s}: {st['lifelong_goals_reached']:>4} goals, "
                  f"throughput {tp:.3f}/timestep, collisions {env.loop.total_collisions}, {ms:.0f} ms/step")
        if tps:
            print(f"{label} {n} agents: mean throughput {np.mean(tps):.3f} over {len(tps)} seeds")
    drv.apply_config({})
    fh.close()


if __name__ == "__main__":
    main()
