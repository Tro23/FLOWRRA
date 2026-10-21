"""
kiva_eval.py -- FLOWRRA on Follower's kiva protocol: the fair comparison with the
published state of the art (build log entry 43).

    # once: the map and the instances (RHCR's kiva.map; POGEMA 1.4.0)
    python convert_grid_map.py kiva.map --name kiva
    python kiva_instances.py

    # our system, with our safety rules (the real product)
    python -u kiva_eval.py --arms ladder6=ladder_run6/flowrra_curriculum.pth:noroute,ladder4=ladder_run4/flowrra_curriculum.pth,RULES,PIBT --out kiva_eval.csv

    # under MAPF's rules: no braking near peers (the like-for-like run)
    python -u kiva_eval.py --arms ... --braking off --out kiva_eval_mapf.csv

THE PROTOCOL (kiva_instances.py has the details): RHCR's kiva map (33 x 46,
1,278 cells), lifelong ("on_target restart"), 512 timesteps, 32-192 agents,
seeds 0-9. Each agent's goal sequence is fixed in advance by POGEMA's own draws,
so every method meets identical tasks. Throughput = goals reached / 512, goals
per timestep -- the unit Learn-to-Follow and RHCR report.

UNITS. A POGEMA agent moves one cell per timestep; a FLOWRRA robot moves half a
cell per step (base_speed 0.5), so 512 timesteps are 1,024 steps. In a lifelong
test every robot always has work, so this conversion is fair (a faster robot
really does finish more tasks); with a fixed order book it would not be.

WHAT DIFFERS, STATED PLAINLY. POGEMA and PIBT move agents cell to cell and only
forbid two in one cell or a swap. FLOWRRA moves in half-cell steps, can collide,
and recovers; with --braking on (default) it also slows near other robots (the
approach-aware bubble), a safety margin the MAPF methods do not carry. Collisions
are reported beside throughput.

ARMS (deploy_eval's syntax): label=checkpoint[:noroute], RULES, and PIBT.
    :noroute   the env's route OFF for this arm: the network drives the floor,
               steered by its Learner at the floor set in the config
               (training.learner.router.scale_end), read once at start-up
    RULES      the shortest-path driver under the same env rules
    PIBT       our own lifelong PIBT (Okumura et al. 2022) on the same graph and
               the same goal sequences: one cell per timestep, vertex and swap
               conflicts forbidden, no safety margin. An in-house yardstick
               beside the published numbers, not a replacement for them.

Every network arm is greedy by default (--epsilon 0), as benchmarks report.
Writes one row per (arm, agents, seed) and prints the means.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import time
from collections import deque
from typing import Any, Dict, List

import numpy as np

import deploy_eval as DE
from config_warehouse import CONFIG

FIELDS = ["arm", "agents", "seed", "timesteps", "steps", "goals", "throughput", "collisions",
          "l3_fired", "guard_stops", "vertex_conflicts", "swap_conflicts", "runtime_s",
          "ms_per_step", "braking", "safety", "router_scale"]


# ----------------------------------------------------------------------------- instances
def load_instance(cache, path: str, protocol: str = "pogema") -> Dict[str, Any]:
    from node_warehouse import precompute_goal_distances_compact
    e = cache.get("kiva")
    raw = json.load(open(path))
    pos = e["pos_dict"]
    xyz = lambda nid: np.array([pos[nid]["X"], pos[nid]["Y"], pos[nid]["Z"]], dtype=np.float32)
    starts = [str(v) for v in raw["starts"]]
    raw_seqs = [[str(g) for g in q] for q in raw["goals"]]

    # Optional: True Kiva Pod <-> Perimeter Workstation alternating cycle
    if protocol == "workstation":
        stations = sorted([nid for nid, p in pos.items()
                           if int(p["X"]) in (1, 44) and int(p["Y"]) % 4 == 0], key=int)
        rng = np.random.default_rng(int(raw["seed"]) + 1000 * int(raw["agents"]))
        seqs = []
        for q in raw_seqs:
            interleaved = []
            for g in q:
                interleaved.append(g)
                interleaved.append(str(rng.choice(stations)))
            seqs.append(interleaved)
    else:
        seqs = raw_seqs

    # CRITICAL FIX: Ensure ALL goals in seqs have precomputed BFS distance maps
    all_goals = sorted({g for q in seqs for g in q})
    missing = [g for g in all_goals if g not in e["goal_distance_maps"]]
    if missing:
        extra, _ = precompute_goal_distances_compact(e["G"], missing, node_index=e["node_index"])
        e["goal_distance_maps"].update(extra)

    missions = [{"id": f"a{i}", "start_node": s, "goal_node": q[0],
                 "start_pos": xyz(s), "goal_pos": xyz(q[0])}
                for i, (s, q) in enumerate(zip(starts, seqs))]
    return {"map": "kiva", "k": int(raw["agents"]), "seed": int(raw["seed"]),
            "scen": os.path.basename(path), "G": e["G"], "pos_dict": pos,
            "missions": missions, "gdm": e["goal_distance_maps"], "bank": e["bank"],
            "goal_pool": {g: xyz(g) for g in all_goals},
            "lifelong": {f"a{i}": q for i, q in enumerate(seqs)},
            "starts": starts, "seqs": seqs}


# ----------------------------------------------------------------------------- PIBT
_BFS: Dict[str, Dict[str, int]] = {}


def _dist(G, goal: str) -> Dict[str, int]:
    d = _BFS.get(goal)
    if d is None:
        d = {goal: 0}
        q = deque([goal])
        while q:
            v = q.popleft()
            for w in G.neighbors(v):
                if w not in d:
                    d[w] = d[v] + 1
                    q.append(w)
        _BFS[goal] = d
    return d


def pibt_lifelong(G, starts: List[str], seqs: List[List[str]], timesteps: int,
                  allow_chain_follow: bool = True) -> Dict[str, int]:
    """Lifelong PIBT. Set allow_chain_follow=False for PIBT_SAFE (enforces the
    1-hop physical safety gap where a robot cannot step into a cell that is
    still occupied at the start of the step)."""
    n = len(starts)
    nbrs = {v: sorted(G.neighbors(v)) for v in G.nodes}
    cur = list(starts)
    gi = [0] * n
    goal = [seqs[i][0] for i in range(n)]
    pri = [i / (10.0 * n) for i in range(n)]
    reached = vconf = sconf = 0
    for _ in range(timesteps):
        occ = {v: i for i, v in enumerate(cur)}
        nxt: List[Any] = [None] * n
        claimed: Dict[str, int] = {}

        def push(i: int, blocker: Any) -> bool:
            h = _dist(G, goal[i])
            for v in sorted(nbrs[cur[i]] + [cur[i]], key=lambda u: (h.get(u, 1 << 30), u)):
                if v in claimed:
                    continue
                if blocker is not None and v == cur[blocker]:
                    continue
                k = occ.get(v)
                if k is not None and k != i:
                    if not allow_chain_follow:
                        continue  # 1-hop physical gap: wait for cell to clear
                    if nxt[k] is None:
                        claimed[v] = i
                        if not push(k, i):
                            del claimed[v]
                            continue
                        nxt[i] = v
                        return True
                    else:
                        continue
                claimed[v] = i
                nxt[i] = v
                return True
            claimed[cur[i]] = i
            nxt[i] = cur[i]
            return False

        for i in sorted(range(n), key=lambda a: -pri[a]):
            if nxt[i] is None:
                push(i, None)
        new = [nxt[i] if nxt[i] is not None else cur[i] for i in range(n)]
        if len(set(new)) < n:
            vconf += 1
        old_at = {v: i for i, v in enumerate(cur)}
        for i in range(n):
            j = old_at.get(new[i])
            if j is not None and j != i and new[j] == cur[i]:
                sconf += 1
        cur = new
        for i in range(n):
            if cur[i] == goal[i]:
                reached += 1
                gi[i] += 1
                goal[i] = seqs[i][min(gi[i], len(seqs[i]) - 1)]
                pri[i] = i / (10.0 * n)
            else:
                pri[i] += 1.0
    return {"goals": reached, "vertex_conflicts": vconf, "swap_conflicts": sconf // 2}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--arms", required=True,
                    help="comma-separated label=checkpoint[:noroute], RULES, PIBT")
    ap.add_argument("--instances", default="kiva_instances")
    ap.add_argument("--agents", default="32,64,96,128,160,192")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--timesteps", type=int, default=512)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v4")
    ap.add_argument("--braking", choices=["on", "off"], default="on",
                    help="on: our approach-aware bubble (the product); off: MAPF's rules")
    ap.add_argument("--safety", choices=["config", "on", "off"], default="on",
                    help="deploy_eval's safety switches (guard, bubbles by kind, hops, route by default)")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--out", default="kiva_eval.csv")
    args = ap.parse_args()

    speed = float(CONFIG["warehouse"]["base_speed"])
    steps = int(round(args.timesteps / speed))
    # Conditions: the clean lifelong floor of the protocol.
    CONFIG.setdefault("stream", {})["enabled"] = False
    CONFIG.setdefault("errors", {})["enabled"] = False
    CONFIG.setdefault("obstacles", {})["enabled"] = False
    CONFIG["warehouse"]["braking"] = (args.braking == "on")
    CONFIG.setdefault("training", {})["max_steps_per_episode"] = steps
    DE.apply_safety_mode(args.safety)
    if args.braking == "off":
        CONFIG.setdefault("safety", {})["bubbles_by_kind"] = False

    from main_runner_warehouse import MapCache
    from benchmark_flowrra import build_agent
    cache = MapCache(args.maps_dir, args.scens_dir)
    agents_list = [int(a) for a in args.agents.split(",") if a.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    paths = [(n, s, os.path.join(args.instances, f"kiva_n{n}_s{s}.json"))
             for n in agents_list for s in seeds]
    missing = [p for _, _, p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"missing instances (run kiva_instances.py): {missing[:3]} ...")

    scale = float(((CONFIG.get("training", {}) or {}).get("learner", {}) or {})
                  .get("router", {}).get("scale_end", 0.0))
    arms = []
    first = load_instance(cache, paths[0][2])
    for spec in [a.strip() for a in args.arms.split(",") if a.strip()]:
        if spec.upper() in ("RULES", "PIBT", "PIBT_SAFE"):
            arms.append((spec.upper(), spec.upper(), False))
            continue
        label, path = spec.split("=", 1)
        noroute = path.endswith(":noroute")
        if noroute:
            path = path[: -len(":noroute")]
        agent = build_agent(first["G"], first["pos_dict"], first["missions"], first["goal_pool"], path)
        arms.append((label, agent, noroute))
        print(f"[kiva] arm {label}: {path}" + (" | route OFF (the network drives)" if noroute else ""))
    print(f"[kiva] {len(paths)} instances x {len(arms)} arms | {args.timesteps} timesteps = {steps} "
          f"steps | braking {args.braking} | safety {args.safety} | router floor {scale} | "
          f"epsilon {args.epsilon}", flush=True)

    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()
    results: Dict[tuple, List[float]] = {}
    for n, s, path in paths:
        inst = load_instance(cache, path)
        for label, agent, noroute in arms:
            t0 = time.time()
            if agent in ("PIBT", "PIBT_SAFE"):
                r = pibt_lifelong(inst["G"], inst["starts"], inst["seqs"], args.timesteps,
                      allow_chain_follow=(agent == "PIBT"))
                row = {"goals": r["goals"], "collisions": 0, "l3_fired": "", "guard_stops": "",
                       "vertex_conflicts": r["vertex_conflicts"], "swap_conflicts": r["swap_conflicts"],
                       "steps": args.timesteps, "braking": "", "safety": "", "router_scale": ""}
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    d = DE.run_one(inst, label, agent, args.epsilon, steps, route=False,
                                   env_route_off=noroute)
                row = {"goals": int(d.get("lifelong_goals_reached", 0)),
                       "collisions": d.get("collisions", ""), "l3_fired": d.get("ladder_l3_fired", ""),
                       "guard_stops": d.get("guard_stops", ""), "vertex_conflicts": "",
                       "swap_conflicts": "", "steps": d.get("steps_run", steps),
                       "braking": args.braking, "safety": args.safety,
                       "router_scale": scale if noroute else ""}
            dt = time.time() - t0
            tp = row["goals"] / args.timesteps
            row.update(arm=label, agents=n, seed=s, timesteps=args.timesteps,
                       throughput=round(tp, 4), runtime_s=round(dt, 1),
                       ms_per_step=round(1000 * dt / max(1, int(row["steps"] or 1)), 1))
            w.writerow(row)
            fh.flush()
            results.setdefault((label, n), []).append(tp)
            print(f"[kiva] {label:<10} {n:>3} agents seed {s}: {row['goals']:>5} goals | "
                  f"throughput {tp:.3f}/timestep | collisions {row['collisions']} | "
                  f"L3 {row['l3_fired']} | {dt:.0f}s", flush=True)
    fh.close()
    print("\n=== mean throughput (goals per timestep), by arm and agents ===")
    for (label, n), v in sorted(results.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"  {label:<10} {n:>3} agents: {np.mean(v):.3f} +- {np.std(v):.3f} over {len(v)} seeds")


if __name__ == "__main__":
    main()
