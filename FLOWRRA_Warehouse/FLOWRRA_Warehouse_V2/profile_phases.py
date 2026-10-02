"""
profile_phases.py

What each phase costs per step, and where the time actually goes.

Runs on a SYNTHETIC warehouse, so it needs no maps or scenarios -- unlike
profile_step.py, which profiles a real instance. Use this to compare
configurations; use profile_step.py to profile a real run.

TWO QUESTIONS IT ANSWERS

1. WHAT DO THE NEW FEATURES COST? Every phase is behind a flag, so each can be
   timed against the baseline it replaces. The conv encoder in particular
   changes the shape of the work: a 3D convolution per fleet per forward pass
   instead of one dense matmul.

2. IS THE BOTTLENECK THE WORLD OR THE NETWORK? The 2026-09-13 profile put
   get_local_affordance and sense_6_axis_rays at ~97% of step time, with NO
   torch entry in the top thirty. That was measured with the flat encoder. The
   conv encoder does materially more GPU work, so the answer may have moved.

WHY MOST OF THE WORLD CANNOT SIMPLY MOVE TO THE GPU. The per-step cost is
dominated by GRAPH TRAVERSAL -- proximity BFS, the structure-mask BFS, ray
walks, intended-path descent. Sequential pointer-chasing with data-dependent
branches is the worst case for a GPU. What CAN batch is the density stamping,
which is the shared-global-field refactor still on the list.

The graph itself is built once and cached, including _structure_mask and the
kernel neighbourhoods. Nothing rebuilds it per step.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import datetime as _dt
import io
import os
import platform
import pstats
import random
import time
from typing import Dict, Tuple

import numpy as np
import networkx as nx

from config_warehouse import CONFIG


CONFIGS: Dict[str, Dict] = {
    "baseline": {},
    "phase1": {
        "proximity.adjacency_metric": "graph",
        "density.ray_transform": "smooth",
    },
    "phase2": {
        "proximity.adjacency_metric": "graph",
        "density.ray_transform": "smooth",
        "density.output_mode": "channels",
        "gnn.encoder": "conv",
        "waiting.enabled": True,
    },
    "phase3": {
        "proximity.adjacency_metric": "graph",
        "density.ray_transform": "smooth",
        "density.output_mode": "channels",
        "gnn.encoder": "conv",
        "waiting.enabled": True,
        "gnn.edge_feature_dim": 4,
    },
    "phase3+obstacles": {
        "proximity.adjacency_metric": "graph",
        "density.ray_transform": "smooth",
        "density.output_mode": "channels",
        "gnn.encoder": "conv",
        "waiting.enabled": True,
        "gnn.edge_feature_dim": 4,
        "obstacles.enabled": True,
        "obstacles.n_humans": 4,
        "obstacles.n_debris": 4,
    },
}

DEFAULTS = {
    "proximity.adjacency_metric": "manhattan",
    "density.ray_transform": "clip25",
    "density.output_mode": "affordance",
    "gnn.encoder": "flat",
    "gnn.edge_feature_dim": 0,
    "waiting.enabled": False,
    "obstacles.enabled": False,
    "obstacles.n_humans": 0,
    "obstacles.n_debris": 0,
}


def apply(overrides: Dict) -> None:
    merged = dict(DEFAULTS)
    merged.update(overrides)
    for dotted, value in merged.items():
        section, key = dotted.split(".")
        CONFIG[section][key] = value


def build(width: int, aisles: int, n_fleets: int, seed: int = 0):
    from node_warehouse import precompute_goal_distances
    G = nx.Graph()
    pos = {}
    ys = list(range(0, aisles * 2, 2))
    for y in ys:
        for x in range(width):
            nid = f"n_{x}_{y}"
            G.add_node(nid)
            pos[nid] = (float(x), float(y), 0.0)
        for x in range(width - 1):
            G.add_edge(f"n_{x}_{y}", f"n_{x+1}_{y}")
    for x in (0, width // 2, width - 1):
        for y in range(ys[0], ys[-1] + 1):
            nid = f"n_{x}_{y}"
            if nid not in G:
                G.add_node(nid)
                pos[nid] = (float(x), float(y), 0.0)
        for y in range(ys[0], ys[-1]):
            G.add_edge(f"n_{x}_{y}", f"n_{x}_{y+1}")

    grid = {(int(a), int(b), int(c)): k for k, (a, b, c) in pos.items()}
    rng = np.random.default_rng(seed)
    nodes = sorted(G.nodes())
    missions, gp = [], {}
    for i in range(n_fleets):
        s = nodes[int(rng.integers(0, len(nodes)))]
        g = nodes[int(rng.integers(0, len(nodes)))]
        missions.append({
            "id": f"f{i}", "start_node": s, "goal_node": g,
            "start_pos": np.array(pos[s], dtype=np.float32),
            "goal_pos": np.array(pos[g], dtype=np.float32)})
        gp[g] = np.array(pos[g], dtype=np.float32)
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in gp])
    return G, grid, missions, gdm, gp


def make_env(n_fleets: int, width: int, aisles: int, seed: int = 7):
    import torch
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    G, grid, missions, gdm, gp = build(width, aisles, n_fleets)
    env = FLOWRRA(G, grid, missions, mode="training", goal_distance_maps=gdm,
                  shared_pool_mode=True, goal_pool=gp)
    n0 = env.nodes[0]
    base = len(n0.get_state_vector(env.nodes))
    rd = CONFIG["reward_decomposition"]
    env.gnn = GNNAgent(
        node_feature_dim=base + env.density.output_dim,
        edge_feature_dim=CONFIG["gnn"]["edge_feature_dim"],
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
        encoder_mode=CONFIG["gnn"]["encoder"],
        base_feature_dim=base,
        density_diamond_mask=env.density._diamond_mask,
    )
    return env, base


def split_cost(env, steps: int) -> Dict[str, float]:
    """
    Where a step's time goes, as a fraction: WORLD against NETWORK.

    _perceive covers rays, the density field and the state vector -- numpy, CPU,
    and mostly graph traversal that does not vectorise. choose_actions covers the
    forward pass, which is the part a GPU accelerates.

    On 2026-09-13 no torch entry appeared in the top thirty at all. The conv
    encoder changes that, so the split is measured per config rather than
    assumed once and quoted forever.
    """
    pr = cProfile.Profile()
    pr.enable()
    timed(env, steps)
    pr.disable()
    st = pstats.Stats(pr)
    want = {"_perceive": "world", "choose_actions": "network",
            "_build_edge_features": "edges", "step": "total"}
    out = {"world": 0.0, "network": 0.0, "edges": 0.0, "total": 0.0}
    for (fn, _line, name), (_cc, _nc, _tt, ct, _cal) in st.stats.items():
        if name in want and "warehouse" in fn:
            k = want[name]
            out[k] = max(out[k], ct)
    base = max(1e-9, out.pop("total"))
    return {k: v / base for k, v in out.items()}


def health(env) -> Dict[str, float]:
    """Counters that should sit near zero, plus the ones saying a feature fired."""
    e = env.get_error_statistics()
    return {
        "collisions": env.loop.total_collisions,
        "completion": round(len(env.claimed_goals) / max(1, len(env.goal_pool)), 4),
        "waits_started": e.get("waits_started", 0),
        "mutual_waits": e.get("mutual_waits", 0),
        "waits_capped": e.get("waits_capped", 0),
        "proj_fallbacks": e.get("projection_fallbacks", 0),
        "kernel_fallbacks": e.get("kernel_manhattan_fallbacks", 0),
        "memory_offgrid_rejected": e.get("memory_offgrid_rejected", 0),
        "ray_origin_recovered": e.get("ray_origin_recovered", 0),
        "override_rate": round(e.get("action_override_rate", 0.0), 4),
        "obstacles": e.get("static_obstacles", 0),
    }


def timed(env, steps: int) -> Tuple[float, int]:
    import torch
    t0 = time.perf_counter()
    done = 0
    for _ in range(steps):
        env.step(episode_step=1, total_episodes=1)
        done += 1
        if env.is_episode_over():
            break
    if torch.cuda.is_available():
        torch.cuda.synchronize()      # GPU work is async; do not time a queue
    return time.perf_counter() - t0, done


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--fleets", type=int, default=40)
    ap.add_argument("--width", type=int, default=30)
    ap.add_argument("--aisles", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--profile", default="phase3",
                    help="which config to break down with cProfile")
    ap.add_argument("--rows", type=int, default=18)
    ap.add_argument("--out", default="profile_results",
                    help="directory for the CSV log; one timestamped file per "
                         "run, nothing overwritten")
    ap.add_argument("--tag", default="",
                    help="short label written into every row, so runs stay "
                         "identifiable after the terminal has scrolled away")
    args = ap.parse_args()

    import torch
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          f"  threads={torch.get_num_threads()}")
    print(f"{args.fleets} fleets on a {args.width}x{args.aisles*2} warehouse\n")

    os.makedirs(args.out, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    tag = args.tag or "untagged"
    csv_path = os.path.join(args.out, f"phases_{tag}_{stamp}.csv")

    fields = ["run_id", "tag", "config", "ms_per_step", "vs_base", "state_dim",
              "adj_shape", "world_frac", "network_frac", "edges_frac",
              "collisions", "completion", "waits_started", "mutual_waits",
              "waits_capped", "proj_fallbacks", "kernel_fallbacks",
              "memory_offgrid_rejected", "ray_origin_recovered",
              "override_rate", "obstacles", "steps", "fleets"]
    fh = open(csv_path, "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields)
    for k, v in (("run_id", stamp), ("tag", tag), ("fleets", args.fleets),
                 ("width", args.width), ("aisles", args.aisles),
                 ("warmup", args.warmup), ("steps", args.steps),
                 ("torch", torch.__version__),
                 ("cuda", torch.cuda.is_available()),
                 ("threads", torch.get_num_threads()),
                 ("python", platform.python_version())):
        fh.write(f"# {k},{v}\n")
    writer.writeheader()
    print(f"logging to {csv_path}\n")

    print(f"{'config':<20}{'ms/step':>10}{'vs base':>9}{'world':>8}{'net':>7}"
          f"{'coll':>7}{'done':>7}{'waits':>7}")
    base_ms = None
    for name, over in CONFIGS.items():
        apply(over)
        env, base = make_env(args.fleets, args.width, args.aisles)
        timed(env, args.warmup)
        dt, n = timed(env, args.steps)
        ms = dt / max(1, n) * 1000
        if base_ms is None:
            base_ms = ms

        # A SECOND, FRESH environment for the cost split. Re-profiling the same
        # one would measure a later episode phase -- more fleets parked, more
        # collapse memory -- so the split would not be comparable across configs.
        env2, _ = make_env(args.fleets, args.width, args.aisles)
        timed(env2, args.warmup)
        frac = split_cost(env2, max(4, args.steps // 5))

        h = health(env)
        edim = CONFIG["gnn"]["edge_feature_dim"]
        adj = "[N,N]" if edim == 0 else f"[N,N,{1+edim}]"
        print(f"{name:<20}{ms:>10.1f}{ms/base_ms:>8.2f}x"
              f"{frac['world']:>8.0%}{frac['network']:>7.0%}"
              f"{h['collisions']:>7}{h['completion']:>7.2f}"
              f"{h['waits_started']:>7}")
        writer.writerow({
            "run_id": stamp, "tag": tag, "config": name,
            "ms_per_step": round(ms, 2), "vs_base": round(ms / base_ms, 3),
            "state_dim": base + env.density.output_dim, "adj_shape": adj,
            "world_frac": round(frac["world"], 4),
            "network_frac": round(frac["network"], 4),
            "edges_frac": round(frac.get("edges", 0.0), 4),
            "steps": n, "fleets": len(env.nodes), **h})
        fh.flush()
    fh.close()

    idx = os.path.join(args.out, "phases_index.csv")
    new = not os.path.exists(idx)
    with open(idx, "a", newline="") as ih:
        w = csv.writer(ih)
        if new:
            w.writerow(["run_id", "tag", "fleets", "cuda", "csv"])
        w.writerow([stamp, tag, args.fleets, torch.cuda.is_available(),
                    os.path.basename(csv_path)])
    print(f"\nwrote {csv_path}")
    print(f"appended one row to {idx}")

    print(f"\nprofiling '{args.profile}' -- where the time goes")
    apply(CONFIGS[args.profile])
    env, _ = make_env(args.fleets, args.width, args.aisles)
    timed(env, args.warmup)
    pr = cProfile.Profile()
    pr.enable()
    timed(env, max(5, args.steps // 4))
    pr.disable()
    buf = io.StringIO()
    pstats.Stats(pr, stream=buf).sort_stats("cumtime").print_stats(args.rows)
    for line in buf.getvalue().splitlines():
        if "/" in line or "{" in line:
            print("   " + line.strip()[:110])

    print("\nCOLLISIONS AND COMPLETION HERE ARE NOISE, NOT RESULTS.")
    print("  Every config builds a FRESH, RANDOMLY INITIALISED network, and the")
    print("  conv encoder has a different parameter shape so it cannot share an")
    print("  init with the flat one. Two random policies wander differently.")
    print("  They are logged because a SUDDEN move -- completion at 0.00, or")
    print("  collisions an order of magnitude up -- means something crashed or")
    print("  froze, which is worth catching. Nothing finer than that.")
    print("  Only ms/step and the world/network split mean anything here.")
    print("\nREADING THIS")
    print("  density / rays on top   -> the WORLD is the cost, as it was on")
    print("     2026-09-13 (~97%, no torch in the top thirty). Most of it is")
    print("     graph traversal, which does not vectorise onto a GPU; the part")
    print("     that would is the shared-field refactor.")
    print("  torch / conv on top     -> the conv encoder has shifted the balance")
    print("     and batching the forward pass is now worth something.")
    print("  _build_edge_features    -> Phase 3 is not free; it is O(edges x k)")
    print("     per step and can be capped by lowering projection_steps.")

    apply({})


if __name__ == "__main__":
    main()