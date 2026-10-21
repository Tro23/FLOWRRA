"""
correlate_ray_repulsion.py -- are rays and the density field measuring the same thing?

THE QUESTION THIS DECIDES
-------------------------
The proposal is a coherence-tempered affordance field:

    A = mask / (1 + R) ** beta ,    beta = 1 + lambda * (1 - C)

where R is local repulsion from the density field and C is a coherence signal
taken from ray clearance. Sharper discrimination when congested, flatter when
clear.

That is only worth a retrain if R and C are INDEPENDENT. If they are not, then
beta(C) is a nonlinear re-weighting of R by something that is mostly R again,
and the whole construction reduces to a different response curve on one
quantity -- which is a thing you can get for free by changing the curve.

A blocked ray usually means a nearby peer or a nearby wall. A nearby peer is
also high R. So the null hypothesis here is "they are the same signal wearing
two hats", and it is a serious null.

WHAT IT MEASURES
----------------
Per fleet-step, over real episodes on a real map:

    C_ray  = mean of the 6 normalised ray distances    (high = clear sightlines)
    R_mean = mean repulsion over the observer's unmasked local cells
    R_max  = max repulsion in the observer's own cell and its graph neighbours

Reports Pearson and Spearman between C_ray and each R, overall and split by
occupancy band, plus the partial correlation of C_ray with R_mean controlling
for peer count -- because if both are really just "how many fleets are near me",
that shows up as the partial collapsing toward zero.

HOW TO READ IT
--------------
    |rho| > 0.8   -> rays and repulsion are one signal. Do not build the
                     coherence term. Change the response curve instead; it is a
                     config number, not an architecture.
    0.4 - 0.8     -> partially redundant. Worth building, but the honest claim
                     is "sharper response under congestion", not "a second
                     sensory channel".
    |rho| < 0.4   -> genuinely independent. The coherence term is carrying
                     information the field cannot express, and graph-rays are
                     worth building as the source of C.

COSTS NOTHING. No training, frozen checkpoint, greedy actions. Run it alongside
the lesion harness on the same afternoon.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from typing import Dict, List

import numpy as np

from config_warehouse import CONFIG


def _rank(xs: List[float]) -> List[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float("nan")


def _spearman(a: List[float], b: List[float]) -> float:
    return _pearson(_rank(a), _rank(b))


def _partial(a: List[float], b: List[float], z: List[float]) -> float:
    """Correlation of a and b with z partialled out."""
    raz, rbz, rab = _pearson(a, z), _pearson(b, z), _pearson(a, b)
    den = math.sqrt(max(1e-12, (1 - raz ** 2) * (1 - rbz ** 2)))
    return (rab - raz * rbz) / den


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--resume", required=True)
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--maps", default="")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--agent-sets", default="25")
    ap.add_argument("--max-steps", type=int,
                    default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--sample-every", type=int, default=5,
                    help="sample every Nth step; consecutive steps are near-duplicates "
                         "and would inflate n without adding information")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lesion_results")
    args = ap.parse_args()

    import torch
    from main_runner_warehouse import MapCache, sample_instance
    from core_warehouse import FLOWRRA
    from agent_warehouse import GNNAgent

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    cache = MapCache(args.maps_dir, args.scens_dir)
    only = [m.strip() for m in args.maps.split(",") if m.strip()] or None
    names = cache.discover(only)
    if not names:
        raise SystemExit("no usable maps")

    rd = CONFIG["reward_decomposition"]
    rng = random.Random(args.seed)
    agent_sets = [int(v) for v in args.agent_sets.split(",") if v.strip()]

    probe = sample_instance(cache, names[0], min(agent_sets), random.Random(args.seed))
    penv = FLOWRRA(probe["G"], probe["pos_dict"], probe["missions"], mode="init",
                   goal_distance_maps=probe["gdm"], shared_pool_mode=True,
                   goal_pool=probe["goal_pool"])
    n0 = penv.nodes[0]
    base_len = len(n0.get_state_vector(penv.nodes))
    density_dim = len(penv.density.get_local_affordance(n0.current_pos, penv.nodes, set()))
    layout = n0.state_layout()
    ray_lo, ray_hi = layout["ray_distances"]

    agent = GNNAgent(
        node_feature_dim=base_len + density_dim, edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"], hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"], n_heads=CONFIG["gnn"]["num_heads"],
        reward_heads=rd["heads"], head_weights=rd["weights"],
        dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
        gamma=CONFIG["training"]["gamma"],
        buffer_capacity=CONFIG["training"]["buffer_capacity"],
        batch_size=CONFIG["training"]["batch_size"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
    )
    agent.load(args.resume)
    agent.epsilon_gaussian = lambda *a, **k: 0.0

    C: List[float] = []
    Rm: List[float] = []
    Rx: List[float] = []
    peers: List[float] = []
    occ: List[float] = []

    for ep in range(1, args.episodes + 1):
        name = rng.choice(names)
        k = rng.choice(agent_sets)
        inst = sample_instance(cache, name, k, rng)
        if inst is None:
            continue
        env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                      goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                      goal_pool=inst["goal_pool"])
        env.gnn = agent
        agent.reset_episode_state()
        occupancy = 100.0 * len(env.nodes) / max(1, len(inst["pos_dict"]))

        for t in range(args.max_steps):
            env.step(episode_step=ep, total_episodes=args.episodes)
            if t % args.sample_every == 0:
                for n in env.get_active_nodes():
                    sv = n.get_state_vector(env.nodes)
                    aff = env.density.get_local_affordance(
                        n.current_pos, env.nodes, env.frozen_nodes,
                        own_goal_pos=n.goal_pos, own_id=n.id)
                    # Affordance is 1/(1+R), so R = 1/A - 1 on unmasked cells.
                    live = aff[aff > 0]
                    if live.size == 0:
                        continue
                    r = 1.0 / live - 1.0
                    C.append(float(np.mean(sv[ray_lo:ray_hi])))
                    Rm.append(float(np.mean(r)))
                    Rx.append(float(np.max(r)))
                    peers.append(float(len(env.proximity.peers_within(n.id, 4.0))))
                    occ.append(occupancy)
            if env.is_episode_over():
                break
        print(f"  ep {ep:02d} {name:<20} k={len(env.nodes):<3} occ={occupancy:.2f}%  "
              f"n={len(C)}")

    if len(C) < 30:
        raise SystemExit(f"only {len(C)} samples; raise --episodes")

    def block(label, a, b, z=None):
        print(f"\n{label}   n={len(a)}")
        print(f"   pearson   {_pearson(a, b):+.3f}")
        print(f"   spearman  {_spearman(a, b):+.3f}")
        if z is not None:
            print(f"   partial (peer count removed)  {_partial(a, b, z):+.3f}")

    print("\n" + "=" * 72)
    print("RAY CLEARANCE  vs  LOCAL REPULSION")
    print("=" * 72)
    block("C_ray  vs  R_mean", C, Rm, peers)
    block("C_ray  vs  R_max", C, Rx, peers)

    bands = {"low (<1%)": [], "mid (1-3%)": [], "high (>3%)": []}
    for c, r, o in zip(C, Rm, occ):
        key = "low (<1%)" if o < 1 else ("mid (1-3%)" if o < 3 else "high (>3%)")
        bands[key].append((c, r))
    print("\nby occupancy band:")
    for label, pts in bands.items():
        if len(pts) < 30:
            print(f"   {label:<14} n={len(pts):<6} (too few)")
            continue
        a = [p[0] for p in pts]; b = [p[1] for p in pts]
        print(f"   {label:<14} n={len(pts):<6} pearson {_pearson(a,b):+.3f}  "
              f"spearman {_spearman(a,b):+.3f}")

    rho = abs(_spearman(C, Rm))
    print("\n" + "-" * 72)
    if rho > 0.8:
        print(f"|rho| = {rho:.3f}  -> ONE SIGNAL. Do not build the coherence term.")
        print("   Change the response curve instead: that is a config number.")
    elif rho > 0.4:
        print(f"|rho| = {rho:.3f}  -> PARTIALLY REDUNDANT. Buildable, but the honest")
        print("   claim is 'sharper response under congestion', not 'a second channel'.")
    else:
        print(f"|rho| = {rho:.3f}  -> INDEPENDENT. The coherence term carries something")
        print("   the field cannot express. Graph-rays are worth building as C.")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "ray_repulsion_correlation.json")
    with open(path, "w") as f:
        json.dump({
            "n": len(C),
            "pearson_C_Rmean": _pearson(C, Rm),
            "spearman_C_Rmean": _spearman(C, Rm),
            "partial_C_Rmean_given_peers": _partial(C, Rm, peers),
            "pearson_C_Rmax": _pearson(C, Rx),
            "spearman_C_Rmax": _spearman(C, Rx),
        }, f, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()