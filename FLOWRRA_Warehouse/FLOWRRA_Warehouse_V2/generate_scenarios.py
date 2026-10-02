"""
generate_scenarios.py

Generates valid start/goal scenario files for the official 3D MAPF Warehouse
maps (Wang, Veerapaneni, Wu, Li & Likhachev, ICAPS 2024).

WHY THIS IS NEEDED
The benchmark's downloadable scenario bundle is degenerate: every row pairs an
agent's start with ITSELF (startNodeId == goalNodeId, verified across
50_10_5_10_10_2, 50_10_5_20_5_2 and 100_20_10_20_10_4, all 2000 rows each). Every
agent is therefore asked to travel zero hops, so any benchmark computed on those
files measures nothing -- a full 450-run sweep on them reported
soc_lower_bound = 0.0 and completion_rate 0.999 for all methods.

The MAPS themselves are fine: connected, correctly sized, with unique integer
coordinates per node. So this regenerates only the missing half of the instance.

WHAT IT GUARANTEES
  * start != goal, and both drawn from the LARGEST connected component, so no
    pair is unreachable by construction.
  * a configurable MINIMUM hop distance. Without a floor, uniform sampling on a
    120k-node warehouse still yields some near-trivial pairs, and a benchmark
    padded with 2-hop tasks flatters every method equally while hiding the
    differences you are trying to measure.
  * reproducibility: (map, seed) fully determines the output, so anyone can
    regenerate byte-identical files.
  * the benchmark's own CSV schema and naming, so nothing downstream changes.

HONESTY NOTE FOR THE WRITE-UP
These are NOT the official scenarios. Say so explicitly: "the published scenario
bundle pairs each start with itself, so we generated random start/goal pairs on
the official maps with seeds 0-9 and a minimum separation of N hops; the
generator is released alongside." That is a defensible, reproducible protocol.
Quietly substituting your own instances while implying they are the benchmark's
is not.
"""

import os
import csv
import random
import argparse
from typing import Dict, List

import numpy as np
import pandas as pd
import networkx as nx


def load_map(nodes_csv: str, edges_csv: str):
    nodes_df = pd.read_csv(nodes_csv, index_col=False)
    edges_df = pd.read_csv(edges_csv, index_col=False)
    nodes_df["NodeId"] = nodes_df["NodeId"].astype(str).str.strip()
    edges_df["nodeFrom"] = edges_df["nodeFrom"].astype(str).str.strip()
    edges_df["nodeTo"] = edges_df["nodeTo"].astype(str).str.strip()

    G = nx.Graph()
    G.add_nodes_from(nodes_df["NodeId"])
    G.add_edges_from(zip(edges_df["nodeFrom"], edges_df["nodeTo"]))
    return G


def build_goal_bank(G, bank_size: int, map_name: str):
    """
    A FIXED set of candidate goal nodes for one map, identical across every seed.

    WHY THIS EXISTS. Training on a fresh scenario every episode is the single
    biggest lever available (400 episodes on one STRESS_TEST file is 400 episodes
    on one instance, and the policy memorised its 25 journeys). But it has a
    cost: precompute_goal_distances() runs one BFS per goal, and if every episode
    invents brand-new goals that is |goals| * O(|E|) of fresh BFS per episode --
    on a 120k-node warehouse, enough to dominate training time.

    Pinning goals to a bank makes the cost one-time per MAP instead of per
    episode. The bank's BFS maps are computed once, cached, and reused forever;
    each episode then samples starts freely (unrestricted, so spawn geometry is
    genuinely fresh) and draws its goals from the bank. Scenario diversity is
    preserved where it matters -- the policy still never sees the same
    start/goal pairing twice -- while the expensive half is amortised.

    Seeded on the map NAME, not the scenario seed, so every seed for a map draws
    from the same bank and the cache stays valid.
    """
    comp = max(nx.connected_components(G), key=len)
    pool = sorted(comp)
    rng = random.Random(f"goalbank::{map_name}")
    if bank_size >= len(pool):
        return pool
    return sorted(rng.sample(pool, bank_size))


def sample_pairs(G, n_agents: int, seed: int, min_hops: int, max_tries_mult: int = 60,
                 goal_bank=None, max_hops: int = 0):
    """
    Samples n_agents (start, goal) pairs with distinct starts, distinct goals,
    and a hop separation inside [min_hops, max_hops].

    Starts and goals are kept mutually exclusive across agents because two agents
    sharing a start is not a valid MAPF instance (they would begin in collision),
    and two sharing a goal is unsatisfiable when agents stay at their targets.

    WHY max_hops MATTERS. A minimum is a floor and cannot produce short journeys.
    On these warehouses two uniformly sampled nodes are almost never close --
    measured p50 separations run from 40 hops on the smallest map to 206 on the
    largest -- so varying only the floor gives a scenario set that is long
    almost everywhere and leaves the policy with essentially no short-journey
    coverage. Getting genuinely MIXED lengths needs a ceiling.

    HOW. With a ceiling, rejection sampling would be hopeless: the fraction of
    random pairs within (say) 20 hops of each other on a 120k-node graph is
    tiny, so almost every draw would be discarded. Instead a single BFS runs
    outward from the start, stopping at max_hops, and the goal is drawn from the
    nodes it reached at depth >= min_hops. That is one bounded traversal per
    agent and it satisfies both constraints by construction rather than by
    retrying.
    """
    rng = random.Random(seed)
    comp = max(nx.connected_components(G), key=len)
    pool = sorted(comp)
    if len(pool) < 2 * n_agents:
        raise ValueError(f"largest component has {len(pool)} nodes; "
                         f"need >= {2*n_agents} for {n_agents} agents")

    # Starts come from the whole component; goals from the bank when one is
    # given (see build_goal_bank). The bank must be a subset of the same
    # component or pairs would be unreachable by construction.
    goal_source = [g for g in goal_bank if g in comp] if goal_bank else pool
    if len(goal_source) < n_agents:
        raise ValueError(f"goal bank has {len(goal_source)} usable nodes; "
                         f"need >= {n_agents}. Raise --goal-bank.")
    goal_set = set(goal_source)

    def candidates_from(s):
        """
        BFS outward from s, stopping at max_hops, returning eligible goals at
        depth >= min_hops. Unbounded (max_hops <= 0) keeps the old cheap path:
        we only need to know the goal is FAR enough, so a bounded search that
        rules out everything nearby is sufficient and much cheaper than a full
        shortest-path call per candidate.
        """
        if max_hops <= 0:
            return None  # signal: use the far-enough rejection path below
        seen = {s}
        frontier = [s]
        out = []
        for depth in range(1, max_hops + 1):
            nxt = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v not in seen:
                        seen.add(v)
                        nxt.append(v)
                        if depth >= min_hops and v in goal_set:
                            out.append(v)
            frontier = nxt
            if not frontier:
                break
        return out

    def far_enough(s, g):
        """True iff g is NOT within (min_hops - 1) hops of s."""
        if min_hops <= 1:
            return s != g
        seen, frontier = {s}, [s]
        for _ in range(min_hops - 1):
            nxt = []
            for u in frontier:
                for v in G.neighbors(u):
                    if v not in seen:
                        if v == g:
                            return False
                        seen.add(v)
                        nxt.append(v)
            frontier = nxt
            if not frontier:
                break
        return True

    starts, goals = [], []
    used_s, used_g = set(), set()
    tries, cap = 0, n_agents * max_tries_mult
    while len(starts) < n_agents and tries < cap:
        tries += 1
        s = rng.choice(pool)
        if s in used_s:
            continue

        cands = candidates_from(s)
        if cands is not None:
            cands = [g for g in cands if g not in used_g and g != s]
            if not cands:
                continue
            g = rng.choice(cands)
        else:
            g = rng.choice(goal_source)
            if g in used_g or s == g:
                continue
            if not far_enough(s, g):
                continue

        starts.append(s); goals.append(g)
        used_s.add(s); used_g.add(g)

    if len(starts) < n_agents:
        raise ValueError(f"only found {len(starts)}/{n_agents} pairs in "
                         f"[{min_hops}, {max_hops or 'inf'}] hops; widen the range "
                         f"or use a larger map")
    return starts, goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--out-dir", default="all_scens_fixed")
    ap.add_argument("--agents", type=int, default=200,
                    help="agents per file; the harness takes the first k, so make "
                         "this >= your largest k")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--min-hops", type=int, default=10,
                    help="floor on start-goal separation. Ignored if "
                         "--min-hops-range is given.")
    ap.add_argument("--min-hops-range", default="",
                    help="MIXED JOURNEY LENGTHS, as 'LO,HI'. Each seed gets its own "
                         "min_hops drawn evenly across [LO, HI] instead of every "
                         "scenario using one fixed floor. A policy trained only at "
                         "min_hops=10 sees one narrow band of journey lengths and has "
                         "no reason to generalise beyond it; the measured failure "
                         "profile was flat across journey length (0.263 success at "
                         "<=25 hops/agent, 0.250 at 25-40), so length diversity costs "
                         "nothing and covers the range the benchmark actually spans. "
                         "Example: --min-hops-range 4,40")
    ap.add_argument("--max-hops", type=int, default=0,
                    help="CEILING on start-goal separation; 0 = unbounded. A floor "
                         "alone cannot produce short journeys: on these warehouses "
                         "two uniformly sampled nodes are almost never close (p50 "
                         "separation runs 40-206 hops across the official maps), so "
                         "varying --min-hops leaves you with long journeys almost "
                         "everywhere and no short-journey coverage at all. Pair this "
                         "with --min-hops-range to get a genuine spread. Sampling "
                         "uses a bounded BFS from each start rather than rejection, "
                         "so a tight ceiling costs nothing.")
    ap.add_argument("--max-hops-range", default="",
                    help="as 'LO,HI'; per-seed ceiling spread across the range, the "
                         "mirror of --min-hops-range. Example: --max-hops-range 25,250")
    ap.add_argument("--goal-bank", type=int, default=400,
                    help="size of the fixed per-map goal bank (see build_goal_bank). "
                         "Must be >= the largest agent count you intend to train on; "
                         "bigger gives more goal diversity at a one-time BFS cost of "
                         "one map traversal per bank node. 0 disables the bank and "
                         "draws goals from the whole graph, which makes per-episode "
                         "BFS caching impossible.")
    ap.add_argument("--maps", default="", help="comma-separated map names; blank = all")
    args = ap.parse_args()

    hops_lo = hops_hi = args.min_hops
    if args.min_hops_range:
        hops_lo, hops_hi = (int(v) for v in args.min_hops_range.split(","))

    max_lo = max_hi = args.max_hops
    if args.max_hops_range:
        max_lo, max_hi = (int(v) for v in args.max_hops_range.split(","))

    names = sorted({
        f[:-len("_Nodes.csv")]
        for f in os.listdir(args.maps_dir) if f.endswith("_Nodes.csv")
    })
    if args.maps:
        keep = {m.strip() for m in args.maps.split(",") if m.strip()}
        names = [n for n in names if n in keep]

    hops_desc = (f"min_hops {hops_lo}-{hops_hi} (spread across seeds)"
                 if hops_hi > hops_lo else f"min_hops={hops_lo}")
    if max_hi:
        hops_desc += (f", max_hops {max_lo}-{max_hi} (spread)"
                      if max_hi > max_lo else f", max_hops={max_lo}")
    print(f"[Gen] {len(names)} maps x {args.seeds} seeds, {args.agents} agents each, "
          f"{hops_desc}, goal bank {args.goal_bank or 'disabled'}\n")

    for name in names:
        nodes_csv = os.path.join(args.maps_dir, f"{name}_Nodes.csv")
        edges_csv = os.path.join(args.maps_dir, f"{name}_Edges.csv")
        if not os.path.exists(edges_csv):
            print(f"  [skip] {name}: no _Edges.csv")
            continue

        G = load_map(nodes_csv, edges_csv)
        out_dir = os.path.join(args.out_dir, name)
        os.makedirs(out_dir, exist_ok=True)

        bank = build_goal_bank(G, args.goal_bank, name) if args.goal_bank else None
        if bank:
            # Written next to the scenarios so the trainer can precompute and
            # cache one BFS map per bank node, once per map, and then reuse them
            # for every episode on that map.
            with open(os.path.join(out_dir, f"{name}_GoalBank.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["goalNodeId"])
                for g in bank:
                    w.writerow([g])

        stats = []
        for seed in range(args.seeds):
            # Mixed journey lengths: spread min_hops evenly across the requested
            # range so the scenario set spans short and long journeys rather than
            # clustering at one floor.
            if args.seeds > 1 and hops_hi > hops_lo:
                mh = int(round(hops_lo + (hops_hi - hops_lo) * seed / (args.seeds - 1)))
            else:
                mh = hops_lo
            if args.seeds > 1 and max_hi > max_lo:
                xh = int(round(max_lo + (max_hi - max_lo) * seed / (args.seeds - 1)))
            else:
                xh = max_lo
            if xh and xh <= mh:
                xh = mh + 1  # keep the window non-empty
            try:
                starts, goals = sample_pairs(G, args.agents, seed, mh,
                                             goal_bank=bank, max_hops=xh)
            except ValueError as exc:
                print(f"  [warn] {name} seed{seed}: {exc}")
                continue

            path = os.path.join(out_dir, f"{name}_StartGoalLocations_Seed{seed}.csv")
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["agentId", "startNodeId", "goalNodeId"])
                for i, (s, g) in enumerate(zip(starts, goals), start=1):
                    w.writerow([i, s, g])

            # Verify a sample rather than trusting the sampler.
            sample = random.Random(seed).sample(range(len(starts)), min(20, len(starts)))
            d = [nx.shortest_path_length(G, starts[i], goals[i]) for i in sample]
            stats.append(np.mean(d))

        print(f"  {name:<22} {G.number_of_nodes():>7} nodes | {len(stats)} scenarios | "
              f"mean hops {np.mean(stats):.1f}" if stats else f"  {name}: none written")

    print(f"\n[Gen] Written to {args.out_dir}/. Point the harness at it:")
    print(f"      python benchmark_all.py --scens-dir {args.out_dir} --agents 25")


if __name__ == "__main__":
    main()