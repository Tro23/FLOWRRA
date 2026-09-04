"""
train_warehouse.py

Curriculum trainer for FLOWRRA. Replaces main_runner_warehouse.py's single-map,
single-scenario training loop.

WHAT WAS WRONG WITH THE OLD LOOP
It loaded ONE map and ONE scenario file (50_5_5_10_5_10 + STRESS_TEST.csv) and
reused them for all 400 episodes. That is not 400 episodes of learning to route;
it is 400 episodes on a single instance with 25 fixed start/goal pairs. The
policy memorised those 25 journeys. Evidence: the resulting checkpoint scores
23/25 on that exact instance, byte-identical across repeated runs, but 2/10 on
FRESH scenarios on THE SAME MAP -- which rules out topology transfer as the
explanation and leaves scenario memorisation as the cause.

WHAT THIS DOES INSTEAD, in descending order of expected effect:

  1. FRESH SCENARIO EVERY EPISODE. The single biggest lever, and the one thing
     400 episodes never varied.
  2. MULTI-MAP SAMPLING. Nearly free once (1) is in: sample (map, seed)
     together rather than sampling a seed within one fixed map.
  3. VARIABLE FLEET COUNT. Draws k per episode from a range. This is only
     possible because GraphReplayBuffer.sample() now pads to the batch max
     instead of discarding every transition whose fleet count is not the modal
     one -- with the old sampler, 26 distinct values of k meant training on a
     rotating 1/26 slice of the buffer.
  4. MIXED JOURNEY LENGTHS, via scenario files generated across a --min-hops
     range (see generate_scenarios.py).

WHY IT IS STILL FAST
Two caches. Graphs are loaded once per map and reused. The expensive part --
one BFS per goal for the distance maps -- is computed once per map over that
map's fixed GOAL BANK and reused for every episode, instead of being recomputed
whenever the scenario changes. Without the bank, per-episode goal randomisation
costs |goals| * O(|E|) of fresh BFS per episode.

Usage:
    python generate_scenarios.py --maps-dir all_maps --out-dir all_scens_fixed \
        --agents 60 --seeds 10 --min-hops-range 4,40 --goal-bank 400

    python train_warehouse.py --maps-dir all_maps --scens-dir all_scens_fixed \
        --episodes 3000 --agents-min 25 --agents-max 50
"""

import argparse
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch

from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from node_warehouse import precompute_goal_distances_compact
from config_warehouse import CONFIG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("Curriculum")


# =============================================================================
# MAP CACHE
# =============================================================================

class MapCache:
    """
    Loads each map's graph, coordinates and goal-bank BFS maps exactly once.

    On a 120k-node warehouse the graph build is seconds and the goal-bank BFS is
    minutes. Doing either per episode would make a multi-thousand-episode run
    impossible; doing both once per map makes map sampling essentially free
    after the first visit.
    """

    def __init__(self, maps_dir: str, scens_dir: str):
        self.maps_dir = maps_dir
        self.scens_dir = scens_dir
        self._cache: Dict[str, Dict[str, Any]] = {}

    def discover(self, only: Optional[List[str]] = None) -> List[str]:
        names = sorted({
            f[: -len("_Nodes.csv")]
            for f in os.listdir(self.maps_dir) if f.endswith("_Nodes.csv")
        })
        usable = []
        for n in names:
            if only and n not in only:
                continue
            if not os.path.exists(os.path.join(self.maps_dir, f"{n}_Edges.csv")):
                continue
            if not os.path.isdir(os.path.join(self.scens_dir, n)):
                logger.warning(f"{n}: no scenario directory, skipping")
                continue
            usable.append(n)
        return usable

    def get(self, name: str) -> Dict[str, Any]:
        if name in self._cache:
            return self._cache[name]

        t0 = time.time()
        nodes_df = pd.read_csv(os.path.join(self.maps_dir, f"{name}_Nodes.csv"),
                               index_col=False)
        edges_df = pd.read_csv(os.path.join(self.maps_dir, f"{name}_Edges.csv"),
                               index_col=False)

        for c in ("X", "Y", "Z"):
            nodes_df[c] = pd.to_numeric(nodes_df[c])
        nodes_df["NodeId"] = nodes_df["NodeId"].astype(str).str.strip()
        pos_dict = nodes_df.set_index("NodeId")[["X", "Y", "Z"]].to_dict("index")

        edges_df["nodeFrom"] = edges_df["nodeFrom"].astype(str).str.strip()
        edges_df["nodeTo"] = edges_df["nodeTo"].astype(str).str.strip()
        G = nx.Graph()
        G.add_nodes_from(nodes_df["NodeId"])
        G.add_edges_from(zip(edges_df["nodeFrom"], edges_df["nodeTo"]))

        # COORDINATE COLLISION CHECK. grid_pos_dict is keyed on integer
        # (X, Y, Z), so two nodes sharing a coordinate means the later one
        # silently wins and any fleet standing on the first resolves to the
        # second -- which is exactly how a distance signal dies without any
        # error being raised (see validate_instance.py). Ladder/shaft-heavy
        # generated maps are the most likely place for this, so it is checked
        # on EVERY map at load rather than trusted.
        coord_count: Dict[Tuple[int, int, int], int] = {}
        for p in pos_dict.values():
            k = (int(round(p["X"])), int(round(p["Y"])), int(round(p["Z"])))
            coord_count[k] = coord_count.get(k, 0) + 1
        lost = sum(c - 1 for c in coord_count.values() if c > 1)
        if lost:
            logger.error(
                f"{name}: {lost} node(s) ({lost/len(pos_dict)*100:.1f}%) share an "
                f"integer coordinate with another node and are UNREACHABLE via "
                f"grid_pos_dict. Fleets on them will resolve to the wrong node and "
                f"read a wrong distance-to-goal. Fix the map before training on it."
            )

        scen_dir = os.path.join(self.scens_dir, name)
        scen_files = sorted(f for f in os.listdir(scen_dir)
                            if f.endswith(".csv") and "StartGoalLocations" in f)

        # Goal bank -> one BFS per bank node, once, reused for every episode.
        bank_path = os.path.join(scen_dir, f"{name}_GoalBank.csv")
        if os.path.exists(bank_path):
            bank = [str(v).strip() for v in
                    pd.read_csv(bank_path)["goalNodeId"].tolist()]
        else:
            # No bank (scenarios generated with --goal-bank 0, or an older
            # bundle): fall back to the union of every goal appearing in this
            # map's scenario files. Still finite and still cacheable.
            logger.warning(f"{name}: no goal bank; deriving one from scenario files.")
            seen = set()
            for f in scen_files:
                df = pd.read_csv(os.path.join(scen_dir, f), index_col=False)
                seen.update(str(v).strip() for v in df["goalNodeId"])
            bank = sorted(seen)

        bank = [g for g in bank if g in pos_dict]
        logger.info(f"{name}: {G.number_of_nodes()} nodes, {len(scen_files)} scenarios, "
                    f"goal bank {len(bank)} -- precomputing BFS maps...")
        # Array-backed, sharing one node->index dict across the whole bank. The
        # dict-per-goal form costs ~3.8 MB per goal on a 120k-node map, i.e.
        # ~1.5 GB for a 400-goal bank on ONE map -- and this cache never evicts,
        # so a 16-map curriculum would accumulate several GB and die partway
        # through a long run.
        gdm, node_index = precompute_goal_distances_compact(G, bank)

        entry = {
            "G": G,
            "pos_dict": pos_dict,
            "scen_dir": scen_dir,
            "scen_files": scen_files,
            "goal_bank": bank,
            "goal_distance_maps": gdm,
            "node_index": node_index,
        }
        self._cache[name] = entry
        logger.info(f"{name}: cached in {time.time()-t0:.1f}s")
        return entry


# =============================================================================
# EPISODE INSTANCE SAMPLING
# =============================================================================

def sample_instance(cache: MapCache, map_name: str, k: int, rng: random.Random):
    """
    Build one episode's instance: a scenario file, the first k of its rows, and
    the pool-mode mission/goal structures.

    The scenario file is drawn fresh each episode. Rows are SHUFFLED before
    taking k rather than always taking the first k, so two episodes at the same
    (map, seed, k) still differ -- with 10 seeds and a few thousand episodes you
    would otherwise see each exact instance hundreds of times, which is a milder
    version of the problem this whole file exists to fix.
    """
    entry = cache.get(map_name)
    scen_file = rng.choice(entry["scen_files"])
    df = pd.read_csv(os.path.join(entry["scen_dir"], scen_file), index_col=False)
    df["startNodeId"] = df["startNodeId"].astype(str).str.strip()
    df["goalNodeId"] = df["goalNodeId"].astype(str).str.strip()

    pos_dict = entry["pos_dict"]
    df = df[df["startNodeId"].isin(pos_dict) & df["goalNodeId"].isin(pos_dict)]
    df = df.sample(frac=1.0, random_state=rng.randrange(1 << 30)).head(k)
    if len(df) < 2:
        return None

    missions = [{
        "id": str(r["agentId"]).strip(),
        "start_node": r["startNodeId"],
        "start_pos": np.array([pos_dict[r["startNodeId"]]["X"],
                               pos_dict[r["startNodeId"]]["Y"],
                               pos_dict[r["startNodeId"]]["Z"]], dtype=np.float32),
    } for _, r in df.iterrows()]

    goal_pool = {}
    for g in df["goalNodeId"].unique():
        goal_pool[g] = np.array([pos_dict[g]["X"], pos_dict[g]["Y"],
                                 pos_dict[g]["Z"]], dtype=np.float32)

    # Only the maps for THIS episode's goals, sliced out of the map-level cache.
    # Every goal came from the bank, so every lookup hits.
    gdm = {g: entry["goal_distance_maps"][g]
           for g in goal_pool if g in entry["goal_distance_maps"]}
    if len(gdm) < len(goal_pool):
        # Goals outside the bank (older scenario bundles, or a bank that has
        # shrunk). Computed in the same array form and reusing the map's shared
        # node index, then cached so the cost is paid once.
        missing = [g for g in goal_pool if g not in gdm]
        extra, _ = precompute_goal_distances_compact(
            entry["G"], missing, node_index=entry["node_index"])
        gdm.update(extra)
        entry["goal_distance_maps"].update(extra)

    return {
        "map": map_name,
        "scen": scen_file,
        "G": entry["G"],
        "pos_dict": pos_dict,
        "missions": missions,
        "goal_pool": goal_pool,
        "gdm": gdm,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_fixed")
    ap.add_argument("--maps", default="", help="comma-separated subset; blank = all")
    ap.add_argument("--episodes", type=int, default=CONFIG["training"]["total_episodes"])
    ap.add_argument("--agents-min", type=int, default=25)
    ap.add_argument("--agents-max", type=int, default=50)
    ap.add_argument("--max-steps", type=int,
                    default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="checkpoint to warm-start from")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--target-sync", type=int, default=1000,
                    help="target-net sync period in GRADIENT STEPS (not episodes)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    only = [m.strip() for m in args.maps.split(",") if m.strip()] or None
    cache = MapCache(args.maps_dir, args.scens_dir)
    map_names = cache.discover(only)
    if not map_names:
        raise SystemExit(f"no usable maps in {args.maps_dir} / {args.scens_dir}")
    logger.info(f"{len(map_names)} maps available: {map_names}")

    # ---- agent, built once from a probe instance -------------------------
    probe = sample_instance(cache, map_names[0], args.agents_min, rng)
    probe_env = FLOWRRA(probe["G"], probe["pos_dict"], probe["missions"], mode="init",
                        goal_distance_maps=probe["gdm"], shared_pool_mode=True,
                        goal_pool=probe["goal_pool"])
    n0 = probe_env.nodes[0]
    input_dim = (len(n0.get_state_vector(probe_env.nodes))
                 + len(probe_env.density.get_local_affordance(
                     n0.current_pos, probe_env.nodes, set())))

    agent = GNNAgent(
        node_feature_dim=input_dim,
        edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        dropout=CONFIG["gnn"]["dropout"],
        lr=CONFIG["gnn"]["learning_rate"],
        gamma=CONFIG["training"]["gamma"],
        buffer_capacity=CONFIG["training"]["buffer_capacity"],
        batch_size=CONFIG["training"]["batch_size"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
    )
    if args.resume:
        agent.load(args.resume)
        logger.info(f"Warm-started from {args.resume}. NOTE: epsilon_gaussian's "
                    f"exploit-explore-exploit shape assumes exactly this -- its low "
                    f"opening epsilon is only meaningful when there is a real policy "
                    f"to exploit. From a cold init the first ~15% of episodes instead "
                    f"follow a randomly-initialised deterministic map.")
    logger.info(f"Agent ready. input_dim={input_dim} gamma={CONFIG['training']['gamma']}")

    logger.info("=" * 78)
    logger.info(f"CURRICULUM: {args.episodes} episodes | k in "
                f"[{args.agents_min}, {args.agents_max}] | {len(map_names)} maps")
    logger.info("=" * 78)

    logs = []
    global_learn_steps = 0
    t_start = time.time()

    for episode in range(1, args.episodes + 1):
        map_name = rng.choice(map_names)
        k = rng.randint(args.agents_min, args.agents_max)
        inst = sample_instance(cache, map_name, k, rng)
        if inst is None:
            continue

        env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                      goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                      goal_pool=inst["goal_pool"])
        env.gnn = agent
        agent.reset_episode_state()

        ep_reward = 0.0
        q_losses, s_losses = [], []

        for step in range(args.max_steps):
            ep_reward += env.step(episode_step=episode, total_episodes=args.episodes)

            if len(agent.memory) >= agent.batch_size:
                agent.learn(node_ids=[n.id for n in env.nodes])
                global_learn_steps += 1
                if global_learn_steps % args.target_sync == 0:
                    agent.update_target_network()
                q_losses.append(agent.last_q_loss)
                s_losses.append(agent.last_stability_loss)

            # is_episode_over(), NOT "all frozen": with error stops those differ.
            # An episode where every other fleet has parked but a pickup is still
            # open must keep running so the rescuer has clock to reach it --
            # cutting it here would delete exactly the handover transitions.
            if env.is_episode_over():
                break

        est = env.get_error_statistics()
        # Deliveries, NOT parked fleets. Those came apart once retired fleets can
        # be recalled to service a pickup: a recalled fleet leaves frozen_nodes
        # but its delivery already happened and stays in claimed_goals. Counting
        # frozen fleets would silently under-report every episode containing a
        # recall. claimed_goals is also the right denominator now that a rescuer
        # can deliver two orders.
        completed = len(env.claimed_goals)
        total_orders = max(1, len(env.goal_pool))
        unfinished = [n for n in env.nodes if n.id not in env.immobile_nodes]
        mean_remaining = float(np.mean([n.get_graph_distance_to_goal()
                                        for n in unfinished])) if unfinished else 0.0

        logger.info(
            f"Ep {episode:04d} | {map_name:<22} k={len(env.nodes):<3} | "
            f"R {ep_reward:8.1f} | done {completed}/{total_orders} | "
            f"coll {env.loop.total_collisions} | "
            f"err {est['errors_injected']} hand {est['handovers_completed']} "f"recall {est['retired_fleets_recalled']} | "
            f"eps {agent.epsilon_gaussian(episode, args.episodes):.3f} | "
            f"qL {np.mean(q_losses) if q_losses else 0:.4f} | "
            f"grad {env.get_gradient_agreement():.3f} | "
            f"left {mean_remaining:.1f}h"
        )

        logs.append({
            "episode": episode, "map": map_name, "scen": inst["scen"],
            "agents": len(env.nodes), "reward": ep_reward,
            "completed": completed, "completion_rate": completed / total_orders,
            "collisions": env.loop.total_collisions,
            "errors_injected": est["errors_injected"],
            "handovers_completed": est["handovers_completed"],
            "retired_recalled": est["retired_fleets_recalled"],
            "avg_q_loss": float(np.mean(q_losses)) if q_losses else 0.0,
            "avg_stability_loss": float(np.mean(s_losses)) if s_losses else 0.0,
            "gradient_agreement": env.get_gradient_agreement(),
            "mean_hops_remaining": mean_remaining,
            "tier1": env.recovery.get_statistics()["spatial_recoveries"],
            "tier2": env.recovery.get_statistics()["temporal_recoveries"],
            "tier3": env.recovery.get_statistics()["yield_recoveries"],
        })

        if episode % 50 == 0:
            os.makedirs(args.out, exist_ok=True)
            agent.save(os.path.join(args.out, "flowrra_curriculum.pth"))
            pd.DataFrame(logs).to_csv(
                os.path.join(args.out, "curriculum_metrics.csv"), index=False)

    os.makedirs(args.out, exist_ok=True)
    agent.save(os.path.join(args.out, "flowrra_curriculum.pth"))
    pd.DataFrame(logs).to_csv(os.path.join(args.out, "curriculum_metrics.csv"),
                              index=False)
    logger.info(f"Done in {(time.time()-t_start)/60:.1f}m. "
                f"Saved to {args.out}/flowrra_curriculum.pth")


if __name__ == "__main__":
    main()