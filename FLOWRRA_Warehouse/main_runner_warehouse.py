"""
main_runner_Warehouse.py

Complete training and benchmarking pipeline for FLOWRRA in a discrete 3D warehouse.
Handles data ingestion (Nodes, Edges, Missions), GNN training loops, and metric tracking.
"""
import time
import logging
import os
import pandas as pd
import networkx as nx
import numpy as np
import torch

from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from node_warehouse import precompute_goal_distances
from config_warehouse import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Warehouse_Runner")

def load_warehouse_data(nodes_csv: str, edges_csv: str, missions_csv: str):
    """
    Parses the warehouse CSV files to build the physical NetworkX graph, 
    the coordinate dictionary, and the benchmarking fleet missions.
    """
    logger.info("Loading Warehouse Data...")
    
    # 1. Load Data
    nodes_df = pd.read_csv(nodes_csv, index_col=False)
    edges_df = pd.read_csv(edges_csv, index_col=False)
    agent_data = pd.read_csv(missions_csv, index_col=False)

    # 2. Clean and Map Nodes
    nodes_df['X'] = pd.to_numeric(nodes_df['X'])
    nodes_df['Y'] = pd.to_numeric(nodes_df['Y'])
    nodes_df['Z'] = pd.to_numeric(nodes_df['Z'])
    nodes_df['NodeId'] = nodes_df['NodeId'].astype(str).str.strip()
    
    pos_dict = nodes_df.set_index('NodeId')[['X', 'Y', 'Z']].to_dict('index')
    
    # 3. Build NetworkX Graph
    edges_df['nodeFrom'] = edges_df['nodeFrom'].astype(str).str.strip()
    edges_df['nodeTo'] = edges_df['nodeTo'].astype(str).str.strip()
    
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['nodeFrom'], row['nodeTo'])

    # 4. Extract Fleet Missions
    agent_data['startNodeId'] = agent_data['startNodeId'].astype(str).str.strip()
    agent_data['goalNodeId'] = agent_data['goalNodeId'].astype(str).str.strip()
    
    fleet_missions = []
    for idx, (_, row) in enumerate(agent_data.iterrows()):
        agent_id = str(row['agentId']).strip()
        start = row['startNodeId']
        goal = row['goalNodeId']
        
        if start in pos_dict and goal in pos_dict:
            fleet_missions.append({
                "id": agent_id,
                "start_node": start,
                "goal_node": goal,
                "start_pos": np.array([pos_dict[start]['X'], pos_dict[start]['Y'], pos_dict[start]['Z']], dtype=np.float32),
                "goal_pos": np.array([pos_dict[goal]['X'], pos_dict[goal]['Y'], pos_dict[goal]['Z']], dtype=np.float32)
            })

    logger.info(f"Loaded {len(G.nodes)} Nodes, {len(G.edges)} Edges, and {len(fleet_missions)} Fleet Missions.")
    return G, pos_dict, fleet_missions


def load_warehouse_data_pool_mode(nodes_csv: str, edges_csv: str, missions_csv: str):
    """
    Same CSV files as load_warehouse_data(), reinterpreted for shared-pool mode:
    startNodeId values are still fleet spawn points (one per fleet, unchanged),
    but goalNodeId values become a shared POOL of claimable goals rather than a
    fixed 1:1 pairing with a specific fleet -- see FLOWRRA's shared_pool_mode
    and node_warehouse.py's retarget_to_nearest_unclaimed() for the actual
    claiming mechanism this feeds.

    Returns: (G, pos_dict, fleet_missions, goal_pool)
      fleet_missions: [{"id", "start_node", "start_pos"}, ...] -- no goal fields,
        goals aren't pre-assigned.
      goal_pool: {goal_node_id: position} -- every DISTINCT goalNodeId value in
        the CSV, available to every fleet.
    """
    logger.info("Loading Warehouse Data (shared-pool mode)...")

    nodes_df = pd.read_csv(nodes_csv, index_col=False)
    edges_df = pd.read_csv(edges_csv, index_col=False)
    agent_data = pd.read_csv(missions_csv, index_col=False)

    nodes_df['X'] = pd.to_numeric(nodes_df['X'])
    nodes_df['Y'] = pd.to_numeric(nodes_df['Y'])
    nodes_df['Z'] = pd.to_numeric(nodes_df['Z'])
    nodes_df['NodeId'] = nodes_df['NodeId'].astype(str).str.strip()
    pos_dict = nodes_df.set_index('NodeId')[['X', 'Y', 'Z']].to_dict('index')

    edges_df['nodeFrom'] = edges_df['nodeFrom'].astype(str).str.strip()
    edges_df['nodeTo'] = edges_df['nodeTo'].astype(str).str.strip()
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['nodeFrom'], row['nodeTo'])

    agent_data['startNodeId'] = agent_data['startNodeId'].astype(str).str.strip()
    agent_data['goalNodeId'] = agent_data['goalNodeId'].astype(str).str.strip()

    fleet_missions = []
    for _, row in agent_data.iterrows():
        agent_id = str(row['agentId']).strip()
        start = row['startNodeId']
        if start in pos_dict:
            fleet_missions.append({
                "id": agent_id,
                "start_node": start,
                "start_pos": np.array([pos_dict[start]['X'], pos_dict[start]['Y'], pos_dict[start]['Z']], dtype=np.float32),
            })

    goal_pool = {}
    for goal in agent_data['goalNodeId'].unique():
        if goal in pos_dict:
            goal_pool[goal] = np.array(
                [pos_dict[goal]['X'], pos_dict[goal]['Y'], pos_dict[goal]['Z']], dtype=np.float32
            )

    logger.info(
        f"Loaded {len(G.nodes)} Nodes, {len(G.edges)} Edges, {len(fleet_missions)} Fleets, "
        f"{len(goal_pool)} pool goals."
    )
    return G, pos_dict, fleet_missions, goal_pool

import argparse
import os
import random
import time
import numpy as np
import pandas as pd
import networkx as nx
import torch
from node_warehouse import precompute_goal_distances_compact

# =============================================================================
# MULTI-MAP CURRICULUM TRAINER
# =============================================================================
# REPLACES the previous single-map, single-scenario loop. That loop loaded ONE
# map and ONE scenario file and reused them for every episode, which is not N
# episodes of learning to route -- it is N episodes on one instance. The
# resulting policy scored 23/25 on that exact instance and 2/10 on fresh
# scenarios on the SAME map, which rules out topology transfer and leaves
# scenario memorisation as the cause.
#
# Every episode now draws a fresh (map, scenario, fleet count). Graphs and
# goal-bank BFS maps are cached per map so that costs one traversal per map
# rather than one per episode.


class MapCache:
    """Loads each map's graph, coordinates and goal-bank distance maps once."""

    def __init__(self, maps_dir, scens_dir):
        self.maps_dir, self.scens_dir = maps_dir, scens_dir
        self._cache = {}

    def discover(self, only=None):
        names = sorted({f[:-len("_Nodes.csv")]
                        for f in os.listdir(self.maps_dir) if f.endswith("_Nodes.csv")})
        out = []
        for n in names:
            if only and n not in only:
                continue
            if not os.path.exists(os.path.join(self.maps_dir, f"{n}_Edges.csv")):
                continue
            if not os.path.isdir(os.path.join(self.scens_dir, n)):
                logger.warning(f"{n}: no scenario directory, skipping")
                continue
            out.append(n)
        return out

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        t0 = time.time()
        nd = pd.read_csv(os.path.join(self.maps_dir, f"{name}_Nodes.csv"), index_col=False)
        ed = pd.read_csv(os.path.join(self.maps_dir, f"{name}_Edges.csv"), index_col=False)
        for c in ("X", "Y", "Z"):
            nd[c] = pd.to_numeric(nd[c])
        nd["NodeId"] = nd["NodeId"].astype(str).str.strip()
        pos_dict = nd.set_index("NodeId")[["X", "Y", "Z"]].to_dict("index")
        ed["nodeFrom"] = ed["nodeFrom"].astype(str).str.strip()
        ed["nodeTo"] = ed["nodeTo"].astype(str).str.strip()
        G = nx.Graph()
        G.add_nodes_from(nd["NodeId"])
        G.add_edges_from(zip(ed["nodeFrom"], ed["nodeTo"]))

        # grid_pos_dict is keyed on integer (X,Y,Z): two nodes sharing a
        # coordinate means the later one silently wins and any fleet on the first
        # resolves to the second, which is how a distance signal dies with no
        # error raised. Checked on every map at load rather than trusted.
        cc = {}
        for pp in pos_dict.values():
            k = (int(round(pp["X"])), int(round(pp["Y"])), int(round(pp["Z"])))
            cc[k] = cc.get(k, 0) + 1
        lost = sum(v - 1 for v in cc.values() if v > 1)
        if lost:
            logger.error(f"{name}: {lost} node(s) share an integer coordinate and are "
                         f"UNREACHABLE via grid_pos_dict. Fix the map before training.")

        scen_dir = os.path.join(self.scens_dir, name)
        scen_files = sorted(f for f in os.listdir(scen_dir)
                            if f.endswith(".csv") and "StartGoalLocations" in f)
        bank_path = os.path.join(scen_dir, f"{name}_GoalBank.csv")
        if os.path.exists(bank_path):
            bank = [str(v).strip() for v in pd.read_csv(bank_path)["goalNodeId"].tolist()]
        else:
            seen = set()
            for f in scen_files:
                seen.update(str(v).strip() for v in
                            pd.read_csv(os.path.join(scen_dir, f), index_col=False)["goalNodeId"])
            bank = sorted(seen)
        bank = [g for g in bank if g in pos_dict]
        logger.info(f"{name}: {G.number_of_nodes()} nodes, {len(scen_files)} scenarios, "
                    f"goal bank {len(bank)} -- precomputing BFS maps...")
        gdm, node_index = precompute_goal_distances_compact(G, bank)
        entry = {"G": G, "pos_dict": pos_dict, "scen_dir": scen_dir,
                 "scen_files": scen_files, "goal_distance_maps": gdm,
                 "node_index": node_index}
        self._cache[name] = entry
        logger.info(f"{name}: cached in {time.time()-t0:.1f}s")
        return entry


def sample_instance(cache, map_name, k, rng):
    """One episode's instance: fresh scenario, shuffled rows, first k."""
    e = cache.get(map_name)
    scen = rng.choice(e["scen_files"])
    df = pd.read_csv(os.path.join(e["scen_dir"], scen), index_col=False)
    df["startNodeId"] = df["startNodeId"].astype(str).str.strip()
    df["goalNodeId"] = df["goalNodeId"].astype(str).str.strip()
    pos = e["pos_dict"]
    df = df[df["startNodeId"].isin(pos) & df["goalNodeId"].isin(pos)]
    # Shuffled before taking k so two episodes at the same (map, seed, k) still
    # differ -- otherwise a few thousand episodes would revisit each exact
    # instance hundreds of times, a milder form of the memorisation this whole
    # loop exists to avoid.
    df = df.sample(frac=1.0, random_state=rng.randrange(1 << 30)).head(k)
    if len(df) < 2:
        return None
    missions = [{"id": str(r["agentId"]).strip(), "start_node": r["startNodeId"],
                 "start_pos": np.array([pos[r["startNodeId"]]["X"],
                                        pos[r["startNodeId"]]["Y"],
                                        pos[r["startNodeId"]]["Z"]], dtype=np.float32)}
                for _, r in df.iterrows()]
    goal_pool = {g: np.array([pos[g]["X"], pos[g]["Y"], pos[g]["Z"]], dtype=np.float32)
                 for g in df["goalNodeId"].unique()}
    gdm = {g: e["goal_distance_maps"][g] for g in goal_pool
           if g in e["goal_distance_maps"]}
    missing = [g for g in goal_pool if g not in gdm]
    if missing:
        extra, _ = precompute_goal_distances_compact(e["G"], missing,
                                                     node_index=e["node_index"])
        gdm.update(extra)
        e["goal_distance_maps"].update(extra)
    return {"map": map_name, "scen": scen, "G": e["G"], "pos_dict": pos,
            "missions": missions, "goal_pool": goal_pool, "gdm": gdm}


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)  # see ablate.py: prefix
                                                     # matching silently
                                                     # mis-assigns flags
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens_v2")
    ap.add_argument("--maps", default="", help="comma-separated subset; blank = all")
    ap.add_argument("--episodes", type=int, default=CONFIG["training"]["total_episodes"])
    ap.add_argument("--agents-min", type=int, default=25)
    ap.add_argument("--agents-max", type=int, default=50)
    ap.add_argument("--agent-sets", default="",
                    help="comma-separated DISCRETE fleet counts to sample from, e.g. "
                         "'25,55'. Overrides --agents-min/--agents-max. Use this when "
                         "you want two distinct regimes rather than a uniform sweep "
                         "between them: sampling uniformly over [25,60] spends most "
                         "episodes at intermediate counts you never intended to test.")
    ap.add_argument("--cold-start", action="store_true",
                    help="peak the exploration schedule at episode 0 and decay from "
                         "there. USE THIS WHENEVER TRAINING FROM SCRATCH. The default "
                         "Gaussian peaks mid-run, so epsilon at episode 1 is ~0.01 -- "
                         "near-greedy on a randomly initialised network, which follows "
                         "an arbitrary DETERMINISTIC map and is worse than random for "
                         "state coverage.")
    ap.add_argument("--max-steps", type=int,
                    default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--target-sync", type=int, default=1000,
                    help="target-net sync period in GRADIENT STEPS, not episodes")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    agent_sets = [int(v) for v in args.agent_sets.split(",") if v.strip()]
    only = [m.strip() for m in args.maps.split(",") if m.strip()] or None
    cache = MapCache(args.maps_dir, args.scens_dir)
    names = cache.discover(only)
    if not names:
        raise SystemExit(f"no usable maps in {args.maps_dir} / {args.scens_dir}")
    logger.info(f"{len(names)} maps available: {names}")

    rd = CONFIG["reward_decomposition"]
    heads, weights = rd["heads"], rd["weights"]

    # PROBE ENVIRONMENT -- built once, purely to measure the state vector width
    # so GNNAgent can be constructed with the right input_dim. It is never given
    # a GNN, never stepped, never pushes a transition, and is discarded on the
    # next line. Its "[Core] Spawned N fleets" line appears BEFORE episode 1 and
    # at a different fleet count, which reads like the run contradicting itself;
    # it does not. Labelled here so the two are distinguishable in the log.
    logger.info("Building throwaway probe environment to measure state vector size "
                "(its spawn/assignment lines below are NOT episode 1)...")
    probe = sample_instance(cache, names[0],
                            min(agent_sets) if agent_sets else args.agents_min, rng)
    penv = FLOWRRA(probe["G"], probe["pos_dict"], probe["missions"], mode="init",
                   goal_distance_maps=probe["gdm"], shared_pool_mode=True,
                   goal_pool=probe["goal_pool"])
    n0 = penv.nodes[0]
    input_dim = (len(n0.get_state_vector(penv.nodes))
                 + len(penv.density.get_local_affordance(n0.current_pos, penv.nodes, set())))

    agent = GNNAgent(
        node_feature_dim=input_dim, edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        reward_heads=heads, head_weights=weights,
        dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
        gamma=CONFIG["training"]["gamma"],
        buffer_capacity=CONFIG["training"]["buffer_capacity"],
        batch_size=CONFIG["training"]["batch_size"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
    )
    agent.cold_start = bool(args.cold_start)
    if args.cold_start and args.resume:
        logger.warning("--cold-start with --resume: opening at maximum exploration "
                       "will discard much of what the checkpoint knows. Intended?")
    if args.resume:
        agent.load(args.resume)
        logger.info(f"Warm-started from {args.resume}")
    logger.info("Probe discarded. Training starts now.")
    logger.info(f"k from {agent_sets or f'[{args.agents_min},{args.agents_max}]'} | "
                f"cold_start={args.cold_start}")
    logger.info(f"input_dim={input_dim} gamma={CONFIG['training']['gamma']} "
                f"heads={heads} weights={weights}")

    logs, learn_steps, t0 = [], 0, time.time()
    for ep in range(1, args.episodes + 1):
        map_name = rng.choice(names)
        k = (rng.choice(agent_sets) if agent_sets
             else rng.randint(args.agents_min, args.agents_max))
        inst = sample_instance(cache, map_name, k, rng)
        if inst is None:
            continue

        env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                      goal_distance_maps=inst["gdm"], shared_pool_mode=True,
                      goal_pool=inst["goal_pool"])
        env.gnn = agent
        agent.reset_episode_state()

        ep_reward, hl = 0.0, []
        for _ in range(args.max_steps):
            ep_reward += env.step(episode_step=ep, total_episodes=args.episodes)
            if len(agent.memory) >= agent.batch_size:
                agent.learn(node_ids=[n.id for n in env.nodes])
                learn_steps += 1
                if learn_steps % args.target_sync == 0:
                    agent.update_target_network()
                hl.append(dict(agent.last_head_losses))
            if env.is_episode_over():
                break

        est = env.get_error_statistics()
        done = len(env.claimed_goals)
        total = max(1, len(env.goal_pool))
        unfinished = [n for n in env.nodes if n.id not in env.immobile_nodes]
        left = float(np.mean([n.get_graph_distance_to_goal() for n in unfinished])) if unfinished else 0.0
        mean_hl = {h: float(np.mean([d[h] for d in hl])) if hl else 0.0 for h in heads}

        logger.info(
            f"Ep {ep:04d} | {map_name:<22} k={len(env.nodes):<3} | R {ep_reward:9.1f} | "
            f"done {done}/{total} | coll {env.loop.total_collisions} | "
            f"rec inv {est['recovery_invocations']} forced {est['recovery_forced']} wasted {est['recovery_wasted']} pre {est['recovery_preemptive']}/{est['recovery_preemptive_success']} | "
            f"risk {est['risk_steps_acted']}/{est['risk_steps']} "
            f"({est['intervention_rate']*100:.0f}%) | "
            f"err {est['errors_injected']} hand {est['handovers_completed']} | "
            f"ovr {est['action_override_rate']*100:.0f}% | "
            f"eps {agent.epsilon_gaussian(ep, args.episodes):.3f} | "
            f"grad {env.get_gradient_agreement():.3f} | left {left:.1f}h | "
            + " ".join(f"{h}:{mean_hl[h]:.3f}" for h in heads)
        )

        row = {"episode": ep, "map": map_name, "scen": inst["scen"],
               "agents": len(env.nodes), "reward": ep_reward,
               "completed": done, "completion_rate": done / total,
               "collisions": env.loop.total_collisions,
               "recovery_invocations": est["recovery_invocations"],
               "recovery_forced": est["recovery_forced"],
               "recovery_resolved": est["recovery_resolved"],
               "recovery_wasted": est["recovery_wasted"],
               "recovery_preemptive": est["recovery_preemptive"],
               "recovery_preemptive_success": est["recovery_preemptive_success"],
               "risk_steps": est["risk_steps"],
               "risk_steps_acted": est["risk_steps_acted"],
               "warning_steps": est["warning_steps"],
               "intervention_rate": est["intervention_rate"],
               # The honest density measure: fraction of fleet-steps spent
               # inside the warning band, i.e. under brake. Occupancy is a
               # map-dependent proxy; this is the mechanism itself.
               "brake_duty_cycle": est["brake_duty_cycle"],
               "mean_peer_gap": est["mean_peer_gap"],
               "errors_injected": est["errors_injected"],
               "handovers_completed": est["handovers_completed"],
               "retired_recalled": est["retired_fleets_recalled"],
               "action_override_rate": est["action_override_rate"],
               "gradient_agreement": env.get_gradient_agreement(),
               "mean_hops_remaining": left}
        row.update({f"loss_{h}": mean_hl[h] for h in heads})
        logs.append(row)

        if ep % 50 == 0:
            os.makedirs(args.out, exist_ok=True)
            agent.save(os.path.join(args.out, "flowrra_curriculum.pth"))
            pd.DataFrame(logs).to_csv(os.path.join(args.out, "curriculum_metrics.csv"), index=False)

    os.makedirs(args.out, exist_ok=True)
    agent.save(os.path.join(args.out, "flowrra_curriculum.pth"))
    pd.DataFrame(logs).to_csv(os.path.join(args.out, "curriculum_metrics.csv"), index=False)
    logger.info(f"Done in {(time.time()-t0)/60:.1f}m -> {args.out}/")


if __name__ == '__main__':
    main()