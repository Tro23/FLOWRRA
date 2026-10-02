"""
main_runner_Warehouse.py

Complete training and benchmarking pipeline for FLOWRRA in a discrete 3D warehouse.
Handles data ingestion (Nodes, Edges, Missions), GNN training loops, and metric tracking.
"""
import time
import logging
import sys
import os
import pandas as pd
import networkx as nx
import numpy as np
import torch

from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from node_warehouse import precompute_goal_distances
from config_warehouse import CONFIG

# stream=sys.stdout: the log goes through the SAME stream as every print().
# Before, the logger wrote to stderr while prints went to stdout, which Python
# buffers in ~8 KB blocks when piped (2>&1 | tee). Episode lines then landed
# wherever the last block ended -- glued mid-line onto a print (cold_run22's
# Ep 0001 and Ep 0002), with the episode's last prints appearing after it. The
# handler flushes stdout on every record, so pending prints go first and each
# log line starts on its own line, in order.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
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
        entry = {"G": G, "pos_dict": pos_dict, "scen_dir": scen_dir, "bank": bank,
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


def _order_window(ep: int, episodes: int):
    """
    How many floors from its dock a new order may lie (STREAM_DESIGN.md). Grows
    linearly from `start` to `end`, reaching `end` at `reach_frac` of the run, so
    the task stops changing before the near-greedy episodes the run is judged on.
    None (no limit) when the config has no order_floor_window.
    """
    w = (CONFIG.get("stream") or {}).get("order_floor_window")
    if not w:
        return None
    s, e = int(w.get("start", 1)), int(w.get("end", 3))
    frac = min(1.0, (ep - 1) / max(1.0, float(w.get("reach_frac", 0.5)) * (episodes - 1)))
    return int(round(s + (e - s) * frac))


def _recovery_value_bound() -> float:
    """
    The largest value the recovery head could ever legitimately hold, from the
    rewards it can actually receive: per step, the holon's coherence (-1..0)
    plus recovery events -- an invocation costs |invocation_cost|, a resolution
    pays resolution_bonus, a preemptive success pays preemptive_bonus. Allowing
    up to TWO of each per step (generous: the "two costs in one step" the
    fixture once showed was a duplicated charge, removed 2026-09-29, so one of
    each is now the real maximum), scaled by the holon column's value scale in
    priority mode, then
    divided by (1 - gamma) for an infinite horizon. With today's config:
    (1 + 2*(4 + 6 + 9)) / 70.3 / 0.01 = 55.5. cold_run21 reached 32,554.
    0.0 when the switch is off.
    """
    if not CONFIG["training"].get("recovery_value_bound", False):
        return 0.0
    rp = CONFIG.get("recovery_policy", {})
    rd = CONFIG.get("reward_decomposition", {})
    hs = (float((rd.get("scales") or {}).get("holon", 1.0))
          if rd.get("mode") == "priority" else 1.0)
    # collision_cost mode: no preemptive bonus, but up to collision_charge_cap
    # new colliding pairs charged per step -- the cap keeps this bound exact.
    if rp.get("reward_mode", "strict_bonus") == "collision_cost":
        event = (abs(float(rp.get("collision_cost", -8.0)))
                 * int(rp.get("collision_charge_cap", 3)))
    else:
        event = float(rp.get("preemptive_bonus", 9.0))
    per_step = (1.0 + 2.0 * (abs(float(rp.get("invocation_cost", -4.0)))
                             + float(rp.get("resolution_bonus", 6.0))
                             + event)) / hs
    return per_step / (1.0 - float(CONFIG["training"]["gamma"]))


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
    base_dim = len(n0.get_state_vector(penv.nodes))
    input_dim = base_dim + penv.density.output_dim

    # ---- FLAG CONSISTENCY, CHECKED BEFORE A LONG RUN STARTS -----------------
    # density.output_mode and gnn.encoder MUST move together. With
    # output_mode="channels" the density half is a packed 2-channel diamond --
    # 462 numbers that only mean anything once scattered back into a cube and
    # convolved. Feed that to the flat encoder and it builds Linear(545, 128)
    # over an unstructured blob: no error, no warning, and WORSE than the
    # 231-dim affordance it replaced, because the mask and the repulsion are now
    # interleaved instead of combined.
    #
    # This would have been silent. It is a hard stop instead: an hour into a
    # 100-episode run is a bad time to discover the encoder was never on.
    _enc = CONFIG["gnn"].get("encoder", "flat")
    _omode = CONFIG["density"].get("output_mode", "affordance")
    if (_enc == "conv") != (_omode == "channels"):
        raise SystemExit(
            f"CONFIG mismatch: gnn.encoder={_enc!r} with "
            f"density.output_mode={_omode!r}. Set them together -- "
            f"('conv','channels') or ('flat','affordance').")

    _edim = int(CONFIG["gnn"].get("edge_feature_dim", 0))
    logger.info(
        f"flags | encoder={_enc} output_mode={_omode} edge_dim={_edim} "
        f"adjacency={CONFIG['proximity'].get('adjacency_metric')} "
        f"rays={CONFIG['density'].get('ray_transform')} "
        f"waiting={CONFIG['waiting'].get('enabled')} "
        f"obstacles={CONFIG.get('obstacles', {}).get('enabled')} "
        f"despawn={CONFIG.get('episode', {}).get('despawn_on_delivery')}")

    agent = GNNAgent(
        node_feature_dim=input_dim, edge_feature_dim=_edim,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        reward_heads=heads, head_weights=weights,
        double_dqn=bool(CONFIG["training"].get("double_dqn", False)),
        target_tau=float(CONFIG["training"].get("target_tau", 0.0)),
        recovery_double_dqn=bool(CONFIG["training"].get("recovery_double_dqn", False)),
        recovery_value_bound=_recovery_value_bound(),
        per_fleet_terminal=bool(CONFIG["training"].get("per_fleet_terminal", False)),
        dropout=CONFIG["gnn"]["dropout"], lr=CONFIG["gnn"]["learning_rate"],
        gamma=CONFIG["training"]["gamma"],
        buffer_capacity=CONFIG["training"]["buffer_capacity"],
        batch_size=CONFIG["training"]["batch_size"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
        # Without these three the conv encoder silently is not built, whatever
        # the config says: encoder_mode defaults to "flat", and the network has
        # no way to know where the base half ends or how to scatter the packed
        # diamond back into a cube.
        encoder_mode=_enc,
        base_feature_dim=base_dim,
        density_diamond_mask=penv.density._diamond_mask,
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
    logger.info(f"recovery head: double_dqn={agent.recovery_double_dqn} "
                f"value_bound={agent.recovery_value_bound:.2f} | target_tau={agent.target_tau} "
                f"| paths_channels={CONFIG['density'].get('paths_channels', False)} "
                f"approach_warning={CONFIG['reward_decomposition'].get('approach_warning', False)} "
                f"entropy_fix={CONFIG['density'].get('entropy_fix', False)}")
    _eps = [agent.epsilon_gaussian(t, args.episodes) for t in range(1, args.episodes + 1)]
    logger.info(f"learning: per_fleet_terminal={agent.per_fleet_terminal} | exploration "
                f"ep1={_eps[0]:.3f} peak={max(_eps):.3f} at ep {1 + _eps.index(max(_eps))} "
                f"last={_eps[-1]:.3f} | below 0.05 from ep "
                f"{next((t for t in range(1 + _eps.index(max(_eps)), args.episodes + 1) if _eps[t - 1] < 0.05), None)} "
                f"| config {agent.explore_cfg or 'historical defaults'}")
    _sc = CONFIG.get("stream") or {}
    logger.info(f"stream: enabled={bool(_sc.get('enabled', False))} "
                f"exit_floors={_sc.get('exit_floors', 'all')} docks/floor={_sc.get('exits_per_floor', 6)} "
                f"respawn_delay={_sc.get('respawn_delay', 0)} "
                f"| order window by episode: {[_order_window(t, args.episodes) for t in (1, args.episodes // 4 or 1, args.episodes // 2 or 1, args.episodes)]} "
                f"| errors={CONFIG['errors'].get('enabled')} "
                f"| frozen_obstacle_severity={CONFIG['density'].get('frozen_obstacle_severity')}")
    # Every log records its own conflict configuration (CONFLICT_DESIGN.md), so
    # a run can be read later without guessing which switches were on.
    _cc = CONFIG.get("conflict") or {}
    logger.info("conflict | " + " ".join(f"{k}={v}" for k, v in _cc.items()))

    logs, learn_steps, t0 = [], 0, time.time()
    for ep in range(1, args.episodes + 1):
        map_name = rng.choice(names)
        k = (rng.choice(agent_sets) if agent_sets
             else rng.randint(args.agents_min, args.agents_max))
        inst = sample_instance(cache, map_name, k, rng)
        if inst is None:
            continue

        # ORDER STREAM (STREAM_DESIGN.md): the whole goal bank's maps, so every
        # future order has one, and an order seed fixed by (seed, episode) --
        # independent of rng, so the instance draws are unchanged.
        _stream = bool((CONFIG.get("stream") or {}).get("enabled", False))
        _e = cache.get(inst["map"]) if _stream else None
        env = FLOWRRA(inst["G"], inst["pos_dict"], inst["missions"], mode="training",
                      goal_distance_maps=(_e["goal_distance_maps"] if _stream else inst["gdm"]),
                      shared_pool_mode=True,
                      goal_pool=inst["goal_pool"],
                      order_bank=(_e["bank"] if _stream else None),
                      order_seed=(args.seed * 1000003 + ep if _stream else None),
                      order_floor_window=(_order_window(ep, args.episodes) if _stream else None))
        env.gnn = agent
        agent.reset_episode_state()

        ep_reward, hl, dg = 0.0, [], []
        for _ in range(args.max_steps):
            ep_reward += env.step(episode_step=ep, total_episodes=args.episodes)
            if len(agent.memory) >= agent.batch_size:
                agent.learn(node_ids=[n.id for n in env.nodes])
                learn_steps += 1
                # tau > 0: blend every learning step. 0: hard copy, as before.
                if agent.target_tau > 0:
                    agent.soft_update_target()
                elif learn_steps % args.target_sync == 0:
                    agent.update_target_network()
                hl.append(dict(agent.last_head_losses))
                dg.append({**{f"qval_{h}": v for h, v in agent.last_head_values.items()},
                           "loss_recovery": agent.last_recovery_loss,
                           "qval_recovery": agent.last_recovery_q})
            if env.is_episode_over():
                break

        est = env.get_error_statistics()
        if getattr(env, "stream", False):
            # A stream retires delivered orders, so claimed_goals and the pool
            # no longer count work: deliveries of orders issued do.
            _ss = env._stream_stats()
            done, total = _ss["stream_deliveries"], max(1, _ss["stream_orders_issued"])
        else:
            done = len(env.claimed_goals)
            total = max(1, len(env.goal_pool))
        unfinished = [n for n in env.nodes if n.id not in env.immobile_nodes]
        left = float(np.mean([n.get_graph_distance_to_goal() for n in unfinished])) if unfinished else 0.0
        mean_hl = {h: float(np.mean([d[h] for d in hl])) if hl else 0.0 for h in heads}

        # EFFICIENCY (measurement fix 4): stream completion cannot reach 1 --
        # orders are always in flight -- so the line reports deliveries against
        # the conflict-free ideal as well. PREVENTION (fix 2): of the preemptive
        # recoveries that resolved, how many saw no collision among the moved
        # fleets within the window, next to the strict one-step success.
        _eff = est.get("stream_efficiency", float("nan"))
        _eff_txt = (f" eff {_eff * 100:.0f}% of {est['stream_ideal_deliveries']:.0f}"
                    if "stream_efficiency" in est and np.isfinite(_eff) else "")
        _prev_res = est["preempt_clear_k"] + est["preempt_collided_k"]
        # WAIT SHARE among fleets at risk: what the network PROPOSED against
        # what was DONE (rules and holds impose waits the policy never chose).
        _wp, _we = est.get("choice_policy_wait_share"), est.get("choice_exec_wait_share")
        _wait_txt = (f" | wait policy {_wp * 100:.0f}% / done {_we * 100:.0f}%"
                     if _wp is not None and np.isfinite(_wp) else "")
        logger.info(
            f"Ep {ep:04d} | {map_name:<22} k={len(env.nodes):<3} | R {ep_reward:9.1f} | "
            f"done {done}/{total}{_eff_txt} | coll {env.loop.total_collisions} | "
            f"rec inv {est['recovery_invocations']} forced {est['recovery_forced']} wasted {est['recovery_wasted']} pre {est['recovery_preemptive']}/{est['recovery_preemptive_success']} "
            f"clear{est['preempt_window']} {est['preempt_clear_k']}/{_prev_res}{_wait_txt} | "
            f"risk {est['risk_steps_acted']}/{est['risk_steps']} "
            f"({est['intervention_rate']*100:.0f}%) | "
            f"err {est['errors_injected']} hand {est['handovers_completed']} | "
            f"ovr {est['action_override_rate']*100:.0f}% | "
            f"eps {agent.epsilon_gaussian(ep, args.episodes):.3f} | "
            f"grad {env.get_gradient_agreement():.3f} | left {left:.1f}h | "
            + " ".join(f"{h}:{mean_hl[h]:.3f}" for h in heads)
        )
        if "stream_deliveries" in est:
            logger.info(
                f"        stream | delivered {est['stream_deliveries']} of {est['stream_orders_issued']} orders "
                f"| exits {est['stream_exits']} re-entries {est['stream_reentries']} "
                f"| on floor {est['stream_on_floor_mean']:.1f} | life {est['stream_life_steps_mean']:.0f} steps "
                f"(exit leg {est['stream_exit_leg_steps_mean']:.0f}) | coll/100 deliveries "
                f"{100.0 * env.loop.total_collisions / max(1, est['stream_deliveries']):.1f} "
                f"| orders within ±{est['stream_order_window']} floors (mean gap {est['stream_order_floor_gap_mean']:.2f}) "
                f"| by quarter {est['stream_deliv_q1']}/{est['stream_deliv_q2']}/{est['stream_deliv_q3']}/{est['stream_deliv_q4']}")
        if "path_views" in est:
            logger.info(
                f"         paths | contested {est['path_contested_pct']:.1f}% of views "
                f"(next cell {est['path_contested_next_pct']:.1f}%) | nearest "
                f"{est['path_contested_soonest']:.2f} cells ahead | shared "
                f"{est['path_shared_pct']:.1f}% | routes in view {est['path_routes_seen_mean']:.2f}")

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

        # Why each undelivered fleet failed, and how far apart fleets ended up.
        # These are the only columns that say anything about the ~8 fleets per
        # episode that quietly do not arrive -- every other metric has been
        # improving while completion sat flat, which means the cause was never
        # in the instrumentation.
        row.update({k: v for k, v in est.items()
                    if k.startswith(("incomplete_", "hops_", "sep_", "orders_",
                                     "tier1_", "tier2_", "recurrence_", "doorstep_", "start_hops_",
                                     "steps_held_", "steps_waiting_",
                                     "convoy_", "held_pair_", "rwd_", "path_", "stream_", "choice_",
                                     "preempt_", "conflict_", "watch_", "corridor_"))})
        row.update({f"loss_{h}": mean_hl[h] for h in heads})
        # Value level per head, and the recovery head's loss and value --
        # episode means over learning steps (0.0 before learning starts).
        row.update({k: (float(np.mean([d[k] for d in dg])) if dg else 0.0)
                    for k in ([f"qval_{h}" for h in heads] + ["loss_recovery", "qval_recovery"])})
        logs.append(row)

        if ep % 5 == 0:
            os.makedirs(args.out, exist_ok=True)
            agent.save(os.path.join(args.out, "flowrra_curriculum.pth"))
            pd.DataFrame(logs).to_csv(os.path.join(args.out, "curriculum_metrics.csv"), index=False)

    os.makedirs(args.out, exist_ok=True)
    agent.save(os.path.join(args.out, "flowrra_curriculum.pth"))
    pd.DataFrame(logs).to_csv(os.path.join(args.out, "curriculum_metrics.csv"), index=False)
    logger.info(f"Done in {(time.time()-t0)/60:.1f}m -> {args.out}/")


if __name__ == '__main__':
    main()