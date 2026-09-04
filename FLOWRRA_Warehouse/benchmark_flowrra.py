"""
benchmark_flowrra.py

Benchmark harness for FLOWRRA on the 3D MAPF Warehouse benchmark
(Wang, Veerapaneni, Wu, Li & Likhachev, ICAPS 2024 -- see mapf.info/Main/Benchmarks).

Runs a trained FLOWRRA checkpoint across every (map, scenario-seed, agent-count)
combination and writes ONE CSV row per instance. Every algorithm you later compare
against writes rows into the same schema, so the comparison table is a groupby
rather than a manual merge.

WHY A SEPARATE HARNESS FROM animated_flowrra.py
That script renders one rollout. A benchmark needs the opposite shape: many
rollouts, no rendering, and a strict separation between "simulate" and "report"
so a second algorithm can be dropped in without touching the metrics code. The
simulation is therefore in run_flowrra_instance(), which returns a plain dict --
implement the same signature for prioritized planning or EECBS and everything
downstream keeps working.

METRIC FAMILIES
  Standard MAPF (comparable to EECBS / MAPF-LNS2 / LaCAM):
    success, completion_rate, makespan, sum_of_costs, soc_lower_bound,
    suboptimality, collisions, runtime_s, decision_ms_per_step
  Maintenance / coherence (FLOWRRA's actual claim -- no planner reports these):
    mean_integrity, integrity_auc, steps_incoherent, mean_time_to_recoherence,
    max_time_to_recoherence, blast_radius, tier1/2/3, max_pair_repeats
  Honesty metrics (report these or a reviewer will find them):
    livelock_override_rate, gradient_agreement

AGENT-COUNT PROTOCOL
The standard MAPF protocol takes the FIRST k agents from a scenario file and
scales k upward until the method fails. That is what produces the canonical
"success rate vs number of agents" curve, and it is the plot reviewers look for
first. AGENT_COUNTS below drives it.
"""

import os
import re
import glob
import time
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import networkx as nx

from core_warehouse import FLOWRRA
from agent_warehouse import GNNAgent
from node_warehouse import precompute_goal_distances
from config_warehouse import CONFIG


# ==========================================================================
# 1. INSTANCE DISCOVERY & LOADING
# ==========================================================================
def discover_instances(maps_dir: str, scens_dir: str) -> List[Dict[str, Any]]:
    """
    Pairs every <name>_Nodes.csv / <name>_Edges.csv in maps_dir with every
    scenario file in scens_dir/<name>/.

    The benchmark ships 15 maps x 10 seeds. Running one map is how you end up
    with a policy that is excellent at 50_5_5_10_5_10 and untested everywhere
    else -- the ICAPS paper's own finding is that warehouse structure noticeably
    influences MAPF performance, so single-map numbers invite exactly that
    objection.
    """
    instances = []
    for nodes_path in sorted(glob.glob(os.path.join(maps_dir, "*_Nodes.csv"))):
        name = os.path.basename(nodes_path)[: -len("_Nodes.csv")]
        edges_path = os.path.join(maps_dir, f"{name}_Edges.csv")
        if not os.path.exists(edges_path):
            print(f"  [skip] {name}: no matching _Edges.csv")
            continue

        scen_glob = os.path.join(scens_dir, name, "*StartGoalLocations*.csv")
        scen_files = sorted(glob.glob(scen_glob))
        if not scen_files:
            print(f"  [skip] {name}: no scenarios under {os.path.join(scens_dir, name)}")
            continue

        for scen_path in scen_files:
            seed_match = re.search(r"Seed(\d+)", os.path.basename(scen_path))
            instances.append({
                "map": name,
                "seed": int(seed_match.group(1)) if seed_match else -1,
                "nodes_csv": nodes_path,
                "edges_csv": edges_path,
                "scen_csv": scen_path,
            })
    return instances


def load_instance(nodes_csv: str, edges_csv: str, scen_csv: str, num_agents: Optional[int]):
    """
    Loads one instance in shared-pool mode, optionally truncated to the first
    num_agents rows of the scenario file.

    Truncation happens HERE rather than after loading, because the goal pool must
    contain exactly the goals of the retained agents. Slicing fleet_missions
    afterwards would leave the pool full-size, so the Hungarian assignment could
    hand a fleet a goal belonging to an agent that is not in the instance -- a
    silently different (and easier) problem than the benchmark defines.
    """
    nodes_df = pd.read_csv(nodes_csv, index_col=False)
    edges_df = pd.read_csv(edges_csv, index_col=False)
    scen_df = pd.read_csv(scen_csv, index_col=False)

    for col in ("X", "Y", "Z"):
        nodes_df[col] = pd.to_numeric(nodes_df[col])
    nodes_df["NodeId"] = nodes_df["NodeId"].astype(str).str.strip()
    pos_dict = nodes_df.set_index("NodeId")[["X", "Y", "Z"]].to_dict("index")

    edges_df["nodeFrom"] = edges_df["nodeFrom"].astype(str).str.strip()
    edges_df["nodeTo"] = edges_df["nodeTo"].astype(str).str.strip()
    G = nx.Graph()
    G.add_nodes_from(pos_dict.keys())
    for _, row in edges_df.iterrows():
        G.add_edge(row["nodeFrom"], row["nodeTo"])

    scen_df["startNodeId"] = scen_df["startNodeId"].astype(str).str.strip()
    scen_df["goalNodeId"] = scen_df["goalNodeId"].astype(str).str.strip()
    scen_df = scen_df[
        scen_df["startNodeId"].isin(pos_dict) & scen_df["goalNodeId"].isin(pos_dict)
    ].reset_index(drop=True)
    if num_agents is not None:
        scen_df = scen_df.iloc[:num_agents].reset_index(drop=True)

    fleet_missions, goal_pool = [], {}
    for _, row in scen_df.iterrows():
        start = row["startNodeId"]
        g = row["goalNodeId"]
        # goal_node / goal_pos are what FIXED-assignment mode reads. They are
        # harmless in shared-pool mode (which ignores them and re-matches via
        # Hungarian), so both protocols load through this one function.
        fleet_missions.append({
            "id": str(row["agentId"]).strip(),
            "start_node": start,
            "goal_node": g,
            "start_pos": np.array(
                [pos_dict[start]["X"], pos_dict[start]["Y"], pos_dict[start]["Z"]],
                dtype=np.float32,
            ),
            "goal_pos": np.array(
                [pos_dict[g]["X"], pos_dict[g]["Y"], pos_dict[g]["Z"]],
                dtype=np.float32,
            ),
        })
        goal = row["goalNodeId"]
        goal_pool[goal] = np.array(
            [pos_dict[goal]["X"], pos_dict[goal]["Y"], pos_dict[goal]["Z"]],
            dtype=np.float32,
        )

    return G, pos_dict, fleet_missions, goal_pool


# ==========================================================================
# 2. THE SIMULATION -- swap this function to benchmark another algorithm
# ==========================================================================
def run_flowrra_instance(G, pos_dict, fleet_missions, goal_pool,
                         agent: GNNAgent, max_steps: int,
                         eval_epsilon: float = 0.02,
                         shared_pool: bool = True) -> Dict[str, Any]:
    """
    Runs ONE instance and returns a flat metrics dict.

    eval_epsilon defaults to 0.02, not 0.0. At exactly zero, once every peer has
    frozen the environment is stationary and the greedy policy becomes a
    deterministic map from position to action -- which on a finite position set
    must eventually cycle. Measured on this benchmark, that stranded 2 of 25
    fleets byte-identically across repeated runs, one of them two hops from its
    goal. Report the value you used; it is part of the method.
    """
    if shared_pool:
        goal_distance_maps = precompute_goal_distances(
            G, [{"goal_node": gid} for gid in goal_pool])
    else:
        goal_distance_maps = precompute_goal_distances(G, fleet_missions)
    env = FLOWRRA(
        G, pos_dict, fleet_missions, mode="eval",
        goal_distance_maps=goal_distance_maps,
        shared_pool_mode=shared_pool,
        goal_pool=goal_pool if shared_pool else None,
    )
    agent.reset_episode_state()          # checkpoint carries a stale frozen set
    agent.epsilon_gaussian = lambda *a, **k: eval_epsilon
    env.gnn = agent

    n = len(env.nodes)
    soc_lb_hops = sum(float(node.initial_graph_distance) for node in env.nodes)

    finish_step: Dict[str, int] = {}
    integrity_trace: List[float] = []
    integrity_strict: List[float] = []
    deadlock_sizes: List[int] = []
    decision_times: List[float] = []

    t_start = time.perf_counter()
    for step in range(max_steps):
        t0 = time.perf_counter()
        env.step(episode_step=1, total_episodes=1)
        decision_times.append((time.perf_counter() - t0) * 1000.0)

        integrity_trace.append(float(env.loop.calculate_integrity()))
        # See baselines_mapf.execute_paths(): mean_integrity must mean "no actual
        # overlap" for every method, or a collision-free planner is penalised for
        # legal proximity. The warning-band score is kept as proximity_margin.
        integrity_strict.append(0.0 if env.loop.deadlocked_nodes else 1.0)
        if env.loop.deadlocked_nodes:
            deadlock_sizes.append(len(env.loop.deadlocked_nodes))

        for node in env.nodes:                       # per-agent completion time
            if node.id in env.frozen_nodes and node.id not in finish_step:
                finish_step[node.id] = step + 1

        if len(env.frozen_nodes) == n:
            break
    runtime_s = time.perf_counter() - t_start
    steps_run = len(integrity_trace)

    # --- standard MAPF costs -------------------------------------------------
    # Unfinished agents are charged the full step budget. This is the usual
    # convention and it keeps SoC comparable across methods that fail differently.
    per_agent_cost = [finish_step.get(node.id, max_steps) for node in env.nodes]
    sum_of_costs = int(sum(per_agent_cost))
    makespan = int(max(per_agent_cost)) if per_agent_cost else 0
    # base_speed 0.5 -> one graph hop costs two timesteps.
    soc_lb = soc_lb_hops / max(CONFIG["warehouse"]["base_speed"], 1e-9)

    # --- coherence -----------------------------------------------------------
    # An "incoherent run" is a maximal stretch of consecutive steps with
    # integrity < 1.0. Its LENGTH is time-to-recoherence: how long the fleet took
    # to return to clear flow. This is the metric an offline planner has no
    # analogue for -- its plan is collision-free by construction, so on a clean
    # instance it scores a trivial 1.0 and the metric only separates methods once
    # execution actually deviates from plan.
    runs, cur = [], 0
    for v in integrity_strict:
        if v < 1.0:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)

    rec = env.recovery.get_statistics()
    total_actions = max(steps_run * n, 1)
    unfinished = [nd for nd in env.nodes if nd.id not in env.frozen_nodes]

    return {
        # --- standard MAPF ---
        "num_agents": n,
        "success": int(len(env.frozen_nodes) == n),
        "completed": len(env.frozen_nodes),
        "completion_rate": len(env.frozen_nodes) / n,
        "makespan": makespan,
        "sum_of_costs": sum_of_costs,
        "soc_lower_bound": round(soc_lb, 1),
        "suboptimality": round(sum_of_costs / soc_lb, 3) if soc_lb > 0 else float("nan"),
        "collisions": env.loop.get_statistics()["total_collisions_occurred"],
        "steps_run": steps_run,
        "runtime_s": round(runtime_s, 3),
        "decision_ms_per_step": round(float(np.mean(decision_times)), 3) if decision_times else 0.0,
        # --- maintenance / coherence ---
        "mean_integrity": round(float(np.mean(integrity_strict)), 4) if integrity_strict else 1.0,
        "proximity_margin": round(float(np.mean(integrity_trace)), 4) if integrity_trace else 1.0,
        "integrity_auc": round(float(np.sum(integrity_trace)), 1),
        "steps_incoherent": int(sum(runs)),
        "mean_time_to_recoherence": round(float(np.mean(runs)), 2) if runs else 0.0,
        "max_time_to_recoherence": int(max(runs)) if runs else 0,
        # blast radius: fleets disturbed per conflict event. A centralized
        # replanner's is N by construction, since it replans everyone.
        "blast_radius": round(float(np.mean(deadlock_sizes)), 2) if deadlock_sizes else 0.0,
        "tier1_spatial": rec["spatial_recoveries"],
        "tier2_temporal": rec["temporal_recoveries"],
        "tier3_yield": rec["yield_recoveries"],
        "max_pair_repeats": max(rec["repeat_pairs"].values()) if rec["repeat_pairs"] else 0,
        # --- honesty ---
        "livelock_override_rate": round(env.livelock_overrides / total_actions, 4),
        "gradient_agreement": round(env.get_gradient_agreement(), 4),
        "mean_hops_remaining": round(
            float(np.mean([nd.get_graph_distance_to_goal() for nd in unfinished])), 2
        ) if unfinished else 0.0,
    }


# ==========================================================================
# 3. DRIVER
# ==========================================================================
def build_agent(G, pos_dict, fleet_missions, goal_pool, checkpoint: str) -> GNNAgent:
    """
    Builds the agent once and reuses it for every instance.

    input_dim is map-INDEPENDENT: 54 state dims + 231 affordance dims = 285,
    regardless of warehouse size or fleet count (the GAT handles variable N via
    the adjacency matrix). That is what makes cross-map evaluation of a single
    checkpoint possible at all.
    """
    gdm = precompute_goal_distances(G, [{"goal_node": g} for g in goal_pool])
    probe = FLOWRRA(G, pos_dict, fleet_missions, mode="init", goal_distance_maps=gdm,
                    shared_pool_mode=True, goal_pool=goal_pool)
    node = probe.nodes[0]
    input_dim = (len(node.get_state_vector(probe.nodes))
                 + len(probe.density.get_local_affordance(node.current_pos, probe.nodes, set())))

    agent = GNNAgent(
        node_feature_dim=input_dim, edge_feature_dim=0,
        action_size=CONFIG["gnn"]["action_size"],
        hidden_dim=CONFIG["gnn"]["hidden_dim"],
        num_layers=CONFIG["gnn"]["num_layers"],
        n_heads=CONFIG["gnn"]["num_heads"],
        dropout=CONFIG["gnn"]["dropout"],
        stability_coef=CONFIG["gnn"]["stability_coef"],
    )
    agent.load(checkpoint)
    print(f"[Bench] Agent loaded (input_dim={input_dim}) from {checkpoint}")
    return agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps-dir", default="all_maps")
    ap.add_argument("--scens-dir", default="all_scens")
    ap.add_argument("--checkpoint", default="checkpoints/flowrra_warehouse_gnn.pth")
    ap.add_argument("--out", default="benchmark_flowrra.csv")
    ap.add_argument("--agents", default="10,25,50",
                    help="comma-separated agent counts (standard MAPF scaling protocol)")
    ap.add_argument("--seeds", default="", help="comma-separated seeds; blank = all")
    ap.add_argument("--maps", default="", help="comma-separated map names; blank = all")
    ap.add_argument("--max-steps", type=int, default=CONFIG["training"]["max_steps_per_episode"])
    ap.add_argument("--epsilon", type=float, default=0.02)
    args = ap.parse_args()

    agent_counts = [int(a) for a in args.agents.split(",") if a.strip()]
    seed_filter = {int(s) for s in args.seeds.split(",") if s.strip()}
    map_filter = {m.strip() for m in args.maps.split(",") if m.strip()}

    print(f"[Bench] Discovering instances in {args.maps_dir} / {args.scens_dir} ...")
    instances = discover_instances(args.maps_dir, args.scens_dir)
    if map_filter:
        instances = [i for i in instances if i["map"] in map_filter]
    if seed_filter:
        instances = [i for i in instances if i["seed"] in seed_filter]
    print(f"[Bench] {len(instances)} (map, seed) pairs x {len(agent_counts)} agent counts "
          f"= {len(instances) * len(agent_counts)} runs\n")
    if not instances:
        print("[Bench] Nothing to run. Check --maps-dir / --scens-dir layout:")
        print("        all_maps/<name>_Nodes.csv, <name>_Edges.csv")
        print("        all_scens/<name>/<name>_StartGoalLocations_Seed<k>.csv")
        return

    first = instances[0]
    G0, pos0, miss0, pool0 = load_instance(
        first["nodes_csv"], first["edges_csv"], first["scen_csv"], agent_counts[0])
    agent = build_agent(G0, pos0, miss0, pool0, args.checkpoint)

    rows, t0 = [], time.time()
    for idx, inst in enumerate(instances, 1):
        for k in agent_counts:
            try:
                G, pos_dict, missions, pool = load_instance(
                    inst["nodes_csv"], inst["edges_csv"], inst["scen_csv"], k)
                if len(missions) < k:
                    continue                     # scenario has fewer agents than requested
                m = run_flowrra_instance(G, pos_dict, missions, pool, agent,
                                         args.max_steps, args.epsilon)
            except Exception as exc:             # one bad instance must not kill the sweep
                print(f"  [error] {inst['map']} seed{inst['seed']} k={k}: {exc}")
                m = {"num_agents": k, "success": 0, "error": str(exc)}

            m.update(algorithm="FLOWRRA", map=inst["map"], seed=inst["seed"],
                     requested_agents=k)
            rows.append(m)
            pd.DataFrame(rows).to_csv(args.out, index=False)   # checkpoint every run

        done = idx * len(agent_counts)
        rate = (time.time() - t0) / max(done, 1)
        print(f"[Bench] {idx}/{len(instances)} maps-seeds | {done} runs | "
              f"{rate:.1f}s/run | eta {rate * (len(instances) - idx) * len(agent_counts) / 60:.0f}m")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[Bench] Wrote {len(df)} rows -> {args.out}\n")

    if "success" in df:
        print("Success rate by map and agent count:")
        print(df.pivot_table(index="map", columns="requested_agents",
                             values="success", aggfunc="mean").round(2).to_string())
        print("\nHeadline means:")
        cols = ["completion_rate", "suboptimality", "collisions", "mean_integrity",
                "mean_time_to_recoherence", "blast_radius", "decision_ms_per_step",
                "livelock_override_rate", "gradient_agreement"]
        print(df[[c for c in cols if c in df]].mean().round(4).to_string())


if __name__ == "__main__":
    main()