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
                         shared_pool: bool = True,
                         failure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
    # ---- MATCHED PERMANENT FAILURE ----------------------------------------
    # FAIRNESS: the baselines were previously the only arm that received the
    # disturbance -- run_flowrra_instance took no such argument, so --disturb hit
    # the planners and left FLOWRRA on a clean instance. Any comparison drawn
    # from that is void.
    #
    # Failures are injected here at the SAME step and on the SAME agent indices
    # the baselines get, through FLOWRRA's own error path so its recovery layer
    # engages exactly as it does in training. Its own stochastic injection is
    # disabled for the run so the two arms see identical failures and nothing
    # else.
    # Progress-triggered, matching baselines_mapf -- see the long note there.
    # A shared step index is not a shared point in the mission when one arm runs
    # 6.5x more steps than the other.
    failure = failure or {}
    # Multiple waves, matching baselines_mapf -- see the note there. The second
    # wave is what can kill a vehicle mid-rescue, which is the case FLOWRRA's
    # re-dispatch exists for and the naive layer has no answer to.
    fail_waves = sorted(failure.get("progress_waves", [])
                        or ([failure["progress"]] if failure.get("progress") is not None else []))
    fail_step = int(failure.get("step", -1))
    fail_idx = list(failure.get("agents", []))
    n_to_fail = int(failure.get("count", len(fail_idx)))
    _wave_i = 0
    rescuer_deaths = 0
    if failure:
        env.errors_enabled = False
    n_orphaned = 0
    orphaned_goal_ids = set()   # the ORDERS stranded, not just the count
    _orphan_step = {}           # goal id -> step it was orphaned
    n_stale_at_injection = 0    # failures whose order was already delivered

    for step in range(max_steps):
        t0 = time.perf_counter()

        if _wave_i < len(fail_waves):
            # Deadline fallback -- see the note in baselines_mapf. Without it an
            # arm that stalls never reaches the later progress thresholds and is
            # rewarded with fewer failures.
            _deadline = int(max_steps * (_wave_i + 1) / (len(fail_waves) + 1))
            # Terminal states, matching baselines_mapf -- delivered OR dead.
            _prog = ((len(env.claimed_goals) + len(env.stopped_nodes))
                     / max(1, len(env.goal_pool)))
            if _prog >= fail_waves[_wave_i] or step >= _deadline:
                fail_idx = [i for i, nd in enumerate(env.nodes)
                            if nd.id not in env.immobile_nodes and nd.current_goal_id][:n_to_fail]
                for i in fail_idx:
                    if env.nodes[i].id in env._pickup_assignment:
                        rescuer_deaths += 1
                fail_step = step
                _wave_i += 1

        if step == fail_step and fail_idx:
            for a in fail_idx:
                if a < len(env.nodes):
                    node = env.nodes[a]
                    if node.id in env.immobile_nodes or not node.current_goal_id:
                        continue
                    env.stopped_nodes.add(node.id)
                    env._error_step[node.id] = env.step_count
                    node.direction = np.zeros(3, dtype=np.float32)
                    env.total_errors += 1
                    # Only an order that is STILL OUTSTANDING is orphaned. Core
                    # already makes this distinction -- it prints "confirmed
                    # dead; no recoverable order" and opens no pickup when the
                    # goal was claimed by someone else while the fleet sat
                    # there.
                    #
                    # BOTH sides of the ratio have to agree. A first attempt
                    # filtered only the numerator's source set and left this
                    # counter incrementing on every failure, which deflated the
                    # recovery rate by exactly the number of stale goals --
                    # measured as one order per instance. An order that was
                    # already delivered was never orphaned, so it belongs in
                    # neither the numerator nor the denominator.
                    if node.current_goal_id not in env.claimed_goals:
                        n_orphaned += 1
                        orphaned_goal_ids.add(node.current_goal_id)
                        _orphan_step[node.current_goal_id] = step
                    else:
                        n_stale_at_injection += 1

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

        if env.is_episode_over():
            break
    runtime_s = time.perf_counter() - t_start
    steps_run = len(integrity_trace)

    # --- standard MAPF costs -------------------------------------------------
    # Unfinished agents are charged the full step budget. This is the usual
    # convention and it keeps SoC comparable across methods that fail differently.
    # SURVIVORS ONLY. The MAPF convention charges an unfinished agent the full
    # step budget, which is correct for an agent that was alive and failed to
    # arrive -- that is a routing failure and should cost. It is NOT correct for
    # a vehicle that was deliberately killed: it did not route badly, it was
    # destroyed, and charging it the cap conflates the two.
    #
    # With 9 deaths and a 780-step cap that penalty is 7,020 before a single
    # vehicle has moved -- measured at 89-95% of the total, so every sum_of_costs
    # comparison was reading a ~5% remainder while the number was dominated by a
    # constant. makespan was worse: a max over agents, so one corpse pinned it at
    # 780 on an episode that actually finished in 146 steps.
    #
    # Excluding the dead makes both quantities mean what they are supposed to:
    # among the vehicles that were alive to finish, how efficiently did they?
    _alive = [n for n in env.nodes if n.id not in env.stopped_nodes]
    per_agent_cost = [finish_step.get(node.id, max_steps) for node in _alive]
    sum_of_costs = int(sum(per_agent_cost))
    makespan = int(max(per_agent_cost)) if per_agent_cost else 0
    agents_costed = len(per_agent_cost)
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

    _est = env.get_error_statistics()

    # RECOVERED = the orphaned ORDER was delivered, by whatever route.
    #
    # This used to be handovers_completed, which counts pickup TRANSFERS and
    # undercounts in three ways: a rescuer inheriting two stranded orders counts
    # once, a transfer whose goal was claimed meanwhile counts as a handover but
    # recovers nothing, and -- the big one -- in shared-pool mode a dead fleet's
    # goal returns to the pool and can be delivered by an ordinary retarget with
    # no handover at all. That order is recovered and the terminal shows it, but
    # the CSV scored it as lost.
    #
    # The baselines have only one route to an orphaned order (goals are fixed per
    # agent, so the rescue is the only way it gets delivered), so counting
    # delivery-by-any-route makes the two arms measure the same thing. The
    # handover count is kept separately, because the split between "recovered by
    # the handover mechanic" and "recovered by ordinary pool retargeting" is
    # exactly what says how much the mechanic is contributing.
    _rec = len(orphaned_goal_ids & set(env.claimed_goals))
    # Rescue latency, matching what the naive baseline reports: steps from the
    # failure that orphaned an order to the delivery that recovered it.
    # Only count a rescue whose delivery came AFTER the failure. A negative
    # latency means the goal was already claimed when the fleet holding it
    # died -- i.e. the fleet was carrying a stale goal id for an order somebody
    # else had already delivered. That is a real condition worth knowing about,
    # not a rounding artefact, so it is counted separately rather than clamped
    # away: `stale_orphans` is the number of "orphaned" orders that were never
    # actually outstanding.
    _pairs = [(env._goal_claim_step[g], _orphan_step[g])
              for g in (orphaned_goal_ids & set(env.claimed_goals))
              if g in getattr(env, "_goal_claim_step", {}) and g in _orphan_step]
    _lat = [c - o for c, o in _pairs if c > o]
    _stale = sum(1 for c, o in _pairs if c <= o)
    return {
        # --- failure recovery (the comparison this harness exists for) ---
        # TWO DIFFERENT COUNTS, deliberately.
        #   failures_injected : vehicles killed. Identical across arms by
        #                       construction, so this is what the harness check
        #                       should compare.
        #   orders_orphaned   : failures that actually stranded an OUTSTANDING
        #                       order, which is the denominator recovery rate is
        #                       measured against.
        # They differ only for FLOWRRA, and only because shared-pool mode lets
        # another fleet claim a goal while the fleet holding it sits dying. The
        # baselines assign a fixed goal per agent, so the case cannot arise for
        # them. Reporting one number for both would either break the matched-
        # failure check or put un-recoverable orders in the denominator.
        "failures_injected": n_orphaned + n_stale_at_injection,
        "orders_orphaned": n_orphaned,
        "orders_recovered": _rec,
        "recovered_via_handover": _est["handovers_completed"],
        "rescuer_deaths": rescuer_deaths,
        "orders_lost": max(0, n_orphaned - _rec),
        "recovery_rate": (_rec / n_orphaned) if n_orphaned else float("nan"),
        "retired_recalled": _est["retired_fleets_recalled"],
        # HOPS, not steps. FLOWRRA spends two simulator steps per hop at
        # base_speed 0.5 while the baseline moves one edge per timestep, so a raw
        # step count silently doubled FLOWRRA's apparent rescue time. Converting
        # here means the CSV column is directly comparable and nothing
        # downstream has to remember the factor.
        "mean_recovery_hops": (float(np.mean(_lat)) * 0.5 if _lat else float("nan")),
        "stale_orphans": _stale + n_stale_at_injection,
        # Density, measured rather than assumed. agents/nodes treats a
        # corridor cell and a junction as equivalent and ignores that
        # traffic concentrates on routes; this is the fraction of
        # fleet-steps actually spent inside the warning band, which is what
        # throttles the fleet. Measured 0.33 on a 1,435-node map versus 0.06
        # on a 6,300-node one at the SAME fleet count.
        # ACTUAL DISTANCE TRAVELLED, summed over fleets, in cell units.
        #
        # sum_of_costs measures TIME (finish step per agent), and converting it
        # to distance by multiplying by base_speed assumes fleets sustain
        # nominal speed. They do not: affordance braking floors them as low as
        # 0.05 cells/step, the final-approach override only lifts the warning-
        # zone floor to 0.7 of the ramp, and dwelling at a pickup costs steps
        # with no movement at all. So the converted figure is an UPPER BOUND on
        # distance, not a measurement of it, and the looser the braking the
        # looser the bound.
        #
        # env already accumulates the real thing per fleet per step
        # (_fleet_travel, Manhattan distance actually moved). Reporting it makes
        # the distance comparison a measurement instead of an inference, and it
        # needs no unit conversion: one cell is one cell in either arm.
        "distance_travelled": float(sum(env._fleet_travel.values()))
        if hasattr(env, "_fleet_travel") else float("nan"),
        # DISTANCE, COUNTED -- not inferred from the clock. soc_hops is
        # sum_of_costs (finish STEPS) x base_speed, i.e. "how far could it have
        # gone at nominal speed". FLOWRRA never sustains nominal speed: affordance
        # braking floors it as low as 0.05, the final-approach override only
        # lifts it to 0.7 of the ramp, and pickup dwell burns steps with no
        # movement at all. _fleet_travel sums the actual Manhattan distance moved
        # each step, so this needs no assumption about what a timestep is worth.
        "distance_travelled": round(float(sum(env._fleet_travel.values())), 1),
        # COLLISIONS AS A RATE. The raw count is collision-STEPS, and the arms run
        # for wildly different durations -- RHCR alone finishes in 22-51 steps,
        # FLOWRRA runs 380-600, naive runs to the 780 cap. Comparing counts
        # rewards whichever arm had least opportunity to register one.
        "collision_rate": round(env.loop.total_collisions / max(steps_run, 1), 5),
        # Measured cells traversed, and collisions as a RATE. A raw collision
        # count is a count of deadlocked STEPS, and the arms run for wildly
        # different lengths -- RHCR finishes in ~22-51 steps, FLOWRRA takes
        # 380-600, naive runs to the 780 cap. An arm that finishes quickly has
        # an order of magnitude fewer chances to register anything, so the raw
        # count flatters it for reasons unrelated to safety.
        "distance_travelled": _est.get("distance_travelled", float("nan")),
        "collision_rate": (env.loop.get_statistics()["total_collisions_occurred"]
                           / max(steps_run, 1)),
        "brake_duty_cycle": _est.get("brake_duty_cycle", float("nan")),
        "mean_peer_gap": _est.get("mean_peer_gap", float("nan")),
        # --- standard MAPF ---
        "num_agents": n,
        # Orders DELIVERED over orders issued, counted the same way as the
        # baselines. claimed_goals, not frozen_nodes: a recalled fleet leaves
        # frozen_nodes although its delivery already happened, and a dead fleet
        # must never be scored as finished.
        "success": int(len(env.claimed_goals) == len(env.goal_pool)),
        "completed": len(env.claimed_goals),
        "completion_rate": len(env.claimed_goals) / max(1, len(env.goal_pool)),
        "makespan": makespan,
        "sum_of_costs": sum_of_costs,
        "agents_costed": agents_costed,   # survivors the cost is averaged over
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

    input_dim is map-INDEPENDENT: 60 state dims + 231 affordance dims = 291,
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
        # The checkpoint carries one decoder per reward head; building the agent
        # without declaring them gives K=1 and load_state_dict fails on
        # "Unexpected key(s) action_decoders.1..4". Sourced from CONFIG so this
        # cannot drift from what the orchestrator emits.
        reward_heads=CONFIG["reward_decomposition"]["heads"],
        head_weights=CONFIG["reward_decomposition"]["weights"],
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