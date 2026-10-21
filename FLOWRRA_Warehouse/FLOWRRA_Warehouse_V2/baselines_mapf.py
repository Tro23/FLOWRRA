"""
baselines_mapf.py

Graph-general MAPF baselines for the 3D MAPF Warehouse benchmark
(Wang, Veerapaneni, Wu, Li & Likhachev, ICAPS 2024).

WHY THESE RUN IN 3D AT ALL
It is tempting to claim "SOTA MAPF methods do not handle 3D". That claim is
false and the benchmark paper says so directly: it finds that the tested 2D MAPF
techniques scaled well to 3D warehouses and that the community's 2D progress
generalizes. CBS, EECBS, PP, PIBT and LaCAM are all defined over an ARBITRARY
graph -- a 3D warehouse is just a graph whose elevator edges happen to change z.
The obstacle is purely that public implementations hardcode 4-connected 2D grids
and MovingAI .map parsing.

So both baselines here operate directly on the networkx graph loaded from the
benchmark's Nodes.csv / Edges.csv. No grid assumption anywhere.

  Prioritized Planning (PP) + space-time A*
      Erdmann & Lozano-Perez 1987. Fast, incomplete, the standard industrial
      approach and the low-level component EECBS is built around. The ICAPS
      paper's own conclusion is that a fast low-level search is critical for 3D
      MAPF, which makes PP the honest floor to beat.

  PIBT (Priority Inheritance with BackTracking)
      Okumura, Machida, Defago & Tamura, AIJ 2022. One-step-lookahead with
      priority inheritance; it is the engine inside LaCAM, the current one-shot
      MAPF state of the art. Fast and scalable, so it is the realistic
      near-SOTA comparison you can actually run.

BOTH RETURN PATHS, NOT METRICS. Scoring happens in execute_paths(), which walks
the plan through the SAME WarehouseLoop used by FLOWRRA. That is what makes the
comparison fair: identical collision model, identical integrity instrument. A
planner scores a trivial mean_integrity of 1.0 on a clean instance -- which is
exactly the point. The metric only separates methods once execution deviates
from the plan, and that is the regime FLOWRRA is built for.
"""

import time
import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ==========================================================================
# SHARED: heuristics
# ==========================================================================
def bfs_distances(G, goal: str) -> Dict[str, int]:
    """Exact distance-to-goal for every node. Admissible + consistent, so A* with
    it expands the minimum possible number of states."""
    dist = {goal: 0}
    frontier = [goal]
    while frontier:
        nxt = []
        for u in frontier:
            for v in G.neighbors(u):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return dist


# ==========================================================================
# BASELINE 1: Prioritized Planning + space-time A*
# ==========================================================================
def space_time_astar(G, start: str, goal: str, h: Dict[str, int],
                     vertex_res: Dict[Tuple[str, int], int],
                     edge_res: Dict[Tuple[str, str, int], int],
                     max_t: int, agent_id: int) -> Optional[List[str]]:
    """
    A* over (vertex, timestep) against a reservation table.

    Reservations encode both conflict types the benchmark forbids: VERTEX
    (two agents on one node at time t) and EDGE/swap (two agents traversing the
    same edge in opposite directions between t and t+1). Omitting the edge case
    is the single most common bug in a PP implementation -- the plan looks
    conflict-free by vertex count while two robots walk through each other.
    """
    if start not in h:
        return None
    open_heap = [(h[start], 0, start, None)]
    came: Dict[Tuple[str, int], Optional[Tuple[str, int]]] = {}
    seen = set()

    while open_heap:
        f, t, v, parent = heapq.heappop(open_heap)
        if (v, t) in seen:
            continue
        seen.add((v, t))
        came[(v, t)] = parent

        if v == goal:
            # Must be able to REST here: a later arrival would collide with us.
            if not any((goal, tt) in vertex_res for tt in range(t, max_t + 1)):
                path, cur = [], (v, t)
                while cur is not None:
                    path.append(cur[0])
                    cur = came[cur]
                return path[::-1]

        if t >= max_t:
            continue

        for nxt in list(G.neighbors(v)) + [v]:          # +[v] = wait action
            if nxt not in h:
                continue
            if (nxt, t + 1) in vertex_res:
                continue
            if (nxt, v, t) in edge_res:                 # head-on swap
                continue
            if (nxt, t + 1) in seen:
                continue
            heapq.heappush(open_heap, (t + 1 + h[nxt], t + 1, nxt, (v, t)))
    return None


def plan_prioritized(G, starts: List[str], goals: List[str],
                     max_t: int, order: Optional[List[int]] = None):
    """
    Plans agents one at a time, each avoiding all previously planned paths.

    PP is INCOMPLETE -- a late agent can be boxed in by earlier ones with no
    valid path. Returns partial results with those agents marked failed rather
    than aborting, because "PP solved 22 of 25" is a real and reportable outcome.
    """
    n = len(starts)
    order = order if order is not None else list(range(n))
    heuristics = [bfs_distances(G, g) for g in goals]

    vertex_res: Dict[Tuple[str, int], int] = {}
    edge_res: Dict[Tuple[str, str, int], int] = {}
    paths: List[Optional[List[str]]] = [None] * n

    for i in order:
        p = space_time_astar(G, starts[i], goals[i], heuristics[i],
                             vertex_res, edge_res, max_t, i)
        paths[i] = p
        if p is None:
            continue
        for t, v in enumerate(p):
            vertex_res[(v, t)] = i
            if t + 1 < len(p):
                edge_res[(v, p[t + 1], t)] = i
        # Reserve the goal cell for the remainder of the horizon -- the agent
        # parks there, so nobody may route through it afterwards.
        for t in range(len(p), max_t + 1):
            vertex_res[(p[-1], t)] = i
    return paths


# ==========================================================================
# BASELINE 2: PIBT
# ==========================================================================
def plan_pibt(G, starts: List[str], goals: List[str], max_t: int):
    """
    PIBT: one-step lookahead with priority inheritance and backtracking.

    At each timestep every agent, in priority order, claims the neighbour that
    most reduces its distance-to-goal. If the claim is blocked by a lower-priority
    agent, that agent INHERITS the higher priority and is asked to move out of
    the way, recursively; failure backtracks and the claimant tries its next
    choice. Priorities grow while an agent is away from its goal and reset on
    arrival, which is what stops any one agent from starving.

    This is the engine inside LaCAM, so it is the closest thing to a current
    one-shot SOTA that is small enough to reimplement faithfully on a general
    graph.
    """
    n = len(starts)
    h = [bfs_distances(G, g) for g in goals]
    cur = list(starts)
    priority = [0.0] * n
    for i in range(n):
        priority[i] = h[i].get(starts[i], 1e9) / max(len(G), 1)

    paths: List[List[str]] = [[s] for s in starts]

    for _ in range(max_t):
        occupied_now = {v: i for i, v in enumerate(cur)}
        nxt: Dict[int, Optional[str]] = {i: None for i in range(n)}
        claimed: Dict[str, int] = {}

        def pibt(i: int, blocker: Optional[int]) -> bool:
            cands = sorted(
                list(G.neighbors(cur[i])) + [cur[i]],
                key=lambda v: (h[i].get(v, 1e9), v),
            )
            for v in cands:
                if v in claimed:
                    continue
                # Forbid swapping through an edge with the agent that pushed us.
                if blocker is not None and v == cur[blocker]:
                    continue
                claimed[v] = i
                k = occupied_now.get(v)
                if k is not None and k != i and nxt[k] is None:
                    if not pibt(k, i):
                        del claimed[v]
                        continue
                nxt[i] = v
                return True
            claimed[cur[i]] = i
            nxt[i] = cur[i]
            return False

        for i in sorted(range(n), key=lambda a: -priority[a]):
            if nxt[i] is None:
                pibt(i, None)

        cur = [nxt[i] if nxt[i] is not None else cur[i] for i in range(n)]
        for i in range(n):
            paths[i].append(cur[i])
            priority[i] = 0.0 if cur[i] == goals[i] else priority[i] + 1.0

        if all(cur[i] == goals[i] for i in range(n)):
            break

    return paths


# ==========================================================================
# SHARED EXECUTION + SCORING
# ==========================================================================
def execute_paths(G, pos_dict, paths, starts, goals, loop, max_t: int,
                  plan_time_s: float, algorithm: str,
                  disturbance: Optional[Dict[str, Any]] = None,
                  replan_fn=None) -> Dict[str, Any]:
    """
    Walks a plan through the SAME WarehouseLoop FLOWRRA uses, so integrity,
    collisions and coherence are measured by identical code.

    disturbance (optional) is what turns this from a planning comparison into a
    MAINTENANCE comparison:
        {"step": t, "agents": [i, ...], "duration": d}
    freezes those agents in place for d timesteps, simulating a stalled robot or
    an occupied elevator. A precomputed plan has no answer to this -- every agent
    behind the stalled one is now desynchronised from its reservation, so the
    plan's collision-freedom guarantee is void and the system must replan.
    replan_fn, if supplied, is called at that moment and its wall-clock cost is
    recorded as replan_s: the latency FLOWRRA does not pay.
    """
    n = len(paths)

    class _P:                        # minimal node stand-in for WarehouseLoop
        __slots__ = ("id", "current_pos")
        def __init__(self, i, p): self.id, self.current_pos = str(i), p

    def coord(node_id):
        d = pos_dict[node_id]
        return np.array([d["X"], d["Y"], d["Z"]], dtype=np.float32)

    def at(i, t):
        p = paths[i]
        if p is None:
            return starts[i]
        return p[min(t, len(p) - 1)]

    stalled_until = {}
    if disturbance:
        for a in disturbance.get("agents", []):
            stalled_until[a] = disturbance["step"] + disturbance.get("duration", 10)

    # TWO integrity traces, deliberately.
    #
    # WarehouseLoop scores 0.5 whenever two fleets are within warning_threshold
    # (2.0 Manhattan) and 0.0 only on an actual overlap. That warning band is
    # FLOWRRA's caution model, not a MAPF rule: routing two agents one or two
    # cells apart is a perfectly legal, collision-free solution. Scoring a
    # planner against it marks it down for correct behaviour -- measured, PP
    # returned a ZERO-collision plan and still scored mean_integrity 0.685.
    #
    #   integrity_strict  : 1.0 unless a genuine overlap occurred. Comparable
    #                       across methods; this is the one that belongs in the
    #                       standard-metrics table.
    #   integrity_margin  : the original warning-band-sensitive score. A real
    #                       safety property (how much clearance a method leaves)
    #                       but a FLOWRRA-defined one, so it is reported
    #                       separately and never as "correctness".
    integrity_trace, integrity_strict, deadlock_sizes, step_ms = [], [], [], []
    finish_step: Dict[int, int] = {}
    offset = {i: 0 for i in range(n)}          # timesteps lost to stalling
    replan_s = 0.0
    horizon = max((len(p) for p in paths if p), default=1)

    for t in range(max_t):
        t0 = time.perf_counter()

        if disturbance and t == disturbance["step"] and replan_fn is not None:
            r0 = time.perf_counter()
            new_paths = replan_fn(t, [at(i, t - offset[i]) for i in range(n)])
            replan_s += time.perf_counter() - r0
            if new_paths is not None:
                paths = new_paths
                offset = {i: t for i in range(n)}

        for i in range(n):
            if t < stalled_until.get(i, -1):
                offset[i] += 1

        positions = [at(i, t - offset[i]) for i in range(n)]
        nodes = [_P(i, coord(v)) for i, v in enumerate(positions)]
        frozen = {str(i) for i in range(n) if positions[i] == goals[i]}

        integ = loop.check_integrity(nodes, t, frozen)
        integrity_trace.append(float(integ))
        integrity_strict.append(0.0 if loop.deadlocked_nodes else 1.0)
        if loop.deadlocked_nodes:
            deadlock_sizes.append(len(loop.deadlocked_nodes))
        step_ms.append((time.perf_counter() - t0) * 1000.0)

        for i in range(n):
            if positions[i] == goals[i] and i not in finish_step:
                finish_step[i] = t + 1

        if len(finish_step) == n:
            break

    per_agent = [finish_step.get(i, max_t) for i in range(n)]
    runs, c = [], 0
    for v in integrity_strict:
        if v < 1.0:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)

    solved = sum(1 for p in paths if p is not None)
    return {
        "algorithm": algorithm,
        "num_agents": n,
        "success": int(len(finish_step) == n),
        "completed": len(finish_step),
        "completion_rate": len(finish_step) / n,
        "agents_planned": solved,
        "makespan": int(max(per_agent)) if per_agent else 0,
        "sum_of_costs": int(sum(per_agent)),
        "collisions": loop.get_statistics()["total_collisions_occurred"],
        "steps_run": len(integrity_trace),
        "plan_time_s": round(plan_time_s, 4),
        "replan_s": round(replan_s, 4),
        "runtime_s": round(plan_time_s + replan_s, 4),
        "decision_ms_per_step": round(float(np.mean(step_ms)), 4) if step_ms else 0.0,
        "mean_integrity": round(float(np.mean(integrity_strict)), 4) if integrity_strict else 1.0,
        "proximity_margin": round(float(np.mean(integrity_trace)), 4) if integrity_trace else 1.0,
        "integrity_auc": round(float(np.sum(integrity_trace)), 1),
        "steps_incoherent": int(sum(runs)),
        "mean_time_to_recoherence": round(float(np.mean(runs)), 2) if runs else 0.0,
        "max_time_to_recoherence": int(max(runs)) if runs else 0,
        "blast_radius": round(float(np.mean(deadlock_sizes)), 2) if deadlock_sizes else 0.0,
    }


def run_baseline_instance(G, pos_dict, fleet_missions, goal_pool, method: str,
                          max_steps: int, assignment: Dict[str, str],
                          disturbance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Entry point mirroring run_flowrra_instance()'s contract: same inputs, same
    metric keys out, so every algorithm lands in one CSV schema.

    `assignment` maps fleet id -> goal node id. Pass FLOWRRA's Hungarian result
    so all methods solve the IDENTICAL task -- otherwise you are comparing task
    allocation as well as planning, and the numbers mean nothing.
    """
    from loop_warehouse import WarehouseLoop
    from config_warehouse import CONFIG

    starts = [m["start_node"] for m in fleet_missions]
    goals = [assignment[m["id"]] for m in fleet_missions]
    horizon = max_steps

    t0 = time.perf_counter()
    if method == "PP":
        paths = plan_prioritized(G, starts, goals, horizon)
    elif method == "PIBT":
        paths = plan_pibt(G, starts, goals, horizon)
    else:
        raise ValueError(f"unknown method {method}")
    plan_time = time.perf_counter() - t0

    def _replan(t, current):
        if method == "PP":
            return plan_prioritized(G, current, goals, horizon)
        return plan_pibt(G, current, goals, horizon)

    loop = WarehouseLoop(
        collision_threshold=CONFIG["warehouse"]["collision_threshold"],
        warning_threshold=CONFIG["warehouse"]["warning_threshold"],
    )
    return execute_paths(G, pos_dict, paths, starts, goals, loop, horizon,
                         plan_time, method, disturbance,
                         _replan if disturbance else None)


# ==========================================================================
# BASELINE 3: RHCR-style rolling-horizon replanning
# ==========================================================================
def space_time_astar_windowed(G, start: str, goal: str, h: Dict[str, int],
                              vertex_res: Dict[Tuple[str, int], int],
                              edge_res: Dict[Tuple[str, str, int], int],
                              window: int) -> Optional[List[str]]:
    """
    Space-time A* truncated at `window` timesteps.

    Differs from the full solver in what it returns on failure to reach the
    goal: rather than None, it returns the best reachable state at the horizon,
    ranked by (distance-to-goal, then time). That IS windowed MAPF -- the point
    of a bounded horizon is to make PROGRESS toward the goal while guaranteeing
    collision-freedom only for the next `window` steps, then replan. Returning
    None instead would make every agent whose goal is further than the window
    fail immediately, which is why a full-horizon solver cannot simply be called
    with a small max_t.
    """
    if start not in h:
        return None
    open_heap = [(h[start], 0, start, None)]
    came: Dict[Tuple[str, int], Optional[Tuple[str, int]]] = {}
    seen = set()
    best_key, best_state = (h[start], 0), (start, 0)

    while open_heap:
        f, t, v, parent = heapq.heappop(open_heap)
        if (v, t) in seen:
            continue
        seen.add((v, t))
        came[(v, t)] = parent

        key = (h.get(v, 1 << 30), -t)      # closer to goal wins; later time breaks ties
        if key < best_key:
            best_key, best_state = key, (v, t)
        if v == goal:
            best_state = (v, t)
            break
        if t >= window:
            continue

        for nxt in list(G.neighbors(v)) + [v]:
            if nxt not in h or (nxt, t + 1) in vertex_res or (nxt, v, t) in edge_res:
                continue
            if (nxt, t + 1) in seen:
                continue
            heapq.heappush(open_heap, (t + 1 + h[nxt], t + 1, nxt, (v, t)))

    path, cur = [], best_state
    while cur is not None:
        path.append(cur[0])
        cur = came.get(cur)
    return path[::-1]


def plan_windowed(G, starts, goals, window: int, method: str):
    """One windowed planning round. Returns paths covering at most `window` steps."""
    if method == "PIBT":
        return plan_pibt(G, starts, goals, window)

    n = len(starts)
    heuristics = [bfs_distances(G, g) for g in goals]
    vertex_res: Dict[Tuple[str, int], int] = {}
    edge_res: Dict[Tuple[str, str, int], int] = {}
    paths = []
    for i in range(n):
        p = space_time_astar_windowed(G, starts[i], goals[i], heuristics[i],
                                      vertex_res, edge_res, window)
        if p is None:
            p = [starts[i]]
        paths.append(p)
        for t, v in enumerate(p):
            vertex_res[(v, t)] = i
            if t + 1 < len(p):
                edge_res[(v, p[t + 1], t)] = i
        for t in range(len(p), window + 1):
            vertex_res[(p[-1], t)] = i
    return paths


def run_rolling_horizon_instance(G, pos_dict, fleet_missions, goal_pool, method: str,
                                 max_steps: int, assignment: Dict[str, str],
                                 window: int = 20, replan_every: int = 5,
                                 disturbance: Optional[Dict[str, Any]] = None,
                                 failure: Optional[Dict[str, Any]] = None,
                                 steps_per_hop: int = 1) -> Dict[str, Any]:
    """
    RHCR-style execution: plan collision-free for a bounded WINDOW, execute
    `replan_every` timesteps, replan from wherever the agents actually are,
    repeat (Li, Tinka, Kiesel, Durham, Kumar & Koenig, AAAI 2021).

    This is the mechanism, not a port. RHCR's contribution is precisely the
    rolling horizon -- bounded-horizon planning keeps computation tractable and
    the agents busy while degrading solution quality only slightly -- so wrapping
    a windowed solver in a replanning loop reproduces it faithfully on a general
    graph. It is NOT the tuned C++ original, and must be labelled
    "RHCR-style, reimplemented" in any write-up.

    It also handles disturbance natively and for free, which is the fair
    comparison FLOWRRA needs: a stalled agent is simply somewhere unexpected at
    the next replan boundary, and the next window is computed from actual
    positions. The cost is real and measured -- replan_s and replan_count -- not
    a failure. Comparing FLOWRRA against a one-shot planner that cannot replan
    would be a strawman; this is the honest opponent.
    """
    from loop_warehouse import WarehouseLoop
    from config_warehouse import CONFIG

    starts = [m["start_node"] for m in fleet_missions]
    goals = [assignment[m["id"]] for m in fleet_missions]
    n = len(starts)

    loop = WarehouseLoop(
        collision_threshold=CONFIG["warehouse"]["collision_threshold"],
        warning_threshold=CONFIG["warehouse"]["warning_threshold"],
    )

    class _P:
        __slots__ = ("id", "current_pos")
        def __init__(self, i, p): self.id, self.current_pos = str(i), p

    def coord(nid):
        d = pos_dict[nid]
        return np.array([d["X"], d["Y"], d["Z"]], dtype=np.float32)

    # ---- PERMANENT FAILURE MODEL ------------------------------------------
    # `disturbance` is a temporary STALL: the vehicle resumes and its order is
    # never in question. That is the setting the existing MAPF malfunction
    # literature covers (k-robust MAPF; Fioravantes et al. 2025 bound the
    # makespan increase after k malfunctions by k turns).
    #
    # `failure` is the setting this harness exists to measure: the vehicle is
    # DEAD and the order it carried is stranded AT ITS CELL. Delivering it means
    # routing another vehicle to that cell FIRST, then to the destination -- a
    # two-leg pickup-and-delivery induced by the failure. That structure is
    # forced by physics, not by design: the package is on the broken robot, not
    # at the goal.
    #
    #   recovery="none"  the order is lost. Not a strawman -- an accurate
    #                    depiction of what a planner alone does.
    #   recovery="naive" the obvious fleet-manager heuristic: nearest vehicle
    #                    already parked at its own goal is retargeted to the dead
    #                    cell, then on to the orphaned destination.
    stalled_until = {}
    if disturbance:
        for a in disturbance.get("agents", []):
            stalled_until[a] = disturbance["step"] + disturbance.get("duration", 10)

    # TRIGGER ON MISSION PROGRESS, NOT CLOCK STEP.
    #
    # A shared step index is not a shared point in the mission. FLOWRRA moves at
    # base_speed 0.5 and ran 287 steps on the same instances RHCR finished in 44,
    # so "hop 25" landed 17% into FLOWRRA's run and 57% into RHCR's. Late means
    # more vehicles already parked, so fewer were still carrying anything to
    # orphan -- measured: FLOWRRA saw 11 failures, the baselines 7, on identical
    # instances. Dividing by base_speed does not fix it, because the step ratio
    # is 6.5x, not 2x.
    #
    # Firing when a FRACTION OF ORDERS HAS BEEN DELIVERED is self-normalising:
    # it needs no pre-pass, no per-arm calibration, and it guarantees vehicles
    # are still active to fail (1 - progress of them). Both arms are then hit at
    # the same point in the work, whatever their clocks are doing.
    failure = failure or {}
    # MULTIPLE WAVES. A rescuer can only die if it is ALREADY rescuing, and at
    # the instant of a single burst nobody is. Raising the failure COUNT gives
    # simultaneous deaths, not a death mid-rescue -- the case that separates
    # re-dispatch (FLOWRRA) from a pickup stranded on a dead rescuer forever
    # (naive). A second wave, after dispatch has happened, is what creates it.
    #
    # Victims are drawn from vehicles still carrying an order, lowest index
    # first. Rescuers on leg 1 or leg 2 ARE carrying an order, so they are in
    # the pool on equal terms -- not preferentially targeted, which would stack
    # the deck for the mechanism under test.
    fail_waves = list(failure.get("progress_waves", []))
    if failure.get("progress") is not None and not fail_waves:
        fail_waves = [failure["progress"]]
    fail_waves = sorted(fail_waves)
    fail_step = int(failure.get("step", -1))
    fail_agents = set(failure.get("agents", []))
    n_to_fail = int(failure.get("count", len(fail_agents)))
    _wave_i = 0
    rescuer_deaths = 0
    recovery_mode = failure.get("recovery", "none")
    dead = set()
    orphan_goal = {}       # dead agent -> goal it never reached
    orphan_cell = {}       # dead agent -> where its load is stranded
    assigned_rescuer = {}  # dead agent -> rescuer index
    leg2 = {}              # rescuer -> destination after pickup
    orders_recovered = 0
    recover_latency = []
    _hcache = {}

    def _h(goal):
        if goal not in _hcache:
            _hcache[goal] = bfs_distances(G, goal)
        return _hcache[goal]

    # DISTANCE, COUNTED, so both arms report the same physical quantity. Each
    # executed move here traverses exactly one edge, so counting moves IS the
    # hop count -- no conversion, no assumption about what a timestep is worth.
    distance_travelled = 0

    # TRANSFER DWELL. FLOWRRA charges pickup_dwell_steps for the physical
    # handover; this arm previously teleported the load, which is not a fairer
    # baseline but an unmodelled one. Matched on DISTANCE, not tick count:
    # FLOWRRA's 4 steps at base_speed 0.5 forgo 2 hops, so this arm waits 2
    # timesteps to forgo the same 2 hops. Matching tick-for-tick would cost it
    # 4 hops and quietly charge the baseline double.
    transfer_dwell_until = {}
    TRANSFER_DWELL = 2

    cur = list(starts)
    finish_step = {}
    integrity_trace, integrity_strict, deadlock_sizes, step_ms = [], [], [], []
    plan_s = replan_s = 0.0
    replans = 0
    paths, path_t = None, 0

    for t in range(max_steps):
        t0 = time.perf_counter()

        # ---- inject permanent failures, and respond ------------------------
        if _wave_i < len(fail_waves):
            # TERMINAL states, not deliveries. Measuring progress as orders
            # DELIVERED gives an arm that cannot recover orphans a ceiling of
            # (n - dead)/n -- RHCR alone tops out near 0.82, so later waves never
            # fire and it received 10.7 failures where the recovering arms took
            # 15.0. The metric rewarded being bad at recovery. Counting a vehicle
            # as resolved when it has delivered OR died always climbs to 1.0, so
            # every arm takes the same waves.
            _done = len(finish_step) + orders_recovered + len(dead)
            # DEADLINE FALLBACK. Progress-triggering alone means an arm that
            # struggles never reaches the later thresholds and so receives FEWER
            # failures than an arm that does well -- measured: 2.67 vs 4.00
            # orphans on identical instances, because one arm stalled at 53%
            # completion and the 0.60 wave never fired. That rewards failing.
            # Each wave therefore also fires at a step deadline spread across the
            # budget, so every arm takes the same number of hits whatever its
            # clock or its competence.
            _deadline = int(max_steps * (_wave_i + 1) / (len(fail_waves) + 1))
            if _done / max(1, n) >= fail_waves[_wave_i] or t >= _deadline:
                fail_agents = set([i for i in range(n)
                                   if i not in dead and cur[i] != goals[i]][:n_to_fail])
                # A victim that was servicing a pickup strands it: the naive
                # policy never re-dispatches, so that load is lost for good.
                for a in fail_agents:
                    if a in assigned_rescuer.values() or a in leg2:
                        rescuer_deaths += 1
                fail_step = t
                _wave_i += 1

        if t == fail_step and fail_agents:
            for a in fail_agents:
                if a < n and cur[a] != goals[a]:
                    dead.add(a)
                    orphan_goal[a] = goals[a]
                    orphan_cell[a] = cur[a]
                    # The dead vehicle is pinned where it stopped. Its goal is
                    # set to its own cell so the planner routes the others
                    # around it as a static obstacle -- which is exactly what
                    # RHCR alone can do about a breakdown.
                    goals[a] = cur[a]
            paths = None  # force an immediate replan around the new obstacles

        if recovery_mode == "naive" and dead:
            for d in list(dead):
                if d in assigned_rescuer:
                    continue
                # Nearest IDLE vehicle by true hop distance to the stranded load.
                # Idle means parked at its own goal: a naive fleet manager will
                # not interrupt a delivery in progress, which is also why it
                # cannot recruit anyone when everyone is still busy.
                hh = _h(orphan_cell[d])
                best, best_d = None, float("inf")
                for i in range(n):
                    if i in dead or i in leg2 or i in assigned_rescuer.values():
                        continue
                    if cur[i] != goals[i]:
                        continue          # busy; naive policy will not divert it
                    dd = hh.get(cur[i])
                    if dd is not None and dd < best_d:
                        best, best_d = i, dd
                if best is None:
                    continue
                assigned_rescuer[d] = best
                leg2[best] = orphan_goal[d]
                goals[best] = orphan_cell[d]      # leg 1: go collect the load
                finish_step.pop(best, None)       # back in service
                paths = None

        # leg 1 complete -> switch the rescuer to the delivery destination.
        #
        # `r not in dead` matters. Killing a vehicle PINS it by setting
        # goals[a] = cur[a], which makes `cur[r] == goals[r]` true -- the same
        # condition this loop uses to mean "arrived at the pickup". A rescuer
        # killed mid-rescue was therefore read as having arrived, had its goal
        # switched to the delivery destination, and was un-pinned as a side
        # effect. It then sat motionless with cur != goals for the rest of the
        # episode, outside the collision exemption, logging a fresh fatal
        # collision every step against the very vehicle it was sent to rescue.
        for r, dest in list(leg2.items()):
            if r in dead:
                continue
            if cur[r] == goals[r] and goals[r] != dest:
                if transfer_dwell_until.get(r, -1) < 0:
                    transfer_dwell_until[r] = t + TRANSFER_DWELL
                    continue          # arrived at the pickup: stand still
                if t < transfer_dwell_until[r]:
                    continue          # still transferring
                goals[r] = dest
                finish_step.pop(r, None)
                paths = None

        if paths is None or path_t >= replan_every:
            p0 = time.perf_counter()
            # Agents already home are pinned so the replan routes around them.
            paths = plan_windowed(G, cur, goals, window, method)
            dt = time.perf_counter() - p0
            if replans == 0:
                plan_s = dt
            else:
                replan_s += dt
            replans += 1
            path_t = 0

        # MATCHED EXECUTION CLOCK.
        #
        # Standard MAPF is discrete: unit-length edges, one vertex move per
        # timestep. These planners follow that convention. FLOWRRA does not --
        # it moves at base_speed 0.5, so it spends TWO simulator steps per hop.
        #
        # Comparing raw step counts across that gap is meaningless, and
        # converting afterwards (steps x base_speed) is worse: sum_of_costs
        # measures TIME, so the conversion produces "hops that could have been
        # covered at nominal speed", which is an upper bound on distance rather
        # than a measurement of it -- and a loose one for FLOWRRA, whose
        # affordance braking means it often covers far less.
        #
        # Equalising the clock at execution instead leaves BOTH algorithms
        # untouched: the planner still plans in unit hops and FLOWRRA still runs
        # its own policy. Only the rate at which a planned hop is consumed
        # changes, so both arms take steps_per_hop simulator steps per graph
        # edge and raw step counts become directly comparable with no conversion
        # anywhere.
        if t % steps_per_hop == 0:
            nxt = []
            for i in range(n):
                if (i in dead or t < stalled_until.get(i, -1)
                        or cur[i] == goals[i]
                        or t < transfer_dwell_until.get(i, -1)):
                    nxt.append(cur[i])   # dead, stalled, parked, or transferring
                else:
                    p = paths[i]
                    nxt.append(p[min(path_t + 1, len(p) - 1)])
            distance_travelled += sum(1 for a, b in zip(cur, nxt) if a != b)
            cur = nxt
            path_t += 1

        nodes = [_P(i, coord(v)) for i, v in enumerate(cur)]
        # EXEMPT EVERY VEHICLE THAT CANNOT MOVE, not just the ones parked on
        # their goal. `cur[i] == goals[i]` alone misses two cases and both occur:
        #
        #   * a DEAD vehicle is pinned by setting goals[a] = cur[a], so it is
        #     covered -- but only until something else changes its goal;
        #   * a dead RESCUER still holds goals[r] = the pickup cell it was
        #     travelling to and will now never reach, so cur != goals forever.
        #
        # The second case leaves two motionless vehicles a cell apart being
        # re-counted as a fresh fatal collision on every step for the rest of the
        # episode -- observed as 140+ logged collisions between one dead fleet
        # and one retired fleet across 700 steps. total_collisions increments per
        # deadlocked STEP, so a single unresolvable pair can dominate the metric.
        frozen = {str(i) for i in range(n) if cur[i] == goals[i] or i in dead}
        integ = loop.check_integrity(nodes, t, frozen)
        integrity_trace.append(float(integ))
        integrity_strict.append(0.0 if loop.deadlocked_nodes else 1.0)
        if loop.deadlocked_nodes:
            deadlock_sizes.append(len(loop.deadlocked_nodes))
        step_ms.append((time.perf_counter() - t0) * 1000.0)

        for i in range(n):
            if i in dead:
                continue
            if cur[i] == goals[i] and i not in finish_step:
                finish_step[i] = t + 1
                # a rescuer arriving at the DESTINATION (not the pickup) has
                # delivered an orphaned order
                if i in leg2 and goals[i] == leg2[i]:
                    orders_recovered += 1
                    recover_latency.append(t + 1 - fail_step)
                    del leg2[i]
        if len(finish_step) + len(dead) == n and not leg2:
            break

    # SURVIVORS ONLY -- see the matching note in benchmark_flowrra. A killed
    # vehicle is charged the full cap under the MAPF convention, and with 9
    # deaths that penalty alone was 89-95% of sum_of_costs, so the metric was
    # measuring how many vehicles died rather than how well the rest routed.
    per_agent = [finish_step.get(i, max_steps) for i in range(n) if i not in dead]
    runs, c = [], 0
    for v in integrity_strict:
        if v < 1.0:
            c += 1
        elif c:
            runs.append(c); c = 0
    if c:
        runs.append(c)

    n_fail = len(dead)
    return {
        # Recovery mode is part of the identity of the arm: "RHCR-PIBT" with no
        # failure response and "RHCR-PIBT+naive" are different systems and must
        # not be averaged together in the summary table.
        "algorithm": f"RHCR-{method}" + ("+naive" if recovery_mode == "naive" else ""),
        "distance_travelled": distance_travelled,
        # Collisions as a RATE. The raw count is a count of deadlocked STEPS,
        # and the arms run for wildly different lengths -- RHCR finishes in
        # ~22-51 steps where naive runs to the 780 cap -- so a short run has an
        # order of magnitude fewer chances to register anything.
        "collision_rate": loop.get_statistics()["total_collisions_occurred"] / max(t + 1, 1),
        "failures_injected": n_fail,
        "orders_orphaned": n_fail,
        "orders_recovered": orders_recovered,
        "rescuer_deaths": rescuer_deaths,
        "orders_lost": n_fail - orders_recovered,
        "recovery_rate": (orders_recovered / n_fail) if n_fail else float("nan"),
        # HOPS. One edge per timestep here, so steps and hops are the same
        # number -- named to match FLOWRRA's converted column so the two are
        # never compared in different units again.
        "mean_recovery_hops": (round(float(np.mean(recover_latency)), 1)
                               if recover_latency else float("nan")),
        "num_agents": n,
        "success": int(len(finish_step) + orders_recovered == n),
        # Orders DELIVERED, matching completion_rate. An order carried to its
        # destination by a rescuer is delivered, and the rate was updated to say
        # so when recovery was added -- but this counter was not, so it read the
        # raw arrival count and undercounted by exactly orders_recovered. Two
        # columns describing the same thing with different definitions is a trap
        # left in the CSV for whoever reads it next.
        "completed": len(finish_step) + orders_recovered,
        # Orders DELIVERED over orders issued. A dead vehicle parked at its own
        # cell must not count as "finished" -- that would score a lost order as a
        # success and erase the entire effect being measured.
        "completion_rate": (len(finish_step) + orders_recovered) / n,
        "agents_planned": n,
        "makespan": int(max(per_agent)) if per_agent else 0,
        "sum_of_costs": int(sum(per_agent)),
        "agents_costed": len(per_agent),
        "collisions": loop.get_statistics()["total_collisions_occurred"],
        "steps_run": len(integrity_trace),
        "plan_time_s": round(plan_s, 4),
        "replan_s": round(replan_s, 4),
        "replan_count": replans,
        "runtime_s": round(plan_s + replan_s, 4),
        "decision_ms_per_step": round(float(np.mean(step_ms)), 4) if step_ms else 0.0,
        "mean_integrity": round(float(np.mean(integrity_strict)), 4) if integrity_strict else 1.0,
        "proximity_margin": round(float(np.mean(integrity_trace)), 4) if integrity_trace else 1.0,
        "steps_incoherent": int(sum(runs)),
        "mean_time_to_recoherence": round(float(np.mean(runs)), 2) if runs else 0.0,
        "max_time_to_recoherence": int(max(runs)) if runs else 0,
        "blast_radius": round(float(np.mean(deadlock_sizes)), 2) if deadlock_sizes else 0.0,
    }