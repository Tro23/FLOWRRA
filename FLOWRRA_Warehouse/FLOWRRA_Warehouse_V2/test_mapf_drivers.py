"""
test_mapf_drivers.py -- the MAPF baselines on our floor (build log entry 44).

Two tiny floors, every baseline arm, our physics and our safety on, our brain
off (exactly as deploy_eval builds a baseline arm):

  1. HEAD-ON. A 7 x 2 floor (two rows joined at every column). A at (0,0) for
     (6,0), B at (6,0) for (0,0): they meet head-on. Each planner must get both
     home (no freeze), with no collision and the 1-hop minimum gap kept.
  2. FOLLOWING. One row. B at (1,0) for (6,0), A behind it at (0,0) for (5,0).
     PIBT's own model lets A step into B's cell the moment B leaves it; on our
     floor the following bubble and the guard keep the gap. Checked: the gap
     never falls below 1 hop, no collision, both home; and how much longer A
     takes than alone (the price of the gap, printed).

No network, no learning, no maps on disk. Run:  python test_mapf_drivers.py
"""

import networkx as nx
import numpy as np

from config_warehouse import CONFIG
from core_warehouse import FLOWRRA
from mapf_drivers import brain_off, make_driver
from node_warehouse import precompute_goal_distances

FAIL = []
ARMS = ["PIBT", "RHCR-PP", "RHCR-PIBT"]


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def floor(width, rows):
    G = nx.Graph()
    pos = {}
    for y in range(rows):
        for x in range(width):
            nid = f"n_{x}_{y}"
            G.add_node(nid)
            pos[nid] = {"X": float(x), "Y": float(y), "Z": 0.0}
            if x:
                G.add_edge(f"n_{x - 1}_{y}", nid)
            if y:
                G.add_edge(f"n_{x}_{y - 1}", nid)
    grid = {(int(v["X"]), int(v["Y"]), 0): k for k, v in pos.items()}
    return G, pos, grid


def run(arm, G, pos, grid, pairs, steps=300):
    """pairs: [(id, start, goal)]. Returns (steps to each goal, collisions, min gap)."""
    missions = [{"id": f, "start_node": s, "goal_node": g,
                 "start_pos": np.array([pos[s]["X"], pos[s]["Y"], 0.0], dtype=np.float32),
                 "goal_pos": np.array([pos[g]["X"], pos[g]["Y"], 0.0], dtype=np.float32)}
                for f, s, g in pairs]
    saved = (CONFIG["stream"].get("enabled"), CONFIG["errors"].get("enabled"),
             CONFIG.get("obstacles", {}).get("enabled"))
    CONFIG["stream"]["enabled"] = False
    CONFIG["errors"]["enabled"] = False
    CONFIG.setdefault("obstacles", {})["enabled"] = False
    try:
        gdm = precompute_goal_distances(G, missions)
        with brain_off():
            env = FLOWRRA(G, grid, missions, mode="training", goal_distance_maps=gdm,
                          shared_pool_mode=False)
    finally:
        (CONFIG["stream"]["enabled"], CONFIG["errors"]["enabled"],
         CONFIG["obstacles"]["enabled"]) = saved
    env.max_steps_per_episode = steps
    if arm == "ALONE":
        from drive_shortest_path import ShortestPathDriver
        drv = ShortestPathDriver(env, "never", 0.5, 0)
    else:
        drv = make_driver(arm, env)
    drv.reset_episode_state()
    drv.epsilon_gaussian = lambda *a, **k: 0.0
    env.gnn = drv
    done, gap = {}, float("inf")
    for t in range(steps):
        env.step(episode_step=1, total_episodes=1)
        live = [n for n in env.nodes]
        for i in range(len(live)):
            for j in range(i + 1, len(live)):
                gap = min(gap, float(np.abs(live[i].current_pos - live[j].current_pos).sum()))
        for n in env.nodes:
            if n.id in env.frozen_nodes and n.id not in done:
                done[n.id] = t + 1
        if len(done) == len(pairs):
            break
    return done, env.loop.total_collisions, gap


def test_head_on():
    G, pos, grid = floor(7, 2)
    for arm in ARMS:
        done, coll, gap = run(arm, G, pos, grid,
                              [("A", "n_0_0", "n_6_0"), ("B", "n_6_0", "n_0_0")])
        print(f"      {arm}: home at steps {done}, collisions {coll}, closest {gap:.2f} hops")
        check(f"1_{arm}_both_home", sorted(done), ["A", "B"])
        check(f"1_{arm}_no_collision", coll, 0)
        check(f"1_{arm}_gap_kept", gap >= 1.0 - 1e-3, True)     # float32 positions


def test_following():
    G, pos, grid = floor(7, 1)
    alone, _, _ = run("ALONE", G, pos, grid, [("A", "n_0_0", "n_5_0")])
    for arm in ARMS:
        done, coll, gap = run(arm, G, pos, grid,
                              [("B", "n_1_0", "n_6_0"), ("A", "n_0_0", "n_5_0")])
        print(f"      {arm}: home at steps {done} (A alone: {alone.get('A')}), "
              f"collisions {coll}, closest {gap:.2f} hops")
        check(f"2_{arm}_both_home", sorted(done), ["A", "B"])
        check(f"2_{arm}_no_collision", coll, 0)
        check(f"2_{arm}_gap_kept", gap >= 1.0 - 1e-3, True)     # float32 positions


if __name__ == "__main__":
    test_head_on()
    test_following()
    print("\nALL PASS" if not FAIL else f"\nFAILED: {FAIL}")
