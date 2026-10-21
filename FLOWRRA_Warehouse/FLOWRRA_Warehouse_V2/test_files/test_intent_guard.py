"""
test_intent_guard.py -- the intent guard (guard_warehouse.py; build log entry 29).

WHAT IT PROVES (numpy + networkx only, no torch; under a minute)
  1. landing_position() is exactly where apply_discrete_action() puts a fleet,
     for every action, speed and position tried (on and off the half-cell
     lattice, node-aligned moves on and off).
  2. THE PROPERTY. After the guard, no two moving fleets end a step closer than
     min_gap (1 hop) -- or, if they started closer than that, closer than they
     started -- and no fleet drives THROUGH another (two fleets on one aisle
     never swap order in a step). Tried on:
       * one aisle: every distance from 0.25 to 3.5 hops in quarter hops, every
         pair of moves (forward, back, wait), speeds 0.5 / 0.35 / 0.1 and the
         Tier-3 winner's 1.1: head-on, following, the leader stopping, the
         leader reversing (a ladder retreat), both reversing;
       * a junction: two fleets reaching the crossing from different aisles,
         every position within 3 hops of it, every pair of moves;
       * a chain of three in one aisle, the leader stopping or reversing;
       * random crowds of 4 to 8 fleets on a sparse grid with gaps and dead
         ends, random moves and speeds (2,000 trials).
  3. NO NEEDLESS BRAKING. A pair whose unguarded moves already keep the gap is
     left at full speed, and a fleet whose own move opens the gap is never
     slowed (two-fleet cases).
  4. HopMetric.distance() equals the true graph distance (BFS) between
     positions on the lattice.

Run:  python test_intent_guard.py      (ends with ALL PASS)
"""

import itertools
import random

import networkx as nx
import numpy as np

import node_warehouse as nw
from node_warehouse import FleetNode, build_spatial_indices
from guard_warehouse import HopMetric, crossed, guard_speeds, landing_position

FAIL = []
MIN_GAP = 1.0
EPS = 1e-6


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_graph(coords, edges):
    G = nx.Graph()
    ids = {c: f"n{i}" for i, c in enumerate(coords)}
    G.add_nodes_from(ids.values())
    for a, b in edges:
        G.add_edge(ids[a], ids[b])
    return G, dict(ids)


def make_fleet(G, grid, pos, fid="f"):
    aisle_index, coords_by_id = build_spatial_indices(grid)
    goal = next(iter(grid))
    return FleetNode(id=fid, current_pos=np.array(pos, dtype=np.float32),
                     goal_pos=np.array(goal, dtype=np.float32),
                     warehouse_bounds=(40.0, 40.0, 20.0), G=G, grid_pos_dict=grid,
                     aisle_index=aisle_index, coords_by_id=coords_by_id,
                     goal_distance_map={grid[goal]: 0}, current_goal_id=grid[goal])


def step(fleets, actions, speeds, metric):
    """Guard, then move every fleet as the env would. Returns (start gaps, end
    gaps, guarded speeds, stats)."""
    movers = {f.id: (f, actions[f.id], speeds[f.id]) for f in fleets}
    pairs = list(itertools.combinations([f.id for f in fleets], 2))
    p0 = {f.id: np.asarray(f.current_pos, dtype=np.float64).copy() for f in fleets}
    start = {(a, b): metric.distance(movers[a][0].current_pos, movers[b][0].current_pos)
             for a, b in pairs}
    new, stats = guard_speeds(movers, pairs, metric, min_gap=MIN_GAP)
    for f in fleets:
        act = actions[f.id] if new[f.id] > 0 else 0
        f.speed = new[f.id] if new[f.id] > 0 else f.speed
        f.apply_discrete_action(act)
    end = {}
    for a, b in pairs:
        pa, pb = movers[a][0].current_pos, movers[b][0].current_pos
        # driving THROUGH each other counts as touching (gap 0)
        end[(a, b)] = 0.0 if crossed(p0[a], pa, p0[b], pb) else metric.distance(pa, pb)
    return start, end, new, stats


def violations(start, end):
    return sum(1 for k in start if end[k] < min(MIN_GAP, start[k]) - EPS)


# ---- an aisle: x = 0..14 ------------------------------------------------------------
def aisle():
    coords = [(x, 0, 0) for x in range(15)]
    return make_graph(coords, list(zip(coords, coords[1:])))


def test_landing_matches_move():
    bad = n = 0
    for aligned in (True, False):
        nw.NODE_ALIGNED = aligned
        G, grid = aisle()
        for x in np.arange(0.0, 14.01, 0.05):
            for act in range(7):
                for s in (0.5, 0.35, 0.1, 1.1, 0.0):
                    f = make_fleet(G, grid, (float(x), 0, 0))
                    f.speed = s
                    pred = landing_position(f, act, s)
                    if s > 0:
                        f.apply_discrete_action(act)
                    n += 1
                    bad += int(not np.allclose(pred, f.current_pos, atol=1e-6))
    nw.NODE_ALIGNED = True
    check(f"landing_equals_the_real_move ({n} cases)", bad, 0)


def test_aisle_pairs():
    nw.NODE_ALIGNED = True
    G, grid = aisle()
    metric = HopMetric(G, grid)
    viol = needless = held_opener = cases = slowed_cases = 0
    speeds = (0.5, 0.35, 0.1, 1.1)
    for gapd in np.arange(0.25, 3.51, 0.25):
        for xa in (3.0, 3.25, 3.5, 3.65):
            xb = xa + gapd
            if xb > 13.5:
                continue
            for aa, ab in itertools.product((0, 1, 2), (0, 1, 2)):
                for sa, sb in itertools.product(speeds, speeds):
                    A = make_fleet(G, grid, (xa, 0, 0), "A")
                    B = make_fleet(G, grid, (xb, 0, 0), "B")
                    lA, lB = landing_position(A, aa, sa), landing_position(B, ab, sb)
                    d0 = metric.distance(A.current_pos, B.current_pos)
                    unguarded = (0.0 if crossed(A.current_pos, lA, B.current_pos, lB)
                                 else metric.distance(lA, lB))
                    # Backing away = moving in the direction away from the other
                    # (on one aisle, the sign of the move). Not "the distance did
                    # not shrink": a fleet driving THROUGH the other (overtaking
                    # from 0.25 apart, or the 1.1 Tier-3 bump) can leave the
                    # distance unchanged while passing straight through it.
                    opens_a = (lA[0] - xa) * (xb - xa) < 0
                    opens_b = (lB[0] - xb) * (xa - xb) < 0
                    start, end, new, _ = step([A, B], {"A": aa, "B": ab}, {"A": sa, "B": sb}, metric)
                    cases += 1
                    viol += violations(start, end)
                    if unguarded >= min(MIN_GAP, d0) - EPS:
                        needless += int(new["A"] < sa - 1e-9 or new["B"] < sb - 1e-9)
                    else:
                        slowed_cases += 1
                        # a fleet whose own move opens the gap is never slowed
                        held_opener += int((opens_a and aa != 0 and new["A"] < sa - 1e-9)
                                           or (opens_b and ab != 0 and new["B"] < sb - 1e-9))
    print(f"      aisle: {cases} cases, guard stepped in on {slowed_cases}")
    check("aisle_never_under_one_hop", viol, 0)
    check("aisle_no_needless_braking", needless, 0)
    check("aisle_never_slows_a_fleet_backing_away", held_opener, 0)


# ---- a junction: aisles x = 0..12 at y = 6, and y = 0..12 at x = 6 -------------------
def junction():
    xs = [(x, 6, 0) for x in range(13)]
    ys = [(6, y, 0) for y in range(13) if y != 6]
    coords = xs + ys
    col = [(6, y, 0) for y in range(13)]
    return make_graph(coords, list(zip(xs, xs[1:])) + list(zip(col, col[1:])))


def test_junction_pairs():
    nw.NODE_ALIGNED = True
    G, grid = junction()
    metric = HopMetric(G, grid)
    viol = cases = 0
    pos_x = [(6.0 - d, 6, 0) for d in np.arange(0.0, 3.01, 0.5)] + [(6.0 + d, 6, 0) for d in (0.5, 1.0, 2.0)]
    pos_y = [(6, 6.0 - d, 0) for d in np.arange(0.5, 3.01, 0.5)] + [(6, 6.0 + d, 0) for d in (0.5, 1.5)]
    for pa, pb in itertools.product(pos_x, pos_y):
        for aa, ab in itertools.product(range(7), range(7)):
            for sa, sb in ((0.5, 0.5), (0.35, 0.5), (1.1, 0.5)):
                A = make_fleet(G, grid, pa, "A")
                B = make_fleet(G, grid, pb, "B")
                start, end, _, _ = step([A, B], {"A": aa, "B": ab}, {"A": sa, "B": sb}, metric)
                cases += 1
                viol += violations(start, end)
    print(f"      junction: {cases} cases")
    check("junction_never_under_one_hop", viol, 0)


def test_chain_of_three():
    nw.NODE_ALIGNED = True
    G, grid = aisle()
    metric = HopMetric(G, grid)
    viol = cases = 0
    for g1, g2 in itertools.product((1.0, 1.25, 1.5, 2.0), (1.0, 1.25, 1.5, 2.0)):
        for lead in (0, 1, 2):                  # the leader waits, goes on, or reverses
            for s in (0.5, 1.1):
                A = make_fleet(G, grid, (2.0, 0, 0), "A")
                B = make_fleet(G, grid, (2.0 + g1, 0, 0), "B")
                C = make_fleet(G, grid, (2.0 + g1 + g2, 0, 0), "C")
                start, end, _, _ = step([A, B, C], {"A": 1, "B": 1, "C": lead},
                                        {"A": s, "B": 0.5, "C": 0.5}, metric)
                cases += 1
                viol += violations(start, end)
    print(f"      chains: {cases} cases")
    check("chain_never_under_one_hop", viol, 0)


# ---- random crowds on a sparse grid ----------------------------------------------------
def sparse_grid(seed=3, n=8):
    rng = random.Random(seed)
    coords = [(x, y, 0) for x in range(n) for y in range(n)]
    cs = set(coords)
    edges = [((x, y, 0), (x + dx, y + dy, 0)) for (x, y, _) in coords for dx, dy in ((1, 0), (0, 1))
             if (x + dx, y + dy, 0) in cs and rng.random() < 0.7]
    G, grid = make_graph(coords, edges)
    keep = max(nx.connected_components(G), key=len)
    G = G.subgraph(keep).copy()
    grid = {c: i for c, i in grid.items() if i in keep}
    return G, grid


def test_random_crowds():
    nw.NODE_ALIGNED = True
    G, grid = sparse_grid()
    metric = HopMetric(G, grid)
    rng = random.Random(11)
    cells = sorted(grid)
    viol = slowed = 0
    for t in range(2000):
        k = rng.randint(4, 8)
        picked = rng.sample(cells, k)
        fleets, acts, speeds = [], {}, {}
        for j, c in enumerate(picked):
            f = make_fleet(G, grid, c, f"r{j}")
            mask = f.get_valid_action_mask()
            opts = [a for a in range(7) if mask[a]]
            acts[f.id] = rng.choice(opts)
            speeds[f.id] = rng.choice((0.5, 0.5, 0.35, 0.1, 1.1))
            fleets.append(f)
        # a couple of moves first, so positions leave the lattice
        for _ in range(rng.randint(0, 3)):
            step(fleets, acts, speeds, metric)
            for f in fleets:
                m = f.get_valid_action_mask()
                acts[f.id] = rng.choice([a for a in range(7) if m[a]])
        start, end, new, st = step(fleets, acts, speeds, metric)
        viol += violations(start, end)
        slowed += st["fleets_slowed"]
    print(f"      crowds: 2000 trials, {slowed} fleet-steps slowed by the guard")
    check("crowds_never_under_one_hop", viol, 0)


def test_order_independent():
    """Same fleets, same moves, pairs and fleets listed in a different order:
    the guard must give exactly the same speeds (the simultaneous step's rule:
    no outcome may depend on a fleet's place in the list)."""
    nw.NODE_ALIGNED = True
    G, grid = sparse_grid()
    metric = HopMetric(G, grid)
    rng = random.Random(23)
    cells = sorted(grid)
    differ = 0
    for t in range(500):
        picked = rng.sample(cells, rng.randint(4, 8))
        movers = {}
        for j, c in enumerate(picked):
            f = make_fleet(G, grid, c, f"r{j}")
            m = f.get_valid_action_mask()
            movers[f.id] = (f, rng.choice([a for a in range(7) if m[a]]),
                            rng.choice((0.5, 0.35, 1.1)))
        pairs = list(itertools.combinations(movers, 2))
        s1, _ = guard_speeds(movers, pairs, metric)
        rev = dict(reversed(list(movers.items())))
        s2, _ = guard_speeds(rev, [(b, a) for a, b in reversed(pairs)], metric)
        differ += int(any(abs(s1[k] - s2[k]) > 1e-12 for k in s1))
    check("guard_independent_of_list_order", differ, 0)


def test_metric_is_graph_distance():
    G, grid = sparse_grid()
    metric = HopMetric(G, grid, cutoff=20)
    inv = {v: k for k, v in grid.items()}
    rng = random.Random(5)
    nodes = list(G.nodes)
    bad = 0
    for _ in range(400):
        a, b = rng.sample(nodes, 2)
        true = nx.shortest_path_length(G, a, b)
        bad += int(abs(metric.distance(np.array(inv[a], float), np.array(inv[b], float)) - true) > 1e-9)
    check("hop_metric_equals_bfs_on_nodes", bad, 0)


if __name__ == "__main__":
    saved = nw.NODE_ALIGNED
    try:
        test_landing_matches_move()
        test_metric_is_graph_distance()
        test_aisle_pairs()
        test_junction_pairs()
        test_chain_of_three()
        test_random_crowds()
        test_order_independent()
    finally:
        nw.NODE_ALIGNED = saved
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))
