"""
test_line_end.py -- the line-end trap (build log entry 25) and its fix.

THE TRAP
  A braked fleet can sit between two nodes, off the half-cell lattice -- e.g.
  at z = 11.65 on a line whose last node is z = 12. The action mask and the goal
  gradient used to test a full 0.5 step ahead (z = 12.15): past the end of the
  line, so "+Z" was marked impossible, although the move itself (NODE_ALIGNED)
  stops on z = 12. The fleet could only back off or wait. Found by
  trace_fleet.py: fleet 22 spent about 40 of its 87 steps in that loop.

THE FIX
  node_warehouse.reach_position(): the position a move actually reaches. The
  move, the mask and the gradient all use it.

WHAT THIS CHECKS (numpy + networkx only, a few seconds):
  1. fleet 22's situation, rebuilt: from z = 11.65 the mask allows +Z, the
     gradient points +Z, and the move lands exactly on z = 12, from where +X
     reaches the goal;
  2. off-lattice, mid-line (z = 10.65, the line goes on): unchanged;
  3. on the half-cell lattice, every node and half-edge of a random grid, every
     action: mask and gradient identical to the old computation;
  4. NODE_ALIGNED off: identical to the old computation everywhere;
  5. off-lattice positions anywhere on that grid: the only verdicts that change
     are moves the old check refused and the move itself performs.

Run:  python test_line_end.py      (ends with ALL PASS)
"""

import random

import networkx as nx
import numpy as np

import node_warehouse as nw
from node_warehouse import ACTION_DELTAS, FleetNode, build_spatial_indices

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"{'PASS' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAIL.append(name)


def make_graph(coords, edges):
    G = nx.Graph()
    ids = {c: f"n{i}" for i, c in enumerate(coords)}
    for c, nid in ids.items():
        G.add_node(nid)
    for a, b in edges:
        G.add_edge(ids[a], ids[b])
    grid = {c: nid for c, nid in ids.items()}
    return G, grid, ids


def make_fleet(G, grid, pos, goal_coord):
    aisle_index, coords_by_id = build_spatial_indices(grid)
    goal_id = grid[goal_coord]
    dist = dict(nx.single_source_shortest_path_length(G, goal_id))
    return FleetNode(id="f", current_pos=np.array(pos, dtype=np.float32),
                     goal_pos=np.array(goal_coord, dtype=np.float32),
                     warehouse_bounds=(40.0, 40.0, 20.0), G=G, grid_pos_dict=grid,
                     aisle_index=aisle_index, coords_by_id=coords_by_id,
                     goal_distance_map=dist, current_goal_id=goal_id)


# ---- the OLD computations, verbatim from before the fix -----------------------
def old_mask(node, reference_speed=0.5):
    mask = np.zeros(len(ACTION_DELTAS), dtype=bool)
    mask[0] = True
    obstacles = node.static_obstacles or ()
    for action in range(1, len(ACTION_DELTAS)):
        proposed_pos = node.current_pos + (ACTION_DELTAS[action] * reference_speed)
        if not node.is_structurally_valid(proposed_pos, action):
            continue
        if obstacles and node._entering_cell(action) in obstacles:
            continue
        mask[action] = True
    return mask


def old_gradient(node):
    saved = nw.reach_position
    nw.reach_position = lambda n, a, s: n.current_pos + ACTION_DELTAS[a] * s
    try:
        return node.get_goal_gradient()
    finally:
        nw.reach_position = saved


# ---- 1, 2: fleet 22, rebuilt ------------------------------------------------------
def test_fleet22():
    nw.NODE_ALIGNED = True
    # A Z line at (29, 9): z = 8..12, 12 is the end. The goal is one step +X
    # from the top: (30, 9, 12).
    line = [(29, 9, z) for z in range(8, 13)]
    coords = line + [(30, 9, 12)]
    edges = list(zip(line, line[1:])) + [((29, 9, 12), (30, 9, 12))]
    G, grid, _ = make_graph(coords, edges)
    goal = (30, 9, 12)

    f = make_fleet(G, grid, (29, 9, 11.65), goal)
    check("trap_old_mask_refused_+Z", bool(old_mask(f)[5]), False)
    check("fixed_mask_allows_+Z", bool(f.get_valid_action_mask()[5]), True)
    check("old_gradient_saw_no_way_home", bool(np.any(old_gradient(f) > 0)), False)
    g = f.get_goal_gradient()
    check("fixed_gradient_points_+Z", int(np.argmax(g)) + 1, 5)
    f.speed = 0.5
    f.apply_discrete_action(5)
    check("move_lands_on_z12", [round(float(v), 6) for v in f.current_pos], [29.0, 9.0, 12.0])
    check("then_+X_is_open", bool(f.get_valid_action_mask()[1]), True)
    check("then_gradient_points_+X", int(np.argmax(f.get_goal_gradient())) + 1, 1)

    # 2: mid-line, the line goes on: no change.
    f2 = make_fleet(G, grid, (29, 9, 10.65), goal)
    check("midline_mask_unchanged", f2.get_valid_action_mask().tolist(), old_mask(f2).tolist())
    check("midline_gradient_unchanged", f2.get_goal_gradient().tolist(), old_gradient(f2).tolist())


# ---- 3, 4, 5: a random grid ---------------------------------------------------------
def random_grid(seed=0, n=7):
    rng = random.Random(seed)
    coords = [(x, y, z) for x in range(n) for y in range(n) for z in range(0, 6, 3)]
    cs = set(coords)
    edges = []
    for (x, y, z) in coords:
        for dx, dy, dz in ((1, 0, 0), (0, 1, 0)):
            b = (x + dx, y + dy, z + dz)
            if b in cs and rng.random() < 0.6:          # sparse: dead ends and gaps
                edges.append(((x, y, z), b))
    # a few lifts between the two floors (z 0 -> 3), with their middle nodes
    lifts = [(x, y) for x in range(n) for y in range(n) if rng.random() < 0.1]
    extra = []
    for (x, y) in lifts:
        col = [(x, y, z) for z in range(0, 4)]
        extra += [c for c in col if c not in cs]
        edges += list(zip(col, col[1:]))
    G, grid, _ = make_graph(coords + extra, edges)
    return G, grid


def positions(grid, lattice=True, seed=1):
    rng = random.Random(seed)
    out = []
    for (x, y, z) in grid:
        out.append((x, y, z))
        for ax in range(3):
            for off in ((0.5,) if lattice else (rng.uniform(0.05, 0.95),)):
                p = [x, y, z]
                p[ax] += off
                out.append(tuple(p))
    return out


def test_random_grid():
    G, grid = random_grid()
    goal = sorted(grid)[len(grid) // 2]
    # 3: lattice, NODE_ALIGNED on
    nw.NODE_ALIGNED = True
    diff_m = diff_g = 0
    for p in positions(grid, lattice=True):
        f = make_fleet(G, grid, p, goal)
        diff_m += int(f.get_valid_action_mask().tolist() != old_mask(f).tolist())
        diff_g += int(f.get_goal_gradient().tolist() != old_gradient(f).tolist())
    check("lattice_masks_identical", diff_m, 0)
    check("lattice_gradients_identical", diff_g, 0)

    # 4: NODE_ALIGNED off, anywhere
    nw.NODE_ALIGNED = False
    diff_m = diff_g = 0
    for p in positions(grid, lattice=False):
        f = make_fleet(G, grid, p, goal)
        diff_m += int(f.get_valid_action_mask().tolist() != old_mask(f).tolist())
        diff_g += int(f.get_goal_gradient().tolist() != old_gradient(f).tolist())
    check("aligned_off_masks_identical", diff_m, 0)
    check("aligned_off_gradients_identical", diff_g, 0)

    # 5: off-lattice, NODE_ALIGNED on: only refused-but-performable moves change
    nw.NODE_ALIGNED = True
    opened = closed = wrong = 0
    for p in positions(grid, lattice=False):
        f = make_fleet(G, grid, p, goal)
        new, old = f.get_valid_action_mask(), old_mask(f)
        for a in range(1, 7):
            if new[a] and not old[a]:
                opened += 1
                # the move itself must actually happen
                g2 = make_fleet(G, grid, p, goal)
                g2.speed = 0.5
                before = g2.current_pos.copy()
                g2.apply_discrete_action(a)
                wrong += int(np.allclose(before, g2.current_pos))
            elif old[a] and not new[a]:
                closed += 1
    print(f"      off-lattice: {opened} moves newly allowed, {closed} newly refused")
    check("off_lattice_nothing_newly_refused", closed, 0)
    check("off_lattice_some_moves_unlocked", opened > 0, True)
    check("every_unlocked_move_really_moves", wrong, 0)


if __name__ == "__main__":
    saved = nw.NODE_ALIGNED
    try:
        test_fleet22()
        test_random_grid()
    finally:
        nw.NODE_ALIGNED = saved
    print("\n" + ("ALL PASS" if not FAIL else f"FAILURES: {FAIL}"))